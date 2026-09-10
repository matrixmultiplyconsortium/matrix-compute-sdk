"""
Pareto Matchmaker and Dynamic Worker Lifecycle Manager for matrix-compute-sdk.

Ranks compute workers via multi-objective Pareto non-dominated sorting across price,
capacity, reputation, and throughput. Maintains dynamic worker reputation with
failure penalties, cooldown blacklisting, and failover candidate selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from matrix_contracts.schemas.discovery import MatchedWorker
except ImportError:
    from matrix_contracts.schemas import MatchedWorker  # type: ignore

from matrix_compute.discovery.relay_client import DiscoveredWorker, RelayClient
from matrix_compute.exceptions import NoWorkersFoundError, WorkerUnavailableError
from matrix_compute.tiling.precision import get_bytes_per_element


# -----------------------------------------------------------------------------
# Data Models
# -----------------------------------------------------------------------------

@dataclass
class MatchResult:
    """
    Result of worker matchmaking and selection.
    Tracks primary worker(s), backup candidates for failover, workload estimates, and costs.
    """
    primary_worker: Optional[DiscoveredWorker] = None
    primary_workers: List[DiscoveredWorker] = field(default_factory=list)
    backup_workers: List[DiscoveredWorker] = field(default_factory=list)
    all_candidates: List[DiscoveredWorker] = field(default_factory=list)
    estimated_gflops: float = 0.0
    estimated_cost_sats: int = 0
    pareto_tier: int = 1
    pareto_tier_counts: Dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.primary_worker is None and self.primary_workers:
            self.primary_worker = self.primary_workers[0]
        elif self.primary_worker is not None and not self.primary_workers:
            self.primary_workers = [self.primary_worker]

    @property
    def total_slots_available(self) -> int:
        """Total slots available aggregated across all selected primary workers."""
        return sum(w.slots_available for w in self.primary_workers)


# -----------------------------------------------------------------------------
# Matchmaker
# -----------------------------------------------------------------------------

class Matchmaker:
    """
    Multi-objective Pareto Matchmaker for decentralized compute workers.
    Solves non-dominated sorting over:
      - Price (sats_per_gflop ascending)
      - VRAM (vram_mb descending)
      - Reputation score (reputation descending)
      - Compute capacity (peak_fp32_tflops descending)
    """

    def __init__(
        self,
        relay_client: Optional[RelayClient] = None,
        blacklist_threshold: int = 3,
        blacklist_cooldown: float = 600.0,
        penalty_per_failure: float = 0.20,
        reward_per_success: float = 0.05,
        min_vram_mb: Optional[int] = None,
        max_price_per_gflop: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        self.relay_client = relay_client
        self.min_vram_mb = min_vram_mb
        self.max_price_per_gflop = max_price_per_gflop
        self.blacklist_threshold = int(blacklist_threshold)
        self.blacklist_cooldown = float(blacklist_cooldown)
        self.penalty_per_failure = float(penalty_per_failure)
        self.reward_per_success = float(reward_per_success)

        self._lock = threading.RLock()
        self._reputation: Dict[str, float] = {}              # pubkey -> float in [0.0, 1.0]
        self._consecutive_failures: Dict[str, int] = {}     # pubkey -> int
        self._blacklisted_until: Dict[str, float] = {}       # pubkey -> expiry timestamp

    # -------------------------------------------------------------------------
    # Dynamic Reputation System
    # -------------------------------------------------------------------------

    def get_reputation(self, worker_pubkey: str) -> float:
        """Returns the current reputation score of the worker, bounded in [0.0, 1.0]."""
        pk = worker_pubkey.strip().lower()
        with self._lock:
            return self._reputation.get(pk, 1.0)

    def is_blacklisted(self, worker_pubkey: str, now: Optional[float] = None) -> bool:
        """Returns True if worker is currently in temporary blacklist cooldown."""
        pk = worker_pubkey.strip().lower()
        current = now if now is not None else time.time()
        with self._lock:
            expiry = self._blacklisted_until.get(pk, 0.0)
            return current < expiry

    def record_failure(
        self,
        worker_pubkey: str,
        penalty: Optional[float] = None,
        reason: Optional[str] = None,
    ) -> float:
        """
        Decrements worker reputation on task failure or verification fraud.
        Triggers temporary blacklist if consecutive failures reach threshold.
        """
        pk = worker_pubkey.strip().lower()
        pen = penalty if penalty is not None else self.penalty_per_failure
        now = time.time()

        with self._lock:
            curr = self._reputation.get(pk, 1.0)
            new_rep = max(0.0, curr - pen)
            self._reputation[pk] = round(new_rep, 4)

            failures = self._consecutive_failures.get(pk, 0) + 1
            self._consecutive_failures[pk] = failures

            if failures >= self.blacklist_threshold:
                self._blacklisted_until[pk] = now + self.blacklist_cooldown

            return self._reputation[pk]

    def report_failure(
        self,
        worker_pubkey: str,
        penalty: Optional[float] = None,
        reason: Optional[str] = None,
    ) -> float:
        """Alias for record_failure."""
        return self.record_failure(worker_pubkey, penalty=penalty, reason=reason)

    def record_success(
        self,
        worker_pubkey: str,
        reward: Optional[float] = None,
    ) -> float:
        """
        Increases worker reputation on successful verified computation.
        Resets consecutive failure count and clears blacklist status.
        """
        pk = worker_pubkey.strip().lower()
        rew = reward if reward is not None else self.reward_per_success

        with self._lock:
            curr = self._reputation.get(pk, 1.0)
            new_rep = min(1.0, curr + rew)
            self._reputation[pk] = round(new_rep, 4)
            self._consecutive_failures[pk] = 0
            self._blacklisted_until.pop(pk, None)

            return self._reputation[pk]

    def report_success(
        self,
        worker_pubkey: str,
        reward: Optional[float] = None,
    ) -> float:
        """Alias for record_success."""
        return self.record_success(worker_pubkey, reward=reward)

    # -------------------------------------------------------------------------
    # Pareto Dominance & Non-Dominated Sorting
    # -------------------------------------------------------------------------

    @staticmethod
    def dominates(a: DiscoveredWorker, b: DiscoveredWorker) -> bool:
        """
        Returns True if candidate `a` strictly dominates candidate `b`.
        a dominates b if:
          1. a is no worse than b in all objectives (price, vram, reputation, tflops).
          2. a is strictly better than b in at least one objective.
        """
        # Price: lower is better
        no_worse_price = a.sats_per_gflop <= b.sats_per_gflop
        strictly_better_price = a.sats_per_gflop < b.sats_per_gflop

        # VRAM: higher is better
        no_worse_vram = a.vram_mb >= b.vram_mb
        strictly_better_vram = a.vram_mb > b.vram_mb

        # Reputation: higher is better
        no_worse_rep = a.reputation_score >= b.reputation_score
        strictly_better_rep = a.reputation_score > b.reputation_score

        # Throughput: higher is better
        no_worse_tflops = a.peak_fp32_tflops >= b.peak_fp32_tflops
        strictly_better_tflops = a.peak_fp32_tflops > b.peak_fp32_tflops

        all_no_worse = (
            no_worse_price and no_worse_vram and no_worse_rep and no_worse_tflops
        )
        at_least_one_better = (
            strictly_better_price
            or strictly_better_vram
            or strictly_better_rep
            or strictly_better_tflops
        )

        return all_no_worse and at_least_one_better

    def compute_pareto_tiers(
        self, candidates: List[DiscoveredWorker]
    ) -> List[List[DiscoveredWorker]]:
        """
        Executes non-dominated sorting over candidates across (price, vram, reputation, tflops).
        Returns list of Pareto tiers [Tier 1, Tier 2, ...].
        """
        if not candidates:
            return []

        remaining = list(candidates)
        tiers: List[List[DiscoveredWorker]] = []

        while remaining:
            current_front: List[DiscoveredWorker] = []
            for candidate in remaining:
                dominated = False
                for other in remaining:
                    if candidate is not other and self.dominates(other, candidate):
                        dominated = True
                        break
                if not dominated:
                    current_front.append(candidate)

            tiers.append(current_front)
            remaining = [c for c in remaining if c not in current_front]

        return tiers

    @staticmethod
    def _lexicographic_sort(candidates: List[DiscoveredWorker]) -> List[DiscoveredWorker]:
        """
        Sorts candidates within a Pareto tier lexicographically:
        price asc -> reputation desc -> vram desc -> tflops desc -> pubkey asc.
        """
        return sorted(
            candidates,
            key=lambda w: (
                w.sats_per_gflop,
                -w.reputation_score,
                -w.vram_mb,
                -w.peak_fp32_tflops,
                w.pubkey,
            ),
        )

    def rank_candidates(
        self, candidates: List[DiscoveredWorker]
    ) -> List[DiscoveredWorker]:
        """
        Ranks candidates by Pareto tier first, and lexicographically within tiers.
        """
        tiers = self.compute_pareto_tiers(candidates)
        ranked: List[DiscoveredWorker] = []
        for tier in tiers:
            ranked.extend(self._lexicographic_sort(tier))
        return ranked

    # -------------------------------------------------------------------------
    # Match Worker Interface
    # -------------------------------------------------------------------------

    def match_worker(
        self,
        workers: Optional[List[DiscoveredWorker]] = None,
        m: int = 1024,
        k: int = 1024,
        n: int = 1024,
        dtype: str = "fp32",
        max_price: Optional[float] = None,
        min_vram_mb: Optional[int] = None,
        required_slots: int = 1,
        num_backups: int = 2,
    ) -> MatchResult:
        """
        Calculates exact required GFLOPs (2 * M * K * N) / 10^9, filters candidate workers,
        computes Pareto tiers, and returns primary + backup worker assignments.
        """
        estimated_gflops = (2.0 * float(m) * float(k) * float(n)) / 1e9

        # Calculate memory requirements in MB if min_vram_mb not explicitly set
        if min_vram_mb is None:
            bytes_per_elem = get_bytes_per_element(dtype)
            total_elements = (m * k) + (k * n) + (m * n)
            required_bytes = total_elements * bytes_per_elem
            min_vram_mb = max(1, int(math.ceil(required_bytes / (1024 * 1024))))

        # Retrieve candidates from relay client if not passed explicitly
        candidate_pool: List[DiscoveredWorker] = []
        if workers is not None:
            candidate_pool = list(workers)
        elif self.relay_client is not None:
            candidate_pool = self.relay_client.query(
                min_vram_mb=min_vram_mb,
                max_price=max_price,
                precision=dtype,
            )

        # Synchronize dynamic reputation scores & filter blacklisted
        now = time.time()
        active_candidates: List[DiscoveredWorker] = []
        for w in candidate_pool:
            w.reputation_score = self.get_reputation(w.pubkey)
            if self.is_blacklisted(w.pubkey, now=now):
                continue
            if required_slots >= 1 and w.slots_available < 1:
                continue
            if w.vram_mb < min_vram_mb:
                continue
            if max_price is not None and w.sats_per_gflop > max_price:
                continue
            if dtype.lower() not in [p.lower() for p in w.supported_precisions]:
                continue
            active_candidates.append(w)

        if not active_candidates:
            raise NoWorkersFoundError(
                "No suitable compute workers available matching task criteria",
                required_vram_mb=min_vram_mb,
                max_price_per_gflop=max_price,
                available_count=0,
            )

        # Compute Pareto tiers and tier counts
        tiers = self.compute_pareto_tiers(active_candidates)
        tier_counts = {idx + 1: len(t) for idx, t in enumerate(tiers)}

        # Rank candidates
        ranked = self.rank_candidates(active_candidates)

        if required_slots <= 0:
            return MatchResult(
                primary_worker=None,
                primary_workers=[],
                backup_workers=[],
                all_candidates=ranked,
                estimated_gflops=estimated_gflops,
                estimated_cost_sats=0,
                pareto_tier=1,
                pareto_tier_counts=tier_counts,
            )

        # Select primary and backup workers
        primary_list = ranked[:required_slots]
        backup_list = ranked[required_slots : required_slots + num_backups]

        primary = primary_list[0] if primary_list else None
        estimated_cost_sats = 0
        if primary:
            price = primary.sats_per_gflop
            if not math.isfinite(price) or price < 0.0:
                price = 0.0
            if not math.isfinite(estimated_gflops) or estimated_gflops < 0.0:
                estimated_gflops = 0.0
            prod = estimated_gflops * price
            if not math.isfinite(prod) or prod < 0.0:
                raw_sats = 0
            else:
                raw_sats = math.ceil(prod)
            min_floor = (
                primary.min_task_sats
                if math.isfinite(primary.min_task_sats) and primary.min_task_sats >= 0
                else 0
            )
            estimated_cost_sats = max(raw_sats, int(min_floor))

        return MatchResult(
            primary_worker=primary,
            primary_workers=primary_list,
            backup_workers=backup_list,
            all_candidates=ranked,
            estimated_gflops=estimated_gflops,
            estimated_cost_sats=estimated_cost_sats,
            pareto_tier=1,
            pareto_tier_counts=tier_counts,
        )

    def find_workers(
        self,
        min_vram_mb: int = 0,
        max_price: Optional[float] = None,
        required_slots: int = 1,
        num_backups: int = 2,
        precision: Optional[str] = None,
        max_dimension: Optional[int] = None,
        candidates: Optional[List[DiscoveredWorker]] = None,
    ) -> MatchResult:
        """
        Finds and ranks compute workers for general slot allocation.
        """
        pool = candidates if candidates is not None else (
            self.relay_client.query(
                min_vram_mb=min_vram_mb,
                max_price=max_price,
                precision=precision,
                max_dimension=max_dimension,
            )
            if self.relay_client
            else []
        )

        return self.match_worker(
            workers=pool,
            m=1024,
            k=1024,
            n=1024,
            dtype=precision or "fp32",
            max_price=max_price,
            min_vram_mb=min_vram_mb,
            required_slots=required_slots,
            num_backups=num_backups,
        )

    def get_failover_worker(
        self,
        match_result: Optional[MatchResult] = None,
        failed_worker_pubkey: str = "",
        candidates: Optional[List[DiscoveredWorker]] = None,
        exclude_pubkeys: Optional[Set[str]] = None,
        min_vram_mb: int = 0,
        max_price: Optional[float] = None,
    ) -> Optional[DiscoveredWorker]:
        """
        Selects the next optimal replacement worker, excluding failed and already-active workers.
        """
        failed_pk = failed_worker_pubkey.strip().lower()
        excluded: Set[str] = set()
        if exclude_pubkeys:
            excluded.update(pk.strip().lower() for pk in exclude_pubkeys)
        if failed_pk:
            excluded.add(failed_pk)

        pool: List[DiscoveredWorker] = []
        if match_result is not None:
            # Check backup workers first, then all candidates
            pool.extend(match_result.backup_workers)
            for c in match_result.all_candidates:
                if c not in pool:
                    pool.append(c)
        elif candidates is not None:
            pool = list(candidates)
        elif self.relay_client is not None:
            pool = self.relay_client.query(min_vram_mb=min_vram_mb, max_price=max_price)

        now = time.time()
        for w in pool:
            pk = w.pubkey.strip().lower()
            if pk in excluded:
                continue
            if self.is_blacklisted(pk, now=now):
                continue
            if w.vram_mb < min_vram_mb:
                continue
            if max_price is not None and w.sats_per_gflop > max_price:
                continue
            return w

        return None
