"""
Parallel Task Scheduler for matrix-compute-sdk Orchestrator subsystem.

Coordinates concurrent execution of tile tasks across a pool of matched workers,
with slot tracking, automated retry, backoff, and dynamic failover upon worker failure.
"""

from __future__ import annotations

from concurrent import futures
import dataclasses
from dataclasses import dataclass
import logging
import math
import random
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np

try:
    from matrix_contracts.schemas import MatchedWorker
except ImportError:
    from matrix_contracts.schemas.discovery import MatchedWorker

from matrix_compute.discovery.relay_client import DiscoveredWorker
from matrix_compute.exceptions import (
    DispatchError,
    WorkerConnectionError,
    WorkerSignatureError,
    WorkerTaskRejectedError,
    WorkerUnavailableError,
)
from matrix_compute.p2p.dispatcher import GrpcTaskDispatcher
from matrix_compute.tiling.partitioner import TileTaskPlan

logger = logging.getLogger(__name__)

WorkerType = Union[MatchedWorker, DiscoveredWorker, Any]


class WorkerSlotTracker:
    """Thread-safe tracker for worker concurrent execution slots."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._cv = threading.Condition(self._lock)
        self._max_slots: Dict[str, int] = {}
        self._active_slots: Dict[str, int] = {}

    def register_worker(self, worker_pubkey: str, max_slots: int = 1) -> None:
        with self._lock:
            self._max_slots[worker_pubkey] = max(int(max_slots), 1)
            if worker_pubkey not in self._active_slots:
                self._active_slots[worker_pubkey] = 0

    def acquire_slot(self, worker_pubkey: str, timeout: Optional[float] = None) -> bool:
        start_time = time.time()
        with self._lock:
            max_s = self._max_slots.get(worker_pubkey, 1)
            while self._active_slots.get(worker_pubkey, 0) >= max_s:
                elapsed = time.time() - start_time
                remaining = (timeout - elapsed) if timeout is not None else None
                if remaining is not None and remaining <= 0:
                    return False
                self._cv.wait(timeout=remaining)
            self._active_slots[worker_pubkey] = self._active_slots.get(worker_pubkey, 0) + 1
            return True

    def acquire_available_worker(
        self,
        candidates: List[WorkerType],
        timeout: Optional[float] = None,
    ) -> Optional[WorkerType]:
        """
        Capacity-aware slot leasing: selects and acquires an execution slot on the least-loaded
        available candidate worker. If all candidates are busy, waits up to timeout seconds.
        Returns the acquired worker, or None if timeout expires.
        """
        if not candidates:
            return None

        start_time = time.time()
        with self._lock:
            # Ensure all candidates are registered
            for w in candidates:
                pk = getattr(w, "worker_pubkey", getattr(w, "pubkey", str(w) if isinstance(w, str) else ""))
                slots = getattr(w, "slots_available", 1)
                self.register_worker(pk, slots)

            while True:
                available: List[Tuple[float, int, WorkerType]] = []
                for idx, w in enumerate(candidates):
                    pk = getattr(w, "worker_pubkey", getattr(w, "pubkey", str(w) if isinstance(w, str) else ""))
                    max_s = self._max_slots.get(pk, 1)
                    act_s = self._active_slots.get(pk, 0)
                    if act_s < max_s:
                        load = act_s / max_s
                        available.append((load, idx, w))

                if available:
                    # Sort by relative load ascending, then original index ascending (preserves primary preference)
                    available.sort(key=lambda x: (x[0], x[1]))
                    selected_worker = available[0][2]
                    selected_pk = getattr(selected_worker, "worker_pubkey", getattr(selected_worker, "pubkey", str(selected_worker) if isinstance(selected_worker, str) else ""))
                    self._active_slots[selected_pk] = self._active_slots.get(selected_pk, 0) + 1
                    return selected_worker

                elapsed = time.time() - start_time
                remaining = (timeout - elapsed) if timeout is not None else None
                if remaining is not None and remaining <= 0:
                    return None
                self._cv.wait(timeout=remaining)

    def release_slot(self, worker_pubkey: str) -> None:
        with self._lock:
            current = self._active_slots.get(worker_pubkey, 0)
            if current > 0:
                self._active_slots[worker_pubkey] = current - 1
            self._cv.notify_all()

    def get_active_slots(self, worker_pubkey: str) -> int:
        with self._lock:
            return self._active_slots.get(worker_pubkey, 0)

    def get_total_active_slots(self) -> int:
        with self._lock:
            return sum(self._active_slots.values())


class ParallelTaskScheduler:
    """
    Schedules and executes tile computation tasks across worker nodes concurrently.
    """

    def __init__(
        self,
        dispatcher: GrpcTaskDispatcher,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
        max_concurrency: int = 16,
    ) -> None:
        self.dispatcher = dispatcher
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.max_concurrency = max_concurrency

        self._pool = futures.ThreadPoolExecutor(max_workers=self.max_concurrency)
        self._lock = threading.RLock()
        self._worker_failure_counts: Dict[str, int] = {}
        self._blacklisted_workers: Dict[str, float] = {}  # pubkey -> unblacklist_time
        self._blacklist_duration_sec = 600.0
        self._slot_tracker = WorkerSlotTracker()
        self._is_shutdown = False

    def get_healthy_workers(self, workers: List[WorkerType]) -> List[WorkerType]:
        """Filters out currently blacklisted workers and returns healthy candidates."""
        now = time.time()
        with self._lock:
            # Clean up expired blacklists
            expired = [pk for pk, expiry in self._blacklisted_workers.items() if now >= expiry]
            for pk in expired:
                del self._blacklisted_workers[pk]
                self._worker_failure_counts.pop(pk, None)

            return [
                w for w in workers
                if getattr(w, "worker_pubkey", getattr(w, "pubkey", "")) not in self._blacklisted_workers
            ]

    def record_worker_failure(self, worker_pubkey: str) -> None:
        """Records a failure for the worker and blacklists after 3 consecutive failures."""
        if not worker_pubkey:
            return
        with self._lock:
            count = self._worker_failure_counts.get(worker_pubkey, 0) + 1
            self._worker_failure_counts[worker_pubkey] = count
            if count >= 3:
                self._blacklisted_workers[worker_pubkey] = time.time() + self._blacklist_duration_sec
                logger.warning(
                    "Worker %s blacklisted for %ds after %d consecutive failures",
                    worker_pubkey,
                    int(self._blacklist_duration_sec),
                    count,
                )

    def record_worker_success(self, worker_pubkey: str) -> None:
        """Resets the failure counter on success."""
        if not worker_pubkey:
            return
        with self._lock:
            self._worker_failure_counts[worker_pubkey] = 0

    def submit_tile(
        self,
        task_plan: TileTaskPlan,
        candidate_workers: List[WorkerType],
        macaroon: str,
        payment_hash: str,
        dtype: Optional[str] = "float32",
        timeout: Optional[float] = None,
        on_failover_callback: Optional[Callable[[WorkerType, Exception], None]] = None,
        escrow_manager: Optional[Any] = None,
        escrow_provider: Optional[Callable[[TileTaskPlan, WorkerType, int], Tuple[TileTaskPlan, str, str]]] = None,
    ) -> futures.Future:
        """
        Submits a single tile execution to the thread pool with failover capability.

        Returns:
            futures.Future resolving to Tuple[np.ndarray, TileResult]
        """
        with self._lock:
            if self._is_shutdown:
                raise RuntimeError("Scheduler is shutdown")
            return self._pool.submit(
                self.execute_with_failover,
                task_plan=task_plan,
                candidate_workers=candidate_workers,
                macaroon=macaroon,
                payment_hash=payment_hash,
                dtype=dtype,
                on_failover_callback=on_failover_callback,
                timeout=timeout,
                escrow_manager=escrow_manager,
                escrow_provider=escrow_provider,
            )

    def execute_with_failover(
        self,
        task_plan: TileTaskPlan,
        candidate_workers: List[WorkerType],
        macaroon: str,
        payment_hash: str,
        dtype: Optional[str] = None,
        on_failover_callback: Optional[Callable[[WorkerType, Exception], None]] = None,
        timeout: Optional[float] = None,
        escrow_manager: Optional[Any] = None,
        escrow_provider: Optional[Callable[[TileTaskPlan, WorkerType, int], Tuple[TileTaskPlan, str, str]]] = None,
    ) -> Tuple[np.ndarray, Any]:
        """
        Executes a single tile task on the primary candidate worker, failing over to backup
        candidates if transient errors or timeouts occur.
        """
        attempts = 0
        last_error: Optional[Exception] = None
        candidate_idx = 0
        excluded_pubkeys: Set[str] = set()

        while attempts <= self.max_retries:
            healthy_candidates = self.get_healthy_workers(candidate_workers)
            if not healthy_candidates:
                # Fallback to all candidates if all are marked blacklisted
                healthy_candidates = candidate_workers

            available_candidates = [
                w for w in healthy_candidates
                if getattr(w, "worker_pubkey", getattr(w, "pubkey", "")) not in excluded_pubkeys
            ]
            if not available_candidates:
                available_candidates = healthy_candidates

            if not available_candidates:
                raise WorkerUnavailableError(
                    f"No healthy workers available for task '{task_plan.task_id}'",
                    available_count=0,
                )

            # Capacity-aware slot acquisition across available candidates
            worker = self._slot_tracker.acquire_available_worker(
                available_candidates, timeout=timeout if timeout is not None else 10.0
            )
            if worker is None:
                attempts += 1
                candidate_idx += 1
                last_error = WorkerUnavailableError(
                    f"Worker slot acquisition timed out after 10.0s for task '{task_plan.task_id}'",
                    available_count=0,
                )
                if self.backoff_factor > 0 and attempts <= self.max_retries:
                    delay = self.backoff_factor * (2 ** (attempts - 1)) + random.uniform(0, 0.1 * self.backoff_factor)
                    time.sleep(delay)
                continue

            pubkey = getattr(worker, "worker_pubkey", getattr(worker, "pubkey", ""))
            target_multiaddr = getattr(
                worker, "p2p_multiaddr",
                getattr(worker, "grpc_endpoint",
                getattr(worker, "multiaddr",
                getattr(worker, "target", str(worker) if isinstance(worker, str) else "")))
            )

            # Escrow and retry handling:
            # If attempts > 0 and escrow_provider or escrow_manager is supplied, re-provision attempt-qualified escrow
            cur_task_plan = task_plan
            cur_macaroon = macaroon
            cur_payment_hash = payment_hash
            new_escrow = None

            if attempts > 0:
                if escrow_provider is not None:
                    cur_task_plan, cur_macaroon, cur_payment_hash = escrow_provider(task_plan, worker, attempts)
                elif escrow_manager is not None:
                    retry_task_id = f"{task_plan.task_id}_retry_{attempts}"
                    sats_per_gflop = getattr(worker, "sats_per_gflop", 0.0005)
                    sats = max(int(math.ceil(task_plan.estimated_gflops * sats_per_gflop)), 10)
                    worker_ln = getattr(
                        worker,
                        "lightning_routing_pubkey",
                        getattr(worker, "lightning_address", "worker@node"),
                    )
                    new_escrow = escrow_manager.create_escrow(
                        task_id=retry_task_id,
                        total_sats=sats,
                        worker_lightning_addr=worker_ln,
                        timeout_seconds=timeout or 30.0,
                        max_gflops=max(task_plan.estimated_gflops, 1.0),
                    )
                    if hasattr(escrow_manager, "hold_escrow"):
                        escrow_manager.hold_escrow(new_escrow.payment_hash)
                    elif hasattr(escrow_manager, "lock_escrow"):
                        escrow_manager.lock_escrow(new_escrow.payment_hash)
                    cur_macaroon = new_escrow.macaroon
                    cur_payment_hash = new_escrow.payment_hash
                    cur_task_plan = dataclasses.replace(task_plan, task_id=retry_task_id)

            try:
                result_matrix, raw_res = self.dispatcher.dispatch_tile(
                    target_multiaddr=target_multiaddr,
                    worker_pubkey=pubkey,
                    task_plan=cur_task_plan,
                    macaroon=cur_macaroon,
                    payment_hash=cur_payment_hash,
                    dtype=dtype,
                )
                self.record_worker_success(pubkey)
                return result_matrix, raw_res
            except Exception as ex:
                attempts += 1
                candidate_idx += 1
                last_error = ex
                excluded_pubkeys.add(pubkey)
                self.record_worker_failure(pubkey)
                logger.info(
                    "Task %s failed on worker %s (attempt %d/%d): %s. Failing over...",
                    task_plan.task_id,
                    pubkey,
                    attempts,
                    self.max_retries,
                    ex,
                )
                if new_escrow is not None and escrow_manager is not None:
                    try:
                        if hasattr(escrow_manager, "cancel_escrow"):
                            escrow_manager.cancel_escrow(new_escrow.payment_hash, reason="failover_retry_failed")
                        elif hasattr(escrow_manager, "abort_and_refund"):
                            escrow_manager.abort_and_refund(new_escrow.payment_hash, reason="failover_retry_failed")
                    except Exception:
                        pass

                if on_failover_callback:
                    try:
                        on_failover_callback(worker, ex)
                    except Exception:
                        pass

                if self.backoff_factor > 0 and attempts <= self.max_retries:
                    delay = self.backoff_factor * (2 ** (attempts - 1)) + random.uniform(0, 0.1 * self.backoff_factor)
                    time.sleep(delay)
            finally:
                self._slot_tracker.release_slot(pubkey)

        raise DispatchError(
            f"All retry and failover attempts exhausted for task '{task_plan.task_id}'",
            task_id=task_plan.task_id,
            cause=last_error,
        )

    def schedule_all(
        self,
        tasks: List[TileTaskPlan],
        executor_fn: Callable[[TileTaskPlan], Any],
    ) -> List[Any]:
        """Schedules all tasks concurrently using the thread pool."""
        with self._lock:
            if self._is_shutdown:
                raise RuntimeError("Scheduler is shutdown")
            futures_list = [self._pool.submit(executor_fn, t) for t in tasks]
        return [f.result() for f in futures_list]

    def cancel(self, reason: str = "cancelled") -> None:
        """Cancels pending tasks in the pool."""
        pass

    def shutdown(self, wait: bool = True) -> None:
        """Shuts down the thread pool executor."""
        with self._lock:
            if not self._is_shutdown:
                self._is_shutdown = True
                self._pool.shutdown(wait=wait)


# Compatibility alias
TaskScheduler = ParallelTaskScheduler
