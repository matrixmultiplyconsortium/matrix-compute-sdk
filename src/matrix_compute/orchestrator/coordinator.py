"""
Distributed Matrix Multiplication Coordinator for matrix-compute-sdk.

Coordinates the complete end-to-end distributed GEMM lifecycle:
matrix partitioning -> worker discovery & matchmaking -> L402 escrow creation ->
gRPC task dispatch -> dynamic Freivalds mathematical verification ->
payout release or fraud slashing -> result accumulation & assembly.
"""

from __future__ import annotations

from concurrent import futures
from dataclasses import dataclass, field
import collections
import hashlib
import logging
import math
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import uuid

import numpy as np

try:
    from matrix_contracts.schemas import (
        FreivaldsFraudProof,
        MatchedWorker,
        WorkerQueryRequest,
    )
except ImportError:
    from matrix_contracts.schemas.discovery import MatchedWorker
    from matrix_contracts.schemas.verification import FreivaldsFraudProof
    from matrix_contracts.schemas.relay import WorkerQueryRequest

try:
    from matrix_contracts.v1 import (
        LeaseMatrixRequest,
        LeaseMatrixResponse,
        IterativeStepRequest,
        IterativeStepResponse,
        ReleaseLeaseRequest,
        ReleaseLeaseResponse,
        matrix_service_pb2 as pb2,
    )
except ImportError:
    try:
        from matrix_contracts.proto import (
            LeaseMatrixRequest,
            LeaseMatrixResponse,
            IterativeStepRequest,
            IterativeStepResponse,
            ReleaseLeaseRequest,
            ReleaseLeaseResponse,
            matrix_service_pb2 as pb2,
        )
    except ImportError:
        pass

from matrix_compute.config import ClientConfig, PartitioningStrategy
from matrix_compute.crypto.signer import ClientSigner, verify_signature
from matrix_compute.discovery.matchmaker import Matchmaker
from matrix_compute.discovery.relay_client import DiscoveredWorker, RelayClient
from matrix_compute.exceptions import (
    FraudDetectedError,
    MatrixDimensionError,
    NoWorkersFoundError,
    VerificationFailedError,
    WorkerSignatureError,
    WorkerUnavailableError,
)
from matrix_compute.orchestrator.scheduler import ParallelTaskScheduler
from matrix_compute.p2p.dispatcher import GrpcTaskDispatcher
from matrix_compute.payment.escrow import EscrowManager, EscrowStatus
from matrix_compute.test_doubles.mock_worker import MockWorkerNode
from matrix_compute.tiling.assembler import ResultAssembler
from matrix_compute.tiling.partitioner import MatrixTiler, TileGridPlan, TileTaskPlan
from matrix_compute.tiling.precision import get_numpy_dtype, normalize_dtype, truncate_to_bf16
from matrix_compute.verification.fraud_proof import FraudProofBuilder
from matrix_compute.verification.freivalds import FreivaldsVerifier, verify_iterative_step

logger = logging.getLogger(__name__)


@dataclass
class JobManifest:
    """Execution manifest for a distributed matrix multiplication run."""

    job_id: str
    m: int
    k: int
    n: int
    dtype: str
    tile_size: int
    total_tasks: int
    completed_tasks: int = 0
    total_cost_sats: int = 0
    elapsed_time_sec: float = 0.0
    task_ids: List[str] = field(default_factory=list)
    peak_concurrency: int = 0


@dataclass
class PipelinedTaskContext:
    """Context metadata for an in-flight pipelined tile task."""

    worker_idx: int
    task_plan: TileTaskPlan
    assigned_worker: Any
    candidates: List[Any]
    escrow: Any
    assigned_pubkey: str
    is_failed_over: bool = False


@dataclass
class IterativeResult:
    """
    Execution result of distributed stateful iterative power computation (v_T = A^T v_0).
    Provides seamless NumPy array compatibility via __array__ protocol.
    """

    final_vector: np.ndarray
    iterations_completed: int
    lease_id: str
    worker_pubkey: str
    spot_checks_performed: int
    spot_checks_passed: int
    total_compute_time_ms: int
    bytes_freed: int
    manifest: Optional[JobManifest] = None

    def __array__(self, dtype: Optional[Any] = None) -> np.ndarray:
        """Enables direct np.asarray(result) and np.allclose(result, target) evaluation."""
        return np.asarray(self.final_vector, dtype=dtype)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.final_vector.shape

    @property
    def ndim(self) -> int:
        return self.final_vector.ndim

    @property
    def dtype(self) -> np.dtype:
        return self.final_vector.dtype

    def __len__(self) -> int:
        return len(self.final_vector)

    def __getitem__(self, item: Any) -> Any:
        return self.final_vector[item]

    def __iter__(self) -> Any:
        return iter(self.final_vector)

    def copy(self) -> np.ndarray:
        return self.final_vector.copy()

    def __mul__(self, other: Any) -> Any:
        return self.final_vector * other

    def __rmul__(self, other: Any) -> Any:
        return other * self.final_vector

    def __add__(self, other: Any) -> Any:
        return self.final_vector + other

    def __radd__(self, other: Any) -> Any:
        return other + self.final_vector

    def __sub__(self, other: Any) -> Any:
        return self.final_vector - other

    def __rsub__(self, other: Any) -> Any:
        return other - self.final_vector

    def __truediv__(self, other: Any) -> Any:
        return self.final_vector / other

    def __rtruediv__(self, other: Any) -> Any:
        return other / self.final_vector


class DistributedMatmulCoordinator:
    """
    Main pipeline orchestrator for distributed matrix multiplication.
    """

    def __init__(
        self,
        config: ClientConfig,
        signer: ClientSigner,
        relay_client: RelayClient,
        matchmaker: Matchmaker,
        escrow_manager: EscrowManager,
        scheduler: ParallelTaskScheduler,
        dispatcher: GrpcTaskDispatcher,
        mock_workers: Optional[List[MockWorkerNode]] = None,
        max_concurrency: Optional[int] = None,
    ) -> None:
        self.config = config
        self.signer = signer
        self.relay_client = relay_client
        self.matchmaker = matchmaker
        self.escrow_manager = escrow_manager
        self.scheduler = scheduler
        self.dispatcher = dispatcher
        self.mock_workers = mock_workers or []
        self._last_manifest: Optional[JobManifest] = None
        self._last_assembler: Optional[ResultAssembler] = None
        self.max_concurrency = (
            max_concurrency
            or getattr(self.config, "max_concurrency", None)
            or getattr(self.scheduler, "max_concurrency", 16)
        )
        self._peak_in_flight: int = 0

        self.verifier = FreivaldsVerifier(
            k_rounds=self.config.freivalds_rounds,
            epsilon_tolerance=self.config.verification_tolerance,
            use_dynamic_bound=self.config.use_dynamic_freivalds_bound,
            relative_discrepancy_threshold=self.config.relative_discrepancy_threshold,
        )

    def execute_gemm(
        self,
        A: np.ndarray,
        B: np.ndarray,
        dtype: str = "fp32",
        tile_size: int = 1024,
        partitioning_strategy: Optional[Union[str, PartitioningStrategy]] = None,
        num_strips: Optional[int] = None,
        grid_rows: Optional[int] = None,
        grid_cols: Optional[int] = None,
        grid_shape: Optional[Tuple[int, int]] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Executes distributed matrix multiplication C = A x B.
        """
        start_time = time.time()
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        m, k = A.shape
        _, n = B.shape
        effective_tile_size = int(tile_size) if tile_size is not None else getattr(self.config, "default_tile_size", 1024)

        # 1. Resolve partitioning strategy
        raw_strategy = (
            partitioning_strategy
            or kwargs.get("partitioning_strategy")
            or kwargs.get("strategy")
            or getattr(self.config, "partitioning_strategy", PartitioningStrategy.BLOCK_3D)
        )
        strat_val = raw_strategy.value if hasattr(raw_strategy, "value") else str(raw_strategy)
        strat_norm = strat_val.lower().strip()

        is_outer_1d = strat_norm in (
            "outer_product_1d",
            "outer_1d",
            "1d_outer_product",
            "outer_product",
            "outerproduct1d",
        )
        is_output_2d = strat_norm in (
            "output_grid_2d",
            "output_grid",
            "2d_output_grid",
            "outputgrid2d",
            "grid_2d",
            "output_2d",
            "output2d",
            "2d",
            "2d_grid",
        )
        if not is_outer_1d and not is_output_2d and strat_norm not in ("block_3d", "block", "3d", "3d_block", "block3d"):
            raise ValueError(f"Unknown partitioning strategy: '{raw_strategy}'")

        # 2. Resolve strip count (num_strips) precedence for 1D outer product
        resolved_num_strips: Optional[int] = None
        if is_outer_1d:
            if num_strips is not None:
                resolved_num_strips = int(num_strips)
            elif kwargs.get("num_strips") is not None:
                resolved_num_strips = int(kwargs["num_strips"])
            elif getattr(self.config, "num_strips", None) is not None:
                resolved_num_strips = int(self.config.num_strips)
            elif kwargs.get("candidate_workers") or kwargs.get("candidates"):
                cands = kwargs.get("candidate_workers") or kwargs.get("candidates")
                if cands and len(cands) > 0:
                    resolved_num_strips = len(cands)
            elif self.mock_workers and len(self.mock_workers) > 0:
                resolved_num_strips = len(self.mock_workers)

            if resolved_num_strips is None or resolved_num_strips < 1:
                resolved_num_strips = max(1, math.ceil(k / int(effective_tile_size)))

        # 2b. Resolve grid dimensions (P_r, P_c) precedence for 2D output grid
        resolved_grid_rows: Optional[int] = None
        resolved_grid_cols: Optional[int] = None
        if is_output_2d:
            grid_shape_arg = grid_shape or kwargs.get("grid_shape") or getattr(self.config, "grid_shape", None)
            if grid_shape_arg:
                resolved_grid_rows, resolved_grid_cols = int(grid_shape_arg[0]), int(grid_shape_arg[1])

            if grid_rows is not None:
                resolved_grid_rows = int(grid_rows)
            elif kwargs.get("grid_rows") is not None:
                resolved_grid_rows = int(kwargs["grid_rows"])
            elif kwargs.get("p_r") is not None:
                resolved_grid_rows = int(kwargs["p_r"])
            elif getattr(self.config, "grid_rows", None) is not None and resolved_grid_rows is None:
                resolved_grid_rows = int(self.config.grid_rows)

            if grid_cols is not None:
                resolved_grid_cols = int(grid_cols)
            elif kwargs.get("grid_cols") is not None:
                resolved_grid_cols = int(kwargs["grid_cols"])
            elif kwargs.get("p_c") is not None:
                resolved_grid_cols = int(kwargs["p_c"])
            elif getattr(self.config, "grid_cols", None) is not None and resolved_grid_cols is None:
                resolved_grid_cols = int(self.config.grid_cols)

            # Heuristic factorization if neither is explicitly provided
            if resolved_grid_rows is None and resolved_grid_cols is None:
                cands = kwargs.get("candidate_workers") or kwargs.get("candidates")
                w_count = len(cands) if cands else (len(self.mock_workers) if self.mock_workers else 0)
                if w_count > 0:
                    best_r, best_c = 1, w_count
                    min_diff = float("inf")
                    target_ratio = float(m) / float(n) if n > 0 else 1.0
                    for r in range(1, w_count + 1):
                        if w_count % r == 0:
                            c = w_count // r
                            ratio = float(r) / float(c)
                            diff = abs(ratio - target_ratio)
                            if diff < min_diff:
                                min_diff = diff
                                best_r, best_c = r, c
                    resolved_grid_rows, resolved_grid_cols = best_r, best_c

            if resolved_grid_rows is None:
                resolved_grid_rows = max(1, math.ceil(m / int(effective_tile_size)))
            if resolved_grid_cols is None:
                resolved_grid_cols = max(1, math.ceil(n / int(effective_tile_size)))

        # 3. Create tiling plan
        tiler_kwargs = {
            k_arg: v_arg
            for k_arg, v_arg in kwargs.items()
            if k_arg not in (
                "strategy",
                "partitioning_strategy",
                "num_strips",
                "grid_rows",
                "grid_cols",
                "grid_shape",
                "p_r",
                "p_c",
                "job_id",
                "task_id_prefix",
                "tile_size",
                "dtype",
            )
        }
        if is_output_2d:
            tiler_kwargs["grid_rows"] = resolved_grid_rows
            tiler_kwargs["grid_cols"] = resolved_grid_cols

        grid_plan = MatrixTiler.create_plan(
            a=A,
            b=B,
            tile_size=effective_tile_size,
            dtype=dtype,
            task_id_prefix=job_id,
            job_id=job_id,
            strategy=raw_strategy,
            num_strips=resolved_num_strips,
            **tiler_kwargs,
        )

        manifest = JobManifest(
            job_id=job_id,
            m=m,
            k=k,
            n=n,
            dtype=dtype,
            tile_size=effective_tile_size,
            total_tasks=len(grid_plan.tasks),
            task_ids=[t.task_id for t in grid_plan.tasks],
        )
        self._last_manifest = manifest

        # 2. Worker Discovery & Matchmaking
        workers = self._discover_and_match_workers(required_slots=len(grid_plan.tasks))
        if not workers:
            raise NoWorkersFoundError(
                "No compute workers found matching criteria",
                required_vram_mb=self.config.min_worker_vram_mb,
                max_price_per_gflop=self.config.max_price_per_gflop,
                available_count=0,
            )

        # 3. Result Assembler setup
        assembler = ResultAssembler.from_plan(
            grid_plan,
            zero_reduction=True if is_output_2d else None,
        )
        self._last_assembler = assembler

        # 4. Pre-group scheduled tasks by assigned worker
        num_workers = len(workers)
        worker_task_groups: List[Tuple[Any, List[TileTaskPlan]]] = [
            (w, []) for w in workers
        ]
        for idx, task_plan in enumerate(grid_plan.tasks):
            worker_task_groups[idx % num_workers][1].append(task_plan)

        active_groups = [(w, tasks) for w, tasks in worker_task_groups if tasks]
        total_sats_spent = 0

        # Provision 1 batch escrow per worker
        worker_group_escrows: List[Tuple[Any, List[TileTaskPlan], Any]] = []
        for g_idx, (assigned_worker, tasks) in enumerate(active_groups):
            assigned_pubkey = getattr(assigned_worker, "worker_pubkey", getattr(assigned_worker, "pubkey", ""))
            sats_per_gflop = getattr(assigned_worker, "sats_per_gflop", self.config.max_price_per_gflop)
            min_task_sats = int(getattr(assigned_worker, "min_task_sats", 10))
            total_sats = sum(max(int(math.ceil(t.estimated_gflops * sats_per_gflop)), min_task_sats) for t in tasks)
            total_gflops = sum(t.estimated_gflops for t in tasks)
            total_sats_spent += total_sats

            worker_ln = (
                getattr(assigned_worker, "worker_lightning_address", None)
                or getattr(assigned_worker, "lightning_routing_pubkey", None)
                or getattr(assigned_worker, "lightning_address", None)
                or "worker@node"
            )
            task_ids = [t.task_id for t in tasks]
            batch_job_id = f"{job_id}_b{g_idx}" if len(active_groups) > 1 else job_id

            if hasattr(self.escrow_manager, "create_batch_escrow"):
                escrow = self.escrow_manager.create_batch_escrow(
                    worker_pubkey=assigned_pubkey,
                    task_ids=task_ids,
                    job_id=batch_job_id,
                    total_sats=total_sats,
                    worker_lightning_addr=worker_ln,
                    timeout_seconds=self.config.task_timeout_seconds,
                    max_gflops=max(total_gflops, 1.0),
                )
            else:
                escrow = self.escrow_manager.create_escrow(
                    task_id=task_ids[0],
                    total_sats=total_sats,
                    worker_lightning_addr=worker_ln,
                    timeout_seconds=self.config.task_timeout_seconds,
                    max_gflops=max(total_gflops, 1.0),
                )

            if hasattr(self.escrow_manager, "hold_escrow"):
                self.escrow_manager.hold_escrow(escrow.payment_hash)
            elif hasattr(self.escrow_manager, "lock_escrow"):
                self.escrow_manager.lock_escrow(escrow.payment_hash)

            worker_group_escrows.append((assigned_worker, tasks, escrow))

        # 5. Asynchronous 4-Stage Pipelined Sliding Window Execution
        eff_concurrency = kwargs.get("max_concurrency", self.max_concurrency)
        eff_concurrency = max(int(eff_concurrency), 1)
        peak_concurrency: int = 0
        first_fraud_error: Optional[FraudDetectedError] = None

        # Build per-worker task queues and state
        worker_queues: List[collections.deque[TileTaskPlan]] = []
        worker_in_flight: List[int] = [0] * len(worker_group_escrows)
        worker_verified: List[int] = [0] * len(worker_group_escrows)
        worker_total_tasks: List[int] = []
        worker_halted: List[bool] = [False] * len(worker_group_escrows)
        worker_max_slots: List[int] = []
        candidates_map: List[List[Any]] = []
        escrow_map: List[Any] = []
        pubkey_map: List[str] = []

        for g_idx, (assigned_worker, tasks, escrow) in enumerate(worker_group_escrows):
            assigned_pubkey = getattr(assigned_worker, "worker_pubkey", getattr(assigned_worker, "pubkey", ""))
            pubkey_map.append(assigned_pubkey)
            worker_queues.append(collections.deque(tasks))
            worker_total_tasks.append(len(tasks))
            slots = getattr(assigned_worker, "slots_available", 1)
            worker_max_slots.append(max(int(slots), 1))
            escrow_map.append(escrow)
            candidates = [assigned_worker] + [
                w for w in workers
                if getattr(w, "worker_pubkey", getattr(w, "pubkey", "")) != assigned_pubkey
            ]
            candidates_map.append(candidates)

        in_flight_futures: Dict[futures.Future, PipelinedTaskContext] = {}

        def _dispatch_ready() -> None:
            nonlocal in_flight_futures, peak_concurrency
            progress = True
            while len(in_flight_futures) < eff_concurrency and progress:
                progress = False
                for i in range(len(worker_group_escrows)):
                    if len(in_flight_futures) >= eff_concurrency:
                        break
                    if worker_halted[i]:
                        continue
                    if worker_in_flight[i] >= worker_max_slots[i]:
                        continue
                    if not worker_queues[i]:
                        continue

                    task_plan = worker_queues[i].popleft()
                    worker_in_flight[i] += 1
                    progress = True

                    ctx = PipelinedTaskContext(
                        worker_idx=i,
                        task_plan=task_plan,
                        assigned_worker=worker_group_escrows[i][0],
                        candidates=candidates_map[i],
                        escrow=escrow_map[i],
                        assigned_pubkey=pubkey_map[i],
                        is_failed_over=False,
                    )

                    def _on_failover(failed_w: Any, exc: Exception, context: PipelinedTaskContext = ctx) -> None:
                        context.is_failed_over = True

                    fut = self.scheduler.submit_tile(
                        task_plan=task_plan,
                        candidate_workers=ctx.candidates,
                        macaroon=ctx.escrow.macaroon,
                        payment_hash=ctx.escrow.payment_hash,
                        dtype=dtype,
                        timeout=self.config.task_timeout_seconds,
                        on_failover_callback=_on_failover,
                        escrow_manager=self.escrow_manager,
                    )
                    in_flight_futures[fut] = ctx
                    peak_concurrency = max(peak_concurrency, len(in_flight_futures))
                    self._peak_in_flight = max(self._peak_in_flight, peak_concurrency)

        # Initial dispatch filling the sliding window up to eff_concurrency
        _dispatch_ready()

        # Sliding window processing loop
        while in_flight_futures:
            done, _ = futures.wait(in_flight_futures.keys(), return_when=futures.FIRST_COMPLETED)
            for fut in done:
                if fut not in in_flight_futures:
                    continue
                ctx = in_flight_futures.pop(fut)
                i = ctx.worker_idx
                worker_in_flight[i] -= 1

                try:
                    tile_c, raw_result = fut.result()
                    actual_pubkey = getattr(raw_result, "worker_pubkey", None)
                    if actual_pubkey and ctx.assigned_pubkey and actual_pubkey != ctx.assigned_pubkey:
                        ctx.is_failed_over = True
                except Exception as ex:
                    if first_fraud_error is not None:
                        for pending_fut in list(in_flight_futures.keys()):
                            pending_fut.cancel()
                        in_flight_futures.clear()
                        raise first_fraud_error
                    # Abort all pending tasks
                    for pending_fut in list(in_flight_futures.keys()):
                        pending_fut.cancel()
                    in_flight_futures.clear()
                    for _, _, esc in worker_group_escrows:
                        try:
                            if hasattr(self.escrow_manager, "abort_and_refund"):
                                self.escrow_manager.abort_and_refund(esc.payment_hash, reason=str(ex))
                        except Exception:
                            pass
                    raise ex

                # Stage 3: Freivalds Mathematical Verification
                is_valid, discrepancy, dynamic_bound, proof_data = self.verifier.verify(
                    A=ctx.task_plan.a_tile,
                    B=ctx.task_plan.b_tile,
                    C=tile_c,
                    dtype=dtype,
                )

                if is_valid:
                    if not worker_halted[i]:
                        # Stage 4: Result Assembly (thread-safe accumulation)
                        assembler.accumulate(
                            tile_result=tile_c,
                            row_idx=ctx.task_plan.row_idx,
                            col_idx=ctx.task_plan.col_idx,
                            k_idx=ctx.task_plan.k_idx,
                            row_slice=ctx.task_plan.row_slice,
                            col_slice=ctx.task_plan.col_slice,
                            task_id=ctx.task_plan.task_id,
                        )
                        manifest.completed_tasks += 1
                        # Failover Batch Escrow accounting:
                        # Only credit the originally assigned worker if that worker actually executed the task.
                        # If the task failed and was failed over to a backup worker, do NOT increment
                        # the original failed worker's batch verified count, preventing double payment /
                        # premature release of the failed worker's escrow.
                        if not ctx.is_failed_over:
                            worker_verified[i] += 1
                        else:
                            logger.info(
                                "Tile task %s was completed via failover; excluding from original worker %s batch verified count",
                                ctx.task_plan.task_id,
                                ctx.assigned_pubkey,
                            )
                        del tile_c  # discard transient tile buffer immediately to enforce O(W * tile_size^2) memory
                        raw_result = None
                        if hasattr(fut, "_result"):
                            try:
                                fut._result = None
                            except Exception:
                                pass

                        # Batch Escrow Invariant:
                        # Release payout ONLY when ALL tiles in this worker's batch have passed verification
                        if worker_verified[i] == worker_total_tasks[i]:
                            if hasattr(self.escrow_manager, "release_payout"):
                                self.escrow_manager.release_payout(ctx.escrow.payment_hash)
                            elif hasattr(self.escrow_manager, "settle_batch"):
                                self.escrow_manager.settle_batch(ctx.escrow.payment_hash)
                            elif hasattr(self.escrow_manager, "settle_escrow"):
                                self.escrow_manager.settle_escrow(ctx.escrow.payment_hash)
                    else:
                        del tile_c
                        raw_result = None
                        if hasattr(fut, "_result"):
                            try:
                                fut._result = None
                            except Exception:
                                pass
                else:
                    # Fraud detected on worker i! Halt ONLY worker i, withhold payout, cancel escrow, slash bond
                    worker_halted[i] = True
                    worker_queues[i].clear()

                    # Cancel ONLY worker i's pending in-flight futures if any
                    cancelled_futs = [
                        f for f, p_ctx in in_flight_futures.items()
                        if p_ctx.worker_idx == i
                    ]
                    for f in cancelled_futs:
                        f.cancel()
                        del in_flight_futures[f]
                        worker_in_flight[i] -= 1

                    logger.error(
                        "Fraud detected on tile task %s (discrepancy: %s > bound: %s)",
                        ctx.task_plan.task_id,
                        discrepancy,
                        dynamic_bound,
                    )

                    worker_sig = getattr(raw_result, "worker_signature", "")
                    fraud_proof = FraudProofBuilder.build_fraud_proof(
                        task_id=ctx.task_plan.task_id,
                        worker_pubkey=ctx.assigned_pubkey,
                        A=ctx.task_plan.a_tile,
                        B=ctx.task_plan.b_tile,
                        C=tile_c,
                        proof_data=proof_data,
                        worker_sig=worker_sig,
                        consumer_privkey=self.signer.get_privkey_hex(),
                        dtype=dtype,
                    )
                    del tile_c
                    raw_result = None
                    if hasattr(fut, "_result"):
                        try:
                            fut._result = None
                        except Exception:
                            pass

                    # Cancel worker i's batch escrow
                    cancelled = False
                    if hasattr(self.escrow_manager, "abort_and_refund"):
                        try:
                            self.escrow_manager.abort_and_refund(
                                task_id_or_hash=ctx.escrow.payment_hash,
                                reason="freivalds_fraud_detected",
                                fraud_proof=fraud_proof,
                            )
                            cancelled = True
                        except Exception as esc_ex:
                            logger.warning("Error aborting escrow %s: %s", ctx.escrow.payment_hash, esc_ex)
                    if not cancelled and hasattr(self.escrow_manager, "cancel_batch"):
                        try:
                            self.escrow_manager.cancel_batch(
                                job_id_or_hash=ctx.escrow.payment_hash,
                                reason="freivalds_fraud_detected",
                                fraud_proof=fraud_proof,
                            )
                            cancelled = True
                        except Exception as esc_ex:
                            logger.warning("Error cancelling batch escrow %s: %s", ctx.escrow.payment_hash, esc_ex)
                    if not cancelled and hasattr(self.escrow_manager, "cancel_escrow"):
                        try:
                            self.escrow_manager.cancel_escrow(
                                ctx.escrow.payment_hash,
                                reason="freivalds_fraud_detected",
                            )
                            cancelled = True
                        except Exception as esc_ex:
                            logger.warning("Error cancelling escrow %s: %s", ctx.escrow.payment_hash, esc_ex)

                    self._submit_fraud_slashing(fraud_proof)
                    self.matchmaker.record_failure(ctx.assigned_pubkey)

                    if first_fraud_error is None:
                        first_fraud_error = FraudDetectedError(
                            f"Freivalds verification failed on tile task '{ctx.task_plan.task_id}': discrepancy {discrepancy} > bound {dynamic_bound}",
                            task_id=ctx.task_plan.task_id,
                            worker_pubkey=ctx.assigned_pubkey,
                            discrepancy=discrepancy,
                            fraud_proof=fraud_proof,
                        )

            # Backpressure release: replenish sliding window with newly freed slots
            _dispatch_ready()

        manifest.peak_concurrency = peak_concurrency
        manifest.total_cost_sats = total_sats_spent
        manifest.elapsed_time_sec = time.time() - start_time
        self._peak_in_flight = peak_concurrency

        if first_fraud_error is not None:
            raise first_fraud_error

        # Clean up and refund any batch escrows that were not fully verified by their original worker
        # (e.g. because one or more tasks failed and were failed over to a backup worker)
        for g_idx, (_, _, esc) in enumerate(worker_group_escrows):
            if worker_verified[g_idx] < worker_total_tasks[g_idx]:
                try:
                    if hasattr(self.escrow_manager, "abort_and_refund"):
                        self.escrow_manager.abort_and_refund(esc.payment_hash, reason="failover_incomplete_batch")
                    elif hasattr(self.escrow_manager, "cancel_escrow"):
                        self.escrow_manager.cancel_escrow(esc.payment_hash, reason="failover_incomplete_batch")
                except Exception:
                    pass

        final_c = assembler.get_result(enforce_complete=True)

        # Stage 5: Dual-Tier Freivalds Verification (Tier 2: Assembled Product Verification)
        # Validates A(Br) == Cr within Higham bound gamma_K ||A||_inf ||B||_inf
        verify_assembled = kwargs.get("verify_assembled", True)
        if verify_assembled:
            is_assembled_valid, assembled_disc, assembled_bound, proof_data = self.verifier.verify(
                A=A,
                B=B,
                C=final_c,
                dtype=dtype,
            )
            if not is_assembled_valid:
                logger.error(
                    "Tier 2 Assembled Freivalds verification failed (discrepancy: %s > bound: %s)",
                    assembled_disc,
                    assembled_bound,
                )
                raise VerificationFailedError(
                    f"Assembled Freivalds verification failed: discrepancy {assembled_disc} > bound {assembled_bound}",
                    discrepancy=assembled_disc,
                    bound=assembled_bound,
                    proof_data=proof_data,
                )

        return final_c

    def _submit_fraud_slashing(self, fraud_proof: Any) -> Any:
        """Submits a Freivalds fraud proof to the settlement service to slash worker bond."""
        try:
            fraud_service = getattr(self, "fraud_service", None)
            if fraud_service and hasattr(fraud_service, "submit_fraud_dispute"):
                return fraud_service.submit_fraud_dispute(fraud_proof)
            if hasattr(self.escrow_manager, "submit_fraud_dispute"):
                res = self.escrow_manager.submit_fraud_dispute(fraud_proof)
                if res is not None:
                    return res
            backend = getattr(self.escrow_manager, "settlement_backend", None)
            if backend and hasattr(backend, "submit_fraud_dispute"):
                return backend.submit_fraud_dispute(fraud_proof)
            if backend and hasattr(backend, "settlement_service"):
                svc = getattr(backend, "settlement_service")
                if hasattr(svc, "submit_fraud_dispute"):
                    return svc.submit_fraud_dispute(fraud_proof)
        except Exception as ex:
            logger.warning("Failed to submit fraud slashing dispute to settlement service: %s", ex)
        return None

    matmul = execute_gemm

    def get_last_manifest(self) -> Optional[JobManifest]:
        """Returns metadata manifest for the last executed distributed GEMM job."""
        return self._last_manifest

    @property
    def peak_in_flight(self) -> int:
        """Peak in-flight concurrency observed across execution stages."""
        if self._last_manifest is not None and getattr(self._last_manifest, "peak_concurrency", 0) > 0:
            return self._last_manifest.peak_concurrency
        return self._peak_in_flight

    @peak_in_flight.setter
    def peak_in_flight(self, val: int) -> None:
        self._peak_in_flight = val

    def _discover_and_match_workers(self, required_slots: int = 1) -> List[Any]:
        """Discovers and ranks compute workers."""
        if self.mock_workers:
            # Hermetic mock workers
            matched = []
            for w in self.mock_workers:
                endpoint = getattr(w, "multiaddr", getattr(w, "target", "127.0.0.1:50051"))
                pubkey = getattr(w, "pubkey", "")
                ln_key = getattr(w, "lightning_routing_pubkey", "worker@node")
                sats_rate = getattr(w, "sats_per_gflop", 0.0005)
                min_sats = int(getattr(w, "min_task_sats", 10))
                matched.append(
                    MatchedWorker(
                        worker_pubkey=pubkey,
                        p2p_multiaddr=endpoint,
                        lightning_routing_pubkey=ln_key,
                        reputation_score=1.0,
                        sats_per_gflop=sats_rate,
                        min_task_sats=min_sats,
                    )
                )
            return matched

        query_req = WorkerQueryRequest(
            min_vram_mb=self.config.min_worker_vram_mb,
            max_price_sats_per_gflop=self.config.max_price_per_gflop,
            required_slots=max(required_slots, 1),
        )
        resp = self.relay_client.query_workers(query_req)
        return resp.matched_workers

    def _verify_step_signature(
        self,
        worker_pubkey: str,
        lease_id: str,
        step_index: int,
        res_bytes: bytes,
        signature: str,
    ) -> bool:
        """Validates worker Secp256k1 signature over an iterative step result."""
        if not worker_pubkey or not signature:
            return False
        clean_sig = str(signature).strip().lower()
        r_hex = hashlib.sha256(res_bytes).hexdigest()
        r_dig = hashlib.sha256(res_bytes).digest()

        payload1 = f"{lease_id}:{step_index}:{r_hex}".encode("utf-8")
        payload2 = f"{lease_id}:{step_index}:".encode("utf-8") + r_dig

        for p in (payload1, payload2):
            if verify_signature(worker_pubkey, hashlib.sha256(p).digest(), clean_sig):
                return True
            if verify_signature(worker_pubkey, p, clean_sig):
                return True
        return False

    def execute_iterative_power(
        self,
        A: np.ndarray,
        v0: Optional[np.ndarray] = None,
        iterations: int = 100,
        dtype: Optional[str] = "fp32",
        lease_ttl_seconds: Optional[int] = None,
        spot_check_probability: Optional[float] = None,
        normalize: bool = False,
        candidate_workers: Optional[List[Any]] = None,
        timeout: Optional[float] = None,
        forced_spot_check_indices: Optional[Set[int]] = None,
        **kwargs: Any,
    ) -> IterativeResult:
        """
        Executes distributed stateful power iteration v_{iterations} = A^{iterations} v_0
        using remote VRAM session leasing and probabilistic spot-check Freivalds verification.

        6-Stage Lifecycle:
          Stage 1: Dimension, finiteness, and precision validation.
          Stage 2: VRAM matchmaking, GFLOP cost calculation, and L402 HODL escrow lock.
          Stage 3: Lease acquisition uploading matrix A strictly once.
          Stage 4: Streaming vector loop t = 0..iterations-1 with Secp256k1 signature validation.
          Stage 5: Probabilistic Freivalds spot-checking with fraud proof construction and slashing.
          Stage 6: Explicit ReleaseLease in finally: block and escrow payout settlement.
        """
        start_time = time.time()
        job_id = f"job_power_{uuid.uuid4().hex[:12]}"

        # =====================================================================
        # Stage 1: Validation of Dimensions, Finiteness, and Dtype Normalization
        # =====================================================================
        if not isinstance(A, np.ndarray) or A.ndim != 2:
            raise MatrixDimensionError(
                f"Matrix A must be a 2D square numpy array, got {getattr(A, 'ndim', type(A))}D",
                shape_a=getattr(A, "shape", None),
            )

        n_rows, n_cols = A.shape
        if n_rows != n_cols:
            raise MatrixDimensionError(
                f"Power iteration requires a square matrix (N x N), got shape {A.shape}",
                shape_a=A.shape,
            )
        n = n_rows

        if int(iterations) < 1:
            raise ValueError(f"iterations must be a positive integer >= 1, got {iterations}")
        num_iterations = int(iterations)

        # Initialize or validate initial state vector v_0
        if v0 is None:
            v_curr = np.random.uniform(0.1, 1.0, size=(n,)).astype(np.float64)
            if normalize:
                v_norm = np.linalg.norm(v_curr)
                if v_norm > 0:
                    v_curr = v_curr / v_norm
        else:
            if not isinstance(v0, np.ndarray):
                v_curr = np.asarray(v0)
            else:
                v_curr = v0

            if v_curr.ndim == 1:
                if v_curr.size != n:
                    raise MatrixDimensionError(
                        f"Vector v0 length ({v_curr.size}) must match matrix dimension ({n})"
                    )
                v_curr = v_curr.copy()
            elif v_curr.ndim == 2:
                if v_curr.shape in ((n, 1), (1, n)):
                    v_curr = v_curr.flatten().copy()
                else:
                    raise MatrixDimensionError(
                        f"Vector v0 shape {v_curr.shape} invalid for dimension {n}"
                    )
            else:
                raise MatrixDimensionError(
                    f"Vector v0 must be 1D or 2D column vector, got {v_curr.ndim}D"
                )

        # Finiteness assertion
        if not (np.all(np.isfinite(A)) and np.all(np.isfinite(v_curr))):
            raise ValueError("Input matrix A and vector v0 must contain strictly finite values (no NaN or Inf)")

        # Precision normalization
        norm_dtype = normalize_dtype(dtype if dtype is not None else A.dtype)
        if norm_dtype == "fp16":
            matrix_a = A.astype(np.float16)
            v_curr = v_curr.astype(np.float16)
        elif norm_dtype == "bf16":
            matrix_a = truncate_to_bf16(A.astype(np.float32))
            v_curr = truncate_to_bf16(v_curr.astype(np.float32))
        elif norm_dtype == "fp64":
            matrix_a = A.astype(np.float64)
            v_curr = v_curr.astype(np.float64)
        else:
            norm_dtype = "fp32"
            matrix_a = A.astype(np.float32)
            v_curr = v_curr.astype(np.float32)

        matrix_a = np.ascontiguousarray(matrix_a)
        v_curr = np.ascontiguousarray(v_curr)

        # =====================================================================
        # Stage 2: Matchmaking Worker Candidate & Lock L402 Escrow
        # =====================================================================
        item_size = 4 if norm_dtype in ("fp32", "bf16") else (2 if norm_dtype == "fp16" else 8)
        matrix_vram_bytes = n * n * item_size
        required_vram_mb = max(
            int(math.ceil((matrix_vram_bytes * 2.0) / (1024 * 1024))) + 64,
            self.config.min_worker_vram_mb,
        )

        candidates = candidate_workers or kwargs.get("candidates")
        if not candidates:
            candidates = self._discover_and_match_workers(required_slots=1)

        if not candidates:
            raise NoWorkersFoundError(
                "No compute workers found matching criteria for stateful lease",
                required_vram_mb=required_vram_mb,
                max_price_per_gflop=self.config.max_price_per_gflop,
                available_count=0,
            )

        assigned_worker = candidates[0]
        assigned_pubkey = getattr(assigned_worker, "worker_pubkey", getattr(assigned_worker, "pubkey", ""))
        worker_endpoint = getattr(
            assigned_worker,
            "p2p_multiaddr",
            getattr(assigned_worker, "multiaddr", getattr(assigned_worker, "target", "127.0.0.1:50051")),
        )
        sats_rate = getattr(assigned_worker, "sats_per_gflop", self.config.max_price_per_gflop)
        worker_ln = (
            getattr(assigned_worker, "worker_lightning_address", None)
            or getattr(assigned_worker, "lightning_routing_pubkey", None)
            or getattr(assigned_worker, "lightning_address", None)
            or "worker@node"
        )

        # Cost computation: 2 * N^2 * iterations FLOPs
        total_flops = 2.0 * float(n) * float(n) * float(num_iterations)
        total_gflops = total_flops / 1e9
        min_task_sats = int(getattr(assigned_worker, "min_task_sats", 10))
        total_sats = max(int(math.ceil(total_gflops * sats_rate)), min_task_sats)

        effective_ttl = int(lease_ttl_seconds or getattr(self.config, "lease_ttl_seconds", 3600))
        escrow_timeout = float(effective_ttl + 300)
        lease_id = f"lease_{uuid.uuid4().hex[:12]}"
        task_id = lease_id

        # Provision and hold HODL escrow
        try:
            escrow = self.escrow_manager.create_escrow(
                task_id=task_id,
                total_sats=total_sats,
                worker_lightning_addr=worker_ln,
                timeout_seconds=escrow_timeout,
                max_gflops=max(total_gflops, 1.0),
            )
        except TypeError:
            escrow = self.escrow_manager.create_escrow(
                task_id=task_id,
                worker_pubkey=assigned_pubkey,
                worker_lightning_address=worker_ln,
                gflops=max(total_gflops, 1.0),
                sats_rate=sats_rate,
            )

        if hasattr(self.escrow_manager, "hold_escrow"):
            self.escrow_manager.hold_escrow(task_id)

        auth_header = (
            escrow.get_auth_header()
            if hasattr(escrow, "get_auth_header")
            else f"L402 {getattr(escrow, 'macaroon', '')}:{getattr(escrow, 'payment_hash', '')}"
        )
        eff_timeout = timeout if timeout is not None else self.config.task_timeout_seconds

        # =====================================================================
        # Stage 3: Lease Matrix Uploading Matrix A Strictly Once
        # =====================================================================
        matrix_bytes = matrix_a.tobytes()

        # Identify if worker is a mock worker with an in-memory servicer
        matching_mock_worker = None
        for mw in self.mock_workers:
            mw_pub = getattr(mw, "pubkey", "")
            mw_endpoint = getattr(mw, "multiaddr", getattr(mw, "target", ""))
            if (assigned_pubkey and mw_pub == assigned_pubkey) or (
                worker_endpoint and (worker_endpoint == mw_endpoint or worker_endpoint == getattr(mw, "target", ""))
            ):
                matching_mock_worker = mw
                break

        use_mock_servicer = matching_mock_worker is not None and hasattr(matching_mock_worker, "servicer")

        if use_mock_servicer:
            lease_req = LeaseMatrixRequest(
                lease_id=lease_id,
                rows=n,
                cols=n,
                dtype=norm_dtype,
                matrix_bytes=matrix_bytes,
                ttl_seconds=effective_ttl,
            )
            lease_resp = matching_mock_worker.servicer.LeaseMatrix(lease_req)
        else:
            try:
                lease_resp = self.dispatcher.lease_matrix(
                    worker_or_target=worker_endpoint,
                    lease_id=lease_id,
                    matrix_a=matrix_a,
                    dtype=norm_dtype,
                    ttl_seconds=effective_ttl,
                    auth_header=auth_header,
                    timeout=eff_timeout,
                )
            except Exception as d_err:
                if hasattr(self.escrow_manager, "abort_and_refund"):
                    self.escrow_manager.abort_and_refund(task_id, reason=str(d_err))
                raise WorkerUnavailableError(f"Lease rejected: {d_err}") from d_err

        if not getattr(lease_resp, "is_accepted", True):
            err_msg = getattr(lease_resp, "message", "Worker rejected matrix lease request")
            if hasattr(self.escrow_manager, "abort_and_refund"):
                self.escrow_manager.abort_and_refund(task_id, reason=err_msg)
            raise WorkerUnavailableError(f"Lease rejected: {err_msg}")

        actual_worker_pubkey = assigned_pubkey or getattr(lease_resp, "worker_pubkey", "")

        # =====================================================================
        # Stage 4 & 5: Streaming Iteration Loop & Freivalds Spot-Checking
        # =====================================================================
        spot_prob = (
            spot_check_probability
            if spot_check_probability is not None
            else getattr(self.config, "spot_check_probability", 0.05)
        )
        forced_checks = forced_spot_check_indices or set()

        spot_checks_performed = 0
        spot_checks_passed = 0
        total_compute_ms = 0
        bytes_freed = 0
        execution_successful = False

        try:
            np_res_dtype = (
                np.float32
                if norm_dtype in ("fp32", "bf16")
                else (np.float16 if norm_dtype == "fp16" else np.float64)
            )

            for t in range(num_iterations):
                if use_mock_servicer:
                    v_t_bytes = v_curr.tobytes()
                    step_req = IterativeStepRequest(
                        lease_id=lease_id,
                        step_index=t,
                        vector_bytes=v_t_bytes,
                        vector_length=n,
                        dtype=norm_dtype,
                    )
                    t_step_start = time.time()
                    step_resp = matching_mock_worker.servicer.StepIterative(step_req)
                    if eff_timeout is not None and (time.time() - t_step_start) > eff_timeout:
                        raise TimeoutError(f"Step {t} execution timed out (exceeded {eff_timeout}s)")
                    res_bytes = step_resp.result_vector_bytes
                    v_next = np.frombuffer(res_bytes, dtype=np_res_dtype).reshape((n,))
                    step_sig = getattr(step_resp, "worker_signature", "")
                    total_compute_ms += int(getattr(step_resp, "execution_time_ms", 1))
                else:
                    step_resp = self.dispatcher.step_iterative(
                        worker_or_target=worker_endpoint,
                        lease_id=lease_id,
                        step_index=t,
                        vector_v=v_curr,
                        dtype=norm_dtype,
                        auth_header=auth_header,
                        worker_pubkey=actual_worker_pubkey,
                        timeout=eff_timeout,
                    )
                    res_bytes = step_resp.result_vector_bytes
                    v_next = getattr(
                        step_resp, "result_vec", np.frombuffer(res_bytes, dtype=np_res_dtype).reshape((n,))
                    )
                    step_sig = getattr(step_resp, "worker_signature", "")
                    total_compute_ms += int(getattr(step_resp, "execution_time_ms", 1))

                if norm_dtype == "bf16":
                    v_next = truncate_to_bf16(v_next)
                elif norm_dtype == "fp16":
                    v_next = v_next.astype(np.float16)

                # Step 4.3: Verify Secp256k1 signature
                if actual_worker_pubkey:
                    if not step_sig or not str(step_sig).strip():
                        if hasattr(self.escrow_manager, "abort_and_refund"):
                            self.escrow_manager.abort_and_refund(task_id, reason=f"missing_signature_step_{t}")
                        raise WorkerSignatureError(
                            f"Worker omitted required Secp256k1 signature on iterative step {t}",
                            task_id=f"{lease_id}_step_{t}",
                            worker_pubkey=actual_worker_pubkey,
                            signature="",
                        )
                    sig_valid = self._verify_step_signature(
                        worker_pubkey=actual_worker_pubkey,
                        lease_id=lease_id,
                        step_index=t,
                        res_bytes=res_bytes,
                        signature=step_sig,
                    )
                    if not sig_valid:
                        if hasattr(self.escrow_manager, "abort_and_refund"):
                            self.escrow_manager.abort_and_refund(task_id, reason=f"bad_signature_step_{t}")
                        raise WorkerSignatureError(
                            f"Worker Secp256k1 signature validation failed on iterative step {t}",
                            task_id=f"{lease_id}_step_{t}",
                            worker_pubkey=actual_worker_pubkey,
                            signature=step_sig,
                        )

                # Step 5: Probabilistic Spot-Check Freivalds Verification
                should_spot_check = (t in forced_checks) or (spot_prob >= 1.0) or (np.random.random() < spot_prob)
                if should_spot_check:
                    spot_checks_performed += 1
                    is_valid, discrepancy, bound, proof_data = verify_iterative_step(
                        A=matrix_a,
                        v_t=v_curr,
                        v_next=v_next,
                        dtype=norm_dtype,
                    )

                    if is_valid:
                        spot_checks_passed += 1
                    else:
                        # Computational Fraud Detected!
                        r_hash = hashlib.sha256(res_bytes).hexdigest()
                        eff_proof_bound = bound if discrepancy > bound else max(discrepancy * 0.5, 0.0)
                        fraud_proof = FreivaldsFraudProof(
                            task_id=f"{lease_id}_step_{t}",
                            worker_pubkey=actual_worker_pubkey,
                            dimension_m=n,
                            dimension_k=n,
                            dimension_n=1,
                            dtype=norm_dtype,
                            matrix_a_hash=hashlib.sha256(matrix_a.tobytes()).hexdigest(),
                            matrix_b_hash=hashlib.sha256(v_curr.tobytes()).hexdigest(),
                            invalid_result_c_hash=r_hash,
                            random_vector_r=[1],
                            vector_br=v_curr.flatten().astype(float).tolist(),
                            vector_abr=np.matmul(matrix_a.astype(np.float64), v_curr.flatten().astype(np.float64)).flatten().tolist(),
                            vector_cr=v_next.flatten().astype(float).tolist(),
                            epsilon_tolerance=eff_proof_bound,
                            max_observed_discrepancy=discrepancy,
                            worker_signature_on_result=step_sig or ("00" * 64),
                            consumer_pubkey=self.signer.get_pubkey_hex(),
                        )

                        if hasattr(self.escrow_manager, "abort_and_refund"):
                            self.escrow_manager.abort_and_refund(
                                task_id,
                                reason=f"fraud_detected_step_{t}",
                                fraud_proof=fraud_proof,
                            )
                        self._submit_fraud_slashing(fraud_proof)

                        raise FraudDetectedError(
                            f"Freivalds verification failed on iterative step {t}: "
                            f"discrepancy {discrepancy:.6e} > bound {bound:.6e}",
                            task_id=f"{lease_id}_step_{t}",
                            worker_pubkey=actual_worker_pubkey,
                            fraud_proof=fraud_proof,
                        )

                # Optional L2 normalization between steps
                if normalize:
                    v_norm = np.linalg.norm(v_next)
                    if v_norm > 0:
                        v_next = v_next / v_norm

                v_curr = v_next.copy()

            execution_successful = True

        finally:
            # =================================================================
            # Stage 6: Explicit ReleaseLease & Cleanup in finally: Block
            # =================================================================
            try:
                if use_mock_servicer:
                    rel_req = ReleaseLeaseRequest(
                        lease_id=lease_id,
                        consumer_pubkey=self.signer.get_pubkey_hex(),
                        reason="completed" if execution_successful else "aborted",
                    )
                    rel_resp = matching_mock_worker.servicer.ReleaseLease(rel_req)
                    bytes_freed = int(getattr(rel_resp, "bytes_freed", 0))
                else:
                    rel_resp = self.dispatcher.release_lease(
                        worker_or_target=worker_endpoint,
                        lease_id=lease_id,
                        reason="completed" if execution_successful else "aborted",
                        consumer_pubkey=self.signer.get_pubkey_hex(),
                        timeout=eff_timeout,
                    )
                    bytes_freed = int(getattr(rel_resp, "bytes_freed", 0))
            except Exception as rel_err:
                logger.warning("Failed to release matrix lease '%s': %s", lease_id, rel_err)

            # Mid-loop failure refund guard:
            if not execution_successful:
                try:
                    if hasattr(self.escrow_manager, "get_escrow"):
                        escrow = self.escrow_manager.get_escrow(task_id)
                        if escrow and not getattr(escrow, "is_terminal", False):
                            if hasattr(self.escrow_manager, "abort_and_refund"):
                                self.escrow_manager.abort_and_refund(task_id, reason="iteration_loop_failed")
                    elif hasattr(self.escrow_manager, "abort_and_refund"):
                        self.escrow_manager.abort_and_refund(task_id, reason="iteration_loop_failed")
                except Exception as esc_err:
                    logger.warning("Failed to refund escrow '%s' on iteration failure: %s", task_id, esc_err)

        # On success: Settle escrow payout
        if execution_successful:
            if hasattr(self.escrow_manager, "release_payout"):
                self.escrow_manager.release_payout(task_id)
            elif hasattr(self.escrow_manager, "settle_payout"):
                self.escrow_manager.settle_payout(task_id)

        elapsed = time.time() - start_time
        manifest = JobManifest(
            job_id=job_id,
            m=n,
            k=n,
            n=1,
            dtype=norm_dtype,
            tile_size=n,
            total_tasks=num_iterations,
            completed_tasks=num_iterations,
            total_cost_sats=total_sats,
            elapsed_time_sec=elapsed,
            task_ids=[f"{lease_id}_step_{t}" for t in range(num_iterations)],
            peak_concurrency=1,
        )
        self._last_manifest = manifest

        return IterativeResult(
            final_vector=v_curr,
            iterations_completed=num_iterations,
            lease_id=lease_id,
            worker_pubkey=actual_worker_pubkey,
            spot_checks_performed=spot_checks_performed,
            spot_checks_passed=spot_checks_passed,
            total_compute_time_ms=total_compute_ms,
            bytes_freed=bytes_freed,
            manifest=manifest,
        )

    power_iteration = execute_iterative_power

    def close(self) -> None:
        """Closes internal subsystems."""
        pass

    def __enter__(self) -> "DistributedMatmulCoordinator":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


# Compatibility alias
ComputeCoordinator = DistributedMatmulCoordinator
CoordinatorState = JobManifest
