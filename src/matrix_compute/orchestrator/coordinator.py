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
import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple, Union
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

from matrix_compute.config import ClientConfig, PartitioningStrategy
from matrix_compute.crypto.signer import ClientSigner
from matrix_compute.discovery.matchmaker import Matchmaker
from matrix_compute.discovery.relay_client import DiscoveredWorker, RelayClient
from matrix_compute.exceptions import (
    FraudDetectedError,
    NoWorkersFoundError,
    VerificationFailedError,
    WorkerUnavailableError,
)
from matrix_compute.orchestrator.scheduler import ParallelTaskScheduler
from matrix_compute.p2p.dispatcher import GrpcTaskDispatcher
from matrix_compute.payment.escrow import EscrowManager, EscrowStatus
from matrix_compute.test_doubles.mock_worker import MockWorkerNode
from matrix_compute.tiling.assembler import ResultAssembler
from matrix_compute.tiling.partitioner import MatrixTiler, TileGridPlan, TileTaskPlan
from matrix_compute.verification.fraud_proof import FraudProofBuilder
from matrix_compute.verification.freivalds import FreivaldsVerifier

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
            total_sats = sum(max(int(math.ceil(t.estimated_gflops * sats_per_gflop)), 10) for t in tasks)
            total_gflops = sum(t.estimated_gflops for t in tasks)
            total_sats_spent += total_sats

            worker_ln = getattr(
                assigned_worker,
                "lightning_routing_pubkey",
                getattr(assigned_worker, "lightning_address", "worker@node"),
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
                matched.append(
                    MatchedWorker(
                        worker_pubkey=pubkey,
                        p2p_multiaddr=endpoint,
                        lightning_routing_pubkey=ln_key,
                        reputation_score=1.0,
                        sats_per_gflop=sats_rate,
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
