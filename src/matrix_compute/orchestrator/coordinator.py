"""
Distributed Matrix Multiplication Coordinator for matrix-compute-sdk.

Coordinates the complete end-to-end distributed GEMM lifecycle:
matrix partitioning -> worker discovery & matchmaking -> L402 escrow creation ->
gRPC task dispatch -> dynamic Freivalds mathematical verification ->
payout release or fraud slashing -> result accumulation & assembly.
"""

from __future__ import annotations

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

from matrix_compute.config import ClientConfig
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
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Executes distributed matrix multiplication C = A x B.
        """
        start_time = time.time()
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        m, k = A.shape
        _, n = B.shape

        # 1. Tiling plan
        grid_plan = MatrixTiler.create_plan(
            a=A,
            b=B,
            tile_size=tile_size,
            dtype=dtype,
            task_id_prefix=job_id,
        )

        manifest = JobManifest(
            job_id=job_id,
            m=m,
            k=k,
            n=n,
            dtype=dtype,
            tile_size=tile_size,
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
        assembler = ResultAssembler.from_plan(grid_plan)

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

        # 5. Dispatch tiles and verify
        for assigned_worker, tasks, escrow in worker_group_escrows:
            assigned_pubkey = getattr(assigned_worker, "worker_pubkey", getattr(assigned_worker, "pubkey", ""))
            candidates = [assigned_worker] + [
                w for w in workers
                if getattr(w, "worker_pubkey", getattr(w, "pubkey", "")) != assigned_pubkey
            ]

            for task_plan in tasks:
                tile_c, raw_result = self.scheduler.execute_with_failover(
                    task_plan=task_plan,
                    candidate_workers=candidates,
                    macaroon=escrow.macaroon,
                    payment_hash=escrow.payment_hash,
                    dtype=dtype,
                )

                # Freivalds Mathematical Verification
                is_valid, discrepancy, dynamic_bound, proof_data = self.verifier.verify(
                    A=task_plan.a_tile,
                    B=task_plan.b_tile,
                    C=tile_c,
                    dtype=dtype,
                )

                if is_valid:
                    assembler.accumulate(
                        tile_result=tile_c,
                        row_idx=task_plan.row_idx,
                        col_idx=task_plan.col_idx,
                        k_idx=task_plan.k_idx,
                        row_slice=task_plan.row_slice,
                        col_slice=task_plan.col_slice,
                        task_id=task_plan.task_id,
                    )
                    manifest.completed_tasks += 1
                else:
                    # Fraud detected! Halt batch, cancel escrow, submit dispute, and raise error
                    logger.error(
                        "Fraud detected on tile task %s (discrepancy: %s > bound: %s)",
                        task_plan.task_id,
                        discrepancy,
                        dynamic_bound,
                    )

                    worker_sig = getattr(raw_result, "worker_signature", "")
                    fraud_proof = FraudProofBuilder.build_fraud_proof(
                        task_id=task_plan.task_id,
                        worker_pubkey=assigned_pubkey,
                        A=task_plan.a_tile,
                        B=task_plan.b_tile,
                        C=tile_c,
                        proof_data=proof_data,
                        worker_sig=worker_sig,
                        consumer_privkey=self.signer.get_privkey_hex(),
                        dtype=dtype,
                    )

                    self.escrow_manager.abort_and_refund(
                        task_id_or_hash=escrow.payment_hash,
                        reason="freivalds_fraud_detected",
                        fraud_proof=fraud_proof,
                    )

                    self._submit_fraud_slashing(fraud_proof)
                    self.matchmaker.record_failure(assigned_pubkey)

                    raise FraudDetectedError(
                        f"Freivalds verification failed on tile task '{task_plan.task_id}': discrepancy {discrepancy} > bound {dynamic_bound}",
                        task_id=task_plan.task_id,
                        worker_pubkey=assigned_pubkey,
                        discrepancy=discrepancy,
                        fraud_proof=fraud_proof,
                    )

            # All tiles for this worker passed: release batch payout atomically
            if hasattr(self.escrow_manager, "release_payout"):
                self.escrow_manager.release_payout(escrow.payment_hash)
            elif hasattr(self.escrow_manager, "settle_batch"):
                self.escrow_manager.settle_batch(escrow.payment_hash)
            elif hasattr(self.escrow_manager, "settle_escrow"):
                self.escrow_manager.settle_escrow(escrow.payment_hash)

        manifest.total_cost_sats = total_sats_spent
        manifest.elapsed_time_sec = time.time() - start_time
        return assembler.get_result(enforce_complete=True)

    def _submit_fraud_slashing(self, fraud_proof: Any) -> Any:
        """Submits a Freivalds fraud proof to the settlement service to slash worker bond."""
        try:
            if hasattr(self.escrow_manager, "submit_fraud_dispute"):
                res = self.escrow_manager.submit_fraud_dispute(fraud_proof)
                if res is not None:
                    return res
            backend = getattr(self.escrow_manager, "settlement_backend", None)
            if backend and hasattr(backend, "submit_fraud_dispute"):
                return backend.submit_fraud_dispute(fraud_proof)
        except Exception as ex:
            logger.warning("Failed to submit fraud slashing dispute to settlement service: %s", ex)
        return None

    matmul = execute_gemm

    def get_last_manifest(self) -> Optional[JobManifest]:
        """Returns metadata manifest for the last executed distributed GEMM job."""
        return self._last_manifest

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
