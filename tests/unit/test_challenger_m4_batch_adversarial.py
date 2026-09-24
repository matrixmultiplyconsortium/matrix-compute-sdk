"""
Empirical Adversarial Challenge Test Suite for Milestone 4:
Coordinator Batch Scheduling, Multi-Task Macaroon Containment & Fraud Slashing Defense.

Author: teamwork_preview_challenger_m4_2
Role: EMPIRICAL CHALLENGER (critic, specialist)
Interface Contracts: ORIGINAL_REQUEST.md (## 2026-09-23T19:21:28Z), planned_work/01_batch_settlement_l402.md
"""

from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import hmac
import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import grpc
import numpy as np
import pytest

from matrix_compute.config import ClientConfig
from matrix_compute.crypto.signer import ClientSigner
from matrix_compute.discovery.matchmaker import Matchmaker
from matrix_compute.discovery.relay_client import RelayClient
from matrix_compute.exceptions import (
    EscrowCreationError,
    EscrowStateError,
    FraudDetectedError,
    MacaroonError,
    MatrixDimensionError,
)
from matrix_compute.orchestrator.coordinator import DistributedMatmulCoordinator
from matrix_compute.orchestrator.scheduler import ParallelTaskScheduler
from matrix_compute.p2p.dispatcher import GrpcTaskDispatcher
from matrix_compute.payment.backend import MockSettlementBackend
from matrix_compute.payment.escrow import (
    EscrowManager,
    EscrowState,
    EscrowStatus,
    generate_escrow_secret,
)
from matrix_compute.payment.l402_client import MacaroonCaveats, MacaroonToken
from matrix_compute.test_doubles.mock_relay import MockNostrRelay
from matrix_compute.test_doubles.mock_settlement import MockSettlementService
from matrix_compute.test_doubles.mock_worker import MockWorkerNode, MockWorkerServicer
from matrix_contracts.schemas.settlement import CreateHodlInvoiceRequest


# ==============================================================================
# Helper Functions
# ==============================================================================

def mint_test_macaroon(
    verification_key: str,
    task_id: str,
    task_ids: Optional[List[str]] = None,
    job_id: Optional[str] = None,
    worker_pubkey: Optional[str] = None,
    payment_hash: str = "a" * 64,
    expiry_seconds: int = 3600,
    max_gflops: float = 100.0,
    locked_sats: int = 500,
) -> str:
    """Mints a canonical L402 Macaroon token for testing."""
    now = time.time()
    caveats = {
        "expiry": int(now + expiry_seconds),
        "locked_sats": locked_sats,
        "max_gflops": float(max_gflops),
        "payment_hash": payment_hash,
        "task_id": task_id,
    }
    if task_ids is not None:
        caveats["task_ids"] = list(task_ids)
    if job_id is not None:
        caveats["job_id"] = str(job_id)
    if worker_pubkey is not None:
        caveats["worker_pubkey"] = str(worker_pubkey)

    caveat_str = json.dumps(caveats, sort_keys=True)
    sig = hmac.new(
        verification_key.encode("utf-8"),
        caveat_str.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    payload = {"caveats": caveats, "signature": sig}
    return base64.b64encode(json.dumps(payload).encode("utf-8")).decode("utf-8")


# ==============================================================================
# 1. Multi-Task Macaroon Containment & Tampering Tests
# ==============================================================================

class TestMacaroonBatchContainmentAdversarial:
    """
    Challenge Dimension 1: Multi-Task Macaroon Containment & Tampering Resistance.
    Verifies that tokens with M tasks authorize any constituent task, but reject
    unauthorized tasks, substring attacks, tampered caveats, and forged signatures.
    """

    VK = "secret-challenger-vk-98765"

    def test_macaroon_authorizes_all_constituent_tasks(self):
        """M-task macaroon validates successfully for all constituent task IDs."""
        tasks = [f"tile_{i}_{j}" for i in range(4) for j in range(4)]  # 16 tasks
        raw_b64 = mint_test_macaroon(
            self.VK, task_id=tasks[0], task_ids=tasks, job_id="job_batch_16"
        )
        token = MacaroonToken.deserialize(raw_b64)
        assert token.verify_hmac(self.VK) is True

        for tid in tasks:
            is_valid, err = token.validate(task_id=tid, m=16, k=16, n=16)
            assert is_valid is True, f"Constituent task '{tid}' failed validation: {err}"

    def test_macaroon_rejects_unauthorized_task_ids(self):
        """M-task macaroon strictly rejects task IDs not in the constituent batch."""
        tasks = ["tile_0_0", "tile_0_1", "tile_1_0", "tile_1_1"]
        raw_b64 = mint_test_macaroon(self.VK, task_id=tasks[0], task_ids=tasks)
        token = MacaroonToken.deserialize(raw_b64)

        unauthorized = ["tile_2_2", "tile_9_9", "evil_task", "tile_0_2"]
        for unauth_id in unauthorized:
            is_valid, err = token.validate(task_id=unauth_id, m=16, k=16, n=16)
            assert is_valid is False
            assert "not authorized in batch" in err or "mismatch" in err

    def test_macaroon_rejects_substring_and_prefix_attacks(self):
        """Substrings or prefixes of authorized task IDs are NOT authorized."""
        tasks = ["task_10", "task_20", "task_300"]
        raw_b64 = mint_test_macaroon(self.VK, task_id=tasks[0], task_ids=tasks)
        token = MacaroonToken.deserialize(raw_b64)

        sub_attacks = ["task_1", "task_2", "task_3", "task_30", "task", "10", "20"]
        for attack_id in sub_attacks:
            is_valid, err = token.validate(task_id=attack_id, m=16, k=16, n=16)
            assert is_valid is False, f"Substring attack succeeded for '{attack_id}'!"

    def test_macaroon_single_task_backward_compatibility(self):
        """Macaroon minted without task_ids list validates only single task_id."""
        raw_b64 = mint_test_macaroon(
            self.VK, task_id="legacy_single_task", task_ids=None
        )
        token = MacaroonToken.deserialize(raw_b64)
        assert token.caveats.task_ids is None

        # Primary task must pass
        is_valid, err = token.validate(task_id="legacy_single_task", m=16, k=16, n=16)
        assert is_valid is True

        # Any other task must fail
        is_valid, err = token.validate(task_id="other_task", m=16, k=16, n=16)
        assert is_valid is False

    def test_macaroon_tampered_task_ids_fails_hmac(self):
        """Modifying task_ids in payload caveats without re-signing fails HMAC check."""
        tasks = ["task_A", "task_B"]
        raw_b64 = mint_test_macaroon(self.VK, task_id="task_A", task_ids=tasks)

        # Deserialize, tamper caveats, re-serialize with original signature
        payload = json.loads(base64.b64decode(raw_b64).decode("utf-8"))
        payload["caveats"]["task_ids"].append("injected_unauthorized_task")
        tampered_b64 = base64.b64encode(json.dumps(payload).encode("utf-8")).decode("utf-8")

        tampered_token = MacaroonToken.deserialize(tampered_b64)
        # HMAC verification must fail
        assert tampered_token.verify_hmac(self.VK) is False

    def test_macaroon_tampered_signature_fails_hmac(self):
        """Forged or flipped HMAC signature fails verification."""
        tasks = ["task_A", "task_B"]
        raw_b64 = mint_test_macaroon(self.VK, task_id="task_A", task_ids=tasks)
        payload = json.loads(base64.b64decode(raw_b64).decode("utf-8"))
        payload["signature"] = "0" * 64
        forged_b64 = base64.b64encode(json.dumps(payload).encode("utf-8")).decode("utf-8")

        token = MacaroonToken.deserialize(forged_b64)
        assert token.verify_hmac(self.VK) is False

    def test_mock_worker_servicer_gate_enforces_batch_containment(self):
        """Worker servicer _verify_l402_gate accepts constituent tasks and rejects outsiders."""
        settlement = MockSettlementService(verification_key=self.VK)
        signer = ClientSigner()
        servicer = MockWorkerServicer(
            signer=signer,
            settlement_service=settlement,
            verification_key=self.VK,
        )

        batch_tasks = ["batch_tile_1", "batch_tile_2", "batch_tile_3"]
        inv_resp = settlement.create_batch_escrow(
            task_ids=batch_tasks,
            total_compute_sats=1000,
        )
        settlement.hold_invoice(inv_resp.payment_hash)

        auth_header = f"L402 {inv_resp.macaroon}:{inv_resp.payment_hash}"

        # Constituent tasks must pass
        for tid in batch_tasks:
            ok, msg, code = servicer._verify_l402_gate(
                auth_header=auth_header, task_id=tid, m=16, k=16, n=16
            )
            assert ok is True, f"Failed for constituent task '{tid}': {msg}"
            assert code == grpc.StatusCode.OK

        # Non-constituent task must be rejected with PERMISSION_DENIED
        ok, msg, code = servicer._verify_l402_gate(
            auth_header=auth_header, task_id="unauthorized_tile_99", m=16, k=16, n=16
        )
        assert ok is False
        assert code == grpc.StatusCode.PERMISSION_DENIED
        assert "not in authorized batch" in msg

    def test_mock_worker_servicer_gate_rejects_unheld_invoice(self):
        """Worker gate rejects UNPAID invoice with FAILED_PRECONDITION."""
        settlement = MockSettlementService(verification_key=self.VK)
        servicer = MockWorkerServicer(
            signer=ClientSigner(),
            settlement_service=settlement,
            verification_key=self.VK,
        )
        batch_tasks = ["unpaid_tile_1"]
        inv_resp = settlement.create_batch_escrow(
            task_ids=batch_tasks,
            total_compute_sats=500,
        )
        # Note: invoice is left in UNPAID state (not held)

        auth_header = f"L402 {inv_resp.macaroon}:{inv_resp.payment_hash}"
        ok, msg, code = servicer._verify_l402_gate(
            auth_header=auth_header, task_id="unpaid_tile_1", m=16, k=16, n=16
        )
        assert ok is False
        assert code == grpc.StatusCode.FAILED_PRECONDITION
        assert "expected 'HELD'" in msg


# ==============================================================================
# 2. Concurrent Batch Coordinator Execution Tests
# ==============================================================================

class TestConcurrentBatchCoordinatorExecution:
    """
    Challenge Dimension 2: Concurrency & Isolation across Batch Executions.
    Verifies that multiple multi-worker batch jobs running simultaneously
    maintain isolated escrows, non-interfering preimages, and exact satoshi conservation.
    """

    VK = "concurrent-batch-vk-54321"

    def test_concurrent_multi_worker_batch_gemm_execution(self):
        """
        4 concurrent threads execute full distributed GEMMs simultaneously across 2 mock workers.
        Verifies:
        1. Zero crosstalk between jobs (unique job IDs, task IDs, and escrows).
        2. All 4 jobs produce mathematically exact results matching np.matmul(A, B).
        3. All batch escrows transition cleanly to SETTLED with valid preimages.
        """
        settlement = MockSettlementService(verification_key=self.VK)
        relay = MockNostrRelay()

        worker1 = MockWorkerNode(
            settlement_service=settlement,
            verification_key=self.VK,
            vram_mb=24576,
            sats_per_gflop=0.0005,
        )
        worker2 = MockWorkerNode(
            settlement_service=settlement,
            verification_key=self.VK,
            vram_mb=24576,
            sats_per_gflop=0.0005,
        )

        try:
            worker1.start()
            worker2.start()
            worker1.publish_to_relay(relay)
            worker2.publish_to_relay(relay)

            config = ClientConfig(
                offline_mock_mode=True,
                nostr_relays=["ws://127.0.0.1:8008"],
                settlement_api_url="http://127.0.0.1:8080",
                settlement_verification_key=self.VK,
                min_worker_vram_mb=8192,
                max_price_per_gflop=0.001,
                default_tile_size=16,
                max_retries=0,
            )

            signer = ClientSigner()
            relay_client = RelayClient(relay_urls=["ws://127.0.0.1:8008"], in_memory_relay=relay)
            matchmaker = Matchmaker()
            backend = MockSettlementBackend(settlement)
            escrow_mgr = EscrowManager(settlement_backend=backend, verification_key=self.VK)
            dispatcher = GrpcTaskDispatcher(consumer_pubkey=signer.get_pubkey_hex())
            scheduler = ParallelTaskScheduler(dispatcher=dispatcher, max_retries=0)

            coordinator = DistributedMatmulCoordinator(
                config=config,
                signer=signer,
                relay_client=relay_client,
                matchmaker=matchmaker,
                escrow_manager=escrow_mgr,
                scheduler=scheduler,
                dispatcher=dispatcher,
                mock_workers=[worker1, worker2],
            )

            # Define 4 distinct matrix pairs for 4 concurrent jobs
            job_inputs = []
            for i in range(4):
                rng = np.random.default_rng(100 + i)
                A = rng.standard_normal((32, 16)).astype(np.float32)
                B = rng.standard_normal((16, 32)).astype(np.float32)
                job_inputs.append((i, A, B))

            def run_job(args):
                job_idx, a_mat, b_mat = args
                res = coordinator.matmul(a_mat, b_mat, dtype="fp32", tile_size=16)
                return job_idx, res, np.matmul(a_mat, b_mat)

            results = {}
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(run_job, inp) for inp in job_inputs]
                for fut in as_completed(futures):
                    j_idx, computed_res, expected_res = fut.result()
                    results[j_idx] = (computed_res, expected_res)

            assert len(results) == 4
            for j_idx, (comp, exp) in results.items():
                np.testing.assert_allclose(comp, exp, rtol=1e-5, atol=1e-5)

            # Verify that all created escrows in EscrowManager settled
            all_escrows = list(escrow_mgr._escrows.values())
            assert len(all_escrows) > 0
            unique_escrows = {e.payment_hash: e for e in all_escrows}

            for phash, esc in unique_escrows.items():
                assert esc.status == EscrowStatus.SETTLED
                assert esc.preimage is not None
                assert settlement.get_status(phash) == "SETTLED"

        finally:
            worker1.stop()
            worker2.stop()

    def test_concurrent_batch_escrow_lifecycle_isolation(self):
        """20 concurrent threads creating, holding, and settling batch escrows have no crosstalk."""
        settlement = MockSettlementService(verification_key=self.VK)
        manager = EscrowManager(settlement_backend=settlement)

        num_threads = 20

        def worker_task(thread_id: int):
            task_ids = [f"th_{thread_id}_tile_{k}" for k in range(5)]
            job_id = f"job_thread_{thread_id}"
            escrow = manager.create_batch_escrow(
                worker_pubkey=f"worker_pubkey_{thread_id}",
                task_ids=task_ids,
                job_id=job_id,
                total_sats=100 + thread_id * 10,
            )
            manager.hold_escrow(escrow.payment_hash)
            # Settle via payment_hash
            settled = manager.settle_batch(escrow.payment_hash)
            return thread_id, escrow.payment_hash, settled.status, settled.preimage

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(worker_task, i) for i in range(num_threads)]
            completed = [f.result() for f in as_completed(futures)]

        assert len(completed) == num_threads
        all_hashes = [item[1] for item in completed]
        assert len(set(all_hashes)) == num_threads  # All payment hashes unique

        for tid, phash, status, preimage in completed:
            assert status == EscrowStatus.SETTLED
            assert preimage is not None
            assert settlement.get_status(phash) == "SETTLED"

    def test_critic_job_index_mapping_discovery(self):
        """
        CRITIC CHALLENGE: Investigates _job_index behavior when multiple workers participate in 1 job.
        In EscrowManager, _job_index is Dict[job_id -> payment_hash].
        If worker 1 and worker 2 are in the same job, _job_index[job_id] is overwritten by worker 2!
        Verifies that settling by payment_hash is safe, but settling by job_id only settles worker 2.
        """
        settlement = MockSettlementService(verification_key=self.VK)
        manager = EscrowManager(settlement_backend=settlement)

        job_id = "shared_job_two_workers"
        escrow_w1 = manager.create_batch_escrow(
            worker_pubkey="worker_1_pk",
            task_ids=["job_t1", "job_t2"],
            job_id=job_id,
            total_sats=200,
        )
        manager.hold_escrow(escrow_w1.payment_hash)

        escrow_w2 = manager.create_batch_escrow(
            worker_pubkey="worker_2_pk",
            task_ids=["job_t3", "job_t4"],
            job_id=job_id,
            total_sats=200,
        )
        manager.hold_escrow(escrow_w2.payment_hash)

        # _job_index[job_id] now points to escrow_w2's payment_hash
        resolved_by_job = manager._resolve_escrow(job_id)
        assert resolved_by_job.payment_hash == escrow_w2.payment_hash

        # Settling by payment_hash correctly settles worker 1
        manager.settle_batch(escrow_w1.payment_hash)
        assert escrow_w1.status == EscrowStatus.SETTLED
        assert escrow_w2.status == EscrowStatus.HELD

        # Settling by job_id settles worker 2
        manager.settle_batch(job_id)
        assert escrow_w2.status == EscrowStatus.SETTLED


# ==============================================================================
# 3. Edge Cases & Boundary Conditions
# ==============================================================================

class TestEdgeCasesBatchScheduling:
    """
    Challenge Dimension 3: Edge Cases, Corner Cases, and Boundary Defenses.
    Tests:
    1. 1-tile batches (backward compatibility when matrix size == tile size).
    2. Empty batches & invalid task_id lists (proper exception).
    3. Duplicate and blank task IDs.
    4. Non-positive satoshi amounts.
    5. Failed worker mid-batch and failover behavior.
    """

    VK = "edge-cases-vk-11223"

    def test_one_tile_batch_backward_compatibility(self):
        """
        1-tile batch: Matrix size 16x16 with tile_size 16 produces exactly 1 tile.
        Verifies:
        - 1 batch escrow is created with task_ids=[task_id].
        - Escrow is_batch returns False (since len(task_ids) == 1).
        - Worker gate authenticates the tile.
        - Result is mathematically exact.
        - Escrow settles atomically.
        """
        settlement = MockSettlementService(verification_key=self.VK)
        relay = MockNostrRelay()

        worker = MockWorkerNode(
            settlement_service=settlement,
            verification_key=self.VK,
            vram_mb=24576,
            sats_per_gflop=0.0005,
        )

        try:
            worker.start()
            worker.publish_to_relay(relay)

            config = ClientConfig(
                offline_mock_mode=True,
                nostr_relays=["ws://127.0.0.1:8008"],
                settlement_api_url="http://127.0.0.1:8080",
                settlement_verification_key=self.VK,
                min_worker_vram_mb=8192,
                max_price_per_gflop=0.001,
                default_tile_size=16,
                max_retries=0,
            )

            signer = ClientSigner()
            relay_client = RelayClient(relay_urls=["ws://127.0.0.1:8008"], in_memory_relay=relay)
            matchmaker = Matchmaker()
            backend = MockSettlementBackend(settlement)
            escrow_mgr = EscrowManager(settlement_backend=backend, verification_key=self.VK)
            dispatcher = GrpcTaskDispatcher(consumer_pubkey=signer.get_pubkey_hex())
            scheduler = ParallelTaskScheduler(dispatcher=dispatcher, max_retries=0)

            coordinator = DistributedMatmulCoordinator(
                config=config,
                signer=signer,
                relay_client=relay_client,
                matchmaker=matchmaker,
                escrow_manager=escrow_mgr,
                scheduler=scheduler,
                dispatcher=dispatcher,
                mock_workers=[worker],
            )

            # 16x16 with tile_size 16 -> exactly 1 tile
            rng = np.random.default_rng(77)
            A = rng.standard_normal((16, 16)).astype(np.float32)
            B = rng.standard_normal((16, 16)).astype(np.float32)

            res = coordinator.matmul(A, B, dtype="fp32", tile_size=16)

            assert res.shape == (16, 16)
            expected = np.matmul(A, B)
            np.testing.assert_allclose(res, expected, rtol=1e-5, atol=1e-5)

            manifest = coordinator.get_last_manifest()
            assert manifest is not None
            assert manifest.total_tasks == 1
            assert manifest.completed_tasks == 1

            # Verify escrow status
            unique_escrows = {e.payment_hash: e for e in escrow_mgr._escrows.values()}
            assert len(unique_escrows) == 1
            esc = list(unique_escrows.values())[0]
            assert esc.is_batch is False  # 1 tile is not a multi-tile batch
            assert esc.status == EscrowStatus.SETTLED
            assert esc.preimage is not None

        finally:
            worker.stop()

    def test_create_batch_escrow_empty_task_ids_rejected(self):
        """EscrowManager.create_batch_escrow raises ValueError if task_ids is empty."""
        settlement = MockSettlementService(verification_key=self.VK)
        manager = EscrowManager(settlement_backend=settlement)

        with pytest.raises(ValueError, match="non-empty list of task identifiers"):
            manager.create_batch_escrow(
                worker_pubkey="w_pk",
                task_ids=[],
                job_id="job_empty",
                total_sats=100,
            )

    def test_create_batch_escrow_duplicate_task_ids_rejected(self):
        """EscrowManager.create_batch_escrow raises ValueError if task_ids contains duplicates."""
        settlement = MockSettlementService(verification_key=self.VK)
        manager = EscrowManager(settlement_backend=settlement)

        with pytest.raises(ValueError, match="unique identifiers"):
            manager.create_batch_escrow(
                worker_pubkey="w_pk",
                task_ids=["task_dup", "task_dup"],
                job_id="job_dup",
                total_sats=100,
            )

    def test_create_batch_escrow_blank_task_id_rejected(self):
        """EscrowManager.create_batch_escrow raises ValueError if any task_id is whitespace."""
        settlement = MockSettlementService(verification_key=self.VK)
        manager = EscrowManager(settlement_backend=settlement)

        with pytest.raises(ValueError, match="empty identifiers"):
            manager.create_batch_escrow(
                worker_pubkey="w_pk",
                task_ids=["valid_task", "   "],
                job_id="job_blank",
                total_sats=100,
            )

    def test_create_batch_escrow_zero_or_negative_sats_rejected(self):
        """create_batch_escrow raises ValueError if total_sats <= 0."""
        settlement = MockSettlementService(verification_key=self.VK)
        manager = EscrowManager(settlement_backend=settlement)

        with pytest.raises(ValueError, match="positive integer"):
            manager.create_batch_escrow(
                worker_pubkey="w_pk",
                task_ids=["t1", "t2"],
                job_id="job_zero_sats",
                total_sats=0,
            )

        with pytest.raises(ValueError, match="positive integer"):
            manager.create_batch_escrow(
                worker_pubkey="w_pk",
                task_ids=["t1", "t2"],
                job_id="job_neg_sats",
                total_sats=-50,
            )

    def test_execute_gemm_zero_or_negative_dimensions_raises(self):
        """Passing empty/zero-dimension matrix to execute_gemm raises MatrixDimensionError."""
        config = ClientConfig(offline_mock_mode=True)
        signer = ClientSigner()
        coordinator = DistributedMatmulCoordinator(
            config=config,
            signer=signer,
            relay_client=RelayClient(in_memory_relay=MockNostrRelay()),
            matchmaker=Matchmaker(),
            escrow_manager=EscrowManager(),
            scheduler=ParallelTaskScheduler(dispatcher=GrpcTaskDispatcher()),
            dispatcher=GrpcTaskDispatcher(),
        )
        with pytest.raises(MatrixDimensionError):
            coordinator.execute_gemm(np.zeros((0, 4)), np.zeros((4, 4)))

    def test_worker_connection_drop_failover_in_batch(self):
        """
        When primary worker drops connection mid-stream, ParallelTaskScheduler
        fails over to backup candidate worker, successfully completing the tile.
        """
        settlement = MockSettlementService(verification_key=self.VK)
        relay = MockNostrRelay()

        worker1 = MockWorkerNode(settlement_service=settlement, verification_key=self.VK)
        worker2 = MockWorkerNode(settlement_service=settlement, verification_key=self.VK)

        try:
            worker1.start()
            worker2.start()

            # Worker 1 drops connection
            worker1.set_drop_connection(True)

            # Provision batch escrow covering failover task
            task_id = "tile_failover_1"
            inv = settlement.create_batch_escrow(
                task_ids=[task_id],
                total_compute_sats=200,
                worker_pubkey=worker1.pubkey,
            )
            settlement.hold_invoice(inv.payment_hash)

            signer = ClientSigner()
            dispatcher = GrpcTaskDispatcher(consumer_pubkey=signer.get_pubkey_hex())
            scheduler = ParallelTaskScheduler(dispatcher=dispatcher, max_retries=1)

            # Create dummy tile task plan
            rng = np.random.default_rng(88)
            A_tile = rng.standard_normal((16, 16)).astype(np.float32)
            B_tile = rng.standard_normal((16, 16)).astype(np.float32)

            from matrix_compute.tiling.partitioner import TileTaskPlan
            task_plan = TileTaskPlan(
                task_id=task_id,
                row_idx=0,
                col_idx=0,
                k_idx=0,
                row_slice=(0, 16),
                col_slice=(0, 16),
                k_slice=(0, 16),
                a_tile=A_tile,
                b_tile=B_tile,
                dtype="fp32",
                estimated_gflops=0.001,
            )

            # Candidates: [worker1 (fails), worker2 (succeeds)]
            from matrix_contracts.schemas import MatchedWorker
            candidates = [
                MatchedWorker(worker_pubkey=worker1.pubkey, p2p_multiaddr=worker1.multiaddr, lightning_routing_pubkey="ln1", reputation_score=1.0, sats_per_gflop=0.0005),
                MatchedWorker(worker_pubkey=worker2.pubkey, p2p_multiaddr=worker2.multiaddr, lightning_routing_pubkey="ln2", reputation_score=0.9, sats_per_gflop=0.0005),
            ]

            tile_c, raw_result = scheduler.execute_with_failover(
                task_plan=task_plan,
                candidate_workers=candidates,
                macaroon=inv.macaroon,
                payment_hash=inv.payment_hash,
                dtype="fp32",
            )

            # Computation completed by backup worker
            expected = np.matmul(A_tile, B_tile)
            np.testing.assert_allclose(tile_c, expected, rtol=1e-5, atol=1e-5)
            assert worker2.servicer.task_count == 1
        finally:
            worker1.stop()
            worker2.stop()


# ==============================================================================
# 4. Bond Balance Conservation & Slashing Tests
# ==============================================================================

class TestBondBalanceConservationAndSlashing:
    """
    Challenge Dimension 4: Bond Collateral Accounting & Conservation.
    Verifies:
    1. Exact satoshi conservation before and after bond slashing: Initial == Remaining + Slashed.
    2. Repeated slashing clamps cleanly at zero without negative satoshis.
    3. Bond deactivation on complete depletion (is_active=False).
    4. Multi-task batch fraud dispute resolution validates constituent task IDs.
    """

    VK = "bond-conservation-vk-33445"

    def test_exact_satoshi_conservation_on_slashing(self):
        """Initial Bond == Remaining Stake + Slashed Sats across all slash amounts."""
        settlement = MockSettlementService(verification_key=self.VK)
        worker_pk = "worker_conservation_pubkey_1"
        initial_stake = 500_000

        settlement.stake_bond(worker_pk, initial_stake)
        assert settlement.get_bond_balance(worker_pk) == initial_stake

        proof_dummy = {
            "task_id": "tile_fraud_conserv_1",
            "worker_pubkey": worker_pk,
        }

        # Slash 150,000 sats
        res = settlement.submit_fraud_dispute(proof_dummy, slash_amount_sats=150_000)
        assert res["success"] is True
        assert res["actual_slashed_sats"] == 150_000
        assert res["remaining_stake_sats"] == 350_000

        # Invariant: Conservation of satoshis
        assert res["actual_slashed_sats"] + res["remaining_stake_sats"] == initial_stake
        assert settlement.get_bond_balance(worker_pk) == 350_000

    def test_repeated_slashing_clamps_to_zero_and_deactivates_bond(self):
        """Slashing exceeding remaining stake clamps at 0 and marks bond inactive."""
        settlement = MockSettlementService(verification_key=self.VK)
        worker_pk = "worker_depleted_pubkey"
        initial_stake = 80_000

        settlement.stake_bond(worker_pk, initial_stake)

        proof_dummy = {"task_id": "tile_deplete_1", "worker_pubkey": worker_pk}

        # Request slash of 100,000 sats (exceeds 80,000 stake)
        res = settlement.submit_fraud_dispute(proof_dummy, slash_amount_sats=100_000)
        assert res["actual_slashed_sats"] == 80_000
        assert res["remaining_stake_sats"] == 0
        assert settlement.get_bond_balance(worker_pk) == 0

        # Bond must be deactivated
        bond_record = settlement._bonds[worker_pk]
        assert bond_record["is_active"] is False
        assert bond_record["staked_amount_sats"] == 0
        assert bond_record["slash_count"] == 1

        # Second dispute on already 0-bond worker
        res2 = settlement.submit_fraud_dispute(proof_dummy, slash_amount_sats=50_000)
        assert res2["actual_slashed_sats"] == 0
        assert res2["remaining_stake_sats"] == 0
        assert bond_record["staked_amount_sats"] == 0

    def test_settlement_dispute_validates_constituent_task_in_batch(self):
        """
        Dispute service validates a fraud proof for constituent task B
        even when the batch invoice primary task_id was task A.
        """
        settlement = MockSettlementService(verification_key=self.VK)
        worker_pk = "worker_batch_dispute_pk"
        settlement.stake_bond(worker_pk, 300_000)

        task_a = "batch_job_tile_0_0"
        task_b = "batch_job_tile_0_1"
        inv = settlement.create_batch_escrow(
            task_ids=[task_a, task_b],
            total_compute_sats=1000,
            worker_pubkey=worker_pk,
        )
        settlement.hold_invoice(inv.payment_hash)
        assert settlement.get_status(inv.payment_hash) == "HELD"

        # Dispute submitted for task_b (constituent, but not primary task_a)
        proof = {"task_id": task_b, "worker_pubkey": worker_pk}
        res = settlement.submit_fraud_dispute(proof, slash_amount_sats=100_000)

        assert res["success"] is True
        assert res["task_id"] == task_b
        assert res["actual_slashed_sats"] == 100_000
        assert res["invoice_cancelled"] is True

        # In-flight batch invoice cancelled
        assert settlement.get_status(inv.payment_hash) == "CANCELLED"
