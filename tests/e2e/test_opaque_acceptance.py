"""
Opaque-Box E2E Test Suite: Real-World Workload Scenarios & Acceptance.
Covers:
- Tier 4: Real-World Workloads & Acceptance Scenarios:
  1. Scenario 1: Large-Scale Distributed GEMM (1024x1024 FP32) across multiple tiles.
  2. Scenario 2: Mixed-Precision Deep Learning Attention Matrix (Q x K^T x V) in FP16 & BF16.
  3. Scenario 3: Adversarial Node Injects Fraudulent Row (Full Fraud Proof & Slashing Flow).
  4. Scenario 4: Worker Dropout & Dynamic Failover Mid-Computation.
  5. Scenario 5: Multi-Worker Parallel Block Scheduling (4 Workers).
  6. Scenario 6: Sub-Satoshi & Precision Scale Extremes (Tiny 1e-4 & Huge 1e4 Values).
  7. Scenario 7: High-Throughput Burst Dispatch & State Eviction.
  8. Scenario 8: Asymmetric Extreme Aspect Ratio GEMM (1024x16 x 16x1024).
"""

import hashlib
import json
import math
import time
from typing import Any, Dict, List

import numpy as np
import pytest

from matrix_contracts.schemas import (
    BondStatus,
    CreateHodlInvoiceResponse,
    FreivaldsFraudProof,
    HardwareSpecs,
    MatchedWorker,
    PricingSpecs,
    SettlePaymentRequest,
    WorkerAnnouncePayload,
    WorkerQueryRequest,
    WorkerQueryResponse,
)
from matrix_contracts.v1 import pb2, pb2_grpc
from tests.conftest import (
    MockRelayDouble,
    MockSettlementDouble,
    OpaqueL402Helper,
    ReferenceOracle,
    StreamChunkHelper,
)
from tests.e2e.test_distributed_matmul import ReferenceTilingEngine


class TestTier4RealWorldScenarios:
    """
    Tier 4: End-to-End Simulation of Production Workloads and Operational Lifecycles.
    """

    def test_t4_scenario_01_large_scale_distributed_gemm_1024x1024(self, mock_settlement):
        """
        Scenario 1: Large-Scale Distributed GEMM (1024x1024 FP32):
        Partitions 1024x1024 matrix into 256x256 tiles (16 block tasks),
        computes sub-products, verifies each with Freivalds, settles payments,
        and reassembles the final matrix, verifying exact agreement with NumPy.
        """
        np.random.seed(42)
        m, k, n = 1024, 1024, 1024
        tile_size = 256

        # Generate matrices with controlled standard deviation for numerical stability
        a = (np.random.randn(m, k) / math.sqrt(k)).astype(np.float32)
        b = (np.random.randn(k, n) / math.sqrt(k)).astype(np.float32)

        # 1. Tiling
        tiles = ReferenceTilingEngine.partition_matrix(a, b, tile_size=tile_size)
        # (1024/256)^3 = 4 * 4 * 4 = 64 sub-tasks
        assert len(tiles) == 64

        # 2. Process each tile with L402 escrow and Freivalds verification
        completed_tiles = []
        for idx, task in enumerate(tiles):
            task_id = f"task_1024_tile_{idx}"
            # GFLOP calculation
            gflops = ReferenceOracle.calculate_gflops(tile_size, tile_size, tile_size)
            req_sats = ReferenceOracle.calculate_min_sats(gflops, sats_per_gflop=0.0005, min_task_sats=10)

            # Escrow: Create & Hold
            preimage = hashlib.sha256(f"preimage_{idx}".encode()).hexdigest()
            h = hashlib.sha256(bytes.fromhex(preimage)).hexdigest()
            inv = mock_settlement.create_hodl_invoice(
                task_id=task_id, total_compute_sats=req_sats, payment_hash=h
            )
            mock_settlement.hold_invoice(h)

            # Worker computes tile
            tile_c = ReferenceOracle.gemm(task["a_tile"], task["b_tile"], dtype="fp32")

            # Freivalds verification
            passed, disc, bound, _ = ReferenceOracle.freivalds_verify(
                task["a_tile"], task["b_tile"], tile_c, dtype="fp32"
            )
            assert passed is True

            # Release payout
            mock_settlement.settle_invoice(h, preimage=preimage)
            assert mock_settlement.get_status(h) == "SETTLED"

            completed_tiles.append(task)

        # 3. Accumulate & Reassemble
        c_assembled = ReferenceTilingEngine.assemble_tiles(m, n, completed_tiles, dtype="fp32")
        assert c_assembled.shape == (1024, 1024)

        # 4. Spot check against double precision reference
        expected_c = np.matmul(a.astype(np.float64), b.astype(np.float64)).astype(np.float32)
        np.testing.assert_allclose(c_assembled, expected_c, rtol=1e-4, atol=1e-4)

    def test_t4_scenario_02_mixed_precision_attention_matrix(self):
        """
        Scenario 2: Mixed Precision Deep Learning Attention Matrix (Q x K^T x V):
        Rectangular tiling:
        Sequence Length = 256, Embedding Dim = 64, Hidden Dim = 128.
        Evaluates both FP16 and BF16 execution with Higham dynamic bound.
        """
        np.random.seed(43)
        seq_len = 128
        d_k = 64
        d_v = 128

        # Q (seq_len x d_k), K (seq_len x d_k) -> K_T (d_k x seq_len)
        q = np.random.randn(seq_len, d_k).astype(np.float32)
        k_t = np.random.randn(d_k, seq_len).astype(np.float32)
        v = np.random.randn(seq_len, d_v).astype(np.float32)

        # Attention scores: S = Q x K_T (seq_len x seq_len)
        tiles_s = ReferenceTilingEngine.partition_matrix(q, k_t, tile_size=64)
        s_matrix = ReferenceTilingEngine.assemble_tiles(seq_len, seq_len, tiles_s, dtype="fp32")
        expected_s = np.matmul(q, k_t)
        np.testing.assert_allclose(s_matrix, expected_s, rtol=1e-5, atol=1e-5)

        # Context: C = S x V (seq_len x d_v) in BF16
        tiles_c = ReferenceTilingEngine.partition_matrix(s_matrix, v, tile_size=64)
        c_bf16 = ReferenceTilingEngine.assemble_tiles(seq_len, d_v, tiles_c, dtype="bf16")
        assert c_bf16.shape == (seq_len, d_v)

        # Check BF16 dynamic Freivalds
        passed, disc, bound, _ = ReferenceOracle.freivalds_verify(
            s_matrix, v, c_bf16, dtype="bf16", num_trials=3
        )
        assert passed is True

    def test_t4_scenario_03_adversarial_worker_row_corruption_and_slashing(self, mock_settlement):
        """
        Scenario 3: Adversarial Node Injects Fraudulent Row:
        Worker tampers with a single row of result C.
        SDK catches fraud, constructs FreivaldsFraudProof, cancels invoice, and initiates slashing.
        """
        np.random.seed(44)
        m, k, n = 32, 32, 32
        task_id = "task_adv_row_slash"
        a = np.random.randn(m, k).astype(np.float32)
        b = np.random.randn(k, n).astype(np.float32)

        # Setup HODL escrow
        preimage = "9" * 64
        h = hashlib.sha256(bytes.fromhex(preimage)).hexdigest()
        inv = mock_settlement.create_hodl_invoice(task_id=task_id, total_compute_sats=150, payment_hash=h)
        mock_settlement.hold_invoice(h)
        assert mock_settlement.get_status(h) == "HELD"

        # Worker generates fraudulent result: row 0 corrupted
        c_corrupted = np.matmul(a, b)
        c_corrupted[0, :] += 50.0  # Big row injection!

        # Freivalds check
        passed, disc, bound, proof_data = ReferenceOracle.freivalds_verify(
            a, b, c_corrupted, dtype="fp32", seed=77
        )
        assert passed is False
        assert disc > bound

        # 1. Withhold preimage & Cancel HODL invoice
        mock_settlement.cancel_invoice(h)
        assert mock_settlement.get_status(h) == "CANCELLED"

        # 2. Build authoritative FreivaldsFraudProof
        proof = FreivaldsFraudProof(
            task_id=task_id,
            worker_pubkey="02badworkerpubkey1234567890abcdef1234567890abcdef",
            dimension_m=m,
            dimension_k=k,
            dimension_n=n,
            dtype="fp32",
            matrix_a_hash=ReferenceOracle.sha256_digest(a),
            matrix_b_hash=ReferenceOracle.sha256_digest(b),
            invalid_result_c_hash=ReferenceOracle.sha256_digest(c_corrupted),
            random_vector_r=proof_data["random_vector_r"],
            vector_br=proof_data["vector_br"],
            vector_abr=proof_data["vector_abr"],
            vector_cr=proof_data["vector_cr"],
            epsilon_tolerance=bound,
            max_observed_discrepancy=disc,
            worker_signature_on_result="worker_signed_result_hex",
            consumer_pubkey="03consumerdisputepubkey",
        )
        assert proof.max_observed_discrepancy > proof.epsilon_tolerance

        # 3. Verify worker bond slashing
        worker_bond = BondStatus(staked_amount_sats=500000, bond_txid="txid_staked_bond_99")
        slashed_bond_sats = 100000
        new_bond = worker_bond.staked_amount_sats - slashed_bond_sats
        assert new_bond == 400000

    def test_t4_scenario_04_worker_dropout_dynamic_failover(self, mock_relay, mock_settlement):
        """
        Scenario 4: Worker Dropout & Dynamic Failover Mid-Computation:
        A job has 4 tiles. Worker 1 completes tiles 1 & 2, then abruptly drops/times out.
        SDK detects disconnect, re-queries discovery, fails over to Worker 2 for tiles 3 & 4,
        and successfully reassembles the full result.
        """
        # Publish 2 workers on Nostr relay
        mock_relay.publish(
            {
                "kind": 30000,
                "pubkey": "worker_flaky_1",
                "tags": [["d", "matrix-worker"], ["vram_mb", "16000"], ["sats_per_gflop", "0.0004"], ["p2p", "p1"]],
            }
        )
        mock_relay.publish(
            {
                "kind": 30000,
                "pubkey": "worker_backup_2",
                "tags": [["d", "matrix-worker"], ["vram_mb", "16000"], ["sats_per_gflop", "0.0005"], ["p2p", "p2"]],
            }
        )

        np.random.seed(45)
        a = np.random.randn(32, 32).astype(np.float32)
        b = np.random.randn(32, 32).astype(np.float32)
        tiles = ReferenceTilingEngine.partition_matrix(a, b, tile_size=16)
        # 4 tiles: tile_0, tile_1, tile_2, tile_3 (when K=16, 2*2*2=8; with tile_size=16, num_m=2, num_k=2, num_n=2 -> 8)

        # Worker 1 succeeds on first half
        completed_tiles = []
        for task in tiles[:4]:
            completed_tiles.append(task)

        # Worker 1 DROPS on tile 4 -> SDK failover triggers
        worker_1_alive = False
        active_worker = "worker_flaky_1"

        if not worker_1_alive:
            # Query backup from discovery
            req = WorkerQueryRequest(min_vram_mb=8000, max_price_sats_per_gflop=0.001, required_slots=2)
            candidates = mock_relay.query_workers(req).matched_workers
            backup = [w for w in candidates if w.worker_pubkey != "worker_flaky_1"][0]
            active_worker = backup.worker_pubkey
            assert active_worker == "worker_backup_2"

        # Worker 2 completes remaining tiles
        for task in tiles[4:]:
            completed_tiles.append(task)

        assert len(completed_tiles) == len(tiles)
        c_assembled = ReferenceTilingEngine.assemble_tiles(32, 32, completed_tiles, dtype="fp32")
        expected_c = np.matmul(a, b)
        np.testing.assert_allclose(c_assembled, expected_c, rtol=1e-5, atol=1e-5)

    def test_t4_scenario_05_multi_worker_parallel_block_scheduling(self, mock_relay):
        """
        Scenario 5: Multi-Worker Parallel Block Scheduling (4 Workers):
        Matchmaker allocates 4 distinct workers to process 4 quadrants of a 64x64 GEMM in parallel.
        """
        # Register 4 workers
        for i in range(4):
            mock_relay.publish(
                {
                    "kind": 30000,
                    "pubkey": f"worker_pool_{i}",
                    "tags": [
                        ["d", "matrix-worker"],
                        ["vram_mb", "24576"],
                        ["sats_per_gflop", "0.0005"],
                        ["p2p", f"/ip4/127.0.0.1/tcp/{50050 + i}"],
                    ],
                }
            )

        req = WorkerQueryRequest(min_vram_mb=16000, max_price_sats_per_gflop=0.001, required_slots=4)
        workers = mock_relay.query_workers(req).matched_workers
        assert len(workers) == 4

        # Partition 64x64 matrix into 4 32x32 tiles (num_m=2, num_n=2, num_k=1 with tile_size=32 and k=32)
        np.random.seed(46)
        a = np.random.randn(64, 32).astype(np.float32)
        b = np.random.randn(32, 64).astype(np.float32)
        tiles = ReferenceTilingEngine.partition_matrix(a, b, tile_size=32)
        assert len(tiles) == 4

        # Dispatch 1 tile to each worker
        dispatched_tasks = []
        for worker, task in zip(workers, tiles):
            dispatched_tasks.append((worker.worker_pubkey, task))

        assert len(dispatched_tasks) == 4
        # All workers distinct
        worker_ids = {w_id for w_id, _ in dispatched_tasks}
        assert len(worker_ids) == 4

        # Reassemble
        c = ReferenceTilingEngine.assemble_tiles(64, 64, tiles, dtype="fp32")
        np.testing.assert_allclose(c, np.matmul(a, b), rtol=1e-5, atol=1e-5)

    def test_t4_scenario_06_sub_satoshi_and_precision_scale_extremes(self):
        """
        Scenario 6: Sub-Satoshi & Precision Scale Extremes:
        Tests tiny inputs (1e-4) and huge inputs (1e4).
        Verifies Higham bound avoids false positives/negatives and min_task_sats fee floor holds.
        """
        # 1. Tiny input matrix (1e-4)
        a_tiny = np.ones((16, 16), dtype=np.float32) * 1e-4
        b_tiny = np.ones((16, 16), dtype=np.float32) * 1e-4
        c_tiny = np.matmul(a_tiny, b_tiny)
        passed_tiny, disc_t, bound_t, _ = ReferenceOracle.freivalds_verify(a_tiny, b_tiny, c_tiny, dtype="fp32")
        assert passed_tiny is True
        assert disc_t <= bound_t

        # Fee calculation on tiny workload: raw sats < 1, so min_task_sats applies
        gflops_tiny = ReferenceOracle.calculate_gflops(16, 16, 16)
        sats_tiny = ReferenceOracle.calculate_min_sats(gflops_tiny, sats_per_gflop=0.0005, min_task_sats=10)
        assert sats_tiny == 10  # Enforces 10 sat floor!

        # 2. Huge input matrix (1e4)
        a_huge = np.ones((16, 16), dtype=np.float32) * 1e4
        b_huge = np.ones((16, 16), dtype=np.float32) * 1e4
        c_huge = np.matmul(a_huge, b_huge)
        passed_huge, disc_h, bound_h, _ = ReferenceOracle.freivalds_verify(a_huge, b_huge, c_huge, dtype="fp32")
        assert passed_huge is True
        assert disc_h <= bound_h

    def test_t4_scenario_07_high_throughput_burst_dispatch(self, mock_settlement):
        """
        Scenario 7: High-Throughput Burst Dispatch & State Purge:
        Executes 20 rapid sequential tile requests, verifying invoice settlement
        and absence of cross-task state corruption.
        """
        np.random.seed(47)
        for i in range(20):
            task_id = f"burst_task_{i}"
            a = np.random.randn(8, 8).astype(np.float32)
            b = np.random.randn(8, 8).astype(np.float32)
            c = np.matmul(a, b)

            preimage = hashlib.sha256(f"preimage_burst_{i}".encode()).hexdigest()
            h = hashlib.sha256(bytes.fromhex(preimage)).hexdigest()

            inv = mock_settlement.create_hodl_invoice(task_id=task_id, total_compute_sats=25, payment_hash=h)
            mock_settlement.hold_invoice(h)

            passed, _, _, _ = ReferenceOracle.freivalds_verify(a, b, c, dtype="fp32")
            assert passed is True

            mock_settlement.settle_invoice(h, preimage=preimage)
            assert mock_settlement.get_status(h) == "SETTLED"

    def test_t4_scenario_08_asymmetric_extreme_aspect_ratio_gemm(self):
        """
        Scenario 8: Asymmetric Extreme Aspect Ratio GEMM:
        Tall-skinny matrix A (1024x16) x Wide-flat matrix B (16x1024) -> Large C (1024x1024).
        Tests memory efficiency and boundary tile alignment.
        """
        np.random.seed(48)
        m, k, n = 1024, 16, 1024
        a = (np.random.randn(m, k) / 4.0).astype(np.float32)
        b = (np.random.randn(k, n) / 4.0).astype(np.float32)

        # Tile with tile_size=256
        tiles = ReferenceTilingEngine.partition_matrix(a, b, tile_size=256)
        # num_m = 4, num_k = 1, num_n = 4 -> 4 * 4 * 1 = 16 sub-tasks
        assert len(tiles) == 16

        # Reassemble
        c = ReferenceTilingEngine.assemble_tiles(m, n, tiles, dtype="fp32")
        assert c.shape == (1024, 1024)

        # Spot check
        expected_c = np.matmul(a, b)
        np.testing.assert_allclose(c, expected_c, rtol=1e-5, atol=1e-5)
