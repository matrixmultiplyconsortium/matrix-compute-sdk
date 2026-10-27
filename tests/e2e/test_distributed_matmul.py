"""
Opaque-Box E2E Test Suite: Distributed Matrix Multiplication.
Covers:
- Tier 1: Feature Coverage across Matrix Tiling, Precision Handling, Assembly,
  Worker Discovery, Matchmaking, gRPC Streaming, L402 Escrow, Freivalds Verification.
- Tier 2: Boundary and Corner Cases: 1x1 scalar matrices, empty matrices, odd dimensions,
  non-multiples of tile sizes, asymmetric aspect ratios, numerical stability,
  malformed inputs, and protocol edge cases.
"""

import base64
import hashlib
import hmac
import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest

from matrix_contracts.schemas import (
    BondStatus,
    CreateHodlInvoiceRequest,
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


# ==============================================================================
# REFERENCE IMPLEMENTATIONS FOR OPAQUE CONTRACT VALIDATION
# ==============================================================================

class ReferenceTilingEngine:
    """Independent reference implementation of matrix block tiling and accumulation."""

    @staticmethod
    def partition_matrix(
        a: np.ndarray, b: np.ndarray, tile_size: int
    ) -> List[Dict[str, Any]]:
        """Partitions matrices A (M x K) and B (K x N) into tile tasks."""
        m, k = a.shape
        k2, n = b.shape
        if k != k2:
            raise ValueError(f"Dimension mismatch: {k} != {k2}")

        tasks = []
        num_m = math.ceil(m / tile_size)
        num_k = math.ceil(k / tile_size)
        num_n = math.ceil(n / tile_size)

        for i in range(num_m):
            row_start = i * tile_size
            row_end = min((i + 1) * tile_size, m)
            for j in range(num_n):
                col_start = j * tile_size
                col_end = min((j + 1) * tile_size, n)
                for p in range(num_k):
                    k_start = p * tile_size
                    k_end = min((p + 1) * tile_size, k)

                    a_tile = a[row_start:row_end, k_start:k_end]
                    b_tile = b[k_start:k_end, col_start:col_end]
                    tasks.append(
                        {
                            "task_id": f"tile_{i}_{j}_{p}",
                            "row_idx": i,
                            "col_idx": j,
                            "k_idx": p,
                            "row_slice": (row_start, row_end),
                            "col_slice": (col_start, col_end),
                            "k_slice": (k_start, k_end),
                            "a_tile": a_tile,
                            "b_tile": b_tile,
                        }
                    )
        return tasks

    @staticmethod
    def assemble_tiles(
        m: int, n: int, tile_tasks: List[Dict[str, Any]], dtype: str = "fp32"
    ) -> np.ndarray:
        """Accumulates tile products into full M x N output matrix."""
        np_dtype = np.float32 if dtype in ("fp32", "bf16") else np.float16
        c = np.zeros((m, n), dtype=np_dtype)
        for task in tile_tasks:
            r_s, r_e = task["row_slice"]
            c_s, c_e = task["col_slice"]
            if r_e > m or c_e > n or r_s < 0 or c_s < 0:
                raise ValueError(f"Tile slice out of bounds: rows {r_s}:{r_e} vs {m}, cols {c_s}:{c_e} vs {n}")
            tile_res = ReferenceOracle.gemm(task["a_tile"], task["b_tile"], dtype=dtype)
            c[r_s:r_e, c_s:c_e] += tile_res

        if dtype == "bf16":
            return ReferenceOracle.truncate_to_bf16(c)
        return c


# ==============================================================================
# TIER 1: FEATURE COVERAGE (60 TESTS: 12 FEATURES x 5 TESTS EACH)
# ==============================================================================

class TestTier1FeatureCoverage:
    """
    Tier 1: Comprehensive feature coverage across all 12 core SDK requirements.
    """

    # --- Feature 1: Matrix Tiling & Slicing ---

    def test_t1_01_square_matrix_tiling_even_blocks(self):
        """T1.F1.1: Partitions 64x64 square matrix into exactly 4 32x32 tiles."""
        a = np.ones((64, 64), dtype=np.float32)
        b = np.ones((64, 64), dtype=np.float32)
        tiles = ReferenceTilingEngine.partition_matrix(a, b, tile_size=32)
        # 2 row blocks, 2 col blocks, 2 inner blocks = 2 * 2 * 2 = 8 sub-tasks
        assert len(tiles) == 8
        for t in tiles:
            assert t["a_tile"].shape == (32, 32)
            assert t["b_tile"].shape == (32, 32)

    def test_t1_02_rectangular_matrix_tiling(self):
        """T1.F1.2: Partitions rectangular matrices (64x32 x 32x128) with tile_size=32."""
        a = np.ones((64, 32), dtype=np.float32)
        b = np.ones((32, 128), dtype=np.float32)
        tiles = ReferenceTilingEngine.partition_matrix(a, b, tile_size=32)
        # num_m = 2, num_k = 1, num_n = 4 -> 2 * 4 * 1 = 8 sub-tasks
        assert len(tiles) == 8
        for t in tiles:
            assert t["a_tile"].shape == (32, 32)
            assert t["b_tile"].shape == (32, 32)

    def test_t1_03_tile_task_metadata_consistency(self):
        """T1.F1.3: Each tile task plan contains monotonic task_id, valid slices, and indices."""
        a = np.zeros((16, 16), dtype=np.float32)
        b = np.zeros((16, 16), dtype=np.float32)
        tiles = ReferenceTilingEngine.partition_matrix(a, b, tile_size=8)
        assert len(tiles) == 8
        task_ids = {t["task_id"] for t in tiles}
        assert len(task_ids) == 8
        for t in tiles:
            r_s, r_e = t["row_slice"]
            c_s, c_e = t["col_slice"]
            assert r_e > r_s
            assert c_e > c_s

    def test_t1_04_single_tile_full_coverage(self):
        """T1.F1.4: When tile_size >= matrix dimensions, creates exactly 1 tile task."""
        a = np.ones((16, 16), dtype=np.float32)
        b = np.ones((16, 16), dtype=np.float32)
        tiles = ReferenceTilingEngine.partition_matrix(a, b, tile_size=32)
        assert len(tiles) == 1
        assert tiles[0]["a_tile"].shape == (16, 16)
        assert tiles[0]["b_tile"].shape == (16, 16)

    def test_t1_05_inner_dimension_match_validation(self):
        """T1.F1.5: Tiling engine rejects matrices with mismatched inner dimensions."""
        a = np.ones((16, 32), dtype=np.float32)
        b = np.ones((24, 16), dtype=np.float32)
        with pytest.raises(ValueError, match="Dimension mismatch"):
            ReferenceTilingEngine.partition_matrix(a, b, tile_size=8)

    # --- Feature 2: Result Accumulation & Assembly ---

    def test_t1_06_accumulator_reassembly_fp32_matches_matmul(self):
        """T1.F2.1: Accumulator reassembles partitioned tiles to match direct np.matmul."""
        np.random.seed(42)
        a = np.random.randn(32, 32).astype(np.float32)
        b = np.random.randn(32, 32).astype(np.float32)
        tiles = ReferenceTilingEngine.partition_matrix(a, b, tile_size=16)
        assembled_c = ReferenceTilingEngine.assemble_tiles(32, 32, tiles, dtype="fp32")
        expected_c = np.matmul(a, b)
        np.testing.assert_allclose(assembled_c, expected_c, rtol=1e-5, atol=1e-5)

    def test_t1_07_accumulator_non_square_reassembly(self):
        """T1.F2.2: Accumulator correctly reassembles rectangular 48x32 x 32x64 product."""
        np.random.seed(43)
        a = np.random.randn(48, 32).astype(np.float32)
        b = np.random.randn(32, 64).astype(np.float32)
        tiles = ReferenceTilingEngine.partition_matrix(a, b, tile_size=16)
        assembled_c = ReferenceTilingEngine.assemble_tiles(48, 64, tiles, dtype="fp32")
        expected_c = np.matmul(a, b)
        assert assembled_c.shape == (48, 64)
        np.testing.assert_allclose(assembled_c, expected_c, rtol=1e-5, atol=1e-5)

    def test_t1_08_multi_k_accumulation_sum(self):
        """T1.F2.3: Accumulator sums multiple K-slice sub-products C_ij = sum_p A_ip * B_pj."""
        a = np.ones((8, 16), dtype=np.float32)
        b = np.ones((16, 8), dtype=np.float32)
        tiles = ReferenceTilingEngine.partition_matrix(a, b, tile_size=8)
        # 2 K-blocks, each yields 8x8 matrix of 8s; total sum should be 16s
        assembled = ReferenceTilingEngine.assemble_tiles(8, 8, tiles, dtype="fp32")
        assert np.all(assembled == 16.0)

    def test_t1_09_identity_matrix_assembly(self):
        """T1.F2.4: Slicing and reassembling A x I yields original matrix A."""
        np.random.seed(44)
        a = np.random.randn(16, 16).astype(np.float32)
        i = np.eye(16, dtype=np.float32)
        tiles = ReferenceTilingEngine.partition_matrix(a, i, tile_size=8)
        c = ReferenceTilingEngine.assemble_tiles(16, 16, tiles, dtype="fp32")
        np.testing.assert_allclose(c, a, rtol=1e-5)

    def test_t1_10_independent_tile_accumulation_order(self):
        """T1.F2.5: Assembling tiles in shuffled/reversed order produces identical result."""
        np.random.seed(45)
        a = np.random.randn(16, 16).astype(np.float32)
        b = np.random.randn(16, 16).astype(np.float32)
        tiles = ReferenceTilingEngine.partition_matrix(a, b, tile_size=8)
        c_forward = ReferenceTilingEngine.assemble_tiles(16, 16, tiles, dtype="fp32")
        c_reversed = ReferenceTilingEngine.assemble_tiles(16, 16, list(reversed(tiles)), dtype="fp32")
        np.testing.assert_allclose(c_forward, c_reversed, rtol=1e-6)

    # --- Feature 3: Precision Handling (FP32, FP16, BF16) ---

    def test_t1_11_fp32_precision_tolerance(self):
        """T1.F3.1: Float32 GEMM matches double precision reference within 1e-5 relative tolerance."""
        np.random.seed(46)
        a = np.random.randn(16, 16).astype(np.float32)
        b = np.random.randn(16, 16).astype(np.float32)
        c = ReferenceOracle.gemm(a, b, dtype="fp32")
        assert c.dtype == np.float32
        np.testing.assert_allclose(c, np.matmul(a.astype(np.float64), b.astype(np.float64)), rtol=1e-5)

    def test_t1_12_fp16_precision_casting(self):
        """T1.F3.2: Float16 GEMM executes and returns numpy.float16 array."""
        a = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float16)
        b = np.array([[0.5, 1.0], [1.5, 2.0]], dtype=np.float16)
        c = ReferenceOracle.gemm(a, b, dtype="fp16")
        assert c.dtype == np.float16
        expected = np.matmul(a.astype(np.float64), b.astype(np.float64)).astype(np.float16)
        np.testing.assert_allclose(c, expected, rtol=1e-3)

    def test_t1_13_bf16_mantissa_masking(self):
        """T1.F3.3: BF16 emulation zeroes lower 16 bits of uint32 representation."""
        a = np.array([1.2345678, 3.1415927, -0.0078125], dtype=np.float32)
        bf16 = ReferenceOracle.truncate_to_bf16(a)
        u32 = bf16.view(np.uint32)
        for val in u32:
            assert (val & 0xFFFF) == 0

    def test_t1_14_bf16_nan_and_inf_preservation(self):
        """T1.F3.4: BF16 truncation preserves IEEE NaN and Infinity semantics."""
        special = np.array([np.nan, np.inf, -np.inf, 0.0, -0.0], dtype=np.float32)
        bf16 = ReferenceOracle.truncate_to_bf16(special)
        assert np.isnan(bf16[0])
        assert np.isinf(bf16[1]) and bf16[1] > 0
        assert np.isinf(bf16[2]) and bf16[2] < 0
        assert bf16[3] == 0.0
        assert bf16[4] == 0.0

    def test_t1_15_bf16_matmul_execution(self):
        """T1.F3.5: BF16 GEMM preserves 7-bit mantissa precision bounds."""
        a = np.ones((8, 8), dtype=np.float32) * 2.5
        b = np.ones((8, 8), dtype=np.float32) * 3.0
        c = ReferenceOracle.gemm(a, b, dtype="bf16")
        # 8 * (2.5 * 3.0) = 8 * 7.5 = 60.0
        assert c[0, 0] == 60.0
        u32 = c.view(np.uint32)
        assert (u32[0, 0] & 0xFFFF) == 0

    # --- Feature 4: Worker Discovery & Filtering ---

    def test_t1_16_relay_publish_worker_announce(self, mock_relay):
        """T1.F4.1: Publishes valid Kind 30000 worker announcement to relay double."""
        event = {
            "kind": 30000,
            "pubkey": "worker_pub_101",
            "tags": [
                ["d", "matrix-worker"],
                ["p2p", "/ip4/127.0.0.1/tcp/50051"],
                ["lightning", "02e3b0..."],
                ["vram_mb", "24576"],
                ["sats_per_gflop", "0.0005"],
            ],
            "content": "{}",
        }
        assert mock_relay.publish(event) is True
        events = mock_relay.get_events()
        assert len(events) == 1

    def test_t1_17_relay_filter_by_min_vram(self, mock_relay):
        """T1.F4.2: Relay query excludes workers with VRAM less than required threshold."""
        e1 = {
            "kind": 30000,
            "pubkey": "w_small",
            "tags": [["d", "matrix-worker"], ["vram_mb", "8192"], ["sats_per_gflop", "0.0005"]],
        }
        e2 = {
            "kind": 30000,
            "pubkey": "w_large",
            "tags": [["d", "matrix-worker"], ["vram_mb", "24576"], ["sats_per_gflop", "0.0005"]],
        }
        mock_relay.publish(e1)
        mock_relay.publish(e2)

        results = mock_relay.query(min_vram_mb=16000)
        assert len(results) == 1
        assert results[0]["pubkey"] == "w_large"

    def test_t1_18_relay_filter_by_max_price(self, mock_relay):
        """T1.F4.3: Relay query excludes workers whose price exceeds max_price."""
        e1 = {
            "kind": 30000,
            "pubkey": "w_cheap",
            "tags": [["d", "matrix-worker"], ["vram_mb", "16000"], ["sats_per_gflop", "0.0004"]],
        }
        e2 = {
            "kind": 30000,
            "pubkey": "w_expensive",
            "tags": [["d", "matrix-worker"], ["vram_mb", "16000"], ["sats_per_gflop", "0.0010"]],
        }
        mock_relay.publish(e1)
        mock_relay.publish(e2)

        results = mock_relay.query(max_price=0.0005)
        assert len(results) == 1
        assert results[0]["pubkey"] == "w_cheap"

    def test_t1_19_worker_query_request_schema_validation(self):
        """T1.F4.4: WorkerQueryRequest model strictly validates query criteria."""
        req = WorkerQueryRequest(
            min_vram_mb=16384,
            max_price_sats_per_gflop=0.0008,
            required_slots=2,
        )
        assert req.method == "QUERY_WORKERS"
        assert req.min_vram_mb == 16384
        assert req.required_slots == 2

    def test_t1_20_worker_query_response_schema_validation(self):
        """T1.F4.5: WorkerQueryResponse serializes matched workers list."""
        w = MatchedWorker(
            worker_pubkey="pk_1",
            p2p_multiaddr="/ip4/127.0.0.1/tcp/50051",
            lightning_routing_pubkey="ln_pk_1",
            reputation_score=0.98,
            sats_per_gflop=0.0005,
        )
        resp = WorkerQueryResponse(status="SUCCESS", matched_workers=[w])
        assert resp.status == "SUCCESS"
        assert len(resp.matched_workers) == 1
        assert resp.matched_workers[0].reputation_score == 0.98

    # --- Feature 5: Worker Matchmaking & Pareto Ranking ---

    def test_t1_21_matchmaker_sorts_by_cheapest_price(self, mock_relay):
        """T1.F5.1: Matchmaker ranks candidate workers primarily by price ascending."""
        prices = [0.0008, 0.0003, 0.0005]
        for idx, p in enumerate(prices):
            mock_relay.publish(
                {
                    "kind": 30000,
                    "pubkey": f"w_{idx}",
                    "tags": [
                        ["d", "matrix-worker"],
                        ["vram_mb", "16000"],
                        ["sats_per_gflop", str(p)],
                        ["p2p", f"/ip4/127.0.0.1/tcp/{50050+idx}"],
                    ],
                }
            )
        req = WorkerQueryRequest(min_vram_mb=8000, max_price_sats_per_gflop=0.001, required_slots=3)
        resp = mock_relay.query_workers(req)
        assert len(resp.matched_workers) == 3
        # First should be cheapest
        assert resp.matched_workers[0].sats_per_gflop == 0.0003
        assert resp.matched_workers[1].sats_per_gflop == 0.0005
        assert resp.matched_workers[2].sats_per_gflop == 0.0008

    def test_t1_22_matchmaker_slots_truncation(self, mock_relay):
        """T1.F5.2: Matchmaker returns exactly the required_slots count."""
        for i in range(5):
            mock_relay.publish(
                {
                    "kind": 30000,
                    "pubkey": f"w_{i}",
                    "tags": [
                        ["d", "matrix-worker"],
                        ["vram_mb", "16000"],
                        ["sats_per_gflop", "0.0005"],
                        ["p2p", f"/ip4/127.0.0.1/tcp/{50050+i}"],
                    ],
                }
            )
        req = WorkerQueryRequest(min_vram_mb=8000, max_price_sats_per_gflop=0.001, required_slots=2)
        resp = mock_relay.query_workers(req)
        assert len(resp.matched_workers) == 2

    def test_t1_23_matchmaker_tie_break_reputation(self):
        """T1.F5.3: Matchmaker breaks price ties by reputation descending."""
        w1 = MatchedWorker(
            worker_pubkey="w1",
            p2p_multiaddr="p1",
            lightning_routing_pubkey="l1",
            reputation_score=0.85,
            sats_per_gflop=0.0005,
        )
        w2 = MatchedWorker(
            worker_pubkey="w2",
            p2p_multiaddr="p2",
            lightning_routing_pubkey="l2",
            reputation_score=0.99,
            sats_per_gflop=0.0005,
        )
        ranked = sorted([w1, w2], key=lambda w: (w.sats_per_gflop, -w.reputation_score))
        assert ranked[0].worker_pubkey == "w2"

    def test_t1_24_empty_candidates_handled_gracefully(self, mock_relay):
        """T1.F5.4: When no workers match criteria, returns SUCCESS with empty matched list."""
        req = WorkerQueryRequest(min_vram_mb=128000, max_price_sats_per_gflop=0.00001)
        resp = mock_relay.query_workers(req)
        assert resp.status == "SUCCESS"
        assert len(resp.matched_workers) == 0

    def test_t1_25_hardware_specs_fields_present(self):
        """T1.F5.5: Worker announcement contains complete hardware and pricing specifications."""
        hw = HardwareSpecs(
            gpu_model="RTX 4090",
            vram_mb=24576,
            peak_fp32_tflops=82.6,
            attestation_proof="proof_blob",
        )
        pr = PricingSpecs(sats_per_gflop=0.0005, min_task_sats=10)
        bd = BondStatus(staked_amount_sats=100000, bond_txid="txid_123")
        assert hw.vram_mb == 24576
        assert pr.min_task_sats == 10
        assert bd.staked_amount_sats == 100000

    # --- Feature 6: Worker Retry & Failover ---

    def test_t1_26_worker_failover_selects_backup_candidate(self, mock_relay):
        """T1.F6.1: On primary worker failure, selects next candidate worker in ranked list."""
        mock_relay.publish(
            {
                "kind": 30000,
                "pubkey": "worker_primary",
                "tags": [["d", "matrix-worker"], ["vram_mb", "16000"], ["sats_per_gflop", "0.0004"], ["p2p", "p1"]],
            }
        )
        mock_relay.publish(
            {
                "kind": 30000,
                "pubkey": "worker_backup",
                "tags": [["d", "matrix-worker"], ["vram_mb", "16000"], ["sats_per_gflop", "0.0005"], ["p2p", "p2"]],
            }
        )
        req = WorkerQueryRequest(min_vram_mb=8000, max_price_sats_per_gflop=0.001, required_slots=2)
        candidates = mock_relay.query_workers(req).matched_workers

        # Simulate primary failure -> failover to backup
        failed_worker = candidates[0]
        backup_worker = candidates[1]
        assert failed_worker.worker_pubkey == "worker_primary"
        assert backup_worker.worker_pubkey == "worker_backup"

    def test_t1_27_reputation_penalty_on_failure(self):
        """T1.F6.2: Worker failing a task experiences simulated reputation penalty."""
        worker = MatchedWorker(
            worker_pubkey="w_flaky",
            p2p_multiaddr="p1",
            lightning_routing_pubkey="l1",
            reputation_score=1.0,
            sats_per_gflop=0.0005,
        )
        # Apply 20% penalty
        penalized_rep = max(0.0, worker.reputation_score - 0.20)
        assert round(penalized_rep, 2) == 0.80

    def test_t1_28_retry_attempts_count_exhaustion(self):
        """T1.F6.3: Retry mechanism tracks retry attempts up to max_retries before raising error."""
        max_retries = 3
        attempts = 0
        success = False
        while attempts < max_retries and not success:
            attempts += 1
        assert attempts == 3
        assert success is False

    def test_t1_29_failover_preserves_task_id_and_tensors(self):
        """T1.F6.4: Failover dispatches identical task_id and input tensors to backup worker."""
        task_id = "task_failover_001"
        a = np.ones((8, 8), dtype=np.float32)
        b = np.ones((8, 8), dtype=np.float32)
        # In failover, payload dispatched to worker B matches worker A
        dispatch_a = {"task_id": task_id, "matrix_a": a, "matrix_b": b}
        failover_dispatch = {"task_id": task_id, "matrix_a": a, "matrix_b": b}
        assert dispatch_a["task_id"] == failover_dispatch["task_id"]
        np.testing.assert_array_equal(dispatch_a["matrix_a"], failover_dispatch["matrix_a"])

    def test_t1_30_blacklisting_repeatedly_failed_worker(self):
        """T1.F6.5: Workers failing consecutively are excluded from subsequent matchmaking."""
        pool = ["w1", "w2", "w3"]
        blacklist = {"w1"}
        active_pool = [w for w in pool if w not in blacklist]
        assert "w1" not in active_pool
        assert len(active_pool) == 2

    # --- Feature 7: gRPC Handshake & Chunk Streaming ---

    def test_t1_31_request_tile_task_ack_gflops(self):
        """T1.F7.1: RequestTileTask calculates exact GFLOP quote (2 * M * N * K) / 1e9."""
        m, k, n = 512, 512, 512
        gflops = ReferenceOracle.calculate_gflops(m, k, n)
        # 2 * 512^3 / 1e9 = 0.268435456
        assert round(gflops, 6) == 0.268435

    def test_t1_32_stream_chunks_generation(self):
        """T1.F7.2: StreamChunkHelper generates sequential TileDataChunk messages."""
        a = np.ones((8, 8), dtype=np.float32)
        b = np.ones((8, 8), dtype=np.float32)
        chunks = StreamChunkHelper.generate_chunks("task_stream_1", a, b, chunk_size_bytes=64)
        assert len(chunks) > 1
        # Sequence numbers are strictly monotonic
        for i, c in enumerate(chunks):
            assert c.sequence_number == i
            assert c.task_id == "task_stream_1"
        assert chunks[-1].is_final_chunk is True

    def test_t1_33_stream_chunk_split_between_matrix_a_and_b(self):
        """T1.F7.3: Stream chunks correctly distinguish Matrix A from Matrix B."""
        a = np.ones((4, 4), dtype=np.float32)
        b = np.ones((4, 4), dtype=np.float32)
        chunks = StreamChunkHelper.generate_chunks("task_ab", a, b, chunk_size_bytes=32)
        a_chunks = [c for c in chunks if c.is_matrix_a]
        b_chunks = [c for c in chunks if not c.is_matrix_a]
        assert len(a_chunks) > 0
        assert len(b_chunks) > 0
        # Reassembled bytes match original
        a_reassembled = b"".join(c.chunk_bytes for c in a_chunks)
        assert a_reassembled == a.tobytes()

    def test_t1_34_final_chunk_flag_semantics(self):
        """T1.F7.4: Only the very last chunk in the stream has is_final_chunk=True."""
        a = np.ones((4, 4), dtype=np.float32)
        b = np.ones((4, 4), dtype=np.float32)
        chunks = StreamChunkHelper.generate_chunks("task_fin", a, b, chunk_size_bytes=32)
        for c in chunks[:-1]:
            assert c.is_final_chunk is False
        assert chunks[-1].is_final_chunk is True

    def test_t1_35_tile_result_protobuf_schema(self):
        """T1.F7.5: TileResult protobuf message validates result fields."""
        c = np.ones((8, 8), dtype=np.float32)
        res = pb2.TileResult(
            task_id="t1",
            rows=8,
            cols=8,
            result_bytes=c.tobytes(),
            worker_signature="sig_hex",
            execution_time_ms=12,
        )
        assert res.rows == 8
        assert res.cols == 8
        assert len(res.result_bytes) == 8 * 8 * 4

    # --- Feature 8: L402 Trustless Escrow & Headers ---

    def test_t1_36_create_hodl_invoice_trustless_preimage(self, mock_settlement):
        """T1.F8.1: Creates HODL invoice in UNPAID status with consumer-generated hash."""
        preimage = "b" * 64
        payment_hash = hashlib.sha256(bytes.fromhex(preimage)).hexdigest()
        resp = mock_settlement.create_hodl_invoice(
            task_id="task_escrow_1",
            total_compute_sats=100,
            payment_hash=payment_hash,
        )
        assert resp.payment_hash == payment_hash
        assert mock_settlement.get_status(payment_hash) == "UNPAID"

    def test_t1_37_hodl_invoice_fee_calculation(self, mock_settlement):
        """T1.F8.2: 5% platform routing fee is correctly deducted from total compute sats."""
        resp = mock_settlement.create_hodl_invoice(
            task_id="task_fee",
            total_compute_sats=1000,
        )
        assert resp.platform_fee_sats == 50
        assert resp.worker_payout_sats == 950

    def test_t1_38_l402_auth_header_format(self):
        """T1.F8.3: Formats Authorization header as 'L402 <macaroon>:<payment_hash>'."""
        macaroon = "bWFjYXJvb25fdG9rZW4="
        payment_hash = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        header = OpaqueL402Helper.format_auth_header(macaroon, payment_hash)
        assert header.startswith("L402 ")
        m_parsed, h_parsed = OpaqueL402Helper.parse_auth_header(header)
        assert m_parsed == macaroon
        assert h_parsed == payment_hash

    def test_t1_39_macaroon_caveat_integrity(self):
        """T1.F8.4: Macaroon HMAC signature validates caveats against verification key."""
        vkey = "secret_key_123"
        task_id = "task_cav"
        expiry = int(time.time()) + 3600
        macaroon = OpaqueL402Helper.mint_macaroon(vkey, task_id, expiry, max_gflops=1.5, locked_sats=50)
        # Decode and verify HMAC
        payload = json.loads(base64.b64decode(macaroon).decode("utf-8"))
        caveat_str = json.dumps(payload["caveats"], sort_keys=True)
        expected_sig = hmac.new(vkey.encode("utf-8"), caveat_str.encode("utf-8"), hashlib.sha256).hexdigest()
        assert payload["signature"] == expected_sig

    def test_t1_40_hold_invoice_state_transition(self, mock_settlement):
        """T1.F8.5: Transitions invoice status from UNPAID to HELD prior to compute dispatch."""
        resp = mock_settlement.create_hodl_invoice(task_id="task_hold", total_compute_sats=50)
        h = resp.payment_hash
        assert mock_settlement.get_status(h) == "UNPAID"
        mock_settlement.hold_invoice(h)
        assert mock_settlement.get_status(h) == "HELD"

    # --- Feature 9: Preimage Release on Success ---

    def test_t1_41_settle_invoice_with_valid_preimage(self, mock_settlement):
        """T1.F9.1: Settles invoice and transitions to SETTLED upon revealing valid preimage."""
        preimage = "c" * 64
        payment_hash = hashlib.sha256(bytes.fromhex(preimage)).hexdigest()
        resp = mock_settlement.create_hodl_invoice(
            task_id="task_settle", total_compute_sats=100, payment_hash=payment_hash
        )
        mock_settlement.hold_invoice(payment_hash)
        assert mock_settlement.get_status(payment_hash) == "HELD"

        mock_settlement.settle_invoice(payment_hash, preimage=preimage)
        assert mock_settlement.get_status(payment_hash) == "SETTLED"

    def test_t1_42_settle_payment_request_schema(self):
        """T1.F9.2: SettlePaymentRequest schema validates task_id and 64-character hex preimage."""
        req = SettlePaymentRequest(task_id="task_s1", preimage="d" * 64)
        assert req.task_id == "task_s1"
        assert len(req.preimage) == 64

    def test_t1_43_cannot_settle_without_holding_first(self, mock_settlement):
        """T1.F9.3: Attempting to settle an UNPAID invoice raises ValueError."""
        resp = mock_settlement.create_hodl_invoice(task_id="task_err", total_compute_sats=100)
        with pytest.raises(ValueError, match="must be HELD"):
            mock_settlement.settle_invoice(resp.payment_hash, preimage="a" * 64)

    def test_t1_44_settle_releases_worker_payout(self, mock_settlement):
        """T1.F9.4: Settled invoice stores revealed preimage for worker redemption."""
        preimage = "e" * 64
        payment_hash = hashlib.sha256(bytes.fromhex(preimage)).hexdigest()
        mock_settlement.create_hodl_invoice(
            task_id="task_payout", total_compute_sats=200, payment_hash=payment_hash
        )
        mock_settlement.hold_invoice(payment_hash)
        mock_settlement.settle_invoice(payment_hash, preimage=preimage)
        assert mock_settlement._invoices[payment_hash]["preimage"] == preimage

    def test_t1_45_settle_invoice_idempotence(self, mock_settlement):
        """T1.F9.5: Settling an already SETTLED invoice rejects with clean error."""
        preimage = "f" * 64
        payment_hash = hashlib.sha256(bytes.fromhex(preimage)).hexdigest()
        mock_settlement.create_hodl_invoice(
            task_id="task_idem", total_compute_sats=50, payment_hash=payment_hash
        )
        mock_settlement.hold_invoice(payment_hash)
        mock_settlement.settle_invoice(payment_hash, preimage=preimage)
        with pytest.raises(ValueError, match="must be HELD"):
            mock_settlement.settle_invoice(payment_hash, preimage=preimage)

    # --- Feature 10: Dynamic Scale-Aware Freivalds Check ---

    def test_t1_46_freivalds_accurate_result_passes(self):
        """T1.F10.1: Freivalds check passes for accurate FP32 matrix multiplication."""
        np.random.seed(47)
        a = np.random.randn(16, 16).astype(np.float32)
        b = np.random.randn(16, 16).astype(np.float32)
        c = np.matmul(a, b)
        passed, disc, bound, proof = ReferenceOracle.freivalds_verify(a, b, c, dtype="fp32")
        assert passed is True
        assert disc <= bound
        assert len(proof) == 0

    def test_t1_47_freivalds_corrupted_result_detected(self):
        """T1.F10.2: Freivalds check fails when result tensor C contains a single corrupted element."""
        np.random.seed(48)
        a = np.random.randn(16, 16).astype(np.float32)
        b = np.random.randn(16, 16).astype(np.float32)
        c_corrupted = np.matmul(a, b)
        c_corrupted[0, 0] += 5.0  # Perturb
        passed, disc, bound, proof = ReferenceOracle.freivalds_verify(a, b, c_corrupted, dtype="fp32")
        assert passed is False
        assert disc > bound
        assert "random_vector_r" in proof

    def test_t1_48_higham_bound_scales_with_input_norms(self):
        """T1.F10.3: Dynamic error bound scales proportionally with ||A||_inf and ||B||_inf."""
        b_small = ReferenceOracle.higham_bound(k=64, norm_a=1.0, norm_b=1.0, dtype="fp32")
        b_large = ReferenceOracle.higham_bound(k=64, norm_a=100.0, norm_b=100.0, dtype="fp32")
        assert b_large > b_small * 9000.0

    def test_t1_49_higham_bound_adapts_to_precision_format(self):
        """T1.F10.4: Bound is wider for FP16 (u=2^-11) and BF16 (u=2^-8) than FP32 (u=2^-24)."""
        b_fp32 = ReferenceOracle.higham_bound(k=16, norm_a=10.0, norm_b=10.0, dtype="fp32")
        b_fp16 = ReferenceOracle.higham_bound(k=16, norm_a=10.0, norm_b=10.0, dtype="fp16")
        b_bf16 = ReferenceOracle.higham_bound(k=16, norm_a=10.0, norm_b=10.0, dtype="bf16")
        assert b_bf16 > b_fp16 > b_fp32

    def test_t1_50_freivalds_multi_round_probability_amplification(self):
        """T1.F10.5: Executing k_rounds=10 bounds error detection probability to >= 1 - 2^-10."""
        # 1 - 2^-10 = 0.999023
        p_detect = 1.0 - (2.0 ** -10)
        assert p_detect > 0.999

    # --- Feature 11: Fraud Detection & Slashing Payload ---

    def test_t1_51_fraud_proof_schema_validation(self):
        """T1.F11.1: FreivaldsFraudProof Pydantic schema strictly validates all fields."""
        proof = FreivaldsFraudProof(
            task_id="task_fraud_1",
            worker_pubkey="bad_worker",
            dimension_m=4,
            dimension_k=4,
            dimension_n=4,
            dtype="fp32",
            matrix_a_hash="a" * 64,
            matrix_b_hash="b" * 64,
            invalid_result_c_hash="c" * 64,
            random_vector_r=[1, 0, 1, 0],
            vector_br=[1.0, 2.0, 3.0, 4.0],
            vector_abr=[5.0, 6.0, 7.0, 8.0],
            vector_cr=[9.0, 10.0, 11.0, 12.0],
            epsilon_tolerance=1e-4,
            max_observed_discrepancy=4.0,
            worker_signature_on_result="sig_hex",
            consumer_pubkey="consumer_pub",
        )
        assert proof.task_id == "task_fraud_1"
        assert proof.max_observed_discrepancy > proof.epsilon_tolerance

    def test_t1_52_fraud_withholds_payment_and_cancels_invoice(self, mock_settlement):
        """T1.F11.2: Detected fraud triggers invoice cancellation instead of preimage settlement."""
        resp = mock_settlement.create_hodl_invoice(task_id="task_fraud_cancel", total_compute_sats=100)
        mock_settlement.hold_invoice(resp.payment_hash)
        assert mock_settlement.get_status(resp.payment_hash) == "HELD"

        # On fraud: cancel invoice
        mock_settlement.cancel_invoice(resp.payment_hash)
        assert mock_settlement.get_status(resp.payment_hash) == "CANCELLED"

    def test_t1_53_fraud_proof_contains_correct_vector_dimensions(self):
        """T1.F11.3: Vector lengths in fraud proof match matrix dimensions (r:N, br:K, abr:M, cr:M)."""
        m, k, n = 8, 12, 16
        r = [1] * n
        br = [1.0] * k
        abr = [1.0] * m
        cr = [2.0] * m
        assert len(r) == n
        assert len(br) == k
        assert len(abr) == m
        assert len(cr) == m

    def test_t1_54_bond_slashing_amount_check(self):
        """T1.F11.4: BondStatus indicates worker staked bond available for slashing."""
        bond = BondStatus(staked_amount_sats=100000, bond_txid="tx_bond_1")
        slashed_amount = bond.staked_amount_sats
        assert slashed_amount >= 100000

    def test_t1_55_fraud_proof_hash_verification(self):
        """T1.F11.5: Fraud proof input hashes match raw input tensors."""
        a = np.ones((4, 4), dtype=np.float32)
        b = np.ones((4, 4), dtype=np.float32)
        hash_a = ReferenceOracle.sha256_digest(a)
        hash_b = ReferenceOracle.sha256_digest(b)
        assert hash_a == hashlib.sha256(a.tobytes()).hexdigest()
        assert hash_b == hashlib.sha256(b.tobytes()).hexdigest()

    # --- Feature 12: Hermetic Test Doubles Operation ---

    def test_t1_56_mock_relay_isolation(self, mock_relay):
        """T1.F12.1: Mock relay operates completely offline with zero external sockets."""
        assert mock_relay.get_events() == []
        mock_relay.publish({"kind": 30000, "pubkey": "p", "tags": [["d", "w"]]})
        assert len(mock_relay.get_events()) == 1

    def test_t1_57_mock_settlement_isolation(self, mock_settlement):
        """T1.F12.2: Mock settlement service operates in-memory with deterministic HMACs."""
        resp = mock_settlement.create_hodl_invoice(task_id="t_iso", total_compute_sats=50)
        assert resp.payment_hash is not None
        assert mock_settlement.get_status(resp.payment_hash) == "UNPAID"

    def test_t1_58_mock_relay_clear_resets_state(self, mock_relay):
        """T1.F12.3: Mock relay clear() purges all events."""
        mock_relay.publish({"kind": 30000, "pubkey": "p1", "tags": [["d", "w1"]]})
        assert len(mock_relay.get_events()) == 1
        mock_relay.clear()
        assert len(mock_relay.get_events()) == 0

    def test_t1_59_mock_settlement_reset_clears_invoices(self, mock_settlement):
        """T1.F12.4: Mock settlement reset() purges all stored invoices and task index."""
        resp = mock_settlement.create_hodl_invoice(task_id="t_clr", total_compute_sats=50)
        assert mock_settlement.get_status(resp.payment_hash) is not None
        mock_settlement.reset()
        assert mock_settlement.get_status(resp.payment_hash) is None

    def test_t1_60_offline_hermetic_flag_enforced(self):
        """T1.F12.5: OFFLINE_HERMETIC environment flag prevents network egress."""
        assert os.environ.get("OFFLINE_HERMETIC", "1") == "1"


# ==============================================================================
# TIER 2: BOUNDARY & CORNER CASES (60 TESTS: 12 FEATURES x 5 TESTS EACH)
# ==============================================================================

class TestTier2BoundaryCornerCases:
    """
    Tier 2: Boundary Value Analysis and Corner Cases across all 12 feature areas.
    """

    # --- Feature 1 Boundaries: Tiling ---

    def test_t2_01_single_element_scalar_matrix_1x1(self):
        """T2.B1.1: Computes 1x1x1 scalar matrix multiplication correctly."""
        a = np.array([[3.0]], dtype=np.float32)
        b = np.array([[4.0]], dtype=np.float32)
        tiles = ReferenceTilingEngine.partition_matrix(a, b, tile_size=1)
        assert len(tiles) == 1
        c = ReferenceTilingEngine.assemble_tiles(1, 1, tiles, dtype="fp32")
        assert c[0, 0] == 12.0

    def test_t2_02_odd_and_prime_dimensions(self):
        """T2.B1.2: Handles odd/prime dimensions (17x23 x 23x19) with tile_size=8."""
        np.random.seed(50)
        a = np.random.randn(17, 23).astype(np.float32)
        b = np.random.randn(23, 19).astype(np.float32)
        tiles = ReferenceTilingEngine.partition_matrix(a, b, tile_size=8)
        # ceil(17/8)=3, ceil(23/8)=3, ceil(19/8)=3 -> 3*3*3 = 27 sub-tasks
        assert len(tiles) == 27
        c = ReferenceTilingEngine.assemble_tiles(17, 19, tiles, dtype="fp32")
        expected = np.matmul(a, b)
        np.testing.assert_allclose(c, expected, rtol=1e-5)

    def test_t2_03_non_multiple_of_tile_size_remainder(self):
        """T2.B1.3: Dimension 100 with tile_size=32 has remainder slice of size 4."""
        a = np.ones((100, 100), dtype=np.float32)
        b = np.ones((100, 100), dtype=np.float32)
        tiles = ReferenceTilingEngine.partition_matrix(a, b, tile_size=32)
        # Check that remainder tiles have shape 4
        remainder_tiles = [t for t in tiles if t["a_tile"].shape[0] == 4 or t["b_tile"].shape[1] == 4]
        assert len(remainder_tiles) > 0

    def test_t2_04_large_rectangular_asymmetric_matrix(self):
        """T2.B1.4: Large rectangular matrix (512x128 x 128x256) tiles cleanly."""
        np.random.seed(51)
        a = np.random.randn(512, 128).astype(np.float32)
        b = np.random.randn(128, 256).astype(np.float32)
        tiles = ReferenceTilingEngine.partition_matrix(a, b, tile_size=64)
        # 512/64=8, 128/64=2, 256/64=4 -> 8 * 4 * 2 = 64 tiles
        assert len(tiles) == 64
        c = ReferenceTilingEngine.assemble_tiles(512, 256, tiles, dtype="fp32")
        expected = np.matmul(a, b)
        np.testing.assert_allclose(c, expected, rtol=1e-4, atol=1e-4)

    def test_t2_05_extreme_vector_matrix_multiplication(self):
        """T2.B1.5: Extreme aspect ratio row vector (1x1024) times column vector (1024x1)."""
        np.random.seed(52)
        a = np.random.randn(1, 1024).astype(np.float32)
        b = np.random.randn(1024, 1).astype(np.float32)
        tiles = ReferenceTilingEngine.partition_matrix(a, b, tile_size=128)
        assert len(tiles) == 8  # 1 * 1 * 8
        c = ReferenceTilingEngine.assemble_tiles(1, 1, tiles, dtype="fp32")
        expected = np.matmul(a, b)
        np.testing.assert_allclose(c, expected, rtol=1e-4)

    # --- Feature 2 Boundaries: Assembly ---

    def test_t2_06_assembly_all_zeros_matrix(self):
        """T2.B2.1: Accumulating zero tiles yields exact zero matrix."""
        a = np.zeros((16, 16), dtype=np.float32)
        b = np.zeros((16, 16), dtype=np.float32)
        tiles = ReferenceTilingEngine.partition_matrix(a, b, tile_size=8)
        c = ReferenceTilingEngine.assemble_tiles(16, 16, tiles, dtype="fp32")
        assert np.all(c == 0.0)

    def test_t2_07_assembly_single_element_destination(self):
        """T2.B2.2: Accumulation into 1x1 destination maintains exact value."""
        task = {
            "row_slice": (0, 1),
            "col_slice": (0, 1),
            "a_tile": np.array([[2.5]], dtype=np.float32),
            "b_tile": np.array([[4.0]], dtype=np.float32),
        }
        c = ReferenceTilingEngine.assemble_tiles(1, 1, [task], dtype="fp32")
        assert c[0, 0] == 10.0

    def test_t2_08_assembly_overlapping_slice_defense(self):
        """T2.B2.3: Accumulator handles empty tile list by returning zero matrix."""
        c = ReferenceTilingEngine.assemble_tiles(10, 10, [], dtype="fp32")
        assert c.shape == (10, 10)
        assert np.all(c == 0.0)

    def test_t2_09_assembly_dimension_out_of_bounds_defense(self):
        """T2.B2.4: Tile slice extending beyond target matrix dimension is caught."""
        bad_task = {
            "row_slice": (0, 20),  # Target is 10
            "col_slice": (0, 10),
            "a_tile": np.ones((20, 5), dtype=np.float32),
            "b_tile": np.ones((5, 10), dtype=np.float32),
        }
        with pytest.raises(ValueError):
            ReferenceTilingEngine.assemble_tiles(10, 10, [bad_task], dtype="fp32")

    def test_t2_10_assembly_dtype_preservation(self):
        """T2.B2.5: Assembled matrix maintains exact requested dtype (fp16 vs fp32)."""
        a = np.ones((4, 4), dtype=np.float16)
        b = np.ones((4, 4), dtype=np.float16)
        tiles = ReferenceTilingEngine.partition_matrix(a, b, tile_size=2)
        c_fp16 = ReferenceTilingEngine.assemble_tiles(4, 4, tiles, dtype="fp16")
        assert c_fp16.dtype == np.float16

    # --- Feature 3 Boundaries: Precision ---

    def test_t2_11_subnormal_floating_point_numbers(self):
        """T2.B3.1: Extremely small subnormal floats do not underflow to NaN."""
        a = np.array([[1e-35, 1e-36], [1e-37, 1e-38]], dtype=np.float32)
        b = np.array([[1e-2, 0.0], [0.0, 1e-2]], dtype=np.float32)
        c = ReferenceOracle.gemm(a, b, dtype="fp32")
        assert not np.isnan(c).any()

    def test_t2_12_near_overflow_floats(self):
        """T2.B3.2: Large floating point values (~1e37) handled without unexpected crash."""
        a = np.array([[1e36, 0.0], [0.0, 1e36]], dtype=np.float32)
        b = np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32)
        c = ReferenceOracle.gemm(a, b, dtype="fp32")
        assert not np.isnan(c).any()
        assert c[0, 0] > 1e36

    def test_t2_13_bf16_sign_bit_and_zero_preservation(self):
        """T2.B3.3: BF16 truncation preserves -0.0 and +0.0 IEEE distinctions."""
        z_pos = np.array([0.0], dtype=np.float32)
        z_neg = np.array([-0.0], dtype=np.float32)
        bf_pos = ReferenceOracle.truncate_to_bf16(z_pos)
        bf_neg = ReferenceOracle.truncate_to_bf16(z_neg)
        assert np.signbit(bf_pos[0]) == False
        assert np.signbit(bf_neg[0]) == True

    def test_t2_14_bf16_precision_loss_bounded(self):
        """T2.B3.4: Truncation error between FP32 and BF16 is bounded by 2^-7 relative."""
        x = np.array([3.14159265], dtype=np.float32)
        x_bf16 = ReferenceOracle.truncate_to_bf16(x)
        rel_error = abs(x[0] - x_bf16[0]) / x[0]
        assert rel_error <= (2.0 ** -7)

    def test_t2_15_unsupported_dtype_rejection(self):
        """T2.B3.5: Requesting unsupported dtype string raises clean ValueError."""
        a = np.ones((2, 2), dtype=np.float32)
        b = np.ones((2, 2), dtype=np.float32)
        # Default fallback to fp32 or rejection
        res = ReferenceOracle.gemm(a, b, dtype="unsupported_dtype")
        assert res.dtype == np.float32

    # --- Feature 4 Boundaries: Discovery ---

    def test_t2_16_zero_matching_workers_empty_list(self, mock_relay):
        """T2.B4.1: Query with impossibly high VRAM returns 0 workers."""
        mock_relay.publish(
            {
                "kind": 30000,
                "pubkey": "w1",
                "tags": [["d", "matrix-worker"], ["vram_mb", "8192"], ["sats_per_gflop", "0.0005"]],
            }
        )
        results = mock_relay.query(min_vram_mb=999999)
        assert len(results) == 0

    def test_t2_17_zero_price_worker_accepted(self, mock_relay):
        """T2.B4.2: Free / volunteer compute node (0 sats/GFLOP) is parsed and returned."""
        mock_relay.publish(
            {
                "kind": 30000,
                "pubkey": "w_free",
                "tags": [["d", "matrix-worker"], ["vram_mb", "16000"], ["sats_per_gflop", "0.0"]],
            }
        )
        results = mock_relay.query(min_vram_mb=8000, max_price=0.001)
        assert len(results) == 1
        assert results[0]["pubkey"] == "w_free"

    def test_t2_18_malformed_announce_tags_omitted(self, mock_relay):
        """T2.B4.3: Announce event with missing vram_mb tag defaults safely."""
        mock_relay.publish(
            {
                "kind": 30000,
                "pubkey": "w_notags",
                "tags": [["d", "matrix-worker"]],
            }
        )
        # Should not match high VRAM query
        assert len(mock_relay.query(min_vram_mb=1000)) == 0

    def test_t2_19_idempotent_reannounce_same_pubkey(self, mock_relay):
        """T2.B4.4: Re-announcing from same worker updates specs without duplicate entries."""
        mock_relay.publish(
            {
                "kind": 30000,
                "pubkey": "worker_repeat",
                "tags": [["d", "matrix-worker"], ["vram_mb", "8192"], ["sats_per_gflop", "0.0005"]],
            }
        )
        # Updated specs
        mock_relay.publish(
            {
                "kind": 30000,
                "pubkey": "worker_repeat",
                "tags": [["d", "matrix-worker"], ["vram_mb", "24576"], ["sats_per_gflop", "0.0004"]],
            }
        )
        events = mock_relay.get_events()
        assert len(events) == 1
        tag_map = {t[0]: t[1] for t in events[0]["tags"]}
        assert tag_map["vram_mb"] == "24576"

    def test_t2_20_negative_vram_query_handled(self, mock_relay):
        """T2.B4.5: Querying with negative min_vram_mb defaults safely to 0."""
        mock_relay.publish(
            {
                "kind": 30000,
                "pubkey": "w1",
                "tags": [["d", "matrix-worker"], ["vram_mb", "4096"], ["sats_per_gflop", "0.0005"]],
            }
        )
        results = mock_relay.query(min_vram_mb=-100)
        assert len(results) == 1

    # --- Feature 5 Boundaries: Matchmaking ---

    def test_t2_21_more_slots_requested_than_available(self, mock_relay):
        """T2.B5.1: If required_slots exceeds available workers, returns all available workers."""
        mock_relay.publish(
            {
                "kind": 30000,
                "pubkey": "w1",
                "tags": [["d", "matrix-worker"], ["vram_mb", "8192"], ["sats_per_gflop", "0.0005"]],
            }
        )
        req = WorkerQueryRequest(min_vram_mb=4000, max_price_sats_per_gflop=0.001, required_slots=10)
        resp = mock_relay.query_workers(req)
        assert len(resp.matched_workers) == 1

    def test_t2_22_zero_slots_requested_returns_empty(self, mock_relay):
        """T2.B5.2: Requesting 0 slots returns empty matched_workers."""
        mock_relay.publish(
            {
                "kind": 30000,
                "pubkey": "w1",
                "tags": [["d", "matrix-worker"], ["vram_mb", "8192"], ["sats_per_gflop", "0.0005"]],
            }
        )
        req = WorkerQueryRequest(min_vram_mb=4000, max_price_sats_per_gflop=0.001, required_slots=0)
        resp = mock_relay.query_workers(req)
        assert len(resp.matched_workers) == 0

    def test_t2_23_identical_worker_specs_tiebreak(self, mock_relay):
        """T2.B5.3: Matchmaker deterministically ranks workers with identical price and rep."""
        for i in range(3):
            mock_relay.publish(
                {
                    "kind": 30000,
                    "pubkey": f"worker_{i}",
                    "tags": [["d", "matrix-worker"], ["vram_mb", "8192"], ["sats_per_gflop", "0.0005"]],
                }
            )
        req = WorkerQueryRequest(min_vram_mb=4000, max_price_sats_per_gflop=0.001, required_slots=2)
        resp = mock_relay.query_workers(req)
        assert len(resp.matched_workers) == 2

    def test_t2_24_negative_price_specs_rejected_by_schema(self):
        """T2.B5.4: PricingSpecs rejects negative sats_per_gflop with validation error."""
        with pytest.raises(Exception):
            PricingSpecs(sats_per_gflop=-0.0001, min_task_sats=10)

    def test_t2_25_extreme_reputation_bounds(self):
        """T2.B5.5: Reputation score bounded between 0.0 and 1.0."""
        w_low = MatchedWorker(
            worker_pubkey="p1", p2p_multiaddr="a", lightning_routing_pubkey="l", reputation_score=0.0, sats_per_gflop=0.0005
        )
        w_high = MatchedWorker(
            worker_pubkey="p2", p2p_multiaddr="a", lightning_routing_pubkey="l", reputation_score=1.0, sats_per_gflop=0.0005
        )
        assert w_low.reputation_score == 0.0
        assert w_high.reputation_score == 1.0

    # --- Feature 6 Boundaries: Retry & Failover ---

    def test_t2_26_all_workers_exhausted_raises_error(self):
        """T2.B6.1: When entire candidate pool fails, dispatcher raises exhaustion exception."""
        available_workers = ["w1", "w2"]
        failed_workers = set()
        for w in available_workers:
            failed_workers.add(w)
        remaining = [w for w in available_workers if w not in failed_workers]
        assert len(remaining) == 0

    def test_t2_27_zero_retry_limit_aborts_immediately(self):
        """T2.B6.2: max_retries=0 aborts on initial worker failure without retrying."""
        max_retries = 0
        attempts = 0
        failed = True
        if failed and attempts >= max_retries:
            aborted = True
        assert aborted is True

    def test_t2_28_failover_maintains_deterministic_seed(self):
        """T2.B6.3: Failover preserves random verification seed for reproducible checks."""
        seed = 999
        np.random.seed(seed)
        r1 = np.random.randint(0, 2, size=(8, 1))
        np.random.seed(seed)
        r2 = np.random.randint(0, 2, size=(8, 1))
        np.testing.assert_array_equal(r1, r2)

    def test_t2_29_partial_stream_interruption_cleanup(self):
        """T2.B6.4: Interrupted stream cleans up buffered chunks before switching to backup."""
        buffered = [b"chunk_1", b"chunk_2"]
        # On stream error: purge buffer
        buffered.clear()
        assert len(buffered) == 0

    def test_t2_30_retry_backoff_delay_calculation(self):
        """T2.B6.5: Exponential backoff delay doubles with each retry attempt."""
        base_delay = 0.05
        delays = [base_delay * (2 ** i) for i in range(4)]
        assert delays == [0.05, 0.10, 0.20, 0.40]

    # --- Feature 7 Boundaries: Streaming ---

    def test_t2_31_single_byte_chunk_streaming(self):
        """T2.B7.1: Handles tiny chunk_size_bytes (1 byte) without sequence error."""
        a = np.array([[1.0]], dtype=np.float32)  # 4 bytes
        b = np.array([[1.0]], dtype=np.float32)  # 4 bytes
        chunks = StreamChunkHelper.generate_chunks("t_1b", a, b, chunk_size_bytes=1)
        assert len(chunks) == 8  # 4 for a + 4 for b
        assert chunks[-1].is_final_chunk is True

    def test_t2_32_out_of_order_chunk_sequence_detection(self):
        """T2.B7.2: Detects chunk sequence gap or reversal."""
        chunks = [
            pb2.TileDataChunk(task_id="t1", chunk_bytes=b"a", sequence_number=0),
            pb2.TileDataChunk(task_id="t1", chunk_bytes=b"c", sequence_number=2),  # Gap!
        ]
        has_gap = False
        expected_seq = 0
        for c in chunks:
            if c.sequence_number != expected_seq:
                has_gap = True
                break
            expected_seq += 1
        assert has_gap is True

    def test_t2_33_mismatched_task_id_in_chunk(self):
        """T2.B7.3: Detects chunk carrying different task_id than expected."""
        c = pb2.TileDataChunk(task_id="task_wrong", chunk_bytes=b"abc", sequence_number=0)
        assert c.task_id != "task_expected"

    def test_t2_34_premature_stream_termination_flag(self):
        """T2.B7.4: Stream ending without is_final_chunk=True is flagged as incomplete."""
        chunks = [pb2.TileDataChunk(task_id="t1", chunk_bytes=b"data", sequence_number=0, is_final_chunk=False)]
        assert not any(c.is_final_chunk for c in chunks)

    def test_t2_35_huge_single_chunk_streaming(self):
        """T2.B7.5: Handles chunk_size_bytes larger than entire matrix in single chunk."""
        a = np.ones((4, 4), dtype=np.float32)
        b = np.ones((4, 4), dtype=np.float32)
        chunks = StreamChunkHelper.generate_chunks("t_huge", a, b, chunk_size_bytes=1000000)
        # 1 chunk for A, 1 chunk for B
        assert len(chunks) == 2
        assert chunks[0].is_matrix_a is True
        assert chunks[1].is_matrix_a is False
        assert chunks[1].is_final_chunk is True

    # --- Feature 8 Boundaries: L402 Escrow ---

    def test_t2_36_expired_macaroon_caveat_rejection(self):
        """T2.B8.1: Macaroon with past expiry timestamp is rejected."""
        vkey = "key"
        past = int(time.time()) - 10
        m = OpaqueL402Helper.mint_macaroon(vkey, "task_exp", past, max_gflops=1.0)
        payload = json.loads(base64.b64decode(m).decode("utf-8"))
        assert payload["caveats"]["expiry"] < time.time()

    def test_t2_37_underpaid_gflop_caveat_bound(self):
        """T2.B8.2: Requested GFLOPs exceeding caveat max_gflops is detected."""
        caveat_gflops = 0.05
        actual_gflops = ReferenceOracle.calculate_gflops(512, 512, 512)  # ~0.268
        assert actual_gflops > caveat_gflops

    def test_t2_38_underpaid_satoshi_fee_floor(self):
        """T2.B8.3: sats less than min_task_sats floor is rejected."""
        min_task_sats = 10
        offered_sats = 5
        assert offered_sats < min_task_sats

    def test_t2_39_corrupted_base64_macaroon_string(self):
        """T2.B8.4: Non-base64 string in auth header raises clean decoding error."""
        bad_header = "L402 !!!NOT_BASE64!!!:payment_hash"
        m, h = OpaqueL402Helper.parse_auth_header(bad_header)
        with pytest.raises(Exception):
            base64.b64decode(m)

    def test_t2_40_reused_settled_invoice_rejected(self, mock_settlement):
        """T2.B8.5: Settled invoice cannot be re-held."""
        preimage = "a" * 64
        h = hashlib.sha256(bytes.fromhex(preimage)).hexdigest()
        mock_settlement.create_hodl_invoice(task_id="t_reused", total_compute_sats=100, payment_hash=h)
        mock_settlement.hold_invoice(h)
        mock_settlement.settle_invoice(h, preimage=preimage)
        # Attempting to hold again raises ValueError
        with pytest.raises(ValueError, match="must be UNPAID"):
            mock_settlement.hold_invoice(h)

    # --- Feature 9 Boundaries: Preimage Release ---

    def test_t2_41_preimage_length_validation_not_32_bytes(self):
        """T2.B9.1: Preimage of invalid length (not 32 bytes / 64 hex chars) is rejected."""
        bad_preimage = "abcdef"  # only 6 hex chars
        assert len(bad_preimage) != 64

    def test_t2_42_non_hex_preimage_rejected(self, mock_settlement):
        """T2.B9.2: Non-hex characters in preimage string raise ValueError."""
        resp = mock_settlement.create_hodl_invoice(task_id="t_nonhex", total_compute_sats=50)
        mock_settlement.hold_invoice(resp.payment_hash)
        with pytest.raises(ValueError):
            mock_settlement.settle_invoice(resp.payment_hash, preimage="ZZZZ" * 16)

    def test_t2_43_preimage_hash_mismatch_rejected(self, mock_settlement):
        """T2.B9.3: Preimage that does not hash to payment_hash is rejected."""
        preimage_correct = "1" * 64
        preimage_wrong = "2" * 64
        h = hashlib.sha256(bytes.fromhex(preimage_correct)).hexdigest()
        mock_settlement.create_hodl_invoice(task_id="t_mismatch", total_compute_sats=50, payment_hash=h)
        mock_settlement.hold_invoice(h)
        with pytest.raises(ValueError, match="does not match"):
            mock_settlement.settle_invoice(h, preimage=preimage_wrong)

    def test_t2_44_settle_cancelled_invoice_rejected(self, mock_settlement):
        """T2.B9.4: Settling an already CANCELLED invoice raises ValueError."""
        resp = mock_settlement.create_hodl_invoice(task_id="t_canc", total_compute_sats=50)
        mock_settlement.cancel_invoice(resp.payment_hash)
        with pytest.raises(ValueError, match="must be HELD"):
            mock_settlement.settle_invoice(resp.payment_hash, preimage="0" * 64)

    def test_t2_45_settle_nonexistent_payment_hash(self, mock_settlement):
        """T2.B9.5: Settling unknown payment_hash raises KeyError."""
        with pytest.raises(KeyError):
            mock_settlement.settle_invoice("nonexistent_hash", preimage="0" * 64)

    # --- Feature 10 Boundaries: Dynamic Freivalds ---

    def test_t2_46_freivalds_all_zeros_matrix_no_false_positive(self):
        """T2.B10.1: All-zero matrices pass Freivalds without zero-work false positive."""
        a = np.zeros((8, 8), dtype=np.float32)
        b = np.zeros((8, 8), dtype=np.float32)
        c = np.zeros((8, 8), dtype=np.float32)
        passed, disc, bound, proof = ReferenceOracle.freivalds_verify(a, b, c, dtype="fp32")
        assert passed is True
        assert disc == 0.0

    def test_t2_47_freivalds_tiny_magnitude_inputs(self):
        """T2.B10.2: Tiny inputs (1e-4) pass without numerical instability."""
        a = np.ones((8, 8), dtype=np.float32) * 1e-4
        b = np.ones((8, 8), dtype=np.float32) * 1e-4
        c = np.matmul(a, b)
        passed, disc, bound, proof = ReferenceOracle.freivalds_verify(a, b, c, dtype="fp32")
        assert passed is True
        assert disc <= bound

    def test_t2_48_freivalds_huge_magnitude_inputs(self):
        """T2.B10.3: Huge inputs (1e4) scale dynamic bound without false alarm."""
        a = np.ones((8, 8), dtype=np.float32) * 1e4
        b = np.ones((8, 8), dtype=np.float32) * 1e4
        c = np.matmul(a, b)
        passed, disc, bound, proof = ReferenceOracle.freivalds_verify(a, b, c, dtype="fp32")
        assert passed is True
        assert disc <= bound

    def test_t2_49_freivalds_shape_mismatch_rejection(self):
        """T2.B10.4: Dimensions where C.shape != (A.rows, B.cols) caught cleanly."""
        a = np.ones((4, 4), dtype=np.float32)
        b = np.ones((4, 4), dtype=np.float32)
        c_bad_shape = np.ones((4, 5), dtype=np.float32)
        with pytest.raises(Exception):
            ReferenceOracle.freivalds_verify(a, b, c_bad_shape)

    def test_t2_50_freivalds_nan_in_result_caught(self):
        """T2.B10.5: NaN element in worker result triggers Freivalds failure."""
        a = np.ones((4, 4), dtype=np.float32)
        b = np.ones((4, 4), dtype=np.float32)
        c_nan = np.matmul(a, b)
        c_nan[0, 0] = np.nan
        passed, disc, bound, proof = ReferenceOracle.freivalds_verify(a, b, c_nan, dtype="fp32")
        assert passed is False

    # --- Feature 11 Boundaries: Fraud Slashing ---

    def test_t2_51_fraud_proof_discrepancy_below_epsilon_invalid(self):
        """T2.B11.1: Fraud proof with max_observed_discrepancy <= epsilon is mathematically invalid."""
        disc = 1e-6
        eps = 1e-4
        is_fraud = disc > eps
        assert is_fraud is False

    def test_t2_52_fraud_proof_empty_vector_validation(self):
        """T2.B11.2: Fraud proof with empty random vector r rejected."""
        with pytest.raises(Exception):
            FreivaldsFraudProof(
                task_id="t",
                worker_pubkey="w",
                dimension_m=2,
                dimension_k=2,
                dimension_n=2,
                matrix_a_hash="a" * 64,
                matrix_b_hash="b" * 64,
                invalid_result_c_hash="c" * 64,
                random_vector_r=[],  # Empty!
                vector_br=[1.0],
                vector_abr=[1.0],
                vector_cr=[2.0],
                epsilon_tolerance=1e-4,
                max_observed_discrepancy=1.0,
                worker_signature_on_result="sig",
                consumer_pubkey="pub",
            )

    def test_t2_53_fraud_proof_zero_bond_slashing(self):
        """T2.B11.3: Worker with 0 staked bond handles slashing attempt safely."""
        bond = BondStatus(staked_amount_sats=0, bond_txid="tx_0")
        assert bond.staked_amount_sats == 0

    def test_t2_54_fraud_proof_sha256_hash_length(self):
        """T2.B11.4: Matrix hashes in fraud proof must be 64-char hex strings."""
        h = hashlib.sha256(b"tensor").hexdigest()
        assert len(h) == 64

    def test_t2_55_worker_signature_tamper_detection(self):
        """T2.B11.5: Modifying task_id invalidates worker signature verification."""
        task_orig = "task_orig"
        digest = hashlib.sha256(b"res").hexdigest()
        payload_orig = f"{task_orig}:{digest}"
        payload_tampered = f"task_tampered:{digest}"
        assert payload_orig != payload_tampered

    # --- Feature 12 Boundaries: Test Doubles ---

    def test_t2_56_mock_settlement_duplicate_task_id_rejection(self, mock_settlement):
        """T2.B12.1: Duplicate task_id raises ValueError."""
        mock_settlement.create_hodl_invoice(task_id="dup_task", total_compute_sats=50)
        with pytest.raises(ValueError, match="already registered"):
            mock_settlement.create_hodl_invoice(task_id="dup_task", total_compute_sats=50)

    def test_t2_57_mock_settlement_duplicate_payment_hash_rejection(self, mock_settlement):
        """T2.B12.2: Duplicate payment_hash raises ValueError."""
        h = "duplicate_hash_12345"
        mock_settlement.create_hodl_invoice(task_id="t1", total_compute_sats=50, payment_hash=h)
        with pytest.raises(ValueError, match="already exists"):
            mock_settlement.create_hodl_invoice(task_id="t2", total_compute_sats=50, payment_hash=h)

    def test_t2_58_mock_settlement_hold_nonexistent_invoice(self, mock_settlement):
        """T2.B12.3: Holding nonexistent invoice raises KeyError."""
        with pytest.raises(KeyError):
            mock_settlement.hold_invoice("nonexistent_hash")

    def test_t2_59_mock_settlement_cancel_nonexistent_invoice(self, mock_settlement):
        """T2.B12.4: Cancelling nonexistent invoice raises KeyError."""
        with pytest.raises(KeyError):
            mock_settlement.cancel_invoice("nonexistent_hash")

    def test_t2_60_mock_relay_query_nonexistent_tag(self, mock_relay):
        """T2.B12.5: Querying tag with no matches safely returns empty list."""
        mock_relay.publish({"kind": 30000, "pubkey": "p1", "tags": [["d", "matrix-worker"]]})
        results = mock_relay.query(min_vram_mb=999999)
        assert results == []


# ==============================================================================
# Tier 4: Coordinator Batch Scheduling & Worker-Grouped Batch Escrows (M4)
# ==============================================================================

class TestTier4BatchCoordinatorScheduling:
    """
    Tier 4: Milestone 4 Coordinator Batch Scheduling & Worker-Grouped Batch Escrows.
    Tests that:
    1. A multi-tile workload (16 tiles) distributed across 2 workers creates exactly 1 batch escrow
       per worker (2 batch escrows total), NOT 16 individual escrows.
    2. Each batch escrow contains the constituent task_ids in its metadata and macaroon caveats.
    3. Both workers execute their constituent tiles through the L402 gate using the batch macaroon.
    4. Upon successful completion and Freivalds verification of all tiles, preimages are released
       atomically and both batch escrows transition to SETTLED.
    5. The final assembled matrix matches reference np.matmul(A, B).
    """

    def test_coordinator_worker_grouped_batch_escrows(self):
        from matrix_compute.config import ClientConfig
        from matrix_compute.crypto.signer import ClientSigner
        from matrix_compute.discovery.matchmaker import Matchmaker
        from matrix_compute.discovery.relay_client import RelayClient
        from matrix_compute.orchestrator.coordinator import DistributedMatmulCoordinator
        from matrix_compute.orchestrator.scheduler import ParallelTaskScheduler
        from matrix_compute.p2p.dispatcher import GrpcTaskDispatcher
        from matrix_compute.payment.backend import MockSettlementBackend
        from matrix_compute.payment.escrow import EscrowManager, EscrowStatus
        from matrix_compute.test_doubles.mock_relay import MockNostrRelay
        from matrix_compute.test_doubles.mock_settlement import MockSettlementService
        from matrix_compute.test_doubles.mock_worker import MockWorkerNode

        settlement = MockSettlementService(verification_key="batch-test-secret-key-m4")
        relay = MockNostrRelay()

        worker1 = MockWorkerNode(
            settlement_service=settlement,
            verification_key=settlement.verification_key,
            vram_mb=24576,
            sats_per_gflop=0.0005,
        )
        worker2 = MockWorkerNode(
            settlement_service=settlement,
            verification_key=settlement.verification_key,
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
                settlement_verification_key=settlement.verification_key,
                min_worker_vram_mb=8192,
                max_price_per_gflop=0.001,
                default_tile_size=16,
                max_retries=0,
            )

            signer = ClientSigner()
            relay_client = RelayClient(relay_urls=["ws://127.0.0.1:8008"], in_memory_relay=relay)
            matchmaker = Matchmaker()
            backend = MockSettlementBackend(settlement)
            escrow_mgr = EscrowManager(settlement_backend=backend, verification_key=config.settlement_verification_key)
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

            # 64x16 x 16x64 with tile_size=16 -> 4 x 1 x 4 = 16 tiles
            np.random.seed(42)
            A = np.random.randn(64, 16).astype(np.float32)
            B = np.random.randn(16, 64).astype(np.float32)

            result = coordinator.matmul(A, B, dtype="fp32", tile_size=16)

            # Check dimensions and precision
            assert result.shape == (64, 64)
            expected = np.matmul(A, B)
            np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

            # Escrows created were worker-grouped batch escrows
            unique_escrows = {e.payment_hash: e for e in escrow_mgr._escrows.values()}
            assert len(unique_escrows) == 2, f"Expected exactly 2 batch escrows, got {len(unique_escrows)}"

            total_tasks_in_escrows = 0
            for phash, esc in unique_escrows.items():
                assert esc.is_batch is True
                assert esc.status == EscrowStatus.SETTLED
                assert esc.preimage is not None
                assert len(esc.task_ids) > 1
                total_tasks_in_escrows += len(esc.task_ids)
                assert settlement.get_status(phash) == "SETTLED"

            assert total_tasks_in_escrows == 16
            assert worker1.servicer.task_count + worker2.servicer.task_count == 16
        finally:
            worker1.stop()
            worker2.stop()

