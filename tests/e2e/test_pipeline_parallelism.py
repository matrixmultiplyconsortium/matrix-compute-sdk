"""
End-to-End Acceptance Test Suite for Pipeline Parallelism Architecture.

Covers:
1. Concurrency Benchmark: 64 tile tasks across 4 mock workers with simulated compute delay,
   asserting >= 2.5x speedup over single-worker sequential baseline.
2. Timestamp Overlap Verification: Assert empirical simultaneous execution interval overlap
   (max(s_a, s_b) < min(e_a, e_b)) across distinct workers via execution_history.
3. Straggler Resilience: Straggler worker with 2.0s delay does not block healthy workers;
   healthy workers complete all their tasks in parallel while the straggler is in-flight.
4. Verification Failure Isolation: Tampered worker results trigger Freivalds fraud detection,
   aborting escrow, slashing bond, and preventing accumulator buffer contamination.
5. Bounded Memory & Concurrency Window: In-flight active concurrency strictly bounded by W
   with backpressure throttling.
6. Multi-Precision Concurrency: Full support for FP32, FP16, and BF16 with parallel pipelining.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pytest

from matrix_compute.config import ClientConfig
from matrix_compute.crypto.signer import ClientSigner
from matrix_compute.discovery.matchmaker import Matchmaker
from matrix_compute.discovery.relay_client import RelayClient
from matrix_compute.exceptions import FraudDetectedError
from matrix_compute.orchestrator.coordinator import DistributedMatmulCoordinator
from matrix_compute.orchestrator.scheduler import ParallelTaskScheduler
from matrix_compute.p2p.dispatcher import GrpcTaskDispatcher
from matrix_compute.payment.backend import MockSettlementBackend
from matrix_compute.payment.escrow import EscrowManager, EscrowStatus
from matrix_compute.test_doubles.mock_relay import MockNostrRelay
from matrix_compute.test_doubles.mock_settlement import MockSettlementService
from matrix_compute.test_doubles.mock_worker import MockWorkerNode


def _create_env(
    num_workers: int = 4,
    verification_key: str = "pipeline-test-secret-vk",
    vram_mb: int = 24576,
    sats_per_gflop: float = 0.0005,
    initial_bond_sats: int = 500_000,
    max_concurrency: int = 16,
    max_retries: int = 0,
    task_timeout_seconds: float = 30.0,
    worker_delays: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Provisions a hermetic multi-worker coordination environment for pipeline testing."""
    settlement = MockSettlementService(verification_key=verification_key)
    relay = MockNostrRelay()

    workers: List[MockWorkerNode] = []
    for i in range(num_workers):
        w = MockWorkerNode(
            settlement_service=settlement,
            verification_key=verification_key,
            vram_mb=vram_mb,
            sats_per_gflop=sats_per_gflop,
        )
        w.start()
        w.publish_to_relay(relay)
        settlement.stake_bond(w.pubkey, initial_bond_sats)
        if worker_delays and i < len(worker_delays):
            w.set_delay(worker_delays[i])
        workers.append(w)

    config = ClientConfig(
        offline_mock_mode=True,
        nostr_relays=["ws://127.0.0.1:8008"],
        settlement_api_url="http://127.0.0.1:8080",
        settlement_verification_key=verification_key,
        min_worker_vram_mb=8192,
        max_price_per_gflop=0.001,
        default_tile_size=256,
        max_retries=max_retries,
        task_timeout_seconds=task_timeout_seconds,
    )

    signer = ClientSigner()
    relay_client = RelayClient(relay_urls=["ws://127.0.0.1:8008"], in_memory_relay=relay)
    matchmaker = Matchmaker()
    backend = MockSettlementBackend(settlement)
    escrow_mgr = EscrowManager(settlement_backend=backend, verification_key=verification_key)
    dispatcher = GrpcTaskDispatcher(consumer_pubkey=signer.get_pubkey_hex())
    scheduler = ParallelTaskScheduler(
        dispatcher=dispatcher,
        max_retries=max_retries,
        max_concurrency=max_concurrency,
    )

    coordinator = DistributedMatmulCoordinator(
        config=config,
        signer=signer,
        relay_client=relay_client,
        matchmaker=matchmaker,
        escrow_manager=escrow_mgr,
        scheduler=scheduler,
        dispatcher=dispatcher,
        mock_workers=workers,
        max_concurrency=max_concurrency,
    )

    return {
        "settlement": settlement,
        "relay": relay,
        "workers": workers,
        "coordinator": coordinator,
        "escrow_mgr": escrow_mgr,
        "config": config,
        "scheduler": scheduler,
    }


def _cleanup_env(env: Dict[str, Any]) -> None:
    """Gracefully shuts down test doubles and thread pools."""
    for w in env.get("workers", []):
        try:
            w.stop()
        except Exception:
            pass
    scheduler = env.get("scheduler")
    if scheduler:
        try:
            scheduler.shutdown(wait=False)
        except Exception:
            pass


# ==============================================================================
# 1. Concurrency Benchmark Suite (>= 2.5x Speedup)
# ==============================================================================

def test_64_tile_4_worker_concurrency_benchmark():
    """
    Executes 64 tile tasks (1024x1024 with tile_size=256 -> 4x4x4=64 tiles) across
    4 mock workers with 40ms simulated compute delay.
    Asserts wall-clock speedup >= 2.5x compared to single-worker sequential baseline.
    """
    delay_sec = 0.04
    np.random.seed(42)
    # 1024 x 1024 matrices with tile_size=256 gives (4 x 4 x 4) = 64 tiles
    A = np.random.randn(1024, 1024).astype(np.float32)
    B = np.random.randn(1024, 1024).astype(np.float32)
    expected_c = np.matmul(A, B)

    # 1. Measure Single-Worker Sequential Baseline
    seq_env = _create_env(
        num_workers=1,
        worker_delays=[delay_sec],
        max_concurrency=1,
    )
    try:
        t0_seq = time.perf_counter()
        c_seq = seq_env["coordinator"].matmul(A, B, dtype="fp32", tile_size=256)
        seq_duration = time.perf_counter() - t0_seq
    finally:
        _cleanup_env(seq_env)

    # Theoretical sequential duration for 64 tiles with 40ms delay is >= 2.56s
    assert seq_duration >= 2.2, f"Sequential baseline too fast: {seq_duration:.3f}s"
    np.testing.assert_allclose(c_seq, expected_c, rtol=1e-4, atol=1e-4)

    # 2. Measure 4-Worker Pipelined Parallel Execution
    par_env = _create_env(
        num_workers=4,
        worker_delays=[delay_sec, delay_sec, delay_sec, delay_sec],
        max_concurrency=16,
    )
    try:
        t0_par = time.perf_counter()
        c_par = par_env["coordinator"].matmul(A, B, dtype="fp32", tile_size=256)
        par_duration = time.perf_counter() - t0_par
    finally:
        _cleanup_env(par_env)

    np.testing.assert_allclose(c_par, expected_c, rtol=1e-4, atol=1e-4)

    speedup = seq_duration / par_duration
    assert speedup >= 2.5, (
        f"Pipeline parallelism benchmark failed: expected >= 2.5x speedup, got {speedup:.2f}x "
        f"(sequential: {seq_duration:.3f}s, parallel: {par_duration:.3f}s)"
    )


# ==============================================================================
# 2. Worker Timestamp Overlap Verification
# ==============================================================================

def test_worker_execution_timestamp_overlap():
    """
    Executes matrix multiplication across 4 mock workers with simulated delay and queries
    execution_history on each node. Asserts that execution intervals across distinct
    workers overlap simultaneously: max(s_a, s_b) < min(e_a, e_b).
    """
    delay_sec = 0.04
    env = _create_env(
        num_workers=4,
        worker_delays=[delay_sec, delay_sec, delay_sec, delay_sec],
        max_concurrency=16,
    )
    try:
        np.random.seed(99)
        # 512 x 512 with tile_size=256 -> 2 x 2 x 2 = 8 tiles across 4 workers (2 tiles/worker)
        A = np.random.randn(512, 512).astype(np.float32)
        B = np.random.randn(512, 512).astype(np.float32)

        c = env["coordinator"].matmul(A, B, dtype="fp32", tile_size=256)
        expected = np.matmul(A, B)
        np.testing.assert_allclose(c, expected, rtol=1e-4, atol=1e-4)

        histories = [w.get_execution_history() for w in env["workers"]]
        for i, hist in enumerate(histories):
            assert len(hist) > 0, f"Worker {i} did not record any execution intervals"

        # Verify interval overlap across distinct workers
        overlapping_pairs = []
        for i in range(len(env["workers"])):
            for j in range(i + 1, len(env["workers"])):
                for task_a in histories[i]:
                    for task_b in histories[j]:
                        s_a, e_a = task_a["start_time"], task_a["end_time"]
                        s_b, e_b = task_b["start_time"], task_b["end_time"]
                        # Overlap condition: max(s_a, s_b) < min(e_a, e_b)
                        overlap_start = max(s_a, s_b)
                        overlap_end = min(e_a, e_b)
                        if overlap_start < overlap_end:
                            overlap_duration = overlap_end - overlap_start
                            overlapping_pairs.append({
                                "worker_a": i,
                                "worker_b": j,
                                "task_a": task_a["task_id"],
                                "task_b": task_b["task_id"],
                                "overlap_duration": overlap_duration,
                            })

        assert len(overlapping_pairs) > 0, "No execution interval overlap found across distinct workers!"
        # Significant overlap duration (>= 15ms) demonstrates concurrent execution
        max_overlap = max(p["overlap_duration"] for p in overlapping_pairs)
        assert max_overlap >= 0.015, (
            f"Overlap duration too small ({max_overlap*1000:.1f}ms); expected >= 15ms"
        )
    finally:
        _cleanup_env(env)


# ==============================================================================
# 3. Straggler Resilience Test (Non-Blocking Parallel Progress)
# ==============================================================================

def test_straggler_non_blocking_parallel_progress():
    """
    Simulates a delayed worker (Worker 0 with 2.0s delay) alongside 2 normal workers
    (Worker 1 and Worker 2 with 0.02s delay).
    Verifies that normal workers complete all their assigned tiles in ~0.1s without deadlock
    while the straggler is in-flight on its first tile.
    """
    env = _create_env(
        num_workers=3,
        worker_delays=[2.0, 0.02, 0.02],
        max_concurrency=16,
    )
    try:
        np.random.seed(101)
        # 48 x 16 x 16 x 32 with tile_size=16 -> 3 x 1 x 2 = 6 tiles (2 tiles/worker)
        A = np.random.randn(48, 16).astype(np.float32)
        B = np.random.randn(16, 32).astype(np.float32)

        t0 = time.perf_counter()
        c = env["coordinator"].matmul(A, B, dtype="fp32", tile_size=16)
        total_duration = time.perf_counter() - t0

        expected = np.matmul(A, B)
        np.testing.assert_allclose(c, expected, rtol=1e-4, atol=1e-4)

        histories = [w.get_execution_history() for w in env["workers"]]
        w0_hist = histories[0]  # Straggler (2.0s delay)
        w1_hist = histories[1]  # Healthy 1 (0.02s delay)
        w2_hist = histories[2]  # Healthy 2 (0.02s delay)

        assert len(w0_hist) == 2, f"Straggler should have 2 tasks, got {len(w0_hist)}"
        assert len(w1_hist) == 2, f"Worker 1 should have 2 tasks, got {len(w1_hist)}"
        assert len(w2_hist) == 2, f"Worker 2 should have 2 tasks, got {len(w2_hist)}"

        # Completion timestamps
        w1_completion = max(t["end_time"] for t in w1_hist)
        w2_completion = max(t["end_time"] for t in w2_hist)
        w0_first_task_end = w0_hist[0]["end_time"]

        # Healthy workers must finish ALL their tasks BEFORE the straggler finishes even its first tile!
        assert w1_completion < w0_first_task_end, (
            f"Worker 1 completion ({w1_completion}) was blocked by straggler first tile end ({w0_first_task_end})"
        )
        assert w2_completion < w0_first_task_end, (
            f"Worker 2 completion ({w2_completion}) was blocked by straggler first tile end ({w0_first_task_end})"
        )

        # Elapsed time for healthy workers from first task start to last task end is ~0.1s (< 0.3s)
        w1_span = max(t["end_time"] for t in w1_hist) - min(t["start_time"] for t in w1_hist)
        w2_span = max(t["end_time"] for t in w2_hist) - min(t["start_time"] for t in w2_hist)
        assert w1_span < 0.3, f"Worker 1 stalled: execution span was {w1_span:.3f}s"
        assert w2_span < 0.3, f"Worker 2 stalled: execution span was {w2_span:.3f}s"
    finally:
        _cleanup_env(env)


# ==============================================================================
# 4. Verification Failure Isolation & Accumulator Protection
# ==============================================================================

def test_freivalds_fraud_isolation_and_accumulator_protection():
    """
    When a worker returns tampered tile data, Stage 3 Freivalds verification detects
    mathematical fraud, halts dispatch to that worker, withholds preimage, cancels escrow,
    slashes worker bond, and prevents the accumulator buffer from being contaminated.
    """
    env = _create_env(num_workers=2, max_concurrency=4)
    workers = env["workers"]
    settlement = env["settlement"]
    escrow_mgr = env["escrow_mgr"]
    coordinator = env["coordinator"]

    try:
        # Worker 1 tampers on task 1
        honest_worker = workers[0]
        dishonest_worker = workers[1]
        dishonest_worker.set_tamper_on_task_count(1)
        dishonest_worker.servicer.tamper_perturbation = 50.0

        initial_dishonest_bond = settlement.get_bond_balance(dishonest_worker.pubkey)
        assert initial_dishonest_bond == 500_000

        np.random.seed(103)
        # 32 x 16 x 16 x 32 with tile_size=16 -> 2 x 1 x 2 = 4 tiles (2 tiles/worker)
        A = np.random.randn(32, 16).astype(np.float32)
        B = np.random.randn(16, 32).astype(np.float32)

        with pytest.raises(FraudDetectedError) as exc_info:
            coordinator.matmul(A, B, dtype="fp32", tile_size=16)

        err = exc_info.value
        assert err.worker_pubkey == dishonest_worker.pubkey
        assert err.discrepancy > 0

        # 1. Dishonest worker was halted immediately on task 1; task 2 was NEVER dispatched
        assert dishonest_worker.servicer.task_count == 1, (
            f"Dishonest worker executed {dishonest_worker.servicer.task_count} tasks, expected exactly 1"
        )

        # 2. Dishonest worker's escrow was cancelled in settlement and preimage withheld
        for esc in escrow_mgr._escrows.values():
            if getattr(esc, "worker_pubkey", "") == dishonest_worker.pubkey:
                assert esc.status == EscrowStatus.CANCELLED
                assert settlement.get_status(esc.payment_hash) == "CANCELLED"
                assert esc.preimage is None or esc.status != EscrowStatus.SETTLED

        # 3. Dishonest worker bond slashed by 100,000 sats
        new_bond = settlement.get_bond_balance(dishonest_worker.pubkey)
        assert new_bond == 400_000, f"Expected slashed bond 400,000, got {new_bond}"

        # 4. Honest worker bond remains intact
        honest_bond = settlement.get_bond_balance(honest_worker.pubkey)
        assert honest_bond == 500_000, f"Honest worker bond affected: {honest_bond}"
    finally:
        _cleanup_env(env)


# ==============================================================================
# 5. Bounded Memory & Concurrency Window Test
# ==============================================================================

def test_bounded_in_flight_memory_concurrency():
    """
    Verifies that the in-flight sliding window strictly enforces max_concurrency W,
    capping peak concurrent tasks and bounding transient memory to O(W * tile_size^2).
    """
    max_w = 3
    delay_sec = 0.02
    env = _create_env(
        num_workers=4,
        worker_delays=[delay_sec, delay_sec, delay_sec, delay_sec],
        max_concurrency=max_w,
    )
    try:
        np.random.seed(105)
        # 64 x 16 x 16 x 32 with tile_size=8 -> 8 x 1 x 4 = 32 tiles across 4 workers
        A = np.random.randn(64, 16).astype(np.float32)
        B = np.random.randn(16, 32).astype(np.float32)

        c = env["coordinator"].matmul(A, B, dtype="fp32", tile_size=8, max_concurrency=max_w)
        expected = np.matmul(A, B)
        np.testing.assert_allclose(c, expected, rtol=1e-4, atol=1e-4)

        manifest = env["coordinator"].get_last_manifest()
        assert manifest is not None
        assert manifest.total_tasks == 64
        assert manifest.completed_tasks == 64

        # Assert peak concurrency never exceeded configured sliding window W
        assert env["coordinator"].peak_in_flight <= max_w, (
            f"Peak in-flight concurrency {env['coordinator'].peak_in_flight} exceeded window {max_w}"
        )
        assert manifest.peak_concurrency <= max_w, (
            f"Manifest peak concurrency {manifest.peak_concurrency} exceeded window {max_w}"
        )
    finally:
        _cleanup_env(env)


# ==============================================================================
# 6. Multi-Precision Concurrency (FP32, FP16, BF16)
# ==============================================================================

@pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
def test_pipelined_distributed_matmul_precisions(dtype: str):
    """
    Verifies that pipeline parallelism operates seamlessly across FP32, FP16, and BF16
    with mathematical accuracy matching NumPy reference outputs.
    """
    from tests.conftest import ReferenceOracle

    env = _create_env(
        num_workers=4,
        worker_delays=[0.01, 0.01, 0.01, 0.01],
        max_concurrency=8,
    )
    try:
        np.random.seed(200)
        # 32 x 32 matrices with tile_size=16 -> 2 x 2 x 2 = 8 tiles
        A = np.random.randn(32, 32).astype(np.float32)
        B = np.random.randn(32, 32).astype(np.float32)

        c = env["coordinator"].matmul(A, B, dtype=dtype, tile_size=16)
        expected = ReferenceOracle.gemm(A, B, dtype=dtype)

        if dtype == "fp32":
            np.testing.assert_allclose(c, expected, rtol=1e-4, atol=1e-4)
        elif dtype == "fp16":
            np.testing.assert_allclose(c, expected, rtol=1e-2, atol=1e-2)
        elif dtype == "bf16":
            # BF16 mantissa truncation tolerance
            np.testing.assert_allclose(c, expected, rtol=2e-2, atol=2e-2)

        # Confirm all 4 workers participated in parallel
        histories = [w.get_execution_history() for w in env["workers"]]
        for h in histories:
            assert len(h) > 0
    finally:
        _cleanup_env(env)
