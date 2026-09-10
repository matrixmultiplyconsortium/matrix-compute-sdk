"""
Unit tests for P2P gRPC Task Dispatcher, ChannelPool, ParallelTaskScheduler, and Client API.
"""

import time
import grpc
import numpy as np
import pytest

from matrix_compute.client import MatrixComputeClient, distributed_matmul
from matrix_compute.config import ClientConfig
from matrix_compute.exceptions import (
    DispatchError,
    MatrixDimensionError,
    StreamingError,
    TaskDispatchError,
    TaskTimeoutError,
    WorkerConnectionError,
    WorkerSignatureError,
    WorkerTaskRejectedError,
)
from matrix_compute.orchestrator.scheduler import ParallelTaskScheduler, WorkerSlotTracker
from matrix_compute.p2p.channel_pool import ChannelPool, parse_multiaddr
from matrix_compute.p2p.dispatcher import GrpcTaskDispatcher
from matrix_compute.test_doubles import MockNostrRelay, MockSettlementService, MockWorkerNode
from matrix_compute.tiling.partitioner import MatrixTiler, TileTaskPlan

try:
    from matrix_contracts.proto import compute_service_pb2 as pb2
except ImportError:
    from matrix_contracts.v1 import pb2


# ==============================================================================
# Multiaddr & ChannelPool Tests
# ==============================================================================

def test_parse_multiaddr_ipv4():
    addr = "/ip4/127.0.0.1/tcp/50051"
    assert parse_multiaddr(addr) == "127.0.0.1:50051"


def test_parse_multiaddr_ipv6():
    addr = "/ip6/::1/tcp/50051"
    assert parse_multiaddr(addr) == "[::1]:50051"
    addr2 = "[2001:db8::1]:8080"
    assert parse_multiaddr(addr2) == "[2001:db8::1]:8080"
    addr3 = "grpc://[::1]:9090"
    assert parse_multiaddr(addr3) == "[::1]:9090"


def test_parse_multiaddr_dns():
    addr = "/dns4/worker.compute.net/tcp/8080"
    assert parse_multiaddr(addr) == "worker.compute.net:8080"


def test_parse_multiaddr_host_port():
    addr = "localhost:50051"
    assert parse_multiaddr(addr) == "localhost:50051"


def test_parse_multiaddr_uri_scheme():
    addr = "grpc://192.168.1.100:9000"
    assert parse_multiaddr(addr) == "192.168.1.100:9000"


def test_parse_multiaddr_object_accessor():
    class DummyWorker:
        p2p_multiaddr = "/ip4/10.0.0.1/tcp/5000"

    assert parse_multiaddr(DummyWorker()) == "10.0.0.1:5000"


def test_parse_multiaddr_invalid():
    with pytest.raises(ValueError):
        parse_multiaddr("")
    with pytest.raises(ValueError):
        parse_multiaddr(None)


def test_channel_pool_caching():
    pool = ChannelPool()
    ch1 = pool.get_channel("127.0.0.1:50051")
    ch2 = pool.get_channel("/ip4/127.0.0.1/tcp/50051")
    assert ch1 is ch2

    stub1 = pool.get_stub("127.0.0.1:50051")
    stub2 = pool.get_stub("/ip4/127.0.0.1/tcp/50051")
    assert stub1 is stub2

    pool.evict("127.0.0.1:50051")
    pool.close()

    with pytest.raises(RuntimeError, match="closed"):
        pool.get_channel("127.0.0.1:50051")


def test_channel_pool_context_manager():
    with ChannelPool() as pool:
        ch = pool.get_channel("127.0.0.1:50052")
        assert ch is not None
    # After exit, pool is closed
    with pytest.raises(RuntimeError):
        pool.get_channel("127.0.0.1:50052")


# ==============================================================================
# GrpcTaskDispatcher Tests
# ==============================================================================

@pytest.fixture
def mock_settlement():
    return MockSettlementService(verification_key="test-secret-key-1234")


@pytest.fixture
def mock_worker(mock_settlement):
    worker = MockWorkerNode(
        port=0,
        settlement_service=mock_settlement,
        verification_key="test-secret-key-1234",
        vram_mb=16384,
        sats_per_gflop=0.0005,
    )
    worker.start()
    yield worker
    worker.stop()


def test_dispatcher_request_tile_task_success(mock_worker):
    dispatcher = GrpcTaskDispatcher()
    ack = dispatcher.request_tile_task(
        worker_or_target=mock_worker.multiaddr,
        task_id="task_test_01",
        m=16,
        k=16,
        n=16,
        dtype="fp32",
        worker_pubkey=mock_worker.pubkey,
    )
    assert ack.is_ready is True
    assert ack.task_id == "task_test_01"


def test_dispatcher_request_tile_task_rejected(mock_worker):
    mock_worker.set_reject_handshake(True, message="Worker out of memory")
    dispatcher = GrpcTaskDispatcher()

    with pytest.raises(WorkerTaskRejectedError) as exc_info:
        dispatcher.request_tile_task(
            worker_or_target=mock_worker.multiaddr,
            task_id="task_test_reject",
            m=16,
            k=16,
            n=16,
            dtype="fp32",
            worker_pubkey=mock_worker.pubkey,
        )
    assert "Worker out of memory" in str(exc_info.value)
    assert exc_info.value.task_id == "task_test_reject"


def test_dispatcher_request_tile_task_connection_error():
    dispatcher = GrpcTaskDispatcher(connect_timeout=0.5)
    with pytest.raises(WorkerConnectionError):
        dispatcher.request_tile_task(
            worker_or_target="127.0.0.1:59999",  # non-existent port
            task_id="task_test_conn",
            m=16,
            k=16,
            n=16,
            dtype="fp32",
        )


def test_dispatcher_full_tile_dispatch(mock_worker, mock_settlement):
    dispatcher = GrpcTaskDispatcher()

    np.random.seed(42)
    a = np.random.randn(8, 8).astype(np.float32)
    b = np.random.randn(8, 8).astype(np.float32)
    task_plan = TileTaskPlan(
        task_id="tile_task_dispatch_1",
        row_idx=0,
        col_idx=0,
        k_idx=0,
        row_slice=(0, 8),
        col_slice=(0, 8),
        k_slice=(0, 8),
        a_tile=a,
        b_tile=b,
        dtype="fp32",
        estimated_gflops=0.001,
    )

    # Mint valid L402 invoice
    inv = mock_settlement.create_hodl_invoice(
        task_id=task_plan.task_id,
        total_compute_sats=50,
        worker_lightning_address="worker@node",
        max_gflops=1.0,
    )
    mock_settlement.hold_invoice(inv.payment_hash)

    c_res, raw_res = dispatcher.dispatch_tile(
        target_multiaddr=mock_worker.multiaddr,
        worker_pubkey=mock_worker.pubkey,
        task_plan=task_plan,
        macaroon=inv.macaroon,
        payment_hash=inv.payment_hash,
        dtype="fp32",
    )

    expected_c = np.matmul(a, b)
    np.testing.assert_allclose(c_res, expected_c, rtol=1e-5)
    assert raw_res.task_id == task_plan.task_id


def test_dispatcher_signature_tamper_detection(mock_worker, mock_settlement):
    dispatcher = GrpcTaskDispatcher()
    mock_worker.set_tamper_signature(True)

    a = np.random.randn(4, 4).astype(np.float32)
    b = np.random.randn(4, 4).astype(np.float32)
    task_plan = TileTaskPlan(
        task_id="tile_task_tamper_sig",
        row_idx=0,
        col_idx=0,
        k_idx=0,
        row_slice=(0, 4),
        col_slice=(0, 4),
        k_slice=(0, 4),
        a_tile=a,
        b_tile=b,
        dtype="fp32",
        estimated_gflops=0.001,
    )

    inv = mock_settlement.create_hodl_invoice(
        task_id=task_plan.task_id,
        total_compute_sats=50,
        max_gflops=1.0,
    )
    mock_settlement.hold_invoice(inv.payment_hash)

    with pytest.raises(WorkerSignatureError):
        dispatcher.dispatch_tile(
            target_multiaddr=mock_worker.multiaddr,
            worker_pubkey=mock_worker.pubkey,
            task_plan=task_plan,
            macaroon=inv.macaroon,
            payment_hash=inv.payment_hash,
            dtype="fp32",
        )


def test_dispatcher_task_id_mismatch_rejection(mock_worker, mock_settlement):
    dispatcher = GrpcTaskDispatcher()
    task_id = "task_legit_mismatch_test"
    inv = mock_settlement.create_hodl_invoice(task_id=task_id, total_compute_sats=50)
    mock_settlement.hold_invoice(inv.payment_hash)

    mock_worker.set_override_result_task_id("task_injected_wrong_id")

    a = np.ones((4, 4), dtype=np.float32)
    b = np.ones((4, 4), dtype=np.float32)
    task_plan = TileTaskPlan(
        task_id=task_id,
        row_idx=0,
        col_idx=0,
        k_idx=0,
        row_slice=(0, 4),
        col_slice=(0, 4),
        k_slice=(0, 4),
        a_tile=a,
        b_tile=b,
        dtype="fp32",
        estimated_gflops=0.001,
    )

    try:
        with pytest.raises(TaskDispatchError) as exc_info:
            dispatcher.dispatch_tile(
                target_multiaddr=mock_worker.multiaddr,
                worker_pubkey=mock_worker.pubkey,
                task_plan=task_plan,
                macaroon=inv.macaroon,
                payment_hash=inv.payment_hash,
                dtype="fp32",
            )
        assert exc_info.value.task_id == task_id
        assert exc_info.value.returned_task_id == "task_injected_wrong_id"
    finally:
        mock_worker.set_override_result_task_id(None)


def test_dispatcher_multi_chunk_streaming(mock_worker, mock_settlement):
    chunk_size = 64
    dispatcher = GrpcTaskDispatcher(chunk_size_bytes=chunk_size)
    task_id = "task_multi_chunk_streaming_test"
    inv = mock_settlement.create_hodl_invoice(task_id=task_id, total_compute_sats=50)
    mock_settlement.hold_invoice(inv.payment_hash)

    np.random.seed(42)
    a = np.random.randn(8, 8).astype(np.float32)
    b = np.random.randn(8, 8).astype(np.float32)
    task_plan = TileTaskPlan(
        task_id=task_id,
        row_idx=0,
        col_idx=0,
        k_idx=0,
        row_slice=(0, 8),
        col_slice=(0, 8),
        k_slice=(0, 8),
        a_tile=a,
        b_tile=b,
        dtype="fp32",
        estimated_gflops=0.001,
    )

    c, raw_res = dispatcher.dispatch_tile(
        target_multiaddr=mock_worker.multiaddr,
        worker_pubkey=mock_worker.pubkey,
        task_plan=task_plan,
        macaroon=inv.macaroon,
        payment_hash=inv.payment_hash,
        dtype="fp32",
    )

    expected = np.matmul(a, b)
    np.testing.assert_allclose(c, expected, rtol=1e-5)
    assert raw_res.task_id == task_id
    assert raw_res.rows == 8
    assert raw_res.cols == 8


def test_dispatcher_execute_tile_contract(mock_worker, mock_settlement):
    dispatcher = GrpcTaskDispatcher()
    task_id = "task_execute_tile_contract_test"
    inv = mock_settlement.create_hodl_invoice(task_id=task_id, total_compute_sats=50)
    mock_settlement.hold_invoice(inv.payment_hash)

    a = np.eye(4, dtype=np.float32)
    b = np.eye(4, dtype=np.float32)
    l402_hdr = f"L402 {inv.macaroon}:{inv.payment_hash}"

    tile_res = dispatcher.execute_tile(
        worker=mock_worker,
        task_id=task_id,
        A_tile=a,
        B_tile=b,
        dtype="fp32",
        l402_header=l402_hdr,
        worker_pubkey=mock_worker.pubkey,
    )
    assert tile_res.task_id == task_id
    assert tile_res.rows == 4
    assert tile_res.cols == 4
    assert len(tile_res.worker_signature) > 0


# ==============================================================================
# Scheduler Tests
# ==============================================================================

def test_scheduler_failover(mock_settlement):
    # Worker 1 drops connection
    w1 = MockWorkerNode(
        port=0,
        settlement_service=mock_settlement,
        verification_key="test-secret-key-1234",
    ).start()
    w1.set_drop_connection(True)

    # Worker 2 is healthy
    w2 = MockWorkerNode(
        port=0,
        settlement_service=mock_settlement,
        verification_key="test-secret-key-1234",
    ).start()

    try:
        dispatcher = GrpcTaskDispatcher(connect_timeout=0.5)
        scheduler = ParallelTaskScheduler(dispatcher=dispatcher, max_retries=2, backoff_factor=0.01)

        np.random.seed(123)
        a = np.random.randn(4, 4).astype(np.float32)
        b = np.random.randn(4, 4).astype(np.float32)
        task_plan = TileTaskPlan(
            task_id="task_failover_test",
            row_idx=0,
            col_idx=0,
            k_idx=0,
            row_slice=(0, 4),
            col_slice=(0, 4),
            k_slice=(0, 4),
            a_tile=a,
            b_tile=b,
            dtype="fp32",
            estimated_gflops=0.001,
        )

        inv = mock_settlement.create_hodl_invoice(task_id=task_plan.task_id, total_compute_sats=50)
        mock_settlement.hold_invoice(inv.payment_hash)

        # Candidates has w1 first, then w2
        candidates = [w1, w2]
        c_res, _ = scheduler.execute_with_failover(
            task_plan=task_plan,
            candidate_workers=candidates,
            macaroon=inv.macaroon,
            payment_hash=inv.payment_hash,
            dtype="fp32",
        )

        np.testing.assert_allclose(c_res, np.matmul(a, b), rtol=1e-5)
    finally:
        w1.stop()
        w2.stop()
        scheduler.shutdown()


def test_scheduler_blacklisting():
    dispatcher = GrpcTaskDispatcher()
    scheduler = ParallelTaskScheduler(dispatcher=dispatcher)

    pk = "worker_bad_apple"
    scheduler.record_worker_failure(pk)
    scheduler.record_worker_failure(pk)
    assert len(scheduler._blacklisted_workers) == 0

    scheduler.record_worker_failure(pk)
    assert pk in scheduler._blacklisted_workers

    class MockW:
        worker_pubkey = pk

    healthy = scheduler.get_healthy_workers([MockW()])
    assert len(healthy) == 0


def test_scheduler_single_worker_retry(mock_worker, mock_settlement):
    dispatcher = GrpcTaskDispatcher()
    scheduler = ParallelTaskScheduler(dispatcher=dispatcher, max_retries=2, backoff_factor=0.01)

    task_id = "task_single_worker_retry_test"
    a = np.eye(4, dtype=np.float32)
    b = np.eye(4, dtype=np.float32)
    plan = TileTaskPlan(
        task_id=task_id,
        row_idx=0,
        col_idx=0,
        k_idx=0,
        row_slice=(0, 4),
        col_slice=(0, 4),
        k_slice=(0, 4),
        a_tile=a,
        b_tile=b,
        dtype="fp32",
        estimated_gflops=0.001,
    )
    inv = mock_settlement.create_hodl_invoice(task_id=task_id, total_compute_sats=50)
    mock_settlement.hold_invoice(inv.payment_hash)

    # Worker initially rejects handshake once
    mock_worker.set_reject_handshake(True, "Worker temporarily busy")
    failover_called = [0]

    def on_failover(w, ex):
        failover_called[0] += 1
        # Clear failure flag so retry succeeds
        mock_worker.set_reject_handshake(False)

    try:
        c, raw_res = scheduler.execute_with_failover(
            task_plan=plan,
            candidate_workers=[mock_worker],
            macaroon=inv.macaroon,
            payment_hash=inv.payment_hash,
            dtype="fp32",
            on_failover_callback=on_failover,
        )
        assert failover_called[0] == 1
        np.testing.assert_allclose(c, np.matmul(a, b), rtol=1e-5)
    finally:
        mock_worker.set_reject_handshake(False)
        scheduler.shutdown()


# ==============================================================================
# MatrixComputeClient Tests
# ==============================================================================

def test_client_dimension_validations():
    config = ClientConfig.for_testing()
    with MatrixComputeClient(config=config) as client:
        # Non-2D
        with pytest.raises(MatrixDimensionError):
            client.matmul(np.ones((4,)), np.ones((4, 4)))

        # Zero dimension
        with pytest.raises(MatrixDimensionError):
            client.matmul(np.ones((0, 4)), np.ones((4, 4)))

        # Inner dimension mismatch
        with pytest.raises(MatrixDimensionError):
            client.matmul(np.ones((4, 8)), np.ones((6, 4)))


def test_client_offline_mock_mode_matmul():
    config = ClientConfig.for_testing(default_tile_size=16)
    np.random.seed(99)
    a = np.random.randn(32, 32).astype(np.float32)
    b = np.random.randn(32, 32).astype(np.float32)

    with MatrixComputeClient(config=config) as client:
        c = client.matmul(a, b, dtype="fp32", tile_size=16)
        expected = np.matmul(a, b)
        np.testing.assert_allclose(c, expected, rtol=1e-4)


def test_distributed_matmul_functional_api():
    config = ClientConfig.for_testing(default_tile_size=8)
    np.random.seed(101)
    a = np.random.randn(16, 16).astype(np.float32)
    b = np.random.randn(16, 16).astype(np.float32)

    c = distributed_matmul(a, b, dtype="fp32", config=config, tile_size=8)
    expected = np.matmul(a, b)
    np.testing.assert_allclose(c, expected, rtol=1e-4)
