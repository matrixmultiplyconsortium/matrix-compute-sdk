"""Comprehensive unit tests for hermetic test doubles: MockNostrRelay, MockSettlementService, MockWorkerNode."""

import base64
import hashlib
import json
import threading
import time
from typing import List

import grpc
import numpy as np
import pytest

from matrix_contracts.proto import compute_service_pb2 as pb2
from matrix_contracts.proto import compute_service_pb2_grpc as pb2_grpc
from matrix_contracts.schemas.discovery import (
    BondStatus,
    HardwareSpecs,
    MatchedWorker,
    PricingSpecs,
    WorkerAnnouncePayload,
    WorkerQueryRequest,
)
from matrix_contracts.schemas.settlement import (
    CreateHodlInvoiceRequest,
    SettlePaymentRequest,
)

from matrix_compute.crypto.signer import verify_tile_result_signature
from matrix_compute.test_doubles.mock_relay import MockNostrRelay
from matrix_compute.test_doubles.mock_settlement import (
    InvoiceStatus,
    MockInvoiceResponse,
    MockSettlementService,
)
from matrix_compute.test_doubles.mock_worker import MockWorkerNode, MockWorkerServicer


# ==============================================================================
# 1. MockNostrRelay Unit Tests (REL-01 through REL-10)
# ==============================================================================

def test_mock_relay_publish_raw_event():
    """REL-01: Ingests and indexes raw Kind 30000 Nostr event dictionary."""
    relay = MockNostrRelay()
    event = {
        "pubkey": "02worker1",
        "kind": 30000,
        "created_at": 1000,
        "tags": [
            ["d", "matrix-worker"],
            ["p2p", "/ip4/127.0.0.1/tcp/50051"],
            ["lightning", "02lnpub1"],
            ["vram_mb", "24576"],
            ["sats_per_gflop", "0.0005"],
            ["gpu_model", "NVIDIA RTX 4090"],
        ],
        "content": "{}",
    }
    assert relay.publish(event) is True
    stored = relay.get_active_announcement("02worker1")
    assert stored is not None
    assert stored["pubkey"] == "02worker1"


def test_mock_relay_publish_announce_payload():
    """REL-02: Serializes WorkerAnnouncePayload to Kind 30000 event with standard tags."""
    relay = MockNostrRelay()
    payload = WorkerAnnouncePayload(
        event="WORKER_ANNOUNCE",
        worker_pubkey="02pub_announce",
        p2p_multiaddr="/ip4/127.0.0.1/tcp/50052",
        lightning_routing_pubkey="02lnpub2",
        hardware=HardwareSpecs(
            gpu_model="NVIDIA RTX 3090",
            vram_mb=24576,
            peak_fp32_tflops=35.6,
            attestation_proof="mock_attest",
            timestamp=1000,
        ),
        pricing=PricingSpecs(
            sats_per_gflop=0.0008,
            min_task_sats=10,
        ),
        bond_status=BondStatus(
            staked_amount_sats=100000,
            bond_txid="mock_bond_1",
        ),
        timestamp=1000,
    )
    event_id = relay.publish_announce(payload)
    assert event_id is not None
    stored = relay.get_event(event_id)
    assert stored is not None
    assert stored["kind"] == 30000
    assert stored["pubkey"] == "02pub_announce"


def test_mock_relay_nip33_overwrite_and_deduplication():
    """REL-03: Later timestamp overwrites existing (kind, pubkey, d_tag) event."""
    relay = MockNostrRelay()
    ev1 = {
        "pubkey": "02node_a",
        "kind": 30000,
        "created_at": 1000,
        "tags": [["d", "matrix-worker"], ["vram_mb", "8192"]],
        "content": "{}",
    }
    ev2 = {
        "pubkey": "02node_a",
        "kind": 30000,
        "created_at": 2000,
        "tags": [["d", "matrix-worker"], ["vram_mb", "16384"]],
        "content": "{}",
    }
    relay.publish(ev1)
    relay.publish(ev2)
    active = relay.get_active_announcement("02node_a")
    assert active["created_at"] == 2000
    specs = MockNostrRelay._parse_event_specs(active)
    assert specs["vram_mb"] == 16384
    assert len(relay.get_events()) == 1


def test_mock_relay_out_of_order_timestamp_discarded():
    """REL-04: Older timestamp event is safely ignored, preserving newest active announcement."""
    relay = MockNostrRelay()
    ev_newer = {
        "pubkey": "02node_b",
        "kind": 30000,
        "created_at": 3000,
        "tags": [["d", "matrix-worker"], ["vram_mb", "24576"]],
        "content": "{}",
    }
    ev_older = {
        "pubkey": "02node_b",
        "kind": 30000,
        "created_at": 1500,
        "tags": [["d", "matrix-worker"], ["vram_mb", "8192"]],
        "content": "{}",
    }
    relay.publish(ev_newer)
    assert relay.publish(ev_older) is True  # returns True per Nostr spec
    active = relay.get_active_announcement("02node_b")
    assert active["created_at"] == 3000
    specs = MockNostrRelay._parse_event_specs(active)
    assert specs["vram_mb"] == 24576


def test_mock_relay_content_fallback_specs():
    """REL-05: Missing tags extract specs from JSON content payload."""
    relay = MockNostrRelay()
    content_obj = {
        "p2p_multiaddr": "/ip4/127.0.0.1/tcp/9000",
        "lightning_routing_pubkey": "02ln_fallback",
        "hardware": {"gpu_model": "RTX A6000", "vram_mb": 49152},
        "pricing": {"sats_per_gflop": 0.0003},
    }
    ev = {
        "pubkey": "02fallback_node",
        "kind": 30000,
        "created_at": 1000,
        "tags": [["d", "matrix-worker"]],  # No specs in tags!
        "content": json.dumps(content_obj),
    }
    relay.publish(ev)
    active = relay.get_active_announcement("02fallback_node")
    specs = MockNostrRelay._parse_event_specs(active)
    assert specs["p2p"] == "/ip4/127.0.0.1/tcp/9000"
    assert specs["lightning"] == "02ln_fallback"
    assert specs["gpu_model"] == "RTX A6000"
    assert specs["vram_mb"] == 49152
    assert specs["price"] == 0.0003


def test_mock_relay_free_tier_preservation():
    """REL-06: 0.0 sat/gflop and 0 VRAM are preserved without fallback corruption."""
    event = {
        "tags": [["sats_per_gflop", "0.0"], ["vram_mb", "0"]],
        "content": json.dumps({"pricing": {"sats_per_gflop": 0.005}, "hardware": {"vram_mb": 16384}}),
    }
    specs = MockNostrRelay._parse_event_specs(event)
    assert specs["price"] == 0.0
    assert specs["vram_mb"] == 0


def test_mock_relay_query_filtering():
    """REL-07: Filters by min VRAM, max price, and GPU model substring."""
    relay = MockNostrRelay()
    ev1 = {
        "pubkey": "node_4090",
        "kind": 30000,
        "tags": [["d", "matrix-worker"], ["vram_mb", "24576"], ["sats_per_gflop", "0.0005"], ["gpu_model", "NVIDIA RTX 4090"]],
        "content": "{}",
    }
    ev2 = {
        "pubkey": "node_3080",
        "kind": 30000,
        "tags": [["d", "matrix-worker"], ["vram_mb", "10240"], ["sats_per_gflop", "0.0002"], ["gpu_model", "NVIDIA RTX 3080"]],
        "content": "{}",
    }
    relay.publish(ev1)
    relay.publish(ev2)

    # Filter by VRAM > 16384
    res_vram = relay.query(min_vram_mb=16384)
    assert len(res_vram) == 1
    assert res_vram[0]["pubkey"] == "node_4090"

    # Filter by max price 0.0003
    res_price = relay.query(max_price=0.0003)
    assert len(res_price) == 1
    assert res_price[0]["pubkey"] == "node_3080"

    # Filter by GPU model
    res_model = relay.query(gpu_model="4090")
    assert len(res_model) == 1
    assert res_model[0]["pubkey"] == "node_4090"


def test_mock_relay_query_workers_pareto_ranking():
    """REL-08: Pareto sorts candidates: cheapest first, tie-break by largest VRAM."""
    relay = MockNostrRelay()
    # 3 candidates:
    # A: 0.001 sats/gflop, 48GB VRAM
    # B: 0.0005 sats/gflop, 24GB VRAM (cheapest)
    # C: 0.0005 sats/gflop, 32GB VRAM (cheapest tie, but more VRAM than B)
    ev_a = {"pubkey": "pk_a", "kind": 30000, "tags": [["d", "matrix-worker"], ["p2p", "p2p_a"], ["lightning", "ln_a"], ["sats_per_gflop", "0.001"], ["vram_mb", "49152"]], "content": "{}"}
    ev_b = {"pubkey": "pk_b", "kind": 30000, "tags": [["d", "matrix-worker"], ["p2p", "p2p_b"], ["lightning", "ln_b"], ["sats_per_gflop", "0.0005"], ["vram_mb", "24576"]], "content": "{}"}
    ev_c = {"pubkey": "pk_c", "kind": 30000, "tags": [["d", "matrix-worker"], ["p2p", "p2p_c"], ["lightning", "ln_c"], ["sats_per_gflop", "0.0005"], ["vram_mb", "32768"]], "content": "{}"}

    relay.publish(ev_a)
    relay.publish(ev_b)
    relay.publish(ev_c)

    req = WorkerQueryRequest(min_vram_mb=8192, max_price_sats_per_gflop=0.01, required_slots=3)
    resp = relay.query_workers(req)
    assert resp.status == "SUCCESS"
    assert len(resp.matched_workers) == 3
    # Candidate C should be #1 (0.0005 price, 32GB VRAM)
    assert resp.matched_workers[0].worker_pubkey == "pk_c"
    # Candidate B should be #2 (0.0005 price, 24GB VRAM)
    assert resp.matched_workers[1].worker_pubkey == "pk_b"
    # Candidate A should be #3 (0.001 price)
    assert resp.matched_workers[2].worker_pubkey == "pk_a"


def test_mock_relay_concurrent_publishing_threads():
    """REL-09: 20 concurrent threads publish/replace events with zero KeyError or deadlock."""
    relay = MockNostrRelay()
    errors = []

    def worker(idx: int):
        try:
            for t in range(10):
                ev = {
                    "pubkey": f"worker_{idx}",
                    "kind": 30000,
                    "created_at": 1000 + t,
                    "tags": [["d", "matrix-worker"], ["vram_mb", str(1024 * (idx + 1))]],
                    "content": "{}",
                }
                relay.publish(ev)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    assert len(errors) == 0
    assert len(relay.get_events()) == 20


def test_mock_relay_reset_and_clear():
    """REL-10: clear() empties all stores and tag indices."""
    relay = MockNostrRelay()
    relay.publish({"pubkey": "test_pk", "kind": 30000, "tags": [["d", "matrix-worker"]], "content": "{}"})
    assert len(relay.get_events()) == 1
    relay.clear()
    assert len(relay.get_events()) == 0


# ==============================================================================
# 2. MockSettlementService Unit Tests (SET-01 through SET-11)
# ==============================================================================

def test_settlement_create_invoice_trustless():
    """SET-01: Consumer provides payment_hash; returns MockInvoiceResponse with preimage=None."""
    settlement = MockSettlementService()
    preimage_bytes = b"\x01" * 32
    p_hash = hashlib.sha256(preimage_bytes).hexdigest()

    resp = settlement.create_hodl_invoice(
        task_id="task_trustless_1",
        total_compute_sats=500,
        payment_hash=p_hash,
    )
    assert resp.payment_hash == p_hash
    assert resp.preimage is None
    assert resp.platform_fee_sats == 10  # 2% of 500
    assert resp.worker_payout_sats == 490
    assert resp["payment_hash"] == p_hash
    assert settlement.get_status(p_hash) == "UNPAID"


def test_settlement_create_invoice_custodial():
    """SET-02: Server generates 32-byte preimage; sha256(preimage) == payment_hash."""
    settlement = MockSettlementService()
    resp = settlement.create_hodl_invoice(task_id="task_custodial_1", total_compute_sats=100)
    assert resp.preimage is not None
    assert len(resp.preimage) == 64
    computed_hash = hashlib.sha256(bytes.fromhex(resp.preimage)).hexdigest()
    assert computed_hash == resp.payment_hash


def test_settlement_duplicate_rejection():
    """SET-03: Duplicate task_id or payment_hash raises ValueError."""
    settlement = MockSettlementService()
    settlement.create_hodl_invoice(task_id="task_dup", payment_hash="11" * 32)

    with pytest.raises(ValueError) as exc:
        settlement.create_hodl_invoice(task_id="task_dup", payment_hash="22" * 32)
    assert "Task ID already registered" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        settlement.create_hodl_invoice(task_id="task_other", payment_hash="11" * 32)
    assert "Duplicate payment hash" in str(exc.value)


def test_settlement_full_lifecycle_transitions():
    """SET-04: Transitions UNPAID -> HELD -> SETTLED."""
    settlement = MockSettlementService()
    secret = b"\x42" * 32
    p_hash = hashlib.sha256(secret).hexdigest()

    resp = settlement.create_hodl_invoice(task_id="task_life_1", payment_hash=p_hash)
    assert settlement.get_status(p_hash) == "UNPAID"
    assert settlement.is_held(p_hash) is False

    # Hold
    assert settlement.hold_invoice(p_hash) is True
    assert settlement.get_status(p_hash) == "HELD"
    assert settlement.is_held(p_hash) is True

    # Settle
    assert settlement.settle_invoice(p_hash, preimage=secret.hex()) is True
    assert settlement.get_status(p_hash) == "SETTLED"
    assert settlement.is_held(p_hash) is False


def test_settlement_illegal_transitions_rejected():
    """SET-05: Cannot settle UNPAID; cannot hold non-UNPAID."""
    settlement = MockSettlementService()
    resp = settlement.create_hodl_invoice(task_id="task_illegal_1")
    p_hash = resp.payment_hash

    # Cannot settle UNPAID invoice directly
    with pytest.raises(ValueError) as exc:
        settlement.settle_invoice(p_hash, preimage=resp.preimage)
    assert "must be HELD" in str(exc.value)

    # Hold it
    settlement.hold_invoice(p_hash)
    # Cannot hold already HELD invoice
    with pytest.raises(ValueError) as exc:
        settlement.hold_invoice(p_hash)
    assert "must be UNPAID" in str(exc.value)


def test_settlement_preimage_validation():
    """SET-06: Rejects non-hex preimage; rejects preimage not matching payment_hash."""
    settlement = MockSettlementService()
    secret = b"\x99" * 32
    p_hash = hashlib.sha256(secret).hexdigest()
    settlement.create_hodl_invoice(task_id="task_preimage_1", payment_hash=p_hash)
    settlement.hold_invoice(p_hash)

    # Non-hex string
    with pytest.raises(ValueError) as exc:
        settlement.settle_invoice(p_hash, preimage="not_a_hex_string")
    assert "valid hexadecimal" in str(exc.value)

    # Mismatched hash
    wrong_secret = b"\x00" * 32
    with pytest.raises(ValueError) as exc:
        settlement.settle_invoice(p_hash, preimage=wrong_secret.hex())
    assert "does not match invoice hash" in str(exc.value)


def test_settlement_settle_via_request_schema():
    """SET-07: Settles using Pydantic SettlePaymentRequest(task_id, preimage)."""
    settlement = MockSettlementService()
    secret = b"\xaa" * 32
    p_hash = hashlib.sha256(secret).hexdigest()
    settlement.create_hodl_invoice(task_id="task_req_schema", payment_hash=p_hash)
    settlement.hold_invoice(p_hash)

    req = SettlePaymentRequest(task_id="task_req_schema", preimage=secret.hex())
    assert settlement.settle_invoice(req) is True
    assert settlement.get_status(p_hash) == "SETTLED"


def test_settlement_cancel_invoice():
    """SET-08: Cancels UNPAID or HELD invoices; cannot cancel already SETTLED invoice."""
    settlement = MockSettlementService()
    resp = settlement.create_hodl_invoice(task_id="task_cancel_1")
    settlement.hold_invoice(resp.payment_hash)
    assert settlement.cancel_invoice(resp.payment_hash) is True
    assert settlement.get_status(resp.payment_hash) == "CANCELLED"

    # Cannot cancel SETTLED
    resp2 = settlement.create_hodl_invoice(task_id="task_cancel_2")
    settlement.hold_invoice(resp2.payment_hash)
    settlement.settle_invoice(resp2.payment_hash, preimage=resp2.preimage)
    with pytest.raises(ValueError) as exc:
        settlement.cancel_invoice(resp2.payment_hash)
    assert "Cannot cancel an already SETTLED" in str(exc.value)


def test_settlement_macaroon_minting_and_caveat_verification():
    """SET-09: Validates HMAC-SHA256, task_id, expiry, and max_gflops bounds."""
    settlement = MockSettlementService()
    resp = settlement.create_hodl_invoice(
        task_id="task_mac_1",
        max_gflops=50.0,
        expiry_seconds=3600,
    )
    ok, msg = settlement.verify_macaroon(
        macaroon_raw=resp.macaroon,
        expected_task_id="task_mac_1",
        required_gflops=40.0,
    )
    assert ok is True
    assert msg == "OK"


def test_settlement_macaroon_tamper_rejection():
    """SET-10: Forged signatures, tampered caveats, or expired tokens return (False, msg)."""
    settlement = MockSettlementService()
    resp = settlement.create_hodl_invoice(task_id="task_tamper_1", max_gflops=10.0, expiry_seconds=10)

    # Decode and tamper caveat
    payload = json.loads(base64.b64decode(resp.macaroon).decode("utf-8"))
    payload["caveats"]["task_id"] = "tampered_task_id"
    tampered_mac = base64.b64encode(json.dumps(payload).encode("utf-8")).decode("utf-8")

    ok, msg = settlement.verify_macaroon(tampered_mac, "tampered_task_id", 5.0)
    assert ok is False
    assert "Invalid macaroon HMAC signature" in msg

    # Expired token
    ok_exp, msg_exp = settlement.verify_macaroon(
        resp.macaroon,
        "task_tamper_1",
        5.0,
        current_time=time.time() + 100,  # 100 seconds in future
    )
    assert ok_exp is False
    assert "expired" in msg_exp


def test_settlement_concurrent_hold_and_settle():
    """SET-11: 20 concurrent tasks held and settled concurrently without race conditions."""
    settlement = MockSettlementService()
    tasks = []
    for i in range(20):
        secret = f"preimage_{i:04d}".encode("utf-8").ljust(32, b"\x00")
        p_hash = hashlib.sha256(secret).hexdigest()
        settlement.create_hodl_invoice(task_id=f"con_task_{i}", payment_hash=p_hash)
        tasks.append((p_hash, secret.hex()))

    errors = []

    def hold_and_settle(p_hash: str, preimage: str):
        try:
            settlement.hold_invoice(p_hash)
            settlement.settle_invoice(p_hash, preimage=preimage)
        except Exception as ex:
            errors.append(ex)

    threads = [threading.Thread(target=hold_and_settle, args=t) for t in tasks]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    assert len(errors) == 0
    for p_hash, _ in tasks:
        assert settlement.get_status(p_hash) == "SETTLED"


# ==============================================================================
# 3. MockWorkerNode Unit Tests (WRK-01 through WRK-17)
# ==============================================================================

def _make_stream_chunks(task_id: str, a_bytes: bytes, b_bytes: bytes, chunk_size: int = 512):
    chunks = []
    seq = 0
    for i in range(0, len(a_bytes), chunk_size):
        sub = a_bytes[i:i + chunk_size]
        chunks.append(pb2.TileDataChunk(
            task_id=task_id,
            sequence_number=seq,
            chunk_bytes=sub,
            is_matrix_a=True,
            is_final_chunk=False,
        ))
        seq += 1
    for i in range(0, len(b_bytes), chunk_size):
        sub = b_bytes[i:i + chunk_size]
        is_final = (i + chunk_size >= len(b_bytes))
        chunks.append(pb2.TileDataChunk(
            task_id=task_id,
            sequence_number=seq,
            chunk_bytes=sub,
            is_matrix_a=False,
            is_final_chunk=is_final,
        ))
        seq += 1
    return chunks


def test_mock_worker_server_lifecycle_and_context_manager():
    """WRK-01: Ephemeral port binding, background startup, graceful shutdown, context manager."""
    with MockWorkerNode(port=0) as worker:
        assert worker.port > 0
        assert worker.target == f"127.0.0.1:{worker.port}"
        assert worker.multiaddr == f"/ip4/127.0.0.1/tcp/{worker.port}"
        channel = worker.create_channel()
        assert channel is not None
        channel.close()
    # Server should be stopped after exiting context
    assert worker.server is None


def test_mock_worker_request_tile_task_success():
    """WRK-02: Returns TileTaskAck with readiness, pubkey, and correct GFLOP quote."""
    with MockWorkerNode() as worker:
        stub = worker.create_stub()
        req = pb2.TileTaskRequest(
            task_id="task_req_1",
            matrix_a_rows=64,
            matrix_a_cols=128,
            matrix_b_cols=64,
            dtype="fp32",
        )
        ack = stub.RequestTileTask(req)
        assert ack.is_ready is True
        assert ack.task_id == "task_req_1"
        assert ack.worker_pubkey == worker.pubkey
        # GFLOPs: (2 * 64 * 128 * 64) / 1e9 = 0.001048576
        expected_gflops = (2.0 * 64 * 128 * 64) / 1e9
        assert abs(ack.estimated_gflops - expected_gflops) < 1e-6


def test_mock_worker_request_tile_task_validation():
    """WRK-03: Rejects non-positive dimensions and unsupported dtypes."""
    with MockWorkerNode() as worker:
        stub = worker.create_stub()
        # Non-positive dimension
        req_bad_dim = pb2.TileTaskRequest(
            task_id="task_bad_dim",
            matrix_a_rows=0,
            matrix_a_cols=10,
            matrix_b_cols=10,
            dtype="fp32",
        )
        with pytest.raises(grpc.RpcError) as exc:
            stub.RequestTileTask(req_bad_dim)
        assert exc.value.code() == grpc.StatusCode.INVALID_ARGUMENT

        # Unsupported dtype
        req_bad_dtype = pb2.TileTaskRequest(
            task_id="task_bad_dtype",
            matrix_a_rows=10,
            matrix_a_cols=10,
            matrix_b_cols=10,
            dtype="int32",
        )
        with pytest.raises(grpc.RpcError) as exc:
            stub.RequestTileTask(req_bad_dtype)
        assert exc.value.code() == grpc.StatusCode.INVALID_ARGUMENT


def test_mock_worker_execute_without_prior_request_aborts():
    """WRK-04: Aborts unregistered task stream with FAILED_PRECONDITION."""
    with MockWorkerNode() as worker:
        worker.set_bypass_l402(True)
        stub = worker.create_stub()
        chunks = [
            pb2.TileDataChunk(task_id="unregistered_task", sequence_number=0, chunk_bytes=b"00", is_matrix_a=True, is_final_chunk=True)
        ]
        with pytest.raises(grpc.RpcError) as exc:
            stub.ExecuteTileTask(iter(chunks))
        assert exc.value.code() == grpc.StatusCode.FAILED_PRECONDITION


def test_mock_worker_l402_gate_missing_auth_aborts():
    """WRK-05: Missing Authorization header aborts with UNAUTHENTICATED."""
    settlement = MockSettlementService()
    with MockWorkerNode(settlement_service=settlement) as worker:
        stub = worker.create_stub()
        stub.RequestTileTask(pb2.TileTaskRequest(task_id="task_no_auth", matrix_a_rows=4, matrix_a_cols=4, matrix_b_cols=4, dtype="fp32"))
        chunks = _make_stream_chunks("task_no_auth", b"\x00" * 64, b"\x00" * 64)
        # Call without auth header
        with pytest.raises(grpc.RpcError) as exc:
            stub.ExecuteTileTask(iter(chunks))
        assert exc.value.code() == grpc.StatusCode.UNAUTHENTICATED


def test_mock_worker_l402_gate_unheld_invoice_aborts():
    """WRK-06: payment_hash in UNPAID status aborts with FAILED_PRECONDITION."""
    settlement = MockSettlementService()
    with MockWorkerNode(settlement_service=settlement) as worker:
        stub = worker.create_stub()
        task_id = "task_unheld"
        stub.RequestTileTask(pb2.TileTaskRequest(task_id=task_id, matrix_a_rows=4, matrix_a_cols=4, matrix_b_cols=4, dtype="fp32"))
        inv = settlement.create_hodl_invoice(task_id=task_id, max_gflops=10.0, expiry_seconds=3600)
        # NOTE: Do NOT call hold_invoice -> invoice remains UNPAID!
        chunks = _make_stream_chunks(task_id, b"\x00" * 64, b"\x00" * 64)
        metadata = [("authorization", f"L402 {inv.macaroon}:{inv.payment_hash}"), ("x-task-id", task_id)]
        with pytest.raises(grpc.RpcError) as exc:
            stub.ExecuteTileTask(iter(chunks), metadata=metadata)
        assert exc.value.code() == grpc.StatusCode.FAILED_PRECONDITION


def test_mock_worker_l402_gate_expired_token_aborts():
    """WRK-07: Expired macaroon aborts with UNAUTHENTICATED."""
    settlement = MockSettlementService()
    with MockWorkerNode(settlement_service=settlement) as worker:
        stub = worker.create_stub()
        task_id = "task_exp"
        stub.RequestTileTask(pb2.TileTaskRequest(task_id=task_id, matrix_a_rows=4, matrix_a_cols=4, matrix_b_cols=4, dtype="fp32"))
        inv = settlement.create_hodl_invoice(task_id=task_id, max_gflops=10.0, expiry_seconds=-10)  # already expired
        settlement.hold_invoice(inv.payment_hash)
        chunks = _make_stream_chunks(task_id, b"\x00" * 64, b"\x00" * 64)
        metadata = [("authorization", f"L402 {inv.macaroon}:{inv.payment_hash}"), ("x-task-id", task_id)]
        with pytest.raises(grpc.RpcError) as exc:
            stub.ExecuteTileTask(iter(chunks), metadata=metadata)
        assert exc.value.code() == grpc.StatusCode.UNAUTHENTICATED


def test_mock_worker_stream_out_of_order_sequence_aborts():
    """WRK-08: Chunk arriving with unexpected sequence number aborts with INVALID_ARGUMENT."""
    with MockWorkerNode() as worker:
        worker.set_bypass_l402(True)
        stub = worker.create_stub()
        task_id = "task_seq_err"
        stub.RequestTileTask(pb2.TileTaskRequest(task_id=task_id, matrix_a_rows=4, matrix_a_cols=4, matrix_b_cols=4, dtype="fp32"))
        chunks = [
            pb2.TileDataChunk(task_id=task_id, sequence_number=0, chunk_bytes=b"\x00" * 32, is_matrix_a=True, is_final_chunk=False),
            pb2.TileDataChunk(task_id=task_id, sequence_number=5, chunk_bytes=b"\x00" * 32, is_matrix_a=True, is_final_chunk=True),  # Out of order!
        ]
        metadata = [("x-task-id", task_id)]
        with pytest.raises(grpc.RpcError) as exc:
            stub.ExecuteTileTask(iter(chunks), metadata=metadata)
        assert exc.value.code() == grpc.StatusCode.INVALID_ARGUMENT


def test_mock_worker_stream_running_byte_overflow_aborts():
    """WRK-09: Accumulated bytes exceeding matrix bounds aborts immediately with zero extra buffering."""
    with MockWorkerNode() as worker:
        worker.set_bypass_l402(True)
        stub = worker.create_stub()
        task_id = "task_byte_overflow"
        # 4x4 FP32 = 64 bytes
        stub.RequestTileTask(pb2.TileTaskRequest(task_id=task_id, matrix_a_rows=4, matrix_a_cols=4, matrix_b_cols=4, dtype="fp32"))
        chunks = [
            # Chunk is 100 bytes (> 64 bytes limit!)
            pb2.TileDataChunk(task_id=task_id, sequence_number=0, chunk_bytes=b"\x00" * 100, is_matrix_a=True, is_final_chunk=False),
        ]
        metadata = [("x-task-id", task_id)]
        with pytest.raises(grpc.RpcError) as exc:
            stub.ExecuteTileTask(iter(chunks), metadata=metadata)
        assert exc.value.code() == grpc.StatusCode.INVALID_ARGUMENT


def test_mock_worker_stream_missing_final_chunk_flag_aborts():
    """WRK-10: Stream terminating without is_final_chunk=True aborts with INVALID_ARGUMENT."""
    with MockWorkerNode() as worker:
        worker.set_bypass_l402(True)
        stub = worker.create_stub()
        task_id = "task_no_final"
        stub.RequestTileTask(pb2.TileTaskRequest(task_id=task_id, matrix_a_rows=4, matrix_a_cols=4, matrix_b_cols=4, dtype="fp32"))
        # 64 bytes A, 64 bytes B, but is_final_chunk is False
        chunks = [
            pb2.TileDataChunk(task_id=task_id, sequence_number=0, chunk_bytes=b"\x00" * 64, is_matrix_a=True, is_final_chunk=False),
            pb2.TileDataChunk(task_id=task_id, sequence_number=1, chunk_bytes=b"\x00" * 64, is_matrix_a=False, is_final_chunk=False),
        ]
        metadata = [("x-task-id", task_id)]
        with pytest.raises(grpc.RpcError) as exc:
            stub.ExecuteTileTask(iter(chunks), metadata=metadata)
        assert exc.value.code() == grpc.StatusCode.INVALID_ARGUMENT


def test_mock_worker_gemm_execution_fp32_fp16_bf16():
    """WRK-11: Streams A and B chunks; result matches NumPy reference within IEEE tolerance."""
    dtypes = ["fp32", "fp16", "bf16"]
    for dt in dtypes:
        with MockWorkerNode() as worker:
            worker.set_bypass_l402(True)
            stub = worker.create_stub()
            task_id = f"task_gemm_{dt}"
            m, k, n = 8, 8, 8

            stub.RequestTileTask(pb2.TileTaskRequest(
                task_id=task_id,
                matrix_a_rows=m,
                matrix_a_cols=k,
                matrix_b_cols=n,
                dtype=dt,
            ))

            np_dtype = np.float32 if dt in ("fp32", "bf16") else np.float16
            np.random.seed(42)
            mat_a = np.random.randn(m, k).astype(np_dtype)
            mat_b = np.random.randn(k, n).astype(np_dtype)

            chunks = _make_stream_chunks(task_id, mat_a.tobytes(), mat_b.tobytes())
            metadata = [("x-task-id", task_id)]
            tile_result = stub.ExecuteTileTask(iter(chunks), metadata=metadata)

            assert tile_result.task_id == task_id
            assert tile_result.rows == m
            assert tile_result.cols == n

            # Reconstruct and compare
            res_mat = np.frombuffer(tile_result.result_bytes, dtype=np_dtype).reshape((m, n))
            if dt == "bf16":
                # Compare against truncated FP32
                ref_c = np.matmul(mat_a.astype(np.float32), mat_b.astype(np.float32))
                ref_masked = (ref_c.view(np.uint32) & 0xFFFF0000).view(np.float32)
                np.testing.assert_allclose(res_mat, ref_masked, atol=1e-5)
            else:
                ref_c = np.matmul(mat_a, mat_b)
                np.testing.assert_allclose(res_mat, ref_c, rtol=1e-3, atol=1e-3)


def test_mock_worker_secp256k1_signature_verification():
    """WRK-12: Validates worker Secp256k1 signature on f'{task_id}:{sha256(result)}'."""
    with MockWorkerNode() as worker:
        worker.set_bypass_l402(True)
        stub = worker.create_stub()
        task_id = "task_sig_test"
        stub.RequestTileTask(pb2.TileTaskRequest(task_id=task_id, matrix_a_rows=4, matrix_a_cols=4, matrix_b_cols=4, dtype="fp32"))
        mat_a = np.eye(4, dtype=np.float32)
        mat_b = np.eye(4, dtype=np.float32)
        chunks = _make_stream_chunks(task_id, mat_a.tobytes(), mat_b.tobytes())
        res = stub.ExecuteTileTask(iter(chunks), metadata=[("x-task-id", task_id)])

        assert verify_tile_result_signature(
            worker.pubkey,
            task_id,
            res.result_bytes,
            res.worker_signature,
        ) is True


def test_mock_worker_fault_injection_tamper_result():
    """WRK-13: set_tamper_result(True) returns perturbed C signed by worker."""
    with MockWorkerNode() as worker:
        worker.set_bypass_l402(True)
        worker.set_tamper_result(True, perturbation=10.0)
        stub = worker.create_stub()
        task_id = "task_tamper_math"
        stub.RequestTileTask(pb2.TileTaskRequest(task_id=task_id, matrix_a_rows=4, matrix_a_cols=4, matrix_b_cols=4, dtype="fp32"))
        mat_a = np.zeros((4, 4), dtype=np.float32)
        mat_b = np.zeros((4, 4), dtype=np.float32)
        chunks = _make_stream_chunks(task_id, mat_a.tobytes(), mat_b.tobytes())
        res = stub.ExecuteTileTask(iter(chunks), metadata=[("x-task-id", task_id)])

        res_mat = np.frombuffer(res.result_bytes, dtype=np.float32).reshape((4, 4))
        # Zero x Zero should be Zero, but perturbed C[0, 0] == 10.0
        assert res_mat[0, 0] == 10.0
        # Worker genuinely signed the perturbed result!
        assert verify_tile_result_signature(
            worker.pubkey,
            task_id,
            res.result_bytes,
            res.worker_signature,
        ) is True


def test_mock_worker_fault_injection_tamper_signature():
    """WRK-14: set_tamper_signature(True) returns corrupted signature string."""
    with MockWorkerNode() as worker:
        worker.set_bypass_l402(True)
        worker.set_tamper_signature(True)
        stub = worker.create_stub()
        task_id = "task_bad_sig"
        stub.RequestTileTask(pb2.TileTaskRequest(task_id=task_id, matrix_a_rows=4, matrix_a_cols=4, matrix_b_cols=4, dtype="fp32"))
        chunks = _make_stream_chunks(task_id, b"\x00" * 64, b"\x00" * 64)
        res = stub.ExecuteTileTask(iter(chunks), metadata=[("x-task-id", task_id)])

        # Signature verification must fail!
        assert verify_tile_result_signature(
            worker.pubkey,
            task_id,
            res.result_bytes,
            res.worker_signature,
        ) is False


def test_mock_worker_fault_injection_reject_handshake():
    """WRK-15: set_reject_handshake(True) returns TileTaskAck(is_ready=False)."""
    with MockWorkerNode() as worker:
        worker.set_reject_handshake(True, message="Worker overloaded")
        stub = worker.create_stub()
        ack = stub.RequestTileTask(pb2.TileTaskRequest(task_id="task_rej", matrix_a_rows=4, matrix_a_cols=4, matrix_b_cols=4, dtype="fp32"))
        assert ack.is_ready is False
        assert ack.message == "Worker overloaded"


def test_mock_worker_fault_injection_drop_connection():
    """WRK-16: set_drop_connection(True) aborts RPC with UNAVAILABLE."""
    with MockWorkerNode() as worker:
        worker.set_drop_connection(True)
        stub = worker.create_stub()
        with pytest.raises(grpc.RpcError) as exc:
            stub.RequestTileTask(pb2.TileTaskRequest(task_id="task_drop", matrix_a_rows=4, matrix_a_cols=4, matrix_b_cols=4, dtype="fp32"))
        assert exc.value.code() == grpc.StatusCode.UNAVAILABLE


def test_mock_worker_publish_to_relay_integration():
    """WRK-17: Publishes announce payload to MockNostrRelay and is discoverable via query_workers."""
    relay = MockNostrRelay()
    with MockWorkerNode(vram_mb=24576, sats_per_gflop=0.0004) as worker:
        event_id = worker.publish_to_relay(relay)
        assert event_id is not None

        req = WorkerQueryRequest(min_vram_mb=16384, max_price_sats_per_gflop=0.001, required_slots=1)
        resp = relay.query_workers(req)
        assert resp.status == "SUCCESS"
        assert len(resp.matched_workers) == 1
        assert resp.matched_workers[0].worker_pubkey == worker.pubkey
        assert resp.matched_workers[0].p2p_multiaddr == worker.multiaddr
        assert resp.matched_workers[0].sats_per_gflop == 0.0004
