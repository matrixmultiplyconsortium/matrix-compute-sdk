"""Adversarial stress test suite for hermetic test doubles: MockNostrRelay, MockSettlementService, MockWorkerNode.
Authored by teamwork_preview_challenger_sdk_m1_2 (Test Doubles Challenger).
"""

import base64
import hashlib
import json
import secrets
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


# Helper to build chunk streams
def _make_stream(task_id: str, a_bytes: bytes, b_bytes: bytes, chunk_size: int = 512):
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


# ==============================================================================
# Group 1: MockNostrRelay Adversarial Tests (ADV-REL-01 through ADV-REL-07)
# ==============================================================================

def test_adv_rel_01_high_volume_concurrent_replacement():
    """ADV-REL-01: 50 concurrent threads publishing 100 updates each.
    Verifies no deadlocks, no iteration corruption, and newest timestamp persists.
    """
    relay = MockNostrRelay()
    errors = []
    num_threads = 50
    updates_per_thread = 100

    def publisher(pubkey_idx: int):
        try:
            pubkey = f"pk_worker_{pubkey_idx % 5}"  # 5 workers contending
            for step in range(updates_per_thread):
                ev = {
                    "pubkey": pubkey,
                    "kind": 30000,
                    "created_at": 1000 + step,
                    "tags": [["d", "matrix-worker"], ["vram_mb", str(1024 * (step + 1))]],
                    "content": "{}",
                }
                relay.publish(ev)
        except Exception as ex:
            errors.append(ex)

    threads = [threading.Thread(target=publisher, args=(i,)) for i in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0
    # There should be exactly 5 active announcements
    events = relay.get_events()
    assert len(events) == 5
    for ev in events:
        assert ev["created_at"] == 1000 + (updates_per_thread - 1)


def test_adv_rel_02_rapid_out_of_order_timestamp_eviction():
    """ADV-REL-02: Out-of-order timestamps: newest event published first, older events discarded."""
    relay = MockNostrRelay()
    pubkey = "pk_eviction_target"

    # Publish newest event first (T = 5000)
    ev_newest = {
        "pubkey": pubkey,
        "kind": 30000,
        "created_at": 5000,
        "tags": [["d", "matrix-worker"], ["vram_mb", "49152"]],
        "content": "{}",
    }
    assert relay.publish(ev_newest) is True

    # Publish progressively older events (T = 4000, 3000, 2000, 1000)
    for old_ts in [4000, 3000, 2000, 1000]:
        ev_old = {
            "pubkey": pubkey,
            "kind": 30000,
            "created_at": old_ts,
            "tags": [["d", "matrix-worker"], ["vram_mb", "8192"]],
            "content": "{}",
        }
        assert relay.publish(ev_old) is True

    active = relay.get_active_announcement(pubkey)
    assert active is not None
    assert active["created_at"] == 5000
    specs = MockNostrRelay._parse_event_specs(active)
    assert specs["vram_mb"] == 49152


def test_adv_rel_03_malformed_tag_resilience():
    """ADV-REL-03: Malformed tag structures (empty lists, 1-tuples, non-numerics) handled safely."""
    relay = MockNostrRelay()
    event = {
        "pubkey": "pk_malformed_tags",
        "kind": 30000,
        "created_at": 1234,
        "tags": [
            [],  # empty
            ["d"],  # 1-tuple missing value
            ["vram_mb", "not_a_valid_number"],  # non-numeric vram
            ["sats_per_gflop", "infinite_sats"],  # non-numeric price
            ["unknown_key", "val", "extra"],  # 3-tuple
            ["gpu_model", "Special-GPU-X"],
        ],
        "content": "{}",
    }
    assert relay.publish(event) is True
    active = relay.get_active_announcement("pk_malformed_tags", d_tag="matrix-worker")
    assert active is not None
    specs = MockNostrRelay._parse_event_specs(active)
    # Defaults should be preserved safely
    assert specs["vram_mb"] == 0
    assert specs["price"] == 0.0
    assert specs["gpu_model"] == "Special-GPU-X"


def test_adv_rel_04_empty_filters_and_boundary_queries():
    """ADV-REL-04: Queries with zero, negative, or substring edge cases."""
    relay = MockNostrRelay()
    relay.publish({
        "pubkey": "free_node",
        "kind": 30000,
        "tags": [["d", "matrix-worker"], ["vram_mb", "0"], ["sats_per_gflop", "0.0"], ["gpu_model", "CPU-Only"]],
        "content": "{}",
    })
    relay.publish({
        "pubkey": "gpu_node",
        "kind": 30000,
        "tags": [["d", "matrix-worker"], ["vram_mb", "24576"], ["sats_per_gflop", "0.0005"], ["gpu_model", "NVIDIA RTX 4090"]],
        "content": "{}",
    })

    # Zero min_vram and zero max_price should find free_node
    res_free = relay.query(min_vram_mb=0, max_price=0.0)
    assert len(res_free) == 1
    assert res_free[0]["pubkey"] == "free_node"

    # Negative min_vram should include both
    res_neg = relay.query(min_vram_mb=-100)
    assert len(res_neg) == 2

    # Case insensitive GPU model
    res_model = relay.query(gpu_model="rtx 4090")
    assert len(res_model) == 1
    assert res_model[0]["pubkey"] == "gpu_node"

    # Non-existent GPU model
    res_none = relay.query(gpu_model="Quantum-TPU-999")
    assert len(res_none) == 0


def test_adv_rel_05_json_content_fallback_corrupted():
    """ADV-REL-05: Corrupted or non-dict content handled without unhandled exceptions."""
    relay = MockNostrRelay()
    # Bad JSON string
    relay.publish({
        "pubkey": "pk_bad_json",
        "kind": 30000,
        "tags": [["d", "matrix-worker"]],
        "content": "{invalid json: not closed",
    })
    active1 = relay.get_active_announcement("pk_bad_json")
    specs1 = MockNostrRelay._parse_event_specs(active1)
    assert specs1["vram_mb"] == 0
    assert specs1["price"] == 0.0

    # JSON Array instead of dict
    relay.publish({
        "pubkey": "pk_array_json",
        "kind": 30000,
        "tags": [["d", "matrix-worker"]],
        "content": "[1, 2, 3]",
    })
    active2 = relay.get_active_announcement("pk_array_json")
    specs2 = MockNostrRelay._parse_event_specs(active2)
    assert specs2["vram_mb"] == 0


def test_adv_rel_06_tag_index_cleanliness_after_replacement():
    """ADV-REL-06: Replacing an event removes old event_id and tag indices."""
    relay = MockNostrRelay()
    ev1 = {
        "pubkey": "pk_replace_target",
        "kind": 30000,
        "created_at": 1000,
        "tags": [["d", "matrix-worker"], ["gpu_model", "OldModel"]],
        "content": "{}",
    }
    id1 = relay.publish_event(ev1)
    assert id1 in relay._events_by_id
    assert id1 in relay._tag_index.get("gpu_model", {}).get("OldModel", set())

    # Replace with newer event
    ev2 = {
        "pubkey": "pk_replace_target",
        "kind": 30000,
        "created_at": 2000,
        "tags": [["d", "matrix-worker"], ["gpu_model", "NewModel"]],
        "content": "{}",
    }
    id2 = relay.publish_event(ev2)
    assert id2 in relay._events_by_id
    # id1 must be purged from _events_by_id and old tag set
    assert id1 not in relay._events_by_id
    assert id1 not in relay._tag_index.get("gpu_model", {}).get("OldModel", set())
    assert id2 in relay._tag_index.get("gpu_model", {}).get("NewModel", set())


def test_adv_rel_07_pareto_matchmaking_tiebreaking():
    """ADV-REL-07: Pareto ranking strictly resolves ties: equal price -> higher VRAM wins."""
    relay = MockNostrRelay()
    # 3 workers with identical price (0.0005) but different VRAM
    relay.publish({"pubkey": "node_v10", "kind": 30000, "tags": [["d", "matrix-worker"], ["p2p", "p1"], ["lightning", "l1"], ["sats_per_gflop", "0.0005"], ["vram_mb", "10000"]], "content": "{}"})
    relay.publish({"pubkey": "node_v30", "kind": 30000, "tags": [["d", "matrix-worker"], ["p2p", "p2"], ["lightning", "l2"], ["sats_per_gflop", "0.0005"], ["vram_mb", "30000"]], "content": "{}"})
    relay.publish({"pubkey": "node_v20", "kind": 30000, "tags": [["d", "matrix-worker"], ["p2p", "p3"], ["lightning", "l3"], ["sats_per_gflop", "0.0005"], ["vram_mb", "20000"]], "content": "{}"})

    req = WorkerQueryRequest(min_vram_mb=8000, max_price_sats_per_gflop=0.01, required_slots=3)
    resp = relay.query_workers(req)
    assert resp.status == "SUCCESS"
    assert [w.worker_pubkey for w in resp.matched_workers] == ["node_v30", "node_v20", "node_v10"]


# ==============================================================================
# Group 2: MockSettlementService Adversarial Tests (ADV-SET-01 through ADV-SET-08)
# ==============================================================================

def test_adv_set_01_duplicate_collisions_rejected():
    """ADV-SET-01: Duplicate task_id or payment_hash raises ValueError and leaves state intact."""
    settlement = MockSettlementService()
    p_hash = hashlib.sha256(b"preimage_1").hexdigest()
    settlement.create_hodl_invoice(task_id="task_unique", payment_hash=p_hash)

    # Re-use task_id
    with pytest.raises(ValueError) as exc1:
        settlement.create_hodl_invoice(task_id="task_unique", payment_hash=hashlib.sha256(b"other").hexdigest())
    assert "Task ID already registered" in str(exc1.value)

    # Re-use payment_hash
    with pytest.raises(ValueError) as exc2:
        settlement.create_hodl_invoice(task_id="task_different", payment_hash=p_hash)
    assert "Duplicate payment hash" in str(exc2.value)


def test_adv_set_02_unauthorized_settlement_bypass_rejected():
    """ADV-SET-02: Direct settlement of UNPAID invoice without prior holding is rejected."""
    settlement = MockSettlementService()
    secret = b"\x77" * 32
    p_hash = hashlib.sha256(secret).hexdigest()
    settlement.create_hodl_invoice(task_id="task_bypass", payment_hash=p_hash)

    # Invoice is UNPAID; attempting to settle must fail
    with pytest.raises(ValueError) as exc:
        settlement.settle_invoice(p_hash, preimage=secret.hex())
    assert "must be HELD" in str(exc.value)
    assert settlement.get_status(p_hash) == "UNPAID"


def test_adv_set_03_double_transitions_and_cancel_rules():
    """ADV-SET-03: Double-hold, double-settle, and cancelling a SETTLED invoice are rejected."""
    settlement = MockSettlementService()
    resp = settlement.create_hodl_invoice(task_id="task_double")
    p_hash = resp.payment_hash
    settlement.hold_invoice(p_hash)

    # Double hold
    with pytest.raises(ValueError) as exc_hold:
        settlement.hold_invoice(p_hash)
    assert "must be UNPAID" in str(exc_hold.value)

    # Settle
    settlement.settle_invoice(p_hash, preimage=resp.preimage)
    assert settlement.get_status(p_hash) == "SETTLED"

    # Double settle
    with pytest.raises(ValueError) as exc_settle:
        settlement.settle_invoice(p_hash, preimage=resp.preimage)
    assert "must be HELD" in str(exc_settle.value)

    # Cancel settled invoice
    with pytest.raises(ValueError) as exc_cancel:
        settlement.cancel_invoice(p_hash)
    assert "Cannot cancel an already SETTLED invoice" in str(exc_cancel.value)


def test_adv_set_04_preimage_verification_edge_cases():
    """ADV-SET-04: Malformed hex, length mismatch, wrong preimage, and missing preimage handled."""
    settlement = MockSettlementService()
    secret = b"\x88" * 32
    p_hash = hashlib.sha256(secret).hexdigest()
    settlement.create_hodl_invoice(task_id="task_preimage_edge", payment_hash=p_hash)
    settlement.hold_invoice(p_hash)

    # Malformed non-hex string
    with pytest.raises(ValueError) as exc_hex:
        settlement.settle_invoice(p_hash, preimage="1234nothex")
    assert "valid hexadecimal" in str(exc_hex.value)

    # Odd length hex string (31 hex chars)
    with pytest.raises(ValueError) as exc_odd:
        settlement.settle_invoice(p_hash, preimage="abcdef1234567890abcdef123456789")
    assert "valid hexadecimal" in str(exc_odd.value)

    # Wrong preimage
    with pytest.raises(ValueError) as exc_wrong:
        settlement.settle_invoice(p_hash, preimage=(b"\x00" * 32).hex())
    assert "does not match invoice hash" in str(exc_wrong.value)

    # Preimage is None on trustless invoice
    with pytest.raises(ValueError) as exc_none:
        settlement.settle_invoice(p_hash, preimage=None)
    assert "Preimage is required" in str(exc_none.value)

    assert settlement.get_status(p_hash) == "HELD"


def test_adv_set_05_macaroon_expiry_boundary():
    """ADV-SET-05: Expired macaroon token is rejected with explicit reason."""
    settlement = MockSettlementService()
    # Expires in 10 seconds
    resp = settlement.create_hodl_invoice(task_id="task_expiry_bnd", expiry_seconds=10)
    # Valid now
    ok_now, _ = settlement.verify_macaroon(resp.macaroon, "task_expiry_bnd", 1.0)
    assert ok_now is True

    # Check 11 seconds later
    ok_later, msg_later = settlement.verify_macaroon(
        resp.macaroon, "task_expiry_bnd", 1.0, current_time=time.time() + 11
    )
    assert ok_later is False
    assert "expired" in msg_later


def test_adv_set_06_macaroon_hmac_tamper_detection():
    """ADV-SET-06: Tampering with any caveat field invalidates HMAC signature."""
    settlement = MockSettlementService(verification_key="secret-key-alpha")
    resp = settlement.create_hodl_invoice(task_id="task_tamper_target", max_gflops=10.0)

    raw_payload = json.loads(base64.b64decode(resp.macaroon).decode("utf-8"))

    for field in ["expiry", "locked_sats", "max_gflops", "payment_hash", "task_id"]:
        bad_payload = json.loads(json.dumps(raw_payload))
        if field in ("expiry", "locked_sats"):
            bad_payload["caveats"][field] += 1
        elif field == "max_gflops":
            bad_payload["caveats"][field] = 9999.0
        else:
            bad_payload["caveats"][field] = f"tampered_{field}"

        tampered_mac = base64.b64encode(json.dumps(bad_payload).encode("utf-8")).decode("utf-8")
        ok, msg = settlement.verify_macaroon(tampered_mac, "task_tamper_target", 1.0)
        assert ok is False
        assert "Invalid macaroon HMAC signature" in msg


def test_adv_set_07_insufficient_gflop_allowance_rejected():
    """ADV-SET-07: Request requiring more GFLOPs than caveat allowance is rejected."""
    settlement = MockSettlementService()
    resp = settlement.create_hodl_invoice(task_id="task_gflop_bnd", max_gflops=5.0)
    # 5.0 GFLOP allowed, 5.1 required -> False
    ok, msg = settlement.verify_macaroon(resp.macaroon, "task_gflop_bnd", required_gflops=5.1)
    assert ok is False
    assert "Insufficient GFLOP allowance" in msg


def test_adv_set_08_concurrent_independent_invoices():
    """ADV-SET-08: 30 concurrent threads holding and settling independent invoices."""
    settlement = MockSettlementService()
    errors = []

    def task_runner(idx: int):
        try:
            secret = f"con_secret_{idx:04d}".encode("utf-8").ljust(32, b"\x00")
            p_hash = hashlib.sha256(secret).hexdigest()
            settlement.create_hodl_invoice(task_id=f"concurrent_task_{idx}", payment_hash=p_hash)
            settlement.hold_invoice(p_hash)
            settlement.settle_invoice(p_hash, preimage=secret.hex())
            assert settlement.get_status(p_hash) == "SETTLED"
        except Exception as ex:
            errors.append(ex)

    threads = [threading.Thread(target=task_runner, args=(i,)) for i in range(30)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0


# ==============================================================================
# Group 3: MockWorkerNode Adversarial Tests (ADV-WRK-01 through ADV-WRK-10)
# ==============================================================================

def test_adv_wrk_01_out_of_order_chunk_sequence_jumping():
    """ADV-WRK-01: Sequence jumping (0 -> 2) aborts immediately with INVALID_ARGUMENT."""
    with MockWorkerNode() as worker:
        worker.set_bypass_l402(True)
        stub = worker.create_stub()
        task_id = "task_seq_jump"
        stub.RequestTileTask(pb2.TileTaskRequest(
            task_id=task_id, matrix_a_rows=4, matrix_a_cols=4, matrix_b_cols=4, dtype="fp32"
        ))

        # Chunk 0 then Chunk 2 (skipping 1)
        chunks = [
            pb2.TileDataChunk(task_id=task_id, sequence_number=0, chunk_bytes=b"\x00" * 32, is_matrix_a=True, is_final_chunk=False),
            pb2.TileDataChunk(task_id=task_id, sequence_number=2, chunk_bytes=b"\x00" * 32, is_matrix_a=True, is_final_chunk=True),
        ]
        with pytest.raises(grpc.RpcError) as exc:
            stub.ExecuteTileTask(iter(chunks), metadata=[("x-task-id", task_id)])
        assert exc.value.code() == grpc.StatusCode.INVALID_ARGUMENT
        assert "Out of order sequence" in exc.value.details()


def test_adv_wrk_02_stream_byte_bound_overrun_matrix_a():
    """ADV-WRK-02: Matrix A accumulated bytes exceeding tensor limit aborts before buffering."""
    with MockWorkerNode() as worker:
        worker.set_bypass_l402(True)
        stub = worker.create_stub()
        task_id = "task_overrun_a"
        # 4x4 FP32 = 64 bytes for Matrix A
        stub.RequestTileTask(pb2.TileTaskRequest(
            task_id=task_id, matrix_a_rows=4, matrix_a_cols=4, matrix_b_cols=4, dtype="fp32"
        ))

        # First chunk 40 bytes (valid), second chunk 30 bytes (40+30=70 > 64 limit!)
        chunks = [
            pb2.TileDataChunk(task_id=task_id, sequence_number=0, chunk_bytes=b"\x00" * 40, is_matrix_a=True, is_final_chunk=False),
            pb2.TileDataChunk(task_id=task_id, sequence_number=1, chunk_bytes=b"\x00" * 30, is_matrix_a=True, is_final_chunk=False),
        ]
        with pytest.raises(grpc.RpcError) as exc:
            stub.ExecuteTileTask(iter(chunks), metadata=[("x-task-id", task_id)])
        assert exc.value.code() == grpc.StatusCode.INVALID_ARGUMENT
        assert "Matrix A stream exceeded limit" in exc.value.details()


def test_adv_wrk_03_stream_byte_bound_overrun_matrix_b():
    """ADV-WRK-03: Matrix B accumulated bytes exceeding tensor limit aborts before buffering."""
    with MockWorkerNode() as worker:
        worker.set_bypass_l402(True)
        stub = worker.create_stub()
        task_id = "task_overrun_b"
        # 4x4 FP32 = 64 bytes each
        stub.RequestTileTask(pb2.TileTaskRequest(
            task_id=task_id, matrix_a_rows=4, matrix_a_cols=4, matrix_b_cols=4, dtype="fp32"
        ))

        # Matrix A valid 64 bytes; Matrix B 70 bytes (> 64)
        chunks = [
            pb2.TileDataChunk(task_id=task_id, sequence_number=0, chunk_bytes=b"\x00" * 64, is_matrix_a=True, is_final_chunk=False),
            pb2.TileDataChunk(task_id=task_id, sequence_number=1, chunk_bytes=b"\x00" * 70, is_matrix_a=False, is_final_chunk=False),
        ]
        with pytest.raises(grpc.RpcError) as exc:
            stub.ExecuteTileTask(iter(chunks), metadata=[("x-task-id", task_id)])
        assert exc.value.code() == grpc.StatusCode.INVALID_ARGUMENT
        assert "Matrix B stream exceeded limit" in exc.value.details()


def test_adv_wrk_04_abrupt_stream_termination_without_final_flag():
    """ADV-WRK-04: Stream terminating without is_final_chunk=True aborts with INVALID_ARGUMENT."""
    with MockWorkerNode() as worker:
        worker.set_bypass_l402(True)
        stub = worker.create_stub()
        task_id = "task_abrupt_term"
        stub.RequestTileTask(pb2.TileTaskRequest(
            task_id=task_id, matrix_a_rows=4, matrix_a_cols=4, matrix_b_cols=4, dtype="fp32"
        ))

        chunks = [
            pb2.TileDataChunk(task_id=task_id, sequence_number=0, chunk_bytes=b"\x00" * 64, is_matrix_a=True, is_final_chunk=False),
            pb2.TileDataChunk(task_id=task_id, sequence_number=1, chunk_bytes=b"\x00" * 64, is_matrix_a=False, is_final_chunk=False),
        ]
        with pytest.raises(grpc.RpcError) as exc:
            stub.ExecuteTileTask(iter(chunks), metadata=[("x-task-id", task_id)])
        assert exc.value.code() == grpc.StatusCode.INVALID_ARGUMENT
        assert "Stream ended without final chunk flag" in exc.value.details()


def test_adv_wrk_05_incomplete_bytes_with_final_flag():
    """ADV-WRK-05: Stream declaring is_final_chunk=True before all bytes sent aborts with byte mismatch."""
    with MockWorkerNode() as worker:
        worker.set_bypass_l402(True)
        stub = worker.create_stub()
        task_id = "task_incomplete"
        # 4x4 FP32 requires 64 bytes each
        stub.RequestTileTask(pb2.TileTaskRequest(
            task_id=task_id, matrix_a_rows=4, matrix_a_cols=4, matrix_b_cols=4, dtype="fp32"
        ))

        chunks = [
            pb2.TileDataChunk(task_id=task_id, sequence_number=0, chunk_bytes=b"\x00" * 32, is_matrix_a=True, is_final_chunk=False),
            pb2.TileDataChunk(task_id=task_id, sequence_number=1, chunk_bytes=b"\x00" * 32, is_matrix_a=False, is_final_chunk=True),
        ]
        with pytest.raises(grpc.RpcError) as exc:
            stub.ExecuteTileTask(iter(chunks), metadata=[("x-task-id", task_id)])
        assert exc.value.code() == grpc.StatusCode.INVALID_ARGUMENT
        assert "Byte count mismatch" in exc.value.details()


def test_adv_wrk_06_unnegotiated_task_execution_aborts():
    """ADV-WRK-06: Calling ExecuteTileTask without prior RequestTileTask aborts with FAILED_PRECONDITION."""
    with MockWorkerNode() as worker:
        worker.set_bypass_l402(True)
        stub = worker.create_stub()
        unnegotiated_id = "ghost_task_id"
        chunks = [
            pb2.TileDataChunk(task_id=unnegotiated_id, sequence_number=0, chunk_bytes=b"\x00" * 16, is_matrix_a=True, is_final_chunk=True)
        ]
        with pytest.raises(grpc.RpcError) as exc:
            stub.ExecuteTileTask(iter(chunks), metadata=[("x-task-id", unnegotiated_id)])
        assert exc.value.code() == grpc.StatusCode.FAILED_PRECONDITION
        assert "was not negotiated" in exc.value.details()


def test_adv_wrk_07_task_metadata_purged_on_finish_and_failure():
    """ADV-WRK-07: Task metadata is purged after execution (successful or failed), preventing memory leaks."""
    with MockWorkerNode() as worker:
        worker.set_bypass_l402(True)
        stub = worker.create_stub()

        # 1. Success case
        task_id_succ = "task_meta_succ"
        stub.RequestTileTask(pb2.TileTaskRequest(task_id=task_id_succ, matrix_a_rows=2, matrix_a_cols=2, matrix_b_cols=2, dtype="fp32"))
        assert task_id_succ in worker.servicer._task_metadata
        chunks = _make_stream(task_id_succ, b"\x00" * 16, b"\x00" * 16)
        stub.ExecuteTileTask(iter(chunks), metadata=[("x-task-id", task_id_succ)])
        assert task_id_succ not in worker.servicer._task_metadata

        # 2. Failure case (overflow)
        task_id_fail = "task_meta_fail"
        stub.RequestTileTask(pb2.TileTaskRequest(task_id=task_id_fail, matrix_a_rows=2, matrix_a_cols=2, matrix_b_cols=2, dtype="fp32"))
        assert task_id_fail in worker.servicer._task_metadata
        bad_chunks = [pb2.TileDataChunk(task_id=task_id_fail, sequence_number=0, chunk_bytes=b"\x00" * 100, is_matrix_a=True, is_final_chunk=False)]
        with pytest.raises(grpc.RpcError):
            stub.ExecuteTileTask(iter(bad_chunks), metadata=[("x-task-id", task_id_fail)])
        # Must be purged even after exception
        assert task_id_fail not in worker.servicer._task_metadata


def test_adv_wrk_08_fault_injection_hooks_comprehensive():
    """ADV-WRK-08: Tests all fault injection hooks: tamper result, tamper sig, drop conn, reject handshake."""
    settlement = MockSettlementService()
    with MockWorkerNode(settlement_service=settlement) as worker:
        stub = worker.create_stub()

        # Hook 1: Reject Handshake
        worker.set_reject_handshake(True, message="Worker out of memory")
        ack = stub.RequestTileTask(pb2.TileTaskRequest(task_id="t_rej", matrix_a_rows=4, matrix_a_cols=4, matrix_b_cols=4, dtype="fp32"))
        assert ack.is_ready is False
        assert ack.message == "Worker out of memory"
        worker.set_reject_handshake(False)

        # Hook 2: Drop Connection on Handshake
        worker.set_drop_connection(True)
        with pytest.raises(grpc.RpcError) as exc_drop:
            stub.RequestTileTask(pb2.TileTaskRequest(task_id="t_drop", matrix_a_rows=4, matrix_a_cols=4, matrix_b_cols=4, dtype="fp32"))
        assert exc_drop.value.code() == grpc.StatusCode.UNAVAILABLE
        worker.set_drop_connection(False)

        # Hook 3: Tamper Result (worker genuinely signs fraudulent math)
        worker.set_bypass_l402(True)
        worker.set_tamper_result(True, perturbation=99.0)
        stub.RequestTileTask(pb2.TileTaskRequest(task_id="t_fraud", matrix_a_rows=2, matrix_a_cols=2, matrix_b_cols=2, dtype="fp32"))
        mat_a = np.zeros((2, 2), dtype=np.float32)
        mat_b = np.zeros((2, 2), dtype=np.float32)
        res = stub.ExecuteTileTask(iter(_make_stream("t_fraud", mat_a.tobytes(), mat_b.tobytes())), metadata=[("x-task-id", "t_fraud")])
        res_mat = np.frombuffer(res.result_bytes, dtype=np.float32).reshape((2, 2))
        assert res_mat[0, 0] == 99.0  # Mathematically invalid!
        # But genuine worker signature present!
        assert verify_tile_result_signature(worker.pubkey, "t_fraud", res.result_bytes, res.worker_signature) is True
        worker.set_tamper_result(False)

        # Hook 4: Tamper Signature (corrupted signature)
        worker.set_tamper_signature(True)
        stub.RequestTileTask(pb2.TileTaskRequest(task_id="t_badsig", matrix_a_rows=2, matrix_a_cols=2, matrix_b_cols=2, dtype="fp32"))
        res_badsig = stub.ExecuteTileTask(iter(_make_stream("t_badsig", mat_a.tobytes(), mat_b.tobytes())), metadata=[("x-task-id", "t_badsig")])
        assert verify_tile_result_signature(worker.pubkey, "t_badsig", res_badsig.result_bytes, res_badsig.worker_signature) is False


def test_adv_wrk_09_l402_pre_allocation_gate_adversarial_rejections():
    """ADV-WRK-09: Unheld invoice, task mismatch, hash mismatch, and underpayment all abort with correct gRPC codes."""
    settlement = MockSettlementService()
    with MockWorkerNode(settlement_service=settlement, sats_per_gflop=0.001, min_task_sats=100) as worker:
        stub = worker.create_stub()

        # Case 1: Unheld Invoice (UNPAID) -> FAILED_PRECONDITION
        task_id = "t_unheld_adv"
        stub.RequestTileTask(pb2.TileTaskRequest(task_id=task_id, matrix_a_rows=4, matrix_a_cols=4, matrix_b_cols=4, dtype="fp32"))
        inv = settlement.create_hodl_invoice(task_id=task_id, total_compute_sats=200, max_gflops=10.0)
        chunks = _make_stream(task_id, b"\x00" * 64, b"\x00" * 64)
        with pytest.raises(grpc.RpcError) as exc_unheld:
            stub.ExecuteTileTask(iter(chunks), metadata=[("authorization", f"L402 {inv.macaroon}:{inv.payment_hash}"), ("x-task-id", task_id)])
        assert exc_unheld.value.code() == grpc.StatusCode.FAILED_PRECONDITION

        # Case 2: Caveat Task ID mismatch -> PERMISSION_DENIED
        settlement.hold_invoice(inv.payment_hash)
        task_id_other = "t_other_adv"
        stub.RequestTileTask(pb2.TileTaskRequest(task_id=task_id_other, matrix_a_rows=4, matrix_a_cols=4, matrix_b_cols=4, dtype="fp32"))
        chunks_other = _make_stream(task_id_other, b"\x00" * 64, b"\x00" * 64)
        with pytest.raises(grpc.RpcError) as exc_task_mismatch:
            stub.ExecuteTileTask(iter(chunks_other), metadata=[("authorization", f"L402 {inv.macaroon}:{inv.payment_hash}"), ("x-task-id", task_id_other)])
        assert exc_task_mismatch.value.code() == grpc.StatusCode.PERMISSION_DENIED

        # Case 3: Caveat Payment Hash mismatch -> PERMISSION_DENIED
        with pytest.raises(grpc.RpcError) as exc_hash_mismatch:
            stub.ExecuteTileTask(iter(chunks), metadata=[("authorization", f"L402 {inv.macaroon}:00112233445566778899001122334455"), ("x-task-id", task_id)])
        assert exc_hash_mismatch.value.code() == grpc.StatusCode.PERMISSION_DENIED

        # Case 4: Underpayment (locked_sats < min_task_sats 100) -> FAILED_PRECONDITION
        task_id_cheap = "t_cheap_adv"
        stub.RequestTileTask(pb2.TileTaskRequest(task_id=task_id_cheap, matrix_a_rows=4, matrix_a_cols=4, matrix_b_cols=4, dtype="fp32"))
        inv_cheap = settlement.create_hodl_invoice(task_id=task_id_cheap, total_compute_sats=10, max_gflops=10.0)
        settlement.hold_invoice(inv_cheap.payment_hash)
        chunks_cheap = _make_stream(task_id_cheap, b"\x00" * 64, b"\x00" * 64)
        with pytest.raises(grpc.RpcError) as exc_underpay:
            stub.ExecuteTileTask(iter(chunks_cheap), metadata=[("authorization", f"L402 {inv_cheap.macaroon}:{inv_cheap.payment_hash}"), ("x-task-id", task_id_cheap)])
        assert exc_underpay.value.code() == grpc.StatusCode.FAILED_PRECONDITION


def test_adv_wrk_10_bf16_mantissa_zeroing_precision():
    """ADV-WRK-10: BF16 emulation zeroes lower 16 bits of FP32 mantissa."""
    with MockWorkerNode() as worker:
        worker.set_bypass_l402(True)
        stub = worker.create_stub()
        task_id = "t_bf16_bits"
        m, k, n = 4, 4, 4
        stub.RequestTileTask(pb2.TileTaskRequest(task_id=task_id, matrix_a_rows=m, matrix_a_cols=k, matrix_b_cols=n, dtype="bf16"))

        # Non-trivial floats
        np.random.seed(123)
        mat_a = np.random.randn(m, k).astype(np.float32)
        mat_b = np.random.randn(k, n).astype(np.float32)

        chunks = _make_stream(task_id, mat_a.tobytes(), mat_b.tobytes())
        res = stub.ExecuteTileTask(iter(chunks), metadata=[("x-task-id", task_id)])

        res_u32 = np.frombuffer(res.result_bytes, dtype=np.uint32)
        # Every single element's lower 16 bits must be strictly 0x0000
        assert np.all((res_u32 & 0x0000FFFF) == 0)
