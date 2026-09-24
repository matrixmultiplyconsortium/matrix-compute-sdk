"""
Comprehensive Unit Tests for Payment Subsystem in matrix-compute-sdk.

Tests L402 challenge parsing, Macaroon base64/JSON token validation, HMAC verification,
BudgetController limits, EscrowManager atomic state transitions (INITIALIZED -> HELD -> SETTLED / CANCELLED / EXPIRED),
timeout sweeps, secret preimage redaction, and Freivalds verify_and_settle integration with fraud proofs.
"""

import base64
import hashlib
import hmac
import json
import threading
import time
from typing import Any, Dict
import numpy as np
import pytest

from matrix_compute.config import ClientConfig
from matrix_compute.crypto.signer import ClientSigner
from matrix_compute.exceptions import (
    BudgetExceededError,
    EscrowCreationError,
    EscrowError,
    EscrowStateError,
    EscrowTimeoutError,
    L402ChallengeError,
    MacaroonError,
    MacaroonValidationError,
    PaymentError,
    PreimageVerificationError,
    SettlementError,
)
from matrix_compute.payment.escrow import (
    EscrowManager,
    EscrowState,
    EscrowStatus,
    generate_escrow_secret,
    validate_preimage,
)
from matrix_compute.payment.l402_client import (
    BudgetController,
    L402Challenge,
    L402PaymentCoordinator,
    MacaroonCaveats,
    MacaroonToken,
    MockSettlementBackend,
    format_auth_header,
    format_grpc_metadata,
)
from matrix_compute.test_doubles.mock_settlement import MockSettlementService


# -----------------------------------------------------------------------------
# Fixtures & Helpers
# -----------------------------------------------------------------------------

@pytest.fixture
def verification_key() -> str:
    return "test-verification-secret-key-12345"


@pytest.fixture
def mock_settlement(verification_key) -> MockSettlementService:
    return MockSettlementService(verification_key=verification_key)


@pytest.fixture
def escrow_mgr(mock_settlement) -> EscrowManager:
    return EscrowManager(settlement_backend=mock_settlement, default_timeout_seconds=60.0)


def create_signed_macaroon(
    task_id: str,
    payment_hash: str,
    expiry: int,
    max_gflops: float,
    locked_sats: int,
    verification_key: str,
) -> str:
    caveats = {
        "expiry": int(expiry),
        "locked_sats": int(locked_sats),
        "max_gflops": float(max_gflops),
        "payment_hash": payment_hash,
        "task_id": task_id,
    }
    caveat_str = json.dumps(caveats, sort_keys=True)
    sig = hmac.new(
        verification_key.encode("utf-8"),
        caveat_str.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    payload = {"caveats": caveats, "signature": sig}
    return base64.b64encode(json.dumps(payload).encode("utf-8")).decode("utf-8")


# =============================================================================
# 1. L402 Challenge Parsing Tests
# =============================================================================

def test_l402_challenge_parse_standard_quoted():
    header = 'WWW-Authenticate: L402 macaroon="b64macaroon123", invoice="lnbc100n1mock"'
    ch = L402Challenge.parse(header)
    assert ch.scheme == "L402"
    assert ch.macaroon == "b64macaroon123"
    assert ch.invoice == "lnbc100n1mock"


def test_l402_challenge_parse_lsat_scheme_unquoted():
    header = "LSAT macaroon=token_xyz, invoice=lnbc500n1mock"
    ch = L402Challenge.parse(header)
    assert ch.scheme == "LSAT"
    assert ch.macaroon == "token_xyz"
    assert ch.invoice == "lnbc500n1mock"


def test_l402_challenge_parse_extra_attributes():
    header = 'L402 macaroon="tok1", invoice="inv1", price_sats=50, realm="compute-matrix"'
    ch = L402Challenge.parse(header)
    assert ch.price_sats == 50
    assert ch.realm == "compute-matrix"


def test_l402_challenge_missing_macaroon_raises():
    header = 'L402 invoice="lnbc100n1mock"'
    with pytest.raises(L402ChallengeError):
        L402Challenge.parse(header)


def test_l402_challenge_missing_invoice_raises():
    header = 'L402 macaroon="b64tok"'
    with pytest.raises(L402ChallengeError):
        L402Challenge.parse(header)


def test_l402_challenge_unsupported_scheme_raises():
    header = 'Bearer token="abcdef123456"'
    with pytest.raises(L402ChallengeError) as exc_info:
        L402Challenge.parse(header)
    assert "scheme must be L402 or LSAT" in str(exc_info.value)


def test_l402_challenge_empty_header_raises():
    with pytest.raises(L402ChallengeError):
        L402Challenge.parse("")


# =============================================================================
# 2. Macaroon Token Parsing & HMAC Verification Tests
# =============================================================================

def test_macaroon_deserialize_valid(verification_key):
    b64 = create_signed_macaroon(
        task_id="task_1",
        payment_hash="a" * 64,
        expiry=int(time.time() + 3600),
        max_gflops=10.0,
        locked_sats=100,
        verification_key=verification_key,
    )
    token = MacaroonToken.deserialize(b64)
    assert token.caveats.task_id == "task_1"
    assert token.caveats.payment_hash == "a" * 64
    assert token.caveats.max_gflops == 10.0
    assert token.caveats.locked_sats == 100


def test_macaroon_deserialize_urlsafe_and_padding(verification_key):
    b64 = create_signed_macaroon(
        task_id="task_pad",
        payment_hash="b" * 64,
        expiry=int(time.time() + 3600),
        max_gflops=5.0,
        locked_sats=50,
        verification_key=verification_key,
    )
    # Strip padding '='
    unpadded = b64.rstrip("=")
    token = MacaroonToken.deserialize(unpadded)
    assert token.caveats.task_id == "task_pad"


def test_macaroon_verify_hmac_pass(verification_key):
    b64 = create_signed_macaroon(
        task_id="task_sec",
        payment_hash="c" * 64,
        expiry=int(time.time() + 3600),
        max_gflops=1.0,
        locked_sats=10,
        verification_key=verification_key,
    )
    token = MacaroonToken.deserialize(b64)
    assert token.verify_hmac(verification_key) is True


def test_macaroon_verify_hmac_wrong_key_fails(verification_key):
    b64 = create_signed_macaroon(
        task_id="task_sec",
        payment_hash="c" * 64,
        expiry=int(time.time() + 3600),
        max_gflops=1.0,
        locked_sats=10,
        verification_key=verification_key,
    )
    token = MacaroonToken.deserialize(b64)
    assert token.verify_hmac("wrong-verification-key") is False


def test_macaroon_malformed_json_raises():
    with pytest.raises(MacaroonError):
        MacaroonToken.deserialize("bm90IGEgSlNPTg==")  # "not a JSON"


# =============================================================================
# 3. Macaroon Caveat Boundary Validation Tests
# =============================================================================

def test_macaroon_caveat_validation_happy_path(verification_key):
    now = time.time()
    b64 = create_signed_macaroon(
        task_id="tile_42",
        payment_hash="d" * 64,
        expiry=int(now + 600),
        max_gflops=20.0,
        locked_sats=500,
        verification_key=verification_key,
    )
    token = MacaroonToken.deserialize(b64)
    # 2 * 1024^3 / 1e9 = ~2.15 GFLOPs
    is_valid, reason = token.validate(
        task_id="tile_42",
        m=1024,
        k=1024,
        n=1024,
        expected_payment_hash="d" * 64,
        max_price_rate=0.0005,
    )
    assert is_valid is True
    assert reason is None


def test_macaroon_caveat_validation_task_id_mismatch(verification_key):
    b64 = create_signed_macaroon(
        task_id="tile_A",
        payment_hash="d" * 64,
        expiry=int(time.time() + 600),
        max_gflops=20.0,
        locked_sats=500,
        verification_key=verification_key,
    )
    token = MacaroonToken.deserialize(b64)
    is_valid, reason = token.validate(task_id="tile_B", m=512, k=512, n=512)
    assert is_valid is False
    assert "Caveat task_id mismatch" in reason


def test_macaroon_caveat_validation_expired(verification_key):
    past = int(time.time() - 100)
    b64 = create_signed_macaroon(
        task_id="tile_exp",
        payment_hash="d" * 64,
        expiry=past,
        max_gflops=20.0,
        locked_sats=500,
        verification_key=verification_key,
    )
    token = MacaroonToken.deserialize(b64)
    is_valid, reason = token.validate(task_id="tile_exp", m=512, k=512, n=512)
    assert is_valid is False
    assert "Macaroon expired" in reason


def test_macaroon_caveat_validation_insufficient_gflops(verification_key):
    b64 = create_signed_macaroon(
        task_id="tile_huge",
        payment_hash="d" * 64,
        expiry=int(time.time() + 600),
        max_gflops=1.0,  # Only 1 GFLOP allowed
        locked_sats=500,
        verification_key=verification_key,
    )
    token = MacaroonToken.deserialize(b64)
    # 2 * 2048^3 / 1e9 = 17.18 GFLOPs > 1.0
    is_valid, reason = token.validate(task_id="tile_huge", m=2048, k=2048, n=2048)
    assert is_valid is False
    assert "Insufficient GFLOP allowance" in reason


def test_macaroon_caveat_validation_underpayment(verification_key):
    b64 = create_signed_macaroon(
        task_id="tile_cheap",
        payment_hash="d" * 64,
        expiry=int(time.time() + 600),
        max_gflops=100.0,
        locked_sats=5,  # only 5 sats locked, but required is 10 (floor)
        verification_key=verification_key,
    )
    token = MacaroonToken.deserialize(b64)
    is_valid, reason = token.validate(
        task_id="tile_cheap",
        m=512,
        k=512,
        n=512,
        max_price_rate=0.0005,
        min_task_sats=10,
    )
    assert is_valid is False
    assert "Insufficient locked satoshis" in reason


# =============================================================================
# 4. Budget Controller Tests
# =============================================================================

def test_budget_controller_rate_limit():
    bc = BudgetController(max_price_per_gflop=0.001)
    bc.check_rate(0.0005)  # Under limit passes
    with pytest.raises(BudgetExceededError):
        bc.check_rate(0.0020)


def test_budget_controller_calculate_cost():
    bc = BudgetController(min_task_sats=10)
    # 2 * 1024^3 / 1e9 = 2.147 GFLOPs. 2.147 * 0.0005 = ~0.00107 sats -> min floor 10
    cost = bc.calculate_cost(1024, 1024, 1024, sats_per_gflop=0.0005)
    assert cost == 10

    # Large matrix: 2 * 4096^3 / 1e9 = 137.4 GFLOPs * 0.1 = 13.74 -> 14 sats
    cost2 = bc.calculate_cost(4096, 4096, 4096, sats_per_gflop=0.1)
    assert cost2 == 14


def test_budget_controller_reserve_commit_release():
    bc = BudgetController(max_budget_sats=1000, max_task_sats=200)

    # Reserve task 1
    bc.reserve("t1", 100)
    assert bc.total_reserved_sats == 100
    assert bc.total_committed_sats == 0

    # Commit task 1 upon pass
    bc.commit("t1")
    assert bc.total_reserved_sats == 0
    assert bc.total_committed_sats == 100

    # Reserve and release task 2 upon failure
    bc.reserve("t2", 50)
    assert bc.total_reserved_sats == 50
    bc.release("t2")
    assert bc.total_reserved_sats == 0
    assert bc.total_committed_sats == 100


def test_budget_controller_exceed_cumulative_budget():
    bc = BudgetController(max_budget_sats=100)
    bc.reserve("t1", 60)
    with pytest.raises(BudgetExceededError):
        bc.reserve("t2", 50)  # 60 + 50 > 100


def test_budget_controller_exceed_task_ceiling():
    bc = BudgetController(max_task_sats=50)
    with pytest.raises(BudgetExceededError):
        bc.reserve("t1", 51)


def test_budget_controller_negative_sats_raises():
    bc = BudgetController()
    with pytest.raises(ValueError):
        bc.reserve("t1", -10)


def test_budget_controller_concurrent_reservations():
    bc = BudgetController(max_budget_sats=10000)
    threads = []

    def worker_reserve(idx: int):
        bc.reserve(f"task_{idx}", 10)

    for i in range(50):
        t = threading.Thread(target=worker_reserve, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    assert bc.total_reserved_sats == 500


# =============================================================================
# 5. Preimage & Payment Hash Utilities Tests
# =============================================================================

def test_generate_escrow_secret():
    preimage, p_hash = generate_escrow_secret()
    assert len(preimage) == 64
    assert len(p_hash) == 64
    # Re-hash preimage to verify SHA-256 binding
    computed = hashlib.sha256(bytes.fromhex(preimage)).hexdigest()
    assert computed == p_hash


def test_validate_preimage():
    preimage, p_hash = generate_escrow_secret()
    assert validate_preimage(preimage, expected_payment_hash=p_hash) is True
    # Wrong hash
    assert validate_preimage(preimage, expected_payment_hash="00" * 32) is False
    # Invalid length
    assert validate_preimage(preimage[:30]) is False
    # Non-hex string
    assert validate_preimage("g" * 64) is False


def test_escrow_state_preimage_masking_repr_and_dict():
    preimage, p_hash = generate_escrow_secret()
    escrow = EscrowState(
        task_id="task_mask",
        payment_hash=p_hash,
        total_compute_sats=100,
        _preimage=preimage,
    )
    # When INITIALIZED or HELD, __repr__ must mask the secret
    assert "[CONFIDENTIAL]" in repr(escrow)
    assert preimage not in repr(escrow)

    # Default to_dict() does not leak preimage
    d = escrow.to_dict()
    assert "preimage" not in d

    # Explicit include_preimage=True includes it
    d_secret = escrow.to_dict(include_preimage=True)
    assert d_secret.get("preimage") == preimage

    # When SETTLED, __repr__ reveals the secret preimage
    escrow.status = EscrowStatus.SETTLED
    assert preimage in repr(escrow)


# =============================================================================
# 6. EscrowManager State Machine Transitions Tests
# =============================================================================

def test_escrow_happy_path(escrow_mgr):
    # 1. Create escrow -> INITIALIZED
    escrow = escrow_mgr.create_escrow("t1", total_sats=200, worker_lightning_addr="worker@test")
    assert escrow.status == EscrowStatus.INITIALIZED
    assert escrow.is_active is True
    assert escrow.is_held is False

    # 2. Hold escrow -> HELD
    held = escrow_mgr.hold_escrow("t1")
    assert held.status == EscrowStatus.HELD
    assert held.is_held is True
    assert held.held_at is not None

    # Pre-allocation auth header presents macaroon:payment_hash (withholding preimage!)
    cred = escrow_mgr.get_auth_credential("t1")
    assert cred.startswith("L402 ")
    assert f":{held.payment_hash}" in cred
    assert held.preimage not in cred

    # 3. Settle payout -> SETTLED
    settled = escrow_mgr.release_payout("t1")
    assert settled.status == EscrowStatus.SETTLED
    assert settled.is_terminal is True
    assert settled.settled_at is not None


def test_escrow_fraud_abort_path(escrow_mgr):
    escrow = escrow_mgr.create_and_hold("t_fraud", total_sats=150)
    assert escrow.status == EscrowStatus.HELD

    # Abort with fraud proof
    mock_proof = {"reason": "discrepancy exceeds bound"}
    cancelled = escrow_mgr.abort_and_refund("t_fraud", reason="fraud", fraud_proof=mock_proof)
    assert cancelled.status == EscrowStatus.CANCELLED
    assert cancelled.is_terminal is True
    assert cancelled.cancelled_at is not None
    assert cancelled.fraud_proof == mock_proof


def test_escrow_illegal_transition_init_to_settled(escrow_mgr):
    escrow_mgr.create_escrow("t_bad", total_sats=100)
    with pytest.raises(EscrowStateError) as exc_info:
        escrow_mgr.release_payout("t_bad")
    assert "must be HELD" in str(exc_info.value)


def test_escrow_illegal_transition_re_settle(escrow_mgr):
    escrow_mgr.create_and_hold("t_resettle", total_sats=100)
    escrow_mgr.release_payout("t_resettle")
    with pytest.raises(EscrowStateError) as exc_info:
        escrow_mgr.release_payout("t_resettle")
    assert "terminal status 'SETTLED'" in str(exc_info.value)


def test_escrow_illegal_transition_settle_cancelled(escrow_mgr):
    escrow_mgr.create_and_hold("t_canc", total_sats=100)
    escrow_mgr.abort_and_refund("t_canc")
    with pytest.raises(EscrowStateError) as exc_info:
        escrow_mgr.release_payout("t_canc")
    assert "terminal status 'CANCELLED'" in str(exc_info.value)


def test_escrow_re_hold_raises(escrow_mgr):
    escrow_mgr.create_and_hold("t_rehold", total_sats=100)
    with pytest.raises(EscrowStateError) as exc_info:
        escrow_mgr.hold_escrow("t_rehold")
    assert "already in HELD status" in str(exc_info.value)


def test_escrow_duplicate_task_id_raises(escrow_mgr):
    escrow_mgr.create_escrow("t_dup", total_sats=100)
    with pytest.raises(EscrowCreationError):
        escrow_mgr.create_escrow("t_dup", total_sats=100)


# =============================================================================
# 7. Escrow Timeout & Expiration Tests
# =============================================================================

def test_escrow_check_timeouts_sweep(escrow_mgr):
    now = time.time()
    # Escrow with 10s deadline
    e1 = escrow_mgr.create_and_hold("t_exp1", total_sats=100, timeout_seconds=10.0)
    # Escrow with 300s deadline
    e2 = escrow_mgr.create_and_hold("t_live", total_sats=100, timeout_seconds=300.0)

    # Advance time by 15s: e1 should expire, e2 should remain HELD
    expired = escrow_mgr.check_timeouts(now=now + 15.0)
    assert len(expired) == 1
    assert expired[0].task_id == "t_exp1"
    assert expired[0].status == EscrowStatus.EXPIRED

    assert e2.status == EscrowStatus.HELD


def test_escrow_lazy_expiration_on_settle(escrow_mgr):
    e = escrow_mgr.create_and_hold("t_lazy", total_sats=100, timeout_seconds=5.0)
    # Simulate past deadline
    e.expires_at = time.time() - 10.0
    with pytest.raises(EscrowTimeoutError):
        escrow_mgr.release_payout("t_lazy")
    assert e.status == EscrowStatus.CANCELLED


# =============================================================================
# 8. Freivalds Verification & Fraud Proof Integration Tests
# =============================================================================

def test_verify_and_settle_accurate_matrix_passes(escrow_mgr):
    # Setup matrices
    rng = np.random.default_rng(42)
    A = rng.standard_normal((16, 16)).astype(np.float32)
    B = rng.standard_normal((16, 16)).astype(np.float32)
    C = np.matmul(A, B)

    escrow_mgr.create_and_hold("task_gemm_pass", total_sats=200)

    is_valid, escrow, fraud_proof = escrow_mgr.verify_and_settle(
        task_id="task_gemm_pass",
        A=A,
        B=B,
        C=C,
        dtype="fp32",
    )
    assert is_valid is True
    assert escrow.status == EscrowStatus.SETTLED
    assert fraud_proof is None


def test_verify_and_settle_corrupted_matrix_generates_fraud_proof(escrow_mgr):
    rng = np.random.default_rng(42)
    A = rng.standard_normal((16, 16)).astype(np.float32)
    B = rng.standard_normal((16, 16)).astype(np.float32)
    C_corrupted = np.matmul(A, B)
    C_corrupted[0, 0] += 50.0  # Inject fraud

    worker_signer = ClientSigner()
    consumer_signer = ClientSigner()

    task_id = "task_gemm_fail"
    escrow_mgr.create_and_hold(task_id, total_sats=200)

    # Worker signs corrupted result
    c_hash = hashlib.sha256(C_corrupted.tobytes()).hexdigest()
    worker_sig = worker_signer.sign_message(f"{task_id}:{c_hash}".encode("utf-8"))

    is_valid, escrow, fraud_proof = escrow_mgr.verify_and_settle(
        task_id=task_id,
        A=A,
        B=B,
        C=C_corrupted,
        worker_pubkey=worker_signer.get_pubkey_hex(),
        worker_sig=worker_sig,
        consumer_privkey=consumer_signer.get_privkey_hex(),
        dtype="fp32",
    )
    assert is_valid is False
    assert escrow.status == EscrowStatus.CANCELLED
    assert fraud_proof is not None
    assert fraud_proof.task_id == task_id
    assert fraud_proof.observed_diff > 1.0


# =============================================================================
# 9. L402 Payment Coordinator End-to-End Tests
# =============================================================================

def test_payment_coordinator_full_lifecycle(mock_settlement):
    config = ClientConfig(max_price_per_gflop=0.001, max_budget_sats=5000)
    coord = L402PaymentCoordinator(config=config, settlement_backend=mock_settlement)

    # 1. Create escrow
    escrow = coord.create_escrow("coord_t1", total_sats=100, worker_lightning_addr="worker@ln")
    assert escrow.status == EscrowStatus.HELD

    # 2. Get gRPC metadata
    metadata = coord.get_grpc_metadata(escrow)
    assert len(metadata) == 2
    assert metadata[0][0] == "authorization"
    assert metadata[0][1].startswith("L402 ")
    assert metadata[1] == ("x-task-id", "coord_t1")

    # 3. Release payout
    assert coord.release_payout(escrow) is True
    assert escrow.status == EscrowStatus.SETTLED


def test_payment_coordinator_abort_and_refund(mock_settlement):
    config = ClientConfig(max_price_per_gflop=0.001, max_budget_sats=5000)
    coord = L402PaymentCoordinator(config=config, settlement_backend=mock_settlement)

    escrow = coord.create_escrow("coord_t2", total_sats=100)
    assert escrow.status == EscrowStatus.HELD

    assert coord.abort_and_refund(escrow, reason="worker_dropped") is True
    assert escrow.status == EscrowStatus.CANCELLED


# =============================================================================
# 10. Batch Escrow & Multi-Task Macaroon Unit Tests
# =============================================================================

def test_create_batch_escrow_multi_task_indexing(mock_settlement):
    manager = EscrowManager(settlement_backend=mock_settlement)
    task_ids = ["batch_t1", "batch_t2", "batch_t3"]
    job_id = "job_test_batch_123"

    escrow = manager.create_batch_escrow(
        worker_pubkey="worker_pubkey_abc",
        task_ids=task_ids,
        job_id=job_id,
        total_sats=300,
        worker_lightning_addr="worker@node",
        max_gflops=50.0,
    )

    assert escrow.status == EscrowStatus.INITIALIZED
    assert escrow.is_batch is True
    assert escrow.task_ids == task_ids
    assert escrow.job_id == job_id
    assert escrow.worker_pubkey == "worker_pubkey_abc"
    assert escrow.total_compute_sats == 300

    # Verify all task_ids resolve to the same batch escrow
    for tid in task_ids:
        resolved = manager.get_escrow(tid)
        assert resolved is not None
        assert resolved.payment_hash == escrow.payment_hash

    # Verify resolution by job_id and payment_hash
    assert manager._resolve_escrow(job_id).payment_hash == escrow.payment_hash
    assert manager.get_by_payment_hash(escrow.payment_hash).job_id == job_id


def test_macaroon_batch_caveat_containment_validation(verification_key):
    # Mint token with task_ids list
    task_ids = ["tile_0_0", "tile_0_1", "tile_1_0"]
    caveats = {
        "expiry": int(time.time() + 3600),
        "locked_sats": 300,
        "max_gflops": 100.0,
        "payment_hash": "a" * 64,
        "task_id": "tile_0_0",
        "task_ids": task_ids,
        "job_id": "job_xyz",
    }
    caveat_str = json.dumps(caveats, sort_keys=True)
    sig = hmac.new(verification_key.encode("utf-8"), caveat_str.encode("utf-8"), hashlib.sha256).hexdigest()
    raw_b64 = base64.b64encode(json.dumps({"caveats": caveats, "signature": sig}).encode("utf-8")).decode("utf-8")

    token = MacaroonToken.deserialize(raw_b64)
    assert token.caveats.task_ids == task_ids
    assert token.caveats.job_id == "job_xyz"

    # All constituent tasks should validate
    for tid in task_ids:
        is_valid, err = token.validate(task_id=tid, m=32, k=32, n=32)
        assert is_valid is True, f"Failed for {tid}: {err}"

    # Non-constituent task must fail
    is_valid, err = token.validate(task_id="outsider_tile", m=32, k=32, n=32)
    assert is_valid is False
    assert "not authorized in batch" in err or "mismatch" in err


def test_settle_batch_releases_preimage(mock_settlement):
    manager = EscrowManager(settlement_backend=mock_settlement)
    task_ids = ["settle_t1", "settle_t2"]
    job_id = "job_settle_batch_456"

    escrow = manager.create_batch_escrow(
        worker_pubkey="worker_pubkey_xyz",
        task_ids=task_ids,
        job_id=job_id,
        total_sats=200,
    )
    manager.hold_escrow(escrow.payment_hash)
    assert escrow.status == EscrowStatus.HELD

    # Settle via job_id
    settled = manager.settle_batch(job_id)
    assert settled.status == EscrowStatus.SETTLED
    assert settled.settled_at is not None
    assert mock_settlement.get_status(escrow.payment_hash) == "SETTLED"


def test_cancel_batch_on_failure(mock_settlement):
    manager = EscrowManager(settlement_backend=mock_settlement)
    task_ids = ["cancel_t1", "cancel_t2"]
    job_id = "job_cancel_batch_789"

    escrow = manager.create_batch_escrow(
        worker_pubkey="worker_pubkey_xyz",
        task_ids=task_ids,
        job_id=job_id,
        total_sats=200,
    )
    manager.hold_escrow(escrow.payment_hash)
    assert escrow.status == EscrowStatus.HELD

    cancelled = manager.cancel_batch(job_id, reason="freivalds_fraud_detected")
    assert cancelled.status == EscrowStatus.CANCELLED
    assert cancelled.cancellation_reason == "freivalds_fraud_detected"
    assert mock_settlement.get_status(escrow.payment_hash) == "CANCELLED"

