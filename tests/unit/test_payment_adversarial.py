"""
Adversarial Stress Test Suite for L402 Payment Client, Budget Controller & Escrow State Machine.
Targeting: Milestone 3 (M3) in matrix-compute-sdk.

Authored by teamwork_preview_challenger_sdk_m3_2 (L402 Payment & Escrow Challenger).

Adversarial Challenge Dimensions:
1. L402 Macaroon Tampering & Cryptographic Forgery:
   - Exhaustive bit-flip attacks on HMAC-SHA256 signatures across all character positions.
   - Truncated, padded, malformed, non-hex, and empty HMAC signatures.
   - Unauthorized caveat mutations (tampered task_id, expiry, max_gflops, locked_sats, payment_hash).
   - Injected unknown caveats and deleted required caveats.
   - Base64/JSON deserialization fuzzing (invalid b64, non-JSON bytes, array/primitive payloads, missing keys).
   - Boundary tests on caveat validation (exact expiry, clock skew tolerance, GFLOP limits, satoshi underpayment).

2. WWW-Authenticate Challenge Header Forgery & Fuzzing:
   - Malformed schemes (Bearer, Basic, Digest, OAuth, Token).
   - Header injection, special characters, whitespace variations, and missing parameters.
   - Quoted vs unquoted attributes, empty values, invalid price_sats.
   - Missing required keys (macaroon, invoice) and malformed headers.

3. Thread-Safe Budget Controller Concurrency Races:
   - 32 concurrent threads racing to exhaust the last remaining satoshis in a budget pool.
   - Assertion of exact cumulative accounting (reserved + committed == total, zero double-spending).
   - High-concurrency simultaneous commit and refund/release cycles.
   - Boundary checks: exact max budget, max budget + 1, task ceiling, negative sats rejection.

4. Trustless Escrow State Machine Concurrency & Race Invariants:
   - Simultaneous settle vs cancel race on the identical tile escrow (timeout vs result delivery).
   - Invariant: Escrow can NEVER be both SETTLED and CANCELLED; exactly one succeeds, the loser raises EscrowStateError.
   - 32-thread settle/cancel stampede.
   - Concurrent race to create escrows with identical task_id (exactly one succeeds).
   - Complete matrix of illegal state transitions (INITIALIZED -> SETTLED, SETTLED -> CANCELLED, CANCELLED -> SETTLED, etc.).
   - Concurrent timeout sweeps across threads.

5. Secret Preimage Secrecy & Redaction Guarantees:
   - Preimage strictly masked as [CONFIDENTIAL] in __repr__ and omitted in to_dict() prior to SETTLED.
   - get_auth_header() strictly withholds preimage even when use_preimage=True is passed on un-settled escrows.
   - Preimage never leaked in exception traces or logs.
   - Preimage revealed only upon successful settlement, cryptographically binding to payment_hash.

6. Freivalds Verification & Escrow Fraud / Settle Integration:
   - verify_and_settle with genuine matrix multiplication: settles invoice, reveals preimage, credits worker.
   - verify_and_settle with mathematical fraud (single-element perturbation, row bias, sign flips, zero-matrix returns):
     withholds preimage, cancels invoice, refunds consumer, generates authoritative signed FreivaldsFraudProof.
   - verify_and_settle with forged worker signatures raises WorkerSignatureError.
   - verify_and_settle with NaN/Inf values catches fraud and cancels escrow.
   - verify_and_settle on unheld or terminal escrows enforces state invariants.

7. L402 Payment Coordinator End-to-End Adversarial Scenarios:
   - Coordinator budget reservation rollback on settlement backend invoice creation failure.
   - Concurrent coordinator workflows with zero budget leakage.
"""

from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import hmac
import json
import logging
import math
import secrets
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest

from matrix_compute.config import ClientConfig
from matrix_compute.crypto.keys import generate_keypair
from matrix_compute.crypto.signer import ClientSigner, format_tile_result_payload, sign_message, verify_signature
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
    WorkerSignatureError,
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
from matrix_compute.verification.fraud_proof import FraudProofBuilder
from matrix_compute.verification.freivalds import FreivaldsVerifier


# =============================================================================
# Helper Utilities
# =============================================================================

def build_signed_macaroon_raw(
    task_id: str,
    payment_hash: str,
    expiry: int,
    max_gflops: float,
    locked_sats: int,
    verification_key: str,
    extra_caveats: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, Any], str]:
    """
    Constructs a signed Macaroon base64 token and returns (raw_b64, caveats_dict, signature_hex).
    """
    caveats: Dict[str, Any] = {
        "expiry": int(expiry),
        "locked_sats": int(locked_sats),
        "max_gflops": float(max_gflops),
        "payment_hash": payment_hash,
        "task_id": task_id,
    }
    if extra_caveats:
        caveats.update(extra_caveats)

    caveat_str = json.dumps(caveats, sort_keys=True)
    sig = hmac.new(
        verification_key.encode("utf-8"),
        caveat_str.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    payload = {"caveats": caveats, "signature": sig}
    b64 = base64.b64encode(json.dumps(payload).encode("utf-8")).decode("utf-8")
    return b64, caveats, sig


# =============================================================================
# Suite 1: L402 Macaroon Parsing, Bit-Flip Attacks & HMAC Signature Tampering
# =============================================================================

class TestMacaroonAdversarialTampering:
    """
    Tests adversarial attacks against Macaroon base64 tokens, including bit-flipping
    HMAC signatures, mutating caveats, and fuzzing deserialization.
    """

    @pytest.fixture
    def v_key(self) -> str:
        return "settlement-super-secret-key-999"

    def test_macaroon_hmac_exhaustive_bit_flips(self, v_key: str):
        """
        Adversarial Test: Flips single bits/nibbles across all 64 characters of HMAC signature.
        Asserts that verify_hmac fails for 100% of tampered signatures.
        """
        raw_b64, caveats, valid_sig = build_signed_macaroon_raw(
            task_id="task_tamper_1",
            payment_hash="e" * 64,
            expiry=int(time.time() + 3600),
            max_gflops=10.0,
            locked_sats=100,
            verification_key=v_key,
        )

        token = MacaroonToken.deserialize(raw_b64)
        assert token.verify_hmac(v_key) is True

        # Systematically tamper each character position in the 64-char hex HMAC
        hex_chars = "0123456789abcdef"
        for i in range(len(valid_sig)):
            original_char = valid_sig[i]
            # Pick a different character
            flipped_char = "0" if original_char != "0" else "1"
            tampered_sig = valid_sig[:i] + flipped_char + valid_sig[i + 1:]

            # Build tampered payload
            tampered_payload = {"caveats": caveats, "signature": tampered_sig}
            tampered_b64 = base64.b64encode(json.dumps(tampered_payload).encode("utf-8")).decode("utf-8")

            tampered_token = MacaroonToken.deserialize(tampered_b64)
            assert tampered_token.verify_hmac(v_key) is False, (
                f"HMAC verification succeeded on tampered signature at index {i}: {tampered_sig}"
            )

    def test_macaroon_hmac_length_and_format_tampering(self, v_key: str):
        """
        Adversarial Test: Tests truncated signatures, extended signatures,
        empty signatures, and non-hex signatures.
        """
        _, caveats, valid_sig = build_signed_macaroon_raw(
            task_id="task_tamper_len",
            payment_hash="f" * 64,
            expiry=int(time.time() + 3600),
            max_gflops=10.0,
            locked_sats=100,
            verification_key=v_key,
        )

        bad_signatures = [
            "",                        # Empty
            valid_sig[:32],            # Truncated half
            valid_sig[:63],            # Truncated 1 char
            valid_sig + "0",           # Extended 1 char
            valid_sig + valid_sig,     # Doubled length
            "z" * 64,                  # Non-hex characters
            "00" * 32,                 # All zero hex
            "!@#$%^&*()" * 6,          # Special characters
        ]

        for bad_sig in bad_signatures:
            payload = {"caveats": caveats, "signature": bad_sig}
            b64 = base64.b64encode(json.dumps(payload).encode("utf-8")).decode("utf-8")
            token = MacaroonToken.deserialize(b64)
            assert token.verify_hmac(v_key) is False, f"Verify passed for invalid signature: {bad_sig!r}"

    def test_macaroon_verification_key_edge_cases(self, v_key: str):
        """
        Tests HMAC verification behavior with empty, wrong, or mismatched verification keys.
        """
        raw_b64, _, _ = build_signed_macaroon_raw(
            task_id="task_vk",
            payment_hash="1" * 64,
            expiry=int(time.time() + 3600),
            max_gflops=10.0,
            locked_sats=100,
            verification_key=v_key,
        )
        token = MacaroonToken.deserialize(raw_b64)

        assert token.verify_hmac("") is False
        assert token.verify_hmac("wrong-key") is False
        assert token.verify_hmac(v_key + "extra") is False
        assert token.verify_hmac(v_key[:-1]) is False
        assert token.verify_hmac(v_key.upper() if v_key != v_key.upper() else v_key + "X") is False

    def test_macaroon_caveats_mutation_breaks_hmac(self, v_key: str):
        """
        Adversarial Test: Tests that mutating any caveat field breaks HMAC verification,
        preventing unauthorized privilege escalation.
        """
        _, original_caveats, sig = build_signed_macaroon_raw(
            task_id="task_orig",
            payment_hash="2" * 64,
            expiry=int(time.time() + 3600),
            max_gflops=10.0,
            locked_sats=100,
            verification_key=v_key,
        )

        # Mutate each field independently while retaining original signature
        mutations = [
            {"task_id": "task_forged"},
            {"expiry": original_caveats["expiry"] + 10000},
            {"max_gflops": 1000.0},
            {"locked_sats": 5000},
            {"payment_hash": "3" * 64},
            {"extra_unauthorized_field": "admin=true"},
        ]

        for mut in mutations:
            tampered_caveats = dict(original_caveats)
            tampered_caveats.update(mut)
            payload = {"caveats": tampered_caveats, "signature": sig}
            b64 = base64.b64encode(json.dumps(payload).encode("utf-8")).decode("utf-8")

            token = MacaroonToken.deserialize(b64)
            assert token.verify_hmac(v_key) is False, f"HMAC verified despite caveat mutation: {mut}"

    def test_macaroon_deserialization_fuzzing(self):
        """
        Adversarial Test: Fuzzes deserialization with malformed payloads, non-JSON bytes,
        missing schema fields, and corrupted base64.
        """
        fuzz_cases = [
            "",                                  # Empty string
            "   ",                               # Whitespace only
            "not-base-64!!!???",                 # Invalid base64 characters
            base64.b64encode(b"not json data").decode(),  # Raw non-JSON bytes
            base64.b64encode(b"12345").decode(),          # JSON integer
            base64.b64encode(b'"just a string"').decode(),# JSON string
            base64.b64encode(b"[1, 2, 3]").decode(),      # JSON array instead of dict
            base64.b64encode(json.dumps({"caveats": {}}).encode()).decode(),  # Missing signature
            base64.b64encode(json.dumps({"signature": "sig"}).encode()).decode(), # Missing caveats
            base64.b64encode(json.dumps({"caveats": "not-a-dict", "signature": "sig"}).encode()).decode(), # Caveats not dict
        ]

        for bad_input in fuzz_cases:
            with pytest.raises(MacaroonError):
                MacaroonToken.deserialize(bad_input)

    def test_macaroon_caveat_validation_boundary_conditions(self, v_key: str):
        """
        Adversarial Test: Tests exact numerical and temporal boundaries in caveat validation.
        """
        now = 1700000000.0  # Fixed timestamp for determinism
        skew = 5.0

        # Boundary 1: Expiration exact bounds
        # Expiry at (now - skew) -> must reject
        b64_exp, _, _ = build_signed_macaroon_raw(
            task_id="t_bound",
            payment_hash="4" * 64,
            expiry=int(now - skew),
            max_gflops=50.0,
            locked_sats=100,
            verification_key=v_key,
        )
        t_exp = MacaroonToken.deserialize(b64_exp)
        valid, msg = t_exp.validate("t_bound", 512, 512, 512, current_time=now, clock_skew_tolerance=skew)
        assert valid is False
        assert "Macaroon expired" in str(msg)

        # Expiry at (now - skew + 1) -> must pass
        b64_live, _, _ = build_signed_macaroon_raw(
            task_id="t_bound",
            payment_hash="4" * 64,
            expiry=int(now - skew + 1),
            max_gflops=50.0,
            locked_sats=100,
            verification_key=v_key,
        )
        t_live = MacaroonToken.deserialize(b64_live)
        valid, msg = t_live.validate("t_bound", 512, 512, 512, current_time=now, clock_skew_tolerance=skew)
        assert valid is True

        # Boundary 2: GFLOPs exact bounds
        # Requested GFLOPs: 2 * 1000 * 1000 * 1000 / 1e9 = 2.0 GFLOPs
        # Allowance: 1.999999 -> must reject
        b64_gf_fail, _, _ = build_signed_macaroon_raw(
            task_id="t_bound",
            payment_hash="4" * 64,
            expiry=int(now + 3600),
            max_gflops=1.999999,
            locked_sats=100,
            verification_key=v_key,
        )
        t_gf_fail = MacaroonToken.deserialize(b64_gf_fail)
        valid, msg = t_gf_fail.validate("t_bound", 1000, 1000, 1000, current_time=now)
        assert valid is False
        assert "Insufficient GFLOP allowance" in str(msg)

        # Allowance: 2.0 -> must pass
        b64_gf_pass, _, _ = build_signed_macaroon_raw(
            task_id="t_bound",
            payment_hash="4" * 64,
            expiry=int(now + 3600),
            max_gflops=2.0,
            locked_sats=100,
            verification_key=v_key,
        )
        t_gf_pass = MacaroonToken.deserialize(b64_gf_pass)
        valid, msg = t_gf_pass.validate("t_bound", 1000, 1000, 1000, current_time=now)
        assert valid is True

        # Boundary 3: Underpayment bounds with min_task_sats floor
        # Rate: 0.001 sats/GFLOP. Requested GFLOPs: 2.0. Raw sats: ceil(0.002) = 1.
        # Min floor: 10 sats.
        # Locked: 9 sats -> must reject
        b64_sats_fail, _, _ = build_signed_macaroon_raw(
            task_id="t_bound",
            payment_hash="4" * 64,
            expiry=int(now + 3600),
            max_gflops=5.0,
            locked_sats=9,
            verification_key=v_key,
        )
        t_sats_fail = MacaroonToken.deserialize(b64_sats_fail)
        valid, msg = t_sats_fail.validate(
            "t_bound", 1000, 1000, 1000, current_time=now, max_price_rate=0.001, min_task_sats=10
        )
        assert valid is False
        assert "Insufficient locked satoshis" in str(msg)

        # Locked: 10 sats -> must pass
        b64_sats_pass, _, _ = build_signed_macaroon_raw(
            task_id="t_bound",
            payment_hash="4" * 64,
            expiry=int(now + 3600),
            max_gflops=5.0,
            locked_sats=10,
            verification_key=v_key,
        )
        t_sats_pass = MacaroonToken.deserialize(b64_sats_pass)
        valid, msg = t_sats_pass.validate(
            "t_bound", 1000, 1000, 1000, current_time=now, max_price_rate=0.001, min_task_sats=10
        )
        assert valid is True


# =============================================================================
# Suite 2: WWW-Authenticate Challenge Header Forgery & Fuzzing
# =============================================================================

class TestL402ChallengeHeaderAdversarial:
    """
    Tests adversarial header injection, invalid schemes, and fuzzed parameter formats.
    """

    def test_challenge_invalid_schemes(self):
        """Adversarial Test: Rejects non-L402/LSAT authentication schemes."""
        invalid_schemes = [
            'Bearer token="abcdef12345"',
            'Basic realm="Secure Area"',
            'Digest username="admin"',
            'OAuth realm="example", oauth_token="xyz"',
            'Token token="custom_token"',
            'MUTATION scheme=attack',
        ]
        for header in invalid_schemes:
            with pytest.raises(L402ChallengeError) as exc_info:
                L402Challenge.parse(header)
            assert "scheme must be L402 or LSAT" in str(exc_info.value)

    def test_challenge_missing_required_params(self):
        """Adversarial Test: Rejects headers missing macaroon, invoice, or both."""
        incomplete_headers = [
            'L402 macaroon="tok_only"',
            'L402 invoice="inv_only"',
            'L402 price_sats=100, realm="compute"',
            'WWW-Authenticate: L402 foo=bar, baz=qux',
            'LSAT',
            'L402 ',
        ]
        for header in incomplete_headers:
            with pytest.raises(L402ChallengeError):
                L402Challenge.parse(header)

    def test_challenge_fuzzed_parameters_and_injection(self):
        """Adversarial Test: Fuzzes parameters with special characters and injection attempts."""
        garbage_headers = [
            'L402 !@#$%^&*()',
            'L402 macaroon="", invoice=""',  # Empty values
            'L402 macaroon="valid_tok", invoice=""',
            'L402 macaroon="", invoice="valid_inv"',
            'L402 ===,,,===',
            'L402 ;;;;;;;;',
        ]
        for header in garbage_headers:
            with pytest.raises(L402ChallengeError):
                L402Challenge.parse(header)

    def test_challenge_malformed_price_sats_graceful_handling(self):
        """
        Adversarial Test: Header contains non-integer price_sats.
        Should parse successfully and set price_sats=None without crashing.
        """
        header = 'L402 macaroon="tok", invoice="inv", price_sats="not-a-number"'
        ch = L402Challenge.parse(header)
        assert ch.macaroon == "tok"
        assert ch.invoice == "inv"
        assert ch.price_sats is None

    def test_challenge_header_case_insensitivity_and_whitespace(self):
        """
        Tests that scheme parsing handles mixed case, extra spaces, and optional prefix.
        """
        headers = [
            'www-authenticate: l402 macaroon="m1", invoice="i1"',
            'WWW-AUTHENTICATE:   LSAT   macaroon="m2", invoice="i2"',
            'lSaT   macaroon=m3, invoice=i3',
            'L402   macaroon="m4"  ,   invoice="i4"  ',
        ]
        for h in headers:
            ch = L402Challenge.parse(h)
            assert ch.scheme in ("L402", "LSAT")
            assert ch.macaroon is not None
            assert ch.invoice is not None


# =============================================================================
# Suite 3: Thread-Safe Budget Controller Concurrency Races
# =============================================================================

class TestBudgetControllerConcurrencyRaces:
    """
    Stress-tests BudgetController under concurrent multi-threaded contention.
    Verifies zero double-spending, atomic reservations, exact accounting, and clean refunds.
    """

    def test_budget_32_threads_exact_pool_exhaustion(self):
        """
        Adversarial Concurrency Test:
        32 concurrent threads race to reserve the last remaining satoshis in a 100-sats pool.
        Pool: max_budget_sats = 100.
        Each thread asks for 10 satoshis (320 satoshis demanded).
        Invariant: EXACTLY 10 threads must succeed; EXACTLY 22 threads must receive BudgetExceededError.
        Cumulative accounting must equal 100, zero double-spending.
        """
        pool_capacity = 100
        cost_per_task = 10
        num_threads = 32
        expected_winners = pool_capacity // cost_per_task  # 10
        expected_losers = num_threads - expected_winners   # 22

        bc = BudgetController(max_budget_sats=pool_capacity, max_task_sats=20)
        barrier = threading.Barrier(num_threads)

        winners: List[str] = []
        losers: List[str] = []
        lock = threading.Lock()

        def racer(thread_id: int):
            task_id = f"race_task_{thread_id}"
            barrier.wait()  # Synchronize all 32 threads to release at the exact same instant
            try:
                bc.reserve(task_id, cost_per_task)
                with lock:
                    winners.append(task_id)
            except BudgetExceededError:
                with lock:
                    losers.append(task_id)

        threads = [threading.Thread(target=racer, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Invariant Assertions
        assert len(winners) == expected_winners, f"Expected {expected_winners} winners, got {len(winners)}"
        assert len(losers) == expected_losers, f"Expected {expected_losers} losers, got {len(losers)}"
        assert bc.total_reserved_sats == pool_capacity, (
            f"Double spending or budget leak: total_reserved={bc.total_reserved_sats}, expected={pool_capacity}"
        )
        assert bc.total_committed_sats == 0
        summary = bc.get_summary()
        assert summary["active_tasks_count"] == expected_winners

    def test_budget_concurrent_commit_and_refund_cycles(self):
        """
        Adversarial Concurrency Test:
        Reserves 10 tasks (100 sats). 5 threads commit, 5 threads refund concurrently.
        Verifies exact final state: 0 reserved, 50 committed.
        Then tests that refunded budget (50 sats) can be immediately re-reserved by 5 new threads.
        """
        bc = BudgetController(max_budget_sats=100)
        for i in range(10):
            bc.reserve(f"task_{i}", 10)

        assert bc.total_reserved_sats == 100

        barrier = threading.Barrier(10)

        def commit_worker(idx: int):
            barrier.wait()
            bc.commit(f"task_{idx}")

        def release_worker(idx: int):
            barrier.wait()
            bc.release(f"task_{idx}")

        threads = []
        # 5 commit threads (0-4)
        for i in range(5):
            threads.append(threading.Thread(target=commit_worker, args=(i,)))
        # 5 release threads (5-9)
        for i in range(5, 10):
            threads.append(threading.Thread(target=release_worker, args=(i,)))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert bc.total_reserved_sats == 0
        assert bc.total_committed_sats == 50

        # Now 50 sats are available: 5 new threads should succeed in reserving 10 sats each
        for i in range(10, 15):
            bc.reserve(f"task_{i}", 10)

        assert bc.total_reserved_sats == 50
        assert bc.total_committed_sats == 50

        # Attempting to reserve 1 more sat must fail
        with pytest.raises(BudgetExceededError):
            bc.reserve("task_overflow", 1)

    def test_budget_controller_boundary_conditions(self):
        """
        Tests strict edge cases: exact budget limits, task ceilings, negative sats.
        """
        bc = BudgetController(max_budget_sats=1000, max_task_sats=250)

        # 1. Negative sats rejected
        with pytest.raises(ValueError):
            bc.reserve("neg_task", -1)

        # 2. Zero sats allowed
        bc.reserve("zero_task", 0)
        assert bc.total_reserved_sats == 0
        bc.release("zero_task")

        # 3. Exact task ceiling allowed, ceiling + 1 rejected
        bc.reserve("ceil_pass", 250)
        assert bc.total_reserved_sats == 250
        bc.release("ceil_pass")

        with pytest.raises(BudgetExceededError):
            bc.reserve("ceil_fail", 251)

        # 4. Exact cumulative budget allowed, budget + 1 rejected
        bc.reserve("t1", 250)
        bc.reserve("t2", 250)
        bc.reserve("t3", 250)
        bc.reserve("t4", 250)
        assert bc.total_reserved_sats == 1000

        with pytest.raises(BudgetExceededError):
            bc.reserve("overflow", 1)


# =============================================================================
# Suite 4: Trustless Escrow State Machine Concurrency & Race Invariants
# =============================================================================

class TestEscrowStateMachineConcurrency:
    """
    Stress-tests EscrowManager state transitions under high-contention concurrent races.
    Verifies mutual exclusion between settlement and cancellation, illegal transition defenses,
    and timeout sweep safety.
    """

    @pytest.fixture
    def mock_backend(self) -> MockSettlementService:
        return MockSettlementService()

    @pytest.fixture
    def escrow_mgr(self, mock_backend: MockSettlementService) -> EscrowManager:
        return EscrowManager(settlement_backend=mock_backend, default_timeout_seconds=60.0)

    def test_escrow_settle_vs_cancel_simultaneous_race_50_iterations(self, escrow_mgr: EscrowManager):
        """
        Adversarial Concurrency Race:
        Simultaneous settlement vs cancellation race on the IDENTICAL tile escrow
        (simulating worker result arrival racing against consumer task timeout).

        Invariant:
        - An escrow can NEVER be both SETTLED and CANCELLED.
        - Exactly ONE action succeeds.
        - The losing action MUST raise EscrowStateError.
        - Executed 50 consecutive times to verify absence of any race condition windows.
        """
        for iteration in range(50):
            task_id = f"race_escrow_{iteration}"
            escrow = escrow_mgr.create_and_hold(task_id, total_sats=100)
            assert escrow.status == EscrowStatus.HELD

            barrier = threading.Barrier(2)
            results: Dict[str, Any] = {}

            def settle_action():
                barrier.wait()
                try:
                    escrow_mgr.release_payout(task_id)
                    results["settle"] = "OK"
                except Exception as ex:
                    results["settle"] = ex

            def cancel_action():
                barrier.wait()
                try:
                    escrow_mgr.abort_and_refund(task_id, reason="timeout")
                    results["cancel"] = "OK"
                except Exception as ex:
                    results["cancel"] = ex

            t1 = threading.Thread(target=settle_action)
            t2 = threading.Thread(target=cancel_action)
            t1.start()
            t2.start()
            t1.join()
            t2.join()

            final_escrow = escrow_mgr.get_escrow(task_id)
            assert final_escrow is not None
            final_status = final_escrow.status

            # Mutual Exclusion Verification
            if results["settle"] == "OK":
                assert final_status == EscrowStatus.SETTLED
                assert isinstance(results["cancel"], EscrowStateError), (
                    f"Iteration {iteration}: cancel did not receive EscrowStateError: {results['cancel']}"
                )
            elif results["cancel"] == "OK":
                assert final_status == EscrowStatus.CANCELLED
                assert isinstance(results["settle"], EscrowStateError), (
                    f"Iteration {iteration}: settle did not receive EscrowStateError: {results['settle']}"
                )
            else:
                pytest.fail(f"Iteration {iteration}: Neither settle nor cancel succeeded: {results}")

    def test_escrow_32_thread_stampede_settle_cancel(self, escrow_mgr: EscrowManager):
        """
        Adversarial Concurrency Stampede:
        16 threads try to settle, 16 threads try to cancel (32 threads total)
        on the exact same escrow simultaneously.
        Invariant: Exactly 1 thread succeeds; 31 threads fail with EscrowStateError.
        """
        task_id = "stampede_escrow"
        escrow_mgr.create_and_hold(task_id, total_sats=500)

        num_threads = 32
        barrier = threading.Barrier(num_threads)
        successes: List[str] = []
        errors: List[Exception] = []
        lock = threading.Lock()

        def worker(thread_id: int):
            is_settler = (thread_id % 2 == 0)
            barrier.wait()
            try:
                if is_settler:
                    escrow_mgr.release_payout(task_id)
                    with lock:
                        successes.append("settle")
                else:
                    escrow_mgr.abort_and_refund(task_id, reason="stampede_cancel")
                    with lock:
                        successes.append("cancel")
            except Exception as ex:
                with lock:
                    errors.append(ex)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(successes) == 1, f"Expected exactly 1 winner, got {len(successes)}: {successes}"
        assert len(errors) == 31
        for err in errors:
            assert isinstance(err, EscrowStateError)

        escrow = escrow_mgr.get_escrow(task_id)
        assert escrow is not None
        assert escrow.is_terminal is True

    def test_escrow_duplicate_creation_concurrency_race(self, escrow_mgr: EscrowManager):
        """
        Adversarial Concurrency Race:
        16 threads attempt to create an escrow with the exact same task_id concurrently.
        Invariant: Exactly 1 thread succeeds; 15 threads raise EscrowCreationError.
        """
        num_threads = 16
        barrier = threading.Barrier(num_threads)
        created_count = 0
        error_count = 0
        lock = threading.Lock()

        def creator():
            nonlocal created_count, error_count
            barrier.wait()
            try:
                escrow_mgr.create_escrow("duplicate_race_task", total_sats=100)
                with lock:
                    created_count += 1
            except EscrowCreationError:
                with lock:
                    error_count += 1

        threads = [threading.Thread(target=creator) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert created_count == 1
        assert error_count == num_threads - 1

    def test_escrow_illegal_state_transitions_comprehensive_matrix(self, escrow_mgr: EscrowManager):
        """
        Adversarial State Machine Test:
        Tests all illegal state transitions across the escrow lifecycle.
        """
        # 1. INITIALIZED -> SETTLE (Must be HELD first)
        e1 = escrow_mgr.create_escrow("t_init_to_settle", total_sats=100)
        with pytest.raises(EscrowStateError) as exc:
            escrow_mgr.release_payout("t_init_to_settle")
        assert "must be HELD" in str(exc.value)

        # 2. Re-holding an already HELD escrow
        escrow_mgr.hold_escrow("t_init_to_settle")
        with pytest.raises(EscrowStateError) as exc:
            escrow_mgr.hold_escrow("t_init_to_settle")
        assert "already in HELD status" in str(exc.value)

        # 3. SETTLED -> CANCEL (Cannot cancel settled invoice)
        escrow_mgr.release_payout("t_init_to_settle")
        with pytest.raises(EscrowStateError) as exc:
            escrow_mgr.abort_and_refund("t_init_to_settle")
        assert "terminal status 'SETTLED'" in str(exc.value)

        # 4. SETTLED -> SETTLE (Cannot re-settle)
        with pytest.raises(EscrowStateError) as exc:
            escrow_mgr.release_payout("t_init_to_settle")
        assert "terminal status 'SETTLED'" in str(exc.value)

        # 5. SETTLED -> HOLD (Cannot hold after settle)
        with pytest.raises(EscrowStateError):
            escrow_mgr.hold_escrow("t_init_to_settle")

        # 6. CANCELLED -> SETTLE (Cannot settle cancelled invoice)
        e2 = escrow_mgr.create_and_hold("t_canc_to_settle", total_sats=100)
        escrow_mgr.abort_and_refund("t_canc_to_settle")
        with pytest.raises(EscrowStateError) as exc:
            escrow_mgr.release_payout("t_canc_to_settle")
        assert "terminal status 'CANCELLED'" in str(exc.value)

        # 7. CANCELLED -> CANCEL (Cannot re-cancel)
        with pytest.raises(EscrowStateError) as exc:
            escrow_mgr.abort_and_refund("t_canc_to_settle")
        assert "terminal status 'CANCELLED'" in str(exc.value)

        # 8. Non-existent task ID
        with pytest.raises(KeyError):
            escrow_mgr.release_payout("non_existent_task")
        with pytest.raises(KeyError):
            escrow_mgr.abort_and_refund("non_existent_task")

    def test_escrow_concurrent_timeout_sweeps(self, escrow_mgr: EscrowManager):
        """
        Adversarial Test: Multiple threads run check_timeouts() concurrently.
        Asserts exact expiration and zero race issues.
        """
        now = time.time()
        # Create 10 escrows with short timeout (expired) and 10 with long timeout (active)
        for i in range(10):
            escrow_mgr.create_and_hold(f"exp_{i}", total_sats=50, timeout_seconds=5.0)
        for i in range(10):
            escrow_mgr.create_and_hold(f"act_{i}", total_sats=50, timeout_seconds=300.0)

        # Simulate advance of time by 10s
        sweep_time = now + 10.0

        all_expired: List[EscrowState] = []
        lock = threading.Lock()

        def sweeper():
            exp = escrow_mgr.check_timeouts(now=sweep_time)
            with lock:
                all_expired.extend(exp)

        threads = [threading.Thread(target=sweeper) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Exactly 10 escrows should have expired across all threads
        assert len(all_expired) == 10
        expired_ids = {e.task_id for e in all_expired}
        assert expired_ids == {f"exp_{i}" for i in range(10)}

        # Long timeout escrows must remain HELD
        for i in range(10):
            e = escrow_mgr.get_escrow(f"act_{i}")
            assert e is not None
            assert e.status == EscrowStatus.HELD


# =============================================================================
# Suite 5: Secret Preimage Secrecy & Redaction Guarantees
# =============================================================================

class TestPreimageSecrecyGuarantees:
    """
    Verifies that under all logging, string representations, dictionary serialization,
    and credential generation, the secret 32-byte preimage is NEVER exposed prior to SETTLED.
    """

    @pytest.fixture
    def escrow_mgr(self) -> EscrowManager:
        return EscrowManager(settlement_backend=MockSettlementService())

    def test_preimage_never_exposed_prior_to_settled(self, escrow_mgr: EscrowManager):
        """
        Adversarial Test:
        Asserts that throughout INITIALIZED, HELD, CANCELLED, and EXPIRED statuses,
        the secret 32-byte preimage is strictly redacted.
        """
        custom_preimage, p_hash = generate_escrow_secret()

        # 1. INITIALIZED State
        escrow = escrow_mgr.create_escrow(
            task_id="sec_test_1",
            total_sats=100,
            custom_preimage=custom_preimage,
        )
        assert escrow.status == EscrowStatus.INITIALIZED

        # Preimage must NOT appear in repr, str, or to_dict()
        assert custom_preimage not in repr(escrow)
        assert "[CONFIDENTIAL]" in repr(escrow)
        assert custom_preimage not in str(escrow)
        assert "preimage" not in escrow.to_dict()
        assert "preimage" not in escrow.to_dict(include_preimage=False)
        assert custom_preimage not in json.dumps(escrow.to_dict())

        # Header generation must withhold preimage even if use_preimage=True
        assert custom_preimage not in escrow.get_auth_header(use_preimage=True)
        assert custom_preimage not in escrow.get_auth_header(use_preimage=False)
        assert p_hash in escrow.get_auth_header(use_preimage=True)

        # 2. HELD State
        escrow_mgr.hold_escrow("sec_test_1")
        assert escrow.status == EscrowStatus.HELD

        assert custom_preimage not in repr(escrow)
        assert "[CONFIDENTIAL]" in repr(escrow)
        assert "preimage" not in escrow.to_dict()
        assert custom_preimage not in escrow.get_auth_header(use_preimage=True)
        assert custom_preimage not in escrow_mgr.get_auth_credential("sec_test_1")

        # 3. CANCELLED State
        escrow_mgr.abort_and_refund("sec_test_1")
        assert escrow.status == EscrowStatus.CANCELLED

        assert custom_preimage not in repr(escrow)
        assert "[CONFIDENTIAL]" in repr(escrow)
        assert "preimage" not in escrow.to_dict()
        assert custom_preimage not in escrow.get_auth_header(use_preimage=True)

    def test_preimage_revealed_only_upon_settlement(self, escrow_mgr: EscrowManager):
        """
        Tests that upon SETTLED, the preimage is released and accurately binds to payment_hash.
        """
        preimage, p_hash = generate_escrow_secret()
        escrow = escrow_mgr.create_and_hold("sec_settle", total_sats=100, custom_preimage=preimage)

        # Before settlement: hidden
        assert preimage not in repr(escrow)

        # Settle invoice
        settled = escrow_mgr.release_payout("sec_settle")
        assert settled.status == EscrowStatus.SETTLED

        # After settlement: revealed in repr and auth header
        assert preimage in repr(settled)
        auth_hdr = settled.get_auth_header(use_preimage=True)
        assert auth_hdr.endswith(f":{preimage}")

        # Verify cryptographic binding
        computed_hash = hashlib.sha256(bytes.fromhex(preimage)).hexdigest()
        assert computed_hash == settled.payment_hash

    def test_preimage_secrecy_in_exceptions(self, escrow_mgr: EscrowManager):
        """
        Adversarial Test: Induces exceptions across operations and verifies that error strings
        never contain the secret preimage.
        """
        preimage, p_hash = generate_escrow_secret()
        escrow_mgr.create_and_hold("t_exc", total_sats=100, custom_preimage=preimage)

        # Induce EscrowStateError
        try:
            escrow_mgr.hold_escrow("t_exc")
        except EscrowStateError as ex:
            assert preimage not in str(ex)
            assert preimage not in repr(ex)

        # Induce EscrowTimeoutError
        escrow = escrow_mgr.get_escrow("t_exc")
        assert escrow is not None
        escrow.expires_at = time.time() - 100.0
        try:
            escrow_mgr.release_payout("t_exc")
        except EscrowTimeoutError as ex:
            assert preimage not in str(ex)
            assert preimage not in repr(ex)


# =============================================================================
# Suite 6: Freivalds Verification & Escrow Fraud / Settle Integration
# =============================================================================

class TestFreivaldsEscrowSettlementIntegration:
    """
    Stress-tests verify_and_settle integration with Freivalds mathematical verification,
    fraud proof construction, and escrow finalization.
    """

    @pytest.fixture
    def escrow_mgr(self) -> EscrowManager:
        return EscrowManager(settlement_backend=MockSettlementService())

    @pytest.fixture
    def signers(self) -> Dict[str, ClientSigner]:
        return {"worker": ClientSigner(), "consumer": ClientSigner()}

    def test_verify_and_settle_honest_gemm_matrix_spectrum(self, escrow_mgr: EscrowManager):
        """
        Tests verify_and_settle with valid matrix multiplications across multiple precision formats.
        Verifies: settles invoice, reveals preimage, returns None fraud proof.
        """
        rng = np.random.default_rng(123)

        for dtype in ("fp32", "fp16", "bf16"):
            task_id = f"honest_gemm_{dtype}"
            escrow = escrow_mgr.create_and_hold(task_id, total_sats=200)

            # Generate honest matrices
            np_dtype = np.float32 if dtype in ("fp32", "bf16") else np.float16
            A = rng.standard_normal((16, 16)).astype(np_dtype)
            B = rng.standard_normal((16, 16)).astype(np_dtype)
            if dtype == "bf16":
                # BF16 mantissa masking emulation
                a_f32 = A.astype(np.float32)
                b_f32 = B.astype(np.float32)
                c_f32 = np.matmul(a_f32, b_f32)
                u32_view = c_f32.view(np.uint32)
                u32_masked = u32_view & 0xFFFF0000
                C = u32_masked.view(np.float32)
            else:
                C = np.matmul(A, B)

            is_valid, settled_escrow, fraud_proof = escrow_mgr.verify_and_settle(
                task_id=task_id,
                A=A,
                B=B,
                C=C,
                dtype=dtype,
            )

            assert is_valid is True
            assert settled_escrow.status == EscrowStatus.SETTLED
            assert fraud_proof is None
            assert settled_escrow.preimage is not None
            assert settled_escrow.get_auth_header(use_preimage=True).endswith(f":{settled_escrow.preimage}")

    def test_verify_and_settle_tampered_gemm_spectrum(
        self, escrow_mgr: EscrowManager, signers: Dict[str, ClientSigner]
    ):
        """
        Adversarial Test: Injects multiple forms of computational fraud into result matrix C:
          1. Single-element perturbation (+1.0)
          2. Row-wide bias (+5.0)
          3. Sign flip (C = -A x B)
          4. Zero-matrix return (C = 0)
          5. Transposed product (C = (A x B)^T)
        Verifies:
          - is_valid == False
          - Escrow is CANCELLED (not settled)
          - Preimage is withheld
          - Signed FreivaldsFraudProof is constructed and valid
        """
        rng = np.random.default_rng(456)
        worker = signers["worker"]
        consumer = signers["consumer"]

        attack_types = ["single_element", "row_bias", "sign_flip", "zero_matrix", "transpose"]

        for attack in attack_types:
            task_id = f"fraud_attack_{attack}"
            escrow = escrow_mgr.create_and_hold(task_id, total_sats=300)
            preimage_secret = escrow.preimage

            A = rng.standard_normal((16, 16)).astype(np.float32)
            B = rng.standard_normal((16, 16)).astype(np.float32)
            honest_C = np.matmul(A, B)

            if attack == "single_element":
                C_fraud = honest_C.copy()
                C_fraud[0, 0] += 1.0
            elif attack == "row_bias":
                C_fraud = honest_C.copy()
                C_fraud[0, :] += 5.0
            elif attack == "sign_flip":
                C_fraud = -honest_C.copy()
            elif attack == "zero_matrix":
                C_fraud = np.zeros_like(honest_C)
            elif attack == "transpose":
                C_fraud = honest_C.T.copy()
            else:
                raise ValueError(attack)

            # Worker signs corrupted output
            c_hash = hashlib.sha256(C_fraud.tobytes()).hexdigest()
            worker_sig = worker.sign_message(f"{task_id}:{c_hash}".encode("utf-8"))

            is_valid, cancelled_escrow, fraud_proof = escrow_mgr.verify_and_settle(
                task_id=task_id,
                A=A,
                B=B,
                C=C_fraud,
                worker_pubkey=worker.get_pubkey_hex(),
                worker_sig=worker_sig,
                consumer_privkey=consumer.get_privkey_hex(),
                dtype="fp32",
            )

            assert is_valid is False, f"Attack {attack} bypassed Freivalds verification!"
            assert cancelled_escrow.status == EscrowStatus.CANCELLED
            assert cancelled_escrow.cancellation_reason == "mathematical_fraud_detected"

            # Preimage must NOT be leaked
            assert preimage_secret not in repr(cancelled_escrow)
            assert "[CONFIDENTIAL]" in repr(cancelled_escrow)
            assert preimage_secret not in cancelled_escrow.get_auth_header(use_preimage=True)

            # Fraud proof must be populated and authoritative
            assert fraud_proof is not None
            assert fraud_proof.task_id == task_id
            assert fraud_proof.worker_pubkey == worker.get_pubkey_hex()
            assert fraud_proof.observed_diff > 0.0

    def test_verify_and_settle_forged_worker_signature_raises(
        self, escrow_mgr: EscrowManager, signers: Dict[str, ClientSigner]
    ):
        """
        Adversarial Test: Worker result is fraudulent, but the worker signature is forged/invalid.
        Asserts that FraudProofBuilder raises WorkerSignatureError.
        """
        rng = np.random.default_rng(789)
        worker = signers["worker"]
        consumer = signers["consumer"]

        task_id = "task_bad_worker_sig"
        escrow_mgr.create_and_hold(task_id, total_sats=100)

        A = rng.standard_normal((8, 8)).astype(np.float32)
        B = rng.standard_normal((8, 8)).astype(np.float32)
        C_fraud = np.matmul(A, B) + 10.0

        forged_sig = "00" * 64

        with pytest.raises(WorkerSignatureError):
            escrow_mgr.verify_and_settle(
                task_id=task_id,
                A=A,
                B=B,
                C=C_fraud,
                worker_pubkey=worker.get_pubkey_hex(),
                worker_sig=forged_sig,
                consumer_privkey=consumer.get_privkey_hex(),
            )

    def test_verify_and_settle_nan_inf_poisoning(
        self, escrow_mgr: EscrowManager, signers: Dict[str, ClientSigner]
    ):
        """
        Adversarial Test: Poisoning matrix C with NaN or Inf.
        Asserts: Detected as invalid, cancels escrow.
        """
        rng = np.random.default_rng(999)
        worker = signers["worker"]
        consumer = signers["consumer"]

        task_id = "task_nan_poison"
        escrow_mgr.create_and_hold(task_id, total_sats=100)

        A = rng.standard_normal((8, 8)).astype(np.float32)
        B = rng.standard_normal((8, 8)).astype(np.float32)
        C_nan = np.matmul(A, B)
        C_nan[0, 0] = np.nan

        c_hash = hashlib.sha256(C_nan.tobytes()).hexdigest()
        worker_sig = worker.sign_message(f"{task_id}:{c_hash}".encode("utf-8"))

        is_valid, cancelled_escrow, _ = escrow_mgr.verify_and_settle(
            task_id=task_id,
            A=A,
            B=B,
            C=C_nan,
            worker_pubkey=worker.get_pubkey_hex(),
            worker_sig=worker_sig,
            consumer_privkey=consumer.get_privkey_hex(),
        )

        assert is_valid is False
        assert cancelled_escrow.status == EscrowStatus.CANCELLED

    def test_verify_and_settle_extreme_dimensions(self, escrow_mgr: EscrowManager):
        """
        Tests boundary dimensions: 1x1 scalar product, 1x32 x 32x1 vector product.
        """
        # 1x1 Matrix
        task_id_1x1 = "task_1x1"
        escrow_mgr.create_and_hold(task_id_1x1, total_sats=50)
        A_1x1 = np.array([[3.0]], dtype=np.float32)
        B_1x1 = np.array([[4.0]], dtype=np.float32)
        C_1x1 = np.array([[12.0]], dtype=np.float32)

        is_valid, escrow_1x1, _ = escrow_mgr.verify_and_settle(task_id_1x1, A_1x1, B_1x1, C_1x1)
        assert is_valid is True
        assert escrow_1x1.status == EscrowStatus.SETTLED

        # 1x32 and 32x1 -> 1x1 dot product
        task_id_vec = "task_vec"
        escrow_mgr.create_and_hold(task_id_vec, total_sats=50)
        rng = np.random.default_rng(42)
        A_vec = rng.standard_normal((1, 32)).astype(np.float32)
        B_vec = rng.standard_normal((32, 1)).astype(np.float32)
        C_vec = np.matmul(A_vec, B_vec)

        is_valid, escrow_vec, _ = escrow_mgr.verify_and_settle(task_id_vec, A_vec, B_vec, C_vec)
        assert is_valid is True
        assert escrow_vec.status == EscrowStatus.SETTLED

    def test_verify_and_settle_unheld_or_terminal_escrow(self, escrow_mgr: EscrowManager):
        """
        Adversarial Test: Calling verify_and_settle on INITIALIZED or terminal escrow.
        """
        A = np.eye(4, dtype=np.float32)
        B = np.eye(4, dtype=np.float32)
        C = np.eye(4, dtype=np.float32)

        # INITIALIZED escrow -> release_payout raises EscrowStateError
        escrow_mgr.create_escrow("t_unheld", total_sats=50)
        with pytest.raises(EscrowStateError) as exc:
            escrow_mgr.verify_and_settle("t_unheld", A, B, C)
        assert "must be HELD" in str(exc.value)

        # Already SETTLED escrow -> raises EscrowStateError
        escrow_mgr.hold_escrow("t_unheld")
        escrow_mgr.release_payout("t_unheld")
        with pytest.raises(EscrowStateError) as exc:
            escrow_mgr.verify_and_settle("t_unheld", A, B, C)
        assert "terminal status 'SETTLED'" in str(exc.value)


# =============================================================================
# Suite 7: L402 Payment Coordinator End-to-End Adversarial Tests
# =============================================================================

class TestPaymentCoordinatorAdversarial:
    """
    Stress-tests L402PaymentCoordinator edge cases, budget refunding on backend failure,
    and concurrent multi-task lifecycles.
    """

    def test_coordinator_budget_rollback_on_backend_failure(self):
        """
        Adversarial Test: Settlement backend throws an error when creating HODL invoice.
        Asserts that coordinator cleanly catches error and refunds the reserved budget,
        preventing budget leak.
        """
        class FailingBackend:
            def create_hodl_invoice(self, *args, **kwargs):
                raise RuntimeError("Settlement node offline")

        bc = BudgetController(max_budget_sats=1000)
        coord = L402PaymentCoordinator(
            settlement_backend=FailingBackend(),
            budget_controller=bc,
        )

        with pytest.raises(EscrowCreationError):
            coord.create_escrow("fail_task", total_sats=250)

        # Budget must be completely refunded! Zero reserved sats.
        assert bc.total_reserved_sats == 0
        assert bc.total_committed_sats == 0

    def test_coordinator_concurrent_multi_task_lifecycle(self):
        """
        Adversarial Concurrency Test:
        Multiple threads creating, releasing payout, and refunding through coordinator.
        Verifies zero budget leakage and exact accounting.
        """
        mock_backend = MockSettlementService()
        bc = BudgetController(max_budget_sats=5000)
        coord = L402PaymentCoordinator(settlement_backend=mock_backend, budget_controller=bc)

        num_tasks = 20
        barrier = threading.Barrier(num_tasks)

        def task_worker(idx: int):
            task_id = f"coord_conc_{idx}"
            barrier.wait()
            escrow = coord.create_escrow(task_id, total_sats=100)

            # Even tasks settle, odd tasks refund
            if idx % 2 == 0:
                coord.release_payout(escrow)
            else:
                coord.abort_and_refund(escrow, reason="worker_cancelled")

        threads = [threading.Thread(target=task_worker, args=(i,)) for i in range(num_tasks)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 10 tasks committed (1000 sats), 10 tasks refunded (0 sats)
        assert bc.total_reserved_sats == 0
        assert bc.total_committed_sats == 1000
