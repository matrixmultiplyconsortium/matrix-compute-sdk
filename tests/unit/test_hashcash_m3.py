"""Unit and Adversarial Tests for SDK Hashcash Solver & Trial Macaroon Deserialization (Milestone 3)."""

from __future__ import annotations

import hashlib
import time
import pytest

from matrix_compute.crypto.hashcash import (
    HashcashSolver,
    HashcashVerifier,
    count_leading_zero_bits,
    solve_hashcash,
    verify_hashcash,
    verify_hashcash_with_reason,
)
from matrix_compute.payment.l402_client import MacaroonToken


class TestSdkHashcash:
    """Tests for client-side SHA-256 Hashcash PoW solver and verifier."""

    def test_leading_zero_bits_calculation(self) -> None:
        assert count_leading_zero_bits(b"\x00\x00\xff" + b"\x00" * 29) == 16
        assert count_leading_zero_bits(b"\x00\x0f\xff" + b"\x00" * 29) == 12
        assert count_leading_zero_bits(b"\x80" + b"\x00" * 31) == 0
        assert count_leading_zero_bits(b"\x01" + b"\x00" * 31) == 7

    def test_solve_and_verify_valid_stamp(self) -> None:
        stamp = HashcashSolver.solve(resource="trial_job_42", bits=12)
        assert stamp.startswith("mc_pow:12:")
        assert HashcashVerifier.verify(stamp, expected_resource="trial_job_42", min_bits=12) is True

        digest = hashlib.sha256(stamp.encode("utf-8")).digest()
        assert count_leading_zero_bits(digest) >= 12

    def test_reject_expired_and_future_timestamps(self) -> None:
        now = int(time.time())
        expired_stamp = solve_hashcash("resource_exp", bits=10, timestamp=now - 400)
        ok_exp, reason_exp = verify_hashcash_with_reason(
            expired_stamp, expected_resource="resource_exp", min_bits=10, max_age_seconds=300, now=now
        )
        assert ok_exp is False
        assert "EXPIRED_TIMESTAMP" in reason_exp

        future_stamp = solve_hashcash("resource_fut", bits=10, timestamp=now + 120)
        ok_fut, reason_fut = verify_hashcash_with_reason(
            future_stamp, expected_resource="resource_fut", min_bits=10, max_age_seconds=300, now=now
        )
        assert ok_fut is False
        assert reason_fut == "FUTURE_TIMESTAMP_REJECTED"

    def test_reject_mismatched_resource_and_insufficient_bits(self) -> None:
        stamp = solve_hashcash("task_alpha", bits=10)
        ok_res, reason_res = verify_hashcash_with_reason(stamp, expected_resource="task_beta", min_bits=10)
        assert ok_res is False
        assert "RESOURCE_MISMATCH" in reason_res

        ok_bits, reason_bits = verify_hashcash_with_reason(stamp, expected_resource="task_alpha", min_bits=16)
        assert ok_bits is False
        assert "INSUFFICIENT_STAMP_BITS" in reason_bits

    def test_reject_malformed_stamps(self) -> None:
        for bad in ["", "invalid", "mc_pow:12:abc:res:0", "other:12:1700000000:res:0", "mc_pow:12:1700000000:res"]:
            assert verify_hashcash(bad, expected_resource="res", min_bits=8) is False


class TestSdkTrialMacaroonDeserialization:
    """Tests for `mc_trial_<base64>` token deserialization in SDK MacaroonToken."""

    def test_deserialize_mc_trial_prefixed_token(self) -> None:
        import base64
        import hmac
        import json

        secret = "trial-secret"
        caveats = {
            "tier": "FREE_TRIAL",
            "task_id": "t_01",
            "allowed_workers": ["02" + "11" * 32],
            "max_matrix_dim": 4096,
            "locked_sats": 0,
            "expiry": int(time.time()) + 3600,
            "max_gflops": 10.0,
            "payment_hash": "0" * 64,
        }
        sig = hmac.new(secret.encode(), json.dumps(caveats, sort_keys=True).encode(), hashlib.sha256).hexdigest()
        envelope = {"caveats": caveats, "signature": sig}
        b64 = base64.urlsafe_b64encode(json.dumps(envelope).encode()).decode().rstrip("=")
        raw_token = f"mc_trial_{b64}"

        tok = MacaroonToken.deserialize(raw_token)
        assert tok.caveats.tier == "FREE_TRIAL"
        assert tok.caveats.max_matrix_dim == 4096
        assert tok.caveats.locked_sats == 0
        assert tok.caveats.allowed_workers == ["02" + "11" * 32]
        assert tok.verify_hmac(secret) is True
