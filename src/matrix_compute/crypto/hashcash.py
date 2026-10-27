"""Client-side SHA-256 Hashcash Proof-of-Work Solver and Verifier for FREE_TRIAL anti-spam."""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import threading
import time
from typing import Optional, Tuple


def count_leading_zero_bits(digest_bytes: bytes) -> int:
    """Counts continuous leading zero bits in a SHA-256 digest."""
    if not isinstance(digest_bytes, (bytes, bytearray)):
        raise TypeError("digest_bytes must be bytes or bytearray")
    zeros = 0
    for b in digest_bytes:
        if b == 0:
            zeros += 8
        else:
            zeros += 8 - int(b).bit_length()
            break
    return zeros


def solve_hashcash(
    resource: str,
    bits: int = 20,
    max_iterations: int = 10_000_000,
    timestamp: Optional[int] = None,
) -> str:
    """
    Solves a SHA-256 Hashcash challenge for a given resource identifier.
    Wire format: `mc_pow:<bits>:<timestamp>:<resource>:<nonce>`

    Args:
        resource: Task or job identifier bound to the proof-of-work.
        bits: Number of leading zero bits required (default 20, ~1M hashes / ~1.5-2s CPU).
        max_iterations: Maximum nonce iterations before raising RuntimeError.
        timestamp: Optional explicit Unix timestamp in seconds (defaults to int(time.time())).

    Returns:
        Valid Hashcash stamp string `mc_pow:<bits>:<timestamp>:<resource>:<nonce>`.
    """
    if not isinstance(resource, str) or not resource.strip():
        raise ValueError("Hashcash resource must be a non-empty string")
    if ":" in resource:
        raise ValueError("Hashcash resource must not contain ':' delimiter")
    if isinstance(bits, bool) or not isinstance(bits, int) or not (0 <= bits <= 256):
        raise ValueError(f"Hashcash bits must be an integer in [0, 256], got {bits}")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations <= 0:
        raise ValueError(f"max_iterations must be a positive integer, got {max_iterations}")

    ts = int(timestamp if timestamp is not None else time.time())
    clean_resource = resource.strip()
    prefix_str = f"mc_pow:{bits}:{ts}:{clean_resource}:"
    prefix_bytes = prefix_str.encode("utf-8")

    if bits == 0:
        return f"{prefix_str}0"

    target = 1 << (256 - bits)
    base_hasher = hashlib.sha256(prefix_bytes)

    for nonce in range(max_iterations):
        h = base_hasher.copy()
        nonce_str = str(nonce)
        h.update(nonce_str.encode("ascii"))
        digest = h.digest()
        if int.from_bytes(digest, "big") < target:
            return f"{prefix_str}{nonce_str}"

    raise RuntimeError(
        f"Hashcash solver exceeded max_iterations ({max_iterations}) for difficulty {bits} bits"
    )


def verify_hashcash_with_reason(
    stamp: str,
    expected_resource: str,
    min_bits: int = 20,
    max_age_seconds: int = 300,
    now: Optional[float] = None,
    max_future_skew_seconds: int = 30,
) -> Tuple[bool, str]:
    """
    Verifies a Hashcash stamp and returns `(is_valid, reason)`.

    Checks:
    1. Wire format `mc_pow:<bits>:<timestamp>:<resource>:<nonce>` (5 colon-separated parts).
    2. Claimed difficulty `bits >= min_bits`.
    3. Resource binding `resource == expected_resource`.
    4. Timestamp freshness (`now - ts <= max_age_seconds` and `ts <= now + max_future_skew_seconds`).
    5. Cryptographic SHA-256 digest has at least `bits` leading zero bits.
    """
    if not isinstance(stamp, str) or not stamp.strip():
        return False, "STAMP_NOT_STRING"

    clean_stamp = stamp.strip()
    parts = clean_stamp.split(":")
    if len(parts) != 5 or parts[0] != "mc_pow":
        return False, "MALFORMED_STAMP_FORMAT"

    bits_str, ts_str, resource, nonce = parts[1], parts[2], parts[3], parts[4]
    if not bits_str or not ts_str or not resource or not nonce:
        return False, "MALFORMED_STAMP_VALUES"

    try:
        claimed_bits = int(bits_str)
        ts = int(ts_str)
    except ValueError:
        return False, "MALFORMED_STAMP_VALUES"

    if claimed_bits < 0 or claimed_bits > 256:
        return False, "INVALID_STAMP_BITS_RANGE"

    if claimed_bits < min_bits:
        return False, f"INSUFFICIENT_STAMP_BITS: got {claimed_bits}, required {min_bits}"

    if resource != str(expected_resource).strip():
        return False, f"RESOURCE_MISMATCH: expected '{expected_resource}', got '{resource}'"

    current_time = float(now if now is not None else time.time())
    if ts > current_time + max_future_skew_seconds:
        return False, "FUTURE_TIMESTAMP_REJECTED"
    if (current_time - ts) > max_age_seconds:
        return (
            False,
            f"EXPIRED_TIMESTAMP: age {int(current_time - ts)}s exceeds {max_age_seconds}s limit",
        )

    digest = hashlib.sha256(clean_stamp.encode("utf-8")).digest()
    actual_bits = count_leading_zero_bits(digest)
    if actual_bits < claimed_bits:
        return (
            False,
            f"PROOF_OF_WORK_INVALID: actual leading zeros {actual_bits} < required {claimed_bits}",
        )

    return True, "VALID_HASHCASH_PROOF"


def verify_hashcash(
    stamp: str,
    expected_resource: str,
    min_bits: int = 20,
    max_age_seconds: int = 300,
    now: Optional[float] = None,
) -> bool:
    """
    Verifies a Hashcash stamp and returns True if valid, False otherwise.
    """
    ok, _ = verify_hashcash_with_reason(
        stamp=stamp,
        expected_resource=expected_resource,
        min_bits=min_bits,
        max_age_seconds=max_age_seconds,
        now=now,
    )
    return ok


class HashcashSolver:
    """Object-oriented Hashcash solver interface."""

    count_leading_zero_bits = staticmethod(count_leading_zero_bits)

    @staticmethod
    def solve(
        resource: str,
        bits: int = 20,
        max_iterations: int = 10_000_000,
        timestamp: Optional[int] = None,
    ) -> str:
        return solve_hashcash(
            resource=resource,
            bits=bits,
            max_iterations=max_iterations,
            timestamp=timestamp,
        )


class HashcashVerifier:
    """
    Hashcash verifier with optional thread-safe replay cache for spent stamps.
    """

    def __init__(
        self,
        min_bits: int = 20,
        max_age_seconds: int = 300,
        max_cache_size: int = 10_000,
    ) -> None:
        self.min_bits = min_bits
        self.max_age_seconds = max_age_seconds
        self.max_cache_size = max_cache_size
        self._lock = threading.RLock()
        self._spent_stamps: OrderedDict[str, float] = OrderedDict()

    count_leading_zero_bits = staticmethod(count_leading_zero_bits)

    @staticmethod
    def verify(
        stamp: str,
        expected_resource: str,
        min_bits: int = 20,
        max_age_seconds: int = 300,
        now: Optional[float] = None,
    ) -> bool:
        return verify_hashcash(
            stamp=stamp,
            expected_resource=expected_resource,
            min_bits=min_bits,
            max_age_seconds=max_age_seconds,
            now=now,
        )

    @staticmethod
    def verify_with_reason(
        stamp: str,
        expected_resource: str,
        min_bits: int = 20,
        max_age_seconds: int = 300,
        now: Optional[float] = None,
    ) -> Tuple[bool, str]:
        return verify_hashcash_with_reason(
            stamp=stamp,
            expected_resource=expected_resource,
            min_bits=min_bits,
            max_age_seconds=max_age_seconds,
            now=now,
        )

    def verify_and_consume(
        self,
        stamp: str,
        expected_resource: str,
        min_bits: Optional[int] = None,
        max_age_seconds: Optional[int] = None,
        now: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """
        Verifies the Hashcash stamp and records it in the replay cache so it cannot be reused.
        """
        req_bits = min_bits if min_bits is not None else self.min_bits
        req_age = max_age_seconds if max_age_seconds is not None else self.max_age_seconds
        current_time = float(now if now is not None else time.time())

        ok, reason = verify_hashcash_with_reason(
            stamp=stamp,
            expected_resource=expected_resource,
            min_bits=req_bits,
            max_age_seconds=req_age,
            now=current_time,
        )
        if not ok:
            return False, reason

        clean_stamp = stamp.strip()
        with self._lock:
            cutoff = current_time - req_age
            expired = [k for k, ts in self._spent_stamps.items() if ts < cutoff]
            for k in expired:
                self._spent_stamps.pop(k, None)

            if clean_stamp in self._spent_stamps:
                return False, "REPLAYED_HASHCASH_STAMP"

            self._spent_stamps[clean_stamp] = current_time
            while len(self._spent_stamps) > self.max_cache_size:
                self._spent_stamps.popitem(last=False)

        return True, "VALID_HASHCASH_PROOF"
