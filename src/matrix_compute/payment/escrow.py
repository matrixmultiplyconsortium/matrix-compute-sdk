"""
Trustless HODL Invoice Escrow State Machine for matrix-compute-sdk.

Coordinates decentralized compute payment escrow via Lightning HODL invoices:
  - Generates secret 32-byte preimages and payment hashes.
  - Manages atomic escrow lifecycle: INITIALIZED -> HELD -> SETTLED / CANCELLED / EXPIRED.
  - Thread-safe concurrency across multi-tile jobs using reentrant locks.
  - Integrates Freivalds verification and authoritative fraud proof construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import logging
import secrets
import threading
import time
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union, runtime_checkable

import numpy as np

from matrix_compute.config import ClientConfig
from matrix_compute.exceptions import (
    EscrowCreationError,
    EscrowError,
    EscrowStateError,
    EscrowTimeoutError,
    PreimageVerificationError,
    SettlementError,
)
from matrix_compute.test_doubles.mock_settlement import MockSettlementService
from matrix_compute.verification.fraud_proof import FraudProofBuilder
from matrix_compute.verification.freivalds import FreivaldsVerifier

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Enums and Protocol Definitions
# -----------------------------------------------------------------------------

class EscrowStatus(str, Enum):
    INITIALIZED = "INITIALIZED"
    HELD = "HELD"
    SETTLED = "SETTLED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"


@runtime_checkable
class SettlementBackendProtocol(Protocol):
    """Abstract protocol for settlement service interactions."""

    def create_hodl_invoice(
        self,
        task_id_or_request: Any,
        total_compute_sats: Optional[int] = None,
        platform_fee_percent: float = 0.02,
        worker_lightning_address: str = "worker@node",
        payment_hash: Optional[str] = None,
        expiry_seconds: int = 3600,
        max_gflops: float = 100.0,
        **kwargs: Any,
    ) -> Any: ...

    def hold_invoice(self, payment_hash: str) -> bool: ...

    def settle_invoice(self, hash_or_request: Any, preimage: Optional[str] = None) -> bool: ...

    def cancel_invoice(self, payment_hash: str) -> bool: ...

    def get_status(self, payment_hash: str) -> Optional[str]: ...


# -----------------------------------------------------------------------------
# Cryptographic Preimage & Payment Hash Utilities
# -----------------------------------------------------------------------------

def generate_escrow_secret() -> Tuple[str, str]:
    """
    Generates a cryptographically secure 32-byte secret preimage and SHA-256 payment hash.
    Returns: (preimage_hex, payment_hash_hex) as 64-character lowercase hex strings.
    """
    preimage_bytes = secrets.token_bytes(32)
    preimage_hex = preimage_bytes.hex().lower()
    payment_hash_hex = hashlib.sha256(preimage_bytes).hexdigest().lower()
    return preimage_hex, payment_hash_hex


def validate_preimage(preimage: str, expected_payment_hash: Optional[str] = None) -> bool:
    """
    Validates that a preimage is a valid 32-byte hex string (64 characters)
    and optionally matches an expected payment hash.
    """
    if not isinstance(preimage, str) or len(preimage) != 64:
        return False
    try:
        raw_bytes = bytes.fromhex(preimage)
        if len(raw_bytes) != 32:
            return False
    except ValueError:
        return False

    if expected_payment_hash is not None:
        computed_hash = hashlib.sha256(raw_bytes).hexdigest().lower()
        if computed_hash != expected_payment_hash.strip().lower():
            return False

    return True


# -----------------------------------------------------------------------------
# Data Model: EscrowState
# -----------------------------------------------------------------------------

@dataclass
class EscrowState:
    """
    Tracks complete lifecycle and cryptographic state of a single compute tile escrow.
    """
    task_id: str
    payment_hash: str
    total_compute_sats: int
    platform_fee_sats: int = 0
    worker_payout_sats: int = 0
    worker_lightning_address: str = "worker@node"
    payment_request: str = ""
    macaroon: str = ""
    status: EscrowStatus = EscrowStatus.INITIALIZED
    created_at: float = field(default_factory=time.time)
    expires_at: float = 0.0
    held_at: Optional[float] = None
    settled_at: Optional[float] = None
    cancelled_at: Optional[float] = None
    cancellation_reason: Optional[str] = None
    fraud_proof: Optional[Any] = None
    max_gflops: float = 100.0
    _preimage: Optional[str] = field(default=None, repr=False)

    @property
    def preimage(self) -> Optional[str]:
        """Confidential accessor for the 32-byte secret preimage."""
        return self._preimage

    @property
    def total_sats(self) -> int:
        """Alias for total_compute_sats."""
        return self.total_compute_sats

    @property
    def is_active(self) -> bool:
        """True if escrow is in flight (INITIALIZED or HELD)."""
        return self.status in (EscrowStatus.INITIALIZED, EscrowStatus.HELD)

    @property
    def is_held(self) -> bool:
        """True if funds are locked into HELD status."""
        return self.status == EscrowStatus.HELD

    @property
    def is_terminal(self) -> bool:
        """True if escrow reached a final state (SETTLED, CANCELLED, EXPIRED)."""
        return self.status in (EscrowStatus.SETTLED, EscrowStatus.CANCELLED, EscrowStatus.EXPIRED)

    def is_expired(self, now: Optional[float] = None) -> bool:
        """True if escrow is active and current time exceeds deadline."""
        current = now if now is not None else time.time()
        return self.is_active and (current >= self.expires_at)

    def get_auth_header(self, use_preimage: bool = False) -> str:
        """
        Formats L402 Authorization header.
        - Prior to settlement: 'L402 <macaroon>:<payment_hash>' (satisfies worker pre-allocation gate).
        - After settlement: 'L402 <macaroon>:<preimage>'.
        """
        if use_preimage and self.status == EscrowStatus.SETTLED and self._preimage:
            return f"L402 {self.macaroon}:{self._preimage}"
        return f"L402 {self.macaroon}:{self.payment_hash}"

    def __repr__(self) -> str:
        secret_display = self._preimage if self.status == EscrowStatus.SETTLED else "[CONFIDENTIAL]"
        return (
            f"EscrowState(task_id='{self.task_id}', status={self.status.value}, "
            f"payment_hash='{self.payment_hash[:12]}...', total_sats={self.total_compute_sats}, "
            f"preimage='{secret_display}')"
        )

    def to_dict(self, include_preimage: bool = False) -> Dict[str, Any]:
        """Serializes escrow state to dictionary with secret redaction by default."""
        data = {
            "task_id": self.task_id,
            "payment_hash": self.payment_hash,
            "status": self.status.value,
            "total_compute_sats": self.total_compute_sats,
            "total_sats": self.total_compute_sats,
            "platform_fee_sats": self.platform_fee_sats,
            "worker_payout_sats": self.worker_payout_sats,
            "worker_lightning_address": self.worker_lightning_address,
            "payment_request": self.payment_request,
            "macaroon": self.macaroon,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "held_at": self.held_at,
            "settled_at": self.settled_at,
            "cancelled_at": self.cancelled_at,
            "cancellation_reason": self.cancellation_reason,
            "has_fraud_proof": self.fraud_proof is not None,
            "max_gflops": self.max_gflops,
        }
        if include_preimage:
            data["preimage"] = self._preimage
        return data


# -----------------------------------------------------------------------------
# Core Class: EscrowManager
# -----------------------------------------------------------------------------

class EscrowManager:
    """
    Thread-safe manager for trustless HODL invoice escrows across distributed tiles.
    Guarantees strict state transitions, preimage secrecy, and timeout sweeps.
    """

    def __init__(
        self,
        settlement_backend: Optional[SettlementBackendProtocol] = None,
        config: Optional[ClientConfig] = None,
        default_timeout_seconds: float = 300.0,
        **kwargs: Any,
    ) -> None:
        self.config = config
        self.verification_key = kwargs.get("verification_key")
        self.auto_settle = kwargs.get("auto_settle", False)
        self.default_timeout_seconds = float(
            default_timeout_seconds
            if default_timeout_seconds is not None
            else (config.task_timeout_seconds if config else 300.0)
        )
        self.settlement_backend: SettlementBackendProtocol = (
            settlement_backend if settlement_backend is not None else MockSettlementService()
        )

        self._lock = threading.RLock()
        self._escrows: Dict[str, EscrowState] = {}          # task_id -> EscrowState
        self._by_payment_hash: Dict[str, str] = {}         # payment_hash -> task_id

    def _resolve_escrow(self, task_id_or_hash: str) -> EscrowState:
        """Resolves an escrow by either task_id or payment_hash under lock."""
        clean = str(task_id_or_hash).strip().lower()
        if task_id_or_hash in self._escrows:
            return self._escrows[task_id_or_hash]
        if clean in self._by_payment_hash:
            tid = self._by_payment_hash[clean]
            return self._escrows[tid]
        raise KeyError(f"Escrow not found for task ID or payment hash: '{task_id_or_hash}'")

    def create_escrow(
        self,
        task_id: str,
        total_sats: int,
        worker_lightning_addr: str = "worker@node",
        timeout_seconds: Optional[float] = None,
        payment_hash: Optional[str] = None,
        max_gflops: float = 100.0,
        custom_preimage: Optional[str] = None,
    ) -> EscrowState:
        """
        Creates and registers a new HODL invoice escrow in INITIALIZED state.
        Generates secret 32-byte preimage R and payment_hash H = SHA256(R).
        Calls settlement backend to mint HODL invoice and Macaroon bound to H.
        """
        if total_sats <= 0:
            raise ValueError(f"total_sats must be positive integer, got {total_sats}")

        with self._lock:
            if task_id in self._escrows:
                raise EscrowCreationError(
                    f"Task ID '{task_id}' already has an active or recorded escrow",
                    task_id=task_id,
                )

            # Generate or validate secret preimage and payment hash
            if custom_preimage is not None:
                if not validate_preimage(custom_preimage, expected_payment_hash=payment_hash):
                    raise PreimageVerificationError(
                        "Invalid custom preimage or hash mismatch",
                        payment_hash=payment_hash,
                        task_id=task_id,
                    )
                preimage_hex = custom_preimage.strip().lower()
                final_payment_hash = (
                    payment_hash.strip().lower()
                    if payment_hash
                    else hashlib.sha256(bytes.fromhex(preimage_hex)).hexdigest().lower()
                )
            elif payment_hash is not None:
                final_payment_hash = payment_hash.strip().lower()
                preimage_hex = None
            else:
                preimage_hex, final_payment_hash = generate_escrow_secret()

            if final_payment_hash in self._by_payment_hash:
                raise EscrowCreationError(
                    f"Payment hash '{final_payment_hash}' already exists in escrow registry",
                    payment_hash=final_payment_hash,
                    task_id=task_id,
                )

            # Request HODL invoice from settlement backend
            timeout_sec = timeout_seconds if timeout_seconds is not None else self.default_timeout_seconds
            now = time.time()
            expires_at = now + timeout_sec

            try:
                inv_resp = self.settlement_backend.create_hodl_invoice(
                    task_id=task_id,
                    total_compute_sats=total_sats,
                    worker_lightning_address=worker_lightning_addr,
                    payment_hash=final_payment_hash,
                    expiry_seconds=int(timeout_sec),
                    max_gflops=max_gflops,
                )
            except Exception as ex:
                raise EscrowCreationError(
                    f"Settlement backend failed to create HODL invoice: {ex}",
                    task_id=task_id,
                    payment_hash=final_payment_hash,
                    amount_sats=total_sats,
                    cause=ex,
                )

            macaroon_str = getattr(inv_resp, "macaroon", "")
            payment_req = getattr(inv_resp, "payment_request", "")
            fee_sats = getattr(inv_resp, "platform_fee_sats", int(round(total_sats * 0.02)))
            payout_sats = getattr(inv_resp, "worker_payout_sats", total_sats - fee_sats)

            # If settlement backend minted its own preimage and none was provided locally
            if preimage_hex is None and getattr(inv_resp, "preimage", None):
                preimage_hex = getattr(inv_resp, "preimage")

            escrow = EscrowState(
                task_id=task_id,
                payment_hash=final_payment_hash,
                total_compute_sats=total_sats,
                platform_fee_sats=fee_sats,
                worker_payout_sats=payout_sats,
                worker_lightning_address=worker_lightning_addr,
                payment_request=payment_req,
                macaroon=macaroon_str,
                status=EscrowStatus.INITIALIZED,
                created_at=now,
                expires_at=expires_at,
                max_gflops=max_gflops,
                _preimage=preimage_hex,
            )

            self._escrows[task_id] = escrow
            self._by_payment_hash[final_payment_hash] = task_id
            return escrow

    def hold_escrow(self, task_id_or_hash: str) -> EscrowState:
        """
        Transitions escrow from INITIALIZED to HELD by locking satoshis with settlement layer.
        """
        with self._lock:
            escrow = self._resolve_escrow(task_id_or_hash)

            if escrow.status == EscrowStatus.HELD:
                raise EscrowStateError(
                    f"Escrow '{escrow.task_id}' is already in HELD status",
                    task_id=escrow.task_id,
                    current_status=escrow.status.value,
                    attempted_transition="HOLD",
                )

            if escrow.status != EscrowStatus.INITIALIZED:
                raise EscrowStateError(
                    f"Cannot hold escrow in status '{escrow.status.value}'; must be INITIALIZED",
                    task_id=escrow.task_id,
                    current_status=escrow.status.value,
                    attempted_transition="HOLD",
                )

            try:
                self.settlement_backend.hold_invoice(escrow.payment_hash)
            except Exception as ex:
                raise EscrowStateError(
                    f"Settlement backend failed to hold invoice: {ex}",
                    task_id=escrow.task_id,
                    current_status=escrow.status.value,
                    attempted_transition="HOLD",
                    cause=ex,
                )

            escrow.status = EscrowStatus.HELD
            escrow.held_at = time.time()
            return escrow

    def create_and_hold(
        self,
        task_id: str,
        total_sats: int,
        worker_lightning_addr: str = "worker@node",
        timeout_seconds: Optional[float] = None,
        max_gflops: float = 100.0,
    ) -> EscrowState:
        """Convenience method: creates escrow and immediately transitions to HELD."""
        with self._lock:
            escrow = self.create_escrow(
                task_id=task_id,
                total_sats=total_sats,
                worker_lightning_addr=worker_lightning_addr,
                timeout_seconds=timeout_seconds,
                max_gflops=max_gflops,
            )
            return self.hold_escrow(escrow.task_id)

    def get_auth_credential(self, task_id: str) -> str:
        """
        Returns L402 authorization credential header value withholding secret preimage.
        Format: 'L402 <macaroon>:<payment_hash>'.
        """
        with self._lock:
            escrow = self._resolve_escrow(task_id)
            return escrow.get_auth_header(use_preimage=False)

    def release_payout(self, task_id_or_hash: str) -> EscrowState:
        """
        Reveals secret preimage to settle HODL invoice and release payout to compute worker.
        Precondition: Escrow must be in HELD status and not expired.
        Postcondition: Escrow status transitions to SETTLED.
        """
        with self._lock:
            escrow = self._resolve_escrow(task_id_or_hash)

            if escrow.is_terminal:
                raise EscrowStateError(
                    f"Cannot settle escrow in terminal status '{escrow.status.value}'",
                    task_id=escrow.task_id,
                    current_status=escrow.status.value,
                    attempted_transition="SETTLE",
                )

            if escrow.status != EscrowStatus.HELD:
                raise EscrowStateError(
                    f"Cannot settle escrow in status '{escrow.status.value}'; must be HELD",
                    task_id=escrow.task_id,
                    current_status=escrow.status.value,
                    attempted_transition="SETTLE",
                )

            # Check expiration deadline
            if escrow.is_expired():
                self.abort_and_refund(escrow.task_id, reason="expired_before_settlement")
                raise EscrowTimeoutError(
                    f"Escrow '{escrow.task_id}' expired before settlement could complete",
                    task_id=escrow.task_id,
                )

            if not escrow._preimage:
                raise SettlementError(
                    f"Secret preimage missing for escrow '{escrow.task_id}'",
                    task_id=escrow.task_id,
                    payment_hash=escrow.payment_hash,
                )

            try:
                self.settlement_backend.settle_invoice(escrow.payment_hash, escrow._preimage)
            except Exception as ex:
                raise SettlementError(
                    f"Settlement backend rejected settlement: {ex}",
                    task_id=escrow.task_id,
                    payment_hash=escrow.payment_hash,
                    cause=ex,
                )

            escrow.status = EscrowStatus.SETTLED
            escrow.settled_at = time.time()
            return escrow

    def abort_and_refund(
        self,
        task_id_or_hash: str,
        reason: str = "cancelled",
        fraud_proof: Optional[Any] = None,
    ) -> EscrowState:
        """
        Strictly withholds preimage, cancels HODL invoice, and refunds locked satoshis.
        Precondition: Escrow must be INITIALIZED or HELD.
        Postcondition: Escrow status transitions to CANCELLED.
        """
        with self._lock:
            escrow = self._resolve_escrow(task_id_or_hash)

            if escrow.is_terminal:
                raise EscrowStateError(
                    f"Cannot abort escrow in terminal status '{escrow.status.value}'",
                    task_id=escrow.task_id,
                    current_status=escrow.status.value,
                    attempted_transition="ABORT",
                )

            try:
                self.settlement_backend.cancel_invoice(escrow.payment_hash)
            except Exception as ex:
                logger.warning("Settlement backend invoice cancel error: %s", ex)

            escrow.status = EscrowStatus.CANCELLED
            escrow.cancelled_at = time.time()
            escrow.cancellation_reason = reason
            escrow.fraud_proof = fraud_proof
            return escrow

    def check_timeouts(self, now: Optional[float] = None) -> List[EscrowState]:
        """
        Sweeps active escrows and auto-cancels any exceeding their execution deadline.
        Returns list of newly expired escrows.
        """
        current = now if now is not None else time.time()
        expired_list: List[EscrowState] = []

        with self._lock:
            for escrow in self._escrows.values():
                if escrow.is_active and current >= escrow.expires_at:
                    try:
                        self.settlement_backend.cancel_invoice(escrow.payment_hash)
                    except Exception as ex:
                        logger.warning("Error cancelling expired invoice %s: %s", escrow.payment_hash, ex)

                    escrow.status = EscrowStatus.EXPIRED
                    escrow.cancelled_at = current
                    escrow.cancellation_reason = "task_timeout"
                    expired_list.append(escrow)

        return expired_list

    def get_escrow(self, task_id: str) -> Optional[EscrowState]:
        """Look up escrow by task_id."""
        with self._lock:
            return self._escrows.get(task_id)

    def get_by_payment_hash(self, payment_hash: str) -> Optional[EscrowState]:
        """Look up escrow by payment_hash."""
        clean = payment_hash.strip().lower()
        with self._lock:
            tid = self._by_payment_hash.get(clean)
            return self._escrows.get(tid) if tid else None

    def verify_and_settle(
        self,
        task_id: str,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        worker_pubkey: Optional[str] = None,
        worker_sig: Optional[str] = None,
        consumer_privkey: Optional[str] = None,
        dtype: str = "fp32",
        k_rounds: int = 5,
    ) -> Tuple[bool, EscrowState, Optional[Any]]:
        """
        Executes end-to-end verification and escrow settlement:
          1. Runs randomized O(n^2) Freivalds verification check.
          2. On pass: reveals secret preimage to finalize payout (HELD -> SETTLED).
          3. On fail: strictly withholds preimage, builds FreivaldsFraudProof,
             and cancels invoice with refund (HELD -> CANCELLED).
        Returns: (is_valid, escrow_state, fraud_proof_or_none)
        """
        with self._lock:
            escrow = self._resolve_escrow(task_id)

        # 1. Run Freivalds verification
        is_valid, discrepancy, bound, proof_data = FreivaldsVerifier.verify(
            A, B, C, dtype=dtype, k_rounds=k_rounds
        )

        if is_valid:
            # Verification passed: release payout to worker
            settled_escrow = self.release_payout(task_id)
            return True, settled_escrow, None
        else:
            # Computational fraud detected
            fraud_proof = None
            if worker_pubkey and worker_sig:
                priv_key = consumer_privkey or (
                    self.config.consumer_privkey_hex if self.config else None
                )
                if not priv_key:
                    from matrix_compute.crypto.signer import ClientSigner
                    priv_key = ClientSigner().get_privkey_hex()
                fraud_proof = FraudProofBuilder.build_fraud_proof(
                    task_id=task_id,
                    worker_pubkey=worker_pubkey,
                    A=A,
                    B=B,
                    C=C,
                    proof_data=proof_data,
                    worker_sig=worker_sig,
                    consumer_privkey=priv_key,
                    dtype=dtype,
                    epsilon_tolerance=bound,
                    verify_worker_sig=True,
                )

            cancelled_escrow = self.abort_and_refund(
                task_id,
                reason="mathematical_fraud_detected",
                fraud_proof=fraud_proof,
            )
            return False, cancelled_escrow, fraud_proof
