"""Exception hierarchy for matrix-compute-sdk."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


class MatrixComputeError(Exception):
    """Base exception for all matrix-compute-sdk errors."""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.cause = cause

    def __str__(self) -> str:
        parts = [self.message]
        if self.details:
            detail_str = ", ".join(f"{k}={v!r}" for k, v in self.details.items() if v is not None)
            if detail_str:
                parts.append(f"({detail_str})")
        if self.cause is not None:
            parts.append(f"[caused by: {type(self.cause).__name__}: {self.cause}]")
        return " ".join(parts)


# -----------------------------------------------------------------------------
# Configuration & Matrix Dimension Errors
# -----------------------------------------------------------------------------

class ConfigurationError(MatrixComputeError):
    """Raised when client configuration is invalid, unparseable, or missing required values."""


class MatrixDimensionError(MatrixComputeError, ValueError):
    """Raised when input matrix shapes are invalid or incompatible for matrix multiplication."""

    def __init__(
        self,
        message: str,
        shape_a: Optional[Tuple[int, ...]] = None,
        shape_b: Optional[Tuple[int, ...]] = None,
        expected_inner_dim: Optional[int] = None,
        **kwargs: Any,
    ):
        details = {
            "shape_a": shape_a,
            "shape_b": shape_b,
            "expected_inner_dim": expected_inner_dim,
        }
        details.update(kwargs.get("details", {}))
        super().__init__(message, details=details, cause=kwargs.get("cause"))
        self.shape_a = shape_a
        self.shape_b = shape_b
        self.expected_inner_dim = expected_inner_dim


class TilingError(MatrixComputeError, ValueError):
    """Raised when matrix partitioning or result accumulation/assembly fails."""


# -----------------------------------------------------------------------------
# Discovery & Worker Availability Errors
# -----------------------------------------------------------------------------

class DiscoveryError(MatrixComputeError):
    """Raised when Nostr relay worker discovery or event parsing encounters an error."""


class WorkerUnavailableError(DiscoveryError):
    """Raised when no suitable compute worker is available matching required VRAM, price, or slots."""

    def __init__(
        self,
        message: str = "No suitable compute workers available matching task criteria",
        required_vram_mb: Optional[int] = None,
        max_price_per_gflop: Optional[float] = None,
        available_count: int = 0,
        **kwargs: Any,
    ):
        details = {
            "required_vram_mb": required_vram_mb,
            "max_price_per_gflop": max_price_per_gflop,
            "available_count": available_count,
        }
        details.update(kwargs.get("details", {}))
        super().__init__(message, details=details, cause=kwargs.get("cause"))
        self.required_vram_mb = required_vram_mb
        self.max_price_per_gflop = max_price_per_gflop
        self.available_count = available_count


class NoWorkersFoundError(WorkerUnavailableError):
    """Raised when no compute workers match the specified hardware, budget, or precision criteria."""


class RelayConnectionError(DiscoveryError):
    """Raised when connection to a specific Nostr relay fails or is unreachable."""

    def __init__(
        self,
        message: str,
        relay_url: Optional[str] = None,
        **kwargs: Any,
    ):
        details = {"relay_url": relay_url}
        details.update(kwargs.get("details", {}))
        super().__init__(message, details=details, cause=kwargs.get("cause"))
        self.relay_url = relay_url


class RelayPoolExhaustedError(DiscoveryError):
    """Raised when all configured Nostr relays in the connection pool fail or are offline."""


class InvalidNostrEventError(DiscoveryError):
    """Raised when a Nostr event has invalid formatting, missing required fields, or fails signature verification."""


# -----------------------------------------------------------------------------
# P2P gRPC Task Dispatch & Streaming Errors
# -----------------------------------------------------------------------------

class DispatchError(MatrixComputeError):
    """Base exception for errors during P2P task negotiation, chunk streaming, or result retrieval."""

    def __init__(
        self,
        message: str,
        task_id: Optional[str] = None,
        **kwargs: Any,
    ):
        details = {"task_id": task_id}
        details.update(kwargs.get("details", {}))
        super().__init__(message, details=details, cause=kwargs.get("cause"))
        self.task_id = task_id


class WorkerConnectionError(DispatchError):
    """Raised when gRPC or transport connection to a selected worker fails or drops."""

    def __init__(
        self,
        message: str,
        task_id: Optional[str] = None,
        worker_pubkey: Optional[str] = None,
        multiaddr: Optional[str] = None,
        **kwargs: Any,
    ):
        details = {"task_id": task_id, "worker_pubkey": worker_pubkey, "multiaddr": multiaddr}
        details.update(kwargs.get("details", {}))
        super().__init__(message, task_id=task_id, details=details, cause=kwargs.get("cause"))
        self.worker_pubkey = worker_pubkey
        self.multiaddr = multiaddr


class WorkerTaskRejectedError(DispatchError):
    """Raised when a worker node explicitly rejects RequestTileTask negotiation."""

    def __init__(
        self,
        message: str,
        task_id: Optional[str] = None,
        worker_pubkey: Optional[str] = None,
        reason: Optional[str] = None,
        **kwargs: Any,
    ):
        details = {"task_id": task_id, "worker_pubkey": worker_pubkey, "reason": reason}
        details.update(kwargs.get("details", {}))
        super().__init__(message, task_id=task_id, details=details, cause=kwargs.get("cause"))
        self.task_id = task_id
        self.worker_pubkey = worker_pubkey
        self.reason = reason


class StreamingError(DispatchError):
    """Raised when streaming tensor chunks to worker fails or is aborted by the server."""

    def __init__(
        self,
        message: str,
        task_id: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(message, task_id=task_id, **kwargs)


class TaskDispatchError(StreamingError):
    """Raised when worker returned mismatched task_id or dispatch fails."""

    def __init__(
        self,
        message: str = "Task dispatch error",
        task_id: Optional[str] = None,
        returned_task_id: Optional[str] = None,
        **kwargs: Any,
    ):
        details = {
            "task_id": task_id,
            "returned_task_id": returned_task_id,
        }
        details.update(kwargs.get("details", {}))
        super().__init__(message, task_id=task_id, details=details, cause=kwargs.get("cause"))
        self.task_id = task_id
        self.returned_task_id = returned_task_id



class TaskTimeoutError(DispatchError):
    """Raised when a tile task exceeds configured execution timeout without completion."""

    def __init__(
        self,
        message: str,
        task_id: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        **kwargs: Any,
    ):
        details = {"task_id": task_id, "timeout_seconds": timeout_seconds}
        details.update(kwargs.get("details", {}))
        super().__init__(message, details=details, cause=kwargs.get("cause"))
        self.task_id = task_id
        self.timeout_seconds = timeout_seconds


class WorkerSignatureError(DispatchError):
    """Raised when worker's cryptographic signature on TileResult is invalid, forged, or missing."""

    def __init__(
        self,
        message: str = "Invalid worker cryptographic signature on TileResult",
        task_id: Optional[str] = None,
        worker_pubkey: Optional[str] = None,
        signature: Optional[str] = None,
        **kwargs: Any,
    ):
        details = {
            "task_id": task_id,
            "worker_pubkey": worker_pubkey,
            "signature": signature,
        }
        details.update(kwargs.get("details", {}))
        super().__init__(message, details=details, cause=kwargs.get("cause"))
        self.task_id = task_id
        self.worker_pubkey = worker_pubkey
        self.signature = signature


# -----------------------------------------------------------------------------
# Payment & L402 Escrow Errors
# -----------------------------------------------------------------------------

class PaymentError(MatrixComputeError):
    """Base exception for L402 HODL invoice, escrow locking, or settlement failures."""

    def __init__(
        self,
        message: str,
        payment_hash: Optional[str] = None,
        task_id: Optional[str] = None,
        amount_sats: Optional[int] = None,
        **kwargs: Any,
    ):
        details = {
            "payment_hash": payment_hash,
            "task_id": task_id,
            "amount_sats": amount_sats,
        }
        details.update(kwargs.get("details", {}))
        super().__init__(message, details=details, cause=kwargs.get("cause"))
        self.payment_hash = payment_hash
        self.task_id = task_id
        self.amount_sats = amount_sats


class L402ChallengeError(PaymentError):
    """Raised when parsing or negotiating an L402/LSAT WWW-Authenticate challenge fails."""


class MacaroonError(PaymentError):
    """Base exception for L402 Macaroon deserialization, caveat extraction, or verification errors."""


class MacaroonValidationError(MacaroonError):
    """Raised when Macaroon caveats (task_id, hash, expiry, gflops, sats) fail verification."""


class BudgetExceededError(PaymentError):
    """Raised when an operation would exceed maximum price per GFLOP, task budget, or cumulative budget."""

    def __init__(
        self,
        message: str,
        current_budget_sats: Optional[int] = None,
        requested_sats: Optional[int] = None,
        max_budget_sats: Optional[int] = None,
        **kwargs: Any,
    ):
        details = {
            "current_budget_sats": current_budget_sats,
            "requested_sats": requested_sats,
            "max_budget_sats": max_budget_sats,
        }
        details.update(kwargs.get("details", {}))
        super().__init__(message, details=details, cause=kwargs.get("cause"))
        self.current_budget_sats = current_budget_sats
        self.requested_sats = requested_sats
        self.max_budget_sats = max_budget_sats


class EscrowError(PaymentError):
    """Base exception for HODL invoice escrow state machine and management errors."""


class EscrowCreationError(EscrowError):
    """Raised when settlement service fails to create a HODL invoice or mint Macaroon."""


class InvoiceHoldError(EscrowError):
    """Raised when consumer funds cannot be locked into HELD state in settlement layer."""


class SettlementError(PaymentError):
    """Raised when releasing preimage to settle HODL invoice fails or is rejected."""


class RefundError(PaymentError):
    """Raised when canceling a HODL invoice or refunding consumer funds fails."""


class EscrowStateError(EscrowError):
    """Raised when an illegal or out-of-sequence state transition is attempted on an escrow."""

    def __init__(
        self,
        message: str,
        task_id: Optional[str] = None,
        current_status: Optional[str] = None,
        attempted_transition: Optional[str] = None,
        **kwargs: Any,
    ):
        details = {
            "task_id": task_id,
            "current_status": current_status,
            "attempted_transition": attempted_transition,
        }
        details.update(kwargs.get("details", {}))
        super().__init__(message, details=details, cause=kwargs.get("cause"))
        self.task_id = task_id
        self.current_status = current_status
        self.attempted_transition = attempted_transition


class EscrowTimeoutError(EscrowError, TaskTimeoutError):
    """Raised when an escrow expires due to worker timeout exceeding task deadline."""

    def __init__(
        self,
        message: str,
        task_id: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        **kwargs: Any,
    ):
        super().__init__(message, task_id=task_id, timeout_seconds=timeout_seconds, **kwargs)


class PreimageVerificationError(PaymentError, ValueError):
    """Raised when a secret preimage fails 32-byte hexadecimal validation or payment_hash matching."""


# -----------------------------------------------------------------------------
# Verification & Fraud Errors
# -----------------------------------------------------------------------------

class VerificationFailedError(MatrixComputeError):
    """
    Raised when mathematical verification (Freivalds) of a tile result fails.
    Indicates that max(|ABr - Cr|) strictly exceeded dynamic error bound.
    """

    def __init__(
        self,
        message: str = "Matrix tile verification failed: discrepancy exceeds dynamic bound",
        task_id: Optional[str] = None,
        worker_pubkey: Optional[str] = None,
        discrepancy: Optional[float] = None,
        bound: Optional[float] = None,
        proof_data: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        details = {
            "task_id": task_id,
            "worker_pubkey": worker_pubkey,
            "discrepancy": discrepancy,
            "bound": bound,
        }
        details.update(kwargs.get("details", {}))
        super().__init__(message, details=details, cause=kwargs.get("cause"))
        self.task_id = task_id
        self.worker_pubkey = worker_pubkey
        self.discrepancy = discrepancy
        self.bound = bound
        self.proof_data = proof_data or {}


class FraudDetectedError(VerificationFailedError):
    """
    Raised when computational fraud is formally proven and an authoritative
    FreivaldsFraudProof instance is constructed for worker bond slashing.
    """

    def __init__(
        self,
        message: str = "Computational fraud detected: worker produced invalid result",
        task_id: Optional[str] = None,
        worker_pubkey: Optional[str] = None,
        discrepancy: Optional[float] = None,
        bound: Optional[float] = None,
        fraud_proof: Any = None,
        **kwargs: Any,
    ):
        super().__init__(
            message=message,
            task_id=task_id,
            worker_pubkey=worker_pubkey,
            discrepancy=discrepancy,
            bound=bound,
            proof_data={"has_fraud_proof": fraud_proof is not None},
            cause=kwargs.get("cause"),
        )
        self.fraud_proof = fraud_proof


# -----------------------------------------------------------------------------
# Confidential Computing (TEE) & Hardware Attestation Errors (Milestone 2)
# -----------------------------------------------------------------------------

class AttestationError(MatrixComputeError):
    """Base exception for all hardware attestation verification failures."""

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        task_id: Optional[str] = None,
        **kwargs: Any,
    ):
        details = {"provider": provider, "task_id": task_id}
        details.update(kwargs.get("details", {}))
        super().__init__(message, details=details, cause=kwargs.get("cause"))
        self.provider = provider
        self.task_id = task_id


class AttestationEvidenceMalformedError(AttestationError):
    """Raised when attestation evidence format is invalid, missing required fields, or unparseable."""


class AttestationCertificateError(AttestationError):
    """Base exception for X.509 certificate decoding, syntax, or extension parsing errors."""


class AttestationChainInvalidError(AttestationCertificateError):
    """Raised when X.509 certificate chain is broken, issuer/subject mismatch, or intermediate CA violation."""


class AttestationUntrustedRootError(AttestationCertificateError):
    """Raised when the root of the certificate chain does not anchor to any trusted Root CA."""


class AttestationExpiredError(AttestationError):
    """Raised when a certificate in the chain is expired/not-yet-valid, or evidence timestamp is stale."""


class AttestationRevokedError(AttestationCertificateError):
    """Raised when a certificate in the chain matches a revoked serial number or CRL."""


class AttestationSignatureError(AttestationError):
    """Raised when cryptographic signature verification fails on a certificate or attestation token."""


class AttestationNonceMismatchError(AttestationError):
    """Raised when attestation claims nonce does not match consumer challenger nonce (replay attack detected)."""

    def __init__(
        self,
        message: str = "Attestation token challenger nonce mismatch",
        expected_nonce: Optional[str] = None,
        actual_nonce: Optional[str] = None,
        provider: Optional[str] = None,
        task_id: Optional[str] = None,
        **kwargs: Any,
    ):
        details = {"expected_nonce": expected_nonce, "actual_nonce": actual_nonce}
        details.update(kwargs.get("details", {}))
        super().__init__(message, provider=provider, task_id=task_id, details=details, **kwargs)
        self.expected_nonce = expected_nonce
        self.actual_nonce = actual_nonce


class AttestationBindingError(AttestationError):
    """Raised when attestation token claims do not bind to the client's ephemeral public key (MITM detected)."""

    def __init__(
        self,
        message: str = "Attestation token ephemeral public key binding mismatch",
        expected_pubkey_hash: Optional[str] = None,
        actual_pubkey_hash: Optional[str] = None,
        provider: Optional[str] = None,
        task_id: Optional[str] = None,
        **kwargs: Any,
    ):
        details = {
            "expected_pubkey_hash": expected_pubkey_hash,
            "actual_pubkey_hash": actual_pubkey_hash,
        }
        details.update(kwargs.get("details", {}))
        super().__init__(message, provider=provider, task_id=task_id, details=details, **kwargs)
        self.expected_pubkey_hash = expected_pubkey_hash
        self.actual_pubkey_hash = actual_pubkey_hash


class AttestationMeasurementMismatchError(AttestationError):
    """Raised when hardware PCR or firmware measurement digests do not match Golden RIM values."""

    def __init__(
        self,
        message: str = "Hardware measurement digest mismatch against Golden RIM",
        mismatches: Optional[Dict[str, Dict[str, str]]] = None,
        missing_claims: Optional[List[str]] = None,
        provider: Optional[str] = None,
        task_id: Optional[str] = None,
        **kwargs: Any,
    ):
        details = {
            "mismatches": mismatches or {},
            "missing_claims": missing_claims or [],
        }
        details.update(kwargs.get("details", {}))
        super().__init__(message, provider=provider, task_id=task_id, details=details, **kwargs)
        self.mismatches = mismatches or {}
        self.missing_claims = missing_claims or []


class AttestationProviderUnsupportedError(AttestationError):
    """Raised when attestation evidence provider is unknown or unsupported."""

