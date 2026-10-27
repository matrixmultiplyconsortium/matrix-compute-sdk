"""
L402 Payment Client, Macaroon Token Parsing & Budget Controller for matrix-compute-sdk.

Implements L402 (LSAT) challenge parsing, Macaroon base64/JSON token deserialization,
canonical HMAC-SHA256 signature verification, caveat boundary assertions,
thread-safe budget control, and high-level L402PaymentCoordinator.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
import hashlib
import hmac
import json
import logging
import math
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from matrix_compute.config import ClientConfig
from matrix_compute.exceptions import (
    BudgetExceededError,
    L402ChallengeError,
    MacaroonError,
    MacaroonValidationError,
    PaymentError,
    SettlementError,
)
from matrix_compute.payment.escrow import (
    EscrowManager,
    EscrowState,
    EscrowStatus,
    SettlementBackendProtocol,
    generate_escrow_secret,
)
from matrix_compute.test_doubles.mock_settlement import MockSettlementService

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# L402 Challenge Parsing
# -----------------------------------------------------------------------------

CHALLENGE_HEADER_REGEX = re.compile(
    r"^(?:WWW-Authenticate:\s*)?(L402|LSAT)\s+(.*)$",
    re.IGNORECASE,
)

PARAM_REGEX = re.compile(
    r'([a-zA-Z0-9_-]+)\s*=\s*(?:"([^"]*)"|([^,\s]+))',
)


@dataclass(frozen=True)
class L402Challenge:
    """
    Parsed HTTP 402 / gRPC UNAUTHENTICATED WWW-Authenticate challenge.
    Supports both L402 and legacy LSAT schemes with quoted and unquoted parameters.
    """
    scheme: str
    macaroon: str
    invoice: str
    price_sats: Optional[int] = None
    realm: Optional[str] = None
    params: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def parse(cls, header_value: str) -> L402Challenge:
        """
        Parses a 'WWW-Authenticate: L402 macaroon="...", invoice="..."' header string.
        Raises L402ChallengeError if header is malformed or missing macaroon/invoice.
        """
        if not header_value or not isinstance(header_value, str):
            raise L402ChallengeError("Header value cannot be empty or non-string")

        clean = header_value.strip()
        match = CHALLENGE_HEADER_REGEX.match(clean)
        if not match:
            raise L402ChallengeError(
                f"Malformed L402 challenge header: scheme must be L402 or LSAT (got {header_value!r})"
            )

        scheme = match.group(1).upper()
        params_str = match.group(2).strip()

        param_matches = PARAM_REGEX.findall(params_str)
        if not param_matches:
            raise L402ChallengeError(f"No parameters found in L402 challenge: {header_value!r}")

        extracted_params: Dict[str, str] = {}
        for key, quoted_val, unquoted_val in param_matches:
            val = quoted_val if quoted_val != "" else unquoted_val
            extracted_params[key.lower()] = val

        macaroon = extracted_params.get("macaroon")
        invoice = extracted_params.get("invoice")

        if not macaroon or not invoice:
            raise L402ChallengeError(
                f"L402 challenge missing required parameters ('macaroon' and 'invoice'): {header_value!r}"
            )

        price_sats = None
        if "price_sats" in extracted_params:
            try:
                price_sats = int(extracted_params["price_sats"])
            except (ValueError, TypeError):
                pass

        realm = extracted_params.get("realm")

        return cls(
            scheme=scheme,
            macaroon=macaroon,
            invoice=invoice,
            price_sats=price_sats,
            realm=realm,
            params=extracted_params,
        )


# -----------------------------------------------------------------------------
# Macaroon Caveats & Token
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class MacaroonCaveats:
    """First-party caveats bound into an L402 compute Macaroon token."""
    task_id: str
    payment_hash: str
    expiry: int
    max_gflops: float
    locked_sats: int
    worker_pubkey: Optional[str] = None
    task_ids: Optional[List[str]] = None
    job_id: Optional[str] = None
    tier: str = "PAID_MAINNET"
    allowed_workers: Optional[List[str]] = None
    max_matrix_dim: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Converts caveats to canonical dictionary."""
        d: Dict[str, Any] = {
            "expiry": int(self.expiry),
            "locked_sats": int(self.locked_sats),
            "max_gflops": float(self.max_gflops),
            "payment_hash": self.payment_hash,
            "task_id": self.task_id,
        }
        if self.worker_pubkey:
            d["worker_pubkey"] = self.worker_pubkey
        if self.task_ids is not None:
            d["task_ids"] = list(self.task_ids)
        if self.job_id is not None:
            d["job_id"] = self.job_id
        if self.tier and self.tier != "PAID_MAINNET":
            d["tier"] = self.tier
        if self.allowed_workers is not None:
            d["allowed_workers"] = list(self.allowed_workers)
        if self.max_matrix_dim is not None:
            d["max_matrix_dim"] = int(self.max_matrix_dim)
        d.update(self.extra)
        return d


class MacaroonToken:
    """
    L402 Macaroon Token container, deserializer, and caveat verifier.
    """

    def __init__(
        self,
        raw_b64: str,
        caveats: MacaroonCaveats,
        signature: str,
        raw_payload: Dict[str, Any],
    ) -> None:
        self.raw_b64 = raw_b64
        self.caveats = caveats
        self.signature = signature
        self.raw_payload = raw_payload

    @classmethod
    def deserialize(cls, raw_b64: str) -> MacaroonToken:
        """
        Decodes base64 JSON token and extracts caveats and signature.
        Auto-corrects missing padding and supports urlsafe base64.
        """
        if not raw_b64 or not isinstance(raw_b64, str):
            raise MacaroonError("Macaroon token string cannot be empty")

        clean = raw_b64.strip()
        if clean.startswith("mc_trial_"):
            clean = clean[len("mc_trial_") :].strip()
        if not clean:
            raise MacaroonError("Macaroon token string cannot be empty")
        # Add padding if needed
        missing_padding = len(clean) % 4
        if missing_padding:
            clean += "=" * (4 - missing_padding)

        try:
            raw_bytes = base64.urlsafe_b64decode(clean)
            payload = json.loads(raw_bytes.decode("utf-8"))
        except Exception as ex:
            raise MacaroonError(f"Failed to decode base64/JSON macaroon: {ex}") from ex

        if not isinstance(payload, dict):
            raise MacaroonError("Macaroon payload must be a JSON object")

        if "caveats" not in payload and "sig" in payload and isinstance(payload.get("sig"), str):
            raw_caveats = {k: v for k, v in payload.items() if k != "sig"}
            sig = str(payload["sig"])
        else:
            if "caveats" not in payload or "signature" not in payload:
                raise MacaroonError("Macaroon payload missing 'caveats' or 'signature'")
            raw_caveats = payload["caveats"]
            sig = str(payload["signature"])

        if not isinstance(raw_caveats, dict):
            raise MacaroonError("Macaroon 'caveats' must be a JSON dictionary")

        task_id = str(raw_caveats.get("task_id", ""))
        p_hash = str(raw_caveats.get("payment_hash", "0" * 64))
        expiry = int(raw_caveats.get("expiry") if "expiry" in raw_caveats else raw_caveats.get("expires_at", 0))
        max_gflops = float(raw_caveats.get("max_gflops", 138.0))
        locked_sats = int(raw_caveats.get("locked_sats", 0))
        worker_pk = raw_caveats.get("worker_pubkey")
        task_ids_val = raw_caveats.get("task_ids")
        task_ids = list(task_ids_val) if isinstance(task_ids_val, (list, tuple)) else None
        job_id = str(raw_caveats["job_id"]) if "job_id" in raw_caveats and raw_caveats["job_id"] is not None else None
        tier = str(raw_caveats.get("tier", "PAID_MAINNET"))
        allowed_workers_val = raw_caveats.get("allowed_workers")
        allowed_workers = list(allowed_workers_val) if isinstance(allowed_workers_val, (list, tuple)) else None
        max_matrix_dim = int(raw_caveats["max_matrix_dim"]) if "max_matrix_dim" in raw_caveats and raw_caveats["max_matrix_dim"] is not None else None

        extra = {
            k: v
            for k, v in raw_caveats.items()
            if k not in (
                "task_id",
                "payment_hash",
                "expiry",
                "max_gflops",
                "locked_sats",
                "worker_pubkey",
                "task_ids",
                "job_id",
                "tier",
                "allowed_workers",
                "max_matrix_dim",
            )
        }

        caveat_obj = MacaroonCaveats(
            task_id=task_id,
            payment_hash=p_hash,
            expiry=expiry,
            max_gflops=max_gflops,
            locked_sats=locked_sats,
            worker_pubkey=worker_pk,
            task_ids=task_ids,
            job_id=job_id,
            tier=tier,
            allowed_workers=allowed_workers,
            max_matrix_dim=max_matrix_dim,
            extra=extra,
        )

        return cls(
            raw_b64=raw_b64,
            caveats=caveat_obj,
            signature=sig,
            raw_payload=payload,
        )

    def verify_hmac(self, verification_key: str) -> bool:
        """
        Verifies HMAC-SHA256 signature against canonical sorted JSON caveats.
        """
        if not verification_key:
            return False

        caveats_dict = self.caveats.to_dict()

        # Try standard sorted JSON format first (matching mock_settlement.py)
        str_candidate_1 = json.dumps(caveats_dict, sort_keys=True)
        sig_1 = hmac.new(
            verification_key.encode("utf-8"),
            str_candidate_1.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        if hmac.compare_digest(self.signature, sig_1):
            return True

        # Try canonical compact sorted JSON format (separators without space)
        str_candidate_2 = json.dumps(caveats_dict, sort_keys=True, separators=(",", ":"))
        sig_2 = hmac.new(
            verification_key.encode("utf-8"),
            str_candidate_2.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        if hmac.compare_digest(self.signature, sig_2):
            return True

        return False

    def validate(
        self,
        task_id: str,
        m: int,
        k: int,
        n: int,
        expected_payment_hash: Optional[str] = None,
        max_price_rate: Optional[float] = None,
        min_task_sats: int = 10,
        current_time: Optional[float] = None,
        clock_skew_tolerance: float = 5.0,
    ) -> Tuple[bool, Optional[str]]:
        """
        Validates caveat constraints:
          1. task_id matches expected task_id
          2. payment_hash matches expected_payment_hash (if specified)
          3. expiry > current_time - clock_skew_tolerance
          4. max_gflops >= requested_gflops
          5. locked_sats >= required_sats (if max_price_rate provided)
        Returns (is_valid, error_message_or_none).
        """
        # 1. Task ID binding
        allowed_tasks = self.caveats.task_ids or self.caveats.extra.get("task_ids") or ([self.caveats.task_id] if self.caveats.task_id else [])
        if task_id not in allowed_tasks and self.caveats.task_id != task_id:
            return False, f"Caveat task_id mismatch: '{task_id}' not authorized in batch {allowed_tasks}"

        # 2. Payment hash binding
        if expected_payment_hash is not None:
            if self.caveats.payment_hash.lower() != expected_payment_hash.strip().lower():
                return False, "Caveat payment_hash mismatch"

        # 3. Expiration window
        now = current_time if current_time is not None else time.time()
        if self.caveats.expiry <= (now - clock_skew_tolerance):
            return False, f"Macaroon expired at {self.caveats.expiry} (now: {int(now)})"

        # 4. Workload allowance
        requested_gflops = (2.0 * float(m) * float(k) * float(n)) / 1e9
        if self.caveats.max_gflops < requested_gflops:
            return False, f"Insufficient GFLOP allowance: {self.caveats.max_gflops} < {requested_gflops}"

        # 5. Locked satoshis sufficiency
        if max_price_rate is not None:
            required_sats = max(math.ceil(requested_gflops * max_price_rate), min_task_sats)
            if self.caveats.locked_sats < required_sats:
                return False, f"Insufficient locked satoshis: {self.caveats.locked_sats} < {required_sats}"

        return True, None


# -----------------------------------------------------------------------------
# Credential Formatting Helpers
# -----------------------------------------------------------------------------

def format_auth_header(macaroon_b64: str, preimage_or_hash: str, scheme: str = "L402") -> str:
    """
    Formats the standard L402 Authorization header string.
    Example: 'L402 <macaroon_b64>:<payment_hash>' or 'L402 <macaroon_b64>:<preimage>'.
    """
    return f"{scheme} {macaroon_b64.strip()}:{preimage_or_hash.strip()}"


def format_grpc_metadata(
    macaroon_b64: str,
    preimage_or_hash: str,
    task_id: Optional[str] = None,
    scheme: str = "L402",
) -> List[Tuple[str, str]]:
    """
    Formats gRPC metadata tuples for MatrixComputeService tasks.
    Returns: [('authorization', 'L402 ...'), ('x-task-id', task_id)].
    """
    metadata = [("authorization", format_auth_header(macaroon_b64, preimage_or_hash, scheme=scheme))]
    if task_id:
        metadata.append(("x-task-id", task_id))
    return metadata


# -----------------------------------------------------------------------------
# Budget Controller
# -----------------------------------------------------------------------------

class BudgetController:
    """
    Thread-safe budget controller enforcing price rate caps, per-task satoshi limits,
    and cumulative job budget constraints.
    """

    def __init__(
        self,
        max_price_per_gflop: float = 0.001,
        max_budget_sats: Optional[int] = None,
        max_task_sats: Optional[int] = None,
        min_task_sats: int = 10,
    ) -> None:
        self.max_price_per_gflop = float(max_price_per_gflop)
        self.max_budget_sats = int(max_budget_sats) if max_budget_sats is not None else None
        self.max_task_sats = int(max_task_sats) if max_task_sats is not None else None
        self.min_task_sats = int(min_task_sats)

        self._lock = threading.RLock()
        self._reserved: Dict[str, int] = {}    # task_id -> reserved satoshis
        self._committed: Dict[str, int] = {}   # task_id -> settled satoshis

    def check_rate(self, sats_per_gflop: float) -> None:
        """Raises BudgetExceededError if worker rate exceeds max_price_per_gflop."""
        if sats_per_gflop > self.max_price_per_gflop:
            raise BudgetExceededError(
                f"Worker price rate {sats_per_gflop} sats/GFLOP exceeds ceiling {self.max_price_per_gflop}",
            )

    def calculate_cost(self, m: int, k: int, n: int, sats_per_gflop: float) -> int:
        """Calculates satoshis required for GEMM tile computation."""
        gflops = (2.0 * float(m) * float(k) * float(n)) / 1e9
        raw_sats = math.ceil(gflops * sats_per_gflop)
        return max(raw_sats, self.min_task_sats)

    @property
    def total_reserved_sats(self) -> int:
        """Total satoshis currently locked in-flight."""
        with self._lock:
            return sum(self._reserved.values())

    @property
    def total_committed_sats(self) -> int:
        """Total satoshis finalized and paid out."""
        with self._lock:
            return sum(self._committed.values())

    def reserve(self, task_id: str, sats: int) -> None:
        """
        Reserves satoshis for an in-flight compute task under budget limits.
        """
        if sats < 0:
            raise ValueError(f"Cannot reserve negative satoshis: {sats}")

        with self._lock:
            if self.max_task_sats is not None and sats > self.max_task_sats:
                raise BudgetExceededError(
                    f"Task cost {sats} sats exceeds per-task ceiling {self.max_task_sats} sats",
                    requested_sats=sats,
                )

            if self.max_budget_sats is not None:
                current_total = self.total_reserved_sats + self.total_committed_sats
                if current_total + sats > self.max_budget_sats:
                    raise BudgetExceededError(
                        f"Cumulative budget exceeded: {current_total + sats} > {self.max_budget_sats} sats",
                        current_budget_sats=current_total,
                        requested_sats=sats,
                        max_budget_sats=self.max_budget_sats,
                    )

            self._reserved[task_id] = sats

    def reserve_budget(self, task_id: str, sats: int) -> None:
        """Alias for reserve."""
        self.reserve(task_id, sats)

    def commit(self, task_id: str, sats: Optional[int] = None) -> None:
        """
        Finalizes task payout on verification pass, moving funds from reserved to committed.
        """
        with self._lock:
            reserved_amt = self._reserved.pop(task_id, 0)
            actual_amt = sats if sats is not None else reserved_amt
            self._committed[task_id] = actual_amt

    def commit_budget(self, task_id: str) -> None:
        """Alias for commit."""
        self.commit(task_id)

    def release(self, task_id: str, sats: Optional[int] = None) -> None:
        """
        Releases reserved satoshis back to available budget on task abort, failure, or fraud.
        """
        with self._lock:
            self._reserved.pop(task_id, None)

    def refund_budget(self, task_id: str) -> None:
        """Alias for release."""
        self.release(task_id)

    def get_summary(self) -> Dict[str, Any]:
        """Returns budget accounting summary."""
        with self._lock:
            return {
                "max_price_per_gflop": self.max_price_per_gflop,
                "max_budget_sats": self.max_budget_sats,
                "max_task_sats": self.max_task_sats,
                "total_reserved_sats": self.total_reserved_sats,
                "total_committed_sats": self.total_committed_sats,
                "active_tasks_count": len(self._reserved),
                "committed_tasks_count": len(self._committed),
            }


# -----------------------------------------------------------------------------
# Settlement Backend Implementations
# -----------------------------------------------------------------------------

class MockSettlementBackend:
    """Settlement backend wrapper targeting in-memory MockSettlementService."""

    def __init__(
        self,
        service: Optional[MockSettlementService] = None,
        verification_key: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        if service is not None:
            self.service = service
        elif verification_key is not None:
            self.service = MockSettlementService(verification_key=verification_key)
        else:
            self.service = MockSettlementService()
        self.verification_key = verification_key or getattr(self.service, "verification_key", None)

    def create_hodl_invoice(
        self,
        task_id: str,
        total_compute_sats: int,
        worker_lightning_address: str = "worker@node",
        payment_hash: Optional[str] = None,
        max_gflops: float = 100.0,
        expiry_seconds: int = 3600,
        **kwargs: Any,
    ) -> Any:
        return self.service.create_hodl_invoice(
            task_id_or_request=task_id,
            total_compute_sats=total_compute_sats,
            worker_lightning_address=worker_lightning_address,
            payment_hash=payment_hash,
            expiry_seconds=expiry_seconds,
            max_gflops=max_gflops,
            **kwargs,
        )

    def hold_invoice(self, payment_hash: str) -> bool:
        return self.service.hold_invoice(payment_hash)

    def settle_invoice(self, payment_hash: str, preimage: str) -> bool:
        return self.service.settle_invoice(payment_hash, preimage)

    def cancel_invoice(self, payment_hash: str) -> bool:
        return self.service.cancel_invoice(payment_hash)

    def get_status(self, payment_hash: str) -> Optional[str]:
        return self.service.get_status(payment_hash)

    def create_batch_escrow(
        self,
        task_ids: List[str],
        total_compute_sats: int,
        worker_lightning_address: str = "worker@node",
        payment_hash: Optional[str] = None,
        expiry_seconds: int = 3600,
        max_gflops: float = 100.0,
        job_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        if hasattr(self.service, "create_batch_escrow"):
            return self.service.create_batch_escrow(
                task_ids=task_ids,
                total_compute_sats=total_compute_sats,
                worker_lightning_address=worker_lightning_address,
                payment_hash=payment_hash,
                expiry_seconds=expiry_seconds,
                max_gflops=max_gflops,
                job_id=job_id,
                **kwargs,
            )
        return self.service.create_hodl_invoice(
            task_ids=task_ids,
            total_compute_sats=total_compute_sats,
            worker_lightning_address=worker_lightning_address,
            payment_hash=payment_hash,
            expiry_seconds=expiry_seconds,
            max_gflops=max_gflops,
            job_id=job_id,
            **kwargs,
        )

    def settle_batch(self, job_id_or_hash: str, preimage: Optional[str] = None, **kwargs: Any) -> bool:
        if hasattr(self.service, "settle_batch"):
            return self.service.settle_batch(job_id_or_hash, preimage=preimage, **kwargs)
        return self.service.settle_invoice(job_id_or_hash, preimage or "")

    def cancel_batch(self, job_id_or_hash: str, **kwargs: Any) -> bool:
        if hasattr(self.service, "cancel_batch"):
            return self.service.cancel_batch(job_id_or_hash, **kwargs)
        return self.service.cancel_invoice(job_id_or_hash)

    def submit_fraud_dispute(
        self, proof: Any, slash_amount_sats: Optional[int] = None, **kwargs: Any
    ) -> Any:
        if hasattr(self.service, "submit_fraud_dispute"):
            return self.service.submit_fraud_dispute(
                proof=proof, slash_amount_sats=slash_amount_sats, **kwargs
            )
        return None


class HttpSettlementBackend:
    """HTTP REST settlement client for production settlement servers."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        timeout: float = 10.0,
        timeout_seconds: Optional[float] = None,
        verification_key: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout_seconds if timeout_seconds is not None else timeout)
        self.verification_key = verification_key

    def create_hodl_invoice(
        self,
        task_id: str,
        total_compute_sats: int,
        worker_lightning_address: str = "worker@node",
        payment_hash: Optional[str] = None,
        max_gflops: float = 100.0,
        expiry_seconds: int = 3600,
        **kwargs: Any,
    ) -> Any:
        try:
            import httpx
            with httpx.Client(base_url=self.base_url, timeout=self.timeout) as client:
                payload = {
                    "task_id": task_id,
                    "total_compute_sats": total_compute_sats,
                    "worker_lightning_address": worker_lightning_address,
                    "payment_hash": payment_hash,
                    "max_gflops": max_gflops,
                    "expiry_seconds": expiry_seconds,
                }
                if "task_ids" in kwargs and kwargs["task_ids"]:
                    payload["task_ids"] = kwargs["task_ids"]
                if "job_id" in kwargs and kwargs["job_id"]:
                    payload["job_id"] = kwargs["job_id"]
                if "worker_pubkey" in kwargs and kwargs["worker_pubkey"]:
                    payload["worker_pubkey"] = kwargs["worker_pubkey"]

                resp = client.post("/invoices", json=payload)
                if resp.status_code == 404:
                    resp = client.post("/invoices/hodl", json=payload)
                resp.raise_for_status()
                data = resp.json()

                class _HttpInvoiceResp:
                    payment_hash = data.get("payment_hash")
                    payment_request = data.get("payment_request")
                    macaroon = data.get("macaroon")
                    platform_fee_sats = data.get("platform_fee_sats", 0)
                    worker_payout_sats = data.get("worker_payout_sats", 0)
                    preimage = data.get("preimage")

                return _HttpInvoiceResp()
        except Exception as ex:
            raise SettlementError(f"HTTP settlement create invoice failed: {ex}") from ex

    def create_batch_escrow(
        self,
        task_ids: List[str],
        total_compute_sats: int,
        worker_lightning_address: str = "worker@node",
        payment_hash: Optional[str] = None,
        max_gflops: float = 100.0,
        expiry_seconds: int = 3600,
        job_id: Optional[str] = None,
        worker_pubkey: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        return self.create_hodl_invoice(
            task_id=task_ids[0] if task_ids else "",
            total_compute_sats=total_compute_sats,
            worker_lightning_address=worker_lightning_address,
            payment_hash=payment_hash,
            max_gflops=max_gflops,
            expiry_seconds=expiry_seconds,
            task_ids=task_ids,
            job_id=job_id,
            worker_pubkey=worker_pubkey,
            **kwargs,
        )

    def hold_invoice(self, payment_hash: str) -> bool:
        try:
            import httpx
            with httpx.Client(base_url=self.base_url, timeout=self.timeout) as client:
                resp = client.post(f"/invoices/{payment_hash}/hold")
                resp.raise_for_status()
                return True
        except Exception as ex:
            raise SettlementError(f"HTTP settlement hold invoice failed: {ex}") from ex

    def settle_invoice(self, payment_hash: str, preimage: str) -> bool:
        try:
            import httpx
            with httpx.Client(base_url=self.base_url, timeout=self.timeout) as client:
                resp = client.post(f"/invoices/{payment_hash}/settle", json={"preimage": preimage})
                resp.raise_for_status()
                return True
        except Exception as ex:
            raise SettlementError(f"HTTP settlement settle invoice failed: {ex}") from ex

    def settle_batch(self, job_id_or_hash: str, preimage: Optional[str] = None, **kwargs: Any) -> bool:
        try:
            import httpx
            with httpx.Client(base_url=self.base_url, timeout=self.timeout) as client:
                payload = {"job_id": job_id_or_hash, "preimage": preimage}
                resp = client.post("/settle", json=payload)
                if resp.status_code == 404:
                    resp = client.post(f"/invoices/{job_id_or_hash}/settle", json={"preimage": preimage})
                resp.raise_for_status()
                return True
        except Exception as ex:
            raise SettlementError(f"HTTP settlement settle batch failed: {ex}") from ex

    def cancel_invoice(self, payment_hash: str) -> bool:
        try:
            import httpx
            with httpx.Client(base_url=self.base_url, timeout=self.timeout) as client:
                resp = client.post(f"/invoices/{payment_hash}/cancel")
                resp.raise_for_status()
                return True
        except Exception as ex:
            raise SettlementError(f"HTTP settlement cancel invoice failed: {ex}") from ex

    def cancel_batch(self, job_id_or_hash: str, **kwargs: Any) -> bool:
        return self.cancel_invoice(job_id_or_hash)

    def submit_fraud_dispute(
        self,
        proof: Any,
        slash_amount_sats: Optional[int] = None,
        consumer_signature: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        try:
            import httpx
            proof_dict = (
                proof.model_dump()
                if hasattr(proof, "model_dump")
                else (proof.dict() if hasattr(proof, "dict") else proof)
            )
            payload = {
                "proof": proof_dict,
                "slash_amount_sats": slash_amount_sats,
                "consumer_signature": consumer_signature,
            }
            with httpx.Client(base_url=self.base_url, timeout=self.timeout) as client:
                resp = client.post("/disputes/slash", json=payload)
                resp.raise_for_status()
                return resp.json()
        except Exception as ex:
            raise SettlementError(f"HTTP settlement submit fraud dispute failed: {ex}") from ex

    def get_status(self, payment_hash: str) -> Optional[str]:
        try:
            import httpx
            with httpx.Client(base_url=self.base_url, timeout=self.timeout) as client:
                resp = client.get(f"/invoices/{payment_hash}")
                if resp.status_code == 404:
                    return None
                resp.raise_for_status()
                return resp.json().get("status")
        except Exception:
            return None


# -----------------------------------------------------------------------------
# L402 Payment Coordinator
# -----------------------------------------------------------------------------

class L402PaymentCoordinator:
    """
    High-level payment coordinator:
      - Interfaces between EscrowManager, BudgetController, and SettlementBackend.
      - Generates L402 gRPC metadata headers for MatrixComputeService.
      - Coordinates pre-allocation locking and post-verification settlement.
    """

    def __init__(
        self,
        config: Optional[ClientConfig] = None,
        settlement_backend: Optional[SettlementBackendProtocol] = None,
        budget_controller: Optional[BudgetController] = None,
        escrow_manager: Optional[EscrowManager] = None,
    ) -> None:
        self.config = config
        self.settlement_backend = (
            settlement_backend
            if settlement_backend is not None
            else MockSettlementBackend(MockSettlementService())
        )
        self.budget_controller = (
            budget_controller
            if budget_controller is not None
            else BudgetController(
                max_price_per_gflop=getattr(config, "max_price_per_gflop", 0.001) if config else 0.001,
                max_budget_sats=getattr(config, "max_budget_sats", None) if config else None,
            )
        )
        self.escrow_manager = (
            escrow_manager
            if escrow_manager is not None
            else EscrowManager(
                settlement_backend=self.settlement_backend,
                config=config,
            )
        )

    def create_escrow(
        self,
        task_id: str,
        total_sats: int,
        worker_lightning_addr: str = "worker@node",
        max_gflops: float = 100.0,
        expiry_seconds: int = 3600,
        preimage_hex: Optional[str] = None,
    ) -> EscrowState:
        """
        1. Reserves budget in BudgetController.
        2. Creates HODL invoice in EscrowManager.
        3. Locks invoice to HELD status.
        4. Returns EscrowState.
        """
        self.budget_controller.reserve(task_id, total_sats)
        try:
            escrow = self.escrow_manager.create_escrow(
                task_id=task_id,
                total_sats=total_sats,
                worker_lightning_addr=worker_lightning_addr,
                timeout_seconds=float(expiry_seconds),
                max_gflops=max_gflops,
                custom_preimage=preimage_hex,
            )
            self.escrow_manager.hold_escrow(escrow.task_id)
            return escrow
        except Exception:
            self.budget_controller.release(task_id)
            raise

    def create_batch_escrow(
        self,
        worker_pubkey: str,
        task_ids: List[str],
        job_id: str,
        total_sats: int,
        worker_lightning_addr: str = "worker@node",
        max_gflops: float = 100.0,
        expiry_seconds: int = 3600,
        preimage_hex: Optional[str] = None,
    ) -> EscrowState:
        """
        1. Reserves cumulative budget in BudgetController for the job.
        2. Creates batch escrow covering task_ids.
        3. Locks escrow into HELD status.
        4. Returns EscrowState.
        """
        self.budget_controller.reserve(job_id, total_sats)
        try:
            escrow = self.escrow_manager.create_batch_escrow(
                worker_pubkey=worker_pubkey,
                task_ids=task_ids,
                job_id=job_id,
                total_sats=total_sats,
                worker_lightning_addr=worker_lightning_addr,
                timeout_seconds=float(expiry_seconds),
                max_gflops=max_gflops,
                custom_preimage=preimage_hex,
            )
            self.escrow_manager.hold_escrow(escrow.payment_hash)
            return escrow
        except Exception:
            self.budget_controller.release(job_id)
            raise

    def format_auth_header(
        self,
        escrow_or_macaroon: Union[EscrowState, str],
        preimage_or_hash: Optional[str] = None,
        use_preimage: bool = False,
    ) -> str:
        """
        Returns 'L402 <macaroon>:<proof>' where proof is payment_hash (pre-settlement)
        or secret preimage (post-settlement).
        """
        if isinstance(escrow_or_macaroon, EscrowState):
            return escrow_or_macaroon.get_auth_header(use_preimage=use_preimage)
        macaroon_str = str(escrow_or_macaroon)
        proof = str(preimage_or_hash or "")
        return format_auth_header(macaroon_str, proof)

    def get_grpc_metadata(
        self,
        escrow: EscrowState,
        use_preimage: bool = False,
    ) -> List[Tuple[str, str]]:
        """
        Generates gRPC metadata list for ExecuteTileTask:
        [('authorization', 'L402 <macaroon>:<proof>'), ('x-task-id', task_id)].
        """
        header_val = escrow.get_auth_header(use_preimage=use_preimage)
        return [
            ("authorization", header_val),
            ("x-task-id", escrow.task_id),
        ]

    def release_payout(self, escrow_or_task_id: Union[EscrowState, str]) -> bool:
        """
        Reveals secret preimage to finalize payout upon verification pass.
        Commits satoshis in BudgetController.
        """
        tid = escrow_or_task_id.task_id if isinstance(escrow_or_task_id, EscrowState) else str(escrow_or_task_id)
        self.escrow_manager.release_payout(tid)
        self.budget_controller.commit(tid)
        return True

    def settle_batch(self, escrow_or_job_id: Union[EscrowState, str], preimage: Optional[str] = None) -> bool:
        """
        Atomic settlement for an entire worker batch.
        Commits budget in BudgetController.
        """
        job_id = (
            (escrow_or_job_id.job_id or escrow_or_job_id.task_id)
            if isinstance(escrow_or_job_id, EscrowState)
            else str(escrow_or_job_id)
        )
        self.escrow_manager.settle_batch(job_id, preimage=preimage)
        self.budget_controller.commit(job_id)
        return True

    def abort_and_refund(self, escrow_or_task_id: Union[EscrowState, str], reason: str = "") -> bool:
        """
        Cancels invoice upon fraud, failure, or timeout. Releases satoshis in BudgetController.
        """
        tid = escrow_or_task_id.task_id if isinstance(escrow_or_task_id, EscrowState) else str(escrow_or_task_id)
        self.escrow_manager.abort_and_refund(tid, reason=reason)
        self.budget_controller.release(tid)
        return True


class L402PaymentClient:
    """
    Client for minting, managing, and verifying L402 payment credentials and macaroons.
    """

    def __init__(
        self,
        settlement_backend: Optional[Any] = None,
        macaroon_caveat_secret: Optional[str] = None,
        verification_key: Optional[str] = None,
    ) -> None:
        self.settlement_backend = settlement_backend
        self.macaroon_caveat_secret = (
            macaroon_caveat_secret
            or verification_key
            or (getattr(settlement_backend, "verification_key", None) if settlement_backend else None)
            or "settlement_secret_key_dev"
        )

    def mint_macaroon(
        self,
        task_id: str,
        payment_hash: str,
        m: int = 2,
        k: int = 2,
        n: int = 2,
        locked_sats: int = 100,
        max_gflops: Optional[float] = None,
        expiry: Optional[int] = None,
        task_ids: Optional[List[str]] = None,
    ) -> str:
        """
        Mints a valid HMAC-SHA256 signed L402 Macaroon token for a matrix compute workload.
        """
        calc_gflops = (2.0 * m * k * n) / 1e9
        effective_gflops = max_gflops if max_gflops is not None else max(100.0, calc_gflops * 10.0)
        exp = int(expiry if expiry is not None else (time.time() + 3600))

        caveats: Dict[str, Any] = {
            "expiry": exp,
            "locked_sats": locked_sats,
            "max_gflops": float(effective_gflops),
            "payment_hash": payment_hash,
            "task_id": task_id,
        }
        if task_ids is not None:
            caveats["task_ids"] = list(task_ids)

        caveat_str = json.dumps(caveats, sort_keys=True)
        sig = hmac.new(
            self.macaroon_caveat_secret.encode("utf-8"),
            caveat_str.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        payload = {"caveats": caveats, "signature": sig}
        return base64.b64encode(json.dumps(payload).encode("utf-8")).decode("utf-8")

