"""In-memory Mock Settlement Service test double for offline L402 testing in matrix-compute-sdk."""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import hmac
import json
import secrets
import threading
import time
from typing import Any, Dict, Optional, Tuple, Union

try:
    from matrix_contracts.schemas.settlement import (
        CreateHodlInvoiceRequest,
        CreateHodlInvoiceResponse,
        SettlePaymentRequest,
    )
except ImportError:
    from matrix_contracts.schemas import (
        CreateHodlInvoiceRequest,
        CreateHodlInvoiceResponse,
        SettlePaymentRequest,
    )


class InvoiceStatus(str, Enum):
    UNPAID = "UNPAID"
    HELD = "HELD"
    SETTLED = "SETTLED"
    CANCELLED = "CANCELLED"


class MockInvoiceResponse(CreateHodlInvoiceResponse):
    preimage: Optional[str] = None

    def __getitem__(self, item: str) -> Any:
        try:
            return getattr(self, item)
        except AttributeError:
            raise KeyError(item)

    def get(self, item: str, default: Any = None) -> Any:
        return getattr(self, item, default)

    def __contains__(self, item: str) -> bool:
        return hasattr(self, item)


@dataclass
class InvoiceRecord:
    task_id: str
    payment_hash: str
    total_compute_sats: int
    platform_fee_percent: float
    platform_fee_sats: int
    worker_payout_sats: int
    worker_lightning_address: str
    payment_request: str
    macaroon: str
    status: InvoiceStatus = InvoiceStatus.UNPAID
    preimage: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    expires_at: float = 0.0
    max_gflops: float = 100.0
    task_ids: Optional[List[str]] = None
    job_id: Optional[str] = None
    worker_pubkey: Optional[str] = None


class MockSettlementService:
    """
    Hermetic in-memory Settlement Service and L402 Macaroon issuer.
    Simulates Lightning HODL invoice lifecycle: UNPAID -> HELD -> SETTLED / CANCELLED.
    """

    _instances: List["MockSettlementService"] = []

    def __init__(self, verification_key: str = "settlement-secret-verification-key-12345") -> None:
        self.verification_key = verification_key
        self._lock = threading.RLock()
        self._invoices: Dict[str, InvoiceRecord] = {}
        self._task_index: Dict[str, str] = {}
        self._job_index: Dict[str, str] = {}
        self._bonds: Dict[str, Dict[str, Any]] = {}
        self.total_settled_sats: int = 0
        self.total_platform_fees: int = 0
        self.total_worker_payouts: int = 0
        self.settlement_calls_count: int = 0
        MockSettlementService._instances.append(self)

    def _mint_macaroon(
        self,
        payment_hash: str,
        task_id: str,
        expires_at: int,
        max_gflops: float,
        locked_sats: int = 100,
        task_ids: Optional[List[str]] = None,
        job_id: Optional[str] = None,
        worker_pubkey: Optional[str] = None,
    ) -> str:
        caveats: Dict[str, Any] = {
            "expiry": int(expires_at),
            "locked_sats": locked_sats,
            "max_gflops": float(max_gflops),
            "payment_hash": payment_hash,
            "task_id": task_id,
        }
        if task_ids is not None:
            caveats["task_ids"] = list(task_ids)
        if job_id is not None:
            caveats["job_id"] = str(job_id)
        if worker_pubkey is not None:
            caveats["worker_pubkey"] = str(worker_pubkey)

        caveat_str = json.dumps(caveats, sort_keys=True)
        sig = hmac.new(
            self.verification_key.encode("utf-8"),
            caveat_str.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        payload = {"caveats": caveats, "signature": sig}
        return base64.b64encode(json.dumps(payload).encode("utf-8")).decode("utf-8")

    def verify_macaroon(
        self,
        macaroon_raw: str,
        expected_task_id: str,
        required_gflops: float,
        current_time: Optional[float] = None,
    ) -> Tuple[bool, str]:
        try:
            raw_bytes = base64.b64decode(macaroon_raw)
            payload = json.loads(raw_bytes.decode("utf-8"))
            if not isinstance(payload, dict):
                return False, "Malformed macaroon token: payload must be a JSON object"
            if "caveats" not in payload or "signature" not in payload:
                return False, "Malformed macaroon token: missing caveats or signature"
            caveats = payload["caveats"]
            signature = payload["signature"]
            if not isinstance(signature, str):
                return False, "Malformed macaroon signature"
            if not isinstance(caveats, dict):
                return False, "Malformed macaroon token: caveats must be a JSON object"

            caveat_str = json.dumps(caveats, sort_keys=True)
            expected_sig = hmac.new(
                self.verification_key.encode("utf-8"),
                caveat_str.encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()

            if not hmac.compare_digest(signature, expected_sig):
                return False, "Invalid macaroon HMAC signature"

            allowed_tasks = caveats.get("task_ids") or [caveats.get("task_id")]
            if expected_task_id not in allowed_tasks:
                return False, f"Caveat task_id mismatch: task '{expected_task_id}' not in authorized batch {allowed_tasks}"

            now = current_time if current_time is not None else time.time()
            expiry = caveats.get("expiry", 0)
            if not isinstance(expiry, (int, float)) or expiry <= now:
                return False, f"Macaroon expired at {expiry} (current: {int(now)})"

            max_gflops = caveats.get("max_gflops", 0.0)
            if not isinstance(max_gflops, (int, float)) or max_gflops < required_gflops:
                return False, f"Insufficient GFLOP allowance: {max_gflops} < {required_gflops}"

            return True, "OK"
        except Exception as ex:
            return False, f"Malformed macaroon token: {ex}"

    def create_hodl_invoice(
        self,
        task_id_or_request: Union[str, CreateHodlInvoiceRequest] = "",
        total_compute_sats: Optional[int] = None,
        platform_fee_percent: float = 0.02,
        worker_lightning_address: str = "worker@testnode.local",
        payment_hash: Optional[str] = None,
        expiry_seconds: int = 3600,
        max_gflops: float = 100.0,
        **kwargs: Any,
    ) -> MockInvoiceResponse:
        task_ids = getattr(task_id_or_request, "task_ids", None) or kwargs.get("task_ids")
        job_id = getattr(task_id_or_request, "job_id", None) or kwargs.get("job_id")
        worker_pk = getattr(task_id_or_request, "worker_pubkey", None) or kwargs.get("worker_pubkey")

        if isinstance(task_id_or_request, CreateHodlInvoiceRequest):
            req = task_id_or_request
            task_id = req.task_id
            compute_sats = req.total_compute_sats
            fee_pct = req.platform_fee_percent
            worker_ln = req.worker_lightning_address
            p_hash = req.payment_hash
        else:
            task_id = str(task_id_or_request or kwargs.get("task_id", ""))
            compute_sats = total_compute_sats if total_compute_sats is not None else kwargs.get("total_compute_sats", 100)
            fee_pct = kwargs.get("platform_fee_percent", platform_fee_percent)
            worker_ln = kwargs.get("worker_lightning_address", worker_lightning_address)
            p_hash = payment_hash if payment_hash is not None else kwargs.get("payment_hash")

        if task_ids:
            clean_task_ids = [str(t).strip() for t in task_ids if str(t).strip()]
            if not task_id and clean_task_ids:
                task_id = clean_task_ids[0]
        else:
            clean_task_ids = [task_id] if task_id else []

        if p_hash:
            final_hash = p_hash.strip().lower()
            preimage = None
        else:
            preimage_bytes = secrets.token_bytes(32)
            preimage = preimage_bytes.hex()
            final_hash = hashlib.sha256(preimage_bytes).hexdigest()

        with self._lock:
            for tid in clean_task_ids:
                if tid in self._task_index:
                    raise ValueError(f"Task ID already registered: '{tid}'")
            if final_hash in self._invoices:
                raise ValueError(f"Duplicate payment hash '{final_hash}' already exists")

            platform_fee_sats = int(round(compute_sats * fee_pct))
            worker_payout_sats = compute_sats - platform_fee_sats
            now = time.time()
            expires_at = int(now + expiry_seconds)

            macaroon_str = self._mint_macaroon(
                payment_hash=final_hash,
                task_id=task_id,
                expires_at=expires_at,
                max_gflops=max_gflops,
                locked_sats=compute_sats,
                task_ids=clean_task_ids if task_ids is not None else None,
                job_id=job_id,
                worker_pubkey=worker_pk,
            )
            payment_request = f"lnbc{compute_sats}000n1p{final_hash[:20]}mockhodl"

            record = InvoiceRecord(
                task_id=task_id,
                payment_hash=final_hash,
                total_compute_sats=compute_sats,
                platform_fee_percent=fee_pct,
                platform_fee_sats=platform_fee_sats,
                worker_payout_sats=worker_payout_sats,
                worker_lightning_address=worker_ln,
                payment_request=payment_request,
                macaroon=macaroon_str,
                status=InvoiceStatus.UNPAID,
                preimage=preimage,
                created_at=now,
                expires_at=float(expires_at),
                max_gflops=max_gflops,
                task_ids=clean_task_ids if task_ids is not None else None,
                job_id=job_id,
                worker_pubkey=worker_pk,
            )
            self._invoices[final_hash] = record
            for tid in clean_task_ids:
                self._task_index[tid] = final_hash
            if job_id:
                self._job_index[job_id] = final_hash

        return MockInvoiceResponse(
            payment_hash=final_hash,
            payment_request=payment_request,
            platform_fee_sats=platform_fee_sats,
            worker_payout_sats=worker_payout_sats,
            macaroon=macaroon_str,
            preimage=preimage,
        )

    def hold_invoice(self, payment_hash: str) -> bool:
        clean_hash = str(payment_hash).strip().lower()
        with self._lock:
            if clean_hash not in self._invoices:
                raise KeyError(f"Payment hash '{payment_hash}' not found")
            record = self._invoices[clean_hash]
            if record.status != InvoiceStatus.UNPAID:
                raise ValueError(f"Cannot hold invoice in status '{record.status.value}'; must be UNPAID")
            record.status = InvoiceStatus.HELD
            return True

    def is_held(self, payment_hash: str) -> bool:
        clean_hash = str(payment_hash).strip().lower()
        with self._lock:
            record = self._invoices.get(clean_hash)
            if record is not None:
                return record.status == InvoiceStatus.HELD
        for inst in list(MockSettlementService._instances):
            if inst is not self:
                try:
                    with inst._lock:
                        record = inst._invoices.get(clean_hash)
                        if record is not None:
                            return record.status == InvoiceStatus.HELD
                except Exception:
                    pass
        return False

    def get_status(self, payment_hash: str) -> Optional[str]:
        clean_hash = str(payment_hash).strip().lower()
        with self._lock:
            record = self._invoices.get(clean_hash)
            if record:
                return record.status.value
        for inst in list(MockSettlementService._instances):
            if inst is not self:
                try:
                    with inst._lock:
                        record = inst._invoices.get(clean_hash)
                        if record:
                            return record.status.value
                except Exception:
                    pass
        return None

    def get_invoice_status(self, payment_hash: str) -> Optional[str]:
        return self.get_status(payment_hash)

    def _invoice_record_to_dict(self, record: InvoiceRecord) -> Dict[str, Any]:
        return {
            "payment_hash": record.payment_hash,
            "preimage": record.preimage,
            "task_id": record.task_id,
            "primary_task_id": record.task_id,
            "task_ids": list(record.task_ids) if record.task_ids else [record.task_id],
            "job_id": record.job_id,
            "total_sats": record.total_compute_sats,
            "amount_sats": record.total_compute_sats,
            "total_compute_sats": record.total_compute_sats,
            "platform_fee_percent": record.platform_fee_percent,
            "platform_fee_sats": record.platform_fee_sats,
            "worker_payout_sats": record.worker_payout_sats,
            "worker_lightning_address": record.worker_lightning_address,
            "worker_address": record.worker_lightning_address,
            "payment_request": record.payment_request,
            "macaroon": record.macaroon,
            "status": record.status.value,
            "max_gflops": record.max_gflops,
            "created_at": record.created_at,
            "expires_at": record.expires_at,
            "expiry": int(record.expires_at),
        }

    def get_invoice(self, payment_hash_or_id: str) -> Optional[Dict[str, Any]]:
        """Queries full invoice record by payment hash, task ID, or job ID."""
        clean = str(payment_hash_or_id).strip()
        clean_lower = clean.lower()
        with self._lock:
            p_hash = None
            if clean_lower in self._invoices:
                p_hash = clean_lower
            elif clean in self._invoices:
                p_hash = clean
            elif clean in self._task_index:
                p_hash = self._task_index[clean]
            elif clean_lower in self._task_index:
                p_hash = self._task_index[clean_lower]
            elif clean in self._job_index:
                p_hash = self._job_index[clean]
            elif clean_lower in self._job_index:
                p_hash = self._job_index[clean_lower]
            if p_hash and p_hash in self._invoices:
                return self._invoice_record_to_dict(self._invoices[p_hash])
        return None

    def lookup_by_task_id(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Multi-index lookup by constituent task_id."""
        clean = str(task_id).strip()
        with self._lock:
            p_hash = self._task_index.get(clean) or self._task_index.get(clean.lower())
            if p_hash and p_hash in self._invoices:
                return self._invoice_record_to_dict(self._invoices[p_hash])
        return None

    def lookup_by_job_id(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Multi-index lookup by job_id."""
        clean = str(job_id).strip()
        with self._lock:
            p_hash = self._job_index.get(clean) or self._job_index.get(clean.lower())
            if p_hash and p_hash in self._invoices:
                return self._invoice_record_to_dict(self._invoices[p_hash])
        return None

    def settle_invoice(
        self,
        hash_or_request: Union[str, SettlePaymentRequest],
        preimage: Optional[str] = None,
    ) -> bool:
        with self._lock:
            if isinstance(hash_or_request, SettlePaymentRequest):
                task_id = hash_or_request.task_id
                clean_hash = self._task_index.get(task_id) or self._job_index.get(task_id)
                if not clean_hash or clean_hash not in self._invoices:
                    raise KeyError(f"No invoice found for task ID '{task_id}'")
                clean_preimage = hash_or_request.preimage.strip().lower()
            else:
                first_arg = str(hash_or_request).strip().lower()
                first_arg_orig = str(hash_or_request).strip()
                if first_arg in self._invoices:
                    clean_hash = first_arg
                elif first_arg_orig in self._invoices:
                    clean_hash = first_arg_orig
                elif first_arg_orig in self._task_index:
                    clean_hash = self._task_index[first_arg_orig]
                elif first_arg in self._task_index:
                    clean_hash = self._task_index[first_arg]
                elif first_arg_orig in self._job_index:
                    clean_hash = self._job_index[first_arg_orig]
                elif first_arg in self._job_index:
                    clean_hash = self._job_index[first_arg]
                else:
                    raise KeyError(f"Payment hash, task ID, or job ID '{hash_or_request}' not found")
                clean_preimage = str(preimage).strip().lower() if preimage is not None else None

            record = self._invoices[clean_hash]
            if record.status != InvoiceStatus.HELD:
                raise ValueError(f"Cannot settle invoice in status '{record.status.value}'; must be HELD")

            if clean_preimage is None:
                if record.preimage:
                    clean_preimage = record.preimage
                else:
                    raise ValueError("Preimage is required to settle invoice")

            try:
                preimage_bytes = bytes.fromhex(clean_preimage)
            except Exception:
                raise ValueError("Preimage is not a valid hexadecimal string")

            computed_hash = hashlib.sha256(preimage_bytes).hexdigest()
            if computed_hash != record.payment_hash:
                raise ValueError(f"Preimage hash {computed_hash} does not match invoice hash {record.payment_hash}")

            record.preimage = clean_preimage
            record.status = InvoiceStatus.SETTLED
            self.settlement_calls_count += 1
            self.total_settled_sats += record.total_compute_sats
            self.total_platform_fees += record.platform_fee_sats
            self.total_worker_payouts += record.worker_payout_sats
            return True

    def create_batch_escrow(
        self,
        task_ids: List[str],
        total_compute_sats: int,
        worker_lightning_address: str = "worker@node",
        payment_hash: Optional[str] = None,
        expiry_seconds: int = 3600,
        max_gflops: float = 100.0,
        job_id: Optional[str] = None,
        worker_pubkey: Optional[str] = None,
        **kwargs: Any,
    ) -> MockInvoiceResponse:
        """Convenience method delegating to create_hodl_invoice for batch escrows."""
        return self.create_hodl_invoice(
            task_id=task_ids[0] if task_ids else "",
            total_compute_sats=total_compute_sats,
            worker_lightning_address=worker_lightning_address,
            payment_hash=payment_hash,
            expiry_seconds=expiry_seconds,
            max_gflops=max_gflops,
            task_ids=task_ids,
            job_id=job_id,
            worker_pubkey=worker_pubkey,
            **kwargs,
        )

    def settle_batch(
        self,
        job_id_or_hash: str,
        preimage: Optional[str] = None,
        **kwargs: Any,
    ) -> bool:
        """Settles an aggregate batch invoice via job_id, payment_hash, or task_id."""
        return self.settle_invoice(job_id_or_hash, preimage=preimage)

    def cancel_invoice(self, payment_hash: str) -> bool:
        clean_hash = str(payment_hash).strip().lower()
        with self._lock:
            if clean_hash not in self._invoices:
                raise KeyError(f"Payment hash '{payment_hash}' not found")
            record = self._invoices[clean_hash]
            if record.status == InvoiceStatus.SETTLED:
                raise ValueError("Cannot cancel an already SETTLED invoice")
            record.status = InvoiceStatus.CANCELLED
            return True

    def cancel_batch(self, job_id_or_hash: str, reason: str = "batch_cancelled", **kwargs: Any) -> bool:
        """Cancels an aggregate batch invoice via job_id, payment_hash, or task_id."""
        clean = str(job_id_or_hash).strip()
        clean_lower = clean.lower()
        with self._lock:
            p_hash = None
            if clean_lower in self._invoices:
                p_hash = clean_lower
            elif clean in self._invoices:
                p_hash = clean
            elif clean in self._task_index:
                p_hash = self._task_index[clean]
            elif clean_lower in self._task_index:
                p_hash = self._task_index[clean_lower]
            elif clean in self._job_index:
                p_hash = self._job_index[clean]
            elif clean_lower in self._job_index:
                p_hash = self._job_index[clean_lower]

            if not p_hash or p_hash not in self._invoices:
                raise KeyError(f"Payment hash, task ID, or job ID '{job_id_or_hash}' not found")
            return self.cancel_invoice(p_hash)

    def stake_bond(
        self,
        worker_pubkey: str,
        amount_sats: int,
        bond_txid: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Registers or deposits worker collateral stake."""
        with self._lock:
            if worker_pubkey not in self._bonds:
                self._bonds[worker_pubkey] = {
                    "worker_pubkey": worker_pubkey,
                    "staked_amount_sats": 0,
                    "is_active": True,
                    "slash_count": 0,
                    "created_at": time.time(),
                }
            self._bonds[worker_pubkey]["staked_amount_sats"] += amount_sats
            return dict(self._bonds[worker_pubkey])

    def get_bond_balance(self, worker_pubkey: str) -> int:
        """Returns currently staked satoshis for a worker."""
        with self._lock:
            record = self._bonds.get(worker_pubkey)
            return record["staked_amount_sats"] if record else 0

    def submit_fraud_dispute(
        self,
        proof: Any,
        slash_amount_sats: Optional[int] = None,
        consumer_signature: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Processes a Freivalds computational fraud dispute:
        1. Identifies the accused worker and task.
        2. Cancels active HELD invoice for the task.
        3. Slashes the worker's staked bond by slash_amount_sats (default 100,000 sats).
        4. Increments slash count and returns resolution summary.
        """
        task_id = str(
            getattr(proof, "task_id", "")
            or (proof.get("task_id") if isinstance(proof, dict) else "")
        )
        worker_pubkey = str(
            getattr(proof, "worker_pubkey", "")
            or (proof.get("worker_pubkey") if isinstance(proof, dict) else "")
        )

        with self._lock:
            # 1. Cancel in-flight held invoice if active
            invoice_cancelled = False
            p_hash = self._task_index.get(task_id)
            if p_hash and p_hash in self._invoices:
                inv = self._invoices[p_hash]
                if inv.status in (InvoiceStatus.UNPAID, InvoiceStatus.HELD):
                    inv.status = InvoiceStatus.CANCELLED
                    invoice_cancelled = True

            # 2. Determine slash penalty
            slash_sats = slash_amount_sats if slash_amount_sats is not None else 100_000

            # 3. Deduct from worker's bonded collateral
            if worker_pubkey not in self._bonds:
                self._bonds[worker_pubkey] = {
                    "worker_pubkey": worker_pubkey,
                    "staked_amount_sats": 100_000,
                    "is_active": True,
                    "slash_count": 0,
                    "created_at": time.time(),
                }

            bond = self._bonds[worker_pubkey]
            actual_slashed = min(bond["staked_amount_sats"], slash_sats)
            bond["staked_amount_sats"] -= actual_slashed
            bond["slash_count"] += 1
            if bond["staked_amount_sats"] <= 0:
                bond["is_active"] = False

            return {
                "success": True,
                "task_id": task_id,
                "worker_pubkey": worker_pubkey,
                "actual_slashed_sats": actual_slashed,
                "remaining_stake_sats": bond["staked_amount_sats"],
                "slash_count": bond["slash_count"],
                "invoice_cancelled": invoice_cancelled,
            }

    def reset(self) -> None:
        with self._lock:
            self._invoices.clear()
            self._task_index.clear()
            self._job_index.clear()
            self._bonds.clear()

    def clear(self) -> None:
        self.reset()
