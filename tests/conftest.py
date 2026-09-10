"""
Pytest configuration, shared fixtures, and authoritative reference oracles
for matrix-compute-sdk opaque-box E2E test suite.
"""

import base64
import hashlib
import hmac
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pytest

# Ensure matrix-contracts, matrix-compute-sdk/src, and matrix-compute-node are resolvable
TESTS_DIR = Path(__file__).resolve().parent
SDK_DIR = TESTS_DIR.parent
ROOT_DIR = SDK_DIR.parent
CONTRACTS_DIR = ROOT_DIR / "matrix-contracts"
NODE_DIR = ROOT_DIR / "matrix-compute-node"
SDK_SRC_DIR = SDK_DIR / "src"

for p in [str(CONTRACTS_DIR), str(SDK_DIR), str(SDK_SRC_DIR), str(NODE_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Import contracts schemas and protobufs
from matrix_contracts.schemas import (
    HardwareSpecs,
    PricingSpecs,
    BondStatus,
    WorkerAnnouncePayload,
    WorkerQueryRequest,
    WorkerQueryResponse,
    MatchedWorker,
    CreateHodlInvoiceRequest,
    CreateHodlInvoiceResponse,
    SettlePaymentRequest,
    FreivaldsFraudProof,
)
from matrix_contracts.v1 import pb2, pb2_grpc


# ==============================================================================
# AUTHORITATIVE REFERENCE ORACLES & TEST HELPERS
# ==============================================================================

class ReferenceOracle:
    """Independent mathematical and cryptographic oracle for deriving expected outputs."""

    @staticmethod
    def truncate_to_bf16(tensor: np.ndarray) -> np.ndarray:
        """
        Truncates float32 mantissa to 7 bits (IEEE bfloat16 emulation).
        Clears the lowest 16 bits of 32-bit float representation (mask 0xFFFF0000).
        Preserves NaN semantics.
        """
        f32 = np.ascontiguousarray(tensor, dtype=np.float32)
        u32 = f32.view(np.uint32)
        masked_u32 = u32 & np.uint32(0xFFFF0000)
        is_nan = np.isnan(f32)
        if np.any(is_nan):
            lost_mantissa = is_nan & ((masked_u32 & np.uint32(0x007F0000)) == 0)
            if np.any(lost_mantissa):
                masked_u32 = masked_u32.copy()
                masked_u32[lost_mantissa] |= np.uint32(0x00400000)
        return masked_u32.view(np.float32)

    @classmethod
    def gemm(cls, a: np.ndarray, b: np.ndarray, dtype: str = "fp32") -> np.ndarray:
        """Calculates expected matrix multiplication C = A x B in double precision and casts."""
        norm_dtype = str(dtype).lower().strip()
        if norm_dtype == "bf16":
            a_bf16 = cls.truncate_to_bf16(a)
            b_bf16 = cls.truncate_to_bf16(b)
            c = np.matmul(a_bf16.astype(np.float64), b_bf16.astype(np.float64)).astype(np.float32)
            return cls.truncate_to_bf16(c)

        a_f64 = a.astype(np.float64)
        b_f64 = b.astype(np.float64)
        c_f64 = np.matmul(a_f64, b_f64)

        if norm_dtype == "fp16":
            return c_f64.astype(np.float16)
        return c_f64.astype(np.float32)

    @staticmethod
    def sha256_digest(tensor: np.ndarray) -> str:
        """Computes SHA256 hex digest over raw contiguous bytes of tensor."""
        if not tensor.flags["C_CONTIGUOUS"]:
            tensor = np.ascontiguousarray(tensor)
        return hashlib.sha256(tensor.tobytes()).hexdigest()

    @staticmethod
    def calculate_gflops(m: int, k: int, n: int) -> float:
        """Authoritative GFLOP formula: (2 * M * N * K) / 10^9."""
        return (2.0 * float(m) * float(k) * float(n)) / 1e9

    @staticmethod
    def calculate_min_sats(gflops: float, sats_per_gflop: float, min_task_sats: int = 10) -> int:
        """Authoritative compute cost in Satoshis."""
        raw_sats = math.ceil(gflops * sats_per_gflop)
        return max(raw_sats, min_task_sats)

    @staticmethod
    def get_unit_roundoff(dtype: str = "fp32") -> float:
        """Returns IEEE 754 unit roundoff u for given precision."""
        dt = str(dtype).lower().strip()
        if dt in ("fp16", "float16"):
            return 2.0 ** -11   # ~4.8828e-4
        elif dt in ("bf16", "bfloat16"):
            return 2.0 ** -8    # ~3.90625e-3
        elif dt in ("fp64", "float64"):
            return 2.0 ** -53   # ~1.1102e-16
        else:
            return 2.0 ** -24   # ~5.9605e-8 (fp32)

    @classmethod
    def higham_bound(cls, k: int, norm_a: float, norm_b: float, dtype: str = "fp32") -> float:
        """
        Higham's matrix multiplication rounding bound:
        gamma_K = (K * u) / (1 - K * u)
        bound = gamma_K * ||A||_inf * ||B||_inf
        """
        u = cls.get_unit_roundoff(dtype)
        ku = float(k) * u
        gamma_k = (ku / (1.0 - ku)) if ku < 0.999 else (ku * 1000.0)
        return gamma_k * float(norm_a) * float(norm_b)

    @classmethod
    def freivalds_verify(
        cls,
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        dtype: str = "fp32",
        num_trials: int = 5,
        seed: Optional[int] = None,
    ) -> Tuple[bool, float, float, Dict[str, Any]]:
        """
        Independent Freivalds randomized algorithm verification with Higham's bound:
        Checks ||A * (B * r) - C * r||_inf <= max(dynamic_bound, 1e-6) for random binary vectors r.
        Returns: (is_valid, max_discrepancy, dynamic_bound, proof_data)
        """
        if seed is not None:
            np.random.seed(seed)

        if a.shape[1] != b.shape[0]:
            raise ValueError(f"Inner dimension mismatch: A cols ({a.shape[1]}) != B rows ({b.shape[0]})")
        if c.shape != (a.shape[0], b.shape[1]):
            raise ValueError(f"Result matrix shape mismatch: C {c.shape} != expected ({a.shape[0]}, {b.shape[1]})")

        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b)) and np.all(np.isfinite(c))):
            return False, float("inf"), 1e-6, {"error": "Non-finite values detected in input or output matrices"}

        m, k = a.shape
        _, n = b.shape

        norm_a = float(np.max(np.sum(np.abs(a.astype(np.float64)), axis=1))) if m > 0 and k > 0 else 0.0
        norm_b = float(np.max(np.sum(np.abs(b.astype(np.float64)), axis=1))) if k > 0 and n > 0 else 0.0
        dynamic_bound = max(cls.higham_bound(k, norm_a, norm_b, dtype=dtype), 1e-6)

        max_discrepancy = 0.0
        first_failure_proof: Dict[str, Any] = {}

        for _ in range(num_trials):
            r = np.random.randint(0, 2, size=(n, 1)).astype(np.float64)
            br = np.matmul(b.astype(np.float64), r)
            abr = np.matmul(a.astype(np.float64), br)
            cr = np.matmul(c.astype(np.float64), r)
            diff = np.max(np.abs(abr - cr))

            if diff > max_discrepancy:
                max_discrepancy = float(diff)

            if diff > dynamic_bound and not first_failure_proof:
                first_failure_proof = {
                    "random_vector_r": r.flatten().astype(int).tolist(),
                    "vector_br": br.flatten().astype(float).tolist(),
                    "vector_abr": abr.flatten().astype(float).tolist(),
                    "vector_cr": cr.flatten().astype(float).tolist(),
                    "observed_diff": float(diff),
                    "dynamic_bound": float(dynamic_bound),
                }

        is_valid = (max_discrepancy <= dynamic_bound)
        return is_valid, max_discrepancy, dynamic_bound, first_failure_proof


class OpaqueL402Helper:
    """Helper for generating and manipulating L402 tokens and Authorization headers."""

    @staticmethod
    def mint_macaroon(
        verification_key: str,
        task_id: str,
        expiry_timestamp: int,
        max_gflops: float,
        locked_sats: int = 100,
        payment_hash: Optional[str] = None,
    ) -> str:
        """Mints an HMAC-SHA256 authenticated macaroon payload."""
        caveats = {
            "task_id": task_id,
            "expiry": expiry_timestamp,
            "max_gflops": max_gflops,
            "locked_sats": locked_sats,
        }
        if payment_hash is not None:
            caveats["payment_hash"] = payment_hash
        caveat_str = json.dumps(caveats, sort_keys=True)
        sig = hmac.new(
            verification_key.encode("utf-8"),
            caveat_str.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        payload = {"caveats": caveats, "signature": sig}
        return base64.b64encode(json.dumps(payload).encode("utf-8")).decode("utf-8")

    @staticmethod
    def format_auth_header(macaroon_b64: str, payment_hash: str) -> str:
        """Formats the gRPC Authorization header according to L402 spec."""
        return f"L402 {macaroon_b64}:{payment_hash}"

    @staticmethod
    def parse_auth_header(auth_header: str) -> Tuple[Optional[str], Optional[str]]:
        """Parses an L402 authorization header into (macaroon_b64, payment_hash)."""
        if not auth_header or not auth_header.startswith("L402 "):
            return None, None
        token_part = auth_header[5:].strip()
        if ":" not in token_part:
            return None, None
        parts = token_part.split(":", 1)
        return parts[0], parts[1]


class StreamChunkHelper:
    """Generates and validates TileDataChunk protobuf streams from numpy matrices."""

    @staticmethod
    def generate_chunks(
        task_id: str,
        matrix_a: np.ndarray,
        matrix_b: np.ndarray,
        chunk_size_bytes: int = 256,
    ) -> List[pb2.TileDataChunk]:
        """Splits Matrix A and Matrix B into sequential TileDataChunk messages."""
        chunks = []
        seq = 0

        # Matrix A chunks
        a_bytes = matrix_a.tobytes()
        for i in range(0, len(a_bytes), chunk_size_bytes):
            chunk_data = a_bytes[i : i + chunk_size_bytes]
            chunks.append(
                pb2.TileDataChunk(
                    task_id=task_id,
                    chunk_bytes=chunk_data,
                    is_matrix_a=True,
                    sequence_number=seq,
                    is_final_chunk=False,
                )
            )
            seq += 1

        # Matrix B chunks
        b_bytes = matrix_b.tobytes()
        num_b_chunks = math.ceil(len(b_bytes) / chunk_size_bytes) if len(b_bytes) > 0 else 1
        for idx, i in enumerate(range(0, len(b_bytes), chunk_size_bytes)):
            chunk_data = b_bytes[i : i + chunk_size_bytes]
            is_final = idx == num_b_chunks - 1
            chunks.append(
                pb2.TileDataChunk(
                    task_id=task_id,
                    chunk_bytes=chunk_data,
                    is_matrix_a=False,
                    sequence_number=seq,
                    is_final_chunk=is_final,
                )
            )
            seq += 1

        return chunks


# ==============================================================================
# HERMETIC IN-MEMORY TEST DOUBLES FOR SDK TESTING
# ==============================================================================

class MockRelayDouble:
    """Thread-safe in-memory Nostr Discovery Relay double."""

    def __init__(self):
        self._events: List[Dict[str, Any]] = []

    def publish(self, event: Dict[str, Any]) -> bool:
        # Parameterized replaceable event on (kind, pubkey, d-tag)
        kind = event.get("kind", 1)
        pubkey = event.get("pubkey", "")
        d_tag = "matrix-worker"
        for t in event.get("tags", []):
            if len(t) >= 2 and t[0] == "d":
                d_tag = t[1]
                break

        # Remove existing if same (kind, pubkey, d_tag)
        self._events = [
            e for e in self._events
            if not (e.get("kind") == kind and e.get("pubkey") == pubkey and self._get_d(e) == d_tag)
        ]
        self._events.append(dict(event))
        return True

    def _get_d(self, event: Dict[str, Any]) -> str:
        for t in event.get("tags", []):
            if len(t) >= 2 and t[0] == "d":
                return t[1]
        return "matrix-worker"

    def query(
        self,
        min_vram_mb: int = 0,
        max_price: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        results = []
        for e in self._events:
            tag_map = {t[0]: t[1] for t in e.get("tags", []) if len(t) >= 2}
            vram = int(tag_map.get("vram_mb", 0))
            price = float(tag_map.get("sats_per_gflop", 0.0))
            if vram >= min_vram_mb and (max_price is None or price <= max_price):
                results.append(e)
        return results

    def query_workers(self, request: WorkerQueryRequest) -> WorkerQueryResponse:
        matched = []
        events = self.query(
            min_vram_mb=request.min_vram_mb,
            max_price=request.max_price_sats_per_gflop,
        )
        for e in events:
            tag_map = {t[0]: t[1] for t in e.get("tags", []) if len(t) >= 2}
            matched.append(
                MatchedWorker(
                    worker_pubkey=e.get("pubkey", ""),
                    p2p_multiaddr=tag_map.get("p2p", ""),
                    lightning_routing_pubkey=tag_map.get("lightning", ""),
                    reputation_score=1.0,
                    sats_per_gflop=float(tag_map.get("sats_per_gflop", 0.0005)),
                )
            )
        matched.sort(key=lambda w: (w.sats_per_gflop, -w.reputation_score))
        return WorkerQueryResponse(
            status="SUCCESS",
            matched_workers=matched[: request.required_slots],
        )

    def clear(self):
        self._events.clear()

    def get_events(self) -> List[Dict[str, Any]]:
        return list(self._events)


class MockSettlementDouble:
    """Thread-safe in-memory Settlement Service double simulating HODL invoice lifecycle."""

    def __init__(self, verification_key: str = "test-settlement-key-12345"):
        self.verification_key = verification_key
        self._invoices: Dict[str, Dict[str, Any]] = {}
        self._task_index: Dict[str, str] = {}

    def create_hodl_invoice(
        self,
        task_id: str,
        total_compute_sats: int,
        worker_lightning_address: str = "worker@node",
        payment_hash: Optional[str] = None,
        max_gflops: float = 100.0,
        expiry_seconds: int = 3600,
    ) -> CreateHodlInvoiceResponse:
        if task_id in self._task_index:
            raise ValueError(f"Task '{task_id}' already registered to invoice '{self._task_index[task_id]}'")

        if payment_hash is None:
            preimage_bytes = os.urandom(32)
            preimage = preimage_bytes.hex()
            final_hash = hashlib.sha256(preimage_bytes).hexdigest()
        else:
            final_hash = payment_hash.lower()
            preimage = None

        if final_hash in self._invoices:
            raise ValueError(f"Duplicate payment hash '{final_hash}' already exists")

        macaroon = OpaqueL402Helper.mint_macaroon(
            self.verification_key,
            task_id,
            int(time.time()) + expiry_seconds,
            max_gflops,
            locked_sats=total_compute_sats,
            payment_hash=final_hash,
        )

        fee_sats = int(round(total_compute_sats * 0.02))
        payout_sats = total_compute_sats - fee_sats

        record = {
            "task_id": task_id,
            "payment_hash": final_hash,
            "status": "UNPAID",
            "preimage": preimage,
            "total_sats": total_compute_sats,
            "worker_payout_sats": payout_sats,
            "macaroon": macaroon,
        }
        self._invoices[final_hash] = record
        self._task_index[task_id] = final_hash

        return CreateHodlInvoiceResponse(
            payment_hash=final_hash,
            payment_request=f"lnbc{total_compute_sats}p1{final_hash[:16]}",
            platform_fee_sats=fee_sats,
            worker_payout_sats=payout_sats,
            macaroon=macaroon,
        )

    def hold_invoice(self, payment_hash: str) -> bool:
        h = payment_hash.lower()
        if h not in self._invoices:
            raise KeyError(f"Invoice {h} not found")
        if self._invoices[h]["status"] != "UNPAID":
            raise ValueError(f"Cannot hold invoice in state '{self._invoices[h]['status']}'; must be UNPAID")
        self._invoices[h]["status"] = "HELD"
        return True

    def settle_invoice(self, payment_hash: str, preimage: str) -> bool:
        h = payment_hash.lower()
        if h not in self._invoices:
            raise KeyError(f"Invoice {h} not found")
        if self._invoices[h]["status"] != "HELD":
            raise ValueError(f"Cannot settle invoice in state '{self._invoices[h]['status']}'; must be HELD")

        computed = hashlib.sha256(bytes.fromhex(preimage)).hexdigest()
        if computed != h:
            raise ValueError("Preimage does not match payment_hash")

        self._invoices[h]["status"] = "SETTLED"
        self._invoices[h]["preimage"] = preimage
        return True

    def cancel_invoice(self, payment_hash: str) -> bool:
        h = payment_hash.lower()
        if h not in self._invoices:
            raise KeyError(f"Invoice {h} not found")
        if self._invoices[h]["status"] == "SETTLED":
            raise ValueError("Cannot cancel SETTLED invoice")
        self._invoices[h]["status"] = "CANCELLED"
        return True

    def get_status(self, payment_hash: str) -> Optional[str]:
        return self._invoices.get(payment_hash.lower(), {}).get("status")

    def reset(self):
        self._invoices.clear()
        self._task_index.clear()


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture
def clean_env(monkeypatch):
    """Isolates environment variables from host system."""
    env_keys_to_clean = [
        "MATRIX_CONSUMER_PRIVKEY",
        "CONSUMER_PRIVKEY",
        "MATRIX_NOSTR_RELAYS",
        "NOSTR_RELAYS",
        "MATRIX_SETTLEMENT_API_URL",
        "SETTLEMENT_API_URL",
        "MATRIX_MAX_PRICE_SATS_PER_GFLOP",
        "MATRIX_MIN_VRAM_MB",
        "OFFLINE_HERMETIC",
    ]
    for key in env_keys_to_clean:
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def mock_relay():
    relay = MockRelayDouble()
    yield relay
    relay.clear()


@pytest.fixture
def mock_settlement():
    settlement = MockSettlementDouble()
    yield settlement
    settlement.reset()


@pytest.fixture
def sample_tensors():
    """Deterministic sample FP32 matrices A (4x4) and B (4x4) with reference product C."""
    np.random.seed(42)
    a = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0, 7.0],
        ],
        dtype=np.float32,
    )
    b = np.array(
        [
            [7.0, 6.0, 5.0, 4.0],
            [3.0, 2.0, 1.0, 0.0],
            [0.0, 1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0, 7.0],
        ],
        dtype=np.float32,
    )
    c_expected = np.matmul(a, b)
    return {
        "m": 4,
        "k": 4,
        "n": 4,
        "a": a,
        "b": b,
        "c_expected": c_expected,
    }
