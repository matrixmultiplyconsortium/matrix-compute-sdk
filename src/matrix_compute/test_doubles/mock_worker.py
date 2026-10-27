"""In-memory Mock Worker Node test double for matrix-compute-sdk testing."""

from __future__ import annotations

import base64
import collections
from concurrent import futures
import hashlib
import hmac
import itertools
import json
import logging
import math
import secrets
import threading
import time
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Union

import grpc
import numpy as np

try:
    from matrix_contracts.proto import compute_service_pb2 as pb2
    from matrix_contracts.proto import compute_service_pb2_grpc as pb2_grpc
except ImportError:
    from matrix_contracts.v1 import pb2, pb2_grpc

try:
    from matrix_contracts.schemas.discovery import (
        BondStatus,
        HardwareSpecs,
        PricingSpecs,
        WorkerAnnouncePayload,
    )
except ImportError:
    from matrix_contracts.schemas import (
        BondStatus,
        HardwareSpecs,
        PricingSpecs,
        WorkerAnnouncePayload,
    )

from matrix_compute.test_doubles.mock_relay import MockNostrRelay
from matrix_compute.test_doubles.mock_settlement import MockSettlementService

logger = logging.getLogger(__name__)

SECP256K1_ORDER = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
SECP256K1_HALF_ORDER = SECP256K1_ORDER // 2

try:
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature
    from cryptography.hazmat.primitives import hashes, serialization
    _HAS_CRYPTOGRAPHY = True
except ImportError:
    _HAS_CRYPTOGRAPHY = False


class _InternalWorkerSigner:
    """Lightweight Secp256k1 signer for mock worker result signing."""

    def __init__(self, privkey_hex: Optional[str] = None) -> None:
        if privkey_hex is None:
            self._privkey_int = secrets.randbelow(SECP256K1_ORDER - 1) + 1
        else:
            self._privkey_int = int(privkey_hex.strip(), 16)

        self._privkey_bytes = self._privkey_int.to_bytes(32, "big")
        if _HAS_CRYPTOGRAPHY:
            self._priv_key_obj = ec.derive_private_key(self._privkey_int, ec.SECP256K1())
            self._pub_key_obj = self._priv_key_obj.public_key()
            self._pubkey_bytes_compressed = self._pub_key_obj.public_bytes(
                encoding=serialization.Encoding.X962,
                format=serialization.PublicFormat.CompressedPoint,
            )
        else:
            raise RuntimeError("cryptography package required for Secp256k1 signing")

    def get_privkey_hex(self) -> str:
        return self._privkey_bytes.hex()

    def get_pubkey_hex(self) -> str:
        return self._pubkey_bytes_compressed.hex()

    def sign_message(self, data: bytes) -> str:
        der_sig = self._priv_key_obj.sign(data, ec.ECDSA(hashes.SHA256()))
        r, s = decode_dss_signature(der_sig)
        if s > SECP256K1_HALF_ORDER:
            s = SECP256K1_ORDER - s
        return (r.to_bytes(32, "big") + s.to_bytes(32, "big")).hex()


class MockWorkerServicer(pb2_grpc.MatrixComputeServiceServicer):
    """
    gRPC Servicer implementing MatrixComputeService with strict L402 gate
    and programmable fault injection hooks.
    """

    def __init__(
        self,
        signer: _InternalWorkerSigner,
        settlement_service: Optional[MockSettlementService] = None,
        verification_key: str = "settlement-secret-verification-key-12345",
        min_price_rate: float = 0.0005,
        min_task_sats: int = 10,
    ) -> None:
        self.signer = signer
        self.settlement_service = settlement_service
        if settlement_service is not None and hasattr(settlement_service, "verification_key"):
            if verification_key == "settlement-secret-verification-key-12345" or not verification_key:
                verification_key = settlement_service.verification_key
        self.verification_key = verification_key
        self.min_price_rate = float(min_price_rate)
        self.min_task_sats = int(min_task_sats)

        self._lock = threading.RLock()
        self._task_metadata: Dict[str, Dict[str, Any]] = {}
        self.task_count = 0
        self.last_task_id: Optional[str] = None
        self.execution_history: List[Dict[str, Any]] = []

        # Fault injection hooks
        self.tamper_result = False
        self.tamper_perturbation = 1.0
        self.tamper_signature = False
        self.drop_connection = False
        self.reject_handshake = False
        self.handshake_reject_message = "Worker busy"
        self.delay_seconds = 0.0
        self.bypass_l402 = False
        self.override_result_task_id: Optional[str] = None
        self.tamper_task_ids: Set[str] = set()
        self.tamper_on_task_count: Optional[int] = None

        # Phase 3: Stateful Iterative Pinned Tensor Sessions
        self._pinned_sessions: Dict[str, Dict[str, Any]] = {}
        self.matrix_upload_counts: Dict[str, int] = collections.defaultdict(int)
        self.recorded_leases: List[str] = []
        self.recorded_steps: List[Tuple[str, int]] = []
        self.recorded_releases: List[str] = []

        # Fault injection hooks for Phase 3
        self.tamper_step_index: Optional[int] = None
        self.tamper_step_indices: Set[int] = set()
        self.tamper_step_perturbation: float = 5.0
        self.tamper_step_signature: bool = False
        self.reject_lease: bool = False
        self.lease_reject_message: str = "Worker VRAM capacity exhausted"
        self.expire_lease_early: bool = False
        self.expire_lease_after_step: Optional[int] = None
        self.delay_step_seconds: float = 0.0

        # Stats for iterative leases
        self.lease_calls: int = 0
        self.step_calls: int = 0
        self.release_calls: int = 0
        self.uploaded_bytes_total: int = 0

    @property
    def tamper_step_iteration(self) -> Optional[int]:
        return self.tamper_step_index

    @tamper_step_iteration.setter
    def tamper_step_iteration(self, val: Optional[int]) -> None:
        self.tamper_step_index = val

    @property
    def reject_lease_message(self) -> str:
        return self.lease_reject_message

    @reject_lease_message.setter
    def reject_lease_message(self, val: str) -> None:
        self.lease_reject_message = val

    @property
    def step_delay_seconds(self) -> float:
        return self.delay_step_seconds

    @step_delay_seconds.setter
    def step_delay_seconds(self, val: float) -> None:
        self.delay_step_seconds = val

    def RequestTileTask(
        self,
        request: pb2.TileTaskRequest,
        context: Optional[grpc.ServicerContext] = None,
    ) -> pb2.TileTaskAck:
        if self.drop_connection:
            if context:
                context.abort(grpc.StatusCode.UNAVAILABLE, "Connection dropped by worker")
            raise RuntimeError("Connection dropped by worker")

        if self.reject_handshake:
            return pb2.TileTaskAck(
                task_id=request.task_id,
                is_ready=False,
                estimated_gflops=0.0,
                worker_pubkey=self.signer.get_pubkey_hex(),
                message=self.handshake_reject_message,
            )

        if request.matrix_a_rows <= 0 or request.matrix_a_cols <= 0 or request.matrix_b_cols <= 0:
            if context:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Matrix dimensions must be positive integers")
            raise ValueError("Matrix dimensions must be positive integers")

        dtype = (request.dtype or "fp32").lower().strip()
        if dtype not in ("fp32", "fp16", "bf16"):
            if context:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"Unsupported dtype: {dtype}")
            raise ValueError(f"Unsupported dtype: {dtype}")

        m = int(request.matrix_a_rows)
        k = int(request.matrix_a_cols)
        n = int(request.matrix_b_cols)
        estimated_gflops = (2.0 * float(m) * float(k) * float(n)) / 1e9

        with self._lock:
            self._task_metadata[request.task_id] = {
                "m": m,
                "k": k,
                "n": n,
                "dtype": dtype,
                "gflops": estimated_gflops,
                "created_at": time.time(),
            }
            self.last_task_id = request.task_id

        return pb2.TileTaskAck(
            task_id=request.task_id,
            is_ready=True,
            estimated_gflops=estimated_gflops,
            worker_pubkey=self.signer.get_pubkey_hex(),
            message="Node ready",
        )

    def _verify_l402_gate(
        self,
        auth_header: str,
        task_id: str,
        m: int,
        k: int,
        n: int,
    ) -> Tuple[bool, str, grpc.StatusCode]:
        if not auth_header:
            return False, "Missing Authorization header with L402 token", grpc.StatusCode.UNAUTHENTICATED

        parts = auth_header.strip().split(" ", 1)
        if len(parts) != 2 or parts[0].upper() not in ("L402", "LSAT"):
            return False, "Malformed Authorization header", grpc.StatusCode.UNAUTHENTICATED

        cred_parts = parts[1].strip().split(":", 1)
        if len(cred_parts) != 2:
            return False, "Malformed credentials (expected macaroon:payment_hash)", grpc.StatusCode.UNAUTHENTICATED

        macaroon_b64, payment_hash = cred_parts[0].strip(), cred_parts[1].strip()
        try:
            payload = json.loads(base64.b64decode(macaroon_b64).decode("utf-8"))
        except Exception:
            return False, "Malformed macaroon base64/JSON", grpc.StatusCode.UNAUTHENTICATED

        if not isinstance(payload, dict) or "caveats" not in payload or "signature" not in payload:
            return False, "Malformed macaroon structure", grpc.StatusCode.UNAUTHENTICATED

        caveats = payload["caveats"]
        signature = payload["signature"]
        if not isinstance(caveats, dict) or not isinstance(signature, str):
            return False, "Malformed macaroon types", grpc.StatusCode.UNAUTHENTICATED

        # 1. HMAC check
        caveat_str = json.dumps(caveats, sort_keys=True)
        keys_to_try = [self.verification_key]
        if self.settlement_service is not None and hasattr(self.settlement_service, "verification_key"):
            if self.settlement_service.verification_key not in keys_to_try:
                keys_to_try.append(self.settlement_service.verification_key)
        for fallback_k in ["settlement-secret-verification-key-12345", "settlement_secret_key_dev"]:
            if fallback_k not in keys_to_try:
                keys_to_try.append(fallback_k)

        valid_hmac = False
        for vk in keys_to_try:
            expected_sig = hmac.new(
                vk.encode("utf-8"),
                caveat_str.encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
            if hmac.compare_digest(signature, expected_sig):
                valid_hmac = True
                break

        if not valid_hmac:
            return False, "Invalid macaroon HMAC signature", grpc.StatusCode.UNAUTHENTICATED

        # 2. Caveat task_id containment
        allowed_tasks = caveats.get("task_ids") or [caveats.get("task_id")]
        if task_id not in allowed_tasks:
            return False, f"Caveat task_id mismatch: task '{task_id}' not in authorized batch {allowed_tasks}", grpc.StatusCode.PERMISSION_DENIED

        # 3. Caveat payment_hash
        if caveats.get("payment_hash") != payment_hash:
            return False, "Macaroon payment_hash mismatch", grpc.StatusCode.PERMISSION_DENIED

        # 4. Caveat expiry
        now = time.time()
        expiry = caveats.get("expiry", 0)
        if not isinstance(expiry, (int, float)) or expiry <= now:
            return False, f"Macaroon expired at {expiry}", grpc.StatusCode.UNAUTHENTICATED

        # 5. Workload bounds
        requested_gflops = (2.0 * float(m) * float(k) * float(n)) / 1e9
        max_gflops = caveats.get("max_gflops", 0.0)
        if max_gflops < requested_gflops:
            return False, f"Insufficient GFLOPs ({max_gflops} < {requested_gflops})", grpc.StatusCode.PERMISSION_DENIED

        if "locked_sats" in caveats:
            locked = caveats["locked_sats"]
            required_sats = max(math.ceil(requested_gflops * self.min_price_rate), self.min_task_sats)
            if locked < required_sats:
                return False, f"Underpayment: {locked} < {required_sats}", grpc.StatusCode.FAILED_PRECONDITION

        # 6. Settlement HELD status
        if not self.settlement_service:
            return False, "Settlement verification unconfigured", grpc.StatusCode.FAILED_PRECONDITION

        status = self.settlement_service.get_status(payment_hash)
        if status != "HELD":
            return False, f"Payment hash is in state '{status}', expected 'HELD'", grpc.StatusCode.FAILED_PRECONDITION

        return True, "OK", grpc.StatusCode.OK

    def ExecuteTileTask(
        self,
        request_iterator: Iterator[pb2.TileDataChunk],
        context: Optional[grpc.ServicerContext] = None,
    ) -> pb2.TileResult:
        start_time = time.perf_counter()
        if self.drop_connection:
            if context:
                context.abort(grpc.StatusCode.UNAVAILABLE, "Connection dropped by worker")
            raise RuntimeError("Connection dropped by worker")

        auth_header = ""
        meta_task_id = None
        if context is not None:
            meta = dict(context.invocation_metadata())
            auth_header = meta.get("authorization") or meta.get("authorization-bin") or ""
            meta_task_id = meta.get("x-task-id")

        task_id = meta_task_id
        if not task_id and auth_header:
            try:
                cred = auth_header.strip().split(" ", 1)[1].split(":", 1)[0]
                payload = json.loads(base64.b64decode(cred).decode("utf-8"))
                task_id = payload.get("caveats", {}).get("task_id")
            except Exception:
                pass

        chunk_iter = iter(request_iterator)
        first_chunk = None
        try:
            first_chunk = next(chunk_iter)
        except StopIteration:
            pass

        if not task_id and first_chunk is not None:
            task_id = first_chunk.task_id

        with self._lock:
            task_info = self._task_metadata.get(task_id or "")

        if not task_info:
            if context:
                context.abort(grpc.StatusCode.FAILED_PRECONDITION, f"Task '{task_id}' was not negotiated")
            raise RuntimeError(f"Task '{task_id}' was not negotiated")

        m = task_info["m"]
        k = task_info["k"]
        n = task_info["n"]
        dtype = task_info["dtype"]

        # 5-step L402 Gate
        if not self.bypass_l402:
            ok, err_msg, status_code = self._verify_l402_gate(auth_header, task_id, m, k, n)
            if not ok:
                if context:
                    context.abort(status_code, err_msg)
                raise PermissionError(f"{status_code}: {err_msg}")

        # Progressive chunk streaming
        np_dtype = np.float32 if dtype in ("fp32", "bf16") else np.float16
        bytes_per_elem = 4 if np_dtype == np.float32 else 2
        expected_a_bytes = m * k * bytes_per_elem
        expected_b_bytes = k * n * bytes_per_elem

        a_chunks: List[bytes] = []
        b_chunks: List[bytes] = []
        running_a = 0
        running_b = 0
        expected_seq = 0
        final_seen = False

        chunks_to_process = itertools.chain([first_chunk], chunk_iter) if first_chunk is not None else []
        try:
            for chunk in chunks_to_process:
                if self.drop_connection:
                    if context:
                        context.abort(grpc.StatusCode.UNAVAILABLE, "Connection dropped mid-stream")
                    raise RuntimeError("Connection dropped mid-stream")

                if chunk.sequence_number != expected_seq:
                    msg = f"Out of order sequence: expected {expected_seq}, got {chunk.sequence_number}"
                    if context:
                        context.abort(grpc.StatusCode.INVALID_ARGUMENT, msg)
                    raise ValueError(msg)

                if chunk.task_id != task_id:
                    msg = f"Chunk task_id mismatch: expected '{task_id}', got '{chunk.task_id}'"
                    if context:
                        context.abort(grpc.StatusCode.INVALID_ARGUMENT, msg)
                    raise ValueError(msg)

                chunk_data = chunk.chunk_bytes or b""
                chunk_len = len(chunk_data)

                if chunk.is_matrix_a:
                    if running_a + chunk_len > expected_a_bytes:
                        msg = f"Matrix A stream exceeded limit ({expected_a_bytes}): got {running_a + chunk_len}"
                        if context:
                            context.abort(grpc.StatusCode.INVALID_ARGUMENT, msg)
                        raise ValueError(msg)
                    running_a += chunk_len
                    a_chunks.append(chunk_data)
                else:
                    if running_b + chunk_len > expected_b_bytes:
                        msg = f"Matrix B stream exceeded limit ({expected_b_bytes}): got {running_b + chunk_len}"
                        if context:
                            context.abort(grpc.StatusCode.INVALID_ARGUMENT, msg)
                        raise ValueError(msg)
                    running_b += chunk_len
                    b_chunks.append(chunk_data)

                expected_seq += 1
                if chunk.is_final_chunk:
                    final_seen = True
                    break

            if not final_seen:
                if context:
                    context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Stream ended without final chunk flag")
                raise ValueError("Stream ended without final chunk flag")

            if running_a != expected_a_bytes or running_b != expected_b_bytes:
                msg = f"Byte count mismatch: A={running_a}/{expected_a_bytes}, B={running_b}/{expected_b_bytes}"
                if context:
                    context.abort(grpc.StatusCode.INVALID_ARGUMENT, msg)
                raise ValueError(msg)

            # Artificial delay if configured
            if self.delay_seconds > 0:
                time.sleep(self.delay_seconds)

            # Reconstruct matrices & compute GEMM
            matrix_a = np.frombuffer(b"".join(a_chunks), dtype=np_dtype).reshape((m, k))
            matrix_b = np.frombuffer(b"".join(b_chunks), dtype=np_dtype).reshape((k, n))

            compute_start = time.perf_counter()
            if dtype == "bf16":
                # BF16 emulation: compute in FP32, mask mantissa
                a_f32 = matrix_a.astype(np.float32)
                b_f32 = matrix_b.astype(np.float32)
                c_f32 = np.matmul(a_f32, b_f32)
                u32_view = c_f32.view(np.uint32)
                u32_masked = u32_view & 0xFFFF0000
                matrix_c = u32_masked.view(np.float32)
            else:
                matrix_c = np.matmul(matrix_a, matrix_b)
            exec_time_ms = int((time.perf_counter() - compute_start) * 1000)

            # Fault injection: tamper result
            should_tamper = self.tamper_result
            if self.tamper_task_ids and task_id in self.tamper_task_ids:
                should_tamper = True
            if self.tamper_on_task_count is not None and (self.task_count + 1) >= self.tamper_on_task_count:
                should_tamper = True

            if should_tamper:
                matrix_c = matrix_c.copy()
                matrix_c += self.tamper_perturbation

            c_bytes = matrix_c.tobytes()
            c_hash = hashlib.sha256(c_bytes).hexdigest()

            # Sign result
            if self.tamper_signature:
                sig = "00" * 64
            else:
                payload_to_sign = f"{task_id}:{c_hash}".encode("utf-8")
                sig = self.signer.sign_message(payload_to_sign)

            end_time = time.perf_counter()
            with self._lock:
                self.task_count += 1
                self.execution_history.append({
                    "task_id": task_id,
                    "start_time": start_time,
                    "end_time": end_time,
                })

            return pb2.TileResult(
                task_id=self.override_result_task_id or task_id,
                rows=m,
                cols=n,
                result_bytes=c_bytes,
                worker_signature=sig,
                execution_time_ms=max(exec_time_ms, 1),
            )
        finally:
            with self._lock:
                self._task_metadata.pop(task_id, None)

    def LeaseMatrix(
        self,
        request: pb2.LeaseMatrixRequest,
        context: Optional[grpc.ServicerContext] = None,
    ) -> pb2.LeaseMatrixResponse:
        """
        Pins matrix A in memory across iterative steps.
        Enforces dimension, dtype, and TTL metadata.
        """
        lease_id = str(request.lease_id).strip()
        if not lease_id:
            msg = "lease_id must not be empty"
            if context:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, msg)
            raise ValueError(msg)

        with self._lock:
            self.lease_calls += 1

        if self.reject_lease:
            msg = self.lease_reject_message or "Worker rejected lease"
            if context:
                context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, msg)
            return pb2.LeaseMatrixResponse(
                lease_id=lease_id,
                is_accepted=False,
                expires_at_unix_ms=0,
                worker_pubkey=self.signer.get_pubkey_hex(),
                message=msg,
            )

        rows = int(request.rows)
        cols = int(request.cols)
        dtype = str(request.dtype).lower().strip() or "fp32"
        raw_bytes = request.matrix_bytes or b""
        ttl_seconds = float(request.ttl_seconds) if request.ttl_seconds > 0 else 3600.0

        np_dtype = np.float32 if dtype in ("fp32", "bf16") else (np.float16 if dtype == "fp16" else np.float64)
        bytes_per_elem = 4 if dtype in ("fp32", "bf16") else (2 if dtype == "fp16" else 8)
        expected_bytes = rows * cols * bytes_per_elem

        if len(raw_bytes) != expected_bytes:
            msg = f"Payload size mismatch: expected {expected_bytes} bytes for {rows}x{cols} ({dtype}), got {len(raw_bytes)} bytes"
            if context:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, msg)
            raise ValueError(msg)

        matrix_arr = np.frombuffer(raw_bytes, dtype=np_dtype).reshape((rows, cols)).copy()
        if dtype == "bf16":
            u32_view = matrix_arr.view(np.uint32)
            u32_masked = u32_view & 0xFFFF0000
            matrix_arr = u32_masked.view(np.float32)

        now = time.time()
        expires_at = now + ttl_seconds
        expires_at_unix_ms = int(expires_at * 1000)

        with self._lock:
            self._pinned_sessions[lease_id] = {
                "matrix": matrix_arr,
                "rows": rows,
                "cols": cols,
                "dtype": dtype,
                "ttl_seconds": ttl_seconds,
                "created_at": now,
                "expires_at": expires_at,
                "expires_at_unix_ms": expires_at_unix_ms,
                "memory_bytes": int(matrix_arr.nbytes),
                "step_count": 0,
                "is_zeroized": False,
            }
            self.matrix_upload_counts[lease_id] += 1
            self.uploaded_bytes_total += len(raw_bytes)
            self.recorded_leases.append(lease_id)

        return pb2.LeaseMatrixResponse(
            lease_id=lease_id,
            is_accepted=True,
            expires_at_unix_ms=expires_at_unix_ms,
            worker_pubkey=self.signer.get_pubkey_hex(),
            message=f"Lease accepted for {rows}x{cols} ({dtype})",
        )

    def StepIterative(
        self,
        request: pb2.IterativeStepRequest,
        context: Optional[grpc.ServicerContext] = None,
    ) -> pb2.IterativeStepResponse:
        """
        Executes single matrix-vector multiplication step v_{t+1} = A * v_t
        using the pinned matrix tensor. Signs result with Secp256k1 key.
        """
        lease_id = str(request.lease_id).strip()
        step_index = int(request.step_index)

        if not lease_id:
            msg = "lease_id must not be empty"
            if context:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, msg)
            raise ValueError(msg)

        with self._lock:
            self.step_calls += 1
            session = self._pinned_sessions.get(lease_id)

        if session is None or session.get("is_zeroized"):
            msg = f"Active lease '{lease_id}' not found or already released"
            if context:
                context.abort(grpc.StatusCode.NOT_FOUND, msg)
            raise KeyError(msg)

        # Expiration check
        now = time.time()
        if (
            self.expire_lease_early
            or now > session["expires_at"]
            or (self.expire_lease_after_step is not None and session["step_count"] >= self.expire_lease_after_step)
        ):
            with self._lock:
                self._pinned_sessions.pop(lease_id, None)
            msg = f"Lease '{lease_id}' has expired"
            if context:
                context.abort(grpc.StatusCode.DEADLINE_EXCEEDED, msg)
            raise RuntimeError(msg)

        if self.delay_step_seconds > 0:
            time.sleep(self.delay_step_seconds)

        cols = session["cols"]
        rows = session["rows"]
        dtype = session["dtype"]
        matrix_a = session["matrix"]

        if request.vector_length and int(request.vector_length) != cols:
            msg = f"Vector length mismatch: matrix cols {cols} != request vector_length {request.vector_length}"
            if context:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, msg)
            raise ValueError(msg)

        np_dtype = np.float32 if dtype in ("fp32", "bf16") else (np.float16 if dtype == "fp16" else np.float64)
        bytes_per_elem = 4 if dtype in ("fp32", "bf16") else (2 if dtype == "fp16" else 8)
        expected_bytes = cols * bytes_per_elem

        raw_vec_bytes = request.vector_bytes or b""
        if len(raw_vec_bytes) != expected_bytes:
            msg = f"Vector bytes size mismatch: expected {expected_bytes} bytes for length {cols} ({dtype}), got {len(raw_vec_bytes)} bytes"
            if context:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, msg)
            raise ValueError(msg)

        v_t = np.frombuffer(raw_vec_bytes, dtype=np_dtype).reshape((cols,))

        t_start = time.perf_counter()
        if dtype == "bf16":
            a_f32 = matrix_a.astype(np.float32)
            v_f32 = v_t.astype(np.float32)
            res_f32 = np.matmul(a_f32, v_f32)
            u32_res = res_f32.view(np.uint32) & 0xFFFF0000
            res_vec = u32_res.view(np.float32)
        elif dtype == "fp16":
            res_vec = np.matmul(matrix_a.astype(np.float32), v_t.astype(np.float32)).astype(np.float16)
        else:
            res_vec = np.matmul(matrix_a, v_t)
        exec_ms = max(int((time.perf_counter() - t_start) * 1000), 1)

        # Fault Injection: Corrupt state vector
        should_tamper = (
            (self.tamper_step_index is not None and self.tamper_step_index == step_index)
            or (step_index in self.tamper_step_indices)
        )
        if should_tamper:
            res_vec = res_vec.copy()
            perturbation = getattr(self, "tamper_step_perturbation", getattr(self, "tamper_perturbation", 5.0))
            res_vec[0] += perturbation

        result_bytes = res_vec.tobytes()
        r_hash = hashlib.sha256(result_bytes).hexdigest()

        # Sign result over (lease_id, step_index, sha256(result_bytes))
        if self.tamper_step_signature:
            sig = "00" * 64
        else:
            payload_to_sign = f"{lease_id}:{step_index}:{r_hash}".encode("utf-8")
            sig = self.signer.sign_message(payload_to_sign)

        with self._lock:
            session["step_count"] += 1
            self.uploaded_bytes_total += len(raw_vec_bytes)
            self.recorded_steps.append((lease_id, step_index))

        return pb2.IterativeStepResponse(
            lease_id=lease_id,
            step_index=step_index,
            result_vector_bytes=result_bytes,
            vector_length=rows,
            worker_signature=sig,
            execution_time_ms=exec_ms,
        )

    def ReleaseLease(
        self,
        request: pb2.ReleaseLeaseRequest,
        context: Optional[grpc.ServicerContext] = None,
    ) -> pb2.ReleaseLeaseResponse:
        """
        Deallocates pinned matrix tensor from memory and reclaims bytes.
        """
        lease_id = str(request.lease_id).strip()
        if not lease_id:
            msg = "lease_id must not be empty"
            if context:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, msg)
            raise ValueError(msg)

        with self._lock:
            self.release_calls += 1
            session = self._pinned_sessions.pop(lease_id, None)
            if session is not None:
                bytes_freed = int(session.get("memory_bytes", 0))
                if "matrix" in session and session["matrix"] is not None and isinstance(session["matrix"], np.ndarray):
                    session["matrix"].fill(0)
                session["is_zeroized"] = True
                self.recorded_releases.append(lease_id)
                is_released = True
                msg = f"Lease '{lease_id}' successfully released; {bytes_freed} bytes freed"
            else:
                bytes_freed = 0
                is_released = False
                msg = f"Lease '{lease_id}' was not active or already released"

        return pb2.ReleaseLeaseResponse(
            lease_id=lease_id,
            is_released=is_released,
            bytes_freed=bytes_freed,
            message=msg,
        )


class MockWorkerNode:
    """
    High-level test double container that runs an in-process gRPC MatrixComputeService
    server on an ephemeral local port with full fault injection control.
    """

    def __init__(
        self,
        port: int = 0,
        privkey_hex: Optional[str] = None,
        settlement_service: Optional[MockSettlementService] = None,
        verification_key: str = "settlement-secret-verification-key-12345",
        vram_mb: int = 24576,
        gpu_model: str = "NVIDIA RTX 4090",
        peak_fp32_tflops: float = 82.6,
        sats_per_gflop: float = 0.0005,
        min_task_sats: int = 10,
        lightning_routing_pubkey: Optional[str] = None,
    ) -> None:
        eff_vk = verification_key
        if settlement_service is not None and hasattr(settlement_service, "verification_key"):
            if verification_key == "settlement-secret-verification-key-12345" or not verification_key:
                eff_vk = settlement_service.verification_key
        self.signer = _InternalWorkerSigner(privkey_hex)
        self.servicer = MockWorkerServicer(
            signer=self.signer,
            settlement_service=settlement_service,
            verification_key=eff_vk,
            min_price_rate=sats_per_gflop,
            min_task_sats=min_task_sats,
        )
        self._requested_port = port
        self.port: int = 0
        self.server: Optional[grpc.Server] = None
        self._thread_pool = futures.ThreadPoolExecutor(max_workers=4)

        # Specifications
        self.vram_mb = vram_mb
        self.gpu_model = gpu_model
        self.peak_fp32_tflops = peak_fp32_tflops
        self.sats_per_gflop = sats_per_gflop
        self.min_task_sats = min_task_sats
        self.lightning_routing_pubkey = lightning_routing_pubkey or f"02ln{self.signer.get_pubkey_hex()[:16]}"
        self.staked_amount_sats = 100000
        self.bond_txid = f"mock_bond_{self.signer.get_pubkey_hex()[:12]}"

    @property
    def pubkey(self) -> str:
        return self.signer.get_pubkey_hex()

    @property
    def privkey(self) -> str:
        return self.signer.get_privkey_hex()

    @property
    def target(self) -> str:
        return f"127.0.0.1:{self.port}"

    @property
    def multiaddr(self) -> str:
        return f"/ip4/127.0.0.1/tcp/{self.port}"

    def start(self) -> MockWorkerNode:
        if self.server is not None:
            return self
        self.server = grpc.server(
            self._thread_pool,
            options=[
                ("grpc.max_receive_message_length", 512 * 1024 * 1024),
                ("grpc.max_send_message_length", 512 * 1024 * 1024),
            ],
        )
        pb2_grpc.add_MatrixComputeServiceServicer_to_server(self.servicer, self.server)
        self.port = self.server.add_insecure_port(f"127.0.0.1:{self._requested_port}")
        self.server.start()
        return self

    def stop(self, grace: Optional[float] = None) -> None:
        if self.server is not None:
            self.server.stop(grace or 0.0)
            self._thread_pool.shutdown(wait=False)
            self.server = None

    def __enter__(self) -> MockWorkerNode:
        return self.start()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()

    def create_channel(self) -> grpc.Channel:
        return grpc.insecure_channel(self.target)

    def create_stub(self) -> pb2_grpc.MatrixComputeServiceStub:
        return pb2_grpc.MatrixComputeServiceStub(self.create_channel())

    def get_stub(self) -> pb2_grpc.MatrixComputeServiceStub:
        """Alias for create_stub."""
        return self.create_stub()

    def publish_to_relay(self, relay: MockNostrRelay, d_tag: str = "matrix-worker") -> str:
        payload = WorkerAnnouncePayload(
            event="WORKER_ANNOUNCE",
            worker_pubkey=self.pubkey,
            p2p_multiaddr=self.multiaddr,
            lightning_routing_pubkey=self.lightning_routing_pubkey,
            hardware=HardwareSpecs(
                gpu_model=self.gpu_model,
                vram_mb=self.vram_mb,
                peak_fp32_tflops=self.peak_fp32_tflops,
                attestation_proof=f"mock_attest_{self.pubkey[:16]}",
                timestamp=int(time.time()),
            ),
            pricing=PricingSpecs(
                sats_per_gflop=self.sats_per_gflop,
                min_task_sats=self.min_task_sats,
            ),
            bond_status=BondStatus(
                staked_amount_sats=self.staked_amount_sats,
                bond_txid=self.bond_txid,
            ),
            timestamp=int(time.time()),
        )
        return relay.publish_announce(payload)

    # Fault injection helper methods
    def set_tamper_result(self, tamper: bool = True, perturbation: float = 1.0) -> None:
        self.servicer.tamper_result = tamper
        self.servicer.tamper_perturbation = perturbation

    def set_tamper_signature(self, tamper: bool = True) -> None:
        self.servicer.tamper_signature = tamper

    def set_drop_connection(self, drop: bool = True) -> None:
        self.servicer.drop_connection = drop

    def set_reject_handshake(self, reject: bool = True, message: str = "Worker busy") -> None:
        self.servicer.reject_handshake = reject
        self.servicer.handshake_reject_message = message

    def set_delay(self, seconds: float) -> None:
        self.servicer.delay_seconds = seconds

    def set_bypass_l402(self, bypass: bool = True) -> None:
        self.servicer.bypass_l402 = bypass

    def set_override_result_task_id(self, task_id: Optional[str]) -> None:
        self.servicer.override_result_task_id = task_id

    def set_tamper_task_ids(self, task_ids: Any) -> None:
        self.servicer.tamper_task_ids = set(task_ids)

    def set_fault_mode(self, mode: str) -> None:
        """Configures worker fault injection mode by name."""
        norm = str(mode).lower().strip()
        if norm in ("corrupt_result", "tamper", "corrupt"):
            self.set_tamper_result(True, perturbation=5.0)
        elif norm in ("disconnect", "drop", "drop_connection"):
            self.set_drop_connection(True)
        elif norm in ("tamper_signature", "bad_signature"):
            self.set_tamper_signature(True)
        elif norm in ("reject_handshake", "busy"):
            self.set_reject_handshake(True)
        elif norm in ("none", "clean", "reset"):
            self.set_tamper_result(False)
            self.set_drop_connection(False)
            self.set_tamper_signature(False)
            self.set_reject_handshake(False)

    def set_tamper_on_task_count(self, count: Optional[int]) -> None:
        self.servicer.tamper_on_task_count = count

    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Returns a copy of the servicer's task execution intervals."""
        with self.servicer._lock:
            return list(self.servicer.execution_history)

    def clear_execution_history(self) -> None:
        """Clears recorded execution history."""
        with self.servicer._lock:
            self.servicer.execution_history.clear()

    # Phase 3 Stateful Iterative Fault Injection & Introspection
    def set_tamper_step_index(self, step_index: Optional[int], perturbation: float = 5.0) -> None:
        """Injects arithmetic corruption into worker output at a specific step index."""
        self.servicer.tamper_step_index = step_index
        self.servicer.tamper_step_perturbation = perturbation

    def set_tamper_step_indices(self, indices: Any, perturbation: float = 5.0) -> None:
        """Injects corruption into multiple step indices."""
        self.servicer.tamper_step_indices = set(indices)
        self.servicer.tamper_step_perturbation = perturbation

    def set_tamper_step_signature(self, tamper: bool = True) -> None:
        """Produces invalid 00-filled Secp256k1 signatures on iterative steps."""
        self.servicer.tamper_step_signature = tamper

    def set_reject_lease(self, reject: bool = True, message: str = "Worker VRAM capacity exhausted") -> None:
        """Simulates worker rejecting LeaseMatrix requests."""
        self.servicer.reject_lease = reject
        self.servicer.lease_reject_message = message

    def set_expire_lease_early(self, expire: bool = True) -> None:
        """Simulates premature lease expiration during iterative execution."""
        self.servicer.expire_lease_early = expire

    def set_expire_lease_after_step(self, step: Optional[int]) -> None:
        """Simulates lease expiration after a given number of steps."""
        self.servicer.expire_lease_after_step = step

    def set_delay_step(self, seconds: float) -> None:
        """Injects artificial delay into StepIterative calls."""
        self.servicer.delay_step_seconds = seconds

    def get_matrix_upload_count(self, lease_id: str) -> int:
        """Returns number of times matrix A was uploaded under lease_id."""
        with self.servicer._lock:
            return self.servicer.matrix_upload_counts.get(lease_id, 0)

    def get_pinned_sessions(self) -> Dict[str, Any]:
        """Returns snapshot of current pinned sessions."""
        with self.servicer._lock:
            return dict(self.servicer._pinned_sessions)

    def get_recorded_steps(self) -> List[Tuple[str, int]]:
        """Returns sequence of executed (lease_id, step_index) steps."""
        with self.servicer._lock:
            return list(self.servicer.recorded_steps)

