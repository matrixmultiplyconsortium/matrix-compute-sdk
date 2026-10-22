"""
P2P gRPC Task Dispatcher for matrix-compute-sdk.

Handles RequestTileTask pre-negotiation, ephemeral X25519 ECDH handshake,
hardware attestation evidence verification via AttestationVerifier,
progressive AES-256-GCM encrypted chunk streaming, result decryption,
L402 metadata formatting, and worker Secp256k1 signature validation.
Supports clean unencrypted fallback when require_confidential_compute is False.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import logging
import math
import threading
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import grpc
import numpy as np

try:
    from matrix_contracts.v1 import pb2, pb2_grpc
except ImportError:
    from matrix_contracts.proto import compute_service_pb2 as pb2
    from matrix_contracts.proto import compute_service_pb2_grpc as pb2_grpc

from matrix_compute.crypto.attestation_verifier import AttestationVerifier
from matrix_compute.crypto.session import (
    CryptoSessionError,
    DecryptionError,
    EncryptedPayload,
    KeyExchangeError,
    SecurityPolicyViolationError,
    SessionContext,
    aes_gcm_decrypt,
    aes_gcm_encrypt,
    decrypt_result,
    derive_session_keys,
    encrypt_chunk,
    generate_challenger_nonce,
    generate_ephemeral_keypair,
)
from matrix_compute.crypto.signer import sha256_digest, verify_signature, verify_tile_result_signature
from matrix_compute.exceptions import (
    AttestationError,
    AttestationEvidenceMalformedError,
    DispatchError,
    StreamingError,
    TaskDispatchError,
    TaskTimeoutError,
    WorkerConnectionError,
    WorkerSignatureError,
    WorkerTaskRejectedError,
)
from matrix_compute.p2p.channel_pool import (
    ChannelPool,
    extract_circuit_target_worker,
    parse_multiaddr,
)
from matrix_compute.tiling.partitioner import TileTaskPlan
from matrix_compute.tiling.precision import get_bytes_per_element, get_numpy_dtype, normalize_dtype

logger = logging.getLogger(__name__)


@dataclass
class DispatchedTileResult:
    """Result of a successfully dispatched and verified tile task."""

    task_id: str
    matrix_c: np.ndarray
    rows: int
    cols: int
    result_bytes: bytes
    worker_signature: str
    execution_time_ms: int
    raw_result: pb2.TileResult


@dataclass
class IterativeSessionContext:
    """Session context tracking state for an active matrix lease."""

    lease_id: str
    rows: int
    cols: int
    dtype: str
    is_confidential: bool = False
    client_ephemeral_priv: Optional[Any] = None
    client_ephemeral_pub_bytes: bytes = b""
    worker_ephemeral_pub_bytes: bytes = b""
    k_c2w: Optional[bytes] = None
    k_w2c: Optional[bytes] = None
    expires_at_unix_ms: int = 0


class IterativeStepResponseWrapper:
    """Wrapper around IterativeStepResponse that supports both direct attribute access and tuple unpacking."""

    def __init__(self, response: pb2.IterativeStepResponse, result_vec: np.ndarray) -> None:
        self._response = response
        self._result_vec = result_vec

    @property
    def result_vec(self) -> np.ndarray:
        return self._result_vec

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    def __iter__(self) -> Iterator[Any]:
        yield self._result_vec
        yield self._response

    def __getitem__(self, index: int) -> Any:
        if index == 0:
            return self._result_vec
        elif index == 1:
            return self._response
        raise IndexError(f"Index out of range: {index}")


class GrpcTaskDispatcher:
    """
    Dispatches individual matrix tile computation tasks to worker nodes over gRPC.
    Manages confidential computing handshakes, attestation verification,
    AES-256-GCM chunk encryption, and result decryption.
    """

    def __init__(
        self,
        channel_pool: Optional[ChannelPool] = None,
        consumer_pubkey: str = "",
        connect_timeout: float = 5.0,
        task_timeout: float = 30.0,
        chunk_size_bytes: int = 1024 * 1024,
        require_confidential_compute: Optional[bool] = None,
        required_tee_type: Optional[str] = None,
        attestation_verifier: Optional[AttestationVerifier] = None,
        config: Optional[Any] = None,
        signer: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        self.channel_pool = channel_pool or ChannelPool()
        if signer is not None and not consumer_pubkey:
            if hasattr(signer, "get_pubkey_hex"):
                consumer_pubkey = signer.get_pubkey_hex()
            elif hasattr(signer, "pubkey"):
                consumer_pubkey = str(signer.pubkey)
        self.consumer_pubkey = consumer_pubkey
        self.connect_timeout = connect_timeout
        self.task_timeout = task_timeout
        self.chunk_size_bytes = max(chunk_size_bytes, 1024)

        # Resolve confidential compute settings from config or explicit arguments
        if config is not None:
            self.require_confidential_compute = (
                config.require_confidential_compute
                if require_confidential_compute is None
                else require_confidential_compute
            )
            self.required_tee_type = (
                config.required_tee_type
                if required_tee_type is None
                else required_tee_type
            )
        else:
            self.require_confidential_compute = (
                False if require_confidential_compute is None else require_confidential_compute
            )
            self.required_tee_type = required_tee_type

        self.attestation_verifier = attestation_verifier
        self._sessions: Dict[str, SessionContext] = {}
        self._iterative_sessions: Dict[str, IterativeSessionContext] = {}
        self._sessions_lock = threading.Lock()

    def _build_metadata(
        self,
        task_or_lease_id: str,
        auth_header: Optional[str] = None,
        worker_pubkey: str = "",
        worker_or_target: Any = None,
        is_lease: bool = False,
        extra_metadata: Optional[List[Tuple[str, str]]] = None,
    ) -> List[Tuple[str, str]]:
        key_name = "x-lease-id" if is_lease else "x-task-id"
        meta = [(key_name, str(task_or_lease_id).strip())]
        if is_lease:
            meta.append(("x-task-id", str(task_or_lease_id).strip()))
        target_worker = (
            worker_pubkey
            or (extract_circuit_target_worker(worker_or_target) if worker_or_target else "")
            or getattr(worker_or_target, "worker_pubkey", "")
            or getattr(worker_or_target, "pubkey", "")
        )
        if target_worker:
            meta.append(("x-target-worker", str(target_worker).strip().lower()))
        if auth_header:
            meta.append(("authorization", auth_header.strip()))
        if extra_metadata:
            meta.extend(extra_metadata)
        return meta

    def request_tile_task(
        self,
        worker_or_target: Union[str, Any],
        task_id: str,
        m: int,
        k: int,
        n: int,
        dtype: str = "fp32",
        worker_pubkey: str = "",
        timeout: Optional[float] = None,
    ) -> pb2.TileTaskAck:
        """
        Step 1: Handshake negotiation via RequestTileTask.
        If require_confidential_compute is True:
          - Generates ephemeral X25519 keypair and challenger nonce.
          - Sends key and nonce in TileTaskRequest.
          - Validates worker attestation evidence fail-closed.
          - Derives directional AES-256-GCM session keys.
        """
        target = parse_multiaddr(worker_or_target)
        stub = self.channel_pool.get_stub(target)
        norm_dtype = normalize_dtype(dtype)

        session_ctx: Optional[SessionContext] = None

        if self.require_confidential_compute:
            client_priv, client_pub = generate_ephemeral_keypair()
            nonce = generate_challenger_nonce()
            session_ctx = SessionContext(
                task_id=task_id,
                client_ephemeral_priv=client_priv,
                client_ephemeral_pub_bytes=client_pub,
                attestation_nonce=nonce,
                is_confidential=True,
                tee_type=self.required_tee_type,
            )
            req = pb2.TileTaskRequest(
                task_id=task_id,
                matrix_a_rows=int(m),
                matrix_a_cols=int(k),
                matrix_b_cols=int(n),
                dtype=norm_dtype,
                require_confidential_compute=True,
                client_ephemeral_pubkey=client_pub,
                attestation_nonce=nonce,
                required_tee_type=self.required_tee_type or "",
            )
        else:
            req = pb2.TileTaskRequest(
                task_id=task_id,
                matrix_a_rows=int(m),
                matrix_a_cols=int(k),
                matrix_b_cols=int(n),
                dtype=norm_dtype,
                require_confidential_compute=False,
            )

        eff_timeout = timeout if timeout is not None else self.connect_timeout
        metadata = self._build_metadata(
            task_id,
            worker_pubkey=worker_pubkey,
            worker_or_target=worker_or_target,
            is_lease=False,
        )
        try:
            try:
                ack = stub.RequestTileTask(req, metadata=metadata, timeout=eff_timeout)
            except TypeError:
                ack = stub.RequestTileTask(req, timeout=eff_timeout)
        except grpc.RpcError as rpc_err:
            self._handle_rpc_error(rpc_err, task_id, "RequestTileTask", worker_pubkey=worker_pubkey, multiaddr=target)

        if not ack.is_ready:
            rejection_msg = ack.message or "Worker not ready"
            raise WorkerTaskRejectedError(
                f"Worker rejected task '{task_id}': {rejection_msg}",
                task_id=task_id,
                worker_pubkey=worker_pubkey or getattr(ack, "worker_pubkey", ""),
                reason=rejection_msg,
            )

        # Handle Confidential Computing session derivation
        if self.require_confidential_compute and session_ctx is not None:
            if not ack.worker_ephemeral_pubkey or len(ack.worker_ephemeral_pubkey) != 32:
                raise KeyExchangeError(f"Worker did not provide 32-byte ephemeral public key for task '{task_id}'")

            if not ack.attestation_evidence:
                raise AttestationEvidenceMalformedError(f"Worker returned empty attestation evidence for task '{task_id}'")

            # Verify hardware attestation evidence fail-closed
            if self.attestation_verifier is None:
                self.attestation_verifier = AttestationVerifier()

            self.attestation_verifier.verify(
                evidence=ack.attestation_evidence,
                expected_nonce=session_ctx.attestation_nonce,
                expected_client_pubkey=ack.worker_ephemeral_pubkey,
                task_id=task_id,
            )

            # Derive directional session keys
            assert session_ctx.client_ephemeral_priv is not None
            keys = derive_session_keys(
                secret_or_priv=session_ctx.client_ephemeral_priv,
                nonce_c_or_pub=ack.worker_ephemeral_pubkey,
                nonce_w_or_salt=session_ctx.attestation_nonce,
            )
            session_ctx.worker_ephemeral_pub_bytes = ack.worker_ephemeral_pubkey
            session_ctx.k_c2w = keys.k_c2w
            session_ctx.k_w2c = keys.k_w2c
            session_ctx.tee_type = ack.tee_type

        with self._sessions_lock:
            self._sessions[task_id] = session_ctx or SessionContext(
                task_id=task_id,
                client_ephemeral_priv=None,
                client_ephemeral_pub_bytes=b"",
                attestation_nonce=b"",
                is_confidential=False,
            )

        return ack

    def stream_tile_chunks(
        self,
        task_id: str,
        matrix_a: np.ndarray,
        matrix_b: np.ndarray,
        dtype: str = "fp32",
        session: Optional[SessionContext] = None,
    ) -> Iterator[pb2.TileDataChunk]:
        """
        Yields serialized TileDataChunk instances.
        Encrypts chunks using AES-256-GCM with fresh 12-byte IVs if session is confidential.
        """
        target_dtype = normalize_dtype(dtype)
        np_dtype = get_numpy_dtype(target_dtype)

        seq = 0
        is_encrypted = (
            session is not None
            and session.is_confidential
            and session.k_c2w is not None
        )

        # Stream Matrix A
        a_bytes = np.ascontiguousarray(matrix_a, dtype=np_dtype).tobytes()
        total_a = len(a_bytes)
        step = self.chunk_size_bytes

        if total_a == 0:
            if is_encrypted and session is not None and session.k_c2w is not None:
                ct, iv, tag = encrypt_chunk(
                    key=session.k_c2w,
                    plaintext=b"",
                    task_id=task_id,
                    sequence_number=seq,
                    is_matrix_a=True,
                    is_final_chunk=False,
                )
                yield pb2.TileDataChunk(
                    task_id=task_id,
                    chunk_bytes=ct,
                    is_matrix_a=True,
                    sequence_number=seq,
                    is_final_chunk=False,
                    is_encrypted=True,
                    iv=iv,
                    tag=tag,
                )
            else:
                yield pb2.TileDataChunk(
                    task_id=task_id,
                    chunk_bytes=b"",
                    is_matrix_a=True,
                    sequence_number=seq,
                    is_final_chunk=False,
                    is_encrypted=False,
                )
            seq += 1
        else:
            for i in range(0, total_a, step):
                chunk_slice = a_bytes[i : i + step]
                if is_encrypted and session is not None and session.k_c2w is not None:
                    ct, iv, tag = encrypt_chunk(
                        key=session.k_c2w,
                        plaintext=chunk_slice,
                        task_id=task_id,
                        sequence_number=seq,
                        is_matrix_a=True,
                        is_final_chunk=False,
                    )
                    yield pb2.TileDataChunk(
                        task_id=task_id,
                        chunk_bytes=ct,
                        is_matrix_a=True,
                        sequence_number=seq,
                        is_final_chunk=False,
                        is_encrypted=True,
                        iv=iv,
                        tag=tag,
                    )
                else:
                    yield pb2.TileDataChunk(
                        task_id=task_id,
                        chunk_bytes=chunk_slice,
                        is_matrix_a=True,
                        sequence_number=seq,
                        is_final_chunk=False,
                        is_encrypted=False,
                    )
                seq += 1

        # Stream Matrix B
        b_bytes = np.ascontiguousarray(matrix_b, dtype=np_dtype).tobytes()
        total_b = len(b_bytes)

        if total_b == 0:
            if is_encrypted and session is not None and session.k_c2w is not None:
                ct, iv, tag = encrypt_chunk(
                    key=session.k_c2w,
                    plaintext=b"",
                    task_id=task_id,
                    sequence_number=seq,
                    is_matrix_a=False,
                    is_final_chunk=True,
                )
                yield pb2.TileDataChunk(
                    task_id=task_id,
                    chunk_bytes=ct,
                    is_matrix_a=False,
                    sequence_number=seq,
                    is_final_chunk=True,
                    is_encrypted=True,
                    iv=iv,
                    tag=tag,
                )
            else:
                yield pb2.TileDataChunk(
                    task_id=task_id,
                    chunk_bytes=b"",
                    is_matrix_a=False,
                    sequence_number=seq,
                    is_final_chunk=True,
                    is_encrypted=False,
                )
            seq += 1
        else:
            total_b_chunks = math.ceil(total_b / step)
            for idx, i in enumerate(range(0, total_b, step)):
                chunk_slice = b_bytes[i : i + step]
                is_final = (idx == total_b_chunks - 1)
                if is_encrypted and session is not None and session.k_c2w is not None:
                    ct, iv, tag = encrypt_chunk(
                        key=session.k_c2w,
                        plaintext=chunk_slice,
                        task_id=task_id,
                        sequence_number=seq,
                        is_matrix_a=False,
                        is_final_chunk=is_final,
                    )
                    yield pb2.TileDataChunk(
                        task_id=task_id,
                        chunk_bytes=ct,
                        is_matrix_a=False,
                        sequence_number=seq,
                        is_final_chunk=is_final,
                        is_encrypted=True,
                        iv=iv,
                        tag=tag,
                    )
                else:
                    yield pb2.TileDataChunk(
                        task_id=task_id,
                        chunk_bytes=chunk_slice,
                        is_matrix_a=False,
                        sequence_number=seq,
                        is_final_chunk=is_final,
                        is_encrypted=False,
                    )
                seq += 1

    def collect_tile_results(
        self,
        tile_result: pb2.TileResult,
        task_id: str,
        m: int,
        n: int,
        dtype: str = "fp32",
        worker_pubkey: str = "",
        matrix_a: Optional[np.ndarray] = None,
        matrix_b: Optional[np.ndarray] = None,
        session: Optional[SessionContext] = None,
    ) -> Tuple[np.ndarray, pb2.TileResult]:
        """
        Validates, decrypts (if encrypted), and unpacks TileResult into NumPy matrix.
        Performs Secp256k1 worker signature verification.
        """
        target_dtype = normalize_dtype(dtype)
        np_dtype = get_numpy_dtype(target_dtype)
        bytes_per_elem = get_bytes_per_element(target_dtype)

        # Validate task_id matches expected task_id
        if str(tile_result.task_id).strip() != str(task_id).strip():
            raise TaskDispatchError(
                f"Worker returned mismatched task_id: expected {task_id}, got {tile_result.task_id}",
                task_id=task_id,
                returned_task_id=tile_result.task_id,
            )

        # Validate dimensions
        if tile_result.rows != m or tile_result.cols != n:
            raise StreamingError(
                f"Worker returned invalid dimensions: expected ({m}, {n}), got ({tile_result.rows}, {tile_result.cols})"
            )

        # Decrypt payload if confidential compute was negotiated or worker returned ciphertext
        if self.require_confidential_compute:
            if not tile_result.is_encrypted:
                raise SecurityPolicyViolationError(
                    f"Confidential computing was required for task '{task_id}', but worker returned unencrypted TileResult",
                    task_id=task_id,
                )
            if session is None or not session.is_confidential or session.k_w2c is None:
                raise DecryptionError(
                    f"Encrypted TileResult received for task '{task_id}', but no valid session key was established",
                    task_id=task_id,
                )
            decrypted_bytes = decrypt_result(
                key=session.k_w2c,
                ciphertext=tile_result.result_bytes,
                iv=tile_result.iv,
                tag=tile_result.tag,
                task_id=task_id,
                rows=tile_result.rows,
                cols=tile_result.cols,
            )
        else:
            if tile_result.is_encrypted:
                if session is not None and session.k_w2c is not None:
                    decrypted_bytes = decrypt_result(
                        key=session.k_w2c,
                        ciphertext=tile_result.result_bytes,
                        iv=tile_result.iv,
                        tag=tile_result.tag,
                        task_id=task_id,
                        rows=tile_result.rows,
                        cols=tile_result.cols,
                    )
                else:
                    raise DecryptionError(
                        f"Unexpected encrypted TileResult for unencrypted task '{task_id}'",
                        task_id=task_id,
                    )
            else:
                decrypted_bytes = tile_result.result_bytes

        # Validate decrypted byte count
        expected_bytes = m * n * bytes_per_elem
        if len(decrypted_bytes) != expected_bytes:
            raise StreamingError(
                f"Worker returned byte count {len(decrypted_bytes)} != expected {expected_bytes}"
            )

        # Cryptographic signature validation
        if worker_pubkey:
            sig_valid = self.verify_worker_signature(
                worker_pubkey=worker_pubkey,
                task_id=task_id,
                result_bytes=decrypted_bytes,
                worker_signature=tile_result.worker_signature,
                matrix_a=matrix_a,
                matrix_b=matrix_b,
                dtype=target_dtype,
                raw_wire_bytes=tile_result.result_bytes if tile_result.is_encrypted else None,
            )
            if not sig_valid:
                raise WorkerSignatureError(
                    f"Worker signature verification failed on tile task '{task_id}'",
                    task_id=task_id,
                    worker_pubkey=worker_pubkey,
                    signature=tile_result.worker_signature,
                )

        c_matrix = np.frombuffer(decrypted_bytes, dtype=np_dtype).reshape((m, n))
        return c_matrix, tile_result

    def stream_execute_tile(
        self,
        worker_or_target: Union[str, Any],
        task_id: str,
        matrix_a: np.ndarray,
        matrix_b: np.ndarray,
        dtype: str = "fp32",
        auth_header: Optional[str] = None,
        worker_pubkey: str = "",
        timeout: Optional[float] = None,
    ) -> Tuple[np.ndarray, pb2.TileResult]:
        """
        Step 2: Progressive chunk streaming, result ingestion, and signature verification.
        """
        target = parse_multiaddr(worker_or_target)
        stub = self.channel_pool.get_stub(target)
        target_dtype = normalize_dtype(dtype)

        m, k = matrix_a.shape
        k2, n = matrix_b.shape
        if k != k2:
            raise ValueError(f"Inner dimension mismatch: {k} != {k2}")

        with self._sessions_lock:
            session = self._sessions.get(task_id)

        metadata = self._build_metadata(
            task_id,
            auth_header=auth_header,
            worker_pubkey=worker_pubkey,
            worker_or_target=worker_or_target,
            is_lease=False,
        )

        eff_timeout = timeout if timeout is not None else self.task_timeout
        try:
            tile_result = stub.ExecuteTileTask(
                self.stream_tile_chunks(
                    task_id=task_id,
                    matrix_a=matrix_a,
                    matrix_b=matrix_b,
                    dtype=target_dtype,
                    session=session,
                ),
                metadata=metadata,
                timeout=eff_timeout,
            )
        except grpc.RpcError as rpc_err:
            self._handle_rpc_error(rpc_err, task_id, "ExecuteTileTask", worker_pubkey=worker_pubkey, multiaddr=target)
        finally:
            with self._sessions_lock:
                self._sessions.pop(task_id, None)

        return self.collect_tile_results(
            tile_result=tile_result,
            task_id=task_id,
            m=m,
            n=n,
            dtype=target_dtype,
            worker_pubkey=worker_pubkey,
            matrix_a=matrix_a,
            matrix_b=matrix_b,
            session=session,
        )

    def execute_tile(
        self,
        worker: Union[str, Any],
        task_id: str,
        A_tile: np.ndarray,
        B_tile: np.ndarray,
        dtype: str = "fp32",
        l402_header: Optional[str] = None,
        worker_pubkey: str = "",
        timeout: Optional[float] = None,
    ) -> pb2.TileResult:
        """
        Executes a tile computation task matching PROJECT.md interface contract:
        execute_tile(worker: MatchedWorker, task_id: str, A_tile: np.ndarray, B_tile: np.ndarray, dtype: str, l402_header: str) -> TileResult
        """
        pubkey = (
            worker_pubkey
            or getattr(worker, "worker_pubkey", getattr(worker, "pubkey", ""))
            or (extract_circuit_target_worker(worker) if worker else "")
            or ""
        )
        m, k = A_tile.shape
        _, n = B_tile.shape
        target_dtype = normalize_dtype(dtype)

        ack = self.request_tile_task(
            worker_or_target=worker,
            task_id=task_id,
            m=m,
            k=k,
            n=n,
            dtype=target_dtype,
            worker_pubkey=pubkey,
            timeout=timeout,
        )
        eff_pubkey = pubkey or getattr(ack, "worker_pubkey", "")

        _, tile_result = self.stream_execute_tile(
            worker_or_target=worker,
            task_id=task_id,
            matrix_a=A_tile,
            matrix_b=B_tile,
            dtype=target_dtype,
            auth_header=l402_header,
            worker_pubkey=eff_pubkey,
            timeout=timeout,
        )
        return tile_result

    def dispatch_tile(
        self,
        target_multiaddr: str,
        worker_pubkey: str,
        task_plan: TileTaskPlan,
        macaroon: str,
        payment_hash: str,
        dtype: Optional[str] = None,
    ) -> Tuple[np.ndarray, pb2.TileResult]:
        """
        Executes a complete tile task over gRPC:
          1. Handshake with RequestTileTask
          2. Progressive chunk streaming to ExecuteTileTask with L402 auth header
          3. Result verification, dimension validation, and Secp256k1 signature verification

        Returns:
            Tuple of (unpacked_result_matrix_c, raw_tile_result_pb2)
        """
        target_dtype = normalize_dtype(dtype or task_plan.dtype)

        # Stage 1: Handshake
        ack = self.request_tile_task(
            worker_or_target=target_multiaddr,
            task_id=task_plan.task_id,
            m=task_plan.m_i,
            k=task_plan.k_j,
            n=task_plan.n_k,
            dtype=target_dtype,
            worker_pubkey=worker_pubkey,
        )
        effective_worker_pubkey = worker_pubkey or getattr(ack, "worker_pubkey", "")

        # Stage 2: Stream Execution
        clean_macaroon = macaroon.strip()
        clean_hash = payment_hash.strip().lower()
        auth_header = f"L402 {clean_macaroon}:{clean_hash}"

        return self.stream_execute_tile(
            worker_or_target=target_multiaddr,
            task_id=task_plan.task_id,
            matrix_a=task_plan.a_tile,
            matrix_b=task_plan.b_tile,
            dtype=target_dtype,
            auth_header=auth_header,
            worker_pubkey=effective_worker_pubkey,
        )

    def verify_worker_signature(
        self,
        worker_pubkey: str,
        task_id: str,
        result_bytes: bytes,
        worker_signature: str,
        matrix_a: Optional[np.ndarray] = None,
        matrix_b: Optional[np.ndarray] = None,
        dtype: Optional[str] = None,
        raw_wire_bytes: Optional[bytes] = None,
    ) -> bool:
        """
        Validates worker's Secp256k1 signature.
        Checks canonical format f"{task_id}:{c_hash}", raw wire bytes, and fallback formats.
        """
        if not worker_pubkey or not worker_signature:
            return False

        # 1. Primary canonical signature check on plaintext result bytes
        if verify_tile_result_signature(
            worker_pubkey=worker_pubkey,
            task_id=task_id,
            result_bytes=result_bytes,
            signature_hex=worker_signature,
        ):
            return True

        # 2. Check if signature was produced on the pre-hashed payload: sha256(f"{task_id}:{c_hash}")
        c_hash = sha256_digest(result_bytes)
        payload = f"{str(task_id).strip()}:{c_hash.strip()}".encode("utf-8")
        if verify_signature(worker_pubkey, hashlib.sha256(payload).digest(), worker_signature):
            return True

        # 3. If raw wire bytes (ciphertext) provided, check signature over ciphertext
        if raw_wire_bytes is not None:
            if verify_tile_result_signature(
                worker_pubkey=worker_pubkey,
                task_id=task_id,
                result_bytes=raw_wire_bytes,
                signature_hex=worker_signature,
            ):
                return True
            raw_hash = sha256_digest(raw_wire_bytes)
            raw_payload = f"{str(task_id).strip()}:{raw_hash.strip()}".encode("utf-8")
            if verify_signature(worker_pubkey, hashlib.sha256(raw_payload).digest(), worker_signature):
                return True

        # 4. Fallback composite signature check
        if matrix_a is not None and matrix_b is not None:
            try:
                np_dtype = get_numpy_dtype(normalize_dtype(dtype or "fp32"))
                a_bytes = np.ascontiguousarray(matrix_a, dtype=np_dtype).tobytes()
                b_bytes = np.ascontiguousarray(matrix_b, dtype=np_dtype).tobytes()

                composite_payload = hashlib.sha256(
                    hashlib.sha256(a_bytes).digest()
                    + hashlib.sha256(b_bytes).digest()
                    + hashlib.sha256(result_bytes).digest()
                    + task_id.encode("utf-8")
                ).digest()

                if verify_signature(worker_pubkey, composite_payload, worker_signature):
                    return True
            except Exception as ex:
                logger.debug("Fallback composite signature verification error: %s", ex)

        return False

    def lease_matrix(
        self,
        worker_or_target: Union[str, Any],
        lease_id: str,
        matrix_a: np.ndarray,
        dtype: str = "fp32",
        ttl_seconds: Optional[int] = None,
        auth_header: Optional[str] = None,
        worker_pubkey: str = "",
        timeout: Optional[float] = None,
        require_confidential_compute: Optional[bool] = None,
    ) -> pb2.LeaseMatrixResponse:
        """
        Requests remote GPU VRAM session lease and uploads matrix A strictly once.
        """
        target = parse_multiaddr(worker_or_target)
        stub = self.channel_pool.get_stub(target)
        target_dtype = normalize_dtype(dtype)
        np_dtype = get_numpy_dtype(target_dtype)

        rows, cols = matrix_a.shape
        eff_ttl = int(ttl_seconds) if ttl_seconds is not None and ttl_seconds > 0 else 3600

        is_confidential = (
            self.require_confidential_compute
            if require_confidential_compute is None
            else require_confidential_compute
        )

        client_priv = None
        client_pub_bytes = b""
        if is_confidential:
            client_priv, client_pub_bytes = generate_ephemeral_keypair()

        # Serialize contiguous matrix bytes
        matrix_bytes = np.ascontiguousarray(matrix_a, dtype=np_dtype).tobytes()

        req = pb2.LeaseMatrixRequest(
            lease_id=str(lease_id).strip(),
            rows=int(rows),
            cols=int(cols),
            dtype=target_dtype,
            matrix_bytes=matrix_bytes,
            ttl_seconds=int(eff_ttl),
            require_confidential_compute=bool(is_confidential),
            client_ephemeral_pubkey=client_pub_bytes,
        )

        metadata = self._build_metadata(
            lease_id,
            auth_header=auth_header,
            worker_pubkey=worker_pubkey,
            worker_or_target=worker_or_target,
            is_lease=True,
        )

        eff_timeout = timeout if timeout is not None else self.task_timeout

        try:
            response = stub.LeaseMatrix(req, metadata=metadata, timeout=eff_timeout)
        except grpc.RpcError as rpc_err:
            self._handle_rpc_error(
                rpc_err, lease_id, "LeaseMatrix", worker_pubkey=worker_pubkey, multiaddr=target
            )

        if not response.is_accepted:
            msg = response.message or "Worker rejected matrix lease"
            raise WorkerTaskRejectedError(
                f"Worker rejected matrix lease '{lease_id}': {msg}",
                task_id=lease_id,
                worker_pubkey=worker_pubkey or getattr(response, "worker_pubkey", ""),
                reason=msg,
            )

        k_c2w = None
        k_w2c = None
        if is_confidential:
            if not response.worker_ephemeral_pubkey or len(response.worker_ephemeral_pubkey) != 32:
                raise KeyExchangeError(
                    f"Worker did not provide 32-byte ephemeral public key for lease '{lease_id}'"
                )
            assert client_priv is not None
            salt = hashlib.sha256(f"iterative:{lease_id}".encode("utf-8")).digest()
            keys = derive_session_keys(
                secret_or_priv=client_priv,
                nonce_c_or_pub=response.worker_ephemeral_pubkey,
                nonce_w_or_salt=salt,
            )
            k_c2w = keys.k_c2w
            k_w2c = keys.k_w2c

        session_ctx = IterativeSessionContext(
            lease_id=lease_id,
            rows=rows,
            cols=cols,
            dtype=target_dtype,
            is_confidential=is_confidential,
            client_ephemeral_priv=client_priv,
            client_ephemeral_pub_bytes=client_pub_bytes,
            worker_ephemeral_pub_bytes=response.worker_ephemeral_pubkey,
            k_c2w=k_c2w,
            k_w2c=k_w2c,
            expires_at_unix_ms=int(response.expires_at_unix_ms),
        )

        with self._sessions_lock:
            self._iterative_sessions[lease_id] = session_ctx

        return response

    def step_iterative(
        self,
        worker_or_target: Union[str, Any],
        lease_id: str,
        step_index: int,
        vector_v: np.ndarray,
        dtype: str = "fp32",
        auth_header: Optional[str] = None,
        worker_pubkey: str = "",
        timeout: Optional[float] = None,
    ) -> IterativeStepResponseWrapper:
        """
        Executes a single remote matrix-vector multiplication step: v_{t+1} = A * v_t.
        """
        target = parse_multiaddr(worker_or_target)
        stub = self.channel_pool.get_stub(target)
        target_dtype = normalize_dtype(dtype)
        np_dtype = get_numpy_dtype(target_dtype)
        bytes_per_elem = get_bytes_per_element(target_dtype)

        # Ensure contiguous 1D array
        vec_1d = np.ascontiguousarray(vector_v.flatten(), dtype=np_dtype)
        vector_len = vec_1d.size
        raw_vec_bytes = vec_1d.tobytes()

        with self._sessions_lock:
            session = self._iterative_sessions.get(lease_id)

        is_encrypted = False
        iv = b""
        tag = b""
        payload_bytes = raw_vec_bytes

        if session is not None and session.is_confidential and session.k_c2w is not None:
            step_aad = f"{lease_id}:step:{step_index}".encode("utf-8")
            enc = aes_gcm_encrypt(
                key=session.k_c2w,
                plaintext=raw_vec_bytes,
                aad=step_aad,
            )
            payload_bytes = enc.ciphertext
            iv = enc.iv
            tag = enc.tag
            is_encrypted = True

        req = pb2.IterativeStepRequest(
            lease_id=str(lease_id).strip(),
            step_index=int(step_index),
            vector_bytes=payload_bytes,
            vector_length=int(vector_len),
            dtype=target_dtype,
            is_encrypted=is_encrypted,
            iv=iv,
            tag=tag,
        )

        metadata = self._build_metadata(
            lease_id,
            auth_header=auth_header,
            worker_pubkey=worker_pubkey,
            worker_or_target=worker_or_target,
            is_lease=True,
            extra_metadata=[("x-step-index", str(step_index))],
        )

        eff_timeout = timeout if timeout is not None else self.task_timeout

        try:
            response = stub.StepIterative(req, metadata=metadata, timeout=eff_timeout)
        except grpc.RpcError as rpc_err:
            self._handle_rpc_error(
                rpc_err, lease_id, "StepIterative", worker_pubkey=worker_pubkey, multiaddr=target
            )

        # Validate step indices and lease_id
        if str(response.lease_id).strip() != str(lease_id).strip():
            raise TaskDispatchError(
                f"Worker returned mismatched lease_id: expected {lease_id}, got {response.lease_id}",
                task_id=lease_id,
                returned_task_id=response.lease_id,
            )

        if response.step_index != step_index:
            raise StreamingError(
                f"Worker returned mismatched step_index: expected {step_index}, got {response.step_index}"
            )

        # Decrypt payload if confidential
        if session is not None and session.is_confidential:
            if not response.is_encrypted:
                raise SecurityPolicyViolationError(
                    f"Confidential compute enabled: worker returned unencrypted step response for lease '{lease_id}'",
                    task_id=lease_id,
                )
            if session.k_w2c is None:
                raise DecryptionError(
                    f"No inbound session key established for lease '{lease_id}'",
                    task_id=lease_id,
                )
            step_resp_aad = f"{lease_id}:step:{step_index}".encode("utf-8")
            decrypted_bytes = aes_gcm_decrypt(
                key=session.k_w2c,
                iv=response.iv,
                ciphertext=response.result_vector_bytes,
                tag=response.tag,
                aad=step_resp_aad,
                fallback_aads=[b"", None],
            )
        else:
            if response.is_encrypted:
                raise DecryptionError(
                    f"Unexpected encrypted response for unencrypted lease '{lease_id}'",
                    task_id=lease_id,
                )
            decrypted_bytes = response.result_vector_bytes

        # Validate byte count
        expected_bytes = response.vector_length * bytes_per_elem
        if len(decrypted_bytes) != expected_bytes:
            raise StreamingError(
                f"Worker returned vector byte count {len(decrypted_bytes)} != expected {expected_bytes} (length={response.vector_length})"
            )

        # Secp256k1 Signature Verification
        if worker_pubkey:
            if not response.worker_signature or not str(response.worker_signature).strip():
                raise WorkerSignatureError(
                    f"Worker omitted required Secp256k1 signature on iterative step {step_index} of lease '{lease_id}'",
                    task_id=lease_id,
                    worker_pubkey=worker_pubkey,
                    signature="",
                )
            sig_valid = self._verify_iterative_step_signature(
                worker_pubkey=worker_pubkey,
                lease_id=lease_id,
                step_index=step_index,
                result_bytes=decrypted_bytes,
                signature=response.worker_signature,
            )
            if not sig_valid:
                raise WorkerSignatureError(
                    f"Worker signature verification failed on iterative step {step_index} of lease '{lease_id}'",
                    task_id=lease_id,
                    worker_pubkey=worker_pubkey,
                    signature=response.worker_signature,
                )

        result_vec = np.frombuffer(decrypted_bytes, dtype=np_dtype)
        return IterativeStepResponseWrapper(response=response, result_vec=result_vec)

    def release_lease(
        self,
        worker_or_target: Union[str, Any],
        lease_id: str,
        reason: str = "completed",
        consumer_pubkey: Optional[str] = None,
        auth_header: Optional[str] = None,
        worker_pubkey: str = "",
        timeout: Optional[float] = None,
    ) -> pb2.ReleaseLeaseResponse:
        """
        Explicitly releases remote VRAM matrix lease and zeroizes memory.
        """
        target = parse_multiaddr(worker_or_target)
        stub = self.channel_pool.get_stub(target)

        with self._sessions_lock:
            self._iterative_sessions.pop(lease_id, None)

        pubkey = consumer_pubkey or self.consumer_pubkey

        req = pb2.ReleaseLeaseRequest(
            lease_id=str(lease_id).strip(),
            consumer_pubkey=str(pubkey).strip(),
            reason=str(reason).strip(),
        )

        metadata = self._build_metadata(
            lease_id,
            auth_header=auth_header,
            worker_pubkey=worker_pubkey,
            worker_or_target=worker_or_target,
            is_lease=True,
        )

        eff_timeout = timeout if timeout is not None else self.connect_timeout

        try:
            response = stub.ReleaseLease(req, metadata=metadata, timeout=eff_timeout)
        except grpc.RpcError as rpc_err:
            self._handle_rpc_error(
                rpc_err, lease_id, "ReleaseLease", worker_pubkey=worker_pubkey, multiaddr=target
            )

        return response

    def _verify_iterative_step_signature(
        self,
        worker_pubkey: str,
        lease_id: str,
        step_index: int,
        result_bytes: bytes,
        signature: str,
    ) -> bool:
        """
        Validates worker Secp256k1 signature over an iterative step result.
        """
        if not worker_pubkey or not signature:
            return False

        clean_sig = str(signature).strip().lower()
        r_hash_hex = hashlib.sha256(result_bytes).hexdigest()
        r_hash_digest = hashlib.sha256(result_bytes).digest()

        # Format 1: Canonical payload with hex hash
        payload1 = f"{lease_id}:{step_index}:{r_hash_hex}".encode("utf-8")
        if verify_signature(worker_pubkey, hashlib.sha256(payload1).digest(), clean_sig):
            return True
        if verify_signature(worker_pubkey, payload1, clean_sig):
            return True

        # Format 2: Payload with binary hash digest
        payload2 = f"{lease_id}:{step_index}:".encode("utf-8") + r_hash_digest
        if verify_signature(worker_pubkey, hashlib.sha256(payload2).digest(), clean_sig):
            return True
        if verify_signature(worker_pubkey, payload2, clean_sig):
            return True

        return False

    def _handle_rpc_error(
        self,
        rpc_err: grpc.RpcError,
        task_id: str,
        stage: str,
        worker_pubkey: str = "",
        multiaddr: str = "",
    ) -> None:
        """Translates gRPC RpcError into domain-specific SDK exception."""
        code = rpc_err.code() if hasattr(rpc_err, "code") else None
        details = rpc_err.details() if hasattr(rpc_err, "details") else str(rpc_err)

        if code == grpc.StatusCode.DEADLINE_EXCEEDED:
            if stage in ("RequestTileTask", "LeaseMatrix"):
                raise WorkerConnectionError(
                    f"Connection timeout during {stage} for task '{task_id}': {details}",
                    task_id=task_id,
                    worker_pubkey=worker_pubkey,
                    multiaddr=multiaddr,
                    cause=rpc_err,
                )
            raise TaskTimeoutError(
                f"Timeout during {stage} for task '{task_id}': {details}",
                task_id=task_id,
                timeout_seconds=self.task_timeout,
                cause=rpc_err,
            )
        elif code in (grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.UNKNOWN):
            raise WorkerConnectionError(
                f"Connection error during {stage} for task '{task_id}': {details}",
                worker_pubkey=worker_pubkey,
                multiaddr=multiaddr,
                cause=rpc_err,
            )
        elif code in (
            grpc.StatusCode.PERMISSION_DENIED,
            grpc.StatusCode.UNAUTHENTICATED,
            grpc.StatusCode.FAILED_PRECONDITION,
            grpc.StatusCode.RESOURCE_EXHAUSTED,
        ):
            raise WorkerTaskRejectedError(
                f"Worker rejected {stage} with {code}: {details}",
                task_id=task_id,
                worker_pubkey=worker_pubkey,
                reason=details,
                cause=rpc_err,
            )
        else:
            raise DispatchError(
                f"gRPC error during {stage} for task '{task_id}': {details}",
                cause=rpc_err,
            )


# Alias per PROJECT.md interface contract
TileDispatcher = GrpcTaskDispatcher
