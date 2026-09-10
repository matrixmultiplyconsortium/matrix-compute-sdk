"""
P2P gRPC Task Dispatcher for matrix-compute-sdk.

Handles RequestTileTask pre-negotiation, ExecuteTileTask progressive chunk streaming,
L402 metadata formatting, and worker Secp256k1 signature validation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import logging
import math
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import grpc
import numpy as np

try:
    from matrix_contracts.v1 import pb2, pb2_grpc
except ImportError:
    from matrix_contracts.proto import compute_service_pb2 as pb2
    from matrix_contracts.proto import compute_service_pb2_grpc as pb2_grpc

from matrix_compute.crypto.signer import sha256_digest, verify_signature, verify_tile_result_signature
from matrix_compute.exceptions import (
    DispatchError,
    StreamingError,
    TaskDispatchError,
    TaskTimeoutError,
    WorkerConnectionError,
    WorkerSignatureError,
    WorkerTaskRejectedError,
)
from matrix_compute.p2p.channel_pool import ChannelPool, parse_multiaddr
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


class GrpcTaskDispatcher:
    """
    Dispatches individual matrix tile computation tasks to worker nodes over gRPC.
    """

    def __init__(
        self,
        channel_pool: Optional[ChannelPool] = None,
        consumer_pubkey: str = "",
        connect_timeout: float = 5.0,
        task_timeout: float = 30.0,
        chunk_size_bytes: int = 1024 * 1024,
    ) -> None:
        self.channel_pool = channel_pool or ChannelPool()
        self.consumer_pubkey = consumer_pubkey
        self.connect_timeout = connect_timeout
        self.task_timeout = task_timeout
        self.chunk_size_bytes = max(chunk_size_bytes, 1024)

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
        Raises WorkerTaskRejectedError if the worker rejects the task.
        """
        target = parse_multiaddr(worker_or_target)
        stub = self.channel_pool.get_stub(target)
        norm_dtype = normalize_dtype(dtype)

        req = pb2.TileTaskRequest(
            task_id=task_id,
            matrix_a_rows=int(m),
            matrix_a_cols=int(k),
            matrix_b_cols=int(n),
            dtype=norm_dtype,
        )

        eff_timeout = timeout if timeout is not None else self.connect_timeout
        try:
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

        return ack

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
        np_dtype = get_numpy_dtype(target_dtype)
        bytes_per_elem = get_bytes_per_element(target_dtype)

        m, k = matrix_a.shape
        k2, n = matrix_b.shape
        if k != k2:
            raise ValueError(f"Inner dimension mismatch: {k} != {k2}")

        def _chunk_generator() -> Iterator[pb2.TileDataChunk]:
            seq = 0
            # Stream Matrix A
            a_bytes = np.ascontiguousarray(matrix_a, dtype=np_dtype).tobytes()
            total_a = len(a_bytes)
            if total_a == 0:
                yield pb2.TileDataChunk(
                    task_id=task_id,
                    chunk_bytes=b"",
                    is_matrix_a=True,
                    sequence_number=seq,
                    is_final_chunk=False,
                )
                seq += 1
            else:
                for i in range(0, total_a, self.chunk_size_bytes):
                    chunk_slice = a_bytes[i : i + self.chunk_size_bytes]
                    yield pb2.TileDataChunk(
                        task_id=task_id,
                        chunk_bytes=chunk_slice,
                        is_matrix_a=True,
                        sequence_number=seq,
                        is_final_chunk=False,
                    )
                    seq += 1

            # Stream Matrix B
            b_bytes = np.ascontiguousarray(matrix_b, dtype=np_dtype).tobytes()
            total_b = len(b_bytes)
            if total_b == 0:
                yield pb2.TileDataChunk(
                    task_id=task_id,
                    chunk_bytes=b"",
                    is_matrix_a=False,
                    sequence_number=seq,
                    is_final_chunk=True,
                )
            else:
                total_b_chunks = math.ceil(total_b / self.chunk_size_bytes)
                for idx, i in enumerate(range(0, total_b, self.chunk_size_bytes)):
                    chunk_slice = b_bytes[i : i + self.chunk_size_bytes]
                    is_final = (idx == total_b_chunks - 1)
                    yield pb2.TileDataChunk(
                        task_id=task_id,
                        chunk_bytes=chunk_slice,
                        is_matrix_a=False,
                        sequence_number=seq,
                        is_final_chunk=is_final,
                    )
                    seq += 1

        metadata: List[Tuple[str, str]] = [("x-task-id", task_id)]
        if auth_header:
            metadata.append(("authorization", auth_header.strip()))

        eff_timeout = timeout if timeout is not None else self.task_timeout
        try:
            tile_result = stub.ExecuteTileTask(
                _chunk_generator(),
                metadata=metadata,
                timeout=eff_timeout,
            )
        except grpc.RpcError as rpc_err:
            self._handle_rpc_error(rpc_err, task_id, "ExecuteTileTask", worker_pubkey=worker_pubkey, multiaddr=target)

        # Validate task_id matches expected task_id
        if str(tile_result.task_id).strip() != str(task_id).strip():
            raise TaskDispatchError(
                f"Worker returned mismatched task_id: expected {task_id}, got {tile_result.task_id}",
                task_id=task_id,
                returned_task_id=tile_result.task_id,
            )

        # Validate dimensions and byte counts
        expected_bytes = m * n * bytes_per_elem
        if tile_result.rows != m or tile_result.cols != n:
            raise StreamingError(
                f"Worker returned invalid dimensions: expected ({m}, {n}), got ({tile_result.rows}, {tile_result.cols})"
            )

        if len(tile_result.result_bytes) != expected_bytes:
            raise StreamingError(
                f"Worker returned byte count {len(tile_result.result_bytes)} != expected {expected_bytes}"
            )

        # Cryptographic signature validation
        if worker_pubkey:
            sig_valid = self.verify_worker_signature(
                worker_pubkey=worker_pubkey,
                task_id=task_id,
                result_bytes=tile_result.result_bytes,
                worker_signature=tile_result.worker_signature,
                matrix_a=matrix_a,
                matrix_b=matrix_b,
                dtype=target_dtype,
            )
            if not sig_valid:
                raise WorkerSignatureError(
                    f"Worker signature verification failed on tile task '{task_id}'",
                    task_id=task_id,
                    worker_pubkey=worker_pubkey,
                    signature=tile_result.worker_signature,
                )

        c_matrix = np.frombuffer(tile_result.result_bytes, dtype=np_dtype).reshape((m, n))
        return c_matrix, tile_result

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
        pubkey = worker_pubkey or getattr(worker, "worker_pubkey", getattr(worker, "pubkey", ""))
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
    ) -> bool:
        """
        Validates worker's Secp256k1 signature.
        Checks canonical format f"{task_id}:{c_hash}", and falls back to composite hash format.
        """
        if not worker_pubkey or not worker_signature:
            return False

        # Primary canonical signature check
        if verify_tile_result_signature(
            worker_pubkey=worker_pubkey,
            task_id=task_id,
            result_bytes=result_bytes,
            signature_hex=worker_signature,
        ):
            return True

        # Check if signature was produced on the pre-hashed payload: sha256(f"{task_id}:{c_hash}")
        c_hash = sha256_digest(result_bytes)
        payload = f"{str(task_id).strip()}:{c_hash.strip()}".encode("utf-8")
        if verify_signature(worker_pubkey, hashlib.sha256(payload).digest(), worker_signature):
            return True

        # Fallback composite signature check
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
            if stage == "RequestTileTask":
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
