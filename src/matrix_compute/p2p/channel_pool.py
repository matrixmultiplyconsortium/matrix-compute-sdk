"""
gRPC Channel and Stub Pool for matrix-compute-sdk P2P subsystem.
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Any, Dict, List, Optional, Tuple, Union

import grpc

try:
    from matrix_contracts.v1 import pb2, pb2_grpc
except ImportError:
    from matrix_contracts.proto import compute_service_pb2 as pb2
    from matrix_contracts.proto import compute_service_pb2_grpc as pb2_grpc

logger = logging.getLogger(__name__)


def parse_multiaddr(endpoint: Union[str, Any]) -> str:
    """
    Parses a multiaddr, URI, or object into a standard gRPC target string (e.g. '127.0.0.1:50051').

    Supported formats:
      - '/ip4/<ip>/tcp/<port>' -> '<ip>:<port>'
      - '/ip6/<ip>/tcp/<port>' -> '<ip>:<port>'
      - '/dns4/<host>/tcp/<port>' -> '<host>:<port>'
      - '/dns/<host>/tcp/<port>' -> '<host>:<port>'
      - 'http://<host>:<port>' -> '<host>:<port>'
      - 'grpc://<host>:<port>' -> '<host>:<port>'
      - '<host>:<port>' -> '<host>:<port>'
      - Objects with .p2p_multiaddr, .grpc_endpoint, .target, or .multiaddr attributes.
    """
    if endpoint is None:
        raise ValueError("Target/multiaddr cannot be None")

    if not isinstance(endpoint, str):
        if hasattr(endpoint, "p2p_multiaddr") and endpoint.p2p_multiaddr:
            endpoint = endpoint.p2p_multiaddr
        elif hasattr(endpoint, "grpc_endpoint") and endpoint.grpc_endpoint:
            endpoint = endpoint.grpc_endpoint
        elif hasattr(endpoint, "target") and endpoint.target:
            endpoint = endpoint.target
        elif hasattr(endpoint, "multiaddr") and endpoint.multiaddr:
            endpoint = endpoint.multiaddr
        else:
            endpoint = str(endpoint)

    clean = str(endpoint).strip()
    if not clean:
        raise ValueError("Multiaddr cannot be empty")

    # Match /ip4/<ip>/tcp/<port> or /dns4/<host>/tcp/<port> or /ip6/<ip>/tcp/<port>
    m = re.match(r"^/(ip4|ip6|dns4|dns)/([^/]+)/tcp/(\d+)", clean)
    if m:
        proto, host, port = m.group(1), m.group(2), m.group(3)
        if proto == "ip6" or ":" in host:
            if not (host.startswith("[") and host.endswith("]")):
                host = f"[{host}]"
        return f"{host}:{port}"

    # Match URI scheme prefix (e.g. grpc://, http://, https://)
    m_scheme = re.match(r"^(?:grpc|grpcs|http|https)://([^/]+)", clean)
    if m_scheme:
        clean = m_scheme.group(1)

    # Match standard host:port or bracketed [ipv6]:port
    if ":" in clean and not clean.startswith("/"):
        m_bracketed = re.match(r"^(\[[^\]]+\]):(\d+)$", clean)
        if m_bracketed:
            return clean
        parts = clean.split(":")
        if len(parts) == 2 and parts[1].isdigit():
            return clean

    return clean


multiaddr_to_target = parse_multiaddr


def extract_circuit_target_worker(endpoint: Union[str, Any]) -> Optional[str]:
    """
    Extracts the target worker pubkey from a circuit multiaddr if present.
    Example:
      '/dns4/relay.local/tcp/9090/p2p-circuit/p2p/<pubkey>' -> '<pubkey>'
      '/ip4/127.0.0.1/tcp/9090/p2p-circuit/p2p/<pubkey>' -> '<pubkey>'
    """
    if endpoint is None:
        return None

    if not isinstance(endpoint, str):
        if hasattr(endpoint, "worker_pubkey") and endpoint.worker_pubkey:
            return str(endpoint.worker_pubkey).strip().lower()
        if hasattr(endpoint, "p2p_multiaddr") and endpoint.p2p_multiaddr:
            endpoint = endpoint.p2p_multiaddr
        elif hasattr(endpoint, "multiaddr") and endpoint.multiaddr:
            endpoint = endpoint.multiaddr
        else:
            endpoint = str(endpoint)

    clean = str(endpoint).strip()
    m = re.search(r"/p2p-circuit/p2p/([a-fA-F0-9]+)", clean)
    if m:
        return m.group(1).lower()

    m2 = re.search(r"/p2p/([a-fA-F0-9]+)", clean)
    if m2 and "p2p-circuit" in clean:
        return m2.group(1).lower()

    return None


def is_circuit_multiaddr(endpoint: Union[str, Any]) -> bool:
    """Returns True if endpoint represents a libp2p circuit relay multiaddr."""
    if endpoint is None:
        return False
    clean = str(getattr(endpoint, "p2p_multiaddr", None) or getattr(endpoint, "multiaddr", None) or endpoint)
    return "p2p-circuit" in clean


class ChannelPool:
    """
    Thread-safe cache of gRPC Channels and MatrixComputeService stubs.
    Reuses existing connections to avoid channel churn and establishes high-throughput
    buffer limits (100MB) with keepalive health management.
    """

    def __init__(
        self,
        max_message_bytes: int = 100 * 1024 * 1024,
        ssl_enabled: bool = False,
        root_certificates: Optional[bytes] = None,
        private_key: Optional[bytes] = None,
        certificate_chain: Optional[bytes] = None,
    ) -> None:
        self.max_message_bytes = max(max_message_bytes, 64 * 1024 * 1024)
        self.ssl_enabled = ssl_enabled
        self.root_certificates = root_certificates
        self.private_key = private_key
        self.certificate_chain = certificate_chain

        self._lock = threading.RLock()
        self._channels: Dict[str, grpc.Channel] = {}
        self._stubs: Dict[str, pb2_grpc.MatrixComputeServiceStub] = {}
        self._is_closed = False

    def _channel_options(self) -> List[Tuple[str, Any]]:
        return [
            ("grpc.max_receive_message_length", self.max_message_bytes),
            ("grpc.max_send_message_length", self.max_message_bytes),
            ("grpc.keepalive_time_ms", 120000),
            ("grpc.keepalive_timeout_ms", 60000),
            ("grpc.keepalive_permit_without_calls", 0),
            ("grpc.http2.max_pings_without_data", 0),
        ]

    def get_channel(
        self,
        target_or_worker: Union[str, Any],
        ssl_enabled: Optional[bool] = None,
    ) -> grpc.Channel:
        """Gets or creates a cached grpc.Channel for the given target address or worker object."""
        target = parse_multiaddr(target_or_worker)
        use_ssl = self.ssl_enabled if ssl_enabled is None else ssl_enabled

        with self._lock:
            if self._is_closed:
                raise RuntimeError("ChannelPool is closed")
            if target not in self._channels:
                if use_ssl:
                    creds = grpc.ssl_channel_credentials(
                        root_certificates=self.root_certificates,
                        private_key=self.private_key,
                        certificate_chain=self.certificate_chain,
                    )
                    channel = grpc.secure_channel(target, creds, options=self._channel_options())
                else:
                    channel = grpc.insecure_channel(target, options=self._channel_options())
                self._channels[target] = channel
            return self._channels[target]

    def get_stub(
        self,
        target_or_worker: Union[str, Any],
        ssl_enabled: Optional[bool] = None,
    ) -> pb2_grpc.MatrixComputeServiceStub:
        """Gets or creates a cached MatrixComputeServiceStub for the target address or worker object."""
        target = parse_multiaddr(target_or_worker)
        with self._lock:
            if self._is_closed:
                raise RuntimeError("ChannelPool is closed")
            if target not in self._stubs:
                channel = self.get_channel(target, ssl_enabled=ssl_enabled)
                self._stubs[target] = pb2_grpc.MatrixComputeServiceStub(channel)
            return self._stubs[target]

    def evict(self, target_or_worker: Union[str, Any]) -> None:
        """Closes and removes the cached channel and stub for the target address."""
        try:
            target = parse_multiaddr(target_or_worker)
        except Exception:
            return

        with self._lock:
            channel = self._channels.pop(target, None)
            self._stubs.pop(target, None)

        if channel is not None:
            try:
                channel.close()
            except Exception as ex:
                logger.debug("Error closing evicted channel to %s: %s", target, ex)

    def close(self) -> None:
        """Closes all cached channels and releases resources."""
        with self._lock:
            if self._is_closed:
                return
            self._is_closed = True
            for target, channel in self._channels.items():
                try:
                    channel.close()
                except Exception as ex:
                    logger.debug("Error closing channel to %s: %s", target, ex)
            self._channels.clear()
            self._stubs.clear()

    def close_all(self) -> None:
        """Alias for close()."""
        self.close()

    def __enter__(self) -> "ChannelPool":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


# Backward and forward compatibility alias
GrpcChannelPool = ChannelPool
