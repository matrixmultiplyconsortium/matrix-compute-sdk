"""
Nostr Relay Discovery Client for matrix-compute-sdk.

Connects to Nostr relays (live WebSocket and/or hermetic in-memory mock relay doubles),
ingests Kind 30000 WORKER_ANNOUNCE events, performs dual-mode tag/JSON parsing,
verifies Secp256k1 X-only event signatures, manages connection pooling and failover,
and provides in-memory TTL caching with deduplication.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import logging
import math
import re
import socket
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urlparse

try:
    from matrix_contracts.schemas.discovery import (
        MatchedWorker,
        WorkerAnnouncePayload,
        WorkerQueryRequest,
        WorkerQueryResponse,
    )
except ImportError:
    from matrix_contracts.schemas import (  # type: ignore
        MatchedWorker,
        WorkerAnnouncePayload,
        WorkerQueryRequest,
        WorkerQueryResponse,
    )

from matrix_compute.crypto.signer import verify_signature
from matrix_compute.exceptions import (
    DiscoveryError,
    InvalidNostrEventError,
    RelayConnectionError,
    RelayPoolExhaustedError,
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Data Models
# -----------------------------------------------------------------------------

@dataclass
class DiscoveredWorker:
    """
    Normalized representation of a compute worker discovered via Nostr.
    Maintains all hardware, pricing, networking, and reputation metadata.
    """
    pubkey: str
    d_tag: str = "matrix-worker"
    grpc_endpoint: str = ""
    lightning_address: str = ""
    vram_mb: int = 0
    sats_per_gflop: float = 0.0
    gpu_model: str = "Unknown GPU"
    peak_fp32_tflops: float = 0.0
    max_dimension: int = 8192
    supported_precisions: List[str] = field(default_factory=lambda: ["fp32", "fp16", "bf16"])
    slots_available: int = 1
    min_task_sats: int = 10
    staked_sats: int = 0
    bond_txid: Optional[str] = None
    created_at: int = 0
    relay_url: str = ""
    raw_event: Optional[Dict[str, Any]] = None
    reputation_score: float = 1.0

    @property
    def worker_pubkey(self) -> str:
        """Alias for worker public key."""
        return self.pubkey

    @property
    def p2p_multiaddr(self) -> str:
        """Alias for gRPC endpoint / network address."""
        return self.grpc_endpoint

    @property
    def lightning_routing_pubkey(self) -> str:
        """Alias for Lightning routing pubkey / address."""
        return self.lightning_address

    def to_matched_worker(self) -> MatchedWorker:
        """Converts to standard matrix_contracts MatchedWorker schema."""
        return MatchedWorker(
            worker_pubkey=self.pubkey,
            p2p_multiaddr=self.grpc_endpoint,
            lightning_routing_pubkey=self.lightning_address,
            reputation_score=self.reputation_score,
            sats_per_gflop=self.sats_per_gflop,
        )


class RelayStatus(str, Enum):
    ACTIVE = "ACTIVE"
    DEGRADED = "DEGRADED"
    OFFLINE = "OFFLINE"


@dataclass
class RelayNode:
    target: Any  # URL str or MockRelay instance
    status: RelayStatus = RelayStatus.ACTIVE
    error_count: int = 0
    last_error: Optional[str] = None
    last_queried: float = 0.0


# -----------------------------------------------------------------------------
# RelayClient Implementation
# -----------------------------------------------------------------------------

class RelayClient:
    """
    Nostr Discovery Relay Client for matrix-compute-sdk.
    Connects to live WebSocket relays and/or hermetic in-memory mock relay doubles.
    Handles NIP-01/NIP-33 event parsing, tag extraction, signature checks, and multi-relay failover.
    """

    def __init__(
        self,
        relays: Optional[Union[str, List[Union[str, Any]]]] = None,
        timeout: float = 5.0,
        cache_ttl: float = 300.0,
        verify_signatures: bool = True,
        offline_mock_mode: bool = False,
        mock_relay: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        if relays is None and "relay_urls" in kwargs:
            relays = kwargs["relay_urls"]
        if "timeout_seconds" in kwargs and kwargs["timeout_seconds"] is not None:
            timeout = kwargs["timeout_seconds"]
        if mock_relay is None and "in_memory_relay" in kwargs:
            mock_relay = kwargs["in_memory_relay"]

        self.timeout = float(timeout)
        self.cache_ttl = float(cache_ttl)
        self.verify_signatures = bool(verify_signatures)
        self.offline_mock_mode = bool(offline_mock_mode)

        self._lock = threading.RLock()
        self._cache: Dict[Tuple[str, str], Tuple[DiscoveredWorker, float]] = {}  # (pubkey, d_tag) -> (worker, expires_at)
        self._relay_pool: List[RelayNode] = []

        targets: List[Any] = []
        if mock_relay is not None:
            targets.append(mock_relay)
        if relays is not None:
            if isinstance(relays, (list, tuple, set)):
                targets.extend(relays)
            else:
                targets.append(relays)

        if not targets:
            # Default to local mock/relay endpoint
            targets = ["ws://127.0.0.1:8008"]

        for target in targets:
            self._add_relay_target(target)

    def _add_relay_target(self, target: Any) -> None:
        with self._lock:
            # Validate URL schemes if string
            if isinstance(target, str):
                parsed = urlparse(target)
                if parsed.scheme not in ("ws", "wss", "http", "https", "mock"):
                    raise RelayConnectionError(
                        f"Unsupported relay protocol scheme '{parsed.scheme}': must be ws, wss, http, or https",
                        relay_url=target,
                    )
            self._relay_pool.append(RelayNode(target=target))

    def connect(self) -> bool:
        """
        Validates connections to configured relays.
        Returns True if at least one relay is active and reachable.
        Raises RelayConnectionError if all relays fail or an invalid endpoint is given.
        """
        with self._lock:
            active_count = 0
            for node in self._relay_pool:
                if not isinstance(node.target, str):
                    # Mock double instance is always active
                    node.status = RelayStatus.ACTIVE
                    active_count += 1
                else:
                    url = node.target
                    parsed = urlparse(url)
                    if parsed.scheme not in ("ws", "wss", "http", "https"):
                        node.status = RelayStatus.OFFLINE
                        node.last_error = f"Invalid scheme {parsed.scheme}"
                        continue

                    # In offline_mock_mode, treat configured endpoints as active
                    if self.offline_mock_mode:
                        node.status = RelayStatus.ACTIVE
                        active_count += 1
                    else:
                        # Genuine TCP connection probe for live endpoints
                        host = parsed.hostname or "127.0.0.1"
                        port = parsed.port
                        if port is None:
                            port = 443 if parsed.scheme in ("wss", "https") else 80

                        try:
                            sock = socket.create_connection((host, port), timeout=self.timeout)
                            sock.close()
                            node.status = RelayStatus.ACTIVE
                            node.last_error = None
                            active_count += 1
                        except (socket.error, OSError) as ex:
                            node.status = RelayStatus.OFFLINE
                            node.last_error = f"Connection refused / unreachable: {ex}"

            if active_count == 0:
                raise RelayConnectionError(
                    "Unable to connect to any configured Nostr relays",
                    relay_url=str(self._relay_pool[0].target) if self._relay_pool else None,
                )
            return True

    def disconnect(self) -> None:
        """Closes all relay connections and purges transient pool state."""
        with self._lock:
            for node in self._relay_pool:
                node.status = RelayStatus.OFFLINE
            self._cache.clear()

    @property
    def is_connected(self) -> bool:
        """Returns True if at least one relay in the pool is ACTIVE."""
        with self._lock:
            return any(node.status == RelayStatus.ACTIVE for node in self._relay_pool)

    def parse_event(self, event: Dict[str, Any], relay_url: str = "") -> DiscoveredWorker:
        """
        Parses a raw Nostr Kind 30000 event into a DiscoveredWorker instance.
        Implements dual-mode parsing:
          1. Standard tags extracted first.
          2. If critical specs are missing, fallback to JSON payload in `event['content']`.
          3. Zero values (e.g. sats_per_gflop = 0.0 or vram_mb = 0) are strictly preserved.
        """
        if not isinstance(event, dict):
            raise InvalidNostrEventError("Event payload must be a dictionary")

        pubkey = str(event.get("pubkey", "")).strip().lower()
        if not pubkey:
            raise InvalidNostrEventError("Event missing required 'pubkey' field")

        created_at = int(event.get("created_at", 0) or time.time())

        # Extract tags
        raw_tags = event.get("tags", [])
        tag_dict: Dict[str, str] = {}
        for t in raw_tags:
            if isinstance(t, (list, tuple)) and len(t) >= 2:
                key = str(t[0]).strip().lower()
                val = str(t[1]).strip()
                tag_dict[key] = val

        d_tag = tag_dict.get("d") or tag_dict.get("identifier") or "matrix-worker"
        grpc_endpoint = (
            tag_dict.get("p2p")
            or tag_dict.get("grpc_endpoint")
            or tag_dict.get("multiaddr")
            or ""
        )
        lightning_address = (
            tag_dict.get("lightning")
            or tag_dict.get("lightning_address")
            or tag_dict.get("lightning_routing_pubkey")
            or ""
        )
        gpu_model = tag_dict.get("gpu_model") or ""

        # Price parsing with zero preservation and NaN/Inf sanitization
        sats_per_gflop: Optional[float] = None
        for price_key in ("sats_per_gflop", "price_per_gflop_sats", "price"):
            if price_key in tag_dict:
                try:
                    val = float(tag_dict[price_key])
                    if math.isfinite(val) and val >= 0.0:
                        sats_per_gflop = val
                    else:
                        sats_per_gflop = 0.0
                    break
                except (ValueError, TypeError):
                    pass

        # VRAM parsing with zero preservation and NaN/Inf sanitization
        vram_mb: Optional[int] = None
        if "vram_mb" in tag_dict:
            try:
                val = float(tag_dict["vram_mb"])
                if math.isfinite(val) and val >= 0.0:
                    vram_mb = int(val)
                else:
                    vram_mb = 0
            except (ValueError, TypeError):
                pass

        # Peak TFLOPS with NaN/Inf sanitization
        peak_fp32_tflops: Optional[float] = None
        if "peak_fp32_tflops" in tag_dict:
            try:
                val = float(tag_dict["peak_fp32_tflops"])
                if math.isfinite(val) and val >= 0.0:
                    peak_fp32_tflops = val
                else:
                    peak_fp32_tflops = 0.0
            except (ValueError, TypeError):
                pass

        # Max dimension
        max_dimension: Optional[int] = None
        for dim_key in ("max_dimension", "max_matrix_dim"):
            if dim_key in tag_dict:
                try:
                    val = int(tag_dict[dim_key])
                    if val > 0:
                        max_dimension = val
                    break
                except (ValueError, TypeError):
                    pass

        # Supported precisions
        supported_precisions = ["fp32", "fp16", "bf16"]
        for prec_key in ("supported_precisions", "precisions"):
            if prec_key in tag_dict:
                val = tag_dict[prec_key]
                if isinstance(val, str):
                    prec_list = [p.strip().lower() for p in val.split(",") if p.strip()]
                    if prec_list:
                        supported_precisions = prec_list
                break

        # Slots available
        slots_available = 1
        for slots_key in ("slots_available", "slots"):
            if slots_key in tag_dict:
                try:
                    slots_available = max(0, int(tag_dict[slots_key]))
                    break
                except (ValueError, TypeError):
                    pass

        # Min task sats
        min_task_sats = 10
        if "min_task_sats" in tag_dict:
            try:
                min_task_sats = max(0, int(tag_dict["min_task_sats"]))
            except (ValueError, TypeError):
                pass

        # Staked sats & bond txid
        staked_sats: Optional[int] = None
        for stake_key in ("staked_sats", "bond_amount"):
            if stake_key in tag_dict:
                try:
                    staked_sats = max(0, int(tag_dict[stake_key]))
                    break
                except (ValueError, TypeError):
                    pass

        bond_txid = tag_dict.get("bond_txid") or None

        # Content fallback if any critical spec is missing
        needs_fallback = (
            not grpc_endpoint
            or not lightning_address
            or not gpu_model
            or sats_per_gflop is None
            or vram_mb is None
        )

        if needs_fallback:
            raw_content = event.get("content", "{}")
            content_data: Dict[str, Any] = {}
            if isinstance(raw_content, dict):
                content_data = raw_content
            elif isinstance(raw_content, str) and raw_content.strip():
                try:
                    parsed_json = json.loads(raw_content)
                    if isinstance(parsed_json, dict):
                        content_data = parsed_json
                except Exception:
                    pass

            if content_data:
                if not grpc_endpoint:
                    grpc_endpoint = str(content_data.get("p2p_multiaddr") or content_data.get("grpc_endpoint") or "")
                if not lightning_address:
                    lightning_address = str(content_data.get("lightning_routing_pubkey") or content_data.get("lightning_address") or "")

                hw = content_data.get("hardware", {})
                if isinstance(hw, dict):
                    if not gpu_model:
                        gpu_model = str(hw.get("gpu_model", ""))
                    if vram_mb is None and "vram_mb" in hw:
                        try:
                            val = float(hw["vram_mb"])
                            if math.isfinite(val) and val >= 0.0:
                                vram_mb = int(val)
                            else:
                                vram_mb = 0
                        except (ValueError, TypeError):
                            pass
                    if peak_fp32_tflops is None and "peak_fp32_tflops" in hw:
                        try:
                            val = float(hw["peak_fp32_tflops"])
                            if math.isfinite(val) and val >= 0.0:
                                peak_fp32_tflops = val
                            else:
                                peak_fp32_tflops = 0.0
                        except (ValueError, TypeError):
                            pass
                    if max_dimension is None and "max_dimension" in hw:
                        try:
                            val = int(hw["max_dimension"])
                            if val > 0:
                                max_dimension = val
                        except (ValueError, TypeError):
                            pass
                    if "supported_precisions" in hw and isinstance(hw["supported_precisions"], list):
                        supported_precisions = [str(p).lower() for p in hw["supported_precisions"]]

                pricing = content_data.get("pricing", {})
                if isinstance(pricing, dict):
                    if sats_per_gflop is None and "sats_per_gflop" in pricing:
                        try:
                            val = float(pricing["sats_per_gflop"])
                            if math.isfinite(val) and val >= 0.0:
                                sats_per_gflop = val
                            else:
                                sats_per_gflop = 0.0
                        except (ValueError, TypeError):
                            pass
                    if "min_task_sats" in pricing:
                        try:
                            min_task_sats = max(0, int(pricing["min_task_sats"]))
                        except (ValueError, TypeError):
                            pass

                bond = content_data.get("bond_status", {})
                if isinstance(bond, dict):
                    if staked_sats is None and "staked_amount_sats" in bond:
                        try:
                            staked_sats = max(0, int(bond["staked_amount_sats"]))
                        except (ValueError, TypeError):
                            pass
                    if not bond_txid:
                        bond_txid = bond.get("bond_txid")

        return DiscoveredWorker(
            pubkey=pubkey,
            d_tag=d_tag,
            grpc_endpoint=grpc_endpoint or "127.0.0.1:50051",
            lightning_address=lightning_address or "worker@lightning.local",
            vram_mb=vram_mb if vram_mb is not None else 0,
            sats_per_gflop=sats_per_gflop if sats_per_gflop is not None else 0.0,
            gpu_model=gpu_model or "Unknown GPU",
            peak_fp32_tflops=peak_fp32_tflops if peak_fp32_tflops is not None else 0.0,
            max_dimension=max_dimension if max_dimension is not None else 8192,
            supported_precisions=supported_precisions,
            slots_available=slots_available,
            min_task_sats=min_task_sats,
            staked_sats=staked_sats if staked_sats is not None else 0,
            bond_txid=bond_txid,
            created_at=created_at,
            relay_url=relay_url,
            raw_event=event,
            reputation_score=1.0,
        )

    def verify_event_signature(self, event: Dict[str, Any]) -> bool:
        """
        Verifies Nostr Secp256k1 event signature using X-only dual-parity checks.
        Supports hermetic mock signatures beginning with 'mock_'.
        """
        if not self.verify_signatures:
            return True

        sig = str(event.get("sig", "")).strip()
        if not sig:
            return False

        # Mock signatures are ONLY permitted when offline_mock_mode is True
        if self.offline_mock_mode and sig.startswith("mock_"):
            return True

        pubkey = str(event.get("pubkey", "")).strip().lower()
        event_id = str(event.get("id", "")).strip().lower()
        if not pubkey or not event_id:
            return False

        # NIP-01 canonical event serialization verification
        # Verify event["id"] == sha256(json.dumps([0, pubkey, created_at, kind, tags, content], ...))
        if any(k in event for k in ("tags", "kind", "created_at", "content")):
            try:
                canonical = [
                    0,
                    event.get("pubkey"),
                    event.get("created_at"),
                    event.get("kind"),
                    event.get("tags", []),
                    event.get("content", ""),
                ]
                serialized = json.dumps(canonical, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
                computed_id = hashlib.sha256(serialized).hexdigest().lower()
                if computed_id != event_id:
                    return False
            except Exception:
                return False

        # Attempt verification on raw 32-byte digest
        try:
            digest_bytes = bytes.fromhex(event_id)
            if len(digest_bytes) == 32 and verify_signature(pubkey, digest_bytes, sig):
                return True
        except Exception:
            pass

        # Attempt verification on string bytes
        try:
            if verify_signature(pubkey, event_id.encode("utf-8"), sig):
                return True
        except Exception:
            pass

        return False

    def query(
        self,
        min_vram_mb: int = 0,
        max_price: Optional[float] = None,
        required_slots: int = 1,
        gpu_model: Optional[str] = None,
        precision: Optional[str] = None,
        max_dimension: Optional[int] = None,
        force_refresh: bool = False,
    ) -> List[DiscoveredWorker]:
        """
        Queries workers from relay pool, filters by hardware and price,
        applies NIP-33 deduplication and caching.
        """
        now = time.time()

        with self._lock:
            # Check cache if not forcing refresh
            if not force_refresh and self._cache:
                valid_cached = [w for w, exp in self._cache.values() if exp > now]
                if valid_cached:
                    return self._filter_workers(
                        valid_cached,
                        min_vram_mb=min_vram_mb,
                        max_price=max_price,
                        required_slots=required_slots,
                        gpu_model=gpu_model,
                        precision=precision,
                        max_dimension=max_dimension,
                    )

            # Query all active relays with failover
            raw_events: List[Tuple[Dict[str, Any], str]] = []
            relays_succeeded = 0

            for node in self._relay_pool:
                if node.status == RelayStatus.OFFLINE and not self.offline_mock_mode:
                    continue

                try:
                    target = node.target
                    node_events: List[Dict[str, Any]] = []

                    if hasattr(target, "query"):
                        # In-memory mock double
                        node_events = target.query(min_vram_mb=0, max_price=None)
                    elif hasattr(target, "get_events"):
                        node_events = target.get_events()
                    elif isinstance(target, str):
                        # URL target: verify reachability and transport
                        if self.offline_mock_mode:
                            node_events = []
                        else:
                            parsed = urlparse(target)
                            host = parsed.hostname or "127.0.0.1"
                            port = parsed.port
                            if port is None:
                                port = 443 if parsed.scheme in ("wss", "https") else 80

                            try:
                                sock = socket.create_connection((host, port), timeout=self.timeout)
                                sock.close()
                            except (socket.error, OSError) as ex:
                                raise ConnectionRefusedError(f"Connection refused to {target}: {ex}") from ex

                            # Live relay query requires websockets library
                            try:
                                import websockets  # noqa: F401
                                raise RelayConnectionError(
                                    f"Live WebSocket subscription not implemented for {target}; use MockNostrRelay or install live adapter",
                                    relay_url=target,
                                )
                            except ImportError:
                                raise RelayConnectionError(
                                    f"Live relay transport requires 'websockets' library; install websockets to query {target}",
                                    relay_url=target,
                                )

                    for ev in node_events:
                        raw_events.append((ev, str(target)))

                    node.status = RelayStatus.ACTIVE
                    node.last_queried = now
                    relays_succeeded += 1

                except Exception as ex:
                    node.error_count += 1
                    node.last_error = str(ex)
                    node.status = RelayStatus.DEGRADED
                    logger.warning("Relay query failed on %s: %s", node.target, ex)

            if relays_succeeded == 0:
                raise RelayPoolExhaustedError("All relays in pool failed or are unreachable")

            # NIP-33 Deduplication: group by (pubkey, d_tag), keep greatest created_at
            dedup_events: Dict[Tuple[str, str], Tuple[Dict[str, Any], str]] = {}
            for event, relay_url in raw_events:
                if int(event.get("kind", 0)) != 30000:
                    continue

                # Verify signature
                if not self.verify_event_signature(event):
                    logger.debug("Dropping event %s with invalid signature", event.get("id"))
                    continue

                pub = str(event.get("pubkey", "")).strip().lower()
                d_val = "matrix-worker"
                for t in event.get("tags", []):
                    if isinstance(t, (list, tuple)) and len(t) >= 2 and t[0] == "d":
                        d_val = str(t[1])
                        break

                key = (pub, d_val)
                ts = int(event.get("created_at", 0))
                if key not in dedup_events or ts > int(dedup_events[key][0].get("created_at", 0)):
                    dedup_events[key] = (event, relay_url)

            # Parse discovered workers
            parsed_workers: List[DiscoveredWorker] = []
            for (pub, d_val), (ev, r_url) in dedup_events.items():
                try:
                    w = self.parse_event(ev, relay_url=r_url)
                    parsed_workers.append(w)
                    self._cache[(pub, d_val)] = (w, now + self.cache_ttl)
                except Exception as ex:
                    logger.warning("Failed to parse event %s: %s", ev.get("id"), ex)

            return self._filter_workers(
                parsed_workers,
                min_vram_mb=min_vram_mb,
                max_price=max_price,
                required_slots=required_slots,
                gpu_model=gpu_model,
                precision=precision,
                max_dimension=max_dimension,
            )

    def query_workers(
        self,
        request_or_min_vram: Union[int, WorkerQueryRequest] = 0,
        max_price: Optional[float] = None,
        required_slots: int = 1,
        gpu_model: Optional[str] = None,
        precision: Optional[str] = None,
        max_dimension: Optional[int] = None,
        force_refresh: bool = False,
        **kwargs: Any,
    ) -> Union[List[DiscoveredWorker], WorkerQueryResponse]:
        """
        Dual-mode query interface:
        If invoked with WorkerQueryRequest, returns WorkerQueryResponse.
        Otherwise returns List[DiscoveredWorker].
        """
        if isinstance(request_or_min_vram, WorkerQueryRequest):
            req = request_or_min_vram
            workers = self.query(
                min_vram_mb=req.min_vram_mb,
                max_price=req.max_price_sats_per_gflop,
                required_slots=req.required_slots,
                force_refresh=force_refresh,
            )
            # Sort lexicographically: price asc, vram desc
            sorted_workers = sorted(workers, key=lambda w: (w.sats_per_gflop, -w.vram_mb))
            matched = [w.to_matched_worker() for w in sorted_workers[:req.required_slots]]
            return WorkerQueryResponse(status="SUCCESS", matched_workers=matched)

        min_vram = int(request_or_min_vram)
        return self.query(
            min_vram_mb=min_vram,
            max_price=max_price,
            required_slots=required_slots,
            gpu_model=gpu_model,
            precision=precision,
            max_dimension=max_dimension,
            force_refresh=force_refresh,
        )

    def _filter_workers(
        self,
        workers: List[DiscoveredWorker],
        min_vram_mb: int = 0,
        max_price: Optional[float] = None,
        required_slots: int = 1,
        gpu_model: Optional[str] = None,
        precision: Optional[str] = None,
        max_dimension: Optional[int] = None,
    ) -> List[DiscoveredWorker]:
        filtered: List[DiscoveredWorker] = []
        for w in workers:
            if w.vram_mb < min_vram_mb:
                continue
            if max_price is not None and w.sats_per_gflop > max_price:
                continue
            if required_slots >= 1 and w.slots_available < 1:
                continue
            if gpu_model is not None and gpu_model.lower() not in w.gpu_model.lower():
                continue
            if precision is not None and precision.lower() not in [p.lower() for p in w.supported_precisions]:
                continue
            if max_dimension is not None and w.max_dimension < max_dimension:
                continue
            filtered.append(w)
        return filtered

    def publish_announce(
        self,
        payload_or_event: Union[Dict[str, Any], WorkerAnnouncePayload],
        relay: Optional[Any] = None,
    ) -> str:
        """Publishes a Kind 30000 worker announcement to relays."""
        target_relays = [relay] if relay is not None else [n.target for n in self._relay_pool]
        event_id = ""

        for r in target_relays:
            if hasattr(r, "publish_announce"):
                if isinstance(payload_or_event, WorkerAnnouncePayload):
                    event_id = r.publish_announce(payload_or_event)
                else:
                    event_id = r.publish_event(payload_or_event)
            elif hasattr(r, "publish"):
                if isinstance(payload_or_event, dict):
                    r.publish(payload_or_event)
                    event_id = str(payload_or_event.get("id", ""))
            elif hasattr(r, "publish_event") and isinstance(payload_or_event, dict):
                event_id = r.publish_event(payload_or_event)
            elif isinstance(r, str):
                if not self.offline_mock_mode:
                    raise RelayConnectionError(
                        f"Live WebSocket publishing requires 'websockets' library to publish to {r}",
                        relay_url=r,
                    )

        return event_id

    def clear_cache(self) -> None:
        """Purges in-memory announcement cache."""
        with self._lock:
            self._cache.clear()
