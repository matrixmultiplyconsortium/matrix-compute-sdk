"""In-memory Mock Nostr Discovery Relay test double for matrix-compute-sdk."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from matrix_contracts.schemas.discovery import (
        MatchedWorker,
        WorkerAnnouncePayload,
        WorkerQueryRequest,
        WorkerQueryResponse,
    )
except ImportError:
    from matrix_contracts.schemas import (
        MatchedWorker,
        WorkerAnnouncePayload,
        WorkerQueryRequest,
        WorkerQueryResponse,
    )


class MockNostrRelay:
    """
    Hermetic in-memory Nostr Discovery Relay test double.
    Implements NIP-01 and NIP-33 parameterized replaceable events (Kind 30000).
    Thread-safe across all publish, query, and administrative operations.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._replaceable_events: Dict[Tuple[int, str, str], Dict[str, Any]] = {}
        self._events_by_id: Dict[str, Dict[str, Any]] = {}
        self._tag_index: Dict[str, Dict[str, Set[str]]] = {}

    def _extract_d_tag(self, tags: List[List[str]]) -> str:
        for t in tags:
            if isinstance(t, (list, tuple)) and len(t) >= 2 and t[0] == "d":
                return str(t[1])
        return "matrix-worker"

    def _extract_timestamp(self, event: Dict[str, Any]) -> int:
        for key in ("created_at", "timestamp", "ts"):
            if key in event and event[key] is not None:
                try:
                    return int(event[key])
                except (ValueError, TypeError):
                    pass
        return int(time.time())

    @staticmethod
    def _parse_event_specs(event: Dict[str, Any]) -> Dict[str, Any]:
        """Parses hardware, pricing, and connectivity specs with JSON content fallback."""
        tag_map: Dict[str, str] = {}
        for t in event.get("tags", []):
            if isinstance(t, (list, tuple)) and len(t) >= 2:
                tag_map[str(t[0])] = str(t[1])

        p2p = tag_map.get("p2p", "")
        lightning = tag_map.get("lightning", "")
        gpu_model = tag_map.get("gpu_model", "")

        price: Optional[float] = None
        if "sats_per_gflop" in tag_map:
            try:
                price = float(tag_map["sats_per_gflop"])
            except (ValueError, TypeError):
                price = None

        vram_mb: Optional[int] = None
        if "vram_mb" in tag_map:
            try:
                vram_mb = int(tag_map["vram_mb"])
            except (ValueError, TypeError):
                vram_mb = None

        # Content fallback if any specification is missing
        if not p2p or not lightning or not gpu_model or price is None or vram_mb is None:
            raw_content = event.get("content", "{}")
            data: Dict[str, Any] = {}
            if isinstance(raw_content, dict):
                data = raw_content
            elif isinstance(raw_content, str):
                try:
                    parsed = json.loads(raw_content)
                    if isinstance(parsed, dict):
                        data = parsed
                except Exception:
                    data = {}

            if data:
                if not p2p:
                    p2p = str(data.get("p2p_multiaddr", ""))
                if not lightning:
                    lightning = str(data.get("lightning_routing_pubkey", ""))
                if not gpu_model:
                    hw = data.get("hardware", {})
                    if isinstance(hw, dict):
                        gpu_model = str(hw.get("gpu_model", ""))
                if price is None:
                    pricing = data.get("pricing", {})
                    if isinstance(pricing, dict) and "sats_per_gflop" in pricing:
                        try:
                            price = float(pricing["sats_per_gflop"])
                        except (ValueError, TypeError):
                            pass
                if vram_mb is None:
                    hw = data.get("hardware", {})
                    if isinstance(hw, dict) and "vram_mb" in hw:
                        try:
                            vram_mb = int(hw["vram_mb"])
                        except (ValueError, TypeError):
                            pass

        return {
            "p2p": p2p,
            "lightning": lightning,
            "gpu_model": gpu_model,
            "price": price if price is not None else 0.0,
            "vram_mb": vram_mb if vram_mb is not None else 0,
        }

    def publish(self, event: Dict[str, Any]) -> bool:
        """Publishes an event according to NIP-33 parameterized replaceable rules."""
        kind = int(event.get("kind", 1))
        pubkey = str(event.get("pubkey", ""))
        tags = event.get("tags", [])
        content = event.get("content", "")
        ts = self._extract_timestamp(event)
        d_tag = self._extract_d_tag(tags)
        key = (kind, pubkey, d_tag)

        event_id = event.get("id")
        if not event_id:
            serialized = json.dumps(
                [0, pubkey, ts, kind, tags, str(content)],
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            event_id = hashlib.sha256(serialized).hexdigest()
            event["id"] = event_id

        with self._lock:
            if key in self._replaceable_events:
                existing = self._replaceable_events[key]
                existing_ts = self._extract_timestamp(existing)
                if ts < existing_ts:
                    return True
                old_id = existing.get("id")
                if old_id:
                    self._events_by_id.pop(old_id, None)
                    self._unindex_tags(old_id, existing.get("tags", []))

            stored_event = dict(event)
            stored_event["id"] = event_id
            stored_event["created_at"] = ts
            self._replaceable_events[key] = stored_event
            self._events_by_id[event_id] = stored_event
            self._index_tags(event_id, tags)
            return True

    def publish_event(self, event: Dict[str, Any]) -> str:
        self.publish(event)
        return str(event.get("id", ""))

    def publish_announce(self, payload: WorkerAnnouncePayload) -> str:
        content_json = payload.model_dump_json()
        tags = [
            ["d", "matrix-worker"],
            ["p2p", payload.p2p_multiaddr],
            ["lightning", payload.lightning_routing_pubkey],
            ["vram_mb", str(payload.hardware.vram_mb)],
            ["sats_per_gflop", str(payload.pricing.sats_per_gflop)],
            ["gpu_model", payload.hardware.gpu_model],
            ["peak_fp32_tflops", str(payload.hardware.peak_fp32_tflops)],
        ]
        created_at = payload.timestamp or int(time.time())
        serialized = json.dumps(
            [0, payload.worker_pubkey, created_at, 30000, tags, content_json],
            separators=(",", ":"),
        ).encode("utf-8")
        event_id = hashlib.sha256(serialized).hexdigest()

        event = {
            "id": event_id,
            "pubkey": payload.worker_pubkey,
            "created_at": created_at,
            "kind": 30000,
            "tags": tags,
            "content": content_json,
            "sig": f"mock_schnorr_{event_id[:16]}",
        }
        self.publish(event)
        return event_id

    def query(
        self,
        min_vram_mb: int = 0,
        max_price: Optional[float] = None,
        gpu_model: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        with self._lock:
            events_snapshot = list(self._replaceable_events.values())

        results: List[Dict[str, Any]] = []
        for event in events_snapshot:
            if int(event.get("kind", 0)) != 30000:
                continue

            specs = self._parse_event_specs(event)
            if specs["vram_mb"] < min_vram_mb:
                continue
            if max_price is not None and specs["price"] > max_price:
                continue
            if gpu_model is not None and gpu_model.lower() not in specs["gpu_model"].lower():
                continue
            results.append(event)
        return results

    def query_workers(self, request: WorkerQueryRequest) -> WorkerQueryResponse:
        matching_events = self.query(
            min_vram_mb=request.min_vram_mb,
            max_price=request.max_price_sats_per_gflop,
        )
        candidates: List[Tuple[float, int, MatchedWorker]] = []
        for event in matching_events:
            specs = self._parse_event_specs(event)
            worker = MatchedWorker(
                worker_pubkey=str(event.get("pubkey", "")),
                p2p_multiaddr=specs["p2p"],
                lightning_routing_pubkey=specs["lightning"],
                reputation_score=1.0,
                sats_per_gflop=specs["price"],
            )
            candidates.append((specs["price"], -specs["vram_mb"], worker))

        candidates.sort(key=lambda x: (x[0], x[1]))
        matched = [c[2] for c in candidates[:request.required_slots]]
        return WorkerQueryResponse(status="SUCCESS", matched_workers=matched)

    def get_events(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._replaceable_events.values())

    def get_event(self, event_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._events_by_id.get(event_id)

    def get_active_announcement(self, pubkey: str, d_tag: str = "matrix-worker") -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._replaceable_events.get((30000, pubkey, d_tag))

    def get_events_by_d_tag(self, d_tag: str) -> List[Dict[str, Any]]:
        with self._lock:
            return [ev for (k, pk, d), ev in self._replaceable_events.items() if d == d_tag]

    def _index_tags(self, event_id: str, tags: List[List[str]]) -> None:
        for t in tags:
            if isinstance(t, (list, tuple)) and len(t) >= 2:
                name, val = str(t[0]), str(t[1])
                if name not in self._tag_index:
                    self._tag_index[name] = {}
                if val not in self._tag_index[name]:
                    self._tag_index[name][val] = set()
                self._tag_index[name][val].add(event_id)

    def _unindex_tags(self, event_id: str, tags: List[List[str]]) -> None:
        for t in tags:
            if isinstance(t, (list, tuple)) and len(t) >= 2:
                name, val = str(t[0]), str(t[1])
                if name in self._tag_index and val in self._tag_index[name]:
                    self._tag_index[name][val].discard(event_id)

    def clear(self) -> None:
        with self._lock:
            self._replaceable_events.clear()
            self._events_by_id.clear()
            self._tag_index.clear()

    def reset(self) -> None:
        self.clear()
