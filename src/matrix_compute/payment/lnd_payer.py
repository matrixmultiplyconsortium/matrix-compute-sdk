"""
LND Wallet Payer for matrix-compute-sdk.

Interfaces with the Consumer's Lightning node (e.g. Polar 'consumer-lnd')
via REST / routerrpc API to pay BOLT11 HODL invoices, locking satoshis into
real HTLCs in the IN_FLIGHT state until Freivalds verification and preimage release.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import ssl
import threading
import time
from typing import Any, Dict, List, Optional, Union

import httpx

from matrix_compute.exceptions import PaymentError

logger = logging.getLogger(__name__)


def hex_to_base64(hex_str: str) -> str:
    """Converts a hex string to standard base64 string."""
    clean = hex_str.strip()
    return base64.b64encode(bytes.fromhex(clean)).decode("ascii")


def base64_to_hex(b64_str: str) -> str:
    """Converts a base64 string to lowercase hex string."""
    clean = b64_str.strip()
    return base64.b64decode(clean).hex().lower()


class LndPayer:
    """
    Consumer Lightning wallet client for locking HODL invoice escrows in Polar / LND.
    Initiates off-chain payments using LND routerrpc (SendPaymentV2) and verifies
    HTLC lock status (IN_FLIGHT).
    """

    def __init__(
        self,
        rest_url: Optional[str] = None,
        macaroon_hex: Optional[str] = None,
        macaroon_path: Optional[str] = None,
        tls_cert: Optional[str] = None,
        tls_path: Optional[str] = None,
        timeout_seconds: float = 30.0,
        verify_ssl: bool = False,
        transport: Optional[httpx.BaseTransport] = None,
    ) -> None:
        self.rest_url = self._resolve_url(rest_url)
        self.macaroon_hex = self._resolve_macaroon(macaroon_hex, macaroon_path)
        self.tls_cert = tls_cert or os.environ.get("CONSUMER_LND_TLS_CERT", os.environ.get("LND_TLS_CERT"))
        self.tls_path = tls_path or os.environ.get("CONSUMER_LND_TLS_PATH", os.environ.get("LND_TLS_PATH"))
        self.timeout_seconds = timeout_seconds
        self.verify_ssl = verify_ssl
        self.transport = transport
        self._lock = threading.RLock()
        self._client: Optional[httpx.Client] = None
        self._in_flight_payments: Dict[str, Dict[str, Any]] = {}

    def _resolve_url(self, explicit_url: Optional[str]) -> str:
        url = (
            explicit_url
            or os.environ.get("CONSUMER_LND_REST_URL")
            or os.environ.get("LND_REST_URL")
            or "https://127.0.0.1:8081"
        )
        url = url.strip().rstrip("/")
        if not url.startswith("http://") and not url.startswith("https://"):
            url = f"https://{url}"
        return url

    def _resolve_macaroon(self, hex_val: Optional[str], path_val: Optional[str]) -> str:
        if hex_val and hex_val.strip():
            return hex_val.strip()
        env_hex = os.environ.get("CONSUMER_LND_MACAROON_HEX") or os.environ.get("LND_MACAROON_HEX")
        if env_hex and env_hex.strip():
            return env_hex.strip()

        path = path_val or os.environ.get("CONSUMER_LND_MACAROON_PATH") or os.environ.get("LND_MACAROON_PATH")
        if path and os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    return f.read().hex()
            except Exception as ex:
                logger.warning("Could not read consumer macaroon at %s: %s", path, ex)
        return ""

    def get_client(self) -> httpx.Client:
        with self._lock:
            if self._client is None or self._client.is_closed:
                headers = {"Content-Type": "application/json"}
                if self.macaroon_hex:
                    headers["Grpc-Metadata-Macaroon"] = self.macaroon_hex

                verify: Union[bool, str, ssl.SSLContext] = False
                if self.tls_path and os.path.exists(self.tls_path):
                    verify = self.tls_path
                elif self.tls_cert:
                    ctx = ssl.create_default_context()
                    ctx.load_verify_locations(cadata=self.tls_cert)
                    verify = ctx
                elif self.verify_ssl:
                    verify = True

                self._client = httpx.Client(
                    base_url=self.rest_url,
                    headers=headers,
                    timeout=self.timeout_seconds,
                    verify=verify,
                    transport=self.transport,
                )
            return self._client

    def close(self) -> None:
        with self._lock:
            if self._client is not None and not self._client.is_closed:
                self._client.close()
                self._client = None

    def pay_invoice(
        self,
        payment_request: str,
        timeout_seconds: Optional[float] = None,
        fee_limit_sat: int = 100,
        wait_for_in_flight: bool = True,
    ) -> Dict[str, Any]:
        """
        Initiates off-chain payment for a BOLT11 invoice via LND routerrpc POST /v2/router/send.
        For HODL invoices, waits until the payment status transitions to 'IN_FLIGHT'
        indicating the HTLC has been routed and held by the settlement platform.
        """
        if not payment_request or not payment_request.strip():
            raise PaymentError("BOLT11 payment_request cannot be empty")

        clean_pr = payment_request.strip()
        timeout = int(timeout_seconds or self.timeout_seconds)

        payload = {
            "payment_request": clean_pr,
            "timeout_seconds": timeout,
            "fee_limit_sat": str(fee_limit_sat),
        }

        client = self.get_client()
        ready_event = threading.Event()
        result_box: Dict[str, Any] = {"update": None, "error": None}

        def _stream_reader() -> None:
            try:
                req = client.build_request("POST", "/v2/router/send", json=payload)
                resp = client.send(req, stream=True)

                if resp.status_code not in (200, 201):
                    resp.read()
                    result_box["error"] = PaymentError(
                        f"LND SendPaymentV2 HTTP error {resp.status_code}: {resp.text}"
                    )
                    resp.close()
                    ready_event.set()
                    return

                payment_info: Dict[str, Any] = {}
                try:
                    for line in resp.iter_lines():
                        if not line:
                            continue
                        try:
                            raw_obj = json.loads(line)
                        except Exception:
                            continue

                        if isinstance(raw_obj, dict) and "error" in raw_obj and raw_obj["error"]:
                            result_box["error"] = PaymentError(
                                f"LND SendPaymentV2 stream error: {raw_obj['error']}"
                            )
                            ready_event.set()
                            return

                        update = (
                            raw_obj.get("result", raw_obj)
                            if isinstance(raw_obj, dict)
                            else raw_obj
                        )
                        if not isinstance(update, dict):
                            continue

                        status = str(update.get("status", "")).upper()
                        payment_info = update

                        if status == "IN_FLIGHT":
                            with self._lock:
                                self._in_flight_payments[clean_pr] = update
                            if result_box["update"] is None:
                                result_box["update"] = update
                            if wait_for_in_flight:
                                ready_event.set()

                        elif status == "SUCCEEDED":
                            with self._lock:
                                self._in_flight_payments[clean_pr] = update
                            result_box["update"] = update
                            ready_event.set()
                            return

                        elif status == "FAILED":
                            failure_reason = update.get("failure_reason", "UNKNOWN_FAILURE")
                            result_box["error"] = PaymentError(
                                f"LND payment failed with status {status}: {failure_reason}"
                            )
                            ready_event.set()
                            return
                finally:
                    resp.close()

                if not ready_event.is_set():
                    if payment_info:
                        result_box["update"] = payment_info
                    else:
                        result_box["error"] = PaymentError(
                            f"LND SendPaymentV2 closed stream without yielding payment confirmation for {clean_pr[:20]}..."
                        )
                    ready_event.set()

            except httpx.HTTPError as ex:
                if not ready_event.is_set():
                    result_box["error"] = PaymentError(
                        f"Network error communicating with Consumer LND: {ex}"
                    )
                    ready_event.set()
            except Exception as ex:
                if not ready_event.is_set():
                    result_box["error"] = PaymentError(
                        f"Unexpected error communicating with Consumer LND: {ex}"
                    )
                    ready_event.set()

        reader_thread = threading.Thread(
            target=_stream_reader,
            daemon=True,
            name=f"lnd-send-{clean_pr[-8:]}",
        )
        reader_thread.start()

        if not ready_event.wait(timeout=float(timeout) + 5.0):
            raise PaymentError(
                f"Timed out waiting {timeout}s for Consumer LND payment status on {clean_pr[:20]}..."
            )

        if result_box["error"] is not None:
            raise result_box["error"]

        return result_box["update"] or {}

    def get_payment_status(self, payment_hash_or_request: str) -> Optional[str]:
        """
        Returns the recorded payment status ('IN_FLIGHT', 'SUCCEEDED', 'FAILED', or None).
        """
        with self._lock:
            info = self._in_flight_payments.get(payment_hash_or_request)
            if info:
                return str(info.get("status", "")).upper()
        return None

    def get_info(self) -> Dict[str, Any]:
        """Queries Consumer LND node info via GET /v1/getinfo."""
        client = self.get_client()
        resp = client.get("/v1/getinfo")
        resp.raise_for_status()
        return resp.json()

    def connect_peer(self, pubkey_hex: str, host_port: str, perm: bool = True) -> Dict[str, Any]:
        """Idempotently connects Consumer LND to a remote Lightning peer via POST /v1/peers."""
        clean_pk = pubkey_hex.strip().lower()
        client = self.get_client()
        peers_resp = client.get("/v1/peers")
        if peers_resp.status_code == 200:
            for p in peers_resp.json().get("peers", []):
                if str(p.get("pub_key", "")).lower() == clean_pk:
                    return {"status": "ALREADY_CONNECTED", "pub_key": clean_pk}

        payload = {"addr": {"pubkey": clean_pk, "host": host_port.strip()}, "perm": perm}
        resp = client.post("/v1/peers", json=payload)
        if resp.status_code in (200, 201) or "already connected" in resp.text.lower():
            return {"status": "CONNECTED", "pub_key": clean_pk}
        resp.raise_for_status()
        return resp.json()

    def list_channels(self, peer_pubkey_hex: Optional[str] = None) -> List[Dict[str, Any]]:
        """Lists open channels on Consumer LND via GET /v1/channels."""
        client = self.get_client()
        resp = client.get("/v1/channels")
        resp.raise_for_status()
        channels = list(resp.json().get("channels", []))
        if peer_pubkey_hex:
            target = peer_pubkey_hex.strip().lower()
            channels = [
                ch for ch in channels
                if str(ch.get("remote_pubkey", "")).lower() == target
            ]
        return channels

    def open_channel(
        self,
        node_pubkey_hex: str,
        local_funding_amount: int = 100_000,
        push_sat: int = 0,
        min_confs: int = 1,
        sat_per_vbyte: int = 1,
        spend_unconfirmed: bool = True,
    ) -> Dict[str, Any]:
        """Opens a new Lightning channel from Consumer LND to Settlement LND via POST /v1/channels."""
        client = self.get_client()
        payload = {
            "node_pubkey_string": node_pubkey_hex.strip(),
            "local_funding_amount": str(int(local_funding_amount)),
            "push_sat": str(int(push_sat)),
            "min_confs": int(min_confs),
            "sat_per_vbyte": str(int(sat_per_vbyte)),
            "spend_unconfirmed": bool(spend_unconfirmed),
        }
        resp = client.post("/v1/channels", json=payload)
        resp.raise_for_status()
        return resp.json()

    def ensure_channel_to_peer(
        self,
        peer_pubkey_hex: str,
        peer_host_port: Optional[str] = None,
        capacity_sats: int = 100_000,
        min_outbound_sats: int = 5_000,
    ) -> Dict[str, Any]:
        """Ensures Consumer LND has an open or pending channel to Settlement LND."""
        clean_pk = peer_pubkey_hex.strip().lower()
        if peer_host_port:
            self.connect_peer(clean_pk, peer_host_port)
        existing = self.list_channels(peer_pubkey_hex=clean_pk)
        for ch in existing:
            if int(ch.get("local_balance", 0)) >= min_outbound_sats:
                return {"status": "ACTIVE" if ch.get("active") else "INACTIVE", "channel": ch}
        funding = self.open_channel(node_pubkey_hex=clean_pk, local_funding_amount=capacity_sats)
        return {"status": "OPENING", "funding": funding}
