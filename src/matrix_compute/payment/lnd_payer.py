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
        try:
            # POST /v2/router/send can stream payment updates as chunked JSON
            req = client.build_request("POST", "/v2/router/send", json=payload)
            resp = client.send(req, stream=True)

            if resp.status_code not in (200, 201):
                resp.read()
                raise PaymentError(f"LND SendPaymentV2 HTTP error {resp.status_code}: {resp.text}")

            payment_info: Dict[str, Any] = {}
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    update = json.loads(line)
                except Exception:
                    continue

                status = str(update.get("status", "")).upper()
                payment_hash = update.get("payment_hash", "")
                payment_info = update

                # For HODL invoices, IN_FLIGHT means the HTLC is actively held!
                if status == "IN_FLIGHT":
                    with self._lock:
                        self._in_flight_payments[clean_pr] = update
                    if wait_for_in_flight:
                        resp.close()
                        return update

                elif status == "SUCCEEDED":
                    with self._lock:
                        self._in_flight_payments[clean_pr] = update
                    resp.close()
                    return update

                elif status == "FAILED":
                    failure_reason = update.get("failure_reason", "UNKNOWN_FAILURE")
                    resp.close()
                    raise PaymentError(f"LND payment failed with status {status}: {failure_reason}")

            resp.close()
            if payment_info:
                return payment_info

            raise PaymentError(
                f"LND SendPaymentV2 closed stream without yielding payment confirmation for {clean_pr[:20]}..."
            )

        except httpx.HTTPError as ex:
            raise PaymentError(f"Network error communicating with Consumer LND: {ex}") from ex

    def get_payment_status(self, payment_hash_or_request: str) -> Optional[str]:
        """
        Returns the recorded payment status ('IN_FLIGHT', 'SUCCEEDED', 'FAILED', or None).
        """
        with self._lock:
            info = self._in_flight_payments.get(payment_hash_or_request)
            if info:
                return str(info.get("status", "")).upper()
        return None
