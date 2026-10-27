"""
Hermetic Unit Tests for NWC Client, NIP-47 Protocol & EscrowManager Integration.
Executes 100% offline using in-process local WebSocket relay servers.
"""

import asyncio
import base64
import hashlib
import json
import secrets
import threading
import time
from typing import Dict, List, Optional
import pytest
import websockets

from matrix_compute.client import MatrixComputeClient
from matrix_compute.config import ClientConfig
from matrix_compute.crypto.nip04 import nip04_decrypt, nip04_encrypt
from matrix_compute.crypto.signer import ClientSigner
from matrix_compute.exceptions import (
    EscrowStateError,
    PaymentError,
    PreimageVerificationError,
)
from matrix_compute.payment.escrow import EscrowManager, EscrowStatus
from matrix_compute.payment.nwc_client import (
    NwcClient,
    NwcConnectionParams,
    PaymentTimeoutError,
)
from matrix_compute.test_doubles.mock_settlement import MockSettlementService


# -----------------------------------------------------------------------------
# Fixtures & Ephemeral Mock Relay Server
# -----------------------------------------------------------------------------

@pytest.fixture
def crypto_keys():
    client_sec = "1111111111111111111111111111111111111111111111111111111111111111"
    wallet_sec = "2222222222222222222222222222222222222222222222222222222222222222"
    client_pub = ClientSigner(client_sec).get_nostr_pubkey().lower()
    wallet_pub = ClientSigner(wallet_sec).get_nostr_pubkey().lower()
    return {
        "client_sec": client_sec,
        "client_pub": client_pub,
        "wallet_sec": wallet_sec,
        "wallet_pub": wallet_pub,
    }


class EphemeralMockNwcRelay:
    """Local in-process WebSocket relay simulating an NWC wallet service."""

    def __init__(self, wallet_sec: str, client_pub: str, behavior: str = "success"):
        self.wallet_sec = wallet_sec
        self.wallet_pub = ClientSigner(wallet_sec).get_nostr_pubkey().lower()
        self.client_pub = client_pub.lower()
        self.behavior = behavior  # "success", "insufficient_balance", "rate_limited", "timeout", "bad_hash"
        self.server = None
        self.port = 0
        self.url = ""
        self.received_requests: List[Dict] = []
        self._preimages: Dict[str, str] = {}
        self._loop = None
        self._thread = None
        self._stop_event = threading.Event()

    async def _ws_handler(self, websocket):
        sub_id = None
        try:
            async for raw in websocket:
                msg = json.loads(raw)
                verb = msg[0]
                if verb == "REQ":
                    sub_id = msg[1]
                    await websocket.send(json.dumps(["EOSE", sub_id]))
                elif verb == "CLOSE":
                    pass
                elif verb == "EVENT":
                    ev = msg[1]
                    self.received_requests.append(ev)
                    req_id = ev["id"]

                    if self.behavior == "timeout":
                        # Simulate wallet ignoring / dropping request
                        continue

                    # Decrypt request
                    content = nip04_decrypt(self.wallet_sec, self.client_pub, ev["content"])
                    req_body = json.loads(content)

                    # Synthesize response
                    if self.behavior == "success":
                        pr = req_body["params"]["invoice"]
                        found_preimage = self._preimages.get(pr)
                        if not found_preimage:
                            for k, v in self._preimages.items():
                                if k in pr or pr in k:
                                    found_preimage = v
                                    break
                        if not found_preimage:
                            for em in getattr(EscrowManager, "_instances", []):
                                for tid, esc in getattr(em, "_escrows", {}).items():
                                    if esc.payment_request == pr or (esc.payment_hash and esc.payment_hash[:20] in pr) or (esc.payment_hash and esc.payment_hash in pr):
                                        found_preimage = esc._preimage
                                        break
                                if found_preimage:
                                    break
                        if not found_preimage:
                            for inst in getattr(MockSettlementService, "_instances", []):
                                for p_hash, inv in inst.invoices.items():
                                    if inv.payment_request == pr or p_hash in pr or p_hash[:20] in pr:
                                        found_preimage = inv.preimage
                                        break
                                if found_preimage:
                                    break
                        if not found_preimage:
                            found_preimage = secrets.token_bytes(32).hex().lower()
                        self._preimages[pr] = found_preimage
                        resp_payload = {"result_type": "pay_invoice", "result": {"preimage": found_preimage}, "error": None}
                    elif self.behavior == "bad_hash":
                        resp_payload = {"result_type": "pay_invoice", "result": {"preimage": "00" * 32}, "error": None}
                    elif self.behavior == "insufficient_balance":
                        resp_payload = {"result_type": "pay_invoice", "result": None, "error": {"code": "INSUFFICIENT_BALANCE", "message": "Low funds"}}
                    elif self.behavior == "rate_limited":
                        resp_payload = {"result_type": "pay_invoice", "result": None, "error": {"code": "RATE_LIMITED", "message": "Slow down"}}
                    else:
                        resp_payload = {"result_type": "pay_invoice", "result": None, "error": {"code": "PAYMENT_FAILED", "message": "Route failed"}}

                    # Encrypt response
                    enc_resp = nip04_encrypt(self.wallet_sec, self.client_pub, json.dumps(resp_payload))
                    resp_event = {
                        "id": hashlib.sha256(f"resp_{req_id}".encode()).hexdigest(),
                        "pubkey": self.wallet_pub,
                        "created_at": int(time.time()),
                        "kind": 23195,
                        "tags": [["p", self.client_pub], ["e", req_id]],
                        "content": enc_resp,
                        "sig": "mock_sig_wallet",
                    }

                    if sub_id:
                        await websocket.send(json.dumps(["EVENT", sub_id, resp_event]))
        except Exception:
            pass

    def start(self):
        ready = threading.Event()

        def _run():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

            async def _start():
                self.server = await websockets.serve(self._ws_handler, "127.0.0.1", 0)
                self.port = self.server.sockets[0].getsockname()[1]
                self.url = f"ws://127.0.0.1:{self.port}"
                ready.set()

            self._loop.run_until_complete(_start())
            self._loop.run_forever()

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        ready.wait(timeout=5.0)

    def stop(self):
        if self._loop and self.server and self._loop.is_running():
            async def _close():
                self.server.close()
                await self.server.wait_closed()
            try:
                asyncio.run_coroutine_threadsafe(_close(), self._loop).result(timeout=2.0)
            except Exception:
                pass
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread:
                self._thread.join(timeout=2.0)


@pytest.fixture
def mock_nwc_relay(crypto_keys):
    relay = EphemeralMockNwcRelay(crypto_keys["wallet_sec"], crypto_keys["client_pub"], behavior="success")
    relay.start()
    yield relay
    relay.stop()


# -----------------------------------------------------------------------------
# Unit Tests
# -----------------------------------------------------------------------------

def test_nwc_params_from_uri_valid(crypto_keys):
    uri = f"nostr+walletconnect://{crypto_keys['wallet_pub']}?relay=wss://relay.damus.io&secret={crypto_keys['client_sec']}&lud16=alice@test.com"
    params = NwcConnectionParams.from_uri(uri)
    assert params.wallet_pubkey == crypto_keys["wallet_pub"]
    assert params.relay_url == "wss://relay.damus.io"
    assert params.secret == crypto_keys["client_sec"]
    assert params.lud16 == "alice@test.com"


def test_nwc_params_from_uri_invalid():
    with pytest.raises(ValueError, match="nostr\\+walletconnect://"):
        NwcConnectionParams.from_uri("https://example.com")
    with pytest.raises(ValueError, match="Missing required 'secret'"):
        NwcConnectionParams.from_uri("nostr+walletconnect://00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff?relay=ws://127.0.0.1")
    with pytest.raises(ValueError, match="Missing required 'relay'"):
        NwcConnectionParams.from_uri("nostr+walletconnect://00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff?secret=00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff")


def test_nwc_client_build_request_event(crypto_keys):
    uri = f"nostr+walletconnect://{crypto_keys['wallet_pub']}?relay=ws://127.0.0.1:8000&secret={crypto_keys['client_sec']}"
    client = NwcClient(uri)

    ev = client.build_request_event("pay_invoice", {"invoice": "lnbc100n1mock"})
    assert ev["kind"] == 23194
    assert ev["pubkey"] == crypto_keys["client_pub"]
    assert ev["tags"] == [["p", crypto_keys["wallet_pub"]]]
    assert "?iv=" in ev["content"]
    assert len(ev["id"]) == 64
    assert len(ev["sig"]) == 128


def test_nwc_client_pay_invoice_success(crypto_keys, mock_nwc_relay):
    uri = f"nostr+walletconnect://{crypto_keys['wallet_pub']}?relay={mock_nwc_relay.url}&secret={crypto_keys['client_sec']}"
    client = NwcClient(uri, timeout_seconds=5.0)

    res = client.pay_invoice("lnbc1000n1p_test_invoice")
    assert res["status"] == "SETTLED"
    assert len(res["preimage"]) == 64
    assert res["payment_hash"] == hashlib.sha256(bytes.fromhex(res["preimage"])).hexdigest().lower()


def test_nwc_client_insufficient_balance_error(crypto_keys):
    relay = EphemeralMockNwcRelay(crypto_keys["wallet_sec"], crypto_keys["client_pub"], behavior="insufficient_balance")
    relay.start()
    try:
        uri = f"nostr+walletconnect://{crypto_keys['wallet_pub']}?relay={relay.url}&secret={crypto_keys['client_sec']}"
        client = NwcClient(uri, timeout_seconds=5.0)

        with pytest.raises(PaymentError, match="INSUFFICIENT_BALANCE"):
            client.pay_invoice("lnbc1000n1p_test_invoice")
    finally:
        relay.stop()


def test_nwc_client_rate_limited_error(crypto_keys):
    relay = EphemeralMockNwcRelay(crypto_keys["wallet_sec"], crypto_keys["client_pub"], behavior="rate_limited")
    relay.start()
    try:
        uri = f"nostr+walletconnect://{crypto_keys['wallet_pub']}?relay={relay.url}&secret={crypto_keys['client_sec']}"
        client = NwcClient(uri, timeout_seconds=5.0)

        with pytest.raises(PaymentError, match="RATE_LIMITED"):
            client.pay_invoice("lnbc1000n1p_test_invoice")
    finally:
        relay.stop()


def test_nwc_client_timeout_error(crypto_keys):
    relay = EphemeralMockNwcRelay(crypto_keys["wallet_sec"], crypto_keys["client_pub"], behavior="timeout")
    relay.start()
    try:
        uri = f"nostr+walletconnect://{crypto_keys['wallet_pub']}?relay={relay.url}&secret={crypto_keys['client_sec']}"
        client = NwcClient(uri, timeout_seconds=0.5)

        with pytest.raises(PaymentTimeoutError, match="timed out"):
            client.pay_invoice("lnbc1000n1p_test_invoice")
    finally:
        relay.stop()


def test_nwc_client_preimage_hash_mismatch(crypto_keys):
    relay = EphemeralMockNwcRelay(crypto_keys["wallet_sec"], crypto_keys["client_pub"], behavior="bad_hash")
    relay.start()
    try:
        uri = f"nostr+walletconnect://{crypto_keys['wallet_pub']}?relay={relay.url}&secret={crypto_keys['client_sec']}"
        client = NwcClient(uri, timeout_seconds=5.0)

        with pytest.raises(PreimageVerificationError):
            client.pay_invoice("lnbc1000n1p_test_invoice", expected_payment_hash="ff" * 32)
    finally:
        relay.stop()


def test_nwc_client_empty_invoice_rejected(crypto_keys):
    uri = f"nostr+walletconnect://{crypto_keys['wallet_pub']}?relay=ws://127.0.0.1:8000&secret={crypto_keys['client_sec']}"
    client = NwcClient(uri)

    with pytest.raises(PaymentError, match="cannot be empty"):
        client.pay_invoice("")


def test_nwc_client_close_idempotent(crypto_keys):
    uri = f"nostr+walletconnect://{crypto_keys['wallet_pub']}?relay=ws://127.0.0.1:8000&secret={crypto_keys['client_sec']}"
    client = NwcClient(uri)
    client.close()
    client.close()


def test_escrow_manager_hold_and_release_with_nwc(crypto_keys, mock_nwc_relay):
    uri = f"nostr+walletconnect://{crypto_keys['wallet_pub']}?relay={mock_nwc_relay.url}&secret={crypto_keys['client_sec']}"
    config = ClientConfig(nwc_uri=uri)
    settlement = MockSettlementService()

    mgr = EscrowManager(settlement_backend=settlement, config=config, default_timeout_seconds=5.0)

    # 1. Create escrow
    escrow = mgr.create_escrow("task_nwc_1", total_sats=500)
    assert escrow.status == EscrowStatus.INITIALIZED

    # 2. Hold escrow -> Routes via NwcClient
    held = mgr.hold_escrow("task_nwc_1")
    assert held.status == EscrowStatus.HELD
    assert held.held_at is not None
    assert held.preimage is not None

    # 3. Release payout -> Settles with revealed preimage
    settled = mgr.release_payout("task_nwc_1")
    assert settled.status == EscrowStatus.SETTLED
    assert settlement.get_status(escrow.payment_hash) == "SETTLED"

    mgr.close()


def test_matrix_compute_client_nwc_uri_integration(crypto_keys, mock_nwc_relay):
    uri = f"nostr+walletconnect://{crypto_keys['wallet_pub']}?relay={mock_nwc_relay.url}&secret={crypto_keys['client_sec']}"
    client = MatrixComputeClient(nwc_connection_uri=uri, offline_mock_mode=True)
    assert client.config.nwc_uri == uri
    assert client.escrow_manager.nwc_client is not None
    client.close()
