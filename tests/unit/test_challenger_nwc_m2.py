"""
Milestone M2 Adversarial Challenge Test Suite: Native Nostr Wallet Connect (NWC / NIP-47) SDK Client.

Empirically challenges matrix-compute-sdk across 6 adversarial attack dimensions:
1. Exhaustive NIP-47 JSON-RPC Error Codes (standard and non-standard) & Malformed Error Payloads.
2. Timeout Adversarial Conditions (silent relays, delayed responses, dead ports, connection drops).
3. Preimage Verification Attacks (rogue preimages, single-bit mutations, invalid lengths/charsets, auto-hash inference).
4. Adversarial Wire & Relay Protocol Injection (tag spoofing, mismatched e/p tags, chatter noise, ciphertext tampering).
5. Concurrency, Contention & Thread Safety Stress (broadcast relays, multi-threaded clients, mixed outcome concurrency).
6. NWC Connection Params & URI Boundary Fuzzing.

Authored by: Challenger 2 (Empirical Challenger)
Integrity mode: Hermetic offline execution
"""

from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import hashlib
import json
import secrets
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import pytest
import websockets

from matrix_compute.client import MatrixComputeClient
from matrix_compute.config import ClientConfig
from matrix_compute.crypto.nip04 import nip04_decrypt, nip04_encrypt
from matrix_compute.crypto.session import KeyExchangeError
from matrix_compute.crypto.signer import ClientSigner
from matrix_compute.exceptions import (
    EscrowStateError,
    PaymentError,
    PreimageVerificationError,
    TaskTimeoutError,
)
from matrix_compute.payment.escrow import EscrowManager, EscrowStatus
from matrix_compute.payment.nwc_client import (
    NwcClient,
    NwcConnectionParams,
    PaymentTimeoutError,
    parse_nwc_uri,
)
from matrix_compute.test_doubles.mock_settlement import MockSettlementService


# -----------------------------------------------------------------------------
# Cryptographic Fixtures & Adversarial Mock Relay
# -----------------------------------------------------------------------------

@pytest.fixture
def crypto_keys():
    """Deterministic Secp256k1 keypairs for client and wallet service."""
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


class AdversarialMockNwcRelay:
    """
    Advanced in-process WebSocket relay simulating an adversarial NWC wallet service.
    Supports injecting malformed responses, errors, corrupt crypto, timeouts, and broadcast mode.
    """

    def __init__(self, wallet_sec: str, client_pub: str):
        self.wallet_sec = wallet_sec
        self.wallet_pub = ClientSigner(wallet_sec).get_nostr_pubkey().lower()
        self.client_pub = client_pub.lower()
        self.server = None
        self.port = 0
        self.url = ""
        self._loop = None
        self._thread = None
        self.active_websockets: Set[Any] = set()
        self.received_requests: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

        # Behavior controls
        self.default_behavior: str = "success"  # "success", "silent", "drop"
        self.custom_responses: Dict[str, Any] = {}
        self.custom_errors: Dict[str, Dict[str, Any]] = {}
        self.silent_invoices: Set[str] = set()
        self.injected_messages: List[str] = []
        self.delay_seconds: float = 0.0
        self.omit_e_tag: bool = False
        self.target_p_tag: Optional[str] = None
        self.target_e_tag: Optional[str] = None
        self.drop_immediately: bool = False
        self.raw_content_override: Optional[str] = None
        self.broadcast_mode: bool = False
        self.preimage_map: Dict[str, str] = {}

    async def _ws_handler(self, websocket):
        with self._lock:
            self.active_websockets.add(websocket)
        sub_id: Optional[str] = None
        try:
            async for raw in websocket:
                if self.drop_immediately:
                    await websocket.close()
                    break

                try:
                    msg = json.loads(raw)
                except Exception:
                    continue

                if not isinstance(msg, list) or len(msg) < 2:
                    continue

                verb = msg[0]
                if verb == "REQ":
                    sub_id = msg[1]
                    await websocket.send(json.dumps(["EOSE", sub_id]))
                elif verb == "CLOSE":
                    pass
                elif verb == "EVENT":
                    ev = msg[1]
                    with self._lock:
                        self.received_requests.append(ev)
                    req_id = ev.get("id", "")

                    if self.default_behavior == "silent":
                        continue

                    # Decrypt request body
                    invoice = ""
                    try:
                        content = nip04_decrypt(self.wallet_sec, self.client_pub, ev["content"])
                        req_body = json.loads(content)
                        invoice = req_body.get("params", {}).get("invoice", "")
                    except Exception:
                        req_body = {}

                    if invoice in self.silent_invoices:
                        continue

                    if self.delay_seconds > 0:
                        await asyncio.sleep(self.delay_seconds)

                    # Send injected messages (noise, chatter, malformed strings)
                    for inj in self.injected_messages:
                        await websocket.send(inj)

                    # Determine payload
                    if invoice in self.custom_errors:
                        err_info = self.custom_errors[invoice]
                        resp_payload = {"result_type": "pay_invoice", "result": None, "error": err_info}
                    elif invoice in self.custom_responses:
                        resp_payload = self.custom_responses[invoice]
                    elif req_id in self.custom_responses:
                        resp_payload = self.custom_responses[req_id]
                    elif self.default_behavior == "success":
                        preimage = self.preimage_map.get(invoice)
                        if not preimage:
                            preimage = secrets.token_bytes(32).hex().lower()
                            self.preimage_map[invoice] = preimage
                        resp_payload = {
                            "result_type": "pay_invoice",
                            "result": {"preimage": preimage, "fees_paid": 10},
                            "error": None,
                        }
                    else:
                        resp_payload = {
                            "result_type": "pay_invoice",
                            "result": None,
                            "error": {"code": "PAYMENT_FAILED", "message": "Simulated default error"},
                        }

                    # Determine content encryption
                    if self.raw_content_override is not None:
                        enc_content = self.raw_content_override
                    else:
                        plaintext_resp = json.dumps(resp_payload)
                        enc_content = nip04_encrypt(self.wallet_sec, self.client_pub, plaintext_resp)

                    # Build tags
                    p_val = self.target_p_tag if self.target_p_tag is not None else self.client_pub
                    tags = [["p", p_val]]
                    if not self.omit_e_tag:
                        e_val = self.target_e_tag if self.target_e_tag is not None else req_id
                        tags.append(["e", e_val])

                    resp_event = {
                        "id": hashlib.sha256(f"resp_{req_id}_{time.time()}".encode()).hexdigest(),
                        "pubkey": self.wallet_pub,
                        "created_at": int(time.time()),
                        "kind": 23195,
                        "tags": tags,
                        "content": enc_content,
                        "sig": "mock_sig_wallet",
                    }

                    target_sockets = list(self.active_websockets) if self.broadcast_mode else [websocket]
                    target_sub = sub_id if sub_id else "sub_default"

                    for ws in target_sockets:
                        try:
                            await ws.send(json.dumps(["EVENT", target_sub, resp_event]))
                        except Exception:
                            pass
        except Exception:
            pass
        finally:
            with self._lock:
                self.active_websockets.discard(websocket)

    def start(self) -> None:
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

    def stop(self) -> None:
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
def adv_relay(crypto_keys):
    relay = AdversarialMockNwcRelay(crypto_keys["wallet_sec"], crypto_keys["client_pub"])
    relay.start()
    yield relay
    relay.stop()


# =============================================================================
# Attack Dimension 1: Exhaustive NIP-47 Error Code Mappings & Malformed Errors
# =============================================================================

class TestAdversarialErrorCodes:
    """Stress tests NIP-47 error code mapping and handling against adversarial inputs."""

    @pytest.mark.parametrize(
        "error_code, error_message",
        [
            ("RATE_LIMITED", "Client rate limited, slow down"),
            ("INSUFFICIENT_BALANCE", "Account balance insufficient for invoice payment"),
            ("PAYMENT_FAILED", "Unable to route payment across lightning network"),
            ("QUOTA_EXCEEDED", "Periodic spending quota exceeded for this application"),
            ("UNAUTHORIZED", "Client public key not authorized to execute pay_invoice"),
            ("RESTRICTED", "Wallet spending policy restricts this destination or amount"),
            ("NOT_FOUND", "Requested payment hash or invoice not found in wallet store"),
            ("INTERNAL", "Internal database error on remote lightning node"),
            ("OTHER", "Unspecified error from lightning backend"),
            ("CUSTOM_VENDOR_CODE_999", "Vendor-specific custom failure reason"),
        ],
    )
    def test_all_nip47_error_codes_raise_payment_error(
        self, crypto_keys, adv_relay, error_code, error_message
    ):
        invoice = f"lnbc_test_{error_code.lower()}"
        adv_relay.custom_errors[invoice] = {"code": error_code, "message": error_message}

        uri = f"nostr+walletconnect://{crypto_keys['wallet_pub']}?relay={adv_relay.url}&secret={crypto_keys['client_sec']}"
        client = NwcClient(uri, timeout_seconds=4.0)

        with pytest.raises(PaymentError) as exc_info:
            client.pay_invoice(invoice)

        err_str = str(exc_info.value)
        assert error_code in err_str
        assert error_message in err_str
        assert exc_info.value.details.get("code") == error_code

    def test_malformed_string_error_object(self, crypto_keys, adv_relay):
        invoice = "lnbc_str_err"
        adv_relay.custom_responses[invoice] = {
            "result_type": "pay_invoice",
            "result": None,
            "error": "Generic wallet error string without dict",
        }

        uri = f"nostr+walletconnect://{crypto_keys['wallet_pub']}?relay={adv_relay.url}&secret={crypto_keys['client_sec']}"
        client = NwcClient(uri, timeout_seconds=4.0)

        with pytest.raises(PaymentError) as exc_info:
            client.pay_invoice(invoice)

        assert "Generic wallet error string" in str(exc_info.value)

    def test_malformed_integer_error_code(self, crypto_keys, adv_relay):
        invoice = "lnbc_int_err"
        adv_relay.custom_responses[invoice] = {
            "result_type": "pay_invoice",
            "result": None,
            "error": {"code": 500, "message": "HTTP style numeric code"},
        }

        uri = f"nostr+walletconnect://{crypto_keys['wallet_pub']}?relay={adv_relay.url}&secret={crypto_keys['client_sec']}"
        client = NwcClient(uri, timeout_seconds=4.0)

        with pytest.raises(PaymentError) as exc_info:
            client.pay_invoice(invoice)

        assert "500" in str(exc_info.value)
        assert "HTTP style numeric code" in str(exc_info.value)

    def test_error_dict_missing_code_field(self, crypto_keys, adv_relay):
        invoice = "lnbc_no_code"
        adv_relay.custom_responses[invoice] = {
            "result_type": "pay_invoice",
            "result": None,
            "error": {"message": "Missing code field completely"},
        }

        uri = f"nostr+walletconnect://{crypto_keys['wallet_pub']}?relay={adv_relay.url}&secret={crypto_keys['client_sec']}"
        client = NwcClient(uri, timeout_seconds=4.0)

        with pytest.raises(PaymentError) as exc_info:
            client.pay_invoice(invoice)

        assert "UNKNOWN_ERROR" in str(exc_info.value)
        assert "Missing code field completely" in str(exc_info.value)

    def test_escrow_manager_hold_escrow_fails_cleanly_on_all_error_codes(
        self, crypto_keys, adv_relay
    ):
        invoice = "lnbc_escrow_quota_exceeded"
        adv_relay.custom_errors[invoice] = {
            "code": "QUOTA_EXCEEDED",
            "message": "Weekly budget limit reached",
        }

        uri = f"nostr+walletconnect://{crypto_keys['wallet_pub']}?relay={adv_relay.url}&secret={crypto_keys['client_sec']}"
        config = ClientConfig(nwc_uri=uri)

        settlement = MockSettlementService()
        mgr = EscrowManager(settlement_backend=settlement, config=config, default_timeout_seconds=4.0)

        escrow = mgr.create_escrow("task_adv_err_1", total_sats=1000)
        escrow.payment_request = invoice
        original_preimage = escrow._preimage

        with pytest.raises(EscrowStateError) as exc_info:
            mgr.hold_escrow("task_adv_err_1")

        assert "QUOTA_EXCEEDED" in str(exc_info.value)
        # Verify escrow status remains INITIALIZED, never HELD
        assert escrow.status == EscrowStatus.INITIALIZED
        assert escrow._preimage == original_preimage
        mgr.close()


# =============================================================================
# Attack Dimension 2: Timeout Adversarial Conditions
# =============================================================================

class TestAdversarialTimeoutHandling:
    """Stress tests timeout handling under unresponsive or degraded relay conditions."""

    def test_silent_relay_triggers_payment_timeout_error(self, crypto_keys, adv_relay):
        adv_relay.default_behavior = "silent"

        uri = f"nostr+walletconnect://{crypto_keys['wallet_pub']}?relay={adv_relay.url}&secret={crypto_keys['client_sec']}"
        client = NwcClient(uri, timeout_seconds=0.3)

        t0 = time.time()
        with pytest.raises(PaymentTimeoutError) as exc_info:
            client.pay_invoice("lnbc_silent_test")
        elapsed = time.time() - t0

        assert 0.25 <= elapsed <= 1.5
        assert isinstance(exc_info.value, PaymentError)
        assert isinstance(exc_info.value, TaskTimeoutError)
        assert exc_info.value.timeout_seconds == 0.3

    def test_delayed_response_exceeding_timeout_deadline(self, crypto_keys, adv_relay):
        # Relay responds at t=0.8s, client timeout is 0.2s
        adv_relay.delay_seconds = 0.8

        uri = f"nostr+walletconnect://{crypto_keys['wallet_pub']}?relay={adv_relay.url}&secret={crypto_keys['client_sec']}"
        client = NwcClient(uri, timeout_seconds=0.2)

        t0 = time.time()
        with pytest.raises(PaymentTimeoutError):
            client.pay_invoice("lnbc_delayed_test")
        elapsed = time.time() - t0

        assert elapsed < 0.7  # Client gave up before the 0.8s response arrived

    def test_unreachable_dead_relay_port_fails_with_payment_error(self, crypto_keys):
        # Port 1 is reserved and guaranteed closed/refused on localhost
        dead_uri = f"nostr+walletconnect://{crypto_keys['wallet_pub']}?relay=ws://127.0.0.1:1&secret={crypto_keys['client_sec']}"
        client = NwcClient(dead_uri, timeout_seconds=0.5)

        with pytest.raises(PaymentError) as exc_info:
            client.pay_invoice("lnbc_dead_port")

        assert "Failed communicating with NWC relay" in str(exc_info.value)

    def test_abrupt_relay_socket_closure(self, crypto_keys, adv_relay):
        adv_relay.drop_immediately = True

        uri = f"nostr+walletconnect://{crypto_keys['wallet_pub']}?relay={adv_relay.url}&secret={crypto_keys['client_sec']}"
        client = NwcClient(uri, timeout_seconds=0.5)

        with pytest.raises(PaymentError):
            client.pay_invoice("lnbc_drop_immediately")

    def test_escrow_manager_hold_escrow_timeout_preserves_initialized_status(
        self, crypto_keys, adv_relay
    ):
        adv_relay.default_behavior = "silent"

        uri = f"nostr+walletconnect://{crypto_keys['wallet_pub']}?relay={adv_relay.url}&secret={crypto_keys['client_sec']}"
        config = ClientConfig(nwc_uri=uri)

        settlement = MockSettlementService()
        mgr = EscrowManager(settlement_backend=settlement, config=config, default_timeout_seconds=0.3)

        escrow = mgr.create_escrow("task_timeout_escrow", total_sats=500)

        with pytest.raises(EscrowStateError) as exc_info:
            mgr.hold_escrow("task_timeout_escrow")

        assert isinstance(exc_info.value.cause, PaymentTimeoutError)
        assert escrow.status == EscrowStatus.INITIALIZED
        mgr.close()


# =============================================================================
# Attack Dimension 3: Preimage Verification Attacks
# =============================================================================

class TestAdversarialPreimageAttacks:
    """Stress tests preimage verification against rogue, corrupted, and manipulated preimages."""

    def test_rogue_random_preimage_rejected(self, crypto_keys, adv_relay):
        expected_preimage = secrets.token_bytes(32).hex().lower()
        expected_hash = hashlib.sha256(bytes.fromhex(expected_preimage)).hexdigest().lower()

        rogue_preimage = secrets.token_bytes(32).hex().lower()
        assert rogue_preimage != expected_preimage

        invoice = "lnbc_rogue_preimage"
        adv_relay.custom_responses[invoice] = {
            "result_type": "pay_invoice",
            "result": {"preimage": rogue_preimage},
            "error": None,
        }

        uri = f"nostr+walletconnect://{crypto_keys['wallet_pub']}?relay={adv_relay.url}&secret={crypto_keys['client_sec']}"
        client = NwcClient(uri, timeout_seconds=4.0)

        with pytest.raises(PreimageVerificationError) as exc_info:
            client.pay_invoice(invoice, expected_payment_hash=expected_hash)

        assert exc_info.value.payment_hash == expected_hash

    @pytest.mark.parametrize("flipped_byte_idx", [0, 15, 31])
    def test_single_bit_mutation_on_preimage_rejected(
        self, crypto_keys, adv_relay, flipped_byte_idx
    ):
        raw_preimage = bytearray(secrets.token_bytes(32))
        expected_hash = hashlib.sha256(raw_preimage).hexdigest().lower()

        # Flip 1 bit in specified byte
        raw_preimage[flipped_byte_idx] ^= 0x01
        corrupted_preimage = raw_preimage.hex().lower()

        invoice = f"lnbc_bitflip_{flipped_byte_idx}"
        adv_relay.custom_responses[invoice] = {
            "result_type": "pay_invoice",
            "result": {"preimage": corrupted_preimage},
            "error": None,
        }

        uri = f"nostr+walletconnect://{crypto_keys['wallet_pub']}?relay={adv_relay.url}&secret={crypto_keys['client_sec']}"
        client = NwcClient(uri, timeout_seconds=4.0)

        with pytest.raises(PreimageVerificationError):
            client.pay_invoice(invoice, expected_payment_hash=expected_hash)

    def test_auto_hash_deduction_from_lnbc_invoice_with_rogue_preimage(
        self, crypto_keys, adv_relay
    ):
        true_preimage = secrets.token_bytes(32).hex().lower()
        true_hash = hashlib.sha256(bytes.fromhex(true_preimage)).hexdigest().lower()

        invoice_69_chars = f"lnbc_{true_hash}"
        assert len(invoice_69_chars) == 69

        # Supply forged preimage
        rogue_preimage = "aa" * 32
        adv_relay.custom_responses[invoice_69_chars] = {
            "result_type": "pay_invoice",
            "result": {"preimage": rogue_preimage},
            "error": None,
        }

        uri = f"nostr+walletconnect://{crypto_keys['wallet_pub']}?relay={adv_relay.url}&secret={crypto_keys['client_sec']}"
        client = NwcClient(uri, timeout_seconds=4.0)

        # Do NOT pass expected_payment_hash explicitly; verify auto-deduction catches mismatch
        with pytest.raises(PreimageVerificationError) as exc_info:
            client.pay_invoice(invoice_69_chars)

        assert exc_info.value.payment_hash == true_hash

    @pytest.mark.parametrize(
        "invalid_preimage, reason",
        [
            ("00" * 31, "62 chars: too short"),
            ("00" * 31 + "0", "63 chars: odd hex length"),
            ("00" * 33, "66 chars: too long"),
            ("", "empty string"),
            ("   ", "whitespace only"),
            ("0123456789abcdef" * 3 + "0123456789abcdeg", "non-hex char 'g' at end"),
            ("zz" * 32, "completely non-hex chars"),
        ],
    )
    def test_preimage_format_violations_raise_payment_error(
        self, crypto_keys, adv_relay, invalid_preimage, reason
    ):
        invoice = f"lnbc_fmt_{hash(invalid_preimage)}"
        adv_relay.custom_responses[invoice] = {
            "result_type": "pay_invoice",
            "result": {"preimage": invalid_preimage},
            "error": None,
        }

        uri = f"nostr+walletconnect://{crypto_keys['wallet_pub']}?relay={adv_relay.url}&secret={crypto_keys['client_sec']}"
        client = NwcClient(uri, timeout_seconds=4.0)

        with pytest.raises(PaymentError, match="Invalid preimage format"):
            client.pay_invoice(invoice)

    def test_case_insensitive_preimage_and_hash_normalization(self, crypto_keys, adv_relay):
        raw_preimage = secrets.token_bytes(32)
        target_hash = hashlib.sha256(raw_preimage).hexdigest()

        # Deliver uppercase preimage from wallet
        uppercase_preimage = raw_preimage.hex().upper()
        # Pass uppercase expected hash
        uppercase_hash = target_hash.upper()

        invoice = "lnbc_case_normalization"
        adv_relay.custom_responses[invoice] = {
            "result_type": "pay_invoice",
            "result": {"preimage": uppercase_preimage},
            "error": None,
        }

        uri = f"nostr+walletconnect://{crypto_keys['wallet_pub']}?relay={adv_relay.url}&secret={crypto_keys['client_sec']}"
        client = NwcClient(uri, timeout_seconds=4.0)

        res = client.pay_invoice(invoice, expected_payment_hash=uppercase_hash)
        assert res["status"] == "SETTLED"
        # Preimage must be normalized to canonical lowercase
        assert res["preimage"] == raw_preimage.hex().lower()
        assert res["payment_hash"] == target_hash.lower()

    @pytest.mark.parametrize(
        "bad_result_payload",
        [
            {"result": None},
            {"result": {}},  # missing "preimage"
            {"result": "not_a_dict"},
            {"result": {"preimage": None}},
            {"result": {"preimage": 123456}},
        ],
    )
    def test_malformed_result_structure_raises_payment_error(
        self, crypto_keys, adv_relay, bad_result_payload
    ):
        invoice = f"lnbc_bad_result_{hash(str(bad_result_payload))}"
        payload = {"result_type": "pay_invoice", "error": None}
        payload.update(bad_result_payload)
        adv_relay.custom_responses[invoice] = payload

        uri = f"nostr+walletconnect://{crypto_keys['wallet_pub']}?relay={adv_relay.url}&secret={crypto_keys['client_sec']}"
        client = NwcClient(uri, timeout_seconds=4.0)

        with pytest.raises(PaymentError):
            client.pay_invoice(invoice)

    def test_escrow_manager_prevents_held_status_on_preimage_attack(
        self, crypto_keys, adv_relay
    ):
        expected_secret = "11" * 32
        expected_hash = hashlib.sha256(bytes.fromhex(expected_secret)).hexdigest()
        rogue_secret = "99" * 32

        invoice = "lnbc_escrow_preimage_attack"
        adv_relay.custom_responses[invoice] = {
            "result_type": "pay_invoice",
            "result": {"preimage": rogue_secret},
            "error": None,
        }

        uri = f"nostr+walletconnect://{crypto_keys['wallet_pub']}?relay={adv_relay.url}&secret={crypto_keys['client_sec']}"
        config = ClientConfig(nwc_uri=uri)

        settlement = MockSettlementService()
        mgr = EscrowManager(settlement_backend=settlement, config=config, default_timeout_seconds=4.0)

        escrow = mgr.create_escrow("task_preimage_attack", total_sats=800)
        escrow.payment_request = invoice
        escrow.payment_hash = expected_hash
        original_preimage = escrow._preimage

        with pytest.raises(EscrowStateError) as exc_info:
            mgr.hold_escrow("task_preimage_attack")

        assert isinstance(exc_info.value.cause, PreimageVerificationError)
        assert escrow.status == EscrowStatus.INITIALIZED
        assert escrow._preimage == original_preimage  # Rogue preimage was NOT accepted into escrow
        mgr.close()


# =============================================================================
# Attack Dimension 4: Wire & Relay Protocol Adversarial Injection
# =============================================================================

class TestAdversarialWireAndRelayProtocol:
    """Stress tests protocol resilience against hijacked tags, noise, and tampered ciphertext."""

    def test_mismatched_p_tag_ignored_until_timeout(self, crypto_keys, adv_relay):
        # Spoof #p tag to a foreign pubkey (Eve)
        eve_pubkey = "ee" * 32
        adv_relay.target_p_tag = eve_pubkey

        uri = f"nostr+walletconnect://{crypto_keys['wallet_pub']}?relay={adv_relay.url}&secret={crypto_keys['client_sec']}"
        client = NwcClient(uri, timeout_seconds=0.3)

        with pytest.raises(PaymentTimeoutError):
            client.pay_invoice("lnbc_spoofed_p_tag")

    def test_mismatched_e_tag_ignored_until_timeout(self, crypto_keys, adv_relay):
        # Spoof #e tag to reference a different request ID
        adv_relay.target_e_tag = "00" * 32

        uri = f"nostr+walletconnect://{crypto_keys['wallet_pub']}?relay={adv_relay.url}&secret={crypto_keys['client_sec']}"
        client = NwcClient(uri, timeout_seconds=0.3)

        with pytest.raises(PaymentTimeoutError):
            client.pay_invoice("lnbc_spoofed_e_tag")

    def test_relay_noise_and_extraneous_messages_handled_transparently(
        self, crypto_keys, adv_relay
    ):
        # Inject extraneous Nostr messages before the real response
        adv_relay.injected_messages = [
            json.dumps(["NOTICE", "Relay maintenance scheduled in 10 minutes"]),
            "MALFORMED_NON_JSON_LINE_FROM_RELAY",
            json.dumps(["EVENT", "fake_sub", {"kind": 1, "content": "Just a chat note", "tags": []}]),
            json.dumps(["EVENT", "fake_sub", {"kind": 23194, "content": "Rogue request", "tags": []}]),
            json.dumps(["EOSE", "fake_sub"]),
            json.dumps(["OK", "some_event_id", True, "saved"]),
        ]

        uri = f"nostr+walletconnect://{crypto_keys['wallet_pub']}?relay={adv_relay.url}&secret={crypto_keys['client_sec']}"
        client = NwcClient(uri, timeout_seconds=4.0)

        # Real response should still be cleanly received despite chatter
        res = client.pay_invoice("lnbc_chatter_test")
        assert res["status"] == "SETTLED"
        assert len(res["preimage"]) == 64

    def test_tampered_ciphertext_payload_fails_cleanly(self, crypto_keys, adv_relay):
        # Provide corrupt ciphertext format
        adv_relay.raw_content_override = "CORRUPTED_BASE64_CIPHERTEXT?iv=AAECAwQFBgcICQoLDA0ODw=="

        uri = f"nostr+walletconnect://{crypto_keys['wallet_pub']}?relay={adv_relay.url}&secret={crypto_keys['client_sec']}"
        client = NwcClient(uri, timeout_seconds=4.0)

        with pytest.raises(PaymentError, match="Failed decrypting NWC response payload"):
            client.pay_invoice("lnbc_tampered_crypto")

    def test_corrupted_iv_payload_fails_cleanly(self, crypto_keys, adv_relay):
        valid_b64_cipher = base64.b64encode(b"0" * 32).decode()
        adv_relay.raw_content_override = f"{valid_b64_cipher}?iv=CORRUPTED_IV"

        uri = f"nostr+walletconnect://{crypto_keys['wallet_pub']}?relay={adv_relay.url}&secret={crypto_keys['client_sec']}"
        client = NwcClient(uri, timeout_seconds=4.0)

        with pytest.raises(PaymentError, match="Failed decrypting NWC response payload"):
            client.pay_invoice("lnbc_corrupt_iv")

    def test_decrypted_content_not_json_fails_cleanly(self, crypto_keys, adv_relay):
        # Valid encryption of non-JSON string
        raw_non_json = "THIS IS NOT VALID JSON AT ALL {[[["
        enc = nip04_encrypt(crypto_keys["wallet_sec"], crypto_keys["client_pub"], raw_non_json)
        adv_relay.raw_content_override = enc

        uri = f"nostr+walletconnect://{crypto_keys['wallet_pub']}?relay={adv_relay.url}&secret={crypto_keys['client_sec']}"
        client = NwcClient(uri, timeout_seconds=4.0)

        with pytest.raises(PaymentError, match="Failed decrypting NWC response payload"):
            client.pay_invoice("lnbc_non_json_plaintext")

    def test_decrypted_content_is_json_array_fails_cleanly(self, crypto_keys, adv_relay):
        raw_json_array = json.dumps(["an", "array", "not", "an", "object"])
        enc = nip04_encrypt(crypto_keys["wallet_sec"], crypto_keys["client_pub"], raw_json_array)
        adv_relay.raw_content_override = enc

        uri = f"nostr+walletconnect://{crypto_keys['wallet_pub']}?relay={adv_relay.url}&secret={crypto_keys['client_sec']}"
        client = NwcClient(uri, timeout_seconds=4.0)

        with pytest.raises(PaymentError, match="NWC response payload must be a JSON object"):
            client.pay_invoice("lnbc_json_array_plaintext")


# =============================================================================
# Attack Dimension 5: Concurrency, Contention & Thread Safety Stress
# =============================================================================

class TestConcurrencyAndContentionStress:
    """Stress tests rapid concurrent payments, broadcast multiplexing, and mixed outcomes."""

    def test_broadcast_relay_concurrency_no_crosstalk(self, crypto_keys, adv_relay):
        """
        In a broadcast relay environment, every response event is broadcast to all connections.
        Verify that 10 concurrent threads on the SAME NwcClient receive only their matching preimage.
        """
        adv_relay.broadcast_mode = True
        num_threads = 10

        invoices = [f"lnbc_concurrent_test_{i}" for i in range(num_threads)]
        preimages = {inv: secrets.token_bytes(32).hex().lower() for inv in invoices}
        for inv, p in preimages.items():
            adv_relay.preimage_map[inv] = p

        uri = f"nostr+walletconnect://{crypto_keys['wallet_pub']}?relay={adv_relay.url}&secret={crypto_keys['client_sec']}"
        client = NwcClient(uri, timeout_seconds=6.0)

        results: Dict[str, str] = {}
        errors: List[Exception] = []
        lock = threading.Lock()

        def _worker(inv: str):
            try:
                res = client.pay_invoice(inv)
                with lock:
                    results[inv] = res["preimage"]
            except Exception as e:
                with lock:
                    errors.append(e)

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as pool:
            futures = [pool.submit(_worker, inv) for inv in invoices]
            concurrent.futures.wait(futures, timeout=10.0)

        assert len(errors) == 0, f"Encountered errors during concurrent execution: {errors}"
        assert len(results) == num_threads

        for inv in invoices:
            expected_p = preimages[inv]
            actual_p = results[inv]
            assert actual_p == expected_p, f"Crosstalk detected on invoice {inv}: expected {expected_p}, got {actual_p}"

    def test_mixed_outcome_concurrency(self, crypto_keys, adv_relay):
        """
        Stress test 10 concurrent requests with mixed outcomes:
        - 4 success
        - 2 insufficient balance
        - 2 rate limited
        - 2 timeout
        """
        adv_relay.broadcast_mode = True

        success_invs = [f"lnbc_mix_succ_{i}" for i in range(4)]
        bal_invs = [f"lnbc_mix_bal_{i}" for i in range(2)]
        rate_invs = [f"lnbc_mix_rate_{i}" for i in range(2)]
        time_invs = [f"lnbc_mix_time_{i}" for i in range(2)]

        for inv in bal_invs:
            adv_relay.custom_errors[inv] = {"code": "INSUFFICIENT_BALANCE", "message": "Low funds"}
        for inv in rate_invs:
            adv_relay.custom_errors[inv] = {"code": "RATE_LIMITED", "message": "Throttled"}
        for inv in time_invs:
            adv_relay.silent_invoices.add(inv)

        all_invoices = success_invs + bal_invs + rate_invs + time_invs

        uri = f"nostr+walletconnect://{crypto_keys['wallet_pub']}?relay={adv_relay.url}&secret={crypto_keys['client_sec']}"
        client = NwcClient(uri, timeout_seconds=1.0)

        outcomes: Dict[str, Any] = {}
        lock = threading.Lock()

        def _worker(inv: str):
            try:
                if inv in time_invs:
                    res = client.pay_invoice(inv, timeout_seconds=0.25)
                else:
                    res = client.pay_invoice(inv, timeout_seconds=4.0)
                with lock:
                    outcomes[inv] = ("SUCCESS", res["preimage"])
            except PaymentTimeoutError:
                with lock:
                    outcomes[inv] = ("TIMEOUT", None)
            except PaymentError as pe:
                code = pe.details.get("code")
                with lock:
                    outcomes[inv] = ("ERROR", code)
            except Exception as e:
                with lock:
                    outcomes[inv] = ("UNEXPECTED", str(e))

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
            futures = [pool.submit(_worker, inv) for inv in all_invoices]
            concurrent.futures.wait(futures, timeout=10.0)

        assert len(outcomes) == 10
        for inv in success_invs:
            assert outcomes[inv][0] == "SUCCESS"
        for inv in bal_invs:
            assert outcomes[inv] == ("ERROR", "INSUFFICIENT_BALANCE")
        for inv in rate_invs:
            assert outcomes[inv] == ("ERROR", "RATE_LIMITED")
        for inv in time_invs:
            assert outcomes[inv] == ("TIMEOUT", None)

    def test_rapid_sequential_payments_stability(self, crypto_keys, adv_relay):
        """Execute 30 rapid sequential payments to ensure no resource leakage or socket exhaustion."""
        uri = f"nostr+walletconnect://{crypto_keys['wallet_pub']}?relay={adv_relay.url}&secret={crypto_keys['client_sec']}"
        client = NwcClient(uri, timeout_seconds=3.0)

        for i in range(30):
            inv = f"lnbc_seq_stress_{i}"
            res = client.pay_invoice(inv)
            assert res["status"] == "SETTLED"
            assert len(res["preimage"]) == 64

    def test_multi_client_instance_concurrency(self, crypto_keys, adv_relay):
        """5 distinct NwcClient instances interacting with the relay simultaneously."""
        num_clients = 5
        clients = [
            NwcClient(
                f"nostr+walletconnect://{crypto_keys['wallet_pub']}?relay={adv_relay.url}&secret={crypto_keys['client_sec']}",
                timeout_seconds=4.0,
            )
            for _ in range(num_clients)
        ]

        results = []
        errors = []

        def _client_task(c: NwcClient, idx: int):
            try:
                res = c.pay_invoice(f"lnbc_multi_client_{idx}")
                results.append(res)
            except Exception as e:
                errors.append(e)

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_clients) as pool:
            futures = [pool.submit(_client_task, c, i) for i, c in enumerate(clients)]
            concurrent.futures.wait(futures, timeout=6.0)

        assert len(errors) == 0
        assert len(results) == num_clients


# =============================================================================
# Attack Dimension 6: NWC Connection Params & URI Boundary Fuzzing
# =============================================================================

class TestNwcConnectionParamsBoundary:
    """Stress tests URI parsing, boundary cases, normalization, and parameter validation."""

    def test_uri_with_multiple_custom_query_params(self, crypto_keys):
        uri = (
            f"nostr+walletconnect://{crypto_keys['wallet_pub']}?"
            f"relay=wss://relay1.com&secret={crypto_keys['client_sec']}&"
            f"custom_param_1=value1&lud16=bob@matrix.org&custom_param_2=999"
        )
        params = NwcConnectionParams.from_uri(uri)
        assert params.wallet_pubkey == crypto_keys["wallet_pub"]
        assert params.relay_url == "wss://relay1.com"
        assert params.lud16 == "bob@matrix.org"

    def test_uri_with_port_and_path(self, crypto_keys):
        uri = f"nostr+walletconnect://{crypto_keys['wallet_pub']}?relay=ws://192.168.1.100:9999/nostr/relay&secret={crypto_keys['client_sec']}"
        params = NwcConnectionParams.from_uri(uri)
        assert params.relay_url == "ws://192.168.1.100:9999/nostr/relay"

    def test_roundtrip_to_uri_and_from_uri(self, crypto_keys):
        original_uri = (
            f"nostr+walletconnect://{crypto_keys['wallet_pub']}?"
            f"relay=wss://relay.damus.io&secret={crypto_keys['client_sec']}&lud16=user@domain.com"
        )
        p1 = NwcConnectionParams.from_uri(original_uri)
        reconstructed_uri = p1.to_uri()
        p2 = NwcConnectionParams.from_uri(reconstructed_uri)

        assert p1.wallet_pubkey == p2.wallet_pubkey
        assert p1.relay_url == p2.relay_url
        assert p1.secret == p2.secret
        assert p1.lud16 == p2.lud16

    @pytest.mark.parametrize(
        "invalid_pubkey",
        [
            "00" * 31,  # 62 chars
            "00" * 33,  # 66 chars
            "not_hex_pubkey_value_0123456789abcdef0123456789abcdef0123456789abc",
        ],
    )
    def test_rejected_invalid_pubkeys(self, crypto_keys, invalid_pubkey):
        uri = f"nostr+walletconnect://{invalid_pubkey}?relay=ws://127.0.0.1&secret={crypto_keys['client_sec']}"
        with pytest.raises(ValueError):
            NwcConnectionParams.from_uri(uri)

    def test_off_curve_pubkey_rejected_by_crypto_layer(self, crypto_keys):
        off_curve_pubkey = "ff" * 32
        uri = f"nostr+walletconnect://{off_curve_pubkey}?relay=ws://127.0.0.1:8000&secret={crypto_keys['client_sec']}"
        client = NwcClient(uri)
        with pytest.raises(KeyExchangeError, match="does not lie on secp256k1 curve"):
            client.build_request_event("pay_invoice", {"invoice": "lnbc100n"})

    @pytest.mark.parametrize(
        "invalid_secret",
        [
            "00" * 32,  # scalar zero
            "fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141",  # secp256k1 order N
            "00" * 31,  # too short
            "zz" * 32,  # non-hex
        ],
    )
    def test_rejected_invalid_secrets(self, crypto_keys, invalid_secret):
        uri = f"nostr+walletconnect://{crypto_keys['wallet_pub']}?relay=ws://127.0.0.1&secret={invalid_secret}"
        with pytest.raises(ValueError):
            NwcConnectionParams.from_uri(uri)
