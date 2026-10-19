"""
Unit tests for LndPayer in matrix-compute-sdk.

Verifies:
- Calling LND routerrpc (SendPaymentV2) via REST POST /v2/router/send
- Detecting HTLC lock state (status: IN_FLIGHT)
- Handling routing/balance failures (status: FAILED)
- EscrowManager integration: hold_escrow invokes lnd_payer.pay_invoice
"""

import json
from typing import Any, Dict

import httpx
import pytest

from matrix_compute.exceptions import EscrowStateError, PaymentError
from matrix_compute.payment.escrow import EscrowManager, EscrowStatus
from matrix_compute.payment.lnd_payer import LndPayer
from matrix_compute.test_doubles.mock_settlement import MockSettlementService


class MockConsumerLndHandler:
    """Mock handler simulating Polar's 'consumer-lnd' node."""

    def __init__(self, simulate_failure: bool = False, failure_reason: str = "FAILURE_REASON_NO_ROUTE") -> None:
        self.simulate_failure = simulate_failure
        self.failure_reason = failure_reason
        self.payment_requests: list = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/v2/router/send":
            body = json.loads(request.content.decode("utf-8"))
            self.payment_requests.append(body)

            if self.simulate_failure:
                stream_content = (
                    json.dumps({"status": "FAILED", "failure_reason": self.failure_reason}) + "\n"
                )
            else:
                stream_content = (
                    json.dumps({
                        "status": "IN_FLIGHT",
                        "payment_hash": "abcd1234",
                        "value_sat": "1000",
                        "fee_sat": "0",
                    }) + "\n"
                )

            return httpx.Response(
                200,
                content=stream_content.encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )

        return httpx.Response(404, json={"error": "not found"})


def test_lnd_payer_pay_invoice_success():
    handler = MockConsumerLndHandler()
    transport = httpx.MockTransport(handler.handle_request)

    payer = LndPayer(
        rest_url="https://consumer-lnd:8081",
        macaroon_hex="deadbeef1234",
        transport=transport,
    )

    bolt11 = "lnbcrt1000u1pabcdef1234567890"
    result = payer.pay_invoice(bolt11, timeout_seconds=15.0)

    assert result["status"] == "IN_FLIGHT"
    assert len(handler.payment_requests) == 1
    assert handler.payment_requests[0]["payment_request"] == bolt11
    assert handler.payment_requests[0]["timeout_seconds"] == 15
    assert payer.get_payment_status(bolt11) == "IN_FLIGHT"


def test_lnd_payer_pay_invoice_failure_raises():
    handler = MockConsumerLndHandler(simulate_failure=True, failure_reason="FAILURE_REASON_INSUFFICIENT_BALANCE")
    transport = httpx.MockTransport(handler.handle_request)

    payer = LndPayer(
        rest_url="https://consumer-lnd:8081",
        macaroon_hex="deadbeef1234",
        transport=transport,
    )

    bolt11 = "lnbcrt1000u1pabcdef1234567890"
    with pytest.raises(PaymentError) as exc_info:
        payer.pay_invoice(bolt11)

    assert "INSUFFICIENT_BALANCE" in str(exc_info.value)


def test_lnd_payer_empty_invoice_raises():
    payer = LndPayer(rest_url="https://consumer-lnd:8081")
    with pytest.raises(PaymentError):
        payer.pay_invoice("")


def test_escrow_manager_integration_with_lnd_payer():
    handler = MockConsumerLndHandler()
    transport = httpx.MockTransport(handler.handle_request)

    payer = LndPayer(
        rest_url="https://consumer-lnd:8081",
        macaroon_hex="deadbeef1234",
        transport=transport,
    )

    mock_settlement = MockSettlementService()
    manager = EscrowManager(
        settlement_backend=mock_settlement,
        lnd_payer=payer,
    )

    # 1. Create escrow
    escrow = manager.create_escrow(
        task_id="task_test_payer_01",
        total_sats=1000,
        worker_lightning_addr="worker@node",
    )
    assert escrow.status == EscrowStatus.INITIALIZED
    assert escrow.payment_request is not None

    # 2. Hold escrow via LndPayer
    held_escrow = manager.hold_escrow("task_test_payer_01")
    assert held_escrow.status == EscrowStatus.HELD
    assert len(handler.payment_requests) == 1
    assert handler.payment_requests[0]["payment_request"] == escrow.payment_request
