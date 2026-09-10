"""Unit tests for exception hierarchy in matrix-compute-sdk."""

import pytest

from matrix_compute.exceptions import (
    ConfigurationError,
    DiscoveryError,
    DispatchError,
    EscrowCreationError,
    FraudDetectedError,
    InvoiceHoldError,
    MatrixComputeError,
    MatrixDimensionError,
    PaymentError,
    RefundError,
    SettlementError,
    StreamingError,
    TaskTimeoutError,
    TilingError,
    VerificationFailedError,
    WorkerConnectionError,
    WorkerSignatureError,
    WorkerTaskRejectedError,
    WorkerUnavailableError,
)

ALL_EXCEPTION_CLASSES = [
    ConfigurationError,
    MatrixDimensionError,
    TilingError,
    DiscoveryError,
    WorkerUnavailableError,
    WorkerConnectionError,
    DispatchError,
    WorkerTaskRejectedError,
    StreamingError,
    TaskTimeoutError,
    WorkerSignatureError,
    PaymentError,
    EscrowCreationError,
    InvoiceHoldError,
    SettlementError,
    RefundError,
    VerificationFailedError,
    FraudDetectedError,
]


def test_all_exceptions_inherit_from_root():
    """Asserts that every exception in the SDK inherits from MatrixComputeError."""
    for exc_cls in ALL_EXCEPTION_CLASSES:
        assert issubclass(exc_cls, MatrixComputeError), f"{exc_cls.__name__} does not inherit from MatrixComputeError"
        assert issubclass(exc_cls, Exception)


def test_catch_all_block():
    """Verifies that a generic except MatrixComputeError catches any child exception."""
    exceptions_to_throw = [
        ConfigurationError("Config missing"),
        MatrixDimensionError("Shape mismatch", shape_a=(4, 5), shape_b=(6, 7)),
        WorkerUnavailableError("No workers", required_vram_mb=8192),
        WorkerTaskRejectedError("Rejected", task_id="t-1", reason="busy"),
        FraudDetectedError("Fraud proven", task_id="t-2", discrepancy=10.5, bound=0.01),
        SettlementError("Preimage rejected", payment_hash="hash123"),
    ]

    for err in exceptions_to_throw:
        caught = False
        try:
            raise err
        except MatrixComputeError as caught_err:
            caught = True
            assert caught_err is err
        assert caught is True


def test_matrix_dimension_error():
    err = MatrixDimensionError(
        "Inner dimensions do not match",
        shape_a=(10, 20),
        shape_b=(30, 40),
        expected_inner_dim=20,
    )
    assert err.shape_a == (10, 20)
    assert err.shape_b == (30, 40)
    assert err.expected_inner_dim == 20
    assert "shape_a=(10, 20)" in str(err)
    assert "expected_inner_dim=20" in str(err)


def test_discovery_and_worker_unavailable_error():
    err = WorkerUnavailableError(
        "No matching worker found",
        required_vram_mb=16384,
        max_price_per_gflop=0.0005,
        available_count=0,
    )
    assert isinstance(err, DiscoveryError)
    assert isinstance(err, MatrixComputeError)
    assert err.required_vram_mb == 16384
    assert err.max_price_per_gflop == 0.0005
    assert err.available_count == 0


def test_dispatch_error_hierarchy():
    rejected = WorkerTaskRejectedError("Worker busy", task_id="task-99", worker_pubkey="02pub1", reason="Overloaded")
    assert isinstance(rejected, DispatchError)
    assert rejected.task_id == "task-99"
    assert rejected.worker_pubkey == "02pub1"
    assert rejected.reason == "Overloaded"

    timeout = TaskTimeoutError("Timed out", task_id="task-100", timeout_seconds=30.0)
    assert isinstance(timeout, DispatchError)
    assert timeout.timeout_seconds == 30.0

    sig_err = WorkerSignatureError("Forged signature", task_id="task-101", worker_pubkey="03pub2", signature="00" * 64)
    assert isinstance(sig_err, DispatchError)
    assert sig_err.worker_pubkey == "03pub2"

    stream_err = StreamingError("Chunk broken")
    assert isinstance(stream_err, DispatchError)


def test_payment_error_hierarchy():
    payment_subclasses = [EscrowCreationError, InvoiceHoldError, SettlementError, RefundError]
    for sub in payment_subclasses:
        assert issubclass(sub, PaymentError)
        inst = sub("Payment failure", payment_hash="hashabc", task_id="task1", amount_sats=500)
        assert inst.payment_hash == "hashabc"
        assert inst.task_id == "task1"
        assert inst.amount_sats == 500
        assert "payment_hash='hashabc'" in str(inst)


def test_verification_and_fraud_error():
    verif_err = VerificationFailedError(
        "Verification check failed",
        task_id="task-xyz",
        worker_pubkey="02pubk",
        discrepancy=5.2,
        bound=0.001,
    )
    assert isinstance(verif_err, MatrixComputeError)
    assert verif_err.task_id == "task-xyz"
    assert verif_err.discrepancy == 5.2
    assert verif_err.bound == 0.001

    fraud_err = FraudDetectedError(
        "Worker cheated",
        task_id="task-xyz",
        worker_pubkey="02pubk",
        discrepancy=5.2,
        bound=0.001,
        fraud_proof={"mock": "proof"},
    )
    assert isinstance(fraud_err, VerificationFailedError)
    assert isinstance(fraud_err, MatrixComputeError)
    assert fraud_err.fraud_proof == {"mock": "proof"}
    assert fraud_err.proof_data["has_fraud_proof"] is True
