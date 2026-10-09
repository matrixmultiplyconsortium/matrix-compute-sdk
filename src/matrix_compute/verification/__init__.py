"""Mathematical verification (Freivalds) and fraud proof generation subsystem for matrix-compute-sdk."""

from .freivalds import (
    DEFAULT_FREIVALDS_ROUNDS,
    DEFAULT_RELATIVE_TOLERANCE_FLOOR,
    UNIT_ROUNDOFFS,
    FreivaldsVerifier,
    freivalds_verify,
    generate_random_binary_vector,
    higham_bound,
    matrix_infinity_norm,
    vector_infinity_norm,
    verify_iterative_step,
)
from .fraud_proof import (
    FraudProofBuilder,
    create_fraud_dispute,
    verify_fraud_proof,
)

__all__ = [
    # Freivalds
    "UNIT_ROUNDOFFS",
    "DEFAULT_FREIVALDS_ROUNDS",
    "DEFAULT_RELATIVE_TOLERANCE_FLOOR",
    "matrix_infinity_norm",
    "vector_infinity_norm",
    "higham_bound",
    "generate_random_binary_vector",
    "FreivaldsVerifier",
    "freivalds_verify",
    "verify_iterative_step",
    # Fraud Proof
    "FraudProofBuilder",
    "verify_fraud_proof",
    "create_fraud_dispute",
]
