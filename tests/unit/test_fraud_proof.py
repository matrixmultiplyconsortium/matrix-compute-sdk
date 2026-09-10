"""
Comprehensive Unit Test Suite for Fraud Proof Generation, Secp256k1 Attestation,
Mathematical Witness Verification, and Dispute Assembly.
"""

from __future__ import annotations

import hashlib
import math
import numpy as np
import pytest

from matrix_contracts.schemas.fraud import FreivaldsFraudProof
from matrix_compute.crypto.keys import generate_keypair
from matrix_compute.crypto.signer import (
    ClientSigner,
    format_tile_result_payload,
    hash_tensor,
    sign_message,
)
from matrix_compute.exceptions import MatrixDimensionError, WorkerSignatureError
from matrix_compute.verification.fraud_proof import (
    FraudProofBuilder,
    create_fraud_dispute,
    verify_fraud_proof,
)


# =============================================================================
# Test Fixtures & Helpers
# =============================================================================

@pytest.fixture
def keys():
    """Generates distinct worker and consumer Secp256k1 keypairs."""
    w_priv, w_pub = generate_keypair()
    c_priv, c_pub = generate_keypair()
    return {
        "w_priv": w_priv,
        "w_pub": w_pub,
        "c_priv": c_priv,
        "c_pub": c_pub,
    }


@pytest.fixture
def matrix_setup():
    """Provides nominal matrices A, B, and a fraudulently corrupted result C."""
    np.random.seed(401)
    a = np.random.randn(8, 8).astype(np.float32)
    b = np.random.randn(8, 8).astype(np.float32)
    c_corrupt = np.matmul(a, b)
    c_corrupt[0, 0] += 10.0  # Fraudulent perturbation
    return a, b, c_corrupt


# =============================================================================
# Group 1: TestFraudProofBuilderValid
# =============================================================================

class TestFraudProofBuilderValid:
    """Tests happy-path proof construction, key derivation, and witness extraction."""

    def test_builder_basic_construction_fp32(self, keys, matrix_setup):
        """Constructs proof from valid FP32 matrices and tampered C with valid worker signature."""
        a, b, c = matrix_setup
        task_id = "task_test_01"
        c_hash = hash_tensor(np.ascontiguousarray(c))
        worker_sig = sign_message(keys["w_priv"], format_tile_result_payload(task_id, c_hash))

        proof = FraudProofBuilder.build_fraud_proof(
            task_id=task_id,
            worker_pubkey=keys["w_pub"],
            A=a,
            B=b,
            C=c,
            worker_sig=worker_sig,
            consumer_privkey=keys["c_priv"],
            verify_worker_sig=True,
        )

        assert isinstance(proof, FreivaldsFraudProof)
        assert proof.task_id == task_id
        assert proof.worker_pubkey == keys["w_pub"]
        assert proof.consumer_pubkey == keys["c_pub"]
        assert proof.dimension_m == 8
        assert proof.dimension_k == 8
        assert proof.dimension_n == 8
        assert proof.dtype == "fp32"
        assert proof.max_observed_discrepancy > proof.epsilon_tolerance

    def test_builder_with_consumer_privkey(self, keys, matrix_setup):
        """Consumer public key is derived automatically from 32-byte private key."""
        a, b, c = matrix_setup
        task_id = "task_test_privkey"
        c_hash = hash_tensor(np.ascontiguousarray(c))
        sig = sign_message(keys["w_priv"], format_tile_result_payload(task_id, c_hash))

        proof = FraudProofBuilder.build_fraud_proof(
            task_id=task_id,
            worker_pubkey=keys["w_pub"],
            A=a,
            B=b,
            C=c,
            worker_sig=sig,
            consumer_privkey=keys["c_priv"],
        )
        assert proof.consumer_pubkey == keys["c_pub"]

    def test_builder_with_consumer_pubkey_directly(self, keys, matrix_setup):
        """Explicit consumer_pubkey is accepted without requiring private key."""
        a, b, c = matrix_setup
        task_id = "task_test_pubkey"
        c_hash = hash_tensor(np.ascontiguousarray(c))
        sig = sign_message(keys["w_priv"], format_tile_result_payload(task_id, c_hash))

        proof = FraudProofBuilder.build_fraud_proof(
            task_id=task_id,
            worker_pubkey=keys["w_pub"],
            A=a,
            B=b,
            C=c,
            worker_sig=sig,
            consumer_pubkey=keys["c_pub"],
        )
        assert proof.consumer_pubkey == keys["c_pub"]

    def test_builder_normalizes_proof_data_numpy_arrays(self, keys, matrix_setup):
        """NumPy arrays in proof_data are converted to standard Python lists of ints and floats."""
        a, b, c = matrix_setup
        task_id = "task_test_norm"
        r_np = np.ones((8, 1), dtype=np.int32)
        br_np = np.ones(8, dtype=np.float32)
        abr_np = np.ones(8, dtype=np.float32)
        cr_np = np.zeros(8, dtype=np.float32)

        proof_data = {
            "random_vector_r": r_np,
            "vector_br": br_np,
            "vector_abr": abr_np,
            "vector_cr": cr_np,
            "epsilon_tolerance": 0.01,
        }
        sig = sign_message(keys["w_priv"], format_tile_result_payload(task_id, hash_tensor(np.ascontiguousarray(c))))

        proof = FraudProofBuilder.build_fraud_proof(
            task_id=task_id,
            worker_pubkey=keys["w_pub"],
            A=a,
            B=b,
            C=c,
            proof_data=proof_data,
            worker_sig=sig,
            consumer_pubkey=keys["c_pub"],
        )
        assert all(isinstance(x, int) for x in proof.random_vector_r)
        assert all(isinstance(x, float) for x in proof.vector_br)
        assert all(isinstance(x, float) for x in proof.vector_abr)
        assert all(isinstance(x, float) for x in proof.vector_cr)

    def test_builder_auto_witness_search_when_proof_data_none(self, keys, matrix_setup):
        """When proof_data is None, builder automatically searches and finds a witness binary vector."""
        a, b, c = matrix_setup
        task_id = "task_test_auto_witness"
        sig = sign_message(keys["w_priv"], format_tile_result_payload(task_id, hash_tensor(np.ascontiguousarray(c))))

        proof = FraudProofBuilder.build_fraud_proof(
            task_id=task_id,
            worker_pubkey=keys["w_pub"],
            A=a,
            B=b,
            C=c,
            proof_data=None,
            worker_sig=sig,
            consumer_pubkey=keys["c_pub"],
        )
        assert proof.max_observed_discrepancy > proof.epsilon_tolerance
        assert len(proof.random_vector_r) == 8

    def test_builder_tensor_hash_accuracy(self, keys, matrix_setup):
        """SHA-256 tensor digests match direct hashlib computations."""
        a, b, c = matrix_setup
        task_id = "task_test_hashes"
        sig = sign_message(keys["w_priv"], format_tile_result_payload(task_id, hash_tensor(np.ascontiguousarray(c))))

        proof = FraudProofBuilder.build_fraud_proof(
            task_id=task_id,
            worker_pubkey=keys["w_pub"],
            A=a,
            B=b,
            C=c,
            worker_sig=sig,
            consumer_pubkey=keys["c_pub"],
        )
        expected_hash_a = hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
        expected_hash_b = hashlib.sha256(np.ascontiguousarray(b).tobytes()).hexdigest()
        expected_hash_c = hashlib.sha256(np.ascontiguousarray(c).tobytes()).hexdigest()

        assert proof.matrix_a_hash == expected_hash_a
        assert proof.matrix_b_hash == expected_hash_b
        assert proof.invalid_result_c_hash == expected_hash_c


# =============================================================================
# Group 2: TestFraudProofBuilderRejections
# =============================================================================

class TestFraudProofBuilderRejections:
    """Tests defensive input validation and mathematical integrity enforcement."""

    def test_builder_rejects_legitimate_result(self, keys):
        """Providing accurate C = A x B where discrepancy <= tolerance raises ValueError."""
        np.random.seed(402)
        a = np.random.randn(8, 8).astype(np.float32)
        b = np.random.randn(8, 8).astype(np.float32)
        c_accurate = np.matmul(a, b)
        task_id = "task_honest"
        sig = sign_message(keys["w_priv"], format_tile_result_payload(task_id, hash_tensor(np.ascontiguousarray(c_accurate))))

        with pytest.raises(ValueError, match="cannot be proven fraudulent"):
            FraudProofBuilder.build_fraud_proof(
                task_id=task_id,
                worker_pubkey=keys["w_pub"],
                A=a,
                B=b,
                C=c_accurate,
                worker_sig=sig,
                consumer_pubkey=keys["c_pub"],
            )

    def test_builder_rejects_dimension_mismatch_inner(self, keys):
        """A cols != B rows raises MatrixDimensionError."""
        a = np.ones((4, 5), dtype=np.float32)
        b = np.ones((6, 4), dtype=np.float32)
        c = np.ones((4, 4), dtype=np.float32)
        with pytest.raises(MatrixDimensionError, match="Dimension mismatch"):
            FraudProofBuilder.build_fraud_proof(
                task_id="task_mismatch",
                worker_pubkey=keys["w_pub"],
                A=a,
                B=b,
                C=c,
                consumer_pubkey=keys["c_pub"],
                verify_worker_sig=False,
            )

    def test_builder_rejects_dimension_mismatch_result(self, keys):
        """C shape != (M, N) raises MatrixDimensionError."""
        a = np.ones((4, 4), dtype=np.float32)
        b = np.ones((4, 4), dtype=np.float32)
        c = np.ones((4, 3), dtype=np.float32)
        with pytest.raises(MatrixDimensionError, match="Result matrix shape mismatch"):
            FraudProofBuilder.build_fraud_proof(
                task_id="task_res_mismatch",
                worker_pubkey=keys["w_pub"],
                A=a,
                B=b,
                C=c,
                consumer_pubkey=keys["c_pub"],
                verify_worker_sig=False,
            )

    def test_builder_rejects_missing_consumer_keys(self, keys, matrix_setup):
        """Providing neither consumer_privkey nor consumer_pubkey raises ValueError."""
        a, b, c = matrix_setup
        with pytest.raises(ValueError, match="Either consumer_pubkey or consumer_privkey must be provided"):
            FraudProofBuilder.build_fraud_proof(
                task_id="task_no_consumer",
                worker_pubkey=keys["w_pub"],
                A=a,
                B=b,
                C=c,
                verify_worker_sig=False,
            )

    def test_builder_rejects_invalid_worker_signature(self, keys, matrix_setup):
        """When verify_worker_sig=True, providing tampered or forged signature raises WorkerSignatureError."""
        a, b, c = matrix_setup
        task_id = "task_bad_sig"
        dummy_sig = "0" * 128

        with pytest.raises(WorkerSignatureError, match="signature"):
            FraudProofBuilder.build_fraud_proof(
                task_id=task_id,
                worker_pubkey=keys["w_pub"],
                A=a,
                B=b,
                C=c,
                worker_sig=dummy_sig,
                consumer_pubkey=keys["c_pub"],
                verify_worker_sig=True,
            )

    def test_builder_rejects_missing_worker_signature(self, keys, matrix_setup):
        """When verify_worker_sig=True, empty worker signature raises WorkerSignatureError."""
        a, b, c = matrix_setup
        task_id = "task_empty_sig"

        with pytest.raises(WorkerSignatureError, match="Missing worker signature"):
            FraudProofBuilder.build_fraud_proof(
                task_id=task_id,
                worker_pubkey=keys["w_pub"],
                A=a,
                B=b,
                C=c,
                worker_sig="",
                consumer_pubkey=keys["c_pub"],
                verify_worker_sig=True,
            )


# =============================================================================
# Group 3: TestVerifyFraudProofStandalone
# =============================================================================

class TestVerifyFraudProofStandalone:
    """Tests independent verification of FreivaldsFraudProof payloads without full matrices."""

    def test_verify_valid_proof_passes(self, keys, matrix_setup):
        """Valid FreivaldsFraudProof passes standalone verification."""
        a, b, c = matrix_setup
        task_id = "task_verify_pass"
        sig = sign_message(keys["w_priv"], format_tile_result_payload(task_id, hash_tensor(np.ascontiguousarray(c))))
        proof = FraudProofBuilder.build_fraud_proof(
            task_id=task_id,
            worker_pubkey=keys["w_pub"],
            A=a,
            B=b,
            C=c,
            worker_sig=sig,
            consumer_pubkey=keys["c_pub"],
        )
        assert verify_fraud_proof(proof) is True

    def test_verify_rejects_discrepancy_below_tolerance(self, keys, matrix_setup):
        """Proof with max_observed_discrepancy <= epsilon_tolerance returns False."""
        a, b, c = matrix_setup
        task_id = "task_tol"
        sig = sign_message(keys["w_priv"], format_tile_result_payload(task_id, hash_tensor(np.ascontiguousarray(c))))
        proof = FraudProofBuilder.build_fraud_proof(
            task_id=task_id,
            worker_pubkey=keys["w_pub"],
            A=a,
            B=b,
            C=c,
            worker_sig=sig,
            consumer_pubkey=keys["c_pub"],
        )
        # Artificially raise epsilon_tolerance above discrepancy
        proof.epsilon_tolerance = proof.max_observed_discrepancy + 1.0
        assert verify_fraud_proof(proof) is False

    def test_verify_rejects_tampered_abr_cr_vectors(self, keys, matrix_setup):
        """Altering vector_abr or vector_cr to eliminate discrepancy returns False."""
        a, b, c = matrix_setup
        task_id = "task_tampered_vec"
        sig = sign_message(keys["w_priv"], format_tile_result_payload(task_id, hash_tensor(np.ascontiguousarray(c))))
        proof = FraudProofBuilder.build_fraud_proof(
            task_id=task_id,
            worker_pubkey=keys["w_pub"],
            A=a,
            B=b,
            C=c,
            worker_sig=sig,
            consumer_pubkey=keys["c_pub"],
        )
        proof.vector_cr = list(proof.vector_abr)  # Force diff to 0.0
        assert verify_fraud_proof(proof) is False

    def test_verify_rejects_inconsistent_max_discrepancy(self, keys, matrix_setup):
        """Reported max_observed_discrepancy disagrees with calculated vector difference."""
        a, b, c = matrix_setup
        task_id = "task_inconsistent"
        sig = sign_message(keys["w_priv"], format_tile_result_payload(task_id, hash_tensor(np.ascontiguousarray(c))))
        proof = FraudProofBuilder.build_fraud_proof(
            task_id=task_id,
            worker_pubkey=keys["w_pub"],
            A=a,
            B=b,
            C=c,
            worker_sig=sig,
            consumer_pubkey=keys["c_pub"],
        )
        proof.max_observed_discrepancy += 50.0  # Falsify reported discrepancy
        assert verify_fraud_proof(proof) is False

    def test_verify_rejects_empty_vectors(self, keys, matrix_setup):
        """Proof with empty vector lists returns False."""
        a, b, c = matrix_setup
        task_id = "task_empty_vec"
        sig = sign_message(keys["w_priv"], format_tile_result_payload(task_id, hash_tensor(np.ascontiguousarray(c))))
        proof = FraudProofBuilder.build_fraud_proof(
            task_id=task_id,
            worker_pubkey=keys["w_pub"],
            A=a,
            B=b,
            C=c,
            worker_sig=sig,
            consumer_pubkey=keys["c_pub"],
        )
        proof.random_vector_r = []
        assert verify_fraud_proof(proof) is False

    def test_verify_rejects_all_zero_vector_r(self, keys, matrix_setup):
        """All-zero binary vector r returns False (cannot prove discrepancy)."""
        a, b, c = matrix_setup
        task_id = "task_zero_r"
        sig = sign_message(keys["w_priv"], format_tile_result_payload(task_id, hash_tensor(np.ascontiguousarray(c))))
        proof = FraudProofBuilder.build_fraud_proof(
            task_id=task_id,
            worker_pubkey=keys["w_pub"],
            A=a,
            B=b,
            C=c,
            worker_sig=sig,
            consumer_pubkey=keys["c_pub"],
        )
        proof.random_vector_r = [0] * len(proof.random_vector_r)
        assert verify_fraud_proof(proof) is False

    def test_verify_rejects_non_binary_vector_r(self, keys, matrix_setup):
        """Non-binary vector r containing values != 0, 1 returns False."""
        a, b, c = matrix_setup
        task_id = "task_non_binary_r"
        sig = sign_message(keys["w_priv"], format_tile_result_payload(task_id, hash_tensor(np.ascontiguousarray(c))))
        proof = FraudProofBuilder.build_fraud_proof(
            task_id=task_id,
            worker_pubkey=keys["w_pub"],
            A=a,
            B=b,
            C=c,
            worker_sig=sig,
            consumer_pubkey=keys["c_pub"],
        )
        proof.random_vector_r[0] = 5
        assert verify_fraud_proof(proof) is False

    def test_verify_rejects_mismatched_vector_lengths(self, keys, matrix_setup):
        """Vector length not matching dimension metadata returns False."""
        a, b, c = matrix_setup
        task_id = "task_len_mismatch"
        sig = sign_message(keys["w_priv"], format_tile_result_payload(task_id, hash_tensor(np.ascontiguousarray(c))))
        proof = FraudProofBuilder.build_fraud_proof(
            task_id=task_id,
            worker_pubkey=keys["w_pub"],
            A=a,
            B=b,
            C=c,
            worker_sig=sig,
            consumer_pubkey=keys["c_pub"],
        )
        proof.vector_br = proof.vector_br[:-1]  # Remove one element
        assert verify_fraud_proof(proof) is False


# =============================================================================
# Group 4: TestVerifyFraudProofCryptographic
# =============================================================================

class TestVerifyFraudProofCryptographic:
    """Tests Secp256k1 signature validation, replay attack prevention, and tamper detection."""

    def test_verify_rejects_forged_worker_signature(self, keys, matrix_setup):
        """Correct vector math but forged worker signature returns False."""
        a, b, c = matrix_setup
        task_id = "task_forged_sig"
        sig = sign_message(keys["w_priv"], format_tile_result_payload(task_id, hash_tensor(np.ascontiguousarray(c))))
        proof = FraudProofBuilder.build_fraud_proof(
            task_id=task_id,
            worker_pubkey=keys["w_pub"],
            A=a,
            B=b,
            C=c,
            worker_sig=sig,
            consumer_pubkey=keys["c_pub"],
        )
        # Replace signature with forged signature from consumer key
        proof.worker_signature_on_result = sign_message(keys["c_priv"], format_tile_result_payload(task_id, proof.invalid_result_c_hash))
        assert verify_fraud_proof(proof) is False

    def test_verify_rejects_replayed_task_id_signature(self, keys, matrix_setup):
        """Worker signed C for task_A, proof presents task_B -> signature fails."""
        a, b, c = matrix_setup
        sig_task_a = sign_message(keys["w_priv"], format_tile_result_payload("task_A", hash_tensor(np.ascontiguousarray(c))))
        proof = FraudProofBuilder.build_fraud_proof(
            task_id="task_A",
            worker_pubkey=keys["w_pub"],
            A=a,
            B=b,
            C=c,
            worker_sig=sig_task_a,
            consumer_pubkey=keys["c_pub"],
        )
        proof.task_id = "task_B"  # Replay attempt
        assert verify_fraud_proof(proof) is False

    def test_verify_rejects_tampered_result_hash(self, keys, matrix_setup):
        """Mutating invalid_result_c_hash invalidates signature verification."""
        a, b, c = matrix_setup
        task_id = "task_tampered_hash"
        sig = sign_message(keys["w_priv"], format_tile_result_payload(task_id, hash_tensor(np.ascontiguousarray(c))))
        proof = FraudProofBuilder.build_fraud_proof(
            task_id=task_id,
            worker_pubkey=keys["w_pub"],
            A=a,
            B=b,
            C=c,
            worker_sig=sig,
            consumer_pubkey=keys["c_pub"],
        )
        proof.invalid_result_c_hash = "0" * 64
        assert verify_fraud_proof(proof) is False


# =============================================================================
# Group 5: TestVerifyFraudProofFullWitness
# =============================================================================

class TestVerifyFraudProofFullWitness:
    """Tests cross-verification of proof vectors against full matrix witnesses."""

    def test_verify_with_full_matrix_witness_passes(self, keys, matrix_setup):
        """Passing matrices A, B, C alongside proof verifies tensor hashes and matrix-vector operations."""
        a, b, c = matrix_setup
        task_id = "task_witness_pass"
        sig = sign_message(keys["w_priv"], format_tile_result_payload(task_id, hash_tensor(np.ascontiguousarray(c))))
        proof = FraudProofBuilder.build_fraud_proof(
            task_id=task_id,
            worker_pubkey=keys["w_pub"],
            A=a,
            B=b,
            C=c,
            worker_sig=sig,
            consumer_pubkey=keys["c_pub"],
        )
        assert verify_fraud_proof(proof, matrix_a=a, matrix_b=b, matrix_c=c) is True

    def test_verify_witness_detects_swapped_matrix(self, keys, matrix_setup):
        """Passing honest C as witness when proof specifies corrupt C fails hash verification."""
        a, b, c_corrupt = matrix_setup
        task_id = "task_swapped_witness"
        c_honest = np.matmul(a, b)
        sig = sign_message(keys["w_priv"], format_tile_result_payload(task_id, hash_tensor(np.ascontiguousarray(c_corrupt))))
        proof = FraudProofBuilder.build_fraud_proof(
            task_id=task_id,
            worker_pubkey=keys["w_pub"],
            A=a,
            B=b,
            C=c_corrupt,
            worker_sig=sig,
            consumer_pubkey=keys["c_pub"],
        )
        # Pass honest matrix C as witness -> hash mismatch
        assert verify_fraud_proof(proof, matrix_a=a, matrix_b=b, matrix_c=c_honest) is False

    def test_create_fraud_dispute_convenience_helper(self, keys, matrix_setup):
        """Tests create_fraud_dispute convenience wrapper using ClientSigner."""
        a, b, c = matrix_setup
        task_id = "task_dispute_helper"
        c_hash = hash_tensor(np.ascontiguousarray(c))
        sig = sign_message(keys["w_priv"], format_tile_result_payload(task_id, c_hash))
        consumer_signer = ClientSigner(keys["c_priv"])

        proof = create_fraud_dispute(
            task_id=task_id,
            worker_pubkey=keys["w_pub"],
            A=a,
            B=b,
            C=c,
            proof_data=None,
            worker_sig=sig,
            consumer_signer=consumer_signer,
            dtype="fp32",
        )
        assert isinstance(proof, FreivaldsFraudProof)
        assert proof.consumer_pubkey == consumer_signer.get_pubkey_hex()
        assert verify_fraud_proof(proof) is True
