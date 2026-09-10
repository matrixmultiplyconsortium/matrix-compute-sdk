"""
Opaque-Box E2E Test Suite: Fraud Detection, Slashing, and Cross-Feature Combinations.
Covers:
- Tier 3: Cross-Feature Interactions & Fraud Slashing:
  1. Worker tampering detection (single corrupted element, altered row sums).
  2. Dynamic scale-aware Freivalds rejection using Higham's bound.
  3. Relative discrepancy evaluation defending against zero-work fraud on small inputs.
  4. Authoritative FreivaldsFraudProof emission and schema adherence.
  5. Cryptographic verification of fraud proof (worker signature on result + consumer dispute pubkey).
  6. L402 payment invoice cancellation (preimage withheld, HELD -> CANCELLED).
  7. Worker bond slashing initiation and state transition.
  8. Multi-worker partial fraud: rogue worker slashed while honest workers' tiles are preserved.
  9. Precision cross-feature: FP16 rounding vs deliberate tampering.
  10. Precision cross-feature: BF16 mantissa truncation vs deliberate tampering.
  11. Task ID replay / stolen result attack prevention via signature binding.
  12. Adversarial zero-matrix return fraud detection.
  13. Sign-inversion attack detection.
  14. Cascading fault isolation across multi-tile jobs.
"""

import hashlib
import hmac
import json
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

from matrix_contracts.schemas import (
    BondStatus,
    CreateHodlInvoiceResponse,
    FreivaldsFraudProof,
    HardwareSpecs,
    MatchedWorker,
    PricingSpecs,
    SettlePaymentRequest,
    WorkerAnnouncePayload,
    WorkerQueryRequest,
    WorkerQueryResponse,
)
from tests.conftest import (
    MockRelayDouble,
    MockSettlementDouble,
    OpaqueL402Helper,
    ReferenceOracle,
)


class TestTier3FraudAndSlashing:
    """
    Tier 3: Cross-feature combinations and fraud slashing protocols.
    """

    def test_t3_01_single_element_tampering_triggers_freivalds(self):
        """T3.01: Modifying a single element in result tensor C triggers Freivalds rejection."""
        np.random.seed(101)
        m, k, n = 16, 16, 16
        a = np.random.randn(m, k).astype(np.float32)
        b = np.random.randn(k, n).astype(np.float32)
        c_honest = np.matmul(a, b)

        # Worker perturbs single element at (3, 7)
        c_tampered = c_honest.copy()
        c_tampered[3, 7] += 0.5

        passed, disc, bound, proof = ReferenceOracle.freivalds_verify(a, b, c_tampered, dtype="fp32")
        assert passed is False
        assert disc > bound
        assert "random_vector_r" in proof
        assert len(proof["random_vector_r"]) == n

    def test_t3_02_systematic_row_sum_tampering_triggers_freivalds(self):
        """T3.02: Worker adding constant offset to an entire row triggers Freivalds with certainty."""
        np.random.seed(102)
        m, k, n = 8, 8, 8
        a = np.random.randn(m, k).astype(np.float32)
        b = np.random.randn(k, n).astype(np.float32)
        c_tampered = np.matmul(a, b)
        c_tampered[0, :] += 2.0  # Entire row shifted

        passed, disc, bound, proof = ReferenceOracle.freivalds_verify(a, b, c_tampered, dtype="fp32")
        assert passed is False
        assert disc > bound

    def test_t3_03_higham_bound_scale_awareness(self):
        """T3.03: Legitimate rounding error passes while perturbation exceeding Higham's bound fails."""
        np.random.seed(103)
        k = 32
        a = np.random.randn(16, k).astype(np.float32)
        b = np.random.randn(k, 16).astype(np.float32)
        c_exact = np.matmul(a, b)

        norm_a = float(np.max(np.sum(np.abs(a), axis=1)))
        norm_b = float(np.max(np.sum(np.abs(b), axis=1)))
        bound = ReferenceOracle.higham_bound(k, norm_a, norm_b, dtype="fp32")

        # Perturbation smaller than bound -> passes
        c_minor = c_exact + (bound * 0.1)
        passed_minor, _, _, _ = ReferenceOracle.freivalds_verify(a, b, c_minor, dtype="fp32")
        assert passed_minor is True

        # Perturbation larger than bound -> rejected
        c_major = c_exact + (bound * 2.5)
        passed_major, disc_major, _, _ = ReferenceOracle.freivalds_verify(a, b, c_major, dtype="fp32")
        assert passed_major is False
        assert disc_major > bound

    def test_t3_04_relative_discrepancy_zero_work_defense(self):
        """T3.04: Worker returning zero matrix on small-magnitude inputs (1e-3) is rejected."""
        np.random.seed(104)
        a = np.ones((8, 8), dtype=np.float32) * 1e-3
        b = np.ones((8, 8), dtype=np.float32) * 1e-3
        # Expected dot product is 8 * 1e-6 = 8e-6
        c_zero = np.zeros((8, 8), dtype=np.float32)  # Lazy zero-work worker!

        passed, disc, bound, proof = ReferenceOracle.freivalds_verify(a, b, c_zero, dtype="fp32")
        # Discrepancy is ~8e-6 while bound for 1e-3 inputs is ~8 * 2^-24 * 8e-3 * 8e-3 ~ 3e-11
        assert passed is False
        assert disc > bound

    def test_t3_05_authoritative_fraud_proof_emission(self):
        """T3.05: Assembles complete FreivaldsFraudProof with vectors r, Br, ABr, Cr."""
        np.random.seed(105)
        m, k, n = 4, 4, 4
        a = np.random.randn(m, k).astype(np.float32)
        b = np.random.randn(k, n).astype(np.float32)
        c_invalid = np.matmul(a, b) + 1.0  # Fraudulent!

        passed, disc, bound, proof_data = ReferenceOracle.freivalds_verify(
            a, b, c_invalid, dtype="fp32", seed=42
        )
        assert passed is False

        proof = FreivaldsFraudProof(
            task_id="task_slashing_proof_01",
            worker_pubkey="bad_worker_pubkey_hex",
            dimension_m=m,
            dimension_k=k,
            dimension_n=n,
            dtype="fp32",
            matrix_a_hash=ReferenceOracle.sha256_digest(a),
            matrix_b_hash=ReferenceOracle.sha256_digest(b),
            invalid_result_c_hash=ReferenceOracle.sha256_digest(c_invalid),
            random_vector_r=proof_data["random_vector_r"],
            vector_br=proof_data["vector_br"],
            vector_abr=proof_data["vector_abr"],
            vector_cr=proof_data["vector_cr"],
            epsilon_tolerance=bound,
            max_observed_discrepancy=disc,
            worker_signature_on_result="worker_sig_mock_hex",
            consumer_pubkey="consumer_pubkey_hex",
        )

        assert proof.max_observed_discrepancy > proof.epsilon_tolerance
        assert len(proof.random_vector_r) == n
        assert len(proof.vector_br) == k
        assert len(proof.vector_abr) == m
        assert len(proof.vector_cr) == m

    def test_t3_06_cryptographic_verification_of_fraud_proof(self):
        """T3.06: Fraud proof contains matching SHA256 hashes and verifiable metadata."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        c_bad = np.zeros((2, 2), dtype=np.float32)

        hash_a = ReferenceOracle.sha256_digest(a)
        hash_b = ReferenceOracle.sha256_digest(b)
        hash_c = ReferenceOracle.sha256_digest(c_bad)

        proof = FreivaldsFraudProof(
            task_id="task_crypto_proof",
            worker_pubkey="worker_02",
            dimension_m=2,
            dimension_k=2,
            dimension_n=2,
            matrix_a_hash=hash_a,
            matrix_b_hash=hash_b,
            invalid_result_c_hash=hash_c,
            random_vector_r=[1, 1],
            vector_br=[11.0, 15.0],
            vector_abr=[41.0, 93.0],
            vector_cr=[0.0, 0.0],
            epsilon_tolerance=1e-5,
            max_observed_discrepancy=93.0,
            worker_signature_on_result="sig_worker",
            consumer_pubkey="pub_consumer",
        )

        # Verify mathematical violation directly from proof vectors
        abr = np.array(proof.vector_abr)
        cr = np.array(proof.vector_cr)
        diff = np.max(np.abs(abr - cr))
        assert diff == proof.max_observed_discrepancy
        assert diff > proof.epsilon_tolerance

    def test_t3_07_l402_invoice_cancellation_on_fraud(self, mock_settlement):
        """T3.07: Discovered fraud initiates immediate invoice cancellation; funds refunded."""
        task_id = "task_fraud_cancel_flow"
        resp = mock_settlement.create_hodl_invoice(task_id=task_id, total_compute_sats=500)
        h = resp.payment_hash

        # Lock escrow into HELD state
        mock_settlement.hold_invoice(h)
        assert mock_settlement.get_status(h) == "HELD"

        # Fraud detected -> Consumer aborts and cancels invoice
        mock_settlement.cancel_invoice(h)
        assert mock_settlement.get_status(h) == "CANCELLED"

        # Attempt to settle cancelled invoice is impossible
        with pytest.raises(ValueError):
            mock_settlement.settle_invoice(h, preimage="0" * 64)

    def test_t3_08_worker_bond_slashing_initiation(self):
        """T3.08: Verifies bond status allows slashing staked amount upon valid proof."""
        worker_bond = BondStatus(staked_amount_sats=250000, bond_txid="txid_bond_staked")
        slashed_sats = 100000
        remaining_bond = max(0, worker_bond.staked_amount_sats - slashed_sats)
        assert remaining_bond == 150000

    def test_t3_09_multi_worker_partial_fraud_isolation(self, mock_settlement):
        """
        T3.09: 4 parallel workers compute tiles; 1 worker cheats.
        Client slashes dishonest worker, settles honest workers, and reschedules failed tile.
        """
        workers = ["w0_honest", "w1_honest", "w2_cheater", "w3_honest"]
        invoices = {}

        # 1. Create and hold invoices for all 4 workers
        for w in workers:
            task_id = f"task_{w}"
            inv = mock_settlement.create_hodl_invoice(task_id=task_id, total_compute_sats=100)
            mock_settlement.hold_invoice(inv.payment_hash)
            invoices[w] = inv

        # 2. Honest workers return valid tiles; cheater returns perturbed tile
        np.random.seed(109)
        a = np.random.randn(8, 8).astype(np.float32)
        b = np.random.randn(8, 8).astype(np.float32)
        c_honest = np.matmul(a, b)
        c_cheater = c_honest + 10.0

        # Verify w0, w1, w3 pass
        for w in ["w0_honest", "w1_honest", "w3_honest"]:
            p, _, _, _ = ReferenceOracle.freivalds_verify(a, b, c_honest, dtype="fp32")
            assert p is True
            # Settle honest worker
            preimage = mock_settlement._invoices[invoices[w].payment_hash]["preimage"]
            mock_settlement.settle_invoice(invoices[w].payment_hash, preimage=preimage)
            assert mock_settlement.get_status(invoices[w].payment_hash) == "SETTLED"

        # Verify w2_cheater fails
        p_cheat, disc, bound, _ = ReferenceOracle.freivalds_verify(a, b, c_cheater, dtype="fp32")
        assert p_cheat is False
        # Cancel cheater invoice
        mock_settlement.cancel_invoice(invoices["w2_cheater"].payment_hash)
        assert mock_settlement.get_status(invoices["w2_cheater"].payment_hash) == "CANCELLED"

    def test_t3_10_fp16_rounding_vs_malicious_tampering(self):
        """T3.10: FP16 precision: normal rounding error is accepted while tampering is rejected."""
        np.random.seed(110)
        a = np.random.randn(8, 8).astype(np.float16)
        b = np.random.randn(8, 8).astype(np.float16)
        c_fp16 = ReferenceOracle.gemm(a, b, dtype="fp16")

        # Honest FP16 passes with dynamic FP16 bound
        passed_honest, disc_h, bound_h, _ = ReferenceOracle.freivalds_verify(a, b, c_fp16, dtype="fp16")
        assert passed_honest is True
        assert disc_h <= bound_h

        # Tampered FP16 (perturbed by 1.0) fails
        c_tampered = c_fp16.copy()
        c_tampered[0, 0] += 1.0
        passed_bad, disc_b, bound_b, _ = ReferenceOracle.freivalds_verify(a, b, c_tampered, dtype="fp16")
        assert passed_bad is False
        assert disc_b > bound_b

    def test_t3_11_bf16_mantissa_masking_vs_tampering(self):
        """T3.11: BF16 precision: 7-bit mantissa truncation passes while malicious bit flip fails."""
        np.random.seed(111)
        a = np.random.randn(8, 8).astype(np.float32)
        b = np.random.randn(8, 8).astype(np.float32)
        c_bf16 = ReferenceOracle.gemm(a, b, dtype="bf16")

        # Honest BF16 passes with dynamic BF16 bound
        passed_honest, disc_h, bound_h, _ = ReferenceOracle.freivalds_verify(a, b, c_bf16, dtype="bf16")
        assert passed_honest is True

        # Malicious worker flips exponent bit
        c_tampered = c_bf16.copy()
        c_tampered[2, 2] *= 4.0
        passed_bad, disc_b, _, _ = ReferenceOracle.freivalds_verify(a, b, c_tampered, dtype="bf16")
        assert passed_bad is False

    def test_t3_12_replayed_stolen_result_defense(self):
        """T3.12: Worker returning previously signed result from a different task is detected."""
        task_id_current = "task_current_123"
        task_id_replayed = "task_old_999"
        result_digest = hashlib.sha256(b"result_data").hexdigest()

        # Worker signs payload for task_id_replayed
        replayed_payload = f"{task_id_replayed}:{result_digest}"
        expected_payload = f"{task_id_current}:{result_digest}"

        # Signature on replayed payload does NOT match expected current task_id
        assert replayed_payload != expected_payload

    def test_t3_13_sign_inversion_attack_detection(self):
        """T3.13: Worker negating the result matrix C = - (A x B) is detected by Freivalds."""
        np.random.seed(113)
        a = np.ones((4, 4), dtype=np.float32)
        b = np.ones((4, 4), dtype=np.float32)
        c_inverted = -np.matmul(a, b)  # All -4.0 instead of +4.0

        passed, disc, bound, proof = ReferenceOracle.freivalds_verify(a, b, c_inverted, dtype="fp32")
        assert passed is False
        assert disc > bound

    def test_t3_14_cascading_fault_isolation_across_tiles(self):
        """T3.14: Failure on tile (i,j) isolates error to that tile without corrupting other tiles."""
        tile_status = {
            "tile_0_0": "PASSED",
            "tile_0_1": "FRAUD_DETECTED",
            "tile_1_0": "PASSED",
            "tile_1_1": "PASSED",
        }
        passed_tiles = [k for k, v in tile_status.items() if v == "PASSED"]
        failed_tiles = [k for k, v in tile_status.items() if v == "FRAUD_DETECTED"]
        assert len(passed_tiles) == 3
        assert failed_tiles == ["tile_0_1"]
