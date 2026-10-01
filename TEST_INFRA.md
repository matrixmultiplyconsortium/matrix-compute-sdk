# TEST_INFRA: 1D Outer-Product (Upload-Optimal) Matrix Partitioning Test Infrastructure

## 1. Test Philosophy & Design Principles

### Opaque-Box & Requirement-Driven Testing
The 1D Outer-Product (Upload-Optimal) test infrastructure validates the distributed matrix multiplication pipeline strictly from an opaque-box, contract-driven, and mathematical invariant perspective. Tests are derived directly from the requirements in:
- `ORIGINAL_REQUEST.md` (specifically Phase 1 Requirements R1 through R4)
- `orchestrator_phase1/PROJECT.md` (Features 1 through 11, Milestones M1 through M4, and Interface Contracts)
- `planned_work/04_configurable_workloads_partitioning.md` (Configurable workload profiles and upload-optimal roofline analysis)

Key testing principles:
1. **Opaque-Box Contract Interfaces**: Tests exercise the system exclusively through public configurations (`ClientConfig`, `PartitioningStrategy`, `WorkloadType`), tiling manifests (`TileGridPlan`, `TileTaskPlan`), result accumulators (`ResultAssembler`), and verification contracts (`FreivaldsVerifier`, `FreivaldsFraudProof`, `MockSettlementDouble`).
2. **Explicit Authoritative Expected Outputs**: Every test assertion is derived from an authoritative mathematical oracle:
   - Double-precision reference calculations (`ReferenceOracle.gemm`) with IEEE 754 precision emulation.
   - Higham dynamic matrix multiplication rounding bound:
     $$\gamma_K = \frac{K \cdot u}{1 - K \cdot u}, \quad \|C - \widehat{C}\|_\infty \le \gamma_K \|A\|_\infty \|B\|_\infty$$
     where $u$ is the unit roundoff: FP32 ($2^{-24} \approx 5.96 \times 10^{-8}$), FP16 ($2^{-11} \approx 4.88 \times 10^{-4}$), and BF16 ($2^{-8} \approx 3.91 \times 10^{-3}$).
   - Exact upload bandwidth invariant:
     $$\text{Total Uploaded Elements} = \sum_{p=0}^{P-1} (M \cdot K_p + K_p \cdot N) = M \cdot K + K \cdot N$$
     $$\text{Redundancy Ratio} = \frac{\text{Total Uploaded Elements}}{M \cdot K + K \cdot N} - 1.0 \equiv 0.0$$
   - Dual-tier Freivalds randomized polynomial verification:
     - Per-strip check: $A_p \cdot (B_p \cdot r) = C_p \cdot r$ before buffer accumulation.
     - Assembled check: $A \cdot (B \cdot r) = C \cdot r$ on the final aggregated product matrix.
3. **Hermetic Offline Execution**: 100% offline execution using in-memory test doubles (`MockSettlementDouble`, `MockRelayDouble`), zero external network calls, zero physical GPU dependencies, and deterministic random seeds.
4. **Adversarial Fraud Isolation**: Injection of corrupted partial products by rogue workers, verifying that per-strip Freivalds flags the fraud, emits authoritative `FreivaldsFraudProof` payloads, cancels the batch L402 HODL invoice, and isolates honest workers' payments.
5. **Zero-Regression Verification**: Guarantees that the baseline 3D block tiling scheme remains fully functional and backward-compatible.

---

## 2. Feature Mapping & Requirements Traceability

| Requirement | Feature Description | Contract / Target Component | Test Coverage |
|-------------|---------------------|-----------------------------|---------------|
| **R1. Configuration & Tiling** | `PartitioningStrategy.OUTER_PRODUCT_1D` & balanced div-mod strip generation ($K_p = \lfloor K/P \rfloor + [p < r]$) | `ClientConfig`, `MatrixTiler.create_plan_outer_product_1d`, `Reference1DOuterProductEngine` | Tier 1 (T1.1-T1.2, T1.9), Tier 2 (T2.1-T2.8) |
| **R1. Dtype Preservation** | FP32, FP16, and BF16 (mantissa truncation `0xFFFF0000`) precision handling | `matrix_compute.tiling.precision`, `truncate_to_bf16` | Tier 1 (T1.4-T1.5), Tier 4 (T4.3-T4.5) |
| **R2. Partial Accumulation** | Streaming in-place accumulation $C = \sum_{p=0}^{P-1} C_p$ with coordinate tracking `(0, 0, p)` | `ResultAssembler.accumulate`, `ResultAssembler.get_result` | Tier 1 (T1.3, T1.6-T1.7), Tier 2 (T2.1-T2.7) |
| **R2. Dual-Tier Freivalds** | Per-strip Freivalds verification before accumulation + assembled verification | `FreivaldsVerifier`, `ReferenceOracle.freivalds_verify` | Tier 3 (T3.2, T3.3, T3.7) |
| **R2. Batch L402 Settlement** | Grouping strip tasks per worker into batch HODL invoices with atomic release or cancellation on fraud | `MockSettlementDouble`, `EscrowManager` | Tier 3 (T3.1, T3.4, T3.6) |
| **R3. Bandwidth Profiling** | Profiler asserting strictly $M \cdot K + K \cdot N$ uploaded elements with zero duplicate uploads | `profile_upload_bandwidth`, `Reference1DOuterProductEngine` | Tier 4 (T4.1, T4.2, T4.6, T4.7) |
| **R3. Higham Equivalence** | Mathematical equivalence against `np.matmul` within Higham bounds across FP32, FP16, and BF16 | Higham bound calculation across prime, odd, scalar, and real-world shapes | Tier 4 (T4.3, T4.4, T4.5) |
| **R4. Zero Regressions** | Verification across all repository unit and integration test suites | Repository test sweep | Platform test suites |

---

## 3. 4-Tier Test Architecture & Inventory

### Tier 1: Feature Coverage (9 Tests)
- `test_t1_01_outer_product_strip_partitioning_even_division`: Partitions $64 \times 64$ matrix into $P=4$ strips of width 16 along $K$.
- `test_t1_02_outer_product_strip_remainder_distribution`: Partitions $K=10$ with $P=3$ into balanced widths $[4, 3, 3]$ with zero gaps.
- `test_t1_03_partial_product_accumulation_fp32`: Accumulates partial products $C_p = A_p \times B_p$ into full $M \times N$ matrix matching `np.matmul`.
- `test_t1_04_precision_fp16_preservation`: Validates FP16 precision preservation and IEEE 754 half-precision rounding.
- `test_t1_05_precision_bf16_preservation`: Validates BF16 precision preservation with 16-bit mantissa masking (`0xFFFF0000`).
- `test_t1_06_result_assembler_in_place_outer_product_slices`: Directly verifies `ResultAssembler` accumulating outer-product slices `(0:M, 0:N, k=p)`.
- `test_t1_07_reentrancy_duplicate_accumulation_rejection`: Enforces duplicate rejection on coordinate `(0, 0, p)`.
- `test_t1_08_matrix_tiler_plan_compatibility_when_present`: Validates `MatrixTiler.create_plan_outer_product_1d` contract conformance when implemented.
- `test_t1_09_configuration_and_enum_contracts_when_present`: Validates `PartitioningStrategy` and `WorkloadType` configuration contracts.

### Tier 2: Boundary & Corner Cases (8 Tests)
- `test_t2_01_prime_dimensions_7x11x13`: Prime dimensions $M=7, K=11, N=13$ partitioned across $P=3$ strips ($[4, 4, 3]$).
- `test_t2_02_odd_dimensions_and_strips`: Odd dimensions $M=15, K=25, N=35$ across $P=4$ strips ($[7, 6, 6, 6]$).
- `test_t2_03_scalar_1x1_matrix`: Scalar $1 \times 1$ matrix: $[[3.0]] \times [[7.0]] = [[21.0]]$ with $P=1$.
- `test_t2_04_asymmetric_tall_skinny_and_wide_flat`: Tall-skinny ($128 \times 16 \times 32$) and wide-flat ($16 \times 128 \times 16$) matrix shapes.
- `test_t2_05_strip_oversubscription_p_greater_than_k`: Strip oversubscription $P=10 > K=3$ clamped to $P_{\text{eff}}=3$ without zero-width strips.
- `test_t2_06_single_strip_p_equals_1`: Single strip $P=1$ covering the entire $K$ dimension without tiling overhead.
- `test_t2_07_extreme_contraction_dimension`: Extreme inner contraction dimension $M=2, K=5000, N=2$ across $P=10$ strips.
- `test_t2_08_invalid_dimensions_rejected`: Rejection of 1D/3D tensors, inner dimension mismatches, and 0-dimension matrices.

### Tier 3: Cross-Feature Combinations (7 Tests)
- `test_t3_01_outer_product_with_mock_worker_pool`: 1D outer product dispatched across simulated workers with batch escrows.
- `test_t3_02_dual_tier_freivalds_per_strip_verification`: Dual-tier verification: per-strip Freivalds before buffer accumulation and assembled Freivalds on final product.
- `test_t3_03_freivalds_fraud_rejection_of_corrupt_strip`: Rogue worker corrupts strip; caught by per-strip Freivalds, emits `FreivaldsFraudProof`, and blocks buffer pollution.
- `test_t3_04_batch_l402_escrow_cancellation_on_fraudulent_strip`: Corrupt strip in a batch escrow causes atomic invoice cancellation and withholding of sats.
- `test_t3_05_sliding_window_concurrency_bounding`: 1D outer product strip dispatch respects sliding window concurrency limit.
- `test_t3_06_batch_l402_escrow_multi_worker_partial_fraud_isolation`: Multi-worker partial fraud: honest worker settles while rogue worker is cancelled.
- `test_t3_07_dual_tier_assembled_corruption_detection`: Assembled Freivalds catches cumulative drift or direct accumulator memory tampering.

### Tier 4: Real-World Scenarios & Bandwidth Profiling (7 Tests)
- `test_t4_01_bandwidth_profiling_zero_upload_redundancy`: Proves total uploaded matrix elements equal strictly $M \cdot K + K \cdot N$ (redundancy ratio $\equiv 0.0$) across multiple shapes.
- `test_t4_02_bandwidth_contrast_with_3d_block_tiling`: Contrasts 1D Outer-Product ($0.0\%$ redundancy) with 3D Block Tiling ($>0.0\%$ upload redundancy, e.g. $300\%$ redundant uploads).
- `test_t4_03_mathematical_equivalence_higham_fp32`: Mathematical equivalence against `np.matmul` within Higham bound $\|C - \widehat{C}\|_\infty \le \gamma_K \|A\|_\infty \|B\|_\infty$ for FP32.
- `test_t4_04_mathematical_equivalence_higham_fp16`: Mathematical equivalence for FP16 within Higham bound $\gamma_K$.
- `test_t4_05_mathematical_equivalence_higham_bf16`: Mathematical equivalence for BF16 with mantissa truncation.
- `test_t4_06_deep_learning_attention_outer_product_projection`: Real-world transformer attention projection ($64 \times 512 \times 64$) with upload-optimal strip slicing.
- `test_t4_07_large_scale_stress_1024x1024`: Large-scale $1024 \times 1024$ distributed outer product verifying exact $2,097,152$ uploaded elements with zero duplicate uploads.

---

## 4. Execution & Runner Commands

```powershell
# Set PYTHONPATH for SDK tests
$env:PYTHONPATH=".;../matrix-contracts;src"

# Run the complete Phase 1 1D Outer-Product E2E test suite (31 tests)
pytest tests/e2e/test_outer_product_e2e.py -v

# Run Tier 1 only (Feature Coverage - 9 tests)
pytest tests/e2e/test_outer_product_e2e.py -k "TestTier1" -v

# Run Tier 2 only (Boundary & Corner Cases - 8 tests)
pytest tests/e2e/test_outer_product_e2e.py -k "TestTier2" -v

# Run Tier 3 only (Cross-Feature Combinations - 7 tests)
pytest tests/e2e/test_outer_product_e2e.py -k "TestTier3" -v

# Run Tier 4 only (Real-World Scenarios & Bandwidth Profiling - 7 tests)
pytest tests/e2e/test_outer_product_e2e.py -k "TestTier4" -v

# Run upload bandwidth profiling tests specifically
pytest tests/e2e/test_outer_product_e2e.py -k "bandwidth" -v

# Run Higham error bound equivalence tests specifically
pytest tests/e2e/test_outer_product_e2e.py -k "higham" -v
```
