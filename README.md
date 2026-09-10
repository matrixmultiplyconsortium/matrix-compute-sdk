# matrix-compute-sdk

Production-grade consumer Python library and orchestrator for distributed matrix multiplication over decentralized compute nodes.

## Features

- **Matrix Tiling & Result Assembly**: Partition arbitrary 2D matrices (FP32, FP16, BF16) into sub-matrix blocks matching worker VRAM constraints and reassemble dot product results accurately.
- **Worker Discovery & Matchmaking**: Query Nostr discovery relays (Kind 30000 `WORKER_ANNOUNCE` events) with Pareto candidate ranking by price, reputation, and VRAM.
- **L402 Trustless Escrow**: Coordinate with settlement services to manage Lightning HODL invoice lifecycles (preimage generation, hold verification, conditional settlement on verification pass, refund on dispute).
- **P2P gRPC Task Dispatch**: Stream chunked tensor blocks to compute workers over gRPC (`MatrixComputeService`).
- **Mathematical Verification**: Randomized $O(n^2)$ Freivalds verification ($ABr = Cr$) using Higham's dynamic scale-aware error bound ($\gamma_K \|A\|_\infty \|B\|_\infty$).
- **Fraud Proof Construction**: Automated generation of cryptographically signed `FreivaldsFraudProof` payloads for worker bond slashing upon detecting malicious or perturbed outputs.
- **Hermetic Test Doubles**: High-fidelity in-memory doubles (`MockNostrRelay`, `MockSettlementService`, `MockWorkerNode`) for 100% offline, deterministic testing.

## Installation

```bash
pip install -e .
```

## Quickstart

### 1. High-Level Distributed Matrix Multiplication

```python
import numpy as np
from matrix_compute import MatrixComputeClient, distributed_matmul

# Generate random input matrices
A = np.random.randn(1024, 1024).astype(np.float32)
B = np.random.randn(1024, 1024).astype(np.float32)

# Using context manager with hermetic test doubles (offline mode)
with MatrixComputeClient.for_testing() as client:
    C = client.matmul(A, B, dtype="fp32")
    assert np.allclose(C, A @ B, atol=1e-4)

# Or using top-level convenience function
C = distributed_matmul(A, B)
```

### 2. Manual Configuration

```python
from matrix_compute.config import ClientConfig
from matrix_compute.client import MatrixComputeClient

config = ClientConfig(
    relay_urls=["wss://relay.damus.io"],
    settlement_url="https://settlement.matrix.network",
    min_vram_mb=8192,
    max_price_sats_per_gflop=0.001,
    freivalds_rounds=5,
)

client = MatrixComputeClient(config=config)
C = client.matmul(A, B)
```

## Architecture & Modules

```
src/matrix_compute/
├── __init__.py          # Public API exports & version
├── client.py            # High-level MatrixComputeClient and distributed_matmul
├── config.py            # ClientConfig & environment resolution
├── exceptions.py        # MatrixComputeError domain hierarchy
├── crypto/              # Secp256k1 key generation, BIP-62 low-S ECDSA, SHA-256 digests
├── test_doubles/        # MockNostrRelay, MockSettlementService, MockWorkerNode
├── tiling/              # 3D grid partitioning, BF16 emulation, and result assembly
├── verification/        # Dynamic Freivalds verification and FreivaldsFraudProof builder
├── discovery/           # Nostr Kind 30000 relay client and Pareto worker matchmaker
├── payment/             # L402 challenge parsing and HODL invoice escrow coordinator
├── p2p/                 # P2P gRPC streaming dispatcher and channel pool
└── orchestrator/        # Parallel task scheduler and compute coordinator
```

## Testing

The SDK contains a comprehensive test suite (590 tests) covering unit, component, opaque-box E2E, and Tier 5 adversarial stress tests:

```bash
# Run all unit and component tests (415 tests)
pytest tests/unit -v

# Run all E2E opaque-box acceptance tests (142 tests)
pytest tests/e2e/test_opaque_acceptance.py tests/e2e/test_distributed_matmul.py tests/e2e/test_fraud_slashing.py -v

# Run Tier 5 adversarial hardening suite (33 tests)
pytest tests/e2e/test_adversarial_hardening.py -v

# Run the complete test suite (590 tests)
pytest tests/ -v
```
