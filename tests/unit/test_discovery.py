"""
Comprehensive Unit Tests for Discovery Subsystem in matrix-compute-sdk.

Tests Nostr RelayClient, NIP-33 event parsing, X-only dual parity signature verification,
multi-relay pooling and failover, in-memory TTL caching, and Pareto Matchmaker
multi-objective ranking, dynamic reputation, and failover selection.
"""

import hashlib
import json
import math
import time
from typing import Any, Dict, List
import pytest

from matrix_compute.crypto.signer import sign_message, ClientSigner
from matrix_compute.discovery.matchmaker import MatchResult, Matchmaker
from matrix_compute.discovery.relay_client import DiscoveredWorker, RelayClient, RelayStatus
from matrix_compute.exceptions import (
    DiscoveryError,
    InvalidNostrEventError,
    NoWorkersFoundError,
    RelayConnectionError,
    RelayPoolExhaustedError,
    WorkerUnavailableError,
)
from matrix_compute.test_doubles.mock_relay import MockNostrRelay
try:
    from matrix_contracts.schemas.discovery import (
        HardwareSpecs,
        PricingSpecs,
        WorkerAnnouncePayload,
        WorkerQueryRequest,
        WorkerQueryResponse,
    )
except ImportError:
    from matrix_contracts.schemas import (  # type: ignore
        HardwareSpecs,
        PricingSpecs,
        WorkerAnnouncePayload,
        WorkerQueryRequest,
        WorkerQueryResponse,
    )


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def mock_relay() -> MockNostrRelay:
    return MockNostrRelay()


@pytest.fixture
def client_signer() -> ClientSigner:
    return ClientSigner()


def make_raw_event(
    pubkey: str,
    d_tag: str = "matrix-worker",
    vram_mb: int = 24576,
    sats_per_gflop: float = 0.0005,
    p2p: str = "/ip4/127.0.0.1/tcp/50051",
    lightning: str = "02" + "a" * 64,
    gpu_model: str = "NVIDIA RTX 4090",
    peak_tflops: float = 82.6,
    created_at: int = 1000,
    sig: str = "mock_schnorr_valid",
    extra_tags: List[List[str]] = None,
    content: str = "{}",
) -> Dict[str, Any]:
    tags = [
        ["d", d_tag],
        ["vram_mb", str(vram_mb)],
        ["sats_per_gflop", str(sats_per_gflop)],
        ["p2p", p2p],
        ["lightning", lightning],
        ["gpu_model", gpu_model],
        ["peak_fp32_tflops", str(peak_tflops)],
        ["max_dimension", "8192"],
        ["supported_precisions", "fp32,fp16,bf16"],
        ["slots_available", "2"],
        ["min_task_sats", "10"],
        ["staked_sats", "50000"],
        ["bond_txid", "mock_txid_001"],
    ]
    if extra_tags:
        tags.extend(extra_tags)

    serialized = json.dumps([0, pubkey, created_at, 30000, tags, content], separators=(",", ":")).encode("utf-8")
    ev_id = hashlib.sha256(serialized).hexdigest()
    return {
        "id": ev_id,
        "pubkey": pubkey,
        "created_at": created_at,
        "kind": 30000,
        "tags": tags,
        "content": content,
        "sig": sig,
    }


# =============================================================================
# 1. DiscoveredWorker Unit Tests
# =============================================================================

def test_discovered_worker_fields_and_aliases():
    worker = DiscoveredWorker(
        pubkey="02" + "1" * 64,
        d_tag="matrix-worker",
        grpc_endpoint="127.0.0.1:50051",
        lightning_address="03" + "2" * 64,
        vram_mb=24576,
        sats_per_gflop=0.0005,
        gpu_model="RTX 4090",
        peak_fp32_tflops=82.6,
        max_dimension=8192,
        supported_precisions=["fp32", "fp16"],
        slots_available=4,
        min_task_sats=10,
        staked_sats=100000,
        bond_txid="tx_123",
        created_at=1680000000,
        relay_url="ws://127.0.0.1:8008",
        reputation_score=0.95,
    )
    assert worker.worker_pubkey == "02" + "1" * 64
    assert worker.p2p_multiaddr == "127.0.0.1:50051"
    assert worker.lightning_routing_pubkey == "03" + "2" * 64
    assert worker.vram_mb == 24576
    assert worker.sats_per_gflop == 0.0005
    assert worker.reputation_score == 0.95

    matched = worker.to_matched_worker()
    assert matched.worker_pubkey == worker.pubkey
    assert matched.p2p_multiaddr == worker.grpc_endpoint
    assert matched.sats_per_gflop == 0.0005


# =============================================================================
# 2. RelayClient Initialization & Pooling Tests
# =============================================================================

def test_relay_client_init_default():
    client = RelayClient(offline_mock_mode=True)
    assert client.cache_ttl == 300.0
    assert client.timeout == 5.0
    assert client.verify_signatures is True


def test_relay_client_mock_relay_injection(mock_relay):
    client = RelayClient(mock_relay=mock_relay)
    assert client.connect() is True
    assert client.is_connected is True


def test_relay_client_invalid_url_scheme():
    with pytest.raises(RelayConnectionError) as exc_info:
        RelayClient(relays=["ftp://invalid.relay.org"])
    assert "Unsupported relay protocol scheme 'ftp'" in str(exc_info.value)


def test_relay_client_unreachable_relay_fails():
    # Genuine socket connection probe refusal on unreachable port
    client = RelayClient(relays=["ws://127.0.0.1:59999"], offline_mock_mode=False, timeout=0.5)
    with pytest.raises(RelayConnectionError):
        client.connect()
    assert client.is_connected is False


def test_relay_client_multi_relay_pool_failover(mock_relay):
    # Pool with one dead relay (genuine socket probe refusal) and one valid in-memory mock relay
    client = RelayClient(relays=["ws://127.0.0.1:59999", mock_relay], offline_mock_mode=False, timeout=0.5)
    assert client.connect() is True
    assert client.is_connected is True


def test_relay_client_disconnect_clears_pool(mock_relay):
    client = RelayClient(mock_relay=mock_relay)
    client.connect()
    assert client.is_connected is True
    client.disconnect()
    assert client.is_connected is False


# =============================================================================
# 3. Dual-Mode Event Parsing Tests
# =============================================================================

def test_relay_client_parse_complete_tags(mock_relay):
    client = RelayClient(mock_relay=mock_relay)
    ev = make_raw_event(
        pubkey="02" + "3" * 64,
        vram_mb=49152,
        sats_per_gflop=0.0008,
        gpu_model="NVIDIA A100",
        peak_tflops=312.0,
    )
    worker = client.parse_event(ev, relay_url="ws://mock")
    assert worker.pubkey == "02" + "3" * 64
    assert worker.vram_mb == 49152
    assert worker.sats_per_gflop == 0.0008
    assert worker.gpu_model == "NVIDIA A100"
    assert worker.peak_fp32_tflops == 312.0
    assert worker.slots_available == 2
    assert worker.staked_sats == 50000
    assert worker.bond_txid == "mock_txid_001"


def test_relay_client_parse_content_json_fallback(mock_relay):
    client = RelayClient(mock_relay=mock_relay)
    # Event with minimal tags; specifications encoded in JSON content
    content_payload = {
        "p2p_multiaddr": "127.0.0.1:9090",
        "lightning_routing_pubkey": "03" + "4" * 64,
        "hardware": {
            "vram_mb": 16384,
            "gpu_model": "NVIDIA RTX 3080",
            "peak_fp32_tflops": 29.8,
            "max_dimension": 4096,
            "supported_precisions": ["fp32", "fp16"],
        },
        "pricing": {
            "sats_per_gflop": 0.0004,
            "min_task_sats": 20,
        },
        "bond_status": {
            "staked_amount_sats": 75000,
            "bond_txid": "tx_bond_content",
        },
    }
    ev = {
        "id": "content_ev_123",
        "pubkey": "02" + "4" * 64,
        "created_at": 2000,
        "kind": 30000,
        "tags": [["d", "matrix-worker"]],
        "content": json.dumps(content_payload),
        "sig": "mock_sig",
    }
    worker = client.parse_event(ev)
    assert worker.vram_mb == 16384
    assert worker.sats_per_gflop == 0.0004
    assert worker.gpu_model == "NVIDIA RTX 3080"
    assert worker.grpc_endpoint == "127.0.0.1:9090"
    assert worker.lightning_address == "03" + "4" * 64
    assert worker.staked_sats == 75000
    assert worker.bond_txid == "tx_bond_content"


def test_relay_client_zero_price_and_vram_preservation(mock_relay):
    client = RelayClient(mock_relay=mock_relay)
    # Free tier worker: sats_per_gflop = 0.0, vram_mb = 0
    ev = {
        "id": "free_tier_ev",
        "pubkey": "02" + "5" * 64,
        "created_at": 3000,
        "kind": 30000,
        "tags": [
            ["d", "matrix-worker"],
            ["p2p", "127.0.0.1:50051"],
            ["lightning", "02" + "5" * 64],
            ["sats_per_gflop", "0.0"],
            ["vram_mb", "0"],
            ["gpu_model", "CPU Host"],
        ],
        # Content has higher price, but tag has 0.0: tag must prevail!
        "content": json.dumps({"pricing": {"sats_per_gflop": 0.05}, "hardware": {"vram_mb": 8192}}),
        "sig": "mock_sig",
    }
    worker = client.parse_event(ev)
    assert worker.sats_per_gflop == 0.0
    assert worker.vram_mb == 0
    assert worker.gpu_model == "CPU Host"


def test_relay_client_malformed_event_missing_pubkey(mock_relay):
    client = RelayClient(mock_relay=mock_relay)
    with pytest.raises(InvalidNostrEventError):
        client.parse_event({"kind": 30000, "tags": []})


# =============================================================================
# 4. Signature Verification Tests
# =============================================================================

def test_relay_client_verify_real_secp256k1_signature():
    signer = ClientSigner()
    client = RelayClient(offline_mock_mode=False, verify_signatures=True)

    pubkey_xonly = signer.get_nostr_pubkey()
    created_at = 12345
    tags = [["d", "matrix-worker"]]
    content = "{}"

    serialized = json.dumps([0, pubkey_xonly, created_at, 30000, tags, content], separators=(",", ":")).encode("utf-8")
    ev_id = hashlib.sha256(serialized).hexdigest()
    sig = signer.sign_message(bytes.fromhex(ev_id))

    event = {
        "id": ev_id,
        "pubkey": pubkey_xonly,
        "created_at": created_at,
        "kind": 30000,
        "tags": tags,
        "content": content,
        "sig": sig,
    }

    assert client.verify_event_signature(event) is True


def test_relay_client_reject_tampered_signature():
    client = RelayClient(offline_mock_mode=False, verify_signatures=True)
    signer = ClientSigner()
    pubkey_xonly = signer.get_nostr_pubkey()

    event = {
        "id": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "pubkey": pubkey_xonly,
        "sig": "00" * 64,  # Forged 64-byte signature
    }
    assert client.verify_event_signature(event) is False


def test_relay_client_mock_signatures_pass():
    client = RelayClient(offline_mock_mode=True, verify_signatures=True)
    event = {
        "id": "abc123",
        "pubkey": "02" + "a" * 64,
        "sig": "mock_schnorr_12345678",
    }
    assert client.verify_event_signature(event) is True


def test_relay_client_mock_signatures_rejected_in_live_mode():
    client = RelayClient(offline_mock_mode=False, verify_signatures=True)
    event = {
        "id": "abc123",
        "pubkey": "02" + "a" * 64,
        "sig": "mock_schnorr_12345678",
    }
    assert client.verify_event_signature(event) is False


def test_relay_client_reject_nip01_id_tampering():
    signer = ClientSigner()
    client = RelayClient(offline_mock_mode=False, verify_signatures=True)
    pubkey_xonly = signer.get_nostr_pubkey()
    created_at = 12345
    tags = [["d", "matrix-worker"], ["p2p", "127.0.0.1:50051"]]
    content = "{}"

    serialized = json.dumps([0, pubkey_xonly, created_at, 30000, tags, content], separators=(",", ":")).encode("utf-8")
    ev_id = hashlib.sha256(serialized).hexdigest()
    sig = signer.sign_message(bytes.fromhex(ev_id))

    # Genuine event passes
    valid_event = {
        "id": ev_id,
        "pubkey": pubkey_xonly,
        "created_at": created_at,
        "kind": 30000,
        "tags": tags,
        "content": content,
        "sig": sig,
    }
    assert client.verify_event_signature(valid_event) is True

    # Tampered event: tags modified to point to malicious host, ID and sig left untouched
    tampered_event = dict(valid_event)
    tampered_event["tags"] = [["d", "matrix-worker"], ["p2p", "192.168.1.100:9999"]]
    assert client.verify_event_signature(tampered_event) is False


# =============================================================================
# 5. NIP-33 Deduplication, Caching & Filters
# =============================================================================

def test_relay_client_nip33_deduplication(mock_relay):
    client = RelayClient(mock_relay=mock_relay)

    pk = "02" + "7" * 64
    # Older event
    ev_old = make_raw_event(pubkey=pk, created_at=1000, sats_per_gflop=0.0010)
    # Newer event with updated price
    ev_new = make_raw_event(pubkey=pk, created_at=2000, sats_per_gflop=0.0003)

    mock_relay.publish(ev_old)
    mock_relay.publish(ev_new)

    workers = client.query(force_refresh=True)
    assert len(workers) == 1
    assert workers[0].created_at == 2000
    assert workers[0].sats_per_gflop == 0.0003


def test_relay_client_caching_and_ttl(mock_relay):
    client = RelayClient(mock_relay=mock_relay, cache_ttl=10.0)

    pk = "02" + "8" * 64
    mock_relay.publish(make_raw_event(pubkey=pk, sats_per_gflop=0.0005))

    # Initial query populates cache
    w1 = client.query(force_refresh=False)
    assert len(w1) == 1

    # Clear mock relay backend; query should still return cached result
    mock_relay.clear()
    w2 = client.query(force_refresh=False)
    assert len(w2) == 1
    assert w2[0].pubkey == pk

    # Force refresh bypasses cache
    w3 = client.query(force_refresh=True)
    assert len(w3) == 0


def test_relay_client_query_filters(mock_relay):
    client = RelayClient(mock_relay=mock_relay)

    mock_relay.publish(make_raw_event(pubkey="02" + "a" * 64, vram_mb=8192, sats_per_gflop=0.0005, gpu_model="RTX 3070"))
    mock_relay.publish(make_raw_event(pubkey="02" + "b" * 64, vram_mb=24576, sats_per_gflop=0.0008, gpu_model="RTX 4090"))
    mock_relay.publish(make_raw_event(pubkey="02" + "c" * 64, vram_mb=49152, sats_per_gflop=0.0015, gpu_model="A100"))

    # VRAM filter
    assert len(client.query(min_vram_mb=16000, force_refresh=True)) == 2

    # Price filter
    assert len(client.query(max_price=0.0006, force_refresh=True)) == 1

    # Combined filter
    assert len(client.query(min_vram_mb=16000, max_price=0.0010, force_refresh=True)) == 1

    # GPU model filter
    assert len(client.query(gpu_model="4090", force_refresh=True)) == 1


def test_relay_client_worker_query_request_adapter(mock_relay):
    client = RelayClient(mock_relay=mock_relay)
    mock_relay.publish(make_raw_event(pubkey="02" + "d" * 64, vram_mb=24576, sats_per_gflop=0.0005))

    req = WorkerQueryRequest(min_vram_mb=8192, max_price_sats_per_gflop=0.0010, required_slots=1)
    resp = client.query_workers(req, force_refresh=True)
    assert isinstance(resp, WorkerQueryResponse)
    assert resp.status == "SUCCESS"
    assert len(resp.matched_workers) == 1
    assert resp.matched_workers[0].worker_pubkey == "02" + "d" * 64


# =============================================================================
# 6. Pareto Matchmaker & Dominance Ranking Tests
# =============================================================================

def make_worker_candidate(
    pubkey: str,
    sats_per_gflop: float,
    vram_mb: int,
    reputation: float = 1.0,
    tflops: float = 80.0,
) -> DiscoveredWorker:
    return DiscoveredWorker(
        pubkey=pubkey,
        grpc_endpoint=f"127.0.0.1:{int(pubkey[-4:], 16) % 1000 + 50000}",
        lightning_address="03" + pubkey[2:],
        sats_per_gflop=sats_per_gflop,
        vram_mb=vram_mb,
        reputation_score=reputation,
        peak_fp32_tflops=tflops,
        supported_precisions=["fp32", "fp16", "bf16"],
        slots_available=2,
    )


def test_matchmaker_pareto_dominance_single_objective():
    # Worker A is strictly cheaper than Worker B with identical capacity and reputation
    a = make_worker_candidate("02" + "a" * 64, sats_per_gflop=0.0005, vram_mb=24576)
    b = make_worker_candidate("02" + "b" * 64, sats_per_gflop=0.0008, vram_mb=24576)

    assert Matchmaker.dominates(a, b) is True
    assert Matchmaker.dominates(b, a) is False


def test_matchmaker_pareto_frontier_tradeoffs():
    # Candidate A is cheaper but has less VRAM; Candidate B has more VRAM but is pricier
    # Neither dominates the other; both belong on the Pareto frontier (Tier 1)
    a = make_worker_candidate("02" + "a" * 64, sats_per_gflop=0.0004, vram_mb=16384)
    b = make_worker_candidate("02" + "b" * 64, sats_per_gflop=0.0008, vram_mb=49152)

    assert Matchmaker.dominates(a, b) is False
    assert Matchmaker.dominates(b, a) is False

    mm = Matchmaker()
    tiers = mm.compute_pareto_tiers([a, b])
    assert len(tiers) == 1
    assert len(tiers[0]) == 2


def test_matchmaker_pareto_tier_partitioning():
    # A dominates C (A is cheaper and has more VRAM than C)
    # B dominates C (B is cheaper and has more VRAM than C)
    a = make_worker_candidate("02" + "a" * 64, sats_per_gflop=0.0004, vram_mb=24576)
    b = make_worker_candidate("02" + "b" * 64, sats_per_gflop=0.0005, vram_mb=49152)
    c = make_worker_candidate("02" + "c" * 64, sats_per_gflop=0.0010, vram_mb=8192)

    mm = Matchmaker()
    tiers = mm.compute_pareto_tiers([a, b, c])
    assert len(tiers) == 2
    assert set(w.pubkey for w in tiers[0]) == {a.pubkey, b.pubkey}  # Tier 1
    assert [w.pubkey for w in tiers[1]] == [c.pubkey]             # Tier 2


def test_matchmaker_lexicographic_sorting_tiebreaks():
    # Within Tier 1:
    # 1. Price ascending: W1 ($0.0004) beats W2 ($0.0005)
    # 2. If same price: reputation descending: W2 (1.0) beats W3 (0.8)
    # 3. If same price and reputation: VRAM descending: W3 (32GB) beats W4 (16GB)
    w1 = make_worker_candidate("02" + "1" * 64, sats_per_gflop=0.0004, vram_mb=16384, reputation=1.0)
    w2 = make_worker_candidate("02" + "2" * 64, sats_per_gflop=0.0005, vram_mb=24576, reputation=1.0)
    w3 = make_worker_candidate("02" + "3" * 64, sats_per_gflop=0.0005, vram_mb=32768, reputation=0.8)
    w4 = make_worker_candidate("02" + "4" * 64, sats_per_gflop=0.0005, vram_mb=16384, reputation=0.8)

    mm = Matchmaker()
    ranked = mm.rank_candidates([w4, w3, w2, w1])
    assert ranked[0].pubkey == w1.pubkey
    assert ranked[1].pubkey == w2.pubkey
    assert ranked[2].pubkey == w3.pubkey
    assert ranked[3].pubkey == w4.pubkey


def test_matchmaker_deterministic_pubkey_tiebreak():
    # Identical price, rep, vram, tflops: tie-broken deterministically by pubkey ascending
    w_a = make_worker_candidate("02" + "a" * 64, sats_per_gflop=0.0005, vram_mb=24576)
    w_b = make_worker_candidate("02" + "b" * 64, sats_per_gflop=0.0005, vram_mb=24576)

    mm = Matchmaker()
    ranked = mm.rank_candidates([w_b, w_a])
    assert ranked[0].pubkey == w_a.pubkey
    assert ranked[1].pubkey == w_b.pubkey


# =============================================================================
# 7. Slot Allocation, Backups & Cost Estimation
# =============================================================================

def test_matchmaker_match_worker_exact_slots_and_backups():
    w1 = make_worker_candidate("02" + "1" * 64, sats_per_gflop=0.0004, vram_mb=24576)
    w2 = make_worker_candidate("02" + "2" * 64, sats_per_gflop=0.0005, vram_mb=24576)
    w3 = make_worker_candidate("02" + "3" * 64, sats_per_gflop=0.0006, vram_mb=24576)
    w4 = make_worker_candidate("02" + "4" * 64, sats_per_gflop=0.0007, vram_mb=24576)

    mm = Matchmaker()
    res = mm.match_worker(
        workers=[w4, w3, w2, w1],
        m=2048,
        k=2048,
        n=2048,
        required_slots=2,
        num_backups=1,
    )
    assert len(res.primary_workers) == 2
    assert res.primary_workers[0].pubkey == w1.pubkey
    assert res.primary_workers[1].pubkey == w2.pubkey
    assert len(res.backup_workers) == 1
    assert res.backup_workers[0].pubkey == w3.pubkey

    # GFLOPs: 2 * 2048^3 / 1e9 = 17.179869184 GFLOPs
    assert res.estimated_gflops > 17.0
    # Cost sats: ceil(17.18 * 0.0004) = 1 (floor min_task_sats = 10)
    assert res.estimated_cost_sats == 10


def test_matchmaker_insufficient_candidates_truncation():
    # Only 1 candidate available when 3 slots requested: returns available without error
    w1 = make_worker_candidate("02" + "1" * 64, sats_per_gflop=0.0004, vram_mb=24576)
    mm = Matchmaker()
    res = mm.match_worker(workers=[w1], required_slots=3)
    assert len(res.primary_workers) == 1
    assert len(res.backup_workers) == 0


def test_matchmaker_zero_slots_returns_empty():
    w1 = make_worker_candidate("02" + "1" * 64, sats_per_gflop=0.0004, vram_mb=24576)
    mm = Matchmaker()
    res = mm.match_worker(workers=[w1], required_slots=0)
    assert res.primary_worker is None
    assert len(res.primary_workers) == 0


def test_matchmaker_no_workers_found_raises():
    mm = Matchmaker()
    with pytest.raises(NoWorkersFoundError) as exc_info:
        mm.match_worker(workers=[], m=1024, k=1024, n=1024)
    assert issubclass(NoWorkersFoundError, WorkerUnavailableError)
    assert issubclass(NoWorkersFoundError, DiscoveryError)


# =============================================================================
# 8. Dynamic Reputation, Blacklisting & Failover Tests
# =============================================================================

def test_matchmaker_reputation_penalty_clamped():
    mm = Matchmaker(penalty_per_failure=0.20)
    pk = "02" + "e" * 64

    # Default is 1.0
    assert mm.get_reputation(pk) == 1.0

    # First failure -> 0.80
    assert mm.record_failure(pk) == 0.80
    # Second failure -> 0.60
    assert mm.record_failure(pk) == 0.60

    # Many failures clamped at 0.0
    for _ in range(5):
        mm.record_failure(pk)
    assert mm.get_reputation(pk) == 0.0


def test_matchmaker_reputation_reward_clamped():
    mm = Matchmaker(penalty_per_failure=0.20, reward_per_success=0.05)
    pk = "02" + "f" * 64

    mm.record_failure(pk)  # 0.80
    assert mm.record_success(pk) == 0.85

    # Reward clamped at 1.0
    for _ in range(10):
        mm.record_success(pk)
    assert mm.get_reputation(pk) == 1.0


def test_matchmaker_consecutive_failures_blacklist():
    mm = Matchmaker(blacklist_threshold=3, blacklist_cooldown=600.0)
    pk = "02" + "9" * 64

    mm.record_failure(pk)
    assert mm.is_blacklisted(pk) is False
    mm.record_failure(pk)
    assert mm.is_blacklisted(pk) is False

    # 3rd failure triggers blacklist
    mm.record_failure(pk)
    assert mm.is_blacklisted(pk) is True

    # Blacklisted worker excluded from matchmaking
    w = make_worker_candidate(pk, sats_per_gflop=0.0004, vram_mb=24576)
    with pytest.raises(NoWorkersFoundError):
        mm.match_worker(workers=[w])

    # Success clears blacklist
    mm.record_success(pk)
    assert mm.is_blacklisted(pk) is False


def test_matchmaker_get_failover_worker():
    w1 = make_worker_candidate("02" + "1" * 64, sats_per_gflop=0.0004, vram_mb=24576)
    w2 = make_worker_candidate("02" + "2" * 64, sats_per_gflop=0.0005, vram_mb=24576)
    w3 = make_worker_candidate("02" + "3" * 64, sats_per_gflop=0.0006, vram_mb=24576)

    mm = Matchmaker()
    match_res = mm.match_worker(workers=[w1, w2, w3], required_slots=1, num_backups=2)

    # Primary worker w1 fails: failover selects w2
    failover = mm.get_failover_worker(match_result=match_res, failed_worker_pubkey=w1.pubkey)
    assert failover is not None
    assert failover.pubkey == w2.pubkey


def test_matchmaker_failover_all_exhausted_returns_none():
    w1 = make_worker_candidate("02" + "1" * 64, sats_per_gflop=0.0004, vram_mb=24576)
    mm = Matchmaker()
    match_res = mm.match_worker(workers=[w1], required_slots=1)

    failover = mm.get_failover_worker(match_result=match_res, failed_worker_pubkey=w1.pubkey)
    assert failover is None


def test_relay_client_multi_worker_slot_matching(mock_relay):
    """
    Verify that multiple workers offering 1 slot each are not filtered out
    when required_slots > 1 is requested, and can fulfill the multi-worker request.
    """
    client = RelayClient(mock_relay=mock_relay)
    # 5 workers each offering 1 slot
    for i in range(5):
        pk = f"02{'a' * 62}{i:02x}"
        ev = make_raw_event(pubkey=pk, sats_per_gflop=0.0001 * (i + 1), extra_tags=[["slots_available", "1"]])
        mock_relay.publish(ev)

    req = WorkerQueryRequest(min_vram_mb=8192, required_slots=3)
    resp = client.query_workers(req, force_refresh=True)
    assert resp.status == "SUCCESS"
    assert len(resp.matched_workers) == 3


def test_relay_client_and_matchmaker_nan_inf_sanitization():
    """
    Verify that NaN/Inf values in events are sanitized and do not crash
    cost estimation (math.ceil) or Pareto matchmaking.
    """
    client = RelayClient(offline_mock_mode=True)
    ev_nan = {
        "id": "ev_nan",
        "pubkey": "02" + "1" * 64,
        "created_at": 1000,
        "kind": 30000,
        "tags": [
            ["d", "matrix-worker"],
            ["sats_per_gflop", "NaN"],
            ["vram_mb", "Infinity"],
            ["peak_fp32_tflops", "-Inf"],
        ],
        "sig": "mock_sig",
    }
    worker = client.parse_event(ev_nan)
    assert worker.sats_per_gflop == 0.0
    assert worker.vram_mb == 0
    assert worker.peak_fp32_tflops == 0.0

    mm = Matchmaker()
    res = mm.match_worker(workers=[worker], m=1024, k=1024, n=1024)
    assert res.estimated_cost_sats >= 0
    assert math.isfinite(res.estimated_cost_sats)


def test_matchmaker_total_slots_available():
    """
    Verify that MatchResult.total_slots_available aggregates slots across matched primary workers.
    """
    w1 = make_worker_candidate("02" + "1" * 64, sats_per_gflop=0.0004, vram_mb=24576)
    w1.slots_available = 2
    w2 = make_worker_candidate("02" + "2" * 64, sats_per_gflop=0.0005, vram_mb=24576)
    w2.slots_available = 3

    mm = Matchmaker()
    res = mm.match_worker(workers=[w1, w2], required_slots=2)
    assert len(res.primary_workers) == 2
    assert res.total_slots_available == 5
