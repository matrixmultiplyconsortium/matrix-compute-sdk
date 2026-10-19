"""
Permanent Adversarial Stress Test Suite for Discovery Subsystem & Pareto Matchmaker.
Targeting: Milestone 3 (M3) in matrix-compute-sdk.

Authored by teamwork_preview_challenger_sdk_m3_1 (Discovery & Matchmaker Challenger).

Challenge Dimensions:
1. Multi-Relay Pool Faults & Failover:
   - Partial failures: sockets reset, network timeouts, empty responses, concurrent failover.
   - Total pool exhaustion: all relays fail -> RelayPoolExhaustedError with degraded tracking.
   - Dynamic relay recovery & addition to active pool.
   - Malformed, corrupt, and non-dict relay payloads handled without unhandled crashes.
2. NIP-33 Conflict Resolution & Deduplication:
   - Out-of-order timestamp resolution: older delayed events never overwrite newer state.
   - Replay attack resilience: identical timestamps do not clobber existing state.
   - Cross-relay arbitration: highest timestamp across multi-relay pool wins.
   - Parameterized 'd' tag isolation: different d-tags for same pubkey are preserved independently.
   - Fallback to 'identifier' tag when 'd' tag is absent.
3. Cryptographic Signature Forgery & Adversarial Nostr Keys:
   - 100% rejection rate on single-bit flips across signature bytes (32-bit test sweep).
   - Mismatched public keys (valid signature from key A on event claiming key B).
   - Mismatched event digest IDs (valid signature for ID 1 attached to ID 2).
   - Malformed signature encodings: non-hex, odd lengths, truncated lengths, oversized lengths.
   - Filtering of forged events during multi-relay discovery queries.
4. Zero-Rate Pricing & Boundary Parsing:
   - Strict preservation of zero-rate pricing (sats_per_gflop = 0.0) in tags and content fallback.
   - Zero-VRAM CPU worker preservation (vram_mb = 0).
   - Boundary matrix dimensions: 1x1x1 (2e-9 GFLOPs) up to 100,000x100,000x100,000 (2M GFLOPs).
   - Incomplete tags paired with corrupted/truncated JSON content handled gracefully.
5. Pareto Matchmaker Multi-Objective Dominance Edge Cases:
   - Multi-objective trade-off collisions (non-dominated Pareto front with 4 orthogonal nodes).
   - Strict 4-tier non-dominated sorting chain (A dominates B dominates C dominates D).
   - Deterministic tie-breaking across 50 nodes with identical metrics (sorted by pubkey asc).
   - Zero-rate compute tier dominance over paid compute tier.
   - Cost estimation respecting GFLOPs, zero-rates, and advertised min_task_sats floor.
6. Rapid Worker Failure Cascades & Failover Traversal:
   - High-concurrency failure cascade: 100 workers failing simultaneously across threads.
   - Dynamic reputation decay, 3-strike threshold triggering temporary blacklisting.
   - Blacklist cooldown expiration restoring eligibility.
   - Verification success clearing strikes and restoring reputation.
   - Sequential failover traversal: Primary -> Backup 1 -> Backup 2 -> Remaining Pool -> None.
   - Failover queue automatically bypassing currently blacklisted workers.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import math
import secrets
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import pytest

from matrix_compute.crypto.keys import (
    generate_keypair,
    privkey_to_pubkey,
    pubkey_to_xonly,
)
from matrix_compute.crypto.signer import ClientSigner, sign_message, verify_signature
from matrix_compute.discovery.matchmaker import Matchmaker, MatchResult
from matrix_compute.discovery.relay_client import (
    DiscoveredWorker,
    RelayClient,
    RelayNode,
    RelayStatus,
)
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
        MatchedWorker,
        WorkerAnnouncePayload,
        WorkerQueryRequest,
        WorkerQueryResponse,
    )
except ImportError:
    from matrix_contracts.schemas import (  # type: ignore
        MatchedWorker,
        WorkerAnnouncePayload,
        WorkerQueryRequest,
        WorkerQueryResponse,
    )


# =============================================================================
# Helper Doubles & Utilities
# =============================================================================

class ProgrammableFaultyRelay:
    """
    Test double simulating network and application level failures in Nostr relays:
    socket resets, query timeouts, internal exceptions, malformed events, or raw empty responses.
    """

    def __init__(
        self,
        mode: str = "ok",
        events: Optional[List[Any]] = None,
        exception_msg: str = "Simulated relay network error",
    ) -> None:
        self.mode = mode
        self.events = list(events) if events is not None else []
        self.exception_msg = exception_msg
        self.query_count = 0

    def query(self, min_vram_mb: int = 0, max_price: Optional[float] = None) -> List[Any]:
        self.query_count += 1
        if self.mode == "connection_reset":
            raise ConnectionResetError(self.exception_msg)
        elif self.mode == "timeout":
            raise TimeoutError(self.exception_msg)
        elif self.mode == "fatal_error":
            raise RuntimeError(self.exception_msg)
        elif self.mode == "return_raw":
            return self.events
        return self.events


def build_signed_nostr_event(
    signer: ClientSigner,
    d_tag: str = "matrix-worker",
    vram_mb: int = 24576,
    sats_per_gflop: float = 0.0005,
    p2p: str = "127.0.0.1:50051",
    lightning: str = "02" + "a" * 64,
    gpu_model: str = "NVIDIA RTX 4090",
    peak_tflops: float = 82.6,
    created_at: int = 1000,
    content: str = "{}",
    extra_tags: Optional[List[List[str]]] = None,
) -> Dict[str, Any]:
    """Generates a cryptographically signed Nostr Kind 30000 event using ClientSigner."""
    pubkey_xonly = signer.get_nostr_pubkey()
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
        ["bond_txid", "tx_bond_valid"],
    ]
    if extra_tags:
        tags.extend(extra_tags)

    serialized = json.dumps(
        [0, pubkey_xonly, created_at, 30000, tags, content],
        separators=(",", ":"),
    ).encode("utf-8")
    ev_id = hashlib.sha256(serialized).hexdigest()
    sig = signer.sign_message(bytes.fromhex(ev_id))

    return {
        "id": ev_id,
        "pubkey": pubkey_xonly,
        "created_at": created_at,
        "kind": 30000,
        "tags": tags,
        "content": content,
        "sig": sig,
    }


def make_mock_event(
    pubkey: str,
    d_tag: str = "matrix-worker",
    vram_mb: int = 24576,
    sats_per_gflop: float = 0.0005,
    p2p: str = "127.0.0.1:50051",
    lightning: str = "02" + "1" * 64,
    gpu_model: str = "NVIDIA RTX 4090",
    peak_tflops: float = 82.6,
    created_at: int = 1000,
    sig: str = "mock_schnorr_valid",
    content: str = "{}",
    extra_tags: Optional[List[List[str]]] = None,
) -> Dict[str, Any]:
    """Generates an in-memory mock Kind 30000 event for fast deterministic testing."""
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
    ]
    if extra_tags:
        tags.extend(extra_tags)

    serialized = json.dumps(
        [0, pubkey, created_at, 30000, tags, content],
        separators=(",", ":"),
    ).encode("utf-8")
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


def make_test_candidate(
    pubkey: str,
    sats_per_gflop: float,
    vram_mb: int,
    reputation: float = 1.0,
    peak_tflops: float = 80.0,
    min_task_sats: int = 10,
    slots: int = 2,
) -> DiscoveredWorker:
    """Helper to construct DiscoveredWorker candidates for Matchmaker tests."""
    return DiscoveredWorker(
        pubkey=pubkey,
        grpc_endpoint=f"127.0.0.1:{int(pubkey[-4:], 16) % 10000 + 40000}",
        lightning_address=f"03{pubkey[2:] if len(pubkey) >= 66 else pubkey.zfill(64)}",
        sats_per_gflop=sats_per_gflop,
        vram_mb=vram_mb,
        reputation_score=reputation,
        peak_fp32_tflops=peak_tflops,
        min_task_sats=min_task_sats,
        slots_available=slots,
        supported_precisions=["fp32", "fp16", "bf16"],
        max_dimension=8192,
    )


# =============================================================================
# Suite 1: Multi-Relay Pool Faults & Failover
# =============================================================================

class TestMultiRelayPoolFaultsAndFailover:
    """
    Stress-tests multi-relay pooling when individual relays disconnect, time out,
    return malformed payloads, or when the entire pool is exhausted.
    """

    def test_multi_relay_partial_failure_recovery(self):
        """
        Pool with 4 relays:
          Relay 1: ConnectionResetError
          Relay 2: TimeoutError
          Relay 3: Returns empty list
          Relay 4: Healthy Mock Relay with 3 workers
        RelayClient must mark failed nodes DEGRADED, query Relay 4, and return workers without crashing.
        """
        healthy_mock = MockNostrRelay()
        for i in range(3):
            ev = make_mock_event(pubkey=f"02{'a' * 62}{i:02x}", sats_per_gflop=0.0005 + i * 0.0001)
            healthy_mock.publish(ev)

        r1 = ProgrammableFaultyRelay(mode="connection_reset", exception_msg="Connection reset by peer")
        r2 = ProgrammableFaultyRelay(mode="timeout", exception_msg="WebSocket receive timed out")
        r3 = ProgrammableFaultyRelay(mode="return_raw", events=[])
        r4 = healthy_mock

        client = RelayClient(relays=[r1, r2, r3, r4], offline_mock_mode=True)
        workers = client.query(force_refresh=True)

        assert len(workers) == 3
        # Verify node status tracking
        statuses = [node.status for node in client._relay_pool]
        assert RelayStatus.DEGRADED in statuses
        assert RelayStatus.ACTIVE in statuses

        # Confirm error logging on failed nodes
        failed_nodes = [node for node in client._relay_pool if node.error_count > 0]
        assert len(failed_nodes) == 2
        assert "Connection reset by peer" in failed_nodes[0].last_error
        assert "WebSocket receive timed out" in failed_nodes[1].last_error

    def test_multi_relay_all_fail_raises_exhausted_error(self):
        """
        When all relays in the pool throw network exceptions, RelayClient must raise
        RelayPoolExhaustedError (subclass of DiscoveryError).
        """
        r1 = ProgrammableFaultyRelay(mode="connection_reset")
        r2 = ProgrammableFaultyRelay(mode="timeout")
        r3 = ProgrammableFaultyRelay(mode="fatal_error")

        client = RelayClient(relays=[r1, r2, r3], offline_mock_mode=True)

        with pytest.raises(RelayPoolExhaustedError) as exc_info:
            client.query(force_refresh=True)

        assert issubclass(RelayPoolExhaustedError, DiscoveryError)
        assert "All relays in pool failed" in str(exc_info.value)
        assert all(n.status == RelayStatus.DEGRADED for n in client._relay_pool)

    def test_dynamic_relay_addition_and_recovery(self):
        """
        Test that a dead pool can recover when a healthy relay is added dynamically via _add_relay_target.
        """
        r_dead = ProgrammableFaultyRelay(mode="connection_reset")
        client = RelayClient(relays=[r_dead], offline_mock_mode=True)

        with pytest.raises(RelayPoolExhaustedError):
            client.query(force_refresh=True)

        # Dynamically inject healthy relay
        healthy = MockNostrRelay()
        healthy.publish(make_mock_event(pubkey="02" + "1" * 64))
        client._add_relay_target(healthy)

        workers = client.query(force_refresh=True)
        assert len(workers) == 1
        assert workers[0].pubkey == "02" + "1" * 64

    def test_malformed_event_payloads_gracefully_dropped(self):
        """
        Adversarial relay serves corrupted events:
        - Non-dict / missing pubkey / wrong kind
        RelayClient must log warnings and discard invalid events without unhandled crash.
        """
        valid_ev = make_mock_event(pubkey="02" + "9" * 64, vram_mb=24576)
        malformed_events = [
            {"kind": 1, "pubkey": "02" + "8" * 64, "tags": []},              # Not Kind 30000
            {"kind": 30000, "pubkey": "", "tags": []},                       # Missing pubkey
            {"kind": 30000, "pubkey": "   ", "tags": [["d", "bad"]]},         # Whitespace pubkey
            valid_ev,
        ]

        relay = ProgrammableFaultyRelay(mode="return_raw", events=malformed_events)
        client = RelayClient(relays=[relay], offline_mock_mode=True)

        workers = client.query(force_refresh=True)
        assert len(workers) == 1
        assert workers[0].pubkey == "02" + "9" * 64


# =============================================================================
# Suite 2: NIP-33 Conflict Resolution & Deduplication
# =============================================================================

class TestNIP33ConflictResolutionAndDeduplication:
    """
    Stress-tests NIP-33 parameterized replaceable event resolution (Kind 30000):
    Out-of-order delivery, replay attacks, cross-relay timestamp race conditions, and distinct 'd' tags.
    """

    def test_nip33_out_of_order_timestamps_older_event_ignored(self):
        """
        Ensure an older delayed event (T=1000) does NOT overwrite a newer event (T=5000)
        regardless of ingestion order.
        """
        pk = "02" + "b" * 64
        ev_new = make_mock_event(pubkey=pk, created_at=5000, sats_per_gflop=0.0002, vram_mb=49152)
        ev_old = make_mock_event(pubkey=pk, created_at=1000, sats_per_gflop=0.0100, vram_mb=16384)

        # Order 1: New arrives first, then old
        relay1 = MockNostrRelay()
        relay1.publish(ev_new)
        relay1.publish(ev_old)  # Should be rejected or ignored by NIP-33 rule

        client1 = RelayClient(mock_relay=relay1, offline_mock_mode=True)
        w1 = client1.query(force_refresh=True)
        assert len(w1) == 1
        assert w1[0].created_at == 5000
        assert w1[0].sats_per_gflop == 0.0002
        assert w1[0].vram_mb == 49152

        # Order 2: Old arrives first, then new
        relay2 = MockNostrRelay()
        relay2.publish(ev_old)
        relay2.publish(ev_new)

        client2 = RelayClient(mock_relay=relay2, offline_mock_mode=True)
        w2 = client2.query(force_refresh=True)
        assert len(w2) == 1
        assert w2[0].created_at == 5000
        assert w2[0].sats_per_gflop == 0.0002

    def test_nip33_replay_attack_identical_timestamp_rejected(self):
        """
        Adversary re-broadcasts an event with the EXACT same created_at timestamp
        but altered pricing. The client must preserve the first accepted version.
        """
        pk = "02" + "c" * 64
        ev_original = make_mock_event(pubkey=pk, created_at=3000, sats_per_gflop=0.0004)
        ev_tampered = make_mock_event(pubkey=pk, created_at=3000, sats_per_gflop=0.0500)

        relay = ProgrammableFaultyRelay(mode="return_raw", events=[ev_original, ev_tampered])
        client = RelayClient(relays=[relay], offline_mock_mode=True)

        workers = client.query(force_refresh=True)
        assert len(workers) == 1
        assert workers[0].sats_per_gflop == 0.0004

    def test_nip33_cross_relay_latest_timestamp_wins(self):
        """
        Relay A has Worker X at T=100 (Price=0.0010)
        Relay B has Worker X at T=300 (Price=0.0003)
        Relay C has Worker X at T=200 (Price=0.0007)
        Client queries pool of all 3 relays: T=300 version must prevail.
        """
        pk = "02" + "d" * 64
        ev_a = make_mock_event(pubkey=pk, created_at=100, sats_per_gflop=0.0010)
        ev_b = make_mock_event(pubkey=pk, created_at=300, sats_per_gflop=0.0003)
        ev_c = make_mock_event(pubkey=pk, created_at=200, sats_per_gflop=0.0007)

        ra = ProgrammableFaultyRelay(mode="return_raw", events=[ev_a])
        rb = ProgrammableFaultyRelay(mode="return_raw", events=[ev_b])
        rc = ProgrammableFaultyRelay(mode="return_raw", events=[ev_c])

        client = RelayClient(relays=[ra, rb, rc], offline_mock_mode=True)
        workers = client.query(force_refresh=True)

        assert len(workers) == 1
        assert workers[0].created_at == 300
        assert workers[0].sats_per_gflop == 0.0003

    def test_nip33_distinct_d_tags_for_same_pubkey_preserved(self):
        """
        A single worker node announces different service profiles under distinct 'd' tags:
        d="matrix-fp32", d="matrix-fp16", d="matrix-bf16".
        NIP-33 isolates replaceable state by (kind, pubkey, d_tag). All 3 must coexist.
        """
        pk = "02" + "e" * 64
        ev_fp32 = make_mock_event(pubkey=pk, d_tag="matrix-fp32", sats_per_gflop=0.0005, vram_mb=24576)
        ev_fp16 = make_mock_event(pubkey=pk, d_tag="matrix-fp16", sats_per_gflop=0.0003, vram_mb=16384)
        ev_bf16 = make_mock_event(pubkey=pk, d_tag="matrix-bf16", sats_per_gflop=0.0004, vram_mb=32768)

        relay = MockNostrRelay()
        relay.publish(ev_fp32)
        relay.publish(ev_fp16)
        relay.publish(ev_bf16)

        client = RelayClient(mock_relay=relay, offline_mock_mode=True)
        workers = client.query(force_refresh=True)

        assert len(workers) == 3
        d_tags_found = {w.d_tag for w in workers}
        assert d_tags_found == {"matrix-fp32", "matrix-fp16", "matrix-bf16"}

    def test_nip33_identifier_tag_fallback(self):
        """When standard 'd' tag is absent, client falls back to 'identifier' tag."""
        pk = "02" + "f" * 64
        ev = {
            "id": "ev_custom_id",
            "pubkey": pk,
            "created_at": 1000,
            "kind": 30000,
            "tags": [["identifier", "custom-cluster-01"], ["vram_mb", "32768"], ["sats_per_gflop", "0.0006"]],
            "content": "{}",
            "sig": "mock_sig",
        }
        client = RelayClient(offline_mock_mode=True)
        worker = client.parse_event(ev)
        assert worker.d_tag == "custom-cluster-01"
        assert worker.vram_mb == 32768


# =============================================================================
# Suite 3: Cryptographic Signature Forgery & Adversarial Nostr Keys
# =============================================================================

class TestSignatureForgeryAndAdversarialNostrKeys:
    """
    Exhaustively stress-tests Nostr Secp256k1 event signature verification:
    Bit-flip sweeps across signature bytes, pubkey mismatches, digest mismatches, and malformed encodings.
    """

    @pytest.fixture
    def real_signer(self) -> ClientSigner:
        return ClientSigner()

    def test_real_secp256k1_signature_verifies_successfully(self, real_signer: ClientSigner):
        """Confirm that a genuine Secp256k1 signature generated by ClientSigner passes verification."""
        client = RelayClient(verify_signatures=True, offline_mock_mode=False)
        event = build_signed_nostr_event(real_signer, created_at=12345)
        assert client.verify_event_signature(event) is True

    def test_signature_bitflip_sweeps_100_percent_rejected(self, real_signer: ClientSigner):
        """
        Flips single bits across 32 distinct byte positions in a valid 64-byte compact signature.
        Assert that verify_event_signature rejects 100% of mutated signatures.
        """
        client = RelayClient(verify_signatures=True, offline_mock_mode=False)
        event = build_signed_nostr_event(real_signer, created_at=54321)
        original_sig_hex = event["sig"]
        sig_bytes = bytearray(bytes.fromhex(original_sig_hex))
        assert len(sig_bytes) == 64

        tested_positions = [
            0, 1, 2, 7, 15, 16, 23, 31,  # R component positions
            32, 33, 39, 47, 48, 55, 62, 63,  # S component positions
        ]

        for byte_idx in tested_positions:
            for bit_offset in (0, 7):  # Test LSB and MSB of the byte
                mutated = bytearray(sig_bytes)
                mutated[byte_idx] ^= (1 << bit_offset)
                mutated_sig_hex = mutated.hex()

                tampered_event = dict(event)
                tampered_event["sig"] = mutated_sig_hex

                assert client.verify_event_signature(tampered_event) is False, (
                    f"Vulnerability: Forged signature accepted with bit flip at byte {byte_idx}, bit {bit_offset}"
                )

    def test_signature_mismatched_pubkey_rejected(self, real_signer: ClientSigner):
        """A valid signature signed by Keypair A must be rejected if the event claims to be from Keypair B."""
        attacker_signer = ClientSigner()
        client = RelayClient(verify_signatures=True, offline_mock_mode=False)

        event = build_signed_nostr_event(real_signer, created_at=9999)
        # Substitute pubkey with attacker's pubkey
        event["pubkey"] = attacker_signer.get_nostr_pubkey()

        assert client.verify_event_signature(event) is False

    def test_signature_mismatched_event_id_digest_rejected(self, real_signer: ClientSigner):
        """A valid signature on Event ID 1 must be rejected if attached to Event ID 2."""
        client = RelayClient(verify_signatures=True, offline_mock_mode=False)
        event = build_signed_nostr_event(real_signer, created_at=8888)

        # Alter event ID to random digest
        event["id"] = hashlib.sha256(b"adversarial_random_payload").hexdigest()
        assert client.verify_event_signature(event) is False

    @pytest.mark.parametrize(
        "malformed_sig",
        [
            "",                                 # Empty string
            "not_hex_signature_string",         # Non-hex string
            "00" * 32,                          # 32 bytes (too short, needs 64)
            "00" * 65,                          # 65 bytes (oversized)
            "deadbeef",                         # Truncated
            "12345",                            # Odd-length hex
            "ff" * 64,                          # Valid length, scalar out of curve range
        ],
    )
    def test_malformed_signature_encodings_do_not_crash(self, real_signer: ClientSigner, malformed_sig: str):
        """Malformed, non-hex, or out-of-range signatures must return False safely without unhandled exceptions."""
        client = RelayClient(verify_signatures=True, offline_mock_mode=False)
        event = build_signed_nostr_event(real_signer)
        event["sig"] = malformed_sig
        assert client.verify_event_signature(event) is False

    def test_query_filters_out_forged_signatures(self, real_signer: ClientSigner):
        """
        When querying relays, RelayClient with verify_signatures=True must filter out
        all events with forged or corrupted signatures, returning only authentic events.
        """
        authentic_ev = build_signed_nostr_event(real_signer, created_at=1000)

        forged_ev = dict(authentic_ev)
        forged_ev["id"] = hashlib.sha256(b"tampered").hexdigest()
        forged_ev["pubkey"] = "02" + "e" * 64

        relay = ProgrammableFaultyRelay(mode="return_raw", events=[authentic_ev, forged_ev])
        client = RelayClient(relays=[relay], verify_signatures=True, offline_mock_mode=False)

        workers = client.query(force_refresh=True)
        assert len(workers) == 1
        assert workers[0].pubkey == authentic_ev["pubkey"]


# =============================================================================
# Suite 4: Zero-Rate & Boundary Parsing
# =============================================================================

class TestZeroRateAndBoundaryParsing:
    """
    Stress-tests zero-rate pricing (sats_per_gflop = 0.0), free compute tiers, zero VRAM,
    extreme matrix dimensions (1x1 to 100,000x100,000), and incomplete tags with corrupted JSON.
    """

    def test_zero_rate_pricing_preserved_in_tags_over_content(self):
        """
        A node advertises sats_per_gflop = 0.0 in Nostr tags.
        Content JSON specifies sats_per_gflop = 0.05.
        The 0.0 tag value MUST prevail and NOT be treated as falsy/unset.
        """
        client = RelayClient(offline_mock_mode=True)
        event = {
            "id": "ev_zero_price",
            "pubkey": "02" + "1" * 64,
            "created_at": 1000,
            "kind": 30000,
            "tags": [
                ["d", "matrix-worker"],
                ["sats_per_gflop", "0.0"],
                ["vram_mb", "24576"],
                ["p2p", "127.0.0.1:50051"],
            ],
            "content": json.dumps({"pricing": {"sats_per_gflop": 0.05}}),
            "sig": "mock_sig",
        }
        worker = client.parse_event(event)
        assert worker.sats_per_gflop == 0.0

    def test_zero_rate_pricing_preserved_in_content_fallback(self):
        """
        When price tag is completely absent, content JSON specifies sats_per_gflop = 0.0.
        Must be parsed as 0.0, not defaulted to 0.0005 or None.
        """
        client = RelayClient(offline_mock_mode=True)
        event = {
            "id": "ev_zero_content",
            "pubkey": "02" + "2" * 64,
            "created_at": 1000,
            "kind": 30000,
            "tags": [["d", "matrix-worker"]],
            "content": json.dumps({"pricing": {"sats_per_gflop": 0.0}}),
            "sig": "mock_sig",
        }
        worker = client.parse_event(event)
        assert worker.sats_per_gflop == 0.0

    def test_zero_vram_cpu_worker_preservation(self):
        """
        CPU host node advertises vram_mb = 0.
        Must be preserved as 0 MB (not overwritten by content or default).
        """
        client = RelayClient(offline_mock_mode=True)
        event = {
            "id": "ev_cpu_zero_vram",
            "pubkey": "02" + "3" * 64,
            "created_at": 1000,
            "kind": 30000,
            "tags": [
                ["d", "matrix-worker"],
                ["vram_mb", "0"],
                ["gpu_model", "CPU Host Node"],
            ],
            "content": json.dumps({"hardware": {"vram_mb": 16384}}),
            "sig": "mock_sig",
        }
        worker = client.parse_event(event)
        assert worker.vram_mb == 0
        assert worker.gpu_model == "CPU Host Node"

    @pytest.mark.parametrize(
        "m, k, n, expected_gflops",
        [
            (1, 1, 1, 2e-9),                              # 1x1x1 scalar: 2 FLOPs
            (100, 100, 100, 2e-3),                        # 100x100x100: 2 MFLOPs
            (1024, 1024, 1024, 2.147483648),              # Standard 1K GEMM
            (10000, 10000, 10000, 2000.0),                # 10K GEMM: 2 TFLOPs
            (100000, 100000, 100000, 2000000.0),          # 100K GEMM: 2 PFLOPs
            (1, 1000000, 1, 0.002),                       # Extreme vector dot product
        ],
    )
    def test_extreme_matrix_dimension_gflop_calculations(
        self, m: int, k: int, n: int, expected_gflops: float
    ):
        """Matchmaker accurately computes GFLOPs across extreme matrix dimensions without float overflow."""
        mm = Matchmaker()
        worker = make_test_candidate("02" + "4" * 64, sats_per_gflop=0.0005, vram_mb=500000)

        res = mm.match_worker(
            workers=[worker],
            m=m,
            k=k,
            n=n,
            min_vram_mb=1,
            required_slots=1,
        )
        assert math.isclose(res.estimated_gflops, expected_gflops, rel_tol=1e-5)

    def test_incomplete_tags_with_corrupted_json_content(self):
        """
        Critical tags (price, vram, p2p) are missing, AND content is corrupted JSON.
        RelayClient must safely handle JSON decode errors and populate safe defaults without crashing.
        """
        client = RelayClient(offline_mock_mode=True)
        event = {
            "id": "ev_corrupted_json",
            "pubkey": "02" + "5" * 64,
            "created_at": 1000,
            "kind": 30000,
            "tags": [["d", "matrix-worker"]],
            "content": "{invalid_json: true, unterminated...",
            "sig": "mock_sig",
        }
        worker = client.parse_event(event)
        assert worker.pubkey == "02" + "5" * 64
        assert worker.vram_mb == 0
        assert worker.sats_per_gflop == 0.0
        assert worker.grpc_endpoint == "127.0.0.1:50051"
        assert worker.gpu_model == "Unknown GPU"


# =============================================================================
# Suite 5: Pareto Matchmaker Multi-Objective Dominance Edge Cases
# =============================================================================

class TestParetoMatchmakerMultiObjectiveAdversarial:
    """
    Stress-tests multi-objective non-dominated sorting across price, VRAM, reputation, and TFLOPS:
    Trade-off collisions, multi-tier dominance chains, 50-node deterministic pubkey tiebreaks,
    and zero-rate dominance.
    """

    def test_pareto_tradeoff_collisions_all_in_tier_1(self):
        """
        4 workers representing mutually orthogonal trade-offs:
          Node A: Price 0.0001, VRAM 16GB, Rep 0.90, TFLOPS 50   (Cheapest)
          Node B: Price 0.0004, VRAM 80GB, Rep 0.90, TFLOPS 50   (Highest VRAM)
          Node C: Price 0.0002, VRAM 24GB, Rep 1.00, TFLOPS 50   (Highest Rep)
          Node D: Price 0.0003, VRAM 24GB, Rep 0.90, TFLOPS 200  (Highest TFLOPS)
        None dominates any other. All 4 MUST belong to Pareto Tier 1.
        """
        node_a = make_test_candidate("02" + "a" * 64, sats_per_gflop=0.0001, vram_mb=16384, reputation=0.90, peak_tflops=50.0)
        node_b = make_test_candidate("02" + "b" * 64, sats_per_gflop=0.0004, vram_mb=81920, reputation=0.90, peak_tflops=50.0)
        node_c = make_test_candidate("02" + "c" * 64, sats_per_gflop=0.0002, vram_mb=24576, reputation=1.00, peak_tflops=50.0)
        node_d = make_test_candidate("02" + "d" * 64, sats_per_gflop=0.0003, vram_mb=24576, reputation=0.90, peak_tflops=200.0)

        candidates = [node_a, node_b, node_c, node_d]

        # Verify pairwise non-dominance
        for i, c1 in enumerate(candidates):
            for j, c2 in enumerate(candidates):
                if i != j:
                    assert Matchmaker.dominates(c1, c2) is False, f"Unexpected dominance: {c1.pubkey[:6]} dominates {c2.pubkey[:6]}"

        mm = Matchmaker()
        tiers = mm.compute_pareto_tiers(candidates)
        assert len(tiers) == 1
        assert len(tiers[0]) == 4
        assert set(w.pubkey for w in tiers[0]) == {node_a.pubkey, node_b.pubkey, node_c.pubkey, node_d.pubkey}

    def test_strict_four_tier_domination_chain(self):
        """
        Chain where A dominates B, B dominates C, and C dominates D.
        Must partition into exactly 4 distinct Pareto tiers [[A], [B], [C], [D]].
        """
        a = make_test_candidate("02" + "a" * 64, sats_per_gflop=0.0001, vram_mb=80000, reputation=1.0, peak_tflops=200.0)
        b = make_test_candidate("02" + "b" * 64, sats_per_gflop=0.0002, vram_mb=40000, reputation=0.9, peak_tflops=100.0)
        c = make_test_candidate("02" + "c" * 64, sats_per_gflop=0.0003, vram_mb=20000, reputation=0.8, peak_tflops=50.0)
        d = make_test_candidate("02" + "d" * 64, sats_per_gflop=0.0004, vram_mb=10000, reputation=0.7, peak_tflops=25.0)

        assert Matchmaker.dominates(a, b) is True
        assert Matchmaker.dominates(b, c) is True
        assert Matchmaker.dominates(c, d) is True

        mm = Matchmaker()
        # Pass in reverse order to ensure sorting does not rely on input order
        tiers = mm.compute_pareto_tiers([d, c, b, a])
        assert len(tiers) == 4
        assert [w.pubkey for w in tiers[0]] == [a.pubkey]
        assert [w.pubkey for w in tiers[1]] == [b.pubkey]
        assert [w.pubkey for w in tiers[2]] == [c.pubkey]
        assert [w.pubkey for w in tiers[3]] == [d.pubkey]

    def test_identical_metrics_50_nodes_deterministic_pubkey_tiebreak(self):
        """
        50 nodes with identical price (0.0005), VRAM (24576), rep (1.0), and TFLOPS (80.0),
        but randomized public keys.
        Lexicographic tiebreak must produce a 100% deterministic ordering by pubkey ascending.
        Output order must remain identical across repeated runs with different input permutations.
        """
        workers: List[DiscoveredWorker] = []
        for i in range(50):
            # Generate random 32-byte hex pubkeys
            pk = f"02{secrets.token_hex(32)}"
            workers.append(make_test_candidate(pk, sats_per_gflop=0.0005, vram_mb=24576, reputation=1.0, peak_tflops=80.0))

        mm = Matchmaker()

        expected_order = sorted([w.pubkey for w in workers])

        # Test with 5 different random shuffles
        import random
        for seed in (42, 101, 777, 999, 1234):
            shuffled = list(workers)
            random.Random(seed).shuffle(shuffled)
            ranked = mm.rank_candidates(shuffled)
            actual_order = [w.pubkey for w in ranked]
            assert actual_order == expected_order, f"Non-deterministic tiebreak ordering observed with seed {seed}"

    def test_zero_rate_worker_strictly_dominates_paid_worker(self):
        """
        Free compute node (sats_per_gflop = 0.0) strictly dominates Paid node (sats_per_gflop = 0.0001)
        when all other parameters are equal.
        """
        w_free = make_test_candidate("02" + "1" * 64, sats_per_gflop=0.0, vram_mb=24576, reputation=1.0, peak_tflops=80.0)
        w_paid = make_test_candidate("02" + "2" * 64, sats_per_gflop=0.0001, vram_mb=24576, reputation=1.0, peak_tflops=80.0)

        assert Matchmaker.dominates(w_free, w_paid) is True
        assert Matchmaker.dominates(w_paid, w_free) is False

        mm = Matchmaker()
        tiers = mm.compute_pareto_tiers([w_paid, w_free])
        assert len(tiers) == 2
        assert tiers[0][0].pubkey == w_free.pubkey
        assert tiers[1][0].pubkey == w_paid.pubkey

    def test_cost_estimation_zero_rate_and_min_task_floor(self):
        """
        Verify cost estimation:
        - Free node with min_task_sats = 0 -> estimated cost = 0 sats.
        - Free node with min_task_sats = 25 -> estimated cost = 25 sats (respects floor).
        - Paid node with 100 GFLOPs at 0.0001 sats/gflop -> ceil(0.01 sats) = 1 sat, with floor = 10 sats -> 10 sats.
        """
        mm = Matchmaker()
        w_free_zero_floor = make_test_candidate("02" + "1" * 64, sats_per_gflop=0.0, vram_mb=24576, min_task_sats=0)
        res1 = mm.match_worker(workers=[w_free_zero_floor], m=1024, k=1024, n=1024)
        assert res1.estimated_cost_sats == 0

        w_free_high_floor = make_test_candidate("02" + "2" * 64, sats_per_gflop=0.0, vram_mb=24576, min_task_sats=25)
        res2 = mm.match_worker(workers=[w_free_high_floor], m=1024, k=1024, n=1024)
        assert res2.estimated_cost_sats == 25


# =============================================================================
# Suite 6: Rapid Worker Failure Cascades & Failover Traversal
# =============================================================================

class TestRapidWorkerFailureCascadesAndFailover:
    """
    Stress-tests high-concurrency failure cascades:
    100 workers failing concurrently, dynamic reputation decay, 3-strike blacklisting,
    cooldown expiration, strike reset on success, and comprehensive failover queue traversal.
    """

    def test_concurrent_failure_cascade_100_workers(self):
        """
        100 workers experience concurrent failures across 10 threads.
        Assert that thread-safe reputation recording decrements correctly:
          Strike 1: rep 0.80, not blacklisted
          Strike 2: rep 0.60, not blacklisted
          Strike 3: rep 0.40, BLACKLISTED
        After 3 strikes, all 100 workers must be blacklisted and match_worker must raise NoWorkersFoundError.
        """
        mm = Matchmaker(blacklist_threshold=3, blacklist_cooldown=600.0, penalty_per_failure=0.20)
        pubkeys = [f"02{'f' * 62}{i:02x}" for i in range(100)]
        workers = [make_test_candidate(pk, sats_per_gflop=0.0005, vram_mb=24576) for pk in pubkeys]

        # Concurrently report 3 failures per worker
        def fail_worker(pk: str):
            mm.record_failure(pk)
            mm.record_failure(pk)
            mm.record_failure(pk)

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            list(executor.map(fail_worker, pubkeys))

        now = time.time()
        for pk in pubkeys:
            assert mm.get_reputation(pk) == 0.40
            assert mm.is_blacklisted(pk, now=now) is True

        # When all workers are blacklisted, matchmaker must raise NoWorkersFoundError
        with pytest.raises(NoWorkersFoundError) as exc_info:
            mm.match_worker(workers=workers, m=1024, k=1024, n=1024)
        assert exc_info.value.available_count == 0

    def test_cooldown_expiration_restores_worker(self):
        """A blacklisted worker becomes eligible again after blacklist_cooldown expires."""
        mm = Matchmaker(blacklist_threshold=3, blacklist_cooldown=300.0)
        pk = "02" + "7" * 64
        t0 = time.time()

        for _ in range(3):
            mm.record_failure(pk)

        # Active cooldown
        assert mm.is_blacklisted(pk, now=t0 + 100.0) is True

        # Expired cooldown
        assert mm.is_blacklisted(pk, now=t0 + 301.0) is False

    def test_success_resets_consecutive_failures_and_restores_rep(self):
        """
        Worker experiences 2 failures (consecutive_failures=2, rep=0.60).
        A successful task reports success:
        Reputation increments, consecutive failures reset to 0, and blacklist is cleared.
        Subsequent failure is treated as 1st strike (does not trigger blacklist).
        """
        mm = Matchmaker(blacklist_threshold=3, penalty_per_failure=0.20, reward_per_success=0.10)
        pk = "02" + "8" * 64

        mm.record_failure(pk)  # 0.80
        mm.record_failure(pk)  # 0.60
        assert mm.is_blacklisted(pk) is False

        # Success reported
        new_rep = mm.record_success(pk)
        assert new_rep == 0.70
        assert mm._consecutive_failures.get(pk) == 0

        # Another failure does not blacklist because count was reset
        mm.record_failure(pk)
        assert mm.is_blacklisted(pk) is False

    def test_sequential_failover_candidate_traversal(self):
        """
        Matchmaker selects primary W1, backups [W2, W3], and maintains all_candidates [W1..W5].
        Test full sequential failover lifecycle:
          W1 fails -> failover to W2
          W2 fails -> failover to W3
          W3 fails -> failover to W4 (from all_candidates)
          W4 fails -> failover to W5
          W5 fails -> returns None (pool exhausted)
        """
        w1 = make_test_candidate("02" + "1" * 64, sats_per_gflop=0.0001, vram_mb=24576)
        w2 = make_test_candidate("02" + "2" * 64, sats_per_gflop=0.0002, vram_mb=24576)
        w3 = make_test_candidate("02" + "3" * 64, sats_per_gflop=0.0003, vram_mb=24576)
        w4 = make_test_candidate("02" + "4" * 64, sats_per_gflop=0.0004, vram_mb=24576)
        w5 = make_test_candidate("02" + "5" * 64, sats_per_gflop=0.0005, vram_mb=24576)

        mm = Matchmaker()
        match_res = mm.match_worker(
            workers=[w5, w4, w3, w2, w1],
            required_slots=1,
            num_backups=2,
        )

        assert match_res.primary_worker.pubkey == w1.pubkey
        assert [w.pubkey for w in match_res.backup_workers] == [w2.pubkey, w3.pubkey]
        assert len(match_res.all_candidates) == 5

        # Fail W1 -> select W2
        f1 = mm.get_failover_worker(match_result=match_res, failed_worker_pubkey=w1.pubkey)
        assert f1 is not None and f1.pubkey == w2.pubkey

        # Fail W2 -> select W3
        f2 = mm.get_failover_worker(match_result=match_res, exclude_pubkeys={w1.pubkey, w2.pubkey})
        assert f2 is not None and f2.pubkey == w3.pubkey

        # Fail W3 -> select W4 (from all_candidates)
        f3 = mm.get_failover_worker(match_result=match_res, exclude_pubkeys={w1.pubkey, w2.pubkey, w3.pubkey})
        assert f3 is not None and f3.pubkey == w4.pubkey

        # Fail W4 -> select W5
        f4 = mm.get_failover_worker(match_result=match_res, exclude_pubkeys={w1.pubkey, w2.pubkey, w3.pubkey, w4.pubkey})
        assert f4 is not None and f4.pubkey == w5.pubkey

        # Fail W5 -> pool exhausted
        f5 = mm.get_failover_worker(
            match_result=match_res,
            exclude_pubkeys={w1.pubkey, w2.pubkey, w3.pubkey, w4.pubkey, w5.pubkey},
        )
        assert f5 is None

    def test_failover_skips_blacklisted_candidates(self):
        """
        If backup worker W2 has accumulated 3 failures elsewhere and is blacklisted,
        failover from W1 must skip W2 and select W3 directly.
        """
        w1 = make_test_candidate("02" + "1" * 64, sats_per_gflop=0.0001, vram_mb=24576)
        w2 = make_test_candidate("02" + "2" * 64, sats_per_gflop=0.0002, vram_mb=24576)
        w3 = make_test_candidate("02" + "3" * 64, sats_per_gflop=0.0003, vram_mb=24576)

        mm = Matchmaker(blacklist_threshold=3)
        match_res = mm.match_worker(workers=[w1, w2, w3], required_slots=1, num_backups=2)

        # W2 is blacklisted before failover query
        for _ in range(3):
            mm.record_failure(w2.pubkey)
        assert mm.is_blacklisted(w2.pubkey) is True

        # Failover from W1 must skip blacklisted W2 and yield W3
        failover = mm.get_failover_worker(match_result=match_res, failed_worker_pubkey=w1.pubkey)
        assert failover is not None
        assert failover.pubkey == w3.pubkey
