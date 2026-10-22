"""
Unit tests for libp2p circuit relay multiaddr resolution and metadata injection
in matrix-compute-sdk.
"""

from unittest.mock import MagicMock
import numpy as np
import pytest

from matrix_compute.p2p.channel_pool import (
    extract_circuit_target_worker,
    is_circuit_multiaddr,
    parse_multiaddr,
)
from matrix_compute.p2p.dispatcher import GrpcTaskDispatcher


def test_parse_circuit_multiaddr():
    circuit_dns = "/dns4/relay.matrix.org/tcp/9090/p2p-circuit/p2p/02abcd1234ef5678"
    assert parse_multiaddr(circuit_dns) == "relay.matrix.org:9090"

    circuit_ip4 = "/ip4/192.168.1.100/tcp/50051/p2p-circuit/p2p/03cdef9876543210"
    assert parse_multiaddr(circuit_ip4) == "192.168.1.100:50051"

    circuit_dns_simple = "/dns/relay-node/tcp/8080/p2p-circuit/p2p/0211223344"
    assert parse_multiaddr(circuit_dns_simple) == "relay-node:8080"


def test_is_circuit_multiaddr():
    circuit = "/dns4/relay.example.com/tcp/9090/p2p-circuit/p2p/02deadbeef"
    direct = "/ip4/127.0.0.1/tcp/50051"

    assert is_circuit_multiaddr(circuit) is True
    assert is_circuit_multiaddr(direct) is False
    assert is_circuit_multiaddr(None) is False
    assert is_circuit_multiaddr("") is False


def test_extract_circuit_target_worker():
    circuit = "/dns4/relay.example.com/tcp/9090/p2p-circuit/p2p/02DeadBeef1234567890abcdef"
    extracted = extract_circuit_target_worker(circuit)
    assert extracted == "02deadbeef1234567890abcdef"

    direct = "/ip4/127.0.0.1/tcp/50051"
    assert extract_circuit_target_worker(direct) is None
    assert extract_circuit_target_worker(None) is None

    # Object with p2p_multiaddr
    class WorkerObj:
        def __init__(self, multiaddr):
            self.p2p_multiaddr = multiaddr

    obj = WorkerObj(circuit)
    assert extract_circuit_target_worker(obj) == "02deadbeef1234567890abcdef"


def test_dispatcher_build_metadata_injection():
    channel_pool = MagicMock()
    dispatcher = GrpcTaskDispatcher(channel_pool=channel_pool)

    circuit_addr = "/dns4/relay.net/tcp/9090/p2p-circuit/p2p/02fa4321abcd"

    # Test task metadata with circuit multiaddr
    meta = dispatcher._build_metadata(
        task_or_lease_id="task-1234",
        auth_header="L402 macaroon:preimage",
        worker_or_target=circuit_addr,
        is_lease=False,
    )
    meta_dict = dict(meta)

    assert meta_dict["x-task-id"] == "task-1234"
    assert meta_dict["x-target-worker"] == "02fa4321abcd"
    assert meta_dict["authorization"] == "L402 macaroon:preimage"
    assert "x-lease-id" not in meta_dict

    # Test lease metadata with explicit worker_pubkey
    lease_meta = dispatcher._build_metadata(
        task_or_lease_id="lease-5678",
        worker_pubkey="03beef1234",
        worker_or_target="127.0.0.1:50051",
        is_lease=True,
        extra_metadata=[("x-step-index", "3")],
    )
    lease_dict = dict(lease_meta)

    assert lease_dict["x-lease-id"] == "lease-5678"
    assert lease_dict["x-task-id"] == "lease-5678"
    assert lease_dict["x-target-worker"] == "03beef1234"
    assert lease_dict["x-step-index"] == "3"


def test_request_tile_task_metadata_passed():
    channel_pool = MagicMock()
    mock_stub = MagicMock()
    channel_pool.get_stub.return_value = mock_stub

    mock_ack = MagicMock()
    mock_ack.is_ready = True
    mock_stub.RequestTileTask.return_value = mock_ack

    dispatcher = GrpcTaskDispatcher(
        channel_pool=channel_pool,
        require_confidential_compute=False,
    )

    circuit_addr = "/dns4/relay.net/tcp/9090/p2p-circuit/p2p/02fa4321abcd"
    ack = dispatcher.request_tile_task(
        worker_or_target=circuit_addr,
        task_id="task-999",
        m=64,
        k=64,
        n=64,
        dtype="fp32",
    )

    assert ack is mock_ack
    mock_stub.RequestTileTask.assert_called_once()
    _, kwargs = mock_stub.RequestTileTask.call_args
    passed_meta = dict(kwargs["metadata"])

    assert passed_meta["x-task-id"] == "task-999"
    assert passed_meta["x-target-worker"] == "02fa4321abcd"
