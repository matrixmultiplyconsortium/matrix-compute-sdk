"""P2P gRPC task dispatch and connection pool modules (Milestone 4)."""

from .channel_pool import (
    ChannelPool,
    GrpcChannelPool,
    multiaddr_to_target,
    parse_multiaddr,
)
from .dispatcher import (
    DispatchedTileResult,
    GrpcTaskDispatcher,
)

__all__ = [
    "ChannelPool",
    "GrpcChannelPool",
    "parse_multiaddr",
    "multiaddr_to_target",
    "GrpcTaskDispatcher",
    "DispatchedTileResult",
]
