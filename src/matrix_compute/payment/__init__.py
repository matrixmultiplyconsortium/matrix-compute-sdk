"""L402 Trustless Escrow and Payment Subsystem."""

from .escrow import (
    EscrowManager,
    EscrowState,
    EscrowStatus,
    SettlementBackendProtocol,
    generate_escrow_secret,
    validate_preimage,
)
from .l402_client import (
    BudgetController,
    HttpSettlementBackend,
    L402Challenge,
    L402PaymentCoordinator,
    MacaroonCaveats,
    MacaroonToken,
    MockSettlementBackend,
    format_auth_header,
    format_grpc_metadata,
)

__all__ = [
    # Escrow State Machine
    "EscrowManager",
    "EscrowState",
    "EscrowStatus",
    "SettlementBackendProtocol",
    "generate_escrow_secret",
    "validate_preimage",
    # L402 & Macaroons
    "L402Challenge",
    "MacaroonToken",
    "MacaroonCaveats",
    "format_auth_header",
    "format_grpc_metadata",
    # Budget & Coordinator
    "BudgetController",
    "L402PaymentCoordinator",
    "MockSettlementBackend",
    "HttpSettlementBackend",
]
