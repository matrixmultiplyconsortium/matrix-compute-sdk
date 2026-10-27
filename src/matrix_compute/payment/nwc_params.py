"""
Nostr Wallet Connect (NWC / NIP-47) Connection Parameters and URI Parser.

Provides structured parsing, strict validation, and canonical serialization
for `nostr+walletconnect://` connection URIs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, quote, unquote, urlparse

SECP256K1_ORDER = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141


@dataclass
class NwcConnectionParams:
    """
    Parsed and cryptographically validated connection parameters for Nostr Wallet Connect.

    Attributes:
        wallet_pubkey: 32-byte hex public key of the remote wallet service (64 hex characters).
        relay_url: Primary WebSocket URL of the Nostr relay (starts with ws:// or wss://).
        secret: 32-byte client private key hex (64 hex characters) used for ECDH and signing.
        lud16: Optional Lightning address (e.g. 'user@getalby.com').
        relay_urls: List of all relay WebSocket URLs specified in the connection string.
    """

    wallet_pubkey: str
    relay_url: str
    secret: str
    lud16: Optional[str] = None
    relay_urls: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Enforces canonical formatting and strict validation upon instantiation."""
        # 1. Clean strings
        self.wallet_pubkey = str(self.wallet_pubkey).strip().lower()
        self.secret = str(self.secret).strip().lower()
        self.relay_url = str(self.relay_url).strip()
        if self.lud16 is not None:
            cleaned_lud16 = str(self.lud16).strip()
            self.lud16 = cleaned_lud16 if cleaned_lud16 else None

        # 2. Normalize relay_urls list
        if not self.relay_urls:
            self.relay_urls = [self.relay_url] if self.relay_url else []
        else:
            cleaned_relays = [str(r).strip() for r in self.relay_urls if str(r).strip()]
            if self.relay_url and self.relay_url not in cleaned_relays:
                cleaned_relays.insert(0, self.relay_url)
            self.relay_urls = cleaned_relays

        # 3. Validate wallet_pubkey
        if len(self.wallet_pubkey) != 64 or not re.fullmatch(r"^[0-9a-f]{64}$", self.wallet_pubkey):
            raise ValueError(
                f"Invalid wallet_pubkey: must be 64-char hex string, got '{self.wallet_pubkey}'"
            )

        # 4. Validate client secret
        if len(self.secret) != 64 or not re.fullmatch(r"^[0-9a-f]{64}$", self.secret):
            raise ValueError(
                f"Invalid secret: must be 64-char hex string, got '{self.secret}'"
            )
        try:
            scalar = int(self.secret, 16)
            if not (1 <= scalar < SECP256K1_ORDER):
                raise ValueError("scalar out of Secp256k1 range")
        except ValueError as err:
            raise ValueError(f"Invalid secret: {err}")

        # 5. Validate relays
        if not self.relay_urls or not self.relay_url:
            raise ValueError("Missing required 'relay' parameter in NWC URI")

        for r in self.relay_urls:
            if not (r.startswith("ws://") or r.startswith("wss://")):
                raise ValueError(f"Relay URL must begin with ws:// or wss://, got '{r}'")
            parsed_r = urlparse(r)
            if not parsed_r.netloc:
                raise ValueError(f"Invalid relay URL: missing host in '{r}'")

        # Synchronize primary relay_url with the first entry of relay_urls
        self.relay_url = self.relay_urls[0]

    @classmethod
    def from_uri(cls, uri: str) -> "NwcConnectionParams":
        """
        Parses and validates a `nostr+walletconnect://` URI into NwcConnectionParams.

        Format:
            nostr+walletconnect://<wallet_pubkey>?relay=<relay_url>&secret=<client_secret>[&lud16=<lud16>]

        Raises:
            ValueError: If URI is malformed, missing required parameters, or invalid.
        """
        if not isinstance(uri, str):
            raise ValueError(f"NWC URI must be string, got {type(uri).__name__}")
        clean = uri.strip()
        if not clean:
            raise ValueError("NWC URI cannot be empty")
        if not (clean.startswith("nostr+walletconnect://") or clean.startswith("nostr+walletconnect:")):
            raise ValueError("URI must start with 'nostr+walletconnect://'")

        parsed = urlparse(clean)
        raw_pubkey = parsed.netloc or parsed.path.lstrip("/")
        if "?" in raw_pubkey:
            raw_pubkey = raw_pubkey.split("?")[0]
        wallet_pubkey = raw_pubkey.rstrip("/").strip().lower()

        if len(wallet_pubkey) != 64 or not re.fullmatch(r"^[0-9a-f]{64}$", wallet_pubkey):
            raise ValueError(
                f"Invalid wallet_pubkey: must be 64-char hex string, got '{wallet_pubkey}'"
            )

        query_params = parse_qs(parsed.query, keep_blank_values=True)

        # Parse relays
        relays_param = query_params.get("relay")
        if not relays_param:
            raise ValueError("Missing required 'relay' parameter in NWC URI")

        valid_relays: List[str] = []
        for r_entry in relays_param:
            for sub_r in r_entry.split(","):
                unquoted = unquote(sub_r.strip())
                if not unquoted:
                    continue
                if not (unquoted.startswith("ws://") or unquoted.startswith("wss://")):
                    raise ValueError(f"Relay URL must begin with ws:// or wss://, got '{unquoted}'")
                parsed_relay = urlparse(unquoted)
                if not parsed_relay.netloc:
                    raise ValueError(f"Invalid relay URL: missing host in '{unquoted}'")
                valid_relays.append(unquoted)

        if not valid_relays:
            raise ValueError("Missing required 'relay' parameter in NWC URI")

        # Parse client secret
        secrets_list = query_params.get("secret")
        if not secrets_list or not any(s.strip() for s in secrets_list):
            raise ValueError("Missing required 'secret' parameter in NWC URI")
        client_secret = secrets_list[0].strip().lower()

        # Parse optional lud16
        lud16_list = query_params.get("lud16")
        lud16 = unquote(lud16_list[0].strip()) if lud16_list and lud16_list[0].strip() else None

        return cls(
            wallet_pubkey=wallet_pubkey,
            relay_url=valid_relays[0],
            secret=client_secret,
            lud16=lud16,
            relay_urls=valid_relays,
        )

    def to_uri(self) -> str:
        """Serializes connection parameters back into standard canonical NWC URI format."""
        base = f"nostr+walletconnect://{self.wallet_pubkey}"
        parts = []
        for r in self.relay_urls:
            parts.append(f"relay={quote(r, safe='')}")
        parts.append(f"secret={self.secret}")
        if self.lud16:
            parts.append(f"lud16={quote(self.lud16, safe='')}")
        return f"{base}?{'&'.join(parts)}"

    @property
    def client_pubkey(self) -> str:
        """Derives the 32-byte X-only Nostr public key hex associated with client secret."""
        from matrix_compute.crypto.keys import privkey_to_pubkey, pubkey_to_xonly
        return pubkey_to_xonly(privkey_to_pubkey(self.secret))

    @property
    def client_pubkey_hex(self) -> str:
        """Alias for client_pubkey."""
        return self.client_pubkey

    def to_dict(self) -> Dict[str, Any]:
        """Returns dictionary representation."""
        return {
            "wallet_pubkey": self.wallet_pubkey,
            "relay_url": self.relay_url,
            "secret": self.secret,
            "lud16": self.lud16,
            "relay_urls": list(self.relay_urls),
        }

    def __getitem__(self, item: str) -> Any:
        """Allows dictionary subscripting for compatibility with test assertions and dict consumers."""
        if hasattr(self, item):
            return getattr(self, item)
        raise KeyError(f"NwcConnectionParams has no field '{item}'")

    def __contains__(self, item: str) -> bool:
        """Membership test for dictionary-like subscripting."""
        return hasattr(self, item)

    def get(self, item: str, default: Any = None) -> Any:
        """Safe getter matching dict interface."""
        return getattr(self, item, default)


def parse_nwc_uri(uri: str) -> NwcConnectionParams:
    """Convenience helper to parse an NWC URI string into NwcConnectionParams."""
    return NwcConnectionParams.from_uri(uri)
