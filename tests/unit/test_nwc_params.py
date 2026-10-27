"""
Unit tests for NwcConnectionParams and NWC URI parser in matrix-compute-sdk.
"""

import pytest
from matrix_compute.payment.nwc_params import (
    NwcConnectionParams,
    parse_nwc_uri,
    SECP256K1_ORDER,
)

SAMPLE_WALLET_PUBKEY = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
SAMPLE_CLIENT_SECRET = "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210"
SAMPLE_RELAY_URL = "wss://relay.damus.io"


def test_standard_nwc_uri_parsing():
    uri = f"nostr+walletconnect://{SAMPLE_WALLET_PUBKEY}?relay={SAMPLE_RELAY_URL}&secret={SAMPLE_CLIENT_SECRET}"
    params = NwcConnectionParams.from_uri(uri)
    assert params.wallet_pubkey == SAMPLE_WALLET_PUBKEY
    assert params.relay_url == SAMPLE_RELAY_URL
    assert params.secret == SAMPLE_CLIENT_SECRET
    assert params.lud16 is None
    assert params.relay_urls == [SAMPLE_RELAY_URL]
    # Verify dict access
    assert params["wallet_pubkey"] == SAMPLE_WALLET_PUBKEY
    assert params["relay_url"] == SAMPLE_RELAY_URL
    assert "wallet_pubkey" in params
    assert params.get("wallet_pubkey") == SAMPLE_WALLET_PUBKEY
    assert params.get("nonexistent", "default") == "default"


def test_nwc_uri_with_lud16():
    uri = f"nostr+walletconnect://{SAMPLE_WALLET_PUBKEY}?relay={SAMPLE_RELAY_URL}&secret={SAMPLE_CLIENT_SECRET}&lud16=user@getalby.com"
    params = parse_nwc_uri(uri)
    assert params.lud16 == "user@getalby.com"
    assert params["lud16"] == "user@getalby.com"


def test_nwc_uri_url_encoded_relay():
    encoded_relay = "wss%3A%2F%2Fnostr.wine%2Fv1"
    uri = f"nostr+walletconnect://{SAMPLE_WALLET_PUBKEY}?relay={encoded_relay}&secret={SAMPLE_CLIENT_SECRET}"
    params = parse_nwc_uri(uri)
    assert params.relay_url == "wss://nostr.wine/v1"


def test_nwc_uri_multiple_relays():
    uri = (
        f"nostr+walletconnect://{SAMPLE_WALLET_PUBKEY}?"
        f"relay=wss://relay1.com&relay=wss://relay2.com&secret={SAMPLE_CLIENT_SECRET}"
    )
    params = NwcConnectionParams.from_uri(uri)
    assert params.relay_url == "wss://relay1.com"
    assert params.relay_urls == ["wss://relay1.com", "wss://relay2.com"]


def test_nwc_uri_comma_separated_relays():
    uri = (
        f"nostr+walletconnect://{SAMPLE_WALLET_PUBKEY}?"
        f"relay=wss://relay1.com,wss://relay2.com&secret={SAMPLE_CLIENT_SECRET}"
    )
    params = NwcConnectionParams.from_uri(uri)
    assert params.relay_url == "wss://relay1.com"
    assert params.relay_urls == ["wss://relay1.com", "wss://relay2.com"]


def test_nwc_uri_roundtrip_to_uri():
    uri = f"nostr+walletconnect://{SAMPLE_WALLET_PUBKEY}?relay={SAMPLE_RELAY_URL}&secret={SAMPLE_CLIENT_SECRET}&lud16=user@getalby.com"
    params = NwcConnectionParams.from_uri(uri)
    serialized = params.to_uri()
    params2 = NwcConnectionParams.from_uri(serialized)
    assert params.wallet_pubkey == params2.wallet_pubkey
    assert params.relay_url == params2.relay_url
    assert params.secret == params2.secret
    assert params.lud16 == params2.lud16


def test_nwc_uri_to_dict():
    uri = f"nostr+walletconnect://{SAMPLE_WALLET_PUBKEY}?relay={SAMPLE_RELAY_URL}&secret={SAMPLE_CLIENT_SECRET}&lud16=user@getalby.com"
    params = NwcConnectionParams.from_uri(uri)
    d = params.to_dict()
    assert isinstance(d, dict)
    assert d["wallet_pubkey"] == SAMPLE_WALLET_PUBKEY
    assert d["relay_url"] == SAMPLE_RELAY_URL
    assert d["secret"] == SAMPLE_CLIENT_SECRET
    assert d["lud16"] == "user@getalby.com"


def test_nwc_uri_client_pubkey_derivation():
    uri = f"nostr+walletconnect://{SAMPLE_WALLET_PUBKEY}?relay={SAMPLE_RELAY_URL}&secret={SAMPLE_CLIENT_SECRET}"
    params = NwcConnectionParams.from_uri(uri)
    assert len(params.client_pubkey) == 64
    assert params.client_pubkey == params.client_pubkey_hex


def test_nwc_uri_case_insensitivity_normalization():
    upper_pub = SAMPLE_WALLET_PUBKEY.upper()
    upper_sec = SAMPLE_CLIENT_SECRET.upper()
    uri = f"nostr+walletconnect://{upper_pub}?relay={SAMPLE_RELAY_URL}&secret={upper_sec}"
    params = NwcConnectionParams.from_uri(uri)
    assert params.wallet_pubkey == SAMPLE_WALLET_PUBKEY.lower()
    assert params.secret == SAMPLE_CLIENT_SECRET.lower()


def test_nwc_uri_trailing_slash_handling():
    uri = f"nostr+walletconnect://{SAMPLE_WALLET_PUBKEY}/?relay={SAMPLE_RELAY_URL}&secret={SAMPLE_CLIENT_SECRET}"
    params = NwcConnectionParams.from_uri(uri)
    assert params.wallet_pubkey == SAMPLE_WALLET_PUBKEY


def test_rejection_missing_scheme():
    uri = f"https://{SAMPLE_WALLET_PUBKEY}?relay={SAMPLE_RELAY_URL}&secret={SAMPLE_CLIENT_SECRET}"
    with pytest.raises(ValueError, match="nostr\\+walletconnect://"):
        NwcConnectionParams.from_uri(uri)


def test_rejection_missing_secret():
    uri = f"nostr+walletconnect://{SAMPLE_WALLET_PUBKEY}?relay={SAMPLE_RELAY_URL}"
    with pytest.raises(ValueError, match="Missing required 'secret'"):
        NwcConnectionParams.from_uri(uri)


def test_rejection_missing_relay():
    uri = f"nostr+walletconnect://{SAMPLE_WALLET_PUBKEY}?secret={SAMPLE_CLIENT_SECRET}"
    with pytest.raises(ValueError, match="Missing required 'relay'"):
        NwcConnectionParams.from_uri(uri)


def test_rejection_empty_relay_param():
    uri = f"nostr+walletconnect://{SAMPLE_WALLET_PUBKEY}?relay=&secret={SAMPLE_CLIENT_SECRET}"
    with pytest.raises(ValueError, match="Missing required 'relay'"):
        NwcConnectionParams.from_uri(uri)


def test_rejection_invalid_relay_scheme():
    uri = f"nostr+walletconnect://{SAMPLE_WALLET_PUBKEY}?relay=http://relay.damus.io&secret={SAMPLE_CLIENT_SECRET}"
    with pytest.raises(ValueError, match="Relay URL must begin with ws:// or wss://"):
        NwcConnectionParams.from_uri(uri)


def test_rejection_malformed_pubkey_length():
    bad_pub = "1234abcd" * 7  # 56 chars
    uri = f"nostr+walletconnect://{bad_pub}?relay={SAMPLE_RELAY_URL}&secret={SAMPLE_CLIENT_SECRET}"
    with pytest.raises(ValueError, match="Invalid wallet_pubkey"):
        NwcConnectionParams.from_uri(uri)


def test_rejection_non_hex_pubkey():
    bad_pub = "z" * 64
    uri = f"nostr+walletconnect://{bad_pub}?relay={SAMPLE_RELAY_URL}&secret={SAMPLE_CLIENT_SECRET}"
    with pytest.raises(ValueError, match="Invalid wallet_pubkey"):
        NwcConnectionParams.from_uri(uri)


def test_rejection_non_hex_secret():
    bad_sec = "g" * 64
    uri = f"nostr+walletconnect://{SAMPLE_WALLET_PUBKEY}?relay={SAMPLE_RELAY_URL}&secret={bad_sec}"
    with pytest.raises(ValueError, match="Invalid secret"):
        NwcConnectionParams.from_uri(uri)


def test_rejection_out_of_range_scalar_secret():
    # Scalar 0
    zero_sec = "0" * 64
    uri = f"nostr+walletconnect://{SAMPLE_WALLET_PUBKEY}?relay={SAMPLE_RELAY_URL}&secret={zero_sec}"
    with pytest.raises(ValueError, match="Invalid secret"):
        NwcConnectionParams.from_uri(uri)

    # Scalar >= SECP256K1_ORDER
    overflow_sec = hex(SECP256K1_ORDER)[2:].zfill(64)
    uri = f"nostr+walletconnect://{SAMPLE_WALLET_PUBKEY}?relay={SAMPLE_RELAY_URL}&secret={overflow_sec}"
    with pytest.raises(ValueError, match="Invalid secret"):
        NwcConnectionParams.from_uri(uri)


def test_rejection_empty_and_whitespace_uris():
    for invalid in ["", "   ", "\t\n"]:
        with pytest.raises(ValueError):
            NwcConnectionParams.from_uri(invalid)


def test_rejection_non_string_uri():
    with pytest.raises(ValueError, match="must be string"):
        NwcConnectionParams.from_uri(12345)


def test_rejection_relay_missing_host():
    uri = f"nostr+walletconnect://{SAMPLE_WALLET_PUBKEY}?relay=ws://&secret={SAMPLE_CLIENT_SECRET}"
    with pytest.raises(ValueError, match="missing host"):
        NwcConnectionParams.from_uri(uri)


def test_direct_instantiation_and_validation():
    params = NwcConnectionParams(
        wallet_pubkey=SAMPLE_WALLET_PUBKEY,
        relay_url=SAMPLE_RELAY_URL,
        secret=SAMPLE_CLIENT_SECRET,
    )
    assert params.relay_urls == [SAMPLE_RELAY_URL]

    with pytest.raises(ValueError, match="Invalid wallet_pubkey"):
        NwcConnectionParams(
            wallet_pubkey="short",
            relay_url=SAMPLE_RELAY_URL,
            secret=SAMPLE_CLIENT_SECRET,
        )


def test_dict_access_keyerror():
    params = NwcConnectionParams(
        wallet_pubkey=SAMPLE_WALLET_PUBKEY,
        relay_url=SAMPLE_RELAY_URL,
        secret=SAMPLE_CLIENT_SECRET,
    )
    with pytest.raises(KeyError, match="no field"):
        _ = params["unknown_field"]
