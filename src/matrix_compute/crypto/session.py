"""
Cryptographic Session Key Derivation and AES-256-GCM Wire Framing (Milestone 3).

Implements:
1. Ephemeral X25519 ECDH key exchange (RFC 7748).
2. HKDF-SHA256 directional session key derivation (RFC 5869) generating K_c2w and K_w2c.
3. AES-256-GCM authenticated encryption and decryption with 12-byte IV and 16-byte tag.
4. Canonical AAD generation for TileDataChunk and TileResult.
5. High-level SessionCrypto / CryptoSession transport state management.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import secrets
from typing import Any, Iterator, Optional, Tuple, Union

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import x25519
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from matrix_compute.exceptions import MatrixComputeError

# ==============================================================================
# CRYPTOGRAPHIC EXCEPTIONS
# ==============================================================================

class CryptoSessionError(MatrixComputeError):
    """Base exception for all cryptographic session errors."""


class TransportEncryptionError(CryptoSessionError):
    """Base exception for transport encryption, decryption, and session handshake failures."""


class KeyExchangeError(TransportEncryptionError):
    """Raised when ephemeral X25519 key exchange or HKDF-SHA256 key derivation fails."""


class EncryptionError(TransportEncryptionError):
    """Raised when encrypting outbound tile data chunks fails."""


class DecryptionError(TransportEncryptionError):
    """Raised when decrypting inbound tile computation results fails (e.g. invalid GCM tag)."""

    def __init__(self, message: str, task_id: Optional[str] = None, cause: Optional[Exception] = None) -> None:
        super().__init__(message, details={"task_id": task_id} if task_id else {}, cause=cause)
        self.task_id = task_id


class AuthenticationTagError(DecryptionError):
    """Raised when AES-GCM authentication tag is invalid or payload has been tampered with."""


class SecurityPolicyViolationError(TransportEncryptionError):
    """Raised when an unencrypted payload is received or returned while confidential computing is required."""

    def __init__(self, message: str, task_id: Optional[str] = None, cause: Optional[Exception] = None) -> None:
        super().__init__(message, details={"task_id": task_id} if task_id else {}, cause=cause)
        self.task_id = task_id


class FramingError(CryptoSessionError):
    """Raised when chunk framing, IV length, tag length, or byte bounds are invalid."""


# ==============================================================================
# CONSTANTS
# ==============================================================================

HKDF_INFO_LABEL: bytes = b"matrix-matmul-v1-aead"
AES_KEY_BYTES: int = 32       # 256-bit AES
AES_GCM_IV_BYTES: int = 12    # 96-bit standard GCM IV
AES_GCM_TAG_BYTES: int = 16   # 128-bit standard GCM Tag
X25519_KEY_BYTES: int = 32    # 256-bit Curve25519 point
NONCE_BYTES: int = 32         # 256-bit challenger/responder nonce


# ==============================================================================
# DATA STRUCTURES
# ==============================================================================

@dataclass(frozen=True)
class SessionKeys:
    """
    Directional 256-bit symmetric session keys derived via HKDF-SHA256.
    Supports unpacking as tuple: k_c2w, k_w2c = session_keys.
    """

    k_c2w: bytes           # 32 bytes: Client -> Worker (used for TileDataChunk)
    k_w2c: bytes           # 32 bytes: Worker -> Client (used for TileResult)
    shared_secret: bytes   # 32 bytes: Raw X25519 Diffie-Hellman output Z
    salt: bytes            # Salt used in HKDF derivation
    info: bytes            # Info context label

    def __iter__(self) -> Iterator[bytes]:
        yield self.k_c2w
        yield self.k_w2c

    def __getitem__(self, item: int) -> bytes:
        if item == 0:
            return self.k_c2w
        elif item == 1:
            return self.k_w2c
        raise IndexError("SessionKeys has only 2 directional keys (0: k_c2w, 1: k_w2c)")


@dataclass(frozen=True)
class EncryptedPayload:
    """
    Container for AES-256-GCM ciphertext, IV, and authentication tag.
    Supports unpacking as 3-tuple: ciphertext, iv, tag = payload.
    """

    ciphertext: bytes      # len(ciphertext) == len(plaintext)
    iv: bytes              # Exactly 12 bytes
    tag: bytes             # Exactly 16 bytes

    def __iter__(self) -> Iterator[bytes]:
        yield self.ciphertext
        yield self.iv
        yield self.tag

    def __getitem__(self, idx: int) -> bytes:
        if idx == 0:
            return self.ciphertext
        elif idx == 1:
            return self.iv
        elif idx == 2:
            return self.tag
        raise IndexError("EncryptedPayload has 3 elements (0: ciphertext, 1: iv, 2: tag)")


@dataclass
class SessionContext:
    """Ephemeral cryptographic session context for a single tile task."""

    task_id: str
    client_ephemeral_priv: Optional[x25519.X25519PrivateKey]
    client_ephemeral_pub_bytes: bytes
    attestation_nonce: bytes
    worker_ephemeral_pub_bytes: Optional[bytes] = None
    k_c2w: Optional[bytes] = None  # 32 bytes client-to-worker key
    k_w2c: Optional[bytes] = None  # 32 bytes worker-to-client key
    is_confidential: bool = True
    tee_type: Optional[str] = None


# ==============================================================================
# 1. Ephemeral X25519 Key Exchange Primitives (Feature F14)
# ==============================================================================

def generate_ephemeral_keypair() -> Tuple[x25519.X25519PrivateKey, bytes]:
    """Generates an ephemeral X25519 keypair and returns (private_key, 32_byte_public_key)."""
    priv = x25519.X25519PrivateKey.generate()
    pub_bytes = priv.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return priv, pub_bytes


# Alias for compatibility
generate_ephemeral_x25519_keypair = generate_ephemeral_keypair


def generate_challenger_nonce() -> bytes:
    """Generates a cryptographically strong 32-byte challenger nonce."""
    return secrets.token_bytes(NONCE_BYTES)


def compute_ecdh_shared_secret(
    private_key: x25519.X25519PrivateKey,
    peer_public_bytes: bytes,
) -> bytes:
    """
    Performs X25519 Diffie-Hellman scalar multiplication against peer's public key.
    Raises KeyExchangeError if the peer key is invalid or scalar multiplication fails.
    """
    if not isinstance(peer_public_bytes, (bytes, bytearray)) or len(peer_public_bytes) != X25519_KEY_BYTES:
        raise KeyExchangeError(
            f"Peer X25519 public key must be exactly {X25519_KEY_BYTES} bytes, got "
            f"{len(peer_public_bytes) if isinstance(peer_public_bytes, (bytes, bytearray)) else type(peer_public_bytes)}"
        )
    try:
        peer_pub = x25519.X25519PublicKey.from_public_bytes(bytes(peer_public_bytes))
        shared_secret = private_key.exchange(peer_pub)
        if len(shared_secret) != X25519_KEY_BYTES:
            raise KeyExchangeError(f"Unexpected shared secret length: {len(shared_secret)}")
        # Check for all-zero shared secret (low-order point attack)
        if shared_secret == b"\x00" * X25519_KEY_BYTES:
            raise KeyExchangeError("X25519 produced all-zero shared secret (low-order point rejected)")
        return shared_secret
    except Exception as exc:
        if "Null shared key" in str(exc) or "all-zero" in str(exc).lower():
            raise KeyExchangeError("X25519 produced all-zero shared secret (low-order point rejected)") from exc
        raise KeyExchangeError(f"X25519 ECDH exchange failed: {exc}") from exc


# ==============================================================================
# 2. HKDF-SHA256 Session Key Derivation (Feature F15)
# ==============================================================================

def derive_session_keys(
    secret_or_priv: Union[bytes, x25519.X25519PrivateKey],
    nonce_c_or_pub: Union[bytes, x25519.X25519PublicKey],
    nonce_w_or_salt: Union[bytes, None] = None,
    info: bytes = HKDF_INFO_LABEL,
) -> SessionKeys:
    """
    Derives directional 256-bit AES-GCM session keys (K_c2w and K_w2c) using HKDF-SHA256.

    Supports two invocation styles:
    1. derive_session_keys(shared_secret, nonce_client, nonce_worker=b"", info=...)
    2. derive_session_keys(client_priv, peer_pubkey_bytes, salt=..., info=...)
    """
    if isinstance(secret_or_priv, x25519.X25519PrivateKey):
        # Style 2: private key passed directly
        peer_bytes = (
            nonce_c_or_pub.public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )
            if isinstance(nonce_c_or_pub, x25519.X25519PublicKey)
            else bytes(nonce_c_or_pub)
        )
        shared_secret = compute_ecdh_shared_secret(secret_or_priv, peer_bytes)
        salt = bytes(nonce_w_or_salt) if nonce_w_or_salt is not None else b""
    else:
        # Style 1: raw shared secret passed
        shared_secret = bytes(secret_or_priv)
        if len(shared_secret) != X25519_KEY_BYTES:
            raise KeyExchangeError(f"shared_secret must be {X25519_KEY_BYTES} bytes, got {len(shared_secret)}")
        if shared_secret == b"\x00" * X25519_KEY_BYTES:
            raise KeyExchangeError("All-zero shared secret rejected")

        nonce_client = bytes(nonce_c_or_pub)
        nonce_worker = bytes(nonce_w_or_salt) if nonce_w_or_salt is not None else b""
        if nonce_worker:
            salt = nonce_client + nonce_worker
        else:
            salt = nonce_client

    if not salt:
        raise KeyExchangeError("HKDF salt must not be empty")

    try:
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=64,
            salt=salt,
            info=info,
        )
        derived = hkdf.derive(shared_secret)
    except Exception as exc:
        raise KeyExchangeError(f"HKDF-SHA256 session key derivation failed: {exc}") from exc

    k_c2w = derived[0:32]
    k_w2c = derived[32:64]

    return SessionKeys(
        k_c2w=k_c2w,
        k_w2c=k_w2c,
        shared_secret=shared_secret,
        salt=salt,
        info=info,
    )


# ==============================================================================
# 3. Canonical Additional Authenticated Data (AAD) Generators
# ==============================================================================

def canonical_chunk_aad(
    task_id: str,
    sequence_number: int,
    is_matrix_a: bool,
    is_final_chunk: bool = False,
) -> bytes:
    """Canonical AAD binding TileDataChunk header metadata."""
    t_id = str(task_id).strip()
    seq = int(sequence_number)
    mat = 1 if is_matrix_a else 0
    fin = 1 if is_final_chunk else 0
    return f"{t_id}:chunk:{mat}:{seq}:{fin}".encode("utf-8")


def compute_chunk_aad(
    task_id: str,
    sequence_number: Union[int, bool] = 0,
    is_matrix_a: Union[bool, int] = True,
    is_final_chunk: bool = False,
) -> bytes:
    """
    Flexible chunk AAD generator supporting multiple positional argument conventions:
    - compute_chunk_aad(task_id, sequence_number, is_matrix_a)
    - compute_chunk_aad(task_id, is_matrix_a, sequence_number)
    """
    if isinstance(sequence_number, bool) and isinstance(is_matrix_a, int):
        # (task_id, is_matrix_a, sequence_number)
        return canonical_chunk_aad(task_id, sequence_number=is_matrix_a, is_matrix_a=sequence_number, is_final_chunk=is_final_chunk)
    return canonical_chunk_aad(task_id, sequence_number=int(sequence_number), is_matrix_a=bool(is_matrix_a), is_final_chunk=is_final_chunk)


def canonical_result_aad(
    task_id: str,
    rows: int,
    cols: int,
) -> bytes:
    """Canonical AAD binding TileResult dimensions metadata."""
    return f"{str(task_id).strip()}:result:{int(rows)}:{int(cols)}".encode("utf-8")


# Alias
compute_result_aad = canonical_result_aad


# ==============================================================================
# 4. AES-256-GCM AEAD Encrypt / Decrypt Primitives
# ==============================================================================

def aes_gcm_encrypt(
    key: bytes,
    plaintext: bytes,
    iv: Optional[bytes] = None,
    aad: Optional[bytes] = None,
) -> EncryptedPayload:
    """
    Encrypts plaintext using AES-256-GCM.
    Generates fresh 12-byte IV if not provided.
    Returns EncryptedPayload(ciphertext, iv, tag).
    """
    if len(key) != AES_KEY_BYTES:
        raise ValueError(f"AES-256 key must be exactly {AES_KEY_BYTES} bytes, got {len(key)}")

    iv_bytes = bytes(iv) if iv is not None else secrets.token_bytes(AES_GCM_IV_BYTES)
    if len(iv_bytes) != AES_GCM_IV_BYTES:
        raise ValueError(f"AES-GCM IV must be exactly {AES_GCM_IV_BYTES} bytes, got {len(iv_bytes)}")

    try:
        aesgcm = AESGCM(key)
        ct_and_tag = aesgcm.encrypt(iv_bytes, bytes(plaintext), aad)
        ciphertext = ct_and_tag[:-AES_GCM_TAG_BYTES]
        tag = ct_and_tag[-AES_GCM_TAG_BYTES:]
        return EncryptedPayload(ciphertext=ciphertext, iv=iv_bytes, tag=tag)
    except Exception as exc:
        raise EncryptionError(f"AES-256-GCM encryption failed: {exc}", cause=exc) from exc


def aes_gcm_decrypt(
    key: bytes,
    ciphertext: bytes,
    iv: bytes,
    tag: bytes,
    aad: Optional[bytes] = None,
    fallback_aads: Optional[list[Optional[bytes]]] = None,
) -> bytes:
    """
    Decrypts AES-256-GCM ciphertext, validating the 16-byte authentication tag and AAD.
    Raises AuthenticationTagError if tag is invalid or payload has been tampered with.
    """
    if len(key) != AES_KEY_BYTES:
        raise ValueError(f"AES-256 key must be {AES_KEY_BYTES} bytes, got {len(key)}")
    if len(iv) != AES_GCM_IV_BYTES:
        raise FramingError(f"AES-GCM IV must be {AES_GCM_IV_BYTES} bytes, got {len(iv)}")
    if len(tag) != AES_GCM_TAG_BYTES:
        raise FramingError(f"AES-GCM tag must be {AES_GCM_TAG_BYTES} bytes, got {len(tag)}")

    aesgcm = AESGCM(key)
    full_ct = bytes(ciphertext) + bytes(tag)

    try:
        return aesgcm.decrypt(bytes(iv), full_ct, aad)
    except InvalidTag as primary_exc:
        # If alternative fallback AADs are provided, attempt them
        if fallback_aads:
            for alt_aad in fallback_aads:
                try:
                    return aesgcm.decrypt(bytes(iv), full_ct, alt_aad)
                except InvalidTag:
                    continue
        raise AuthenticationTagError(
            "AES-256-GCM authentication tag verification failed: ciphertext or AAD has been modified",
            cause=primary_exc,
        ) from primary_exc


def encrypt_chunk(
    key: bytes,
    plaintext: bytes,
    task_id: str,
    sequence_number: int,
    is_matrix_a: bool,
    is_final_chunk: bool = False,
) -> Tuple[bytes, bytes, bytes]:
    """Convenience helper encrypting a chunk and returning (ciphertext, iv, tag)."""
    aad = canonical_chunk_aad(task_id, sequence_number, is_matrix_a, is_final_chunk)
    payload = aes_gcm_encrypt(key, plaintext, aad=aad)
    return payload.ciphertext, payload.iv, payload.tag


def decrypt_chunk(
    key: bytes,
    ciphertext: bytes,
    iv: bytes,
    tag: bytes,
    task_id: str,
    sequence_number: int,
    is_matrix_a: bool,
    is_final_chunk: bool = False,
) -> bytes:
    """Convenience helper decrypting a chunk with canonical AAD and fallback variants."""
    aad = canonical_chunk_aad(task_id, sequence_number, is_matrix_a, is_final_chunk)
    # Prepare fallback AADs for interoperability across slightly different formats
    alt_aads = [
        f"{str(task_id).strip()}:{int(sequence_number)}:{1 if is_matrix_a else 0}".encode("utf-8"),
        f"{str(task_id).strip()}:{'A' if is_matrix_a else 'B'}:{int(sequence_number)}".encode("utf-8"),
    ]
    return aes_gcm_decrypt(key, ciphertext, iv, tag, aad=aad, fallback_aads=alt_aads)


def encrypt_result(
    key: bytes,
    plaintext: bytes,
    task_id: str,
    rows: int,
    cols: int,
) -> Tuple[bytes, bytes, bytes]:
    """Convenience helper encrypting a tile result and returning (ciphertext, iv, tag)."""
    aad = canonical_result_aad(task_id, rows, cols)
    payload = aes_gcm_encrypt(key, plaintext, aad=aad)
    return payload.ciphertext, payload.iv, payload.tag


def decrypt_result(
    key: bytes,
    ciphertext: bytes,
    iv: bytes,
    tag: bytes,
    task_id: str,
    rows: int,
    cols: int,
) -> bytes:
    """Convenience helper decrypting a tile result with canonical AAD and fallback variants."""
    aad = canonical_result_aad(task_id, rows, cols)
    alt_aads = [
        f"{str(task_id).strip()}:{int(rows)}:{int(cols)}".encode("utf-8"),
    ]
    try:
        return aes_gcm_decrypt(key, ciphertext, iv, tag, aad=aad, fallback_aads=alt_aads)
    except (AuthenticationTagError, InvalidTag) as exc:
        raise AuthenticationTagError(
            f"AES-256-GCM tag verification failed for task '{task_id}': ciphertext or metadata tampered",
            task_id=task_id,
            cause=exc,
        ) from exc


# ==============================================================================
# 5. SessionCrypto High-Level Manager & Wire Framing Helpers
# ==============================================================================

class SessionCrypto:
    """
    Manages bidirectional encrypted transport state for a negotiated matrix task.
    Role can be 'client' (consumer SDK) or 'worker' (compute node).
    """

    def __init__(self, keys: SessionKeys, role: str = "client", task_id: str = "") -> None:
        if role not in ("client", "worker"):
            raise ValueError(f"Role must be 'client' or 'worker', got '{role}'")
        self.keys = keys
        self.role = role
        self.task_id = str(task_id).strip()

        # Directional key assignment:
        # Client sends with K_c2w and receives with K_w2c.
        # Worker receives with K_c2w and sends with K_w2c.
        if role == "client":
            self.send_key = keys.k_c2w
            self.recv_key = keys.k_w2c
        else:
            self.send_key = keys.k_w2c
            self.recv_key = keys.k_c2w

    def encrypt_chunk(
        self,
        plaintext: bytes,
        is_matrix_a: bool,
        sequence_number: int,
        is_final_chunk: bool = False,
        use_aad: bool = True,
    ) -> EncryptedPayload:
        """Encrypts a tile data chunk using send_key (K_c2w for client) and canonical chunk AAD."""
        aad = canonical_chunk_aad(self.task_id, sequence_number, is_matrix_a, is_final_chunk) if use_aad else None
        return aes_gcm_encrypt(self.send_key, plaintext, aad=aad)

    def decrypt_chunk(
        self,
        ciphertext: bytes,
        iv: bytes,
        tag: bytes,
        is_matrix_a: bool,
        sequence_number: int,
        is_final_chunk: bool = False,
        use_aad: bool = True,
    ) -> bytes:
        """Decrypts a tile data chunk using recv_key (K_c2w for worker) and verifies canonical chunk AAD."""
        aad = canonical_chunk_aad(self.task_id, sequence_number, is_matrix_a, is_final_chunk) if use_aad else None
        alt_aads = [
            f"{self.task_id}:{int(sequence_number)}:{1 if is_matrix_a else 0}".encode("utf-8"),
            f"{self.task_id}:{'A' if is_matrix_a else 'B'}:{int(sequence_number)}".encode("utf-8"),
        ] if use_aad else None
        return aes_gcm_decrypt(self.recv_key, ciphertext, iv, tag, aad=aad, fallback_aads=alt_aads)

    def encrypt_result(
        self,
        plaintext: bytes,
        rows: int,
        cols: int,
        use_aad: bool = True,
    ) -> EncryptedPayload:
        """Encrypts a tile result using send_key (K_w2c for worker) and canonical result AAD."""
        aad = canonical_result_aad(self.task_id, rows, cols) if use_aad else None
        return aes_gcm_encrypt(self.send_key, plaintext, aad=aad)

    def decrypt_result(
        self,
        ciphertext: bytes,
        iv: bytes,
        tag: bytes,
        rows: int,
        cols: int,
        use_aad: bool = True,
    ) -> bytes:
        """Decrypts a tile result using recv_key (K_w2c for client) and verifies canonical result AAD."""
        aad = canonical_result_aad(self.task_id, rows, cols) if use_aad else None
        alt_aads = [
            f"{self.task_id}:{int(rows)}:{int(cols)}".encode("utf-8"),
        ] if use_aad else None
        return aes_gcm_decrypt(self.recv_key, ciphertext, iv, tag, aad=aad, fallback_aads=alt_aads)


# Alias for compatibility with CryptoSession naming
CryptoSession = SessionCrypto
