"""
Encryption Module for Secure Data Storage

Provides AES-256 encryption using the Fernet scheme (AES-128-CBC with HMAC).
Used for encrypting sensitive metadata and context in the Memory Bank.

Features:
- Password-based key derivation using PBKDF2 with SHA-256
- Configurable iteration count for key derivation
- Automatic salt generation and management
- JSON encryption/decryption helpers
- Graceful degradation when cryptography library unavailable

Security Notes:
- Encryption key should be stored securely (env var, secret manager)
- Salt is stored alongside encrypted data for key derivation
- Never log or expose encryption keys
"""

import base64
import json
import logging
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)

# Optional cryptography support
try:
    from cryptography.fernet import Fernet, InvalidToken
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False
    InvalidToken = Exception  # Fallback for type hints
    logger.warning(
        "cryptography library not installed. Encryption disabled. "
        "Install with: pip install cryptography"
    )


@dataclass
class EncryptionConfig:
    """Configuration for encryption settings."""
    
    # Key derivation settings
    iterations: int = 480_000  # OWASP recommended minimum
    salt_length: int = 16  # 128 bits
    key_length: int = 32  # 256 bits
    
    # Storage settings
    salt_filename: str = ".encryption_salt"
    
    # Behavior settings
    fail_on_decrypt_error: bool = False  # If True, raise on decrypt failure
    log_encryption_events: bool = False  # If True, log encrypt/decrypt calls


class SecureStorage:
    """
    Provides AES encryption/decryption for sensitive data.
    
    Uses Fernet symmetric encryption (AES-128-CBC with HMAC-SHA256).
    Key is derived from a password using PBKDF2.
    
    Example:
        >>> storage = SecureStorage("my_secret_password")
        >>> encrypted = storage.encrypt("sensitive data")
        >>> decrypted = storage.decrypt(encrypted)
        >>> assert decrypted == "sensitive data"
    """
    
    def __init__(
        self,
        encryption_key: Optional[str] = None,
        salt: Optional[bytes] = None,
        storage_dir: Optional[Union[str, Path]] = None,
        config: Optional[EncryptionConfig] = None
    ):
        """
        Initialize secure storage.
        
        Args:
            encryption_key: Password for encryption. None disables encryption.
            salt: Pre-existing salt bytes. If None, generates or loads from file.
            storage_dir: Directory to store/load salt file. Required if salt not provided.
            config: Encryption configuration settings.
        """
        self.config = config or EncryptionConfig()
        self.storage_dir = Path(storage_dir) if storage_dir else None
        
        self._encryption_enabled = (
            encryption_key is not None and 
            CRYPTOGRAPHY_AVAILABLE
        )
        self._fernet: Optional[Any] = None
        self._salt: Optional[bytes] = None
        
        if encryption_key and not CRYPTOGRAPHY_AVAILABLE:
            logger.warning(
                "Encryption requested but cryptography not installed. "
                "Data will NOT be encrypted."
            )
        
        if self._encryption_enabled:
            self._salt = salt or self._load_or_create_salt()
            self._init_fernet(encryption_key)
            logger.info("SecureStorage initialized with encryption enabled")
    
    @property
    def encryption_enabled(self) -> bool:
        """Check if encryption is currently enabled."""
        return self._encryption_enabled
    
    @property
    def salt(self) -> Optional[bytes]:
        """Get the salt used for key derivation."""
        return self._salt
    
    def _load_or_create_salt(self) -> bytes:
        """Load existing salt from file or create new one."""
        if self.storage_dir:
            salt_path = self.storage_dir / self.config.salt_filename
            if salt_path.exists():
                return salt_path.read_bytes()
            
            # Create new salt and save
            salt = secrets.token_bytes(self.config.salt_length)
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            salt_path.write_bytes(salt)
            return salt
        
        # No storage dir, generate ephemeral salt
        return secrets.token_bytes(self.config.salt_length)
    
    def _init_fernet(self, password: str) -> None:
        """Initialize Fernet cipher with derived key."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=self.config.key_length,
            salt=self._salt,
            iterations=self.config.iterations,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        self._fernet = Fernet(key)
    
    def encrypt(self, plaintext: str) -> str:
        """
        Encrypt a string.
        
        Args:
            plaintext: The string to encrypt.
            
        Returns:
            Base64-encoded encrypted string, or original if encryption disabled.
        """
        if not self._encryption_enabled or not plaintext:
            return plaintext
        
        try:
            encrypted = self._fernet.encrypt(plaintext.encode())
            result = base64.urlsafe_b64encode(encrypted).decode()
            
            if self.config.log_encryption_events:
                logger.debug(f"Encrypted {len(plaintext)} chars")
            
            return result
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            return plaintext
    
    def decrypt(self, ciphertext: str) -> str:
        """
        Decrypt an encrypted string.
        
        Args:
            ciphertext: Base64-encoded encrypted string.
            
        Returns:
            Decrypted plaintext, or original if decryption fails/disabled.
        """
        if not self._encryption_enabled or not ciphertext:
            return ciphertext
        
        try:
            decoded = base64.urlsafe_b64decode(ciphertext.encode())
            decrypted = self._fernet.decrypt(decoded)
            
            if self.config.log_encryption_events:
                logger.debug("Decrypted data successfully")
            
            return decrypted.decode()  # type: ignore[union-attr]
        except InvalidToken:
            if self.config.fail_on_decrypt_error:
                raise
            logger.warning("Decryption failed - invalid token (wrong key?)")
            return ciphertext
        except Exception as e:
            if self.config.fail_on_decrypt_error:
                raise
            logger.error(f"Decryption failed: {e}")
            return ciphertext
    
    def encrypt_dict(self, data: Dict[str, Any]) -> str:
        """
        Encrypt a dictionary as JSON.
        
        Args:
            data: Dictionary to encrypt.
            
        Returns:
            Encrypted JSON string.
        """
        json_str = json.dumps(data, ensure_ascii=False, default=str)
        return self.encrypt(json_str)
    
    def decrypt_dict(self, ciphertext: str) -> Dict[str, Any]:
        """
        Decrypt an encrypted JSON dictionary.
        
        Args:
            ciphertext: Encrypted JSON string.
            
        Returns:
            Decrypted dictionary, or empty dict on failure.
        """
        if not ciphertext:
            return {}
        
        decrypted = self.decrypt(ciphertext)
        try:
            result: Dict[str, Any] = json.loads(decrypted)
            return result
        except json.JSONDecodeError:
            logger.warning("Failed to parse decrypted JSON")
            return {}
    
    def rotate_key(self, new_password: str, data_to_migrate: list[str]) -> list[str]:
        """
        Rotate encryption key and re-encrypt data.
        
        Args:
            new_password: New encryption password.
            data_to_migrate: List of encrypted strings to re-encrypt.
            
        Returns:
            List of re-encrypted strings with new key.
        """
        if not self._encryption_enabled:
            return data_to_migrate
        
        # Decrypt with old key
        decrypted = [self.decrypt(d) for d in data_to_migrate]
        
        # Generate new salt and initialize with new key
        self._salt = secrets.token_bytes(self.config.salt_length)
        if self.storage_dir:
            salt_path = self.storage_dir / self.config.salt_filename
            salt_path.write_bytes(self._salt)
        
        self._init_fernet(new_password)
        
        # Re-encrypt with new key
        return [self.encrypt(d) for d in decrypted]


def generate_encryption_key(length: int = 32) -> str:
    """
    Generate a secure random encryption key.
    
    Args:
        length: Length of the key in bytes.
        
    Returns:
        Hex-encoded random key.
    """
    return secrets.token_hex(length)

