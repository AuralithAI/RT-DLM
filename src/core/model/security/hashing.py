"""
Identifier Hashing Module

Provides secure one-way hashing for identifiers (user_id, session_id, etc.).
Ensures that raw identifiers are never stored while still allowing lookups.

Features:
- SHA-256 based hashing with configurable salt
- Consistent hashing (same input = same output)
- Salt management (load/save to file)
- HMAC option for additional security
- Truncation options for storage efficiency

Security Notes:
- Hashes are one-way - original values cannot be recovered
- Salt should be kept secret and consistent across sessions
- Use different salts for different identifier types if needed

Usage:
    >>> hasher = IdentifierHasher(salt="my_secret_salt")
    >>> hashed = hasher.hash("user_12345")
    >>> print(hashed)  # "a1b2c3d4..."
    >>> 
    >>> # Verification
    >>> hasher.verify("user_12345", hashed)  # True
"""

import hashlib
import hmac
import logging
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class HashConfig:
    """Configuration for identifier hashing."""
    
    # Hash algorithm
    algorithm: str = "sha256"
    
    # Output length (truncate hash to this many chars, 0 = full)
    output_length: int = 32
    
    # Use HMAC instead of simple hash
    use_hmac: bool = True
    
    # Salt file name
    salt_filename: str = ".hash_salt"
    
    # Salt length in bytes
    salt_length: int = 32


class IdentifierHasher:
    """
    Secure one-way hashing for identifiers.
    
    Example:
        >>> hasher = IdentifierHasher()
        >>> user_hash = hasher.hash("user_john_doe")
        >>> session_hash = hasher.hash("session_abc123")
        >>> 
        >>> # Same input always produces same output
        >>> hasher.hash("user_john_doe") == user_hash  # True
        >>> 
        >>> # Verify an identifier against a hash
        >>> hasher.verify("user_john_doe", user_hash)  # True
    """
    
    def __init__(
        self,
        salt: Optional[str] = None,
        storage_dir: Optional[Union[str, Path]] = None,
        config: Optional[HashConfig] = None
    ):
        """
        Initialize identifier hasher.
        
        Args:
            salt: Salt string for hashing. If None, loads/generates from file.
            storage_dir: Directory to store salt file.
            config: Hashing configuration.
        """
        self.config = config or HashConfig()
        self.storage_dir = Path(storage_dir) if storage_dir else None
        
        # Load or create salt
        if salt:
            self._salt = salt.encode() if isinstance(salt, str) else salt
        else:
            self._salt = self._load_or_create_salt()
        
        logger.debug("IdentifierHasher initialized")
    
    @property
    def salt(self) -> bytes:
        """Get the salt (as bytes)."""
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
            logger.info(f"Created new hash salt at {salt_path}")
            return salt
        
        # No storage dir, generate ephemeral salt
        logger.warning("No storage_dir provided - using ephemeral salt")
        return secrets.token_bytes(self.config.salt_length)
    
    def hash(self, identifier: Optional[str]) -> Optional[str]:
        """
        Hash an identifier.
        
        Args:
            identifier: The identifier to hash.
            
        Returns:
            Hashed identifier string, or None if input is None.
        """
        if identifier is None:
            return None
        
        if not identifier:
            return ""
        
        if self.config.use_hmac:
            hash_obj = hmac.new(
                self._salt,
                identifier.encode(),
                self.config.algorithm
            )
            result = hash_obj.hexdigest()
        else:
            # Simple salted hash
            salted = self._salt + identifier.encode()
            simple_hash = hashlib.new(self.config.algorithm)
            simple_hash.update(salted)
            result = simple_hash.hexdigest()
        
        # Truncate if needed
        if self.config.output_length > 0:
            result = result[:self.config.output_length]
        
        return result
    
    def verify(self, identifier: str, hashed: str) -> bool:
        """
        Verify an identifier against a hash.
        
        Args:
            identifier: The original identifier.
            hashed: The hash to verify against.
            
        Returns:
            True if the identifier produces the same hash.
        """
        return self.hash(identifier) == hashed
    
    def hash_with_pepper(self, identifier: str, pepper: str) -> str:
        """
        Hash with additional pepper (per-record secret).
        
        Args:
            identifier: The identifier to hash.
            pepper: Additional secret for this specific hash.
            
        Returns:
            Hashed identifier.
        """
        combined = f"{pepper}:{identifier}"
        return self.hash(combined)
    
    def batch_hash(self, identifiers: list[Optional[str]]) -> list[Optional[str]]:
        """
        Hash multiple identifiers.
        
        Args:
            identifiers: List of identifiers to hash.
            
        Returns:
            List of hashed identifiers.
        """
        return [self.hash(i) for i in identifiers]


class IdentifierManager:
    """
    Manages hashing for different types of identifiers.
    
    Uses separate hashers for different identifier types to prevent
    cross-correlation attacks.
    
    Example:
        >>> manager = IdentifierManager(storage_dir="./hashes")
        >>> user_hash = manager.hash_user_id("user_123")
        >>> session_hash = manager.hash_session_id("session_abc")
    """
    
    def __init__(
        self,
        storage_dir: Optional[Union[str, Path]] = None,
        user_salt: Optional[str] = None,
        session_salt: Optional[str] = None
    ):
        """
        Initialize identifier manager.
        
        Args:
            storage_dir: Directory for salt files.
            user_salt: Salt for user IDs.
            session_salt: Salt for session IDs.
        """
        self.storage_dir = Path(storage_dir) if storage_dir else None
        
        # Create separate hashers for different ID types
        user_config = HashConfig(salt_filename=".user_salt")
        session_config = HashConfig(salt_filename=".session_salt")
        
        self._user_hasher = IdentifierHasher(
            salt=user_salt,
            storage_dir=storage_dir,
            config=user_config
        )
        
        self._session_hasher = IdentifierHasher(
            salt=session_salt,
            storage_dir=storage_dir,
            config=session_config
        )
    
    def hash_user_id(self, user_id: Optional[str]) -> Optional[str]:
        """Hash a user ID."""
        return self._user_hasher.hash(user_id)
    
    def hash_session_id(self, session_id: Optional[str]) -> Optional[str]:
        """Hash a session ID."""
        return self._session_hasher.hash(session_id)
    
    def verify_user_id(self, user_id: str, hashed: str) -> bool:
        """Verify a user ID against a hash."""
        return self._user_hasher.verify(user_id, hashed)
    
    def verify_session_id(self, session_id: str, hashed: str) -> bool:
        """Verify a session ID against a hash."""
        return self._session_hasher.verify(session_id, hashed)


def generate_salt(length: int = 32) -> str:
    """
    Generate a random salt string.
    
    Args:
        length: Length in bytes.
        
    Returns:
        Hex-encoded random salt.
    """
    return secrets.token_hex(length)

