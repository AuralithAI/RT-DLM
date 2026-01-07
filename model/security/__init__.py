"""
Model Security Module

Provides security utilities for the Memory Bank system:
- Encryption: AES-256 encryption for sensitive data storage
- PII Detection: Detects personally identifiable information using proper libraries
- Sanitizer: Scrubs/masks PII from text and structured data
- Hashing: Secure one-way hashing for identifiers

Security Philosophy:
- Never store raw PII (emails, SSNs, credit cards, etc.)
- Always hash user/session identifiers
- Optionally encrypt sensitive context and metadata
- Fail safely - if detection fails, err on the side of caution

Usage:
    from model.security import SecureStorage, PIIDetector, DataSanitizer, IdentifierHasher
    
    # Encryption
    storage = SecureStorage(encryption_key="secret")
    encrypted = storage.encrypt("sensitive data")
    
    # PII Detection
    detector = PIIDetector()
    findings = detector.detect("Contact john@example.com")
    
    # Data Sanitization
    sanitizer = DataSanitizer()
    clean_text = sanitizer.sanitize("SSN: 123-45-6789")
    
    # Identifier Hashing
    hasher = IdentifierHasher()
    hashed_id = hasher.hash("user_123")
"""

from .encryption import SecureStorage, EncryptionConfig
from .pii_detector import PIIDetector, PIIType, PIIFinding
from .sanitizer import DataSanitizer, SanitizationConfig
from .hashing import IdentifierHasher, HashConfig

__all__ = [
    # Encryption
    "SecureStorage",
    "EncryptionConfig",
    # PII Detection
    "PIIDetector",
    "PIIType",
    "PIIFinding",
    # Sanitization
    "DataSanitizer", 
    "SanitizationConfig",
    # Hashing
    "IdentifierHasher",
    "HashConfig",
]
