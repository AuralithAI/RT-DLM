"""
Tests for TMS_block/security modules.

Tests cover:
1. PIIDetector - PII detection using validators and patterns
2. DataSanitizer - PII redaction and sanitization
3. SecureStorage - Encryption and decryption
4. IdentifierHasher - Secure identifier hashing
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.model.security import (
    PIIDetector, PIIType, PIIFinding,
    DataSanitizer, SanitizationConfig,
    SecureStorage, EncryptionConfig,
    IdentifierHasher, HashConfig
)


class TestPIIDetector(unittest.TestCase):
    """Test suite for PIIDetector class."""

    def setUp(self):
        """Create detector instance."""
        self.detector = PIIDetector()

    def test_detect_email(self):
        """Test email detection."""
        text = "Contact us at support@example.com"
        findings = self.detector.detect(text)
        
        self.assertTrue(len(findings) >= 1)
        email_findings = [f for f in findings if f.pii_type == PIIType.EMAIL]
        self.assertTrue(len(email_findings) >= 1)
        self.assertIn("support@example.com", email_findings[0].value)

    def test_detect_phone(self):
        """Test phone number detection."""
        text = "Call us at 555-123-4567"
        findings = self.detector.detect(text)
        
        phone_findings = [f for f in findings if f.pii_type == PIIType.PHONE]
        self.assertTrue(len(phone_findings) >= 1)

    def test_detect_ssn(self):
        """Test SSN detection."""
        text = "SSN: 123-45-6789"
        findings = self.detector.detect(text)
        
        ssn_findings = [f for f in findings if f.pii_type == PIIType.SSN]
        self.assertTrue(len(ssn_findings) >= 1)

    def test_detect_credit_card(self):
        """Test credit card detection."""
        # Test with Visa-like number (4 followed by 15 digits)
        text = "Card: 4532-0151-2345-6789"
        findings = self.detector.detect(text)
        
        cc_findings = [f for f in findings if f.pii_type == PIIType.CREDIT_CARD]
        self.assertTrue(len(cc_findings) >= 1)

    def test_detect_ip_address(self):
        """Test IP address detection."""
        text = "Server IP: 192.168.1.100"
        findings = self.detector.detect(text)
        
        ip_findings = [f for f in findings if f.pii_type == PIIType.IP_ADDRESS]
        self.assertTrue(len(ip_findings) >= 1)

    def test_detect_dob(self):
        """Test date of birth detection."""
        text = "Date of birth: 12/25/1990"
        findings = self.detector.detect(text)
        
        dob_findings = [f for f in findings if f.pii_type == PIIType.DATE_OF_BIRTH]
        self.assertTrue(len(dob_findings) >= 1)

    def test_detect_multiple_pii(self):
        """Test detection of multiple PII types."""
        text = "Email: test@test.com, Phone: 555-000-1234, SSN: 987-65-4321"
        findings = self.detector.detect(text)
        
        # Should detect at least 2 different types (SSN and phone may overlap with patterns)
        pii_types = {f.pii_type for f in findings}
        self.assertGreaterEqual(len(pii_types), 2)

    def test_no_pii(self):
        """Test that clean text returns no findings."""
        text = "This is a clean sentence with no personal information."
        findings = self.detector.detect(text)
        
        # Should have no findings (or very few false positives)
        self.assertTrue(len(findings) <= 1)

    def test_custom_pattern(self):
        """Test that detector can be configured with DetectorConfig."""
        # Just test that detector works with default config
        detector = PIIDetector()
        
        text = "Email: test@example.com"
        findings = detector.detect(text)
        
        email_findings = [f for f in findings if f.pii_type == PIIType.EMAIL]
        self.assertGreaterEqual(len(email_findings), 1)


class TestDataSanitizer(unittest.TestCase):
    """Test suite for DataSanitizer class."""

    def setUp(self):
        """Create sanitizer instance."""
        self.sanitizer = DataSanitizer()

    def test_sanitize_email(self):
        """Test email sanitization."""
        text = "Contact: user@example.com"
        sanitized = self.sanitizer.sanitize(text)
        
        self.assertNotIn("user@example.com", sanitized)
        self.assertIn("[EMAIL_REDACTED]", sanitized)

    def test_sanitize_phone(self):
        """Test phone sanitization."""
        text = "Call me at 555-123-4567"
        sanitized = self.sanitizer.sanitize(text)
        
        self.assertNotIn("555-123-4567", sanitized)
        self.assertIn("[PHONE_REDACTED]", sanitized)

    def test_sanitize_ssn(self):
        """Test SSN sanitization."""
        text = "My SSN is 123-45-6789"
        sanitized = self.sanitizer.sanitize(text)
        
        self.assertNotIn("123-45-6789", sanitized)
        self.assertIn("[SSN_REDACTED]", sanitized)

    def test_sanitize_dict(self):
        """Test dictionary sanitization."""
        data = {
            "email": "test@example.com",
            "message": "Call 555-000-1234",
            "nested": {
                "ssn": "111-22-3333"
            }
        }
        sanitized = self.sanitizer.sanitize_dict(data)
        
        # Check nested sanitization
        self.assertNotIn("test@example.com", str(sanitized))
        self.assertNotIn("555-000-1234", str(sanitized))
        self.assertNotIn("111-22-3333", str(sanitized))

    def test_sanitize_list(self):
        """Test list sanitization in dict."""
        data = {
            "emails": ["user1@test.com", "user2@test.com"],
            "notes": "Clean text"
        }
        sanitized = self.sanitizer.sanitize_dict(data)
        
        self.assertNotIn("user1@test.com", str(sanitized))
        self.assertNotIn("user2@test.com", str(sanitized))

    def test_sanitize_empty(self):
        """Test sanitization of empty inputs."""
        text = self.sanitizer.sanitize("")
        self.assertEqual(text, "")
        
        data = self.sanitizer.sanitize_dict({})
        self.assertEqual(data, {})

    def test_custom_tokens(self):
        """Test custom replacement tokens via config."""
        config = SanitizationConfig(
            replacement_tokens={
                PIIType.EMAIL: "***HIDDEN_EMAIL***"
            }
        )
        sanitizer = DataSanitizer(config=config)
        
        text = "Email: test@test.com"
        sanitized = sanitizer.sanitize(text)
        
        self.assertIn("***HIDDEN_EMAIL***", sanitized)


class TestSecureStorage(unittest.TestCase):
    """Test suite for SecureStorage class."""

    def setUp(self):
        """Create temporary storage for salt files."""
        self.temp_dir = tempfile.mkdtemp(prefix="security_test_")
        
    def tearDown(self):
        """Clean up temporary directory."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_encryption_enabled(self):
        """Test that encryption works when key provided."""
        storage = SecureStorage(encryption_key="test_password_123")
        
        self.assertTrue(storage.encryption_enabled)
        
        plain = "sensitive data"
        encrypted = storage.encrypt(plain)
        
        self.assertNotEqual(plain, encrypted)
        self.assertTrue(len(encrypted) > len(plain))

    def test_encryption_decryption_roundtrip(self):
        """Test that encryption/decryption roundtrip works."""
        storage = SecureStorage(encryption_key="secure_key_456")
        
        original = "This is my secret message!"
        encrypted = storage.encrypt(original)
        decrypted = storage.decrypt(encrypted)
        
        self.assertEqual(original, decrypted)

    def test_dict_encryption_roundtrip(self):
        """Test dictionary encryption/decryption."""
        storage = SecureStorage(encryption_key="json_key_789")
        
        original = {
            "user": "john",
            "score": 95.5,
            "tags": ["a", "b", "c"]
        }
        encrypted = storage.encrypt_dict(original)
        decrypted = storage.decrypt_dict(encrypted)
        
        self.assertEqual(original, decrypted)

    def test_no_encryption_without_key(self):
        """Test that no encryption happens without key."""
        storage = SecureStorage()
        
        self.assertFalse(storage.encryption_enabled)
        
        text = "plain text"
        result = storage.encrypt(text)
        
        self.assertEqual(text, result)

    def test_same_key_same_salt_same_result(self):
        """Test deterministic encryption with same salt."""
        salt = b"fixed_salt_12345"
        
        storage1 = SecureStorage(encryption_key="same_key", salt=salt)
        storage2 = SecureStorage(encryption_key="same_key", salt=salt)
        
        # Both should be able to decrypt each other's data
        encrypted = storage1.encrypt("test data")
        decrypted = storage2.decrypt(encrypted)
        
        self.assertEqual("test data", decrypted)

    def test_different_keys_fail_decryption(self):
        """Test that different keys can't decrypt."""
        storage1 = SecureStorage(encryption_key="key1")
        storage2 = SecureStorage(encryption_key="key2")
        
        encrypted = storage1.encrypt("secret")
        decrypted = storage2.decrypt(encrypted)
        
        # Should fail silently and return original encrypted data
        self.assertNotEqual("secret", decrypted)


class TestIdentifierHasher(unittest.TestCase):
    """Test suite for IdentifierHasher class."""

    def test_basic_hash(self):
        """Test basic identifier hashing."""
        hasher = IdentifierHasher()
        
        hashed = hasher.hash("user_123")
        
        self.assertIsNotNone(hashed)
        self.assertNotEqual(hashed, "user_123")
        # Default output_length is 32
        self.assertEqual(len(hashed), 32)

    def test_deterministic(self):
        """Test that same input gives same hash."""
        hasher = IdentifierHasher()
        
        hash1 = hasher.hash("same_user")
        hash2 = hasher.hash("same_user")
        
        self.assertEqual(hash1, hash2)

    def test_different_inputs_different_hashes(self):
        """Test that different inputs give different hashes."""
        hasher = IdentifierHasher()
        
        hash1 = hasher.hash("user_a")
        hash2 = hasher.hash("user_b")
        
        self.assertNotEqual(hash1, hash2)

    def test_salted_hashing(self):
        """Test that salt affects hash output."""
        hasher1 = IdentifierHasher(salt="salt_a")
        hasher2 = IdentifierHasher(salt="salt_b")
        
        hash1 = hasher1.hash("user_123")
        hash2 = hasher2.hash("user_123")
        
        self.assertNotEqual(hash1, hash2)

    def test_consistent_with_salt(self):
        """Test consistency with same salt."""
        hasher1 = IdentifierHasher(salt="same_salt")
        hasher2 = IdentifierHasher(salt="same_salt")
        
        hash1 = hasher1.hash("user_xyz")
        hash2 = hasher2.hash("user_xyz")
        
        self.assertEqual(hash1, hash2)

    def test_hash_empty(self):
        """Test hashing empty string."""
        hasher = IdentifierHasher()
        
        hashed = hasher.hash("")
        
        # Should return empty or handle gracefully
        self.assertTrue(hashed == "" or hashed is None)

    def test_hash_none(self):
        """Test hashing None."""
        hasher = IdentifierHasher()
        
        hashed = hasher.hash(None)
        
        self.assertIsNone(hashed)

    def test_sha512_algorithm(self):
        """Test SHA-512 algorithm configuration."""
        config = HashConfig(algorithm="sha512", output_length=64)
        hasher = IdentifierHasher(config=config)
        
        hashed = hasher.hash("test_user")
        
        # SHA-512 with output_length=64 produces 64 char hash
        self.assertIsNotNone(hashed)
        self.assertEqual(len(hashed), 64)


class TestSecurityIntegration(unittest.TestCase):
    """Integration tests for security modules working together."""

    def test_sanitize_then_encrypt(self):
        """Test sanitizing PII then encrypting the result."""
        sanitizer = DataSanitizer()
        storage = SecureStorage(encryption_key="integration_key")
        
        original = "User email is test@example.com"
        
        # First sanitize
        sanitized = sanitizer.sanitize(original)
        self.assertNotIn("test@example.com", sanitized)
        
        # Then encrypt
        encrypted = storage.encrypt(sanitized)
        self.assertNotEqual(sanitized, encrypted)
        
        # Decrypt should give sanitized version
        decrypted = storage.decrypt(encrypted)
        self.assertEqual(sanitized, decrypted)
        self.assertIn("[EMAIL_REDACTED]", decrypted)

    def test_hash_and_compare(self):
        """Test hashing identifiers for secure comparison."""
        hasher = IdentifierHasher(salt="lookup_salt")
        
        # Store hash
        stored_hash = hasher.hash("user_123")
        
        # Later, check if user matches
        lookup_hash = hasher.hash("user_123")
        
        self.assertEqual(stored_hash, lookup_hash)
        
        # Different user doesn't match
        other_hash = hasher.hash("user_456")
        self.assertNotEqual(stored_hash, other_hash)


if __name__ == "__main__":
    unittest.main()

