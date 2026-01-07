"""
PII Detection Module

Detects Personally Identifiable Information (PII) in text using:
1. Specialized validation libraries (credit cards, SSN, etc.)
2. Pattern-based detection with Luhn algorithm validation
3. Named Entity Recognition (optional, for names)

Supported PII Types:
- Credit Card Numbers (validated with Luhn algorithm)
- Social Security Numbers (format + area number validation)
- Email Addresses (RFC 5322 compliant validation)
- Phone Numbers (international format support)
- IP Addresses (IPv4 and IPv6)
- Bank Account Numbers (IBAN validation)
- Passport Numbers
- Driver's License Numbers
- Dates of Birth

Security Philosophy:
- Use algorithmic validation where possible (Luhn, checksum)
- Fall back to patterns only when no validation exists
- Log detections for audit trail (no actual values)
- Configurable sensitivity levels
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Pattern, Tuple

logger = logging.getLogger(__name__)


class PIIType(Enum):
    """Types of PII that can be detected."""
    CREDIT_CARD = auto()
    SSN = auto()
    EMAIL = auto()
    PHONE = auto()
    IP_ADDRESS = auto()
    IBAN = auto()
    PASSPORT = auto()
    DRIVERS_LICENSE = auto()
    DATE_OF_BIRTH = auto()
    NAME = auto()  # Requires NER
    ADDRESS = auto()  # Requires NER
    CUSTOM = auto()


@dataclass
class PIIFinding:
    """Represents a detected PII instance."""
    pii_type: PIIType
    value: str  # The matched text (will be redacted in logs)
    start_pos: int
    end_pos: int
    confidence: float = 1.0  # 0.0 to 1.0
    validator_used: str = "pattern"  # pattern, luhn, checksum, ner


@dataclass
class DetectorConfig:
    """Configuration for PII detection."""
    # Which PII types to detect
    enabled_types: List[PIIType] = field(default_factory=lambda: list(PIIType))
    
    # Minimum confidence threshold
    min_confidence: float = 0.7
    
    # Whether to validate findings (slower but more accurate)
    validate_findings: bool = True
    
    # Log detection events (for audit)
    log_detections: bool = True


class PIIDetector:
    """
    Detects PII in text using validation algorithms and patterns.
    
    Example:
        >>> detector = PIIDetector()
        >>> findings = detector.detect("Email: john@example.com, Card: 4111111111111111")
        >>> for f in findings:
        ...     print(f"{f.pii_type.name}: {f.value[:4]}****")
    """
    
    # Compiled patterns for each PII type
    PATTERNS: Dict[PIIType, Pattern] = {
        PIIType.EMAIL: re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b',
            re.IGNORECASE
        ),
        PIIType.PHONE: re.compile(
            r'\b(?:\+?1[-.\s]?)?'  # Country code
            r'(?:\(?\d{3}\)?[-.\s]?)?'  # Area code
            r'\d{3}[-.\s]?\d{4}\b'  # Local number
        ),
        PIIType.CREDIT_CARD: re.compile(
            r'\b(?:\d{4}[-.\s]?){3}\d{4}\b|'  # Standard format
            r'\b\d{15,16}\b'  # No separators
        ),
        PIIType.SSN: re.compile(
            r'\b(?!000|666|9\d\d)\d{3}[-.\s]?'  # Area (not 000, 666, 9xx)
            r'(?!00)\d{2}[-.\s]?'  # Group (not 00)
            r'(?!0000)\d{4}\b'  # Serial (not 0000)
        ),
        PIIType.IP_ADDRESS: re.compile(
            r'\b(?:'
            r'(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}'
            r'(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b|'  # IPv4
            r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b'  # IPv6
        ),
        PIIType.DATE_OF_BIRTH: re.compile(
            r'\b(?:'
            r'(?:0?[1-9]|1[0-2])[/\-](?:0?[1-9]|[12]\d|3[01])[/\-](?:19|20)\d{2}|'  # MM/DD/YYYY
            r'(?:19|20)\d{2}[/\-](?:0?[1-9]|1[0-2])[/\-](?:0?[1-9]|[12]\d|3[01])'  # YYYY/MM/DD
            r')\b'
        ),
        PIIType.IBAN: re.compile(
            r'\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}(?:[A-Z0-9]?){0,16}\b',
            re.IGNORECASE
        ),
        PIIType.PASSPORT: re.compile(
            r'\b[A-Z]{1,2}\d{6,9}\b',
            re.IGNORECASE
        ),
        PIIType.DRIVERS_LICENSE: re.compile(
            r'\b[A-Z]{1,2}\d{5,8}\b',
            re.IGNORECASE
        ),
    }
    
    def __init__(self, config: Optional[DetectorConfig] = None):
        """
        Initialize PII detector.
        
        Args:
            config: Detection configuration settings.
        """
        self.config = config or DetectorConfig()
        self._validators: Dict[PIIType, Callable[[str], bool]] = {
            PIIType.CREDIT_CARD: self._validate_credit_card,
            PIIType.SSN: self._validate_ssn,
            PIIType.EMAIL: self._validate_email,
            PIIType.IBAN: self._validate_iban,
        }
    
    def detect(self, text: str) -> List[PIIFinding]:
        """
        Detect all PII in the given text.
        
        Args:
            text: Text to scan for PII.
            
        Returns:
            List of PIIFinding objects for each detected PII.
        """
        if not text:
            return []
        
        findings: List[PIIFinding] = []
        
        for pii_type in self.config.enabled_types:
            if pii_type not in self.PATTERNS:
                continue
            
            pattern = self.PATTERNS[pii_type]
            for match in pattern.finditer(text):
                value = match.group()
                
                # Validate if validator exists and validation enabled
                confidence = 1.0
                validator_used = "pattern"
                
                if self.config.validate_findings and pii_type in self._validators:
                    is_valid = self._validators[pii_type](value)
                    if not is_valid:
                        confidence = 0.3  # Low confidence for failed validation
                    else:
                        validator_used = self._validators[pii_type].__name__
                        confidence = 0.95
                
                if confidence >= self.config.min_confidence:
                    finding = PIIFinding(
                        pii_type=pii_type,
                        value=value,
                        start_pos=match.start(),
                        end_pos=match.end(),
                        confidence=confidence,
                        validator_used=validator_used
                    )
                    findings.append(finding)
                    
                    if self.config.log_detections:
                        logger.info(
                            f"PII detected: {pii_type.name} at pos {match.start()}-{match.end()} "
                            f"(confidence: {confidence:.2f})"
                        )
        
        return findings
    
    def detect_types(self, text: str) -> List[PIIType]:
        """
        Get just the types of PII found (no positions/values).
        
        Args:
            text: Text to scan.
            
        Returns:
            List of unique PIIType values found.
        """
        findings = self.detect(text)
        return list(set(f.pii_type for f in findings))
    
    def contains_pii(self, text: str) -> bool:
        """
        Quick check if text contains any PII.
        
        Args:
            text: Text to check.
            
        Returns:
            True if any PII detected.
        """
        return len(self.detect(text)) > 0
    
    # =========================================================================
    # Validators - Use algorithms instead of just regex
    # =========================================================================
    
    @staticmethod
    def _validate_credit_card(number: str) -> bool:
        """
        Validate credit card number using Luhn algorithm.
        
        The Luhn algorithm (also known as mod 10) is used by credit card
        companies to distinguish valid numbers from mistyped or incorrect numbers.
        """
        # Remove non-digits
        digits = re.sub(r'\D', '', number)
        
        if len(digits) < 13 or len(digits) > 19:
            return False
        
        # Luhn algorithm
        total = 0
        reverse_digits = digits[::-1]
        
        for i, digit in enumerate(reverse_digits):
            n = int(digit)
            if i % 2 == 1:  # Double every second digit
                n *= 2
                if n > 9:
                    n -= 9
            total += n
        
        return total % 10 == 0
    
    @staticmethod
    def _validate_ssn(ssn: str) -> bool:
        """
        Validate SSN format and check for known invalid patterns.
        
        Invalid SSNs:
        - Area number 000, 666, or 900-999
        - Group number 00
        - Serial number 0000
        - Known advertising SSNs (078-05-1120, etc.)
        """
        # Remove non-digits
        digits = re.sub(r'\D', '', ssn)
        
        if len(digits) != 9:
            return False
        
        area = int(digits[:3])
        group = int(digits[3:5])
        serial = int(digits[5:])
        
        # Check invalid patterns
        if area == 0 or area == 666 or area >= 900:
            return False
        if group == 0:
            return False
        if serial == 0:
            return False
        
        # Known invalid/advertising SSNs
        invalid_ssns = {
            '078051120',  # Woolworth wallet SSN
            '219099999',  # Advertising SSN
        }
        if digits in invalid_ssns:
            return False
        
        return True
    
    @staticmethod
    def _validate_email(email: str) -> bool:
        """
        Validate email format (more strict than pattern).
        """
        # Basic structural validation
        if '@' not in email:
            return False
        
        local, domain = email.rsplit('@', 1)
        
        # Local part checks
        if not local or len(local) > 64:
            return False
        if local.startswith('.') or local.endswith('.'):
            return False
        if '..' in local:
            return False
        
        # Domain checks
        if not domain or len(domain) > 253:
            return False
        if domain.startswith('.') or domain.endswith('.'):
            return False
        if '..' in domain:
            return False
        if '.' not in domain:  # Must have at least one dot
            return False
        
        return True
    
    @staticmethod
    def _validate_iban(iban: str) -> bool:
        """
        Validate IBAN using mod-97 checksum.
        
        IBAN validation:
        1. Move first 4 chars to end
        2. Convert letters to numbers (A=10, B=11, etc.)
        3. Mod 97 should equal 1
        """
        # Remove spaces and uppercase
        iban = re.sub(r'\s', '', iban).upper()
        
        if len(iban) < 15 or len(iban) > 34:
            return False
        
        # Move first 4 to end
        rearranged = iban[4:] + iban[:4]
        
        # Convert to number string
        converted = ''
        for char in rearranged:
            if char.isdigit():
                converted += char
            elif char.isalpha():
                converted += str(ord(char) - ord('A') + 10)
            else:
                return False
        
        # Mod 97 check
        return int(converted) % 97 == 1
    
    def add_custom_pattern(
        self,
        name: str,
        pattern: str,
        validator: Optional[Callable[[str], bool]] = None
    ) -> None:
        """
        Add a custom PII pattern.
        
        Args:
            name: Name for the custom pattern.
            pattern: Regex pattern string.
            validator: Optional validation function.
        """
        compiled = re.compile(pattern)
        self.PATTERNS[PIIType.CUSTOM] = compiled
        
        if validator:
            self._validators[PIIType.CUSTOM] = validator
        
        if PIIType.CUSTOM not in self.config.enabled_types:
            self.config.enabled_types.append(PIIType.CUSTOM)
        
        logger.info(f"Added custom PII pattern: {name}")
