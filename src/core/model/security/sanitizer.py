"""
Data Sanitizer Module

Orchestrates PII detection and scrubbing/masking from text and structured data.
Uses the PIIDetector for finding sensitive data and replaces it with safe tokens.

Features:
- Multiple redaction strategies (token replacement, masking, hashing)
- Recursive dictionary/list sanitization
- Preserves data structure while removing PII
- Configurable replacement tokens
- Audit logging of sanitization events

Usage:
    >>> sanitizer = DataSanitizer()
    >>> clean = sanitizer.sanitize("Email: john@example.com")
    >>> print(clean)  # "Email: [EMAIL_REDACTED]"
"""

import hashlib
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Union

from .pii_detector import PIIDetector, PIIFinding, PIIType, DetectorConfig

logger = logging.getLogger(__name__)


class RedactionStrategy(Enum):
    """How to redact detected PII."""
    TOKEN = auto()      # Replace with [TYPE_REDACTED]
    MASK = auto()       # Replace with ****
    HASH = auto()       # Replace with hash of value
    PARTIAL = auto()    # Show partial (e.g., ****1234)
    REMOVE = auto()     # Remove entirely


@dataclass
class SanitizationConfig:
    """Configuration for data sanitization."""
    
    # Default redaction strategy
    default_strategy: RedactionStrategy = RedactionStrategy.TOKEN
    
    # Strategy overrides per PII type
    type_strategies: Dict[PIIType, RedactionStrategy] = field(default_factory=dict)
    
    # Custom replacement tokens
    replacement_tokens: Dict[PIIType, str] = field(default_factory=lambda: {
        PIIType.EMAIL: "[EMAIL_REDACTED]",
        PIIType.PHONE: "[PHONE_REDACTED]",
        PIIType.CREDIT_CARD: "[CC_REDACTED]",
        PIIType.SSN: "[SSN_REDACTED]",
        PIIType.IP_ADDRESS: "[IP_REDACTED]",
        PIIType.DATE_OF_BIRTH: "[DOB_REDACTED]",
        PIIType.IBAN: "[IBAN_REDACTED]",
        PIIType.PASSPORT: "[PASSPORT_REDACTED]",
        PIIType.DRIVERS_LICENSE: "[DL_REDACTED]",
        PIIType.NAME: "[NAME_REDACTED]",
        PIIType.ADDRESS: "[ADDRESS_REDACTED]",
        PIIType.CUSTOM: "[REDACTED]",
    })
    
    # Masking character for MASK strategy
    mask_char: str = "*"
    
    # Number of visible chars for PARTIAL strategy
    partial_visible: int = 4
    
    # Whether to log sanitization events
    log_sanitization: bool = True
    
    # Hash salt for HASH strategy
    hash_salt: str = ""


class DataSanitizer:
    """
    Sanitizes data by detecting and redacting PII.
    
    Example:
        >>> sanitizer = DataSanitizer()
        >>> 
        >>> # Text sanitization
        >>> clean = sanitizer.sanitize("Contact: john@example.com")
        >>> print(clean)  # "Contact: [EMAIL_REDACTED]"
        >>> 
        >>> # Dictionary sanitization
        >>> data = {"email": "test@example.com", "name": "John"}
        >>> clean_data = sanitizer.sanitize_dict(data)
    """
    
    def __init__(
        self,
        config: Optional[SanitizationConfig] = None,
        detector: Optional[PIIDetector] = None,
        detector_config: Optional[DetectorConfig] = None
    ):
        """
        Initialize data sanitizer.
        
        Args:
            config: Sanitization configuration.
            detector: Custom PIIDetector instance.
            detector_config: Configuration for default detector.
        """
        self.config = config or SanitizationConfig()
        self.detector = detector or PIIDetector(detector_config)
        self._sanitization_count = 0
    
    @property
    def sanitization_count(self) -> int:
        """Number of sanitization operations performed."""
        return self._sanitization_count
    
    def sanitize(self, text: str) -> str:
        """
        Remove PII from text.
        
        Args:
            text: Input text potentially containing PII.
            
        Returns:
            Sanitized text with PII redacted.
        """
        if not text:
            return text
        
        # Detect PII
        findings = self.detector.detect(text)
        
        if not findings:
            return text
        
        # Sort by position (reverse) to replace from end to start
        findings.sort(key=lambda f: f.start_pos, reverse=True)
        
        # Replace each finding
        result = text
        redacted_types = []
        
        for finding in findings:
            replacement = self._get_replacement(finding)
            result = result[:finding.start_pos] + replacement + result[finding.end_pos:]
            redacted_types.append(finding.pii_type.name)
        
        self._sanitization_count += 1
        
        if self.config.log_sanitization and redacted_types:
            logger.info(f"Sanitized {len(findings)} PII items: {', '.join(set(redacted_types))}")
        
        return result
    
    def sanitize_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively sanitize all string values in a dictionary.
        
        Args:
            data: Dictionary with potentially sensitive values.
            
        Returns:
            New dictionary with all string values sanitized.
        """
        if not data:
            return data
        
        result: Dict[str, Any] = self._sanitize_recursive(data)
        return result
    
    def sanitize_list(self, data: List[Any]) -> List[Any]:
        """
        Sanitize all items in a list.
        
        Args:
            data: List with potentially sensitive values.
            
        Returns:
            New list with all string values sanitized.
        """
        if not data:
            return data
        
        return [self._sanitize_value(item) for item in data]
    
    def _sanitize_recursive(self, data: Any) -> Any:
        """Recursively sanitize data structures."""
        if isinstance(data, dict):
            return {
                key: self._sanitize_value(value)
                for key, value in data.items()
            }
        elif isinstance(data, list):
            return [self._sanitize_value(item) for item in data]
        elif isinstance(data, str):
            return self.sanitize(data)
        else:
            return data
    
    def _sanitize_value(self, value: Any) -> Any:
        """Sanitize a single value based on its type."""
        if isinstance(value, str):
            return self.sanitize(value)
        elif isinstance(value, dict):
            return self._sanitize_recursive(value)
        elif isinstance(value, list):
            return [self._sanitize_value(item) for item in value]
        else:
            return value
    
    def _get_replacement(self, finding: PIIFinding) -> str:
        """Get the replacement text for a PII finding."""
        strategy = self.config.type_strategies.get(
            finding.pii_type,
            self.config.default_strategy
        )
        
        if strategy == RedactionStrategy.TOKEN:
            return self.config.replacement_tokens.get(
                finding.pii_type,
                "[REDACTED]"
            )
        
        elif strategy == RedactionStrategy.MASK:
            return self.config.mask_char * len(finding.value)
        
        elif strategy == RedactionStrategy.HASH:
            salted = f"{self.config.hash_salt}{finding.value}"
            hash_val = hashlib.sha256(salted.encode()).hexdigest()[:12]
            return f"[HASH:{hash_val}]"
        
        elif strategy == RedactionStrategy.PARTIAL:
            visible = self.config.partial_visible
            if len(finding.value) <= visible:
                return self.config.mask_char * len(finding.value)
            masked = self.config.mask_char * (len(finding.value) - visible)
            return masked + finding.value[-visible:]
        
        elif strategy == RedactionStrategy.REMOVE:
            return ""
        
        return "[REDACTED]"
    
    def get_findings(self, text: str) -> List[PIIFinding]:
        """
        Get PII findings without sanitizing.
        
        Args:
            text: Text to analyze.
            
        Returns:
            List of PII findings.
        """
        return self.detector.detect(text)
    
    def contains_pii(self, text: str) -> bool:
        """
        Quick check if text contains PII.
        
        Args:
            text: Text to check.
            
        Returns:
            True if PII detected.
        """
        return self.detector.contains_pii(text)


def create_strict_sanitizer() -> DataSanitizer:
    """
    Create a sanitizer with strict settings.
    
    Uses token replacement and logs all sanitization events.
    """
    config = SanitizationConfig(
        default_strategy=RedactionStrategy.TOKEN,
        log_sanitization=True
    )
    
    detector_config = DetectorConfig(
        min_confidence=0.5,  # Lower threshold = more aggressive
        validate_findings=True,
        log_detections=True
    )
    
    return DataSanitizer(config=config, detector_config=detector_config)


def create_minimal_sanitizer() -> DataSanitizer:
    """
    Create a sanitizer that only handles high-risk PII.
    
    Only detects credit cards, SSN, and email.
    """
    detector_config = DetectorConfig(
        enabled_types=[PIIType.CREDIT_CARD, PIIType.SSN, PIIType.EMAIL],
        min_confidence=0.8,
        validate_findings=True,
        log_detections=False
    )
    
    return DataSanitizer(detector_config=detector_config)

