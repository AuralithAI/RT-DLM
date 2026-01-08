"""
Data Preprocessing Module

Handles data cleaning and quality filtering:
- Deduplication (exact, MinHash, SimHash)
- Quality filtering (length, language, content)
- PII removal
- Toxicity detection
- Text normalization
"""

import re
import hashlib
import logging
from typing import List, Dict, Optional, Generator, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import unicodedata

import numpy as np

from .config import PipelineConfig, QualityConfig
from .ingestion import IngestedItem

logger = logging.getLogger(__name__)


class DeduplicationMethod(Enum):
    """Deduplication methods."""
    EXACT = "exact"
    MINHASH = "minhash"
    SIMHASH = "simhash"


@dataclass
class QualityMetrics:
    """Quality metrics for a piece of content."""
    # Text metrics
    char_count: int = 0
    word_count: int = 0
    unique_word_ratio: float = 0.0
    special_char_ratio: float = 0.0
    avg_word_length: float = 0.0
    
    # Language
    detected_language: Optional[str] = None
    language_confidence: float = 0.0
    
    # Content quality
    has_pii: bool = False
    toxicity_score: float = 0.0
    is_duplicate: bool = False
    duplicate_of: Optional[str] = None
    
    # Overall
    quality_score: float = 0.0
    passed_filter: bool = True
    rejection_reasons: List[str] = field(default_factory=list)


@dataclass
class PreprocessedItem:
    """Preprocessed data item with quality metrics."""
    original: IngestedItem
    content: str
    metrics: QualityMetrics
    normalized: bool = False


class TextNormalizer:
    """Text normalization utilities."""
    
    # Common patterns
    URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
    EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    PHONE_PATTERN = re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b')
    SSN_PATTERN = re.compile(r'\b\d{3}[-]?\d{2}[-]?\d{4}\b')
    WHITESPACE_PATTERN = re.compile(r'\s+')
    
    @staticmethod
    def normalize_unicode(text: str) -> str:
        """Normalize unicode characters."""
        # NFKC normalization
        text = unicodedata.normalize("NFKC", text)
        return text
    
    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Normalize whitespace."""
        text = TextNormalizer.WHITESPACE_PATTERN.sub(" ", text)
        return text.strip()
    
    @staticmethod
    def remove_urls(text: str) -> str:
        """Remove URLs from text."""
        return TextNormalizer.URL_PATTERN.sub("[URL]", text)
    
    @staticmethod
    def remove_emails(text: str) -> str:
        """Remove email addresses from text."""
        return TextNormalizer.EMAIL_PATTERN.sub("[EMAIL]", text)
    
    @staticmethod
    def remove_phone_numbers(text: str) -> str:
        """Remove phone numbers from text."""
        return TextNormalizer.PHONE_PATTERN.sub("[PHONE]", text)
    
    @staticmethod
    def remove_ssn(text: str) -> str:
        """Remove Social Security Numbers from text."""
        return TextNormalizer.SSN_PATTERN.sub("[SSN]", text)
    
    @classmethod
    def remove_pii(cls, text: str) -> Tuple[str, bool]:
        """
        Remove personally identifiable information from text.
        
        Returns:
            Tuple of (cleaned text, whether PII was found)
        """
        original = text
        text = cls.remove_emails(text)
        text = cls.remove_phone_numbers(text)
        text = cls.remove_ssn(text)
        
        has_pii = text != original
        return text, has_pii
    
    @classmethod
    def normalize(cls, text: str, remove_pii: bool = True) -> Tuple[str, bool]:
        """
        Full text normalization.
        
        Returns:
            Tuple of (normalized text, whether PII was found)
        """
        text = cls.normalize_unicode(text)
        text = cls.normalize_whitespace(text)
        
        has_pii = False
        if remove_pii:
            text, has_pii = cls.remove_pii(text)
        
        return text, has_pii


class MinHashDeduplicator:
    """
    MinHash-based near-duplicate detection.
    
    Uses Locality Sensitive Hashing for efficient approximate matching.
    """
    
    def __init__(self, num_perm: int = 128, threshold: float = 0.8):
        """
        Initialize MinHash deduplicator.
        
        Args:
            num_perm: Number of permutations (more = more accurate, slower)
            threshold: Jaccard similarity threshold for duplicates
        """
        self.num_perm = num_perm
        self.threshold = threshold
        
        # Generate random hash functions (using fixed seed for reproducibility)
        np.random.seed(42)
        self.a = np.random.randint(1, 2**31 - 1, size=num_perm, dtype=np.int64)
        self.b = np.random.randint(0, 2**31 - 1, size=num_perm, dtype=np.int64)
        self.prime = 2**31 - 1
        
        # Storage for seen hashes
        self.signatures: Dict[str, np.ndarray] = {}
        self.buckets: Dict[int, Set[str]] = defaultdict(set)
        
        # LSH bands
        self.num_bands = 16
        self.rows_per_band = num_perm // self.num_bands
    
    def _get_shingles(self, text: str, k: int = 5) -> Set[int]:
        """Get k-shingles (character n-grams) as hashes."""
        text = text.lower()
        shingles = set()
        for i in range(len(text) - k + 1):
            shingle = text[i:i+k]
            shingle_hash = hash(shingle) & 0x7FFFFFFF
            shingles.add(shingle_hash)
        return shingles
    
    def _compute_signature(self, shingles: Set[int]) -> np.ndarray:
        """Compute MinHash signature for a set of shingles."""
        if not shingles:
            return np.full(self.num_perm, np.inf)
        
        signature = np.full(self.num_perm, np.inf)
        
        for shingle in shingles:
            # Compute all hash values at once
            hashes = (self.a * shingle + self.b) % self.prime
            signature = np.minimum(signature, hashes)
        
        return signature.astype(np.int64)
    
    def _get_lsh_buckets(self, signature: np.ndarray) -> List[int]:
        """Get LSH bucket IDs for a signature."""
        buckets = []
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band = signature[start:end]
            bucket_hash = hash(tuple(band))
            buckets.append(bucket_hash)
        return buckets
    
    def is_duplicate(self, doc_id: str, text: str) -> Tuple[bool, Optional[str]]:
        """
        Check if text is a near-duplicate of any seen text.
        
        Args:
            doc_id: Unique identifier for this document
            text: Text content to check
            
        Returns:
            Tuple of (is_duplicate, duplicate_of_id)
        """
        shingles = self._get_shingles(text)
        signature = self._compute_signature(shingles)
        buckets = self._get_lsh_buckets(signature)
        
        # Check for potential duplicates in the same buckets
        candidates = set()
        for band_idx, bucket_hash in enumerate(buckets):
            bucket_key = (band_idx, bucket_hash)
            candidates.update(self.buckets.get(bucket_key, set()))
        
        # Check actual similarity with candidates
        for candidate_id in candidates:
            if candidate_id in self.signatures:
                candidate_sig = self.signatures[candidate_id]
                similarity = np.mean(signature == candidate_sig)
                
                if similarity >= self.threshold:
                    return True, candidate_id
        
        # Not a duplicate - add to index
        self.signatures[doc_id] = signature
        for band_idx, bucket_hash in enumerate(buckets):
            bucket_key = (band_idx, bucket_hash)
            self.buckets[bucket_key].add(doc_id)
        
        return False, None
    
    def clear(self):
        """Clear all stored signatures."""
        self.signatures.clear()
        self.buckets.clear()


class ExactDeduplicator:
    """Exact hash-based deduplication."""
    
    def __init__(self):
        self.seen_hashes: Dict[str, str] = {}  # hash -> doc_id
    
    def is_duplicate(self, doc_id: str, text: str) -> Tuple[bool, Optional[str]]:
        """Check if text is an exact duplicate."""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        if text_hash in self.seen_hashes:
            return True, self.seen_hashes[text_hash]
        
        self.seen_hashes[text_hash] = doc_id
        return False, None
    
    def clear(self):
        """Clear all stored hashes."""
        self.seen_hashes.clear()


class DataPreprocessor:
    """
    Production-ready data preprocessor.
    
    Features:
    - Multiple deduplication methods
    - Quality filtering
    - PII removal
    - Toxicity detection (optional)
    - Language detection (optional)
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.quality_config = config.quality
        
        # Initialize normalizer
        self.normalizer = TextNormalizer()
        
        # Initialize deduplicator based on config
        self._init_deduplicator()
        
        # Optional: language detector
        self.lang_detector = None
        if self.quality_config.detect_language:
            self._init_language_detector()
        
        # Optional: toxicity detector
        self.toxicity_detector = None
        if self.quality_config.remove_toxicity:
            self._init_toxicity_detector()
        
        # Statistics
        self.stats = {
            "total_processed": 0,
            "passed": 0,
            "duplicates": 0,
            "quality_rejected": 0,
            "pii_found": 0,
            "toxic_rejected": 0,
            "rejection_reasons": defaultdict(int),
        }
    
    def _init_deduplicator(self):
        """Initialize deduplicator based on config."""
        method = self.quality_config.dedup_method
        
        if method == "minhash":
            self.deduplicator = MinHashDeduplicator(
                num_perm=self.quality_config.minhash_num_perm,
                threshold=self.quality_config.minhash_threshold,
            )
        else:  # exact
            self.deduplicator = ExactDeduplicator()
    
    def _init_language_detector(self):
        """Initialize language detector."""
        try:
            import langdetect
            self.lang_detector = langdetect
            logger.info("Language detection enabled")
        except ImportError:
            logger.warning("langdetect not installed. Language detection disabled.")
    
    def _init_toxicity_detector(self):
        """Initialize toxicity detector."""
        try:
            from transformers import pipeline
            self.toxicity_detector = pipeline(
                "text-classification",
                model="unitary/toxic-bert",
                device=-1,  # CPU
            )
            logger.info("Toxicity detection enabled")
        except ImportError:
            logger.warning("transformers not installed. Toxicity detection disabled.")
        except Exception as e:
            logger.warning(f"Could not load toxicity model: {e}")
    
    def compute_quality_metrics(self, text: str, doc_id: str) -> QualityMetrics:
        """Compute quality metrics for text content."""
        metrics = QualityMetrics()
        
        # Basic counts
        metrics.char_count = len(text)
        words = text.split()
        metrics.word_count = len(words)
        
        if metrics.word_count > 0:
            unique_words = set(w.lower() for w in words)
            metrics.unique_word_ratio = len(unique_words) / metrics.word_count
            metrics.avg_word_length = sum(len(w) for w in words) / metrics.word_count
        
        # Special character ratio
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        metrics.special_char_ratio = special_chars / max(1, metrics.char_count)
        
        # Language detection
        if self.lang_detector and metrics.word_count >= 10:
            try:
                result = self.lang_detector.detect_langs(text[:1000])
                if result:
                    metrics.detected_language = result[0].lang
                    metrics.language_confidence = result[0].prob
            except Exception:
                pass
        
        # Deduplication check
        if self.quality_config.deduplicate:
            is_dup, dup_of = self.deduplicator.is_duplicate(doc_id, text)
            metrics.is_duplicate = is_dup
            metrics.duplicate_of = dup_of
        
        # Toxicity check
        if self.toxicity_detector and metrics.word_count >= 5:
            try:
                result = self.toxicity_detector(text[:512])[0]
                if result["label"] == "toxic":
                    metrics.toxicity_score = result["score"]
            except Exception:
                pass
        
        return metrics
    
    def check_quality(self, metrics: QualityMetrics) -> Tuple[bool, List[str]]:
        """
        Check if content passes quality filters.
        
        Returns:
            Tuple of (passed, rejection_reasons)
        """
        reasons = []
        
        # Length checks
        if metrics.char_count < self.quality_config.min_text_length:
            reasons.append(f"too_short:{metrics.char_count}")
        if metrics.char_count > self.quality_config.max_text_length:
            reasons.append(f"too_long:{metrics.char_count}")
        
        # Word count checks
        if metrics.word_count < self.quality_config.min_word_count:
            reasons.append(f"too_few_words:{metrics.word_count}")
        if metrics.word_count > self.quality_config.max_word_count:
            reasons.append(f"too_many_words:{metrics.word_count}")
        
        # Quality checks
        if metrics.unique_word_ratio < self.quality_config.min_unique_words_ratio:
            reasons.append(f"low_unique_ratio:{metrics.unique_word_ratio:.2f}")
        if metrics.special_char_ratio > self.quality_config.max_special_char_ratio:
            reasons.append(f"high_special_chars:{metrics.special_char_ratio:.2f}")
        
        # Duplicate check
        if metrics.is_duplicate:
            reasons.append(f"duplicate:{metrics.duplicate_of}")
        
        # Language check
        if self.quality_config.detect_language and metrics.detected_language:
            if metrics.detected_language not in self.quality_config.allowed_languages:
                reasons.append(f"wrong_language:{metrics.detected_language}")
        
        # Toxicity check
        if metrics.toxicity_score >= self.quality_config.toxicity_threshold:
            reasons.append(f"toxic:{metrics.toxicity_score:.2f}")
        
        passed = len(reasons) == 0
        return passed, reasons
    
    def preprocess(self, item: IngestedItem) -> Optional[PreprocessedItem]:
        """
        Preprocess a single item.
        
        Args:
            item: Ingested item to preprocess
            
        Returns:
            PreprocessedItem if passes filters, None otherwise
        """
        self.stats["total_processed"] += 1
        
        # Get text content
        if isinstance(item.content, bytes):
            try:
                text = item.content.decode("utf-8", errors="ignore")
            except Exception:
                return None
        else:
            text = item.content
        
        # Normalize text
        text, has_pii = self.normalizer.normalize(
            text, 
            remove_pii=self.quality_config.remove_pii
        )
        
        if has_pii:
            self.stats["pii_found"] += 1
        
        # Generate doc ID
        doc_id = item.source.checksum or hashlib.md5(text.encode()).hexdigest()
        
        # Compute quality metrics
        metrics = self.compute_quality_metrics(text, doc_id)
        metrics.has_pii = has_pii
        
        # Check quality
        passed, reasons = self.check_quality(metrics)
        metrics.passed_filter = passed
        metrics.rejection_reasons = reasons
        
        # Update statistics
        if not passed:
            for reason in reasons:
                reason_type = reason.split(":")[0]
                self.stats["rejection_reasons"][reason_type] += 1
            
            if metrics.is_duplicate:
                self.stats["duplicates"] += 1
            elif metrics.toxicity_score >= self.quality_config.toxicity_threshold:
                self.stats["toxic_rejected"] += 1
            else:
                self.stats["quality_rejected"] += 1
            
            return None
        
        self.stats["passed"] += 1
        
        return PreprocessedItem(
            original=item,
            content=text,
            metrics=metrics,
            normalized=True,
        )
    
    def preprocess_batch(
        self, 
        items: List[IngestedItem],
        yield_rejected: bool = False,
    ) -> Generator[PreprocessedItem, None, None]:
        """
        Preprocess a batch of items.
        
        Args:
            items: List of items to preprocess
            yield_rejected: Whether to yield rejected items too
            
        Yields:
            PreprocessedItem objects
        """
        for item in items:
            try:
                result = self.preprocess(item)
                if result is not None:
                    yield result
                elif yield_rejected:
                    # Create a rejected item with metrics
                    text = item.content if isinstance(item.content, str) else ""
                    doc_id = hashlib.md5(text.encode()).hexdigest()
                    metrics = self.compute_quality_metrics(text, doc_id)
                    _, reasons = self.check_quality(metrics)
                    metrics.passed_filter = False
                    metrics.rejection_reasons = reasons
                    
                    yield PreprocessedItem(
                        original=item,
                        content=text,
                        metrics=metrics,
                        normalized=False,
                    )
            except Exception as e:
                logger.error(f"Error preprocessing item: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get preprocessing statistics."""
        stats = dict(self.stats)
        stats["rejection_reasons"] = dict(stats["rejection_reasons"])
        
        if stats["total_processed"] > 0:
            stats["pass_rate"] = stats["passed"] / stats["total_processed"]
            stats["duplicate_rate"] = stats["duplicates"] / stats["total_processed"]
        
        return stats
    
    def reset_statistics(self):
        """Reset statistics counters."""
        self.stats = {
            "total_processed": 0,
            "passed": 0,
            "duplicates": 0,
            "quality_rejected": 0,
            "pii_found": 0,
            "toxic_rejected": 0,
            "rejection_reasons": defaultdict(int),
        }
    
    def reset_deduplicator(self):
        """Reset deduplication state."""
        self.deduplicator.clear()
