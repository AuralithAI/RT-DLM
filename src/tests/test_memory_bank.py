"""
Tests for Memory Bank Module

Tests for memory storage, retrieval, forgetting curves,
and contextual memory indexing with FAISS.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import time


class TestMemoryItem(unittest.TestCase):
    """Test MemoryItem dataclass."""
    
    def test_memory_item_creation(self):
        """Test creating memory items."""
        from src.core.model.memory_bank import MemoryItem
        
        item = MemoryItem(
            key=np.ones(64),
            value=np.ones(128),
            timestamp=time.time(),
            access_count=0,
            importance_score=0.8,
            context_tags=["test", "example"],
            emotional_valence=0.5,
            consolidation_level=0
        )
        
        self.assertEqual(item.key.shape, (64,))
        self.assertEqual(item.value.shape, (128,))
        self.assertEqual(item.importance_score, 0.8)
        self.assertEqual(len(item.context_tags), 2)
        self.assertEqual(item.consolidation_level, 0)


class TestAdaptiveForgettingCurve(unittest.TestCase):
    """Test AdaptiveForgettingCurve class."""
    
    def setUp(self):
        """Set up test fixtures."""
        from src.core.model.memory_bank import AdaptiveForgettingCurve
        self.curve = AdaptiveForgettingCurve(base_decay=0.1, importance_weight=0.5)
    
    def test_curve_initialization(self):
        """Test forgetting curve initialization."""
        self.assertEqual(self.curve.base_decay, 0.1)
        self.assertEqual(self.curve.importance_weight, 0.5)
    
    def test_retention_probability_decay(self):
        """Test that retention probability decays with time."""
        from src.core.model.memory_bank import MemoryItem
        
        old_time = time.time() - 1000
        recent_time = time.time() - 10
        
        old_item = MemoryItem(
            key=np.zeros(64),
            value=np.zeros(64),
            timestamp=old_time,
            access_count=1,
            importance_score=0.5,
            context_tags=[],
            emotional_valence=0.0,
            consolidation_level=0
        )
        
        recent_item = MemoryItem(
            key=np.zeros(64),
            value=np.zeros(64),
            timestamp=recent_time,
            access_count=1,
            importance_score=0.5,
            context_tags=[],
            emotional_valence=0.0,
            consolidation_level=0
        )
        
        current_time = time.time()
        old_retention = self.curve.calculate_retention_probability(old_item, current_time)
        recent_retention = self.curve.calculate_retention_probability(recent_item, current_time)
        
        # Older memories should have lower retention
        self.assertLess(old_retention, recent_retention)
    
    def test_importance_boosts_retention(self):
        """Test that importance boosts retention probability."""
        from src.core.model.memory_bank import MemoryItem
        
        base_time = time.time() - 100
        
        low_importance = MemoryItem(
            key=np.zeros(64),
            value=np.zeros(64),
            timestamp=base_time,
            access_count=1,
            importance_score=0.1,
            context_tags=[],
            emotional_valence=0.0,
            consolidation_level=0
        )
        
        high_importance = MemoryItem(
            key=np.zeros(64),
            value=np.zeros(64),
            timestamp=base_time,
            access_count=1,
            importance_score=0.9,
            context_tags=[],
            emotional_valence=0.0,
            consolidation_level=0
        )
        
        current_time = time.time()
        low_retention = self.curve.calculate_retention_probability(low_importance, current_time)
        high_retention = self.curve.calculate_retention_probability(high_importance, current_time)
        
        self.assertLess(low_retention, high_retention)
    
    def test_access_count_boosts_retention(self):
        """Test that higher access count boosts retention."""
        from src.core.model.memory_bank import MemoryItem
        
        base_time = time.time() - 100
        
        rarely_accessed = MemoryItem(
            key=np.zeros(64),
            value=np.zeros(64),
            timestamp=base_time,
            access_count=1,
            importance_score=0.5,
            context_tags=[],
            emotional_valence=0.0,
            consolidation_level=0
        )
        
        frequently_accessed = MemoryItem(
            key=np.zeros(64),
            value=np.zeros(64),
            timestamp=base_time,
            access_count=100,
            importance_score=0.5,
            context_tags=[],
            emotional_valence=0.0,
            consolidation_level=0
        )
        
        current_time = time.time()
        rare_retention = self.curve.calculate_retention_probability(rarely_accessed, current_time)
        freq_retention = self.curve.calculate_retention_probability(frequently_accessed, current_time)
        
        self.assertLess(rare_retention, freq_retention)


class TestContextualMemoryIndex(unittest.TestCase):
    """Test ContextualMemoryIndex class."""
    
    def test_index_initialization(self):
        """Test memory index initialization."""
        from src.core.model.memory_bank import ContextualMemoryIndex
        
        index = ContextualMemoryIndex(embedding_dim=64, num_clusters=8)
        
        self.assertEqual(index.embedding_dim, 64)
        self.assertEqual(index.num_clusters, 8)


class TestSecurityStubs(unittest.TestCase):
    """Test security module stubs."""
    
    def test_data_sanitizer_stub(self):
        """Test DataSanitizer stub or real implementation."""
        try:
            from src.core.model.memory_bank import DataSanitizer
            
            sanitizer = DataSanitizer()
            result = sanitizer.sanitize("test text")
            
            self.assertIsInstance(result, str)
        except ImportError:
            self.skipTest("DataSanitizer not available")
    
    def test_secure_storage_stub(self):
        """Test SecureStorage stub or real implementation."""
        try:
            from src.core.model.memory_bank import SecureStorage
            
            storage = SecureStorage()
            
            # Test encrypt/decrypt
            data = "test data"
            encrypted = storage.encrypt(data)
            
            self.assertIsNotNone(encrypted)
        except ImportError:
            self.skipTest("SecureStorage not available")
    
    def test_identifier_hasher_stub(self):
        """Test IdentifierHasher stub or real implementation."""
        try:
            from src.core.model.memory_bank import IdentifierHasher
            
            hasher = IdentifierHasher()
            hashed = hasher.hash("user123")
            
            self.assertIsInstance(hashed, str)
            # Hash should be deterministic
            self.assertEqual(hashed, hasher.hash("user123"))
        except ImportError:
            self.skipTest("IdentifierHasher not available")


class TestMemoryBankIntegration(unittest.TestCase):
    """Integration tests for memory bank functionality."""
    
    def test_memory_lifecycle(self):
        """Test complete memory lifecycle."""
        from src.core.model.memory_bank import MemoryItem, AdaptiveForgettingCurve
        
        curve = AdaptiveForgettingCurve()
        
        # Create memory items with different characteristics
        memories = []
        for i in range(5):
            item = MemoryItem(
                key=np.random.randn(64),
                value=np.random.randn(128),
                timestamp=time.time() - i * 100,
                access_count=i + 1,
                importance_score=0.5 + i * 0.1,
                context_tags=[f"tag_{i}"],
                emotional_valence=0.0,
                consolidation_level=i % 3
            )
            memories.append(item)
        
        # Calculate retention for all
        current_time = time.time()
        retentions = [
            curve.calculate_retention_probability(m, current_time)
            for m in memories
        ]
        
        # All should be valid probabilities
        for r in retentions:
            self.assertGreaterEqual(r, 0.0)
            self.assertLessEqual(r, 1.0)


if __name__ == "__main__":
    unittest.main()
