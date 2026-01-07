"""
Tests for PersistentLTMStorage in TMS_block/memory_bank.py

Tests cover:
1. Basic store/retrieve operations
2. Save/load persistence cycles
3. Statistics and clear operations
4. Edge cases and error handling
"""

import os
import sys
import tempfile
import shutil
import unittest
from pathlib import Path

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from TMS_block.memory_bank import PersistentLTMStorage, MemoryBank


class TestPersistentLTMStorage(unittest.TestCase):
    """Test suite for PersistentLTMStorage class."""

    def setUp(self):
        """Create a temporary directory for test storage."""
        self.test_dir = tempfile.mkdtemp(prefix="ltm_test_")
        self.d_model = 64
        self.storage = PersistentLTMStorage(
            d_model=self.d_model,
            storage_dir=self.test_dir
        )

    def tearDown(self):
        """Clean up temporary directory after tests."""
        if hasattr(self, 'test_dir') and os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_initialization(self):
        """Test that storage initializes correctly."""
        self.assertIsNotNone(self.storage)
        self.assertEqual(self.storage.d_model, self.d_model)
        self.assertEqual(len(self.storage.embeddings), 0)
        self.assertEqual(self.storage.ltm_index.ntotal, 0)

    def test_store_single_memory(self):
        """Test storing a single memory entry."""
        embedding = np.random.randn(self.d_model).astype(np.float32)

        memory_id = self.storage.store_ltm(
            embedding,
            context="test memory",
            importance_score=0.8,
            emotional_valence=0.5
        )

        self.assertIsNotNone(memory_id)
        self.assertEqual(len(self.storage.embeddings), 1)
        stats = self.storage.get_stats()
        self.assertEqual(stats["total_entries"], 1)

    def test_store_multiple_memories(self):
        """Test storing multiple memory entries."""
        num_memories = 10

        for i in range(num_memories):
            embedding = np.random.randn(self.d_model).astype(np.float32)
            self.storage.store_ltm(
                embedding,
                importance_score=i / num_memories,
                context=f"tag_{i}",
                emotional_valence=np.sin(i)
            )

        stats = self.storage.get_stats()
        self.assertEqual(stats["total_entries"], num_memories)

    def test_retrieve_similar_memories(self):
        """Test retrieving memories by similarity search."""
        # Store some memories with known embeddings
        base_embedding = np.random.randn(self.d_model).astype(np.float32)

        # Store the original
        self.storage.store_ltm(
            base_embedding,
            context="original",
            importance_score=1.0
        )

        # Store a similar one (slightly perturbed)
        similar = base_embedding + np.random.randn(self.d_model).astype(np.float32) * 0.1
        self.storage.store_ltm(
            similar,
            context="similar",
            importance_score=0.9
        )

        # Store some random ones
        for _ in range(5):
            random_emb = np.random.randn(self.d_model).astype(np.float32)
            self.storage.store_ltm(
                random_emb,
                context="random",
                importance_score=0.5
            )

        # Query with original - should return similar memories first
        results = self.storage.retrieve_ltm(base_embedding, k=2)

        self.assertEqual(len(results), 2)
        # First result should be the original (exact match)
        self.assertEqual(results[0][1]["context"], "original")

    def test_retrieve_with_filters(self):
        """Test retrieving memories with importance filtering."""
        # Store memories with varying importance
        for importance in [0.1, 0.3, 0.5, 0.7, 0.9]:
            embedding = np.random.randn(self.d_model).astype(np.float32)
            self.storage.store_ltm(
                embedding,
                importance_score=importance,
                context=f"imp_{importance}"
            )

        # Query with min_importance filter
        query = np.random.randn(self.d_model).astype(np.float32)
        results = self.storage.retrieve_ltm(query, k=10, min_importance=0.6)

        # Should only return memories with importance >= 0.6
        for _, metadata, _ in results:
            self.assertGreaterEqual(metadata["importance_score"], 0.6)

    def test_save_and_load_cycle(self):
        """Test that memories persist across save/load cycles."""
        # Store some memories
        stored_ids = []
        for i in range(5):
            embedding = np.random.randn(self.d_model).astype(np.float32)
            memory_id = self.storage.store_ltm(
                embedding,
                importance_score=0.5 + i * 0.1,
                context=f"persistent_{i}",
                emotional_valence=i * 0.2
            )
            stored_ids.append(memory_id)

        # Save to disk
        self.storage.save()

        # Create a new storage instance and load
        new_storage = PersistentLTMStorage(
            d_model=self.d_model,
            storage_dir=self.test_dir
        )
        new_storage.load()

        # Verify memories are restored
        stats = new_storage.get_stats()
        self.assertEqual(stats["total_entries"], 5)

        # Verify embeddings are preserved
        self.assertEqual(len(new_storage.embeddings), 5)

    def test_persistence_across_sessions(self):
        """Test that FAISS index and embeddings persist correctly."""
        # Store a known embedding
        known_embedding = np.ones(self.d_model, dtype=np.float32) * 0.5
        self.storage.store_ltm(
            known_embedding,
            context="known",
            importance_score=1.0
        )
        self.storage.save()

        # Simulate new session
        del self.storage

        new_storage = PersistentLTMStorage(
            d_model=self.d_model,
            storage_dir=self.test_dir
        )
        new_storage.load()

        # Query with the known embedding - should find exact match
        results = new_storage.retrieve_ltm(known_embedding, k=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][1]["context"], "known")
        # Distance should be very small (near 0) for exact match
        self.assertLess(results[0][2], 0.001)

    def test_access_count_tracking(self):
        """Test that access counts are updated on retrieval."""
        embedding = np.random.randn(self.d_model).astype(np.float32)
        self.storage.store_ltm(
            embedding,
            context="access_test",
            importance_score=1.0
        )

        # Retrieve multiple times
        for _ in range(3):
            self.storage.retrieve_ltm(embedding, k=1)

        # Check access count in results
        results = self.storage.retrieve_ltm(embedding, k=1)
        # Access count should be >= 3 (might be 4 since this is also a retrieval)
        self.assertGreaterEqual(results[0][1].get("access_count", 0), 3)

    def test_statistics(self):
        """Test get_stats method."""
        # Empty storage
        stats = self.storage.get_stats()
        self.assertEqual(stats["total_entries"], 0)

        # Add some memories
        for i in range(3):
            embedding = np.random.randn(self.d_model).astype(np.float32)
            self.storage.store_ltm(
                embedding,
                importance_score=0.5 + i * 0.2,
                context=f"stat_{i}"
            )

        stats = self.storage.get_stats()
        self.assertEqual(stats["total_entries"], 3)
        self.assertIn("avg_importance", stats)
        self.assertIn("storage_dir", stats)

    def test_clear_all(self):
        """Test clearing all memories."""
        # Store some memories
        for _ in range(5):
            embedding = np.random.randn(self.d_model).astype(np.float32)
            self.storage.store_ltm(embedding, importance_score=0.5)

        self.assertEqual(self.storage.get_stats()["total_entries"], 5)

        # Clear all
        self.storage.clear()

        # Verify empty
        stats = self.storage.get_stats()
        self.assertEqual(stats["total_entries"], 0)
        self.assertEqual(len(self.storage.embeddings), 0)
        self.assertEqual(self.storage.ltm_index.ntotal, 0)

    def test_empty_query(self):
        """Test querying empty storage."""
        query = np.random.randn(self.d_model).astype(np.float32)
        results = self.storage.retrieve_ltm(query, k=5)

        self.assertEqual(len(results), 0)

    def test_dimension_validation(self):
        """Test that embedding dimensions are validated."""
        # Wrong dimension should raise an error or be handled gracefully
        wrong_dim_embedding = np.random.randn(self.d_model * 2).astype(np.float32)

        with self.assertRaises((ValueError, AssertionError, Exception)):
            self.storage.store_ltm(wrong_dim_embedding, importance_score=0.5)


class TestMemoryBankWithPersistence(unittest.TestCase):
    """Test MemoryBank integration with PersistentLTMStorage."""

    def setUp(self):
        """Create a temporary directory and initialize MemoryBank with persistence."""
        self.test_dir = tempfile.mkdtemp(prefix="memory_bank_test_")
        self.d_model = 64

        # Create MemoryBank with persistent storage enabled
        self.memory_bank = MemoryBank(
            memory_size=256,
            embedding_dim=self.d_model,
            retrieval_k=5,
            enable_persistent_ltm=True,
            ltm_storage_dir=self.test_dir
        )

    def tearDown(self):
        """Clean up temporary directory after tests."""
        if hasattr(self, 'test_dir') and os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_memory_bank_with_persistence_enabled(self):
        """Test that MemoryBank correctly uses persistent storage."""
        self.assertTrue(self.memory_bank.enable_persistent_ltm)
        self.assertIsNotNone(self.memory_bank.persistent_ltm)

    def test_store_to_ltm_via_memory_bank(self):
        """Test storing to LTM through MemoryBank interface."""
        embedding = np.random.randn(self.d_model).astype(np.float32)

        memory_id = self.memory_bank.store_ltm(
            embedding,
            context="via_bank",
            importance_score=0.9
        )

        self.assertIsNotNone(memory_id)
        stats = self.memory_bank.persistent_ltm.get_stats()
        self.assertEqual(stats["total_entries"], 1)

    def test_retrieve_from_ltm_via_memory_bank(self):
        """Test retrieving from LTM through MemoryBank interface."""
        # Store a memory
        embedding = np.random.randn(self.d_model).astype(np.float32)
        self.memory_bank.store_ltm(
            embedding,
            context="retrieve_test",
            importance_score=0.9
        )

        # Retrieve it
        results = self.memory_bank.retrieve_ltm(embedding, k=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][1]["context"], "retrieve_test")

    def test_save_and_load_via_memory_bank(self):
        """Test save/load cycle through MemoryBank interface."""
        # Store memories
        for i in range(3):
            embedding = np.random.randn(self.d_model).astype(np.float32)
            self.memory_bank.store_ltm(
                embedding,
                context=f"bank_persist_{i}",
                importance_score=0.8
            )

        # Save
        self.memory_bank.persistent_ltm.save()

        # Create new MemoryBank with same storage path
        new_memory_bank = MemoryBank(
            memory_size=256,
            embedding_dim=self.d_model,
            retrieval_k=5,
            enable_persistent_ltm=True,
            ltm_storage_dir=self.test_dir
        )

        # Load (should auto-load on init, but call explicitly)
        new_memory_bank.persistent_ltm.load()

        # Verify
        stats = new_memory_bank.persistent_ltm.get_stats()
        self.assertEqual(stats["total_entries"], 3)

    def test_memory_bank_without_persistence(self):
        """Test that MemoryBank works without persistent storage."""
        non_persistent_bank = MemoryBank(
            memory_size=256,
            embedding_dim=self.d_model,
            retrieval_k=5,
            enable_persistent_ltm=False
        )

        self.assertFalse(non_persistent_bank.enable_persistent_ltm)

        # Store/retrieve should return None or empty
        embedding = np.random.randn(self.d_model).astype(np.float32)
        result = non_persistent_bank.store_ltm(embedding, importance_score=0.5)
        self.assertIsNone(result)

        results = non_persistent_bank.retrieve_ltm(embedding)
        self.assertEqual(len(results), 0)


class TestPersistenceEdgeCases(unittest.TestCase):
    """Test edge cases for persistent storage."""

    def setUp(self):
        """Create a temporary directory for test storage."""
        self.test_dir = tempfile.mkdtemp(prefix="ltm_edge_test_")
        self.d_model = 64

    def tearDown(self):
        """Clean up temporary directory after tests."""
        if hasattr(self, 'test_dir') and os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_load_without_save(self):
        """Test loading from empty/non-existent storage."""
        storage = PersistentLTMStorage(
            d_model=self.d_model,
            storage_dir=self.test_dir
        )

        # Should not raise error, just be empty
        storage.load()

        stats = storage.get_stats()
        self.assertEqual(stats["total_entries"], 0)

    def test_large_number_of_memories(self):
        """Test storage with many memories."""
        storage = PersistentLTMStorage(
            d_model=self.d_model,
            storage_dir=self.test_dir
        )

        num_memories = 100
        for i in range(num_memories):
            embedding = np.random.randn(self.d_model).astype(np.float32)
            storage.store_ltm(
                embedding,
                importance_score=np.random.random(),
                context=f"large_{i}"
            )

        stats = storage.get_stats()
        self.assertEqual(stats["total_entries"], num_memories)

        # Test retrieval performance
        query = np.random.randn(self.d_model).astype(np.float32)
        results = storage.retrieve_ltm(query, k=10)
        self.assertEqual(len(results), 10)

    def test_unicode_metadata(self):
        """Test storing metadata with unicode characters."""
        storage = PersistentLTMStorage(
            d_model=self.d_model,
            storage_dir=self.test_dir
        )

        embedding = np.random.randn(self.d_model).astype(np.float32)
        metadata = {
            "tags": ["日本語", "中文", "한국어", "🎉"],
            "description": "Unicode test: αβγδ"
        }

        memory_id = storage.store_ltm(
            embedding,
            metadata=metadata,
            importance_score=0.8,
            context="unicode_test"
        )
        self.assertIsNotNone(memory_id)

        # Save and reload
        storage.save()

        new_storage = PersistentLTMStorage(
            d_model=self.d_model,
            storage_dir=self.test_dir
        )
        new_storage.load()

        results = new_storage.retrieve_ltm(embedding, k=1)
        self.assertEqual(len(results), 1)

    def test_repeated_save_cycles(self):
        """Test multiple save/load cycles don't corrupt data."""
        storage = PersistentLTMStorage(
            d_model=self.d_model,
            storage_dir=self.test_dir
        )

        # First cycle
        for _ in range(5):
            embedding = np.random.randn(self.d_model).astype(np.float32)
            storage.store_ltm(embedding, importance_score=0.5, context="cycle_1")
        storage.save()

        # Second cycle - add more
        storage.load()
        for _ in range(5):
            embedding = np.random.randn(self.d_model).astype(np.float32)
            storage.store_ltm(embedding, importance_score=0.6, context="cycle_2")
        storage.save()

        # Third cycle - verify
        final_storage = PersistentLTMStorage(
            d_model=self.d_model,
            storage_dir=self.test_dir
        )
        final_storage.load()

        stats = final_storage.get_stats()
        self.assertEqual(stats["total_entries"], 10)


class TestSecurityFeatures(unittest.TestCase):
    """Test suite for encryption and PII scrubbing features."""

    def setUp(self):
        """Create a temporary directory for test storage."""
        self.test_dir = tempfile.mkdtemp(prefix="ltm_security_test_")
        self.d_model = 64

    def tearDown(self):
        """Clean up temporary directory after tests."""
        if hasattr(self, 'test_dir') and os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_pii_scrubbing_email(self):
        """Test that email addresses are scrubbed from context."""
        storage = PersistentLTMStorage(
            d_model=self.d_model,
            storage_dir=self.test_dir,
            enable_pii_scrubbing=True
        )

        embedding = np.random.randn(self.d_model).astype(np.float32)
        storage.store_ltm(
            embedding,
            context="Contact john.doe@example.com for more info",
            importance_score=0.8
        )

        # Retrieve and check context is scrubbed
        results = storage.retrieve_ltm(embedding, k=1)
        self.assertEqual(len(results), 1)
        context = results[0][1]["context"]
        self.assertNotIn("john.doe@example.com", context)
        self.assertIn("[EMAIL_REDACTED]", context)

    def test_pii_scrubbing_phone(self):
        """Test that phone numbers are scrubbed from context."""
        storage = PersistentLTMStorage(
            d_model=self.d_model,
            storage_dir=self.test_dir,
            enable_pii_scrubbing=True
        )

        embedding = np.random.randn(self.d_model).astype(np.float32)
        storage.store_ltm(
            embedding,
            context="Call me at 555-123-4567",
            importance_score=0.8
        )

        results = storage.retrieve_ltm(embedding, k=1)
        context = results[0][1]["context"]
        self.assertNotIn("555-123-4567", context)
        self.assertIn("[PHONE_REDACTED]", context)

    def test_pii_scrubbing_ssn(self):
        """Test that SSNs are scrubbed from context."""
        storage = PersistentLTMStorage(
            d_model=self.d_model,
            storage_dir=self.test_dir,
            enable_pii_scrubbing=True
        )

        embedding = np.random.randn(self.d_model).astype(np.float32)
        storage.store_ltm(
            embedding,
            context="SSN is 123-45-6789",
            importance_score=0.8
        )

        results = storage.retrieve_ltm(embedding, k=1)
        context = results[0][1]["context"]
        self.assertNotIn("123-45-6789", context)
        self.assertIn("[SSN_REDACTED]", context)

    def test_pii_scrubbing_metadata(self):
        """Test that PII is scrubbed from metadata dictionary."""
        storage = PersistentLTMStorage(
            d_model=self.d_model,
            storage_dir=self.test_dir,
            enable_pii_scrubbing=True
        )

        embedding = np.random.randn(self.d_model).astype(np.float32)
        storage.store_ltm(
            embedding,
            metadata={"email": "test@example.com", "notes": "Call 555-000-1234"},
            context="test",
            importance_score=0.8
        )

        results = storage.retrieve_ltm(embedding, k=1)
        extra = results[0][1]["extra"]
        self.assertNotIn("test@example.com", str(extra))
        self.assertNotIn("555-000-1234", str(extra))

    def test_identifier_hashing(self):
        """Test that user_id and session_id are hashed."""
        storage = PersistentLTMStorage(
            d_model=self.d_model,
            storage_dir=self.test_dir
        )

        embedding = np.random.randn(self.d_model).astype(np.float32)
        storage.store_ltm(
            embedding,
            context="test",
            session_id="my_session_123",
            user_id="user_john_doe",
            importance_score=0.8
        )

        results = storage.retrieve_ltm(embedding, k=1)
        # IDs should be hashed, not stored as plain text
        self.assertNotEqual(results[0][1]["session_id"], "my_session_123")
        self.assertNotEqual(results[0][1]["user_id"], "user_john_doe")
        # Should be 32-char hex hash
        self.assertEqual(len(results[0][1]["session_id"]), 32)
        self.assertEqual(len(results[0][1]["user_id"]), 32)

    def test_filter_by_hashed_session(self):
        """Test that session filtering works with hashed IDs."""
        storage = PersistentLTMStorage(
            d_model=self.d_model,
            storage_dir=self.test_dir
        )

        # Store memories with different sessions
        for session in ["session_a", "session_b"]:
            embedding = np.random.randn(self.d_model).astype(np.float32)
            storage.store_ltm(
                embedding,
                context=f"Memory from {session}",
                session_id=session,
                importance_score=0.8
            )

        # Query with session filter - should work with original ID
        query = np.random.randn(self.d_model).astype(np.float32)
        results = storage.retrieve_ltm(query, k=10, session_id="session_a")
        
        # All results should be from session_a (hashed)
        for _, metadata, _ in results:
            # Verify same hashed value
            self.assertEqual(
                metadata["session_id"],
                storage._hash_id("session_a")
            )

    def test_encryption_enabled(self):
        """Test that encryption works when enabled."""
        storage = PersistentLTMStorage(
            d_model=self.d_model,
            storage_dir=self.test_dir,
            encryption_key="my_secret_password_123"
        )

        embedding = np.random.randn(self.d_model).astype(np.float32)
        storage.store_ltm(
            embedding,
            context="This is sensitive data",
            metadata={"secret": "classified_info"},
            importance_score=0.8
        )

        # Retrieve and verify decryption works
        results = storage.retrieve_ltm(embedding, k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][1]["context"], "This is sensitive data")
        self.assertEqual(results[0][1]["extra"]["secret"], "classified_info")

    def test_encryption_persists_across_sessions(self):
        """Test that encrypted data can be read after restart."""
        encryption_key = "persistent_secret_key"
        
        # Create and store
        storage1 = PersistentLTMStorage(
            d_model=self.d_model,
            storage_dir=self.test_dir,
            encryption_key=encryption_key
        )
        
        embedding = np.random.randn(self.d_model).astype(np.float32)
        storage1.store_ltm(
            embedding,
            context="Encrypted persistent data",
            importance_score=0.9
        )
        storage1.save()
        
        # Simulate restart with new instance
        del storage1
        
        storage2 = PersistentLTMStorage(
            d_model=self.d_model,
            storage_dir=self.test_dir,
            encryption_key=encryption_key
        )
        storage2.load()
        
        # Verify can decrypt
        results = storage2.retrieve_ltm(embedding, k=1)
        self.assertEqual(results[0][1]["context"], "Encrypted persistent data")

    def test_wrong_encryption_key_fails_gracefully(self):
        """Test that wrong key returns encrypted data, not crash."""
        # Store with one key
        storage1 = PersistentLTMStorage(
            d_model=self.d_model,
            storage_dir=self.test_dir,
            encryption_key="correct_key"
        )
        
        embedding = np.random.randn(self.d_model).astype(np.float32)
        storage1.store_ltm(
            embedding,
            context="Secret message",
            importance_score=0.9
        )
        storage1.save()
        del storage1
        
        # Try to read with wrong key - should not crash
        storage2 = PersistentLTMStorage(
            d_model=self.d_model,
            storage_dir=self.test_dir,
            encryption_key="wrong_key"
        )
        storage2.load()
        
        # Should not crash, but context won't be correctly decrypted
        results = storage2.retrieve_ltm(embedding, k=1)
        self.assertEqual(len(results), 1)
        # Context won't match original (decryption fails gracefully)
        self.assertNotEqual(results[0][1]["context"], "Secret message")

    def test_pii_scrubbing_disabled(self):
        """Test that PII is preserved when scrubbing is disabled."""
        storage = PersistentLTMStorage(
            d_model=self.d_model,
            storage_dir=self.test_dir,
            enable_pii_scrubbing=False
        )

        embedding = np.random.randn(self.d_model).astype(np.float32)
        email = "test@example.com"
        storage.store_ltm(
            embedding,
            context=f"Contact {email}",
            importance_score=0.8
        )

        results = storage.retrieve_ltm(embedding, k=1)
        context = results[0][1]["context"]
        # Email should be preserved when scrubbing is disabled
        self.assertIn(email, context)


if __name__ == "__main__":
    # Run tests with verbosity
    unittest.main(verbosity=2)