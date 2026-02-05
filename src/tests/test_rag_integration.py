"""Tests for Retrieval Augmented Generation (RAG) integration."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk
from typing import Dict, List
from dataclasses import dataclass


# Test configuration
@dataclass
class RAGTestConfig:
    """Minimal config for RAG testing."""
    d_model: int = 64
    vocab_size: int = 1000
    max_seq_length: int = 128
    num_heads: int = 4
    num_layers: int = 2
    batch_size: int = 4
    moe_experts: int = 4
    moe_top_k: int = 2
    memory_size: int = 100
    retrieval_k: int = 3
    ltm_weight: float = 0.3
    stm_weight: float = 0.3
    mtm_weight: float = 0.3
    max_reasoning_steps: int = 3
    spike_threshold: float = 0.1
    EPSILON: float = 1e-8


class TestRetrievalConfig:
    """Test RetrievalConfig settings."""
    
    def test_retrieval_config_defaults(self):
        """Test default configuration values."""
        from src.config.retrieval_config import RetrievalConfig
        
        config = RetrievalConfig()
        
        assert config.enabled == False
        assert config.top_k == 5
        assert config.use_hybrid == True
        assert 0 <= config.augmentation_probability <= 1
    
    def test_retrieval_config_for_training(self):
        """Test training preset."""
        from src.config.retrieval_config import RetrievalConfig
        
        config = RetrievalConfig.for_training()
        
        assert config.enabled == True
        assert config.augmentation_probability > 0
        assert config.use_hybrid == True
    
    def test_retrieval_config_for_inference(self):
        """Test inference preset."""
        from src.config.retrieval_config import RetrievalConfig
        
        config = RetrievalConfig.for_inference()
        
        assert config.enabled == True
    
    def test_retrieval_config_disabled(self):
        """Test disabled preset."""
        from src.config.retrieval_config import RetrievalConfig
        
        config = RetrievalConfig.disabled()
        
        assert config.enabled == False
    
    @pytest.mark.parametrize("aug_prob", [0.0, 0.1, 0.3, 0.5, 1.0])
    def test_augmentation_probability_values(self, aug_prob):
        """Test various augmentation probability values."""
        from src.config.retrieval_config import RetrievalConfig
        
        config = RetrievalConfig(enabled=True, augmentation_probability=aug_prob)
        
        assert config.augmentation_probability == aug_prob


class TestHybridRetriever:
    """Test HybridRetriever functionality."""
    
    @pytest.fixture
    def retriever(self):
        from src.modules.retrieval import HybridRetriever
        return HybridRetriever(embedding_dim=64)
    
    @pytest.fixture
    def sample_documents(self):
        return [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "Neural networks are inspired by biological neurons.",
            "Deep learning uses multiple layers of neural networks.",
            "Transformers have revolutionized natural language processing.",
        ]
    
    @pytest.fixture
    def sample_embeddings(self, sample_documents):
        rng = np.random.default_rng(42)
        return rng.standard_normal((len(sample_documents), 64)).astype(np.float32)
    
    def test_add_documents(self, retriever, sample_documents, sample_embeddings):
        """Test adding documents to retriever."""
        retriever.add_documents(
            documents=sample_documents,
            embeddings=sample_embeddings,
            doc_ids=[f"doc_{i}" for i in range(len(sample_documents))]
        )
        
        assert len(retriever) == len(sample_documents)
    
    def test_search_sparse_only(self, retriever, sample_documents):
        """Test sparse-only retrieval."""
        retriever.add_documents(
            documents=sample_documents,
            doc_ids=[f"doc_{i}" for i in range(len(sample_documents))]
        )
        
        results = retriever.search("neural networks", top_k=2)
        
        assert len(results) <= 2
        assert all(hasattr(r, 'text') for r in results)
        assert all(hasattr(r, 'score') for r in results)
    
    def test_search_hybrid(self, retriever, sample_documents, sample_embeddings):
        """Test hybrid retrieval."""
        retriever.add_documents(
            documents=sample_documents,
            embeddings=sample_embeddings,
            doc_ids=[f"doc_{i}" for i in range(len(sample_documents))]
        )
        
        query_embedding = np.random.randn(64).astype(np.float32)
        
        results = retriever.search(
            "neural networks",
            query_embedding=query_embedding,
            top_k=3
        )
        
        assert len(results) <= 3
    
    def test_clear(self, retriever, sample_documents, sample_embeddings):
        """Test clearing the retriever."""
        retriever.add_documents(
            documents=sample_documents,
            embeddings=sample_embeddings
        )
        
        assert len(retriever) > 0
        
        retriever.clear()
        
        assert len(retriever) == 0


class TestDocumentIngester:
    """Test DocumentIngester functionality."""
    
    @pytest.fixture
    def ingester(self):
        from src.modules.retrieval import DocumentIngester
        return DocumentIngester(
            chunk_size=100,
            chunk_overlap=20,
            embedding_dim=64
        )
    
    @pytest.fixture
    def sample_long_document(self):
        return " ".join([
            "This is sentence number {}.".format(i) 
            for i in range(50)
        ])
    
    def test_ingest_document(self, ingester, sample_long_document):
        """Test document ingestion creates chunks."""
        docs = [{"text": sample_long_document, "doc_id": "test_doc"}]
        
        stats = ingester.ingest_documents(docs, store_to_memory=False)
        
        assert stats["documents_processed"] == 1
        assert stats["chunks_created"] > 0
    
    def test_chunk_iteration(self, ingester, sample_long_document):
        """Test iterating over chunks."""
        docs = [{"text": sample_long_document, "doc_id": "test_doc"}]
        ingester.ingest_documents(docs, store_to_memory=False)
        
        chunks = list(ingester.iter_chunks())
        
        assert len(chunks) > 0
        assert all(hasattr(c, 'text') for c in chunks)
        assert all(hasattr(c, 'chunk_id') for c in chunks)
    
    def test_embedding_function(self, ingester):
        """Test setting custom embedding function."""
        def custom_embed(texts):
            return np.random.randn(len(texts), 64).astype(np.float32)

        ingester.set_embedding_fn(custom_embed)

        assert ingester.embedding_fn is not None
class TestRetrievalAugmentedTraining:
    """Test RAG integration with training."""
    
    @pytest.fixture
    def retrieval_training(self):
        from src.config.retrieval_config import RetrievalConfig
        from src.modules.retrieval import HybridRetriever, RetrievalAugmentedTraining
        
        config = RetrievalConfig.for_training()
        config.augmentation_probability = 0.5
        retriever = HybridRetriever(embedding_dim=64)
        
        docs = ["Document one", "Document two", "Document three"]
        embeddings = np.random.randn(3, 64).astype(np.float32)
        retriever.add_documents(docs, embeddings)
        
        return RetrievalAugmentedTraining(config=config, retriever=retriever)
    
    @pytest.fixture
    def sample_batch(self):
        return {
            "input_ids": jnp.ones((4, 32), dtype=jnp.int32),
            "targets": jnp.ones((4, 32), dtype=jnp.int32),
        }
    
    def test_prepare_augmented_batch(self, retrieval_training, sample_batch):
        """Test batch augmentation."""
        rng = jax.random.PRNGKey(42)
        
        augmented = retrieval_training.prepare_augmented_batch(sample_batch, rng)
        
        assert hasattr(augmented, 'augmentation_applied')


class TestAGITrainerRAGIntegration:
    """Test RAG integration with AGITrainer."""
    
    @pytest.fixture
    def config(self):
        from src.config.agi_config import AGIConfig
        config = AGIConfig()
        config.d_model = 64
        config.vocab_size = 1000
        config.max_seq_length = 64
        config.num_heads = 4
        config.num_layers = 2
        config.moe_experts = 4
        config.moe_top_k = 2
        return config
    
    @pytest.fixture
    def sample_documents(self):
        return [
            "Artificial intelligence is transforming industries.",
            "Machine learning models require large datasets.",
            "Neural networks can learn complex patterns.",
        ]
    
    def test_configure_retrieval(self, config, sample_documents):
        """Test configuring retrieval on trainer."""
        from src.train import AGITrainer
        from src.config.retrieval_config import RetrievalConfig
        
        trainer = AGITrainer(config)
        retrieval_config = RetrievalConfig.for_training()
        retrieval_config.augmentation_probability = 0.3
        trainer.configure_retrieval(
            retrieval_config,
            documents=sample_documents
        )
        
        assert trainer.retrieval_config is not None
        assert trainer.retrieval_config.enabled == True
        assert trainer.retriever is not None
        assert trainer.document_ingester is not None
    
    def test_get_rag_statistics(self, config, sample_documents):
        """Test RAG statistics retrieval."""
        from src.train import AGITrainer
        from src.config.retrieval_config import RetrievalConfig
        
        trainer = AGITrainer(config)
        
        stats_disabled = trainer.get_rag_statistics()
        assert stats_disabled["rag_enabled"] == False
        
        retrieval_config = RetrievalConfig.for_training()
        retrieval_config.augmentation_probability = 0.3
        trainer.configure_retrieval(
            retrieval_config,
            documents=sample_documents
        )
        
        stats_enabled = trainer.get_rag_statistics()
        assert stats_enabled["rag_enabled"] == True
        assert abs(stats_enabled["augmentation_probability"] - 0.3) < 1e-6
        assert stats_enabled["total_chunks_indexed"] > 0
    
    def test_hash_text_to_embedding(self, config):
        """Test deterministic hash-based embeddings."""
        from src.train import AGITrainer
        
        trainer = AGITrainer(config)
        
        text = "test input text"
        
        emb1 = trainer._hash_text_to_embedding(text)
        emb2 = trainer._hash_text_to_embedding(text)
        
        np.testing.assert_array_equal(emb1, emb2)
        
        norm = np.linalg.norm(emb1)
        assert abs(norm - 1.0) < 1e-5
        
        assert emb1.shape == (config.d_model,)
    
    def test_simple_tokenize(self, config):
        """Test simple word-based tokenization."""
        from src.train import AGITrainer
        
        trainer = AGITrainer(config)
        
        text = "hello world this is a test"
        tokens = trainer._simple_tokenize(text, max_length=10)
        
        assert len(tokens) == 10
        assert all(0 <= t < config.vocab_size for t in tokens)
        assert tokens[-1] == 0 or len(text.split()) >= 10


class TestRAGAblationStudy:
    """Test utilities for RAG ablation studies."""
    
    @pytest.fixture
    def config(self):
        from src.config.agi_config import AGIConfig
        config = AGIConfig()
        config.d_model = 64
        config.vocab_size = 1000
        config.max_seq_length = 64
        config.num_heads = 4
        config.num_layers = 2
        return config
    
    def test_rag_on_off_comparison_structure(self, config):
        """Test that we can create trainers with and without RAG."""
        from src.train import AGITrainer
        from src.config.retrieval_config import RetrievalConfig
        
        trainer_no_rag = AGITrainer(config)
        stats_no_rag = trainer_no_rag.get_rag_statistics()
        
        trainer_with_rag = AGITrainer(config)
        trainer_with_rag.configure_retrieval(
            RetrievalConfig.for_training(),
            documents=["Sample document for testing."]
        )
        stats_with_rag = trainer_with_rag.get_rag_statistics()
        
        assert stats_no_rag["rag_enabled"] == False
        assert stats_with_rag["rag_enabled"] == True
    
    @pytest.mark.parametrize("aug_prob", [0.0, 0.1, 0.3, 0.5])
    def test_different_augmentation_probabilities(self, config, aug_prob):
        """Test RAG with different augmentation probabilities."""
        from src.train import AGITrainer
        from src.config.retrieval_config import RetrievalConfig
        
        trainer = AGITrainer(config)
        trainer.configure_retrieval(
            RetrievalConfig(enabled=True, augmentation_probability=aug_prob),
            documents=["Test document."]
        )
        
        stats = trainer.get_rag_statistics()
        assert stats["augmentation_probability"] == aug_prob


class TestModelEmbeddingIntegration:
    """Test integration between model embeddings and RAG."""
    
    @pytest.fixture
    def config(self):
        from src.config.agi_config import AGIConfig
        config = AGIConfig()
        config.d_model = 64
        config.vocab_size = 1000
        config.max_seq_length = 64
        config.num_heads = 4
        config.num_layers = 2
        config.moe_experts = 4
        config.moe_top_k = 2
        return config
    
    def test_create_model_embeddings_requires_init(self, config):
        """Test that create_model_embeddings requires initialized model."""
        from src.train import AGITrainer
        
        trainer = AGITrainer(config)
        
        token_ids = jnp.ones((2, 32), dtype=jnp.int32)
        
        with pytest.raises(RuntimeError, match="Model must be initialized"):
            trainer.create_model_embeddings(token_ids)
    
    def test_schedule_retrieval_update(self, config):
        """Test scheduling retrieval index updates."""
        from src.train import AGITrainer
        from src.config.retrieval_config import RetrievalConfig
        
        trainer = AGITrainer(config)
        trainer.configure_retrieval(
            RetrievalConfig.for_training(),
            documents=["Test doc."]
        )
        
        trainer.schedule_retrieval_update(update_every_n_epochs=3)
        
        assert hasattr(trainer, '_retrieval_update_frequency')
        assert trainer._retrieval_update_frequency == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
