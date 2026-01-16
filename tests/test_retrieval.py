"""
Tests for the retrieval augmentation module.

Tests:
- Configuration
- Document chunking
- Hybrid retrieval (sparse + dense)
- Cross-attention
- Training integration
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp


class TestRetrievalConfig:
    """Test RetrievalConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        from config.retrieval_config import RetrievalConfig
        
        config = RetrievalConfig()
        assert config.enabled is False
        assert config.top_k == 5
        assert config.use_hybrid is True
        
    def test_training_preset(self):
        """Test training preset."""
        from config.retrieval_config import RetrievalConfig
        
        config = RetrievalConfig.for_training()
        assert config.enabled is True
        assert abs(config.augmentation_probability - 0.2) < 1e-6
        assert config.use_contrastive_loss is True
        
    def test_inference_preset(self):
        """Test inference preset."""
        from config.retrieval_config import RetrievalConfig
        
        config = RetrievalConfig.for_inference()
        assert config.enabled is True
        assert abs(config.augmentation_probability - 1.0) < 1e-6
        assert config.rerank_results is True
        
    def test_config_validation(self):
        """Test configuration validation."""
        from config.retrieval_config import RetrievalConfig
        
        with pytest.raises(AssertionError):
            RetrievalConfig(top_k=0)  # Must be positive
            
        with pytest.raises(AssertionError):
            RetrievalConfig(sparse_weight=0.5, dense_weight=0.3)  # Must sum to 1


class TestDocumentChunking:
    """Test document chunking."""
    
    def test_fixed_chunking(self):
        """Test fixed-size chunking."""
        from modules.retrieval.document_ingester import TextChunker, ChunkingStrategy
        
        chunker = TextChunker(
            chunk_size=100,
            chunk_overlap=20,
            strategy=ChunkingStrategy.FIXED
        )
        
        text = "A" * 250
        chunks = chunker.chunk_text(text)
        
        assert len(chunks) >= 2
        assert all(len(c) <= 100 for c in chunks)
        
    def test_sentence_chunking(self):
        """Test sentence-based chunking."""
        from modules.retrieval.document_ingester import TextChunker, ChunkingStrategy
        
        chunker = TextChunker(
            chunk_size=50,
            chunk_overlap=10,
            strategy=ChunkingStrategy.SENTENCE
        )
        
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = chunker.chunk_text(text)
        
        assert len(chunks) >= 1
        # Sentences should not be split mid-word
        for chunk in chunks:
            assert "sentenc" not in chunk or "sentence" in chunk
            
    def test_recursive_chunking(self):
        """Test recursive text splitting."""
        from modules.retrieval.document_ingester import TextChunker, ChunkingStrategy
        
        chunker = TextChunker(
            chunk_size=50,
            chunk_overlap=10,
            strategy=ChunkingStrategy.RECURSIVE
        )
        
        text = "Paragraph one with content.\n\nParagraph two with more content.\n\nParagraph three."
        chunks = chunker.chunk_text(text)
        
        assert len(chunks) >= 1
        
    def test_sliding_window_chunking(self):
        """Test sliding window chunking."""
        from modules.retrieval.document_ingester import TextChunker, ChunkingStrategy
        
        chunker = TextChunker(
            chunk_size=50,
            chunk_overlap=25,
            strategy=ChunkingStrategy.SLIDING_WINDOW
        )
        
        text = "A" * 100
        chunks = chunker.chunk_text(text)
        
        # With 50% overlap, should have overlapping chunks
        assert len(chunks) >= 2


class TestDocumentIngester:
    """Test DocumentIngester."""
    
    def test_ingester_creation(self):
        """Test ingester creation."""
        from modules.retrieval import DocumentIngester
        
        ingester = DocumentIngester(chunk_size=100)
        assert ingester.num_chunks == 0
        assert ingester.num_documents == 0
        
    def test_document_processing(self):
        """Test processing documents without storage."""
        from modules.retrieval import DocumentIngester
        
        ingester = DocumentIngester(chunk_size=100)
        
        docs = [
            {"text": "This is document one with some content."},
            {"text": "This is document two with different content."},
        ]
        
        stats = ingester.ingest_documents(docs, store_to_memory=False)
        
        assert stats["documents_processed"] == 2
        assert stats["chunks_created"] >= 2
        assert ingester.num_documents == 2
        
    def test_document_metadata(self):
        """Test metadata preservation."""
        from modules.retrieval import DocumentIngester
        
        ingester = DocumentIngester(chunk_size=500)
        
        docs = [
            {
                "text": "Content here.",
                "source": "wiki",
                "topic": "AI",
            }
        ]
        
        ingester.ingest_documents(docs, store_to_memory=False)
        
        # Get chunks and verify metadata
        chunks = list(ingester.iter_chunks())
        assert len(chunks) >= 1
        assert chunks[0].metadata.get("source") == "wiki"


class TestSparseRetriever:
    """Test BM25 sparse retrieval."""
    
    def test_sparse_retriever_creation(self):
        """Test sparse retriever creation."""
        from modules.retrieval.hybrid_retriever import SparseRetriever
        
        retriever = SparseRetriever()
        assert len(retriever.documents) == 0
        
    def test_sparse_indexing(self):
        """Test document indexing."""
        from modules.retrieval.hybrid_retriever import SparseRetriever
        
        retriever = SparseRetriever()
        
        docs = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "Python is a programming language.",
        ]
        
        retriever.add_documents(docs)
        assert len(retriever.documents) == 3
        
    def test_sparse_search(self):
        """Test BM25 search."""
        from modules.retrieval.hybrid_retriever import SparseRetriever
        
        retriever = SparseRetriever()
        
        docs = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "Python is a programming language.",
        ]
        
        retriever.add_documents(docs)
        
        results = retriever.search("machine learning AI", top_k=2)
        
        assert len(results) == 2
        assert results[0].score >= results[1].score
        # ML doc should be top result
        assert "machine learning" in results[0].text.lower()


class TestDenseRetriever:
    """Test dense (embedding-based) retrieval."""
    
    def test_dense_retriever_creation(self):
        """Test dense retriever creation."""
        from modules.retrieval.hybrid_retriever import DenseRetriever
        
        retriever = DenseRetriever(embedding_dim=64)
        assert len(retriever.documents) == 0
        
    def test_dense_indexing(self):
        """Test document indexing with embeddings."""
        from modules.retrieval.hybrid_retriever import DenseRetriever
        
        retriever = DenseRetriever(embedding_dim=64, use_faiss=False)
        
        docs = ["Doc 1", "Doc 2", "Doc 3"]
        embeddings = np.random.randn(3, 64).astype(np.float32)
        
        retriever.add_documents(embeddings, docs)
        assert len(retriever.documents) == 3
        
    def test_dense_search(self):
        """Test embedding-based search."""
        from modules.retrieval.hybrid_retriever import DenseRetriever
        
        retriever = DenseRetriever(embedding_dim=64, use_faiss=False)
        
        # Create documents with distinct embeddings
        docs = ["Doc A", "Doc B", "Doc C"]
        embeddings = np.eye(3, 64).astype(np.float32)  # Orthogonal embeddings
        
        retriever.add_documents(embeddings, docs)
        
        # Query should match first document
        query = embeddings[0]
        results = retriever.search(query, top_k=2)
        
        assert len(results) == 2
        assert results[0].text == "Doc A"  # Should match first doc


class TestHybridRetriever:
    """Test hybrid retrieval."""
    
    def test_hybrid_retriever_creation(self):
        """Test hybrid retriever creation."""
        from modules.retrieval import HybridRetriever
        
        retriever = HybridRetriever(embedding_dim=64)
        assert retriever.num_documents == 0
        
    def test_hybrid_indexing(self):
        """Test hybrid document indexing."""
        from modules.retrieval import HybridRetriever
        
        retriever = HybridRetriever(embedding_dim=64)
        
        docs = ["Machine learning basics", "Deep learning tutorial", "Python programming"]
        embeddings = np.random.randn(3, 64).astype(np.float32)
        
        retriever.add_documents(docs, embeddings=embeddings)
        assert retriever.num_documents == 3
        
    def test_hybrid_search_sparse_only(self):
        """Test hybrid search with sparse only (no embeddings)."""
        from modules.retrieval import HybridRetriever
        
        retriever = HybridRetriever(embedding_dim=64)
        
        docs = ["Machine learning basics", "Deep learning tutorial", "Python programming"]
        retriever.add_documents(docs)  # No embeddings
        
        results = retriever.search("machine learning", top_k=2)
        
        assert len(results) == 2
        # Results are still "hybrid" source even if only sparse contributed
        assert results[0].source in ("sparse", "hybrid")
        
    def test_hybrid_search_combined(self):
        """Test hybrid search with both sparse and dense."""
        from modules.retrieval import HybridRetriever
        
        retriever = HybridRetriever(embedding_dim=64)
        
        docs = ["Machine learning basics", "Deep learning tutorial", "Python programming"]
        embeddings = np.random.randn(3, 64).astype(np.float32)
        
        retriever.add_documents(docs, embeddings=embeddings)
        
        query_embedding = embeddings[0]  # Similar to first doc
        results = retriever.search("machine learning", query_embedding, top_k=2)
        
        assert len(results) == 2
        assert results[0].source == "hybrid"  # Combined results


class TestRetrievalAugmentedAttention:
    """Test cross-attention for retrieval."""
    
    def test_cross_attention_output_shape(self):
        """Test cross-attention output shape."""
        from modules.retrieval.augmented_attention import CrossAttentionRetrieval
        import haiku as hk
        
        def forward(query, retrieved):
            cross_attn = CrossAttentionRetrieval(
                d_model=64,
                num_heads=4,
            )
            return cross_attn(query, retrieved)
        
        forward_fn = hk.transform(forward)
        
        rng = jax.random.PRNGKey(0)
        query = jnp.ones((2, 10, 64))  # [batch, seq, d_model]
        retrieved = jnp.ones((2, 5, 64))  # [batch, num_retrieved, d_model]
        
        params = forward_fn.init(rng, query, retrieved)
        output = forward_fn.apply(params, rng, query, retrieved)
        
        assert output.shape == query.shape
        
    def test_retrieval_augmented_attention(self):
        """Test full retrieval-augmented attention block."""
        from modules.retrieval.augmented_attention import RetrievalAugmentedAttention
        import haiku as hk
        
        def forward(x, retrieved):
            block = RetrievalAugmentedAttention(
                d_model=64,
                num_heads=4,
            )
            return block(x, retrieved)
        
        forward_fn = hk.transform(forward)
        
        rng = jax.random.PRNGKey(0)
        x = jnp.ones((2, 10, 64))
        retrieved = jnp.ones((2, 5, 64))
        
        params = forward_fn.init(rng, x, retrieved)
        output, aux = forward_fn.apply(params, rng, x, retrieved)
        
        assert output.shape == x.shape
        assert "retrieval_gate" in aux


class TestContrastiveLoss:
    """Test contrastive loss for retrieval."""
    
    def test_contrastive_loss_basic(self):
        """Test basic contrastive loss computation."""
        from modules.retrieval.training_integration import RetrievalContrastiveLoss
        
        loss_fn = RetrievalContrastiveLoss()
        
        # Create embeddings
        query = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        positive = jnp.array([[1.0, 0.0], [0.0, 1.0]])  # Same as query
        
        loss, metrics = loss_fn(query, positive)
        
        assert loss >= 0
        assert "positive_similarity" in metrics
        
    def test_contrastive_loss_with_negatives(self):
        """Test contrastive loss with negative samples."""
        from modules.retrieval.training_integration import RetrievalContrastiveLoss
        
        loss_fn = RetrievalContrastiveLoss()
        
        query = jnp.array([[1.0, 0.0]])
        positive = jnp.array([[0.9, 0.1]])  # Similar
        negative = jnp.array([[[0.0, 1.0], [-1.0, 0.0]]])  # Dissimilar
        
        loss, metrics = loss_fn(query, positive, negative)
        
        assert loss >= 0
        assert "negative_similarity" in metrics
        assert metrics["positive_similarity"] > metrics["negative_similarity"]


class TestTrainingIntegration:
    """Test training integration utilities."""
    
    def test_augmented_batch_creation(self):
        """Test creating augmented batches."""
        from modules.retrieval.training_integration import RetrievalAugmentedTraining
        from config.retrieval_config import RetrievalConfig
        
        config = RetrievalConfig(
            enabled=False,  # Disabled for this test
        )
        
        training = RetrievalAugmentedTraining(config)
        
        batch = {"input_ids": jnp.ones((4, 32), dtype=jnp.int32)}
        rng = jax.random.PRNGKey(0)
        
        augmented = training.prepare_augmented_batch(batch, rng)
        
        assert not augmented.augmentation_applied
        assert augmented.input_ids.shape == (4, 32)
        
    def test_augmentation_probability(self):
        """Test that augmentation respects probability setting."""
        from modules.retrieval.training_integration import RetrievalAugmentedTraining
        from config.retrieval_config import RetrievalConfig
        from modules.retrieval import HybridRetriever
        
        config = RetrievalConfig(
            enabled=True,
            augmentation_probability=0.0,  # Never augment
        )
        
        retriever = HybridRetriever(embedding_dim=64)
        retriever.add_documents(["Test doc"])
        
        training = RetrievalAugmentedTraining(config, retriever=retriever)
        
        # Should never augment with 0 probability
        for i in range(10):
            rng = jax.random.PRNGKey(i)
            # With 0 probability, should_augment returns False
            assert not training.should_augment(rng)
            
    def test_retrieval_stats(self):
        """Test training statistics tracking."""
        from modules.retrieval.training_integration import RetrievalAugmentedTraining
        from config.retrieval_config import RetrievalConfig
        
        config = RetrievalConfig(enabled=False)
        training = RetrievalAugmentedTraining(config)
        
        # Simulate some batches
        for i in range(5):
            batch = {"input_ids": jnp.ones((4, 32), dtype=jnp.int32)}
            training.prepare_augmented_batch(batch, jax.random.PRNGKey(i))
            
        stats = training.get_stats()
        assert stats["total_batches"] == 5
        assert stats["augmentation_rate"] < 0.01  # Disabled, should be ~0


class TestIntegration:
    """Integration tests."""
    
    def test_full_pipeline(self):
        """Test full retrieval pipeline."""
        from modules.retrieval import (
            DocumentIngester,
            HybridRetriever,
        )
        
        # Create components
        ingester = DocumentIngester(chunk_size=100)
        retriever = HybridRetriever(embedding_dim=64)
        
        # Ingest documents
        docs = [
            {"text": "Machine learning is a field of AI."},
            {"text": "Deep learning uses neural networks."},
            {"text": "Python is popular for data science."},
        ]
        ingester.ingest_documents(docs, store_to_memory=False)
        
        # Add to retriever
        chunks = list(ingester.iter_chunks())
        chunk_texts = [c.text for c in chunks]
        embeddings = np.random.randn(len(chunks), 64).astype(np.float32)
        
        retriever.add_documents(chunk_texts, embeddings=embeddings)
        
        # Search
        query_emb = embeddings[0]
        results = retriever.search("machine learning", query_emb, top_k=2)
        
        assert len(results) == 2
        assert all(hasattr(r, "text") for r in results)
        assert all(hasattr(r, "score") for r in results)
