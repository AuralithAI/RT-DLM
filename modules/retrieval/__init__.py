"""
RT-DLM Retrieval Augmentation Module

Production-grade retrieval-augmented generation (RAG) capabilities that
integrate with the existing memory systems rather than replacing them.

Architecture follows industry best practices:
- Retrieval is optional and external to core model weights
- Hybrid dense + sparse retrieval for best quality
- Seamless integration with existing MemoryBank infrastructure
- Cross-attention based augmentation (differentiable)

Usage:
    from modules.retrieval import (
        DocumentIngester,
        HybridRetriever,
        RetrievalAugmentedForward,
        RetrievalConfig,
    )

Industry Context:
    - GPT-4, Claude, Gemini: Keep retrieval external/optional
    - Perplexity: RAG-native with hybrid retrieval
    - This implementation: Optional augmentation layer on top of 
      RT-DLM's existing tiered memory system
"""

# Configuration is in config/ folder (alongside AGIConfig)
from config.retrieval_config import (
    RetrievalConfig,
    RetrievalProvider,
    ChunkingStrategy,
)

from modules.retrieval.document_ingester import (
    DocumentIngester,
    DocumentChunk,
)

from modules.retrieval.hybrid_retriever import (
    HybridRetriever,
    RetrievalResult,
    SparseRetriever,
    DenseRetriever,
)

from modules.retrieval.augmented_attention import (
    RetrievalAugmentedAttention,
    CrossAttentionRetrieval,
)

from modules.retrieval.training_integration import (
    RetrievalAugmentedTraining,
    RetrievalContrastiveLoss,
)

__all__ = [
    # Configuration (from config/)
    'RetrievalConfig',
    'RetrievalProvider',
    'ChunkingStrategy',
    
    # Document Processing
    'DocumentIngester',
    'DocumentChunk',
    
    # Retrieval
    'HybridRetriever',
    'RetrievalResult',
    'SparseRetriever',
    'DenseRetriever',
    
    # Augmented Attention
    'RetrievalAugmentedAttention',
    'CrossAttentionRetrieval',
    
    # Training Integration
    'RetrievalAugmentedTraining',
    'RetrievalContrastiveLoss',
]
