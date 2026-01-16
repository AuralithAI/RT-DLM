"""
Hybrid Retrieval System

Combines dense (embedding-based) and sparse (BM25) retrieval for
best-in-class retrieval quality. Industry standard approach.

Why Hybrid?
- Dense: Good for semantic similarity
- Sparse: Good for keyword/entity matching
- Combined: Best of both worlds (Perplexity, Google, etc. use this)
"""

import logging
import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """
    Result from retrieval operation.
    
    Attributes:
        text: Retrieved text content
        score: Relevance score (higher = more relevant)
        chunk_id: Unique identifier for the chunk
        metadata: Additional metadata
        source: Which retriever found this ("dense", "sparse", "hybrid")
    """
    text: str
    score: float
    chunk_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = "hybrid"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "score": self.score,
            "chunk_id": self.chunk_id,
            "metadata": self.metadata,
            "source": self.source,
        }


class SparseRetriever:
    """
    BM25-based sparse retrieval.
    
    BM25 is the industry standard for keyword-based retrieval.
    Good for:
    - Exact term matching
    - Named entities
    - Technical terms
    """
    
    # BM25 parameters (standard values)
    K1 = 1.5  # Term frequency saturation
    B = 0.75  # Document length normalization
    
    def __init__(self):
        """Initialize sparse retriever."""
        self.documents: List[str] = []
        self.doc_ids: List[str] = []
        self.metadata: List[Dict] = []
        
        # BM25 index
        self.doc_freqs: Counter = Counter()  # Term -> doc count
        self.doc_lens: List[int] = []
        self.avg_doc_len: float = 0.0
        self.term_freqs: List[Counter] = []  # Per-doc term frequencies
        
    def add_documents(
        self,
        documents: List[str],
        doc_ids: Optional[List[str]] = None,
        metadata: Optional[List[Dict]] = None,
    ) -> None:
        """
        Add documents to the index.
        
        Args:
            documents: List of document texts
            doc_ids: Optional list of document IDs
            metadata: Optional list of metadata dicts
        """
        if doc_ids is None:
            start_idx = len(self.documents)
            doc_ids = [f"doc_{start_idx + i}" for i in range(len(documents))]
            
        if metadata is None:
            metadata = [{} for _ in documents]
            
        for doc, doc_id, meta in zip(documents, doc_ids, metadata):
            # Tokenize
            tokens = self._tokenize(doc)
            
            # Update index
            self.documents.append(doc)
            self.doc_ids.append(doc_id)
            self.metadata.append(meta)
            
            # Term frequencies for this doc
            tf = Counter(tokens)
            self.term_freqs.append(tf)
            
            # Update document frequencies
            for term in set(tokens):
                self.doc_freqs[term] += 1
                
            # Document length
            self.doc_lens.append(len(tokens))
            
        # Update average document length
        if self.doc_lens:
            self.avg_doc_len = sum(self.doc_lens) / len(self.doc_lens)
            
        logger.debug(f"Sparse index: {len(self.documents)} documents")
    
    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        """
        Search for documents matching query.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of RetrievalResult sorted by score
        """
        if not self.documents:
            return []
            
        query_tokens = self._tokenize(query)
        scores = []
        
        num_docs = len(self.documents)
        
        for doc_idx in range(num_docs):
            score = self._compute_bm25_score(query_tokens, doc_idx, num_docs)
            scores.append((doc_idx, score))
            
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Build results
        results = []
        for doc_idx, score in scores[:top_k]:
            results.append(RetrievalResult(
                text=self.documents[doc_idx],
                score=score,
                chunk_id=self.doc_ids[doc_idx],
                metadata=self.metadata[doc_idx],
                source="sparse",
            ))
            
        return results
    
    def _compute_bm25_score(
        self,
        query_tokens: List[str],
        doc_idx: int,
        num_docs: int,
    ) -> float:
        """Compute BM25 score for a document."""
        score = 0.0
        doc_len = self.doc_lens[doc_idx]
        tf = self.term_freqs[doc_idx]
        
        for term in query_tokens:
            if term not in tf:
                continue
                
            # Term frequency in document
            term_freq = tf[term]
            
            # Document frequency
            doc_freq = self.doc_freqs.get(term, 0)
            
            # IDF component
            idf = math.log((num_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1)
            
            # TF component with length normalization
            tf_component = (
                term_freq * (self.K1 + 1) /
                (term_freq + self.K1 * (1 - self.B + self.B * doc_len / self.avg_doc_len))
            )
            
            score += idf * tf_component
            
        return score
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        # Lowercase and split on non-alphanumeric
        import re
        tokens = re.findall(r'\b\w+\b', text.lower())
        # Remove very short tokens
        return [t for t in tokens if len(t) > 2]


class DenseRetriever:
    """
    Embedding-based dense retrieval.
    
    Uses cosine similarity or dot product between query and document embeddings.
    Good for:
    - Semantic similarity
    - Paraphrase detection
    - Cross-lingual retrieval
    """
    
    def __init__(
        self,
        embedding_dim: int = 384,
        use_faiss: bool = True,
        normalize: bool = True,
    ):
        """
        Initialize dense retriever.
        
        Args:
            embedding_dim: Dimension of embeddings
            use_faiss: Use FAISS for efficient search
            normalize: L2 normalize embeddings
        """
        self.embedding_dim = embedding_dim
        self.use_faiss = use_faiss
        self.normalize = normalize
        
        # Document storage
        self.embeddings: List[np.ndarray] = []
        self.documents: List[str] = []
        self.doc_ids: List[str] = []
        self.metadata: List[Dict] = []
        
        # FAISS index
        self.index = None
        if use_faiss:
            try:
                import faiss
                self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product (cosine if normalized)
                logger.info("Dense retriever using FAISS")
            except ImportError:
                logger.warning("FAISS not available, using numpy")
                self.use_faiss = False
                
    def add_documents(
        self,
        embeddings: np.ndarray,
        documents: List[str],
        doc_ids: Optional[List[str]] = None,
        metadata: Optional[List[Dict]] = None,
    ) -> None:
        """
        Add documents with their embeddings.
        
        Args:
            embeddings: Document embeddings [num_docs, embedding_dim]
            documents: Document texts
            doc_ids: Optional document IDs
            metadata: Optional metadata
        """
        if doc_ids is None:
            start_idx = len(self.documents)
            doc_ids = [f"doc_{start_idx + i}" for i in range(len(documents))]
            
        if metadata is None:
            metadata = [{} for _ in documents]
            
        # Normalize if needed
        if self.normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-8)
            
        # Store
        for emb, doc, doc_id, meta in zip(embeddings, documents, doc_ids, metadata):
            self.embeddings.append(emb)
            self.documents.append(doc)
            self.doc_ids.append(doc_id)
            self.metadata.append(meta)
            
        # Update FAISS index
        if self.use_faiss and self.index is not None:
            self.index.add(embeddings.astype(np.float32))
            
        logger.debug(f"Dense index: {len(self.documents)} documents")
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding [embedding_dim]
            top_k: Number of results
            
        Returns:
            List of RetrievalResult sorted by score
        """
        if not self.documents:
            return []
            
        # Normalize query
        if self.normalize:
            query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
            
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        
        if self.use_faiss and self.index is not None:
            scores, indices = self.index.search(query_embedding, min(top_k, len(self.documents)))
            scores = scores[0]
            indices = indices[0]
        else:
            # Numpy fallback
            embeddings_matrix = np.array(self.embeddings)
            scores = np.dot(embeddings_matrix, query_embedding.T).flatten()
            indices = np.argsort(scores)[::-1][:top_k]
            scores = scores[indices]
            
        # Build results
        results = []
        for idx, score in zip(indices, scores):
            if idx < 0:  # FAISS returns -1 for empty results
                continue
            results.append(RetrievalResult(
                text=self.documents[idx],
                score=float(score),
                chunk_id=self.doc_ids[idx],
                metadata=self.metadata[idx],
                source="dense",
            ))
            
        return results


class HybridRetriever:
    """
    Hybrid retrieval combining dense and sparse methods.
    
    Industry best practice used by:
    - Perplexity AI
    - Google Search
    - Microsoft Bing
    - OpenAI (with browsing)
    
    Fusion strategy: Reciprocal Rank Fusion (RRF)
    """
    
    def __init__(
        self,
        embedding_dim: int = 384,
        sparse_weight: float = 0.3,
        dense_weight: float = 0.7,
        use_rrf: bool = True,
        rrf_k: int = 60,
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            embedding_dim: Dimension for dense embeddings
            sparse_weight: Weight for sparse retrieval
            dense_weight: Weight for dense retrieval
            use_rrf: Use Reciprocal Rank Fusion (recommended)
            rrf_k: RRF constant (60 is standard)
        """
        self.sparse_weight = sparse_weight
        self.dense_weight = dense_weight
        self.use_rrf = use_rrf
        self.rrf_k = rrf_k
        
        # Initialize retrievers
        self.sparse_retriever = SparseRetriever()
        self.dense_retriever = DenseRetriever(embedding_dim=embedding_dim)
        
        logger.info(
            f"HybridRetriever: sparse={sparse_weight:.1%}, dense={dense_weight:.1%}, "
            f"rrf={use_rrf}"
        )
    
    def add_documents(
        self,
        documents: List[str],
        embeddings: Optional[np.ndarray] = None,
        doc_ids: Optional[List[str]] = None,
        metadata: Optional[List[Dict]] = None,
    ) -> None:
        """
        Add documents to both retrievers.
        
        Args:
            documents: Document texts
            embeddings: Dense embeddings (required for dense retrieval)
            doc_ids: Document IDs
            metadata: Document metadata
        """
        # Add to sparse
        self.sparse_retriever.add_documents(documents, doc_ids, metadata)
        
        # Add to dense if embeddings provided
        if embeddings is not None:
            self.dense_retriever.add_documents(embeddings, documents, doc_ids, metadata)
        else:
            logger.warning("No embeddings provided, dense retrieval will be disabled")
    
    def search(
        self,
        query: str,
        query_embedding: Optional[np.ndarray] = None,
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        """
        Hybrid search combining sparse and dense retrieval.
        
        Args:
            query: Query text (for sparse)
            query_embedding: Query embedding (for dense)
            top_k: Number of results
            
        Returns:
            List of RetrievalResult sorted by combined score
        """
        # Get more results from each to allow for fusion
        fetch_k = top_k * 3
        
        # Sparse retrieval
        sparse_results = self.sparse_retriever.search(query, top_k=fetch_k)
        
        # Dense retrieval
        dense_results = []
        if query_embedding is not None and self.dense_retriever.documents:
            dense_results = self.dense_retriever.search(query_embedding, top_k=fetch_k)
            
        # Combine results
        if self.use_rrf:
            combined = self._reciprocal_rank_fusion(sparse_results, dense_results)
        else:
            combined = self._weighted_fusion(sparse_results, dense_results)
            
        return combined[:top_k]
    
    def _reciprocal_rank_fusion(
        self,
        sparse_results: List[RetrievalResult],
        dense_results: List[RetrievalResult],
    ) -> List[RetrievalResult]:
        """
        Combine results using Reciprocal Rank Fusion.
        
        RRF score = sum(1 / (k + rank)) for each list where doc appears.
        This is robust to score calibration differences between retrievers.
        """
        scores: Dict[str, float] = {}
        results_map: Dict[str, RetrievalResult] = {}
        
        # Process sparse results
        for rank, result in enumerate(sparse_results):
            key = result.chunk_id or result.text[:100]
            rrf_score = self.sparse_weight / (self.rrf_k + rank + 1)
            scores[key] = scores.get(key, 0) + rrf_score
            results_map[key] = result
            
        # Process dense results
        for rank, result in enumerate(dense_results):
            key = result.chunk_id or result.text[:100]
            rrf_score = self.dense_weight / (self.rrf_k + rank + 1)
            scores[key] = scores.get(key, 0) + rrf_score
            if key not in results_map:
                results_map[key] = result
                
        # Sort by combined score
        sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
        
        # Build results with updated scores
        results = []
        for key in sorted_keys:
            result = results_map[key]
            results.append(RetrievalResult(
                text=result.text,
                score=scores[key],
                chunk_id=result.chunk_id,
                metadata=result.metadata,
                source="hybrid",
            ))
            
        return results
    
    def _weighted_fusion(
        self,
        sparse_results: List[RetrievalResult],
        dense_results: List[RetrievalResult],
    ) -> List[RetrievalResult]:
        """
        Combine results using weighted score fusion.
        
        Simple but requires score normalization.
        """
        scores: Dict[str, float] = {}
        results_map: Dict[str, RetrievalResult] = {}
        
        # Normalize sparse scores
        if sparse_results:
            max_sparse = max(r.score for r in sparse_results) or 1.0
            for result in sparse_results:
                key = result.chunk_id or result.text[:100]
                normalized = (result.score / max_sparse) * self.sparse_weight
                scores[key] = scores.get(key, 0) + normalized
                results_map[key] = result
                
        # Normalize dense scores
        if dense_results:
            max_dense = max(r.score for r in dense_results) or 1.0
            for result in dense_results:
                key = result.chunk_id or result.text[:100]
                normalized = (result.score / max_dense) * self.dense_weight
                scores[key] = scores.get(key, 0) + normalized
                if key not in results_map:
                    results_map[key] = result
                    
        # Sort by combined score
        sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
        
        # Build results
        results = []
        for key in sorted_keys:
            result = results_map[key]
            results.append(RetrievalResult(
                text=result.text,
                score=scores[key],
                chunk_id=result.chunk_id,
                metadata=result.metadata,
                source="hybrid",
            ))
            
        return results
    
    @property
    def num_documents(self) -> int:
        """Number of indexed documents."""
        return len(self.sparse_retriever.documents)
