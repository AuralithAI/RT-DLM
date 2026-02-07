"""
Document Ingestion Pipeline

Handles document chunking, embedding, and storage into the RT-DLM memory systems.
Integrates with existing PersistentLTMStorage rather than replacing it.

Industry Best Practices:
- Recursive text splitting (like LangChain)
- Semantic chunking for better retrieval quality
- Overlap to maintain context across chunks
- Metadata preservation for filtering
"""

import logging
import hashlib
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Iterator
from pathlib import Path
import json

import numpy as np

from src.config.retrieval_config import ChunkingStrategy

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """
    A chunk of a document with metadata.
    
    Attributes:
        text: The chunk text content
        embedding: Vector embedding (None until computed)
        chunk_id: Unique identifier for this chunk
        doc_id: Parent document identifier
        chunk_index: Position in original document
        metadata: Additional metadata (source, title, etc.)
    """
    text: str
    embedding: Optional[np.ndarray] = None
    chunk_id: str = ""
    doc_id: str = ""
    chunk_index: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Generate chunk_id if not provided."""
        if not self.chunk_id:
            # MD5 used only for content-based ID generation, not security
            content_hash = hashlib.md5(self.text.encode(), usedforsecurity=False).hexdigest()[:8]
            self.chunk_id = f"{self.doc_id}_{self.chunk_index}_{content_hash}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "text": self.text,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "chunk_index": self.chunk_index,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentChunk":
        """Create from dictionary."""
        return cls(
            text=data["text"],
            chunk_id=data.get("chunk_id", ""),
            doc_id=data.get("doc_id", ""),
            chunk_index=data.get("chunk_index", 0),
            metadata=data.get("metadata", {}),
        )


class TextChunker:
    """
    Intelligent text chunking with multiple strategies.
    
    Follows industry best practices from LangChain, LlamaIndex.
    """
    
    # Separators for recursive splitting (order matters)
    RECURSIVE_SEPARATORS = [
        "\n\n\n",   # Triple newlines (section breaks)
        "\n\n",     # Paragraph breaks
        "\n",       # Line breaks
        ". ",       # Sentences
        "? ",
        "! ",
        "; ",       # Clauses
        ", ",       # Phrases
        " ",        # Words
        "",         # Characters (last resort)
    ]
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
        length_function: Optional[Callable[[str], int]] = None,
    ):
        """
        Initialize chunker.
        
        Args:
            chunk_size: Target size for chunks (in tokens or characters)
            chunk_overlap: Overlap between consecutive chunks
            strategy: Chunking strategy to use
            length_function: Function to measure text length (default: char count)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy
        self.length_function = length_function or len
        
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks using configured strategy.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        if self.strategy == ChunkingStrategy.FIXED:
            return self._fixed_chunking(text)
        elif self.strategy == ChunkingStrategy.SENTENCE:
            return self._sentence_chunking(text)
        elif self.strategy == ChunkingStrategy.SLIDING_WINDOW:
            return self._sliding_window_chunking(text)
        elif self.strategy == ChunkingStrategy.RECURSIVE:
            return self._recursive_chunking(text)
        elif self.strategy == ChunkingStrategy.SEMANTIC:
            return self._semantic_chunking(text)
        else:
            logger.warning(f"Unknown strategy {self.strategy}, using recursive")
            return self._recursive_chunking(text)
    
    def _fixed_chunking(self, text: str) -> List[str]:
        """Simple fixed-size chunking."""
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk.strip())
            start = end - self.chunk_overlap
            
        return [c for c in chunks if c]
    
    def _sentence_chunking(self, text: str) -> List[str]:
        """Split by sentences, then combine to target size."""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks: List[str] = []
        current_chunk: List[str] = []
        current_length = 0
        
        for sentence in sentences:
            sentence_len = self.length_function(sentence)
            
            if current_length + sentence_len > self.chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                # Keep overlap
                overlap_text = " ".join(current_chunk)
                while self.length_function(overlap_text) > self.chunk_overlap:
                    current_chunk.pop(0)
                    overlap_text = " ".join(current_chunk)
                current_length = self.length_function(overlap_text)
                
            current_chunk.append(sentence)
            current_length += sentence_len
            
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return [c.strip() for c in chunks if c.strip()]
    
    def _sliding_window_chunking(self, text: str) -> List[str]:
        """Overlapping window chunking."""
        chunks = []
        step = self.chunk_size - self.chunk_overlap
        
        for i in range(0, len(text), step):
            chunk = text[i:i + self.chunk_size]
            if chunk.strip():
                chunks.append(chunk.strip())
                
        return chunks
    
    def _recursive_chunking(
        self, 
        text: str, 
        separators: Optional[List[str]] = None
    ) -> List[str]:
        """
        Recursive text splitting - industry standard approach.
        
        Tries to split on larger separators first, falls back to smaller.
        """
        if separators is None:
            separators = self.RECURSIVE_SEPARATORS.copy()
            
        if not separators:
            # Base case: no more separators, do fixed chunking
            return self._fixed_chunking(text)
        
        separator = separators[0]
        remaining_separators = separators[1:]
        
        # Try splitting on current separator
        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)  # Character-level split
            
        chunks: List[str] = []
        current_chunk: List[str] = []
        current_length = 0
        
        for split in splits:
            split_len = self.length_function(split)
            
            if split_len > self.chunk_size:
                # This split is too large, recursively chunk it
                if current_chunk:
                    chunks.append(separator.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                    
                sub_chunks = self._recursive_chunking(split, remaining_separators)
                chunks.extend(sub_chunks)
                
            elif current_length + split_len + len(separator) > self.chunk_size:
                # Current chunk is full
                if current_chunk:
                    chunks.append(separator.join(current_chunk))
                    
                # Handle overlap
                overlap_chunk: List[str] = []
                overlap_len = 0
                for item in reversed(current_chunk):
                    item_len = self.length_function(item) + len(separator)
                    if overlap_len + item_len > self.chunk_overlap:
                        break
                    overlap_chunk.insert(0, item)
                    overlap_len += item_len
                    
                current_chunk = overlap_chunk + [split]
                current_length = overlap_len + split_len
                
            else:
                current_chunk.append(split)
                current_length += split_len + len(separator)
                
        if current_chunk:
            chunks.append(separator.join(current_chunk))
            
        return [c.strip() for c in chunks if c.strip()]
    
    def _semantic_chunking(self, text: str) -> List[str]:
        """
        Semantic-aware chunking based on content boundaries.
        
        Uses heuristics to detect topic changes:
        - Headers (markdown, numbered)
        - Long paragraphs
        - Significant whitespace
        """
        # Detect semantic boundaries
        patterns = [
            r'\n#{1,6}\s',          # Markdown headers
            r'\n\d+\.\s',           # Numbered lists
            r'\n[A-Z][^.!?]*:\s*\n', # Section labels
            r'\n{3,}',              # Multiple blank lines
        ]
        
        # Mark boundaries
        boundary_positions = {0, len(text)}
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                boundary_positions.add(match.start())
                
        # Sort boundaries
        boundaries = sorted(boundary_positions)
        
        # Create initial segments
        segments = []
        for i in range(len(boundaries) - 1):
            segment = text[boundaries[i]:boundaries[i+1]].strip()
            if segment:
                segments.append(segment)
                
        # Merge small segments, split large ones
        chunks = []
        current = ""
        
        for segment in segments:
            if self.length_function(current) + self.length_function(segment) <= self.chunk_size:
                current = f"{current}\n\n{segment}" if current else segment
            else:
                if current:
                    chunks.append(current)
                    
                if self.length_function(segment) > self.chunk_size:
                    # Large segment, use recursive chunking
                    chunks.extend(self._recursive_chunking(segment))
                    current = ""
                else:
                    current = segment
                    
        if current:
            chunks.append(current)
            
        return chunks


class DocumentIngester:
    """
    Document ingestion pipeline for RT-DLM retrieval.
    
    Handles:
    - Document chunking (multiple strategies)
    - Embedding computation (using model or external)
    - Storage to MemoryBank/PersistentLTMStorage
    - Batch processing for efficiency
    
    Example:
        from src.modules.retrieval import DocumentIngester
        from src.core.model.memory_bank import MemoryBank
        
        ingester = DocumentIngester(
            memory_bank=memory_bank,
            embedding_fn=model_embed_fn,
            chunk_size=512,
        )
        
        # Ingest documents
        stats = ingester.ingest_documents([
            {"text": "Document 1 content...", "source": "wiki"},
            {"text": "Document 2 content...", "source": "arxiv"},
        ])
    """
    
    def __init__(
        self,
        memory_bank: Optional[Any] = None,  # MemoryBank instance
        embedding_fn: Optional[Callable[[List[str]], np.ndarray]] = None,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        chunking_strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
        batch_size: int = 32,
        embedding_dim: int = 384,
        storage_dir: str = "./retrieval_store",
    ):
        """
        Initialize document ingester.
        
        Args:
            memory_bank: Optional MemoryBank for storage (can add later)
            embedding_fn: Function to embed text (can add later)
            chunk_size: Target chunk size
            chunk_overlap: Overlap between chunks
            chunking_strategy: How to split documents
            batch_size: Batch size for embedding
            embedding_dim: Dimension of embeddings
            storage_dir: Directory for local storage
        """
        self.memory_bank = memory_bank
        self.embedding_fn = embedding_fn
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize chunker
        self.chunker = TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            strategy=chunking_strategy,
        )
        
        # Track ingested documents
        self.chunk_store: Dict[str, DocumentChunk] = {}
        self.doc_index: Dict[str, List[str]] = {}  # doc_id -> [chunk_ids]
        
        logger.info(f"DocumentIngester initialized: strategy={chunking_strategy.value}")
    
    def set_memory_bank(self, memory_bank: Any) -> None:
        """Set or update the memory bank."""
        self.memory_bank = memory_bank
        logger.info("Memory bank configured for ingester")
    
    def set_embedding_fn(self, embedding_fn: Callable[[List[str]], np.ndarray]) -> None:
        """Set or update the embedding function."""
        self.embedding_fn = embedding_fn
        logger.info("Embedding function configured for ingester")
    
    def ingest_documents(
        self,
        documents: List[Dict[str, Any]],
        store_to_memory: bool = True,
    ) -> Dict[str, Any]:
        """
        Ingest multiple documents.
        
        Args:
            documents: List of dicts with 'text' and optional metadata
            store_to_memory: Whether to store in memory bank
            
        Returns:
            Statistics about ingestion
        """
        stats: Dict[str, Any] = {
            "documents_processed": 0,
            "chunks_created": 0,
            "chunks_embedded": 0,
            "chunks_stored": 0,
            "errors": [],
        }
        
        all_chunks: List[DocumentChunk] = []
        
        for doc in documents:
            try:
                chunks = self._process_document(doc)
                all_chunks.extend(chunks)
                stats["documents_processed"] += 1
                stats["chunks_created"] += len(chunks)
            except Exception as e:
                logger.error(f"Error processing document: {e}")
                stats["errors"].append(str(e))
                
        # Batch embed all chunks
        if all_chunks and self.embedding_fn:
            try:
                self._embed_chunks(all_chunks)
                stats["chunks_embedded"] = len(all_chunks)
            except Exception as e:
                logger.error(f"Error embedding chunks: {e}")
                stats["errors"].append(f"Embedding error: {e}")
                
        # Store to memory bank
        if store_to_memory and self.memory_bank:
            try:
                stored = self._store_chunks(all_chunks)
                stats["chunks_stored"] = stored
            except Exception as e:
                logger.error(f"Error storing chunks: {e}")
                stats["errors"].append(f"Storage error: {e}")
                
        # Save local index
        self._save_index()
        
        logger.info(f"Ingestion complete: {stats}")
        return stats
    
    def _process_document(self, doc: Dict[str, Any]) -> List[DocumentChunk]:
        """Process a single document into chunks."""
        text = doc.get("text", "")
        if not text:
            raise ValueError("Document must have 'text' field")
            
        # Generate doc_id (MD5 used only for content-based ID generation, not security)
        doc_id = doc.get("doc_id", hashlib.md5(text.encode(), usedforsecurity=False).hexdigest()[:16])
        
        # Extract metadata
        metadata = {k: v for k, v in doc.items() if k not in ("text", "doc_id")}
        
        # Chunk the text
        chunk_texts = self.chunker.chunk_text(text)
        
        # Create chunk objects
        chunks = []
        for i, chunk_text in enumerate(chunk_texts):
            chunk = DocumentChunk(
                text=chunk_text,
                doc_id=doc_id,
                chunk_index=i,
                metadata=metadata.copy(),
            )
            chunks.append(chunk)
            self.chunk_store[chunk.chunk_id] = chunk
            
        # Update doc index
        self.doc_index[doc_id] = [c.chunk_id for c in chunks]
        
        return chunks
    
    def _embed_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Embed chunks in batches."""
        texts = [c.text for c in chunks]
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_chunks = chunks[i:i + self.batch_size]
            
            embeddings = self.embedding_fn(batch_texts)
            
            # Handle JAX arrays
            if hasattr(embeddings, 'block_until_ready'):
                embeddings = np.array(embeddings)
                
            for chunk, emb in zip(batch_chunks, embeddings):
                chunk.embedding = emb
                
    def _store_chunks(self, chunks: List[DocumentChunk]) -> int:
        """Store chunks in memory bank."""
        stored = 0
        
        for chunk in chunks:
            if chunk.embedding is None:
                continue
                
            try:
                # Store in memory bank
                if hasattr(self.memory_bank, 'store_ltm'):
                    # Use LTM storage
                    self.memory_bank.store_ltm(
                        embedding=chunk.embedding,
                        metadata={
                            "chunk_id": chunk.chunk_id,
                            "doc_id": chunk.doc_id,
                            **chunk.metadata,
                        },
                        context=chunk.text[:500],  # Truncate for storage
                    )
                elif hasattr(self.memory_bank, 'store'):
                    # Generic store method
                    self.memory_bank.store(chunk.embedding, chunk.text)
                else:
                    logger.warning("Memory bank has no store method")
                    break
                    
                stored += 1
            except Exception as e:
                logger.warning(f"Failed to store chunk {chunk.chunk_id}: {e}")
                
        return stored
    
    def _save_index(self) -> None:
        """Save chunk index to disk."""
        index_path = self.storage_dir / "chunk_index.json"
        
        index_data = {
            "doc_index": self.doc_index,
            "chunks": {
                cid: chunk.to_dict() 
                for cid, chunk in self.chunk_store.items()
            },
        }
        
        with open(index_path, "w") as f:
            json.dump(index_data, f)
            
    def load_index(self) -> bool:
        """Load chunk index from disk."""
        index_path = self.storage_dir / "chunk_index.json"
        
        if not index_path.exists():
            return False
            
        try:
            with open(index_path, "r") as f:
                index_data = json.load(f)
                
            self.doc_index = index_data.get("doc_index", {})
            self.chunk_store = {
                cid: DocumentChunk.from_dict(data)
                for cid, data in index_data.get("chunks", {}).items()
            }
            
            logger.info(f"Loaded index: {len(self.chunk_store)} chunks")
            return True
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False
    
    def get_chunk(self, chunk_id: str) -> Optional[DocumentChunk]:
        """Retrieve a chunk by ID."""
        return self.chunk_store.get(chunk_id)
    
    def get_document_chunks(self, doc_id: str) -> List[DocumentChunk]:
        """Get all chunks for a document."""
        chunk_ids = self.doc_index.get(doc_id, [])
        return [self.chunk_store[cid] for cid in chunk_ids if cid in self.chunk_store]
    
    def iter_chunks(self) -> Iterator[DocumentChunk]:
        """Iterate over all chunks."""
        yield from self.chunk_store.values()
    
    @property
    def num_chunks(self) -> int:
        """Total number of chunks."""
        return len(self.chunk_store)
    
    @property
    def num_documents(self) -> int:
        """Total number of documents."""
        return len(self.doc_index)
