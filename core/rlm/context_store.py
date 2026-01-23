import time
import hashlib
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Iterator
from enum import Enum
import numpy as np

from config.rlm_config import PartitionStrategy


@dataclass
class ContextMetadata:
    source: str = ""
    content_type: str = "text"
    total_length: int = 0
    num_chunks: int = 0
    embedding_dim: Optional[int] = None
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    tags: List[str] = field(default_factory=list)
    parent_var: Optional[str] = None
    chunk_index: Optional[int] = None


@dataclass
class ContextVariable:
    name: str
    content: str
    metadata: ContextMetadata
    embedding: Optional[np.ndarray] = None
    chunks: Optional[List[str]] = None
    chunk_embeddings: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return len(self.content)

    def content_hash(self) -> str:
        return hashlib.sha256(self.content.encode()).hexdigest()[:16]


class ContextStore:
    def __init__(
        self,
        max_variables: int = 100,
        max_total_size: int = 10_000_000,
        enable_caching: bool = True,
        cache_ttl: float = 300.0,
    ):
        self._variables: Dict[str, ContextVariable] = {}
        self._max_variables = max_variables
        self._max_total_size = max_total_size
        self._enable_caching = enable_caching
        self._cache_ttl = cache_ttl
        self._total_size = 0
        self._access_order: List[str] = []
        self._embedding_fn: Optional[Any] = None

    def set_embedding_function(self, fn: Any) -> None:
        self._embedding_fn = fn

    def store(
        self,
        name: str,
        content: str,
        source: str = "",
        tags: Optional[List[str]] = None,
        parent_var: Optional[str] = None,
        chunk_index: Optional[int] = None,
    ) -> ContextVariable:
        if len(content) > self._max_total_size:
            raise ValueError(f"Content exceeds max size: {len(content)} > {self._max_total_size}")

        self._evict_if_needed(len(content))

        metadata = ContextMetadata(
            source=source,
            total_length=len(content),
            tags=tags or [],
            parent_var=parent_var,
            chunk_index=chunk_index,
        )

        embedding = None
        if self._embedding_fn is not None:
            try:
                embedding = self._embedding_fn(content)
                metadata.embedding_dim = embedding.shape[-1] if hasattr(embedding, 'shape') else None
            except Exception:
                pass

        var = ContextVariable(
            name=name,
            content=content,
            metadata=metadata,
            embedding=embedding,
        )

        if name in self._variables:
            self._total_size -= len(self._variables[name].content)

        self._variables[name] = var
        self._total_size += len(content)
        self._update_access_order(name)

        return var

    def get(self, name: str) -> Optional[ContextVariable]:
        var = self._variables.get(name)
        if var is not None:
            var.metadata.last_accessed = time.time()
            var.metadata.access_count += 1
            self._update_access_order(name)
        return var

    def peek(self, name: str, start: int = 0, length: int = 2000) -> Optional[str]:
        var = self.get(name)
        if var is None:
            return None
        end = min(start + length, len(var.content))
        return var.content[start:end]

    def delete(self, name: str) -> bool:
        if name in self._variables:
            self._total_size -= len(self._variables[name].content)
            del self._variables[name]
            if name in self._access_order:
                self._access_order.remove(name)
            return True
        return False

    def list_variables(self) -> List[str]:
        return list(self._variables.keys())

    def get_metadata(self, name: str) -> Optional[ContextMetadata]:
        var = self._variables.get(name)
        return var.metadata if var else None

    def partition(
        self,
        name: str,
        strategy: PartitionStrategy = PartitionStrategy.FIXED_SIZE,
        chunk_size: int = 2000,
        overlap: int = 200,
    ) -> List[str]:
        var = self.get(name)
        if var is None:
            return []

        content = var.content

        if strategy == PartitionStrategy.FIXED_SIZE:
            chunks = self._partition_fixed_size(content, chunk_size, overlap)
        elif strategy == PartitionStrategy.PARAGRAPH:
            chunks = self._partition_paragraph(content, chunk_size)
        elif strategy == PartitionStrategy.SENTENCE:
            chunks = self._partition_sentence(content, chunk_size)
        elif strategy == PartitionStrategy.SEMANTIC:
            chunks = self._partition_semantic(content, chunk_size)
        else:
            chunks = self._partition_fixed_size(content, chunk_size, overlap)

        chunk_names = []
        for i, chunk in enumerate(chunks):
            chunk_name = f"{name}_chunk_{i}"
            self.store(
                name=chunk_name,
                content=chunk,
                source=var.metadata.source,
                tags=var.metadata.tags + [f"chunk_{i}"],
                parent_var=name,
                chunk_index=i,
            )
            chunk_names.append(chunk_name)

        var.chunks = chunks
        var.metadata.num_chunks = len(chunks)

        return chunk_names

    def grep(
        self,
        name: str,
        pattern: str,
        regex: bool = False,
        context_lines: int = 0,
    ) -> List[Dict[str, Any]]:
        var = self.get(name)
        if var is None:
            return []

        results = []
        lines = var.content.split('\n')

        if regex:
            try:
                compiled = re.compile(pattern, re.IGNORECASE)
            except re.error:
                return []
            match_fn = lambda line: compiled.search(line) is not None
        else:
            pattern_lower = pattern.lower()
            match_fn = lambda line: pattern_lower in line.lower()

        for i, line in enumerate(lines):
            if match_fn(line):
                start_idx = max(0, i - context_lines)
                end_idx = min(len(lines), i + context_lines + 1)
                context = '\n'.join(lines[start_idx:end_idx])
                results.append({
                    'line_number': i,
                    'line': line,
                    'context': context if context_lines > 0 else line,
                })

        return results

    def count(self, name: str, pattern: str, regex: bool = False) -> int:
        var = self.get(name)
        if var is None:
            return 0

        if regex:
            try:
                return len(re.findall(pattern, var.content, re.IGNORECASE))
            except re.error:
                return 0
        return var.content.lower().count(pattern.lower())

    def clear(self) -> None:
        self._variables.clear()
        self._access_order.clear()
        self._total_size = 0

    def stats(self) -> Dict[str, Any]:
        return {
            'num_variables': len(self._variables),
            'total_size': self._total_size,
            'max_variables': self._max_variables,
            'max_total_size': self._max_total_size,
        }

    def _evict_if_needed(self, new_size: int) -> None:
        while (
            len(self._variables) >= self._max_variables
            or self._total_size + new_size > self._max_total_size
        ):
            if not self._access_order:
                break
            oldest = self._access_order[0]
            self.delete(oldest)

    def _update_access_order(self, name: str) -> None:
        if name in self._access_order:
            self._access_order.remove(name)
        self._access_order.append(name)

    def _partition_fixed_size(
        self, content: str, chunk_size: int, overlap: int
    ) -> List[str]:
        chunks = []
        start = 0
        while start < len(content):
            end = start + chunk_size
            chunks.append(content[start:end])
            start = end - overlap
            if start >= len(content):
                break
        return chunks

    def _partition_paragraph(self, content: str, max_size: int) -> List[str]:
        paragraphs = re.split(r'\n\s*\n', content)
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk) + len(para) + 2 <= max_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks if chunks else [content]

    def _partition_sentence(self, content: str, max_size: int) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', content)
        chunks = []
        current_chunk = ""

        for sent in sentences:
            if len(current_chunk) + len(sent) + 1 <= max_size:
                current_chunk += sent + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sent + " "

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks if chunks else [content]

    def _partition_semantic(self, content: str, max_size: int) -> List[str]:
        paragraphs = re.split(r'\n\s*\n', content)

        if self._embedding_fn is None:
            return self._partition_paragraph(content, max_size)

        chunks = []
        current_chunk = ""
        current_embedding = None

        for para in paragraphs:
            if not para.strip():
                continue

            para_embedding = self._get_embedding_safe(para)
            current_chunk, current_embedding, new_chunk = self._process_semantic_paragraph(
                para, para_embedding, current_chunk, current_embedding, max_size
            )
            if new_chunk:
                chunks.append(new_chunk)

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks if chunks else [content]

    def _get_embedding_safe(self, text: str) -> Optional[np.ndarray]:
        if self._embedding_fn is None:
            return None
        try:
            return self._embedding_fn(text)
        except Exception:
            return None

    def _process_semantic_paragraph(
        self,
        para: str,
        para_embedding: Optional[np.ndarray],
        current_chunk: str,
        current_embedding: Optional[np.ndarray],
        max_size: int,
    ) -> tuple:
        if len(current_chunk) + len(para) + 2 > max_size:
            new_chunk = current_chunk.strip() if current_chunk else None
            return para + "\n\n", para_embedding, new_chunk

        if current_embedding is not None and para_embedding is not None:
            similarity = self._cosine_similarity(current_embedding, para_embedding)
            if similarity > 0.7:
                return current_chunk + para + "\n\n", current_embedding, None
            new_chunk = current_chunk.strip() if current_chunk else None
            return para + "\n\n", para_embedding, new_chunk

        return current_chunk + para + "\n\n", para_embedding, None

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a).flatten()
        b = np.asarray(b).flatten()
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def __iter__(self) -> Iterator[ContextVariable]:
        return iter(self._variables.values())

    def __len__(self) -> int:
        return len(self._variables)

    def __contains__(self, name: str) -> bool:
        return name in self._variables
