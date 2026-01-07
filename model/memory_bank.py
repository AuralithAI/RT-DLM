import platform
import logging
import hashlib
import secrets

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Platform-aware FAISS import
# Use faiss-gpu on Linux/Mac, faiss-cpu on Windows
_USE_GPU_FAISS = platform.system() != "Windows"

try:
    import faiss
    if _USE_GPU_FAISS and hasattr(faiss, 'StandardGpuResources'):
        logger.info("FAISS GPU support available")
    elif _USE_GPU_FAISS:
        logger.info("FAISS GPU not available, using CPU")
except ImportError:
    raise ImportError(
        "FAISS not installed. Install with:\n"
        "  Windows: pip install faiss-cpu\n"
        "  Linux/Mac: pip install faiss-gpu (or faiss-cpu)"
    )

# Security imports - only import what's actually used
try:
    from model.security import (
        SecureStorage,
        DataSanitizer,
        IdentifierHasher
    )
    _SECURITY_AVAILABLE = True
except ImportError:
    # Fallback for direct imports
    try:
        from .security import (
            SecureStorage,
            DataSanitizer,
            IdentifierHasher
        )
        _SECURITY_AVAILABLE = True
    except ImportError:
        _SECURITY_AVAILABLE = False
        logger.warning("Security modules not available. PII scrubbing and encryption disabled.")

import jax
import jax.numpy as jnp
import numpy as np
import sqlite3
import json
from typing import Optional, Tuple, Dict, List, Any
from dataclasses import dataclass
from pathlib import Path
import time


# =============================================================================
# Security: PIIScrubber and SecureStorage imported from model.security
# =============================================================================

# Backward compatibility aliases for any code that imports from this module
# These are re-exported from the security module for convenience
if _SECURITY_AVAILABLE:
    # PIIScrubber is replaced by the new DataSanitizer which uses PIIDetector
    # SecureStorage, IdentifierHasher are directly imported from security module
    pass  # Classes already imported above
else:
    # Fallback: minimal stub if security module unavailable
    class PIIDetector:
        """Stub when security module unavailable."""
        def detect(self, text): return []  # noqa: ARG002
    
    class DataSanitizer:
        """Stub when security module unavailable."""
        def __init__(self, *args, **kwargs): 
            """Initialize stub."""  # noqa: ARG002
        def sanitize(self, text: str) -> str:
            return text
        def sanitize_dict(self, data): 
            return data
    
    class SecureStorage:
        """Stub when security module unavailable."""
        def __init__(self, *args, **kwargs): 
            """Initialize stub."""  # noqa: ARG002
            self.encryption_enabled = False
        def encrypt(self, data): return data
        def decrypt(self, data): return data
        def encrypt_dict(self, data): return json.dumps(data)
        def decrypt_dict(self, data): 
            try: return json.loads(data)
            except json.JSONDecodeError: return {}
    
    class IdentifierHasher:
        """Stub when security module unavailable."""
        def __init__(self, *args, **kwargs): 
            """Initialize stub."""  # noqa: ARG002
        def hash(self, identifier):
            if not identifier: return identifier
            return hashlib.sha256(identifier.encode()).hexdigest()[:32]


@dataclass
class MemoryItem:
    """Enhanced memory item with metadata."""
    key: np.ndarray
    value: np.ndarray
    timestamp: float
    access_count: int
    importance_score: float
    context_tags: List[str]
    emotional_valence: float  # -1 to 1
    consolidation_level: int  # 0=STM, 1=MTM, 2=LTM


class AdaptiveForgettingCurve:
    """Implements sophisticated forgetting mechanisms."""
    
    def __init__(self, base_decay: float = 0.1, importance_weight: float = 0.5):
        self.base_decay = base_decay
        self.importance_weight = importance_weight
    
    def calculate_retention_probability(self, memory_item: MemoryItem, 
                                     current_time: float) -> float:
        """Calculate probability of retaining a memory."""
        # Time-based decay
        time_delta = current_time - memory_item.timestamp
        time_decay = np.exp(-self.base_decay * time_delta)
        
        # Importance-based retention
        importance_boost = 1.0 + self.importance_weight * memory_item.importance_score
        
        # Access frequency boost
        access_boost = 1.0 + 0.1 * np.log1p(memory_item.access_count)
        
        # Emotional significance boost
        emotional_boost = 1.0 + 0.2 * abs(memory_item.emotional_valence)
        
        # Combined retention probability
        retention_prob = time_decay * importance_boost * access_boost * emotional_boost
        return min(1.0, retention_prob)


class ContextualMemoryIndex:
    """Enhanced memory indexing with contextual clustering."""
    
    def __init__(self, embedding_dim: int, num_clusters: int = 8):
        self.embedding_dim = embedding_dim
        self.num_clusters = num_clusters
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.context_clusters = {}
        self.memory_items: List[MemoryItem] = []
        
    def add_memory(self, memory_item: MemoryItem):
        """Add memory with contextual clustering."""
        self.memory_items.append(memory_item)
        key_array = memory_item.key.reshape(1, -1).astype(np.float32)
        self.index.add(key_array)
        
        # Update context clusters
        for tag in memory_item.context_tags:
            if tag not in self.context_clusters:
                self.context_clusters[tag] = []
            self.context_clusters[tag].append(len(self.memory_items) - 1)
    
    def retrieve_contextual(self, query: np.ndarray, context_tags: List[str], 
                          k: int = 5) -> List[MemoryItem]:
        """Retrieve memories with contextual filtering."""
        if self.index.ntotal == 0:
            return []
            
        # Get candidate memories
        query_array = query.reshape(1, -1).astype(np.float32)
        _, indices = self.index.search(query_array, min(k * 3, self.index.ntotal))
        
        candidates = []
        for idx in indices[0]:
            if idx < len(self.memory_items):
                memory = self.memory_items[idx]
                # Calculate context relevance
                context_overlap = len(set(memory.context_tags) & set(context_tags))
                relevance_score = context_overlap / (len(context_tags) + 1e-8)
                candidates.append((memory, relevance_score))
        
        # Sort by relevance and return top-k
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [item[0] for item in candidates[:k]]


@dataclass
class LTMEntry:
    """Entry for Long-Term Memory with metadata for persistence."""
    embedding_id: int
    embedding: np.ndarray
    timestamp: float
    access_count: int
    importance_score: float
    context: str
    session_id: str
    user_id: Optional[str]
    emotional_valence: float
    consolidation_level: int
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON storage."""
        return {
            'embedding_id': self.embedding_id,
            'timestamp': self.timestamp,
            'access_count': self.access_count,
            'importance_score': self.importance_score,
            'context': self.context,
            'session_id': self.session_id,
            'user_id': self.user_id,
            'emotional_valence': self.emotional_valence,
            'consolidation_level': self.consolidation_level,
            'metadata': self.metadata
        }


class PersistentLTMStorage:
    """
    Persistent Long-Term Memory storage backend using FAISS and SQLite.
    
    Enables cross-session recall for applications like:
    - Personalized education: Remember student progress and learning patterns
    - Mental health support: Track emotional patterns and coping strategies
    - Personal assistants: Remember user preferences and history
    
    Features:
    - FAISS IndexFlatL2 for efficient vector similarity search
    - SQLite for metadata persistence and querying
    - Automatic save/load with configurable paths
    - Session and user-based filtering
    - **PII Detection and Scrubbing**: Automatically removes personal data
    - **Encryption**: Optional AES encryption for sensitive metadata
    - **Identifier Hashing**: User/session IDs are hashed, never stored raw
    
    Security:
    - All text fields are scrubbed for PII before storage
    - user_id and session_id are always hashed (one-way)
    - Optional encryption for context and metadata fields
    """
    
    def __init__(
        self,
        d_model: int,
        storage_dir: str = "./ltm_storage",
        db_name: str = "ltm.db",
        index_name: str = "ltm_index.faiss",
        auto_save: bool = True,
        save_interval: int = 100,
        encryption_key: Optional[str] = None,
        enable_pii_scrubbing: bool = True,
        hash_salt: Optional[str] = None
    ):
        """
        Initialize persistent LTM storage with security features.
        
        Args:
            d_model: Embedding dimension
            storage_dir: Directory for storage files
            db_name: SQLite database filename
            index_name: FAISS index filename
            auto_save: Whether to auto-save periodically
            save_interval: Number of operations between auto-saves
            encryption_key: Password for encrypting sensitive fields (optional)
            enable_pii_scrubbing: Whether to automatically remove PII from text
            hash_salt: Salt for hashing identifiers (optional, random if not provided)
        """
        self.d_model = d_model
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.db_path = self.storage_dir / db_name
        self.index_path = self.storage_dir / index_name
        self.embeddings_path = self.storage_dir / "embeddings.npy"
        self.salt_path = self.storage_dir / ".salt"
        
        self.auto_save = auto_save
        self.save_interval = save_interval
        self.operation_count = 0
        
        # Security: PII Scrubbing using DataSanitizer
        self.enable_pii_scrubbing = enable_pii_scrubbing
        self._data_sanitizer = DataSanitizer() if enable_pii_scrubbing else None
        
        # Security: Identifier hashing using IdentifierHasher
        self._hash_salt = hash_salt or self._load_or_create_salt()
        self._id_hasher = IdentifierHasher(salt=self._hash_salt)
        
        # Security: Encryption
        self._secure_storage: Optional[SecureStorage] = None
        if encryption_key:
            encryption_salt = self._load_or_create_encryption_salt()
            self._secure_storage = SecureStorage(
                encryption_key=encryption_key,
                salt=encryption_salt
            )
        
        # Initialize FAISS index
        self.ltm_index = faiss.IndexFlatL2(d_model)
        
        # Track embeddings (FAISS doesn't store them retrievably)
        self.embeddings: List[np.ndarray] = []
        
        # Initialize SQLite database
        self._init_database()
        
        # Load existing data if available
        self._load_if_exists()
        
        security_status = []
        if enable_pii_scrubbing:
            security_status.append("PII-scrubbing")
        if encryption_key:
            security_status.append("encrypted")
        security_status.append("hashed-identifiers")
        
        logger.info(f"PersistentLTMStorage initialized at {storage_dir} [{', '.join(security_status)}]")
    
    def _load_or_create_salt(self) -> str:
        """Load existing hash salt or create a new one."""
        if self.salt_path.exists():
            return self.salt_path.read_text().strip()
        salt = secrets.token_hex(16)
        self.salt_path.write_text(salt)
        return salt
    
    def _load_or_create_encryption_salt(self) -> bytes:
        """Load existing encryption salt or create a new one."""
        enc_salt_path = self.storage_dir / ".enc_salt"
        if enc_salt_path.exists():
            return enc_salt_path.read_bytes()
        salt = secrets.token_bytes(16)
        enc_salt_path.write_bytes(salt)
        return salt
    
    def _hash_id(self, identifier: Optional[str]) -> Optional[str]:
        """Hash an identifier (user_id, session_id) for secure storage."""
        if not identifier:
            return None
        return self._id_hasher.hash(identifier)
    
    def _scrub_text(self, text: str) -> str:
        """Remove PII from text if scrubbing is enabled."""
        if not text or not self._data_sanitizer:
            return text
        return self._data_sanitizer.sanitize(text)
    
    def _scrub_metadata(self, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Remove PII from metadata dictionary."""
        if not metadata:
            return {}
        if not self._data_sanitizer:
            return metadata
        return self._data_sanitizer.sanitize_dict(metadata)
    
    def _encrypt_text(self, text: str) -> str:
        """Encrypt text if encryption is enabled."""
        if not self._secure_storage or not text:
            return text
        return self._secure_storage.encrypt(text)
    
    def _decrypt_text(self, encrypted: str) -> str:
        """Decrypt text if encryption is enabled."""
        if not self._secure_storage or not encrypted:
            return encrypted
        return self._secure_storage.decrypt(encrypted)
    
    def _encrypt_metadata(self, metadata: Dict[str, Any]) -> str:
        """Encrypt metadata to JSON string."""
        if not self._secure_storage:
            return json.dumps(metadata)
        return self._secure_storage.encrypt_dict(metadata)
    
    def _decrypt_metadata(self, encrypted: str) -> Dict[str, Any]:
        """Decrypt metadata from JSON string."""
        if not self._secure_storage:
            try:
                return json.loads(encrypted) if encrypted else {}
            except json.JSONDecodeError:
                return {}
        return self._secure_storage.decrypt_dict(encrypted)
        
    def _init_database(self) -> None:
        """Initialize SQLite database schema."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Create LTM entries table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ltm_entries (
                embedding_id INTEGER PRIMARY KEY,
                timestamp REAL NOT NULL,
                access_count INTEGER DEFAULT 0,
                importance_score REAL DEFAULT 0.5,
                context TEXT,
                session_id TEXT,
                user_id TEXT,
                emotional_valence REAL DEFAULT 0.0,
                consolidation_level INTEGER DEFAULT 2,
                metadata TEXT
            )
        ''')
        
        # Create indexes for efficient querying
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_session ON ltm_entries(session_id)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_user ON ltm_entries(user_id)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_timestamp ON ltm_entries(timestamp)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_importance ON ltm_entries(importance_score)
        ''')
        
        conn.commit()
        conn.close()
        
    def _load_if_exists(self) -> None:
        """Load existing index and embeddings if available."""
        if self.index_path.exists() and self.embeddings_path.exists():
            try:
                self.ltm_index = faiss.read_index(str(self.index_path))
                self.embeddings = list(np.load(str(self.embeddings_path)))
                logger.info(f"Loaded {self.ltm_index.ntotal} LTM entries from disk")
            except Exception as e:
                logger.warning(f"Failed to load existing LTM data: {e}")
                self.ltm_index = faiss.IndexFlatL2(self.d_model)
                self.embeddings = []
    
    def store_ltm(
        self,
        embedding: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
        context: str = "",
        session_id: str = "default",
        user_id: Optional[str] = None,
        importance_score: float = 0.5,
        emotional_valence: float = 0.0
    ) -> int:
        """
        Store an embedding with metadata in Long-Term Memory.
        
        Security:
        - PII is automatically scrubbed from context and metadata
        - user_id and session_id are hashed (one-way, not reversible)
        - context and metadata are encrypted if encryption_key was provided
        
        Args:
            embedding: Vector embedding [d_model]
            metadata: Optional additional metadata (will be scrubbed for PII)
            context: Context description (will be scrubbed for PII)
            session_id: Session identifier (will be hashed)
            user_id: Optional user identifier (will be hashed)
            importance_score: Importance score [0, 1]
            emotional_valence: Emotional valence [-1, 1]
            
        Returns:
            embedding_id: Unique identifier for the stored embedding
        """
        # Ensure proper shape
        embedding = np.asarray(embedding, dtype=np.float32)
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        
        assert embedding.shape[1] == self.d_model, \
            f"Expected dim {self.d_model}, got {embedding.shape[1]}"
        
        # Get next ID
        embedding_id = len(self.embeddings)
        
        # Add to FAISS index
        self.ltm_index.add(embedding)
        self.embeddings.append(embedding.squeeze())
        
        # SECURITY: Scrub PII from text fields
        safe_context = self._scrub_text(context)
        safe_metadata = self._scrub_metadata(metadata)
        
        # SECURITY: Hash identifiers (one-way, not reversible)
        hashed_session = self._hash_id(session_id)
        hashed_user = self._hash_id(user_id)
        
        # SECURITY: Encrypt sensitive fields
        encrypted_context = self._encrypt_text(safe_context)
        encrypted_metadata = self._encrypt_metadata(safe_metadata)
        
        # Store metadata in SQLite
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO ltm_entries 
            (embedding_id, timestamp, access_count, importance_score, context,
             session_id, user_id, emotional_valence, consolidation_level, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            embedding_id,
            time.time(),
            0,
            importance_score,
            encrypted_context,
            hashed_session,
            hashed_user,
            emotional_valence,
            2,  # LTM consolidation level
            encrypted_metadata
        ))
        
        conn.commit()
        conn.close()
        
        # Auto-save check
        self.operation_count += 1
        if self.auto_save and self.operation_count % self.save_interval == 0:
            self.save()
        
        logger.debug(f"Stored LTM entry {embedding_id} [secured]")
        return embedding_id
    
    def retrieve_ltm(
        self,
        query: np.ndarray,
        k: int = 5,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        min_importance: float = 0.0
    ) -> List[Tuple[np.ndarray, Dict[str, Any], float]]:
        """
        Retrieve similar memories from Long-Term Memory.
        
        Note: session_id and user_id filters use hashed values internally.
        Pass the original identifiers - they will be hashed for comparison.
        
        Args:
            query: Query embedding [d_model]
            k: Number of results to return
            session_id: Optional filter by session (will be hashed for lookup)
            user_id: Optional filter by user (will be hashed for lookup)
            min_importance: Minimum importance score
            
        Returns:
            List of (embedding, metadata, distance) tuples
            Note: context and metadata are decrypted if encryption was enabled
        """
        if self.ltm_index.ntotal == 0:
            return []
        
        # Ensure proper shape
        query = np.asarray(query, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        
        # Hash identifiers for comparison (they're stored hashed)
        hashed_session = self._hash_id(session_id) if session_id else None
        hashed_user = self._hash_id(user_id) if user_id else None
        
        # Search FAISS index
        search_k = min(k * 3, self.ltm_index.ntotal)  # Get extra for filtering
        distances, indices = self.ltm_index.search(query, search_k)
        
        # Get metadata and filter
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < 0 or idx >= len(self.embeddings):
                continue
            
            # Get metadata
            cursor.execute('''
                SELECT embedding_id, timestamp, access_count, importance_score,
                       context, session_id, user_id, emotional_valence, 
                       consolidation_level, metadata
                FROM ltm_entries WHERE embedding_id = ?
            ''', (int(idx),))
            
            row = cursor.fetchone()
            if row is None:
                continue
            
            # Apply filters (using hashed values)
            if min_importance > 0 and row[3] < min_importance:
                continue
            if hashed_session is not None and row[5] != hashed_session:
                continue
            if hashed_user is not None and row[6] != hashed_user:
                continue
            
            # Update access count
            cursor.execute('''
                UPDATE ltm_entries SET access_count = access_count + 1
                WHERE embedding_id = ?
            ''', (int(idx),))
            
            # Decrypt context and metadata
            decrypted_context = self._decrypt_text(row[4]) if row[4] else ""
            decrypted_extra = self._decrypt_metadata(row[9]) if row[9] else {}
            
            metadata = {
                'embedding_id': row[0],
                'timestamp': row[1],
                'access_count': row[2] + 1,
                'importance_score': row[3],
                'context': decrypted_context,
                'session_id': row[5],  # Returns hashed ID (original not recoverable)
                'user_id': row[6],  # Returns hashed ID (original not recoverable)
                'emotional_valence': row[7],
                'consolidation_level': row[8],
                'extra': decrypted_extra
            }
            
            results.append((self.embeddings[idx], metadata, float(dist)))
            
            if len(results) >= k:
                break
        
        conn.commit()
        conn.close()
        
        return results
    
    def retrieve_by_context(
        self,
        context_keywords: List[str],
        k: int = 5
    ) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Retrieve memories matching context keywords.
        
        Args:
            context_keywords: Keywords to search in context
            k: Maximum results
            
        Returns:
            List of (embedding, metadata) tuples
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Build LIKE query for keywords
        conditions = " OR ".join(["context LIKE ?" for _ in context_keywords])
        params = [f"%{kw}%" for kw in context_keywords]
        
        cursor.execute(f'''
            SELECT embedding_id, timestamp, access_count, importance_score,
                   context, session_id, user_id, emotional_valence,
                   consolidation_level, metadata
            FROM ltm_entries
            WHERE {conditions}
            ORDER BY importance_score DESC, timestamp DESC
            LIMIT ?
        ''', params + [k])
        
        results = []
        for row in cursor.fetchall():
            if row[0] < len(self.embeddings):
                metadata = {
                    'embedding_id': row[0],
                    'timestamp': row[1],
                    'access_count': row[2],
                    'importance_score': row[3],
                    'context': row[4],
                    'session_id': row[5],
                    'user_id': row[6],
                    'emotional_valence': row[7],
                    'consolidation_level': row[8],
                    'extra': json.loads(row[9]) if row[9] else {}
                }
                results.append((self.embeddings[row[0]], metadata))
        
        conn.close()
        return results
    
    def save(self) -> None:
        """Save FAISS index and embeddings to disk."""
        try:
            faiss.write_index(self.ltm_index, str(self.index_path))
            np.save(str(self.embeddings_path), np.array(self.embeddings))
            logger.info(f"Saved {self.ltm_index.ntotal} LTM entries to disk")
        except Exception as e:
            logger.error(f"Failed to save LTM data: {e}")
    
    def load(self) -> bool:
        """Load FAISS index and embeddings from disk."""
        try:
            self._load_if_exists()
            return True
        except Exception as e:
            logger.error(f"Failed to load LTM data: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM ltm_entries')
        total_entries = cursor.fetchone()[0]
        
        cursor.execute('SELECT AVG(importance_score) FROM ltm_entries')
        avg_importance = cursor.fetchone()[0] or 0
        
        cursor.execute('SELECT COUNT(DISTINCT session_id) FROM ltm_entries')
        unique_sessions = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(DISTINCT user_id) FROM ltm_entries WHERE user_id IS NOT NULL')
        unique_users = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_entries': total_entries,
            'index_size': self.ltm_index.ntotal,
            'avg_importance': avg_importance,
            'unique_sessions': unique_sessions,
            'unique_users': unique_users,
            'storage_dir': str(self.storage_dir)
        }
    
    def clear(self) -> None:
        """Clear all stored memories."""
        self.ltm_index = faiss.IndexFlatL2(self.d_model)
        self.embeddings = []
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute('DELETE FROM ltm_entries')
        conn.commit()
        conn.close()
        
        # Remove files
        if self.index_path.exists():
            self.index_path.unlink()
        if self.embeddings_path.exists():
            self.embeddings_path.unlink()
        
        logger.info("Cleared all LTM entries")


class MemoryBank:
    def __init__(
        self, 
        memory_size: int, 
        embedding_dim: int, 
        retrieval_k: int,
        enable_persistent_ltm: bool = False,
        ltm_storage_dir: str = "./ltm_storage",
        ltm_auto_save: bool = True
    ):
        """
        Memory Bank using FAISS for efficient retrieval.
        
        Args:
            memory_size: Maximum number of memories to store
            embedding_dim: Dimension of embeddings
            retrieval_k: Number of memories to retrieve
            enable_persistent_ltm: Enable persistent Long-Term Memory storage
            ltm_storage_dir: Directory for LTM persistence
            ltm_auto_save: Auto-save LTM periodically
        """
        self.memory_size = memory_size
        self.embedding_dim = embedding_dim
        self.retrieval_k = retrieval_k
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.values = []
        self.feedback_scores = []
        
        # Persistent LTM storage
        self.enable_persistent_ltm = enable_persistent_ltm
        self.persistent_ltm: Optional[PersistentLTMStorage] = None
        
        if enable_persistent_ltm:
            self.persistent_ltm = PersistentLTMStorage(
                d_model=embedding_dim,
                storage_dir=ltm_storage_dir,
                auto_save=ltm_auto_save
            )
            logger.info("Persistent LTM storage enabled")

    def apply_spiking_attention_jnp(self, x, spike_threshold, epsilon):
        """
        Apply Spiking Attention to JAX array.
        """
        scores = jnp.mean(x, axis=-1, keepdims=True)
        spiking_mask = scores > spike_threshold
        spiked_x = jnp.where(spiking_mask, x, 0.0)
        return spiked_x / (jnp.sum(spiked_x, axis=-1, keepdims=True) + epsilon)
    
    def store_ltm(
        self,
        embedding: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
        context: str = "",
        session_id: str = "default",
        user_id: Optional[str] = None,
        importance_score: float = 0.5
    ) -> Optional[int]:
        """
        Store an embedding in persistent Long-Term Memory.
        
        This enables cross-session recall for personalized applications.
        
        Args:
            embedding: Vector embedding
            metadata: Optional additional metadata
            context: Context description
            session_id: Session identifier for filtering
            user_id: Optional user identifier
            importance_score: Importance score [0, 1]
            
        Returns:
            embedding_id if successful, None if LTM not enabled
        """
        if self.persistent_ltm is None:
            logger.warning("Persistent LTM not enabled")
            return None
        
        return self.persistent_ltm.store_ltm(
            embedding=embedding,
            metadata=metadata,
            context=context,
            session_id=session_id,
            user_id=user_id,
            importance_score=importance_score
        )
    
    def retrieve_ltm(
        self,
        query: np.ndarray,
        k: int = 5,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> List[Tuple[np.ndarray, Dict[str, Any], float]]:
        """
        Retrieve similar memories from persistent Long-Term Memory.
        
        Args:
            query: Query embedding
            k: Number of results
            session_id: Optional filter by session
            user_id: Optional filter by user
            
        Returns:
            List of (embedding, metadata, distance) tuples
        """
        if self.persistent_ltm is None:
            return []
        
        return self.persistent_ltm.retrieve_ltm(
            query=query,
            k=k,
            session_id=session_id,
            user_id=user_id
        )
    
    def consolidate_to_ltm(
        self,
        importance_threshold: float = 0.7,
        context: str = "auto_consolidated",
        session_id: str = "default"
    ) -> int:
        """
        Consolidate high-importance memories from working memory to LTM.
        
        This mimics biological memory consolidation during sleep.
        
        Args:
            importance_threshold: Minimum importance to consolidate
            context: Context for consolidated memories
            session_id: Session identifier
            
        Returns:
            Number of memories consolidated
        """
        if self.persistent_ltm is None or len(self.values) == 0:
            return 0
        
        consolidated = 0
        for i, (value, score) in enumerate(zip(self.values, self.feedback_scores)):
            if score >= importance_threshold:
                self.store_ltm(
                    embedding=np.array(value),
                    context=context,
                    session_id=session_id,
                    importance_score=score,
                    metadata={'consolidation_source': 'working_memory', 'index': i}
                )
                consolidated += 1
        
        logger.info(f"Consolidated {consolidated} memories to LTM")
        return consolidated
    
    def save_ltm(self) -> None:
        """Explicitly save persistent LTM to disk."""
        if self.persistent_ltm is not None:
            self.persistent_ltm.save()
    
    def load_ltm(self) -> bool:
        """Load persistent LTM from disk."""
        if self.persistent_ltm is not None:
            return self.persistent_ltm.load()
        return False
    
    def get_ltm_stats(self) -> Optional[Dict[str, Any]]:
        """Get persistent LTM statistics."""
        if self.persistent_ltm is not None:
            return self.persistent_ltm.get_stats()
        return None

    def store(self, keys, values, feedback_scores=None, spike_threshold=0.1, epsilon=1e-8):
        """
        Stores key-value pairs in the memory bank.
        :param keys: Batch of mean-pooled embeddings (shape: [batch_size, embedding_dim])
        :param values: Corresponding transformer activations (shape: [batch_size, embedding_dim])
        """
        # **Ensure computation is complete before conversion**
        keys = jax.block_until_ready(keys)
        values = jax.block_until_ready(values)

        # **Detach from JAX before passing to FAISS -- Apply Spiking Attention to keys before storage**
        keys = self.apply_spiking_attention_jnp(keys, spike_threshold, epsilon)
        keys_np = np.asarray(jax.device_get(keys), dtype=np.float32)
        values_np = np.asarray(jax.device_get(values), dtype=np.float32)

        # **Ensure 2D shape**
        if keys_np.ndim == 1:
            keys_np = keys_np.reshape(1, -1)
        if values_np.ndim == 1:
            values_np = values_np.reshape(1, -1)

        assert keys_np.shape[1] == self.embedding_dim, f"Expected dim {self.embedding_dim}, got {keys_np.shape[1]}"
        if len(self.values) + len(keys_np) > self.memory_size:
            remove_count = len(self.values) + len(keys_np) - self.memory_size
            self.values = self.values[remove_count:]
            self.index.reset()
            self.index.add(np.asarray(self.values, dtype=np.float32))

        self.values.extend(values_np.tolist())
        self.feedback_scores.extend([feedback_scores] * len(keys_np) if feedback_scores else [0.0] * len(keys_np))
        self.index.add(keys_np)

    def retrieve(self, queries_np, spike_threshold=0.1, epsilon=1e-8):
        """
        Retrieves closest memory values.
        :param queries: Query embeddings (shape: [batch_size, embedding_dim])
        :return: Retrieved memory activations (shape: [batch_size, embedding_dim])
        """
        queries_np = jax.block_until_ready(queries_np)
        queries_np = self.apply_spiking_attention_jnp(queries_np, spike_threshold, epsilon)
        queries_np = np.asarray(jax.device_get(queries_np), dtype=np.float32)
        if queries_np.ndim == 1:
            queries_np = queries_np.reshape(1, -1)
        assert queries_np.shape[1] == self.embedding_dim
        if self.index.ntotal == 0:
            return np.zeros((queries_np.shape[0], self.embedding_dim), dtype=np.float32)
        distances, indices = self.index.search(queries_np, self.retrieval_k)
        retrieved_values = np.array([self.values[idx] for idx in indices.flatten()]).reshape(
            indices.shape[0], indices.shape[1], self.embedding_dim
        )
        norms = np.linalg.norm(retrieved_values, axis=-1, keepdims=True) + epsilon
        retrieved_values = retrieved_values / norms
        noise = np.random.normal(0, 0.1, retrieved_values.shape).astype(np.float32)
        retrieved_values = retrieved_values + noise
        final_norms = np.linalg.norm(retrieved_values, axis=-1, keepdims=True) + epsilon
        retrieved_values = retrieved_values / final_norms
        #logger.info(f"[MemoryBank] Retrieved norm: {np.linalg.norm(retrieved_values.mean(axis=1)):.4f}")
        return np.mean(retrieved_values, axis=1)
    
class ShortTermMemory:
    def __init__(self, buffer_size: int, embedding_dim: int):
        """Short-Term Memory as a per-batch buffer."""
        self.buffer_size = buffer_size  
        self.embedding_dim = embedding_dim
        self.buffer = []
        self.feedback_scores = []  

    def apply_spiking_attention_jnp(self, x, spike_threshold, epsilon):
        """
        Apply Spiking Attention to JAX array.
        """
        scores = jnp.mean(x, axis=-1, keepdims=True)
        spiking_mask = scores > spike_threshold
        spiked_x = jnp.where(spiking_mask, x, 0.0)
        return spiked_x / (jnp.sum(spiked_x, axis=-1, keepdims=True) + epsilon)
    
    def store(self, keys, values, spike_threshold=0.1, epsilon=1e-8):
        """Store embeddings for the current batch, resetting each call."""
        keys = jax.block_until_ready(keys)
        values = jax.block_until_ready(values)
        keys = self.apply_spiking_attention_jnp(keys, spike_threshold, epsilon)
        keys_np = np.asarray(jax.device_get(keys), dtype=np.float32)
        values_np = np.asarray(jax.device_get(values), dtype=np.float32)

        if keys_np.ndim == 1:
            keys_np = keys_np.reshape(1, -1)
        if values_np.ndim == 1:
            values_np = values_np.reshape(1, -1)

        assert keys_np.shape[1] == self.embedding_dim
        self.buffer = list(values_np[:self.buffer_size])  

    def retrieve(self, queries_np, spike_threshold=0.1, epsilon=1e-8):
        """Retrieve all embeddings from the current batch buffer."""
        queries_np = jax.block_until_ready(queries_np)
        queries_np = self.apply_spiking_attention_jnp(queries_np, spike_threshold, epsilon)
        queries_np = np.asarray(jax.device_get(queries_np), dtype=np.float32)
        if queries_np.ndim == 1:
            queries_np = queries_np.reshape(1, -1)
        assert queries_np.shape[1] == self.embedding_dim
        if not self.buffer:
            return np.zeros((queries_np.shape[0], self.embedding_dim), dtype=np.float32)
        retrieved_values = np.mean(self.buffer, axis=0, keepdims=True)
        norms = np.linalg.norm(retrieved_values, axis=-1, keepdims=True) + epsilon
        retrieved_values = retrieved_values / norms
        return np.repeat(retrieved_values, queries_np.shape[0], axis=0)
    
class MidTermMemory:
    def __init__(self, buffer_size: int, embedding_dim: int, retention_steps: int):
        self.buffer_size = buffer_size  
        self.embedding_dim = embedding_dim
        self.buffer = []
        self.feedback_scores = []  
        self.step_count = 0
        self.retention_steps = retention_steps  

    def apply_spiking_attention_jnp(self, x, spike_threshold, epsilon):
        """
        Apply Spiking Attention to JAX array.
        """
        scores = jnp.mean(x, axis=-1, keepdims=True)
        spiking_mask = scores > spike_threshold
        spiked_x = jnp.where(spiking_mask, x, 0.0)
        return spiked_x / (jnp.sum(spiked_x, axis=-1, keepdims=True) + epsilon)
    
    def store(self, keys, values, spike_threshold=0.1, epsilon=1e-8):
        keys = jax.block_until_ready(keys)
        values = jax.block_until_ready(values)
        keys = self.apply_spiking_attention_jnp(keys, spike_threshold, epsilon)
        keys_np = np.asarray(jax.device_get(keys), dtype=np.float32)
        values_np = np.asarray(jax.device_get(values), dtype=np.float32)
        if keys_np.ndim == 1:
            keys_np = keys_np.reshape(1, -1)
        if values_np.ndim == 1:
            values_np = values_np.reshape(1, -1)
        assert keys_np.shape[1] == self.embedding_dim

        self.buffer.extend(values_np)
        self.buffer = self.buffer[-self.buffer_size:]
        self.step_count += 1

    def retrieve(self, queries_np, spike_threshold=0.1, epsilon=1e-8):
        queries_np = jax.block_until_ready(queries_np)
        queries_np = self.apply_spiking_attention_jnp(queries_np, spike_threshold, epsilon)
        queries_np = np.asarray(jax.device_get(queries_np), dtype=np.float32)
        if queries_np.ndim == 1:
            queries_np = queries_np.reshape(1, -1)
        assert queries_np.shape[1] == self.embedding_dim
        if not self.buffer or self.step_count % self.retention_steps == 0:
            self.buffer = []
            return np.zeros((queries_np.shape[0], self.embedding_dim), dtype=np.float32)
        retrieved_values = np.mean(self.buffer, axis=0, keepdims=True)
        norms = np.linalg.norm(retrieved_values, axis=-1, keepdims=True) + epsilon
        retrieved_values = retrieved_values / norms
        return np.repeat(retrieved_values, queries_np.shape[0], axis=0)