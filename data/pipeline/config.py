"""
Pipeline Configuration Module

Defines all configuration classes for the RT-DLM data pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
from pathlib import Path
import os


class StorageBackend(Enum):
    """Supported storage backends."""
    LOCAL = "local"
    S3 = "s3"
    GCS = "gcs"
    AZURE = "azure"


class TokenizerType(Enum):
    """Tokenizer types."""
    BPE = "bpe"
    UNIGRAM = "unigram"
    WORDPIECE = "wordpiece"
    SENTENCEPIECE = "sentencepiece"


@dataclass
class QualityConfig:
    """Configuration for data quality filtering."""
    # Text quality
    min_text_length: int = 50
    max_text_length: int = 100000
    min_word_count: int = 10
    max_word_count: int = 50000
    min_unique_words_ratio: float = 0.1
    max_special_char_ratio: float = 0.3
    
    # Language filtering
    allowed_languages: List[str] = field(default_factory=lambda: ["en"])
    detect_language: bool = True
    
    # Content filtering
    remove_pii: bool = True
    remove_toxicity: bool = True
    toxicity_threshold: float = 0.7
    
    # Deduplication
    deduplicate: bool = True
    dedup_method: str = "minhash"  # "exact", "minhash", "simhash"
    minhash_threshold: float = 0.8
    minhash_num_perm: int = 128
    
    # Image quality
    min_image_size: Tuple[int, int] = (64, 64)
    max_image_size: Tuple[int, int] = (4096, 4096)
    min_image_quality: float = 0.3  # Sharpness threshold
    
    # Audio quality  
    min_audio_duration: float = 0.5  # seconds
    max_audio_duration: float = 3600.0  # 1 hour
    min_audio_sample_rate: int = 8000


@dataclass
class ShardConfig:
    """Configuration for data sharding."""
    # Shard sizing
    max_shard_size_bytes: int = 1_000_000_000  # 1 GB
    max_tokens_per_shard: int = 10_000_000  # 10M tokens
    max_samples_per_shard: int = 100_000
    
    # Sequence packing
    sequence_length: int = 2048
    pack_sequences: bool = True  # Pack multiple short sequences
    
    # Format
    format: str = "safetensors"  # "safetensors", "numpy", "arrow"
    compression: Optional[str] = None  # "zstd", "lz4", "gzip"
    
    # Storage
    output_dir: str = "data/shards"
    index_file: str = "shard_index.json"
    
    # Splitting
    train_ratio: float = 0.95
    val_ratio: float = 0.04
    test_ratio: float = 0.01


@dataclass  
class TokenizationPipelineConfig:
    """Configuration for tokenization in the pipeline."""
    # Text tokenizer
    tokenizer_type: TokenizerType = TokenizerType.BPE
    vocab_size: int = 50000
    tokenizer_path: str = "data/rt_dlm_sp"
    
    # Special tokens
    pad_token_id: int = 0
    unk_token_id: int = 1
    bos_token_id: int = 2
    eos_token_id: int = 3
    sep_token_id: int = 4
    
    # Modality tokens (reserved range 10-50)
    modality_token_start: int = 10
    
    # Image tokenization
    image_encoder: str = "vit"  # "vit", "clip", "resnet"
    image_size: Tuple[int, int] = (224, 224)
    image_patch_size: int = 16
    image_embed_dim: int = 768
    
    # Audio tokenization
    audio_encoder: str = "wav2vec"  # "wav2vec", "whisper", "mfcc"
    audio_sample_rate: int = 16000
    audio_embed_dim: int = 768
    
    # Video tokenization
    video_fps: int = 8
    max_video_frames: int = 64


@dataclass
class IngestionConfig:
    """Configuration for data ingestion."""
    # Source directories
    raw_data_dir: str = "data/raw"
    staging_dir: str = "data/staging"
    
    # Cloud storage (optional)
    storage_backend: StorageBackend = StorageBackend.LOCAL
    s3_bucket: Optional[str] = None
    gcs_bucket: Optional[str] = None
    
    # Web scraping
    max_concurrent_downloads: int = 10
    download_timeout: float = 30.0
    respect_robots_txt: bool = True
    user_agent: str = "RT-DLM-DataPipeline/1.0"
    
    # Rate limiting
    requests_per_second: float = 5.0
    retry_attempts: int = 3
    retry_delay: float = 1.0
    
    # File handling
    supported_extensions: List[str] = field(default_factory=lambda: [
        # Text
        ".txt", ".md", ".rst", ".csv", ".json", ".jsonl",
        # Code
        ".py", ".js", ".ts", ".java", ".cpp", ".c", ".h", ".go", ".rs",
        # Documents
        ".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx",
        # Web
        ".html", ".htm", ".xml",
        # Images
        ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp",
        # Audio
        ".mp3", ".wav", ".flac", ".ogg", ".m4a",
        # Video
        ".mp4", ".avi", ".mkv", ".mov", ".webm",
    ])


@dataclass
class ProcessingConfig:
    """Configuration for data processing."""
    # Parallelism
    num_workers: int = field(default_factory=lambda: os.cpu_count() or 4)
    batch_size: int = 1000
    use_multiprocessing: bool = True
    
    # Memory management
    max_memory_gb: float = 16.0
    streaming_mode: bool = True  # Process without loading all data
    
    # Progress tracking
    log_interval: int = 1000
    save_checkpoints: bool = True
    checkpoint_interval: int = 10000
    
    # Error handling
    skip_errors: bool = True
    max_errors: int = 1000
    error_log_file: str = "data/pipeline_errors.log"


@dataclass
class PipelineConfig:
    """Master configuration for the entire data pipeline."""
    # Sub-configurations
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    tokenization: TokenizationPipelineConfig = field(default_factory=TokenizationPipelineConfig)
    sharding: ShardConfig = field(default_factory=ShardConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    
    # Pipeline settings
    name: str = "rt_dlm_data_pipeline"
    version: str = "1.0.0"
    
    # Directories
    base_dir: str = "data"
    cache_dir: str = "data/.cache"
    temp_dir: str = "data/.temp"
    
    # Manifest
    manifest_file: str = "data/manifest.json"
    
    # Modes
    dry_run: bool = False
    verbose: bool = True
    
    def __post_init__(self):
        """Create necessary directories."""
        dirs = [
            self.base_dir,
            self.cache_dir,
            self.temp_dir,
            self.ingestion.raw_data_dir,
            self.ingestion.staging_dir,
            self.sharding.output_dir,
        ]
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        """Load configuration from YAML file."""
        import yaml
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls._from_dict(data)
    
    @classmethod
    def from_json(cls, path: str) -> "PipelineConfig":
        """Load configuration from JSON file."""
        import json
        with open(path, "r") as f:
            data = json.load(f)
        return cls._from_dict(data)
    
    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
        """Create config from dictionary."""
        ingestion = IngestionConfig(**data.get("ingestion", {}))
        quality = QualityConfig(**data.get("quality", {}))
        tokenization = TokenizationPipelineConfig(**data.get("tokenization", {}))
        sharding = ShardConfig(**data.get("sharding", {}))
        processing = ProcessingConfig(**data.get("processing", {}))
        
        return cls(
            ingestion=ingestion,
            quality=quality,
            tokenization=tokenization,
            sharding=sharding,
            processing=processing,
            **{k: v for k, v in data.items() 
               if k not in ["ingestion", "quality", "tokenization", "sharding", "processing"]}
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        from dataclasses import asdict
        return asdict(self)
    
    def save(self, path: str):
        """Save configuration to file."""
        import json
        
        # Convert enums to strings
        data = self.to_dict()
        
        def convert_enums(obj):
            if isinstance(obj, dict):
                return {k: convert_enums(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_enums(v) for v in obj]
            elif isinstance(obj, Enum):
                return obj.value
            return obj
        
        data = convert_enums(data)
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


# Preset configurations for common use cases
def create_small_scale_config() -> PipelineConfig:
    """Configuration for small-scale testing (< 10GB data)."""
    return PipelineConfig(
        processing=ProcessingConfig(
            num_workers=4,
            batch_size=100,
            max_memory_gb=4.0,
        ),
        sharding=ShardConfig(
            max_shard_size_bytes=100_000_000,  # 100 MB
            max_tokens_per_shard=1_000_000,
            sequence_length=1024,
        ),
        quality=QualityConfig(
            deduplicate=True,
            remove_pii=True,
            remove_toxicity=False,  # Skip for speed
        ),
    )


def create_production_config() -> PipelineConfig:
    """Configuration for production-scale processing (100GB+ data)."""
    return PipelineConfig(
        processing=ProcessingConfig(
            num_workers=os.cpu_count() or 8,
            batch_size=1000,
            max_memory_gb=64.0,
            streaming_mode=True,
        ),
        sharding=ShardConfig(
            max_shard_size_bytes=1_000_000_000,  # 1 GB
            max_tokens_per_shard=10_000_000,
            sequence_length=2048,
            compression="zstd",
        ),
        quality=QualityConfig(
            deduplicate=True,
            remove_pii=True,
            remove_toxicity=True,
            minhash_threshold=0.85,
        ),
        tokenization=TokenizationPipelineConfig(
            vocab_size=50000,
            image_encoder="clip",
            audio_encoder="whisper",
        ),
    )


def create_multimodal_config() -> PipelineConfig:
    """Configuration optimized for multimodal data."""
    return PipelineConfig(
        processing=ProcessingConfig(
            num_workers=8,
            batch_size=500,
            max_memory_gb=32.0,
        ),
        sharding=ShardConfig(
            max_shard_size_bytes=500_000_000,  # 500 MB (larger due to embeddings)
            sequence_length=4096,  # Longer for multimodal
        ),
        tokenization=TokenizationPipelineConfig(
            vocab_size=50000,
            image_encoder="clip",
            image_embed_dim=768,
            audio_encoder="whisper",
            audio_embed_dim=768,
            video_fps=4,  # Lower fps for efficiency
            max_video_frames=32,
        ),
    )
