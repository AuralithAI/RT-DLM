"""
RT-DLM Production Data Pipeline

A scalable, modular pipeline for processing multimodal data into training-ready shards.

Pipeline Stages:
    1. Ingestion - Collect data from various sources
    2. Preprocessing - Clean, deduplicate, filter
    3. Extraction - Convert file formats to raw content
    4. Tokenization - Convert to token IDs/embeddings
    5. Sharding - Split into efficient binary shards
    6. Loading - Stream shards for training
"""

from .config import PipelineConfig, ShardConfig, QualityConfig
from .ingestion import DataIngester, DataSource, SourceType
from .preprocessing import DataPreprocessor, DeduplicationMethod
from .extraction import ContentExtractor, ExtractedContent
from .sharding import ShardWriter, ShardReader, ShardIndex
from .dataloader import ShardedDataLoader, TrainingBatch
from .pipeline import DataPipeline, PipelineStatus

__all__ = [
    # Config
    "PipelineConfig",
    "ShardConfig", 
    "QualityConfig",
    # Ingestion
    "DataIngester",
    "DataSource",
    "SourceType",
    # Preprocessing
    "DataPreprocessor",
    "DeduplicationMethod",
    # Extraction
    "ContentExtractor",
    "ExtractedContent",
    # Sharding
    "ShardWriter",
    "ShardReader",
    "ShardIndex",
    # DataLoader
    "ShardedDataLoader",
    "TrainingBatch",
    # Pipeline
    "DataPipeline",
    "PipelineStatus",
]
