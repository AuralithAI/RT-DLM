"""
Comprehensive tests for the RT-DLM production data pipeline.

Tests cover:
- Pipeline configuration
- Data ingestion from various sources
- Preprocessing (deduplication, quality filtering)
- Content extraction from various formats
- Sharding and storage
- Training dataloader
"""

import pytest
import tempfile
import shutil
import json
import numpy as np
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from dataclasses import asdict
import sys
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestPipelineConfig:
    """Test pipeline configuration classes."""
    
    def test_pipeline_config_defaults(self):
        """Test default configuration values."""
        from data.pipeline.config import PipelineConfig
        
        config = PipelineConfig()
        
        assert config.name == "rt_dlm_data_pipeline"
        assert config.version == "1.0.0"
        assert config.dry_run is False
        assert config.verbose is True
        
    def test_quality_config_defaults(self):
        """Test quality configuration defaults."""
        from data.pipeline.config import QualityConfig
        
        config = QualityConfig()
        
        assert config.min_text_length == 50
        assert config.max_text_length == 100000
        assert config.remove_pii is True
        assert config.deduplicate is True
        assert config.minhash_threshold == 0.8
        
    def test_shard_config_defaults(self):
        """Test shard configuration defaults."""
        from data.pipeline.config import ShardConfig
        
        config = ShardConfig()
        
        assert config.max_shard_size_bytes == 1_000_000_000
        assert config.sequence_length == 2048
        assert config.pack_sequences is True
        assert config.format == "safetensors"
        assert config.train_ratio == 0.95
        
    def test_ingestion_config_defaults(self):
        """Test ingestion configuration defaults."""
        from data.pipeline.config import IngestionConfig
        
        config = IngestionConfig()
        
        assert config.max_concurrent_downloads == 10
        assert config.download_timeout == 30.0
        assert config.respect_robots_txt is True
        assert ".txt" in config.supported_extensions
        assert ".pdf" in config.supported_extensions
        
    def test_processing_config_defaults(self):
        """Test processing configuration defaults."""
        from data.pipeline.config import ProcessingConfig
        
        config = ProcessingConfig()
        
        assert config.num_workers >= 1
        assert config.batch_size == 1000
        assert config.streaming_mode is True
        assert config.skip_errors is True
        
    def test_tokenization_config_defaults(self):
        """Test tokenization configuration defaults."""
        from data.pipeline.config import TokenizationPipelineConfig, TokenizerType
        
        config = TokenizationPipelineConfig()
        
        assert config.vocab_size == 50000
        assert config.tokenizer_type == TokenizerType.BPE
        assert config.image_size == (224, 224)
        assert config.audio_sample_rate == 16000
        
    def test_pipeline_config_custom_values(self):
        """Test configuration with custom values."""
        from data.pipeline.config import (
            PipelineConfig, QualityConfig, 
            ShardConfig, ProcessingConfig
        )
        
        config = PipelineConfig(
            name="custom_pipeline",
            dry_run=True,
            quality=QualityConfig(minhash_threshold=0.9),
            sharding=ShardConfig(sequence_length=4096),
            processing=ProcessingConfig(batch_size=500)
        )
        
        assert config.name == "custom_pipeline"
        assert config.dry_run is True
        assert config.quality.minhash_threshold == 0.9
        assert config.sharding.sequence_length == 4096
        assert config.processing.batch_size == 500
        
    def test_create_small_scale_config(self):
        """Test preset small scale configuration."""
        from data.pipeline.config import create_small_scale_config
        
        config = create_small_scale_config()
        
        assert config.processing.num_workers == 4
        assert config.processing.max_memory_gb == 4.0
        assert config.sharding.max_shard_size_bytes == 100_000_000
        
    def test_create_production_config(self):
        """Test preset production configuration."""
        from data.pipeline.config import create_production_config
        
        config = create_production_config()
        
        assert config.sharding.max_shard_size_bytes == 1_000_000_000
        assert config.sharding.compression == "zstd"
        assert config.quality.remove_toxicity is True
        
    def test_create_multimodal_config(self):
        """Test preset multimodal configuration."""
        from data.pipeline.config import create_multimodal_config
        
        config = create_multimodal_config()
        
        assert config.sharding.sequence_length == 4096
        assert config.tokenization.image_encoder == "clip"
        assert config.tokenization.audio_encoder == "whisper"


class TestDataIngestion:
    """Test data ingestion functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp = tempfile.mkdtemp()
        yield Path(temp)
        shutil.rmtree(temp, ignore_errors=True)
        
    def test_source_type_enum(self):
        """Test SourceType enumeration."""
        from data.pipeline.ingestion import SourceType
        
        assert SourceType.FILE.value == "file"
        assert SourceType.DIRECTORY.value == "directory"
        assert SourceType.URL.value == "url"
        assert SourceType.S3.value == "s3"
        assert SourceType.HUGGINGFACE.value == "huggingface"
        
    def test_data_source_creation(self):
        """Test DataSource dataclass creation."""
        from data.pipeline.ingestion import DataSource, SourceType
        
        source = DataSource(
            path="/data/texts",
            source_type=SourceType.DIRECTORY,
            metadata={"category": "text"}
        )
        
        assert source.path == "/data/texts"
        assert source.source_type == SourceType.DIRECTORY
        assert source.processed is False
        assert source.error is None
        
    def test_ingested_item_properties(self):
        """Test IngestedItem properties."""
        from data.pipeline.ingestion import DataSource, SourceType, IngestedItem
        
        source = DataSource(path="/test.txt", source_type=SourceType.FILE)
        
        text_item = IngestedItem(
            source=source,
            content="Hello world",
            content_type="text/plain",
            metadata={}
        )
        
        assert text_item.is_text is True
        assert text_item.is_binary is False
        
        binary_item = IngestedItem(
            source=source,
            content=b"\x89PNG",
            content_type="image/png",
            metadata={}
        )
        
        assert binary_item.is_text is False
        assert binary_item.is_binary is True
        
    def test_data_ingester_initialization(self):
        """Test DataIngester initialization."""
        from data.pipeline.config import PipelineConfig
        from data.pipeline.ingestion import DataIngester
        
        config = PipelineConfig()
        ingester = DataIngester(config)
        
        assert ingester.config == config
        assert ingester.ingestion_config == config.ingestion


class TestPreprocessing:
    """Test data preprocessing functionality."""
    
    def test_deduplication_method_enum(self):
        """Test DeduplicationMethod enum."""
        from data.pipeline.preprocessing import DeduplicationMethod
        
        assert DeduplicationMethod.EXACT.value == "exact"
        assert DeduplicationMethod.MINHASH.value == "minhash"
        assert DeduplicationMethod.SIMHASH.value == "simhash"
        
    def test_data_preprocessor_initialization(self):
        """Test DataPreprocessor initialization."""
        from data.pipeline.config import PipelineConfig
        from data.pipeline.preprocessing import DataPreprocessor
        
        config = PipelineConfig()
        preprocessor = DataPreprocessor(config)
        
        assert preprocessor.config == config
        
    def test_quality_metrics(self):
        """Test quality metrics from config."""
        from data.pipeline.config import QualityConfig
        
        config = QualityConfig(
            min_text_length=100,
            max_text_length=10000,
            remove_pii=True,
            toxicity_threshold=0.8
        )
        
        assert config.min_text_length == 100
        assert config.toxicity_threshold == 0.8


class TestContentExtraction:
    """Test content extraction functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp = tempfile.mkdtemp()
        yield Path(temp)
        shutil.rmtree(temp, ignore_errors=True)
        
    def test_extracted_content_dataclass(self):
        """Test ExtractedContent dataclass."""
        from data.pipeline.extraction import ExtractedContent
        
        content = ExtractedContent(
            content_type="text",
            text="Hello world",
            source_path="/path/to/file.txt"
        )
        
        assert content.content_type == "text"
        assert content.text == "Hello world"
        
    def test_content_extractor_initialization(self):
        """Test ContentExtractor initialization."""
        from data.pipeline.config import PipelineConfig
        from data.pipeline.extraction import ContentExtractor
        
        config = PipelineConfig()
        extractor = ContentExtractor(config)
        
        assert extractor.config == config


class TestSharding:
    """Test data sharding functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp = tempfile.mkdtemp()
        yield Path(temp)
        shutil.rmtree(temp, ignore_errors=True)
        
    def test_shard_index_dataclass(self):
        """Test ShardIndex dataclass."""
        from data.pipeline.sharding import ShardIndex
        
        index = ShardIndex(
            num_shards=10,
            total_samples=100000,
            total_tokens=10000000
        )
        
        assert index.num_shards == 10
        assert index.total_samples == 100000
        
    def test_shard_writer_initialization(self, temp_dir):
        """Test ShardWriter initialization."""
        from data.pipeline.config import PipelineConfig, ShardConfig
        from data.pipeline.sharding import ShardWriter
        
        config = PipelineConfig(
            sharding=ShardConfig(output_dir=str(temp_dir))
        )
        writer = ShardWriter(config)
        
        assert writer.config == config
        
    def test_shard_reader_initialization(self, temp_dir):
        """Test ShardReader initialization."""
        from data.pipeline.sharding import ShardReader
        
        reader = ShardReader(str(temp_dir))
        
        assert reader.shard_dir == temp_dir


class TestDataLoader:
    """Test training dataloader functionality."""
    
    def test_training_batch_dataclass(self):
        """Test TrainingBatch dataclass."""
        from data.pipeline.dataloader import TrainingBatch
        
        batch = TrainingBatch(
            input_ids=np.array([[1, 2, 3], [4, 5, 6]]),
            attention_mask=np.array([[1, 1, 1], [1, 1, 1]]),
            batch_idx=0,
            shard_ids=["shard_001", "shard_001"]
        )
        
        assert batch.input_ids.shape == (2, 3)
        assert batch.batch_idx == 0
        
    def test_sharded_dataloader_initialization(self):
        """Test ShardedDataLoader initialization."""
        from data.pipeline.config import PipelineConfig
        from data.pipeline.dataloader import ShardedDataLoader
        
        config = PipelineConfig()
        loader = ShardedDataLoader(config)
        
        assert loader.config == config


class TestPipelineIntegration:
    """Test full pipeline integration."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp = tempfile.mkdtemp()
        yield Path(temp)
        shutil.rmtree(temp, ignore_errors=True)
        
    def test_pipeline_status_enum(self):
        """Test PipelineStatus enum."""
        from data.pipeline.pipeline import PipelineStatus
        
        assert PipelineStatus.IDLE.value == "idle"
        assert PipelineStatus.RUNNING.value == "running"
        assert PipelineStatus.COMPLETED.value == "completed"
        assert PipelineStatus.FAILED.value == "failed"
        
    def test_data_pipeline_initialization(self, temp_dir):
        """Test DataPipeline initialization."""
        from data.pipeline.config import PipelineConfig
        from data.pipeline.pipeline import DataPipeline
        
        config = PipelineConfig(base_dir=str(temp_dir))
        pipeline = DataPipeline(config)
        
        assert pipeline.config == config


class TestPackageImports:
    """Test that all pipeline modules can be imported."""
    
    def test_import_config(self):
        """Test config module import."""
        from data.pipeline.config import (
            PipelineConfig,
            QualityConfig,
            ShardConfig,
            IngestionConfig,
            ProcessingConfig,
            TokenizationPipelineConfig,
            StorageBackend,
            TokenizerType
        )
        
    def test_import_ingestion(self):
        """Test ingestion module import."""
        from data.pipeline.ingestion import (
            DataIngester,
            DataSource,
            SourceType,
            IngestedItem
        )
        
    def test_import_preprocessing(self):
        """Test preprocessing module import."""
        from data.pipeline.preprocessing import (
            DataPreprocessor,
            DeduplicationMethod
        )
        
    def test_import_extraction(self):
        """Test extraction module import."""
        from data.pipeline.extraction import (
            ContentExtractor,
            ExtractedContent
        )
        
    def test_import_sharding(self):
        """Test sharding module import."""
        from data.pipeline.sharding import (
            ShardWriter,
            ShardReader,
            ShardIndex
        )
        
    def test_import_dataloader(self):
        """Test dataloader module import."""
        from data.pipeline.dataloader import (
            ShardedDataLoader,
            TrainingBatch
        )
        
    def test_import_pipeline(self):
        """Test main pipeline module import."""
        from data.pipeline.pipeline import (
            DataPipeline,
            PipelineStatus
        )
        
    def test_import_from_package(self):
        """Test imports from package __init__."""
        from data.pipeline import (
            PipelineConfig,
            ShardConfig,
            QualityConfig,
            DataIngester,
            DataSource,
            SourceType,
            DataPreprocessor,
            DeduplicationMethod,
            ContentExtractor,
            ExtractedContent,
            ShardWriter,
            ShardReader,
            ShardIndex,
            ShardedDataLoader,
            TrainingBatch,
            DataPipeline,
            PipelineStatus
        )


class TestConfigurationFactories:
    """Test configuration factory functions."""
    
    def test_small_scale_config_validity(self):
        """Test small scale config is valid."""
        from data.pipeline.config import create_small_scale_config
        
        config = create_small_scale_config()
        
        assert config.ingestion is not None
        assert config.quality is not None
        assert config.sharding is not None
        assert config.processing is not None
        assert config.tokenization is not None
        
    def test_production_config_validity(self):
        """Test production config is valid."""
        from data.pipeline.config import create_production_config
        
        config = create_production_config()
        
        assert config.sharding.max_shard_size_bytes >= 1_000_000_000
        assert config.processing.streaming_mode is True
        
    def test_multimodal_config_validity(self):
        """Test multimodal config is valid."""
        from data.pipeline.config import create_multimodal_config
        
        config = create_multimodal_config()
        
        assert config.tokenization.image_encoder in ["vit", "clip", "resnet"]
        assert config.tokenization.audio_encoder in ["wav2vec", "whisper", "mfcc"]


class TestStorageBackend:
    """Test storage backend configuration."""
    
    def test_storage_backend_enum(self):
        """Test StorageBackend enumeration."""
        from data.pipeline.config import StorageBackend
        
        assert StorageBackend.LOCAL.value == "local"
        assert StorageBackend.S3.value == "s3"
        assert StorageBackend.GCS.value == "gcs"
        assert StorageBackend.AZURE.value == "azure"
        
    def test_ingestion_with_storage_backend(self):
        """Test ingestion config with different backends."""
        from data.pipeline.config import IngestionConfig, StorageBackend
        
        local_config = IngestionConfig(storage_backend=StorageBackend.LOCAL)
        assert local_config.storage_backend == StorageBackend.LOCAL
        
        s3_config = IngestionConfig(
            storage_backend=StorageBackend.S3,
            s3_bucket="my-bucket"
        )
        assert s3_config.storage_backend == StorageBackend.S3
        assert s3_config.s3_bucket == "my-bucket"


class TestTokenizerType:
    """Test tokenizer type configuration."""
    
    def test_tokenizer_type_enum(self):
        """Test TokenizerType enumeration."""
        from data.pipeline.config import TokenizerType
        
        assert TokenizerType.BPE.value == "bpe"
        assert TokenizerType.UNIGRAM.value == "unigram"
        assert TokenizerType.WORDPIECE.value == "wordpiece"
        assert TokenizerType.SENTENCEPIECE.value == "sentencepiece"
        
    def test_tokenization_with_type(self):
        """Test tokenization config with different types."""
        from data.pipeline.config import TokenizationPipelineConfig, TokenizerType
        
        bpe_config = TokenizationPipelineConfig(tokenizer_type=TokenizerType.BPE)
        assert bpe_config.tokenizer_type == TokenizerType.BPE
        
        sp_config = TokenizationPipelineConfig(tokenizer_type=TokenizerType.SENTENCEPIECE)
        assert sp_config.tokenizer_type == TokenizerType.SENTENCEPIECE


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp = tempfile.mkdtemp()
        yield Path(temp)
        shutil.rmtree(temp, ignore_errors=True)
        
    def test_empty_data_source(self):
        """Test handling of empty data source."""
        from data.pipeline.ingestion import DataSource, SourceType
        
        source = DataSource(
            path="",
            source_type=SourceType.FILE
        )
        
        assert source.path == ""
        assert source.processed is False
        
    def test_quality_config_boundary_values(self):
        """Test quality config with boundary values."""
        from data.pipeline.config import QualityConfig
        
        config = QualityConfig(
            min_text_length=0,
            max_text_length=1,
            minhash_threshold=0.0
        )
        
        assert config.min_text_length == 0
        assert config.minhash_threshold == 0.0
        
        config2 = QualityConfig(
            minhash_threshold=1.0,
            toxicity_threshold=1.0
        )
        
        assert config2.minhash_threshold == 1.0
        assert config2.toxicity_threshold == 1.0


class TestSerialization:
    """Test configuration serialization."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp = tempfile.mkdtemp()
        yield Path(temp)
        shutil.rmtree(temp, ignore_errors=True)
        
    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        from data.pipeline.config import PipelineConfig
        
        config = PipelineConfig(name="test_pipeline")
        data = config.to_dict()
        
        assert isinstance(data, dict)
        assert data["name"] == "test_pipeline"
        
    def test_config_save_and_load(self, temp_dir):
        """Test saving and loading config."""
        from data.pipeline.config import PipelineConfig
        
        config = PipelineConfig(
            name="serialization_test",
            verbose=False
        )
        
        save_path = str(temp_dir / "config.json")
        config.save(save_path)
        
        assert Path(save_path).exists()
        
        loaded = PipelineConfig.from_json(save_path)
        assert loaded.name == config.name
        assert loaded.verbose == config.verbose


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
