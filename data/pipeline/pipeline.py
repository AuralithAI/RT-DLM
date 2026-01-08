"""
Data Pipeline Orchestrator

Main entry point that ties all pipeline components together:
- Ingestion → Preprocessing → Extraction → Tokenization → Sharding

Supports:
- Full pipeline execution
- Stage-by-stage execution
- Resumable processing
- Progress tracking
- CLI interface
"""

import os
import sys
import json
import logging
import argparse
import time
from pathlib import Path
from typing import List, Dict, Optional, Any, Generator
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import random

import numpy as np

from .config import (
    PipelineConfig, 
    create_small_scale_config,
    create_production_config,
    create_multimodal_config,
)
from .ingestion import DataIngester, DataSource, IngestedItem
from .preprocessing import DataPreprocessor, PreprocessedItem
from .extraction import ContentExtractor, ExtractedContent
from .sharding import ShardWriter, TokenizedSample, ShardIndex

# Try to import tokenizer
try:
    from modules.tokenization.multimodal_tokenizer import MultiModalTokenizer
    HAS_TOKENIZER = True
except ImportError:
    HAS_TOKENIZER = False

logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """Pipeline execution stages."""
    INGEST = "ingest"
    PREPROCESS = "preprocess"
    EXTRACT = "extract"
    TOKENIZE = "tokenize"
    SHARD = "shard"
    COMPLETE = "complete"


@dataclass
class PipelineStatus:
    """Status of pipeline execution."""
    stage: PipelineStage
    started_at: str
    updated_at: str
    
    # Counts
    total_sources: int = 0
    processed_sources: int = 0
    
    ingested_items: int = 0
    preprocessed_items: int = 0
    extracted_items: int = 0
    tokenized_items: int = 0
    
    total_tokens: int = 0
    shards_written: int = 0
    
    # Errors
    errors: int = 0
    error_log: List[str] = None
    
    # Timing
    elapsed_seconds: float = 0.0
    items_per_second: float = 0.0
    
    def __post_init__(self):
        if self.error_log is None:
            self.error_log = []
    
    def save(self, path: str):
        """Save status to file."""
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "PipelineStatus":
        """Load status from file."""
        with open(path, "r") as f:
            data = json.load(f)
        data["stage"] = PipelineStage(data["stage"])
        return cls(**data)


class DataPipeline:
    """
    Main data pipeline orchestrator.
    
    Coordinates all stages of data processing from raw sources
    to training-ready shards.
    
    Example:
        ```python
        from data.pipeline import DataPipeline, PipelineConfig
        
        # Create pipeline
        config = PipelineConfig()
        pipeline = DataPipeline(config)
        
        # Run full pipeline
        pipeline.run(sources=["data/raw/"])
        
        # Or run stages individually
        pipeline.ingest(["data/raw/"])
        pipeline.preprocess()
        pipeline.extract()
        pipeline.tokenize()
        pipeline.shard()
        ```
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize the data pipeline.
        
        Args:
            config: Pipeline configuration. Uses defaults if not provided.
        """
        self.config = config or PipelineConfig()
        
        # Initialize components
        self.ingester = DataIngester(self.config)
        self.preprocessor = DataPreprocessor(self.config)
        self.extractor = ContentExtractor(self.config)
        self.shard_writer = ShardWriter(self.config)
        
        # Initialize tokenizer
        self.tokenizer = None
        if HAS_TOKENIZER:
            try:
                self.tokenizer = MultiModalTokenizer()
                tokenizer_path = self.config.tokenization.tokenizer_path
                if os.path.exists(f"{tokenizer_path}.model"):
                    self.tokenizer.load_text_tokenizer(tokenizer_path)
                    logger.info("Loaded existing tokenizer")
            except Exception as e:
                logger.warning(f"Could not initialize tokenizer: {e}")
        
        # Status tracking
        self.status = PipelineStatus(
            stage=PipelineStage.INGEST,
            started_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
        )
        
        # Data buffers for stage-by-stage processing
        self._ingested_items: List[IngestedItem] = []
        self._preprocessed_items: List[PreprocessedItem] = []
        self._extracted_items: List[ExtractedContent] = []
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup pipeline logging."""
        log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        logging.basicConfig(level=logging.INFO, format=log_format)
        
        # File handler for errors
        error_log_path = self.config.processing.error_log_file
        os.makedirs(os.path.dirname(error_log_path), exist_ok=True)
        
        file_handler = logging.FileHandler(error_log_path)
        file_handler.setLevel(logging.ERROR)
        file_handler.setFormatter(logging.Formatter(log_format))
        
        logging.getLogger().addHandler(file_handler)
    
    def run(
        self,
        sources: List[str],
        resume: bool = False,
    ) -> ShardIndex:
        """
        Run the complete pipeline from sources to shards.
        
        Args:
            sources: List of source paths (files, directories, URLs)
            resume: Whether to resume from last checkpoint
            
        Returns:
            ShardIndex with metadata about created shards
        """
        start_time = time.time()
        
        logger.info("=" * 60)
        logger.info("RT-DLM Data Pipeline")
        logger.info("=" * 60)
        logger.info(f"Sources: {sources}")
        logger.info(f"Output: {self.config.sharding.output_dir}")
        
        try:
            # Stage 1: Ingest
            logger.info("\n[1/5] INGESTION")
            self.ingest(sources)
            
            # Stage 2: Preprocess
            logger.info("\n[2/5] PREPROCESSING")
            self.preprocess()
            
            # Stage 3: Extract
            logger.info("\n[3/5] EXTRACTION")
            self.extract()
            
            # Stage 4: Tokenize
            logger.info("\n[4/5] TOKENIZATION")
            self.tokenize()
            
            # Stage 5: Shard
            logger.info("\n[5/5] SHARDING")
            index = self.shard()
            
            # Update status
            elapsed = time.time() - start_time
            self.status.stage = PipelineStage.COMPLETE
            self.status.elapsed_seconds = elapsed
            self.status.items_per_second = self.status.tokenized_items / max(1, elapsed)
            self.status.updated_at = datetime.now().isoformat()
            
            # Save status
            status_path = os.path.join(self.config.base_dir, "pipeline_status.json")
            self.status.save(status_path)
            
            logger.info("\n" + "=" * 60)
            logger.info("PIPELINE COMPLETE")
            logger.info(f"Elapsed: {elapsed:.1f}s")
            logger.info(f"Shards: {self.status.shards_written}")
            logger.info(f"Tokens: {self.status.total_tokens:,}")
            logger.info("=" * 60)
            
            return index
        
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            self.status.errors += 1
            self.status.error_log.append(str(e))
            raise
    
    def ingest(self, sources: List[str]) -> List[IngestedItem]:
        """
        Stage 1: Ingest data from sources.
        
        Args:
            sources: List of source paths
            
        Returns:
            List of ingested items
        """
        self.status.stage = PipelineStage.INGEST
        self.status.updated_at = datetime.now().isoformat()
        
        # Discover sources
        discovered = self.ingester.discover_sources(sources)
        self.status.total_sources = len(discovered)
        
        logger.info(f"Discovered {len(discovered)} sources")
        
        # Ingest
        self._ingested_items = []
        for item in self.ingester.ingest(discovered):
            self._ingested_items.append(item)
            self.status.ingested_items += 1
            self.status.processed_sources += 1
            
            if self.status.ingested_items % self.config.processing.log_interval == 0:
                logger.info(f"Ingested {self.status.ingested_items} items")
        
        # Save manifest
        self.ingester.save_manifest()
        
        logger.info(f"Ingestion complete: {self.status.ingested_items} items")
        
        return self._ingested_items
    
    def preprocess(
        self, 
        items: Optional[List[IngestedItem]] = None,
    ) -> List[PreprocessedItem]:
        """
        Stage 2: Preprocess and filter data.
        
        Args:
            items: Items to preprocess. Uses buffered items if not provided.
            
        Returns:
            List of preprocessed items
        """
        self.status.stage = PipelineStage.PREPROCESS
        self.status.updated_at = datetime.now().isoformat()
        
        if items is None:
            items = self._ingested_items
        
        logger.info(f"Preprocessing {len(items)} items")
        
        self._preprocessed_items = []
        for result in self.preprocessor.preprocess_batch(items):
            self._preprocessed_items.append(result)
            self.status.preprocessed_items += 1
            
            if self.status.preprocessed_items % self.config.processing.log_interval == 0:
                logger.info(f"Preprocessed {self.status.preprocessed_items} items")
        
        # Log statistics
        stats = self.preprocessor.get_statistics()
        logger.info(f"Preprocessing complete: {stats['passed']}/{stats['total_processed']} passed")
        logger.info(f"  - Duplicates removed: {stats['duplicates']}")
        logger.info(f"  - Quality rejected: {stats['quality_rejected']}")
        logger.info(f"  - PII found: {stats['pii_found']}")
        
        return self._preprocessed_items
    
    def extract(
        self,
        items: Optional[List[PreprocessedItem]] = None,
    ) -> List[ExtractedContent]:
        """
        Stage 3: Extract content from files.
        
        Args:
            items: Items to extract from. Uses buffered items if not provided.
            
        Returns:
            List of extracted content
        """
        self.status.stage = PipelineStage.EXTRACT
        self.status.updated_at = datetime.now().isoformat()
        
        if items is None:
            items = self._preprocessed_items
        
        logger.info(f"Extracting content from {len(items)} items")
        
        self._extracted_items = []
        for item in items:
            try:
                extracted = self.extractor.extract(item)
                if extracted.extraction_success and (extracted.has_text() or extracted.has_embeddings()):
                    self._extracted_items.append(extracted)
                    self.status.extracted_items += 1
            except Exception as e:
                self.status.errors += 1
                if self.config.processing.skip_errors:
                    logger.warning(f"Extraction error: {e}")
                else:
                    raise
            
            if self.status.extracted_items % self.config.processing.log_interval == 0:
                logger.info(f"Extracted {self.status.extracted_items} items")
        
        logger.info(f"Extraction complete: {self.status.extracted_items} items")
        
        return self._extracted_items
    
    def tokenize(
        self,
        items: Optional[List[ExtractedContent]] = None,
    ) -> Generator[TokenizedSample, None, None]:
        """
        Stage 4: Tokenize content.
        
        Args:
            items: Items to tokenize. Uses buffered items if not provided.
            
        Yields:
            TokenizedSample objects
        """
        self.status.stage = PipelineStage.TOKENIZE
        self.status.updated_at = datetime.now().isoformat()
        
        if items is None:
            items = self._extracted_items
        
        logger.info(f"Tokenizing {len(items)} items")
        
        for item in items:
            try:
                sample = self._tokenize_item(item)
                if sample is not None:
                    self.status.tokenized_items += 1
                    self.status.total_tokens += len(sample.token_ids)
                    yield sample
                    
                    if self.status.tokenized_items % self.config.processing.log_interval == 0:
                        logger.info(f"Tokenized {self.status.tokenized_items} items, "
                                   f"{self.status.total_tokens:,} tokens")
            except Exception as e:
                self.status.errors += 1
                if self.config.processing.skip_errors:
                    logger.warning(f"Tokenization error: {e}")
                else:
                    raise
        
        logger.info(f"Tokenization complete: {self.status.tokenized_items} items, "
                   f"{self.status.total_tokens:,} tokens")
    
    def _tokenize_item(self, item: ExtractedContent) -> Optional[TokenizedSample]:
        """Tokenize a single extracted content item."""
        if not item.has_text() and not item.has_embeddings():
            return None
        
        token_ids = []
        modalities = []
        
        # Tokenize text
        if item.has_text() and self.tokenizer is not None:
            try:
                text_tokens = self.tokenizer.tokenize(item.text)
                token_ids.extend(text_tokens)
                modalities.append("text")
            except Exception as e:
                logger.warning(f"Text tokenization failed: {e}")
                # Fallback: simple character-level
                token_ids = [ord(c) % 50000 for c in item.text[:2048]]
                modalities.append("text")
        elif item.has_text():
            # No tokenizer - use simple encoding
            token_ids = [ord(c) % 50000 for c in item.text[:2048]]
            modalities.append("text")
        
        if not token_ids:
            return None
        
        # Create sample
        sample = TokenizedSample(
            token_ids=np.array(token_ids, dtype=np.int32),
            embeddings=item.image_embeddings if item.image_embeddings is not None 
                       else item.audio_embeddings if item.audio_embeddings is not None
                       else item.video_embeddings,
            source_id=item.source_path,
            modalities=modalities,
        )
        
        return sample
    
    def shard(
        self,
        samples: Optional[Generator[TokenizedSample, None, None]] = None,
    ) -> ShardIndex:
        """
        Stage 5: Create shards from tokenized samples.
        
        Args:
            samples: Generator of samples. Runs tokenization if not provided.
            
        Returns:
            ShardIndex with shard metadata
        """
        self.status.stage = PipelineStage.SHARD
        self.status.updated_at = datetime.now().isoformat()
        
        if samples is None:
            samples = self.tokenize()
        
        # Collect samples for splitting
        all_samples = list(samples)
        random.shuffle(all_samples)
        
        total = len(all_samples)
        train_end = int(total * self.config.sharding.train_ratio)
        val_end = train_end + int(total * self.config.sharding.val_ratio)
        
        train_samples = all_samples[:train_end]
        val_samples = all_samples[train_end:val_end]
        test_samples = all_samples[val_end:]
        
        logger.info(f"Splitting: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")
        
        # Write shards
        self.shard_writer.set_split("train")
        for sample in train_samples:
            self.shard_writer.add_sample(sample)
        
        self.shard_writer.set_split("val")
        for sample in val_samples:
            self.shard_writer.add_sample(sample)
        
        self.shard_writer.set_split("test")
        for sample in test_samples:
            self.shard_writer.add_sample(sample)
        
        # Finalize
        index = self.shard_writer.finalize()
        self.status.shards_written = self.shard_writer.stats["shards_written"]
        
        logger.info(f"Sharding complete: {self.status.shards_written} shards")
        
        return index
    
    def get_status(self) -> PipelineStatus:
        """Get current pipeline status."""
        return self.status


def main():
    """CLI entry point for the data pipeline."""
    parser = argparse.ArgumentParser(
        description="RT-DLM Data Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process local directory
  python -m data.pipeline.pipeline --sources data/raw/
  
  # Process multiple sources
  python -m data.pipeline.pipeline --sources data/texts/ data/images/ --multimodal
  
  # Use production config
  python -m data.pipeline.pipeline --sources data/raw/ --preset production
  
  # Custom output directory
  python -m data.pipeline.pipeline --sources data/raw/ --output data/shards_v2
        """
    )
    
    parser.add_argument(
        "--sources", "-s",
        nargs="+",
        required=True,
        help="Source paths (files, directories, URLs)"
    )
    
    parser.add_argument(
        "--output", "-o",
        default="data/shards",
        help="Output directory for shards"
    )
    
    parser.add_argument(
        "--preset",
        choices=["small", "production", "multimodal"],
        default="small",
        help="Configuration preset"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training"
    )
    
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=2048,
        help="Sequence length for tokenization"
    )
    
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Disable deduplication"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose logging"
    )
    
    args = parser.parse_args()
    
    # Create config
    if args.preset == "production":
        config = create_production_config()
    elif args.preset == "multimodal":
        config = create_multimodal_config()
    else:
        config = create_small_scale_config()
    
    # Override with CLI args
    config.sharding.output_dir = args.output
    config.sharding.sequence_length = args.sequence_length
    config.quality.deduplicate = not args.no_dedup
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run pipeline
    pipeline = DataPipeline(config)
    
    try:
        index = pipeline.run(args.sources)
        
        print("\n" + "=" * 40)
        print("SUCCESS")
        print("=" * 40)
        print(f"Shards: {len(index.train_shards)} train, {len(index.val_shards)} val, {len(index.test_shards)} test")
        print(f"Total tokens: {index.total_tokens:,}")
        print(f"Output: {args.output}")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
