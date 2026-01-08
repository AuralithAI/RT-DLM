# RT-DLM Data Processing Pipeline

A production-ready, scalable pipeline for processing multimodal data into training-ready shards.

## Overview

The RT-DLM data pipeline transforms raw data from various sources into efficient binary shards optimized for distributed training. It handles text, images, audio, and video with built-in quality filtering, deduplication, and tokenization.

```
┌─────────────┐    ┌──────────────┐    ┌────────────┐    ┌─────────────┐    ┌──────────┐
│  Ingestion  │───▶│ Preprocessing│───▶│ Extraction │───▶│ Tokenization│───▶│ Sharding │
└─────────────┘    └──────────────┘    └────────────┘    └─────────────┘    └──────────┘
      │                   │                  │                  │                 │
      ▼                   ▼                  ▼                  ▼                 ▼
 Multi-source       Quality Filter      Content Parse      Token IDs        Binary Shards
   Collect          Deduplicate         All Formats        Embeddings       SafeTensors
```

## Pipeline Stages

### 1. Data Ingestion

Collects data from multiple sources with parallel processing and rate limiting.

**Supported Sources:**
| Source | Description | Configuration |
|--------|-------------|---------------|
| Local Files | Directories and files on disk | `SourceType.DIRECTORY`, `SourceType.FILE` |
| Amazon S3 | AWS S3 buckets | `storage_backend=StorageBackend.S3` |
| Google Cloud | GCS buckets | `storage_backend=StorageBackend.GCS` |
| Azure Blob | Azure storage containers | `storage_backend=StorageBackend.AZURE` |
| HuggingFace | HuggingFace datasets | `SourceType.HUGGINGFACE` |
| Web URLs | Web scraping with rate limiting | `SourceType.URL` |
| APIs | REST API endpoints | `SourceType.API` |

**Features:**
- Parallel file discovery and download
- Automatic content type detection
- Configurable rate limiting (default: 5 req/s)
- Retry logic with exponential backoff
- Manifest tracking for incremental updates

```python
from data.pipeline import DataIngester, DataSource, SourceType, PipelineConfig

config = PipelineConfig()
ingester = DataIngester(config)

# Add data sources
sources = [
    DataSource(path="/data/texts", source_type=SourceType.DIRECTORY),
    DataSource(path="s3://my-bucket/data", source_type=SourceType.S3),
    DataSource(path="https://example.com/data.json", source_type=SourceType.URL),
]

# Ingest all sources
for item in ingester.ingest(sources):
    process(item)
```

### 2. Preprocessing

Cleans and filters data to ensure quality training samples.

**Quality Filters:**
| Filter | Purpose | Default Threshold |
|--------|---------|-------------------|
| Text Length | Remove too short/long texts | 50 - 100,000 chars |
| Word Count | Ensure meaningful content | 10 - 50,000 words |
| Unique Words | Filter repetitive content | > 10% unique |
| Special Chars | Remove noisy text | < 30% special chars |
| Language | Keep target languages | English only |

**Deduplication Methods:**
- **Exact**: Hash-based exact duplicate detection
- **MinHash**: Fuzzy matching for near-duplicates (default, threshold: 0.8)
- **SimHash**: Locality-sensitive hashing for similar content

**Privacy & Safety:**
- **PII Removal**: Automatic detection and redaction of emails, phone numbers, SSNs, addresses
- **Toxicity Filtering**: Content moderation using transformer-based classifiers

```python
from data.pipeline import DataPreprocessor, PipelineConfig
from data.pipeline.config import QualityConfig

config = PipelineConfig(
    quality=QualityConfig(
        min_text_length=100,
        deduplicate=True,
        dedup_method="minhash",
        minhash_threshold=0.85,
        remove_pii=True,
        remove_toxicity=True,
        toxicity_threshold=0.7
    )
)

preprocessor = DataPreprocessor(config)
cleaned_data = preprocessor.process(raw_data)
```

### 3. Content Extraction

Extracts text and embeddings from various file formats.

**Supported Formats:**

| Category | Formats | Extraction Method |
|----------|---------|-------------------|
| **Text** | `.txt`, `.md`, `.rst`, `.csv`, `.json`, `.jsonl` | Direct parsing |
| **Code** | `.py`, `.js`, `.ts`, `.java`, `.cpp`, `.go`, `.rs` | Syntax-aware parsing |
| **Documents** | `.pdf`, `.doc`, `.docx`, `.ppt`, `.pptx`, `.xls`, `.xlsx` | pdfplumber, python-docx |
| **Web** | `.html`, `.htm`, `.xml` | BeautifulSoup parsing |
| **Images** | `.jpg`, `.png`, `.gif`, `.bmp`, `.tiff`, `.webp` | Patch-based encoding |
| **Audio** | `.mp3`, `.wav`, `.flac`, `.ogg`, `.m4a` | MFCC / Wav2Vec / Whisper |
| **Video** | `.mp4`, `.avi`, `.mkv`, `.mov`, `.webm` | Frame extraction + audio |

**Multimodal Extraction:**
```python
from data.pipeline import ContentExtractor, PipelineConfig
from data.pipeline.config import TokenizationPipelineConfig

config = PipelineConfig(
    tokenization=TokenizationPipelineConfig(
        image_encoder="clip",      # ViT, CLIP, or ResNet
        image_size=(224, 224),
        audio_encoder="whisper",   # Wav2Vec, Whisper, or MFCC
        audio_sample_rate=16000,
        video_fps=8,
        max_video_frames=64
    )
)

extractor = ContentExtractor(config)
content = extractor.extract("/path/to/file.pdf")
```

### 4. Tokenization

Converts extracted content into token IDs and embeddings.

**Tokenizer Options:**
| Type | Description | Use Case |
|------|-------------|----------|
| BPE | Byte-Pair Encoding | General text (default) |
| Unigram | Unigram language model | Multilingual |
| WordPiece | BERT-style tokenization | Fine-tuning |
| SentencePiece | Subword tokenization | Production |

**Special Tokens:**
```
PAD: 0    UNK: 1    BOS: 2    EOS: 3    SEP: 4
Modality tokens: 10+ (reserved for [IMG], [AUD], [VID], etc.)
```

**Multimodal Token Interleaving:**
```
[BOS] This is an image: [IMG] <image_embeddings> [/IMG] showing a cat. [EOS]
```

### 5. Sharding

Converts tokenized data into efficient binary shards for distributed training.

**Shard Configuration:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_shard_size_bytes` | 1 GB | Maximum shard file size |
| `max_tokens_per_shard` | 10M | Token limit per shard |
| `sequence_length` | 2048 | Training sequence length |
| `pack_sequences` | True | Pack short sequences together |
| `format` | SafeTensors | Output format |
| `compression` | None | Optional: zstd, lz4, gzip |

**Output Formats:**
- **SafeTensors** (recommended): Fast, memory-mapped, secure
- **NumPy**: Standard `.npy` arrays
- **Arrow**: Apache Arrow columnar format

**Sequence Packing:**
Efficiently packs multiple short sequences into fixed-length training samples:
```
Sample 1: [seq1_tokens...][SEP][seq2_tokens...][SEP][seq3_tokens...][PAD...]
Sample 2: [seq4_tokens...][SEP][seq5_tokens...][PAD...]
```

```python
from data.pipeline import ShardWriter, PipelineConfig
from data.pipeline.config import ShardConfig

config = PipelineConfig(
    sharding=ShardConfig(
        max_shard_size_bytes=500_000_000,  # 500 MB
        sequence_length=2048,
        pack_sequences=True,
        format="safetensors",
        compression="zstd",
        train_ratio=0.95,
        val_ratio=0.04,
        test_ratio=0.01
    )
)

writer = ShardWriter(config)
writer.write(tokenized_samples)
index = writer.finalize()
```

**Shard Index:**
Each shard directory contains an index file (`shard_index.json`):
```json
{
  "num_shards": 100,
  "total_samples": 10000000,
  "total_tokens": 20480000000,
  "shards": [
    {"id": "shard_0000", "path": "train/shard_0000.safetensors", "samples": 100000},
    {"id": "shard_0001", "path": "train/shard_0001.safetensors", "samples": 100000}
  ],
  "splits": {"train": 95, "val": 4, "test": 1}
}
```

### 6. Training DataLoader

Efficiently streams shards during training with prefetching.

**Features:**
- Streaming reads (no full dataset in memory)
- Prefetching with configurable buffer
- Dynamic batching
- Multi-worker parallel loading
- Shuffle across and within shards
- Resume from checkpoint support

```python
from data.pipeline import ShardedDataLoader, PipelineConfig

config = PipelineConfig()
loader = ShardedDataLoader(config)
loader.load_shards("/path/to/shards")

for batch in loader.iter_batches(batch_size=32, split="train"):
    # batch.input_ids: [batch_size, seq_len]
    # batch.attention_mask: [batch_size, seq_len]
    train_step(batch)
```

## Quick Start

### Basic Usage

```python
from data.pipeline import DataPipeline, PipelineConfig

# Create pipeline with default config
config = PipelineConfig()
pipeline = DataPipeline(config)

# Process data directory
stats = pipeline.run(
    input_paths=["/data/raw/texts", "/data/raw/documents"],
    output_dir="/data/shards"
)

print(f"Processed {stats.total_files} files")
print(f"Created {stats.num_shards} shards")
print(f"Total tokens: {stats.total_tokens:,}")
```

### Production Configuration

```python
from data.pipeline.config import create_production_config

# Optimized for large-scale processing
config = create_production_config()

# Customize as needed
config.sharding.compression = "zstd"
config.quality.minhash_threshold = 0.9
config.processing.num_workers = 16

pipeline = DataPipeline(config)
```

### Multimodal Configuration

```python
from data.pipeline.config import create_multimodal_config

# Optimized for mixed text/image/audio/video
config = create_multimodal_config()

pipeline = DataPipeline(config)
pipeline.run(
    input_paths=["/data/images", "/data/audio", "/data/video"],
    output_dir="/data/multimodal_shards"
)
```

## Configuration Reference

### Preset Configurations

| Preset | Use Case | Memory | Shard Size |
|--------|----------|--------|------------|
| `create_small_scale_config()` | Testing, <10GB data | 4 GB | 100 MB |
| `create_production_config()` | Production, 100GB+ | 64 GB | 1 GB |
| `create_multimodal_config()` | Mixed modalities | 32 GB | 500 MB |

### Full Configuration Example

```python
from data.pipeline.config import (
    PipelineConfig,
    IngestionConfig,
    QualityConfig,
    TokenizationPipelineConfig,
    ShardConfig,
    ProcessingConfig,
    StorageBackend,
    TokenizerType
)

config = PipelineConfig(
    name="my_pipeline",
    
    ingestion=IngestionConfig(
        storage_backend=StorageBackend.S3,
        s3_bucket="my-data-bucket",
        max_concurrent_downloads=20,
        supported_extensions=[".txt", ".json", ".pdf"]
    ),
    
    quality=QualityConfig(
        min_text_length=100,
        max_text_length=50000,
        deduplicate=True,
        minhash_threshold=0.85,
        remove_pii=True,
        remove_toxicity=True,
        allowed_languages=["en", "es", "fr"]
    ),
    
    tokenization=TokenizationPipelineConfig(
        tokenizer_type=TokenizerType.BPE,
        vocab_size=50000,
        image_encoder="clip",
        audio_encoder="whisper"
    ),
    
    sharding=ShardConfig(
        max_shard_size_bytes=1_000_000_000,
        sequence_length=4096,
        pack_sequences=True,
        format="safetensors",
        compression="zstd"
    ),
    
    processing=ProcessingConfig(
        num_workers=16,
        batch_size=1000,
        streaming_mode=True,
        max_memory_gb=64.0
    )
)
```

## Directory Structure

After running the pipeline, your data directory will look like:

```
data/
├── raw/                    # Original input data
│   ├── texts/
│   ├── documents/
│   └── images/
├── staging/                # Intermediate processing
├── shards/                 # Output shards
│   ├── train/
│   │   ├── shard_0000.safetensors
│   │   ├── shard_0001.safetensors
│   │   └── ...
│   ├── val/
│   │   └── shard_0000.safetensors
│   ├── test/
│   │   └── shard_0000.safetensors
│   └── shard_index.json
├── .cache/                 # Cached embeddings
└── manifest.json           # Processing manifest
```

## Performance Tips

1. **Use streaming mode** for datasets larger than available RAM
2. **Enable sequence packing** to maximize GPU utilization
3. **Use SafeTensors format** for fastest loading
4. **Enable zstd compression** for 30-50% storage savings
5. **Increase num_workers** based on CPU cores
6. **Use S3/GCS** for distributed processing

## API Reference

See the module docstrings for detailed API documentation:

- [`data.pipeline.config`](../data/pipeline/config.py) - Configuration classes
- [`data.pipeline.ingestion`](../data/pipeline/ingestion.py) - Data ingestion
- [`data.pipeline.preprocessing`](../data/pipeline/preprocessing.py) - Data preprocessing
- [`data.pipeline.extraction`](../data/pipeline/extraction.py) - Content extraction
- [`data.pipeline.sharding`](../data/pipeline/sharding.py) - Shard creation
- [`data.pipeline.dataloader`](../data/pipeline/dataloader.py) - Training dataloader
- [`data.pipeline.pipeline`](../data/pipeline/pipeline.py) - Pipeline orchestrator

---

[← Back to Main README](../README.md) | [Architecture →](./ARCHITECTURE.md) | [Quick Start →](./QUICKSTART.md)
