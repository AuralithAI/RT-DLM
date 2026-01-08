"""
Data Ingestion Module

Handles data collection from various sources:
- Local file systems
- Cloud storage (S3, GCS, Azure)
- Web scraping
- APIs (weather, news, etc.)
- Datasets (HuggingFace, Common Crawl, etc.)
"""

import os
import json
import hashlib
import mimetypes
import asyncio
import aiohttp
import aiofiles
from pathlib import Path
from typing import List, Dict, Optional, Generator, Any, Union, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

from .config import PipelineConfig, IngestionConfig

logger = logging.getLogger(__name__)


class SourceType(Enum):
    """Types of data sources."""
    FILE = "file"
    DIRECTORY = "directory"
    URL = "url"
    S3 = "s3"
    GCS = "gcs"
    HUGGINGFACE = "huggingface"
    API = "api"
    COMMON_CRAWL = "common_crawl"


@dataclass
class DataSource:
    """Represents a data source with metadata."""
    path: str
    source_type: SourceType
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Auto-detected properties
    content_type: Optional[str] = None
    size_bytes: Optional[int] = None
    last_modified: Optional[datetime] = None
    checksum: Optional[str] = None
    
    # Processing state
    processed: bool = False
    error: Optional[str] = None


@dataclass
class IngestedItem:
    """Represents an ingested data item."""
    source: DataSource
    content: Union[bytes, str]
    content_type: str
    metadata: Dict[str, Any]
    
    @property
    def is_text(self) -> bool:
        return self.content_type.startswith("text/") or self.content_type in [
            "application/json", "application/xml"
        ]
    
    @property
    def is_binary(self) -> bool:
        return not self.is_text


class DataIngester:
    """
    Production-ready data ingester for multimodal data.
    
    Features:
    - Parallel file discovery
    - Streaming for large files
    - Cloud storage support
    - Web scraping with rate limiting
    - Manifest tracking
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.ingestion_config = config.ingestion
        
        # Initialize mimetypes
        mimetypes.init()
        
        # Manifest tracking
        self.manifest: List[DataSource] = []
        self._load_manifest()
        
        # Statistics
        self.stats = {
            "total_discovered": 0,
            "total_ingested": 0,
            "total_bytes": 0,
            "errors": 0,
            "by_type": {},
        }
    
    def _load_manifest(self):
        """Load existing manifest if available."""
        manifest_path = Path(self.config.manifest_file)
        if manifest_path.exists():
            try:
                with open(manifest_path, "r") as f:
                    data = json.load(f)
                    for item in data.get("sources", []):
                        self.manifest.append(DataSource(
                            path=item["path"],
                            source_type=SourceType(item["source_type"]),
                            metadata=item.get("metadata", {}),
                            content_type=item.get("content_type"),
                            size_bytes=item.get("size_bytes"),
                            checksum=item.get("checksum"),
                            processed=item.get("processed", False),
                        ))
                logger.info(f"Loaded {len(self.manifest)} sources from manifest")
            except Exception as e:
                logger.warning(f"Could not load manifest: {e}")
    
    def save_manifest(self):
        """Save manifest to disk."""
        manifest_path = Path(self.config.manifest_file)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "version": self.config.version,
            "created": datetime.now().isoformat(),
            "stats": self.stats,
            "sources": [
                {
                    "path": s.path,
                    "source_type": s.source_type.value,
                    "metadata": s.metadata,
                    "content_type": s.content_type,
                    "size_bytes": s.size_bytes,
                    "checksum": s.checksum,
                    "processed": s.processed,
                    "error": s.error,
                }
                for s in self.manifest
            ]
        }
        
        with open(manifest_path, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved manifest with {len(self.manifest)} sources")
    
    def discover_sources(
        self,
        paths: List[str],
        recursive: bool = True,
        pattern: Optional[str] = None,
    ) -> List[DataSource]:
        """
        Discover data sources from given paths.
        
        Args:
            paths: List of file paths, directories, URLs, or cloud URIs
            recursive: Whether to search directories recursively
            pattern: Optional glob pattern to filter files
            
        Returns:
            List of discovered DataSource objects
        """
        sources = []
        
        for path in paths:
            if path.startswith("s3://"):
                sources.extend(self._discover_s3(path, pattern))
            elif path.startswith("gs://"):
                sources.extend(self._discover_gcs(path, pattern))
            elif path.startswith(("http://", "https://")):
                sources.append(self._create_url_source(path))
            elif os.path.isfile(path):
                source = self._create_file_source(path)
                if source:
                    sources.append(source)
            elif os.path.isdir(path):
                sources.extend(self._discover_directory(path, recursive, pattern))
            else:
                logger.warning(f"Unknown source type: {path}")
        
        # Update manifest
        existing_paths = {s.path for s in self.manifest}
        new_sources = [s for s in sources if s.path not in existing_paths]
        self.manifest.extend(new_sources)
        
        self.stats["total_discovered"] = len(self.manifest)
        logger.info(f"Discovered {len(new_sources)} new sources ({len(self.manifest)} total)")
        
        return sources
    
    def _discover_directory(
        self, 
        directory: str, 
        recursive: bool = True,
        pattern: Optional[str] = None,
    ) -> List[DataSource]:
        """Discover files in a directory."""
        sources = []
        dir_path = Path(directory)
        
        if pattern:
            if recursive:
                files = dir_path.rglob(pattern)
            else:
                files = dir_path.glob(pattern)
        else:
            if recursive:
                files = dir_path.rglob("*")
            else:
                files = dir_path.glob("*")
        
        supported_exts = set(self.ingestion_config.supported_extensions)
        
        for file_path in files:
            if file_path.is_file():
                ext = file_path.suffix.lower()
                if ext in supported_exts or not supported_exts:
                    source = self._create_file_source(str(file_path))
                    if source:
                        sources.append(source)
        
        return sources
    
    def _create_file_source(self, file_path: str) -> Optional[DataSource]:
        """Create a DataSource from a file path."""
        try:
            path = Path(file_path)
            stat = path.stat()
            content_type, _ = mimetypes.guess_type(file_path)
            
            return DataSource(
                path=str(path.absolute()),
                source_type=SourceType.FILE,
                content_type=content_type or "application/octet-stream",
                size_bytes=stat.st_size,
                last_modified=datetime.fromtimestamp(stat.st_mtime),
                metadata={
                    "filename": path.name,
                    "extension": path.suffix.lower(),
                    "parent": str(path.parent),
                }
            )
        except Exception as e:
            logger.error(f"Error creating source for {file_path}: {e}")
            return None
    
    def _create_url_source(self, url: str) -> DataSource:
        """Create a DataSource from a URL."""
        return DataSource(
            path=url,
            source_type=SourceType.URL,
            metadata={"url": url},
        )
    
    def _discover_s3(self, uri: str, pattern: Optional[str] = None) -> List[DataSource]:
        """Discover files in S3 bucket."""
        sources = []
        try:
            import boto3
            
            # Parse S3 URI
            parts = uri.replace("s3://", "").split("/", 1)
            bucket = parts[0]
            prefix = parts[1] if len(parts) > 1 else ""
            
            s3 = boto3.client("s3")
            paginator = s3.get_paginator("list_objects_v2")
            
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    ext = Path(key).suffix.lower()
                    
                    if ext in self.ingestion_config.supported_extensions:
                        sources.append(DataSource(
                            path=f"s3://{bucket}/{key}",
                            source_type=SourceType.S3,
                            content_type=mimetypes.guess_type(key)[0] or "application/octet-stream",
                            size_bytes=obj["Size"],
                            last_modified=obj["LastModified"],
                            metadata={
                                "bucket": bucket,
                                "key": key,
                                "etag": obj.get("ETag", "").strip('"'),
                            }
                        ))
        except ImportError:
            logger.warning("boto3 not installed. S3 discovery unavailable.")
        except Exception as e:
            logger.error(f"Error discovering S3 sources: {e}")
        
        return sources
    
    def _discover_gcs(self, uri: str, pattern: Optional[str] = None) -> List[DataSource]:
        """Discover files in Google Cloud Storage."""
        sources = []
        try:
            from google.cloud import storage
            
            # Parse GCS URI
            parts = uri.replace("gs://", "").split("/", 1)
            bucket_name = parts[0]
            prefix = parts[1] if len(parts) > 1 else ""
            
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            
            for blob in bucket.list_blobs(prefix=prefix):
                ext = Path(blob.name).suffix.lower()
                
                if ext in self.ingestion_config.supported_extensions:
                    sources.append(DataSource(
                        path=f"gs://{bucket_name}/{blob.name}",
                        source_type=SourceType.GCS,
                        content_type=blob.content_type or "application/octet-stream",
                        size_bytes=blob.size,
                        last_modified=blob.updated,
                        metadata={
                            "bucket": bucket_name,
                            "name": blob.name,
                            "md5_hash": blob.md5_hash,
                        }
                    ))
        except ImportError:
            logger.warning("google-cloud-storage not installed. GCS discovery unavailable.")
        except Exception as e:
            logger.error(f"Error discovering GCS sources: {e}")
        
        return sources
    
    def ingest(
        self, 
        sources: Optional[List[DataSource]] = None,
        num_workers: Optional[int] = None,
    ) -> Generator[IngestedItem, None, None]:
        """
        Ingest data from sources.
        
        Args:
            sources: List of sources to ingest (uses manifest if None)
            num_workers: Number of parallel workers
            
        Yields:
            IngestedItem objects with content and metadata
        """
        if sources is None:
            sources = [s for s in self.manifest if not s.processed]
        
        num_workers = num_workers or self.config.processing.num_workers
        
        logger.info(f"Ingesting {len(sources)} sources with {num_workers} workers")
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(self._ingest_source, source): source 
                for source in sources
            }
            
            for future in as_completed(futures):
                source = futures[future]
                try:
                    item = future.result()
                    if item:
                        self.stats["total_ingested"] += 1
                        self.stats["total_bytes"] += len(item.content) if item.content else 0
                        
                        # Track by type
                        ext = source.metadata.get("extension", "unknown")
                        self.stats["by_type"][ext] = self.stats["by_type"].get(ext, 0) + 1
                        
                        source.processed = True
                        yield item
                except Exception as e:
                    logger.error(f"Error ingesting {source.path}: {e}")
                    source.error = str(e)
                    self.stats["errors"] += 1
    
    def _ingest_source(self, source: DataSource) -> Optional[IngestedItem]:
        """Ingest a single source."""
        try:
            if source.source_type == SourceType.FILE:
                return self._ingest_file(source)
            elif source.source_type == SourceType.URL:
                return self._ingest_url(source)
            elif source.source_type == SourceType.S3:
                return self._ingest_s3(source)
            elif source.source_type == SourceType.GCS:
                return self._ingest_gcs(source)
            else:
                logger.warning(f"Unsupported source type: {source.source_type}")
                return None
        except Exception as e:
            logger.error(f"Error ingesting source {source.path}: {e}")
            return None
    
    def _ingest_file(self, source: DataSource) -> Optional[IngestedItem]:
        """Ingest a local file."""
        try:
            # Check if text or binary
            is_text = source.content_type and (
                source.content_type.startswith("text/") or
                source.content_type in ["application/json", "application/xml", "application/javascript"]
            )
            
            mode = "r" if is_text else "rb"
            encoding = "utf-8" if is_text else None
            
            with open(source.path, mode, encoding=encoding, errors="ignore" if is_text else None) as f:
                content = f.read()
            
            # Calculate checksum
            if isinstance(content, str):
                checksum = hashlib.md5(content.encode()).hexdigest()
            else:
                checksum = hashlib.md5(content).hexdigest()
            
            source.checksum = checksum
            
            return IngestedItem(
                source=source,
                content=content,
                content_type=source.content_type or "application/octet-stream",
                metadata={
                    **source.metadata,
                    "checksum": checksum,
                    "size_bytes": len(content) if isinstance(content, bytes) else len(content.encode()),
                }
            )
        except Exception as e:
            logger.error(f"Error reading file {source.path}: {e}")
            return None
    
    def _ingest_url(self, source: DataSource) -> Optional[IngestedItem]:
        """Ingest content from URL."""
        import requests
        
        try:
            headers = {"User-Agent": self.ingestion_config.user_agent}
            response = requests.get(
                source.path,
                headers=headers,
                timeout=self.ingestion_config.download_timeout,
            )
            response.raise_for_status()
            
            content_type = response.headers.get("Content-Type", "text/html").split(";")[0]
            
            return IngestedItem(
                source=source,
                content=response.text if "text" in content_type else response.content,
                content_type=content_type,
                metadata={
                    **source.metadata,
                    "status_code": response.status_code,
                    "content_length": len(response.content),
                }
            )
        except Exception as e:
            logger.error(f"Error fetching URL {source.path}: {e}")
            return None
    
    def _ingest_s3(self, source: DataSource) -> Optional[IngestedItem]:
        """Ingest content from S3."""
        try:
            import boto3
            
            bucket = source.metadata["bucket"]
            key = source.metadata["key"]
            
            s3 = boto3.client("s3")
            response = s3.get_object(Bucket=bucket, Key=key)
            content = response["Body"].read()
            
            return IngestedItem(
                source=source,
                content=content,
                content_type=response.get("ContentType", "application/octet-stream"),
                metadata={
                    **source.metadata,
                    "etag": response.get("ETag", "").strip('"'),
                }
            )
        except Exception as e:
            logger.error(f"Error reading S3 object {source.path}: {e}")
            return None
    
    def _ingest_gcs(self, source: DataSource) -> Optional[IngestedItem]:
        """Ingest content from Google Cloud Storage."""
        try:
            from google.cloud import storage
            
            bucket_name = source.metadata["bucket"]
            blob_name = source.metadata["name"]
            
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            content = blob.download_as_bytes()
            
            return IngestedItem(
                source=source,
                content=content,
                content_type=blob.content_type or "application/octet-stream",
                metadata={
                    **source.metadata,
                    "md5_hash": blob.md5_hash,
                }
            )
        except Exception as e:
            logger.error(f"Error reading GCS object {source.path}: {e}")
            return None
    
    async def ingest_async(
        self, 
        sources: Optional[List[DataSource]] = None,
        max_concurrent: Optional[int] = None,
    ) -> AsyncGenerator[IngestedItem, None]:
        """
        Async version of ingest for better performance with I/O-bound sources.
        """
        if sources is None:
            sources = [s for s in self.manifest if not s.processed]
        
        max_concurrent = max_concurrent or self.ingestion_config.max_concurrent_downloads
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def ingest_with_semaphore(source: DataSource):
            async with semaphore:
                return await self._ingest_source_async(source)
        
        tasks = [ingest_with_semaphore(source) for source in sources]
        
        for coro in asyncio.as_completed(tasks):
            item = await coro
            if item:
                yield item
    
    async def _ingest_source_async(self, source: DataSource) -> Optional[IngestedItem]:
        """Async ingestion of a single source."""
        if source.source_type == SourceType.FILE:
            return await self._ingest_file_async(source)
        elif source.source_type == SourceType.URL:
            return await self._ingest_url_async(source)
        else:
            # Fall back to sync for cloud storage
            return self._ingest_source(source)
    
    async def _ingest_file_async(self, source: DataSource) -> Optional[IngestedItem]:
        """Async file reading."""
        try:
            is_text = source.content_type and source.content_type.startswith("text/")
            mode = "r" if is_text else "rb"
            
            async with aiofiles.open(source.path, mode) as f:
                content = await f.read()
            
            return IngestedItem(
                source=source,
                content=content,
                content_type=source.content_type or "application/octet-stream",
                metadata=source.metadata,
            )
        except Exception as e:
            logger.error(f"Error reading file async {source.path}: {e}")
            return None
    
    async def _ingest_url_async(self, source: DataSource) -> Optional[IngestedItem]:
        """Async URL fetching."""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {"User-Agent": self.ingestion_config.user_agent}
                async with session.get(
                    source.path,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self.ingestion_config.download_timeout),
                ) as response:
                    content = await response.text() if "text" in response.content_type else await response.read()
                    
                    return IngestedItem(
                        source=source,
                        content=content,
                        content_type=response.content_type,
                        metadata={
                            **source.metadata,
                            "status_code": response.status,
                        }
                    )
        except Exception as e:
            logger.error(f"Error fetching URL async {source.path}: {e}")
            return None
    
    @staticmethod
    def from_huggingface(
        dataset_name: str,
        split: str = "train",
        max_samples: Optional[int] = None,
        text_column: str = "text",
    ) -> Generator[IngestedItem, None, None]:
        """
        Ingest data from a HuggingFace dataset.
        
        Args:
            dataset_name: Name of the dataset (e.g., "togethercomputer/RedPajama-Data-1T-Sample")
            split: Dataset split to use
            max_samples: Maximum number of samples to ingest
            text_column: Column containing text data
        """
        try:
            from datasets import load_dataset
            
            if max_samples:
                split = f"{split}[:{max_samples}]"
            
            dataset = load_dataset(dataset_name, split=split, trust_remote_code=True)
            
            for idx, item in enumerate(dataset):
                text = item.get(text_column, "")
                if text:
                    yield IngestedItem(
                        source=DataSource(
                            path=f"huggingface://{dataset_name}/{idx}",
                            source_type=SourceType.HUGGINGFACE,
                            metadata={
                                "dataset": dataset_name,
                                "index": idx,
                                **{k: v for k, v in item.items() if k != text_column and isinstance(v, (str, int, float))}
                            }
                        ),
                        content=text,
                        content_type="text/plain",
                        metadata={"source": "huggingface", "dataset": dataset_name}
                    )
        except ImportError:
            logger.error("datasets library not installed. Run: pip install datasets")
        except Exception as e:
            logger.error(f"Error loading HuggingFace dataset: {e}")
