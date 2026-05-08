"""Production benchmark dataset downloader with HF / HTTP fetch, checksum, and caching."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


DEFAULT_CACHE_DIR = Path(
    os.environ.get("RTDLM_DATASET_CACHE", str(Path.home() / ".cache" / "rtdlm" / "datasets"))
)


@dataclass
class DatasetSpec:
    """Declarative dataset descriptor for the registry."""
    name: str
    source: str
    repo_id: Optional[str] = None
    subset: Optional[str] = None
    split: Optional[str] = None
    urls: List[str] = field(default_factory=list)
    sha256: Dict[str, str] = field(default_factory=dict)
    license: str = "unknown"
    homepage: str = ""
    citation: str = ""
    requires_auth: bool = False


class DatasetCache:
    """Local on-disk cache with integrity verification."""

    def __init__(self, root: Optional[Path] = None):
        """Initialize cache rooted at `root` (default: ~/.cache/rtdlm/datasets)."""
        self.root = Path(root) if root is not None else DEFAULT_CACHE_DIR
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, dataset_name: str, filename: str) -> Path:
        """Return the on-disk cache path for a dataset file."""
        return self.root / dataset_name / filename

    def has(self, dataset_name: str, filename: str, expected_sha256: Optional[str] = None) -> bool:
        """Return True iff file is present and (optionally) checksum matches."""
        p = self.path_for(dataset_name, filename)
        if not p.exists():
            return False
        if expected_sha256 is None:
            return True
        return self._sha256(p) == expected_sha256.lower()

    def store(self, dataset_name: str, filename: str, src: Path) -> Path:
        """Atomically move `src` into the cache and return the destination path."""
        dst = self.path_for(dataset_name, filename)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        return dst

    def write_metadata(self, dataset_name: str, meta: Dict[str, Any]) -> Path:
        """Persist a JSON metadata sidecar for the dataset."""
        p = self.path_for(dataset_name, "_meta.json")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(meta, indent=2, sort_keys=True))
        return p

    @staticmethod
    def _sha256(path: Path, chunk: int = 1 << 20) -> str:
        """Compute SHA-256 of a file in streaming chunks."""
        h = hashlib.sha256()
        with path.open("rb") as f:
            for buf in iter(lambda: f.read(chunk), b""):
                h.update(buf)
        return h.hexdigest()


class HTTPFetcher:
    """Minimal urllib-based downloader with retry + integrity check."""

    def __init__(self, max_retries: int = 3, backoff_seconds: float = 1.5, timeout: int = 60):
        """Configure retry/backoff/timeout policy."""
        self.max_retries = max_retries
        self.backoff = backoff_seconds
        self.timeout = timeout

    def fetch(self, url: str, dst: Path, expected_sha256: Optional[str] = None) -> Path:
        """Download `url` to `dst` with retries; verify checksum if provided."""
        dst.parent.mkdir(parents=True, exist_ok=True)
        last_err: Optional[BaseException] = None
        for attempt in range(self.max_retries):
            try:
                self._download_once(url, dst)
                if expected_sha256 is not None:
                    actual = DatasetCache._sha256(dst)
                    if actual != expected_sha256.lower():
                        raise ValueError(
                            f"checksum mismatch for {url}: {actual} != {expected_sha256}"
                        )
                return dst
            except (urllib.error.URLError, ValueError) as exc:
                last_err = exc
                logger.warning("fetch attempt %d failed for %s: %s", attempt + 1, url, exc)
                time.sleep(self.backoff * (attempt + 1))
        raise RuntimeError(f"failed to download {url} after {self.max_retries} attempts: {last_err}")

    def _download_once(self, url: str, dst: Path) -> None:
        """Single download attempt to a temp file then atomic rename."""
        with tempfile.NamedTemporaryFile(delete=False, dir=dst.parent) as tmp:
            tmp_path = Path(tmp.name)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "rtdlm-dataset-fetcher/1.0"})
            with urllib.request.urlopen(req, timeout=self.timeout) as resp, tmp_path.open("wb") as out:
                shutil.copyfileobj(resp, out)
            shutil.move(str(tmp_path), str(dst))
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)


class HuggingFaceFetcher:
    """Thin wrapper around the optional `datasets` library."""

    def __init__(self, cache_dir: Optional[Path] = None):
        """Optionally direct HF cache to a custom dir."""
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None

    def available(self) -> bool:
        """Return True iff the `datasets` library can be imported."""
        try:
            import datasets  # noqa: F401
            return True
        except ImportError:
            return False

    def load(
        self,
        repo_id: str,
        subset: Optional[str] = None,
        split: Optional[str] = None,
        token: Optional[str] = None,
    ) -> Any:
        """Load a HF dataset; raises RuntimeError if `datasets` isn't installed."""
        if not self.available():
            raise RuntimeError("`datasets` package not installed; pip install datasets")
        from datasets import load_dataset  # type: ignore

        kwargs: Dict[str, Any] = {}
        if self.cache_dir is not None:
            kwargs["cache_dir"] = str(self.cache_dir)
        if split is not None:
            kwargs["split"] = split
        if token is not None:
            kwargs["token"] = token
        return load_dataset(repo_id, subset, **kwargs)


_REGISTRY: Dict[str, DatasetSpec] = {
    "gpqa_diamond": DatasetSpec(
        name="gpqa_diamond",
        source="huggingface",
        repo_id="Idavidrein/gpqa",
        subset="gpqa_diamond",
        split="train",
        license="CC-BY-4.0",
        homepage="https://huggingface.co/datasets/Idavidrein/gpqa",
        citation="Rein et al. 2023 — GPQA: A Graduate-Level Google-Proof Q&A Benchmark",
        requires_auth=True,
    ),
    "aime": DatasetSpec(
        name="aime",
        source="huggingface",
        repo_id="di-dimitrov/aime-problems",
        split="train",
        license="research-only",
        homepage="https://huggingface.co/datasets/di-dimitrov/aime-problems",
        citation="MAA — American Invitational Mathematics Examination",
    ),
    "swe_bench_verified": DatasetSpec(
        name="swe_bench_verified",
        source="huggingface",
        repo_id="princeton-nlp/SWE-bench_Verified",
        split="test",
        license="MIT",
        homepage="https://www.swebench.com",
        citation="Jimenez et al. 2024 — SWE-bench: Can LMs Resolve Real-World GitHub Issues?",
    ),
    "livecodebench": DatasetSpec(
        name="livecodebench",
        source="huggingface",
        repo_id="livecodebench/code_generation_lite",
        split="test",
        license="CC-BY-4.0",
        homepage="https://livecodebench.github.io",
        citation="Jain et al. 2024 — LiveCodeBench",
    ),
    "mmlu": DatasetSpec(
        name="mmlu",
        source="huggingface",
        repo_id="cais/mmlu",
        subset="all",
        split="test",
        license="MIT",
        homepage="https://huggingface.co/datasets/cais/mmlu",
        citation="Hendrycks et al. 2021 — MMLU",
    ),
}


def list_registered() -> List[str]:
    """Return names of all registered datasets."""
    return sorted(_REGISTRY.keys())


def get_spec(name: str) -> DatasetSpec:
    """Return spec for a registered dataset; raises KeyError otherwise."""
    if name not in _REGISTRY:
        raise KeyError(f"unknown dataset '{name}'; known: {list_registered()}")
    return _REGISTRY[name]


def register_dataset(spec: DatasetSpec) -> None:
    """Add a dataset spec to the in-process registry (idempotent)."""
    _REGISTRY[spec.name] = spec


def _fetch_huggingface(
    spec: DatasetSpec,
    cache: DatasetCache,
    token: Optional[str],
    progress_callback: Optional[Callable[[str, int, int], None]],
) -> List[Path]:
    """Download a HF dataset and persist as a flat JSON file in the cache."""
    hf = HuggingFaceFetcher(cache_dir=cache.root / spec.name)
    if not hf.available():
        raise RuntimeError(f"dataset '{spec.name}' requires `datasets` (pip install datasets)")
    if spec.repo_id is None:
        raise ValueError(f"dataset '{spec.name}' has no repo_id but source=huggingface")
    ds = hf.load(spec.repo_id, spec.subset, spec.split, token=token)
    out = cache.path_for(spec.name, "data.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    try:
        for i, row in enumerate(ds):
            rows.append(dict(row))
            if progress_callback is not None and i % 256 == 0:
                progress_callback(spec.name, i, -1)
    except TypeError:
        rows = [dict(r) for r in ds]
    out.write_text(json.dumps(rows))
    return [out]


def _fetch_http(spec: DatasetSpec, cache: DatasetCache) -> List[Path]:
    """Download every URL in `spec.urls` with retry + checksum verification."""
    fetcher = HTTPFetcher()
    paths: List[Path] = []
    for url in spec.urls:
        filename = Path(url).name or hashlib.sha1(url.encode()).hexdigest()
        expected = spec.sha256.get(filename)
        if cache.has(spec.name, filename, expected):
            paths.append(cache.path_for(spec.name, filename))
            continue
        tmp = cache.path_for(spec.name, filename + ".tmp")
        fetcher.fetch(url, tmp, expected_sha256=expected)
        paths.append(cache.store(spec.name, filename, tmp))
    return paths


def download_dataset(
    name: str,
    cache: Optional[DatasetCache] = None,
    token: Optional[str] = None,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> Tuple[DatasetSpec, List[Path]]:
    """Download a registered dataset; returns (spec, list_of_local_paths)."""
    spec = get_spec(name)
    cache = cache or DatasetCache()

    if spec.source == "huggingface" and spec.repo_id is not None:
        paths = _fetch_huggingface(spec, cache, token, progress_callback)
    elif spec.source == "http":
        paths = _fetch_http(spec, cache)
    else:
        raise ValueError(f"unsupported source '{spec.source}' for dataset '{name}'")

    cache.write_metadata(
        name,
        {
            "name": spec.name,
            "source": spec.source,
            "repo_id": spec.repo_id,
            "subset": spec.subset,
            "split": spec.split,
            "license": spec.license,
            "homepage": spec.homepage,
            "citation": spec.citation,
            "files": [str(p) for p in paths],
            "downloaded_at": time.time(),
        },
    )
    return spec, paths


def verify_cached_dataset(name: str, cache: Optional[DatasetCache] = None) -> bool:
    """Re-hash all cached files for a dataset and verify against registered checksums."""
    spec = get_spec(name)
    cache = cache or DatasetCache()
    if not spec.sha256:
        return True
    for filename, expected in spec.sha256.items():
        if not cache.has(name, filename, expected):
            return False
    return True
