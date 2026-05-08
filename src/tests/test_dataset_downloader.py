"""Tests for benchmark dataset downloader registry, cache, and HTTP fetcher."""

from __future__ import annotations

import hashlib
import http.server
import json
import socketserver
import threading
from pathlib import Path

import pytest

from src.core.benchmarks.dataset_downloader import (
    DatasetCache,
    DatasetSpec,
    HTTPFetcher,
    download_dataset,
    get_spec,
    list_registered,
    register_dataset,
    verify_cached_dataset,
)


@pytest.fixture
def tmp_cache(tmp_path: Path) -> DatasetCache:
    """Per-test cache rooted in tmp_path."""
    return DatasetCache(root=tmp_path / "cache")


def test_registry_contains_known_benchmarks():
    """Built-in registry must include the core benchmark suite."""
    names = list_registered()
    for required in ("gpqa_diamond", "aime", "swe_bench_verified", "livecodebench", "mmlu"):
        assert required in names


def test_get_spec_unknown_raises():
    """Looking up a missing dataset raises KeyError."""
    with pytest.raises(KeyError):
        get_spec("does_not_exist")


def test_register_dataset_idempotent():
    """register_dataset should overwrite duplicates without error."""
    spec = DatasetSpec(name="custom_test", source="http", urls=["http://x"], license="MIT")
    register_dataset(spec)
    register_dataset(spec)
    assert get_spec("custom_test").name == "custom_test"


def test_cache_path_for_returns_under_root(tmp_cache: DatasetCache):
    """Cache should isolate datasets in their own subdirs."""
    p = tmp_cache.path_for("d1", "file.bin")
    assert p.parent.name == "d1"
    assert tmp_cache.root in p.parents


def test_cache_has_returns_false_when_missing(tmp_cache: DatasetCache):
    """has() must return False for non-existent files."""
    assert not tmp_cache.has("nope", "missing.bin")


def test_cache_has_with_checksum_match(tmp_cache: DatasetCache, tmp_path: Path):
    """Checksum-verified has() returns True only when sha256 matches."""
    src = tmp_path / "src.bin"
    src.write_bytes(b"hello world")
    expected = hashlib.sha256(b"hello world").hexdigest()
    tmp_cache.store("d1", "f.bin", src)
    assert tmp_cache.has("d1", "f.bin", expected)
    assert not tmp_cache.has("d1", "f.bin", "0" * 64)


def test_cache_write_metadata_round_trip(tmp_cache: DatasetCache):
    """Metadata sidecar must be valid JSON."""
    p = tmp_cache.write_metadata("d1", {"x": 1, "y": "z"})
    loaded = json.loads(p.read_text())
    assert loaded == {"x": 1, "y": "z"}


def _start_http_server(directory: Path) -> tuple[int, threading.Thread, socketserver.TCPServer]:
    """Spin up a localhost HTTP server serving `directory`."""

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(directory), **kwargs)

        def log_message(self, *_: object) -> None:
            return

    server = socketserver.TCPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return port, thread, server


def test_http_fetcher_downloads_and_verifies(tmp_path: Path):
    """End-to-end HTTP download with checksum verification."""
    served = tmp_path / "served"
    served.mkdir()
    payload = b"benchmark sample content"
    (served / "file.bin").write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()

    port, _, server = _start_http_server(served)
    try:
        dst = tmp_path / "downloaded.bin"
        HTTPFetcher(max_retries=2, backoff_seconds=0.1).fetch(
            f"http://127.0.0.1:{port}/file.bin", dst, expected_sha256=expected
        )
        assert dst.read_bytes() == payload
    finally:
        server.shutdown()
        server.server_close()


def test_http_fetcher_checksum_mismatch_raises(tmp_path: Path):
    """Mismatched sha256 must raise after retries."""
    served = tmp_path / "served"
    served.mkdir()
    (served / "f.bin").write_bytes(b"abc")
    port, _, server = _start_http_server(served)
    try:
        with pytest.raises(RuntimeError):
            HTTPFetcher(max_retries=2, backoff_seconds=0.05).fetch(
                f"http://127.0.0.1:{port}/f.bin",
                tmp_path / "f.bin",
                expected_sha256="0" * 64,
            )
    finally:
        server.shutdown()
        server.server_close()


def test_download_dataset_http_round_trip(tmp_path: Path):
    """download_dataset() should pull, cache, and report HTTP-source datasets."""
    served = tmp_path / "served"
    served.mkdir()
    payload = b"row1\nrow2\n"
    (served / "rows.txt").write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()
    port, _, server = _start_http_server(served)
    try:
        url = f"http://127.0.0.1:{port}/rows.txt"
        spec = DatasetSpec(
            name="local_test_ds",
            source="http",
            urls=[url],
            sha256={"rows.txt": expected},
            license="CC0",
        )
        register_dataset(spec)
        cache = DatasetCache(root=tmp_path / "cache")
        result_spec, paths = download_dataset("local_test_ds", cache=cache)
        assert result_spec.name == "local_test_ds"
        assert len(paths) == 1
        assert paths[0].read_bytes() == payload
        assert verify_cached_dataset("local_test_ds", cache)
        meta = json.loads((cache.root / "local_test_ds" / "_meta.json").read_text())
        assert meta["license"] == "CC0"
    finally:
        server.shutdown()
        server.server_close()


def test_download_dataset_unknown_source_raises():
    """Unsupported source string must raise ValueError."""
    register_dataset(DatasetSpec(name="bad_src", source="ftp", license="?"))
    with pytest.raises(ValueError):
        download_dataset("bad_src")
