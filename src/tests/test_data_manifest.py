import json

import pytest

from src.core.data.manifest import (
    DataManifest,
    ShardEntry,
    build_manifest,
    sha256_file,
)


def _write(path, content: bytes) -> str:
    path.write_bytes(content)
    return str(path)


def test_sha256_file_deterministic(tmp_path):
    p = tmp_path / "a.bin"
    p.write_bytes(b"hello world")
    h1 = sha256_file(p)
    h2 = sha256_file(p)
    assert h1 == h2
    assert len(h1) == 64


def test_sha256_file_differs_on_content_change(tmp_path):
    p = tmp_path / "a.bin"
    p.write_bytes(b"hello")
    h1 = sha256_file(p)
    p.write_bytes(b"world")
    h2 = sha256_file(p)
    assert h1 != h2


def test_build_manifest_sorted_by_path(tmp_path):
    a = _write(tmp_path / "z.bin", b"za")
    b = _write(tmp_path / "a.bin", b"ab")
    c = _write(tmp_path / "m.bin", b"mc")
    mf = build_manifest([a, b, c], order_seed=42)
    paths = [s.path for s in mf.shards]
    assert paths == sorted(paths)
    assert mf.order_seed == 42


def test_build_manifest_records_sizes_and_hashes(tmp_path):
    p = _write(tmp_path / "a.bin", b"abcdef")
    mf = build_manifest([p], order_seed=0)
    assert len(mf.shards) == 1
    s = mf.shards[0]
    assert s.size_bytes == 6
    assert s.sha256 == sha256_file(p)


def test_build_manifest_with_num_examples(tmp_path):
    p1 = _write(tmp_path / "a.bin", b"a")
    p2 = _write(tmp_path / "b.bin", b"b")
    mf = build_manifest([p1, p2], order_seed=1, num_examples=[10, 20])
    by_path = {s.path: s for s in mf.shards}
    assert by_path[p1].num_examples == 10
    assert by_path[p2].num_examples == 20


def test_build_manifest_rejects_mismatched_num_examples(tmp_path):
    p = _write(tmp_path / "a.bin", b"a")
    with pytest.raises(ValueError, match="num_examples"):
        build_manifest([p], order_seed=0, num_examples=[1, 2])


def test_build_manifest_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_manifest([str(tmp_path / "nope.bin")], order_seed=0)


def test_digest_is_stable(tmp_path):
    p = _write(tmp_path / "a.bin", b"abc")
    mf = build_manifest([p], order_seed=7)
    assert mf.digest() == mf.digest()
    assert len(mf.digest()) == 64


def test_digest_changes_with_seed(tmp_path):
    p = _write(tmp_path / "a.bin", b"abc")
    mf1 = build_manifest([p], order_seed=1)
    mf2 = build_manifest([p], order_seed=2)
    assert mf1.digest() != mf2.digest()


def test_save_load_roundtrip(tmp_path):
    p = _write(tmp_path / "a.bin", b"abc")
    mf = build_manifest([p], order_seed=11)
    out = tmp_path / "manifest.json"
    mf.save(out)
    loaded = DataManifest.load(out)
    assert loaded.digest() == mf.digest()
    assert loaded.order_seed == 11
    assert len(loaded.shards) == 1


def test_verify_clean(tmp_path):
    p = _write(tmp_path / "a.bin", b"abc")
    mf = build_manifest([p], order_seed=0)
    assert mf.verify() == []


def test_verify_detects_tampering(tmp_path):
    p = tmp_path / "a.bin"
    p.write_bytes(b"original")
    mf = build_manifest([str(p)], order_seed=0)
    p.write_bytes(b"tampered")
    broken = mf.verify()
    assert broken == [str(p)]


def test_verify_detects_missing(tmp_path):
    p = tmp_path / "a.bin"
    p.write_bytes(b"x")
    mf = build_manifest([str(p)], order_seed=0)
    p.unlink()
    broken = mf.verify()
    assert broken == [str(p)]


def test_from_dict_handles_missing_optional_fields():
    raw = {"order_seed": 5, "shards": []}
    mf = DataManifest.from_dict(raw)
    assert mf.order_seed == 5
    assert mf.shards == []
    assert mf.version == 1


def test_shard_entry_immutable():
    s = ShardEntry(path="a", size_bytes=1, sha256="x")
    with pytest.raises(Exception):
        s.path = "b"  # type: ignore[misc]


def test_save_produces_valid_json(tmp_path):
    p = _write(tmp_path / "a.bin", b"abc")
    mf = build_manifest([p], order_seed=3)
    out = tmp_path / "m.json"
    mf.save(out)
    data = json.loads(out.read_text())
    assert "shards" in data
    assert data["order_seed"] == 3
