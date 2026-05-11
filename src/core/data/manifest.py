"""Deterministic training data shard manifest with sha256 + order seed."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Optional


_CHUNK = 1 << 20


def sha256_file(path: str | os.PathLike) -> str:
    """Streaming sha256 over a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(_CHUNK)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class ShardEntry:
    """A single shard descriptor."""

    path: str
    size_bytes: int
    sha256: str
    num_examples: Optional[int] = None


@dataclass
class DataManifest:
    """Deterministic data manifest."""

    shards: List[ShardEntry]
    order_seed: int
    version: int = 1

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "order_seed": self.order_seed,
            "shards": [asdict(s) for s in self.shards],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DataManifest":
        return cls(
            version=int(data.get("version", 1)),
            order_seed=int(data["order_seed"]),
            shards=[ShardEntry(**s) for s in data.get("shards", [])],
        )

    def save(self, path: str | os.PathLike) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: str | os.PathLike) -> "DataManifest":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def digest(self) -> str:
        """Stable manifest digest (sha256 over canonical JSON)."""
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def verify(self) -> List[str]:
        """Return list of shard paths whose on-disk sha256 disagrees with the manifest."""
        broken: List[str] = []
        for shard in self.shards:
            if not os.path.exists(shard.path):
                broken.append(shard.path)
                continue
            if sha256_file(shard.path) != shard.sha256:
                broken.append(shard.path)
        return broken


def build_manifest(
    shard_paths: Iterable[str | os.PathLike],
    order_seed: int,
    num_examples: Optional[List[int]] = None,
) -> DataManifest:
    """Compute a manifest from a list of shard paths."""
    paths = [str(p) for p in shard_paths]
    if num_examples is not None and len(num_examples) != len(paths):
        raise ValueError("num_examples length must match shard_paths length")
    entries: List[ShardEntry] = []
    for i, p in enumerate(paths):
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        entries.append(
            ShardEntry(
                path=p,
                size_bytes=os.path.getsize(p),
                sha256=sha256_file(p),
                num_examples=num_examples[i] if num_examples is not None else None,
            )
        )
    entries.sort(key=lambda s: s.path)
    return DataManifest(shards=entries, order_seed=order_seed)
