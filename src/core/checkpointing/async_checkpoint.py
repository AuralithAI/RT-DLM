"""Async checkpointing with optional Orbax backend, threaded SafeTensors fallback."""

from __future__ import annotations

import concurrent.futures
import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.core.checkpoint_manager import (
    CheckpointMetadata,
    CheckpointManager,
    flatten_params,
)

logger = logging.getLogger(__name__)

_MANIFEST_SUFFIX = ".manifest.json"


@dataclass
class CheckpointManifest:
    """Async checkpoint manifest capturing reproducibility state."""

    step: int
    epoch: int
    rng_state: List[int] = field(default_factory=list)
    data_cursor: int = 0
    data_manifest_digest: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "epoch": self.epoch,
            "rng_state": list(self.rng_state),
            "data_cursor": self.data_cursor,
            "data_manifest_digest": self.data_manifest_digest,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointManifest":
        return cls(
            step=int(data["step"]),
            epoch=int(data["epoch"]),
            rng_state=list(data.get("rng_state", [])),
            data_cursor=int(data.get("data_cursor", 0)),
            data_manifest_digest=data.get("data_manifest_digest"),
            extra=dict(data.get("extra", {})),
        )


class AsyncCheckpointer:
    """Non-blocking checkpointer with a single background worker thread."""

    def __init__(self, checkpoint_dir: str, keep_last: int = 3):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last = keep_last
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="ckpt")
        self._pending: List[concurrent.futures.Future] = []
        self._lock = threading.Lock()
        self._sync_manager = CheckpointManager(str(self.checkpoint_dir))

    def save(
        self,
        params: Dict,
        opt_state: Any,
        epoch: int,
        step: int,
        manifest: CheckpointManifest,
        metrics: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> concurrent.futures.Future:
        """Schedule an async save; returns a Future that resolves when the write is durable."""
        snapshot = self._snapshot(params)
        opt_snapshot = self._snapshot(opt_state)
        future = self._executor.submit(
            self._do_save,
            snapshot,
            opt_snapshot,
            epoch,
            step,
            manifest.to_dict(),
            metrics or {},
            config or {},
        )
        with self._lock:
            self._pending.append(future)
        future.add_done_callback(self._on_done)
        return future

    def wait(self, timeout: Optional[float] = None) -> None:
        """Block until all pending writes complete."""
        with self._lock:
            pending = list(self._pending)
        for fut in pending:
            fut.result(timeout=timeout)

    def shutdown(self) -> None:
        """Wait for outstanding work and stop the worker."""
        self.wait()
        self._executor.shutdown(wait=True)

    def list_checkpoints(self) -> List[Path]:
        return sorted(self.checkpoint_dir.glob("rtdlm_agi_epoch_*.safetensors"))

    @staticmethod
    def _snapshot(tree: Any) -> Any:
        """Materialize a numpy-only copy on the calling thread."""
        if tree is None:
            return None
        try:
            import jax
            return jax.tree_util.tree_map(lambda x: np.asarray(x) if hasattr(x, "shape") else x, tree)
        except Exception:
            return tree

    def _do_save(
        self,
        params: Dict,
        opt_state: Any,
        epoch: int,
        step: int,
        manifest_dict: Dict[str, Any],
        metrics: Dict[str, Any],
        config: Dict[str, Any],
    ) -> str:
        path = self._sync_manager.save_checkpoint(
            params=params,
            opt_state=opt_state,
            epoch=epoch,
            step_count=step,
            metrics=metrics,
            config=config,
        )
        manifest_path = Path(path).with_suffix(_MANIFEST_SUFFIX)
        manifest_path.write_text(json.dumps(manifest_dict, indent=2, sort_keys=True))
        self._prune()
        return str(path)

    def _on_done(self, fut: concurrent.futures.Future) -> None:
        with self._lock:
            if fut in self._pending:
                self._pending.remove(fut)
        if fut.exception() is not None:
            logger.error("Async checkpoint failed: %s", fut.exception())

    def _prune(self) -> None:
        ckpts = self.list_checkpoints()
        if len(ckpts) <= self.keep_last:
            return
        for old in ckpts[: len(ckpts) - self.keep_last]:
            try:
                old.unlink()
                manifest = old.with_suffix(_MANIFEST_SUFFIX)
                if manifest.exists():
                    manifest.unlink()
            except OSError as e:
                logger.warning("Failed to prune %s: %s", old, e)

    def load_manifest(self, checkpoint_path: str) -> Optional[CheckpointManifest]:
        manifest_path = Path(checkpoint_path).with_suffix(_MANIFEST_SUFFIX)
        if not manifest_path.exists():
            return None
        return CheckpointManifest.from_dict(json.loads(manifest_path.read_text()))


__all__ = ["AsyncCheckpointer", "CheckpointManifest"]
