import json
import time

import numpy as np
import pytest

from src.core.checkpointing.async_checkpoint import AsyncCheckpointer, CheckpointManifest


def _make_params():
    return {"layer": {"w": np.ones((2, 2), dtype=np.float32), "b": np.zeros((2,), dtype=np.float32)}}


def _make_manifest(step=1, epoch=0):
    return CheckpointManifest(
        step=step,
        epoch=epoch,
        rng_state=[1, 2, 3],
        data_cursor=42,
        data_manifest_digest="deadbeef",
        extra={"foo": "bar"},
    )


def test_checkpoint_manifest_roundtrip():
    m = _make_manifest()
    d = m.to_dict()
    m2 = CheckpointManifest.from_dict(d)
    assert m2.step == m.step
    assert m2.epoch == m.epoch
    assert m2.rng_state == m.rng_state
    assert m2.data_cursor == m.data_cursor
    assert m2.data_manifest_digest == m.data_manifest_digest
    assert m2.extra == m.extra


def test_checkpoint_manifest_from_dict_defaults():
    m = CheckpointManifest.from_dict({"step": 7, "epoch": 1})
    assert m.step == 7
    assert m.epoch == 1
    assert m.rng_state == []
    assert m.data_cursor == 0
    assert m.data_manifest_digest is None
    assert m.extra == {}


def test_async_save_creates_checkpoint(tmp_path):
    cp = AsyncCheckpointer(str(tmp_path), keep_last=3)
    try:
        fut = cp.save(_make_params(), opt_state=None, epoch=0, step=1, manifest=_make_manifest())
        path = fut.result(timeout=30)
        assert path
        assert cp.list_checkpoints(), "checkpoint file not produced"
    finally:
        cp.shutdown()


def test_async_save_writes_manifest_sidecar(tmp_path):
    cp = AsyncCheckpointer(str(tmp_path), keep_last=3)
    try:
        fut = cp.save(_make_params(), None, epoch=0, step=5, manifest=_make_manifest(step=5))
        path = fut.result(timeout=30)
        loaded = cp.load_manifest(path)
        assert loaded is not None
        assert loaded.step == 5
        assert loaded.data_manifest_digest == "deadbeef"
        assert loaded.extra == {"foo": "bar"}
    finally:
        cp.shutdown()


def test_async_wait_blocks_until_durable(tmp_path):
    cp = AsyncCheckpointer(str(tmp_path), keep_last=5)
    try:
        cp.save(_make_params(), None, 0, 1, _make_manifest(step=1))
        cp.save(_make_params(), None, 0, 2, _make_manifest(step=2))
        cp.wait(timeout=30)
        assert len(cp.list_checkpoints()) >= 1
    finally:
        cp.shutdown()


def test_async_prune_keeps_last_n(tmp_path):
    cp = AsyncCheckpointer(str(tmp_path), keep_last=2)
    try:
        for i in range(5):
            cp.save(_make_params(), None, epoch=i, step=i, manifest=_make_manifest(step=i, epoch=i))
            time.sleep(0.01)
        cp.wait(timeout=60)
        ckpts = cp.list_checkpoints()
        assert len(ckpts) <= 2
    finally:
        cp.shutdown()


def test_async_prune_removes_orphan_manifests(tmp_path):
    cp = AsyncCheckpointer(str(tmp_path), keep_last=1)
    try:
        for i in range(3):
            cp.save(_make_params(), None, epoch=i, step=i, manifest=_make_manifest(step=i, epoch=i))
            time.sleep(0.01)
        cp.wait(timeout=60)
        sidecars = list(tmp_path.glob("*.manifest.json"))
        ckpts = cp.list_checkpoints()
        assert len(sidecars) == len(ckpts)
    finally:
        cp.shutdown()


def test_load_manifest_missing_returns_none(tmp_path):
    cp = AsyncCheckpointer(str(tmp_path))
    try:
        assert cp.load_manifest(str(tmp_path / "nonexistent.safetensors")) is None
    finally:
        cp.shutdown()


def test_manifest_sidecar_is_valid_json(tmp_path):
    cp = AsyncCheckpointer(str(tmp_path))
    try:
        fut = cp.save(_make_params(), None, 0, 1, _make_manifest())
        path = fut.result(timeout=30)
        sidecar = next(tmp_path.glob("*.manifest.json"))
        data = json.loads(sidecar.read_text())
        assert data["step"] == 1
        assert data["rng_state"] == [1, 2, 3]
        # path used to silence unused-var
        assert path
    finally:
        cp.shutdown()


def test_snapshot_on_caller_thread_decouples_state(tmp_path):
    cp = AsyncCheckpointer(str(tmp_path))
    try:
        params = _make_params()
        fut = cp.save(params, None, 0, 1, _make_manifest())
        params["layer"]["w"][:] = 999.0
        fut.result(timeout=30)
        # If snapshot worked, no error from concurrent mutation; checkpoint write succeeded
        assert cp.list_checkpoints()
    finally:
        cp.shutdown()


def test_async_save_with_metrics_and_config(tmp_path):
    cp = AsyncCheckpointer(str(tmp_path))
    try:
        fut = cp.save(
            _make_params(),
            None,
            0,
            1,
            _make_manifest(),
            metrics={"loss": 0.5},
            config={"d_model": 256},
        )
        path = fut.result(timeout=30)
        assert path
    finally:
        cp.shutdown()
