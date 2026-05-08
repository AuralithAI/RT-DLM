"""Tests for AWQ / GPTQ post-training quantization and GGUF writer."""

from pathlib import Path

import numpy as np
import pytest

from src.core.quantization.awq_gptq import (
    AWQConfig,
    GGUFFile,
    GGUFTensor,
    GPTQConfig,
    awq_calibrate_scales,
    awq_dequantize,
    awq_quantize_layer,
    gptq_dequantize,
    gptq_quantize_layer,
    write_gguf,
)


def test_awq_calibrate_scales_normalized():
    """AWQ scales should be mean-normalized to ~1."""
    rng = np.random.default_rng(0)
    w = rng.normal(size=(32, 16)).astype(np.float32)
    a = np.abs(rng.normal(size=16)).astype(np.float32) + 0.1
    scales = awq_calibrate_scales(w, a)
    assert scales.shape == (16,)
    assert float(scales.mean()) == pytest.approx(1.0, abs=1e-5)


def test_awq_quantize_dequantize_roundtrip():
    """AWQ dequantization should approximately recover the original weights."""
    rng = np.random.default_rng(1)
    w = rng.normal(size=(16, 32)).astype(np.float32)
    act = np.abs(rng.normal(size=32)).astype(np.float32) + 1.0
    qt = awq_quantize_layer(w, act, AWQConfig(bits=4, group_size=16))
    deq = awq_dequantize(qt)
    assert deq.shape == w.shape
    err = float(np.mean(np.abs(deq - w)))
    assert err < float(np.std(w)) * 1.5


def test_awq_codes_in_range():
    """AWQ codes must respect the integer range for the requested bit width."""
    rng = np.random.default_rng(2)
    w = rng.normal(size=(8, 16)).astype(np.float32)
    act = np.abs(rng.normal(size=16)).astype(np.float32) + 1.0
    qt = awq_quantize_layer(w, act, AWQConfig(bits=4, group_size=8))
    assert int(qt.codes.min()) >= 0
    assert int(qt.codes.max()) <= 15


def test_gptq_quantize_shape():
    """GPTQ output codes should have the right packed shape."""
    rng = np.random.default_rng(3)
    w = rng.normal(size=(8, 32)).astype(np.float32)
    h = (w.T @ w) + np.eye(32) * 0.1
    qt = gptq_quantize_layer(w, h, GPTQConfig(bits=4, block_size=16))
    assert qt.codes.shape == (8, 1, 32)


def test_gptq_dequantize_shape():
    """GPTQ dequantization should recover input shape."""
    rng = np.random.default_rng(4)
    w = rng.normal(size=(8, 32)).astype(np.float32)
    h = (w.T @ w) + np.eye(32) * 0.1
    qt = gptq_quantize_layer(w, h, GPTQConfig(bits=4, block_size=16))
    deq = gptq_dequantize(qt)
    assert deq.shape == w.shape


def test_gguf_writer_roundtrip(tmp_path: Path):
    """GGUF writer should produce a non-empty binary blob with the magic header."""
    f = GGUFFile(
        tensors=[GGUFTensor(name="w0", array=np.zeros((4, 4), dtype=np.int8))],
        metadata={"arch": "rt-dlm"},
    )
    out = tmp_path / "model.gguf"
    write_gguf(f, out)
    data = out.read_bytes()
    assert data.startswith(b"GGUF")
    assert len(data) > 32
