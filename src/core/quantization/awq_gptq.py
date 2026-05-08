"""AWQ / GPTQ post-training quantization and a minimal GGUF-compatible writer."""

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import jax.numpy as jnp
import numpy as np


@dataclass
class AWQConfig:
    """Configuration for activation-aware weight quantization."""

    bits: int = 4
    group_size: int = 128
    zero_point: bool = True
    scale_clip_min: float = 1e-5


@dataclass
class GPTQConfig:
    """Configuration for GPTQ ordered weight quantization."""

    bits: int = 4
    block_size: int = 128
    percdamp: float = 0.01
    actorder: bool = True


@dataclass
class QuantTensor:
    """Quantized weight bundle: codes + scales + zero points + metadata."""

    codes: np.ndarray
    scales: np.ndarray
    zeros: Optional[np.ndarray]
    bits: int
    group_size: int
    shape: Tuple[int, ...]


def _q_range(bits: int) -> Tuple[int, int]:
    """Inclusive integer range for `bits` quantization."""
    return 0, (1 << bits) - 1


def _grouped_minmax(weight: np.ndarray, group_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Per-group min/max along the last axis."""
    flat = weight.reshape(weight.shape[0], -1)
    cols = flat.shape[1]
    pad = (group_size - cols % group_size) % group_size
    if pad:
        flat = np.pad(flat, ((0, 0), (0, pad)), constant_values=0.0)
    grouped = flat.reshape(flat.shape[0], -1, group_size)
    return grouped.min(axis=-1), grouped.max(axis=-1)


def awq_calibrate_scales(weight: np.ndarray, activation_stats: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Per-channel AWQ scaling factor sqrt(|act|^alpha / |w|^(1-alpha))."""
    a = np.maximum(np.abs(activation_stats), 1e-5)
    w = np.maximum(np.abs(weight).mean(axis=0), 1e-5)
    s = np.power(a, alpha) / np.power(w, 1.0 - alpha)
    return s / s.mean()


def awq_quantize_layer(
    weight: np.ndarray,
    activation_stats: np.ndarray,
    config: AWQConfig,
) -> QuantTensor:
    """Apply AWQ scaling then group-wise asymmetric quantization."""
    qmin, qmax = _q_range(config.bits)
    scale_act = awq_calibrate_scales(weight, activation_stats)
    scaled = weight * scale_act[None, :]
    g_min, g_max = _grouped_minmax(scaled, config.group_size)
    scale = np.maximum((g_max - g_min) / float(qmax - qmin), config.scale_clip_min)
    zero = np.round(-g_min / scale) if config.zero_point else np.zeros_like(scale)

    flat = scaled.reshape(scaled.shape[0], -1)
    pad = (config.group_size - flat.shape[1] % config.group_size) % config.group_size
    if pad:
        flat = np.pad(flat, ((0, 0), (0, pad)), constant_values=0.0)
    grouped = flat.reshape(flat.shape[0], -1, config.group_size)
    codes = np.clip(np.round(grouped / scale[:, :, None]) + zero[:, :, None], qmin, qmax).astype(np.int32)
    return QuantTensor(
        codes=codes,
        scales=(
            (scale / scale_act[: scale.shape[1]][None, :].repeat(scale.shape[0], axis=0))
            if scale.shape[1] == scale_act.shape[0]
            else scale
        ),
        zeros=zero.astype(np.int32) if config.zero_point else None,
        bits=config.bits,
        group_size=config.group_size,
        shape=weight.shape,
    )


def awq_dequantize(qt: QuantTensor) -> np.ndarray:
    """Inverse of `awq_quantize_layer` (lossy reconstruction)."""
    zeros = qt.zeros if qt.zeros is not None else np.zeros_like(qt.scales, dtype=np.int32)
    deq = (qt.codes - zeros[:, :, None]) * qt.scales[:, :, None]
    flat = deq.reshape(qt.shape[0], -1)
    return flat[:, : np.prod(qt.shape[1:])].reshape(qt.shape)


def _add_diag_damp(matrix: np.ndarray, damp: float) -> np.ndarray:
    """Add a diagonal damping term scaled by mean diagonal."""
    diag_mean = float(np.mean(np.diag(matrix))) + 1e-8
    n = matrix.shape[0]
    return matrix + damp * diag_mean * np.eye(n, dtype=matrix.dtype)


def gptq_quantize_layer(
    weight: np.ndarray,
    hessian: np.ndarray,
    config: GPTQConfig,
) -> QuantTensor:
    """GPTQ ordered quantization with iterative error compensation."""
    out_dim, in_dim = weight.shape
    qmin, qmax = _q_range(config.bits)
    h = _add_diag_damp(hessian.astype(np.float64), config.percdamp)
    if config.actorder:
        order = np.argsort(-np.diag(h))
    else:
        order = np.arange(in_dim)
    inv_order = np.argsort(order)
    w = weight[:, order].astype(np.float64).copy()
    h = h[order][:, order]

    try:
        h_inv = np.linalg.cholesky(np.linalg.inv(h)).T
    except np.linalg.LinAlgError:
        h_inv = np.eye(in_dim, dtype=np.float64)

    codes = np.zeros_like(w, dtype=np.int32)
    scales: List[np.ndarray] = []
    zeros: List[np.ndarray] = []

    for start in range(0, in_dim, config.block_size):
        end = min(start + config.block_size, in_dim)
        block = w[:, start:end]
        block_min = block.min(axis=1, keepdims=True)
        block_max = block.max(axis=1, keepdims=True)
        scale = np.maximum((block_max - block_min) / float(qmax - qmin), 1e-8)
        zero = np.round(-block_min / scale)
        for j in range(start, end):
            x = w[:, j : j + 1]
            q = np.clip(np.round(x / scale) + zero, qmin, qmax)
            codes[:, j : j + 1] = q.astype(np.int32)
            dequant = (q - zero) * scale
            err = (x - dequant) / max(h_inv[j, j], 1e-8)
            if j + 1 < end:
                w[:, j + 1 : end] -= err @ h_inv[j : j + 1, j + 1 : end]
        scales.append(scale.squeeze(-1))
        zeros.append(zero.squeeze(-1).astype(np.int32))

    codes = codes[:, inv_order]
    return QuantTensor(
        codes=codes.reshape(out_dim, 1, in_dim),
        scales=np.stack(scales, axis=1),
        zeros=np.stack(zeros, axis=1),
        bits=config.bits,
        group_size=config.block_size,
        shape=weight.shape,
    )


def gptq_dequantize(qt: QuantTensor) -> np.ndarray:
    """Approximate inverse of GPTQ quantization."""
    out_dim, _, in_dim = qt.codes.shape
    deq = np.zeros((out_dim, in_dim), dtype=np.float32)
    for i, start in enumerate(range(0, in_dim, qt.group_size)):
        end = min(start + qt.group_size, in_dim)
        scale = qt.scales[:, i : i + 1]
        zero = qt.zeros[:, i : i + 1] if qt.zeros is not None else 0
        deq[:, start:end] = (qt.codes[:, 0, start:end] - zero) * scale
    return deq.reshape(qt.shape)


_GGUF_MAGIC = b"GGUF"
_GGUF_VERSION = 3


@dataclass
class GGUFTensor:
    """Single tensor entry in a GGUF blob."""

    name: str
    array: np.ndarray
    quant_type: int = 0


@dataclass
class GGUFFile:
    """In-memory GGUF container."""

    tensors: List[GGUFTensor] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)


def _write_str(buf: List[bytes], s: str) -> None:
    """Write length-prefixed UTF-8 string to buffer."""
    encoded = s.encode("utf-8")
    buf.append(struct.pack("<Q", len(encoded)))
    buf.append(encoded)


def write_gguf(file: GGUFFile, path: str | Path) -> None:
    """Minimal GGUF serializer: header + tensor names + raw quantized data."""
    buf: List[bytes] = []
    buf.append(_GGUF_MAGIC)
    buf.append(struct.pack("<I", _GGUF_VERSION))
    buf.append(struct.pack("<Q", len(file.tensors)))
    buf.append(struct.pack("<Q", len(file.metadata)))
    for k, v in file.metadata.items():
        _write_str(buf, k)
        _write_str(buf, str(v))
    for t in file.tensors:
        _write_str(buf, t.name)
        buf.append(struct.pack("<I", t.array.ndim))
        for d in t.array.shape:
            buf.append(struct.pack("<Q", int(d)))
        buf.append(struct.pack("<I", t.quant_type))
        buf.append(t.array.astype(np.int8).tobytes())
    Path(path).write_bytes(b"".join(buf))


def quantize_params_awq(
    params: Dict[str, jnp.ndarray],
    activation_stats: Dict[str, jnp.ndarray],
    config: AWQConfig,
) -> Dict[str, QuantTensor]:
    """Quantize a dict of weight matrices using AWQ."""
    out: Dict[str, QuantTensor] = {}
    for name, w in params.items():
        if w.ndim != 2 or name not in activation_stats:
            continue
        out[name] = awq_quantize_layer(np.asarray(w), np.asarray(activation_stats[name]), config)
    return out


def quantize_params_gptq(
    params: Dict[str, jnp.ndarray],
    hessians: Dict[str, jnp.ndarray],
    config: GPTQConfig,
) -> Dict[str, QuantTensor]:
    """Quantize a dict of weight matrices using GPTQ."""
    out: Dict[str, QuantTensor] = {}
    for name, w in params.items():
        if w.ndim != 2 or name not in hessians:
            continue
        out[name] = gptq_quantize_layer(np.asarray(w), np.asarray(hessians[name]), config)
    return out
