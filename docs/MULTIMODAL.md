# Multimodal Architecture

RT-DLM supports a heterogeneous set of input modalities through a unified
fusion stack. All encoders project into a common `d_model` representation that
feeds `MultiModalRTDLM` (see `src/modules/multimodal/fusion_module.py`).

## Modalities

| Modality       | Encoder                              | File                             |
|----------------|--------------------------------------|----------------------------------|
| Text           | `RTDLM`                              | `src/rtdlm.py`                   |
| Image (RGB/D)  | `VisionEncoder(in_channels=3 or 4)`  | `fusion_module.py`               |
| Video          | `VideoEncoder` (spatiotemporal+RoPE) | `fusion_module.py`               |
| Audio          | `HybridAudioEncoder` + `stft` + `LearnableFilterbank` | `hybrid_audio_module.py` |
| Document       | `DocumentEncoder` + `TableStructureEncoder` + `ChartDecoder` | `document_encoder.py` |
| Point Cloud    | `PointCloudEncoder` (PointNet++)     | `point_cloud_encoder.py`         |
| Biosignal      | `BiosignalEncoder` (channel attn)    | `biosignal_encoder.py`           |
| Tactile        | `TactileEncoder`                     | `tactile_encoder.py`             |
| Action         | `ActionEncoder` (discretized bins)   | `action_encoder.py`              |

## Reconstruction & Tokenization Heads

| Head                       | Purpose                              | File                            |
|----------------------------|--------------------------------------|---------------------------------|
| `SpectrogramDecoder`       | Reconstruct mel spectrograms         | `spectrogram_decoder.py`        |
| `VQVAEImageTokenizer`      | Discrete image codes (8192-token codebook by default) | `vqvae_image_tokenizer.py` |
| `StreamingVideoBuffer`     | Circular buffer for long video       | `streaming_video_buffer.py`     |

## Audio Pipeline

The waveform path is fully real (no synthetic features):

1. `stft(wave, n_fft=512, hop=160, win=400)` produces a magnitude spectrogram.
2. `LearnableFilterbank(n_mels=80)` initializes from a mel-slaney filterbank
   and adds a learned residual clipped at zero (preserves dynamic range
   without the saturation issues of `softplus`).
3. The mel features feed the existing `HybridAudioEncoder` (Conv + transformer).

Validation: a 440 Hz sine sampled at 16 kHz produces an STFT magnitude peak at
bin 14 (theoretical: 440·512/16000 ≈ 14.08).

## Video Pipeline

`VideoEncoder` preserves patch tokens across time:

- Per-frame `VisionEncoder` produces `(B, F, P, d)` patch tokens.
- A motion branch computes frame differences, applies two stride-2 Conv2D
  blocks, resizes to match patch grid, and projects to `d_model // 4`.
- Appearance and motion features concatenate and project back to `d_model`.
- `apply_3d_rope(x, axis_lens=(T, H, W))` injects 3D positional bias.
- A sparse temporal causal mask is applied: full intra-frame attention plus
  inter-frame attention restricted to a `motion_window` (default 4) of
  preceding frames.
- 3 stacked attention layers refine the spatiotemporal token sequence.

## Cross-Modal Synchronization

Two helpers in `fusion_module.py`:

- `sinusoidal_timestamp_embedding(timestamps, dim, max_period=10000.0)` builds a
  scale-invariant timestamp embedding usable as additive bias.
- `modality_synchronization_loss(feat_a, feat_b, ts_a, ts_b, tolerance=0.05)`
  penalizes feature divergence when timestamps fall within `tolerance` seconds.

## Configuration

All modalities are gated by flags on `AGIConfig`:

```python
from src.config.agi_config import AGIConfig

cfg = AGIConfig(
    vision_in_channels=4,                 # RGBD
    video_patch_size=16,
    video_motion_window=4,
    enable_document_modality=True,
    enable_pointcloud_modality=True,
    enable_biosignal_modality=True,
    enable_tactile_modality=True,
    enable_action_modality=True,
    action_num_axes=15,
    action_num_bins=256,
    enable_image_vq=True,
    image_vq_codes=8192,
    image_vq_code_dim=256,
    image_vq_downsample=16,
    enable_spectrogram_decoder=True,
    spectrogram_n_mels=128,
    streaming_video_max_frames=32,
    streaming_compressed_size=64,
    contrastive_loss_weight=0.1,
    modality_sync_loss_weight=0.05,
    modality_sync_tolerance=0.05,
    contrastive_temperature=0.07,
)
```

## Evaluation

`MultimodalEvaluator` (in `src/core/benchmark_evaluation.py`) measures per-
modality accuracy in isolation and jointly, then reports interference:

```python
from src.core.benchmark_evaluation import MultimodalEvaluator

evaluator = MultimodalEvaluator(model_apply_fn=model.apply)
report = evaluator.evaluate_with_interference(
    params, rng,
    modality_tasks={"vision": vision_task, "audio": audio_task},
    solo_samples={"vision": [...], "audio": [...]},
    joint_samples={"vision": [...], "audio": [...]},
)
print(report.interference, report.aggregate_interference)
```

## Tests

See `src/tests/test_multimodal_extensions.py` and
`src/tests/test_multimodal_evaluator.py`.
