# RT-DLM Sampling Guide

> **⚠️ Development Utility**: This module is for **testing and development only**.
> For production inference, use optimized serving frameworks (vLLM, TGI, etc.)
> that load RT-DLM checkpoints. Production inference requires KV caching,
> continuous batching, and optimized kernels not present in this module.

## Overview

The `src/core/sampling.py` module provides token sampling strategies for **testing** model outputs during development. It helps validate that trained models can generate coherent text.

## Key Components

### SamplingConfig

Dataclass containing all sampling parameters:

- `temperature` - Controls randomness (0.0 = deterministic, 1.0 = standard, >1.0 = more random)
- `top_k` - Keep only top-k tokens
- `top_p` - Nucleus sampling threshold
- `min_p` - Minimum probability threshold
- `repetition_penalty` - Penalty for repeated tokens (1.0 = no penalty)
- `do_sample` - Whether to sample or use greedy decoding
- `seed` - Random seed for reproducibility

### TokenSampler

Main class for applying sampling strategies. Import from `src.core.sampling` and use with `SamplingConfig`.

## Sampling Strategies

### Temperature Scaling

Controls the sharpness of the probability distribution:
- `temperature < 1.0`: More deterministic, focuses on high-probability tokens
- `temperature = 1.0`: Standard softmax probabilities
- `temperature > 1.0`: More random, flattens the distribution

### Top-K Filtering

Keeps only the `k` most probable tokens. Lower values = more focused output.

### Top-P (Nucleus) Sampling

Dynamically selects tokens until cumulative probability reaches `p`. Adaptive vocab size based on probability mass.

### Min-P Filtering

Removes tokens below a relative probability threshold compared to the top token.

### Repetition Penalty

Reduces probability of recently generated tokens to avoid repetition.

## Sampling Presets

| Preset | Temperature | Top-P | Top-K | Use Case |
|--------|-------------|-------|-------|----------|
| Creative | 1.2 | 0.95 | 100 | Creative writing, brainstorming |
| Balanced | 0.7 | 0.9 | 50 | General purpose |
| Precise | 0.3 | 0.9 | 20 | Factual, focused responses |
| Deterministic | 0.0 | - | - | Greedy decoding, reproducible output |

## Best Practices

1. **Start with Balanced**: Use the balanced preset as a baseline
2. **Adjust Temperature First**: Primary control for output diversity
3. **Combine Strategies**: Use top-k AND top-p together for quality
4. **Enable Repetition Penalty**: Prevents degenerate loops
5. **Set Seeds**: Use fixed seeds for reproducible testing

## Speculative Decoding

For faster generation during development testing:

- **SpeculativeDecoder**: Uses draft model for token prediction
- **SelfSpeculativeDecoder**: Uses early exit for self-speculation

Both use draft-then-verify approach for 2-4x speedup while maintaining output quality.

## Production Notes

This sampling module is **not** suitable for production serving. For production:

- Use **vLLM** or **TGI** for optimized inference
- Export to **ONNX** for cross-platform deployment
- Use **TensorRT** for NVIDIA-optimized serving

These frameworks provide KV caching, continuous batching, and GPU-optimized kernels essential for production throughput.

