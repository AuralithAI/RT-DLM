# RT-DLM Sampling Guide

> **⚠️ Development Utility**: This module is for **testing and development only**.
> For production inference, use optimized serving frameworks (vLLM, TGI, etc.)
> that load RT-DLM checkpoints. Production inference requires KV caching,
> continuous batching, and optimized kernels not present in this module.

## Overview

The `core/sampling.py` module provides token sampling strategies for **testing** model outputs during development. It helps validate that trained models can generate coherent text.

## Key Components

### SamplingConfig

Dataclass containing all sampling parameters:

```python
from core.sampling import SamplingConfig

config = SamplingConfig(
    temperature=0.7,      # Controls randomness (0.0 = deterministic, 1.0 = standard, >1.0 = more random)
    top_k=50,             # Keep only top-k tokens
    top_p=0.9,            # Nucleus sampling threshold
    min_p=0.0,            # Minimum probability threshold
    repetition_penalty=1.0,  # Penalty for repeated tokens (1.0 = no penalty)
    do_sample=True,       # Whether to sample or use greedy decoding
    seed=42,              # Random seed for reproducibility
)
```

### TokenSampler

Main class for applying sampling strategies:

```python
from core.sampling import TokenSampler, SamplingConfig

sampler = TokenSampler()
config = SamplingConfig(temperature=0.7, top_k=50, top_p=0.9)

# Sample from logits
result = sampler.sample(logits, config, token_history=[])

# Access results
selected_token = result.token_id
probability = result.probability
all_probs = result.all_probabilities
```

## Sampling Strategies

### Temperature Scaling

Controls the sharpness of the probability distribution:
- `temperature < 1.0`: More deterministic, focuses on high-probability tokens
- `temperature = 1.0`: Standard softmax probabilities
- `temperature > 1.0`: More random, flattens the distribution

```python
# Low temperature for focused, deterministic output
config = SamplingConfig(temperature=0.3)

# High temperature for creative, diverse output
config = SamplingConfig(temperature=1.2)
```

### Top-K Filtering

Keeps only the `k` most probable tokens:

```python
# Only consider top 50 tokens
config = SamplingConfig(top_k=50)

# More focused (fewer options)
config = SamplingConfig(top_k=10)
```

### Top-P (Nucleus) Sampling

Dynamically selects tokens until cumulative probability reaches `p`:

```python
# Include tokens until 90% probability mass
config = SamplingConfig(top_p=0.9)

# More focused
config = SamplingConfig(top_p=0.7)
```

### Min-P Filtering

Removes tokens below a relative probability threshold:

```python
# Remove tokens with probability < 10% of the top token
config = SamplingConfig(min_p=0.1)
```

### Repetition Penalty

Penalizes tokens that have already been generated:

```python
# Light penalty
config = SamplingConfig(repetition_penalty=1.1)

# Strong penalty
config = SamplingConfig(repetition_penalty=1.5)

# Use with token history
result = sampler.sample(logits, config, token_history=[prev_token_1, prev_token_2])
```

## Preset Configurations

For convenience, preset configurations are available:

```python
from core.sampling import (
    create_sampling_config_creative,
    create_sampling_config_precise,
    create_sampling_config_balanced,
    create_sampling_config_deterministic,
)

# Creative writing, storytelling
creative_config = create_sampling_config_creative()
# temperature=1.2, top_p=0.95, top_k=100, repetition_penalty=1.1

# Factual, precise responses
precise_config = create_sampling_config_precise()
# temperature=0.3, top_p=0.9, top_k=20, repetition_penalty=1.05

# General purpose
balanced_config = create_sampling_config_balanced()
# temperature=0.7, top_p=0.9, top_k=50, repetition_penalty=1.0

# Fully deterministic (greedy)
deterministic_config = create_sampling_config_deterministic()
# temperature=0.0, do_sample=False
```

## Using Sampling in Training

The sampling module can be used for evaluation during training:

```python
from core.sampling import TokenSampler, SamplingConfig

sampler = TokenSampler()
config = SamplingConfig(temperature=0.7, top_k=50)

# During model evaluation
result = sampler.sample(model_logits, config, token_history=[])
```

## Best Practices

1. **For factual Q&A**: Use `precise` preset or low temperature (0.2-0.5)
2. **For creative writing**: Use `creative` preset or high temperature (0.8-1.2)
3. **For code generation**: Use `balanced` preset with repetition penalty
4. **For deterministic output**: Use `deterministic` preset
5. **Combine strategies**: Top-K + Top-P often works well together
6. **Token history**: Keep recent tokens for effective repetition penalty

## Debugging

Enable probability logging for debugging:

```python
result = sampler.sample(logits, config, token_history=[])
print(f"Selected token: {result.token_id}")
print(f"Probability: {result.probability:.4f}")
print(f"Top 10 probabilities: {result.all_probabilities[:10]}")
```

## Speculative Decoding

RT-DLM supports speculative decoding for faster inference. This technique uses a smaller "draft" model to propose tokens, then verifies them with the main model.

### How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                    Speculative Decoding                          │
│                                                                  │
│   1. Draft Model generates K candidate tokens quickly            │
│   2. Main Model verifies all K tokens in one forward pass        │
│   3. Accept matching tokens, reject divergent ones               │
│   4. Speedup: 2-3x faster than autoregressive decoding           │
│                                                                  │
│   ┌─────────┐        ┌─────────────┐        ┌─────────────────┐  │
│   │  Draft  │──K────▶│   Verify    │──────▶│ Accept/Reject   │  │
│   │  Model  │ tokens │ (Main Model)│        │ + Resample      │  │
│   └─────────┘        └─────────────┘        └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Usage

```python
from core.sampling import SpeculativeDecoder, SelfSpeculativeDecoder

# With separate draft model (e.g., smaller version)
speculative = SpeculativeDecoder(
    target_model=main_model,
    draft_model=draft_model,
    num_speculative_tokens=4
)

# Generate tokens
tokens = speculative.generate(input_ids, max_new_tokens=100)

# Self-speculative (uses early exit from same model)
self_spec = SelfSpeculativeDecoder(
    model=model,
    early_exit_layer=6,  # Use layer 6 as draft
    num_speculative_tokens=4
)
```

### When to Use

| Method | Best For | Speedup |
|--------|----------|---------|
| **Standard Autoregressive** | Short outputs, streaming | 1x |
| **Speculative Decoding** | Long outputs, batch processing | 2-3x |
| **Self-Speculative** | When no draft model available | 1.5-2x |

