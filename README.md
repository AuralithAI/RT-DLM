# RT-DLM: Real-Time Deep Learning Model

A JAX/Haiku-based neural architecture for training, combining transformer models, mixture of experts, quantum-inspired computing, and multi-paradigm hybrid learning.

> **Note**: This repository focuses on **model architecture and training**. Data collection, tokenization, and processing are handled by the standalone [Auralith-Data-Pipeline](https://github.com/AuralithAI/Auralith-Data-Pipeline) repository.

## Overview

RT-DLM provides a unified architecture for building and training advanced AI models. The system combines classical deep learning with symbolic reasoning, probabilistic inference, and quantum-ready modules.

## Core Components

### Model Architecture

#### Transformer-Memory-Sparse (TMS) Model
A three-tier memory system with sparse mixture of experts:
- **Long-term Memory (LTM)**: Persistent knowledge storage
- **Short-term Memory (STM)**: Context-sensitive processing
- **Meta-task Memory (MTM)**: Adaptive task handling
- **Sparse MoE**: Configurable expert routing with load balancing

#### Advanced Attention System
State-of-the-art attention mechanisms for efficiency and scalability:
- **RoPE (Rotary Position Embedding)**: Better long-context handling and extrapolation
- **GQA (Grouped-Query Attention)**: 2-4x KV cache reduction for faster inference
- **MQA (Multi-Query Attention)**: Maximum efficiency with shared KV heads
- **Sliding Window Attention**: O(n) complexity for very long sequences (>8k tokens)
- **Linear Attention**: Approximate attention with O(n) complexity
- **Spiking Attention**: Sparse activation for efficiency

```python
# Configure attention variant in TMSModel
model = TMSModel(
    d_model=512,
    num_heads=8,
    attention_type="gqa",      # "standard", "gqa", "mqa", "sliding", "linear"
    num_kv_heads=2,            # For GQA: 4x KV cache reduction
    position_encoding="rope",  # "rope", "learned", "none"
    ...
)
```

#### Graph-Based Neural Components
Relational reasoning with graph neural networks:
- **GraphNeuron**: Message-passing neural units
- **GraphAttentionUnit**: Graph attention mechanisms
- **MultiHopGraphReasoner**: Multi-step relational inference
- **GraphMoE**: Graph-structured mixture of experts

#### Hybrid Architecture
Multi-paradigm learning combining:
- **Traditional ML**: SVM, Random Forest, Naive Bayes branches
- **Deep Learning**: CNN, RNN, Transformer branches
- **Symbolic Reasoning**: Rule-based inference and logical operations
- **Probabilistic Models**: Uncertainty quantification and Bayesian methods
- **Ensemble Fusion**: Cross-paradigm interaction via outer-product attention

#### Quantum-Inspired Modules
Classical simulation of quantum computing primitives:
- QuantumSimulator with PHASE, CNOT, Hadamard gates
- VariationalQuantumCircuit for parameterized quantum ML
- **100+ qubit simulation** with tensor network approximations (MPS, TTN)
- Quantum attention mechanisms

#### Multimodal Processing
Cross-modal fusion capabilities:
- Audio emotion detection and hybrid audio module
- Video understanding with temporal modeling

### Training Pipeline

- **Epoch-based training loop** with configurable batch size
- **SafeTensors checkpointing** for model persistence
- **Gradient optimization** via Optax (AdamW, learning rate scheduling)
- **Ethics module** with feedback collection and reward modeling
- **Mixed-precision training** (bfloat16/float16) for faster training
- **Gradient checkpointing** for memory efficiency
- **Distributed training** support (data parallelism with pmap)
- **Model parallelism** for very large models (tensor/pipeline parallelism)
- **Speculative decoding** for faster inference

### Model Scale Presets

Pre-configured model sizes for different use cases:

| Preset | d_model | Heads | Layers | Parameters |
|--------|---------|-------|--------|------------|
| `tiny` | 256 | 4 | 4 | ~10M |
| `small` | 512 | 8 | 6 | ~50M |
| `base` | 768 | 12 | 12 | ~125M |
| `large` | 1024 | 16 | 24 | ~350M |
| `xlarge` | 2048 | 32 | 32 | ~1.3B |
| `xxlarge` | 4096 | 64 | 48 | ~7B |

```python
from config import AGIConfig

# Use a preset
config = AGIConfig.from_preset("large")
```

## Quick Start

### Installation
```bash
git clone https://github.com/AuralithAI/RT-DLM.git
cd RT-DLM
pip install -r requirements.txt
```

### Training

The model accepts pre-tokenized tensors (from [Auralith-Data-Pipeline](https://github.com/AuralithAI/Auralith-Data-Pipeline)).

```bash
# Train model
python train.py --data-dir /path/to/tokenized/shards

# Train with custom settings
python train.py --epochs 50 --batch-size 32 --lr 1e-4

# Resume training from a checkpoint
python train.py --resume checkpoints/rtdlm_agi_epoch_10.safetensors

# Resume with extended epochs
python train.py --resume checkpoints/rtdlm_agi_epoch_10.safetensors --epochs 100
```

### Training Modes

RT-DLM supports three training modes:

| Mode | Config | Use Case |
|------|--------|----------|
| **Standard** | `model_parallel=False` | Single GPU, full AGI model |
| **Data Parallel** | `distributed_training=True` | Multiple GPUs, same model replicated |
| **Model Parallel** | `model_parallel=True` | Model too large for single device |

```python
from config import AGIConfig

# Standard training (default) - Full AGI model
config = AGIConfig()

# Data parallel - Replicate across GPUs
config = AGIConfig(distributed_training=True, num_devices=4)

# Model parallel - Shard layers across GPUs
config = AGIConfig(model_parallel=True, num_devices=8)
```

**Note**: Model parallel mode uses a simplified transformer architecture optimized for sharding. Standard mode includes all AGI features (consciousness, quantum, multimodal, etc.).

### Running Tests
```bash
pytest tests/
```

## Implementation Status

### Completed
- TMS block with three-tier memory and sparse MoE
- **Advanced Attention System** (RoPE, GQA, MQA, Sliding Window, Linear Attention)
- **Graph-Based Neural Components** (GraphNeuron, MultiHopGraphReasoner, GraphMoE)
- **Model Scale Presets** (tiny to xxlarge configurations)
- Hybrid architecture with four ML paradigms
- Ensemble fusion with cross-paradigm interaction
- Quantum simulator (100+ qubits with tensor network approximations)
- Variational quantum circuit
- Multimodal fusion
- Production-ready token sampling (Top-K, Top-P, temperature, repetition penalty)
- **Speculative decoding** for faster inference
- SafeTensors checkpoint management
- Training pipeline with epoch-based loop and checkpoint resumption
- Ethics module with feedback collection and reward modeling
- Mixed-precision training (bfloat16/float16)
- Gradient checkpointing for memory efficiency
- Distributed training support (data parallelism)
- Comprehensive test suite (375+ tests)
- Model parallelism (tensor parallelism, pipeline parallelism)
- Tensor network approximations for quantum simulation (MPS, TTN)

## Requirements

- Python 3.10+
- JAX 0.4.35+
- Haiku 0.0.13+
- Optax (optimizer)
- SafeTensors (checkpoints)
- NumPy

## Related Repositories

| Repository | Description |
|------------|-------------|
| [Auralith-Data-Pipeline](https://github.com/AuralithAI/Auralith-Data-Pipeline) | Data collection, tokenization, processing, and sharding |

## Documentation

- [Architecture Overview](docs/ARCHITECTURE.md) - System design and data flow
- [Sampling Strategies](docs/SAMPLING.md) - Token generation configuration
- [Quick Start Guide](docs/QUICKSTART.md) - Getting started

## License

See [LICENSE](LICENSE) file for details.
