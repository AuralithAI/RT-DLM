# RT-DLM: Real-Time Deep Learning Architecture and Training System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-0.4.35+-orange.svg)](https://github.com/google/jax)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-740%2B%20passing-brightgreen.svg)](src/tests/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![CI](https://github.com/AuralithAI/RT-DLM/actions/workflows/test.yml/badge.svg)](https://github.com/AuralithAI/RT-DLM/actions/workflows/test.yml)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](Dockerfile.train)

A JAX/Haiku-based neural architecture for training, combining transformer models, mixture of experts, quantum-inspired computing, and multi-paradigm hybrid learning.

> **Note**: This repository focuses on **model architecture and training**. Data collection, tokenization, and processing are handled by the standalone [Auralith-Data-Pipeline](https://github.com/AuralithAI/Auralith-Data-Pipeline) repository.

## Project Structure

```
RT-DLM/
├── src/                 # Core model package
│   ├── rtdlm.py         # Model definitions
│   ├── train.py         # Training script
│   ├── core/            # Core components
│   ├── config/          # Configuration
│   ├── modules/         # Feature modules
│   └── tests/           # Test suite
├── helm/                # Kubernetes Helm chart
├── monitoring/          # Prometheus/Grafana
├── scripts/             # CLI utilities
├── docs/                # Documentation
└── .github/workflows/   # CI/CD pipelines
```

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

Configure via `TMSModel(attention_type="gqa", num_kv_heads=2, position_encoding="rope", ...)`.

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
- **Gradient clipping** for training stability
- **Ethics module** with feedback collection and reward modeling
- **Mixed-precision training** (bfloat16/float16) for faster training
- **Gradient checkpointing** for memory efficiency

### Distributed Training

RT-DLM provides production-ready distributed training via `ScalableMesh`:

| Strategy | Use Case | API |
|----------|----------|-----|
| **Data Parallel** | Multiple GPUs, same model | `jax.pmap` with gradient sync |
| **Tensor Parallel** | Large models (>7B params) | Sharded parameters via `NamedSharding` |
| **Combined** | Production scale | Data + Tensor parallelism |

Key utilities in `core.scalable_training`:
- `recommend_parallelism()` - Auto-detect optimal strategy for your hardware
- `estimate_model_memory()` - Estimate GPU memory requirements
- `profile_collective_communication()` - Measure all-reduce latency/bandwidth

### Quantum Cost Analysis

The quantum module provides resource estimation via `estimate_quantum_overhead()`:

| Qubits | Mode | Memory |
|--------|------|--------|
| ≤16 | Full State | ~1 MB |
| 17-24 | Full State | 256 MB - 16 GB |
| 30+ | Tensor Network | O(n×χ²) |

**Note**: Quantum simulation is classical emulation. Set `quantum_layers=0` in config to disable.

### Evaluation Metrics

Production-grade training with comprehensive evaluation via `TrainingEvaluator`:

- **Perplexity**: Core LM quality metric (lower = better)
- **Token Accuracy**: Top-1 and Top-5 prediction accuracy
- **Gradient Norms**: Health monitoring (NaN/Inf/exploding/vanishing detection)
- **Throughput**: Tokens per second
- **Loss Curves**: Structured JSON logging for analysis

### Model Scale Presets

Pre-configured model sizes via `AGIConfig.from_preset()`:

| Preset | d_model | Heads | Layers | Parameters |
|--------|---------|-------|--------|------------|
| `tiny` | 256 | 4 | 4 | ~10M |
| `small` | 512 | 8 | 6 | ~50M |
| `base` | 768 | 12 | 12 | ~125M |
| `large` | 1024 | 16 | 24 | ~350M |
| `xlarge` | 2048 | 32 | 32 | ~1.3B |
| `xxlarge` | 4096 | 64 | 48 | ~7B |

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
python src/train.py --data-dir /path/to/tokenized/shards

# Train with custom settings
python src/train.py --epochs 50 --batch-size 32 --lr 1e-4

# Resume training from a checkpoint
python src/train.py --resume checkpoints/rtdlm_agi_epoch_10.safetensors

# Resume with extended epochs
python src/train.py --resume checkpoints/rtdlm_agi_epoch_10.safetensors --epochs 100
```

### Training Modes

RT-DLM supports multiple training modes with automatic optimization:

| Mode | Config | Use Case | Devices |
|------|--------|----------|---------|
| **Standard** | Default | Development, single GPU | 1 |
| **Data Parallel** | `distributed_training=True` | Faster training, batch scaling | 2-8 |
| **Tensor Parallel** | `tensor_parallel=True` | Models >7B params | 4+ |
| **Combined** | Both flags | Production scale | 8+ |

Set `quantum_layers=0` to disable quantum simulation for faster training.

## Implementation Status

### Completed
- TMS block with three-tier memory and sparse MoE
- **Advanced Attention System** (RoPE, GQA, MQA, Sliding Window, Linear Attention)
- **Graph-Based Neural Components** (GraphNeuron, MultiHopGraphReasoner, GraphMoE)
- **Model Scale Presets** (tiny to xxlarge configurations)
- Hybrid architecture with four ML paradigms
- Ensemble fusion with cross-paradigm interaction
- Quantum simulator (100+ qubits with tensor network approximations)
- Variational quantum circuit with cost estimation API
- Multimodal fusion
- SafeTensors checkpoint management
- Training pipeline with epoch-based loop and checkpoint resumption
- Ethics module with feedback collection and reward modeling
- Mixed-precision training (bfloat16/float16)
- Gradient checkpointing for memory efficiency
- Gradient clipping for training stability
- **Distributed Training** (`ScalableMesh` with data + tensor parallelism)
- **Memory & Parallelism Estimation** (`estimate_model_memory`, `recommend_parallelism`)
- **Communication Profiling** (`profile_collective_communication`)
- Tensor network approximations for quantum simulation (MPS, TTN)
- **Evaluation Metrics** (perplexity, gradient norms, structured logging)
- **Gradient Health Monitoring** (NaN/Inf detection, exploding/vanishing detection)
- **Validation Runner** for periodic evaluation
- Comprehensive test suite (600+ tests)

### Architecture Notes
- **Training Focus**: This repository focuses on model architecture and training completeness
- **Inference**: Token sampling/generation utilities are marked as `@dev_utility` for testing purposes only
- **Production Inference**: For production deployment, use optimized serving frameworks (vLLM, TGI) that load RT-DLM checkpoints

## Related Repositories

| Repository | Description |
|------------|-------------|
| [Auralith-Data-Pipeline](https://github.com/AuralithAI/Auralith-Data-Pipeline) | Data collection, tokenization, processing, and sharding |

## Documentation

- [Architecture Overview](docs/ARCHITECTURE.md) - System design and data flow
- [Deployment Guide](docs/DEPLOYMENT.md) - Deployment architecture and how-to
- [Sampling Strategies](docs/SAMPLING.md) - Token generation configuration
- [Quick Start Guide](docs/QUICKSTART.md) - Getting started
- [Changelog](CHANGELOG.md) - Version history and changes

## Deployment & Infrastructure

### CI/CD Pipeline

Automated testing and deployment via GitHub Actions:

```bash
# Workflow triggers:
.github/workflows/test.yml     # Runs on any branch (pytest with coverage)
.github/workflows/lint.yml     # Runs on any branch (code quality checks)
.github/workflows/docker-build.yml  # Runs on main only (production builds)
```

### Docker

Build and run training environments:

```bash
# Build training image (GPU)
docker build -t rtdlm:train -f Dockerfile.train .

# Build CPU-only image
docker build -t rtdlm:train-cpu -f Dockerfile.train --target cpu .

# Run with Docker Compose
docker-compose up training

# Run with monitoring stack (Prometheus + Grafana)
docker-compose --profile monitoring up
```

### Kubernetes (Helm)

Deploy training jobs to Kubernetes using Helm:

```bash
# Install the Helm chart
helm install rtdlm ./helm/rtdlm -n rtdlm --create-namespace

# Install with custom values
helm install rtdlm ./helm/rtdlm -n rtdlm \
  --set training.config.preset=large \
  --set training.resources.limits."nvidia\.com/gpu"=4 \
  --set storage.checkpoints.size=100Gi

# Enable distributed training
helm install rtdlm ./helm/rtdlm -n rtdlm \
  --set distributed_training.enabled=true \
  --set distributed_training.num_nodes=4

# Upgrade existing deployment
helm upgrade rtdlm ./helm/rtdlm -n rtdlm --reuse-values

# Check status
kubectl get pods -n rtdlm
kubectl logs -f -l app.kubernetes.io/name=rtdlm -n rtdlm

# Uninstall
helm uninstall rtdlm -n rtdlm
```

Helm chart structure:
```
helm/rtdlm/
├── Chart.yaml          # Chart metadata
├── values.yaml         # Default configuration
└── templates/
    ├── _helpers.tpl    # Template helpers
    ├── configmap.yaml  # Training configuration
    ├── secrets.yaml    # Credentials (WandB, AWS, HF)
    ├── storage.yaml    # PVCs for checkpoints/logs
    ├── training_deployment.yaml  # Main training
    ├── training_service.yaml     # Metrics service
    ├── training_ingress.yaml     # Optional ingress
    ├── distributed_training.yaml # Multi-node jobs
    └── networkpolicy.yaml        # Network policies
```

### Monitoring

Prometheus metrics exposed on port 8000:

```python
from monitoring.prometheus_exporter import PrometheusTrainingCallback

# Add to training loop
callback = PrometheusTrainingCallback(port=8000)
callback.on_batch_end(loss=loss, batch_time=0.5)
```

Access metrics at `http://localhost:8000/metrics`

### Model Export

Export trained models for deployment:

```bash
# Quantize to INT8
python scripts/quantize_model.py \
    --checkpoint checkpoints/model.safetensors \
    --output checkpoints/model_int8.safetensors \
    --precision int8

# Export to ONNX
python scripts/export_to_onnx.py \
    --checkpoint checkpoints/model.safetensors \
    --output models/model.onnx \
    --opset 15
```

### Make Commands

Common tasks via Makefile:

```bash
make help           # Show all commands
make install-dev    # Install dev dependencies
make test           # Run tests
make test-cov       # Run tests with coverage
make lint           # Check code quality
make format         # Auto-format code
make docker-build   # Build Docker image
make train-tiny     # Run tiny model training
make validate-model # Validate model init
```

## License

See [LICENSE](LICENSE) file for details.
