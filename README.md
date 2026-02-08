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

## Overview

RT-DLM provides a unified architecture for building and training advanced AI models. The system combines classical deep learning with symbolic reasoning, probabilistic inference, and quantum-ready modules.

## Core Components

### Compute Controller

RT-DLM features a **learned Compute Controller** that dynamically allocates compute across modules under a budget constraint:

- **Adaptive Module Selection**: Controller learns which modules to invoke based on input complexity
- **Budget-Aware Execution**: Allocates compute budget across modules, halts when exhausted
- **Confidence-Based Halting**: Stops early when confidence threshold is reached
- **K-Step Execution Loop**: Iteratively refines output up to max steps
- **Module Contracts**: Standardized interface for cost, capabilities, and dependencies
- **Configuration Presets**: `FAST_CONFIG`, `BALANCED_CONFIG`, `THOROUGH_CONFIG` for different use cases
- **Multi-Objective Training**: Efficiency, calibration, utilization, and ponder losses
- **RL Reward Shaping**: Dense rewards for controller optimization
- **Full AGI Integration**: Controller-driven forward pass with execution tracing

Enable via config: `AGIConfig(use_compute_controller=True, controller_strategy="balanced")`. See [Architecture](docs/ARCHITECTURE.md#enabling-the-controller) for full options.

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

Clone the repository and install with `pip install -r requirements.txt`.

### Training

The model accepts pre-tokenized tensors (from [Auralith-Data-Pipeline](https://github.com/AuralithAI/Auralith-Data-Pipeline)).

Run `python src/train.py --data-dir /path/to/tokenized/shards` to train. Use `--epochs`, `--batch-size`, `--lr` for hyperparameters and `--resume` to continue from a checkpoint.

### Training Modes

RT-DLM supports multiple training modes with automatic optimization:

| Mode | Config | Use Case | Devices |
|------|--------|----------|---------|
| **Standard** | Default | Development, single GPU | 1 |
| **Data Parallel** | `distributed_training=True` | Faster training, batch scaling | 2-8 |
| **Tensor Parallel** | `tensor_parallel=True` | Models >7B params | 4+ |
| **Combined** | Both flags | Production scale | 8+ |

Set `quantum_layers=0` to disable quantum simulation for faster training.

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

Automated testing and deployment via GitHub Actions with workflows for testing (any branch), linting (any branch), and Docker builds (main only).

### Docker

Build training images with `docker build -f Dockerfile.train`. Supports GPU and CPU-only targets. Use `docker-compose` for local development with optional monitoring stack (Prometheus + Grafana).

### Kubernetes (Helm)

Deploy training jobs to Kubernetes using the Helm chart in `helm/rtdlm/`. Install with `helm install rtdlm ./helm/rtdlm -n rtdlm --create-namespace` and customize with `--set` flags for model preset, GPU count, storage, and distributed training options.

### Monitoring

Prometheus metrics exposed on port 8000 via `PrometheusTrainingCallback`. Access at `http://localhost:8000/metrics`.

### Model Export

Export trained models using `scripts/quantize_model.py` for INT8 quantization and `scripts/export_to_onnx.py` for ONNX export.

### Make Commands

Common tasks via Makefile: `make install-dev`, `make test`, `make test-cov`, `make lint`, `make format`, `make docker-build`, `make train-tiny`, `make validate-model`.

## License

See [LICENSE](LICENSE) file for details.
