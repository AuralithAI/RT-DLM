# RT-DLM Deployment Architecture

This document describes the deployment architecture and infrastructure for RT-DLM (Real-Time Deep Learning Model).

## Project Structure

```
RT-DLM/
├── src/                     # Core model package
│   ├── __init__.py          # Package exports
│   ├── rtdlm.py             # Main model definitions
│   ├── train.py             # Training script
│   ├── core/                # Core components
│   │   ├── model/           # Model architecture
│   │   ├── training/        # Training utilities
│   │   ├── quantum/         # Quantum modules
│   │   ├── agi/             # AGI system
│   │   ├── ethics/          # Ethics & fairness
│   │   ├── rlm/             # Recursive language model
│   │   ├── export/          # ONNX export
│   │   └── quantization/    # Model quantization
│   ├── config/              # Configuration classes
│   ├── modules/             # Feature modules
│   │   ├── multimodal/      # Multi-modal processing
│   │   ├── retrieval/       # RAG integration
│   │   ├── hybrid_architecture/
│   │   └── capabilities/    # Advanced algorithms
│   └── tests/               # Test suite
├── helm/                    # Kubernetes Helm chart
│   └── rtdlm/
│       ├── Chart.yaml
│       ├── values.yaml
│       └── templates/
├── monitoring/              # Prometheus/Grafana
├── scripts/                 # CLI utilities
├── docs/                    # Documentation
├── .github/workflows/       # CI/CD pipelines
├── Dockerfile.train         # Training container
├── docker-compose.yml       # Local development
├── Makefile                 # Development tasks
└── pyproject.toml           # Package configuration
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RT-DLM Deployment Architecture                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                         GitHub Repository                           │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐                         │    │
│  │  │ test.yml │  │ lint.yml │  │ docker-  │                         │    │
│  │  │ (any br) │  │ (any br) │  │ build.yml│                         │    │
│  │  └────┬─────┘  └────┬─────┘  │ (main)   │                         │    │
│  │       │             │        └────┬─────┘                         │    │
│  └───────┼─────────────┼─────────────┼──────────────────────────────┘    │
│          │             │             │                                     │
│          ▼             ▼             ▼                                     │
│  ┌───────────────────────────────────────────────────────────────────┐    │
│  │                        CI/CD Pipeline                              │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐   │    │
│  │  │  Run Tests  │  │ Code Quality│  │ Build & Push Docker     │   │    │
│  │  │  + Coverage │  │  Checks     │  │ Images to GHCR          │   │    │
│  │  └─────────────┘  └─────────────┘  └────────────┬────────────┘   │    │
│  └────────────────────────────────────────────────┼─────────────────┘    │
│                                                    │                       │
│                                                    ▼                       │
│  ┌────────────────────────────────────────────────────────────────────┐   │
│  │                    Container Registry (GHCR)                        │   │
│  │   ghcr.io/auralithai/rt-dlm:train-latest                           │   │
│  │   ghcr.io/auralithai/rt-dlm:train-<sha>                            │   │
│  └─────────────────────────────┬──────────────────────────────────────┘   │
│                                │                                           │
│                                ▼                                           │
│  ┌────────────────────────────────────────────────────────────────────┐   │
│  │                      Kubernetes Cluster                             │   │
│  │  ┌──────────────────────────────────────────────────────────────┐  │   │
│  │  │                    Helm Chart (rtdlm)                         │  │   │
│  │  │  ┌─────────────────────────────────────────────────────────┐ │  │   │
│  │  │  │                  rtdlm Namespace                         │ │  │   │
│  │  │  │                                                         │ │  │   │
│  │  │  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │ │  │   │
│  │  │  │  │  ConfigMap   │  │   Secrets    │  │   Storage    │ │ │  │   │
│  │  │  │  │  (config)    │  │  (creds)     │  │   (PVCs)     │ │ │  │   │
│  │  │  │  └──────────────┘  └──────────────┘  └──────────────┘ │ │  │   │
│  │  │  │                                                         │ │  │   │
│  │  │  │  ┌────────────────────────────────────────────────────┐│ │  │   │
│  │  │  │  │              Training Deployment                    ││ │  │   │
│  │  │  │  │  ┌──────────────────────────────────────────────┐ ││ │  │   │
│  │  │  │  │  │  Pod: rtdlm-training                          │ ││ │  │   │
│  │  │  │  │  │  - Container: trainer (python train.py)       │ ││ │  │   │
│  │  │  │  │  │  - GPU: nvidia-a100 x4                        │ ││ │  │   │
│  │  │  │  │  │  - Memory: 64Gi                               │ ││ │  │   │
│  │  │  │  │  └──────────────────────────────────────────────┘ ││ │  │   │
│  │  │  │  └────────────────────────────────────────────────────┘│ │  │   │
│  │  │  │                                                         │ │  │   │
│  │  │  │  ┌────────────────────────────────────────────────────┐│ │  │   │
│  │  │  │  │           Distributed Training Job                  ││ │  │   │
│  │  │  │  │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐              ││ │  │   │
│  │  │  │  │  │Node 0│ │Node 1│ │Node 2│ │Node 3│              ││ │  │   │
│  │  │  │  │  │Master│ │Worker│ │Worker│ │Worker│              ││ │  │   │
│  │  │  │  │  └──────┘ └──────┘ └──────┘ └──────┘              ││ │  │   │
│  │  │  │  └────────────────────────────────────────────────────┘│ │  │   │
│  │  │  │                                                         │ │  │   │
│  │  │  │  ┌──────────────┐  ┌──────────────────────────────────┐│ │  │   │
│  │  │  │  │   Service    │  │        Metrics Server            ││ │  │   │
│  │  │  │  │  (ClusterIP) │  │  (Prometheus Exporter :8000)     ││ │  │   │
│  │  │  │  └──────────────┘  └──────────────────────────────────┘│ │  │   │
│  │  │  └─────────────────────────────────────────────────────────┘ │  │   │
│  │  └──────────────────────────────────────────────────────────────┘  │   │
│  └────────────────────────────────────────────────────────────────────┘   │
│                                                                           │
│  ┌────────────────────────────────────────────────────────────────────┐   │
│  │                     Monitoring Stack                                │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │   │
│  │  │  Prometheus  │─▶│   Grafana    │  │    AlertManager          │  │   │
│  │  │  (scrape)    │  │  (visualize) │  │    (notifications)       │  │   │
│  │  └──────────────┘  └──────────────┘  └──────────────────────────┘  │   │
│  └────────────────────────────────────────────────────────────────────┘   │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
```

## CI/CD Pipeline

### Workflow Triggers

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `test.yml` | Any branch push/PR | Run pytest with coverage |
| `lint.yml` | Any branch push/PR | Code quality checks |
| `docker-build.yml` | Main branch only | Build & push production images |

### Test Workflow (`test.yml`)

```yaml
# Runs on any branch
on:
  push:
    branches: ['**']
  pull_request:
    branches: ['**']

# Matrix testing
strategy:
  matrix:
    python-version: ["3.10", "3.11", "3.12"]

# Jobs:
# 1. test - Run pytest with coverage
# 2. smoke-test - Verify model initialization
```

### Docker Build Workflow (`docker-build.yml`)

```yaml
# Runs only on main branch
on:
  push:
    branches: [main]
    tags: ['v*']

# Builds:
# 1. GPU image (default)
# 2. CPU image (for testing)
# 3. Pushes to ghcr.io
```

## Docker Architecture

### Multi-Stage Build

```dockerfile
# Stage 1: Base - System dependencies
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 AS base

# Stage 2: GPU Dependencies
FROM base AS gpu-deps
# JAX with CUDA 12

# Stage 3: CPU Build
FROM base AS cpu
# JAX CPU only

# Stage 4: GPU Build (Default)
FROM gpu-deps AS gpu
# Final training image
```

### Running Locally

```bash
# GPU training
docker-compose up training

# CPU training (for testing)
docker-compose up training-cpu

# With monitoring stack
docker-compose --profile monitoring up
```

## Kubernetes Deployment

### Helm Chart Structure

```
helm/rtdlm/
├── Chart.yaml              # Chart metadata
├── values.yaml             # Configuration
└── templates/
    ├── _helpers.tpl        # Template helpers
    ├── configmap.yaml      # Training config
    ├── secrets.yaml        # Credentials
    ├── serviceaccount.yaml # RBAC
    ├── storage.yaml        # PVCs
    ├── training_deployment.yaml    # Single-node training
    ├── training_service.yaml       # Metrics service
    ├── training_ingress.yaml       # Optional ingress
    ├── distributed_training.yaml   # Multi-node training
    ├── metrics_deployment.yaml     # Prometheus exporter
    └── networkpolicy.yaml          # Network security
```

### Installation

```bash
# Basic installation
helm install rtdlm ./helm/rtdlm -n rtdlm --create-namespace

# With custom values
helm install rtdlm ./helm/rtdlm -n rtdlm \
  --set training.model.preset=large \
  --set training.resources.limits."nvidia\.com/gpu"=4

# Enable distributed training
helm install rtdlm ./helm/rtdlm -n rtdlm \
  --set distributed_training.enabled=true \
  --set distributed_training.num_nodes=4
```

### Key Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `training.enabled` | Enable training deployment | `true` |
| `training.model.preset` | Model size (tiny/small/medium/large) | `medium` |
| `training.gpu.count` | Number of GPUs per node | `4` |
| `training.gpu.type` | GPU type | `nvidia-a100` |
| `distributed_training.enabled` | Enable multi-node training | `false` |
| `distributed_training.workers` | Number of nodes | `4` |

## Monitoring

### Prometheus Metrics

The training process exposes metrics on port 8000:

```python
from monitoring.prometheus_exporter import PrometheusTrainingCallback

callback = PrometheusTrainingCallback(port=8000)
callback.on_batch_end(loss=loss, batch_time=elapsed)
```

Available metrics:
- `rtdlm_training_loss` - Current training loss
- `rtdlm_training_perplexity` - Current perplexity
- `rtdlm_training_learning_rate` - Learning rate
- `rtdlm_batch_time_seconds` - Batch processing time
- `rtdlm_tokens_per_second` - Training throughput
- `rtdlm_gpu_memory_used_bytes` - GPU memory usage
- `rtdlm_gradient_norm` - Gradient L2 norm

### Alert Rules

Pre-configured alerts:
- High training loss
- NaN gradients detected
- GPU memory exhaustion
- Low training throughput
- Exploding/vanishing gradients

## Model Export

### Quantization

```bash
# INT8 quantization
python scripts/quantize_model.py \
  --checkpoint checkpoints/model.safetensors \
  --output checkpoints/model_int8.safetensors \
  --precision int8
```

### ONNX Export

```bash
# Export to ONNX
python scripts/export_to_onnx.py \
  --checkpoint checkpoints/model.safetensors \
  --output models/model.onnx \
  --opset 15
```

## Development Workflow

### Quick Commands

```bash
# Install dependencies
make install-dev

# Run tests
make test

# Run tests with coverage
make test-cov

# Format code
make format

# Lint code
make lint

# Build Docker image
make docker-build

# Deploy to Kubernetes
make helm-install

# Run tiny model training
make train-tiny
```

### Local Development

```bash
# 1. Set up environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -e ".[dev]"

# 2. Run tests
pytest src/tests/ -v

# 3. Start training locally
python src/train.py --preset tiny --epochs 10
```

## Separation of Concerns

### Training Repository (RT-DLM)

This repository focuses on:
- Model architecture definition
- Training pipeline
- Checkpointing
- Evaluation metrics
- Distributed training

### Inference Repository (Separate)

Production inference should use a separate repository with:
- vLLM or TGI for serving
- ONNX Runtime for cross-platform
- TensorRT for NVIDIA optimization
- API endpoints for inference

### Data Pipeline (Auralith-Data-Pipeline)

Data processing is handled separately:
- Data collection
- Tokenization
- Preprocessing
- Sharding for distributed training

## Security Considerations

1. **Non-root containers**: All containers run as non-root user `rtdlm`
2. **Secret management**: Kubernetes secrets for credentials
3. **Network policies**: Restrict pod-to-pod communication
4. **Service accounts**: Minimal RBAC permissions
5. **Image scanning**: CI includes security scanning with Bandit

## Troubleshooting

### Common Issues

1. **GPU not detected**
   ```bash
   kubectl logs -l app=rtdlm-training -n rtdlm
   # Check nvidia-smi in init container
   ```

2. **OOM errors**
   - Reduce batch size in values.yaml
   - Enable gradient checkpointing
   - Use smaller model preset

3. **Slow training**
   - Check GPU utilization: `nvidia-smi dmon`
   - Enable XLA compilation caching
   - Check data loading bottlenecks

4. **Checkpoint issues**
   - Verify PVC is mounted correctly
   - Check storage class supports ReadWriteOnce
   - Ensure sufficient storage space

## References

- [JAX Documentation](https://jax.readthedocs.io/)
- [Haiku Documentation](https://dm-haiku.readthedocs.io/)
- [Kubernetes GPU Scheduling](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/)
- [Helm Charts](https://helm.sh/docs/)
