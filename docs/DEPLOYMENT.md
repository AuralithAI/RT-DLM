# RT-DLM Deployment Architecture

This document describes the deployment architecture and infrastructure for RT-DLM (Real-Time Deep Learning Model).

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

### Test Workflow

Runs on any branch push/PR with matrix testing across Python 3.10, 3.11, and 3.12. Includes pytest with coverage and smoke test for model initialization.

### Docker Build Workflow

Runs on main branch and version tags. Builds GPU and CPU images and pushes to ghcr.io.

## Docker Architecture

### Multi-Stage Build

Uses NVIDIA CUDA 12.1 base image with four stages: Base (system dependencies), GPU Dependencies (JAX with CUDA), CPU Build (JAX CPU-only), and GPU Build (final training image).

### Running Locally

Use `docker-compose` with `training` service for GPU, `training-cpu` for CPU testing, and `--profile monitoring` for the full monitoring stack.

## Kubernetes Deployment

### Helm Chart

Located in `helm/rtdlm/` with templates for ConfigMap, Secrets, ServiceAccount, PVCs, Training Deployment, Distributed Training, Service, Ingress, Metrics, and NetworkPolicy.

### Installation

Install with `helm install rtdlm ./helm/rtdlm -n rtdlm --create-namespace`. Customize with `--set` flags for model preset, GPU count, and distributed training options.

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

The training process exposes metrics on port 8000 via `PrometheusTrainingCallback`.

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

Use `scripts/quantize_model.py` for INT8 quantization of trained checkpoints.

### ONNX Export

Use `scripts/export_to_onnx.py` to export models to ONNX format.

## Development Workflow

### Quick Commands

Common Makefile targets: `make install-dev`, `make test`, `make test-cov`, `make format`, `make lint`, `make docker-build`, `make helm-install`, `make train-tiny`.

### Local Development

Set up a Python virtual environment, install with `pip install -e ".[dev]"`, run tests with pytest, and start training with the desired preset.

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

1. **GPU not detected**: Check pod logs and nvidia-smi in init container

2. **OOM errors**: Reduce batch size, enable gradient checkpointing, or use smaller model preset

3. **Slow training**: Check GPU utilization, enable XLA compilation caching, check data loading bottlenecks

4. **Checkpoint issues**: Verify PVC mount, check storage class supports ReadWriteOnce, ensure sufficient space

## References

- [JAX Documentation](https://jax.readthedocs.io/)
- [Haiku Documentation](https://dm-haiku.readthedocs.io/)
- [Kubernetes GPU Scheduling](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/)
- [Helm Charts](https://helm.sh/docs/)
