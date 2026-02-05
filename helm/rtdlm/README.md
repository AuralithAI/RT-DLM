# RT-DLM Helm Chart

A Helm chart for deploying RT-DLM (Real-Time Deep Learning Model) training infrastructure on Kubernetes.

## Prerequisites

- Kubernetes 1.25+
- Helm 3.10+
- NVIDIA GPU Operator (for GPU nodes)
- PV provisioner support in the cluster
- Ingress controller (nginx recommended)

## Installation

### Add the Helm repository

Add with `helm repo add rtdlm https://auralithai.github.io/rt-dlm-charts` and `helm repo update`.

### Install the chart

Install with `helm install rtdlm rtdlm/rtdlm -n rtdlm --create-namespace`. Use `-f my-values.yaml` for custom values or install from local with `helm install rtdlm ./helm/rtdlm -n rtdlm --create-namespace`.

### Upgrade

Use `helm upgrade rtdlm rtdlm/rtdlm -n rtdlm -f my-values.yaml`.

### Uninstall

Use `helm uninstall rtdlm -n rtdlm`.

## Configuration

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `Name` | Release name | `rtdlm` |
| `Namespace` | Kubernetes namespace | `rtdlm` |
| `training.enabled` | Enable training deployment | `true` |
| `training.replicaCount` | Number of training replicas | `1` |
| `training.gpu.enabled` | Enable GPU support | `true` |
| `training.gpu.count` | Number of GPUs per pod | `4` |
| `training.model.preset` | Model preset (tiny/small/medium/large) | `medium` |
| `training.model.batch_size` | Training batch size | `32` |
| `training.model.epochs` | Number of training epochs | `100` |
| `distributed_training.enabled` | Enable distributed training | `false` |
| `distributed_training.workers` | Number of distributed workers | `4` |

### Example Values

Customize training configuration via values file with options for GPU settings (`training.gpu.count`, `training.gpu.type`), model settings (`training.model.preset`, `training.model.batch_size`), resource limits, distributed training, and WandB integration.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Kubernetes Cluster                        │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                  rtdlm namespace                         │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │ │
│  │  │  ConfigMap  │  │   Secret    │  │    PVCs     │     │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘     │ │
│  │         │                │                │              │ │
│  │         ▼                ▼                ▼              │ │
│  │  ┌─────────────────────────────────────────────────┐    │ │
│  │  │              Training Deployment                 │    │ │
│  │  │  ┌─────────────────────────────────────────┐    │    │ │
│  │  │  │              Pod (GPU)                   │    │    │ │
│  │  │  │  ┌─────────┐  ┌─────────┐  ┌─────────┐ │    │    │ │
│  │  │  │  │ Trainer │  │ Metrics │  │  Logs   │ │    │    │ │
│  │  │  │  └─────────┘  └─────────┘  └─────────┘ │    │    │ │
│  │  │  └─────────────────────────────────────────┘    │    │ │
│  │  └─────────────────────────────────────────────────┘    │ │
│  │         │                                                │ │
│  │         ▼                                                │ │
│  │  ┌─────────────┐      ┌─────────────┐                   │ │
│  │  │   Service   │ ──── │   Ingress   │                   │ │
│  │  └─────────────┘      └─────────────┘                   │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Monitoring

The chart exposes Prometheus metrics on port 8000. Configure your Prometheus instance with kubernetes_sd_configs to scrape pods in the rtdlm namespace with the `prometheus.io/scrape: true` annotation.

## License

Apache 2.0
