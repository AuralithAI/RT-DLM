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

```bash
helm repo add rtdlm https://auralithai.github.io/rt-dlm-charts
helm repo update
```

### Install the chart

```bash
# Install with default values
helm install rtdlm rtdlm/rtdlm -n rtdlm --create-namespace

# Install with custom values
helm install rtdlm rtdlm/rtdlm -n rtdlm --create-namespace -f my-values.yaml

# Install from local chart
helm install rtdlm ./helm/rtdlm -n rtdlm --create-namespace
```

### Upgrade

```bash
helm upgrade rtdlm rtdlm/rtdlm -n rtdlm -f my-values.yaml
```

### Uninstall

```bash
helm uninstall rtdlm -n rtdlm
```

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

```yaml
# Custom training configuration
training:
  enabled: true
  replicaCount: 1
  
  gpu:
    enabled: true
    count: 8
    type: nvidia-a100
  
  model:
    preset: large
    batch_size: 64
    epochs: 200
    learning_rate: "5e-5"
  
  resources:
    requests:
      memory: "64Gi"
      cpu: "16"
      nvidia.com/gpu: 8
    limits:
      memory: "128Gi"
      cpu: "32"
      nvidia.com/gpu: 8

# Enable distributed training
distributed_training:
  enabled: true
  workers: 4

# Enable WandB logging
training:
  wandb:
    enabled: true
    project: my-project
    mode: online

secrets:
  wandb_api_key: "your-api-key"
```

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

The chart exposes Prometheus metrics on port 8000. Configure your Prometheus instance to scrape:

```yaml
scrape_configs:
  - job_name: 'rtdlm-training'
    kubernetes_sd_configs:
      - role: pod
        namespaces:
          names:
            - rtdlm
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
```

## License

Apache 2.0
