# RT-DLM Kubernetes & Helm Architecture

> Reference guide for how RT-DLM is packaged, deployed, and scheduled on
> Kubernetes — both via the Helm chart (`helm/rtdlm/`) and via raw
> Kustomize manifests (`k8s/base/` + `k8s/training/`).

---

## Table of Contents

1. [TL;DR](#tldr)
2. [Mental model: what Kubernetes is doing](#mental-model-what-kubernetes-is-doing)
3. [Mental model: what Helm adds on top](#mental-model-what-helm-adds-on-top)
4. [Repository layout](#repository-layout)
5. [Resource graph](#resource-graph)
6. [GPU scheduling pipeline](#gpu-scheduling-pipeline)
7. [Distributed training topology](#distributed-training-topology)
8. [Security posture](#security-posture)
9. [Operational workflow](#operational-workflow)
10. [Choosing Helm vs raw Kustomize](#choosing-helm-vs-raw-kustomize)
11. [Glossary](#glossary)

---

## TL;DR

| Question | Answer |
|---|---|
| What is Kubernetes (k8s)? | A cluster OS that schedules containers onto machines, gives them networking, storage, and self-healing. |
| What is Helm? | A package manager for k8s. It templates YAML manifests with values and tracks installed "releases". |
| Why ship both? | Helm is great for production releases (one command, parameterized). Raw Kustomize is great for GitOps / air-gapped clusters and easier auditing. |
| Where does RT-DLM run? | As a `Job` (single-node) or an indexed `Job` (multi-node, NCCL/RDMA-enabled) on GPU nodes selected by `nvidia.com/gpu.product` labels. |
| How are GPUs requested? | Through the standard `nvidia.com/gpu` resource exposed by the NVIDIA device plugin, plus a `RuntimeClass: nvidia` that ensures the NVIDIA container runtime is used. |

---

## Mental model: what Kubernetes is doing

Think of Kubernetes as a **distributed cron + scheduler + load balancer + secret store + storage broker**, all wrapped behind a declarative API. You don't tell it *how* to run things; you tell it *what* you want, and a control loop reconciles toward that desired state.

```
                ┌────────────────────────────────────────────────┐
                │                  kube-apiserver                │
                │  (single source of truth — every object goes  │
                │   through here, validated, persisted in etcd) │
                └───────────────┬────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
┌──────────────┐       ┌─────────────────┐     ┌─────────────────┐
│  scheduler   │       │   controllers   │     │   kubelets      │
│ (assigns     │       │ (Job, ReplicaSet│     │ (run containers │
│  pods → node)│       │  ServiceAccount │     │  on each node)  │
│              │       │  reconcilers)   │     │                 │
└──────┬───────┘       └─────────────────┘     └────────┬────────┘
       │                                                │
       │   "this pod needs 4×nvidia.com/gpu             │
       │    + 32Gi RAM + matches nodeSelector"          │
       ▼                                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                       GPU Worker Nodes                          │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│   │ Node A (8×H100) Node B (8×H100) │  Node C (8×H100) ...      │
│   │  • kubelet   │  • kubelet      │  • kubelet                 │
│   │  • container │  • container    │  • container               │
│   │    runtime   │    runtime      │    runtime                 │
│   │  • NVIDIA    │  • NVIDIA       │  • NVIDIA                  │
│   │    device    │    device       │    device                  │
│   │    plugin    │    plugin       │    plugin                  │
│   └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

### Core objects RT-DLM uses

| Kind | Purpose | Where defined |
|---|---|---|
| `Namespace` | Logical isolation (`rtdlm`) with Pod Security restrictions | `k8s/base/namespace.yaml`, helm namespace template |
| `ConfigMap` | Non-secret training hyperparameters + JAX/NCCL env vars | `k8s/base/configmap.yaml`, `helm/rtdlm/templates/configmap.yaml` |
| `Secret` | HF token, AWS keys, WandB key | `helm/rtdlm/templates/secrets.yaml` |
| `ServiceAccount` + `Role` + `RoleBinding` | Workload identity with least-privilege RBAC | `serviceaccount.yaml` |
| `Job` (single-node) | One-shot training run | `training-job.yaml` |
| `Job` (`completionMode: Indexed`, multi-node) | Distributed training; index injected as `JOB_COMPLETION_INDEX` | `distributed-training.yaml` |
| `Service` (headless, `clusterIP: None`) | Stable DNS for inter-worker NCCL discovery | `headless-service.yaml` |
| `PersistentVolumeClaim` | Checkpoints, logs, training data | `storage.yaml` |
| `NetworkPolicy` | Default-deny + scoped allow for monitoring scrapes | `networkpolicy.yaml` |
| `PriorityClass` | Lets training jobs preempt lower-priority workloads on saturated GPU pools | `priorityclass.yaml` |
| `RuntimeClass: nvidia` | Tells the kubelet to use the NVIDIA container runtime (mounts CUDA libs, exposes `/dev/nvidia*`) | `runtimeclass.yaml` |
| `PodDisruptionBudget` | Prevents the cluster autoscaler / node drains from killing in-flight workers | `pdb.yaml` |

### The control loop

For every object you POST to the API server:

1. The **scheduler** picks a node that satisfies `resources.requests`, `nodeSelector`, `affinity`, `tolerations`, `topologySpreadConstraints`, and `runtimeClassName`.
2. The **kubelet** on that node pulls the image, applies the `RuntimeClass`, mounts volumes/secrets/configmaps, and starts the container.
3. **Controllers** keep reconciling — if a pod dies, a Job's controller starts a new one (up to `backoffLimit`).
4. **kube-proxy / CNI** programs node iptables / eBPF so pod-to-pod traffic resolves through the headless service's DNS.

---

## Mental model: what Helm adds on top

Helm is **client-side templating + a small server-side release record**. It does not extend the Kubernetes API.

```
┌─────────────────────┐        ┌──────────────────────────┐
│  helm/rtdlm/        │        │                          │
│  ├── Chart.yaml     │        │   helm install rtdlm     │
│  ├── values.yaml    │  ───▶  │     ./helm/rtdlm \       │
│  ├── templates/     │        │     -f values.prod.yaml  │
│  │   ├── *.yaml     │        │                          │
│  │   └── _helpers.tpl       │ │                          │
│  └── configmaps/    │        │                          │
└─────────────────────┘        └────────────┬─────────────┘
                                            │
                       ┌────────────────────┴────────────────────┐
                       │  1. Render Go templates with values     │
                       │  2. Validate against k8s OpenAPI schema │
                       │  3. POST every manifest to kube-apiserver
                       │  4. Save a "release" Secret in the      │
                       │     namespace (so `helm upgrade` knows  │
                       │     what to diff against next time)     │
                       └────────────────────┬────────────────────┘
                                            ▼
                            ┌──────────────────────────────┐
                            │  Kubernetes objects created  │
                            │  (identical to what kubectl  │
                            │   apply -k k8s/training      │
                            │   would produce)             │
                            └──────────────────────────────┘
```

### What lives in the chart

```
helm/rtdlm/
├── Chart.yaml          # name, version, appVersion, deps
├── values.yaml         # default knobs (overridable per env)
└── templates/
    ├── _helpers.tpl                    # named templates: rtdlm.fullname, labels
    ├── namespace.yaml
    ├── serviceaccount.yaml
    ├── configmap.yaml
    ├── secrets.yaml
    ├── networkpolicy.yaml
    ├── storage.yaml                    # PVCs
    ├── training_deployment.yaml        # single-node Job
    ├── distributed_training.yaml       # multi-node Indexed Job + headless svc
    ├── training_service.yaml           # metrics service
    ├── training_ingress.yaml
    ├── metrics_deployment.yaml         # Prometheus exporter
    ├── priorityclass.yaml              # GPU scheduling priority
    ├── poddisruptionbudget.yaml        # protects in-flight workers
    └── runtimeclass.yaml               # NVIDIA container runtime
```

### Why use Helm here

- **One toggle, many manifests.** Setting `gpuScheduling.spreadAcrossZones: true` in `values.yaml` injects a `topologySpreadConstraints` block into the right Job spec without touching every YAML.
- **Environment overlays.** `values.dev.yaml` (1 GPU) vs `values.prod.yaml` (32× H100, RDMA on) reuse the same templates.
- **Rollback.** `helm rollback rtdlm 3` restores release revision 3 from the chart's release secret.
- **Dependency management.** `Chart.yaml` can declare sub-charts (e.g. Prometheus) that Helm pulls and installs together.

---

## Repository layout

```
RT-DLM/
├── helm/
│   └── rtdlm/                ← Helm chart (parameterized)
│       ├── Chart.yaml
│       ├── values.yaml
│       └── templates/
└── k8s/                      ← Raw Kustomize manifests (no templating)
    ├── base/                 ← Cluster-scope + ns-scope basics
    │   ├── kustomization.yaml
    │   ├── namespace.yaml
    │   ├── serviceaccount.yaml
    │   ├── configmap.yaml
    │   ├── networkpolicy.yaml
    │   ├── storage.yaml
    │   ├── priorityclass.yaml
    │   └── runtimeclass.yaml
    └── training/             ← Workload overlay (depends on base/)
        ├── kustomization.yaml
        ├── training-job.yaml
        ├── distributed-training.yaml
        ├── headless-service.yaml
        └── pdb.yaml
```

**Both paths produce equivalent objects.** Pick whichever fits your delivery story (see [§ Choosing Helm vs raw Kustomize](#choosing-helm-vs-raw-kustomize)).

---

## Resource graph

This is the dependency graph for a distributed training run, regardless of whether it was created via Helm or Kustomize:

```
                        ┌─────────────────┐
                        │   Namespace     │
                        │     rtdlm       │
                        └────────┬────────┘
                                 │
        ┌────────────────────────┼────────────────────────────┐
        ▼                        ▼                            ▼
┌──────────────┐        ┌─────────────────┐         ┌────────────────┐
│ ConfigMap    │        │ ServiceAccount  │         │ NetworkPolicy  │
│ rtdlm-config │        │  + Role + RB    │         │ default-deny + │
│ (hyperparams)│        │                 │         │ scoped-allow   │
└──────┬───────┘        └────────┬────────┘         └────────────────┘
       │                         │
       │ envFrom                 │ serviceAccountName
       │                         │
       │  ┌──────────────────────┴────────────────────────┐
       │  │                                                │
       │  │              Job (Indexed, parallelism=N)      │
       │  │  ┌───────────┬───────────┬───────────┐         │
       │  │  │  Pod 0    │  Pod 1    │  Pod N-1  │         │
       └──┤  │ ──────────│ ──────────│ ────────── │ ◀──────┤ runtimeClassName: nvidia
          │  │ JOB_INDEX │ JOB_INDEX │ JOB_INDEX │         │ priorityClassName: rtdlm-high
          │  │   = 0     │   = 1     │   = N-1   │         │
          │  │           │           │           │         │
          │  │ requests: │ requests: │ requests: │         │
          │  │ 4×GPU,    │ 4×GPU,    │ 4×GPU,    │         │
          │  │ 1×RDMA    │ 1×RDMA    │ 1×RDMA    │         │
          │  └─────┬─────┴─────┬─────┴─────┬─────┘         │
          │        │           │           │               │
          │        └───────────┼───────────┘               │
          │                    │ NCCL all-reduce           │
          │                    │ (port 29500)              │
          └────────────────────┼───────────────────────────┘
                               │
                ┌──────────────▼──────────────┐
                │   Service (headless)         │
                │   rtdlm-workers              │
                │   clusterIP: None            │
                │   → DNS: pod-N.subdomain     │
                └──────────────────────────────┘

        ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
        │ PVC          │  │ PVC          │  │ PVC          │
        │ checkpoints  │  │ logs         │  │ data (RO)    │
        └──────────────┘  └──────────────┘  └──────────────┘
                ▲                ▲                ▲
                └────────────────┴────────────────┘
                          mounted into every pod
```

---

## GPU scheduling pipeline

This is what happens between `kubectl apply` and `nvidia-smi` showing utilization.

```
1.  YOU APPLY MANIFEST
    └─ kubectl / helm  ──▶  kube-apiserver  ──▶  etcd

2.  SCHEDULER PHASE  (kube-scheduler)
    For each pod, filter nodes by:
       ✓ resources.requests.nvidia.com/gpu  (must have free GPUs)
       ✓ nodeSelector.nvidia.com/gpu.present = "true"
       ✓ affinity.nodeAffinity → in (A100-80GB | H100 | H200 | …)
       ✓ tolerations match node taint  nvidia.com/gpu:NoSchedule
       ✓ podAntiAffinity → no two trainer pods on same hostname
       ✓ topologySpreadConstraints → balance across AZs
       ✓ runtimeClassName "nvidia" → node supports it
    Score remaining nodes, pick best, bind pod → node.

3.  KUBELET PHASE  (on chosen node)
    └─ Reads RuntimeClass "nvidia" → uses nvidia-container-runtime
    └─ NVIDIA Device Plugin allocates specific GPU UUIDs to the pod
    └─ Mounts /dev/nvidia0..N, /usr/local/nvidia/{lib64,bin}
    └─ Starts container with NVIDIA_VISIBLE_DEVICES set

4.  CONTAINER PHASE
    └─ JAX/PyTorch sees only the allocated GPUs
    └─ NCCL initializes using NCCL_SOCKET_IFNAME / NCCL_IB_HCA
    └─ Workers discover each other via headless Service DNS
    └─ All-reduce begins
```

### Where each policy is enforced

| Concern | Mechanism | RT-DLM example |
|---|---|---|
| "Only schedule on GPU nodes" | `nodeSelector` | `nvidia.com/gpu.present: "true"` |
| "Only A100 or newer" | `nodeAffinity` + `nvidia.com/gpu.product` label | `In: [A100-80GB, H100-80GB-HBM3, H200]` |
| "Tolerate the GPU taint" | `tolerations` | `key: nvidia.com/gpu, operator: Exists` |
| "One trainer per node" | `podAntiAffinity` (required) | `topologyKey: kubernetes.io/hostname` |
| "Spread across AZs" | `topologySpreadConstraints` | `topologyKey: topology.kubernetes.io/zone` |
| "Use NVIDIA runtime" | `runtimeClassName` | `nvidia` |
| "Preempt batch jobs if needed" | `priorityClassName` | `rtdlm-high-priority` (value 1_000_000) |
| "Don't get evicted mid-run" | `PodDisruptionBudget` | `minAvailable: "100%"` |
| "RDMA InfiniBand for NCCL" | extended resource | `rdma/hca_shared_devices_a: 1` + `IPC_LOCK` cap |
| "Slice a single GPU into MIG" | resource name + label | `nvidia.com/mig-1g.10gb: 1` (`gpuScheduling.migProfile`) |

---

## Distributed training topology

```
              ┌─────────────────────────────────────────────────┐
              │           Job (parallelism=4, completions=4)    │
              │                                                 │
              │  Each pod gets:                                 │
              │   • JOB_COMPLETION_INDEX env  (0..3)            │
              │   • Same hostname pattern: <job>-<idx>          │
              │   • Resolved DNS via headless Service:          │
              │       <job>-0.rtdlm-workers.rtdlm.svc...        │
              └────┬────────────┬────────────┬────────────┬─────┘
                   │            │            │            │
                   ▼            ▼            ▼            ▼
            ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
            │ rank 0   │ │ rank 1   │ │ rank 2   │ │ rank 3   │
            │ (master) │ │          │ │          │ │          │
            │          │ │          │ │          │ │          │
            │ 4×GPU    │ │ 4×GPU    │ │ 4×GPU    │ │ 4×GPU    │
            │ Node A   │ │ Node B   │ │ Node C   │ │ Node D   │
            └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘
                 │            │            │            │
                 │            │            │            │
                 └────────────┴───┬────────┴────────────┘
                                  │
                          NCCL ring/tree all-reduce
                          • TCP fallback on eth0 (port 29500)
                          • InfiniBand (`mlx5`) when RDMA enabled
```

### Master discovery

The training command uses:

```
--master_addr=rtdlm-training-distributed-0.rtdlm-workers.rtdlm.svc.cluster.local
--master_port=29500
--node_rank=$(JOB_COMPLETION_INDEX)
--world_size=4
```

That FQDN resolves only because the **headless Service** publishes per-pod DNS records (`clusterIP: None` + `publishNotReadyAddresses: true`). Without it, rank 0's hostname would not be discoverable until it became Ready, which deadlocks NCCL initialization.

---

## Security posture

The chart and base ship with conservative defaults:

| Layer | Default |
|---|---|
| **Pod Security Standards** | `restricted` (enforce + audit + warn) on the namespace |
| **Service account token** | `automountServiceAccountToken: false` |
| **Container user** | `runAsNonRoot: true`, uid/gid 1000 |
| **Capabilities** | `drop: ["ALL"]`; only `IPC_LOCK` added when RDMA is enabled (justified — required for memory pinning during InfiniBand zero-copy) |
| **Seccomp** | `RuntimeDefault` |
| **Privilege escalation** | `allowPrivilegeEscalation: false` |
| **Network egress** | Default-deny; only DNS, HTTPS to public, and same-namespace traffic allowed |
| **RBAC** | `Role` scoped to read `configmaps`, `secrets`, and own pods; no cluster-wide verbs |

---

## Operational workflow

### Install (Helm)

```bash
helm install rtdlm ./helm/rtdlm \
  --namespace rtdlm --create-namespace \
  -f ./helm/rtdlm/values.yaml \
  --set distributed_training.enabled=true \
  --set distributed_training.workers=4 \
  --set gpuScheduling.spreadAcrossNodes=true \
  --set gpuScheduling.rdma.enabled=true \
  --set priorityClass.enabled=true \
  --set runtimeClass.enabled=true
```

### Install (Kustomize)

```bash
kubectl apply -k k8s/base
kubectl apply -k k8s/training
```

### Inspect

```bash
kubectl -n rtdlm get pods,jobs,svc,pvc
kubectl -n rtdlm logs -f job/rtdlm-training-distributed
kubectl -n rtdlm describe pod rtdlm-training-distributed-0   # see scheduling decisions
```

### Upgrade / rollback (Helm only)

```bash
helm upgrade rtdlm ./helm/rtdlm -f values.prod.yaml
helm history rtdlm
helm rollback rtdlm 2
```

### Tear down

```bash
helm uninstall rtdlm -n rtdlm                 # Helm path
kubectl delete -k k8s/training                # Kustomize path
kubectl delete -k k8s/base
```

---

## Choosing Helm vs raw Kustomize

| You want… | Use |
|---|---|
| One-command install with knobs (`--set`, `-f`) | **Helm** |
| `helm rollback`, release history | **Helm** |
| Sub-chart dependencies (Prometheus, etc.) | **Helm** |
| GitOps with ArgoCD/Flux applying raw manifests | **Kustomize** |
| Minimal supply chain (no Go-template engine) | **Kustomize** |
| Per-environment overlays (dev/staging/prod) | **Kustomize** (overlays directory) |
| Air-gapped clusters (no Helm in image) | **Kustomize** |

Both produce equivalent objects, so you can switch without re-architecting.

---

## Glossary

| Term | One-liner |
|---|---|
| **Pod** | Smallest deployable unit; one or more containers sharing network + storage. |
| **Job** | Run-to-completion workload. `Indexed` mode gives each pod a stable rank via `JOB_COMPLETION_INDEX`. |
| **Service** | Stable virtual IP (or DNS-only when "headless") that load-balances to pods. |
| **PVC** | Persistent Volume Claim — request for durable storage; bound by the cluster to a real volume. |
| **ConfigMap / Secret** | Key-value blobs mounted into pods as env vars or files. Secrets are base64 + RBAC-restricted. |
| **NodeSelector / Affinity** | Hard / soft constraints on which nodes a pod may land on. |
| **Toleration** | Counterpart to a node taint; lets a pod schedule onto an otherwise-tainted node. |
| **Topology Spread** | Soft balancing across failure domains (zones, racks). |
| **PriorityClass** | Higher-value pods can preempt lower-value ones when the cluster is full. |
| **PodDisruptionBudget** | Floor on how many pods of a workload must remain Ready during voluntary disruptions. |
| **RuntimeClass** | Selects the container runtime (e.g. `nvidia` for GPU workloads). |
| **NCCL** | NVIDIA Collective Communications Library — does the all-reduce across GPUs/nodes. |
| **RDMA / InfiniBand** | Zero-copy NIC-level transfers; enabled in the chart by `gpuScheduling.rdma.enabled`. |
| **MIG** | Multi-Instance GPU — partition one A100/H100 into smaller virtual GPUs. |
| **Helm release** | A named installation of a chart, tracked by a Secret in the namespace. |
| **Kustomize overlay** | A directory that references a `base/` and patches/adds resources on top. |
