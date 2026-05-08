"""Model card + compute disclosure generator (HF model-card spec compatible)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

GPU_TDP_WATTS: Dict[str, float] = {
    "nvidia-a100-40gb": 400.0,
    "nvidia-a100-80gb": 400.0,
    "nvidia-h100-80gb": 700.0,
    "nvidia-h200": 700.0,
    "nvidia-v100-32gb": 300.0,
    "nvidia-t4": 70.0,
    "tpu-v4": 200.0,
    "tpu-v5e": 175.0,
    "tpu-v5p": 250.0,
}

DEFAULT_PUE = 1.12
DEFAULT_GRID_KGCO2_PER_KWH = 0.387


@dataclass
class HardwareInventory:
    """Hardware used during a training run."""

    accelerator: str
    accelerator_count: int
    accelerator_hours: float
    cpu_cores: Optional[int] = None
    memory_gb: Optional[int] = None
    interconnect: Optional[str] = None
    cloud_provider: Optional[str] = None
    region: Optional[str] = None


@dataclass
class ComputeDisclosure:
    """Compute, energy, and carbon footprint of a training run."""

    total_flops: float
    training_tokens: int
    hardware: HardwareInventory
    pue: float = DEFAULT_PUE
    grid_kgco2_per_kwh: float = DEFAULT_GRID_KGCO2_PER_KWH
    energy_kwh: Optional[float] = None
    co2_kg: Optional[float] = None

    def estimate_energy_and_carbon(self) -> "ComputeDisclosure":
        """Populate energy_kwh and co2_kg from hardware + PUE + grid intensity."""
        tdp = GPU_TDP_WATTS.get(self.hardware.accelerator.lower(), 400.0)
        kwh_raw = tdp * self.hardware.accelerator_count * self.hardware.accelerator_hours / 1000.0
        self.energy_kwh = kwh_raw * self.pue
        self.co2_kg = self.energy_kwh * self.grid_kgco2_per_kwh
        return self


@dataclass
class DatasetReference:
    """Single training/eval dataset reference for the model card."""

    name: str
    license: str = "unknown"
    homepage: str = ""
    citation: str = ""
    split: Optional[str] = None
    num_examples: Optional[int] = None
    role: str = "training"


@dataclass
class EvaluationResult:
    """Single benchmark result."""

    benchmark: str
    metric: str
    value: float
    split: Optional[str] = None
    notes: str = ""


@dataclass
class ModelCard:
    """HF-compatible model card with safety + compute sections."""

    model_name: str
    model_version: str
    description: str
    architecture: str
    parameters: int
    license: str
    authors: List[str] = field(default_factory=list)
    contact: str = ""
    languages: List[str] = field(default_factory=lambda: ["en"])
    datasets: List[DatasetReference] = field(default_factory=list)
    evaluations: List[EvaluationResult] = field(default_factory=list)
    intended_use: str = ""
    out_of_scope_use: str = ""
    limitations: str = ""
    bias_risks: str = ""
    safety_mitigations: List[str] = field(default_factory=list)
    compute: Optional[ComputeDisclosure] = None
    training_config: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dict representation."""
        return asdict(self)

    def to_json(self, path: Optional[Path] = None) -> str:
        """Serialize to JSON; writes to disk if `path` is provided."""
        text = json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    def to_markdown(self, path: Optional[Path] = None) -> str:
        """Render the card in HF model-card markdown format."""
        text = _render_markdown(self)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text


def _render_yaml_frontmatter(card: ModelCard) -> str:
    """YAML frontmatter accepted by the HF Hub model-card parser."""
    langs = "\n".join(f"  - {l}" for l in card.languages)
    datasets = "\n".join(f"  - {d.name}" for d in card.datasets)
    metrics = sorted({e.metric for e in card.evaluations})
    metrics_block = "\n".join(f"  - {m}" for m in metrics) if metrics else "  - accuracy"
    return (
        "---\n"
        f"language:\n{langs}\n"
        f"license: {card.license}\n"
        f"library_name: rt-dlm\n"
        f"tags:\n  - rt-dlm\n  - jax\n  - haiku\n"
        f"datasets:\n{datasets if datasets else '  - none'}\n"
        f"metrics:\n{metrics_block}\n"
        "---\n\n"
    )


def _render_eval_table(card: ModelCard) -> str:
    """Render the evaluation block as a markdown table."""
    if not card.evaluations:
        return "_No evaluations recorded._\n"
    rows = ["| Benchmark | Metric | Value | Split | Notes |", "|---|---|---|---|---|"]
    for e in card.evaluations:
        rows.append(f"| {e.benchmark} | {e.metric} | {e.value:.4f} | {e.split or '-'} | {e.notes or '-'} |")
    return "\n".join(rows) + "\n"


def _render_compute_block(card: ModelCard) -> str:
    """Render the compute disclosure section."""
    if card.compute is None:
        return "_Compute disclosure not provided._\n"
    c = card.compute
    h = c.hardware
    energy = f"{c.energy_kwh:.2f} kWh" if c.energy_kwh is not None else "n/a"
    co2 = f"{c.co2_kg:.2f} kg CO₂eq" if c.co2_kg is not None else "n/a"
    return (
        f"- **Total FLOPs**: {c.total_flops:.3e}\n"
        f"- **Training tokens**: {c.training_tokens:,}\n"
        f"- **Hardware**: {h.accelerator_count} × {h.accelerator}\n"
        f"- **Accelerator hours**: {h.accelerator_hours:,.1f}\n"
        f"- **Interconnect**: {h.interconnect or 'n/a'}\n"
        f"- **Cloud / region**: {h.cloud_provider or 'n/a'} / {h.region or 'n/a'}\n"
        f"- **PUE**: {c.pue}\n"
        f"- **Grid intensity**: {c.grid_kgco2_per_kwh} kg CO₂/kWh\n"
        f"- **Estimated energy**: {energy}\n"
        f"- **Estimated CO₂**: {co2}\n"
    )


def _render_dataset_block(card: ModelCard) -> str:
    """Render the dataset references."""
    if not card.datasets:
        return "_No datasets recorded._\n"
    rows = ["| Dataset | Role | License | Examples | Homepage |", "|---|---|---|---|---|"]
    for d in card.datasets:
        n = f"{d.num_examples:,}" if d.num_examples else "?"
        rows.append(f"| {d.name} | {d.role} | {d.license} | {n} | {d.homepage or '-'} |")
    return "\n".join(rows) + "\n"


def _render_markdown(card: ModelCard) -> str:
    """Compose the full markdown document."""
    parts = [
        _render_yaml_frontmatter(card),
        f"# {card.model_name} ({card.model_version})\n\n",
        f"{card.description}\n\n",
        "## Model Details\n",
        f"- **Architecture**: {card.architecture}\n",
        f"- **Parameters**: {card.parameters:,}\n",
        f"- **License**: {card.license}\n",
        f"- **Authors**: {', '.join(card.authors) or 'n/a'}\n",
        f"- **Contact**: {card.contact or 'n/a'}\n",
        f"- **Created**: {card.created_at}\n\n",
        "## Intended Use\n",
        f"{card.intended_use or '_Not specified._'}\n\n",
        "## Out of Scope\n",
        f"{card.out_of_scope_use or '_Not specified._'}\n\n",
        "## Limitations\n",
        f"{card.limitations or '_Not specified._'}\n\n",
        "## Bias, Risks & Safety\n",
        f"{card.bias_risks or '_Not specified._'}\n\n",
        "**Safety Mitigations:**\n",
    ]
    if card.safety_mitigations:
        parts.extend(f"- {m}\n" for m in card.safety_mitigations)
    else:
        parts.append("- _None recorded._\n")
    parts.extend(
        [
            "\n## Training Datasets\n",
            _render_dataset_block(card),
            "\n## Evaluation Results\n",
            _render_eval_table(card),
            "\n## Compute Disclosure\n",
            _render_compute_block(card),
            "\n## Training Configuration\n",
            "```json\n",
            json.dumps(card.training_config, indent=2, default=str),
            "\n```\n",
        ]
    )
    return "".join(parts)


def estimate_transformer_flops(num_params: int, training_tokens: int, include_backward: bool = True) -> float:
    """Standard 6N×T (fwd+bwd) or 2N×T (fwd-only) FLOPs estimate."""
    multiplier = 6.0 if include_backward else 2.0
    return multiplier * float(num_params) * float(training_tokens)


def build_compute_disclosure(
    num_params: int,
    training_tokens: int,
    accelerator: str,
    accelerator_count: int,
    accelerator_hours: float,
    interconnect: Optional[str] = None,
    cloud_provider: Optional[str] = None,
    region: Optional[str] = None,
    pue: float = DEFAULT_PUE,
    grid_kgco2_per_kwh: float = DEFAULT_GRID_KGCO2_PER_KWH,
) -> ComputeDisclosure:
    """Build a ComputeDisclosure with FLOPs + energy + CO₂ filled in."""
    hw = HardwareInventory(
        accelerator=accelerator,
        accelerator_count=accelerator_count,
        accelerator_hours=accelerator_hours,
        interconnect=interconnect,
        cloud_provider=cloud_provider,
        region=region,
    )
    cd = ComputeDisclosure(
        total_flops=estimate_transformer_flops(num_params, training_tokens),
        training_tokens=training_tokens,
        hardware=hw,
        pue=pue,
        grid_kgco2_per_kwh=grid_kgco2_per_kwh,
    )
    return cd.estimate_energy_and_carbon()
