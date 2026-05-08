"""Tests for the model card + compute disclosure generator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.core.utils.model_card import (
    ComputeDisclosure,
    DatasetReference,
    EvaluationResult,
    HardwareInventory,
    ModelCard,
    build_compute_disclosure,
    estimate_transformer_flops,
)


def _sample_card() -> ModelCard:
    """Construct a complete ModelCard for serialization tests."""
    compute = build_compute_disclosure(
        num_params=1_000_000_000,
        training_tokens=10_000_000_000,
        accelerator="nvidia-h100-80gb",
        accelerator_count=8,
        accelerator_hours=240.0,
        interconnect="NVLink + InfiniBand",
        cloud_provider="aws",
        region="us-east-1",
    )
    return ModelCard(
        model_name="rt-dlm",
        model_version="0.1.0",
        description="A research language model.",
        architecture="Transformer + retrieval + RWKV hybrid",
        parameters=1_000_000_000,
        license="Apache-2.0",
        authors=["AuralithAI"],
        contact="hello@auralithai.com",
        languages=["en"],
        datasets=[
            DatasetReference(
                name="gpqa_diamond",
                license="CC-BY-4.0",
                homepage="https://huggingface.co/datasets/Idavidrein/gpqa",
                role="evaluation",
                num_examples=198,
            )
        ],
        evaluations=[
            EvaluationResult(benchmark="gpqa_diamond", metric="accuracy", value=0.4123),
            EvaluationResult(benchmark="aime", metric="exact_match", value=0.5),
        ],
        intended_use="Research and benchmarking.",
        out_of_scope_use="High-stakes decisions.",
        limitations="May hallucinate.",
        bias_risks="May reflect web biases.",
        safety_mitigations=["RLHF", "DPO", "abstention training"],
        compute=compute,
        training_config={"optimizer": "adamw", "lr": 1e-4},
    )


def test_estimate_transformer_flops_default_includes_backward():
    """Default 6N×T multiplier matches Kaplan et al. accounting."""
    f = estimate_transformer_flops(num_params=10, training_tokens=20)
    assert f == pytest.approx(6.0 * 10 * 20)


def test_estimate_transformer_flops_forward_only():
    """include_backward=False yields the 2N×T forward-only estimate."""
    f = estimate_transformer_flops(num_params=10, training_tokens=20, include_backward=False)
    assert f == pytest.approx(2.0 * 10 * 20)


def test_compute_disclosure_energy_and_carbon_populated():
    """Energy + CO₂ should be filled by build_compute_disclosure helper."""
    cd = build_compute_disclosure(
        num_params=1_000_000,
        training_tokens=1_000_000,
        accelerator="nvidia-a100-80gb",
        accelerator_count=4,
        accelerator_hours=10.0,
    )
    assert cd.energy_kwh is not None and cd.energy_kwh > 0.0
    assert cd.co2_kg is not None and cd.co2_kg > 0.0
    assert cd.total_flops == pytest.approx(6.0 * 1_000_000 * 1_000_000)


def test_unknown_accelerator_falls_back_to_default_tdp():
    """Unknown accelerator strings fall back to a 400 W default rather than crashing."""
    cd = ComputeDisclosure(
        total_flops=1.0,
        training_tokens=1,
        hardware=HardwareInventory(
            accelerator="custom-fpga-xyz",
            accelerator_count=1,
            accelerator_hours=1.0,
        ),
    ).estimate_energy_and_carbon()
    assert cd.energy_kwh is not None
    assert cd.energy_kwh == pytest.approx(0.4 * cd.pue, abs=1e-6)


def test_to_json_round_trips(tmp_path: Path):
    """to_json must produce a valid JSON document containing the expected keys."""
    card = _sample_card()
    out = tmp_path / "card.json"
    card.to_json(path=out)
    data = json.loads(out.read_text())
    assert data["model_name"] == "rt-dlm"
    assert data["compute"]["hardware"]["accelerator"] == "nvidia-h100-80gb"
    assert data["compute"]["energy_kwh"] is not None
    assert data["compute"]["co2_kg"] is not None
    assert len(data["evaluations"]) == 2


def test_to_markdown_contains_required_sections(tmp_path: Path):
    """Markdown output must include the standard model-card sections."""
    card = _sample_card()
    md = card.to_markdown(path=tmp_path / "card.md")
    for section in (
        "# rt-dlm (0.1.0)",
        "## Model Details",
        "## Intended Use",
        "## Limitations",
        "## Bias, Risks & Safety",
        "## Training Datasets",
        "## Evaluation Results",
        "## Compute Disclosure",
        "## Training Configuration",
    ):
        assert section in md
    assert "license: Apache-2.0" in md
    assert "gpqa_diamond" in md
    assert "kg CO₂eq" in md


def test_markdown_handles_minimal_card():
    """Cards without compute / datasets / evals must still render."""
    card = ModelCard(
        model_name="m",
        model_version="0",
        description="d",
        architecture="a",
        parameters=1,
        license="MIT",
    )
    md = card.to_markdown()
    assert "_No evaluations recorded._" in md
    assert "_No datasets recorded._" in md
    assert "_Compute disclosure not provided._" in md


def test_compute_disclosure_pue_scaling():
    """Energy must scale linearly with PUE."""
    a = build_compute_disclosure(
        num_params=1, training_tokens=1, accelerator="nvidia-a100-80gb",
        accelerator_count=1, accelerator_hours=1.0, pue=1.0,
    )
    b = build_compute_disclosure(
        num_params=1, training_tokens=1, accelerator="nvidia-a100-80gb",
        accelerator_count=1, accelerator_hours=1.0, pue=2.0,
    )
    assert b.energy_kwh == pytest.approx(2.0 * (a.energy_kwh or 0.0))
