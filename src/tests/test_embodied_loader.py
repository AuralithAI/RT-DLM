"""Tests for the embodied data loader and modality mixer."""

import jax.numpy as jnp
import numpy as np
import pytest

from src.core.training.embodied_loader import (
    EmbodiedDataLoader,
    EmbodiedSample,
    MixedModalityBatcher,
    MixerSpec,
    action_token_loss,
)


def _provider(i: int) -> EmbodiedSample:
    """Synthetic robot sample provider for tests."""
    rng = np.random.default_rng(i)
    return EmbodiedSample(
        images=jnp.asarray(rng.random((4, 4, 3), dtype=np.float32)),
        proprio=jnp.asarray(rng.random(8, dtype=np.float32)),
        actions=jnp.asarray(rng.random(7, dtype=np.float32)),
    )


def test_loader_yields_correct_batch_shape():
    """Loader should yield batched arrays with matching first axis."""
    loader = EmbodiedDataLoader(_provider, num_samples=8, batch_size=4, shuffle=False)
    batches = list(loader)
    assert len(batches) == 2
    assert batches[0]["images"].shape == (4, 4, 4, 3)
    assert batches[0]["proprio"].shape == (4, 8)
    assert batches[0]["actions"].shape == (4, 7)


def test_loader_handles_partial_final_batch():
    """Loader should still emit a partial batch at the end."""
    loader = EmbodiedDataLoader(_provider, num_samples=5, batch_size=2, shuffle=False)
    batches = list(loader)
    assert batches[-1]["images"].shape[0] == 1


def test_mixer_invalid_weights():
    """Mixer must reject zero-sum weights."""
    spec = MixerSpec(name="x", loader=iter([]), weight=0.0)
    with pytest.raises(ValueError):
        MixedModalityBatcher([spec])


def test_mixer_round_robin_recovers():
    """Mixer must restart exhausted loaders cleanly."""
    loader1 = EmbodiedDataLoader(_provider, num_samples=2, batch_size=2, shuffle=False)
    loader2 = EmbodiedDataLoader(_provider, num_samples=2, batch_size=2, shuffle=False)
    mixer = MixedModalityBatcher(
        [
            MixerSpec("a", loader1, 1.0),
            MixerSpec("b", loader2, 1.0),
        ]
    )
    seen = []
    for _ in range(6):
        seen.append(next(mixer)["source"])
    assert set(seen).issubset({"a", "b"})
    assert len(seen) == 6


def test_action_token_loss_zero_on_match():
    """Per-axis CE should be near 0 when logits saturate the correct bin."""
    bins = jnp.zeros((2, 4), dtype=jnp.int32)
    logits = jnp.zeros((2, 4, 5)).at[..., 0].set(50.0)
    loss = action_token_loss(logits, bins)
    assert float(loss) < 1e-3
