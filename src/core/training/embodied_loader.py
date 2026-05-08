"""Embodied data loader and modality-mixed batching for robotics co-training."""

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np


@dataclass
class EmbodiedSample:
    """One step of robot trajectory data."""
    images: Optional[jnp.ndarray] = None
    proprio: Optional[jnp.ndarray] = None
    actions: Optional[jnp.ndarray] = None
    instruction_tokens: Optional[jnp.ndarray] = None
    rewards: Optional[jnp.ndarray] = None
    metadata: Dict[str, object] = field(default_factory=dict)


SampleProvider = Callable[[int], EmbodiedSample]


class EmbodiedDataLoader:
    """Iterates embodied samples in fixed batches."""

    def __init__(
        self,
        provider: SampleProvider,
        num_samples: int,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
    ):
        self.provider = provider
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._rng = np.random.default_rng(seed)

    def _iter_indices(self) -> Iterable[int]:
        """Yield sample indices according to shuffle policy."""
        idx = np.arange(self.num_samples)
        if self.shuffle:
            self._rng.shuffle(idx)
        return idx.tolist()

    def __iter__(self) -> Iterator[Dict[str, jnp.ndarray]]:
        buffer: List[EmbodiedSample] = []
        for i in self._iter_indices():
            buffer.append(self.provider(int(i)))
            if len(buffer) == self.batch_size:
                yield self._collate(buffer)
                buffer = []
        if buffer:
            yield self._collate(buffer)

    @staticmethod
    def _stack(values: List[jnp.ndarray]) -> jnp.ndarray:
        """Stack along batch axis after promoting all elements to arrays."""
        return jnp.stack([jnp.asarray(v) for v in values], axis=0)

    def _collate(self, samples: List[EmbodiedSample]) -> Dict[str, jnp.ndarray]:
        """Collate a list of samples into a dict of batched JAX arrays."""
        out: Dict[str, jnp.ndarray] = {}
        for key in ("images", "proprio", "actions", "instruction_tokens", "rewards"):
            vals = [getattr(s, key) for s in samples]
            if all(v is not None for v in vals):
                out[key] = self._stack(vals)
        return out


@dataclass
class MixerSpec:
    """A named loader plus its sampling probability."""
    name: str
    loader: Iterable[Dict[str, jnp.ndarray]]
    weight: float


class MixedModalityBatcher:
    """Round-robin / weighted-mix iterator across multiple loaders."""

    def __init__(self, specs: Sequence[MixerSpec], seed: int = 0):
        weights = jnp.asarray([s.weight for s in specs], dtype=jnp.float32)
        if float(weights.sum()) <= 0:
            raise ValueError("mixer weights must sum > 0")
        self.specs = list(specs)
        self.weights = weights / weights.sum()
        self._rng = np.random.default_rng(seed)
        self._iters = [iter(s.loader) for s in self.specs]

    def _refresh(self, i: int) -> None:
        """Restart iterator `i` after exhaustion."""
        self._iters[i] = iter(self.specs[i].loader)

    def __iter__(self) -> Iterator[Dict[str, object]]:
        return self

    def __next__(self) -> Dict[str, object]:
        choice = int(self._rng.choice(len(self.specs), p=np.asarray(self.weights)))
        try:
            batch = next(self._iters[choice])
        except StopIteration:
            self._refresh(choice)
            batch = next(self._iters[choice])
        return {"source": self.specs[choice].name, "batch": batch}


def action_token_loss(
    predicted_logits: jnp.ndarray, target_bins: jnp.ndarray
) -> jnp.ndarray:
    """Per-axis cross-entropy on discretized action tokens."""
    log_p = jax.nn.log_softmax(predicted_logits, axis=-1)
    one_hot = jax.nn.one_hot(target_bins, predicted_logits.shape[-1])
    return -jnp.mean(jnp.sum(one_hot * log_p, axis=-1))
