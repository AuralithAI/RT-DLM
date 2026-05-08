import jax
import jax.numpy as jnp
from typing import Optional, Tuple


class StreamingVideoBuffer:
    """Circular spatiotemporal token buffer for streaming inference.

    Holds the most recent `max_frames` of encoded patch tokens plus an optional
    compressed-history latent produced by an external RLM compressor. Pure JAX
    arrays + Python state — JIT-friendly when frame count is fixed.
    """

    def __init__(
        self,
        d_model: int,
        max_frames: int,
        patches_per_frame: int,
        compressed_size: int = 64,
    ):
        self.d_model = d_model
        self.max_frames = max_frames
        self.patches_per_frame = patches_per_frame
        self.compressed_size = compressed_size
        self.buffer = jnp.zeros((max_frames, patches_per_frame, d_model))
        self.timestamps = jnp.zeros((max_frames,))
        self.length = 0
        self.compressed_history: Optional[jnp.ndarray] = None

    def append(self, frame_tokens: jnp.ndarray, timestamp: float) -> None:
        """frame_tokens: [patches_per_frame, d_model]."""
        if frame_tokens.shape != (self.patches_per_frame, self.d_model):
            raise ValueError("frame_tokens shape mismatch")
        if self.length < self.max_frames:
            self.buffer = self.buffer.at[self.length].set(frame_tokens)
            self.timestamps = self.timestamps.at[self.length].set(timestamp)
            self.length += 1
        else:
            self.buffer = jnp.concatenate(
                [self.buffer[1:], frame_tokens[None, ...]], axis=0
            )
            self.timestamps = jnp.concatenate(
                [self.timestamps[1:], jnp.asarray([timestamp])], axis=0
            )

    def view(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Returns (tokens [F*P, d_model], timestamps [F])."""
        f = self.length
        tokens = self.buffer[:f].reshape(f * self.patches_per_frame, self.d_model)
        return tokens, self.timestamps[:f]

    def attach_compressed_history(self, compressed: jnp.ndarray) -> None:
        if compressed.shape[-1] != self.d_model:
            raise ValueError("compressed history dim mismatch")
        self.compressed_history = compressed

    def full_context(self) -> jnp.ndarray:
        tokens, _ = self.view()
        if self.compressed_history is None:
            return tokens
        return jnp.concatenate([self.compressed_history, tokens], axis=0)

    def reset(self) -> None:
        self.buffer = jnp.zeros_like(self.buffer)
        self.timestamps = jnp.zeros_like(self.timestamps)
        self.length = 0
        self.compressed_history = None
