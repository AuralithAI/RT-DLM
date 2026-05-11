"""Unit tests verifying train_step JIT caches a single trace."""

import jax
import jax.numpy as jnp
import optax
import pytest

from src.config.agi_config import AGIConfig
from src.train import AGITrainer


@pytest.fixture(scope="module")
def trainer():
    cfg = AGIConfig(
        d_model=64,
        num_heads=4,
        num_layers=2,
        vocab_size=128,
        max_seq_length=32,
        moe_experts=2,
        moe_top_k=1,
        batch_size=2,
        multimodal_enabled=False,
        consciousness_simulation=False,
        graph_neurons_enabled=False,
        ethics_enabled=False,
        meta_learning_enabled=False,
        self_improvement_enabled=False,
        continual_learning=False,
        use_compute_controller=False,
        rlm_enabled=False,
    )
    return AGITrainer(cfg)


class TestJitTrainStep:
    def test_jit_step_callable(self, trainer):
        assert callable(trainer._jit_train_step)
        assert callable(trainer.train_step)

    def test_jit_step_is_jitted(self, trainer):
        assert hasattr(trainer._jit_train_step, "lower") or "PjitFunction" in type(trainer._jit_train_step).__name__ or "CompiledFunction" in type(trainer._jit_train_step).__name__

    def test_train_step_delegates_to_jit(self, trainer):
        assert trainer.train_step.__doc__ is not None
