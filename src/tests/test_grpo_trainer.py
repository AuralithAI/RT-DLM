"""
Tests for GRPO Training Script (train_controller_grpo.py)

Covers:
- RewardComputer: reward components, edge cases
- GRPOTrainer: init, trajectory sampling, scoring, training step
- Integration: smoke test end-to-end, config wiring
- Wiring: GRPOValueHead in RTDLMAGISystem, self-critique in loss
"""

import pytest
import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np

from src.config.agi_config import AGIConfig
from src.train_controller_grpo import (
    RewardComputer,
    GRPOTrainer,
    Trajectory,
    TrajectoryGroup,
    create_dummy_batch,
)

# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def small_config():
    """Minimal config for fast testing."""
    return AGIConfig(
        d_model=64,
        use_grpo=True,
        use_compute_controller=True,
        controller_max_steps=3,
        grpo_num_groups=2,
        grpo_group_size=2,
    )


@pytest.fixture
def reward_computer():
    return RewardComputer()


@pytest.fixture
def trainer(small_config):
    return GRPOTrainer(
        config=small_config,
        num_groups=2,
        group_size=2,
        learning_rate=1e-3,
    )


@pytest.fixture
def rng():
    return jax.random.PRNGKey(42)


# =========================================================================
# RewardComputer Tests
# =========================================================================


class TestRewardComputer:
    """Tests for trajectory reward computation."""

    def test_efficiency_reward(self, reward_computer):
        """Low budget usage earns efficiency bonus."""
        traj = Trajectory(
            hidden_states=jnp.zeros(64),
            log_prob=0.0,
            reward=0.0,
            modules_called=[],
            steps_taken=2,
            budget_used=0.3,  # < 0.6 threshold
        )
        reward = reward_computer.compute_reward(traj)
        assert reward >= reward_computer.efficiency_weight

    def test_no_efficiency_reward_high_budget(self, reward_computer):
        """High budget usage gets no efficiency bonus."""
        traj = Trajectory(
            hidden_states=jnp.zeros(64),
            log_prob=0.0,
            reward=0.0,
            modules_called=[],
            steps_taken=5,
            budget_used=0.9,  # > 0.6 threshold
        )
        reward = reward_computer.compute_reward(traj)
        assert reward < reward_computer.efficiency_weight

    def test_unnecessary_module_penalty(self, reward_computer):
        """Using too many modules incurs penalty."""
        from src.core.agi.compute_controller import ModuleType

        traj = Trajectory(
            hidden_states=jnp.zeros(64),
            log_prob=0.0,
            reward=0.0,
            modules_called=[
                ModuleType.MEMORY_RETRIEVAL,
                ModuleType.GRAPH_REASONING,
                ModuleType.SYMBOLIC_REASONING,
                ModuleType.PROBABILISTIC,
                ModuleType.QUANTUM_SIMULATION,
            ],
            steps_taken=5,
            budget_used=0.9,
        )
        reward = reward_computer.compute_reward(traj)
        # 5 unique modules, penalty for 2 unnecessary
        assert reward < 0.0

    def test_correctness_reward_with_matching_answer(self, reward_computer):
        """Matching majority answer earns correctness reward."""
        answer = jnp.ones(64) * 0.5
        majority = jnp.ones(64) * 0.5  # Same as answer
        traj = Trajectory(
            hidden_states=jnp.zeros(64),
            log_prob=0.0,
            reward=0.0,
            modules_called=[],
            steps_taken=2,
            budget_used=0.3,
        )
        reward = reward_computer.compute_reward(traj, majority, answer)
        assert reward >= reward_computer.correctness_weight

    def test_correctness_no_reward_different_answer(self, reward_computer):
        """Different answer from majority gets no correctness reward."""
        answer = jnp.ones(64) * 0.5
        majority = -jnp.ones(64) * 0.5  # Opposite
        traj = Trajectory(
            hidden_states=jnp.zeros(64),
            log_prob=0.0,
            reward=0.0,
            modules_called=[],
            steps_taken=2,
            budget_used=0.9,  # Also no efficiency
        )
        reward = reward_computer.compute_reward(traj, majority, answer)
        assert reward < reward_computer.correctness_weight

    def test_confidence_bonus(self, reward_computer):
        """High confidence on correct answer earns bonus."""
        answer = jnp.ones(64) * 0.5
        majority = jnp.ones(64) * 0.5
        traj = Trajectory(
            hidden_states=jnp.zeros(64),
            log_prob=0.0,
            reward=0.0,
            modules_called=[],
            steps_taken=2,
            budget_used=0.3,
            value_estimate=0.95,  # High confidence
        )
        reward = reward_computer.compute_reward(traj, majority, answer)
        assert reward >= (
            reward_computer.correctness_weight
            + reward_computer.confidence_bonus
            + reward_computer.efficiency_weight
        )

    def test_no_majority_answer(self, reward_computer):
        """Without majority answer, only efficiency counts."""
        traj = Trajectory(
            hidden_states=jnp.zeros(64),
            log_prob=0.0,
            reward=0.0,
            modules_called=[],
            steps_taken=2,
            budget_used=0.3,
        )
        reward = reward_computer.compute_reward(traj, None, None)
        assert reward == reward_computer.efficiency_weight


# =========================================================================
# GRPOTrainer Tests
# =========================================================================


class TestGRPOTrainer:
    """Tests for the GRPO training pipeline."""

    def test_trainer_init(self, trainer, small_config):
        """Trainer initialises with correct config."""
        assert trainer.num_groups == 2
        assert trainer.group_size == 2
        assert trainer.config.use_grpo is True
        assert trainer.controller_fn is not None

    def test_init_params(self, trainer, rng):
        """Params and opt_state are initialised."""
        dummy = jax.random.normal(rng, (2, 8, 64))
        params, opt_state = trainer.init_params(rng, dummy)
        assert params is not None
        assert opt_state is not None
        param_count = sum(p.size for p in jax.tree_util.tree_leaves(params))
        assert param_count > 0

    def test_sample_trajectories(self, trainer, rng):
        """Trajectory sampling produces correct structure."""
        dummy = jax.random.normal(rng, (2, 8, 64))
        rng, init_rng, sample_rng = jax.random.split(rng, 3)
        params, _ = trainer.init_params(init_rng, dummy)

        groups = trainer.sample_trajectories(params, sample_rng, dummy)
        assert len(groups) == trainer.num_groups
        for group in groups:
            assert isinstance(group, TrajectoryGroup)
            assert len(group.trajectories) == trainer.group_size
            for traj in group.trajectories:
                assert isinstance(traj, Trajectory)
                assert traj.hidden_states.shape == (64,)
                assert isinstance(traj.reward, float)

    def test_score_trajectories(self, trainer, rng):
        """Scoring produces correct shaped advantages/returns."""
        dummy = jax.random.normal(rng, (2, 8, 64))
        rng, init_rng, sample_rng = jax.random.split(rng, 3)
        params, _ = trainer.init_params(init_rng, dummy)

        groups = trainer.sample_trajectories(params, sample_rng, dummy)
        advantages, returns, values = trainer.score_trajectories(groups)

        total = trainer.num_groups * trainer.group_size
        assert advantages.shape == (total,)
        assert returns.shape == (total,)
        assert values.shape == (total,)
        # Advantages should be mean-zero within each group
        grouped = advantages.reshape(trainer.num_groups, trainer.group_size)
        for g in range(trainer.num_groups):
            assert abs(float(grouped[g].mean())) < 1.0  # Approximately centered

    def test_grpo_train_step(self, trainer, rng):
        """Single training step executes and returns metrics."""
        dummy = jax.random.normal(rng, (2, 8, 64))
        rng, init_rng, step_rng = jax.random.split(rng, 3)
        params, opt_state = trainer.init_params(init_rng, dummy)

        new_params, new_opt_state, metrics = trainer.grpo_train_step(
            params, opt_state, step_rng, dummy
        )

        assert new_params is not None
        assert new_opt_state is not None
        assert "grpo_total_loss" in metrics
        assert "mean_reward" in metrics
        assert "value_loss" in metrics
        assert "num_trajectories" in metrics
        assert metrics["num_trajectories"] == trainer.num_groups * trainer.group_size

    def test_params_change_after_step(self, trainer, rng):
        """Parameters should change after a training step."""
        dummy = jax.random.normal(rng, (2, 8, 64))
        rng, init_rng, step_rng = jax.random.split(rng, 3)
        params, opt_state = trainer.init_params(init_rng, dummy)

        old_leaves = jax.tree_util.tree_leaves(params)
        new_params, _, _ = trainer.grpo_train_step(params, opt_state, step_rng, dummy)
        new_leaves = jax.tree_util.tree_leaves(new_params)

        # At least some parameters should have changed
        changed = sum(1 for o, n in zip(old_leaves, new_leaves) if not jnp.allclose(o, n))
        assert changed > 0

    def test_two_training_steps(self, trainer, rng):
        """Two consecutive steps execute without error."""
        dummy = jax.random.normal(rng, (2, 8, 64))
        rng, init_rng = jax.random.split(rng)
        params, opt_state = trainer.init_params(init_rng, dummy)

        for _ in range(2):
            rng, step_rng, batch_rng = jax.random.split(rng, 3)
            batch = jax.random.normal(batch_rng, (2, 8, 64))
            params, opt_state, metrics = trainer.grpo_train_step(params, opt_state, step_rng, batch)
        assert metrics["mean_reward"] is not None


# =========================================================================
# Dummy Batch Tests
# =========================================================================


class TestDummyBatch:
    """Tests for create_dummy_batch utility."""

    def test_shape(self, rng):
        batch = create_dummy_batch(4, 16, 128, rng)
        assert batch.shape == (4, 16, 128)

    def test_dtype(self, rng):
        batch = create_dummy_batch(2, 8, 64, rng)
        assert batch.dtype == jnp.float32

    def test_deterministic(self, rng):
        b1 = create_dummy_batch(2, 8, 64, rng)
        b2 = create_dummy_batch(2, 8, 64, rng)
        np.testing.assert_array_equal(b1, b2)


# =========================================================================
# Wiring Integration Tests
# =========================================================================


class TestRTDLMWiring:
    """Tests that GRPO modules are correctly wired into RTDLMAGISystem."""

    def test_grpo_value_head_in_agi_system(self):
        """GRPOValueHead is created when use_grpo=True."""
        config = AGIConfig(d_model=64, use_grpo=True, use_compute_controller=True)

        def _forward(x):
            system = __import__("src.rtdlm", fromlist=["RTDLMAGISystem"]).RTDLMAGISystem(config)
            assert hasattr(system, "grpo_value_head")
            return system.grpo_value_head(x, is_training=False)

        fn = hk.transform_with_state(_forward)
        rng = jax.random.PRNGKey(0)
        dummy = jax.random.normal(rng, (2, 64))
        params, state = fn.init(rng, dummy)
        output, _ = fn.apply(params, state, rng, dummy)
        assert output.shape == (2, 1)

    def test_verify_reflect_in_reasoning_engine(self):
        """ReasoningEngine uses VerifyReflectReasoning when enabled."""
        config = AGIConfig(
            d_model=64,
            enable_verify_reflect=True,
            max_verify_steps=2,
            verify_confidence_threshold=0.85,
        )

        from src.core.reasoning import ReasoningEngine, VerifyReflectReasoning

        def _forward(query, context):
            engine = ReasoningEngine(config)
            assert isinstance(engine.chain_of_thought, VerifyReflectReasoning)
            return engine.chain_of_thought(query, context)

        fn = hk.transform_with_state(_forward)
        rng = jax.random.PRNGKey(0)
        q = jax.random.normal(rng, (2, 4, 64))
        c = jax.random.normal(rng, (2, 8, 64))
        params, state = fn.init(rng, q, c)
        result, _ = fn.apply(params, state, rng, q, c)
        assert "verification_scores" in result
        assert "final_answer" in result

    def test_verify_reflect_disabled_uses_cot(self):
        """ReasoningEngine uses plain CoT when verify_reflect is disabled."""
        config = AGIConfig(d_model=64, enable_verify_reflect=False)

        from src.core.reasoning import (
            ReasoningEngine,
            ChainOfThoughtReasoning,
            VerifyReflectReasoning,
        )

        def _forward(query, context):
            engine = ReasoningEngine(config)
            assert isinstance(engine.chain_of_thought, ChainOfThoughtReasoning)
            assert not isinstance(engine.chain_of_thought, VerifyReflectReasoning)
            return engine.chain_of_thought(query, context)

        fn = hk.transform_with_state(_forward)
        rng = jax.random.PRNGKey(0)
        q = jax.random.normal(rng, (2, 4, 64))
        c = jax.random.normal(rng, (2, 8, 64))
        params, state = fn.init(rng, q, c)
        result, _ = fn.apply(params, state, rng, q, c)
        assert "final_answer" in result
        # Standard CoT doesn't have verification_scores
        assert "verification_scores" not in result

    def test_self_critique_in_agi_system(self):
        """SelfCritiqueHead is created when enable_self_critique=True."""
        config = AGIConfig(
            d_model=64,
            enable_self_critique=True,
            use_compute_controller=True,
        )

        def _forward(x):
            system = __import__("src.rtdlm", fromlist=["RTDLMAGISystem"]).RTDLMAGISystem(config)
            assert hasattr(system, "self_critique_head")
            result = system.self_critique_head(x)
            return result["quality_score"]

        fn = hk.transform_with_state(_forward)
        rng = jax.random.PRNGKey(0)
        dummy = jax.random.normal(rng, (2, 64))
        params, state = fn.init(rng, dummy)
        output, _ = fn.apply(params, state, rng, dummy)
        assert output.shape == (2, 1)
        assert jnp.all(output >= 0) and jnp.all(output <= 1)

    def test_self_critique_loss_in_compute_agi_loss(self):
        """Self-critique loss is added to compute_agi_loss when enabled."""
        config = AGIConfig(
            d_model=64,
            enable_self_critique=True,
            critique_loss_coeff=0.1,
            vocab_size=100,
        )
        from src.rtdlm import compute_agi_loss

        logits = jax.random.normal(jax.random.PRNGKey(0), (2, 10, 100))
        targets = jax.random.randint(jax.random.PRNGKey(1), (2, 10), 0, 100)
        aux_outputs = {
            "critique_quality_score": jnp.array([[0.3], [0.7]]),
        }

        loss = compute_agi_loss(logits, targets, aux_outputs, config)
        assert "self_critique_loss" in aux_outputs.get("loss_components", {})


class TestKVCacheWiring:
    """Tests that KV cache is wired into GroupedQueryAttention."""

    def test_gqa_accepts_kv_cache_params(self):
        """GQA __call__ accepts kv_cache, layer_id, prefix_id kwargs."""
        from src.core.model.advanced_attention import (
            GroupedQueryAttention,
            KVPrefixCache,
        )

        def _forward(x):
            gqa = GroupedQueryAttention(
                num_heads=4,
                num_kv_heads=2,
                head_dim=16,
                use_rope=False,
            )
            # Without cache
            out1, _ = gqa(x, is_training=False)

            # With cache (store prefix)
            cache = KVPrefixCache(
                num_layers=1,
                max_prefix_len=32,
                num_kv_heads=2,
                head_dim=16,
                max_entries=4,
            )
            out2, _ = gqa(x, is_training=False, kv_cache=cache, layer_id=0, prefix_id="test_prefix")
            return out1, out2

        fn = hk.transform(_forward)
        rng = jax.random.PRNGKey(0)
        x = jax.random.normal(rng, (2, 8, 64))
        params = fn.init(rng, x)
        out1, out2 = fn.apply(params, rng, x)
        assert out1.shape == out2.shape == (2, 8, 64)

    def test_kv_cache_hit_concatenates(self):
        """Second call with same prefix_id uses cached K/V."""
        from src.core.model.advanced_attention import (
            GroupedQueryAttention,
            KVPrefixCache,
        )

        cache = KVPrefixCache(
            num_layers=1,
            max_prefix_len=32,
            num_kv_heads=2,
            head_dim=16,
            max_entries=4,
        )

        def _forward(x, use_cache):
            gqa = GroupedQueryAttention(
                num_heads=4,
                num_kv_heads=2,
                head_dim=16,
                use_rope=False,
            )
            if use_cache:
                return gqa(x, is_training=False, kv_cache=cache, layer_id=0, prefix_id="prompt_1")
            return gqa(x, is_training=False)

        fn = hk.transform(_forward)
        rng = jax.random.PRNGKey(0)
        x = jax.random.normal(rng, (2, 8, 64))
        params = fn.init(rng, x, False)

        # First call — cache miss, stores prefix
        _ = fn.apply(params, rng, x, True)
        assert cache.has("prompt_1")

        # Second call — cache hit
        out, _ = fn.apply(params, rng, x, True)
        # Output seq_len still 8 but attention was over 8 + 8 = 16
        assert out.shape == (2, 8, 64)
        stats = cache.get_stats()
        # After two calls with same prefix, access_counts > 1
        assert stats["num_entries"] >= 1
        assert "prompt_1" in stats["access_counts"]
        assert stats["access_counts"]["prompt_1"] >= 2  # init + 2nd lookup
