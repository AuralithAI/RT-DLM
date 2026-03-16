"""
Tests: Self-Critique Module + Synthetic Data + Code Modality Routing
=====================================================================

Covers:
- SelfCritiqueModule (closed-loop generate→critique→revise)
- SyntheticDataGenerator (hard-example mining, critique filtering, shard writing)
- LanguageAwareRetrievalFilter (code detection MLP)
- ComputeState.code_confidence field
- ModuleRegistry.get_code_boosted_costs
- AGIConfig flags (synthetic data, code routing)
"""

import sys
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Ensure src is on path
SRC_DIR = Path(__file__).parent.parent.resolve()
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# =============================================================================
# Test: SelfCritiqueModule
# =============================================================================


class TestSelfCritiqueModule:
    """Tests for the closed-loop self-critique module."""

    def _make_module(self, d_model=64, threshold=0.6, max_rev=2):
        import haiku as hk
        from core.reasoning import SelfCritiqueModule

        def forward(x):
            m = SelfCritiqueModule(
                d_model=d_model,
                quality_threshold=threshold,
                max_revisions=max_rev,
                name="test_scm",
            )
            return m(x, is_training=True)

        return hk.transform(forward)

    def _init_and_apply(self, d_model=64, threshold=0.6, max_rev=2, batch=2):
        model = self._make_module(d_model, threshold, max_rev)
        rng = jax.random.PRNGKey(0)
        x = jax.random.normal(rng, (batch, d_model))
        params = model.init(rng, x)
        result = model.apply(params, rng, x)
        return result

    def test_output_keys(self):
        result = self._init_and_apply()
        expected_keys = {
            "revised_output",
            "quality_scores",
            "num_revisions_applied",
            "process_rewards",
            "total_process_reward",
            "accepted_early",
            "final_quality",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_revised_output_shape(self):
        d_model = 64
        batch = 3
        result = self._init_and_apply(d_model=d_model, batch=batch)
        assert result["revised_output"].shape == (batch, d_model)

    def test_quality_scores_list(self):
        result = self._init_and_apply(max_rev=2)
        # Should have max_rev+1 scores (initial + each revision)
        assert len(result["quality_scores"]) == 3

    def test_quality_scores_bounded(self):
        result = self._init_and_apply()
        for qs in result["quality_scores"]:
            assert float(jnp.min(qs)) >= 0.0
            assert float(jnp.max(qs)) <= 1.0

    def test_process_rewards_list(self):
        result = self._init_and_apply(max_rev=3)
        # process_rewards has one entry per revision step (not the initial)
        assert len(result["process_rewards"]) == 3

    def test_process_reward_values(self):
        result = self._init_and_apply()
        for pr in result["process_rewards"]:
            val = float(pr)
            assert val == 0.0 or val == pytest.approx(0.3, abs=0.01)

    def test_total_process_reward(self):
        result = self._init_and_apply()
        expected = sum(float(r) for r in result["process_rewards"])
        assert float(result["total_process_reward"]) == pytest.approx(expected, abs=1e-5)

    def test_3d_input(self):
        """Module should handle [batch, seq, d_model] input by pooling."""
        import haiku as hk
        from core.reasoning import SelfCritiqueModule

        d_model = 64

        def forward(x):
            m = SelfCritiqueModule(d_model=d_model, name="test_3d")
            return m(x, is_training=True)

        model = hk.transform(forward)
        rng = jax.random.PRNGKey(0)
        x = jax.random.normal(rng, (2, 10, d_model))
        params = model.init(rng, x)
        result = model.apply(params, rng, x)
        assert result["revised_output"].shape == (2, d_model)

    def test_gradients_flow(self):
        """Ensure gradients flow through the critique module."""
        import haiku as hk
        from core.reasoning import SelfCritiqueModule

        d_model = 32

        def forward(x):
            m = SelfCritiqueModule(d_model=d_model, name="grad_test")
            out = m(x, is_training=True)
            return jnp.mean(out["revised_output"])

        model = hk.transform(forward)
        rng = jax.random.PRNGKey(42)
        x = jax.random.normal(rng, (1, d_model))
        params = model.init(rng, x)

        grad_fn = jax.grad(lambda p, r, x: model.apply(p, r, x))
        grads = grad_fn(params, rng, x)

        # At least some gradients should be non-zero
        flat_grads = jax.tree_util.tree_leaves(grads)
        has_nonzero = any(jnp.any(g != 0) for g in flat_grads)
        assert has_nonzero, "All gradients are zero — critique module not differentiable"

    def test_zero_max_revisions(self):
        """With max_revisions=0, should just critique once (no revisions)."""
        result = self._init_and_apply(max_rev=0)
        assert len(result["quality_scores"]) == 1
        assert len(result["process_rewards"]) == 0
        assert result["num_revisions_applied"] == 0


# =============================================================================
# Test: SyntheticDataGenerator
# =============================================================================


class TestSyntheticDataGenerator:
    """Tests for synthetic hard-example mining and shard generation."""

    def _make_config(self, **overrides):
        """Create a minimal config-like object."""
        defaults = {
            "synthetic_data_difficulty_threshold": 0.6,
            "synthetic_data_batch_multiplier": 0.5,
            "synthetic_data_quality_improvement_min": 0.1,
            "synthetic_data_output_dir": tempfile.mkdtemp(),
            "vocab_size": 100,
            "max_seq_length": 32,
        }
        defaults.update(overrides)

        class Config:
            pass

        cfg = Config()
        for k, v in defaults.items():
            setattr(cfg, k, v)
        return cfg

    def _make_seed_batches(self, n=5, batch_size=4, seq_len=16):
        return [
            {
                "input_ids": jnp.ones((batch_size, seq_len), dtype=jnp.int32) * i,
                "targets": jnp.ones((batch_size, seq_len), dtype=jnp.int32) * i,
                "text": jnp.ones((batch_size, seq_len), dtype=jnp.int32) * i,
            }
            for i in range(n)
        ]

    def test_init(self):
        from core.synthetic_data_loop import SyntheticDataGenerator

        cfg = self._make_config()
        gen = SyntheticDataGenerator(cfg)
        assert gen.difficulty_threshold == 0.6
        assert gen.batch_multiplier == 0.5

    def test_generate_hard_examples_low_confidence(self):
        """Model returning low confidence → examples kept."""
        from core.synthetic_data_loop import SyntheticDataGenerator

        cfg = self._make_config(synthetic_data_difficulty_threshold=0.8)
        gen = SyntheticDataGenerator(cfg)

        # Mock model that always returns low confidence
        def model_fn(params, state, rng, inputs):
            batch_size = inputs["text"].shape[0]
            return {
                "confidence": jnp.array([[0.3]]),
                "logits": jnp.zeros((batch_size, 16, 100)),
            }, {}

        batches = self._make_seed_batches(n=5)
        rng = jax.random.PRNGKey(0)

        hard = gen.generate_hard_examples(model_fn, {}, {}, rng, batches)
        assert len(hard) == 5

    def test_generate_hard_examples_high_confidence(self):
        """Model returning high confidence → no hard examples."""
        from core.synthetic_data_loop import SyntheticDataGenerator

        cfg = self._make_config(synthetic_data_difficulty_threshold=0.5)
        gen = SyntheticDataGenerator(cfg)

        def model_fn(params, state, rng, inputs):
            return {"confidence": jnp.array([[0.9]])}, {}

        batches = self._make_seed_batches(n=5)
        rng = jax.random.PRNGKey(0)

        hard = gen.generate_hard_examples(model_fn, {}, {}, rng, batches)
        assert len(hard) == 0

    def test_generate_hard_max_examples(self):
        """Respect max_examples cap."""
        from core.synthetic_data_loop import SyntheticDataGenerator

        cfg = self._make_config(synthetic_data_difficulty_threshold=0.9)
        gen = SyntheticDataGenerator(cfg)

        def model_fn(params, state, rng, inputs):
            return {"confidence": jnp.array([[0.1]])}, {}

        batches = self._make_seed_batches(n=10)
        rng = jax.random.PRNGKey(0)

        hard = gen.generate_hard_examples(model_fn, {}, {}, rng, batches, max_examples=3)
        assert len(hard) == 3

    def test_filter_with_critique(self):
        """Critique filter keeps samples where quality improves."""
        from core.synthetic_data_loop import SyntheticDataGenerator

        cfg = self._make_config(synthetic_data_quality_improvement_min=0.05)
        gen = SyntheticDataGenerator(cfg)

        call_count = 0

        def critique_fn(hidden):
            nonlocal call_count
            call_count += 1
            return {
                "quality_scores": [jnp.array([[0.3]]), jnp.array([[0.5]])],
                "final_quality": jnp.array([[0.5]]),
            }

        batches = self._make_seed_batches(n=3)
        rng = jax.random.PRNGKey(0)

        filtered = gen.filter_with_critique(batches, critique_fn, rng)
        assert len(filtered) == 3

    def test_filter_with_critique_no_improvement(self):
        """Critique filter rejects samples with no quality improvement."""
        from core.synthetic_data_loop import SyntheticDataGenerator

        cfg = self._make_config(synthetic_data_quality_improvement_min=0.5)
        gen = SyntheticDataGenerator(cfg)

        def critique_fn(hidden):
            return {
                "quality_scores": [jnp.array([[0.4]]), jnp.array([[0.41]])],
                "final_quality": jnp.array([[0.41]]),
            }

        batches = self._make_seed_batches(n=3)
        rng = jax.random.PRNGKey(0)

        filtered = gen.filter_with_critique(batches, critique_fn, rng)
        assert len(filtered) == 0

    def test_augment_training_shard(self):
        """Writes a valid .safetensors shard."""
        from core.synthetic_data_loop import SyntheticDataGenerator

        out_dir = tempfile.mkdtemp()
        cfg = self._make_config(synthetic_data_output_dir=out_dir)
        gen = SyntheticDataGenerator(cfg)

        batches = [
            {
                "input_ids": np.ones((4, 16), dtype=np.int32),
                "targets": np.zeros((4, 16), dtype=np.int32),
            },
            {
                "input_ids": np.ones((4, 16), dtype=np.int32) * 2,
                "targets": np.zeros((4, 16), dtype=np.int32),
            },
        ]

        shard_path = gen.augment_training_shard(batches, epoch=1)

        assert shard_path is not None
        assert shard_path.exists()
        assert "epoch1" in shard_path.name

    def test_augment_empty_list(self):
        """Empty examples → no shard written."""
        from core.synthetic_data_loop import SyntheticDataGenerator

        cfg = self._make_config()
        gen = SyntheticDataGenerator(cfg)

        result = gen.augment_training_shard([], epoch=0)
        assert result is None

    def test_run_epoch_end_pipeline(self):
        """Full pipeline: mine → filter → save."""
        from core.synthetic_data_loop import SyntheticDataGenerator

        out_dir = tempfile.mkdtemp()
        cfg = self._make_config(
            synthetic_data_output_dir=out_dir,
            synthetic_data_difficulty_threshold=0.9,
            synthetic_data_batch_multiplier=1.0,
        )
        gen = SyntheticDataGenerator(cfg)

        def model_fn(params, state, rng, inputs):
            return {"confidence": jnp.array([[0.1]])}, {}

        batches = self._make_seed_batches(n=5)
        rng = jax.random.PRNGKey(0)

        shard = gen.run_epoch_end(
            model_fn=model_fn,
            params={},
            state={},
            rng=rng,
            seed_batches=batches,
            epoch=0,
        )

        assert shard is not None
        assert shard.exists()

    def test_run_epoch_end_no_hard_examples(self):
        """Pipeline with confident model → no shard."""
        from core.synthetic_data_loop import SyntheticDataGenerator

        cfg = self._make_config(synthetic_data_difficulty_threshold=0.1)
        gen = SyntheticDataGenerator(cfg)

        def model_fn(params, state, rng, inputs):
            return {"confidence": jnp.array([[0.9]])}, {}

        batches = self._make_seed_batches(n=3)
        rng = jax.random.PRNGKey(0)

        result = gen.run_epoch_end(model_fn, {}, {}, rng, batches, epoch=0)
        assert result is None


# =============================================================================
# Test: LanguageAwareRetrievalFilter
# =============================================================================


class TestLanguageAwareRetrievalFilter:
    """Tests for code vs natural language detection."""

    def test_init(self):
        from core.model.memory_bank import LanguageAwareRetrievalFilter

        filt = LanguageAwareRetrievalFilter(embedding_dim=128)
        assert filt.embedding_dim == 128
        assert filt.hidden_dim == 32  # 128 // 4

    def test_init_params(self):
        from core.model.memory_bank import LanguageAwareRetrievalFilter

        filt = LanguageAwareRetrievalFilter(embedding_dim=64)
        params = filt.init_params()
        assert "w1" in params
        assert "b1" in params
        assert "w2" in params
        assert "b2" in params
        assert params["w1"].shape == (64, 32)
        assert params["w2"].shape == (32, 1)

    def test_is_code_query_returns_float(self):
        from core.model.memory_bank import LanguageAwareRetrievalFilter

        filt = LanguageAwareRetrievalFilter(embedding_dim=32)
        filt.init_params()
        query = np.random.randn(32).astype(np.float32)
        result = filt.is_code_query(query)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_is_code_query_truncation(self):
        """Query longer than embedding_dim should be truncated."""
        from core.model.memory_bank import LanguageAwareRetrievalFilter

        filt = LanguageAwareRetrievalFilter(embedding_dim=16)
        filt.init_params()
        long_query = np.random.randn(100).astype(np.float32)
        result = filt.is_code_query(long_query)
        assert 0.0 <= result <= 1.0

    def test_is_code_query_padding(self):
        """Query shorter than embedding_dim should be padded."""
        from core.model.memory_bank import LanguageAwareRetrievalFilter

        filt = LanguageAwareRetrievalFilter(embedding_dim=64)
        filt.init_params()
        short_query = np.random.randn(10).astype(np.float32)
        result = filt.is_code_query(short_query)
        assert 0.0 <= result <= 1.0

    def test_filter_memories_below_threshold(self):
        """When code_confidence < threshold, memories unchanged."""
        from core.model.memory_bank import LanguageAwareRetrievalFilter

        filt = LanguageAwareRetrievalFilter(embedding_dim=16, threshold=0.99)
        params = filt.init_params()

        class FakeMem:
            def __init__(self, tags):
                self.context_tags = tags

        memories = [FakeMem(["text"]), FakeMem(["code"]), FakeMem(["text"])]
        query = np.zeros(16, dtype=np.float32)

        result, conf = filt.filter_memories(query, memories, params)
        assert len(result) == 3

    def test_filter_memories_code_boost(self):
        """When code_confidence > threshold, code-tagged memories boosted."""
        from core.model.memory_bank import LanguageAwareRetrievalFilter

        filt = LanguageAwareRetrievalFilter(embedding_dim=16, threshold=0.01)
        # Initialize with weights that produce high code_confidence
        rng = np.random.default_rng(42)
        params = filt.init_params(rng)
        params["b2"] = np.array([10.0], dtype=np.float32)

        class FakeMem:
            def __init__(self, tags):
                self.context_tags = tags

        memories = [FakeMem(["text"]), FakeMem(["code"]), FakeMem(["text"])]
        query = np.zeros(16, dtype=np.float32)

        result, conf = filt.filter_memories(query, memories, params)
        assert result[0].context_tags == ["code"]
        assert conf > 0.5

    def test_auto_init_params(self):
        """If no params set, init_params is called automatically."""
        from core.model.memory_bank import LanguageAwareRetrievalFilter

        filt = LanguageAwareRetrievalFilter(embedding_dim=32)
        result = filt.is_code_query(np.zeros(32, dtype=np.float32))
        assert 0.0 <= result <= 1.0

    def test_custom_hidden_dim(self):
        from core.model.memory_bank import LanguageAwareRetrievalFilter

        filt = LanguageAwareRetrievalFilter(embedding_dim=128, hidden_dim=64)
        params = filt.init_params()
        assert params["w1"].shape == (128, 64)
        assert params["w2"].shape == (64, 1)


# =============================================================================
# Test: ComputeState.code_confidence
# =============================================================================


class TestComputeStateCodeConfidence:
    """Tests for the new code_confidence field on ComputeState."""

    def test_default_code_confidence(self):
        from core.agi.compute_controller import ComputeState

        state = ComputeState(
            hidden=jnp.zeros((1, 4, 64)),
            hidden_pooled=jnp.zeros((1, 64)),
            memory_summary=jnp.zeros((1, 64)),
            uncertainty=jnp.array([[0.5]]),
            confidence=jnp.array([[0.5]]),
            budget_remaining=1.0,
            step=0,
            modules_called=[],
            module_outputs=[],
        )
        assert state.code_confidence == 0.0

    def test_custom_code_confidence(self):
        from core.agi.compute_controller import ComputeState

        state = ComputeState(
            hidden=jnp.zeros((1, 4, 64)),
            hidden_pooled=jnp.zeros((1, 64)),
            memory_summary=jnp.zeros((1, 64)),
            uncertainty=jnp.array([[0.5]]),
            confidence=jnp.array([[0.5]]),
            budget_remaining=1.0,
            step=0,
            modules_called=[],
            module_outputs=[],
            code_confidence=0.85,
        )
        assert state.code_confidence == 0.85

    def test_code_confidence_in_tuple(self):
        """Ensure code_confidence is accessible as a tuple field."""
        from core.agi.compute_controller import ComputeState

        state = ComputeState(
            hidden=jnp.zeros((1, 4, 64)),
            hidden_pooled=jnp.zeros((1, 64)),
            memory_summary=jnp.zeros((1, 64)),
            uncertainty=jnp.array([[0.5]]),
            confidence=jnp.array([[0.5]]),
            budget_remaining=1.0,
            step=0,
            modules_called=[],
            module_outputs=[],
            code_confidence=0.42,
        )
        # NamedTuple fields
        assert "code_confidence" in state._fields


# =============================================================================
# Test: ModuleRegistry.get_code_boosted_costs
# =============================================================================


class TestModuleRegistryCodeBoosted:
    """Tests for code-modality routing boost in ModuleRegistry."""

    def test_no_boost_below_threshold(self):
        from core.agi.compute_controller import ModuleRegistry

        registry = ModuleRegistry()
        costs = registry.get_code_boosted_costs(code_confidence=0.3, threshold=0.6)
        for mt, contract in registry.get_all().items():
            assert costs[mt] == contract.base_cost

    def test_boost_above_threshold(self):
        from core.agi.compute_controller import ModuleRegistry, ModuleType

        registry = ModuleRegistry()
        costs = registry.get_code_boosted_costs(
            code_confidence=0.8, threshold=0.6, boost_factor=2.0
        )
        orig = registry.get(ModuleType.MEMORY_RETRIEVAL).base_cost
        assert costs[ModuleType.MEMORY_RETRIEVAL] == pytest.approx(orig / 2.0)

        orig_sym = registry.get(ModuleType.SYMBOLIC_REASONING).base_cost
        assert costs[ModuleType.SYMBOLIC_REASONING] == pytest.approx(orig_sym / 2.0)

        orig_graph = registry.get(ModuleType.GRAPH_REASONING).base_cost
        assert costs[ModuleType.GRAPH_REASONING] == pytest.approx(orig_graph / 2.0)

    def test_non_code_modules_unchanged(self):
        from core.agi.compute_controller import ModuleRegistry, ModuleType

        registry = ModuleRegistry()
        costs = registry.get_code_boosted_costs(code_confidence=0.9, threshold=0.5)
        for mt in [
            ModuleType.QUANTUM_SIMULATION,
            ModuleType.CREATIVE_GENERATION,
            ModuleType.CONSCIOUSNESS,
            ModuleType.OUTPUT_GENERATION,
        ]:
            contract = registry.get(mt)
            if contract:
                assert costs[mt] == contract.base_cost

    def test_custom_boost_factor(self):
        from core.agi.compute_controller import ModuleRegistry, ModuleType

        registry = ModuleRegistry()
        costs = registry.get_code_boosted_costs(
            code_confidence=0.9, threshold=0.1, boost_factor=3.0
        )
        orig = registry.get(ModuleType.MEMORY_RETRIEVAL).base_cost
        assert costs[ModuleType.MEMORY_RETRIEVAL] == pytest.approx(orig / 3.0)


# =============================================================================
# Test: AGIConfig synthetic data + code routing flags
# =============================================================================


class TestAGIConfigSyntheticCodeRouting:
    """Tests for synthetic data and code routing config flags."""

    def test_synthetic_data_defaults(self):
        from config.agi_config import AGIConfig

        cfg = AGIConfig()
        assert cfg.enable_synthetic_data is False
        assert cfg.synthetic_data_difficulty_threshold == 0.6
        assert cfg.synthetic_data_batch_multiplier == 0.2
        assert cfg.synthetic_data_quality_improvement_min == 0.1

    def test_synthetic_data_enabled(self):
        from config.agi_config import AGIConfig

        cfg = AGIConfig(
            enable_synthetic_data=True,
            synthetic_data_difficulty_threshold=0.5,
            synthetic_data_batch_multiplier=0.3,
        )
        assert cfg.enable_synthetic_data is True
        assert cfg.synthetic_data_difficulty_threshold == 0.5
        assert cfg.synthetic_data_batch_multiplier == 0.3

    def test_synthetic_data_invalid_threshold(self):
        from config.agi_config import AGIConfig

        with pytest.raises(AssertionError):
            AGIConfig(
                enable_synthetic_data=True,
                synthetic_data_difficulty_threshold=1.5,
            )

    def test_synthetic_data_invalid_multiplier(self):
        from config.agi_config import AGIConfig

        with pytest.raises(AssertionError):
            AGIConfig(
                enable_synthetic_data=True,
                synthetic_data_batch_multiplier=2.0,
            )

    def test_code_routing_defaults(self):
        from config.agi_config import AGIConfig

        cfg = AGIConfig()
        assert cfg.enable_code_routing is False
        assert cfg.code_routing_threshold == 0.6
        assert cfg.code_routing_boost == 1.5

    def test_code_routing_enabled(self):
        from config.agi_config import AGIConfig

        cfg = AGIConfig(
            enable_code_routing=True,
            code_routing_threshold=0.7,
            code_routing_boost=2.0,
        )
        assert cfg.enable_code_routing is True
        assert cfg.code_routing_threshold == 0.7
        assert cfg.code_routing_boost == 2.0

    def test_code_routing_invalid_threshold(self):
        from config.agi_config import AGIConfig

        with pytest.raises(AssertionError):
            AGIConfig(
                enable_code_routing=True,
                code_routing_threshold=0.0,  # Must be > 0
            )

    def test_code_routing_invalid_boost(self):
        from config.agi_config import AGIConfig

        with pytest.raises(AssertionError):
            AGIConfig(
                enable_code_routing=True,
                code_routing_boost=0.5,  # Must be >= 1.0
            )

    def test_print_summary_includes_synthetic_code_routing(self, capsys):
        from config.agi_config import AGIConfig

        cfg = AGIConfig(
            enable_synthetic_data=True,
            enable_code_routing=True,
        )
        cfg.print_summary()
        output = capsys.readouterr().out
        assert "Synthetic Data" in output
        assert "Code Modality Routing" in output

    def test_config_to_dict_includes_synthetic_code_routing(self):
        from config.agi_config import AGIConfig

        cfg = AGIConfig(
            enable_synthetic_data=True,
            enable_code_routing=True,
        )
        d = cfg.to_dict()
        assert "enable_synthetic_data" in d
        assert "enable_code_routing" in d
        assert d["enable_synthetic_data"] is True
        assert d["enable_code_routing"] is True


# =============================================================================
# Test: Code modality tagging in ShardedDataLoader
# =============================================================================


class TestShardedDataLoaderCodeModality:
    """Tests for code modality detection in shard loading."""

    def test_load_shard_with_modality(self, tmp_path):
        """Shard with modality=4 should flag _is_code_shard."""
        from safetensors.numpy import save_file

        # Create a shard with code modality
        shard_path = tmp_path / "test.safetensors"
        save_file(
            {
                "input_ids": np.ones((8, 16), dtype=np.int32),
                "targets": np.ones((8, 16), dtype=np.int32),
                "modality": np.array([4, 4, 4, 4, 1, 1, 1, 1], dtype=np.int32),
            },
            str(shard_path),
        )

        from train import ShardedDataLoader

        loader = ShardedDataLoader(
            data_dir=str(tmp_path),
            batch_size=4,
            seq_length=16,
        )
        shard = loader._load_shard(shard_path)
        assert "_is_code_shard" in shard
        assert shard["_is_code_shard"][0] == 1

    def test_load_shard_without_modality(self, tmp_path):
        """Shard without modality tensor should not flag code."""
        from safetensors.numpy import save_file

        shard_path = tmp_path / "test.safetensors"
        save_file(
            {
                "input_ids": np.ones((8, 16), dtype=np.int32),
                "targets": np.ones((8, 16), dtype=np.int32),
            },
            str(shard_path),
        )

        from train import ShardedDataLoader

        loader = ShardedDataLoader(
            data_dir=str(tmp_path),
            batch_size=4,
            seq_length=16,
        )
        shard = loader._load_shard(shard_path)
        assert "_is_code_shard" not in shard

    def test_batch_code_confidence(self, tmp_path):
        """Batches from code shard should have code_confidence."""
        from safetensors.numpy import save_file

        shard_path = tmp_path / "test.safetensors"
        save_file(
            {
                "input_ids": np.ones((8, 16), dtype=np.int32),
                "targets": np.ones((8, 16), dtype=np.int32),
                "modality": np.array([4, 4, 4, 4, 4, 4, 4, 4], dtype=np.int32),
            },
            str(shard_path),
        )

        from train import ShardedDataLoader

        loader = ShardedDataLoader(
            data_dir=str(tmp_path),
            batch_size=4,
            seq_length=16,
        )
        shard = loader._load_shard(shard_path)
        batches = loader._create_batches_from_shard(shard)

        assert len(batches) > 0
        assert "code_confidence" in batches[0]
        assert batches[0]["code_confidence"] == 1.0

    def test_batch_no_code_confidence(self, tmp_path):
        """Batches from non-code shard should have code_confidence=0."""
        from safetensors.numpy import save_file

        shard_path = tmp_path / "test.safetensors"
        save_file(
            {
                "input_ids": np.ones((8, 16), dtype=np.int32),
                "targets": np.ones((8, 16), dtype=np.int32),
            },
            str(shard_path),
        )

        from train import ShardedDataLoader

        loader = ShardedDataLoader(
            data_dir=str(tmp_path),
            batch_size=4,
            seq_length=16,
        )
        shard = loader._load_shard(shard_path)
        batches = loader._create_batches_from_shard(shard)

        assert "code_confidence" in batches[0]
        assert batches[0]["code_confidence"] == 0.0


# =============================================================================
# Test: Integration — self-critique wired into RTDLMAGISystem
# =============================================================================


class TestSelfCritiqueIntegration:
    """Integration test: SelfCritiqueModule in the full model forward pass."""

    def test_self_critique_module_in_rtdlm(self):
        """When enable_self_critique=True, RTDLMAGISystem has SelfCritiqueModule."""
        from config.agi_config import AGIConfig
        from rtdlm import RTDLMAGISystem
        import haiku as hk

        config = AGIConfig(
            d_model=64,
            num_heads=4,
            num_layers=2,
            vocab_size=100,
            enable_self_critique=True,
            max_revisions=2,
            use_compute_controller=True,
        )

        def forward(inputs):
            model = RTDLMAGISystem(config)
            return model(inputs, is_training=True)

        model = hk.transform_with_state(forward)
        rng = jax.random.PRNGKey(0)
        dummy_input = {"text": jax.random.randint(rng, (1, 8), 0, 100)}

        params, state = model.init(rng, dummy_input)
        output, new_state = model.apply(params, state, rng, dummy_input)

        # Should have critique outputs
        assert "critique_quality_score" in output or "critique_quality_scores" in output

    def test_self_critique_disabled(self):
        """When enable_self_critique=False, no critique output."""
        from config.agi_config import AGIConfig
        from rtdlm import RTDLMAGISystem
        import haiku as hk

        config = AGIConfig(
            d_model=64,
            num_heads=4,
            num_layers=2,
            vocab_size=100,
            enable_self_critique=False,
            use_compute_controller=True,
        )

        def forward(inputs):
            model = RTDLMAGISystem(config)
            return model(inputs, is_training=True)

        model = hk.transform_with_state(forward)
        rng = jax.random.PRNGKey(0)
        dummy_input = {"text": jax.random.randint(rng, (1, 8), 0, 100)}

        params, state = model.init(rng, dummy_input)
        output, _ = model.apply(params, state, rng, dummy_input)

        # Should NOT have critique outputs
        assert "critique_quality_scores" not in output


# =============================================================================
# Test: SelfCritiqueModule process reward in compute_agi_loss
# =============================================================================


class TestComputeAGILossProcessReward:
    """Tests for process reward loss in compute_agi_loss."""

    def test_process_reward_loss(self):
        """Ensure process_reward_loss is included when present."""
        from rtdlm import compute_agi_loss
        from config.agi_config import AGIConfig

        config = AGIConfig(
            enable_self_critique=True,
            critique_loss_coeff=0.1,
            vocab_size=100,
        )

        logits = jnp.zeros((2, 8, 100))
        targets = jnp.zeros((2, 8), dtype=jnp.int32)
        aux = {
            "critique_quality_score": jnp.array([[0.5], [0.6]]),
            "critique_process_reward": jnp.array(0.3),
        }

        loss = compute_agi_loss(logits, targets, aux_outputs=aux, config=config)
        assert jnp.isfinite(loss)

        # Check loss components
        assert "self_critique_loss" in aux["loss_components"]
        assert "self_critique_process_reward_loss" in aux["loss_components"]

    def test_no_process_reward_without_key(self):
        """No process reward loss when key not in aux_outputs."""
        from rtdlm import compute_agi_loss
        from config.agi_config import AGIConfig

        config = AGIConfig(
            enable_self_critique=True,
            critique_loss_coeff=0.1,
            vocab_size=100,
        )

        logits = jnp.zeros((2, 8, 100))
        targets = jnp.zeros((2, 8), dtype=jnp.int32)
        aux = {
            "critique_quality_score": jnp.array([[0.5], [0.6]]),
        }

        loss = compute_agi_loss(logits, targets, aux_outputs=aux, config=config)
        assert jnp.isfinite(loss)
        assert "self_critique_process_reward_loss" not in aux["loss_components"]
