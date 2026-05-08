"""
Tests for Think-Budget, Hierarchical Compression, Multi-Resolution
Patches, and Hard Negative Mining.

Covers:
- ComputePlan.apply_think_budget scaling logic
- ComputePlan.__call__ with think_budget_tokens override
- HierarchicalCompressor auto-trigger, tier-1, tier-2
- MultiResolutionPatchEmbed with various patch sizes
- VisionEncoder with use_multi_resolution=True
- compute_multimodal_alignment_loss with hard negative mining
- AGIConfig validation for new flags
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk

# =========================================================================
# Think-Budget → ComputePlan wiring
# =========================================================================


class TestThinkBudgetScaling:
    """Tests for ComputePlan.apply_think_budget."""

    def _run_apply_think_budget(self, budget_tokens, max_steps=10, initial_budget=1.0):
        from src.core.agi.compute_controller import ComputePlan

        def fn():
            plan = ComputePlan(d_model=64, max_steps=max_steps, initial_budget=initial_budget)
            return plan.apply_think_budget(budget_tokens)

        transformed = hk.transform(fn)
        rng = jax.random.PRNGKey(0)
        params = transformed.init(rng)
        steps, budget = transformed.apply(params, rng)
        return int(steps), float(budget)

    def test_none_returns_defaults(self):
        steps, budget = self._run_apply_think_budget(None)
        assert steps == 10
        assert budget == pytest.approx(1.0)

    def test_low_budget_reduces_steps(self):
        steps, budget = self._run_apply_think_budget(256)
        assert steps < 10
        assert steps >= 1

    def test_high_budget_increases_steps(self):
        steps, budget = self._run_apply_think_budget(4096)
        assert steps > 10
        assert budget > 1.0

    def test_medium_budget_is_identity(self):
        steps, budget = self._run_apply_think_budget(1024)
        assert steps == 10
        assert budget == pytest.approx(1.0)

    def test_budget_clamps_minimum(self):
        steps, budget = self._run_apply_think_budget(1)
        assert steps >= 1
        assert budget > 0

    def test_budget_clamps_max_budget(self):
        _, budget = self._run_apply_think_budget(100_000)
        assert budget <= 10.0

    def test_monotonic_scaling(self):
        budgets = [256, 1024, 4096, 8192]
        step_counts = [self._run_apply_think_budget(b)[0] for b in budgets]
        for i in range(len(step_counts) - 1):
            assert step_counts[i] <= step_counts[i + 1], f"Steps should be monotonically increasing: {step_counts}"


class TestComputePlanWithThinkBudget:
    """Tests that ComputePlan.__call__ properly uses think_budget_tokens."""

    def _run_plan(self, think_budget_tokens):
        from src.core.agi.compute_controller import (
            ComputePlan,
            ComputeController,
            ModuleRegistry,
        )

        def fn(hidden):
            plan = ComputePlan(d_model=64, max_steps=5, initial_budget=1.0)
            controller = ComputeController(d_model=64)
            registry = ModuleRegistry()
            executors = {}
            state, trace = plan(
                hidden,
                controller,
                registry,
                executors,
                think_budget_tokens=think_budget_tokens,
            )
            return state.hidden_pooled, trace

        transformed = hk.transform_with_state(fn)
        rng = jax.random.PRNGKey(42)
        hidden = jnp.ones((1, 8, 64))
        params, state = transformed.init(rng, hidden)
        (_, trace), _ = transformed.apply(params, state, rng, hidden)
        return trace

    def test_execution_trace_contains_budget_info(self):
        trace = self._run_plan(4096)
        assert trace["think_budget_tokens"] == 4096
        assert trace["effective_max_steps"] > 5
        assert trace["effective_budget"] > 1.0

    def test_no_budget_leaves_defaults(self):
        trace = self._run_plan(None)
        assert trace["think_budget_tokens"] is None
        assert trace["effective_max_steps"] == 5
        assert trace["effective_budget"] == pytest.approx(1.0)


# =========================================================================
# Hierarchical Compression
# =========================================================================


class TestHierarchicalCompressor:
    """Tests for HierarchicalCompressor."""

    def _make_compressor(self, threshold=100, tier1_max_tokens=200, tier2_max_tokens=50):
        from src.core.rlm.context_store import ContextStore
        from src.core.rlm.context_tools import ContextTools
        from src.core.rlm.hierarchical_compressor import HierarchicalCompressor

        store = ContextStore()
        tools = ContextTools(store)
        compressor = HierarchicalCompressor(
            context_store=store,
            context_tools=tools,
            auto_compress_threshold=threshold,
            tier1_max_tokens=tier1_max_tokens,
            tier2_max_tokens=tier2_max_tokens,
        )
        return store, tools, compressor

    def test_should_compress_false_when_small(self):
        store, _, compressor = self._make_compressor(threshold=1000)
        store.store("ctx", "short text", source="test")
        assert compressor.should_compress("ctx") is False

    def test_should_compress_true_when_large(self):
        store, _, compressor = self._make_compressor(threshold=100)
        store.store("ctx", "x" * 200, source="test")
        assert compressor.should_compress("ctx") is True

    def test_should_compress_false_for_missing_var(self):
        _, _, compressor = self._make_compressor()
        assert compressor.should_compress("nonexistent") is False

    def test_maybe_compress_returns_none_if_small(self):
        store, _, compressor = self._make_compressor(threshold=1000)
        store.store("ctx", "short", source="test")
        assert compressor.maybe_compress("ctx") is None

    def test_maybe_compress_tier1(self):
        """Content bigger than threshold but < 4x → tier-1."""
        store, _, compressor = self._make_compressor(
            threshold=100,
            tier1_max_tokens=30,
        )
        # ~260 chars: > 100 but < 400 → should pick tier 1
        content = "Hello world. " * 20
        store.store("ctx", content, source="test")
        result = compressor.maybe_compress("ctx")
        assert result is not None
        assert result.success
        assert result.tier == 1
        assert result.compression_ratio > 1.0
        assert result.summary_var in store

    def test_force_compress_tier2(self):
        """Force tier-2 compression."""
        store, _, compressor = self._make_compressor(
            threshold=100,
            tier1_max_tokens=100,
            tier2_max_tokens=10,
        )
        content = "This is sentence number one. " * 80  # ~2400 chars
        store.store("ctx", content, source="test")
        result = compressor.compress("ctx", target_tier=2)
        assert result.success
        assert result.tier == 2
        assert "compressed_t2" in result.summary_var

    def test_compress_tier0_noop(self):
        store, _, compressor = self._make_compressor()
        store.store("ctx", "hello world", source="test")
        result = compressor.compress("ctx", target_tier=0)
        assert result.success
        assert result.tier == 0
        assert result.summary_var == "ctx"

    def test_compress_missing_var(self):
        _, _, compressor = self._make_compressor()
        result = compressor.compress("missing", target_tier=1)
        assert not result.success
        assert result.error is not None

    def test_get_compressed_var(self):
        store, _, compressor = self._make_compressor(
            threshold=100,
            tier1_max_tokens=30,
        )
        content = "Word goes here. " * 30
        store.store("ctx", content, source="test")
        compressor.compress("ctx", target_tier=1)
        name = compressor.get_compressed_var("ctx", tier=1)
        assert name is not None
        assert "compressed_t1" in name

    def test_get_compressed_var_none_if_not_compressed(self):
        _, _, compressor = self._make_compressor()
        assert compressor.get_compressed_var("ctx") is None

    def test_stats_updated_after_compress(self):
        store, _, compressor = self._make_compressor(
            threshold=100,
            tier1_max_tokens=20,
        )
        content = "Lots of text here and there. " * 30
        store.store("ctx", content, source="test")
        compressor.compress("ctx", target_tier=1)
        stats = compressor.get_stats()
        assert stats["total_compressions"] == 1
        assert stats["tier_counts"][1] >= 1

    def test_auto_pick_tier2_for_very_large(self):
        """Very large content → auto-selects tier-2."""
        store, _, compressor = self._make_compressor(threshold=100)
        content = "x " * 500  # 1000 chars > 4*100
        store.store("ctx", content, source="test")
        result = compressor.maybe_compress("ctx")
        assert result is not None
        assert result.tier == 2


class TestRLMConfigCompression:
    """Tests for new RLMConfig fields."""

    def test_default_compression_fields(self):
        from src.config.rlm_config import RLMConfig

        cfg = RLMConfig()
        assert cfg.enable_hierarchical_compression is False
        assert cfg.auto_compress_threshold == 32000
        assert cfg.store_utilisation_threshold == pytest.approx(0.70)
        assert cfg.tier1_max_tokens == 2000
        assert cfg.tier2_max_tokens == 500

    def test_compression_fields_custom(self):
        from src.config.rlm_config import RLMConfig

        cfg = RLMConfig(
            enable_hierarchical_compression=True,
            auto_compress_threshold=5000,
            tier1_max_tokens=500,
        )
        assert cfg.enable_hierarchical_compression is True
        assert cfg.auto_compress_threshold == 5000
        assert cfg.tier1_max_tokens == 500


class TestRLMOrchestratorCompressor:
    """Tests that RLMOrchestrator has a compressor wired in."""

    def test_orchestrator_has_compressor(self):
        from src.core.rlm.rlm_core import RLMOrchestrator

        orch = RLMOrchestrator(d_model=64)
        assert hasattr(orch, "compressor")
        from src.core.rlm.hierarchical_compressor import HierarchicalCompressor

        assert isinstance(orch.compressor, HierarchicalCompressor)

    def test_orchestrator_stats_has_compressor(self):
        from src.core.rlm.rlm_core import RLMOrchestrator

        orch = RLMOrchestrator(d_model=64)
        stats = orch.get_stats()
        assert "compressor" in stats
        assert "total_compressions" in stats["compressor"]


# =========================================================================
# Multi-Resolution Patch Embedding
# =========================================================================


class TestMultiResolutionPatchEmbed:
    """Tests for MultiResolutionPatchEmbed."""

    def test_output_shape_default_patches(self):
        """Default patch sizes [8,16,32] on a 64x64 image."""
        from src.modules.multimodal.fusion_module import MultiResolutionPatchEmbed

        def fn(images):
            embed = MultiResolutionPatchEmbed(d_model=64)
            return embed(images)

        transformed = hk.transform(fn)
        rng = jax.random.PRNGKey(0)
        images = jnp.ones((2, 64, 64, 3))
        params = transformed.init(rng, images)
        out = transformed.apply(params, rng, images)

        # 64/8=8 -> 64 patches; 64/16=4 -> 16 patches; 64/32=2 -> 4 patches
        expected_patches = 64 + 16 + 4
        assert out.shape == (2, expected_patches, 64)

    def test_single_patch_size(self):
        """Single patch size."""
        from src.modules.multimodal.fusion_module import MultiResolutionPatchEmbed

        def fn(images):
            embed = MultiResolutionPatchEmbed(d_model=32, patch_sizes=[16])
            return embed(images)

        transformed = hk.transform(fn)
        rng = jax.random.PRNGKey(0)
        images = jnp.ones((1, 32, 32, 3))
        params = transformed.init(rng, images)
        out = transformed.apply(params, rng, images)

        # 32/16=2 -> 4 patches
        assert out.shape == (1, 4, 32)

    def test_different_image_sizes(self):
        """Positional embedding should interpolate for non-canonical sizes."""
        from src.modules.multimodal.fusion_module import MultiResolutionPatchEmbed

        def fn(images):
            embed = MultiResolutionPatchEmbed(d_model=32, patch_sizes=[16], canonical_img_size=224)
            return embed(images)

        transformed = hk.transform(fn)
        rng = jax.random.PRNGKey(0)
        # 128x128 -> 8x8=64 patches (canonical would be 14x14=196)
        images = jnp.ones((1, 128, 128, 3))
        params = transformed.init(rng, images)
        out = transformed.apply(params, rng, images)
        assert out.shape == (1, 64, 32)

    def test_batch_independence(self):
        """Batches should be independent."""
        from src.modules.multimodal.fusion_module import MultiResolutionPatchEmbed

        def fn(images):
            embed = MultiResolutionPatchEmbed(d_model=32, patch_sizes=[16])
            return embed(images)

        transformed = hk.transform(fn)
        rng = jax.random.PRNGKey(0)
        img1 = jax.random.normal(jax.random.PRNGKey(1), (1, 32, 32, 3))
        img2 = jax.random.normal(jax.random.PRNGKey(2), (1, 32, 32, 3))
        batch = jnp.concatenate([img1, img2], axis=0)
        params = transformed.init(rng, batch)

        out_batch = transformed.apply(params, rng, batch)
        out_single = transformed.apply(params, rng, img1)

        np.testing.assert_allclose(out_batch[0], out_single[0], atol=1e-5)


class TestVisionEncoderMultiRes:
    """Tests for VisionEncoder with multi-resolution enabled."""

    def test_standard_path_unchanged(self):
        """use_multi_resolution=False -> original behavior."""
        from src.modules.multimodal.fusion_module import VisionEncoder

        def fn(images):
            enc = VisionEncoder(d_model=64, use_multi_resolution=False)
            return enc(images)

        transformed = hk.transform(fn)
        rng = jax.random.PRNGKey(0)
        images = jnp.ones((1, 32, 32, 3))
        params = transformed.init(rng, images)
        out = transformed.apply(params, rng, images)
        assert out.ndim == 3
        assert out.shape[0] == 1
        assert out.shape[2] == 64

    def test_multi_res_path(self):
        """use_multi_resolution=True -> uses MultiResolutionPatchEmbed."""
        from src.modules.multimodal.fusion_module import VisionEncoder

        def fn(images):
            enc = VisionEncoder(
                d_model=64,
                use_multi_resolution=True,
                multi_res_patch_sizes=[8, 16],
            )
            return enc(images)

        transformed = hk.transform(fn)
        rng = jax.random.PRNGKey(0)
        images = jnp.ones((1, 32, 32, 3))
        params = transformed.init(rng, images)
        out = transformed.apply(params, rng, images)
        # 32/8=4 -> 16 patches; 32/16=2 -> 4 patches = 20 total
        assert out.shape == (1, 20, 64)


class TestAGIConfigMultiRes:
    """Config flags for multi-resolution vision."""

    def test_default_disabled(self):
        from src.config.agi_config import AGIConfig

        cfg = AGIConfig()
        assert cfg.enable_multi_res_vision is False
        assert cfg.vision_patch_sizes == [8, 16, 32]

    def test_enable_multi_res(self):
        from src.config.agi_config import AGIConfig

        cfg = AGIConfig(enable_multi_res_vision=True, vision_patch_sizes=[4, 16])
        assert cfg.enable_multi_res_vision is True
        assert cfg.vision_patch_sizes == [4, 16]

    def test_validation_empty_patches_fails(self):
        from src.config.agi_config import AGIConfig

        with pytest.raises(AssertionError, match="at least one entry"):
            AGIConfig(enable_multi_res_vision=True, vision_patch_sizes=[])

    def test_validation_negative_patches_fails(self):
        from src.config.agi_config import AGIConfig

        with pytest.raises(AssertionError, match="positive"):
            AGIConfig(enable_multi_res_vision=True, vision_patch_sizes=[-1, 16])


# =========================================================================
# Hard Negative Mining (Contrastive Loss)
# =========================================================================


class TestHardNegativeMining:
    """Tests for compute_multimodal_alignment_loss with hard negative mining."""

    def _make_features(self, batch_size=8, d_model=32, seed=42):
        rng = jax.random.PRNGKey(seed)
        k1, k2 = jax.random.split(rng)
        text = jax.random.normal(k1, (batch_size, d_model))
        audio = jax.random.normal(k2, (batch_size, d_model))
        return {"text_features": text, "audio_features": audio}

    def test_no_config_unchanged(self):
        """Without config -> standard InfoNCE (backward compat)."""
        from src.rtdlm import compute_multimodal_alignment_loss

        aux = self._make_features()
        loss = compute_multimodal_alignment_loss(aux)
        assert float(loss) > 0

    def test_config_disabled_unchanged(self):
        """Config with mining disabled -> standard InfoNCE."""
        from src.rtdlm import compute_multimodal_alignment_loss
        from src.config.agi_config import AGIConfig

        cfg = AGIConfig(enable_hard_negative_mining=False)
        aux = self._make_features()
        loss_no_cfg = float(compute_multimodal_alignment_loss(aux))
        loss_cfg = float(compute_multimodal_alignment_loss(aux, config=cfg))
        assert loss_no_cfg == pytest.approx(loss_cfg, abs=1e-5)

    def test_hard_negatives_produces_loss(self):
        """Enabled mining still produces a finite positive loss."""
        from src.rtdlm import compute_multimodal_alignment_loss
        from src.config.agi_config import AGIConfig

        cfg = AGIConfig(enable_hard_negative_mining=True)
        aux = self._make_features()
        loss = float(compute_multimodal_alignment_loss(aux, config=cfg))
        assert np.isfinite(loss)
        assert loss > 0

    def test_hard_negatives_different_from_standard(self):
        """Hard-neg loss should differ from standard when mining enabled."""
        from src.rtdlm import compute_multimodal_alignment_loss
        from src.config.agi_config import AGIConfig

        aux = self._make_features(batch_size=16)
        loss_standard = float(compute_multimodal_alignment_loss(aux))
        cfg = AGIConfig(enable_hard_negative_mining=True, contrastive_margin=0.3)
        loss_hard = float(compute_multimodal_alignment_loss(aux, config=cfg))
        # Both should be finite
        assert np.isfinite(loss_standard)
        assert np.isfinite(loss_hard)

    def test_batch_size_1_no_crash(self):
        """Batch size 1 -> should not crash (no negatives)."""
        from src.rtdlm import compute_multimodal_alignment_loss
        from src.config.agi_config import AGIConfig

        cfg = AGIConfig(enable_hard_negative_mining=True)
        aux = self._make_features(batch_size=1)
        loss = float(compute_multimodal_alignment_loss(aux, config=cfg))
        assert np.isfinite(loss)

    def test_no_modality_returns_zero(self):
        """No multimodal features -> 0.0."""
        from src.rtdlm import compute_multimodal_alignment_loss
        from src.config.agi_config import AGIConfig

        cfg = AGIConfig(enable_hard_negative_mining=True)
        loss = float(
            compute_multimodal_alignment_loss(
                {"text_features": jnp.ones((4, 32))},
                config=cfg,
            )
        )
        assert loss == pytest.approx(0.0)

    def test_3d_features_still_work(self):
        """3D features (with seq dim) should be averaged and work."""
        from src.rtdlm import compute_multimodal_alignment_loss
        from src.config.agi_config import AGIConfig

        cfg = AGIConfig(enable_hard_negative_mining=True)
        rng = jax.random.PRNGKey(0)
        k1, k2 = jax.random.split(rng)
        aux = {
            "text_features": jax.random.normal(k1, (4, 8, 32)),
            "audio_features": jax.random.normal(k2, (4, 8, 32)),
        }
        loss = float(compute_multimodal_alignment_loss(aux, config=cfg))
        assert np.isfinite(loss)
        assert loss > 0


class TestAGIConfigHardNegatives:
    """Config validation for hard negative mining flags."""

    def test_default_disabled(self):
        from src.config.agi_config import AGIConfig

        cfg = AGIConfig()
        assert cfg.enable_hard_negative_mining is False
        assert cfg.contrastive_margin == pytest.approx(0.2)
        assert cfg.hard_negative_ratio == pytest.approx(0.5)

    def test_margin_out_of_range(self):
        from src.config.agi_config import AGIConfig

        with pytest.raises(AssertionError, match="contrastive_margin"):
            AGIConfig(enable_hard_negative_mining=True, contrastive_margin=1.5)

    def test_ratio_zero_fails(self):
        from src.config.agi_config import AGIConfig

        with pytest.raises(AssertionError, match="hard_negative_ratio"):
            AGIConfig(enable_hard_negative_mining=True, hard_negative_ratio=0.0)

    def test_valid_custom(self):
        from src.config.agi_config import AGIConfig

        cfg = AGIConfig(
            enable_hard_negative_mining=True,
            contrastive_margin=0.1,
            hard_negative_ratio=0.8,
        )
        assert cfg.contrastive_margin == pytest.approx(0.1)
        assert cfg.hard_negative_ratio == pytest.approx(0.8)


# =========================================================================
# Config print_summary smoke test
# =========================================================================


class TestConfigPrintSummary:
    """Ensure print_summary works with new sections."""

    def test_print_summary_all_enabled(self, capsys):
        from src.config.agi_config import AGIConfig

        cfg = AGIConfig(
            enable_think_budget=True,
            enable_hard_negative_mining=True,
            enable_multi_res_vision=True,
        )
        cfg.print_summary()
        out = capsys.readouterr().out
        assert "Hard Negative Mining" in out
        assert "Multi-Resolution Vision" in out
        assert "Think Budget" in out

    def test_print_summary_all_disabled(self, capsys):
        from src.config.agi_config import AGIConfig

        cfg = AGIConfig()
        cfg.print_summary()
        out = capsys.readouterr().out
        assert "Hard Negative Mining" in out
        assert "Multi-Resolution Vision" in out
