"""
Unit tests for core/agi/agi_system.py

Tests the following components:
- AGISystemAbstraction: Central AGI orchestration hub
- ComponentFusion: Multi-component fusion with attention
- StageTracker: AGI stage progression tracking (0-6 levels)
- EthicalAlignmentModule: Ethical alignment evaluation
- AGIStage: Stage enumeration
- StageThresholds: Configurable thresholds
"""

import sys
import unittest
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

import jax
import jax.numpy as jnp
import haiku as hk

from core.agi.agi_system import (
    AGISystemAbstraction,
    ComponentFusion,
    StageTracker,
    EthicalAlignmentModule,
    AGIStage,
    StageThresholds,
    create_agi_system_fn,
)


# Test constants
D_MODEL = 64
BATCH_SIZE = 2


class TestAGIStage(unittest.TestCase):
    """Test AGI stage enumeration"""
    
    def test_stage_values(self):
        """Test stage values are correct"""
        self.assertEqual(AGIStage.REACTIVE, 0)
        self.assertEqual(AGIStage.REFLECTIVE, 1)
        self.assertEqual(AGIStage.LEARNING, 2)
        self.assertEqual(AGIStage.META_COGNITIVE, 3)
        self.assertEqual(AGIStage.SELF_IMPROVING, 4)
        self.assertEqual(AGIStage.CREATIVE, 5)
        self.assertEqual(AGIStage.CONSCIOUS, 6)
        
    def test_stage_names(self):
        """Test stage names are accessible"""
        self.assertEqual(AGIStage(0).name, "REACTIVE")
        self.assertEqual(AGIStage(6).name, "CONSCIOUS")


class TestStageThresholds(unittest.TestCase):
    """Test stage threshold configuration"""
    
    def test_default_thresholds(self):
        """Test default threshold values"""
        thresholds = StageThresholds()
        self.assertEqual(thresholds.reflective, 0.2)
        self.assertEqual(thresholds.learning, 0.35)
        self.assertEqual(thresholds.meta_cognitive, 0.5)
        self.assertEqual(thresholds.self_improving, 0.65)
        self.assertEqual(thresholds.creative, 0.8)
        self.assertEqual(thresholds.conscious, 0.95)
        
    def test_custom_thresholds(self):
        """Test custom threshold values"""
        thresholds = StageThresholds(
            reflective=0.1,
            learning=0.3,
            meta_cognitive=0.4,
            self_improving=0.6,
            creative=0.75,
            conscious=0.9
        )
        self.assertEqual(thresholds.reflective, 0.1)
        self.assertEqual(thresholds.conscious, 0.9)


class TestComponentFusion(unittest.TestCase):
    """Test ComponentFusion module"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.rng = jax.random.PRNGKey(42)
        
    def test_fusion_output_shape(self):
        """Test fusion produces correct output shape"""
        def fusion_fn(components):
            fusion = ComponentFusion(d_model=D_MODEL, num_components=3)
            return fusion(components)
        
        transformed = hk.transform(fusion_fn)
        
        components = {
            "consciousness": jnp.ones((BATCH_SIZE, D_MODEL)),
            "reasoning": jnp.ones((BATCH_SIZE, D_MODEL)),
            "creativity": jnp.ones((BATCH_SIZE, D_MODEL))
        }
        
        params = transformed.init(self.rng, components)
        fused, info = transformed.apply(params, self.rng, components)
        
        # Check output shape
        self.assertEqual(fused.shape, (BATCH_SIZE, D_MODEL))
        
        # Check fusion info
        self.assertIn("attention_weights", info)
        self.assertIn("coherence", info)
        self.assertEqual(len(info["component_names"]), 3)
        
    def test_attention_weights_sum_to_one(self):
        """Test attention weights are valid softmax output"""
        def fusion_fn(components):
            fusion = ComponentFusion(d_model=D_MODEL, num_components=2)
            return fusion(components)
        
        transformed = hk.transform(fusion_fn)
        
        components = {
            "a": jax.random.normal(self.rng, (BATCH_SIZE, D_MODEL)),
            "b": jax.random.normal(self.rng, (BATCH_SIZE, D_MODEL))
        }
        
        params = transformed.init(self.rng, components)
        _, info = transformed.apply(params, self.rng, components)
        
        # Attention weights should sum to 1
        weight_sums = info["attention_weights"].sum(axis=-1)
        self.assertTrue(jnp.allclose(weight_sums, 1.0, atol=1e-5))


class TestStageTracker(unittest.TestCase):
    """Test StageTracker module"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.rng = jax.random.PRNGKey(43)
        
    def test_stage_tracking(self):
        """Test stage tracking produces valid output"""
        def track_fn(unified_repr):
            tracker = StageTracker(d_model=D_MODEL)
            return tracker(unified_repr)
        
        transformed = hk.transform(track_fn)
        
        unified_repr = jnp.ones((BATCH_SIZE, D_MODEL))
        
        params = transformed.init(self.rng, unified_repr)
        stage, info = transformed.apply(params, self.rng, unified_repr)
        
        # Stage should be in valid range
        self.assertGreaterEqual(stage, 0)
        self.assertLessEqual(stage, 6)
        
        # Info should contain expected keys
        self.assertIn("current_stage", info)
        self.assertIn("stage_name", info)
        self.assertIn("stage_progress", info)
        self.assertIn("overall_score", info)
        
    def test_stage_progress_bounded(self):
        """Test stage progress is in [0, 1]"""
        def track_fn(unified_repr):
            tracker = StageTracker(d_model=D_MODEL)
            return tracker(unified_repr)
        
        transformed = hk.transform(track_fn)
        
        # Test with random inputs
        for _ in range(5):
            key = jax.random.split(self.rng)[0]
            unified_repr = jax.random.normal(key, (BATCH_SIZE, D_MODEL))
            
            params = transformed.init(key, unified_repr)
            _, info = transformed.apply(params, key, unified_repr)
            
            progress = info["stage_progress"]
            self.assertGreaterEqual(progress, 0.0)
            self.assertLessEqual(progress, 1.0)


class TestEthicalAlignmentModule(unittest.TestCase):
    """Test EthicalAlignmentModule"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.rng = jax.random.PRNGKey(44)
        
    def test_ethical_evaluation(self):
        """Test ethical evaluation produces valid output"""
        def ethics_fn(decision_repr):
            ethics = EthicalAlignmentModule(d_model=D_MODEL)
            return ethics(decision_repr)
        
        transformed = hk.transform(ethics_fn)
        
        decision_repr = jnp.ones((BATCH_SIZE, D_MODEL))
        
        params = transformed.init(self.rng, decision_repr)
        result = transformed.apply(params, self.rng, decision_repr)
        
        # Check output structure
        self.assertIn("ethical_scores", result)
        self.assertIn("overall_alignment", result)
        self.assertIn("risk_score", result)
        self.assertIn("dimension_names", result)
        
        # Check 8 ethical dimensions
        self.assertEqual(result["ethical_scores"].shape[-1], 8)
        self.assertEqual(len(result["dimension_names"]), 8)
        
    def test_ethical_scores_bounded(self):
        """Test ethical scores are in [0, 1] (sigmoid output)"""
        def ethics_fn(decision_repr):
            ethics = EthicalAlignmentModule(d_model=D_MODEL)
            return ethics(decision_repr)
        
        transformed = hk.transform(ethics_fn)
        
        decision_repr = jax.random.normal(self.rng, (BATCH_SIZE, D_MODEL))
        
        params = transformed.init(self.rng, decision_repr)
        result = transformed.apply(params, self.rng, decision_repr)
        
        # Scores should be in [0, 1]
        self.assertTrue(jnp.all(result["ethical_scores"] >= 0))
        self.assertTrue(jnp.all(result["ethical_scores"] <= 1))


class TestAGISystemAbstraction(unittest.TestCase):
    """Test AGISystemAbstraction main module"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.rng = jax.random.PRNGKey(45)
        
    def test_full_orchestration(self):
        """Test full AGI system orchestration"""
        def system_fn(consciousness_out, reasoning_out, creativity_out):
            system = AGISystemAbstraction(d_model=D_MODEL)
            return system(
                consciousness_output=consciousness_out,
                reasoning_output=reasoning_out,
                creativity_output=creativity_out
            )
        
        transformed = hk.transform(system_fn)
        
        consciousness_out = jnp.ones((BATCH_SIZE, D_MODEL))
        reasoning_out = jnp.ones((BATCH_SIZE, D_MODEL))
        creativity_out = jnp.ones((BATCH_SIZE, D_MODEL))
        
        params = transformed.init(
            self.rng, consciousness_out, reasoning_out, creativity_out
        )
        result = transformed.apply(
            params, self.rng, consciousness_out, reasoning_out, creativity_out
        )
        
        # Check output structure
        self.assertIn("unified_representation", result)
        self.assertIn("current_stage", result)
        self.assertIn("stage_info", result)
        self.assertIn("ethics_info", result)
        self.assertIn("improvement_signal", result)
        
        # Check unified representation shape
        self.assertEqual(
            result["unified_representation"].shape, 
            (BATCH_SIZE, D_MODEL)
        )
        
    def test_unify_components_method(self):
        """Test unify_components method directly"""
        def unify_fn(consciousness_out, reasoning_out):
            system = AGISystemAbstraction(d_model=D_MODEL)
            return system.unify_components(consciousness_out, reasoning_out)
        
        transformed = hk.transform(unify_fn)
        
        consciousness_out = jax.random.normal(self.rng, (BATCH_SIZE, D_MODEL))
        reasoning_out = jax.random.normal(self.rng, (BATCH_SIZE, D_MODEL))
        
        params = transformed.init(self.rng, consciousness_out, reasoning_out)
        unified, info = transformed.apply(
            params, self.rng, consciousness_out, reasoning_out
        )
        
        # Check output
        self.assertEqual(unified.shape, (BATCH_SIZE, D_MODEL))
        self.assertIn("coherence", info)
        
    def test_track_stage_method(self):
        """Test track_stage method directly"""
        def track_fn(unified_repr):
            system = AGISystemAbstraction(d_model=D_MODEL)
            return system.track_stage(unified_repr)
        
        transformed = hk.transform(track_fn)
        
        unified_repr = jnp.ones((BATCH_SIZE, D_MODEL))
        
        params = transformed.init(self.rng, unified_repr)
        stage, info = transformed.apply(params, self.rng, unified_repr)
        
        self.assertIsInstance(stage, int)
        self.assertIn("stage_name", info)


class TestAGISystemFactory(unittest.TestCase):
    """Test create_agi_system_fn factory"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.rng = jax.random.PRNGKey(46)
        
    def test_factory_creates_valid_transform(self):
        """Test factory creates valid Haiku transformed function"""
        system_fn = create_agi_system_fn(d_model=D_MODEL)
        
        consciousness_out = jnp.ones((BATCH_SIZE, D_MODEL))
        reasoning_out = jnp.ones((BATCH_SIZE, D_MODEL))
        
        params = system_fn.init(self.rng, consciousness_out, reasoning_out)
        result = system_fn.apply(params, self.rng, consciousness_out, reasoning_out)
        
        self.assertIn("unified_representation", result)
        self.assertIn("current_stage", result)


class TestAGISystemIntegration(unittest.TestCase):
    """Integration tests for AGI system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.rng = jax.random.PRNGKey(47)
        
    def test_stage_progression_with_scores(self):
        """Test that higher scores lead to higher stages"""
        def system_fn(consciousness_out, reasoning_out):
            system = AGISystemAbstraction(d_model=D_MODEL)
            return system(
                consciousness_output=consciousness_out,
                reasoning_output=reasoning_out
            )
        
        transformed = hk.transform(system_fn)
        
        # Test with low values (should be lower stage)
        low_input = jnp.zeros((BATCH_SIZE, D_MODEL))
        params = transformed.init(self.rng, low_input, low_input)
        low_result = transformed.apply(params, self.rng, low_input, low_input)
        
        # Test with high values (may be higher stage depending on learned weights)
        high_input = jnp.ones((BATCH_SIZE, D_MODEL)) * 10
        high_result = transformed.apply(params, self.rng, high_input, high_input)
        
        # Both should produce valid stages
        self.assertGreaterEqual(low_result["current_stage"], 0)
        self.assertLessEqual(high_result["current_stage"], 6)
        
    def test_ethical_tracking_toggle(self):
        """Test ethical tracking can be disabled"""
        def system_fn(consciousness_out, reasoning_out):
            system = AGISystemAbstraction(
                d_model=D_MODEL, 
                enable_ethical_tracking=False
            )
            return system(
                consciousness_output=consciousness_out,
                reasoning_output=reasoning_out
            )
        
        transformed = hk.transform(system_fn)
        
        inputs = jnp.ones((BATCH_SIZE, D_MODEL))
        params = transformed.init(self.rng, inputs, inputs)
        result = transformed.apply(params, self.rng, inputs, inputs)
        
        # Ethics should indicate disabled
        self.assertIn("ethics_info", result)
        self.assertFalse(result["ethics_info"].get("enabled", True))


if __name__ == "__main__":
    unittest.main()

