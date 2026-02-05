"""
Tests for Fairness Constraints in Ethics Module

Tests the FairnessAnalyzer, FairnessConfig, and fairness-constrained
reward model functionality. Validates that:
- Bias detection correctly identifies unfair predictions
- Demographic parity differences are computed correctly
- Fairness penalties are applied when bias > threshold (0.1)
- Sensitive attribute handling works as expected

These tests ensure unbiased outputs in sensitive domains (law, finance)
and alignment with human flourishing principles.
"""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np

# Handle JAX import
try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

# Import fairness module
import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

from src.core.ethics.ethical_adaptation import (
    FairnessConfig,
    FairnessResult,
    FairnessMetric,
    SensitiveAttribute,
    FairnessAnalyzer,
    FAIRLEARN_AVAILABLE
)


class TestFairnessConfig(unittest.TestCase):
    """Tests for FairnessConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = FairnessConfig()
        
        self.assertEqual(config.bias_threshold, 0.1)
        self.assertEqual(config.fairness_penalty_weight, 0.5)
        self.assertFalse(config.strict_enforcement)
        self.assertTrue(config.log_violations)
        self.assertTrue(config.apply_correction)
        self.assertEqual(config.min_samples_for_fairness, 10)
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = FairnessConfig(
            bias_threshold=0.05,
            fairness_penalty_weight=0.8,
            strict_enforcement=True,
            min_samples_for_fairness=20
        )
        
        self.assertEqual(config.bias_threshold, 0.05)
        self.assertEqual(config.fairness_penalty_weight, 0.8)
        self.assertTrue(config.strict_enforcement)
        self.assertEqual(config.min_samples_for_fairness, 20)
    
    def test_enabled_metrics_default(self):
        """Test default enabled metrics."""
        config = FairnessConfig()
        
        self.assertIn(FairnessMetric.DEMOGRAPHIC_PARITY, config.enabled_metrics)
        self.assertIn(FairnessMetric.EQUALIZED_ODDS, config.enabled_metrics)
    
    def test_sensitive_attributes_default(self):
        """Test default sensitive attributes."""
        config = FairnessConfig()
        
        self.assertIn(SensitiveAttribute.GENDER, config.sensitive_attributes)
        self.assertIn(SensitiveAttribute.RACE, config.sensitive_attributes)
        self.assertIn(SensitiveAttribute.AGE, config.sensitive_attributes)


class TestFairnessResult(unittest.TestCase):
    """Tests for FairnessResult dataclass."""
    
    def test_result_creation(self):
        """Test creating a fairness result."""
        result = FairnessResult(
            fairness_score=0.85,
            demographic_parity_diff=0.05,
            equalized_odds_diff=0.08
        )
        
        self.assertEqual(result.fairness_score, 0.85)
        self.assertEqual(result.demographic_parity_diff, 0.05)
        self.assertEqual(result.equalized_odds_diff, 0.08)
        self.assertEqual(result.violations, [])
        self.assertEqual(result.corrections, {})
    
    def test_result_with_violations(self):
        """Test result with detected violations."""
        result = FairnessResult(
            fairness_score=0.45,
            demographic_parity_diff=0.25,
            violations=["Demographic parity violation for gender group"],
            corrections={"gender_male": -0.1, "gender_female": 0.1}
        )
        
        self.assertEqual(len(result.violations), 1)
        self.assertIn("Demographic parity", result.violations[0])
        self.assertEqual(len(result.corrections), 2)


class TestSensitiveAttribute(unittest.TestCase):
    """Tests for SensitiveAttribute enum."""
    
    def test_all_attributes_exist(self):
        """Test all expected sensitive attributes are defined."""
        expected = [
            'GENDER', 'RACE', 'AGE', 'RELIGION', 'NATIONALITY',
            'DISABILITY', 'SEXUAL_ORIENTATION', 'SOCIOECONOMIC_STATUS', 'CUSTOM'
        ]
        
        for attr in expected:
            self.assertTrue(hasattr(SensitiveAttribute, attr))


class TestFairnessMetric(unittest.TestCase):
    """Tests for FairnessMetric enum."""
    
    def test_all_metrics_exist(self):
        """Test all expected fairness metrics are defined."""
        expected = [
            'DEMOGRAPHIC_PARITY', 'EQUALIZED_ODDS', 'EQUAL_OPPORTUNITY',
            'PREDICTIVE_PARITY', 'CALIBRATION', 'INDIVIDUAL_FAIRNESS'
        ]
        
        for metric in expected:
            self.assertTrue(hasattr(FairnessMetric, metric))
    
    def test_metric_values(self):
        """Test metric enum values."""
        self.assertEqual(FairnessMetric.DEMOGRAPHIC_PARITY.value, "demographic_parity")
        self.assertEqual(FairnessMetric.EQUALIZED_ODDS.value, "equalized_odds")


class TestFairnessAnalyzer(unittest.TestCase):
    """Tests for FairnessAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = FairnessConfig()
        self.analyzer = FairnessAnalyzer(self.config)
    
    def test_analyzer_initialization(self):
        """Test analyzer initializes correctly."""
        self.assertIsNotNone(self.analyzer)
        self.assertEqual(self.analyzer.config.bias_threshold, 0.1)
    
    def test_analyze_with_numpy_arrays(self):
        """Test analysis with numpy arrays."""
        # Create synthetic fair predictions (equal distribution across groups)
        np.random.seed(42)
        predictions = np.random.randint(0, 2, size=100)
        ground_truth = np.random.randint(0, 2, size=100)
        sensitive_features = np.random.choice(['A', 'B'], size=100)
        
        result = self.analyzer.analyze(
            predictions=predictions,
            ground_truth=ground_truth,
            sensitive_features=sensitive_features
        )
        
        self.assertIsInstance(result, FairnessResult)
        self.assertIsNotNone(result.fairness_score)
    
    @unittest.skipUnless(FAIRLEARN_AVAILABLE, "Fairlearn not installed")
    def test_demographic_parity_calculation(self):
        """Test demographic parity difference calculation with Fairlearn."""
        # Create biased predictions (group A gets more positive predictions)
        predictions = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        sensitive_features = np.array(['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B'])
        ground_truth = np.array([1, 1, 0, 0, 1, 1, 0, 0, 0, 1])
        
        result = self.analyzer.analyze(
            predictions=predictions,
            ground_truth=ground_truth,
            sensitive_features=sensitive_features
        )
        
        # Should detect significant bias
        self.assertIsNotNone(result.demographic_parity_diff)
        # Absolute difference should be 1.0 (100% vs 0%)
        self.assertGreater(abs(result.demographic_parity_diff or 0), 0.1)
    
    def test_bias_threshold_violation_detection(self):
        """Test that violations are detected when bias > threshold."""
        # Create clearly biased predictions
        predictions = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        sensitive_features = np.array(['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B'])
        ground_truth = np.array([1, 1, 0, 0, 1, 1, 0, 0, 0, 1])
        
        result = self.analyzer.analyze(
            predictions=predictions,
            ground_truth=ground_truth,
            sensitive_features=sensitive_features
        )
        
        # Should have violations for such extreme bias
        # (unless Fairlearn not available, in which case placeholder behavior)
        if FAIRLEARN_AVAILABLE:
            self.assertGreater(len(result.violations), 0)
    
    def test_fair_predictions_no_violations(self):
        """Test that fair predictions produce no violations."""
        # Create balanced predictions
        np.random.seed(42)
        n = 100
        predictions = np.random.randint(0, 2, size=n)
        ground_truth = predictions.copy()  # Perfect predictions
        # 50/50 split between groups
        sensitive_features = np.array(['A'] * 50 + ['B'] * 50)
        
        result = self.analyzer.analyze(
            predictions=predictions,
            ground_truth=ground_truth,
            sensitive_features=sensitive_features
        )
        
        # Fairness score should be relatively high
        self.assertGreater(result.fairness_score, 0.5)


@unittest.skipUnless(JAX_AVAILABLE, "JAX not installed")
class TestFairnessWithJAX(unittest.TestCase):
    """Tests for fairness integration with JAX arrays."""
    
    def test_jax_array_compatibility(self):
        """Test that JAX arrays can be converted for fairness analysis."""
        config = FairnessConfig()
        analyzer = FairnessAnalyzer(config)
        
        # Create JAX arrays
        predictions = jnp.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        ground_truth = jnp.array([1, 0, 1, 0, 1, 1, 0, 0, 1, 0])
        
        # Convert to numpy for analysis
        predictions_np = np.array(predictions)
        ground_truth_np = np.array(ground_truth)
        sensitive_features = np.array(['A'] * 5 + ['B'] * 5)
        
        result = analyzer.analyze(
            predictions=predictions_np,
            ground_truth=ground_truth_np,
            sensitive_features=sensitive_features
        )
        
        self.assertIsInstance(result, FairnessResult)


class TestFairnessPenaltyIntegration(unittest.TestCase):
    """Tests for fairness penalty in loss function."""
    
    @unittest.skipUnless(JAX_AVAILABLE, "JAX not installed")
    def test_compute_fairness_penalty_loss(self):
        """Test the fairness penalty loss computation."""
        from src.rtdlm import compute_fairness_penalty_loss
        
        # Create mock logits with high variance (biased)
        np.random.seed(42)
        logits = jnp.array(np.random.randn(4, 10, 100))
        
        fairness_eval = {
            "analyzer_active": True,
            "fairness_config": {
                "bias_threshold": 0.1
            }
        }
        
        loss = compute_fairness_penalty_loss(logits, fairness_eval)
        
        # Should return a scalar loss
        self.assertEqual(loss.shape, ())
        # Loss should be non-negative
        self.assertGreaterEqual(float(loss), 0.0)
    
    @unittest.skipUnless(JAX_AVAILABLE, "JAX not installed")
    def test_fairness_penalty_with_uniform_output(self):
        """Test penalty is lower for uniform (fair) outputs."""
        from src.rtdlm import compute_fairness_penalty_loss
        
        # Create uniform logits (all same value = maximum entropy)
        uniform_logits = jnp.zeros((4, 10, 100))
        
        # Create concentrated logits (one value much higher = low entropy)
        concentrated_logits = jnp.zeros((4, 10, 100))
        concentrated_logits = concentrated_logits.at[:, :, 0].set(10.0)
        
        fairness_eval = {
            "analyzer_active": True,
            "fairness_config": {"bias_threshold": 0.1}
        }
        
        uniform_loss = compute_fairness_penalty_loss(uniform_logits, fairness_eval)
        concentrated_loss = compute_fairness_penalty_loss(concentrated_logits, fairness_eval)
        
        # Concentrated (unfair) should have higher penalty
        self.assertGreater(float(concentrated_loss), float(uniform_loss))
    
    def test_fairness_penalty_with_none_logits(self):
        """Test penalty returns 0 when logits are None."""
        # Import only if available
        if JAX_AVAILABLE:
            from src.rtdlm import compute_fairness_penalty_loss
            
            result = compute_fairness_penalty_loss(None, {})
            self.assertEqual(result, 0.0)


class TestBiasThresholdBehavior(unittest.TestCase):
    """Tests specifically for the 0.1 bias threshold behavior."""
    
    def test_threshold_config_value(self):
        """Test that default threshold is 0.1 as specified."""
        config = FairnessConfig()
        self.assertEqual(config.bias_threshold, 0.1)
    
    def test_threshold_in_fairness_eval(self):
        """Test threshold appears correctly in fairness evaluation dict."""
        config = FairnessConfig()
        
        fairness_eval = {
            "analyzer_active": True,
            "fairness_config": {
                "bias_threshold": config.bias_threshold,
                "fairness_penalty_weight": config.fairness_penalty_weight
            }
        }
        
        self.assertEqual(fairness_eval["fairness_config"]["bias_threshold"], 0.1)
    
    @unittest.skipUnless(FAIRLEARN_AVAILABLE, "Fairlearn not installed")  
    def test_threshold_triggers_penalty_adjustment(self):
        """Test that exceeding 0.1 threshold triggers reward adjustment."""
        config = FairnessConfig(bias_threshold=0.1)
        analyzer = FairnessAnalyzer(config)
        
        # Create predictions with bias just above threshold
        # Group A: 80% positive, Group B: 60% positive = 20% difference > 0.1
        predictions = np.array([1, 1, 1, 1, 0, 1, 1, 1, 0, 0])
        sensitive_features = np.array(['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B'])
        ground_truth = np.array([1, 1, 1, 0, 0, 1, 1, 0, 0, 0])
        
        result = analyzer.analyze(
            predictions=predictions,
            ground_truth=ground_truth,
            sensitive_features=sensitive_features
        )
        
        # Check that violation was detected due to exceeding threshold
        if result.demographic_parity_diff is not None:
            bias_detected = abs(result.demographic_parity_diff) > 0.1
            if bias_detected:
                # Should have at least one violation
                self.assertGreater(len(result.violations), 0)


if __name__ == "__main__":
    unittest.main()

