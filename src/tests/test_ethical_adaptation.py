"""
Tests for Ethical Adaptation Module

Tests for fairness analysis, bias detection, and ethical
adaptation using Fairlearn-based metrics.
"""

import unittest
import jax.numpy as jnp
import numpy as np


class TestFairnessConfig(unittest.TestCase):
    """Test FairnessConfig dataclass."""
    
    def test_default_config(self):
        """Test default fairness configuration."""
        from src.core.ethics.ethical_adaptation import FairnessConfig
        
        config = FairnessConfig()
        
        self.assertIsNotNone(config.bias_threshold)
        self.assertIsNotNone(config.enabled_metrics)
        self.assertTrue(config.log_violations)
    
    def test_custom_config(self):
        """Test custom fairness configuration."""
        from src.core.ethics.ethical_adaptation import FairnessConfig, FairnessMetric
        
        config = FairnessConfig(
            bias_threshold=0.2,
            fairness_penalty_weight=0.8,
            strict_enforcement=True
        )
        
        self.assertEqual(config.bias_threshold, 0.2)
        self.assertEqual(config.fairness_penalty_weight, 0.8)
        self.assertTrue(config.strict_enforcement)


class TestSensitiveAttribute(unittest.TestCase):
    """Test SensitiveAttribute enum."""
    
    def test_sensitive_attribute_values(self):
        """Test SensitiveAttribute enum values."""
        from src.core.ethics.ethical_adaptation import SensitiveAttribute
        
        # Check that enum values exist
        self.assertIsNotNone(SensitiveAttribute.GENDER)
        self.assertIsNotNone(SensitiveAttribute.RACE)
        self.assertIsNotNone(SensitiveAttribute.AGE)
        self.assertIsNotNone(SensitiveAttribute.CUSTOM)


class TestFairnessAnalyzer(unittest.TestCase):
    """Test FairnessAnalyzer class."""
    
    def test_analyzer_initialization(self):
        """Test FairnessAnalyzer initialization."""
        from src.core.ethics.ethical_adaptation import (
            FairnessConfig, FairnessAnalyzer
        )
        
        config = FairnessConfig()
        analyzer = FairnessAnalyzer(config)
        
        self.assertIsNotNone(analyzer.config)
    
    def test_analyze_predictions(self):
        """Test analyzing predictions for fairness."""
        from src.core.ethics.ethical_adaptation import (
            FairnessConfig, FairnessAnalyzer
        )
        
        config = FairnessConfig()
        analyzer = FairnessAnalyzer(config)
        
        # Synthetic predictions and sensitive features
        predictions = np.array([1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        sensitive_features = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
        ground_truth = np.array([1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1])
        
        result = analyzer.analyze(
            predictions=predictions,
            sensitive_features=sensitive_features,
            ground_truth=ground_truth
        )
        
        self.assertIsNotNone(result)
        self.assertIsInstance(result.fairness_score, float)
    
    def test_analyze_with_insufficient_samples(self):
        """Test analysis with insufficient samples."""
        from src.core.ethics.ethical_adaptation import (
            FairnessConfig, FairnessAnalyzer
        )
        
        config = FairnessConfig(min_samples_for_fairness=100)
        analyzer = FairnessAnalyzer(config)
        
        # Too few samples
        predictions = np.array([1, 0, 1])
        sensitive_features = np.array([0, 1, 0])
        
        result = analyzer.analyze(
            predictions=predictions,
            sensitive_features=sensitive_features
        )
        
        # Should return default result with warning
        self.assertIsNotNone(result)
        self.assertTrue(len(result.violations) > 0 or result.is_fair)
    
    def test_compute_fairness_penalty(self):
        """Test fairness penalty computation."""
        from src.core.ethics.ethical_adaptation import (
            FairnessConfig, FairnessAnalyzer
        )
        
        config = FairnessConfig()
        analyzer = FairnessAnalyzer(config)
        
        # Create biased predictions
        predictions = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
        sensitive_features = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
        
        penalty = analyzer.compute_fairness_penalty(
            predictions, sensitive_features
        )
        
        # Penalty should be non-negative
        self.assertGreaterEqual(penalty, 0.0)


class TestFairnessResult(unittest.TestCase):
    """Test FairnessResult dataclass."""
    
    def test_fairness_result_creation(self):
        """Test creating FairnessResult."""
        from src.core.ethics.ethical_adaptation import FairnessResult
        
        result = FairnessResult(
            fairness_score=0.85,
            demographic_parity_diff=0.05,
            equalized_odds_diff=0.03,
            is_fair=True
        )
        
        self.assertEqual(result.fairness_score, 0.85)
        self.assertEqual(result.demographic_parity_diff, 0.05)
        self.assertTrue(result.is_fair)
    
    def test_fairness_result_with_violations(self):
        """Test FairnessResult with violations."""
        from src.core.ethics.ethical_adaptation import FairnessResult
        
        result = FairnessResult(
            fairness_score=0.5,
            violations=["Demographic parity violation: 0.25"],
            corrections={"demographic_parity": -0.125},
            is_fair=False
        )
        
        self.assertFalse(result.is_fair)
        self.assertEqual(len(result.violations), 1)
        self.assertIn("demographic_parity", result.corrections)


class TestBiasDetection(unittest.TestCase):
    """Test bias detection using analyze method."""
    
    def test_detect_bias_in_predictions(self):
        """Test detecting bias in model predictions via analysis."""
        from src.core.ethics.ethical_adaptation import (
            FairnessConfig, FairnessAnalyzer
        )
        
        config = FairnessConfig(bias_threshold=0.05)
        analyzer = FairnessAnalyzer(config)
        
        # Create biased predictions (group 1 always positive)
        predictions = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
        sensitive_features = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
        
        result = analyzer.analyze(
            predictions=predictions,
            sensitive_features=sensitive_features
        )
        
        # Should detect violations (is_fair should be False)
        self.assertFalse(result.is_fair)
    
    def test_no_bias_detection(self):
        """Test no bias detected for fair predictions."""
        from src.core.ethics.ethical_adaptation import (
            FairnessConfig, FairnessAnalyzer
        )
        
        config = FairnessConfig(bias_threshold=0.5)  # Higher threshold
        analyzer = FairnessAnalyzer(config)
        
        # Create roughly fair predictions
        predictions = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        sensitive_features = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
        
        result = analyzer.analyze(
            predictions=predictions,
            sensitive_features=sensitive_features
        )
        
        # Should be fair with balanced predictions
        self.assertTrue(result.is_fair)


class TestFairnessMetricEnum(unittest.TestCase):
    """Test FairnessMetric enum."""
    
    def test_fairness_metric_values(self):
        """Test FairnessMetric enum values."""
        from src.core.ethics.ethical_adaptation import FairnessMetric
        
        self.assertEqual(FairnessMetric.DEMOGRAPHIC_PARITY.value, "demographic_parity")
        self.assertEqual(FairnessMetric.EQUALIZED_ODDS.value, "equalized_odds")
        self.assertEqual(FairnessMetric.EQUAL_OPPORTUNITY.value, "equal_opportunity")


class TestGroupStatistics(unittest.TestCase):
    """Test per-group fairness statistics."""
    
    def test_group_statistics_in_result(self):
        """Test that analysis includes group statistics."""
        from src.core.ethics.ethical_adaptation import (
            FairnessConfig, FairnessAnalyzer
        )
        
        config = FairnessConfig()
        analyzer = FairnessAnalyzer(config)
        
        predictions = np.array([1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1])
        sensitive_features = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
        
        result = analyzer.analyze(
            predictions=predictions,
            sensitive_features=sensitive_features
        )
        
        # Should have group statistics
        self.assertIsInstance(result.group_statistics, dict)
    
    def test_fairness_score_range(self):
        """Test that fairness score is bounded."""
        from src.core.ethics.ethical_adaptation import (
            FairnessConfig, FairnessAnalyzer
        )
        
        config = FairnessConfig()
        analyzer = FairnessAnalyzer(config)
        
        predictions = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        sensitive_features = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
        
        result = analyzer.analyze(
            predictions=predictions,
            sensitive_features=sensitive_features
        )
        
        # Fairness score should be between 0 and 1
        self.assertGreaterEqual(result.fairness_score, 0.0)
        self.assertLessEqual(result.fairness_score, 1.0)


if __name__ == "__main__":
    unittest.main()


if __name__ == "__main__":
    unittest.main()
