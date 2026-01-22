"""
Tests for Benchmark Evaluation Module.

Tests production evaluation metrics including:
- Perplexity tracking
- Calibration metrics (ECE, MCE)
- Compute efficiency tracking
- Benchmark evaluation framework
"""

import unittest
import numpy as np
import jax
import jax.numpy as jnp

from core.benchmark_evaluation import (
    BenchmarkResult,
    CalibrationResult,
    ComputeMetrics,
    ProductionMetrics,
    CalibrationTracker,
    ComputeEfficiencyTracker,
    PerplexityTracker,
    BenchmarkEvaluator,
)


class TestBenchmarkResult(unittest.TestCase):
    """Tests for BenchmarkResult dataclass."""
    
    def test_basic_result(self):
        """Test basic benchmark result creation."""
        result = BenchmarkResult(
            benchmark_name="MMLU",
            accuracy=0.75,
            num_correct=75,
            num_total=100,
        )
        
        self.assertEqual(result.benchmark_name, "MMLU")
        self.assertEqual(result.accuracy, 0.75)
        self.assertEqual(result.num_correct, 75)
        self.assertEqual(result.num_total, 100)
    
    def test_with_categories(self):
        """Test result with category breakdown."""
        result = BenchmarkResult(
            benchmark_name="MMLU",
            accuracy=0.70,
            num_correct=70,
            num_total=100,
            category_scores={"math": 0.8, "science": 0.6},
        )
        
        self.assertIn("math", result.category_scores)
        self.assertEqual(result.category_scores["math"], 0.8)
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = BenchmarkResult(
            benchmark_name="Test",
            accuracy=0.5,
            num_correct=5,
            num_total=10,
            total_time_sec=1.5,
        )
        
        d = result.to_dict()
        self.assertEqual(d['benchmark'], "Test")
        self.assertEqual(d['accuracy'], 0.5)
        self.assertEqual(d['total_time_sec'], 1.5)


class TestCalibrationResult(unittest.TestCase):
    """Tests for CalibrationResult dataclass."""
    
    def test_basic_calibration(self):
        """Test basic calibration result."""
        result = CalibrationResult(
            expected_calibration_error=0.05,
            maximum_calibration_error=0.1,
            average_confidence=0.8,
            average_accuracy=0.75,
        )
        
        self.assertEqual(result.expected_calibration_error, 0.05)
        self.assertEqual(result.maximum_calibration_error, 0.1)
    
    def test_with_bins(self):
        """Test calibration with bin data."""
        result = CalibrationResult(
            expected_calibration_error=0.05,
            maximum_calibration_error=0.1,
            average_confidence=0.8,
            average_accuracy=0.75,
            bin_accuracies=[0.1, 0.3, 0.5, 0.7, 0.9],
            bin_confidences=[0.1, 0.3, 0.5, 0.7, 0.9],
            bin_counts=[10, 20, 30, 20, 10],
        )
        
        self.assertEqual(len(result.bin_accuracies), 5)
        self.assertEqual(sum(result.bin_counts), 90)


class TestCalibrationTracker(unittest.TestCase):
    """Tests for CalibrationTracker class."""
    
    def setUp(self):
        """Set up test tracker."""
        self.tracker = CalibrationTracker(num_bins=10)
    
    def test_initialization(self):
        """Test tracker initialization."""
        self.assertEqual(self.tracker.num_bins, 10)
        self.assertEqual(len(self.tracker.confidences), 0)
    
    def test_update_with_classification(self):
        """Test update with classification logits."""
        # Create simple classification logits [batch, num_classes]
        logits = jnp.array([[2.0, 0.5, 0.1],  # Confident class 0
                           [0.1, 3.0, 0.2],  # Confident class 1
                           [0.5, 0.5, 0.5]])  # Uncertain
        targets = jnp.array([0, 1, 2])
        
        self.tracker.update(logits, targets)
        
        self.assertEqual(len(self.tracker.confidences), 3)
        self.assertEqual(len(self.tracker.predictions), 3)
    
    def test_update_with_sequence(self):
        """Test update with sequence logits."""
        # Create sequence logits [batch, seq_len, vocab_size]
        batch_size, seq_len, vocab_size = 2, 4, 10
        logits = jax.random.normal(jax.random.PRNGKey(42), (batch_size, seq_len, vocab_size))
        targets = jax.random.randint(jax.random.PRNGKey(43), (batch_size, seq_len), 0, vocab_size)
        
        self.tracker.update(logits, targets)
        
        # Should have batch_size * seq_len predictions
        self.assertEqual(len(self.tracker.confidences), batch_size * seq_len)
    
    def test_compute_empty(self):
        """Test compute with no data."""
        result = self.tracker.compute()
        
        self.assertEqual(result.expected_calibration_error, 0.0)
        self.assertEqual(result.average_confidence, 0.0)
    
    def test_compute_with_data(self):
        """Test compute with accumulated data."""
        # Add some predictions
        logits = jnp.array([[5.0, 0.0],  # Very confident class 0
                           [0.0, 5.0],  # Very confident class 1
                           [0.0, 5.0]]) # Very confident class 1
        targets = jnp.array([0, 1, 0])  # One wrong (last one)
        
        self.tracker.update(logits, targets)
        result = self.tracker.compute()
        
        # Average accuracy should be 2/3
        self.assertAlmostEqual(result.average_accuracy, 2/3, places=2)
        # Confidence should be high (close to 1.0)
        self.assertGreater(result.average_confidence, 0.9)
    
    def test_reset(self):
        """Test reset clears data."""
        logits = jnp.ones((2, 3))
        targets = jnp.zeros((2,), dtype=jnp.int32)
        
        self.tracker.update(logits, targets)
        self.assertGreater(len(self.tracker.confidences), 0)
        
        self.tracker.reset()
        self.assertEqual(len(self.tracker.confidences), 0)


class TestComputeEfficiencyTracker(unittest.TestCase):
    """Tests for ComputeEfficiencyTracker class."""
    
    def setUp(self):
        """Set up test tracker."""
        self.tracker = ComputeEfficiencyTracker()
    
    def test_initialization(self):
        """Test tracker initialization."""
        self.assertEqual(len(self.tracker.latencies), 0)
        self.assertEqual(self.tracker.total_time, 0.0)
    
    def test_batch_timing(self):
        """Test batch timing measurement."""
        import time
        
        self.tracker.start_batch()
        time.sleep(0.01)  # 10ms
        self.tracker.end_batch(num_tokens=100, num_samples=4)
        
        self.assertEqual(len(self.tracker.latencies), 1)
        self.assertEqual(self.tracker.token_counts[0], 100)
        self.assertGreater(self.tracker.latencies[0], 5)  # > 5ms
    
    def test_compute_metrics(self):
        """Test compute metrics calculation."""
        import time
        
        # Add some measurements
        for i in range(5):
            self.tracker.start_batch()
            time.sleep(0.005)  # 5ms
            self.tracker.end_batch(num_tokens=100, num_samples=4)
        
        metrics = self.tracker.compute()
        
        self.assertGreater(metrics.tokens_per_second, 0)
        self.assertGreater(metrics.samples_per_second, 0)
        self.assertGreater(metrics.avg_latency_ms, 0)
    
    def test_compute_empty(self):
        """Test compute with no measurements."""
        metrics = self.tracker.compute()
        
        self.assertEqual(metrics.tokens_per_second, 0.0)
        self.assertEqual(metrics.samples_per_second, 0.0)
    
    def test_flops_estimation(self):
        """Test FLOPs estimation with model config."""
        tracker = ComputeEfficiencyTracker(
            model_config={'d_model': 512, 'num_layers': 6}
        )
        
        tracker.start_batch()
        tracker.end_batch(100, 1)
        
        metrics = tracker.compute()
        
        # FLOPs should be estimated
        self.assertIsNotNone(metrics.flops_per_token)
        self.assertGreater(metrics.flops_per_token, 0)
    
    def test_reset(self):
        """Test reset clears measurements."""
        self.tracker.start_batch()
        self.tracker.end_batch(100, 1)
        
        self.assertGreater(len(self.tracker.latencies), 0)
        
        self.tracker.reset()
        self.assertEqual(len(self.tracker.latencies), 0)


class TestPerplexityTracker(unittest.TestCase):
    """Tests for PerplexityTracker class."""
    
    def setUp(self):
        """Set up test tracker."""
        self.tracker = PerplexityTracker(window_size=10)
    
    def test_initialization(self):
        """Test tracker initialization."""
        self.assertEqual(self.tracker.window_size, 10)
        self.assertEqual(len(self.tracker.losses), 0)
    
    def test_update(self):
        """Test update with loss values."""
        self.tracker.update(loss=2.0, num_tokens=100)
        self.tracker.update(loss=1.8, num_tokens=100)
        
        self.assertEqual(len(self.tracker.losses), 2)
    
    def test_get_perplexity(self):
        """Test perplexity calculation."""
        # Loss of 1.0 -> perplexity of e ≈ 2.718
        self.tracker.update(loss=1.0, num_tokens=100)
        
        ppl = self.tracker.get_perplexity()
        self.assertAlmostEqual(ppl, np.exp(1.0), places=2)
    
    def test_perplexity_empty(self):
        """Test perplexity with no data."""
        ppl = self.tracker.get_perplexity()
        self.assertEqual(ppl, float('inf'))
    
    def test_weighted_average(self):
        """Test weighted averaging by token count."""
        # Add losses with different token counts
        self.tracker.update(loss=1.0, num_tokens=100)  # Weight: 100
        self.tracker.update(loss=2.0, num_tokens=100)  # Weight: 100
        
        # Average loss should be (1.0*100 + 2.0*100) / 200 = 1.5
        loss = self.tracker.get_loss()
        self.assertAlmostEqual(loss, 1.5, places=4)
    
    def test_window_sliding(self):
        """Test that window slides correctly."""
        tracker = PerplexityTracker(window_size=3)
        
        for i in range(5):
            tracker.update(loss=float(i), num_tokens=100)
        
        # Should only keep last 3 values (2, 3, 4)
        self.assertEqual(len(tracker.losses), 3)
    
    def test_reset(self):
        """Test reset clears data."""
        self.tracker.update(loss=1.0, num_tokens=100)
        self.tracker.reset()
        
        self.assertEqual(len(self.tracker.losses), 0)


class TestProductionMetrics(unittest.TestCase):
    """Tests for ProductionMetrics dataclass."""
    
    def test_basic_metrics(self):
        """Test basic production metrics."""
        metrics = ProductionMetrics(
            perplexity=10.5,
            validation_loss=2.3,
            token_accuracy=0.85,
        )
        
        self.assertEqual(metrics.perplexity, 10.5)
        self.assertEqual(metrics.validation_loss, 2.3)
        self.assertEqual(metrics.token_accuracy, 0.85)
    
    def test_with_benchmarks(self):
        """Test with benchmark results."""
        mmlu = BenchmarkResult("MMLU", 0.7, 70, 100)
        
        metrics = ProductionMetrics(
            perplexity=10.0,
            validation_loss=2.0,
            token_accuracy=0.8,
            benchmark_results={'mmlu': mmlu},
        )
        
        self.assertIn('mmlu', metrics.benchmark_results)
        self.assertEqual(metrics.benchmark_results['mmlu'].accuracy, 0.7)
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = ProductionMetrics(
            perplexity=10.0,
            validation_loss=2.0,
            token_accuracy=0.8,
        )
        
        d = metrics.to_dict()
        self.assertEqual(d['perplexity'], 10.0)
        self.assertEqual(d['validation_loss'], 2.0)
    
    def test_summary(self):
        """Test summary generation."""
        cal = CalibrationResult(0.05, 0.1, 0.8, 0.75)
        compute = ComputeMetrics(1000.0, 50.0, avg_latency_ms=10.0)
        
        metrics = ProductionMetrics(
            perplexity=10.0,
            validation_loss=2.0,
            token_accuracy=0.8,
            calibration=cal,
            compute=compute,
        )
        
        summary = metrics.summary()
        
        self.assertIn("Perplexity", summary)
        self.assertIn("10.0", summary)
        self.assertIn("Calibration", summary)
        self.assertIn("ECE", summary)


class TestBenchmarkEvaluator(unittest.TestCase):
    """Tests for BenchmarkEvaluator class."""
    
    def setUp(self):
        """Set up test evaluator."""
        # Mock model function
        def mock_model_fn(params, rng, **kwargs):
            batch_size = kwargs['inputs']['text'].shape[0]
            seq_len = kwargs['inputs']['text'].shape[1]
            vocab_size = 100
            return {'logits': jax.random.normal(rng, (batch_size, seq_len, vocab_size))}
        
        self.model_fn = mock_model_fn
        self.evaluator = BenchmarkEvaluator(
            model_apply_fn=mock_model_fn,
            vocab_size=100,
        )
    
    def test_initialization(self):
        """Test evaluator initialization."""
        self.assertIsNotNone(self.evaluator.calibration_tracker)
        self.assertIsNotNone(self.evaluator.compute_tracker)
    
    def test_evaluate_multiple_choice(self):
        """Test multiple choice evaluation."""
        questions = [
            {'prompt': 'What is 2+2?', 'choices': ['3', '4', '5'], 'answer': 1},
            {'prompt': 'What is 3+3?', 'choices': ['5', '6', '7'], 'answer': 1},
        ]
        
        params = {}
        rng = jax.random.PRNGKey(42)
        
        result = self.evaluator.evaluate_multiple_choice(
            params, questions, 'TestBench', rng
        )
        
        self.assertEqual(result.benchmark_name, 'TestBench')
        self.assertEqual(result.num_total, 2)
        # Accuracy is random due to placeholder implementation
        self.assertGreaterEqual(result.accuracy, 0.0)
        self.assertLessEqual(result.accuracy, 1.0)
    
    def test_evaluate_perplexity(self):
        """Test perplexity evaluation."""
        # Create mock validation data
        val_data = [
            {
                'input_ids': jnp.ones((2, 16), dtype=jnp.int32),
                'targets': jnp.ones((2, 16), dtype=jnp.int32),
            }
            for _ in range(3)
        ]
        
        params = {}
        rng = jax.random.PRNGKey(42)
        
        ppl, loss = self.evaluator.evaluate_perplexity(params, val_data, rng)
        
        self.assertGreater(ppl, 0)
        self.assertTrue(np.isfinite(ppl))


class TestIntegration(unittest.TestCase):
    """Integration tests for evaluation components."""
    
    def test_full_evaluation_flow(self):
        """Test complete evaluation flow."""
        # Create trackers
        ppl_tracker = PerplexityTracker(window_size=10)
        cal_tracker = CalibrationTracker(num_bins=5)
        compute_tracker = ComputeEfficiencyTracker()
        
        # Simulate training loop
        for i in range(10):
            # Create mock batch
            batch_size, seq_len, vocab_size = 4, 16, 100
            logits = jax.random.normal(
                jax.random.PRNGKey(i), 
                (batch_size, seq_len, vocab_size)
            )
            targets = jax.random.randint(
                jax.random.PRNGKey(i+100), 
                (batch_size, seq_len), 
                0, vocab_size
            )
            
            # Track compute
            compute_tracker.start_batch()
            
            # Update perplexity
            loss = float(jnp.mean(jnp.abs(logits)))
            num_tokens = batch_size * seq_len
            ppl_tracker.update(loss, num_tokens)
            
            # Update calibration
            cal_tracker.update(logits, targets)
            
            # End compute tracking
            compute_tracker.end_batch(num_tokens, batch_size)
        
        # Get final metrics
        ppl = ppl_tracker.get_perplexity()
        cal_result = cal_tracker.compute()
        compute_result = compute_tracker.compute()
        
        # Create production metrics
        metrics = ProductionMetrics(
            perplexity=ppl,
            validation_loss=ppl_tracker.get_loss(),
            token_accuracy=cal_result.average_accuracy,
            calibration=cal_result,
            compute=compute_result,
        )
        
        # Verify all metrics are computed
        self.assertGreater(metrics.perplexity, 0)
        self.assertIsNotNone(metrics.calibration)
        self.assertIsNotNone(metrics.compute)
        
        # Verify summary generation
        summary = metrics.summary()
        self.assertIn("PRODUCTION EVALUATION", summary)


if __name__ == "__main__":
    unittest.main(verbosity=2)
