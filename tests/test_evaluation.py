"""
Tests for evaluation module and decorators.

Tests:
- Decorators (core.utils.decorators)
- Evaluation metrics (core.training.evaluation)
- Gradient monitoring
- Logging and validation
"""

import pytest
import jax
import jax.numpy as jnp
import tempfile
import json
import warnings
from pathlib import Path


class TestDevUtilityDecorator:
    """Test the @dev_utility decorator."""
    
    def test_decorator_marks_function(self):
        """Test that decorator properly marks functions."""
        from core.utils import dev_utility, is_dev_utility, get_dev_utility_reason
        
        @dev_utility("Testing purposes only")
        def my_func():
            return 42
        
        assert is_dev_utility(my_func)
        assert get_dev_utility_reason(my_func) == "Testing purposes only"
        assert my_func() == 42  # Function still works
    
    def test_decorator_marks_class(self):
        """Test that decorator properly marks classes."""
        from core.utils import dev_utility, is_dev_utility
        
        @dev_utility("Internal testing class")
        class MyClass:
            def method(self):
                return "works"
        
        assert is_dev_utility(MyClass)
        obj = MyClass()
        assert obj.method() == "works"
    
    def test_non_decorated_not_marked(self):
        """Test that non-decorated items are not marked."""
        from core.utils import is_dev_utility
        
        def regular_func():
            return 1
        
        class RegularClass:
            pass
        
        assert not is_dev_utility(regular_func)
        assert not is_dev_utility(RegularClass)
    
    def test_backward_compat_import(self):
        """Test backward compatibility import from core.evaluation."""
        from core.evaluation import dev_utility, is_dev_utility
        
        @dev_utility("Test")
        def func():
            pass
        
        assert is_dev_utility(func)


class TestDeprecatedDecorator:
    """Test the @deprecated decorator."""
    
    def test_deprecated_warns(self):
        """Test that deprecated decorator emits warning."""
        from core.utils import deprecated, is_deprecated
        
        @deprecated("Use new_func instead")
        def old_func():
            return 42
        
        assert is_deprecated(old_func)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = old_func()
            assert result == 42
            assert len(w) == 1
            assert "deprecated" in str(w[0].message).lower()
    
    def test_deprecated_class(self):
        """Test deprecated class warns on instantiation."""
        from core.utils import deprecated
        
        @deprecated("Use NewClass instead", version="2.0")
        class OldClass:
            def __init__(self):
                self.value = 1
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            obj = OldClass()
            assert obj.value == 1
            assert len(w) == 1


class TestExperimentalDecorator:
    """Test the @experimental decorator."""
    
    def test_experimental_marks(self):
        """Test that experimental decorator marks functions."""
        from core.utils import experimental, is_experimental
        
        @experimental("New quantum backend")
        def quantum_func():
            return "quantum"
        
        assert is_experimental(quantum_func)
        assert quantum_func() == "quantum"


class TestInternalDecorator:
    """Test the @internal decorator."""
    
    def test_internal_marks(self):
        """Test that internal decorator marks functions."""
        from core.utils import internal, is_internal
        
        @internal("Low-level implementation")
        def _internal_func():
            return "internal"
        
        assert is_internal(_internal_func)
        assert _internal_func() == "internal"


class TestRequiresDecorator:
    """Test the @requires decorator."""
    
    def test_requires_with_available_deps(self):
        """Test requires passes with available dependencies."""
        from core.utils import requires
        
        @requires("json", "os")  # Standard library, always available
        def func_with_deps():
            return "works"
        
        assert func_with_deps() == "works"
    
    def test_requires_with_missing_deps(self):
        """Test requires raises with missing dependencies."""
        from core.utils import requires
        
        @requires("nonexistent_package_xyz")
        def func_missing_dep():
            return "never"
        
        with pytest.raises(ImportError) as exc_info:
            func_missing_dep()
        
        assert "nonexistent_package_xyz" in str(exc_info.value)


class TestBatchMetrics:
    """Test BatchMetrics dataclass."""
    
    def test_basic_creation(self):
        """Test basic BatchMetrics creation."""
        from core.training import BatchMetrics
        
        metrics = BatchMetrics(
            loss=0.5,
            perplexity=1.65,
            token_accuracy=0.85,
            num_tokens=1024,
            sequence_length=128,
        )
        
        assert metrics.loss == 0.5
        assert metrics.perplexity == 1.65
        assert metrics.token_accuracy == 0.85
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        from core.training import BatchMetrics
        
        metrics = BatchMetrics(
            loss=0.5,
            perplexity=1.65,
            token_accuracy=0.85,
            num_tokens=1024,
            sequence_length=128,
            top5_accuracy=0.95,
        )
        
        d = metrics.to_dict()
        assert 'loss' in d
        assert 'top5_accuracy' in d
        assert d['top5_accuracy'] == 0.95


class TestEvaluationMetrics:
    """Test EvaluationMetrics computation."""
    
    @pytest.fixture
    def evaluator(self):
        """Create evaluator instance."""
        from core.training import EvaluationMetrics
        return EvaluationMetrics(vocab_size=100, pad_token_id=0)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample logits and targets."""
        batch_size, seq_len, vocab_size = 2, 10, 100
        key = jax.random.PRNGKey(42)
        
        # Create logits with some structure
        logits = jax.random.normal(key, (batch_size, seq_len, vocab_size))
        
        # Create targets
        targets = jax.random.randint(
            jax.random.PRNGKey(43), 
            (batch_size, seq_len), 
            0, vocab_size
        )
        
        return logits, targets
    
    def test_perplexity_computation(self, evaluator, sample_data):
        """Test perplexity is computed correctly."""
        logits, targets = sample_data
        
        perplexity, loss = evaluator.compute_perplexity(logits, targets)
        
        assert perplexity >= 1.0  # Perplexity is always >= 1
        assert loss >= 0  # Cross entropy is non-negative
        assert perplexity == pytest.approx(jnp.exp(loss), rel=0.01)
    
    def test_perplexity_perfect_prediction(self, evaluator):
        """Test perplexity approaches 1 for perfect predictions."""
        batch_size, seq_len, vocab_size = 2, 10, 100
        
        # Create one-hot logits (very confident correct predictions)
        targets = jnp.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                           [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]])
        
        # Create logits that strongly favor the correct tokens
        logits = jnp.ones((batch_size, seq_len, vocab_size)) * -100  # Very low for all
        # Set high values for correct tokens
        for b in range(batch_size):
            for s in range(seq_len):
                logits = logits.at[b, s, targets[b, s]].set(100)
        
        perplexity, loss = evaluator.compute_perplexity(logits, targets)
        
        # Should be very close to 1
        assert perplexity < 1.1
    
    def test_token_accuracy(self, evaluator, sample_data):
        """Test token accuracy computation."""
        logits, targets = sample_data
        
        accuracy = evaluator.compute_token_accuracy(logits, targets)
        
        assert 0 <= accuracy <= 1.0
    
    def test_token_accuracy_perfect(self, evaluator):
        """Test 100% accuracy when predictions match targets."""
        batch_size, seq_len, vocab_size = 2, 10, 100
        targets = jnp.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                           [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]])
        
        logits = jnp.ones((batch_size, seq_len, vocab_size)) * -100
        for b in range(batch_size):
            for s in range(seq_len):
                logits = logits.at[b, s, targets[b, s]].set(100)
        
        accuracy = evaluator.compute_token_accuracy(logits, targets)
        
        assert accuracy > 0.99
    
    def test_top5_accuracy(self, evaluator, sample_data):
        """Test top-5 accuracy is >= top-1."""
        logits, targets = sample_data
        
        top1 = evaluator.compute_token_accuracy(logits, targets, top_k=1)
        top5 = evaluator.compute_token_accuracy(logits, targets, top_k=5)
        
        assert top5 >= top1
    
    def test_entropy_computation(self, evaluator, sample_data):
        """Test entropy is computed correctly."""
        logits, targets = sample_data
        
        entropy = evaluator.compute_entropy(logits)
        
        assert entropy >= 0  # Entropy is non-negative
    
    def test_entropy_uniform_high(self, evaluator):
        """Test that uniform distribution has high entropy."""
        batch_size, seq_len, vocab_size = 2, 10, 100
        
        # Uniform logits = high entropy
        uniform_logits = jnp.zeros((batch_size, seq_len, vocab_size))
        
        # Peaked logits = low entropy
        peaked_logits = jnp.ones((batch_size, seq_len, vocab_size)) * -100
        peaked_logits = peaked_logits.at[:, :, 0].set(100)
        
        uniform_entropy = evaluator.compute_entropy(uniform_logits)
        peaked_entropy = evaluator.compute_entropy(peaked_logits)
        
        assert uniform_entropy > peaked_entropy
    
    def test_batch_metrics_computation(self, evaluator, sample_data):
        """Test full batch metrics computation."""
        logits, targets = sample_data
        
        metrics = evaluator.compute_batch_metrics(
            logits, targets, loss=0.5, compute_entropy=True
        )
        
        assert metrics.loss == 0.5
        assert metrics.perplexity >= 1.0
        assert 0 <= metrics.token_accuracy <= 1.0
        assert metrics.entropy is not None
        assert metrics.entropy >= 0
    
    def test_mask_handling(self, evaluator):
        """Test that masking works correctly."""
        batch_size, seq_len, vocab_size = 2, 10, 100
        
        logits = jax.random.normal(
            jax.random.PRNGKey(42), 
            (batch_size, seq_len, vocab_size)
        )
        targets = jax.random.randint(
            jax.random.PRNGKey(43), 
            (batch_size, seq_len), 
            1, vocab_size  # Note: starts at 1, not 0
        )
        
        # Set some targets to padding (0)
        targets = targets.at[:, -3:].set(0)  # Last 3 tokens are padding
        
        # Compute with automatic mask (ignores pad_token_id=0)
        perplexity_auto, _ = evaluator.compute_perplexity(logits, targets)
        
        # Compute with explicit mask
        mask = (targets != 0).astype(jnp.float32)
        perplexity_explicit, _ = evaluator.compute_perplexity(logits, targets, mask)
        
        # Should be approximately equal
        assert perplexity_auto == pytest.approx(perplexity_explicit, rel=0.01)


class TestGradientMonitor:
    """Test gradient monitoring functionality."""
    
    @pytest.fixture
    def monitor(self):
        """Create gradient monitor."""
        from core.evaluation import GradientMonitor
        return GradientMonitor(
            exploding_threshold=100.0,
            vanishing_threshold=1e-7,
        )
    
    @pytest.fixture
    def sample_grads(self):
        """Create sample gradients."""
        return {
            'layer1': {'weights': jnp.ones((64, 64)) * 0.1},
            'layer2': {'weights': jnp.ones((64, 64)) * 0.05},
        }
    
    def test_gradient_norm_computation(self, monitor, sample_grads):
        """Test gradient norm is computed."""
        metrics = monitor.compute_gradient_metrics(sample_grads)
        
        assert metrics.global_norm > 0
        assert metrics.max_norm > 0
        assert metrics.min_norm >= 0
        assert metrics.mean_norm > 0
    
    def test_nan_detection(self, monitor):
        """Test NaN gradient detection."""
        grads_with_nan = {
            'layer1': {'weights': jnp.array([[float('nan'), 1.0], [1.0, 1.0]])},
        }
        
        metrics = monitor.compute_gradient_metrics(grads_with_nan)
        
        assert metrics.has_nan
    
    def test_inf_detection(self, monitor):
        """Test Inf gradient detection."""
        grads_with_inf = {
            'layer1': {'weights': jnp.array([[float('inf'), 1.0], [1.0, 1.0]])},
        }
        
        metrics = monitor.compute_gradient_metrics(grads_with_inf)
        
        assert metrics.has_inf
    
    def test_exploding_detection(self, monitor):
        """Test exploding gradient detection."""
        exploding_grads = {
            'layer1': {'weights': jnp.ones((100, 100)) * 10.0},  # Large grads
        }
        
        metrics = monitor.compute_gradient_metrics(exploding_grads)
        
        assert metrics.is_exploding
    
    def test_vanishing_detection(self, monitor):
        """Test vanishing gradient detection."""
        vanishing_grads = {
            'layer1': {'weights': jnp.ones((10, 10)) * 1e-10},  # Tiny grads
        }
        
        metrics = monitor.compute_gradient_metrics(vanishing_grads)
        
        assert metrics.is_vanishing
    
    def test_healthy_gradients(self, monitor, sample_grads):
        """Test healthy gradients are properly identified."""
        metrics = monitor.compute_gradient_metrics(sample_grads)
        
        assert not metrics.has_nan
        assert not metrics.has_inf
        assert not metrics.is_exploding
        assert not metrics.is_vanishing
    
    def test_trend_analysis(self, monitor, sample_grads):
        """Test gradient trend analysis."""
        # Compute several times to build history
        for _ in range(10):
            monitor.compute_gradient_metrics(sample_grads)
        
        trend = monitor.get_trend()
        
        assert 'trend' in trend
        assert 'volatility' in trend
        assert 'recent_mean' in trend
    
    def test_to_dict(self, monitor, sample_grads):
        """Test metrics conversion to dict."""
        metrics = monitor.compute_gradient_metrics(sample_grads)
        
        d = metrics.to_dict()
        
        assert 'global_norm' in d
        assert 'has_nan' in d
        assert 'is_exploding' in d


class TestMetricLogger:
    """Test structured metric logging."""
    
    @pytest.fixture
    def temp_log_dir(self):
        """Create temporary log directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @pytest.fixture
    def logger(self, temp_log_dir):
        """Create metric logger."""
        from core.evaluation import MetricLogger
        return MetricLogger(
            log_dir=temp_log_dir,
            experiment_name="test_run",
            log_every_n_steps=1,
            console_log_every_n_steps=100,  # Reduce console noise
        )
    
    def test_logger_creates_directories(self, temp_log_dir):
        """Test logger creates necessary directories."""
        from core.evaluation import MetricLogger
        
        logger = MetricLogger(
            log_dir=temp_log_dir,
            experiment_name="test_experiment"
        )
        
        experiment_dir = Path(temp_log_dir) / "test_experiment"
        assert experiment_dir.exists()
    
    def test_config_logging(self, logger, temp_log_dir):
        """Test configuration logging."""
        config = {'learning_rate': 0.001, 'batch_size': 32}
        logger.log_config(config)
        
        config_file = Path(temp_log_dir) / "test_run" / "config.json"
        assert config_file.exists()
        
        with open(config_file) as f:
            saved = json.load(f)
        
        assert saved['config']['learning_rate'] == 0.001
    
    def test_step_logging(self, logger, temp_log_dir):
        """Test training step logging."""
        from core.evaluation import TrainingStepMetrics, BatchMetrics
        
        batch_metrics = BatchMetrics(
            loss=0.5,
            perplexity=1.65,
            token_accuracy=0.85,
            num_tokens=1024,
            sequence_length=128,
        )
        
        step_metrics = TrainingStepMetrics(
            step=100,
            timestamp=1234567890.0,
            batch_metrics=batch_metrics,
            learning_rate=0.001,
        )
        
        logger.log_step(100, step_metrics)
        
        metrics_file = Path(temp_log_dir) / "test_run" / "metrics.jsonl"
        assert metrics_file.exists()
        
        with open(metrics_file) as f:
            line = f.readline()
            logged = json.loads(line)
        
        assert logged['step'] == 100
        assert logged['loss'] == 0.5
    
    def test_validation_logging(self, logger, temp_log_dir):
        """Test validation metrics logging."""
        from core.evaluation import ValidationMetrics
        
        val_metrics = ValidationMetrics(
            total_loss=0.4,
            perplexity=1.5,
            token_accuracy=0.88,
            num_batches=50,
            num_tokens=50000,
            total_time_sec=30.5,
        )
        
        logger.log_validation(1000, val_metrics)
        
        val_file = Path(temp_log_dir) / "test_run" / "validation.jsonl"
        assert val_file.exists()
    
    def test_best_metrics_tracking(self, logger):
        """Test best metrics are tracked."""
        from core.evaluation import TrainingStepMetrics, BatchMetrics
        
        # Log several steps with varying loss
        for i, loss in enumerate([0.5, 0.3, 0.4, 0.2, 0.6]):
            batch_metrics = BatchMetrics(
                loss=loss,
                perplexity=float(jnp.exp(loss)),
                token_accuracy=1.0 - loss,
                num_tokens=1024,
                sequence_length=128,
            )
            step_metrics = TrainingStepMetrics(
                step=i,
                timestamp=float(i),
                batch_metrics=batch_metrics,
                learning_rate=0.001,
            )
            logger.log_step(i, step_metrics, force=True)
        
        best = logger.get_best_metrics()
        
        assert best['best_loss'] == 0.2
    
    def test_summary_generation(self, logger):
        """Test training summary generation."""
        from core.evaluation import TrainingStepMetrics, BatchMetrics
        
        batch_metrics = BatchMetrics(
            loss=0.5,
            perplexity=1.65,
            token_accuracy=0.85,
            num_tokens=1024,
            sequence_length=128,
        )
        step_metrics = TrainingStepMetrics(
            step=100,
            timestamp=1234567890.0,
            batch_metrics=batch_metrics,
            learning_rate=0.001,
        )
        logger.log_step(100, step_metrics, force=True)
        
        summary = logger.summary()
        
        assert "TRAINING SUMMARY" in summary
        assert "test_run" in summary


class TestValidationRunner:
    """Test validation loop runner."""
    
    def test_validation_run(self):
        """Test validation loop execution."""
        from core.evaluation import ValidationRunner, EvaluationMetrics
        
        evaluator = EvaluationMetrics(vocab_size=100)
        
        # Mock model function
        def mock_model(params, inputs):
            batch_size, seq_len = inputs.shape
            return jax.random.normal(
                jax.random.PRNGKey(0), 
                (batch_size, seq_len, 100)
            )
        
        runner = ValidationRunner(mock_model, evaluator)
        
        # Create small validation set
        val_data = [
            (jnp.ones((2, 10), dtype=jnp.int32), 
             jnp.ones((2, 10), dtype=jnp.int32))
            for _ in range(5)
        ]
        
        metrics = runner.run_validation(params={}, val_data=val_data)
        
        assert metrics.num_batches == 5
        assert metrics.perplexity >= 1.0
        assert 0 <= metrics.token_accuracy <= 1.0


class TestTrainingEvaluator:
    """Test high-level TrainingEvaluator integration."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    def test_evaluator_creation(self, temp_dir):
        """Test TrainingEvaluator can be created."""
        from core.evaluation import TrainingEvaluator
        
        evaluator = TrainingEvaluator(
            vocab_size=100,
            log_dir=temp_dir,
            experiment_name="test"
        )
        
        assert evaluator is not None
    
    def test_on_train_step(self, temp_dir):
        """Test processing a training step."""
        from core.evaluation import TrainingEvaluator
        
        evaluator = TrainingEvaluator(
            vocab_size=100,
            log_dir=temp_dir,
            log_every_n_steps=1,
            console_log_every_n_steps=1000,  # Suppress console
        )
        
        # Create sample data
        logits = jax.random.normal(jax.random.PRNGKey(0), (2, 10, 100))
        targets = jax.random.randint(jax.random.PRNGKey(1), (2, 10), 0, 100)
        grads = {'layer': jnp.ones((10, 10)) * 0.1}
        
        metrics = evaluator.on_train_step(
            step=1,
            loss=0.5,
            logits=logits,
            targets=targets,
            learning_rate=0.001,
            grads=grads,
        )
        
        assert metrics.step == 1
        assert metrics.batch_metrics.loss == 0.5
        assert metrics.gradient_metrics is not None
    
    def test_should_validate(self, temp_dir):
        """Test validation timing check."""
        from core.evaluation import TrainingEvaluator
        
        evaluator = TrainingEvaluator(
            vocab_size=100,
            log_dir=temp_dir,
            validate_every_n_steps=100,
        )
        
        assert not evaluator.should_validate(0)
        assert not evaluator.should_validate(50)
        assert evaluator.should_validate(100)
        assert not evaluator.should_validate(150)
        assert evaluator.should_validate(200)
    
    def test_full_integration(self, temp_dir):
        """Test full training loop integration."""
        from core.evaluation import TrainingEvaluator
        
        evaluator = TrainingEvaluator(
            vocab_size=100,
            log_dir=temp_dir,
            experiment_name="integration_test",
            log_every_n_steps=1,
            console_log_every_n_steps=1000,  # Suppress console
            validate_every_n_steps=5,
            config={'test': True}
        )
        
        # Simulate training loop
        for step in range(10):
            logits = jax.random.normal(jax.random.PRNGKey(step), (2, 10, 100))
            targets = jax.random.randint(jax.random.PRNGKey(step+100), (2, 10), 0, 100)
            grads = {'layer': jnp.ones((10, 10)) * (0.1 / (step + 1))}
            
            evaluator.on_train_step(
                step=step,
                loss=1.0 / (step + 1),
                logits=logits,
                targets=targets,
                learning_rate=0.001 * (0.99 ** step),
                grads=grads,
            )
        
        # Check summary works
        summary = evaluator.summary()
        assert "integration_test" in summary
        
        # Check best metrics
        best = evaluator.get_best_metrics()
        assert 'best_loss' in best


class TestSamplingDevUtility:
    """Test that sampling module is marked as dev utility."""
    
    def test_sampling_module_marked(self):
        """Test sampling module has dev utility markers."""
        import core.sampling as sampling
        
        assert hasattr(sampling, '_MODULE_DEV_UTILITY')
        assert sampling._MODULE_DEV_UTILITY is True
        assert hasattr(sampling, '_MODULE_DEV_REASON')


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
