"""
Tests for Advanced Algorithms Module

Tests for continual learning algorithms including EWC, SI, 
Progressive Neural Networks, and MAS.
"""

import unittest
import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk


class TestTaskMemory(unittest.TestCase):
    """Test TaskMemory dataclass."""
    
    def test_task_memory_creation(self):
        """Test creating TaskMemory."""
        from src.modules.capabilities.advanced_algorithms import TaskMemory
        
        memory = TaskMemory(
            task_id="task_1",
            params_snapshot={"layer": jnp.zeros((10, 10))},
            fisher_matrix={"layer": jnp.ones((10, 10))},
            importance_weights={"layer": jnp.ones((10, 10))},
            performance_metrics={"accuracy": 0.95},
            num_samples=1000
        )
        
        self.assertEqual(memory.task_id, "task_1")
        self.assertEqual(memory.num_samples, 1000)
        self.assertIn("accuracy", memory.performance_metrics)
    
    def test_task_memory_params(self):
        """Test TaskMemory parameter storage."""
        from src.modules.capabilities.advanced_algorithms import TaskMemory
        
        params = {
            "layer1": jnp.ones((5, 5)),
            "layer2": jnp.zeros((3, 3))
        }
        
        memory = TaskMemory(
            task_id="test",
            params_snapshot=params,
            fisher_matrix={},
            importance_weights={},
            performance_metrics={},
            num_samples=100
        )
        
        self.assertEqual(memory.params_snapshot["layer1"].shape, (5, 5))


class TestFisherInformation(unittest.TestCase):
    """Test Fisher Information computation."""
    
    def test_compute_fisher_information_structure(self):
        """Test Fisher information computation returns correct structure."""
        from src.modules.capabilities.advanced_algorithms import compute_fisher_information
        
        # Create simple model that matches expected signature
        def model_fn(params, rng, inputs):
            x = inputs["text"].astype(jnp.float32)
            return jnp.matmul(x, params["w"]) + params["b"]
        
        params = {
            "w": jax.random.normal(jax.random.PRNGKey(0), (10, 5)),
            "b": jnp.zeros(5)
        }
        
        # Data should be 2D: (num_samples, seq_len) for token IDs or features
        data = jax.random.normal(jax.random.PRNGKey(1), (16, 10))
        # Targets should match: (num_samples, seq_len) for sequence tasks
        targets = jax.random.randint(jax.random.PRNGKey(2), (16, 10), 0, 5)
        
        rng = jax.random.PRNGKey(42)
        
        fisher = compute_fisher_information(
            model_fn=model_fn,
            params=params,
            data_samples=data,
            targets=targets,
            rng=rng,
            num_samples=10
        )
        
        # Fisher should have same structure as params
        self.assertIn("w", fisher)
        self.assertIn("b", fisher)


class TestEWCLoss(unittest.TestCase):
    """Test Elastic Weight Consolidation loss."""
    
    def test_ewc_loss_computation(self):
        """Test EWC loss computation."""
        try:
            from src.modules.capabilities.advanced_algorithms import compute_ewc_loss
            
            # Old params (from previous task)
            params_star = {
                "w": jnp.ones((5, 5)),
                "b": jnp.zeros(5)
            }
            
            # New params (current)
            params = {
                "w": jnp.ones((5, 5)) * 1.1,
                "b": jnp.ones(5) * 0.1
            }
            
            # Fisher diagonal
            fisher_matrix = {
                "w": jnp.ones((5, 5)) * 100,
                "b": jnp.ones(5) * 50
            }
            
            ewc_loss = compute_ewc_loss(
                params=params,
                params_star=params_star,
                fisher_matrix=fisher_matrix,
                lambda_ewc=1.0
            )
            
            # EWC loss should be positive when params differ
            self.assertGreater(float(ewc_loss), 0)
        except ImportError:
            self.skipTest("compute_ewc_loss not available")
    
    def test_ewc_loss_zero_when_same_params(self):
        """Test EWC loss is zero when params unchanged."""
        try:
            from src.modules.capabilities.advanced_algorithms import compute_ewc_loss
            
            params = {
                "w": jnp.ones((5, 5)),
                "b": jnp.zeros(5)
            }
            
            fisher_matrix = {
                "w": jnp.ones((5, 5)),
                "b": jnp.ones(5)
            }
            
            ewc_loss = compute_ewc_loss(
                params=params,
                params_star=params,
                fisher_matrix=fisher_matrix,
                lambda_ewc=1.0
            )
            
            # Should be zero or very small
            self.assertLess(float(ewc_loss), 1e-6)
        except ImportError:
            self.skipTest("compute_ewc_loss not available")


class TestSynapticIntelligence(unittest.TestCase):
    """Test Synaptic Intelligence (SI) algorithm."""
    
    def test_si_module_exists(self):
        """Test SynapticIntelligence module exists and can be instantiated."""
        import haiku as hk
        from src.modules.capabilities.advanced_algorithms import SynapticIntelligence
        
        def forward(features, gradients):
            si = SynapticIntelligence(d_model=64, lambda_si=1.0, damping=0.1)
            return si(features, gradients)
        
        init = hk.transform(forward)
        rng = jax.random.PRNGKey(42)
        features = jax.random.normal(rng, (2, 16, 64))
        gradients = jax.random.normal(rng, (2, 16, 64))
        
        params = init.init(rng, features, gradients)
        self.assertIsNotNone(params)


class TestProgressiveNetworks(unittest.TestCase):
    """Test Progressive Neural Networks."""
    
    def test_progressive_network_exists(self):
        """Test ProgressiveNeuralNetwork class exists."""
        import haiku as hk
        from src.modules.capabilities.advanced_algorithms import ProgressiveNeuralNetwork
        
        def forward(x):
            pnn = ProgressiveNeuralNetwork(d_model=64, max_columns=5)
            return pnn(x, column_idx=0)
        
        init = hk.transform(forward)
        rng = jax.random.PRNGKey(42)
        x = jax.random.normal(rng, (2, 16, 64))
        
        params = init.init(rng, x)
        self.assertIsNotNone(params)


class TestMemoryAwareSynapses(unittest.TestCase):
    """Test Memory-Aware Synapses (MAS)."""
    
    def test_mas_module_exists(self):
        """Test MemoryAwareSynapses module exists."""
        import haiku as hk
        from src.modules.capabilities.advanced_algorithms import MemoryAwareSynapses
        
        def forward(features, output_gradients):
            mas = MemoryAwareSynapses(d_model=64)
            return mas(features, output_gradients)
        
        init = hk.transform(forward)
        rng = jax.random.PRNGKey(42)
        features = jax.random.normal(rng, (2, 16, 64))
        output_gradients = jax.random.normal(rng, (2, 16, 64))
        
        params = init.init(rng, features, output_gradients)
        self.assertIsNotNone(params)


class TestContinualLearner(unittest.TestCase):
    """Test ContinualLearner class (replaces ContinualLearningManager)."""
    
    def test_learner_initialization(self):
        """Test ContinualLearner initialization."""
        import haiku as hk
        from src.modules.capabilities.advanced_algorithms import ContinualLearner
        
        def forward(x):
            learner = ContinualLearner(
                d_model=64,
                lambda_ewc=400.0,
                lambda_si=1.0,
                max_tasks=5
            )
            return learner(x, task_embedding=None)
        
        init = hk.transform(forward)
        rng = jax.random.PRNGKey(42)
        x = jax.random.normal(rng, (2, 16, 64))
        
        params = init.init(rng, x)
        self.assertIsNotNone(params)


class TestImportanceWeights(unittest.TestCase):
    """Test importance weight computations."""
    
    def test_importance_weight_shape(self):
        """Test importance weights match parameter shapes."""
        from src.modules.capabilities.advanced_algorithms import TaskMemory
        
        params = {"layer": jnp.ones((10, 10))}
        importance = {"layer": jnp.ones((10, 10)) * 0.5}
        
        memory = TaskMemory(
            task_id="test",
            params_snapshot=params,
            fisher_matrix={},
            importance_weights=importance,
            performance_metrics={},
            num_samples=100
        )
        
        self.assertEqual(
            memory.params_snapshot["layer"].shape,
            memory.importance_weights["layer"].shape
        )


class TestRegularizationStrength(unittest.TestCase):
    """Test regularization strength control."""
    
    def test_lambda_scaling(self):
        """Test EWC lambda affects loss magnitude."""
        try:
            from src.modules.capabilities.advanced_algorithms import compute_ewc_loss
            
            params_star = {"w": jnp.zeros((3, 3))}
            params = {"w": jnp.ones((3, 3))}
            fisher_matrix = {"w": jnp.ones((3, 3))}
            
            loss_low = compute_ewc_loss(
                params=params,
                params_star=params_star,
                fisher_matrix=fisher_matrix,
                lambda_ewc=1.0
            )
            
            loss_high = compute_ewc_loss(
                params=params,
                params_star=params_star,
                fisher_matrix=fisher_matrix,
                lambda_ewc=10.0
            )
            
            # Higher lambda should give higher loss
            self.assertGreater(float(loss_high), float(loss_low))
        except ImportError:
            self.skipTest("compute_ewc_loss not available")


if __name__ == "__main__":
    unittest.main()
