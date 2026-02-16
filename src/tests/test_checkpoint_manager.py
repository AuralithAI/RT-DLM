"""
Tests for Checkpoint Manager Module

Tests for secure model checkpointing using SafeTensors format,
including save, load, and metadata management.
"""

import unittest
import tempfile
import shutil
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np


class TestCheckpointMetadata(unittest.TestCase):
    """Test CheckpointMetadata dataclass."""
    
    def test_metadata_creation(self):
        """Test creating checkpoint metadata."""
        from src.core.checkpoint_manager import CheckpointMetadata
        
        metadata = CheckpointMetadata(
            epoch=10,
            step_count=5000,
            timestamp="2024-01-01T12:00:00",
            model_name="test_model",
            training_losses=[0.5, 0.4, 0.3],
            metrics={"accuracy": 0.95}
        )
        
        self.assertEqual(metadata.epoch, 10)
        self.assertEqual(metadata.step_count, 5000)
        self.assertEqual(metadata.model_name, "test_model")
        self.assertEqual(metadata.framework, "jax")
    
    def test_metadata_to_dict(self):
        """Test converting metadata to dictionary."""
        from src.core.checkpoint_manager import CheckpointMetadata
        
        metadata = CheckpointMetadata(
            epoch=5,
            step_count=1000,
            timestamp="2024-01-01",
            training_losses=[0.5, 0.4]
        )
        
        d = metadata.to_dict()
        
        self.assertIsInstance(d, dict)
        self.assertEqual(d["epoch"], 5)
        self.assertEqual(d["step_count"], 1000)
        self.assertEqual(d["training_losses"], [0.5, 0.4])
    
    def test_metadata_from_dict(self):
        """Test creating metadata from dictionary."""
        from src.core.checkpoint_manager import CheckpointMetadata
        
        data = {
            "epoch": 15,
            "step_count": 7500,
            "timestamp": "2024-02-01",
            "model_name": "loaded_model",
            "metrics": {"loss": 0.1}
        }
        
        metadata = CheckpointMetadata.from_dict(data)
        
        self.assertEqual(metadata.epoch, 15)
        self.assertEqual(metadata.step_count, 7500)
        self.assertEqual(metadata.model_name, "loaded_model")


class TestFlattenParams(unittest.TestCase):
    """Test parameter flattening utilities."""
    
    def test_flatten_simple_params(self):
        """Test flattening simple parameter dictionary."""
        from src.core.checkpoint_manager import flatten_params
        
        params = {
            "layer1": {
                "weight": jnp.ones((10, 10)),
                "bias": jnp.zeros(10)
            },
            "layer2": {
                "weight": jnp.ones((5, 10)),
                "bias": jnp.zeros(5)
            }
        }
        
        flat = flatten_params(params, prefix="params")
        
        self.assertIn("params.layer1.weight", flat)
        self.assertIn("params.layer1.bias", flat)
        self.assertIn("params.layer2.weight", flat)
        self.assertEqual(flat["params.layer1.weight"].shape, (10, 10))
    
    def test_flatten_deep_params(self):
        """Test flattening deeply nested parameters."""
        from src.core.checkpoint_manager import flatten_params
        
        params = {
            "encoder": {
                "layer0": {
                    "attention": {
                        "query": jnp.ones((64, 64)),
                        "key": jnp.ones((64, 64)),
                        "value": jnp.ones((64, 64))
                    }
                }
            }
        }
        
        flat = flatten_params(params)
        
        self.assertIn("encoder.layer0.attention.query", flat)
        self.assertIn("encoder.layer0.attention.key", flat)


class TestUnflattenParams(unittest.TestCase):
    """Test parameter unflattening utilities."""
    
    def test_unflatten_params(self):
        """Test unflattening parameters back to nested structure."""
        from src.core.checkpoint_manager import flatten_params, unflatten_params
        
        original = {
            "layer1": {
                "weight": jnp.ones((10, 10)),
                "bias": jnp.zeros(10)
            }
        }
        
        flat = flatten_params(original)
        restored = unflatten_params(flat)
        
        self.assertIn("layer1", restored)
        self.assertIn("weight", restored["layer1"])
        self.assertEqual(restored["layer1"]["weight"].shape, (10, 10))


class TestCheckpointManager(unittest.TestCase):
    """Test CheckpointManager class."""
    
    def setUp(self):
        """Set up test fixtures with temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_dir = Path(self.temp_dir) / "checkpoints"
    
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_manager_initialization(self):
        """Test checkpoint manager initialization."""
        from src.core.checkpoint_manager import CheckpointManager
        
        manager = CheckpointManager(
            checkpoint_dir=str(self.checkpoint_dir),
            model_name="test_model",
            keep_last_n=3
        )
        
        self.assertEqual(manager.model_name, "test_model")
        self.assertEqual(manager.keep_last_n, 3)
        self.assertTrue(self.checkpoint_dir.exists())
    
    def test_save_checkpoint(self):
        """Test saving checkpoint."""
        from src.core.checkpoint_manager import CheckpointManager
        
        manager = CheckpointManager(
            checkpoint_dir=str(self.checkpoint_dir),
            model_name="save_test"
        )
        
        params = {
            "layer": {
                "weight": jnp.ones((10, 10)),
                "bias": jnp.zeros(10)
            }
        }
        
        # Create simple optimizer state
        import optax
        optimizer = optax.adam(0.001)
        opt_state = optimizer.init(params)
        
        path = manager.save_checkpoint(
            params=params,
            opt_state=opt_state,
            epoch=5,
            step_count=1000,
            metrics={"loss": 0.1}
        )
        
        self.assertTrue(Path(path).exists())
        self.assertTrue(Path(path).with_suffix(".json").exists())
    
    def test_load_checkpoint(self):
        """Test loading checkpoint."""
        from src.core.checkpoint_manager import CheckpointManager
        
        manager = CheckpointManager(
            checkpoint_dir=str(self.checkpoint_dir),
            model_name="load_test"
        )
        
        params = {
            "layer": {
                "weight": jnp.array([[1.0, 2.0], [3.0, 4.0]]),
                "bias": jnp.array([0.1, 0.2])
            }
        }
        
        import optax
        optimizer = optax.adam(0.001)
        opt_state = optimizer.init(params)
        
        # Save
        path = manager.save_checkpoint(
            params=params,
            opt_state=opt_state,
            epoch=3,
            step_count=500
        )
        
        # Load
        checkpoint = manager.load_checkpoint(
            checkpoint_path=path,
            load_opt_state=False
        )
        
        self.assertIn("params", checkpoint)
        self.assertIn("epoch", checkpoint)
        self.assertEqual(checkpoint["epoch"], 3)
    
    def test_load_latest_checkpoint(self):
        """Test loading latest checkpoint when no path specified."""
        from src.core.checkpoint_manager import CheckpointManager
        
        manager = CheckpointManager(
            checkpoint_dir=str(self.checkpoint_dir),
            model_name="latest_test"
        )
        
        params = {"w": jnp.ones((5, 5))}
        import optax
        opt_state = optax.adam(0.001).init(params)
        
        # Save multiple checkpoints
        for epoch in [1, 2, 3]:
            manager.save_checkpoint(
                params=params,
                opt_state=opt_state,
                epoch=epoch
            )
        
        # Load latest (should be epoch 3)
        checkpoint = manager.load_checkpoint(load_opt_state=False)
        
        self.assertEqual(checkpoint["epoch"], 3)
    
    def test_checkpoint_cleanup(self):
        """Test old checkpoint cleanup."""
        from src.core.checkpoint_manager import CheckpointManager
        
        manager = CheckpointManager(
            checkpoint_dir=str(self.checkpoint_dir),
            model_name="cleanup_test",
            keep_last_n=2
        )
        
        params = {"w": jnp.ones((3, 3))}
        import optax
        opt_state = optax.adam(0.001).init(params)
        
        # Save more checkpoints than keep_last_n
        for epoch in range(5):
            manager.save_checkpoint(
                params=params,
                opt_state=opt_state,
                epoch=epoch
            )
        
        # Check that only keep_last_n checkpoints remain
        checkpoints = list(self.checkpoint_dir.glob("*.safetensors"))
        self.assertLessEqual(len(checkpoints), 2)


class TestCheckpointWithMetrics(unittest.TestCase):
    """Test checkpoint saving with various metrics."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_save_with_training_history(self):
        """Test saving checkpoint with training history."""
        from src.core.checkpoint_manager import CheckpointManager
        
        manager = CheckpointManager(
            checkpoint_dir=self.temp_dir,
            model_name="history_test"
        )
        
        params = {"w": jnp.ones((2, 2))}
        import optax
        opt_state = optax.adam(0.001).init(params)
        
        training_losses = [1.0, 0.8, 0.6, 0.4, 0.2]
        validation_losses = [1.1, 0.9, 0.7, 0.5, 0.3]
        
        path = manager.save_checkpoint(
            params=params,
            opt_state=opt_state,
            epoch=5,
            training_losses=training_losses,
            validation_losses=validation_losses,
            metrics={"final_loss": 0.2, "accuracy": 0.95}
        )
        
        # Load and verify
        checkpoint = manager.load_checkpoint(path, load_opt_state=False)
        
        self.assertIn("metadata", checkpoint)


class TestCheckpointExtraTensors(unittest.TestCase):
    """Test saving extra tensors with checkpoint."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_save_extra_tensors(self):
        """Test saving additional tensors."""
        from src.core.checkpoint_manager import CheckpointManager
        
        manager = CheckpointManager(
            checkpoint_dir=self.temp_dir,
            model_name="extra_test"
        )
        
        params = {"w": jnp.ones((2, 2))}
        import optax
        opt_state = optax.adam(0.001).init(params)
        
        extra = {
            "running_mean": np.zeros(10),
            "running_var": np.ones(10)
        }
        
        path = manager.save_checkpoint(
            params=params,
            opt_state=opt_state,
            epoch=1,
            extra_tensors=extra
        )
        
        self.assertTrue(Path(path).exists())


if __name__ == "__main__":
    unittest.main()
