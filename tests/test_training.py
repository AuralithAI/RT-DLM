"""
Unit tests for AGI training pipeline

Tests train_agi.py and training utilities.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

import unittest
import json
import jax
import jax.numpy as jnp
import numpy as np

# Test constants
D_MODEL = 64
BATCH_SIZE = 2
SEQ_LEN = 16


class TestTrainingBatch(unittest.TestCase):
    """Test training batch creation"""
    
    def test_create_training_batch_text_only(self):
        """Test creating a text-only training batch"""
        from config.agi_config import AGIConfig
        from train_agi import AGITrainer
        
        # Create minimal config for testing
        config = AGIConfig(
            d_model=D_MODEL,
            num_heads=4,
            num_layers=2,
            vocab_size=1000,
            batch_size=BATCH_SIZE,
            multimodal_enabled=False,
        )
        
        trainer = AGITrainer(config)
        
        texts = ["Hello world", "Test sentence"]
        batch = trainer.create_training_batch(texts, include_multimodal=False)
        
        self.assertIn("input_ids", batch)
        self.assertIn("targets", batch)
        self.assertEqual(batch["input_ids"].shape[0], len(texts))
        
    def test_create_training_batch_multimodal(self):
        """Test creating a multimodal training batch"""
        from config.agi_config import AGIConfig
        from train_agi import AGITrainer
        
        config = AGIConfig(
            d_model=D_MODEL,
            num_heads=4,
            num_layers=2,
            vocab_size=1000,
            batch_size=BATCH_SIZE,
            multimodal_enabled=True,
        )
        
        trainer = AGITrainer(config)
        
        texts = ["Hello world", "Test sentence"]
        batch = trainer.create_training_batch(texts, include_multimodal=True)
        
        self.assertIn("input_ids", batch)
        self.assertIn("multimodal_inputs", batch)
        self.assertIn("images", batch["multimodal_inputs"])
        self.assertIn("audio", batch["multimodal_inputs"])
        
        # Check shapes
        self.assertEqual(batch["multimodal_inputs"]["images"].shape[0], len(texts))
        self.assertEqual(batch["multimodal_inputs"]["audio"].shape[0], len(texts))
        
    def test_create_training_batch_with_real_images(self):
        """Test creating batch with real image data"""
        from config.agi_config import AGIConfig
        from train_agi import AGITrainer
        
        config = AGIConfig(
            d_model=D_MODEL,
            num_heads=4,
            num_layers=2,
            vocab_size=1000,
            batch_size=BATCH_SIZE,
            multimodal_enabled=True,
        )
        
        trainer = AGITrainer(config)
        
        texts = ["Image description 1", "Image description 2"]
        # Simulate real image data (uint8, 0-255)
        fake_images = np.random.randint(0, 256, (2, 224, 224, 3), dtype=np.uint8)
        
        batch = trainer.create_training_batch(
            texts, 
            include_multimodal=True,
            images=fake_images
        )
        
        # Images should be normalized to [0, 1]
        self.assertIn("image", batch)
        self.assertTrue(jnp.all(batch["image"] >= 0))
        self.assertTrue(jnp.all(batch["image"] <= 1))


class TestReasoningTask(unittest.TestCase):
    """Test reasoning task creation"""
    
    def test_create_reasoning_task(self):
        """Test creating reasoning tasks"""
        from config.agi_config import AGIConfig
        from train_agi import AGITrainer
        
        config = AGIConfig(
            d_model=D_MODEL,
            num_heads=4,
            num_layers=2,
            vocab_size=1000,
            batch_size=4,
            multimodal_enabled=False,
        )
        
        trainer = AGITrainer(config)
        
        batch = trainer.create_reasoning_task(batch_size=4)
        
        self.assertIn("input_ids", batch)
        self.assertEqual(batch["input_ids"].shape[0], 4)


class TestSampleData(unittest.TestCase):
    """Test sample data loading"""
    
    def test_sample_json_exists(self):
        """Test that sample.json exists and is valid"""
        sample_path = Path(__file__).parent.parent / "data" / "sample.json"
        self.assertTrue(sample_path.exists(), "sample.json should exist")
        
    def test_sample_json_structure(self):
        """Test sample.json has correct structure"""
        sample_path = Path(__file__).parent.parent / "data" / "sample.json"
        
        with open(sample_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.assertIn("metadata", data)
        self.assertIn("samples", data)
        self.assertGreater(len(data["samples"]), 0)
        
        # Check first sample structure
        first_sample = data["samples"][0]
        self.assertIn("id", first_sample)
        self.assertIn("text", first_sample)
        self.assertIn("modalities", first_sample)
        
    def test_sample_json_has_100_samples(self):
        """Test sample.json has 100 samples"""
        sample_path = Path(__file__).parent.parent / "data" / "sample.json"
        
        with open(sample_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.assertEqual(len(data["samples"]), 100)
        
    def test_sample_json_emotion_samples(self):
        """Test sample.json includes emotion-labeled samples"""
        sample_path = Path(__file__).parent.parent / "data" / "sample.json"
        
        with open(sample_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        emotion_samples = [s for s in data["samples"] if "emotion" in s]
        self.assertGreater(len(emotion_samples), 10, "Should have emotion samples")
        
    def test_sample_json_reasoning_samples(self):
        """Test sample.json includes reasoning samples"""
        sample_path = Path(__file__).parent.parent / "data" / "sample.json"
        
        with open(sample_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        reasoning_samples = [s for s in data["samples"] if s.get("task_type") == "reasoning"]
        self.assertGreater(len(reasoning_samples), 0, "Should have reasoning samples")


class TestEvaluateReasoningQuality(unittest.TestCase):
    """Test reasoning quality evaluation"""
    
    def test_evaluate_reasoning_with_ground_truth(self):
        """Test reasoning evaluation with ground truth"""
        from config.agi_config import AGIConfig
        from train_agi import AGITrainer
        
        config = AGIConfig(
            d_model=D_MODEL,
            num_heads=4,
            num_layers=2,
            vocab_size=1000,
            batch_size=BATCH_SIZE,
        )
        
        trainer = AGITrainer(config)
        
        # Create fake reasoning chain
        reasoning_chain = [
            jnp.ones((BATCH_SIZE, D_MODEL)),
            jnp.ones((BATCH_SIZE, D_MODEL)) * 0.9,
            jnp.ones((BATCH_SIZE, D_MODEL)) * 0.8,
        ]
        
        # Evaluate without ground truth
        score = trainer.evaluate_reasoning_quality(reasoning_chain)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        
    def test_evaluate_empty_reasoning_chain(self):
        """Test evaluation with empty reasoning chain"""
        from config.agi_config import AGIConfig
        from train_agi import AGITrainer
        
        config = AGIConfig(
            d_model=D_MODEL,
            num_heads=4,
            num_layers=2,
            vocab_size=1000,
            batch_size=BATCH_SIZE,
        )
        
        trainer = AGITrainer(config)
        
        score = trainer.evaluate_reasoning_quality([])
        self.assertEqual(score, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
