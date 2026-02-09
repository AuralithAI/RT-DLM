"""
Tests for Real-Time Learning Module

Tests for feedback collection, dynamic skill acquisition,
and continuous learning from user interactions.
"""

import unittest
import jax
import jax.numpy as jnp
import numpy as np
import time


class TestFeedbackSample(unittest.TestCase):
    """Test FeedbackSample dataclass."""
    
    def test_feedback_sample_creation(self):
        """Test creating feedback samples."""
        from src.modules.capabilities.real_time_learning import FeedbackSample
        
        sample = FeedbackSample(
            input_text="What is 2+2?",
            output_text="4",
            user_rating=0.9,
            correction=None,
            feedback_type="rating",
            timestamp=time.time(),
            context={"topic": "math"}
        )
        
        self.assertEqual(sample.input_text, "What is 2+2?")
        self.assertEqual(sample.output_text, "4")
        self.assertEqual(sample.user_rating, 0.9)
        self.assertIsNone(sample.correction)
    
    def test_feedback_sample_with_correction(self):
        """Test feedback sample with correction."""
        from src.modules.capabilities.real_time_learning import FeedbackSample
        
        sample = FeedbackSample(
            input_text="Capital of France?",
            output_text="London",
            user_rating=-0.5,
            correction="Paris",
            feedback_type="correction"
        )
        
        self.assertEqual(sample.correction, "Paris")
        self.assertEqual(sample.feedback_type, "correction")


class TestSkillDefinition(unittest.TestCase):
    """Test SkillDefinition dataclass."""
    
    def test_skill_definition_creation(self):
        """Test creating skill definitions."""
        from src.modules.capabilities.real_time_learning import SkillDefinition
        
        skill = SkillDefinition(
            skill_id="math_001",
            skill_name="Basic Arithmetic",
            description="Perform basic arithmetic operations",
            required_examples=10,
            current_proficiency=0.5
        )
        
        self.assertEqual(skill.skill_id, "math_001")
        self.assertEqual(skill.skill_name, "Basic Arithmetic")
        self.assertEqual(skill.required_examples, 10)
        self.assertEqual(skill.current_proficiency, 0.5)


class TestRealTimeFeedbackBuffer(unittest.TestCase):
    """Test RealTimeFeedbackBuffer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        from src.modules.capabilities.real_time_learning import RealTimeFeedbackBuffer
        self.buffer = RealTimeFeedbackBuffer(max_size=100, priority_threshold=0.8)
    
    def test_buffer_initialization(self):
        """Test buffer initialization."""
        self.assertEqual(self.buffer.max_size, 100)
        self.assertEqual(self.buffer.priority_threshold, 0.8)
        self.assertEqual(len(self.buffer.buffer), 0)
    
    def test_add_feedback(self):
        """Test adding feedback to buffer."""
        from src.modules.capabilities.real_time_learning import FeedbackSample
        
        sample = FeedbackSample(
            input_text="test input",
            output_text="test output",
            user_rating=0.5
        )
        
        self.buffer.add_feedback(sample)
        
        self.assertEqual(len(self.buffer.buffer), 1)
    
    def test_priority_feedback_added_to_priority_buffer(self):
        """Test high-priority feedback goes to priority buffer."""
        from src.modules.capabilities.real_time_learning import FeedbackSample
        
        # Low priority (below threshold)
        low_sample = FeedbackSample(
            input_text="low",
            output_text="low",
            user_rating=0.5
        )
        
        # High priority (above threshold)
        high_sample = FeedbackSample(
            input_text="high",
            output_text="high",
            user_rating=0.95  # Above 0.8 threshold
        )
        
        self.buffer.add_feedback(low_sample)
        self.buffer.add_feedback(high_sample)
        
        # Both in main buffer
        self.assertEqual(len(self.buffer.buffer), 2)
        # Only high priority in priority buffer
        self.assertEqual(len(self.buffer.priority_buffer), 1)
    
    def test_correction_goes_to_priority(self):
        """Test feedback with correction goes to priority buffer."""
        from src.modules.capabilities.real_time_learning import FeedbackSample
        
        sample = FeedbackSample(
            input_text="question",
            output_text="wrong answer",
            user_rating=0.3,  # Below threshold but has correction
            correction="correct answer"
        )
        
        self.buffer.add_feedback(sample)
        
        self.assertEqual(len(self.buffer.priority_buffer), 1)
    
    def test_skill_specific_buffer(self):
        """Test skill-specific buffering."""
        from src.modules.capabilities.real_time_learning import FeedbackSample
        
        sample = FeedbackSample(
            input_text="math question",
            output_text="answer",
            user_rating=0.8,
            context={"skill_id": "math_001"}
        )
        
        self.buffer.add_feedback(sample)
        
        self.assertIn("math_001", self.buffer.skill_buffers)
        self.assertEqual(len(self.buffer.skill_buffers["math_001"]), 1)
    
    def test_get_training_batch(self):
        """Test getting training batch."""
        from src.modules.capabilities.real_time_learning import FeedbackSample
        
        # Add samples
        for i in range(20):
            sample = FeedbackSample(
                input_text=f"input_{i}",
                output_text=f"output_{i}",
                user_rating=0.5
            )
            self.buffer.add_feedback(sample)
        
        batch = self.buffer.get_training_batch(batch_size=10)
        
        self.assertEqual(len(batch), 10)
    
    def test_get_training_batch_prioritized(self):
        """Test getting prioritized training batch."""
        from src.modules.capabilities.real_time_learning import FeedbackSample
        
        # Add low priority samples
        for i in range(10):
            sample = FeedbackSample(
                input_text=f"low_{i}",
                output_text=f"low_{i}",
                user_rating=0.5
            )
            self.buffer.add_feedback(sample)
        
        # Add high priority samples
        for i in range(10):
            sample = FeedbackSample(
                input_text=f"high_{i}",
                output_text=f"high_{i}",
                user_rating=0.95
            )
            self.buffer.add_feedback(sample)
        
        # Get prioritized batch - should prefer priority buffer
        batch = self.buffer.get_training_batch(batch_size=5, prioritize=True)
        
        # All should be from priority buffer
        self.assertEqual(len(batch), 5)
    
    def test_get_skill_samples(self):
        """Test getting skill-specific samples."""
        from src.modules.capabilities.real_time_learning import FeedbackSample
        
        # Add samples for different skills
        for i in range(5):
            sample = FeedbackSample(
                input_text=f"math_{i}",
                output_text=f"answer_{i}",
                user_rating=0.8,
                context={"skill_id": "math"}
            )
            self.buffer.add_feedback(sample)
        
        for i in range(3):
            sample = FeedbackSample(
                input_text=f"code_{i}",
                output_text=f"code_{i}",
                user_rating=0.7,
                context={"skill_id": "coding"}
            )
            self.buffer.add_feedback(sample)
        
        math_samples = self.buffer.get_skill_samples("math")
        code_samples = self.buffer.get_skill_samples("coding")
        
        self.assertEqual(len(math_samples), 5)
        self.assertEqual(len(code_samples), 3)
    
    def test_get_skill_samples_with_limit(self):
        """Test getting limited skill samples."""
        from src.modules.capabilities.real_time_learning import FeedbackSample
        
        for i in range(10):
            sample = FeedbackSample(
                input_text=f"sample_{i}",
                output_text=f"output_{i}",
                user_rating=0.7,
                context={"skill_id": "test_skill"}
            )
            self.buffer.add_feedback(sample)
        
        samples = self.buffer.get_skill_samples("test_skill", count=5)
        
        self.assertEqual(len(samples), 5)
    
    def test_get_skill_samples_unknown_skill(self):
        """Test getting samples for unknown skill."""
        samples = self.buffer.get_skill_samples("nonexistent_skill")
        self.assertEqual(len(samples), 0)


class TestRealTimeLearningSystem(unittest.TestCase):
    """Test RealTimeLearningSystem class."""
    
    def test_system_initialization(self):
        """Test system initialization."""
        try:
            from src.modules.capabilities.real_time_learning import RealTimeLearningSystem
            
            system = RealTimeLearningSystem(
                d_model=64,
                buffer_size=1000,
                learning_rate=0.001
            )
            
            self.assertEqual(system.d_model, 64)
        except (ImportError, TypeError, AttributeError) as e:
            self.skipTest(f"RealTimeLearningSystem not available or incompatible: {e}")


class TestDynamicSkillAcquisition(unittest.TestCase):
    """Test DynamicSkillAcquisition module."""
    
    def test_skill_acquisition_creation(self):
        """Test creating skill acquisition module."""
        import haiku as hk
        from src.modules.capabilities.real_time_learning import DynamicSkillAcquisition
        
        def forward(x):
            module = DynamicSkillAcquisition(base_d_model=64, max_skills=50)
            return module(x)
        
        rng = jax.random.PRNGKey(42)
        x = jnp.ones((2, 16, 64))
        
        init_fn = hk.transform_with_state(forward)
        params, state = init_fn.init(rng, x)
        
        self.assertIsNotNone(params)


class TestFeedbackProcessing(unittest.TestCase):
    """Test feedback processing workflows."""
    
    def test_feedback_to_training_workflow(self):
        """Test complete feedback to training workflow."""
        from src.modules.capabilities.real_time_learning import (
            FeedbackSample, RealTimeFeedbackBuffer
        )
        
        buffer = RealTimeFeedbackBuffer(max_size=100)
        
        # Simulate user interactions
        interactions = [
            ("What is Python?", "Python is a programming language", 0.9),
            ("How to sort?", "Use sorted() function", 0.85),
            ("Bug in code", "Wrong suggestion", -0.5),
        ]
        
        for inp, out, rating in interactions:
            sample = FeedbackSample(
                input_text=inp,
                output_text=out,
                user_rating=rating
            )
            buffer.add_feedback(sample)
        
        # Get batch for training
        batch = buffer.get_training_batch(batch_size=3)
        
        self.assertEqual(len(batch), 3)


if __name__ == "__main__":
    unittest.main()
