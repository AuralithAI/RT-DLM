"""
Tests for Hybrid Video Module

Tests for video processing, temporal encoding, object tracking,
action recognition, and scene understanding.
"""

import unittest
import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk


class TestHybridVideoEncoder(unittest.TestCase):
    """Test HybridVideoEncoder class."""
    
    def test_encoder_initialization(self):
        """Test HybridVideoEncoder initialization."""
        from src.modules.multimodal.hybrid_video_module import HybridVideoEncoder
        
        def init_fn():
            encoder = HybridVideoEncoder(
                d_model=64,
                frame_height=224,
                frame_width=224
            )
            # Video: [batch, frames, height, width, channels]
            video = jnp.zeros((1, 8, 224, 224, 3))
            return encoder(video)
        
        init = hk.transform(init_fn)
        rng = jax.random.PRNGKey(42)
        params = init.init(rng)
        
        self.assertIsNotNone(params)
    
    def test_encoder_output_structure(self):
        """Test encoder output structure."""
        from src.modules.multimodal.hybrid_video_module import HybridVideoEncoder
        
        def forward_fn(video):
            encoder = HybridVideoEncoder(d_model=64)
            return encoder(video)
        
        init = hk.transform(forward_fn)
        rng = jax.random.PRNGKey(42)
        video = jax.random.normal(rng, (1, 8, 224, 224, 3))
        
        params = init.init(rng, video)
        output = init.apply(params, rng, video)
        
        # Output should be a dictionary
        self.assertIsInstance(output, dict)
        self.assertIn('primary_features', output)
        self.assertIn('frame_features', output)
    
    def test_task_hint_tracking(self):
        """Test task hint for tracking."""
        from src.modules.multimodal.hybrid_video_module import HybridVideoEncoder
        
        def forward_fn(video):
            encoder = HybridVideoEncoder(d_model=64)
            return encoder(video, task_hint='tracking')
        
        init = hk.transform(forward_fn)
        rng = jax.random.PRNGKey(42)
        video = jax.random.normal(rng, (1, 8, 224, 224, 3))
        
        params = init.init(rng, video)
        output = init.apply(params, rng, video)
        
        self.assertIn('primary_features', output)
    
    def test_task_hint_action(self):
        """Test task hint for action recognition."""
        from src.modules.multimodal.hybrid_video_module import HybridVideoEncoder
        
        def forward_fn(video):
            encoder = HybridVideoEncoder(d_model=64)
            return encoder(video, task_hint='action')
        
        init = hk.transform(forward_fn)
        rng = jax.random.PRNGKey(42)
        video = jax.random.normal(rng, (1, 8, 224, 224, 3))
        
        params = init.init(rng, video)
        output = init.apply(params, rng, video)
        
        self.assertIn('primary_features', output)


class TestHybridFrameEncoder(unittest.TestCase):
    """Test HybridFrameEncoder class."""
    
    def test_frame_encoder_exists(self):
        """Test HybridFrameEncoder exists."""
        try:
            from src.modules.multimodal.hybrid_video_module import HybridFrameEncoder
            self.assertIsNotNone(HybridFrameEncoder)
        except ImportError:
            self.skipTest("HybridFrameEncoder not available")


class TestTemporalEncoder(unittest.TestCase):
    """Test TemporalEncoder class."""
    
    def test_temporal_encoder_exists(self):
        """Test TemporalEncoder exists."""
        try:
            from src.modules.multimodal.hybrid_video_module import TemporalEncoder
            self.assertIsNotNone(TemporalEncoder)
        except ImportError:
            self.skipTest("TemporalEncoder not available")


class TestObjectTrackingModule(unittest.TestCase):
    """Test ObjectTrackingModule class."""
    
    def test_object_tracker_exists(self):
        """Test ObjectTrackingModule exists."""
        try:
            from src.modules.multimodal.hybrid_video_module import ObjectTrackingModule
            self.assertIsNotNone(ObjectTrackingModule)
        except ImportError:
            self.skipTest("ObjectTrackingModule not available")


class TestActionRecognitionModule(unittest.TestCase):
    """Test ActionRecognitionModule class."""
    
    def test_action_recognizer_exists(self):
        """Test ActionRecognitionModule exists."""
        try:
            from src.modules.multimodal.hybrid_video_module import ActionRecognitionModule
            self.assertIsNotNone(ActionRecognitionModule)
        except ImportError:
            self.skipTest("ActionRecognitionModule not available")


class TestSceneUnderstandingModule(unittest.TestCase):
    """Test SceneUnderstandingModule class."""
    
    def test_scene_analyzer_exists(self):
        """Test SceneUnderstandingModule exists."""
        try:
            from src.modules.multimodal.hybrid_video_module import SceneUnderstandingModule
            self.assertIsNotNone(SceneUnderstandingModule)
        except ImportError:
            self.skipTest("SceneUnderstandingModule not available")


class TestMotionAnalysisModule(unittest.TestCase):
    """Test MotionAnalysisModule class."""
    
    def test_motion_analyzer_exists(self):
        """Test MotionAnalysisModule exists."""
        try:
            from src.modules.multimodal.hybrid_video_module import MotionAnalysisModule
            self.assertIsNotNone(MotionAnalysisModule)
        except ImportError:
            self.skipTest("MotionAnalysisModule not available")


class TestMultiScaleFeatureFusion(unittest.TestCase):
    """Test MultiScaleFeatureFusion class."""
    
    def test_feature_fusion_exists(self):
        """Test MultiScaleFeatureFusion exists."""
        try:
            from src.modules.multimodal.hybrid_video_module import MultiScaleFeatureFusion
            self.assertIsNotNone(MultiScaleFeatureFusion)
        except ImportError:
            self.skipTest("MultiScaleFeatureFusion not available")


class TestVideoInputShapes(unittest.TestCase):
    """Test video input shape handling."""
    
    def test_various_frame_counts(self):
        """Test encoder with different frame counts."""
        from src.modules.multimodal.hybrid_video_module import HybridVideoEncoder
        
        for num_frames in [4, 8, 16]:
            def forward_fn(video):
                encoder = HybridVideoEncoder(d_model=64)
                return encoder(video)
            
            init = hk.transform(forward_fn)
            rng = jax.random.PRNGKey(42)
            video = jax.random.normal(rng, (1, num_frames, 224, 224, 3))
            
            params = init.init(rng, video)
            output = init.apply(params, rng, video)
            
            self.assertIsNotNone(output['primary_features'])


class TestNumericalStability(unittest.TestCase):
    """Test numerical stability of video processing."""
    
    def test_no_nan_output(self):
        """Test encoder doesn't produce NaN."""
        from src.modules.multimodal.hybrid_video_module import HybridVideoEncoder
        
        def forward_fn(video):
            encoder = HybridVideoEncoder(d_model=64)
            return encoder(video)
        
        init = hk.transform(forward_fn)
        rng = jax.random.PRNGKey(42)
        video = jax.random.normal(rng, (1, 8, 224, 224, 3))
        
        params = init.init(rng, video)
        output = init.apply(params, rng, video)
        
        self.assertFalse(jnp.any(jnp.isnan(output['primary_features'])))


if __name__ == "__main__":
    unittest.main()
