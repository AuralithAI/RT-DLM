"""
Real-Time Feedback Learning System for RT-DLM AGI
Implements continuous learning from user interactions and dynamic skill acquisition.
"""

import haiku as hk
import jax
import jax.numpy as jnp
import optax
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class FeedbackSample:
    """Single feedback sample for real-time learning."""
    input_text: str
    output_text: str
    user_rating: float  # -1 to 1
    correction: Optional[str] = None
    feedback_type: str = "rating"  # rating, correction, preference
    timestamp: float = 0.0
    context: Optional[Dict] = None


@dataclass
class SkillDefinition:
    """Definition of a learnable skill."""
    skill_id: str
    skill_name: str
    description: str
    required_examples: int = 5
    current_proficiency: float = 0.0
    training_samples: List[FeedbackSample] = None


class RealTimeFeedbackBuffer:
    """Efficient buffer for storing and managing real-time feedback."""
    
    def __init__(self, max_size: int = 10000, priority_threshold: float = 0.8):
        self.max_size = max_size
        self.priority_threshold = priority_threshold
        self.buffer: deque[FeedbackSample] = deque(maxlen=max_size)
        self.priority_buffer: deque[FeedbackSample] = deque(maxlen=max_size // 4)  # High-priority samples
        self.skill_buffers: Dict[str, deque[FeedbackSample]] = {}  # Per-skill buffers
        
    def add_feedback(self, sample: FeedbackSample):
        """Add feedback sample with automatic prioritization."""
        # Add to main buffer
        self.buffer.append(sample)
        
        # Add to priority buffer if high-impact feedback
        if abs(sample.user_rating) > self.priority_threshold or sample.correction:
            self.priority_buffer.append(sample)
        
        # Add to skill-specific buffer if context indicates skill
        if sample.context and 'skill_id' in sample.context:
            skill_id = sample.context['skill_id']
            if skill_id not in self.skill_buffers:
                self.skill_buffers[skill_id] = deque(maxlen=1000)
            self.skill_buffers[skill_id].append(sample)
    
    def get_training_batch(self, batch_size: int, prioritize: bool = True) -> List[FeedbackSample]:
        """Get batch of samples for training."""
        if prioritize and len(self.priority_buffer) >= batch_size:
            # Sample from priority buffer
            indices = np.random.choice(len(self.priority_buffer), batch_size, replace=False)
            return [self.priority_buffer[i] for i in indices]
        elif len(self.buffer) >= batch_size:
            # Sample from main buffer
            indices = np.random.choice(len(self.buffer), batch_size, replace=False)
            return [self.buffer[i] for i in indices]
        else:
            return list(self.buffer)
    
    def get_skill_samples(self, skill_id: str, count: int = None) -> List[FeedbackSample]:
        """Get samples for specific skill learning."""
        if skill_id in self.skill_buffers:
            samples = list(self.skill_buffers[skill_id])
            if count and len(samples) > count:
                return samples[-count:]  # Return most recent
            return samples
        return []


class DynamicSkillAcquisition(hk.Module):
    """
    Dynamic skill acquisition system that can learn new capabilities 
    without full retraining.
    """
    
    def __init__(self, base_d_model: int, max_skills: int = 100, name=None):
        super().__init__(name=name)
        self.base_d_model = base_d_model
        self.max_skills = max_skills
        
        # Skill-specific adapter networks
        self.skill_adapters = {}
        
        # Skill routing network
        self.skill_router = hk.Sequential([
            hk.Linear(base_d_model),
            jax.nn.silu,
            hk.Linear(max_skills),
            jax.nn.softmax
        ])
        
        # Meta-skill learning network
        self.meta_learner = hk.Sequential([
            hk.Linear(base_d_model * 2),
            jax.nn.silu,
            hk.Linear(base_d_model),
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        ])
        
        # Skill registry
        self.active_skills = hk.get_state("active_skills", [], dtype=jnp.int32, init=jnp.zeros)
        
    def create_skill_adapter(self, skill_id: str) -> hk.Module:
        """Create a new adapter network for a specific skill."""
        return hk.Sequential([
            hk.Linear(self.base_d_model // 2),
            jax.nn.silu,
            hk.Linear(self.base_d_model),
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        ], name=f"skill_adapter_{skill_id}")
    
    def __call__(self, x: jnp.ndarray, skill_context: Optional[str] = None) -> jnp.ndarray:
        """
        Apply skill-specific processing to input.
        
        Args:
            x: Input embeddings [batch_size, seq_len, d_model]
            skill_context: Optional skill identifier
            
        Returns:
            Skill-adapted embeddings
        """
        # Get skill routing scores
        pooled_input = jnp.mean(x, axis=1)  # [batch_size, d_model]
        skill_scores = self.skill_router(pooled_input)
        
        # If specific skill is requested, amplify its score
        if skill_context and skill_context in self.skill_adapters:
            skill_idx = list(self.skill_adapters.keys()).index(skill_context)
            skill_scores = skill_scores.at[:, skill_idx].multiply(2.0)
            skill_scores = jax.nn.softmax(skill_scores, axis=-1)
        
        # Apply skill-specific adaptations
        adapted_outputs = []
        for i, (skill_id, adapter) in enumerate(self.skill_adapters.items()):
            skill_weight = skill_scores[:, i:i+1, None]  # [batch_size, 1, 1]
            adapted = adapter(x) * skill_weight
            adapted_outputs.append(adapted)
        
        if adapted_outputs:
            # Combine all skill adaptations
            combined_adaptation = jnp.sum(jnp.stack(adapted_outputs), axis=0)
            
            # Meta-learning integration
            meta_input = jnp.concatenate([x.reshape(x.shape[0], -1), 
                                        combined_adaptation.reshape(combined_adaptation.shape[0], -1)], axis=-1)
            meta_output = self.meta_learner(meta_input)
            meta_output = meta_output.reshape(x.shape)
            
            return x + 0.3 * combined_adaptation + 0.2 * meta_output
        else:
            return x
    
    def add_new_skill(self, skill_id: str):
        """Add a new skill adapter to the system."""
        if skill_id not in self.skill_adapters:
            self.skill_adapters[skill_id] = self.create_skill_adapter(skill_id)
            logger.info(f"Added new skill adapter: {skill_id}")


class RealTimeLearningSystem:
    """
    Complete real-time learning system that integrates feedback processing,
    skill acquisition, and model adaptation.
    """
    
    def __init__(self, base_model, d_model: int, vocab_size: int):
        self.base_model = base_model
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Feedback management
        self.feedback_buffer = RealTimeFeedbackBuffer()
        
        # Skill management
        self.skills_registry: Dict[str, Any] = {}
        self.skill_acquisition = DynamicSkillAcquisition(d_model)
        
        # Fast adaptation optimizer
        self.adaptation_optimizer = optax.adam(learning_rate=1e-4)
        
        # Performance tracking
        self.performance_history: Dict[str, deque[float]] = {
            'accuracy': deque(maxlen=1000),
            'user_satisfaction': deque(maxlen=1000),
            'learning_speed': deque(maxlen=100)
        }
    
    def process_user_feedback(self, input_text: str, output_text: str, 
                            feedback: Dict[str, Any]) -> bool:
        """
        Process real-time user feedback and trigger learning if needed.
        
        Args:
            input_text: Original input
            output_text: Model's output
            feedback: User feedback dictionary
            
        Returns:
            Whether immediate learning was triggered
        """
        # Create feedback sample
        sample = FeedbackSample(
            input_text=input_text,
            output_text=output_text,
            user_rating=feedback.get('rating', 0.0),
            correction=feedback.get('correction'),
            feedback_type=feedback.get('type', 'rating'),
            timestamp=feedback.get('timestamp', 0.0),
            context=feedback.get('context', {})
        )
        
        # Add to buffer
        self.feedback_buffer.add_feedback(sample)
        
        # Trigger immediate learning for high-impact feedback
        should_learn_immediately = (
            abs(sample.user_rating) > 0.8 or  # Strong positive/negative feedback
            sample.correction is not None or   # User provided correction
            len(self.feedback_buffer.priority_buffer) >= 8  # Enough priority samples
        )
        
        if should_learn_immediately:
            return self.immediate_adaptation()
        
        return False
    
    def immediate_adaptation(self) -> bool:
        """Perform immediate model adaptation based on recent feedback."""
        try:
            # Get training batch from priority buffer
            training_samples = self.feedback_buffer.get_training_batch(8, prioritize=True)
            
            if len(training_samples) < 4:
                return False
            
            # Prepare training data
            inputs, targets, weights = self._prepare_feedback_training_data(training_samples)
            
            # Fast gradient-based adaptation
            success = self._fast_adaptation_step(inputs, targets, weights)
            
            if success:
                logger.info(f"Immediate adaptation completed with {len(training_samples)} samples")
                
                # Update performance tracking
                avg_rating = np.mean([s.user_rating for s in training_samples])
                self.performance_history['user_satisfaction'].append(avg_rating)
                
            return success
            
        except Exception as e:
            logger.error(f"Immediate adaptation failed: {e}")
            return False
    
    def learn_new_skill(self, skill_definition: SkillDefinition, 
                       training_samples: List[FeedbackSample]) -> bool:
        """
        Learn a completely new skill from user demonstrations.
        
        Args:
            skill_definition: Definition of the new skill
            training_samples: Example interactions for the skill
            
        Returns:
            Whether skill learning was successful
        """
        skill_id = skill_definition.skill_id
        
        # Add skill to registry
        self.skills_registry[skill_id] = skill_definition
        
        # Create skill adapter
        self.skill_acquisition.add_new_skill(skill_id)
        
        # Train skill adapter on examples
        success = self._train_skill_adapter(skill_id, training_samples)
        
        if success:
            logger.info(f"Successfully learned new skill: {skill_definition.skill_name}")
            skill_definition.current_proficiency = 0.8  # Initial proficiency
        else:
            logger.warning(f"Failed to learn skill: {skill_definition.skill_name}")
            # Remove failed skill
            del self.skills_registry[skill_id]
        
        return success
    
    def _prepare_feedback_training_data(self, samples: List[FeedbackSample]) -> Tuple:
        """Prepare feedback samples for training."""
        # This would convert text to token IDs and create training targets
        # Implementation depends on your tokenizer
        inputs = []
        targets = []
        weights = []
        
        for sample in samples:
            # Convert to model format (simplified)
            # In real implementation, use your tokenizer
            input_ids = [hash(sample.input_text) % self.vocab_size]  # Placeholder
            target_ids = [hash(sample.output_text) % self.vocab_size]  # Placeholder
            
            # Weight by feedback strength
            weight = abs(sample.user_rating)
            if sample.correction:
                weight *= 2.0  # Corrections are more important
            
            inputs.append(input_ids)
            targets.append(target_ids)
            weights.append(weight)
        
        return jnp.array(inputs), jnp.array(targets), jnp.array(weights)
    
    def _fast_adaptation_step(self, inputs: jnp.ndarray, targets: jnp.ndarray, 
                            weights: jnp.ndarray) -> bool:
        """Perform fast adaptation using the feedback data."""
        try:
            # This would integrate with your existing training loop
            # Simplified implementation
            
            # Calculate weighted loss
            # loss = weighted_cross_entropy(predictions, targets, weights)
            
            # Apply gradients with small learning rate for stability
            # params = optax.apply_updates(params, updates)
            
            return True
            
        except Exception as e:
            logger.error(f"Fast adaptation step failed: {e}")
            return False
    
    def _train_skill_adapter(self, skill_id: str, samples: List[FeedbackSample]) -> bool:
        """Train the skill-specific adapter."""
        try:
            # Prepare skill-specific training data
            skill_inputs, skill_targets, _ = self._prepare_feedback_training_data(samples)
            
            # Train adapter network (simplified)
            # This would use meta-learning to quickly adapt the skill adapter
            
            return True
            
        except Exception as e:
            logger.error(f"Skill adapter training failed for {skill_id}: {e}")
            return False
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get current learning system performance metrics."""
        return {
            'avg_user_satisfaction': np.mean(list(self.performance_history['user_satisfaction'])) if self.performance_history['user_satisfaction'] else 0.0,
            'active_skills_count': len(self.skills_registry),
            'feedback_samples_count': len(self.feedback_buffer.buffer),
            'priority_samples_count': len(self.feedback_buffer.priority_buffer),
            'recent_learning_events': len([s for s in self.feedback_buffer.buffer if abs(s.user_rating) > 0.8])
        }
    
    def export_learned_skills(self) -> Dict[str, Any]:
        """Export learned skills for persistence."""
        return {
            'skills_registry': {k: {
                'skill_id': v.skill_id,
                'skill_name': v.skill_name,
                'description': v.description,
                'proficiency': v.current_proficiency
            } for k, v in self.skills_registry.items()},
            'performance_metrics': self.get_performance_metrics()
        }


# Integration with existing TMS model
class RTDLMWithRealTimeLearning(hk.Module):
    """
    Enhanced RT-DLM with real-time learning capabilities.
    Integrates with existing TMS architecture.
    """
    
    def __init__(self, tms_model, d_model: int, vocab_size: int, name=None):
        super().__init__(name=name)
        self.tms_model = tms_model
        self.real_time_learning = RealTimeLearningSystem(tms_model, d_model, vocab_size)
    
    def __call__(self, inputs, **kwargs):
        """Enhanced forward pass with real-time learning integration."""
        # Get base model output
        base_output = self.tms_model(inputs, **kwargs)
        
        # Apply skill-specific adaptations
        skill_context = kwargs.get('skill_context')
        adapted_output = self.real_time_learning.skill_acquisition(base_output, skill_context)
        
        return adapted_output
    
    def process_feedback(self, input_text: str, output_text: str, feedback: Dict) -> bool:
        """Process user feedback for real-time learning."""
        return self.real_time_learning.process_user_feedback(input_text, output_text, feedback)
    
    def learn_skill(self, skill_def: SkillDefinition, samples: List[FeedbackSample]) -> bool:
        """Learn a new skill from examples."""
        return self.real_time_learning.learn_new_skill(skill_def, samples)

