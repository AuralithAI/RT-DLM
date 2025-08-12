import haiku as hk
import jax
import jax.numpy as jnp
import optax
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

class ReasoningStep(hk.Module):
    """Single reasoning step with thought tracking"""
    
    def __init__(self, d_model: int, name=None):
        super().__init__(name=name)
        self.d_model = d_model
        
        # Question analysis
        self.question_encoder = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(d_model),
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        ], name="question_encoder")
        
        # Working memory
        self.working_memory = hk.MultiHeadAttention(
            num_heads=8, key_size=d_model//8, name="working_memory",
            w_init=hk.initializers.TruncatedNormal(stddev=0.02)
        )
        
        # Hypothesis generation
        self.hypothesis_generator = hk.Sequential([
            hk.Linear(d_model * 2),
            jax.nn.silu,
            hk.Linear(d_model),
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        ], name="hypothesis_gen")
        
        # Evidence integration
        self.evidence_integrator = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(d_model),
            jax.nn.sigmoid  # Confidence score
        ], name="evidence_integrator")
        
        # Thought tracking
        self.thought_tracker = hk.Linear(d_model, name="thought_tracker")
        
    def __call__(self, query, context, previous_thoughts=None):
        """
        Single reasoning step
        
        Args:
            query: Current question/problem [batch, seq_len, d_model]
            context: Available context/knowledge [batch, context_len, d_model]
            previous_thoughts: Previous reasoning steps [batch, num_thoughts, d_model]
        """
        # Encode the question
        encoded_query = self.question_encoder(query)
        
        # Working memory: attend to relevant context
        working_mem, attention_weights = self.working_memory(
            encoded_query, context, context, return_attention_weights=True
        )
        
        # Combine query with working memory
        combined_input = jnp.concatenate([encoded_query, working_mem], axis=-1)
        
        # Generate hypothesis
        hypothesis = self.hypothesis_generator(combined_input)
        
        # Integrate with previous thoughts if available
        if previous_thoughts is not None:
            thought_context = jnp.concatenate([previous_thoughts, hypothesis], axis=1)
            integrated_thoughts = self.working_memory(hypothesis, thought_context, thought_context)
            hypothesis = hypothesis + integrated_thoughts
        
        # Compute confidence
        confidence = self.evidence_integrator(hypothesis)
        
        # Track thought
        thought_representation = self.thought_tracker(hypothesis)
        
        return {
            "hypothesis": hypothesis,
            "confidence": confidence,
            "thought_representation": thought_representation,
            "attention_weights": attention_weights,
            "working_memory": working_mem
        }

class ChainOfThoughtReasoning(hk.Module):
    """Chain-of-thought reasoning with explicit step tracking"""
    
    def __init__(self, d_model: int, max_reasoning_steps: int = 10, name=None):
        super().__init__(name=name)
        self.d_model = d_model
        self.max_reasoning_steps = max_reasoning_steps
        
        # Reasoning steps
        self.reasoning_steps = [
            ReasoningStep(d_model, name=f"step_{i}") 
            for i in range(max_reasoning_steps)
        ]
        
        # Step selector (decides when to stop reasoning)
        self.step_selector = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(1),
            jax.nn.sigmoid  # Probability of stopping
        ], name="step_selector")
        
        # Final answer synthesis
        self.answer_synthesizer = hk.Sequential([
            hk.Linear(d_model * 2),
            jax.nn.silu,
            hk.Linear(d_model),
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        ], name="answer_synthesizer")
        
    def __call__(self, query, context, max_steps=None):
        """
        Perform chain-of-thought reasoning
        
        Args:
            query: Question to reason about [batch, seq_len, d_model]
            context: Available knowledge [batch, context_len, d_model]
            max_steps: Override max reasoning steps
        """
        if max_steps is None:
            max_steps = self.max_reasoning_steps
            
        thoughts = []
        confidences = []
        attention_maps = []
        
        current_query = query
        previous_thoughts = None
        
        for step in range(max_steps):
            # Execute reasoning step
            step_result = self.reasoning_steps[step](
                current_query, context, previous_thoughts
            )
            
            thoughts.append(step_result["thought_representation"])
            confidences.append(step_result["confidence"])
            attention_maps.append(step_result["attention_weights"])
            
            # Update for next step
            current_query = step_result["hypothesis"]
            if previous_thoughts is None:
                previous_thoughts = step_result["thought_representation"]
            else:
                previous_thoughts = jnp.concatenate([
                    previous_thoughts, step_result["thought_representation"]
                ], axis=1)
            
            # Decide whether to continue
            _ = self.step_selector(step_result["hypothesis"].mean(axis=1, keepdims=True))
            
            # For simplicity, we'll continue until max_steps
            # In practice, you could implement early stopping based on stop_probability
        
        # Synthesize final answer
        all_thoughts = jnp.stack(thoughts, axis=1)  # [batch, num_steps, seq_len, d_model]
        thought_summary = all_thoughts.mean(axis=(1, 2))  # [batch, d_model]
        query_summary = query.mean(axis=1)  # [batch, d_model]
        
        final_input = jnp.concatenate([thought_summary, query_summary], axis=-1)
        final_answer = self.answer_synthesizer(final_input)
        
        return {
            "final_answer": final_answer,
            "reasoning_chain": thoughts,
            "confidences": confidences,
            "attention_maps": attention_maps,
            "thought_summary": thought_summary
        }

class MetaLearningController(hk.Module):
    """Meta-learning controller for few-shot adaptation"""
    
    def __init__(self, d_model: int, name=None):
        super().__init__(name=name)
        self.d_model = d_model
        
        # Task encoder
        self.task_encoder = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(d_model),
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        ], name="task_encoder")
        
        # Few-shot adaptation
        self.adaptation_controller = hk.Sequential([
            hk.Linear(d_model * 2),
            jax.nn.silu,
            hk.Linear(d_model),
            jax.nn.tanh
        ], name="adaptation_controller")
        
        # Learning rate modulation
        self.lr_modulator = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(1),
            jax.nn.sigmoid
        ], name="lr_modulator")
        
    def __call__(self, support_examples, query_example, task_description=None):
        """
        Meta-learning for few-shot adaptation
        
        Args:
            support_examples: [batch, num_examples, seq_len, d_model]
            query_example: [batch, seq_len, d_model]
            task_description: Optional task description
        """
        # Encode task from support examples
        task_representation = self.task_encoder(support_examples.mean(axis=(1, 2)))
        
        # Encode query
        query_representation = query_example.mean(axis=1)
        
        # Generate adaptation signal
        combined_input = jnp.concatenate([task_representation, query_representation], axis=-1)
        adaptation_signal = self.adaptation_controller(combined_input)
        
        # Modulate learning rate
        adaptive_lr = self.lr_modulator(task_representation)
        
        return {
            "task_representation": task_representation,
            "adaptation_signal": adaptation_signal,
            "adaptive_learning_rate": adaptive_lr,
            "query_encoding": query_representation
        }

class SelfImprovementModule(hk.Module):
    """Self-improvement through experience replay and meta-optimization"""
    
    def __init__(self, d_model: int, memory_size: int = 10000, name=None):
        super().__init__(name=name)
        self.d_model = d_model
        self.memory_size = memory_size
        
        # Experience encoder
        self.experience_encoder = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(d_model),
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        ], name="experience_encoder")
        
        # Performance predictor
        self.performance_predictor = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(1),
            jax.nn.sigmoid
        ], name="performance_predictor")
        
        # Improvement strategy generator
        self.strategy_generator = hk.Sequential([
            hk.Linear(d_model * 2),
            jax.nn.silu,
            hk.Linear(d_model),
            jax.nn.tanh
        ], name="strategy_generator")
        
        # Initialize experience memory
        self.experience_memory = hk.get_state(
            "experience_memory", 
            [memory_size, d_model], 
            init=jnp.zeros,
            dtype=jnp.float32
        )
        self.memory_scores = hk.get_state(
            "memory_scores",
            [memory_size],
            init=jnp.zeros,
            dtype=jnp.float32
        )
        
    def store_experience(self, experience, performance_score):
        """Store successful experiences for replay"""
        # Encode experience
        encoded_exp = self.experience_encoder(experience.mean(axis=1))
        
        # Find lowest scoring memory slot to replace
        min_idx = jnp.argmin(self.memory_scores)
        
        # Update memory
        new_memory = self.experience_memory.at[min_idx].set(encoded_exp[0])
        new_scores = self.memory_scores.at[min_idx].set(performance_score)
        
        hk.set_state("experience_memory", new_memory)
        hk.set_state("memory_scores", new_scores)
        
    def generate_improvement_strategy(self, current_performance, target_task):
        """Generate strategy for self-improvement"""
        # Get best experiences from memory
        top_k = 5
        top_indices = jnp.argsort(self.memory_scores)[-top_k:]
        best_experiences = self.experience_memory[top_indices].mean(axis=0)
        
        # Encode current task
        task_encoding = self.experience_encoder(target_task.mean(axis=1))
        
        # Generate improvement strategy
        combined_input = jnp.concatenate([best_experiences, task_encoding], axis=-1)
        strategy = self.strategy_generator(combined_input)
        
        # Predict expected performance improvement
        expected_improvement = self.performance_predictor(strategy)
        
        return {
            "improvement_strategy": strategy,
            "expected_improvement": expected_improvement,
            "best_experiences": best_experiences
        }
        
    def __call__(self, current_input, performance_feedback=None):
        """Main forward pass with optional experience storage"""
        if performance_feedback is not None:
            self.store_experience(current_input, performance_feedback)
            
        strategy = self.generate_improvement_strategy(0.0, current_input)
        return strategy

class ReasoningEngine(hk.Module):
    """Complete reasoning engine combining all components"""
    
    def __init__(self, config, name=None):
        super().__init__(name=name)
        self.config = config
        
        # Core reasoning components
        self.chain_of_thought = ChainOfThoughtReasoning(
            config.d_model, 
            max_reasoning_steps=config.max_reasoning_steps
        )
        self.meta_controller = MetaLearningController(config.d_model)
        self.self_improvement = SelfImprovementModule(config.d_model)
        
        # Integration layer
        self.reasoning_integrator = hk.Sequential([
            hk.Linear(config.d_model * 3),
            jax.nn.silu,
            hk.Linear(config.d_model),
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        ], name="reasoning_integrator")
        
    def __call__(self, query, context, support_examples=None, performance_feedback=None):
        """
        Complete reasoning pipeline
        
        Args:
            query: Question/problem to solve
            context: Available knowledge
            support_examples: Few-shot examples (optional)
            performance_feedback: Feedback for self-improvement (optional)
        """
        # Chain-of-thought reasoning
        reasoning_result = self.chain_of_thought(query, context)
        
        # Meta-learning adaptation
        if support_examples is not None:
            meta_result = self.meta_controller(support_examples, query)
            adapted_query = query + meta_result["adaptation_signal"]
        else:
            meta_result = None
            adapted_query = query
        
        # Self-improvement
        improvement_result = self.self_improvement(adapted_query, performance_feedback)
        
        # Integrate all reasoning components
        reasoning_features = reasoning_result["thought_summary"]
        meta_features = meta_result["task_representation"] if meta_result else jnp.zeros_like(reasoning_features)
        improvement_features = improvement_result["improvement_strategy"]
        
        integrated_reasoning = self.reasoning_integrator(
            jnp.concatenate([reasoning_features, meta_features, improvement_features], axis=-1)
        )
        
        return {
            "reasoning_output": integrated_reasoning,
            "chain_of_thought": reasoning_result,
            "meta_learning": meta_result,
            "self_improvement": improvement_result,
            "reasoning_chain": reasoning_result["reasoning_chain"],
            "confidence_scores": reasoning_result["confidences"]
        }
