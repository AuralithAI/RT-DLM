import haiku as hk
import jax
import jax.numpy as jnp
import optax
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

# Import SemanticParser for graph-based multi-hop reasoning
try:
    from modules.capabilities.comprehension_modules import SemanticParser
    SEMANTIC_PARSER_AVAILABLE = True
except ImportError:
    SEMANTIC_PARSER_AVAILABLE = False

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
            query: Current question/problem [batch, seq_len, d_model] or [batch, d_model]
            context: Available context/knowledge [batch, context_len, d_model]
            previous_thoughts: Previous reasoning steps [batch, num_thoughts, d_model] or [batch, d_model]
        """
        if query.ndim == 2:
            query = query[:, None, :]
        
        # Encode the question
        encoded_query = self.question_encoder(query)
        
        # Working memory: attend to relevant context
        working_mem = self.working_memory(encoded_query, context, context)
        
        # Combine query with working memory
        combined_input = jnp.concatenate([encoded_query, working_mem], axis=-1)
        
        # Generate hypothesis
        hypothesis = self.hypothesis_generator(combined_input)
        
        # Integrate with previous thoughts if available
        if previous_thoughts is not None:
            if previous_thoughts.ndim == 2:
                previous_thoughts = previous_thoughts[:, None, :]
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
            "working_memory": working_mem
        }

class ChainOfThoughtReasoning(hk.Module):
    """Chain-of-thought reasoning with explicit step tracking and graph-based multi-hop support"""
    
    def __init__(self, d_model: int, max_reasoning_steps: int = 10, use_semantic_graph: bool = True, name=None):
        super().__init__(name=name)
        self.d_model = d_model
        self.max_reasoning_steps = max_reasoning_steps
        self.use_semantic_graph = use_semantic_graph and SEMANTIC_PARSER_AVAILABLE
        
        # Reasoning steps
        self.reasoning_steps = [
            ReasoningStep(d_model, name=f"step_{i}") 
            for i in range(max_reasoning_steps)
        ]
        
        # SemanticParser for graph-based multi-hop reasoning
        if self.use_semantic_graph:
            self.semantic_parser = SemanticParser(
                d_model=d_model,
                max_nodes=32,
                num_hops=3,
                num_heads=8,
                edge_threshold=0.3,
                name="semantic_parser"
            )
            
            # Graph-enhanced reasoning integrator
            self.graph_integrator = hk.Sequential([
                hk.Linear(d_model * 2),
                jax.nn.silu,
                hk.Linear(d_model),
                hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
            ], name="graph_integrator")
        
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
            query: Question to reason about [batch, seq_len, d_model] or [batch, d_model]
            context: Available knowledge [batch, context_len, d_model]
            max_steps: Override max reasoning steps
        """
        if max_steps is None:
            max_steps = self.max_reasoning_steps
        
        # Ensure query is 3D for consistent processing
        if query.ndim == 2:
            query = query[:, None, :]
            
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
            # Update for next step
            current_query = step_result["hypothesis"]
            if previous_thoughts is None:
                previous_thoughts = step_result["thought_representation"]
            else:
                previous_thoughts = jnp.concatenate([
                    previous_thoughts, step_result["thought_representation"]
                ], axis=1)
            
            # Early stopping based on confidence and stop probability
            stop_input = step_result["hypothesis"].mean(axis=1, keepdims=True)
            stop_probability = self.step_selector(stop_input).mean()
            
            # Stop if stop probability is high (>0.8) and we've done at least 2 steps
            if step >= 2 and stop_probability > 0.8:
                break
        
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
    
    def multi_hop_reasoning(
        self, 
        query: jnp.ndarray, 
        context: jnp.ndarray,
        is_training: bool = True
    ) -> Dict[str, Any]:
        """
        Perform graph-based multi-hop reasoning using SemanticParser.
        
        This method builds a conceptual graph from context, extracts knowledge,
        and performs multi-hop traversal to answer complex queries.
        
        Useful for niches like health diagnostics where multi-step inference
        and abstraction from multimodal data are critical.
        
        Args:
            query: Question to reason about [batch, seq_len, d_model]
            context: Available knowledge [batch, context_len, d_model]
            is_training: Whether in training mode
            
        Returns:
            Dictionary with reasoning results and graph structure
        """
        if not self.use_semantic_graph:
            # Fall back to standard chain-of-thought
            return self(query, context)
        
        # Build conceptual graph from context using SemanticParser
        query_vector = query.mean(axis=1)  # [batch, d_model]
        
        # Full semantic parsing with graph-based reasoning
        semantic_result = self.semantic_parser.parse(
            context, 
            query=query_vector,
            mask=None,
            is_training=is_training
        )
        
        # Extract graph-based reasoning output
        graph_answer = semantic_result["reasoning"]["answer_embedding"] if semantic_result["reasoning"] else None
        semantic_representation = semantic_result["semantic_representation"]
        
        # Also run chain-of-thought reasoning for comparison
        cot_result = self(query, context)
        
        # Integrate graph-based and chain-of-thought reasoning
        if graph_answer is not None:
            integrated_answer = self.graph_integrator(
                jnp.concatenate([cot_result["final_answer"], graph_answer], axis=-1)
            )
        else:
            integrated_answer = cot_result["final_answer"]
        
        return {
            "final_answer": integrated_answer,
            "reasoning_chain": cot_result["reasoning_chain"],
            "confidences": cot_result["confidences"],
            "attention_maps": cot_result["attention_maps"],
            "thought_summary": cot_result["thought_summary"],
            # Graph-based reasoning outputs
            "graph": semantic_result["graph"],
            "knowledge": semantic_result["knowledge"],
            "graph_reasoning": semantic_result["reasoning"],
            "semantic_representation": semantic_representation,
            "hop_embeddings": semantic_result["reasoning"]["hop_embeddings"] if semantic_result["reasoning"] else None
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
        if target_task.ndim == 2:
            task_encoding = self.experience_encoder(target_task)
        else:
            task_encoding = self.experience_encoder(target_task.mean(axis=1))
        
        # Broadcast best_experiences to match batch size
        batch_size = task_encoding.shape[0]
        best_experiences_batched = jnp.broadcast_to(
            best_experiences, (batch_size, best_experiences.shape[-1])
        )
        
        # Generate improvement strategy
        combined_input = jnp.concatenate([best_experiences_batched, task_encoding], axis=-1)
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

        self._rlm_enabled = getattr(config, 'rlm_enabled', False)
        if self._rlm_enabled:
            from core.rlm import RecursiveLanguageModel
            from config.rlm_config import RLMConfig
            rlm_config = RLMConfig(
                max_recursion_depth=getattr(config, 'rlm_max_recursion_depth', 5),
                context_peek_size=getattr(config, 'rlm_context_peek_size', 2000),
                tool_budget=getattr(config, 'rlm_tool_budget', 20),
                auto_partition_threshold=getattr(config, 'rlm_auto_partition_threshold', 8000),
                direct_context_threshold=getattr(config, 'rlm_direct_context_threshold', 2000),
            )
            self.rlm = RecursiveLanguageModel(config.d_model, rlm_config)

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

    def recursive_context_reasoning(
        self,
        query: jnp.ndarray,
        context: jnp.ndarray,
        context_length: int,
    ) -> Dict[str, Any]:
        if not self._rlm_enabled:
            return self(query, context)

        tool_probs, term_prob, parameters, encoded_query = self.rlm(
            query, context_length, recursion_depth=0, tool_calls_used=0
        )

        cot_result = self.chain_of_thought(query, context)

        return {
            "reasoning_output": cot_result["final_answer"],
            "chain_of_thought": cot_result,
            "rlm_tool_probs": tool_probs,
            "rlm_termination_prob": term_prob,
            "rlm_parameters": parameters,
            "rlm_encoded_query": encoded_query,
            "reasoning_chain": cot_result["reasoning_chain"],
            "confidence_scores": cot_result["confidences"],
        }
