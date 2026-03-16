import haiku as hk
import jax
import jax.numpy as jnp
import optax
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

# Import SemanticParser for graph-based multi-hop reasoning
try:
    from src.modules.capabilities.comprehension_modules import SemanticParser
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


# =============================================================================
# Verify/Reflect Loop
# =============================================================================

class VerificationHead(hk.Module):
    """
    Verification head that evaluates whether a reasoning output is correct.
    
    Takes a candidate answer and its reasoning chain, produces a verification
    score indicating confidence that the answer is correct. Used as the
    'verify' step in the verify/reflect loop.
    
    Architecture:
        [answer, thought_summary, query] → Linear(3*d, d) → SiLU
            → LayerNorm → Linear(d, d//2) → SiLU → Linear(d//2, 1) → sigmoid
    """
    
    def __init__(self, d_model: int, name: Optional[str] = None):
        super().__init__(name=name)
        self.d_model = d_model
        
        self.verify_net = hk.Sequential([
            hk.Linear(d_model, name="verify_fc1"),
            jax.nn.silu,
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name="verify_ln"),
            hk.Linear(d_model // 2, name="verify_fc2"),
            jax.nn.silu,
            hk.Linear(1, name="verify_out"),
            jax.nn.sigmoid
        ], name="verify_net")
        
        self.input_proj = hk.Linear(d_model, name="verify_input_proj")
    
    def __call__(
        self,
        answer: jnp.ndarray,
        thought_summary: jnp.ndarray,
        query_summary: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Verify a candidate answer.
        
        Args:
            answer: Candidate answer [batch, d_model]
            thought_summary: Summary of reasoning chain [batch, d_model]
            query_summary: Original query summary [batch, d_model]
            
        Returns:
            verification_score: Confidence the answer is correct [batch, 1]
        """
        combined = jnp.concatenate([answer, thought_summary, query_summary], axis=-1)
        projected = self.input_proj(combined)
        return self.verify_net(projected)


class ReflectionModule(hk.Module):
    """
    Reflection module that generates corrective signals when verification fails.
    
    When the VerificationHead produces a low score, the ReflectionModule
    analyzes what went wrong and produces a correction delta to refine
    the answer. This implements the 'reflect' step.
    
    Architecture:
        [answer, thought_summary, verification_score] → Linear(2d+1, d) → SiLU
            → LayerNorm → Linear(d, d) → tanh (correction bounded to [-1, 1])
    """
    
    def __init__(self, d_model: int, name: Optional[str] = None):
        super().__init__(name=name)
        self.d_model = d_model
        
        self.reflect_net = hk.Sequential([
            hk.Linear(d_model, name="reflect_fc1"),
            jax.nn.silu,
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name="reflect_ln"),
            hk.Linear(d_model, name="reflect_fc2"),
            jax.nn.tanh  # Bounded correction
        ], name="reflect_net")
        
        self.input_proj = hk.Linear(d_model, name="reflect_input_proj")
        
        # Reflection gate: controls how much correction to apply
        self.gate = hk.Sequential([
            hk.Linear(d_model // 2, name="gate_fc1"),
            jax.nn.silu,
            hk.Linear(d_model, name="gate_fc2"),
            jax.nn.sigmoid
        ], name="reflect_gate")
    
    def __call__(
        self,
        answer: jnp.ndarray,
        thought_summary: jnp.ndarray,
        verification_score: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Generate reflection-based correction.
        
        Args:
            answer: Current candidate answer [batch, d_model]
            thought_summary: Reasoning chain summary [batch, d_model]
            verification_score: How confident the verifier is [batch, 1]
            
        Returns:
            corrected_answer: Answer with reflection correction applied [batch, d_model]
            correction_delta: The raw correction signal [batch, d_model]
        """
        # Expand verification score to d_model for concatenation
        score_expanded = jnp.broadcast_to(verification_score, (answer.shape[0], self.d_model))
        
        combined = jnp.concatenate([answer, thought_summary, score_expanded], axis=-1)
        projected = self.input_proj(combined)
        
        # Generate correction
        correction = self.reflect_net(projected)
        
        # Gate the correction (apply more correction when verification is low)
        gate_input = jnp.concatenate([answer, correction], axis=-1)
        gate_proj = hk.Linear(self.d_model // 2, name="gate_proj")(gate_input)
        gate_value = self.gate(gate_proj)
        
        # Scale correction by inverse of verification confidence
        # Low verification → more correction
        correction_scale = 1.0 - verification_score
        scaled_correction = correction * gate_value * correction_scale
        
        corrected_answer = answer + scaled_correction
        
        return corrected_answer, scaled_correction


class VerifyReflectReasoning(hk.Module):
    """
    Iterative Verify/Reflect reasoning loop.
    
    Wraps ChainOfThoughtReasoning with a verification and reflection cycle:
    1. Generate initial answer via CoT reasoning
    2. Verify: Check if answer meets confidence threshold
    3. If verification fails: Reflect and generate correction
    4. Repeat until verified or max_verify_steps reached
    
    This enables the model to catch and correct its own mistakes,
    similar to how humans re-check their work.
    
    Args:
        d_model: Model dimension
        max_reasoning_steps: Steps for initial CoT reasoning
        max_verify_steps: Maximum verify/reflect iterations
        confidence_threshold: Verification score to accept answer
        use_semantic_graph: Whether to use graph-based reasoning
    """
    
    def __init__(
        self,
        d_model: int,
        max_reasoning_steps: int = 10,
        max_verify_steps: int = 3,
        confidence_threshold: float = 0.85,
        use_semantic_graph: bool = True,
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.d_model = d_model
        self.max_verify_steps = max_verify_steps
        self.confidence_threshold = confidence_threshold
        
        # Core reasoning engine
        self.chain_of_thought = ChainOfThoughtReasoning(
            d_model, max_reasoning_steps, use_semantic_graph, name="cot"
        )
        
        # Verify/Reflect components
        self.verifier = VerificationHead(d_model, name="verifier")
        self.reflector = ReflectionModule(d_model, name="reflector")
    
    def __call__(
        self,
        query: jnp.ndarray,
        context: jnp.ndarray,
        max_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Perform reasoning with verify/reflect loop.
        
        Args:
            query: Question to reason about [batch, seq_len, d_model] or [batch, d_model]
            context: Available knowledge [batch, context_len, d_model]
            max_steps: Override max reasoning steps for CoT
            
        Returns:
            Dictionary containing:
                - final_answer: Best answer after verification [batch, d_model]
                - reasoning_chain: CoT reasoning steps
                - confidences: Step confidences from CoT
                - verification_scores: Scores from each verify step
                - num_reflections: Number of reflection corrections applied
                - thought_summary: Summary of reasoning chain
                - reflection_deltas: Correction vectors applied
        """
        # Ensure query is 3D
        if query.ndim == 2:
            query_3d = query[:, None, :]
        else:
            query_3d = query
        
        # Step 1: Initial reasoning
        cot_result = self.chain_of_thought(query_3d, context, max_steps)
        
        current_answer = cot_result["final_answer"]
        thought_summary = cot_result["thought_summary"]
        query_summary = query_3d.mean(axis=1)  # [batch, d_model]
        
        verification_scores = []
        reflection_deltas = []
        num_reflections = 0
        # Track whether we've already accepted the answer
        accepted = jnp.array(False)
        
        # Step 2-4: Verify/Reflect loop
        # Always run all steps to keep the computation graph static
        # (required for jax.grad compatibility), but conditionally apply updates.
        for _ in range(self.max_verify_steps):
            # Verify current answer
            v_score = self.verifier(current_answer, thought_summary, query_summary)
            verification_scores.append(v_score)
            
            # Check if verification passes (JAX-traceable comparison)
            passes = v_score.mean() >= self.confidence_threshold
            accepted = accepted | passes
            
            # Reflect and correct (always compute, conditionally apply)
            corrected, delta = self.reflector(current_answer, thought_summary, v_score)
            reflection_deltas.append(delta)
            
            # Only apply correction if not yet accepted
            # jnp.where is differentiable and trace-safe
            should_apply = ~accepted
            apply_mask = jnp.where(should_apply, 1.0, 0.0)
            
            current_answer = current_answer + apply_mask * (corrected - current_answer)
            num_reflections += int(jnp.where(should_apply, 1, 0))
            
            # Update thought summary with reflection
            delta_mean = delta.mean(axis=-1, keepdims=True)
            thought_update = jnp.broadcast_to(delta_mean, thought_summary.shape) * 0.1
            thought_summary = thought_summary + apply_mask * thought_update
        
        return {
            "final_answer": current_answer,
            "reasoning_chain": cot_result["reasoning_chain"],
            "confidences": cot_result["confidences"],
            "attention_maps": cot_result.get("attention_maps", []),
            "thought_summary": thought_summary,
            "verification_scores": verification_scores,
            "num_reflections": num_reflections,
            "reflection_deltas": reflection_deltas,
        }


class SelfCritiqueHead(hk.Module):
    """
    Self-critique head for output quality assessment and iterative revision.
    
    Evaluates the quality of generated output and produces a critique signal.
    If quality is below threshold, triggers re-generation with the critique
    as additional context, enabling iterative self-improvement.
    
    Architecture:
        hidden_state → Linear(d, d//2) → SiLU → LayerNorm
                     → Linear(d//2, 2) → [quality_score, revision_signal]
    
    The quality_score (sigmoid) indicates output quality [0, 1].
    The revision_signal (tanh) provides directional feedback [-1, 1].
    """
    
    def __init__(
        self,
        d_model: int,
        quality_threshold: float = 0.6,
        max_revisions: int = 2,
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.d_model = d_model
        self.quality_threshold = quality_threshold
        self.max_revisions = max_revisions
        
        self.critique_net = hk.Sequential([
            hk.Linear(d_model // 2, name="critique_fc1"),
            jax.nn.silu,
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name="critique_ln"),
        ], name="critique_backbone")
        
        # Separate heads for quality and revision
        self.quality_head = hk.Sequential([
            hk.Linear(1, name="quality_out"),
            jax.nn.sigmoid
        ], name="quality_head")
        
        self.revision_head = hk.Sequential([
            hk.Linear(d_model, name="revision_out"),
            jax.nn.tanh
        ], name="revision_head")
    
    def __call__(
        self,
        hidden_state: jnp.ndarray,
        is_training: bool = True
    ) -> Dict[str, jnp.ndarray]:
        """
        Evaluate output quality and generate revision signal.
        
        Args:
            hidden_state: Output hidden state [batch, d_model] or [batch, seq, d_model]
            is_training: Whether in training mode
            
        Returns:
            Dictionary with:
                - quality_score: Output quality [batch, 1]
                - revision_signal: Directional revision feedback [batch, d_model]
                - needs_revision: Boolean mask [batch, 1]
        """
        # Pool if sequence input
        if hidden_state.ndim == 3:
            hidden_state = hidden_state.mean(axis=1)
        
        features = self.critique_net(hidden_state)
        quality_score = self.quality_head(features)
        revision_signal = self.revision_head(features)
        
        needs_revision = quality_score < self.quality_threshold
        
        return {
            "quality_score": quality_score,
            "revision_signal": revision_signal,
            "needs_revision": needs_revision,
        }
    
    def revise(
        self,
        hidden_state: jnp.ndarray,
        revision_signal: jnp.ndarray,
        iteration: int = 0
    ) -> jnp.ndarray:
        """
        Apply revision signal to hidden state.
        
        Uses decaying strength for subsequent revisions to prevent
        oscillation.
        
        Args:
            hidden_state: Current hidden state [batch, d_model]
            revision_signal: Correction direction [batch, d_model]
            iteration: Current revision iteration (0-indexed)
            
        Returns:
            revised_state: Corrected hidden state [batch, d_model]
        """
        # Decay strength with each revision to prevent oscillation
        decay = 0.5 ** iteration
        return hidden_state + decay * revision_signal


class SelfCritiqueModule(hk.Module):
    """
    Closed-loop self-critique module: generate → critique → revise.
    
    Wraps SelfCritiqueHead in an iterative loop that:
    1. Evaluates output quality via the critique head
    2. If quality < threshold, applies revision signal
    3. Re-evaluates quality after revision
    4. Repeats until quality passes OR max_revisions reached
    
    Also tracks a process reward signal: +0.3 when a revision step
    improves the quality score (fed back to GRPO training).
    
    Args:
        d_model: Model dimension
        quality_threshold: Quality score needed to accept output
        max_revisions: Maximum revision iterations
    """
    
    def __init__(
        self,
        d_model: int,
        quality_threshold: float = 0.6,
        max_revisions: int = 2,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.d_model = d_model
        self.quality_threshold = quality_threshold
        self.max_revisions = max_revisions
        
        self.critique_head = SelfCritiqueHead(
            d_model=d_model,
            quality_threshold=quality_threshold,
            max_revisions=max_revisions,
            name="critique_head",
        )
    
    def __call__(
        self,
        hidden_state: jnp.ndarray,
        is_training: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the generate → critique → revise loop.
        
        Args:
            hidden_state: Output hidden state [batch, d_model] or [batch, seq, d_model]
            is_training: Whether in training mode
            
        Returns:
            Dictionary with:
                - revised_output: Final (possibly revised) hidden state [batch, d_model]
                - quality_scores: List of quality scores at each step
                - num_revisions_applied: Number of revisions that were applied
                - process_rewards: List of reward signals per revision step
                - total_process_reward: Sum of all process rewards
                - accepted_early: Whether quality passed before max_revisions
        """
        # Pool if sequence input
        if hidden_state.ndim == 3:
            pooled = hidden_state.mean(axis=1)
        else:
            pooled = hidden_state
        
        current = pooled
        quality_scores = []
        process_rewards = []
        num_revisions_applied = 0
        accepted = jnp.array(False)
        
        # Always run all iterations for static computation graph (JAX tracing)
        for i in range(self.max_revisions + 1):
            # Critique current output
            critique_result = self.critique_head(current, is_training=is_training)
            quality = critique_result["quality_score"]
            quality_scores.append(quality)
            
            if i == 0:
                # First pass — no revision yet
                prev_quality = quality
                # Check if already good enough
                passes = quality.mean() >= self.quality_threshold
                accepted = accepted | passes
                continue
            
            # Apply revision (always compute, conditionally apply)
            revised = self.critique_head.revise(
                current,
                critique_result["revision_signal"],
                iteration=i - 1,
            )
            
            # Compute process reward: +0.3 if quality improved
            quality_delta = quality.mean() - prev_quality.mean()
            step_reward = jnp.where(quality_delta > 0.0, 0.3, 0.0)
            process_rewards.append(step_reward)
            
            # Only apply if not yet accepted
            should_apply = ~accepted
            apply_mask = jnp.where(should_apply, 1.0, 0.0)
            
            current = current + apply_mask * (revised - current)
            num_revisions_applied += int(jnp.where(should_apply, 1, 0))
            
            # Check acceptance after revision
            passes = quality.mean() >= self.quality_threshold
            accepted = accepted | passes
            
            prev_quality = quality
        
        total_process_reward = sum(process_rewards) if process_rewards else jnp.array(0.0)
        
        return {
            "revised_output": current,
            "quality_scores": quality_scores,
            "num_revisions_applied": num_revisions_applied,
            "process_rewards": process_rewards,
            "total_process_reward": total_process_reward,
            "accepted_early": accepted,
            "final_quality": quality_scores[-1] if quality_scores else jnp.array(0.0),
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
        
        # Verify/Reflect reasoning — replaces plain CoT when enabled
        self._verify_reflect_enabled = getattr(config, 'enable_verify_reflect', False)
        
        if self._verify_reflect_enabled:
            self.chain_of_thought = VerifyReflectReasoning(
                config.d_model,
                max_reasoning_steps=config.max_reasoning_steps,
                max_verify_steps=getattr(config, 'max_verify_steps', 3),
                confidence_threshold=getattr(config, 'verify_confidence_threshold', 0.85),
                name="verify_reflect_cot"
            )
        else:
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
            from src.core.rlm import RecursiveLanguageModel
            from src.config.rlm_config import RLMConfig
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
        
        result = {
            "reasoning_output": integrated_reasoning,
            "chain_of_thought": reasoning_result,
            "meta_learning": meta_result,
            "self_improvement": improvement_result,
            "reasoning_chain": reasoning_result["reasoning_chain"],
            "confidence_scores": reasoning_result["confidences"],
        }
        
        # Include verify/reflect metadata when enabled
        if self._verify_reflect_enabled:
            result["verification_scores"] = reasoning_result.get("verification_scores", [])
            result["num_reflections"] = reasoning_result.get("num_reflections", 0)
            result["reflection_deltas"] = reasoning_result.get("reflection_deltas", [])
        
        return result

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

    def get_rlm_outputs(
        self,
        query: jnp.ndarray,
        context_length: int,
    ) -> Optional[Dict[str, Any]]:
        if not self._rlm_enabled:
            return None
        tool_probs, term_prob, parameters, encoded_query = self.rlm(
            query, context_length, recursion_depth=0, tool_calls_used=0
        )
        return {
            "tool_probs": tool_probs,
            "termination_prob": term_prob,
            "parameters": parameters,
            "encoded_query": encoded_query,
        }
