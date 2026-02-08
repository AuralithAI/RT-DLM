"""
Module Adapters for Compute Controller

This module provides adapters that wrap existing RT-DLM modules to comply
with the ModuleContract interface required by the ComputeController.

Each adapter:
1. Takes a ComputeState as input
2. Runs the underlying module
3. Returns a standardized ModuleOutput

This allows the controller to orchestrate all modules uniformly.
"""

import haiku as hk
import jax
import jax.numpy as jnp
from typing import Dict, Optional, Any, Callable
import logging

from src.core.agi.compute_controller import (
    ModuleType, 
    ModuleOutput, 
    ComputeState
)

logger = logging.getLogger(__name__)


class ModuleAdapter(hk.Module):
    """
    Base class for module adapters.
    
    Provides common functionality for wrapping modules
    to comply with the ModuleContract interface.
    """
    
    def __init__(
        self,
        d_model: int,
        module_type: ModuleType,
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.d_model = d_model
        self.module_type = module_type
        
        # Confidence estimator - estimates reliability of module output
        self.confidence_estimator = hk.Sequential([
            hk.Linear(d_model // 2),
            jax.nn.silu,
            hk.Linear(1),
            jax.nn.sigmoid
        ], name="confidence_estimator")
        
        # Uncertainty estimator - estimates how much we don't know
        self.uncertainty_estimator = hk.Sequential([
            hk.Linear(d_model // 2),
            jax.nn.silu,
            hk.Linear(1),
            jax.nn.sigmoid
        ], name="uncertainty_estimator")
        
        # Delta projector - ensures output delta has correct shape
        self.delta_projector = hk.Linear(d_model, name="delta_projector")
        
        # Halt suggestion network
        self.halt_suggester = hk.Sequential([
            hk.Linear(d_model // 4),
            jax.nn.silu,
            hk.Linear(1),
            jax.nn.sigmoid
        ], name="halt_suggester")
    
    def _compute_confidence(self, output: jnp.ndarray) -> jnp.ndarray:
        """Compute confidence score for module output."""
        if output.ndim > 2:
            pooled = output.mean(axis=1)
        else:
            pooled = output
        return self.confidence_estimator(pooled)
    
    def _compute_uncertainty(
        self, 
        output: jnp.ndarray, 
        input_state: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute uncertainty based on input-output relationship."""
        if output.ndim > 2:
            output_pooled = output.mean(axis=1)
        else:
            output_pooled = output
            
        if input_state.ndim > 2:
            input_pooled = input_state.mean(axis=1)
        else:
            input_pooled = input_state
        
        # Uncertainty based on how different output is from input
        diff = output_pooled - input_pooled
        return self.uncertainty_estimator(diff)
    
    def _compute_delta(
        self, 
        output: jnp.ndarray, 
        input_state: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute delta to apply to hidden state."""
        if output.ndim > 2:
            output_pooled = output.mean(axis=1)
        else:
            output_pooled = output
            
        if input_state.ndim > 2:
            input_pooled = input_state.mean(axis=1)
        else:
            input_pooled = input_state
        
        # Delta is the projected difference
        diff = output_pooled - input_pooled
        return self.delta_projector(diff)
    
    def _should_suggest_halt(self, confidence: jnp.ndarray) -> bool:
        """Determine if module suggests halting."""
        # Suggest halt if very confident
        return bool(confidence.mean() > 0.9)
    
    def wrap_output(
        self,
        raw_output: Any,
        state: ComputeState,
        actual_cost: float,
        evidence: Optional[Dict[str, Any]] = None
    ) -> ModuleOutput:
        """
        Wrap raw module output into standardized ModuleOutput.
        
        Args:
            raw_output: The raw output from the underlying module
            state: Current compute state
            actual_cost: Actual compute cost
            evidence: Optional evidence/explanation
            
        Returns:
            Standardized ModuleOutput
        """
        # Extract output tensor
        if isinstance(raw_output, dict):
            # Try common keys
            for key in ["output", "hidden", "features", "result", "fused_memory"]:
                if key in raw_output:
                    output_tensor = raw_output[key]
                    break
            else:
                # Use first tensor value
                output_tensor = next(
                    (v for v in raw_output.values() if isinstance(v, jnp.ndarray)),
                    state.hidden_pooled
                )
        elif isinstance(raw_output, jnp.ndarray):
            output_tensor = raw_output
        else:
            output_tensor = state.hidden_pooled
        
        # Compute standardized outputs
        confidence = self._compute_confidence(output_tensor)
        uncertainty = self._compute_uncertainty(output_tensor, state.hidden_pooled)
        delta = self._compute_delta(output_tensor, state.hidden_pooled)
        suggests_halt = self._should_suggest_halt(confidence)
        
        return ModuleOutput(
            hidden_delta=delta,
            confidence=confidence,
            uncertainty=uncertainty,
            actual_cost=actual_cost,
            evidence=evidence or {"raw_output_keys": list(raw_output.keys()) if isinstance(raw_output, dict) else ["tensor"]},
            suggests_halt=suggests_halt
        )


class MemoryRetrievalAdapter(ModuleAdapter):
    """
    Adapter for memory retrieval (LTM/STM/MTM).
    
    Wraps HierarchicalMemoryFusion or similar memory modules.
    """
    
    def __init__(self, d_model: int, memory_module: Optional[hk.Module] = None, name: str = "memory_adapter"):
        super().__init__(d_model, ModuleType.MEMORY_RETRIEVAL, name)
        self.memory_module = memory_module
        
        # Fallback memory attention if no module provided
        self.fallback_attention = hk.MultiHeadAttention(
            num_heads=4,
            key_size=d_model // 4,
            w_init=hk.initializers.TruncatedNormal(stddev=0.02),
            name="fallback_memory_attention"
        )
    
    def __call__(
        self,
        state: ComputeState,
        ltm: Optional[jnp.ndarray] = None,
        stm: Optional[jnp.ndarray] = None,
        mtm: Optional[jnp.ndarray] = None,
        is_training: bool = True
    ) -> ModuleOutput:
        """
        Execute memory retrieval.
        
        Args:
            state: Current compute state
            ltm: Long-term memory [batch, mem_size, d_model]
            stm: Short-term memory [batch, mem_size, d_model]
            mtm: Medium-term memory [batch, mem_size, d_model]
            is_training: Training mode flag
        """
        hidden = state.hidden_pooled
        
        if self.memory_module is not None:
            # Use provided memory module
            raw_output = self.memory_module(
                ltm=ltm or jnp.zeros((hidden.shape[0], 1, self.d_model)),
                stm=stm or jnp.zeros((hidden.shape[0], 1, self.d_model)),
                mtm=mtm or jnp.zeros((hidden.shape[0], 1, self.d_model)),
                context=hidden,
                is_training=is_training
            )
        else:
            # Fallback: simple attention over available memories
            memories = []
            if ltm is not None:
                memories.append(ltm)
            if stm is not None:
                memories.append(stm)
            if mtm is not None:
                memories.append(mtm)
            
            if memories:
                combined_memory = jnp.concatenate(memories, axis=1)
                query = hidden[:, None, :]
                attended = self.fallback_attention(query, combined_memory, combined_memory)
                raw_output = {"fused_memory": attended.squeeze(1)}
            else:
                raw_output = {"fused_memory": hidden}
        
        # Update memory summary in evidence
        evidence = {
            "ltm_used": ltm is not None,
            "stm_used": stm is not None,
            "mtm_used": mtm is not None
        }
        
        return self.wrap_output(raw_output, state, 0.05, evidence)


class GraphReasoningAdapter(ModuleAdapter):
    """
    Adapter for graph-based reasoning.
    
    Wraps MultiHopGraphReasoner, SemanticParser, etc.
    """
    
    def __init__(
        self, 
        d_model: int, 
        graph_module: Optional[hk.Module] = None,
        num_hops: int = 3,
        name: str = "graph_reasoning_adapter"
    ):
        super().__init__(d_model, ModuleType.GRAPH_REASONING, name)
        self.graph_module = graph_module
        self.num_hops = num_hops
        
        # Fallback graph reasoning
        self.graph_attention = hk.MultiHeadAttention(
            num_heads=8,
            key_size=d_model // 8,
            w_init=hk.initializers.TruncatedNormal(stddev=0.02),
            name="graph_attention"
        )
        
        self.hop_processor = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(d_model),
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        ], name="hop_processor")
    
    def __call__(
        self,
        state: ComputeState,
        adjacency: Optional[jnp.ndarray] = None,
        query: Optional[jnp.ndarray] = None,
        is_training: bool = True
    ) -> ModuleOutput:
        """
        Execute graph reasoning.
        
        Args:
            state: Current compute state
            adjacency: Optional adjacency matrix [batch, nodes, nodes]
            query: Optional query for guided reasoning
            is_training: Training mode flag
        """
        hidden = state.hidden
        
        if hidden.ndim == 2:
            hidden = hidden[:, None, :]
        
        if self.graph_module is not None:
            # Use provided graph module
            raw_output = self.graph_module(
                node_features=hidden,
                adjacency=adjacency,
                query=query,
                is_training=is_training
            )
        else:
            # Fallback: multi-hop attention
            current = hidden
            reasoning_trace = []
            
            for _ in range(self.num_hops):
                # Self-attention as proxy for graph propagation
                attended = self.graph_attention(current, current, current)
                current = self.hop_processor(attended)
                reasoning_trace.append(current.mean(axis=1))
            
            raw_output = {
                "output": current,
                "reasoning_trace": jnp.stack(reasoning_trace, axis=1)
            }
        
        evidence = {
            "num_hops": self.num_hops,
            "has_adjacency": adjacency is not None,
            "has_query": query is not None
        }
        
        return self.wrap_output(raw_output, state, 0.15, evidence)


class SymbolicReasoningAdapter(ModuleAdapter):
    """
    Adapter for symbolic reasoning.
    
    Wraps SymbolicReasoningBackbone, RuleBasedEngine, etc.
    """
    
    def __init__(
        self, 
        d_model: int, 
        symbolic_module: Optional[hk.Module] = None,
        name: str = "symbolic_reasoning_adapter"
    ):
        super().__init__(d_model, ModuleType.SYMBOLIC_REASONING, name)
        self.symbolic_module = symbolic_module
        
        # Fallback: differentiable rule application
        self.rule_encoder = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(d_model)
        ], name="rule_encoder")
        
        self.rule_applier = hk.Sequential([
            hk.Linear(d_model * 2),
            jax.nn.silu,
            hk.Linear(d_model),
            jax.nn.tanh
        ], name="rule_applier")
    
    def __call__(
        self,
        state: ComputeState,
        rules: Optional[jnp.ndarray] = None,
        is_training: bool = True
    ) -> ModuleOutput:
        """Execute symbolic reasoning."""
        hidden = state.hidden_pooled
        
        if self.symbolic_module is not None:
            raw_output = self.symbolic_module(hidden)
        else:
            # Fallback: encode input as "rules" and apply
            encoded_rules = self.rule_encoder(hidden)
            rule_input = jnp.concatenate([hidden, encoded_rules], axis=-1)
            raw_output = {"output": self.rule_applier(rule_input)}
        
        evidence = {"rules_provided": rules is not None}
        return self.wrap_output(raw_output, state, 0.10, evidence)


class ProbabilisticAdapter(ModuleAdapter):
    """
    Adapter for probabilistic inference.
    
    Wraps ProbabilisticBackbone, BayesianNeuralNetwork, etc.
    """
    
    def __init__(
        self, 
        d_model: int, 
        probabilistic_module: Optional[hk.Module] = None,
        name: str = "probabilistic_adapter"
    ):
        super().__init__(d_model, ModuleType.PROBABILISTIC, name)
        self.probabilistic_module = probabilistic_module
        
        # Fallback: dropout-based uncertainty
        self.mean_network = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(d_model)
        ], name="mean_network")
        
        self.variance_network = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(d_model),
            jax.nn.softplus  # Ensure positive variance
        ], name="variance_network")
    
    def __call__(
        self,
        state: ComputeState,
        is_training: bool = True
    ) -> ModuleOutput:
        """Execute probabilistic inference."""
        hidden = state.hidden_pooled
        
        if self.probabilistic_module is not None:
            raw_output = self.probabilistic_module(hidden)
        else:
            # Fallback: predict mean and variance
            mean = self.mean_network(hidden)
            variance = self.variance_network(hidden)
            
            # Sample if training (reparameterization trick)
            if is_training:
                rng = hk.next_rng_key()
                eps = jax.random.normal(rng, mean.shape)
                output = mean + jnp.sqrt(variance) * eps
            else:
                output = mean
            
            raw_output = {
                "output": output,
                "mean": mean,
                "variance": variance,
                "epistemic_uncertainty": variance.mean(axis=-1, keepdims=True)
            }
        
        # Override uncertainty with probabilistic estimate
        module_output = self.wrap_output(raw_output, state, 0.08, {"probabilistic": True})
        
        # Use variance as uncertainty if available
        if isinstance(raw_output, dict) and "variance" in raw_output:
            uncertainty = jnp.clip(raw_output["variance"].mean(axis=-1, keepdims=True), 0, 1)
            module_output = ModuleOutput(
                hidden_delta=module_output.hidden_delta,
                confidence=module_output.confidence,
                uncertainty=uncertainty,
                actual_cost=module_output.actual_cost,
                evidence=module_output.evidence,
                suggests_halt=module_output.suggests_halt
            )
        
        return module_output


class MoERoutingAdapter(ModuleAdapter):
    """
    Adapter for Mixture of Experts routing.
    
    Wraps SparseMoE module.
    """
    
    def __init__(
        self, 
        d_model: int, 
        moe_module: Optional[hk.Module] = None,
        num_experts: int = 8,
        top_k: int = 2,
        name: str = "moe_routing_adapter"
    ):
        super().__init__(d_model, ModuleType.MOE_ROUTING, name)
        self.moe_module = moe_module
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Fallback: simple gated experts
        self.gate = hk.Linear(num_experts, name="expert_gate")
        self.experts = [
            hk.Sequential([
                hk.Linear(d_model * 2),
                jax.nn.silu,
                hk.Linear(d_model)
            ], name=f"expert_{i}")
            for i in range(num_experts)
        ]
    
    def __call__(
        self,
        state: ComputeState,
        is_training: bool = True
    ) -> ModuleOutput:
        """Execute MoE routing."""
        hidden = state.hidden_pooled
        
        if self.moe_module is not None:
            raw_output = self.moe_module(hidden[:, None, :] if hidden.ndim == 2 else hidden)
            if isinstance(raw_output, tuple):
                raw_output = {"output": raw_output[0], "aux_loss": raw_output[1] if len(raw_output) > 1 else 0.0}
        else:
            # Fallback: compute gate and route to top-k experts
            gate_logits = self.gate(hidden)
            gate_probs = jax.nn.softmax(gate_logits, axis=-1)
            
            # Get top-k experts
            top_k_probs, top_k_indices = jax.lax.top_k(gate_probs, self.top_k)
            top_k_probs = top_k_probs / top_k_probs.sum(axis=-1, keepdims=True)
            
            # Compute expert outputs (simplified)
            expert_outputs = jnp.stack([expert(hidden) for expert in self.experts], axis=1)
            
            # Weighted combination of top-k
            output = jnp.zeros_like(hidden)
            for i in range(self.top_k):
                expert_idx = top_k_indices[:, i]
                expert_out = expert_outputs[jnp.arange(hidden.shape[0]), expert_idx]
                output = output + top_k_probs[:, i:i+1] * expert_out
            
            raw_output = {
                "output": output,
                "gate_probs": gate_probs,
                "top_k_indices": top_k_indices
            }
        
        evidence = {
            "num_experts": self.num_experts,
            "top_k": self.top_k
        }
        return self.wrap_output(raw_output, state, 0.12, evidence)


class ConsciousnessAdapter(ModuleAdapter):
    """
    Adapter for consciousness simulation.
    
    Wraps ConsciousnessSimulator module.
    """
    
    def __init__(
        self, 
        d_model: int, 
        consciousness_module: Optional[hk.Module] = None,
        name: str = "consciousness_adapter"
    ):
        super().__init__(d_model, ModuleType.CONSCIOUSNESS, name)
        self.consciousness_module = consciousness_module
        
        # Fallback: self-attention introspection
        self.introspection = hk.MultiHeadAttention(
            num_heads=4,
            key_size=d_model // 4,
            w_init=hk.initializers.TruncatedNormal(stddev=0.02),
            name="introspection"
        )
        
        self.meta_awareness = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(d_model),
            jax.nn.tanh
        ], name="meta_awareness")
    
    def __call__(
        self,
        state: ComputeState,
        external_input: Optional[jnp.ndarray] = None,
        is_training: bool = True
    ) -> ModuleOutput:
        """Execute consciousness simulation."""
        hidden = state.hidden
        if hidden.ndim == 2:
            hidden = hidden[:, None, :]
        
        if self.consciousness_module is not None:
            raw_output = self.consciousness_module(
                internal_state=hidden,
                external_input=external_input if external_input is not None else hidden
            )
        else:
            # Fallback: introspection + meta-awareness
            introspected = self.introspection(hidden, hidden, hidden)
            meta = self.meta_awareness(introspected.mean(axis=1))
            
            raw_output = {
                "introspection": introspected,
                "meta_awareness": meta,
                "output": meta
            }
        
        evidence = {"introspection_steps": 1}
        return self.wrap_output(raw_output, state, 0.10, evidence)


class OutputGenerationAdapter(ModuleAdapter):
    """
    Adapter for final output generation.
    
    This module is always called last and cannot be skipped.
    """
    
    def __init__(
        self, 
        d_model: int, 
        vocab_size: int,
        output_head: Optional[hk.Module] = None,
        name: str = "output_generation_adapter"
    ):
        super().__init__(d_model, ModuleType.OUTPUT_GENERATION, name)
        self.vocab_size = vocab_size
        self.output_head = output_head
        
        # Fallback output projection
        self.fallback_head = hk.Linear(vocab_size, name="fallback_output_head")
    
    def __call__(
        self,
        state: ComputeState,
        is_training: bool = True
    ) -> ModuleOutput:
        """Generate final output."""
        hidden = state.hidden_pooled
        
        if self.output_head is not None:
            logits = self.output_head(hidden)
        else:
            logits = self.fallback_head(hidden)
        
        # Output generation always suggests halt
        confidence = self._compute_confidence(hidden)
        
        return ModuleOutput(
            hidden_delta=jnp.zeros_like(hidden),  # No more updates
            confidence=confidence,
            uncertainty=1.0 - confidence,  # Inverse confidence
            actual_cost=0.05,
            evidence={"logits_shape": logits.shape},
            suggests_halt=True,  # Always halt after output
            recommended_next=None
        )


def create_module_executors(
    d_model: int,
    vocab_size: int,
    existing_modules: Optional[Dict[str, hk.Module]] = None
) -> Dict[ModuleType, Callable]:
    """
    Create executor functions for all module types.
    
    Args:
        d_model: Model dimension
        vocab_size: Vocabulary size for output
        existing_modules: Optional dict of existing modules to wrap
        
    Returns:
        Dict mapping ModuleType to callable executors
    """
    existing_modules = existing_modules or {}
    
    # Create adapters
    memory_adapter = MemoryRetrievalAdapter(
        d_model=d_model,
        memory_module=existing_modules.get("memory_fusion")
    )
    
    graph_adapter = GraphReasoningAdapter(
        d_model=d_model,
        graph_module=existing_modules.get("graph_reasoner")
    )
    
    symbolic_adapter = SymbolicReasoningAdapter(
        d_model=d_model,
        symbolic_module=existing_modules.get("symbolic_reasoning")
    )
    
    probabilistic_adapter = ProbabilisticAdapter(
        d_model=d_model,
        probabilistic_module=existing_modules.get("probabilistic")
    )
    
    moe_adapter = MoERoutingAdapter(
        d_model=d_model,
        moe_module=existing_modules.get("moe")
    )
    
    consciousness_adapter = ConsciousnessAdapter(
        d_model=d_model,
        consciousness_module=existing_modules.get("consciousness")
    )
    
    output_adapter = OutputGenerationAdapter(
        d_model=d_model,
        vocab_size=vocab_size,
        output_head=existing_modules.get("output_head")
    )
    
    # Create executor functions
    def memory_executor(state: ComputeState, is_training: bool) -> ModuleOutput:
        return memory_adapter(state, is_training=is_training)
    
    def graph_executor(state: ComputeState, is_training: bool) -> ModuleOutput:
        return graph_adapter(state, is_training=is_training)
    
    def symbolic_executor(state: ComputeState, is_training: bool) -> ModuleOutput:
        return symbolic_adapter(state, is_training=is_training)
    
    def probabilistic_executor(state: ComputeState, is_training: bool) -> ModuleOutput:
        return probabilistic_adapter(state, is_training=is_training)
    
    def moe_executor(state: ComputeState, is_training: bool) -> ModuleOutput:
        return moe_adapter(state, is_training=is_training)
    
    def consciousness_executor(state: ComputeState, is_training: bool) -> ModuleOutput:
        return consciousness_adapter(state, is_training=is_training)
    
    def output_executor(state: ComputeState, is_training: bool) -> ModuleOutput:
        return output_adapter(state, is_training=is_training)
    
    return {
        ModuleType.MEMORY_RETRIEVAL: memory_executor,
        ModuleType.GRAPH_REASONING: graph_executor,
        ModuleType.SYMBOLIC_REASONING: symbolic_executor,
        ModuleType.PROBABILISTIC: probabilistic_executor,
        ModuleType.MOE_ROUTING: moe_executor,
        ModuleType.CONSCIOUSNESS: consciousness_executor,
        ModuleType.OUTPUT_GENERATION: output_executor,
    }
