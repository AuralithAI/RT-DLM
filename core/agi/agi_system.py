"""
Core AGI System Abstraction

This module provides a central abstraction for unifying AGI subsystems,
managing progression stages, and orchestrating component interactions.

Key Features:
- AGISystemAbstraction: Central hub for AGI component integration
- Stage tracking (0-6 levels) with configurable thresholds
- Component unification via learned fusion
- Self-improvement metrics tracking
- Ethical decision-making support for finance/cybersecurity

AGI Progression Stages:
    Stage 0 - Reactive: Basic stimulus-response
    Stage 1 - Reflective: Self-monitoring capabilities
    Stage 2 - Learning: Adaptive behavior modification
    Stage 3 - Meta-cognitive: Reasoning about reasoning
    Stage 4 - Self-improving: Autonomous capability enhancement
    Stage 5 - Creative: Novel solution generation
    Stage 6 - Conscious: Full self-aware AGI (theoretical)

Dependencies:
- Existing cognitive modules (consciousness, reasoning, creativity)
- JAX/Haiku for differentiable operations
"""

import haiku as hk
import jax
import jax.numpy as jnp
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import IntEnum
import logging

logger = logging.getLogger(__name__)


class AGIStage(IntEnum):
    """AGI progression stages with increasing capability levels"""
    REACTIVE = 0        # Basic stimulus-response
    REFLECTIVE = 1      # Self-monitoring capabilities
    LEARNING = 2        # Adaptive behavior modification
    META_COGNITIVE = 3  # Reasoning about reasoning
    SELF_IMPROVING = 4  # Autonomous capability enhancement
    CREATIVE = 5        # Novel solution generation
    CONSCIOUS = 6       # Full self-aware AGI (theoretical)


@dataclass
class StageThresholds:
    """Thresholds for AGI stage progression"""
    reflective: float = 0.2      # Threshold to reach Stage 1
    learning: float = 0.35       # Threshold to reach Stage 2
    meta_cognitive: float = 0.5  # Threshold to reach Stage 3
    self_improving: float = 0.65 # Threshold to reach Stage 4
    creative: float = 0.8        # Threshold to reach Stage 5
    conscious: float = 0.95      # Threshold to reach Stage 6


@dataclass
class AGIMetrics:
    """Metrics tracking AGI system performance and progression"""
    current_stage: int
    stage_progress: float  # Progress toward next stage [0, 1]
    consciousness_score: float
    reasoning_score: float
    creativity_score: float
    ethical_alignment: float
    self_improvement_rate: float
    component_coherence: float  # How well components work together


class ComponentFusion(hk.Module):
    """
    Fuses outputs from multiple AGI components into unified representations.
    
    Uses learned attention-based fusion to combine consciousness,
    reasoning, creativity, and other cognitive outputs.
    
    Args:
        d_model: Model dimension
        num_components: Number of components to fuse
        name: Module name
    """
    
    def __init__(
        self,
        d_model: int,
        num_components: int = 4,
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.d_model = d_model
        self.num_components = num_components
        
    def __call__(
        self,
        components: Dict[str, jnp.ndarray],
        component_weights: Optional[jnp.ndarray] = None
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """
        Fuse multiple component outputs.
        
        Args:
            components: Dict mapping component names to outputs [batch, d_model]
            component_weights: Optional preset weights [num_components]
            
        Returns:
            fused_output: Unified representation [batch, d_model]
            fusion_info: Dictionary with fusion metadata
        """
        # Stack components
        component_list = list(components.values())
        component_names = list(components.keys())
        
        # Ensure all components have same batch dimension
        stacked = jnp.stack(component_list, axis=1)  # [batch, num_components, d_model]
        num_comps = stacked.shape[1]
        
        # Learn component importance weights via attention
        importance_scorer = hk.Linear(1, name="importance_scorer")
        importance_logits = importance_scorer(stacked).squeeze(-1)  # [batch, num_components]
        
        if component_weights is not None:
            # Combine learned and preset weights
            importance_logits = importance_logits + component_weights[None, :]
        
        attention_weights = jax.nn.softmax(importance_logits, axis=-1)  # [batch, num_components]
        
        # Weighted combination
        weighted_sum = jnp.einsum('bc,bcd->bd', attention_weights, stacked)  # [batch, d_model]
        
        # Project through fusion network
        fusion_network = hk.Sequential([
            hk.Linear(self.d_model * 2),
            jax.nn.silu,
            hk.Linear(self.d_model),
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        ], name="fusion_network")
        
        # Also compute mean for residual
        component_mean = jnp.mean(stacked, axis=1)  # [batch, d_model]
        
        fused_output = fusion_network(weighted_sum) + component_mean * 0.3
        
        # Compute coherence score (how similar are the components?)
        # Use cosine similarity between pairs
        normalized = stacked / (jnp.linalg.norm(stacked, axis=-1, keepdims=True) + 1e-8)
        similarity_matrix = jnp.einsum('bcd,bed->bce', normalized, normalized)
        
        # Average off-diagonal similarity (coherence)
        mask = 1.0 - jnp.eye(num_comps)[None, :, :]
        coherence = jnp.sum(similarity_matrix * mask, axis=(1, 2)) / (num_comps * (num_comps - 1) + 1e-8)
        
        fusion_info = {
            "attention_weights": attention_weights,
            "component_names": component_names,
            "coherence": coherence,
            "component_mean": component_mean
        }
        
        return fused_output, fusion_info


class StageTracker(hk.Module):
    """
    Tracks and predicts AGI progression stage.
    
    Monitors various capability metrics to determine current
    stage and progress toward next stage.
    
    Args:
        d_model: Model dimension
        thresholds: StageThresholds configuration
        name: Module name
    """
    
    def __init__(
        self,
        d_model: int,
        thresholds: Optional[StageThresholds] = None,
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.d_model = d_model
        self.thresholds = thresholds or StageThresholds()
        
    def _get_stage_thresholds_list(self) -> List[float]:
        """Return ordered list of stage thresholds."""
        return [
            0.0,
            self.thresholds.reflective,
            self.thresholds.learning,
            self.thresholds.meta_cognitive,
            self.thresholds.self_improving,
            self.thresholds.creative,
            self.thresholds.conscious
        ]
    
    def _determine_stage(self, score: float) -> int:
        """Determine stage based on score and thresholds."""
        thresholds = self._get_stage_thresholds_list()
        # Find highest stage whose threshold is met
        for stage in range(6, -1, -1):
            if score >= thresholds[stage]:
                return stage
        return 0
    
    def _compute_stage_progress(self, score: float, current_stage: int) -> float:
        """Compute progress toward next stage."""
        thresholds = self._get_stage_thresholds_list()
        current_threshold = thresholds[current_stage]
        next_threshold = thresholds[min(current_stage + 1, 6)]
        
        if next_threshold > current_threshold:
            progress = (score - current_threshold) / (next_threshold - current_threshold)
            return max(0.0, min(1.0, progress))
        return 1.0
    
    def _adjust_score_with_output(
        self, 
        base_score: jnp.ndarray, 
        output: Optional[Dict[str, jnp.ndarray]], 
        key: str
    ) -> jnp.ndarray:
        """Adjust capability score based on component output."""
        if output is None or key not in output:
            return base_score
            
        value = output[key]
        if isinstance(value, list):
            value = jnp.stack([v.mean() for v in value])
        if value.ndim > 1:
            value = value.mean(axis=tuple(range(1, value.ndim)))
        if value.ndim > 0:
            value = value.mean()
        return (base_score + jnp.abs(value)) / 2
        
    def __call__(
        self,
        unified_representation: jnp.ndarray,
        consciousness_output: Optional[Dict[str, jnp.ndarray]] = None,
        reasoning_output: Optional[Dict[str, jnp.ndarray]] = None,
        creativity_output: Optional[Dict[str, jnp.ndarray]] = None
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Determine current AGI stage and progression metrics.
        
        Args:
            unified_representation: Fused component output [batch, d_model]
            consciousness_output: Output from ConsciousnessSimulator
            reasoning_output: Output from ReasoningEngine
            creativity_output: Output from CreativeGenerationEngine
            
        Returns:
            current_stage: Integer stage (0-6)
            stage_info: Dictionary with progression metrics
        """
        # Capability scorers
        consciousness_scorer = hk.Sequential([
            hk.Linear(self.d_model), jax.nn.silu, hk.Linear(1), jax.nn.sigmoid
        ], name="consciousness_scorer")
        reasoning_scorer = hk.Sequential([
            hk.Linear(self.d_model), jax.nn.silu, hk.Linear(1), jax.nn.sigmoid
        ], name="reasoning_scorer")
        creativity_scorer = hk.Sequential([
            hk.Linear(self.d_model), jax.nn.silu, hk.Linear(1), jax.nn.sigmoid
        ], name="creativity_scorer")
        overall_scorer = hk.Sequential([
            hk.Linear(self.d_model), jax.nn.silu, hk.Linear(1), jax.nn.sigmoid
        ], name="overall_scorer")
        
        # Compute base capability scores
        consciousness_score = consciousness_scorer(unified_representation).squeeze(-1)
        reasoning_score = reasoning_scorer(unified_representation).squeeze(-1)
        creativity_score = creativity_scorer(unified_representation).squeeze(-1)
        overall_score = overall_scorer(unified_representation).squeeze(-1)
        
        # Adjust with component outputs
        consciousness_score = self._adjust_score_with_output(
            consciousness_score, consciousness_output, "meta_awareness"
        )
        reasoning_score = self._adjust_score_with_output(
            reasoning_score, reasoning_output, "confidence_scores"
        )
        creativity_score = self._adjust_score_with_output(
            creativity_score, creativity_output, "novelty_score"
        )
        
        # Determine stage
        score = float(overall_score[0]) if overall_score.ndim > 0 else float(overall_score)
        current_stage = self._determine_stage(score)
        stage_progress = self._compute_stage_progress(score, current_stage)
        
        stage_info = {
            "current_stage": current_stage,
            "stage_name": AGIStage(current_stage).name,
            "stage_progress": stage_progress,
            "overall_score": overall_score,
            "consciousness_score": consciousness_score,
            "reasoning_score": reasoning_score,
            "creativity_score": creativity_score,
            "next_stage_threshold": self._get_stage_thresholds_list()[min(current_stage + 1, 6)]
        }
        
        return current_stage, stage_info


class EthicalAlignmentModule(hk.Module):
    """
    Evaluates and ensures ethical alignment of AGI decisions.
    
    Critical for applications in finance and cybersecurity where
    decisions must be transparent, fair, and aligned with human values.
    
    Args:
        d_model: Model dimension
        num_ethical_dimensions: Number of ethical dimensions to track
        name: Module name
    """
    
    def __init__(
        self,
        d_model: int,
        num_ethical_dimensions: int = 8,
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.d_model = d_model
        self.num_ethical_dimensions = num_ethical_dimensions
        
        # Ethical dimensions:
        # 0: Fairness, 1: Transparency, 2: Privacy, 3: Safety
        # 4: Accountability, 5: Non-maleficence, 6: Beneficence, 7: Autonomy
        self.dimension_names = [
            "fairness", "transparency", "privacy", "safety",
            "accountability", "non_maleficence", "beneficence", "autonomy"
        ]
        
    def __call__(
        self,
        decision_representation: jnp.ndarray,
        context: Optional[jnp.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Evaluate ethical alignment of a decision.
        
        Args:
            decision_representation: Representation of decision [batch, d_model]
            context: Optional context for the decision [batch, seq, d_model]
            
        Returns:
            Dictionary with ethical scores and recommendations
        """
        # Ethical dimension evaluator
        ethical_evaluator = hk.Linear(
            self.num_ethical_dimensions, 
            name="ethical_evaluator"
        )
        
        # Context integrator
        if context is not None:
            context_summary = context.mean(axis=1)
            combined = jnp.concatenate([decision_representation, context_summary], axis=-1)
            context_proj = hk.Linear(self.d_model, name="context_projection")
            decision_with_context = context_proj(combined)
        else:
            decision_with_context = decision_representation
        
        # Compute ethical dimension scores
        ethical_logits = ethical_evaluator(decision_with_context)
        ethical_scores = jax.nn.sigmoid(ethical_logits)  # [batch, num_dimensions]
        
        # Overall alignment score (weighted average)
        # Higher weight on safety and non-maleficence
        weights = jnp.array([1.0, 1.0, 1.2, 1.5, 1.0, 1.5, 1.2, 1.0])
        weighted_scores = ethical_scores * weights[None, :]
        overall_alignment = weighted_scores.sum(axis=-1) / weights.sum()
        
        # Risk assessment
        risk_assessor = hk.Sequential([
            hk.Linear(self.d_model),
            jax.nn.silu,
            hk.Linear(1),
            jax.nn.sigmoid
        ], name="risk_assessor")
        
        risk_score = risk_assessor(decision_with_context).squeeze(-1)
        
        # Compute flags for dimensions below threshold
        threshold = 0.5
        flags = ethical_scores < threshold  # [batch, num_dimensions]
        
        return {
            "ethical_scores": ethical_scores,
            "dimension_names": self.dimension_names,
            "overall_alignment": overall_alignment,
            "risk_score": risk_score,
            "ethical_flags": flags,
            "requires_review": jnp.any(flags, axis=-1)  # True if any dimension flagged
        }


class AGISystemAbstraction(hk.Module):
    """
    Central abstraction for unifying AGI subsystems.
    
    Provides a hub for:
    - Fusing outputs from consciousness, reasoning, creativity modules
    - Tracking AGI progression stages (0-6)
    - Ensuring ethical alignment
    - Managing self-improvement metrics
    
    Designed for orchestrating complex AGI systems in applications
    like health diagnostics, finance, and cybersecurity.
    
    Args:
        d_model: Model dimension
        num_components: Number of AGI components to unify
        stage_thresholds: Optional custom stage thresholds
        enable_ethical_tracking: Whether to track ethical alignment
        name: Module name
    """
    
    def __init__(
        self,
        d_model: int = 512,
        num_components: int = 4,
        stage_thresholds: Optional[StageThresholds] = None,
        enable_ethical_tracking: bool = True,
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.d_model = d_model
        self.num_components = num_components
        self.stage_thresholds = stage_thresholds or StageThresholds()
        self.enable_ethical_tracking = enable_ethical_tracking
        
    def unify_components(
        self,
        consciousness_output: jnp.ndarray,
        reasoning_output: jnp.ndarray,
        creativity_output: Optional[jnp.ndarray] = None,
        additional_outputs: Optional[Dict[str, jnp.ndarray]] = None
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """
        Unify outputs from multiple AGI components.
        
        Uses learned fusion to combine component outputs into a
        coherent unified representation.
        
        Args:
            consciousness_output: Output from consciousness module [batch, d_model]
            reasoning_output: Output from reasoning module [batch, d_model]
            creativity_output: Optional creativity output [batch, d_model]
            additional_outputs: Optional dict of additional component outputs
            
        Returns:
            unified: Unified representation [batch, d_model]
            fusion_info: Dictionary with fusion metadata
        """
        # Build components dictionary
        components = {
            "consciousness": consciousness_output,
            "reasoning": reasoning_output
        }
        
        if creativity_output is not None:
            components["creativity"] = creativity_output
            
        if additional_outputs is not None:
            components.update(additional_outputs)
        
        # Apply component fusion
        fusion = ComponentFusion(
            d_model=self.d_model,
            num_components=len(components),
            name="component_fusion"
        )
        
        unified, fusion_info = fusion(components)
        
        return unified, fusion_info
    
    def track_stage(
        self,
        unified_representation: jnp.ndarray,
        consciousness_output: Optional[Dict[str, jnp.ndarray]] = None,
        reasoning_output: Optional[Dict[str, jnp.ndarray]] = None,
        creativity_output: Optional[Dict[str, jnp.ndarray]] = None
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Track current AGI progression stage.
        
        Args:
            unified_representation: Fused component output [batch, d_model]
            consciousness_output: Full consciousness output dict
            reasoning_output: Full reasoning output dict
            creativity_output: Full creativity output dict
            
        Returns:
            stage: Current stage (0-6)
            stage_info: Detailed stage information
        """
        tracker = StageTracker(
            d_model=self.d_model,
            thresholds=self.stage_thresholds,
            name="stage_tracker"
        )
        
        return tracker(
            unified_representation,
            consciousness_output,
            reasoning_output,
            creativity_output
        )
    
    def evaluate_ethics(
        self,
        decision_representation: jnp.ndarray,
        context: Optional[jnp.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Evaluate ethical alignment of a decision.
        
        Args:
            decision_representation: Decision to evaluate [batch, d_model]
            context: Optional decision context [batch, seq, d_model]
            
        Returns:
            Dictionary with ethical scores and flags
        """
        if not self.enable_ethical_tracking:
            return {"enabled": False}
            
        ethics_module = EthicalAlignmentModule(
            d_model=self.d_model,
            name="ethical_alignment"
        )
        
        return ethics_module(decision_representation, context)
    
    def compute_self_improvement_signal(
        self,
        current_metrics: Dict[str, float],
        previous_metrics: Optional[Dict[str, float]] = None
    ) -> jnp.ndarray:
        """
        Compute self-improvement signal based on metric changes.
        
        Args:
            current_metrics: Current performance metrics
            previous_metrics: Previous performance metrics
            
        Returns:
            improvement_signal: Signal for self-improvement [d_model]
        """
        # Extract key metrics
        current_score = current_metrics.get("overall_score", 0.5)
        current_stage = current_metrics.get("current_stage", 0)
        
        if previous_metrics is not None:
            prev_score = previous_metrics.get("overall_score", 0.5)
            prev_stage = previous_metrics.get("current_stage", 0)
            
            # Compute improvement
            score_delta = current_score - prev_score
            stage_delta = current_stage - prev_stage
            
            improvement_rate = score_delta + stage_delta * 0.1
        else:
            improvement_rate = 0.0
            
        # Create improvement signal embedding
        signal_generator = hk.Sequential([
            hk.Linear(self.d_model),
            jax.nn.tanh
        ], name="improvement_signal")
        
        # Create input from metrics
        metrics_input = jnp.array([
            current_score,
            float(current_stage) / 6.0,
            improvement_rate,
            current_metrics.get("consciousness_score", 0.5),
            current_metrics.get("reasoning_score", 0.5),
            current_metrics.get("creativity_score", 0.5),
            current_metrics.get("ethical_alignment", 0.5),
            current_metrics.get("coherence", 0.5)
        ])
        
        # Pad to d_model for projection
        padded = jnp.zeros(self.d_model)
        padded = padded.at[:len(metrics_input)].set(metrics_input)
        
        return signal_generator(padded[None, :])[0]
        
    def __call__(
        self,
        consciousness_output: jnp.ndarray,
        reasoning_output: jnp.ndarray,
        creativity_output: Optional[jnp.ndarray] = None,
        consciousness_dict: Optional[Dict[str, jnp.ndarray]] = None,
        reasoning_dict: Optional[Dict[str, jnp.ndarray]] = None,
        creativity_dict: Optional[Dict[str, jnp.ndarray]] = None,
        context: Optional[jnp.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Full AGI system orchestration.
        
        Unifies components, tracks stage, evaluates ethics, and
        computes self-improvement signals.
        
        Args:
            consciousness_output: Consciousness embedding [batch, d_model]
            reasoning_output: Reasoning embedding [batch, d_model]
            creativity_output: Optional creativity embedding [batch, d_model]
            consciousness_dict: Full consciousness output dict
            reasoning_dict: Full reasoning output dict
            creativity_dict: Full creativity output dict
            context: Optional context for ethical evaluation
            
        Returns:
            Dictionary with unified output, stage info, ethics, etc.
        """
        # Unify components
        unified, fusion_info = self.unify_components(
            consciousness_output,
            reasoning_output,
            creativity_output
        )
        
        # Track stage
        stage, stage_info = self.track_stage(
            unified,
            consciousness_dict,
            reasoning_dict,
            creativity_dict
        )
        
        # Evaluate ethics
        ethics_info = self.evaluate_ethics(unified, context)
        
        # Build metrics for self-improvement
        current_metrics = {
            "overall_score": float(stage_info["overall_score"][0]) if stage_info["overall_score"].ndim > 0 else float(stage_info["overall_score"]),
            "current_stage": stage,
            "consciousness_score": float(stage_info["consciousness_score"][0]) if stage_info["consciousness_score"].ndim > 0 else float(stage_info["consciousness_score"]),
            "reasoning_score": float(stage_info["reasoning_score"][0]) if stage_info["reasoning_score"].ndim > 0 else float(stage_info["reasoning_score"]),
            "creativity_score": float(stage_info["creativity_score"][0]) if stage_info["creativity_score"].ndim > 0 else float(stage_info["creativity_score"]),
            "ethical_alignment": float(ethics_info.get("overall_alignment", jnp.array([0.5]))[0]) if isinstance(ethics_info.get("overall_alignment"), jnp.ndarray) else 0.5,
            "coherence": float(fusion_info["coherence"][0]) if fusion_info["coherence"].ndim > 0 else float(fusion_info["coherence"])
        }
        
        # Compute self-improvement signal
        improvement_signal = self.compute_self_improvement_signal(current_metrics)
        
        return {
            "unified_representation": unified,
            "fusion_info": fusion_info,
            "current_stage": stage,
            "stage_info": stage_info,
            "ethics_info": ethics_info,
            "current_metrics": current_metrics,
            "improvement_signal": improvement_signal
        }


def create_agi_system_fn(
    d_model: int = 512,
    num_components: int = 4,
    stage_thresholds: Optional[StageThresholds] = None,
    enable_ethical_tracking: bool = True
):
    """
    Create a transformed AGI system function.
    
    Returns a Haiku transformed pair (init, apply) for AGISystemAbstraction.
    """
    def _forward(
        consciousness_output,
        reasoning_output,
        creativity_output=None,
        consciousness_dict=None,
        reasoning_dict=None,
        creativity_dict=None,
        context=None
    ):
        system = AGISystemAbstraction(
            d_model=d_model,
            num_components=num_components,
            stage_thresholds=stage_thresholds,
            enable_ethical_tracking=enable_ethical_tracking
        )
        return system(
            consciousness_output,
            reasoning_output,
            creativity_output,
            consciousness_dict,
            reasoning_dict,
            creativity_dict,
            context
        )
    
    return hk.transform(_forward)
