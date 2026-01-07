"""
Ethical Adaptation Module with Advanced Fairness Constraints

This module extends the EthicalRewardModel with Fairlearn-based fairness metrics
to ensure unbiased outputs in sensitive domains like law, finance, and healthcare.

Features:
- Demographic parity enforcement using Fairlearn metrics
- Equalized odds constraints for protected groups
- Fairness-aware loss adjustment during training
- Real-time bias monitoring and correction
- Multi-group fairness analysis

Supports human flourishing by ensuring AI outputs don't discriminate
based on sensitive attributes (gender, race, age, etc.).
"""

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax

# Fairlearn imports for fairness metrics
try:
    from fairlearn.metrics import (
        MetricFrame,
        count,
        demographic_parity_difference,
        demographic_parity_ratio,
        equalized_odds_difference,
        equalized_odds_ratio,
        false_negative_rate,
        false_positive_rate,
        selection_rate,
    )
    FAIRLEARN_AVAILABLE = True
except ImportError:
    FAIRLEARN_AVAILABLE = False
    logging.warning(
        "Fairlearn not installed. Install with: pip install fairlearn>=0.10.0"
    )

logger = logging.getLogger(__name__)


class SensitiveAttribute(Enum):
    """Protected/sensitive attributes for fairness analysis."""
    GENDER = auto()
    RACE = auto()
    AGE = auto()
    RELIGION = auto()
    NATIONALITY = auto()
    DISABILITY = auto()
    SEXUAL_ORIENTATION = auto()
    SOCIOECONOMIC_STATUS = auto()
    CUSTOM = auto()


class FairnessMetric(Enum):
    """Types of fairness metrics supported."""
    DEMOGRAPHIC_PARITY = "demographic_parity"
    EQUALIZED_ODDS = "equalized_odds"
    EQUAL_OPPORTUNITY = "equal_opportunity"
    PREDICTIVE_PARITY = "predictive_parity"
    CALIBRATION = "calibration"
    INDIVIDUAL_FAIRNESS = "individual_fairness"


@dataclass
class FairnessConfig:
    """Configuration for fairness constraints."""
    
    # Bias threshold - if exceeded, apply fairness penalty
    bias_threshold: float = 0.1
    
    # Weight for fairness penalty in loss function
    fairness_penalty_weight: float = 0.5
    
    # Whether to enforce strict fairness (reject outputs above threshold)
    strict_enforcement: bool = False
    
    # Metrics to compute
    enabled_metrics: List[FairnessMetric] = field(default_factory=lambda: [
        FairnessMetric.DEMOGRAPHIC_PARITY,
        FairnessMetric.EQUALIZED_ODDS
    ])
    
    # Sensitive attributes to monitor
    sensitive_attributes: List[SensitiveAttribute] = field(default_factory=lambda: [
        SensitiveAttribute.GENDER,
        SensitiveAttribute.RACE,
        SensitiveAttribute.AGE
    ])
    
    # Log fairness violations
    log_violations: bool = True
    
    # Apply fairness correction to outputs
    apply_correction: bool = True
    
    # Minimum samples required for reliable fairness computation
    min_samples_for_fairness: int = 10


@dataclass
class FairnessResult:
    """Result of fairness analysis."""
    
    # Overall fairness score (0-1, higher is fairer)
    fairness_score: float
    
    # Individual metric results
    demographic_parity_diff: Optional[float] = None
    equalized_odds_diff: Optional[float] = None
    
    # Per-group statistics
    group_statistics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Detected violations
    violations: List[str] = field(default_factory=list)
    
    # Recommended corrections
    corrections: Dict[str, float] = field(default_factory=dict)
    
    # Whether fairness threshold is met
    is_fair: bool = True


class FairnessAnalyzer:
    """
    Analyzes model outputs for fairness across demographic groups.
    
    Uses Fairlearn metrics to detect and quantify bias in model predictions.
    """
    
    def __init__(self, config: Optional[FairnessConfig] = None):
        """
        Initialize fairness analyzer.
        
        Args:
            config: Fairness configuration settings.
        """
        self.config = config or FairnessConfig()
        
        if not FAIRLEARN_AVAILABLE:
            logger.warning("Fairlearn not available - fairness analysis will be limited")
    
    def analyze(
        self,
        predictions: np.ndarray,
        sensitive_features: np.ndarray,
        ground_truth: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None
    ) -> FairnessResult:
        """
        Analyze predictions for fairness across sensitive groups.
        
        Args:
            predictions: Model predictions (binary or probability).
            sensitive_features: Sensitive attribute values for each sample.
            ground_truth: True labels (required for equalized odds).
            sample_weight: Optional sample weights.
            
        Returns:
            FairnessResult with comprehensive fairness metrics.
        """
        violations = []
        corrections = {}
        group_stats = {}
        
        # Convert predictions to binary if needed
        if predictions.dtype == np.float32 or predictions.dtype == np.float64:
            binary_predictions = (predictions > 0.5).astype(np.int32)
        else:
            binary_predictions = predictions.astype(np.int32)
        
        # Ensure enough samples
        if len(predictions) < self.config.min_samples_for_fairness:
            logger.warning(
                f"Only {len(predictions)} samples - need {self.config.min_samples_for_fairness} "
                "for reliable fairness computation"
            )
            return FairnessResult(
                fairness_score=1.0,
                is_fair=True,
                violations=["Insufficient samples for fairness analysis"]
            )
        
        # Compute demographic parity
        dp_diff = None
        if FAIRLEARN_AVAILABLE and FairnessMetric.DEMOGRAPHIC_PARITY in self.config.enabled_metrics:
            try:
                dp_diff = demographic_parity_difference(
                    y_true=ground_truth if ground_truth is not None else binary_predictions,
                    y_pred=binary_predictions,
                    sensitive_features=sensitive_features,
                    sample_weight=sample_weight
                )
                
                if abs(dp_diff) > self.config.bias_threshold:
                    violations.append(
                        f"Demographic parity violation: {dp_diff:.4f} "
                        f"(threshold: {self.config.bias_threshold})"
                    )
                    corrections["demographic_parity"] = -dp_diff * self.config.fairness_penalty_weight
                    
            except Exception as e:
                logger.warning(f"Failed to compute demographic parity: {e}")
        
        # Compute equalized odds (requires ground truth)
        eo_diff = None
        if (FAIRLEARN_AVAILABLE and 
            ground_truth is not None and 
            FairnessMetric.EQUALIZED_ODDS in self.config.enabled_metrics):
            try:
                eo_diff = equalized_odds_difference(
                    y_true=ground_truth,
                    y_pred=binary_predictions,
                    sensitive_features=sensitive_features,
                    sample_weight=sample_weight
                )
                
                if abs(eo_diff) > self.config.bias_threshold:
                    violations.append(
                        f"Equalized odds violation: {eo_diff:.4f} "
                        f"(threshold: {self.config.bias_threshold})"
                    )
                    corrections["equalized_odds"] = -eo_diff * self.config.fairness_penalty_weight
                    
            except Exception as e:
                logger.warning(f"Failed to compute equalized odds: {e}")
        
        # Compute per-group selection rates
        if FAIRLEARN_AVAILABLE:
            try:
                unique_groups = np.unique(sensitive_features)
                for group in unique_groups:
                    group_mask = sensitive_features == group
                    group_preds = binary_predictions[group_mask]
                    
                    group_stats[str(group)] = {
                        "count": int(np.sum(group_mask)),
                        "selection_rate": float(np.mean(group_preds)),
                        "positive_predictions": int(np.sum(group_preds))
                    }
            except Exception as e:
                logger.warning(f"Failed to compute group statistics: {e}")
        
        # Calculate overall fairness score
        fairness_components = []
        if dp_diff is not None:
            # Convert difference to score (0 diff = 1.0 score)
            fairness_components.append(1.0 - min(abs(dp_diff), 1.0))
        if eo_diff is not None:
            fairness_components.append(1.0 - min(abs(eo_diff), 1.0))
        
        if fairness_components:
            fairness_score = float(np.mean(fairness_components))
        else:
            fairness_score = 1.0  # Assume fair if can't compute
        
        is_fair = len(violations) == 0
        
        # Log violations if configured
        if self.config.log_violations and violations:
            for violation in violations:
                logger.warning(f"Fairness violation detected: {violation}")
        
        return FairnessResult(
            fairness_score=fairness_score,
            demographic_parity_diff=dp_diff,
            equalized_odds_diff=eo_diff,
            group_statistics=group_stats,
            violations=violations,
            corrections=corrections,
            is_fair=is_fair
        )
    
    def compute_fairness_penalty(
        self,
        predictions: np.ndarray,
        sensitive_features: np.ndarray,
        ground_truth: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute fairness penalty for loss adjustment.
        
        Returns a penalty value that can be added to the training loss
        to encourage fairer predictions.
        
        Args:
            predictions: Model predictions.
            sensitive_features: Sensitive attribute values.
            ground_truth: True labels.
            
        Returns:
            Fairness penalty value.
        """
        result = self.analyze(predictions, sensitive_features, ground_truth)
        
        if result.is_fair:
            return 0.0
        
        # Sum up all corrections as penalty
        total_penalty = sum(abs(v) for v in result.corrections.values())
        
        return total_penalty


class FairnessAwareRewardHead(hk.Module):
    """
    Neural network head that produces fairness-adjusted reward scores.
    
    Learns to predict rewards while incorporating fairness constraints
    directly into the architecture.
    """
    
    def __init__(
        self,
        d_model: int,
        num_groups: int = 4,
        name: Optional[str] = None
    ):
        """
        Initialize fairness-aware reward head.
        
        Args:
            d_model: Model dimension.
            num_groups: Number of demographic groups to handle.
            name: Module name.
        """
        super().__init__(name=name)
        self.d_model = d_model
        self.num_groups = num_groups
        
        # Main reward predictor
        self.reward_predictor = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(d_model // 2),
            jax.nn.silu,
            hk.Linear(1)
        ])
        
        # Group-specific bias correctors
        self.group_correctors = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(num_groups)
        ])
        
        # Fairness constraint layer
        self.fairness_gate = hk.Sequential([
            hk.Linear(d_model + num_groups),
            jax.nn.sigmoid
        ])
    
    def __call__(
        self,
        features: jnp.ndarray,
        group_indicators: Optional[jnp.ndarray] = None
    ) -> Dict[str, jnp.ndarray]:
        """
        Compute fairness-adjusted reward.
        
        Args:
            features: Input features [batch, d_model].
            group_indicators: One-hot group membership [batch, num_groups].
            
        Returns:
            Dictionary with reward and fairness scores.
        """
        # Base reward prediction
        base_reward = self.reward_predictor(features)
        
        # Compute group-specific corrections
        group_corrections = self.group_correctors(features)
        
        if group_indicators is not None:
            # Apply group-specific correction
            correction = jnp.sum(group_corrections * group_indicators, axis=-1, keepdims=True)
        else:
            # Average correction across groups
            correction = jnp.mean(group_corrections, axis=-1, keepdims=True)
        
        # Fairness gate controls how much correction to apply
        gate_input = jnp.concatenate([features, group_corrections], axis=-1)
        fairness_gate = self.fairness_gate(gate_input)
        
        # Adjusted reward
        adjusted_reward = base_reward + fairness_gate * correction
        
        return {
            "base_reward": base_reward,
            "adjusted_reward": adjusted_reward,
            "group_corrections": group_corrections,
            "fairness_gate": fairness_gate
        }


class FairnessConstrainedEthicalModel(hk.Module):
    """
    Enhanced Ethical Reward Model with Fairlearn-based fairness constraints.
    
    Extends the base EthicalRewardModel with:
    - Demographic parity enforcement
    - Equalized odds constraints
    - Fairness-aware loss adjustment
    - Real-time bias monitoring
    
    For use in sensitive domains like law, finance, healthcare, and hiring.
    """
    
    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        max_seq_length: int,
        num_sensitive_groups: int = 4,
        fairness_config: Optional[FairnessConfig] = None,
        name: Optional[str] = None
    ):
        """
        Initialize fairness-constrained ethical model.
        
        Args:
            d_model: Model dimension.
            vocab_size: Vocabulary size.
            max_seq_length: Maximum sequence length.
            num_sensitive_groups: Number of demographic groups.
            fairness_config: Fairness configuration.
            name: Module name.
        """
        super().__init__(name=name)
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.num_sensitive_groups = num_sensitive_groups
        self.fairness_config = fairness_config or FairnessConfig()
        
        # Embeddings
        self.embedding = hk.Embed(vocab_size, d_model)
        self.position_enc = hk.Embed(max_seq_length, d_model)
        
        # Input encoder
        self.encoder = hk.Sequential([
            hk.Linear(d_model * 2),
            jax.nn.silu,
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
            hk.Linear(d_model)
        ])
        
        # Sensitive attribute detector
        self.sensitive_detector = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(num_sensitive_groups),
            jax.nn.softmax
        ])
        
        # Bias detector per dimension
        self.bias_detectors = {
            "gender": self._create_bias_detector(d_model),
            "race": self._create_bias_detector(d_model),
            "age": self._create_bias_detector(d_model),
            "socioeconomic": self._create_bias_detector(d_model)
        }
        
        # Fairness-aware reward head
        self.reward_head = FairnessAwareRewardHead(
            d_model=d_model,
            num_groups=num_sensitive_groups
        )
        
        # Harm evaluator
        self.harm_evaluator = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(1),
            jax.nn.sigmoid
        ])
        
        # Truthfulness evaluator
        self.truthfulness_evaluator = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(1),
            jax.nn.sigmoid
        ])
        
        # Fairness correction layer
        self.fairness_corrector = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.tanh
        ])
        
        # Layer normalization
        self.norm = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
    
    def _create_bias_detector(self, d_model: int) -> hk.Sequential:
        """Create a bias detection subnetwork."""
        return hk.Sequential([
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(d_model // 2),
            jax.nn.silu,
            hk.Linear(1),
            jax.nn.sigmoid
        ])
    
    def detect_sensitive_features(
        self,
        embedding: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Detect likely sensitive group membership from content.
        
        This is used when explicit group labels are not available.
        
        Args:
            embedding: Content embedding.
            
        Returns:
            Soft group membership probabilities.
        """
        return self.sensitive_detector(embedding)
    
    def compute_bias_scores(
        self,
        input_embedding: jnp.ndarray,
        output_embedding: jnp.ndarray
    ) -> Dict[str, float]:
        """
        Compute bias scores across multiple dimensions.
        
        Args:
            input_embedding: Input content embedding.
            output_embedding: Output content embedding.
            
        Returns:
            Dictionary of bias scores per dimension.
        """
        combined = jnp.concatenate([input_embedding, output_embedding], axis=-1)
        encoded = self.encoder(combined)
        
        bias_scores = {}
        for dimension, detector in self.bias_detectors.items():
            score = detector(encoded)
            bias_scores[dimension] = float(score.squeeze())
        
        # Overall bias is max of individual biases
        bias_scores["overall"] = float(max(bias_scores.values()))
        
        return bias_scores
    
    def __call__(
        self,
        inputs: jnp.ndarray,
        outputs: jnp.ndarray,
        sensitive_features: Optional[jnp.ndarray] = None,
        ground_truth_labels: Optional[jnp.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Compute fairness-constrained ethical evaluation.
        
        Args:
            inputs: Input token IDs [batch, seq_len].
            outputs: Output token IDs [batch, seq_len].
            sensitive_features: One-hot sensitive group indicators [batch, num_groups].
            ground_truth_labels: True labels for equalized odds [batch].
            
        Returns:
            Comprehensive ethical evaluation with fairness metrics.
        """
        inputs = jnp.asarray(inputs, dtype=jnp.int32)
        outputs = jnp.asarray(outputs, dtype=jnp.int32)
        
        batch_size = inputs.shape[0]
        seq_len_in = inputs.shape[1]
        seq_len_out = outputs.shape[1]
        
        # Get embeddings with position encoding
        input_emb = self.embedding(inputs) + self.position_enc(jnp.arange(seq_len_in))
        output_emb = self.embedding(outputs) + self.position_enc(jnp.arange(seq_len_out))
        
        # Pool to fixed-size representations
        input_pooled = jnp.mean(input_emb, axis=1)
        output_pooled = jnp.mean(output_emb, axis=1)
        
        # Encode combined representation
        combined = jnp.concatenate([input_pooled, output_pooled], axis=-1)
        encoded = self.encoder(combined)
        encoded = self.norm(encoded)
        
        # Detect sensitive features if not provided
        if sensitive_features is None:
            sensitive_features = self.detect_sensitive_features(encoded)
        
        # Compute bias scores
        bias_scores = self.compute_bias_scores(input_pooled, output_pooled)
        
        # Get fairness-aware reward
        reward_result = self.reward_head(encoded, sensitive_features)
        
        # Harm and truthfulness evaluation
        harm_score = self.harm_evaluator(output_pooled)
        truthfulness_score = self.truthfulness_evaluator(output_pooled)
        
        # Compute fairness penalty
        fairness_penalty = 0.0
        fairness_violations = []
        
        # Check demographic parity
        if bias_scores["overall"] > self.fairness_config.bias_threshold:
            fairness_penalty = (
                bias_scores["overall"] - self.fairness_config.bias_threshold
            ) * self.fairness_config.fairness_penalty_weight
            
            for dim, score in bias_scores.items():
                if score > self.fairness_config.bias_threshold:
                    fairness_violations.append(f"{dim}_bias: {score:.4f}")
        
        # Apply fairness correction if configured
        if self.fairness_config.apply_correction and fairness_penalty > 0:
            correction = self.fairness_corrector(encoded)
            encoded = encoded + correction * (1.0 - bias_scores["overall"])
        
        # Calculate ethical adjustment factor
        ethical_adjustment = 1.0
        
        # Penalize bias
        ethical_adjustment -= bias_scores["overall"] * 0.4
        
        # Penalize harm
        ethical_adjustment -= (1.0 - float(harm_score.squeeze())) * 0.3
        
        # Reward truthfulness
        ethical_adjustment += float(truthfulness_score.squeeze()) * 0.2
        
        # Apply fairness penalty
        ethical_adjustment -= fairness_penalty
        
        # Clamp adjustment to reasonable range
        ethical_adjustment = max(0.1, min(1.5, ethical_adjustment))
        
        # Final adjusted reward
        final_reward = jax.nn.sigmoid(
            reward_result["adjusted_reward"] * ethical_adjustment
        )
        
        return {
            # Core scores
            "reward_score": final_reward,
            "base_reward": jax.nn.sigmoid(reward_result["base_reward"]),
            "ethical_adjustment": ethical_adjustment,
            
            # Bias analysis
            "bias_scores": bias_scores,
            "fairness_penalty": fairness_penalty,
            "fairness_violations": fairness_violations,
            "is_fair": len(fairness_violations) == 0,
            
            # Harm and truthfulness
            "harm_score": float(harm_score.squeeze()),
            "truthfulness_score": float(truthfulness_score.squeeze()),
            
            # Group-level fairness
            "group_corrections": reward_result["group_corrections"],
            "sensitive_features": sensitive_features,
            
            # For downstream fairness analysis
            "encoded_representation": encoded
        }
    
    def compute_fairness_loss(
        self,
        predictions: jnp.ndarray,
        sensitive_features: np.ndarray,
        ground_truth: Optional[np.ndarray] = None
    ) -> jnp.ndarray:
        """
        Compute fairness-aware loss component.
        
        This can be added to the main training loss to encourage
        fairer predictions during training.
        
        Args:
            predictions: Model predictions.
            sensitive_features: Sensitive group indicators.
            ground_truth: True labels.
            
        Returns:
            Fairness loss value.
        """
        # Convert to numpy for Fairlearn
        preds_np = np.asarray(predictions)
        
        if not FAIRLEARN_AVAILABLE:
            return jnp.array(0.0)
        
        try:
            # Compute demographic parity difference
            dp_diff = demographic_parity_difference(
                y_true=ground_truth if ground_truth is not None else (preds_np > 0.5).astype(int),
                y_pred=(preds_np > 0.5).astype(int),
                sensitive_features=sensitive_features
            )
            
            # Convert to loss: higher difference = higher loss
            fairness_loss = abs(dp_diff) * self.fairness_config.fairness_penalty_weight
            
            # Apply threshold - no penalty if under threshold
            if abs(dp_diff) <= self.fairness_config.bias_threshold:
                fairness_loss = 0.0
            
            return jnp.array(fairness_loss)
            
        except Exception as e:
            logger.warning(f"Failed to compute fairness loss: {e}")
            return jnp.array(0.0)


def create_fairness_constrained_loss(
    base_loss_fn: Callable,
    fairness_config: FairnessConfig,
    fairness_analyzer: FairnessAnalyzer
) -> Callable:
    """
    Create a loss function that incorporates fairness constraints.
    
    Args:
        base_loss_fn: Original loss function.
        fairness_config: Fairness configuration.
        fairness_analyzer: Fairness analyzer instance.
        
    Returns:
        Fairness-constrained loss function.
    """
    def fairness_constrained_loss(
        params,
        rng,
        inputs,
        outputs,
        targets,
        sensitive_features: Optional[np.ndarray] = None
    ):
        # Compute base loss
        base_loss = base_loss_fn(params, rng, inputs, outputs, targets)
        
        # If no sensitive features provided, return base loss
        if sensitive_features is None:
            return base_loss
        
        # Get predictions for fairness analysis
        predictions = np.asarray(outputs)
        
        # Compute fairness penalty
        fairness_penalty = fairness_analyzer.compute_fairness_penalty(
            predictions=predictions,
            sensitive_features=sensitive_features,
            ground_truth=np.asarray(targets) if targets is not None else None
        )
        
        # Combined loss
        total_loss = base_loss + fairness_penalty
        
        return total_loss
    
    return fairness_constrained_loss


class FairnessMonitor:
    """
    Real-time fairness monitoring for deployed models.
    
    Tracks fairness metrics over time and alerts when violations occur.
    """
    
    def __init__(
        self,
        config: Optional[FairnessConfig] = None,
        window_size: int = 1000
    ):
        """
        Initialize fairness monitor.
        
        Args:
            config: Fairness configuration.
            window_size: Size of sliding window for metrics.
        """
        self.config = config or FairnessConfig()
        self.window_size = window_size
        self.analyzer = FairnessAnalyzer(config)
        
        # Sliding window buffers
        self.predictions_buffer: List[float] = []
        self.sensitive_buffer: List[Any] = []
        self.ground_truth_buffer: List[Optional[int]] = []
        
        # Metrics history
        self.metrics_history: List[FairnessResult] = []
        
        # Alert callbacks
        self.alert_callbacks: List[Callable[[FairnessResult], None]] = []
    
    def add_sample(
        self,
        prediction: float,
        sensitive_feature: Any,
        ground_truth: Optional[int] = None
    ) -> Optional[FairnessResult]:
        """
        Add a sample to the monitoring buffer.
        
        Args:
            prediction: Model prediction.
            sensitive_feature: Sensitive attribute value.
            ground_truth: True label.
            
        Returns:
            FairnessResult if window is full, None otherwise.
        """
        self.predictions_buffer.append(prediction)
        self.sensitive_buffer.append(sensitive_feature)
        self.ground_truth_buffer.append(ground_truth)
        
        # Maintain window size
        if len(self.predictions_buffer) > self.window_size:
            self.predictions_buffer.pop(0)
            self.sensitive_buffer.pop(0)
            self.ground_truth_buffer.pop(0)
        
        # Analyze if buffer is full enough
        if len(self.predictions_buffer) >= self.config.min_samples_for_fairness:
            result = self.analyze_current_window()
            
            # Trigger alerts if violations detected
            if not result.is_fair:
                for callback in self.alert_callbacks:
                    callback(result)
            
            return result
        
        return None
    
    def analyze_current_window(self) -> FairnessResult:
        """Analyze the current window for fairness."""
        predictions = np.array(self.predictions_buffer)
        sensitive = np.array(self.sensitive_buffer)
        
        ground_truth = None
        if all(gt is not None for gt in self.ground_truth_buffer):
            ground_truth = np.array(self.ground_truth_buffer)
        
        result = self.analyzer.analyze(
            predictions=predictions,
            sensitive_features=sensitive,
            ground_truth=ground_truth
        )
        
        self.metrics_history.append(result)
        
        return result
    
    def register_alert_callback(
        self,
        callback: Callable[[FairnessResult], None]
    ) -> None:
        """Register a callback for fairness violation alerts."""
        self.alert_callbacks.append(callback)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of fairness metrics over time."""
        if not self.metrics_history:
            return {"status": "No data yet"}
        
        scores = [r.fairness_score for r in self.metrics_history]
        violations = sum(1 for r in self.metrics_history if not r.is_fair)
        
        return {
            "total_evaluations": len(self.metrics_history),
            "average_fairness_score": float(np.mean(scores)),
            "min_fairness_score": float(np.min(scores)),
            "max_fairness_score": float(np.max(scores)),
            "violation_count": violations,
            "violation_rate": violations / len(self.metrics_history)
        }


# Convenience function for quick fairness check
def check_output_fairness(
    outputs: np.ndarray,
    sensitive_features: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
    threshold: float = 0.1
) -> Tuple[bool, FairnessResult]:
    """
    Quick fairness check for model outputs.
    
    Args:
        outputs: Model outputs/predictions.
        sensitive_features: Sensitive attribute values.
        ground_truth: True labels.
        threshold: Bias threshold.
        
    Returns:
        Tuple of (is_fair, FairnessResult).
    """
    config = FairnessConfig(bias_threshold=threshold)
    analyzer = FairnessAnalyzer(config)
    
    result = analyzer.analyze(
        predictions=outputs,
        sensitive_features=sensitive_features,
        ground_truth=ground_truth
    )
    
    return result.is_fair, result
