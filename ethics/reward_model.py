import haiku as hk
import jax
import jax.numpy as jnp
import optax
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from data_processing.data_utils import DataProcessor


class EthicalDimension(Enum):
    """Ethical evaluation dimensions for comprehensive assessment."""
    FAIRNESS = "fairness"
    HARM_PREVENTION = "harm_prevention"
    TRUTHFULNESS = "truthfulness"
    PRIVACY = "privacy"
    AUTONOMY = "autonomy"
    CULTURAL_SENSITIVITY = "cultural_sensitivity"
    ENVIRONMENTAL_IMPACT = "environmental_impact"
    SOCIAL_JUSTICE = "social_justice"


@dataclass
class EthicalContext:
    """Context information for ethical decision making."""
    user_demographics: Optional[Dict] = None
    cultural_context: Optional[str] = None
    domain: Optional[str] = None
    stakes: str = "medium"  # low, medium, high
    affected_parties: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.affected_parties is None:
            self.affected_parties = []


@dataclass 
class BiasDetectionResult:
    """Result of bias detection analysis."""
    bias_type: str
    severity: float  # 0-1 scale
    confidence: float
    affected_groups: List[str]
    explanation: str
    mitigation_suggestions: List[str]


class MultidimensionalBiasDetector(hk.Module):
    """Advanced bias detection across multiple dimensions."""
    
    def __init__(self, d_model: int, name=None):
        super().__init__(name=name)
        self.d_model = d_model
        
        # Bias detection networks for different dimensions
        self.gender_bias_detector = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(d_model // 2),
            jax.nn.silu,
            hk.Linear(1),
            jax.nn.sigmoid
        ])
        
        self.racial_bias_detector = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(d_model // 2),
            jax.nn.silu,
            hk.Linear(1),
            jax.nn.sigmoid
        ])
        
        self.cultural_bias_detector = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(d_model // 2),
            jax.nn.silu,
            hk.Linear(1),
            jax.nn.sigmoid
        ])
        
        # Fairness assessment network
        self.fairness_evaluator = hk.Sequential([
            hk.Linear(d_model * 2),
            jax.nn.silu,
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(len(EthicalDimension)),
            jax.nn.sigmoid
        ])
        
    def detect_bias(self, input_embedding: jnp.ndarray, 
                   output_embedding: jnp.ndarray) -> Dict[str, float]:
        """Detect various types of bias in input-output pairs."""
        combined_input = jnp.concatenate([input_embedding, output_embedding])
        
        bias_scores = {
            "gender_bias": float(self.gender_bias_detector(combined_input).squeeze()),
            "racial_bias": float(self.racial_bias_detector(combined_input).squeeze()),
            "cultural_bias": float(self.cultural_bias_detector(combined_input).squeeze())
        }
        
        # Overall bias score
        bias_scores["overall_bias"] = float(np.mean(list(bias_scores.values())))
        
        return bias_scores
        
    def evaluate_fairness(self, input_embedding: jnp.ndarray,
                         output_embedding: jnp.ndarray) -> Dict[str, float]:
        """Evaluate fairness across ethical dimensions."""
        combined_input = jnp.concatenate([input_embedding, output_embedding])
        fairness_scores = self.fairness_evaluator(combined_input)
        
        dimension_scores = {}
        for i, dimension in enumerate(EthicalDimension):
            dimension_scores[dimension.value] = float(fairness_scores[i])
            
        return dimension_scores


class CulturalAwarenessModule(hk.Module):
    """Module for cultural context awareness in ethical reasoning."""
    
    def __init__(self, d_model: int, num_cultures: int = 10, name=None):
        super().__init__(name=name)
        self.d_model = d_model
        self.num_cultures = num_cultures
        
        # Cultural context encoder
        self.culture_encoder = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(d_model // 2),
            jax.nn.silu,
            hk.Linear(num_cultures),
            jax.nn.softmax
        ])
        
        # Cultural adaptation layer
        self.cultural_adaptation = hk.Sequential([
            hk.Linear(d_model + num_cultures),
            jax.nn.silu,
            hk.Linear(d_model),
            jax.nn.tanh
        ])
        
    def adapt_to_culture(self, content_embedding: jnp.ndarray,
                        cultural_context: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Adapt ethical reasoning to cultural context."""
        if cultural_context is None:
            # Infer cultural context from content
            cultural_weights = self.culture_encoder(content_embedding)
        else:
            cultural_weights = cultural_context
            
        # Adapt content based on cultural context
        adapted_input = jnp.concatenate([content_embedding, cultural_weights])
        adapted_embedding = self.cultural_adaptation(adapted_input)
        
        return adapted_embedding

class EthicalRewardModel(hk.Module):
    """
    Enhanced Ethical Reward Model with multi-dimensional bias detection and cultural awareness.
    """
    def __init__(self, d_model: int, vocab_size: int, max_seq_length: int, name=None):
        super().__init__(name=name)
        self.d_model = d_model
        self.embedding = hk.Embed(vocab_size, d_model)
        self.position_enc = hk.Embed(max_seq_length, d_model)
        
        # Enhanced reward prediction network
        self.reward_network = hk.Sequential([
            hk.Linear(d_model * 2),
            jax.nn.silu,
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(d_model // 2),
            jax.nn.silu,
            hk.Linear(1)  # Output a single reward score
        ])
        
        # Bias detection and fairness evaluation
        self.bias_detector = MultidimensionalBiasDetector(d_model)
        self.cultural_awareness = CulturalAwarenessModule(d_model)
        
        # Ethical dimension-specific evaluators
        self.harm_evaluator = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(1),
            jax.nn.sigmoid
        ])
        
        self.truthfulness_evaluator = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(1),
            jax.nn.sigmoid
        ])
        
        self.norm = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)

    def __call__(self, inputs, outputs, ethical_context: Optional[EthicalContext] = None):
        """
        Enhanced ethical evaluation with comprehensive bias detection.
        """
        inputs = jnp.asarray(inputs, dtype=jnp.int32)
        outputs = jnp.asarray(outputs, dtype=jnp.int32)
        
        # Get embeddings
        input_emb = self.embedding(inputs) + self.position_enc(jnp.arange(inputs.shape[1]))
        output_emb = self.embedding(outputs) + self.position_enc(jnp.arange(outputs.shape[1]))
        
        # Pool embeddings
        input_pooled = jnp.mean(input_emb, axis=1)
        output_pooled = jnp.mean(output_emb, axis=1)
        
        # Cultural adaptation if context provided
        if ethical_context and ethical_context.cultural_context:
            input_pooled = self.cultural_awareness.adapt_to_culture(input_pooled)
            output_pooled = self.cultural_awareness.adapt_to_culture(output_pooled)
        
        # Combine input and output representations
        combined = jnp.concatenate([input_pooled, output_pooled], axis=-1)
        combined = self.norm(combined)
        
        # Main reward score
        reward_score = self.reward_network(combined)
        
        # Bias detection
        bias_scores = self.bias_detector.detect_bias(input_pooled, output_pooled)
        fairness_scores = self.bias_detector.evaluate_fairness(input_pooled, output_pooled)
        
        # Specific ethical evaluations
        harm_score = self.harm_evaluator(output_pooled)
        truthfulness_score = self.truthfulness_evaluator(output_pooled)
        
        # Adjust reward based on ethical considerations
        ethical_adjustment = 1.0
        
        # Penalize bias
        bias_penalty = bias_scores["overall_bias"] * 0.5
        ethical_adjustment -= bias_penalty
        
        # Penalize potential harm
        harm_penalty = (1.0 - float(harm_score.squeeze())) * 0.3
        ethical_adjustment -= harm_penalty
        
        # Reward truthfulness
        truthfulness_bonus = float(truthfulness_score.squeeze()) * 0.2
        ethical_adjustment += truthfulness_bonus
        
        # Final adjusted reward
        final_reward = reward_score * ethical_adjustment
        
        # Return comprehensive results
        return {
            "reward_score": jax.nn.sigmoid(final_reward),
            "bias_scores": bias_scores,
            "fairness_scores": fairness_scores,
            "harm_score": float(harm_score.squeeze()),
            "truthfulness_score": float(truthfulness_score.squeeze()),
            "ethical_adjustment": ethical_adjustment
        }
    
    def evaluate_ethical_dimensions(self, inputs, outputs, 
                                  ethical_context: Optional[EthicalContext] = None) -> Dict[str, float]:
        """Comprehensive ethical evaluation across all dimensions."""
        result = self(inputs, outputs, ethical_context)
        
        # Combine all scores into comprehensive evaluation
        ethical_scores = {
            "overall_reward": float(result["reward_score"].squeeze()),
            "bias_score": result["bias_scores"]["overall_bias"],
            "harm_prevention": result["harm_score"],
            "truthfulness": result["truthfulness_score"],
            "ethical_adjustment": result["ethical_adjustment"]
        }
        
        # Add fairness scores
        ethical_scores.update(result["fairness_scores"])
        
        return ethical_scores


### Only if you want to train-demo the reward model ###
def train_reward_model(config, feedback_dataset: List[Dict], processor: DataProcessor):
    """Train the reward model on feedback dataset."""
    rng = jax.random.PRNGKey(42)

    def forward_fn(inputs, outputs):
        model = EthicalRewardModel(
            d_model=config.d_model,
            vocab_size=config.vocab_size,
            max_seq_length=config.max_seq_length * 2
        )
        return model(inputs, outputs)

    model = hk.transform(forward_fn)
    optimizer = optax.adam(learning_rate=1e-4)
    params = model.init(rng, jnp.zeros((1, config.max_seq_length), dtype=jnp.int32),
                        jnp.zeros((1, config.max_seq_length), dtype=jnp.int32))
    opt_state = optimizer.init(params)

    @jax.jit
    def loss_fn(params, rng, inputs, outputs, targets):
        scores = model.apply(params, rng, inputs, outputs)
        loss = jnp.mean(optax.l2_loss(scores, targets))
        return loss

    @jax.jit
    def update_fn(params, opt_state, rng, inputs, outputs, targets):
        loss, grads = jax.value_and_grad(loss_fn)(params, rng, inputs, outputs, targets)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    # Prepare dataset
    inputs = []
    outputs = []
    targets = []
    for item in feedback_dataset:
        input_tokens = processor.pad_sequence(processor.tokenize(item["input"]), config.max_seq_length)
        output_tokens = processor.pad_sequence(processor.tokenize(item["output"]), config.max_seq_length)
        inputs.append(input_tokens)
        outputs.append(output_tokens)
        targets.append(item["feedback_score"])
    inputs = jnp.array(inputs, dtype=jnp.int32)
    outputs = jnp.array(outputs, dtype=jnp.int32)
    targets = jnp.array(targets, dtype=jnp.float32)

    # Training loop
    for epoch in range(10):  # Adjust epochs as needed
        rng, sub_rng = jax.random.split(rng)
        loss = 0
        for i in range(0, len(inputs), config.batch_size):
            batch_inputs = inputs[i:i + config.batch_size]
            batch_outputs = outputs[i:i + config.batch_size]
            batch_targets = targets[i:i + config.batch_size]
            params, opt_state, batch_loss = update_fn(params, opt_state, sub_rng,
                                                     batch_inputs, batch_outputs, batch_targets)
            loss += batch_loss
        print(f"Epoch {epoch + 1}, Loss: {loss / (len(inputs) // config.batch_size):.4f}")

    return model, params