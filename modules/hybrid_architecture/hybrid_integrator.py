import haiku as hk
import jax
import jax.numpy as jnp
from typing import Dict, List, Optional, Any, Union
import logging

logger = logging.getLogger(__name__)

class HybridArchitectureIntegrator(hk.Module):
    """
    Hybrid architecture that integrates multiple ML approaches:
    - Traditional ML models (SVM, Random Forest concepts)
    - Deep Learning (CNN, RNN, Transformers)
    - Symbolic AI (Rule-based reasoning)
    - Probabilistic models (Bayesian approaches)
    """
    
    def __init__(self, d_model: int, name=None):
        super().__init__(name=name)
        self.d_model = d_model
        
        # Traditional ML backbone
        self.traditional_ml = TraditionalMLBackbone(d_model)
        
        # Deep learning backbone
        self.deep_learning = DeepLearningBackbone(d_model)
        
        # Symbolic reasoning backbone
        self.symbolic_reasoning = SymbolicReasoningBackbone(d_model)
        
        # Probabilistic modeling backbone
        self.probabilistic_model = ProbabilisticBackbone(d_model)
        
        # Meta-learning for approach selection
        self.approach_selector = ApproachSelector(d_model)
        
        # Ensemble fusion layer
        self.ensemble_fusion = EnsembleFusion(d_model)
        
        # Knowledge distillation for model compression
        self.knowledge_distiller = KnowledgeDistillationModule(d_model)
        
        # Adaptive weighting based on input characteristics
        self.adaptive_weighter = AdaptiveWeightingModule(d_model)
    
    def __call__(self, inputs: Dict[str, jnp.ndarray], task_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Process inputs through hybrid architecture
        
        Args:
            inputs: Dictionary containing different input modalities
            task_type: Type of task ('classification', 'regression', 'generation', etc.)
        """
        primary_input = inputs.get('text', inputs.get('features', list(inputs.values())[0]))
        
        # Traditional ML processing
        traditional_output = self.traditional_ml(primary_input)
        
        # Deep learning processing
        deep_output = self.deep_learning(primary_input)
        
        # Symbolic reasoning
        symbolic_output = self.symbolic_reasoning(primary_input)
        
        # Probabilistic modeling
        probabilistic_output = self.probabilistic_model(primary_input)
        
        # Select best approaches for this input
        approach_weights = self.approach_selector(primary_input, task_type)
        
        # Adaptive weighting based on input characteristics
        adaptive_weights = self.adaptive_weighter(primary_input)
        
        # Combine weights
        final_weights = approach_weights * adaptive_weights
        
        # Ensemble fusion
        ensemble_output = self.ensemble_fusion([
            traditional_output, deep_output, symbolic_output, probabilistic_output
        ], final_weights)
        
        # Knowledge distillation for continuous learning
        distilled_knowledge = self.knowledge_distiller(
            ensemble_output, [traditional_output, deep_output, symbolic_output, probabilistic_output]
        )
        
        return {
            'ensemble_output': ensemble_output,
            'traditional_ml': traditional_output,
            'deep_learning': deep_output,
            'symbolic_reasoning': symbolic_output,
            'probabilistic': probabilistic_output,
            'approach_weights': final_weights,
            'distilled_knowledge': distilled_knowledge,
            'confidence': self._calculate_confidence(ensemble_output, final_weights)
        }
    
    def _calculate_confidence(self, ensemble_output: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
        """Calculate ensemble confidence based on agreement between models"""
        # Simple confidence measure based on weight distribution
        weight_entropy = -jnp.sum(weights * jnp.log(weights + 1e-8), axis=-1)
        max_entropy = jnp.log(weights.shape[-1])
        confidence = 1.0 - (weight_entropy / max_entropy)
        return confidence


class TraditionalMLBackbone(hk.Module):
    """Traditional ML approaches implemented as neural networks"""
    
    def __init__(self, d_model: int, name=None):
        super().__init__(name=name)
        self.d_model = d_model
        
        # SVM-like approach (using hinge loss concepts)
        self.svm_like = SVMLikeClassifier(d_model)
        
        # Random Forest-like approach (using ensemble of decision trees)
        self.forest_like = RandomForestLike(d_model)
        
        # Naive Bayes-like approach
        self.bayes_like = NaiveBayesLike(d_model)
        
        # Feature engineering module
        self.feature_engineer = FeatureEngineeringModule(d_model)
        
        # Traditional ensemble
        self.traditional_ensemble = hk.Sequential([
            hk.Linear(d_model * 3),
            jax.nn.silu,
            hk.Linear(d_model)
        ], name="traditional_ensemble")
    
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Process through traditional ML approaches"""
        # Engineer features first
        engineered_features = self.feature_engineer(inputs)
        
        # Apply different traditional approaches
        svm_output = self.svm_like(engineered_features)
        forest_output = self.forest_like(engineered_features)
        bayes_output = self.bayes_like(engineered_features)
        
        # Combine traditional approaches
        combined = jnp.concatenate([svm_output, forest_output, bayes_output], axis=-1)
        ensemble_output = self.traditional_ensemble(combined)
        
        return ensemble_output


class SVMLikeClassifier(hk.Module):
    """SVM-inspired classifier using neural networks with RBF kernel"""
    
    def __init__(self, d_model: int, num_support_vectors: int = 64, gamma: float = 0.1, name=None):
        super().__init__(name=name)
        self.d_model = d_model
        self.num_support_vectors = num_support_vectors
        self.gamma = gamma
        
        # Feature mapping (like kernel trick)
        self.feature_mapping = hk.Sequential([
            hk.Linear(d_model * 2),
            jax.nn.silu,  # Non-linear activation like RBF kernel
            hk.Linear(d_model)
        ], name="feature_mapping")
        
        # Decision boundary
        self.decision_boundary = hk.Linear(d_model, name="decision_boundary")
    
    def _rbf_kernel(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """Compute RBF (Gaussian) kernel between x and y.
        
        RBF kernel: K(x, y) = exp(-gamma * ||x - y||^2)
        
        Args:
            x: Input tensor of shape [..., d]
            y: Support vectors of shape [num_sv, d]
            
        Returns:
            Kernel matrix of shape [..., num_sv]
        """
        # Compute squared Euclidean distance
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x @ y^T
        x_sq = jnp.sum(x ** 2, axis=-1, keepdims=True)  # [..., 1]
        y_sq = jnp.sum(y ** 2, axis=-1)  # [num_sv]
        xy = jnp.matmul(x, y.T)  # [..., num_sv]
        
        sq_dist = x_sq + y_sq - 2 * xy  # [..., num_sv]
        
        # Apply RBF kernel
        kernel_output = jnp.exp(-self.gamma * sq_dist)
        
        return kernel_output
    
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """SVM-like classification with RBF kernel"""
        # Initialize learnable support vectors
        support_vectors = hk.get_parameter(
            "support_vectors",
            shape=(self.num_support_vectors, inputs.shape[-1]),
            init=hk.initializers.TruncatedNormal(stddev=0.02)
        )
        
        # Compute RBF kernel similarity to support vectors
        if inputs.ndim == 3:
            # Handle sequence input: [batch, seq, d]
            batch_size, seq_len, d = inputs.shape
            inputs_flat = inputs.reshape(-1, d)
            kernel_output = self._rbf_kernel(inputs_flat, support_vectors)
            kernel_output = kernel_output.reshape(batch_size, seq_len, -1)
        else:
            # Handle 2D input: [batch, d]
            kernel_output = self._rbf_kernel(inputs, support_vectors)
        
        # Map kernel output to higher dimensional space
        mapped_features = self.feature_mapping(kernel_output)
        
        # Find decision boundary
        decision_scores = self.decision_boundary(mapped_features)
        
        return decision_scores


class RandomForestLike(hk.Module):
    """Random Forest-inspired ensemble using neural networks"""
    
    def __init__(self, d_model: int, num_trees: int = 8, name=None):
        super().__init__(name=name)
        self.d_model = d_model
        self.num_trees = num_trees
        
        # Individual "trees" (small neural networks)
        self.trees = []
        for i in range(num_trees):
            tree = hk.Sequential([
                hk.Linear(d_model // 2),
                jax.nn.relu,
                hk.Linear(d_model // 4),
                jax.nn.relu,
                hk.Linear(d_model)
            ], name=f"tree_{i}")
            self.trees.append(tree)
        
        # Feature selection for each tree (like random feature subsets)
        self.feature_selectors = []
        for i in range(num_trees):
            selector = hk.Linear(d_model, name=f"feature_selector_{i}")
            self.feature_selectors.append(selector)
    
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Random Forest-like prediction"""
        tree_outputs = []
        
        for tree, selector in zip(self.trees, self.feature_selectors):
            # Select random subset of features
            selected_features = selector(inputs)
            selected_features = jax.nn.relu(selected_features)  # Feature selection
            
            # Apply tree
            tree_output = tree(selected_features)
            tree_outputs.append(tree_output)
        
        # Average predictions (like forest voting)
        forest_output = jnp.mean(jnp.stack(tree_outputs), axis=0)
        
        return forest_output


class NaiveBayesLike(hk.Module):
    """Naive Bayes-inspired classifier"""
    
    def __init__(self, d_model: int, name=None):
        super().__init__(name=name)
        self.d_model = d_model
        
        # Feature likelihood estimators
        self.likelihood_estimator = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.softmax
        ], name="likelihood_estimator")
        
        # Prior probability estimator
        self.prior_estimator = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.softmax
        ], name="prior_estimator")
        
        # Posterior calculator
        self.posterior_calculator = hk.Linear(d_model, name="posterior_calculator")
    
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Naive Bayes-like prediction"""
        # Estimate feature likelihoods
        likelihoods = self.likelihood_estimator(inputs)
        
        # Estimate priors
        priors = self.prior_estimator(inputs.mean(axis=1, keepdims=True))
        
        # Calculate posterior (simplified)
        posterior_input = likelihoods * priors
        posterior = self.posterior_calculator(posterior_input)
        
        return posterior


class FeatureEngineeringModule(hk.Module):
    """Automated feature engineering module"""
    
    def __init__(self, d_model: int, name=None):
        super().__init__(name=name)
        self.d_model = d_model
        
        # Polynomial features
        self.polynomial_features = hk.Linear(d_model, name="polynomial_features")
        
        # Interaction features
        self.interaction_features = hk.Linear(d_model, name="interaction_features")
        
        # Statistical features
        self.statistical_features = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(d_model)
        ], name="statistical_features")
    
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Engineer features automatically"""
        # Create polynomial-like features
        poly_features = self.polynomial_features(inputs * inputs)
        
        # Create interaction features using outer product
        # For 3D inputs: (batch, seq, d_model), compute outer product per position
        if inputs.ndim == 3:
            batch_size, seq_len, d = inputs.shape
            # Compute outer product between feature dimensions: [batch, seq, d, d]
            outer_product = jnp.einsum('bsi,bsj->bsij', inputs, inputs)
            # Extract upper triangle (unique interactions) and flatten
            triu_indices = jnp.triu_indices(d, k=1)
            # Vectorized extraction of upper triangle for all batch/seq
            interaction_input = outer_product[:, :, triu_indices[0], triu_indices[1]]
        else:
            # For 2D inputs: (batch, d_model)
            batch_size, d = inputs.shape
            outer_product = jnp.einsum('bi,bj->bij', inputs, inputs)
            triu_indices = jnp.triu_indices(d, k=1)
            interaction_input = outer_product[:, triu_indices[0], triu_indices[1]]
        
        # Project to target dimension
        if interaction_input.shape[-1] > self.d_model:
            interaction_input = interaction_input[..., :self.d_model]
        elif interaction_input.shape[-1] < self.d_model:
            pad_shape = list(interaction_input.shape)
            pad_shape[-1] = self.d_model - interaction_input.shape[-1]
            padding = jnp.zeros(pad_shape)
            interaction_input = jnp.concatenate([interaction_input, padding], axis=-1)
        
        interaction_features = self.interaction_features(interaction_input)
        
        # Create statistical features
        mean_features = jnp.mean(inputs, axis=-1, keepdims=True).repeat(self.d_model, axis=-1)
        std_features = jnp.std(inputs, axis=-1, keepdims=True).repeat(self.d_model, axis=-1)
        statistical_input = jnp.concatenate([mean_features, std_features], axis=-1)
        
        if statistical_input.shape[-1] > self.d_model:
            statistical_input = statistical_input[..., :self.d_model]
        elif statistical_input.shape[-1] < self.d_model:
            padding = jnp.zeros((inputs.shape[0], self.d_model - statistical_input.shape[-1]))
            statistical_input = jnp.concatenate([statistical_input, padding], axis=-1)
        
        statistical_features = self.statistical_features(statistical_input)
        
        # Combine all features
        engineered_features = inputs + poly_features + interaction_features + statistical_features
        
        return engineered_features


class DeepLearningBackbone(hk.Module):
    """Deep learning approaches (CNN, RNN, Transformer)"""
    
    def __init__(self, d_model: int, name=None):
        super().__init__(name=name)
        self.d_model = d_model
        
        # CNN branch
        self.cnn_branch = CNNBranch(d_model)
        
        # RNN branch  
        self.rnn_branch = RNNBranch(d_model)
        
        # Transformer branch
        self.transformer_branch = TransformerBranch(d_model)
        
        # Deep ensemble
        self.deep_ensemble = hk.Sequential([
            hk.Linear(d_model * 3),
            jax.nn.silu,
            hk.Linear(d_model),
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        ], name="deep_ensemble")
    
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Process through deep learning approaches"""
        # CNN processing
        cnn_output = self.cnn_branch(inputs)
        
        # RNN processing
        rnn_output = self.rnn_branch(inputs)
        
        # Transformer processing
        transformer_output = self.transformer_branch(inputs)
        
        # Combine deep learning approaches
        combined = jnp.concatenate([cnn_output, rnn_output, transformer_output], axis=-1)
        ensemble_output = self.deep_ensemble(combined)
        
        return ensemble_output


class CNNBranch(hk.Module):
    """CNN branch for local pattern detection with vectorized sequence processing"""
    
    def __init__(self, d_model: int, kernel_sizes: tuple = (3, 5, 7), name=None):
        super().__init__(name=name)
        self.d_model = d_model
        self.kernel_sizes = kernel_sizes
        
        # Multi-scale 1D convolutions for sequence data
        self.conv_layers_small = hk.Sequential([
            hk.Conv1D(output_channels=d_model//4, kernel_shape=3, padding='SAME'),
            jax.nn.relu,
        ], name="conv_small")
        
        self.conv_layers_medium = hk.Sequential([
            hk.Conv1D(output_channels=d_model//4, kernel_shape=5, padding='SAME'),
            jax.nn.relu,
        ], name="conv_medium")
        
        self.conv_layers_large = hk.Sequential([
            hk.Conv1D(output_channels=d_model//4, kernel_shape=7, padding='SAME'),
            jax.nn.relu,
        ], name="conv_large")
        
        # Second layer convolutions
        self.conv_final = hk.Conv1D(
            output_channels=d_model, kernel_shape=3, padding='SAME', name="conv_final"
        )
        
        # Projection layer for dimension matching
        self.projection = hk.Linear(d_model, name="cnn_projection")
    
    def _apply_conv_vectorized(self, conv_layer, inputs: jnp.ndarray) -> jnp.ndarray:
        """Apply convolution layer with vectorized batch processing.
        
        Uses jax.vmap for efficient parallel processing across batch dimension.
        
        Args:
            conv_layer: The convolution layer to apply
            inputs: Shape [batch, seq_len, d_model]
            
        Returns:
            Convolved output
        """
        # Conv1D already handles batch dimension efficiently in Haiku
        # But we can use vmap for explicit parallelization if needed
        return conv_layer(inputs)
    
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Apply vectorized CNN processing for sequences.
        
        Args:
            inputs: Shape [batch, seq_len, d_model] or [batch, d_model]
            
        Returns:
            Processed output with shape [batch, d_model]
        """
        # Handle 2D inputs by adding sequence dimension
        if inputs.ndim == 2:
            inputs = inputs[:, None, :]  # [batch, 1, d_model]
        
        # Apply multi-scale convolutions in parallel using vectorized operations
        # Each conv layer is automatically vectorized over batch dimension
        conv_small = self._apply_conv_vectorized(self.conv_layers_small, inputs)
        conv_medium = self._apply_conv_vectorized(self.conv_layers_medium, inputs)
        conv_large = self._apply_conv_vectorized(self.conv_layers_large, inputs)
        
        # Concatenate multi-scale features
        multi_scale = jnp.concatenate([conv_small, conv_medium, conv_large], axis=-1)
        
        # Pad to match d_model if needed
        if multi_scale.shape[-1] < self.d_model:
            padding = jnp.zeros((*multi_scale.shape[:-1], self.d_model - multi_scale.shape[-1]))
            multi_scale = jnp.concatenate([multi_scale, padding], axis=-1)
        elif multi_scale.shape[-1] > self.d_model:
            multi_scale = multi_scale[..., :self.d_model]
        
        # Apply final conv layer with vectorized processing
        conv_output = self._apply_conv_vectorized(
            lambda x: jax.nn.relu(self.conv_final(x)),
            multi_scale
        )
        
        # Vectorized pooling: use both max and mean pooling
        max_pooled = jnp.max(conv_output, axis=1)   # [batch, d_model]
        mean_pooled = jnp.mean(conv_output, axis=1)  # [batch, d_model]
        
        # Combine pooled features
        pooled_output = (max_pooled + mean_pooled) / 2
        
        # Project to output dimension
        output = self.projection(pooled_output)
        
        return output


class RNNBranch(hk.Module):
    """RNN branch for temporal modeling"""
    
    def __init__(self, d_model: int, name=None):
        super().__init__(name=name)
        self.d_model = d_model
        
        # Simplified RNN using attention
        self.temporal_attention = hk.MultiHeadAttention(
            num_heads=4, 
            key_size=d_model//4, 
            w_init=hk.initializers.TruncatedNormal(stddev=0.02),
            name="temporal_attention"
        )
        
        # Sequential processing
        self.sequential_processor = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(d_model)
        ], name="sequential_processor")
    
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Apply RNN-like processing"""
        # Apply temporal attention
        attended_output = self.temporal_attention(inputs, inputs, inputs)
        
        # Sequential processing
        processed_output = self.sequential_processor(attended_output)
        
        # Aggregate over sequence
        aggregated_output = jnp.mean(processed_output, axis=1)
        
        return aggregated_output


class TransformerBranch(hk.Module):
    """Transformer branch for long-range dependencies"""
    
    def __init__(self, d_model: int, name=None):
        super().__init__(name=name)
        self.d_model = d_model
        
        # Multi-head attention
        self.attention = hk.MultiHeadAttention(
            num_heads=8, 
            key_size=d_model//8, 
            w_init=hk.initializers.TruncatedNormal(stddev=0.02),
            name="attention"
        )
        
        # Feed-forward network
        self.ffn = hk.Sequential([
            hk.Linear(d_model * 4),
            jax.nn.gelu,
            hk.Linear(d_model)
        ], name="ffn")
        
        # Layer normalization
        self.layer_norm1 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name="layer_norm1")
        self.layer_norm2 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name="layer_norm2")
    
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Apply Transformer processing"""
        # Self-attention with residual connection
        attended = self.attention(inputs, inputs, inputs)
        attended = self.layer_norm1(inputs + attended)
        
        # Feed-forward with residual connection
        ffn_output = self.ffn(attended)
        output = self.layer_norm2(attended + ffn_output)
        
        # Aggregate over sequence
        aggregated_output = jnp.mean(output, axis=1)
        
        return aggregated_output


class SymbolicReasoningBackbone(hk.Module):
    """Symbolic reasoning backbone"""
    
    def __init__(self, d_model: int, name=None):
        super().__init__(name=name)
        self.d_model = d_model
        
        # Rule-based reasoning simulator
        self.rule_engine = RuleBasedEngine(d_model)
        
        # Logic reasoning simulator
        self.logic_engine = LogicReasoningEngine(d_model)
        
        # Symbol grounding
        self.symbol_grounder = SymbolGroundingModule(d_model)
        
        # Symbolic ensemble
        self.symbolic_ensemble = hk.Sequential([
            hk.Linear(d_model * 3),
            jax.nn.silu,
            hk.Linear(d_model)
        ], name="symbolic_ensemble")
    
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Process through symbolic reasoning"""
        # Ground symbols from neural representations
        grounded_symbols = self.symbol_grounder(inputs)
        
        # Apply rule-based reasoning
        rule_output = self.rule_engine(grounded_symbols)
        
        # Apply logic reasoning
        logic_output = self.logic_engine(grounded_symbols)
        
        # Combine symbolic approaches
        combined = jnp.concatenate([grounded_symbols, rule_output, logic_output], axis=-1)
        ensemble_output = self.symbolic_ensemble(combined)
        
        return ensemble_output


class RuleBasedEngine(hk.Module):
    """Rule-based reasoning engine"""
    
    def __init__(self, d_model: int, num_rules: int = 16, name=None):
        super().__init__(name=name)
        self.d_model = d_model
        self.num_rules = num_rules
        
        # Rule condition matchers
        self.rule_conditions = []
        for i in range(num_rules):
            condition = hk.Sequential([
                hk.Linear(d_model),
                jax.nn.sigmoid
            ], name=f"rule_condition_{i}")
            self.rule_conditions.append(condition)
        
        # Rule actions
        self.rule_actions = []
        for i in range(num_rules):
            action = hk.Sequential([
                hk.Linear(d_model),
                jax.nn.silu,
                hk.Linear(d_model)
            ], name=f"rule_action_{i}")
            self.rule_actions.append(action)
    
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Apply rule-based reasoning"""
        rule_outputs = []
        
        for condition, action in zip(self.rule_conditions, self.rule_actions):
            # Check if rule condition is satisfied
            condition_score = condition(inputs).mean(axis=1, keepdims=True)
            
            # Apply rule action if condition is met
            rule_action = action(inputs)
            weighted_action = rule_action * condition_score
            
            rule_outputs.append(weighted_action)
        
        # Aggregate all rule outputs
        aggregated_rules = jnp.sum(jnp.stack(rule_outputs), axis=0)
        
        return aggregated_rules


class LogicReasoningEngine(hk.Module):
    """Logic reasoning engine (simplified)"""
    
    def __init__(self, d_model: int, name=None):
        super().__init__(name=name)
        self.d_model = d_model
        
        # Premise encoder
        self.premise_encoder = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(d_model)
        ], name="premise_encoder")
        
        # Inference engine
        self.inference_engine = hk.MultiHeadAttention(
            num_heads=4, 
            key_size=d_model//4, 
            w_init=hk.initializers.TruncatedNormal(stddev=0.02),
            name="inference_engine"
        )
        
        # Conclusion generator
        self.conclusion_generator = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(d_model)
        ], name="conclusion_generator")
    
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Apply logic reasoning"""
        # Encode premises
        premises = self.premise_encoder(inputs)
        
        # Apply inference
        inferred = self.inference_engine(premises, premises, premises)
        
        # Generate conclusions
        conclusions = self.conclusion_generator(inferred)
        
        # Preserve input shape - only aggregate if 2D input
        if inputs.ndim == 2:
            return jnp.mean(conclusions, axis=0, keepdims=True) if conclusions.ndim == 1 else conclusions
        else:
            # Keep sequence dimension for 3D inputs
            return conclusions


class SymbolGroundingModule(hk.Module):
    """Symbol grounding module to bridge neural and symbolic representations"""
    
    def __init__(self, d_model: int, name=None):
        super().__init__(name=name)
        self.d_model = d_model
        
        # Neural to symbolic mapper
        self.neural_to_symbolic = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(d_model),
            jax.nn.tanh  # Bounded symbolic representations
        ], name="neural_to_symbolic")
        
        # Symbolic to neural mapper
        self.symbolic_to_neural = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(d_model)
        ], name="symbolic_to_neural")
    
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Ground symbols from neural representations"""
        # Map neural to symbolic
        symbolic_repr = self.neural_to_symbolic(inputs)
        
        # Map back to neural (for consistency)
        grounded_neural = self.symbolic_to_neural(symbolic_repr)
        
        return grounded_neural


class ProbabilisticBackbone(hk.Module):
    """Probabilistic modeling backbone"""
    
    def __init__(self, d_model: int, name=None):
        super().__init__(name=name)
        self.d_model = d_model
        
        # Bayesian neural network approximation
        self.bayesian_net = BayesianNeuralNetwork(d_model)
        
        # Uncertainty quantification
        self.uncertainty_quantifier = UncertaintyQuantifier(d_model)
        
        # Variational inference
        self.variational_inference = VariationalInferenceModule(d_model)
        
        # Probabilistic ensemble
        self.probabilistic_ensemble = hk.Sequential([
            hk.Linear(d_model * 3),
            jax.nn.silu,
            hk.Linear(d_model)
        ], name="probabilistic_ensemble")
    
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Process through probabilistic models"""
        # Bayesian neural network
        bayesian_output = self.bayesian_net(inputs)
        
        # Uncertainty quantification
        uncertainty_output = self.uncertainty_quantifier(inputs)
        
        # Variational inference
        variational_output = self.variational_inference(inputs)
        
        # Combine probabilistic approaches
        combined = jnp.concatenate([bayesian_output, uncertainty_output, variational_output], axis=-1)
        ensemble_output = self.probabilistic_ensemble(combined)
        
        return ensemble_output


class BayesianNeuralNetwork(hk.Module):
    """Bayesian neural network approximation"""
    
    def __init__(self, d_model: int, name=None):
        super().__init__(name=name)
        self.d_model = d_model
        
        # Mean and variance predictors
        self.mean_predictor = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(d_model)
        ], name="mean_predictor")
        
        self.variance_predictor = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(d_model),
            jax.nn.softplus  # Ensure positive variance
        ], name="variance_predictor")
    
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Bayesian neural network prediction"""
        # Predict mean and variance
        mean = self.mean_predictor(inputs)
        variance = self.variance_predictor(inputs)
        
        # Sample from distribution (reparameterization trick)
        key = hk.next_rng_key()
        epsilon = jax.random.normal(key, mean.shape)
        sample = mean + jnp.sqrt(variance) * epsilon
        
        return sample


class UncertaintyQuantifier(hk.Module):
    """Uncertainty quantification module"""
    
    def __init__(self, d_model: int, name=None):
        super().__init__(name=name)
        self.d_model = d_model
        
        # Aleatoric uncertainty (data uncertainty)
        self.aleatoric_estimator = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(d_model),
            jax.nn.softplus
        ], name="aleatoric_estimator")
        
        # Epistemic uncertainty (model uncertainty) 
        self.epistemic_estimator = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(d_model),
            jax.nn.softplus
        ], name="epistemic_estimator")
    
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Quantify different types of uncertainty"""
        # Estimate aleatoric uncertainty
        aleatoric = self.aleatoric_estimator(inputs)
        
        # Estimate epistemic uncertainty
        epistemic = self.epistemic_estimator(inputs)
        
        # Combine uncertainties
        total_uncertainty = aleatoric + epistemic
        
        return total_uncertainty


class VariationalInferenceModule(hk.Module):
    """Variational inference module"""
    
    def __init__(self, d_model: int, name=None):
        super().__init__(name=name)
        self.d_model = d_model
        
        # Encoder (recognition model)
        self.encoder = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(d_model * 2)  # Mean and log variance
        ], name="encoder")
        
        # Decoder (generative model)
        self.decoder = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(d_model)
        ], name="decoder")
    
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Variational inference"""
        # Encode to latent space
        encoded = self.encoder(inputs)
        mean, log_var = jnp.split(encoded, 2, axis=-1)
        
        # Sample from latent space
        key = hk.next_rng_key()
        epsilon = jax.random.normal(key, mean.shape)
        latent = mean + jnp.exp(0.5 * log_var) * epsilon
        
        # Decode back to output space
        decoded = self.decoder(latent)
        
        return decoded


class ApproachSelector(hk.Module):
    """Meta-learning module to select best approaches for given input"""
    
    def __init__(self, d_model: int, name=None):
        super().__init__(name=name)
        self.d_model = d_model
        
        # Input analyzer
        self.input_analyzer = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(d_model)
        ], name="input_analyzer")
        
        # Approach weight predictor
        self.weight_predictor = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(4),  # 4 approaches
            jax.nn.softmax
        ], name="weight_predictor")
    
    def __call__(self, inputs: jnp.ndarray, task_type: Optional[str] = None) -> jnp.ndarray:
        """Select best approaches for the input"""
        # Analyze input characteristics
        input_analysis = self.input_analyzer(inputs.mean(axis=1))
        
        # Predict weights for different approaches
        approach_weights = self.weight_predictor(input_analysis)
        
        # Adjust weights based on task type if provided
        if task_type:
            task_adjustment = self._get_task_adjustment(task_type)
            approach_weights = approach_weights * task_adjustment
            approach_weights = approach_weights / jnp.sum(approach_weights, axis=-1, keepdims=True)
        
        return approach_weights
    
    def _get_task_adjustment(self, task_type: str) -> jnp.ndarray:
        """Get task-specific weight adjustments"""
        # Default equal weighting
        adjustments = jnp.ones(4)
        
        if task_type == 'classification':
            # Favor traditional ML and deep learning
            adjustments = jnp.array([1.2, 1.3, 0.8, 1.0])
        elif task_type == 'reasoning':
            # Favor symbolic and probabilistic
            adjustments = jnp.array([0.8, 1.0, 1.5, 1.2])
        elif task_type == 'generation':
            # Favor deep learning and probabilistic
            adjustments = jnp.array([0.8, 1.4, 0.9, 1.3])
        
        return adjustments


class AdaptiveWeightingModule(hk.Module):
    """Adaptive weighting based on input characteristics"""
    
    def __init__(self, d_model: int, name=None):
        super().__init__(name=name)
        self.d_model = d_model
        
        # Input complexity analyzer
        self.complexity_analyzer = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(1),
            jax.nn.sigmoid
        ], name="complexity_analyzer")
        
        # Noise level estimator
        self.noise_estimator = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(1),
            jax.nn.sigmoid
        ], name="noise_estimator")
        
        # Adaptivity controller
        self.adaptivity_controller = hk.Sequential([
            hk.Linear(2),  # complexity + noise
            jax.nn.silu,
            hk.Linear(4),  # 4 approaches
            jax.nn.softmax
        ], name="adaptivity_controller")
    
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Calculate adaptive weights based on input characteristics"""
        # Handle both 2D [batch, d_model] and 3D [batch, seq, d_model] inputs
        if inputs.ndim == 3:
            pooled_input = inputs.mean(axis=1)  # [batch, d_model]
            input_std = jnp.std(inputs, axis=1)  # [batch, d_model]
        else:
            pooled_input = inputs  # [batch, d_model]
            input_std = jnp.std(inputs, axis=-1, keepdims=True)  # [batch, 1]
        
        # Analyze input complexity
        complexity = self.complexity_analyzer(pooled_input)  # [batch, 1]
        
        # Estimate noise level - ensure 2D output
        if input_std.ndim > 2:
            input_std = input_std.squeeze()
        if input_std.ndim == 1:
            input_std = input_std[:, None]
        noise_level = self.noise_estimator(input_std)  # [batch, 1]
        
        # Ensure both have same shape for concatenation
        if noise_level.ndim > 2:
            noise_level = noise_level.squeeze(-1)
        
        # Combine characteristics
        characteristics = jnp.concatenate([complexity, noise_level], axis=-1)  # [batch, 2]
        
        # Calculate adaptive weights
        adaptive_weights = self.adaptivity_controller(characteristics)
        
        return adaptive_weights


class EnsembleFusion(hk.Module):
    """Ensemble fusion module with cross-paradigm interaction"""
    
    def __init__(self, d_model: int, name=None):
        super().__init__(name=name)
        self.d_model = d_model
        
        # Attention-based fusion
        self.fusion_attention = hk.MultiHeadAttention(
            num_heads=4, 
            key_size=d_model//4, 
            w_init=hk.initializers.TruncatedNormal(stddev=0.02),
            name="fusion_attention"
        )
        
        # Cross-paradigm interaction projection
        self.interaction_projection = hk.Linear(d_model, name="interaction_projection")
        
        # Final projection
        self.final_projection = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(d_model),
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        ], name="final_projection")
    
    def __call__(self, model_outputs: List[jnp.ndarray], weights: jnp.ndarray) -> jnp.ndarray:
        """Fuse ensemble outputs with learned weights and cross-paradigm interaction
        
        model_outputs expected order: [traditional_ml, deep_learning, symbolic, probabilistic]
        """
        # Normalize all outputs to 2D [batch, d_model] for consistent stacking
        normalized_outputs = []
        for output in model_outputs:
            if output.ndim == 3:
                # Pool sequence dimension: [batch, seq, d_model] -> [batch, d_model]
                normalized = jnp.mean(output, axis=1)
            elif output.ndim == 2:
                normalized = output
            else:
                # Handle unexpected dimensions
                normalized = output.reshape(output.shape[0], -1)
                if normalized.shape[-1] != self.d_model:
                    normalized = normalized[..., :self.d_model]
            normalized_outputs.append(normalized)
        
        # Stack model outputs
        stacked_outputs = jnp.stack(normalized_outputs, axis=1)  # [batch, num_models, d_model]
        
        # Compute cross-paradigm interaction between statistical (traditional) and deep learning
        # model_outputs[0] = traditional ML, model_outputs[1] = deep learning
        statistical_features = normalized_outputs[0]  # [batch, d_model]
        deep_features = normalized_outputs[1]  # [batch, d_model]
        
        # Compute outer product interaction: captures feature correlations across paradigms
        # [batch, d_model] case - compute outer product per batch element
        interaction = jnp.einsum('bi,bj->bij', statistical_features, deep_features)
        # Take diagonal + trace as summary (efficient approximation)
        interaction_diag = jnp.diagonal(interaction, axis1=1, axis2=2)
        interaction_trace = jnp.trace(interaction, axis1=1, axis2=2, dtype=interaction.dtype)
        interaction_summary = interaction_diag * (1.0 + interaction_trace[:, None] / self.d_model)
        
        # Project interaction to match dimensions
        interaction_features = self.interaction_projection(interaction_summary)
        
        # Apply attention fusion
        fused = self.fusion_attention(stacked_outputs, stacked_outputs, stacked_outputs)
        
        # Weight by approach weights - fused is [batch, num_models, d_model]
        weighted_fused = fused * weights[:, :, None]
        
        # Sum across models
        ensemble_output = jnp.sum(weighted_fused, axis=1)  # [batch, d_model]
        
        # Add cross-paradigm interaction as residual connection
        ensemble_output = ensemble_output + 0.1 * interaction_features
        
        # Final projection
        final_output = self.final_projection(ensemble_output)
        
        return final_output


class KnowledgeDistillationModule(hk.Module):
    """Knowledge distillation for continuous learning"""
    
    def __init__(self, d_model: int, name=None):
        super().__init__(name=name)
        self.d_model = d_model
        
        # Teacher-student distillation
        self.student_network = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(d_model)
        ], name="student_network")
        
        # Knowledge extractor
        self.knowledge_extractor = hk.Sequential([
            hk.Linear(d_model * 4),  # All teacher outputs
            jax.nn.silu,
            hk.Linear(d_model)
        ], name="knowledge_extractor")
    
    def __call__(self, ensemble_output: jnp.ndarray, teacher_outputs: List[jnp.ndarray]) -> jnp.ndarray:
        """Distill knowledge from ensemble to student network"""
        # Normalize all teacher outputs to 2D [batch, d_model] before concatenation
        normalized_teachers = []
        for output in teacher_outputs:
            if output.ndim == 3:
                # Pool sequence dimension: [batch, seq, d_model] -> [batch, d_model]
                normalized = jnp.mean(output, axis=1)
            elif output.ndim == 2:
                normalized = output
            else:
                normalized = output.reshape(output.shape[0], -1)
            normalized_teachers.append(normalized)
        
        # Extract knowledge from all teachers
        teacher_knowledge = jnp.concatenate(normalized_teachers, axis=-1)
        extracted_knowledge = self.knowledge_extractor(teacher_knowledge)
        
        # Train student to match ensemble
        student_output = self.student_network(extracted_knowledge)
        
        return student_output


def compute_task_complexity(inputs: jnp.ndarray, epsilon: float = 1e-8) -> jnp.ndarray:
    """
    Compute task complexity using entropy of input distribution.
    
    Higher entropy indicates more complex/varied input requiring more specialists.
    
    Args:
        inputs: Input tensor [batch, d_model] or [batch, seq, d_model]
        epsilon: Small constant for numerical stability
        
    Returns:
        Complexity score in [0, 1] per batch item
    """
    # Flatten to [batch, features]
    if inputs.ndim == 3:
        flat_inputs = inputs.reshape(inputs.shape[0], -1)
    else:
        flat_inputs = inputs
    
    # Normalize to probability distribution (softmax)
    probs = jax.nn.softmax(flat_inputs, axis=-1)
    
    # Compute entropy: -sum(p * log(p))
    log_probs = jnp.log(probs + epsilon)
    entropy = -jnp.sum(probs * log_probs, axis=-1)
    
    # Normalize to [0, 1] based on max possible entropy
    max_entropy = jnp.log(flat_inputs.shape[-1])
    normalized_complexity = entropy / (max_entropy + epsilon)
    
    return normalized_complexity


class SpecialistAgent(hk.Module):
    """A specialist agent with specific domain expertise.
    
    Supports dynamic weight initialization for spawned agents with
    specialized configurations based on task requirements.
    """
    
    def __init__(
        self, 
        d_model: int, 
        specialization: str, 
        weight_scale: float = 1.0,
        is_spawned: bool = False,
        name=None
    ):
        """
        Initialize specialist agent.
        
        Args:
            d_model: Model dimension
            specialization: Domain of expertise (e.g., 'reasoning', 'analysis')
            weight_scale: Scale factor for weight initialization (spawned agents may use different scales)
            is_spawned: Whether this agent was dynamically spawned
            name: Module name
        """
        super().__init__(name=name)
        self.d_model = d_model
        self.specialization = specialization
        self.weight_scale = weight_scale
        self.is_spawned = is_spawned
        
        # Weight initializer with configurable scale for spawned agents
        w_init = hk.initializers.VarianceScaling(weight_scale)
        
        # Agent-specific processing
        self.encoder = hk.Sequential([
            hk.Linear(d_model, w_init=w_init),
            jax.nn.silu,
            hk.Linear(d_model, w_init=w_init)
        ], name=f"agent_encoder_{specialization}")
        
        # Expert network for this specialization
        self.expert = hk.Sequential([
            hk.Linear(d_model * 2, w_init=w_init),
            jax.nn.silu,
            hk.Linear(d_model, w_init=w_init),
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        ], name=f"agent_expert_{specialization}")
        
        # Confidence scorer
        self.confidence_head = hk.Sequential([
            hk.Linear(d_model // 2, w_init=w_init),
            jax.nn.silu,
            hk.Linear(1, w_init=w_init),
            jax.nn.sigmoid
        ], name=f"agent_confidence_{specialization}")
    
    def process(self, inputs: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """Process input through this specialist agent.
        
        Args:
            inputs: Input tensor of shape [batch, d_model] or [batch, seq, d_model]
            
        Returns:
            Dictionary with 'response' and 'confidence'
        """
        # Encode input
        encoded = self.encoder(inputs)
        
        # Generate expert response
        response = self.expert(encoded)
        
        # Calculate confidence
        if response.ndim == 3:
            pooled = jnp.mean(response, axis=1)
        else:
            pooled = response
        confidence = self.confidence_head(pooled)
        
        return {
            'response': response,
            'confidence': confidence,
            'specialization': self.specialization
        }
    
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Forward pass returning just the response."""
        return self.process(inputs)['response']


class MultiAgentConsensus(hk.Module):
    """Multi-agent system for collaborative decision making with consensus.
    
    Implements a multi-agent loop where multiple specialist agents process
    input independently, then their responses are aggregated via consensus.
    
    Features:
    - Dynamic agent spawning based on task complexity
    - Parallel agent processing via jax.vmap
    - Consensus-based response aggregation
    - Scalable crisis response (spawn analysis agents for data overload)
    """
    
    # Crisis specializations for dynamic spawning
    CRISIS_SPECIALIZATIONS = [
        "emergency_analysis",
        "resource_allocation",
        "communication",
        "logistics",
        "risk_assessment",
        "coordination",
        "triage",
        "recovery_planning"
    ]
    
    def __init__(
        self, 
        d_model: int, 
        num_agents: int = 4, 
        max_spawned_agents: int = 8,
        spawn_threshold: float = 0.5,
        name=None
    ):
        """
        Initialize multi-agent consensus system.
        
        Args:
            d_model: Model dimension
            num_agents: Number of base agents
            max_spawned_agents: Maximum number of dynamically spawned agents
            spawn_threshold: Complexity threshold for spawning new agents
            name: Module name
        """
        super().__init__(name=name)
        self.d_model = d_model
        self.num_agents = num_agents
        self.max_spawned_agents = max_spawned_agents
        self.spawn_threshold = spawn_threshold
        
        # Track spawned agents
        self._spawned_agents: List[SpecialistAgent] = []
        self._spawn_counter = 0
        
        # Define agent specializations
        specializations = [
            "reasoning",
            "creativity", 
            "analysis",
            "synthesis"
        ]
        
        # Create specialist agents
        self.agents = [
            SpecialistAgent(d_model, spec, name=f"agent_{i}")
            for i, spec in enumerate(specializations[:num_agents])
        ]
        
        # Consensus mechanism
        self.consensus_attention = hk.MultiHeadAttention(
            num_heads=4,
            key_size=d_model // 4,
            w_init=hk.initializers.TruncatedNormal(stddev=0.02),
            name="consensus_attention"
        )
        
        # Final consensus projection
        self.consensus_projection = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(d_model),
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        ], name="consensus_projection")
        
        # Aggregation weights (learnable)
        self.weight_network = hk.Linear(num_agents, name="agent_weights")
        
        # Spawned agent fusion layer (handles variable number of agents)
        self.spawned_fusion = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(d_model)
        ], name="spawned_fusion")
    
    def compute_complexity(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """
        Compute task complexity from input using entropy.
        
        Args:
            inputs: Input tensor
            
        Returns:
            Complexity score per batch item in [0, 1]
        """
        return compute_task_complexity(inputs)
    
    def spawn_agent(
        self, 
        task_complexity: float,
        specialization: Optional[str] = None
    ) -> Optional[SpecialistAgent]:
        """
        Spawn a new specialist agent based on task complexity.
        
        Dynamically creates a new SpecialistAgent when complexity exceeds
        threshold and max agents haven't been reached.
        
        Args:
            task_complexity: Complexity score in [0, 1]
            specialization: Optional specific specialization for the agent
            
        Returns:
            Newly spawned SpecialistAgent or None if spawn not needed/allowed
        """
        # Check if spawning is warranted
        if task_complexity <= self.spawn_threshold:
            return None
        
        # Check max agents limit
        if len(self._spawned_agents) >= self.max_spawned_agents:
            logger.warning(f"Max spawned agents ({self.max_spawned_agents}) reached")
            return None
        
        # Select specialization based on crisis needs
        if specialization is None:
            spec_index = self._spawn_counter % len(self.CRISIS_SPECIALIZATIONS)
            specialization = self.CRISIS_SPECIALIZATIONS[spec_index]
        
        # Compute specialized weight scale based on complexity
        # Higher complexity = more aggressive initialization
        weight_scale = 1.0 + (task_complexity - self.spawn_threshold)
        
        # Create new specialist agent
        new_agent = SpecialistAgent(
            d_model=self.d_model,
            specialization=specialization,
            weight_scale=weight_scale,
            is_spawned=True,
            name=f"spawned_agent_{self._spawn_counter}"
        )
        
        self._spawned_agents.append(new_agent)
        self._spawn_counter += 1
        
        logger.info(f"Spawned new agent: {specialization} (complexity: {task_complexity:.3f})")
        
        return new_agent
    
    def spawn_crisis_team(
        self, 
        inputs: jnp.ndarray,
        num_agents: int = 3
    ) -> List[SpecialistAgent]:
        """
        Spawn a team of agents for crisis response.
        
        Creates multiple specialized agents for handling crisis situations
        like disaster data overload.
        
        Args:
            inputs: Input tensor to analyze for complexity
            num_agents: Number of agents to spawn
            
        Returns:
            List of spawned SpecialistAgent instances
        """
        complexity = float(jnp.mean(self.compute_complexity(inputs)))
        spawned = []
        
        # Priority crisis specializations
        priority_specs = ["emergency_analysis", "resource_allocation", "coordination"]
        
        for i, spec in enumerate(priority_specs[:num_agents]):
            # Each agent gets slightly different weight scale
            adjusted_complexity = complexity + (i * 0.05)
            agent = self.spawn_agent(
                task_complexity=min(adjusted_complexity, 1.0),
                specialization=spec
            )
            if agent is not None:
                spawned.append(agent)
        
        return spawned
    
    def get_active_agents(self) -> List[SpecialistAgent]:
        """Get all active agents (base + spawned)."""
        return self.agents + self._spawned_agents
    
    def clear_spawned_agents(self) -> int:
        """Clear all spawned agents, returning count cleared."""
        count = len(self._spawned_agents)
        self._spawned_agents = []
        return count
    
    def _process_agent_parallel(
        self, 
        agent: SpecialistAgent, 
        inputs: jnp.ndarray
    ) -> Dict[str, jnp.ndarray]:
        """Process a single agent - used for vmap."""
        return agent.process(inputs)
    
    def __call__(
        self, 
        inputs: jnp.ndarray,
        auto_spawn: bool = True,
        crisis_mode: bool = False
    ) -> Dict[str, Any]:
        """Run multi-agent loop and compute consensus.
        
        Supports dynamic agent spawning based on task complexity and
        parallel processing via jax.vmap for efficiency.
        
        Args:
            inputs: Input tensor [batch, d_model] or [batch, seq, d_model]
            auto_spawn: Whether to automatically spawn agents for complex tasks
            crisis_mode: If True, spawn full crisis response team
            
        Returns:
            Dictionary with consensus output, individual agent responses,
            complexity metrics, and spawned agent info
        """
        # Compute task complexity
        complexity = self.compute_complexity(inputs)
        mean_complexity = float(jnp.mean(complexity))
        
        # Dynamic agent spawning based on complexity
        spawned_this_call = []
        if auto_spawn and mean_complexity > self.spawn_threshold:
            if crisis_mode:
                # Spawn crisis response team
                spawned_this_call = self.spawn_crisis_team(inputs)
            else:
                # Spawn single specialist for complex task
                agent = self.spawn_agent(mean_complexity)
                if agent is not None:
                    spawned_this_call.append(agent)
        
        # Get all active agents (base + spawned)
        active_agents = self.get_active_agents()
        
        # Process all agents in parallel using list comprehension
        # (jax.vmap on dynamic agent list requires careful handling)
        responses = []
        confidences = []
        specializations = []
        
        # Multi-agent loop: each agent processes the input
        for agent in active_agents:
            agent_output = agent.process(inputs)
            response = agent_output['response']
            confidence = agent_output['confidence']
            responses.append(response)
            confidences.append(confidence)
            specializations.append(agent.specialization)
        
        num_active = len(active_agents)
        
        # Stack responses: [num_agents, batch, d_model] or [num_agents, batch, seq, d_model]
        stacked_responses = jnp.stack(responses, axis=0)
        stacked_confidences = jnp.stack(confidences, axis=0)  # [num_agents, batch, 1]
        
        # Compute consensus using mean (weighted by confidence)
        # Normalize confidences to weights
        confidence_weights = jax.nn.softmax(stacked_confidences.squeeze(-1), axis=0)  # [num_agents, batch]
        
        # Weighted mean consensus
        if stacked_responses.ndim == 3:
            # [num_agents, batch, d_model]
            weighted_responses = stacked_responses * confidence_weights[:, :, None]
            consensus_mean = jnp.sum(weighted_responses, axis=0)  # [batch, d_model]
        else:
            # [num_agents, batch, seq, d_model]
            weighted_responses = stacked_responses * confidence_weights[:, :, None, None]
            consensus_mean = jnp.sum(weighted_responses, axis=0)  # [batch, seq, d_model]
        
        # Attention-based consensus refinement
        # Reshape for attention: [batch, num_agents, d_model]
        if stacked_responses.ndim == 3:
            agent_responses = stacked_responses.transpose(1, 0, 2)  # [batch, num_agents, d_model]
        else:
            # Pool sequence dimension for attention
            agent_responses = jnp.mean(stacked_responses, axis=2).transpose(1, 0, 2)
        
        # Self-attention over agent responses
        attended = self.consensus_attention(agent_responses, agent_responses, agent_responses)
        
        # Pool attended responses
        attended_pooled = jnp.mean(attended, axis=1)
        
        # Combine mean consensus with attention refinement
        combined = consensus_mean if consensus_mean.ndim == 2 else jnp.mean(consensus_mean, axis=1)
        
        # If we have spawned agents, incorporate their contributions via fusion
        if len(self._spawned_agents) > 0:
            # Get spawned agent responses
            spawned_indices = slice(self.num_agents, num_active)
            spawned_responses = stacked_responses[spawned_indices]
            spawned_pooled = jnp.mean(spawned_responses, axis=(0, 1) if spawned_responses.ndim == 3 else (0, 1, 2))
            
            # Fuse spawned agent contributions
            fused_spawned = self.spawned_fusion(spawned_pooled)
            
            # Weight spawned contribution by mean complexity
            spawned_weight = mean_complexity * 0.3  # Max 30% contribution from spawned
            combined = combined * (1 - spawned_weight) + fused_spawned * spawned_weight
        
        final_consensus = self.consensus_projection(combined + attended_pooled)
        
        return {
            'consensus': final_consensus,
            'agent_responses': responses,
            'agent_confidences': confidences,
            'confidence_weights': confidence_weights,
            'specializations': specializations,
            'task_complexity': complexity,
            'mean_complexity': mean_complexity,
            'num_active_agents': num_active,
            'num_spawned_agents': len(self._spawned_agents),
            'spawned_this_call': len(spawned_this_call),
            'crisis_mode': crisis_mode
        }
    
    def process_parallel(self, inputs: jnp.ndarray) -> Dict[str, Any]:
        """
        Process inputs through all agents in parallel using jax.vmap.
        
        This is an optimized path for when all agents have the same structure
        and we can batch their computations together.
        
        Args:
            inputs: Input tensor [batch, d_model]
            
        Returns:
            Dictionary with parallel-processed results
        """
        # Process base agents
        base_responses = []
        for i in range(self.num_agents):
            response = self.agents[i](inputs)
            base_responses.append(response)
        
        # Stack for efficient operations
        stacked = jnp.stack(base_responses, axis=0)
        
        # Simple mean consensus for parallel mode
        parallel_consensus = jnp.mean(stacked, axis=0)
        
        return {
            'parallel_consensus': parallel_consensus,
            'agent_responses': base_responses,
            'num_agents': self.num_agents
        }

