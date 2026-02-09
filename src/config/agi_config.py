import jax.numpy as jnp


# Model scale presets for different deployment scenarios
MODEL_PRESETS = {
    "tiny": {
        "d_model": 256,
        "num_heads": 4,
        "num_layers": 6,
        "moe_experts": 4,
        "vocab_size": 32000,
        "description": "Tiny model for testing (~50M params)",
    },
    "small": {
        "d_model": 384,
        "num_heads": 8,
        "num_layers": 12,
        "moe_experts": 8,
        "vocab_size": 32000,
        "description": "Small model for development (~150M params)",
    },
    "base": {
        "d_model": 768,
        "num_heads": 12,
        "num_layers": 12,
        "moe_experts": 8,
        "vocab_size": 50000,
        "description": "Base model for fine-tuning (~350M params)",
    },
    "large": {
        "d_model": 1024,
        "num_heads": 16,
        "num_layers": 24,
        "moe_experts": 16,
        "vocab_size": 50000,
        "description": "Large model for production (~1B params)",
    },
    "xlarge": {
        "d_model": 2048,
        "num_heads": 32,
        "num_layers": 32,
        "moe_experts": 32,
        "vocab_size": 100000,
        "description": "XLarge model for advanced tasks (~7B params)",
    },
    "xxlarge": {
        "d_model": 4096,
        "num_heads": 64,
        "num_layers": 48,
        "moe_experts": 64,
        "vocab_size": 150000,
        "description": "XXLarge model for SOTA performance (~70B params)",
    },
}


class AGIConfig:
    """
    Configuration class for RT-DLM AGI model with multi-modal processing,
    quantum-inspired components, and meta-learning capabilities.
    
    Note: Tokenization is handled externally by Auralith-Data-Pipeline.
    This config focuses on model architecture parameters.
    """

    def __init__(self, **kwargs):
        # --- Model Architecture Parameters ---
        self.vocab_size = kwargs.get("vocab_size", 32000)  # Base vocabulary size
        self.total_vocab_size = kwargs.get("total_vocab_size", 50000)  # Total vocab including all modalities
        self.max_seq_length = kwargs.get("max_seq_length", 2048)  # Maximum sequence length
        self.d_model = kwargs.get("d_model", 384)  # Embedding dimension (model width)
        self.num_heads = kwargs.get("num_heads", 8)  # Number of attention heads
        self.num_layers = kwargs.get("num_layers", 12)  # Number of transformer layers
        self.moe_experts = kwargs.get("moe_experts", 8)  # Number of experts in Mixture of Experts
        self.moe_top_k = kwargs.get("moe_top_k", 2)  # Top-k experts to select in MoE
        self.task_size = kwargs.get("task_size", 15)  # Task size for support and query sets (batches)
        self.prune_threshold = kwargs.get("prune_threshold", 0.01)  # Pruning threshold for MoE/Transformer/Self-Attention neurons.
        self.prune_interval = kwargs.get("prune_interval", 100)  # Pruning interval for MoE/Transformer/Self-Attention neurons.

        # --- Advanced Attention Parameters ---
        self.attention_type = kwargs.get("attention_type", "standard")  # "standard", "gqa", "mqa", "linear", "sliding"
        self.num_kv_heads = kwargs.get("num_kv_heads", None)  # KV heads for GQA (None=MHA, 1=MQA)
        self.position_encoding = kwargs.get("position_encoding", "rope")  # "rope", "learned", "alibi", "none"
        self.rope_theta = kwargs.get("rope_theta", 10000.0)  # RoPE base frequency
        self.rope_scaling = kwargs.get("rope_scaling", None)  # Extended context scaling (e.g., 2.0 for 2x length)
        self.sliding_window_size = kwargs.get("sliding_window_size", 512)  # Window size for sliding attention
        self.use_flash_attention = kwargs.get("use_flash_attention", False)  # Enable Flash Attention if available

        # --- Graph Neural Network Parameters ---
        self.graph_neurons_enabled = kwargs.get("graph_neurons_enabled", True)  # Enable graph-based neurons
        self.graph_max_nodes = kwargs.get("graph_max_nodes", 64)  # Maximum nodes in dynamic graphs
        self.graph_edge_threshold = kwargs.get("graph_edge_threshold", 0.3)  # Edge creation threshold
        self.graph_num_hops = kwargs.get("graph_num_hops", 3)  # Multi-hop reasoning steps
        self.graph_num_edge_types = kwargs.get("graph_num_edge_types", 8)  # Relational edge types
        self.graph_moe_routing = kwargs.get("graph_moe_routing", True)  # Graph-based MoE routing

        # --- Advanced AGI Features ---
        self.max_reasoning_steps = kwargs.get("max_reasoning_steps", 10)  # Chain-of-thought reasoning steps
        self.quantum_qubits = kwargs.get("quantum_qubits", 16)  # Number of qubits for quantum simulation
        self.quantum_layers = kwargs.get("quantum_layers", 4)  # Number of quantum-inspired layers
        self.meta_learning_enabled = kwargs.get("meta_learning_enabled", True)  # Enable meta-learning
        self.self_improvement_enabled = kwargs.get("self_improvement_enabled", True)  # Enable self-improvement

        # --- Recursive Language Model (RLM) Parameters ---
        self.rlm_enabled = kwargs.get("rlm_enabled", True)  # Enable RLM for long context
        self.rlm_max_recursion_depth = kwargs.get("rlm_max_recursion_depth", 5)  # Max recursion depth
        self.rlm_context_peek_size = kwargs.get("rlm_context_peek_size", 2000)  # Chars per peek
        self.rlm_tool_budget = kwargs.get("rlm_tool_budget", 20)  # Max tool calls per query
        self.rlm_auto_partition_threshold = kwargs.get("rlm_auto_partition_threshold", 8000)  # Auto-partition above this
        self.rlm_direct_context_threshold = kwargs.get("rlm_direct_context_threshold", 2000)  # Use direct pass below this

        # --- Multi-Modal Parameters ---
        self.multimodal_enabled = kwargs.get("multimodal_enabled", True)  # Enable multi-modal processing
        self.vision_patch_size = kwargs.get("vision_patch_size", 16)  # ViT patch size
        self.vision_layers = kwargs.get("vision_layers", 6)  # Number of vision transformer layers
        self.audio_freq_bins = kwargs.get("audio_freq_bins", 128)  # Audio frequency bins
        self.video_frames = kwargs.get("video_frames", 16)  # Number of video frames to process
        
        # --- Ethical AI Parameters ---
        self.ethics_enabled = kwargs.get("ethics_enabled", True)  # Enable ethical reasoning
        self.ethics_weight = kwargs.get("ethics_weight", 0.1)  # Weight for ethical loss
        self.bias_detection_enabled = kwargs.get("bias_detection_enabled", True)  # Enable bias detection
        self.fairness_constraints = kwargs.get("fairness_constraints", True)  # Apply fairness constraints
        
        # --- Self-Evolution Parameters ---
        self.auto_architecture_search = kwargs.get("auto_architecture_search", False)  # Neural architecture search
        self.dynamic_layer_creation = kwargs.get("dynamic_layer_creation", False)  # Dynamic layer addition
        self.capability_expansion = kwargs.get("capability_expansion", True)  # Expand capabilities over time
        self.knowledge_distillation = kwargs.get("knowledge_distillation", True)  # Self-teaching
        
        # --- Advanced Memory Parameters ---
        self.episodic_memory_enabled = kwargs.get("episodic_memory_enabled", True)  # Episodic memory
        self.semantic_memory_size = kwargs.get("semantic_memory_size", 50000)  # Semantic memory size
        self.working_memory_capacity = kwargs.get("working_memory_capacity", 7)  # Working memory slots
        self.memory_consolidation = kwargs.get("memory_consolidation", True)  # Memory consolidation
        
        # --- AGI-Scale Attention Parameters ---
        # Ring Attention for infinite context distributed across devices
        self.use_agi_attention = kwargs.get("use_agi_attention", False)  # Enable AGI attention features
        self.enable_ring_attention = kwargs.get("enable_ring_attention", True)  # Ring Attention for distributed infinite context
        self.ring_block_size = kwargs.get("ring_block_size", 512)  # Block size for Ring Attention
        self.num_ring_devices = kwargs.get("num_ring_devices", 1)  # Number of devices for distributed attention
        
        # Cross-Memory Attention for LTM/STM/MTM interaction
        self.enable_memory_cross_attention = kwargs.get("enable_memory_cross_attention", True)  # Memory banks interact via attention
        self.memory_attention_heads = kwargs.get("memory_attention_heads", 4)  # Heads for memory cross-attention
        self.memory_dropout = kwargs.get("memory_dropout", 0.1)  # Dropout for memory attention
        
        # Infinite Context via hierarchical compression
        self.enable_infinite_context = kwargs.get("enable_infinite_context", False)  # Infinite context mode
        self.context_chunk_size = kwargs.get("context_chunk_size", 1024)  # Chunk size for infinite context
        self.global_context_size = kwargs.get("global_context_size", 256)  # Compressed global context tokens
        self.context_compression_ratio = kwargs.get("context_compression_ratio", 4)  # Compression ratio for chunks
        
        # --- Continual Learning Parameters ---
        self.continual_learning = kwargs.get("continual_learning", True)  # Enable continual learning (EWC)
        self.lambda_ewc = kwargs.get("lambda_ewc", 1000.0)  # EWC regularization strength
        self.lambda_si = kwargs.get("lambda_si", 1.0)  # Synaptic Intelligence strength
        self.max_task_memories = kwargs.get("max_task_memories", 10)  # Max tasks to remember
        
        # --- Consciousness Simulation Parameters ---
        self.consciousness_simulation = kwargs.get("consciousness_simulation", False)  # Simulate consciousness
        self.self_awareness_level = kwargs.get("self_awareness_level", 0.3)  # Self-awareness simulation
        self.introspection_enabled = kwargs.get("introspection_enabled", True)  # Self-monitoring
        self.goal_setting_enabled = kwargs.get("goal_setting_enabled", True)  # Autonomous goal setting

        # Ensure d_model is divisible by num_heads for MultiHeadAttention compatibility
        if self.d_model % self.num_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})")

        # --- Training Hyperparameters ---
        self.batch_size = kwargs.get("batch_size", 32)  # Batch size for training
        self.learning_rate = kwargs.get("learning_rate", 1e-4)  # Initial learning rate
        self.inner_learning_rate = kwargs.get("inner_learning_rate", 0.01)  # Inner loop learning rate for MAML
        self.num_inner_steps = kwargs.get("num_inner_steps", 10)  # Number of inner loop steps for MAML
        self.num_epochs = kwargs.get("num_epochs", 3)  # Number of training epochs
        self.eval_interval = kwargs.get("eval_interval", 25)  # Frequency of evaluation during training
        self.temperature = kwargs.get("temperature", 1.2)  # Temperature for sampling (if applicable)
        self.label_smoothing = kwargs.get("label_smoothing", 0.1)  # Label smoothing factor for loss

        # Optimizer parameters
        self.warmup_steps = kwargs.get("warmup_steps", 5000)  # Warmup steps for learning rate schedule
        self.decay_steps = kwargs.get("decay_steps", 200000)  # Decay steps for cosine decay
        self.init_lr = kwargs.get("init_lr", 2e-6)  # Initial learning rate for warmup
        self.end_lr = kwargs.get("end_lr", 2e-6)  # End learning rate after decay
        self.weight_decay = kwargs.get("weight_decay", 1e-3)  # Weight decay for AdamW
        self.clip_norm = kwargs.get("clip_norm", 0.5)  # Global norm clipping value

        # --- Data Processing Parameters ---
        self.max_seq_length = kwargs.get("max_seq_length", 4096)  # Maximum sequence length for input
        self.pad_token_id = kwargs.get("pad_token_id", 0)  # Token ID used for padding
        self.max_sentence_length = kwargs.get("max_sentence_length", 2048)  # Maximum allowed sentence length
        self.input_sentence_size = kwargs.get("input_sentence_size", 500000)  # Total number of sentences in input data
        self.num_threads = kwargs.get("num_threads", 16)  # Number of threads for data processing

        # --- Memory Bank Parameters ---
        self.memory_size = kwargs.get("memory_size", 5000)  # Size of the long-term memory bank
        self.retrieval_k = kwargs.get("retrieval_k", 3)  # Number of top-k items to retrieve from LTM
        self.stm_buffer_size = kwargs.get("stm_buffer_size", self.batch_size)  # Default STM buffer size (tunable)
        self.ltm_weight = kwargs.get("ltm_weight", 0.5)  # Weight for long-term memory contribution
        self.stm_weight = kwargs.get("stm_weight", 0.5)  # Weight for short-term memory contribution
        self.mtm_weight = kwargs.get("mtm_weight", 0.5)  # Weight for mid-term memory contribution

        # --- Advanced Training Features ---
        self.curriculum_learning = kwargs.get("curriculum_learning", True)  # Progressive difficulty
        self.adversarial_training = kwargs.get("adversarial_training", False)  # Robustness training
        self.continual_learning = kwargs.get("continual_learning", True)  # Learn without forgetting
        self.few_shot_adaptation = kwargs.get("few_shot_adaptation", True)  # Few-shot learning
        
        # --- AGI Capabilities Flags ---
        self.scientific_reasoning = kwargs.get("scientific_reasoning", True)  # Scientific discovery
        self.creative_generation = kwargs.get("creative_generation", True)  # Creative content
        self.social_intelligence = kwargs.get("social_intelligence", True)  # Social understanding
        self.emotional_intelligence = kwargs.get("emotional_intelligence", True)  # Emotional reasoning
        
        # --- Safety and Alignment ---
        self.alignment_training = kwargs.get("alignment_training", True)  # Human alignment
        self.value_learning = kwargs.get("value_learning", True)  # Learn human values
        self.interpretability = kwargs.get("interpretability", True)  # Model interpretability
        self.safety_constraints = kwargs.get("safety_constraints", True)  # Safety guardrails
        
        # --- Spike Attention Parameters ---
        self.spike_threshold = kwargs.get("spike_threshold", 0.1)  # Spiking attention threshold
        self.EPSILON = kwargs.get("epsilon", 1e-8)  # Small value for numerical stability
        
        # --- Performance Optimization Parameters ---
        # Mixed Precision Training
        self.mixed_precision = kwargs.get("mixed_precision", False)  # Enable mixed precision
        self.precision_dtype = kwargs.get("precision_dtype", "float32")  # float32, bfloat16, float16
        self.compute_dtype = kwargs.get("compute_dtype", "float32")  # Compute precision
        
        # Gradient Checkpointing (Memory Efficiency)
        self.gradient_checkpointing = kwargs.get("gradient_checkpointing", False)  # Enable gradient checkpointing
        self.checkpoint_every_n_layers = kwargs.get("checkpoint_every_n_layers", 2)  # Checkpoint frequency
        
        # Distributed Training
        self.distributed_training = kwargs.get("distributed_training", False)  # Enable distributed training
        self.num_devices = kwargs.get("num_devices", 1)  # Number of devices for training
        self.data_parallel = kwargs.get("data_parallel", True)  # Data parallelism
        self.model_parallel = kwargs.get("model_parallel", False)  # Model parallelism
        self.gradient_accumulation_steps = kwargs.get("gradient_accumulation_steps", 1)  # Gradient accumulation
        self.enable_memory_profiling = kwargs.get("enable_memory_profiling", False)  # Enable memory profiling
        
        # Production Evaluation Metrics
        self.enable_fairness_tracking = kwargs.get("enable_fairness_tracking", False)  # Track fairness metrics
        self.calibration_bins = kwargs.get("calibration_bins", 10)  # Bins for calibration tracking
        self.perplexity_window = kwargs.get("perplexity_window", 100)  # Window for running perplexity
        
        # Extended Quantum Simulation
        self.quantum_max_qubits = kwargs.get("quantum_max_qubits", 64)  # Extended qubit simulation limit
        self.quantum_chunked_simulation = kwargs.get("quantum_chunked_simulation", True)  # Enable chunked simulation
        self.quantum_sparse_mode = kwargs.get("quantum_sparse_mode", True)  # Sparse state representation

        # --- Compute Controller Parameters (Dynamic Module Orchestration) ---
        self.use_compute_controller = kwargs.get("use_compute_controller", False)  # Enable dynamic compute allocation
        self.controller_max_steps = kwargs.get("controller_max_steps", 10)  # Max steps per forward pass
        self.controller_initial_budget = kwargs.get("controller_initial_budget", 1.0)  # Initial compute budget
        self.controller_halt_threshold = kwargs.get("controller_halt_threshold", 0.8)  # Halt when confidence exceeds
        self.controller_min_budget = kwargs.get("controller_min_budget", 0.05)  # Minimum budget per step
        self.controller_temperature = kwargs.get("controller_temperature", 1.0)  # Module selection temperature
        
        # Controller Training Losses
        self.controller_lambda_compute = kwargs.get("controller_lambda_compute", 0.01)  # Compute efficiency weight
        self.controller_lambda_utilization = kwargs.get("controller_lambda_utilization", 0.005)  # Module utilization weight
        self.controller_lambda_calibration = kwargs.get("controller_lambda_calibration", 0.1)  # Confidence calibration weight
        self.controller_lambda_budget = kwargs.get("controller_lambda_budget", 0.05)  # Budget adherence weight
        self.controller_lambda_ponder = kwargs.get("controller_lambda_ponder", 0.01)  # Ponder cost weight
        
        # Controller Strategy
        self.controller_strategy = kwargs.get("controller_strategy", "balanced")  # "fast", "balanced", "thorough", "adaptive"

        # Validate configuration
        self._validate_config()

    def _validate_config(self):
        """Validate configuration parameters"""
        assert self.d_model > 0, "d_model must be positive"
        assert self.num_heads > 0, "num_heads must be positive"
        assert self.vocab_size > 0, "vocab_size must be positive"
        assert 0 <= self.spike_threshold <= 1, "spike_threshold must be between 0 and 1"
        assert self.quantum_qubits > 0, "quantum_qubits must be positive"
        assert self.max_reasoning_steps > 0, "max_reasoning_steps must be positive"
        
        # Validate multi-modal parameters
        if self.multimodal_enabled:
            assert self.vision_patch_size > 0, "vision_patch_size must be positive"
            assert self.audio_freq_bins > 0, "audio_freq_bins must be positive"
            assert self.video_frames > 0, "video_frames must be positive"
        
        # Validate precision dtype
        valid_dtypes = ["float32", "bfloat16", "float16"]
        assert self.precision_dtype in valid_dtypes, f"precision_dtype must be one of {valid_dtypes}"
        assert self.compute_dtype in valid_dtypes, f"compute_dtype must be one of {valid_dtypes}"
        
        # Validate gradient checkpointing
        if self.gradient_checkpointing:
            assert self.checkpoint_every_n_layers > 0, "checkpoint_every_n_layers must be positive"
            
        # Validate distributed settings
        if self.distributed_training:
            assert self.num_devices >= 1, "num_devices must be at least 1"
            assert self.gradient_accumulation_steps >= 1, "gradient_accumulation_steps must be at least 1"
        
        # Validate compute controller settings
        if self.use_compute_controller:
            assert self.controller_max_steps >= 1, "controller_max_steps must be at least 1"
            assert 0 < self.controller_initial_budget <= 10.0, "controller_initial_budget must be between 0 and 10"
            assert 0 < self.controller_halt_threshold <= 1.0, "controller_halt_threshold must be between 0 and 1"
            valid_strategies = ["fast", "balanced", "thorough", "adaptive"]
            assert self.controller_strategy in valid_strategies, f"controller_strategy must be one of {valid_strategies}"
            
    def to_dict(self):
        """Convert config to dictionary"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def update(self, **kwargs):
        """Update configuration with new values"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: Unknown configuration parameter: {key}")
        self._validate_config()
        
    def save(self, filepath):
        """Save configuration to file"""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
            
    @classmethod
    def load(cls, filepath):
        """Load configuration from file"""
        import json
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    @classmethod
    def from_preset(cls, preset_name: str, **overrides):
        """
        Create configuration from a preset scale.
        
        Available presets: tiny, small, base, large, xlarge, xxlarge
        
        Args:
            preset_name: Name of the preset (e.g., 'large', 'xlarge')
            **overrides: Additional parameters to override preset values
            
        Returns:
            AGIConfig instance
            
        Example:
            # Create large model config
            config = AGIConfig.from_preset('large')
            
            # Create xlarge with custom learning rate
            config = AGIConfig.from_preset('xlarge', learning_rate=5e-5)
        """
        if preset_name not in MODEL_PRESETS:
            available = ', '.join(MODEL_PRESETS.keys())
            raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")
        
        preset = MODEL_PRESETS[preset_name].copy()
        preset.pop('description', None)  # Remove description from config params
        preset.update(overrides)
        
        return cls(**preset)
    
    @staticmethod
    def list_presets():
        """List available model presets with descriptions."""
        print("Available Model Presets:")
        print("-" * 60)
        for name, preset in MODEL_PRESETS.items():
            desc = preset.get('description', 'No description')
            params = f"d_model={preset['d_model']}, layers={preset['num_layers']}"
            print(f"  {name:10s} - {desc}")
            print(f"             {params}")
        print("-" * 60)
    
    def get_model_size_estimate(self):
        """Estimate model size in parameters"""
        # Rough estimate based on transformer architecture
        embedding_params = self.vocab_size * self.d_model
        attention_params = self.num_layers * (4 * self.d_model * self.d_model)
        ffn_params = self.num_layers * (self.d_model * self.d_model * 4)
        moe_params = self.moe_experts * (self.d_model * self.d_model * 2)
        
        total_params = embedding_params + attention_params + ffn_params + moe_params
        
        # Add quantum and multi-modal components
        if self.multimodal_enabled:
            total_params += self.d_model * self.d_model * 4  # Vision/audio encoders
            
        if self.quantum_layers > 0:
            total_params += self.quantum_layers * self.d_model * self.quantum_qubits
            
        return total_params
    
    def print_summary(self):
        """Print configuration summary"""
        print("=" * 60)
        print("RT-DLM Advanced AGI Configuration Summary")
        print("=" * 60)
        print("Model Architecture:")
        print(f"  - d_model: {self.d_model}")
        print(f"  - num_heads: {self.num_heads}")
        print(f"  - num_layers: {self.num_layers}")
        print(f"  - vocab_size: {self.vocab_size}")
        print(f"  - MoE experts: {self.moe_experts}")
        print(f"  - Estimated parameters: {self.get_model_size_estimate():,}")
        
        print("\nGraph Neural Networks:")
        print(f"  - Graph neurons: {self.graph_neurons_enabled}")
        if self.graph_neurons_enabled:
            print(f"  - Max nodes: {self.graph_max_nodes}")
            print(f"  - Multi-hop reasoning: {self.graph_num_hops} hops")
            print(f"  - Graph MoE routing: {self.graph_moe_routing}")
        
        print("\nAdvanced Features:")
        print(f"  - Multi-modal: {self.multimodal_enabled}")
        print(f"  - Quantum layers: {self.quantum_layers}")
        print(f"  - Meta-learning: {self.meta_learning_enabled}")
        print(f"  - Self-improvement: {self.self_improvement_enabled}")
        print(f"  - Ethical AI: {self.ethics_enabled}")
        print(f"  - Reasoning steps: {self.max_reasoning_steps}")
        
        print("\nMemory System:")
        print(f"  - Memory size: {self.memory_size}")
        print(f"  - Retrieval k: {self.retrieval_k}")
        print(f"  - Working memory: {self.working_memory_capacity}")
        
        print("\nPerformance Optimization:")
        print(f"  - Mixed precision: {self.mixed_precision} ({self.precision_dtype})")
        print(f"  - Gradient checkpointing: {self.gradient_checkpointing}")
        print(f"  - Distributed training: {self.distributed_training}")
        if self.distributed_training:
            print(f"    - Devices: {self.num_devices}")
            print(f"    - Gradient accumulation: {self.gradient_accumulation_steps}")
        print(f"  - Quantum max qubits: {self.quantum_max_qubits}")
        print(f"  - Quantum chunked sim: {self.quantum_chunked_simulation}")
        
        print("\nCompute Controller:")
        print(f"  - Enabled: {self.use_compute_controller}")
        if self.use_compute_controller:
            print(f"    - Strategy: {self.controller_strategy}")
            print(f"    - Max steps: {self.controller_max_steps}")
            print(f"    - Initial budget: {self.controller_initial_budget}")
            print(f"    - Halt threshold: {self.controller_halt_threshold}")
        
        print("\nTraining:")
        print(f"  - Batch size: {self.batch_size}")
        print(f"  - Learning rate: {self.learning_rate}")
        print(f"  - Clip norm: {self.clip_norm}")
        print(f"  - Epochs: {self.num_epochs}")
        print("=" * 60)

