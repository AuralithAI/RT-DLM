# Model scale presets for different deployment scenarios
MODEL_PRESETS = {
    "tiny": {
        "d_model": 256,
        "num_heads": 4,
        "num_layers": 6,
        "moe_experts": 4,
        "vocab_size": 32000,
        "base_d_model": 256,
        "description": "Tiny model for testing (~50M params)",
    },
    "small": {
        "d_model": 384,
        "num_heads": 8,
        "num_layers": 12,
        "moe_experts": 8,
        "vocab_size": 32000,
        "base_d_model": 256,
        "description": "Small model for development (~150M params)",
    },
    "base": {
        "d_model": 768,
        "num_heads": 12,
        "num_layers": 12,
        "moe_experts": 8,
        "vocab_size": 50000,
        "base_d_model": 256,
        "description": "Base model for fine-tuning (~350M params)",
    },
    "large": {
        "d_model": 1024,
        "num_heads": 16,
        "num_layers": 24,
        "moe_experts": 16,
        "vocab_size": 50000,
        "base_d_model": 256,
        "description": "Large model for production (~1B params)",
    },
    "xlarge": {
        "d_model": 2048,
        "num_heads": 32,
        "num_layers": 32,
        "moe_experts": 32,
        "vocab_size": 100000,
        "base_d_model": 256,
        "description": "XLarge model for advanced tasks (~7B params)",
    },
    "xxlarge": {
        "d_model": 4096,
        "num_heads": 64,
        "num_layers": 48,
        "moe_experts": 64,
        "vocab_size": 150000,
        "base_d_model": 256,
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
        self.prune_threshold = kwargs.get(
            "prune_threshold", 0.01
        )  # Pruning threshold for MoE/Transformer/Self-Attention neurons.
        self.prune_interval = kwargs.get(
            "prune_interval", 100
        )  # Pruning interval for MoE/Transformer/Self-Attention neurons.

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
        self.quantum_qubits = kwargs.get("quantum_qubits", 0)  # Number of qubits for quantum simulation (0 = disabled)
        self.quantum_layers = kwargs.get("quantum_layers", 0)  # Number of quantum-inspired layers (0 = disabled)
        self.meta_learning_enabled = kwargs.get("meta_learning_enabled", True)  # Enable meta-learning
        self.self_improvement_enabled = kwargs.get("self_improvement_enabled", True)  # Enable self-improvement

        # --- muP / MoE regularization ---
        self.base_d_model = kwargs.get("base_d_model", 256)
        self.use_mup = kwargs.get("use_mup", False)
        self.moe_z_loss_weight = kwargs.get("moe_z_loss_weight", 1e-4)
        self.moe_router_z_loss_weight = kwargs.get("moe_router_z_loss_weight", 1e-3)

        # --- Recursive Language Model (RLM) Parameters ---
        self.rlm_enabled = kwargs.get("rlm_enabled", True)  # Enable RLM for long context
        self.rlm_max_recursion_depth = kwargs.get("rlm_max_recursion_depth", 5)  # Max recursion depth
        self.rlm_context_peek_size = kwargs.get("rlm_context_peek_size", 2000)  # Chars per peek
        self.rlm_tool_budget = kwargs.get("rlm_tool_budget", 20)  # Max tool calls per query
        self.rlm_auto_partition_threshold = kwargs.get(
            "rlm_auto_partition_threshold", 8000
        )  # Auto-partition above this
        self.rlm_direct_context_threshold = kwargs.get(
            "rlm_direct_context_threshold", 2000
        )  # Use direct pass below this

        # --- Multi-Modal Parameters ---
        self.multimodal_enabled = kwargs.get("multimodal_enabled", True)  # Enable multi-modal processing
        self.vision_patch_size = kwargs.get("vision_patch_size", 16)  # ViT patch size
        self.vision_layers = kwargs.get("vision_layers", 6)  # Number of vision transformer layers
        self.audio_freq_bins = kwargs.get("audio_freq_bins", 128)  # Audio frequency bins
        self.video_frames = kwargs.get("video_frames", 16)  # Number of video frames to process
        # Multi-resolution vision
        self.enable_multi_res_vision = kwargs.get("enable_multi_res_vision", False)
        self.vision_patch_sizes = kwargs.get("vision_patch_sizes", [8, 16, 32])
        self.vision_in_channels = kwargs.get("vision_in_channels", 3)

        # Video spatiotemporal config
        self.video_patch_size = kwargs.get("video_patch_size", 16)
        self.video_motion_window = kwargs.get("video_motion_window", 4)

        # Extended modality flags
        self.enable_document_modality = kwargs.get("enable_document_modality", False)
        self.enable_pointcloud_modality = kwargs.get("enable_pointcloud_modality", False)
        self.enable_biosignal_modality = kwargs.get("enable_biosignal_modality", False)
        self.enable_tactile_modality = kwargs.get("enable_tactile_modality", False)
        self.enable_action_modality = kwargs.get("enable_action_modality", False)
        self.action_num_axes = kwargs.get("action_num_axes", 15)
        self.action_num_bins = kwargs.get("action_num_bins", 256)

        # VQ-VAE image tokenization
        self.enable_image_vq = kwargs.get("enable_image_vq", False)
        self.image_vq_codes = kwargs.get("image_vq_codes", 8192)
        self.image_vq_code_dim = kwargs.get("image_vq_code_dim", 256)
        self.image_vq_downsample = kwargs.get("image_vq_downsample", 16)

        # Spectrogram decoder (audio reconstruction head)
        self.enable_spectrogram_decoder = kwargs.get("enable_spectrogram_decoder", False)
        self.spectrogram_n_mels = kwargs.get("spectrogram_n_mels", 128)

        # Streaming video buffer
        self.streaming_video_max_frames = kwargs.get("streaming_video_max_frames", 32)
        self.streaming_compressed_size = kwargs.get("streaming_compressed_size", 64)

        # Cross-modal alignment losses
        self.contrastive_loss_weight = kwargs.get("contrastive_loss_weight", 0.1)
        self.modality_sync_loss_weight = kwargs.get("modality_sync_loss_weight", 0.05)
        self.modality_sync_tolerance = kwargs.get("modality_sync_tolerance", 0.05)
        self.contrastive_temperature = kwargs.get("contrastive_temperature", 0.07)

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
        self.enable_ring_attention = kwargs.get(
            "enable_ring_attention", True
        )  # Ring Attention for distributed infinite context
        self.ring_block_size = kwargs.get("ring_block_size", 512)  # Block size for Ring Attention
        self.num_ring_devices = kwargs.get("num_ring_devices", 1)  # Number of devices for distributed attention

        # Cross-Memory Attention for LTM/STM/MTM interaction
        self.enable_memory_cross_attention = kwargs.get(
            "enable_memory_cross_attention", True
        )  # Memory banks interact via attention
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
        self.controller_lambda_utilization = kwargs.get(
            "controller_lambda_utilization", 0.005
        )  # Module utilization weight
        self.controller_lambda_calibration = kwargs.get(
            "controller_lambda_calibration", 0.1
        )  # Confidence calibration weight
        self.controller_lambda_budget = kwargs.get("controller_lambda_budget", 0.05)  # Budget adherence weight
        self.controller_lambda_ponder = kwargs.get("controller_lambda_ponder", 0.01)  # Ponder cost weight

        # Controller Strategy
        self.controller_strategy = kwargs.get(
            "controller_strategy", "balanced"
        )  # "fast", "balanced", "thorough", "adaptive"

        # --- GRPO (Group Relative Policy Optimization) ---
        self.use_grpo = kwargs.get("use_grpo", False)  # Enable GRPO value head & training
        self.grpo_num_groups = kwargs.get("grpo_num_groups", 4)  # Number of response groups per prompt
        self.grpo_group_size = kwargs.get("grpo_group_size", 4)  # Responses per group
        self.grpo_clip_eps = kwargs.get("grpo_clip_eps", 0.2)  # PPO-style clip epsilon
        self.grpo_kl_coeff = kwargs.get("grpo_kl_coeff", 0.01)  # KL penalty coefficient
        self.grpo_value_loss_coeff = kwargs.get("grpo_value_loss_coeff", 0.5)  # Value head loss weight
        self.grpo_entropy_coeff = kwargs.get("grpo_entropy_coeff", 0.01)  # Entropy bonus coefficient
        self.grpo_gamma = kwargs.get("grpo_gamma", 1.0)  # Discount factor for returns
        self.grpo_lam = kwargs.get("grpo_lam", 0.95)  # GAE lambda for advantage estimation
        self.grpo_normalize_advantages = kwargs.get("grpo_normalize_advantages", True)  # Normalize advantages per group
        self.grpo_reward_model = kwargs.get("grpo_reward_model", "internal")  # "internal", "external", "rule_based"

        # --- Verify / Reflect Loop ---
        self.enable_verify_reflect = kwargs.get("enable_verify_reflect", False)  # Enable verify/reflect reasoning loop
        self.max_verify_steps = kwargs.get("max_verify_steps", 3)  # Max verification iterations
        self.verify_confidence_threshold = kwargs.get(
            "verify_confidence_threshold", 0.85
        )  # Confidence to accept without reflection
        self.reflect_temperature = kwargs.get("reflect_temperature", 0.7)  # Temperature for reflection sampling
        self.verify_reward_bonus = kwargs.get("verify_reward_bonus", 0.1)  # Bonus reward for passing verification
        self.reflect_penalty = kwargs.get("reflect_penalty", -0.05)  # Penalty for requiring reflection

        # --- KV Prefix Cache ---
        self.enable_kv_cache = kwargs.get("enable_kv_cache", False)  # Enable KV prefix caching
        self.kv_cache_prefix_len = kwargs.get("kv_cache_prefix_len", 256)  # Max prefix tokens to cache
        self.kv_cache_max_batch = kwargs.get("kv_cache_max_batch", 32)  # Max batch size for cache
        self.kv_cache_eviction = kwargs.get("kv_cache_eviction", "lru")  # Cache eviction: "lru", "fifo", "lfu"
        self.kv_cache_dtype = kwargs.get("kv_cache_dtype", "bfloat16")  # Cache storage dtype

        # --- Self-Critique ---
        self.enable_self_critique = kwargs.get("enable_self_critique", False)  # Enable self-critique head
        self.self_critique_threshold = kwargs.get(
            "self_critique_threshold", 0.6
        )  # Quality threshold to trigger revision
        self.max_revisions = kwargs.get("max_revisions", 2)  # Maximum self-revision iterations
        self.critique_loss_coeff = kwargs.get("critique_loss_coeff", 0.1)  # Weight for critique loss

        # --- Think Budget ---
        self.enable_think_budget = kwargs.get("enable_think_budget", False)  # Enable adaptive think budget
        self.think_budget_max_tokens = kwargs.get("think_budget_max_tokens", 1024)  # Max reasoning tokens
        self.think_budget_min_tokens = kwargs.get("think_budget_min_tokens", 32)  # Min reasoning tokens
        self.think_budget_difficulty_scale = kwargs.get(
            "think_budget_difficulty_scale", True
        )  # Scale budget by difficulty

        # --- Hard Negative Mining (Contrastive Loss) ---
        self.enable_hard_negative_mining = kwargs.get("enable_hard_negative_mining", False)
        self.contrastive_margin = kwargs.get("contrastive_margin", 0.2)  # Margin for semi-hard selection
        self.hard_negative_ratio = kwargs.get("hard_negative_ratio", 0.5)  # Fraction of negatives to keep

        # --- MLflow Experiment Tracking ---
        self.mlflow_enabled = kwargs.get("mlflow_enabled", False)  # Enable MLflow tracking
        self.mlflow_tracking_uri = kwargs.get("mlflow_tracking_uri", None)  # MLflow server URI (None → local ./mlruns)
        self.mlflow_experiment_name = kwargs.get("mlflow_experiment_name", "rtdlm_training")  # Experiment name
        self.mlflow_run_name = kwargs.get("mlflow_run_name", None)  # Optional run name
        self.mlflow_log_interval = kwargs.get("mlflow_log_interval", 10)  # Steps between metric logs

        # --- Synthetic Data Self-Improvement ---
        self.enable_synthetic_data = kwargs.get("enable_synthetic_data", False)  # Enable synthetic hard-example mining
        self.synthetic_data_difficulty_threshold = kwargs.get(
            "synthetic_data_difficulty_threshold", 0.6
        )  # Confidence below this → "hard"
        self.synthetic_data_batch_multiplier = kwargs.get(
            "synthetic_data_batch_multiplier", 0.2
        )  # Fraction of epoch size to generate
        self.synthetic_data_quality_improvement_min = kwargs.get(
            "synthetic_data_quality_improvement_min", 0.1
        )  # Min quality gain to keep sample
        self.synthetic_data_output_dir = kwargs.get(
            "synthetic_data_output_dir", "synthetic_shards"
        )  # Dir for generated shards

        # --- Code Modality Routing ---
        self.enable_code_routing = kwargs.get("enable_code_routing", False)  # Enable code-aware module routing
        self.code_routing_threshold = kwargs.get(
            "code_routing_threshold", 0.6
        )  # Code confidence threshold for routing boost
        self.code_routing_boost = kwargs.get(
            "code_routing_boost", 1.5
        )  # Multiplier for code-relevant module probabilities

        # --- Benchmark Evaluation ---
        self.benchmark_enabled = kwargs.get("benchmark_enabled", False)  # Enable benchmark harness
        self.benchmark_names = kwargs.get("benchmark_names", ["gpqa"])  # Benchmarks to run: gpqa, aime, swe, livecode
        self.benchmark_max_samples = kwargs.get("benchmark_max_samples", None)  # Max samples per benchmark (None=all)
        self.benchmark_think_budget = kwargs.get("benchmark_think_budget", "medium")  # Think-budget preset for eval
        self.benchmark_output_dir = kwargs.get("benchmark_output_dir", "eval_results")  # Output directory for results
        self.benchmark_eval_interval = kwargs.get("benchmark_eval_interval", 1)  # Run benchmarks every N epochs

        # Validate configuration
        self._validate_config()
        self._compose_subconfigs()

    def _compose_subconfigs(self):
        """Build composed sub-config dataclasses mirroring flat fields."""
        from src.config.architecture_config import ArchitectureConfig
        from src.config.training_config import TrainingConfig
        from src.config.precision_config import PrecisionConfig
        from src.config.parallelism_config import ParallelismConfig
        from src.config.multimodal_config import MultimodalConfig
        from src.config.safety_config import SafetyConfig

        self.architecture = ArchitectureConfig(
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            vocab_size=self.vocab_size,
            max_seq_length=self.max_seq_length,
            base_d_model=self.base_d_model,
            moe_experts=self.moe_experts,
            moe_top_k=self.moe_top_k,
            attention_type=self.attention_type,
            num_kv_heads=self.num_kv_heads,
            position_encoding=self.position_encoding,
            rope_theta=self.rope_theta,
            rope_scaling=self.rope_scaling,
            sliding_window_size=self.sliding_window_size,
            use_flash_attention=self.use_flash_attention,
        )
        self.training = TrainingConfig(
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            num_epochs=self.num_epochs,
            warmup_steps=self.warmup_steps,
            decay_steps=self.decay_steps,
            init_lr=self.init_lr,
            end_lr=self.end_lr,
            weight_decay=self.weight_decay,
            clip_norm=self.clip_norm,
            label_smoothing=self.label_smoothing,
            eval_interval=self.eval_interval,
            moe_z_loss_weight=self.moe_z_loss_weight,
            moe_router_z_loss_weight=self.moe_router_z_loss_weight,
            use_mup=self.use_mup,
        )
        self.precision = PrecisionConfig(
            mixed_precision=self.mixed_precision,
            precision_dtype=self.precision_dtype,
            compute_dtype=self.compute_dtype,
            gradient_checkpointing=self.gradient_checkpointing,
            checkpoint_every_n_layers=self.checkpoint_every_n_layers,
        )
        self.parallelism = ParallelismConfig(
            distributed_training=self.distributed_training,
            num_devices=self.num_devices,
            data_parallel=self.data_parallel,
            model_parallel=self.model_parallel,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
        )
        self.multimodal = MultimodalConfig(
            multimodal_enabled=self.multimodal_enabled,
            vision_patch_size=self.vision_patch_size,
            vision_layers=self.vision_layers,
            audio_freq_bins=self.audio_freq_bins,
            video_frames=self.video_frames,
            enable_multi_res_vision=self.enable_multi_res_vision,
            vision_patch_sizes=list(self.vision_patch_sizes),
            vision_in_channels=self.vision_in_channels,
            video_patch_size=self.video_patch_size,
            video_motion_window=self.video_motion_window,
            enable_document_modality=self.enable_document_modality,
            enable_pointcloud_modality=self.enable_pointcloud_modality,
            enable_biosignal_modality=self.enable_biosignal_modality,
            enable_tactile_modality=self.enable_tactile_modality,
            enable_action_modality=self.enable_action_modality,
            action_num_axes=self.action_num_axes,
            action_num_bins=self.action_num_bins,
            enable_image_vq=self.enable_image_vq,
            image_vq_codes=self.image_vq_codes,
            image_vq_code_dim=self.image_vq_code_dim,
            image_vq_downsample=self.image_vq_downsample,
            enable_spectrogram_decoder=self.enable_spectrogram_decoder,
            spectrogram_n_mels=self.spectrogram_n_mels,
            streaming_video_max_frames=self.streaming_video_max_frames,
            streaming_compressed_size=self.streaming_compressed_size,
        )
        self.safety = SafetyConfig(
            ethics_enabled=self.ethics_enabled,
            ethics_weight=self.ethics_weight,
            bias_detection_enabled=self.bias_detection_enabled,
            fairness_constraints=self.fairness_constraints,
            alignment_training=self.alignment_training,
            value_learning=self.value_learning,
            interpretability=self.interpretability,
            safety_constraints=self.safety_constraints,
        )
        self.architecture.validate()
        self.training.validate()
        self.precision.validate()
        self.parallelism.validate()
        self.multimodal.validate()
        self.safety.validate()

    def _validate_config(self):
        """Validate configuration parameters"""
        assert self.d_model > 0, "d_model must be positive"
        assert self.num_heads > 0, "num_heads must be positive"
        assert self.vocab_size > 0, "vocab_size must be positive"
        assert 0 <= self.spike_threshold <= 1, "spike_threshold must be between 0 and 1"
        assert self.quantum_qubits >= 0, "quantum_qubits must be non-negative"
        assert self.quantum_layers >= 0, "quantum_layers must be non-negative"
        if self.quantum_layers > 0:
            assert self.quantum_qubits > 0, "quantum_qubits must be > 0 when quantum_layers > 0"
        assert self.base_d_model > 0, "base_d_model must be positive"
        assert self.moe_z_loss_weight >= 0, "moe_z_loss_weight must be non-negative"
        assert self.moe_router_z_loss_weight >= 0, "moe_router_z_loss_weight must be non-negative"
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
            assert (
                self.controller_strategy in valid_strategies
            ), f"controller_strategy must be one of {valid_strategies}"

        # Validate GRPO settings
        if self.use_grpo:
            assert self.grpo_num_groups >= 1, "grpo_num_groups must be at least 1"
            assert self.grpo_group_size >= 2, "grpo_group_size must be at least 2"
            assert 0 < self.grpo_clip_eps < 1.0, "grpo_clip_eps must be between 0 and 1"
            assert self.grpo_kl_coeff >= 0, "grpo_kl_coeff must be non-negative"
            assert self.grpo_value_loss_coeff >= 0, "grpo_value_loss_coeff must be non-negative"
            valid_reward_models = ["internal", "external", "rule_based"]
            assert (
                self.grpo_reward_model in valid_reward_models
            ), f"grpo_reward_model must be one of {valid_reward_models}"

        # Validate verify/reflect settings
        if self.enable_verify_reflect:
            assert self.max_verify_steps >= 1, "max_verify_steps must be at least 1"
            assert 0 < self.verify_confidence_threshold <= 1.0, "verify_confidence_threshold must be between 0 and 1"
            assert 0 < self.reflect_temperature <= 2.0, "reflect_temperature must be between 0 and 2"

        # Validate KV cache settings
        if self.enable_kv_cache:
            assert self.kv_cache_prefix_len >= 1, "kv_cache_prefix_len must be at least 1"
            assert self.kv_cache_max_batch >= 1, "kv_cache_max_batch must be at least 1"
            valid_evictions = ["lru", "fifo", "lfu"]
            assert self.kv_cache_eviction in valid_evictions, f"kv_cache_eviction must be one of {valid_evictions}"

        # Validate self-critique settings
        if self.enable_self_critique:
            assert 0 < self.self_critique_threshold <= 1.0, "self_critique_threshold must be between 0 and 1"
            assert self.max_revisions >= 1, "max_revisions must be at least 1"
            assert self.critique_loss_coeff >= 0, "critique_loss_coeff must be non-negative"

        # Validate synthetic data settings
        if self.enable_synthetic_data:
            assert (
                0 < self.synthetic_data_difficulty_threshold <= 1.0
            ), "synthetic_data_difficulty_threshold must be between 0 and 1"
            assert (
                0 < self.synthetic_data_batch_multiplier <= 1.0
            ), "synthetic_data_batch_multiplier must be between 0 and 1"
            assert (
                0 <= self.synthetic_data_quality_improvement_min <= 1.0
            ), "synthetic_data_quality_improvement_min must be between 0 and 1"

        # Validate code routing settings
        if self.enable_code_routing:
            assert 0 < self.code_routing_threshold <= 1.0, "code_routing_threshold must be between 0 and 1"
            assert self.code_routing_boost >= 1.0, "code_routing_boost must be at least 1.0"

        # Validate think budget settings
        if self.enable_think_budget:
            assert (
                self.think_budget_max_tokens >= self.think_budget_min_tokens
            ), "think_budget_max_tokens must be >= think_budget_min_tokens"
            assert self.think_budget_min_tokens >= 1, "think_budget_min_tokens must be at least 1"

        # Validate hard negative mining
        if self.enable_hard_negative_mining:
            assert 0.0 <= self.contrastive_margin <= 1.0, "contrastive_margin must be in [0.0, 1.0]"
            assert 0.0 < self.hard_negative_ratio <= 1.0, "hard_negative_ratio must be in (0.0, 1.0]"

        # Validate multi-resolution vision
        if self.enable_multi_res_vision:
            assert len(self.vision_patch_sizes) >= 1, "vision_patch_sizes must have at least one entry"
            assert all(ps > 0 for ps in self.vision_patch_sizes), "All vision_patch_sizes must be positive"

        # Validate MLflow settings
        if self.mlflow_enabled:
            assert self.mlflow_experiment_name, "mlflow_experiment_name must be non-empty when MLflow is enabled"
            assert self.mlflow_log_interval >= 1, "mlflow_log_interval must be at least 1"

        # Validate benchmark settings
        if self.benchmark_enabled:
            valid_benchmarks = {"gpqa", "aime", "swe", "livecode", "all"}
            if isinstance(self.benchmark_names, list):
                for b in self.benchmark_names:
                    assert b in valid_benchmarks, f"Unknown benchmark: {b}. Must be one of {valid_benchmarks}"
            assert self.benchmark_eval_interval >= 1, "benchmark_eval_interval must be at least 1"
            valid_budgets = {"low", "medium", "high", "max"}
            if isinstance(self.benchmark_think_budget, str):
                assert (
                    self.benchmark_think_budget in valid_budgets
                ), f"benchmark_think_budget must be one of {valid_budgets}"

    def to_dict(self):
        """Convert config to dictionary"""
        skip = {"architecture", "training", "precision", "parallelism", "multimodal", "safety"}
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_") and k not in skip}

    def update(self, **kwargs):
        """Update configuration with new values"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: Unknown configuration parameter: {key}")
        self._validate_config()
        self._compose_subconfigs()

    def save(self, filepath):
        """Save configuration to file"""
        import json

        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath):
        """Load configuration from file"""
        import json

        with open(filepath, "r") as f:
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
            available = ", ".join(MODEL_PRESETS.keys())
            raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")

        preset = MODEL_PRESETS[preset_name].copy()
        preset.pop("description", None)  # Remove description from config params
        preset.update(overrides)

        return cls(**preset)

    @staticmethod
    def list_presets():
        """List available model presets with descriptions."""
        print("Available Model Presets:")
        print("-" * 60)
        for name, preset in MODEL_PRESETS.items():
            desc = preset.get("description", "No description")
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

        if self.quantum_layers > 0 and self.quantum_qubits > 0:
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

        print("\nGRPO (Group Relative Policy Optimization):")
        print(f"  - Enabled: {self.use_grpo}")
        if self.use_grpo:
            print(f"    - Num groups: {self.grpo_num_groups}")
            print(f"    - Group size: {self.grpo_group_size}")
            print(f"    - Clip epsilon: {self.grpo_clip_eps}")
            print(f"    - KL coefficient: {self.grpo_kl_coeff}")
            print(f"    - Reward model: {self.grpo_reward_model}")

        print("\nVerify/Reflect Loop:")
        print(f"  - Enabled: {self.enable_verify_reflect}")
        if self.enable_verify_reflect:
            print(f"    - Max verify steps: {self.max_verify_steps}")
            print(f"    - Confidence threshold: {self.verify_confidence_threshold}")
            print(f"    - Reflect temperature: {self.reflect_temperature}")

        print("\nKV Prefix Cache:")
        print(f"  - Enabled: {self.enable_kv_cache}")
        if self.enable_kv_cache:
            print(f"    - Prefix length: {self.kv_cache_prefix_len}")
            print(f"    - Max batch: {self.kv_cache_max_batch}")
            print(f"    - Eviction: {self.kv_cache_eviction}")

        print("\nSelf-Critique:")
        print(f"  - Enabled: {self.enable_self_critique}")
        if self.enable_self_critique:
            print(f"    - Quality threshold: {self.self_critique_threshold}")
            print(f"    - Max revisions: {self.max_revisions}")

        print("\nSynthetic Data Self-Improvement:")
        print(f"  - Enabled: {self.enable_synthetic_data}")
        if self.enable_synthetic_data:
            print(f"    - Difficulty threshold: {self.synthetic_data_difficulty_threshold}")
            print(f"    - Batch multiplier: {self.synthetic_data_batch_multiplier}")
            print(f"    - Quality improvement min: {self.synthetic_data_quality_improvement_min}")
            print(f"    - Output dir: {self.synthetic_data_output_dir}")

        print("\nCode Modality Routing:")
        print(f"  - Enabled: {self.enable_code_routing}")
        if self.enable_code_routing:
            print(f"    - Code confidence threshold: {self.code_routing_threshold}")
            print(f"    - Routing boost: {self.code_routing_boost}x")

        print("\nThink Budget:")
        print(f"  - Enabled: {self.enable_think_budget}")
        if self.enable_think_budget:
            print(f"    - Token range: {self.think_budget_min_tokens}-{self.think_budget_max_tokens}")
            print(f"    - Difficulty scaling: {self.think_budget_difficulty_scale}")

        print("\nHard Negative Mining (Contrastive Loss):")
        print(f"  - Enabled: {self.enable_hard_negative_mining}")
        if self.enable_hard_negative_mining:
            print(f"    - Contrastive margin: {self.contrastive_margin}")
            print(f"    - Hard negative ratio: {self.hard_negative_ratio}")

        print("\nMulti-Resolution Vision:")
        print(f"  - Enabled: {self.enable_multi_res_vision}")
        if self.enable_multi_res_vision:
            print(f"    - Patch sizes: {self.vision_patch_sizes}")

        print("\nMLflow Tracking:")
        print(f"  - Enabled: {self.mlflow_enabled}")
        if self.mlflow_enabled:
            print(f"    - Tracking URI: {self.mlflow_tracking_uri or 'local (./mlruns)'}")
            print(f"    - Experiment: {self.mlflow_experiment_name}")
            print(f"    - Log interval: {self.mlflow_log_interval}")

        print("\nBenchmark Evaluation:")
        print(f"  - Enabled: {self.benchmark_enabled}")
        if self.benchmark_enabled:
            print(f"    - Benchmarks: {', '.join(self.benchmark_names)}")
            print(f"    - Think budget: {self.benchmark_think_budget}")
            print(f"    - Max samples: {self.benchmark_max_samples or 'all'}")
            print(f"    - Eval interval: every {self.benchmark_eval_interval} epoch(s)")
        print("=" * 60)
