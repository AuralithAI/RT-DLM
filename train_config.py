import jax.numpy as jnp

class TrainConfig:
    """
    Configuration class for training the TMS model, with support for dynamic hyperparameter updates
    from hyperparameter tuning (e.g., via Optuna in hyper_param_tune).
    """

    def __init__(self, **kwargs):
        # --- Model Architecture Parameters ---
        self.vocab_size = kwargs.get("vocab_size", 8000)  # Vocabulary size for token embeddings
        self.d_model = kwargs.get("d_model", 384)  # Embedding dimension (model width)
        self.num_heads = kwargs.get("num_heads", 8)  # Number of attention heads
        self.num_layers = kwargs.get("num_layers", 12)  # Number of transformer layers
        self.moe_experts = kwargs.get("moe_experts", 8)  # Number of experts in Mixture of Experts
        self.moe_top_k = kwargs.get("moe_top_k", 2)  # Top-k experts to select in MoE
        self.task_size = kwargs.get("task_size", 15)  # Task size for support and query sets (batches)

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
        self.max_seq_length = kwargs.get("max_seq_length", 64)  # Maximum sequence length for input
        self.pad_token_id = kwargs.get("pad_token_id", 0)  # Token ID used for padding
        self.max_sentence_length = kwargs.get("max_sentence_length", 5192)  # Maximum allowed sentence length
        self.input_sentence_size = kwargs.get("input_sentence_size", 500000)  # Total number of sentences in input data
        self.character_coverage = kwargs.get("character_coverage", 0.9999)  # Character coverage for tokenizer
        self.num_threads = kwargs.get("num_threads", 16)  # Number of threads for data processing

        # --- Memory Bank Parameters ---
        self.memory_size = kwargs.get("memory_size", 5000)  # Size of the long-term memory bank
        self.retrieval_k = kwargs.get("retrieval_k", 3)  # Number of top-k items to retrieve from LTM
        self.stm_buffer_size = kwargs.get("stm_buffer_size", self.batch_size)  # Default STM buffer size (tunable)
        self.ltm_weight = kwargs.get("ltm_weight", 0.5)  # Weight for long-term memory contribution
        self.stm_weight = kwargs.get("stm_weight", 0.5)  # Weight for short-term memory contribution
        self.mtm_weight = kwargs.get("mtm_weight", 0.5)  # Weight for mid-term memory contribution
        self.mtm_buffer_size = kwargs.get("mtm_buffer_size", 1000)  # Size of the mid-term memory buffer
        self.retention_steps = kwargs.get("retention_steps", 100)  # Number of steps to retain in mid-term memory

        # --- Numerical Stability ---
        self.EPSILON = kwargs.get("EPSILON", 1e-8)  # Small constant to avoid division by zero

        # --- XLA/GPU Configuration ---
        self.xla_gpu_parallelism = kwargs.get("xla_gpu_parallelism", 10)  # Number of parallel GPU compilations (tuned for GH200)

        # --- Data Type Enforcement ---
        self.input_dtype = jnp.int32  # Default dtype for inputs (e.g., token IDs)
        self.embedding_dtype = jnp.float32  # Default dtype for embeddings and activations

        # Validate and adjust parameters for compatibility
        self._validate_config()

    def _validate_config(self):
        """
        Validate configuration parameters for compatibility and consistency.
        """
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if self.d_model <= 0 or self.d_model % self.num_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be positive and divisible by num_heads ({self.num_heads})")
        if self.num_heads <= 0 or self.num_heads > self.d_model:
            raise ValueError(f"num_heads ({self.num_heads}) must be positive and not exceed d_model ({self.d_model})")
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}")
        if self.moe_experts <= 0 or self.moe_top_k <= 0 or self.moe_top_k > self.moe_experts:
            raise ValueError(f"moe_experts ({self.moe_experts}) and moe_top_k ({self.moe_top_k}) must be positive, and moe_top_k must not exceed moe_experts")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.learning_rate <= 0 or self.inner_learning_rate <= 0:
            raise ValueError(f"learning_rate ({self.learning_rate}) and inner_learning_rate ({self.inner_learning_rate}) must be positive")
        if self.max_seq_length <= 0:
            raise ValueError(f"max_seq_length must be positive, got {self.max_seq_length}")
        if self.memory_size <= 0 or self.retrieval_k <= 0 or self.retrieval_k > self.memory_size:
            raise ValueError(f"memory_size ({self.memory_size}) and retrieval_k ({self.retrieval_k}) must be positive, and retrieval_k must not exceed memory_size")
        if self.stm_buffer_size <= 0 or self.mtm_buffer_size <= 0:
            raise ValueError(f"stm_buffer_size ({self.stm_buffer_size}) and mtm_buffer_size ({self.mtm_buffer_size}) must be positive")
        if self.retention_steps <= 0:
            raise ValueError(f"retention_steps must be positive, got {self.retention_steps}")
        if not (0 <= self.ltm_weight <= 1 and 0 <= self.stm_weight <= 1 and 0 <= self.mtm_weight <= 1):
            raise ValueError("Memory weights (ltm_weight, stm_weight, mtm_weight) must be between 0 and 1")
        if self.xla_gpu_parallelism <= 0:
            raise ValueError(f"xla_gpu_parallelism must be positive, got {self.xla_gpu_parallelism}")
