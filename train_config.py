class TrainConfig:
    def __init__(self):
        # --- Model Architecture Parameters ---
        self.vocab_size = 8000          # Vocabulary size for token embeddings
        self.d_model = 384              # Embedding dimension (model width)
        self.num_heads = 8              # Number of attention heads
        self.num_layers = 12            # Number of transformer layers
        self.moe_experts = 8            # Number of experts in Mixture of Experts
        self.moe_top_k = 2              # Top-k experts to select in MoE

        # --- Training Hyperparameters ---
        self.batch_size = 128           # Batch size for training
        self.learning_rate = 1e-4       # Initial learning rate
        self.num_epochs = 3             # Number of training epochs
        self.eval_interval = 25         # Frequency of evaluation during training
        self.temperature = 1.2          # Temperature for sampling (if applicable)
        self.label_smoothing = 0.1      # Label smoothing factor for loss

        # --- Data Processing Parameters ---
        self.max_seq_length = 64        # Maximum sequence length for input
        self.pad_token_id = 0           # Token ID used for padding
        self.max_sentence_length = 5192 # Maximum allowed sentence length
        self.input_sentence_size = 500000  # Total number of sentences in input data
        self.character_coverage = 0.9999   # Character coverage for tokenizer
        self.num_threads = 16           # Number of threads for data processing

        # --- Memory Bank Parameters ---
        self.memory_size = 5000         # Size of the long-term memory bank
        self.retrieval_k = 3            # Number of top-k items to retrieve from LTM
        self.buffer_size = self.batch_size  # Default STM buffer size (tunable)
        self.ltm_weight = 0.5           # Weight for long-term memory contribution
        self.stm_weight = 0.5           # Weight for short-term memory contribution

        # --- Numerical Stability ---
        self.EPSILON = 1e-8             # Small constant to avoid division by zero

        # --- XLA/GPU Configuration ---
        self.xla_gpu_parallelism = 8    # Number of parallel GPU compilations (tuned for GH200)