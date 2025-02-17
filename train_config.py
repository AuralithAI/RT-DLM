class TrainConfig:
    def __init__(self):
        self.vocab_size = 6145
        self.d_model = 256
        self.num_heads = 8
        self.num_layers = 12
        self.moe_experts = 8
        self.moe_top_k = 2
        self.max_seq_length = 64
        self.batch_size = 32
        self.learning_rate = 2e-4
        self.num_epochs = 150
        self.eval_interval = 50
        self.temperature = 1.5
        self.pad_token_id = 0
        self.label_smoothing = 0.1