class TrainConfig:
    def __init__(self):
        self.vocab_size = 6145
        self.d_model = 128
        self.num_heads = 4
        self.num_layers = 4
        self.moe_experts = 8
        self.moe_top_k = 2
        self.max_seq_length = 64
        self.batch_size = 64
        self.learning_rate = 2e-4
        self.num_epochs = 50
        self.eval_interval = 50
        self.temperature = 2.5
        self.pad_token_id = 0