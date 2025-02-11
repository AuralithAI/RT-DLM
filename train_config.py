class TrainConfig:
    def __init__(self):
        self.vocab_size = 4000
        self.d_model = 64
        self.num_heads = 2
        self.num_layers = 2
        self.moe_experts = 8
        self.moe_top_k = 2
        self.max_seq_length = 64
        self.batch_size = 4
        self.learning_rate = 3e-4
        self.num_epochs = 3
        self.eval_interval = 50