class TrainConfig:
    def __init__(self):
        self.vocab_size = 8000
        self.d_model = 384
        self.num_heads = 8
        self.num_layers = 12
        self.moe_experts = 8
        self.moe_top_k = 2
        self.max_seq_length = 64
        self.batch_size = 128
        self.learning_rate = 1e-4
        self.num_epochs = 200
        self.eval_interval = 25
        self.temperature = 1.2
        self.pad_token_id = 0
        self.label_smoothing = 0.1
        self.max_sentence_length = 5192
        self.input_sentence_size = 500000
        self.character_coverage=0.9999
        self.num_threads=16