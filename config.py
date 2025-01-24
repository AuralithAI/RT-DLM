####################################################################################
#                   RT - DLM Model Configuration File
####################################################################################

class RTDLMConfig:

    # Constructor for RTDLMConfig:
    # Parameters:
    #   vocab_size: int - Vocabulary size
    #   d_model: int - Embedding dimension
    #   max_seq_length: int - Maximum sequence length
    def __init__(self, vocab_size: int, d_model: int, max_seq_length: int):
        self.vocab_size = vocab_size  
        self.d_model = d_model  
        self.max_seq_length = max_seq_length