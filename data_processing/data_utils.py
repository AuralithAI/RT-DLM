import os
import re
import numpy as np
import sentencepiece as spm
import jax.numpy as jnp
from typing import List
from train_config import TrainConfig

config = TrainConfig()

class DataProcessor:
    def __init__(self, vocab_size: int = config.vocab_size, model_prefix: str = "data/rt_dlm_sp"):
        self.vocab_size = vocab_size
        self.model_prefix = model_prefix
        self.sp = spm.SentencePieceProcessor()

        # Load trained SentencePiece model if exists
        if os.path.exists(f"{self.model_prefix}.model"):
            self.sp.load(f"{self.model_prefix}.model")
        else:
            print(f"[WARNING] SentencePiece model not found! Train first using `train_tokenizer`.")

    def preprocess_text(self, text: str) -> str:
        """Cleans and normalizes text."""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def train_tokenizer(self, input_file: str):
        """Train a SentencePiece tokenizer using Unigram model."""
        spm.SentencePieceTrainer.train(
            input=input_file,
            model_prefix=self.model_prefix,
            vocab_size=self.vocab_size,
            model_type="unigram",  # Change to 'bpe' if needed
            pad_id=0, unk_id=1, bos_id=2, eos_id=3,  
            max_sentence_length=config.max_sentence_length,
            input_sentence_size=config.input_sentence_size,
            character_coverage=config.character_coverage,
            num_threads=config.num_threads
        )
        self.sp.load(f"{self.model_prefix}.model")
        print(f"[INFO] Tokenizer trained and saved as {self.model_prefix}.model")

    def tokenize(self, text: str) -> List[int]:
        """Tokenizes text into subword token IDs."""
        return self.sp.encode(text, out_type=int)

    def decode_tokens(self, token_ids: List[int]) -> str:
        """Decodes a list of token IDs back into text."""
        return self.sp.decode(token_ids)

    def pad_sequence(self, tokens: List[int], max_length: int) -> List[int]:
        """Pads or truncates a sequence to max_length."""
        tokens = tokens[:max_length]  # Truncate if too long
        tokens += [config.pad_token_id] * (max_length - len(tokens))  # Pad if too short
        return tokens  

def load_data(file_path: str) -> List[str]:
    """Loads dataset from file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found.")
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]
    
def create_batches(inputs: jnp.ndarray, targets: jnp.ndarray, batch_size: int, shuffle: bool = True):
    """Yield batches of input and target data."""
    n_samples = inputs.shape[0]
    indices = np.arange(n_samples)
    if shuffle:
        np.random.shuffle(indices)
    for start_idx in range(0, n_samples, batch_size):
        batch_indices = indices[start_idx:start_idx + batch_size]
        yield inputs[batch_indices], targets[batch_indices]