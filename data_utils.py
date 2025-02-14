import os
import re
import jax.numpy as jnp
from typing import List, Dict, Tuple, Optional

"""
    DataProcessor class is used to preprocess and tokenize the text data.
"""
class DataProcessor:
    def __init__(self, vocab_size: int = 6145):
        self.vocab = {}
        self.vocab_size = vocab_size  

    def preprocess_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def tokenize(self, text: str) -> List[str]:
        return text.split()

    def build_vocab(self, texts: List[str]) -> None:
        word_freq = {}

        for text in texts:
            tokens = self.tokenize(self.preprocess_text(text))
            for token in tokens:
                word_freq[token] = word_freq.get(token, 0) + 1

        sorted_vocab = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

        limited_vocab = sorted_vocab[: self.vocab_size - 2]
        self.vocab = {word: idx + 2 for idx, (word, _) in enumerate(limited_vocab)}

        self.vocab['<PAD>'] = 0
        self.vocab['<UNK>'] = 1

    def convert_text_to_tokens(self, text: str) -> List[int]:
        tokens = self.tokenize(self.preprocess_text(text))
        
        token_ids = []
        for word in tokens:
            token_id = self.vocab.get(word, self.vocab['<UNK>'])
            
            if token_id < 0 or token_id >= self.vocab_size:
                print(f"[ERROR] Invalid token detected: '{word}' -> {token_id}")
                token_id = self.vocab['<UNK>']  # Fallback to <UNK>

            token_ids.append(int(token_id))

        return token_ids


    def pad_sequence(self, tokens: List[int], max_length: int) -> List[int]:
        tokens = tokens[:max_length]  # Truncate if too long
        tokens += [self.vocab['<PAD>']] * (max_length - len(tokens))  # Pad if too short

        if any(t < 0 or t >= self.vocab_size for t in tokens):
            print(f"[ERROR] Corrupted padded sequence: {tokens}")

        return [int(t) for t in tokens]  


def load_data(file_path: str) -> List[str]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found.")
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def preprocess_batch(batch, processor, max_seq_length):
    """
    Converts a batch of text into tokenized, padded input-target tensors.
    """
    inputs, targets = [], []

    for text in batch:
        tokens = processor.convert_text_to_tokens(text)

        if len(tokens) == 0:
            tokens = [processor.vocab['<UNK>']] 

        padded_tokens = processor.pad_sequence(tokens, max_seq_length)

        if any(t < 0 or t >= processor.vocab_size for t in padded_tokens):
            print(f"[ERROR] preprocess_batch(): Invalid token IDs detected -> {padded_tokens}")
            raise ValueError("[FATAL ERROR] Token ID out of range in preprocess_batch!")

        inputs.append(padded_tokens)
        targets.append(padded_tokens)

    inputs = jnp.array(inputs, dtype=jnp.int32)
    targets = jnp.array(targets, dtype=jnp.int32)

    return inputs, targets

