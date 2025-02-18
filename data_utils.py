import os
import re
import json
import jax.numpy as jnp
from typing import List
from train_config import TrainConfig

config = TrainConfig()
class DataProcessor:
    def __init__(self, vocab_size: int = 6145, vocab_file: str = "data/vocab.json"):
        self.vocab_size = vocab_size
        self.vocab_file = vocab_file
        self.vocab = {}
        self.inverse_vocab = {}
    
    def preprocess_text(self, text: str) -> str:
        """Cleans and normalizes text."""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def tokenize(self, text: str) -> List[str]:
        """Splits text into tokens (whitespace-based)."""
        return text.split()

    def decode_tokens(self, token_ids: List[int]) -> str:
        """ Converts a list of token IDs back into a string. """
        words = [self.inverse_vocab.get(token_id, '<UNK>') for token_id in token_ids]
        return ' '.join(words).replace(' <PAD>', '')

    def build_vocab(self, texts: List[str]) -> None:
        """Builds vocabulary from training dataset and saves it."""
        word_freq = {}

        for text in texts:
            tokens = self.tokenize(self.preprocess_text(text))
            for token in tokens:
                word_freq[token] = word_freq.get(token, 0) + 1

        # Sort vocabulary by frequency (descending)
        sorted_vocab = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

        # Ensure special tokens are added first
        self.vocab = {
            '<PAD>': 0,  # Padding Token
            '<UNK>': 1,  # Unknown Token
            '<BOS>': 2,  # Beginning of Sequence
            '<EOS>': 3,  # End of Sequence
            '<SEP>': 4,  # Separator (for sentence pairs)
            '<CLS>': 5,  # Classifier token (useful for classification tasks)
            '<MASK>': 6  # Masking Token (for MLM / masked language modeling)
        }


        # Add top frequent words while keeping vocab within limit
        for idx, (word, _) in enumerate(sorted_vocab[: self.vocab_size - len(self.vocab)]):
            self.vocab[word] = idx + len(self.vocab)

        # Save vocabulary
        self.save_vocab()
        print(f"[INFO] Vocabulary built and saved to {self.vocab_file}.")

    def save_vocab(self):
        """Saves vocabulary to a JSON file."""
        os.makedirs(os.path.dirname(self.vocab_file), exist_ok=True)
        with open(self.vocab_file, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, indent=4)
        print(f"[INFO] Vocabulary saved to {self.vocab_file}")

    def load_vocab(self):
        """Loads vocabulary from a JSON file and creates inverse mapping."""
        if not os.path.exists(self.vocab_file):
            raise FileNotFoundError(f"[ERROR] Vocabulary file '{self.vocab_file}' not found! Train model first.")

        with open(self.vocab_file, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)

        # Create inverse mapping for decoding
        self.inverse_vocab = {idx: word for word, idx in self.vocab.items()}
        print(f"[INFO] Loaded vocabulary from {self.vocab_file} (size: {len(self.vocab)})")

    def convert_text_to_tokens(self, text: str) -> List[int]:
        """Converts text into a list of token IDs, replacing OOV words with <UNK>."""
        tokens = self.tokenize(self.preprocess_text(text))
        token_ids = [self.vocab.get(word, self.vocab['<UNK>']) for word in tokens]

        if max(token_ids, default=0) >= self.vocab_size:
            print(f"[WARNING] Some token IDs exceed vocab size! Adjusting...")
            token_ids = [tid if tid < self.vocab_size else self.vocab['<UNK>'] for tid in token_ids]

        return token_ids
    
    def pad_sequence(self, tokens: List[int], max_length: int) -> List[int]:
        """Pads or truncates a sequence to max_length."""
        tokens = tokens[:max_length]  # Truncate if too long
        tokens += [self.vocab['<PAD>']] * (max_length - len(tokens))  # Pad if too short
        return tokens  

def load_data(file_path: str) -> List[str]:
    """Loads dataset from file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found.")
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def preprocess_batch(batch, processor, max_seq_length):
    """Converts batch of text into tokenized, padded input-target tensors."""
    inputs, targets = [], []

    for text in batch:
        tokens = processor.convert_text_to_tokens(text)
        if len(tokens) == 0:
            tokens = [processor.vocab['<UNK>']]  

        padded_tokens = processor.pad_sequence(tokens, max_seq_length)
        inputs.append(padded_tokens)
        targets.append(padded_tokens)

    inputs = jnp.array(inputs, dtype=jnp.int32)
    targets = jnp.array(targets, dtype=jnp.int32)

    return inputs, targets

