import os,re
import jax.numpy as jnp
from typing import List, Dict, Tuple, Optional
import requests

"""
DataProcessor class for text preprocessing and tokenization.
"""

class DataProcessor:
    """
        Constructor for DataProcessor class.
    """
    def __init__(self, vocab: Optional[Dict[str, int]] = None):
        self.vocab = vocab or {}

    def preprocess_text(self, text: str) -> str:
        """
        Remove special characters and extra whitespaces.
        """
        #print(f"[DEBUG] Preprocessing text: {text}")
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  
        text = re.sub(r'\s+', ' ', text).strip()   
        #print(f"[DEBUG] Preprocessed text: {text}") 
        return text

    def tokenize(self, text: str) -> List[str]:
        tokens = text.split()
        #print(f"[DEBUG] Tokenized words: {tokens}")
        return tokens

    def build_vocab(self, texts: List[str]) -> None:
        word_set = set()
        for text in texts:
            tokens = self.tokenize(self.preprocess_text(text))
            word_set.update(tokens)

        if len(word_set) < 2000:
            print(f"[WARNING] Vocabulary is too small ({len(word_set)} words)! Consider adding more data.")

        self.vocab = {word: idx for idx, word in enumerate(sorted(word_set), start=2)}
        self.vocab['<PAD>'] = 0  
        self.vocab['<UNK>'] = 1 
        #print(f"[DEBUG] Vocabulary Size: {len(self.vocab)}")
        #print(f"[DEBUG] Sample Vocabulary Entries: {list(self.vocab.items())[:20]}")

    def convert_text_to_tokens(self, text: str) -> List[int]:
        if not self.vocab:
            raise ValueError("Vocabulary is not initialized. Call `build_vocab` first.")
        tokens = self.tokenize(self.preprocess_text(text))
        token_ids = [self.vocab.get(word, self.vocab['<UNK>']) for word in tokens]
        #print(f"[DEBUG] Tokenized Text: {tokens}")
        #print(f"[DEBUG] Token IDs: {token_ids}")
        return token_ids

    def pad_sequence(self, tokens: List[int], max_length: int) -> List[int]:
        original_length = len(tokens)
        if len(tokens) < max_length:
            tokens += [self.vocab['<PAD>']] * (max_length - len(tokens))

        #print(f"[DEBUG] Original Length: {original_length}, Padded Length: {len(tokens)}")
        return tokens[:max_length]


def load_data(file_path: str) -> List[str]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found.")
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f]
    if not lines:
        raise ValueError(f"No data found in {file_path}.")
    return lines


def fetch_wikipedia_data():
    """Fetches Wikipedia data for training."""
    print("[INFO] Fetching Wikipedia dataset...")
    url = "https://raw.githubusercontent.com/attardi/wikiextractor/master/test/wiki_00"
    
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("Failed to fetch Wikipedia data")

    raw_text = response.text
    articles = raw_text.split("\n\n")[:500]  # Limit to 500 articles for now
    return articles

def fetch_commoncrawl_data():
    """Fetches sample data from Common Crawl."""
    print("[INFO] Fetching Common Crawl dataset...")
    url = "https://data.commoncrawl.org/crawl-data/CC-MAIN-2023-10/wet.paths.gz"

    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("Failed to fetch Common Crawl data")

    raw_text = response.text.split("\n")[:500]  # Limit to 500 samples
    return raw_text

def save_dataset(data, file_path="data/dataset.txt"):
    """Saves dataset to file."""
    with open(file_path, "w", encoding="utf-8") as f:
        for line in data:
            f.write(line + "\n")
    print(f"[INFO] Dataset saved to {file_path}")

def load_data(file_path: str) -> List[str]:
    """Loads text data from file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found.")
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def preprocess_batch(batch, processor, max_seq_length):
    """
    Preprocess a batch of text data into tokenized and padded input and target tensors.
    Args:
        batch (List[str]): List of text strings.
        processor (DataProcessor): DataProcessor instance with an initialized vocabulary.
        max_seq_length (int): Maximum sequence length for padding.
    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: Tokenized and padded input and target tensors.
    """
    inputs, targets = [], []
    for text in batch:
        tokens = processor.convert_text_to_tokens(text)
        padded = processor.pad_sequence(tokens, max_seq_length)
        inputs.append(padded)
        targets.append(padded)

    inputs = jnp.array(inputs, dtype=jnp.int32)
    targets = jnp.array(targets, dtype=jnp.int32)

    if len(inputs.shape) == 1:
        inputs = inputs.reshape(1, -1)

    if len(targets.shape) == 1:
        targets = targets.reshape(1, -1)

    if inputs.shape[1] != max_seq_length or targets.shape[1] != max_seq_length:
        pad_width = max_seq_length - inputs.shape[1]
        inputs = jnp.pad(inputs, ((0, 0), (0, pad_width)), constant_values=0)
        targets = jnp.pad(targets, ((0, 0), (0, pad_width)), constant_values=0)

    return inputs, targets