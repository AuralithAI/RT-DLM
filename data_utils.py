import os,re
import jax.numpy as jnp
from typing import List, Dict, Tuple, Optional
import requests
import json
import gzip

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


def fetch_wikipedia_data(num_articles=500):
    """Fetches Wikipedia articles dynamically from Wikipedia API."""
    print("[INFO] Fetching Wikipedia dataset...")

    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "list": "random",
        "rnlimit": num_articles,
        "rnnamespace": 0 
    }

    response = requests.get(url, params=params)
    
    if response.status_code != 200:
        raise Exception("Failed to fetch Wikipedia data")

    data = response.json()
    articles = []

    for page in data["query"]["random"]:
        title = page["title"]

        # Fetch content for each article
        content_url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + title.replace(" ", "_")
        content_response = requests.get(content_url)

        if content_response.status_code == 200:
            content_data = content_response.json()
            if "extract" in content_data:
                articles.append(content_data["extract"])

    print(f"[INFO] Fetched {len(articles)} Wikipedia articles.")
    return articles

def fetch_commoncrawl_data(num_samples=500):
    """Fetches raw text data from Common Crawl."""
    print("[INFO] Fetching Common Crawl dataset paths...")

    # URL for Common Crawl WET file index
    url = "https://data.commoncrawl.org/crawl-data/CC-MAIN-2023-40/wet.paths.gz"

    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("Failed to fetch Common Crawl dataset paths.")

    wet_paths = response.text.strip().split("\n")
    if not wet_paths or len(wet_paths[0]) < 10:
        raise Exception("Invalid WET file paths received.")

    wet_file_path = wet_paths[0].strip()  # First available WET file path
    wet_file_url = f"https://data.commoncrawl.org/{wet_file_path}"

    print(f"[INFO] Fetching Common Crawl text data from: {wet_file_url}")

    wet_response = requests.get(wet_file_url, stream=True)
    if wet_response.status_code != 200:
        raise Exception("Failed to fetch Common Crawl text data.")

    # Read and extract plain text from the WET file
    raw_texts = []
    with gzip.open(wet_response.raw, "rt", encoding="utf-8") as f:
        for line in f:
            # Skip metadata lines (WARC headers) and empty lines
            if line.strip() and not line.startswith(("WARC", "Content-Length", "Content-Type", "WARC-Target-URI")):
                raw_texts.append(line.strip())

            if len(raw_texts) >= num_samples:
                break

    print(f"[INFO] Successfully fetched {len(raw_texts)} text samples from Common Crawl.")
    return raw_texts

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