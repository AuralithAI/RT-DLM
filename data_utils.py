import os
import sys
import random
import re
import numpy as np
import sentencepiece as spm
import jax.numpy as jnp
import librosa
import cv2
import logging
from typing import List
from train_config import TrainConfig

config = TrainConfig()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('train.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, vocab_size: int = config.vocab_size, model_prefix: str = "data/rt_dlm_sp"):
        self.vocab_size = vocab_size
        self.model_prefix = model_prefix
        self.sp = spm.SentencePieceProcessor()

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
            model_type="unigram",
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
        tokens = tokens[:max_length]
        tokens += [config.pad_token_id] * (max_length - len(tokens))
        return tokens

class MultimodalData:
    def __init__(self, inputs, modality_types, targets=None, target_modality="text"):
        assert len(inputs) == len(modality_types), "Inputs and modality_types must have the same length"
        self.inputs = inputs
        self.modality_types = modality_types
        self.targets = targets
        self.target_modality = target_modality

    def validate(self):
        valid_modalities = {"text", "audio", "image", "video"}
        for m_type in self.modality_types:
            if m_type not in valid_modalities:
                raise ValueError(f"Invalid modality type: {m_type}, must be one of {valid_modalities}")
        if self.target_modality not in valid_modalities:
            raise ValueError(f"Invalid target modality: {self.target_modality}, must be one of {valid_modalities}")

def load_text_data(file_path: str, config=None) -> jnp.ndarray:
    """Loads and tokenizes text data."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found.")
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_texts = [line.strip() for line in f if line.strip()]
    processor = DataProcessor(vocab_size=config.vocab_size if config else 8000)
    tokenized_texts = [processor.tokenize(text) for text in raw_texts]
    return jnp.array([processor.pad_sequence(tokens, config.max_seq_length if config else 64) for tokens in tokenized_texts], dtype=jnp.int32)

def load_audio_data(directory: str, config=None) -> jnp.ndarray:
    """Loads and preprocesses audio data from WAV and MP3 files."""
    if not os.path.exists(directory):
        return None
    audio_files = [f for f in os.listdir(directory) if f.lower().endswith(('.wav', '.mp3'))]
    if not audio_files:
        return None
    audio_data = []
    sample_rate = config.audio_sample_rate if config else 16000
    max_length = config.max_audio_length if config else 16000
    for f in audio_files:
        audio, sr = librosa.load(os.path.join(directory, f), sr=sample_rate)
        if len(audio) > max_length:
            audio = audio[:max_length]
        elif len(audio) < max_length:
            audio = np.pad(audio, (0, max_length - len(audio)))
        audio_data.append(audio)
    return jnp.array(audio_data, dtype=jnp.float32)[:, :, None]

def load_image_data(directory: str, config=None) -> jnp.ndarray:
    """Loads and preprocesses image data from JPG/PNG files."""
    if not os.path.exists(directory):
        return None
    image_files = [f for f in os.listdir(directory) if f.endswith(('.jpg', '.png'))]
    image_data = []
    image_size = config.image_size if config else 64
    for f in image_files:
        img = cv2.imread(os.path.join(directory, f))
        img = cv2.resize(img, (image_size, image_size))
        image_data.append(img)
    return jnp.array(image_data, dtype=jnp.float32) / 255.0

def load_video_data(directory: str, config=None) -> jnp.ndarray:
    """Loads and preprocesses video data from MP4 files."""
    if not os.path.exists(directory):
        return None
    video_files = [f for f in os.listdir(directory) if f.endswith('.mp4')]
    video_data = []
    image_size = config.image_size if config else 64
    max_frames = config.max_video_frames if config else 300
    for f in video_files:
        cap = cv2.VideoCapture(os.path.join(directory, f))
        frames = []
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (image_size, image_size))
            frames.append(frame)
        cap.release()
        if len(frames) < max_frames:
            frames.extend([np.zeros((image_size, image_size, 3))] * (max_frames - len(frames)))
        video_data.append(frames[:max_frames])
    return jnp.array(video_data, dtype=jnp.float32) / 255.0

def load_multimodal_data(data_dir: str, config=None) -> List[MultimodalData]:
    """Loads all modality data from the specified directory."""
    datasets = []
    text_data = load_text_data(os.path.join(data_dir, "train_data.txt"), config)
    if text_data is not None:
        datasets.append(MultimodalData(inputs=[text_data], modality_types=["text"], targets=text_data, target_modality="text"))

    audio_data = load_audio_data(os.path.join(data_dir, "audio"), config)
    if audio_data is not None:
        datasets.append(MultimodalData(inputs=[audio_data], modality_types=["audio"], targets=audio_data, target_modality="audio"))

    image_data = load_image_data(os.path.join(data_dir, "images"), config)
    if image_data is not None:
        datasets.append(MultimodalData(inputs=[image_data], modality_types=["image"], targets=image_data, target_modality="image"))

    video_data = load_video_data(os.path.join(data_dir, "videos"), config)
    if video_data is not None:
        datasets.append(MultimodalData(inputs=[video_data], modality_types=["video"], targets=video_data, target_modality="video"))

    return datasets

def create_batches(inputs: jnp.ndarray, targets: jnp.ndarray, batch_size: int, shuffle: bool = True):
    """Yield batches of input and target data (for text only)."""
    n_samples = inputs.shape[0]
    indices = np.arange(n_samples)
    if shuffle:
        np.random.shuffle(indices)
    for start_idx in range(0, n_samples, batch_size):
        batch_indices = indices[start_idx:start_idx + batch_size]
        yield inputs[batch_indices], targets[batch_indices]

def create_multimodal_batches(multimodal_datasets: List[MultimodalData], batch_size: int, shuffle: bool = True):
    """Yield batches with combined multimodal inputs, one at a time."""
    modality_map = {d.target_modality: d for d in multimodal_datasets if d.inputs[0].shape[0] > 0}
    max_samples = max(d.inputs[0].shape[0] for d in multimodal_datasets)
    modalities = list(modality_map.keys())
    
    logger.info(f"Creating batches with max_samples={max_samples}, batch_size={batch_size}")
    for modality in modalities:
        logger.info(f"Modality {modality}: {modality_map[modality].inputs[0].shape[0]} samples")
    
    indices = np.arange(max_samples)
    if shuffle:
        np.random.shuffle(indices)
    
    for i in range(0, max_samples, batch_size):
        batch_inputs = []
        batch_modality_types = []
        for modality in modalities:
            num_samples = modality_map[modality].inputs[0].shape[0]
            modality_indices = np.arange(num_samples)
            if shuffle:
                np.random.shuffle(modality_indices)
            batch_start = i % num_samples
            batch_end = (i + batch_size) % num_samples if (i + batch_size) > num_samples else i + batch_size
            if batch_end <= batch_start:
                batch_indices = np.concatenate([modality_indices[batch_start:], modality_indices[:batch_end]])
            else:
                batch_indices = modality_indices[batch_start:batch_end]
            if len(batch_indices) < batch_size:
                batch_indices = np.tile(batch_indices, (batch_size // len(batch_indices) + 1))[:batch_size]
            batch_inputs.append(modality_map[modality].inputs[0][batch_indices])
            batch_modality_types.append(modality)
        
        # Randomly choose output modality
        output_modality = random.choice(modalities)
        num_target_samples = modality_map[output_modality].targets.shape[0]
        target_start = i % num_target_samples
        target_end = (i + batch_size) % num_target_samples if (i + batch_size) > num_target_samples else i + batch_size
        if target_end <= target_start:
            batch_target_indices = np.concatenate([np.arange(target_start, num_target_samples), np.arange(0, target_end)])
        else:
            batch_target_indices = np.arange(target_start, target_end)
        if len(batch_target_indices) < batch_size:
            batch_target_indices = np.tile(batch_target_indices, (batch_size // len(batch_target_indices) + 1))[:batch_size]
        batch_targets = modality_map[output_modality].targets[batch_target_indices]
        
        logger.info(f"Yielding batch: modalities={batch_modality_types}, output={output_modality}, input_shapes={[inp.shape for inp in batch_inputs]}")
        yield (batch_inputs, batch_modality_types, batch_targets, output_modality)