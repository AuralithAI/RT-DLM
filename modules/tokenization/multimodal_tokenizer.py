"""
Advanced Multi-Modal Tokenization System for RT-DLM AGI
Handles: Text, Images, Audio, Video, PDFs, XML, ZIP, Binary files, and more
"""

import os
import io
import json
import zipfile
import hashlib
import mimetypes
from pathlib import Path
from typing import List, Dict, Tuple, Union, Any, Optional
from dataclasses import dataclass
from enum import Enum

import numpy as np
import jax.numpy as jnp

# Text processing
import sentencepiece as spm

# Image processing
try:
    from PIL import Image
    import cv2
    HAS_VISION = True
except ImportError:
    HAS_VISION = False
    print("Vision libraries not installed. Image/Video processing will be limited.")

# Audio processing
try:
    import librosa
    import soundfile as sf
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False
    print("Audio libraries not installed. Audio processing will be limited.")

# Document processing
try:
    import PyPDF2
    HAS_PDF = True
except ImportError:
    HAS_PDF = False
    print("PyPDF2 not installed. PDF processing will be limited.")

try:
    import xml.etree.ElementTree as ET
    from bs4 import BeautifulSoup
    HAS_DOC_PARSING = True
except ImportError:
    HAS_DOC_PARSING = False
    print("Document parsing libraries not installed.")

class ModalityType(Enum):
    """Different data modalities supported by the tokenizer."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    PDF = "pdf"
    XML = "xml"
    HTML = "html"
    JSON = "json"
    CSV = "csv"
    ZIP = "zip"
    BINARY = "binary"
    CODE = "code"

@dataclass
class TokenizationConfig:
    """Configuration for multi-modal tokenization."""
    # Text tokenization
    text_vocab_size: int = 32000
    text_model_type: str = "bpe"  # "bpe", "unigram"
    max_text_length: int = 2048
    
    # Image tokenization
    image_patch_size: int = 16
    image_vocab_size: int = 8192
    image_resize: Tuple[int, int] = (224, 224)
    
    # Audio tokenization
    audio_sample_rate: int = 16000
    audio_hop_length: int = 512
    audio_n_mels: int = 80
    audio_vocab_size: int = 1024
    
    # Video tokenization
    video_fps: int = 8
    video_max_frames: int = 64
    
    # Document processing
    pdf_max_pages: int = 100
    xml_max_depth: int = 20
    
    # General
    max_sequence_length: int = 4096
    pad_token_id: int = 0
    unk_token_id: int = 1
    bos_token_id: int = 2
    eos_token_id: int = 3
    sep_token_id: int = 4
    
    # Special modality tokens
    def get_modality_tokens(self) -> Dict[str, int]:
        return {
            "TEXT_START": 10,
            "TEXT_END": 11,
            "IMAGE_START": 12,
            "IMAGE_END": 13,
            "AUDIO_START": 14,
            "AUDIO_END": 15,
            "VIDEO_START": 16,
            "VIDEO_END": 17,
            "PDF_START": 18,
            "PDF_END": 19,
            "XML_START": 20,
            "XML_END": 21,
            "BINARY_START": 22,
            "BINARY_END": 23,
            "CODE_START": 24,
            "CODE_END": 25,
        }

class MultiModalTokenizer:
    """
    Comprehensive tokenizer for all data modalities.
    Converts any input type into a unified token sequence.
    """
    
    def __init__(self, config: Optional[TokenizationConfig] = None):
        self.config = config or TokenizationConfig()
        self.modality_tokens = self.config.get_modality_tokens()
        
        # Initialize text tokenizer
        self.text_tokenizer = None
        
        # File type detection
        mimetypes.init()
        
    def train_text_tokenizer(self, texts: List[str], model_path: str = "tokenizers/text_model"):
        """Train the text tokenizer on provided texts."""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Write texts to temporary file
        temp_file = f"{model_path}_temp.txt"
        with open(temp_file, "w", encoding="utf-8") as f:
            for text in texts:
                f.write(text + "\n")
        
        # Train SentencePiece
        spm.SentencePieceTrainer.train(
            input=temp_file,
            model_prefix=model_path,
            vocab_size=self.config.text_vocab_size,
            model_type=self.config.text_model_type,
            pad_id=self.config.pad_token_id,
            unk_id=self.config.unk_token_id,
            bos_id=self.config.bos_token_id,
            eos_id=self.config.eos_token_id,
        )
        
        # Load the trained model
        self.text_tokenizer = spm.SentencePieceProcessor()
        self.text_tokenizer.load(f"{model_path}.model")
        
        # Clean up temp file
        os.remove(temp_file)
        print(f"Text tokenizer trained and saved to {model_path}.model")
    
    def load_text_tokenizer(self, model_path: str):
        """Load pre-trained text tokenizer."""
        self.text_tokenizer = spm.SentencePieceProcessor()
        self.text_tokenizer.load(f"{model_path}.model")
        print(f"Text tokenizer loaded from {model_path}.model")
    
    def detect_modality(self, input_data) -> ModalityType:
        """Automatically detect the modality of input data."""
        if isinstance(input_data, str):
            if os.path.isfile(input_data):
                # File path - detect by extension and MIME type
                mime_type, _ = mimetypes.guess_type(input_data)
                return self._mime_to_modality(mime_type, input_data)
            else:
                # Text string
                return ModalityType.TEXT
        
        elif isinstance(input_data, bytes):
            # Binary data - analyze header
            return self._detect_binary_modality(input_data)
        
        elif isinstance(input_data, Path):
            return self.detect_modality(str(input_data))
        
        else:
            return ModalityType.BINARY
    
    def _mime_to_modality(self, mime_type: Optional[str], file_path: str) -> ModalityType:
        """Convert MIME type to modality."""
        if mime_type is None:
            # Check file extension
            ext = Path(file_path).suffix.lower()
            extension_map = {
                '.txt': ModalityType.TEXT,
                '.md': ModalityType.TEXT,
                '.py': ModalityType.CODE,
                '.js': ModalityType.CODE,
                '.cpp': ModalityType.CODE,
                '.java': ModalityType.CODE,
                '.zip': ModalityType.ZIP,
                '.rar': ModalityType.ZIP,
                '.7z': ModalityType.ZIP,
                '.tar': ModalityType.ZIP,
                '.gz': ModalityType.ZIP,
                '.jpg': ModalityType.IMAGE,
                '.jpeg': ModalityType.IMAGE,
                '.png': ModalityType.IMAGE,
                '.gif': ModalityType.IMAGE,
                '.mp3': ModalityType.AUDIO,
                '.wav': ModalityType.AUDIO,
                '.mp4': ModalityType.VIDEO,
                '.avi': ModalityType.VIDEO,
                '.pdf': ModalityType.PDF,
                '.xml': ModalityType.XML,
                '.html': ModalityType.HTML,
                '.json': ModalityType.JSON,
                '.csv': ModalityType.CSV,
            }
            return extension_map.get(ext, ModalityType.BINARY)
        
        if mime_type.startswith('text/'):
            if 'xml' in mime_type:
                return ModalityType.XML
            elif 'html' in mime_type:
                return ModalityType.HTML
            return ModalityType.TEXT
        elif mime_type.startswith('image/'):
            return ModalityType.IMAGE
        elif mime_type.startswith('audio/'):
            return ModalityType.AUDIO
        elif mime_type.startswith('video/'):
            return ModalityType.VIDEO
        elif mime_type == 'application/pdf':
            return ModalityType.PDF
        elif mime_type == 'application/json':
            return ModalityType.JSON
        elif mime_type in ['application/zip', 'application/x-zip-compressed']:
            return ModalityType.ZIP
        else:
            return ModalityType.BINARY
    
    def _detect_binary_modality(self, data: bytes) -> ModalityType:
        """Detect modality from binary data headers."""
        if data.startswith(b'\x89PNG'):
            return ModalityType.IMAGE
        elif data.startswith(b'\xff\xd8\xff'):  # JPEG
            return ModalityType.IMAGE
        elif data.startswith(b'GIF8'):
            return ModalityType.IMAGE
        elif data.startswith(b'RIFF') and b'WAVE' in data[:20]:
            return ModalityType.AUDIO
        elif data.startswith(b'ID3') or data.startswith(b'\xff\xfb'):  # MP3
            return ModalityType.AUDIO
        elif data.startswith(b'%PDF'):
            return ModalityType.PDF
        elif data.startswith(b'PK\x03\x04'):  # ZIP
            return ModalityType.ZIP
        else:
            return ModalityType.BINARY
    
    def tokenize(self, input_data, modality: Optional[ModalityType] = None) -> List[int]:
        """
        Tokenize input data based on its modality.
        Returns a unified token sequence with modality markers.
        """
        if modality is None:
            modality = self.detect_modality(input_data)
        
        # Add modality start token
        tokens = [self.modality_tokens[f"{modality.value.upper()}_START"]]
        
        try:
            if modality == ModalityType.TEXT:
                tokens.extend(self._tokenize_text(input_data))
            elif modality == ModalityType.IMAGE:
                tokens.extend(self._tokenize_image(input_data))
            elif modality == ModalityType.AUDIO:
                tokens.extend(self._tokenize_audio(input_data))
            elif modality == ModalityType.VIDEO:
                tokens.extend(self._tokenize_video(input_data))
            elif modality in [ModalityType.PDF, ModalityType.XML, ModalityType.HTML, ModalityType.JSON]:
                tokens.extend(self._tokenize_document(input_data, modality))
            elif modality == ModalityType.ZIP:
                tokens.extend(self._tokenize_zip(input_data))
            elif modality == ModalityType.CODE:
                tokens.extend(self._tokenize_code(input_data))
            else:
                tokens.extend(self._tokenize_binary(input_data))
        
        except Exception as e:
            print(f"Error tokenizing {modality}: {e}")
            # Fallback to binary tokenization
            tokens.extend(self._tokenize_binary(input_data))
        
        # Add modality end token
        tokens.append(self.modality_tokens[f"{modality.value.upper()}_END"])
        
        return tokens[:self.config.max_sequence_length]
    
    def _tokenize_text(self, text) -> List[int]:
        """Tokenize text using the trained text tokenizer."""
        if isinstance(text, (str, Path)) and os.path.isfile(str(text)):
            with open(text, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
        
        if isinstance(text, bytes):
            text = text.decode('utf-8', errors='ignore')
        
        if self.text_tokenizer is None:
            # Fallback: simple character-level tokenization
            return [ord(c) % 1000 + 100 for c in str(text)[:self.config.max_text_length]]
        
        try:
            return self.text_tokenizer.encode_as_ids(str(text))[:self.config.max_text_length]
        except Exception:
            # Fallback if encoding fails
            return [ord(c) % 1000 + 100 for c in str(text)[:self.config.max_text_length]]
    
    def _tokenize_image(self, image_path) -> List[int]:
        """Tokenize image using patch-based approach."""
        if not HAS_VISION:
            return [self.config.unk_token_id]
        
        try:
            # Load and preprocess image
            if isinstance(image_path, bytes):
                image = Image.open(io.BytesIO(image_path)).convert('RGB')
            else:
                image = Image.open(image_path).convert('RGB')
            
            image = image.resize(self.config.image_resize)
            image_array = np.array(image)
            
            # Create patches
            patch_size = self.config.image_patch_size
            h, w, c = image_array.shape
            
            tokens = []
            for i in range(0, h, patch_size):
                for j in range(0, w, patch_size):
                    patch = image_array[i:i+patch_size, j:j+patch_size]
                    if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                        # Pad patch
                        padded_patch = np.zeros((patch_size, patch_size, c), dtype=np.uint8)
                        padded_patch[:patch.shape[0], :patch.shape[1]] = patch
                        patch = padded_patch
                    
                    # Simple patch encoding: average RGB values
                    avg_rgb = np.mean(patch, axis=(0, 1))
                    token = int(avg_rgb[0] * 1000 + avg_rgb[1] * 100 + avg_rgb[2])
                    tokens.append(token % self.config.image_vocab_size + 1000)
            
            return tokens
        
        except Exception as e:
            print(f"Error tokenizing image: {e}")
            return [self.config.unk_token_id]
    
    def _tokenize_audio(self, audio_path) -> List[int]:
        """Tokenize audio using spectrogram-based approach."""
        if not HAS_AUDIO:
            return [self.config.unk_token_id]
        
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.config.audio_sample_rate)
            
            # Extract mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=sr,
                hop_length=self.config.audio_hop_length,
                n_mels=self.config.audio_n_mels
            )
            
            # Convert to log scale
            log_mel = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Quantize to tokens
            tokens = []
            for frame in log_mel.T:  # Transpose to get time frames
                # Simple quantization
                quantized = ((frame + 80) / 80 * self.config.audio_vocab_size).astype(int)
                quantized = np.clip(quantized, 0, self.config.audio_vocab_size - 1)
                tokens.extend(quantized.tolist())
            
            return [t + 2000 for t in tokens]  # Offset to avoid conflicts
        
        except Exception as e:
            print(f"Error tokenizing audio: {e}")
            return [self.config.unk_token_id]
    
    def _tokenize_video(self, video_path) -> List[int]:
        """Tokenize video by sampling frames."""
        if not HAS_VISION:
            return [self.config.unk_token_id]
        
        try:
            # Load video
            cap = cv2.VideoCapture(str(video_path))
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Sample frames uniformly
            frame_indices = np.linspace(0, frame_count - 1, 
                                      min(self.config.video_max_frames, frame_count), 
                                      dtype=int)
            
            tokens = []
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    # Convert frame to PIL Image
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_pil = Image.fromarray(frame_rgb)
                    
                    # Save temporarily and tokenize
                    temp_path = "/tmp/temp_frame.jpg"
                    frame_pil.save(temp_path)
                    frame_tokens = self._tokenize_image(temp_path)
                    tokens.extend(frame_tokens[:50])  # Limit tokens per frame
                    
                    # Clean up
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
            
            cap.release()
            return tokens
        
        except Exception as e:
            print(f"Error tokenizing video: {e}")
            return [self.config.unk_token_id]
    
    def _tokenize_document(self, doc_path, modality: ModalityType) -> List[int]:
        """Tokenize structured documents."""
        try:
            if modality == ModalityType.PDF:
                return self._tokenize_pdf(doc_path)
            elif modality == ModalityType.XML:
                return self._tokenize_xml(doc_path)
            elif modality == ModalityType.HTML:
                return self._tokenize_html(doc_path)
            elif modality == ModalityType.JSON:
                return self._tokenize_json(doc_path)
            else:
                return [self.config.unk_token_id]
        
        except Exception as e:
            print(f"Error tokenizing document: {e}")
            return [self.config.unk_token_id]
    
    def _tokenize_pdf(self, pdf_path) -> List[int]:
        """Extract text from PDF and tokenize."""
        if not HAS_PDF:
            return [self.config.unk_token_id]
        
        text = ""
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num, page in enumerate(reader.pages[:self.config.pdf_max_pages]):
                text += page.extract_text() + "\n"
        
        return self._tokenize_text(text)
    
    def _tokenize_xml(self, xml_path) -> List[int]:
        """Parse XML structure and tokenize."""
        if not HAS_DOC_PARSING:
            return [self.config.unk_token_id]
        
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        def extract_structure(element, depth=0):
            if depth > self.config.xml_max_depth:
                return []
            
            tokens = []
            # Tag name
            tokens.extend([ord(c) % 100 + 4000 for c in element.tag[:20]])
            
            # Text content
            if element.text:
                tokens.extend([ord(c) % 100 + 4100 for c in element.text[:100]])
            
            # Recursively process children
            for child in element:
                tokens.extend(extract_structure(child, depth + 1))
            
            return tokens
        
        return extract_structure(root)[:self.config.max_sequence_length // 2]
    
    def _tokenize_html(self, html_path) -> List[int]:
        """Parse HTML and extract meaningful content."""
        if not HAS_DOC_PARSING:
            return [self.config.unk_token_id]
        
        with open(html_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        soup = BeautifulSoup(content, 'html.parser')
        text = soup.get_text()
        
        return self._tokenize_text(text)
    
    def _tokenize_json(self, json_path) -> List[int]:
        """Parse JSON structure."""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert to string and tokenize
        json_str = json.dumps(data, ensure_ascii=False)[:self.config.max_text_length]
        return self._tokenize_text(json_str)
    
    def _tokenize_zip(self, zip_path) -> List[int]:
        """Tokenize ZIP archives by extracting and tokenizing contents."""
        tokens = []
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_file:
                for file_info in zip_file.filelist[:10]:  # Limit to first 10 files
                    if file_info.file_size < 1024 * 1024:  # Limit to 1MB files
                        try:
                            content = zip_file.read(file_info.filename)
                            file_modality = self._detect_binary_modality(content)
                            file_tokens = self.tokenize(content, file_modality)
                            tokens.extend(file_tokens[:100])  # Limit tokens per file
                        except Exception:
                            continue
        except Exception:
            # Fallback to binary tokenization
            return self._tokenize_binary(zip_path)
        
        return tokens[:self.config.max_sequence_length // 2]
    
    def _tokenize_code(self, code_path) -> List[int]:
        """Tokenize source code with syntax awareness."""
        with open(code_path, 'r', encoding='utf-8', errors='ignore') as f:
            code = f.read()
        
        # Simple syntax-aware tokenization
        # This could be enhanced with AST parsing
        return self._tokenize_text(code)
    
    def _tokenize_binary(self, data) -> List[int]:
        """Tokenize arbitrary binary data."""
        if isinstance(data, (str, Path)):
            with open(data, 'rb') as f:
                data = f.read()
        
        # Simple byte-level tokenization
        tokens = []
        for i in range(0, len(data), 4):  # Process 4 bytes at a time
            chunk = data[i:i+4]
            if len(chunk) < 4:
                chunk = chunk + b'\x00' * (4 - len(chunk))  # Pad
            
            # Convert 4 bytes to int
            token = int.from_bytes(chunk, byteorder='big') % 10000 + 7000
            tokens.append(token)
            
            if len(tokens) >= self.config.max_sequence_length // 2:
                break
        
        return tokens
    
    def detokenize_text(self, tokens: List[int]) -> str:
        """Detokenize text tokens back to string."""
        if self.text_tokenizer is None:
            # Fallback: character-level detokenization
            return "".join([chr((t - 100) % 256) for t in tokens if 100 <= t < 1100])
        
        try:
            return self.text_tokenizer.decode_ids(tokens)
        except Exception:
            # Fallback if decoding fails
            return "".join([chr((t - 100) % 256) for t in tokens if 100 <= t < 1100])
    
    def get_vocab_size(self) -> int:
        """Get the total vocabulary size including all modalities."""
        return max([
            self.config.text_vocab_size + 1000,
            self.config.image_vocab_size + 2000,
            self.config.audio_vocab_size + 3000,
            10000 + 7000,  # Binary tokens
        ])

# Example usage and testing
if __name__ == "__main__":
    # Initialize tokenizer
    config = TokenizationConfig()
    tokenizer = MultiModalTokenizer(config)
    
    # Example texts for training
    sample_texts = [
        "This is a sample text for training the tokenizer.",
        "Advanced multi-modal AI systems can process various data types.",
        "Tokenization is crucial for converting raw data into model inputs.",
        "The RT-DLM model can understand text, images, audio, and video simultaneously.",
        "Artificial General Intelligence requires unified representation of all data modalities."
    ]
    
    # Train text tokenizer
    tokenizer.train_text_tokenizer(sample_texts, "tokenizers/agi_text_model")
    
    # Test different modalities
    test_cases = [
        ("Hello, world! This is a test of the AGI tokenizer.", ModalityType.TEXT),
        ("{'key': 'value', 'numbers': [1, 2, 3]}", ModalityType.JSON),
    ]
    
    print("=" * 60)
    print("ADVANCED MULTI-MODAL TOKENIZER TEST")
    print("=" * 60)
    
    for data, modality in test_cases:
        print(f"\nTesting {modality.value.upper()}:")
        print(f"Input: {data}")
        
        tokens = tokenizer.tokenize(data, modality)
        print(f"Tokens (first 20): {tokens[:20]}")
        print(f"Total tokens: {len(tokens)}")
        
        # Test text detokenization
        if modality == ModalityType.TEXT:
            # Extract only text tokens (between TEXT_START and TEXT_END)
            text_tokens = []
            in_text = False
            for token in tokens:
                if token == tokenizer.modality_tokens["TEXT_START"]:
                    in_text = True
                elif token == tokenizer.modality_tokens["TEXT_END"]:
                    in_text = False
                elif in_text:
                    text_tokens.append(token)
            
            detokenized = tokenizer.detokenize_text(text_tokens)
            print(f"Detokenized: {detokenized}")
    
    print(f"\nTotal vocabulary size: {tokenizer.get_vocab_size()}")
    print("Tokenizer ready for AGI training!")

