"""
Advanced Data Processing for RT-DLM AGI with Multi-Modal Tokenization
"""

import os
import json
import pickle
import numpy as np
import jax.numpy as jnp
from pathlib import Path
from typing import List, Dict, Tuple, Union, Any, Optional
from dataclasses import dataclass

from tokenization.multimodal_tokenizer import AdvancedMultiModalTokenizer, ModalityType, TokenizationConfig
from agi_config import AdvancedAGIConfig

@dataclass
class MultiModalDataSample:
    """Container for multi-modal data samples."""
    text: Optional[str] = None
    image_path: Optional[str] = None
    audio_path: Optional[str] = None
    video_path: Optional[str] = None
    document_path: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    tokens: Optional[List[int]] = None
    modalities: Optional[List[ModalityType]] = None

class AdvancedDataProcessor:
    """
    Advanced data processor for RT-DLM AGI that handles all data modalities
    and creates unified token sequences for training and inference.
    """
    
    def __init__(self, config: AdvancedAGIConfig):
        self.config = config
        self.tokenizer = AdvancedMultiModalTokenizer(config.tokenization_config)
        
        # Initialize tokenizer if not already trained
        self._ensure_tokenizer_ready()
    
    def _ensure_tokenizer_ready(self):
        """Ensure text tokenizer is trained and ready."""
        tokenizer_path = "tokenizers/agi_text_model"
        
        if not os.path.exists(f"{tokenizer_path}.model"):
            print("🔧 Training text tokenizer for AGI...")
            
            # Create sample training data
            sample_texts = self._get_sample_training_texts()
            
            # Train tokenizer
            self.tokenizer.train_text_tokenizer(sample_texts, tokenizer_path)
            print("✅ Text tokenizer training complete!")
        else:
            # Load existing tokenizer
            self.tokenizer.load_text_tokenizer(tokenizer_path)
            print("✅ Text tokenizer loaded successfully!")
    
    def _get_sample_training_texts(self) -> List[str]:
        """Get sample texts for training the tokenizer."""
        sample_texts = [
            "The RT-DLM AGI system processes multiple data modalities simultaneously.",
            "Advanced artificial intelligence requires understanding of text, images, audio, and video.",
            "Multi-modal learning enables more comprehensive understanding of the world.",
            "Consciousness simulation involves self-awareness and introspective capabilities.",
            "Quantum-enhanced neural networks leverage quantum computing principles.",
            "Scientific discovery through AI involves hypothesis generation and testing.",
            "Creative intelligence generates novel content across different modalities.",
            "Social-emotional intelligence enables empathetic and culturally aware interactions.",
            "Ethical AI frameworks ensure responsible and fair decision making.",
            "Self-improvement mechanisms allow continuous learning and adaptation.",
            "Chain-of-thought reasoning breaks down complex problems into steps.",
            "Meta-learning enables rapid adaptation to new tasks and domains.",
            "Hierarchical memory systems organize information at different time scales.",
            "Cross-modal attention allows information flow between different modalities.",
            "Unified representation spaces enable seamless multi-modal processing.",
            "Real-time dynamic learning adapts to changing environments instantly.",
            "Artificial general intelligence surpasses narrow AI in versatility.",
            "Cognitive architectures model human-like thinking processes.",
            "Neural architecture search optimizes model structures automatically.",
            "Continual learning prevents catastrophic forgetting of previous knowledge.",
            "Transfer learning leverages knowledge from one domain to another.",
            "Few-shot learning enables quick adaptation with minimal examples.",
            "Zero-shot learning performs tasks without explicit training examples.",
            "Reinforcement learning from human feedback aligns AI with human values.",
            "Constitutional AI embeds ethical principles into decision making.",
            "Interpretable AI provides explanations for its reasoning processes.",
            "Robust AI handles adversarial inputs and edge cases gracefully.",
            "Scalable AI architectures grow with increasing computational resources.",
            "Efficient AI optimizes performance while minimizing resource usage.",
            "Distributed AI coordinates multiple agents for complex tasks."
        ]
        
        # Load additional texts from data files if they exist
        data_files = [
            "data/train_data.txt",
            "data/validation_data.txt",
            "data/dataset.txt"
        ]
        
        for file_path in data_files:
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        file_texts = [line.strip() for line in f if line.strip()]
                        sample_texts.extend(file_texts[:1000])  # Limit to 1000 lines per file
                except Exception as e:
                    print(f"Warning: Could not read {file_path}: {e}")
        
        return sample_texts
    
    def create_multimodal_sample(
        self,
        text: Optional[str] = None,
        image_path: Optional[str] = None,
        audio_path: Optional[str] = None,
        video_path: Optional[str] = None,
        document_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> MultiModalDataSample:
        """Create a multi-modal data sample."""
        
        sample = MultiModalDataSample(
            text=text,
            image_path=image_path,
            audio_path=audio_path,
            video_path=video_path,
            document_path=document_path,
            metadata=metadata or {}
        )
        
        # Tokenize all available modalities
        all_tokens = []
        modalities = []
        
        if text is not None:
            text_tokens = self.tokenizer.tokenize(text, ModalityType.TEXT)
            all_tokens.extend(text_tokens)
            modalities.append(ModalityType.TEXT)
        
        if image_path is not None and os.path.exists(image_path):
            image_tokens = self.tokenizer.tokenize(image_path, ModalityType.IMAGE)
            all_tokens.extend(image_tokens)
            modalities.append(ModalityType.IMAGE)
        
        if audio_path is not None and os.path.exists(audio_path):
            audio_tokens = self.tokenizer.tokenize(audio_path, ModalityType.AUDIO)
            all_tokens.extend(audio_tokens)
            modalities.append(ModalityType.AUDIO)
        
        if video_path is not None and os.path.exists(video_path):
            video_tokens = self.tokenizer.tokenize(video_path, ModalityType.VIDEO)
            all_tokens.extend(video_tokens)
            modalities.append(ModalityType.VIDEO)
        
        if document_path is not None and os.path.exists(document_path):
            doc_modality = self.tokenizer.detect_modality(document_path)
            doc_tokens = self.tokenizer.tokenize(document_path, doc_modality)
            all_tokens.extend(doc_tokens)
            modalities.append(doc_modality)
        
        sample.tokens = all_tokens[:self.config.max_seq_length]
        sample.modalities = modalities
        
        return sample
    
    def process_text_file(self, file_path: str) -> List[MultiModalDataSample]:
        """Process a text file into multi-modal samples."""
        samples = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
            
            for line in lines:
                sample = self.create_multimodal_sample(text=line)
                samples.append(sample)
        
        except Exception as e:
            print(f"Error processing text file {file_path}: {e}")
        
        return samples
    
    def process_json_dataset(self, file_path: str) -> List[MultiModalDataSample]:
        """Process a JSON dataset with multi-modal entries."""
        samples = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                for item in data:
                    sample = self._json_item_to_sample(item)
                    if sample:
                        samples.append(sample)
            elif isinstance(data, dict):
                sample = self._json_item_to_sample(data)
                if sample:
                    samples.append(sample)
        
        except Exception as e:
            print(f"Error processing JSON dataset {file_path}: {e}")
        
        return samples
    
    def _json_item_to_sample(self, item: Dict[str, Any]) -> Optional[MultiModalDataSample]:
        """Convert a JSON item to a multi-modal sample."""
        try:
            return self.create_multimodal_sample(
                text=item.get('text'),
                image_path=item.get('image_path'),
                audio_path=item.get('audio_path'),
                video_path=item.get('video_path'),
                document_path=item.get('document_path'),
                metadata=item.get('metadata', {})
            )
        except Exception as e:
            print(f"Error converting JSON item to sample: {e}")
            return None
    
    def process_directory(
        self,
        directory_path: str,
        recursive: bool = True,
        max_files: int = 1000
    ) -> List[MultiModalDataSample]:
        """Process all supported files in a directory."""
        samples = []
        file_count = 0
        
        directory = Path(directory_path)
        if not directory.exists():
            print(f"Directory {directory_path} does not exist")
            return samples
        
        # Get all files
        if recursive:
            pattern = "**/*"
        else:
            pattern = "*"
        
        for file_path in directory.glob(pattern):
            if file_count >= max_files:
                break
            
            if file_path.is_file():
                try:
                    # Detect modality and create sample
                    modality = self.tokenizer.detect_modality(str(file_path))
                    
                    if modality == ModalityType.TEXT:
                        file_samples = self.process_text_file(str(file_path))
                        samples.extend(file_samples)
                    else:
                        # For non-text files, create a single sample
                        sample = MultiModalDataSample()
                        
                        if modality == ModalityType.IMAGE:
                            sample.image_path = str(file_path)
                        elif modality == ModalityType.AUDIO:
                            sample.audio_path = str(file_path)
                        elif modality == ModalityType.VIDEO:
                            sample.video_path = str(file_path)
                        else:
                            sample.document_path = str(file_path)
                        
                        # Tokenize the file
                        tokens = self.tokenizer.tokenize(str(file_path), modality)
                        sample.tokens = tokens[:self.config.max_seq_length]
                        sample.modalities = [modality]
                        samples.append(sample)
                    
                    file_count += 1
                
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")
                    continue
        
        print(f"Processed {file_count} files, created {len(samples)} samples")
        return samples
    
    def create_training_batches(
        self,
        samples: List[MultiModalDataSample],
        batch_size: Optional[int] = None,
        shuffle: bool = True
    ) -> List[Dict[str, jnp.ndarray]]:
        """Create training batches from multi-modal samples."""
        
        if batch_size is None:
            batch_size = self.config.batch_size
        
        # Filter out samples without tokens
        valid_samples = [s for s in samples if s.tokens and len(s.tokens) > 1]
        
        if shuffle:
            np.random.shuffle(valid_samples)
        
        batches = []
        
        for i in range(0, len(valid_samples), batch_size):
            batch_samples = valid_samples[i:i + batch_size]
            
            # Pad sequences to same length
            max_length = min(
                max(len(s.tokens) for s in batch_samples),
                self.config.max_seq_length
            )
            
            batch_tokens = []
            batch_targets = []
            
            for sample in batch_samples:
                tokens = sample.tokens[:max_length]
                
                # Pad if necessary
                while len(tokens) < max_length:
                    tokens.append(self.config.pad_token_id)
                
                # Create input and target sequences
                input_tokens = tokens[:-1]
                target_tokens = tokens[1:]
                
                # Pad if still needed
                while len(input_tokens) < max_length - 1:
                    input_tokens.append(self.config.pad_token_id)
                    target_tokens.append(self.config.pad_token_id)
                
                batch_tokens.append(input_tokens)
                batch_targets.append(target_tokens)
            
            # Convert to JAX arrays
            batch = {
                'inputs': jnp.array(batch_tokens),
                'targets': jnp.array(batch_targets)
            }
            
            batches.append(batch)
        
        return batches
    
    def save_processed_dataset(
        self,
        samples: List[MultiModalDataSample],
        output_path: str
    ):
        """Save processed dataset to disk."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert samples to serializable format
        serializable_samples = []
        for sample in samples:
            serializable_sample = {
                'text': sample.text,
                'image_path': sample.image_path,
                'audio_path': sample.audio_path,
                'video_path': sample.video_path,
                'document_path': sample.document_path,
                'metadata': sample.metadata,
                'tokens': sample.tokens,
                'modalities': [m.value for m in sample.modalities] if sample.modalities else None
            }
            serializable_samples.append(serializable_sample)
        
        with open(output_path, 'wb') as f:
            pickle.dump(serializable_samples, f)
        
        print(f"Saved {len(samples)} samples to {output_path}")
    
    def load_processed_dataset(self, input_path: str) -> List[MultiModalDataSample]:
        """Load processed dataset from disk."""
        samples = []
        
        try:
            with open(input_path, 'rb') as f:
                serializable_samples = pickle.load(f)
            
            for item in serializable_samples:
                sample = MultiModalDataSample(
                    text=item['text'],
                    image_path=item['image_path'],
                    audio_path=item['audio_path'],
                    video_path=item['video_path'],
                    document_path=item['document_path'],
                    metadata=item['metadata'],
                    tokens=item['tokens'],
                    modalities=[ModalityType(m) for m in item['modalities']] if item['modalities'] else None
                )
                samples.append(sample)
            
            print(f"Loaded {len(samples)} samples from {input_path}")
        
        except Exception as e:
            print(f"Error loading dataset from {input_path}: {e}")
        
        return samples
    
    def get_vocab_size(self) -> int:
        """Get the total vocabulary size for all modalities."""
        return self.tokenizer.get_vocab_size()
    
    def decode_tokens(self, tokens: List[int]) -> str:
        """Decode tokens back to text (for text modality)."""
        return self.tokenizer.detokenize_text(tokens)

# Example usage and testing
if __name__ == "__main__":
    # Initialize configuration and processor
    config = AdvancedAGIConfig()
    processor = AdvancedDataProcessor(config)
    
    print("🚀 TESTING ADVANCED DATA PROCESSOR")
    print("=" * 50)
    
    # Test single text sample
    print("\n1. Testing single text sample:")
    text_sample = processor.create_multimodal_sample(
        text="This is a test of the advanced RT-DLM AGI data processor."
    )
    print(f"Text: {text_sample.text}")
    print(f"Tokens: {len(text_sample.tokens)} tokens")
    print(f"Modalities: {[m.value for m in text_sample.modalities]}")
    
    # Test multi-modal sample
    print("\n2. Testing multi-modal sample:")
    multimodal_sample = processor.create_multimodal_sample(
        text="A beautiful sunset over the mountains.",
        metadata={"source": "test", "quality": "high"}
    )
    print(f"Text: {multimodal_sample.text}")
    print(f"Tokens: {len(multimodal_sample.tokens)} tokens")
    print(f"Modalities: {[m.value for m in multimodal_sample.modalities]}")
    
    # Test batch creation
    print("\n3. Testing batch creation:")
    samples = [text_sample, multimodal_sample]
    batches = processor.create_training_batches(samples, batch_size=2)
    
    if batches:
        batch = batches[0]
        print(f"Batch inputs shape: {batch['inputs'].shape}")
        print(f"Batch targets shape: {batch['targets'].shape}")
        print(f"Sample input tokens: {batch['inputs'][0][:10]}")
    
    print(f"\nTotal vocabulary size: {processor.get_vocab_size():,}")
    print("✅ Data processor testing complete!")
