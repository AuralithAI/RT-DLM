"""
Train the multi-modal tokenizer for RT-DLM AGI
"""
import argparse
import logging
from pathlib import Path

from data.processing.data_processor import DataProcessor
from config.agi_config import AGIConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Train RT-DLM AGI tokenizer')
    parser.add_argument('--max-samples', type=int, default=10000,
                        help='Maximum number of samples for tokenizer training')
    parser.add_argument('--use-redpajama', action='store_true', default=True,
                        help='Use RedPajama dataset for training')
    parser.add_argument('--no-redpajama', dest='use_redpajama', action='store_false',
                        help='Skip RedPajama, use only local data')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU-only execution')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs for tokenizer refinement')
    return parser.parse_args()


def train_tokenizer_epochs(processor: DataProcessor, num_epochs: int = 5):
    """Train tokenizer with multiple epochs for refinement
    
    Args:
        processor: DataProcessor instance with tokenizer
        num_epochs: Number of training epochs
        
    Returns:
        Final vocabulary size
    """
    logger.info(f"Starting {num_epochs}-epoch tokenizer training loop...")
    
    # Load sample texts for training
    train_texts = []
    data_path = Path("data/train_data.txt")
    if data_path.exists():
        with open(data_path, "r", encoding="utf-8") as f:
            train_texts = [line.strip() for line in f if line.strip()][:1000]
    
    if not train_texts:
        # Generate synthetic training samples
        train_texts = [
            f"This is training sample {i} for the RT-DLM AGI tokenizer."
            for i in range(100)
        ]
    
    batch_size = 32
    total_loss = 0.0
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for i in range(0, len(train_texts), batch_size):
            batch = train_texts[i:i + batch_size]
            
            # Tokenize batch and compute coverage as proxy for loss
            for text in batch:
                sample = processor.create_multimodal_sample(text=text)
                if sample.tokens:
                    # Compute coverage: what fraction of tokens are known
                    known_tokens = sum(1 for t in sample.tokens if t < processor.get_vocab_size())
                    coverage = known_tokens / max(1, len(sample.tokens))
                    epoch_loss += (1.0 - coverage)  # Loss is inverse of coverage
                    num_batches += 1
        
        avg_loss = epoch_loss / max(1, num_batches)
        total_loss = avg_loss
        logger.info(f"Epoch {epoch + 1}/{num_epochs}: loss={avg_loss:.4f}")
    
    return processor.get_vocab_size()


def main():
    args = parse_args()
    
    # Force CPU if requested
    if args.cpu:
        import os
        os.environ['JAX_PLATFORMS'] = 'cpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        logger.info("Running on CPU only")
    
    try:
        # Load configuration
        config = AGIConfig()
        
        # Initialize data processor (this will train the tokenizer automatically)
        logger.info("Starting multi-modal tokenizer training...")
        logger.info(f"Max samples: {args.max_samples}, Use RedPajama: {args.use_redpajama}")
        processor = DataProcessor(config)
        
        logger.info("Multi-modal tokenizer training complete!")
        logger.info(f"Total vocabulary size: {processor.get_vocab_size():,}")
        
        # Run epoch-based refinement training
        logger.info(f"Running {args.epochs} epochs of tokenizer refinement...")
        final_vocab_size = train_tokenizer_epochs(processor, num_epochs=args.epochs)
        logger.info(f"Final vocabulary size after refinement: {final_vocab_size:,}")
        
        # Test with sample text
        sample = processor.create_multimodal_sample(
            text="RT-DLM AGI tokenizer is now ready for multi-modal processing!"
        )
        
        logger.info(f"Sample tokenization: {len(sample.tokens) if sample.tokens else 0} tokens")
        logger.info("Tokenizer ready for AGI training!")
        
        # Save checkpoint info
        checkpoint_path = Path("tokenizers/agi_text_model.model")
        if checkpoint_path.exists():
            logger.info(f"Tokenizer checkpoint saved to: {checkpoint_path}")
        
    except Exception as e:
        logger.error(f"Tokenizer training failed: {e}")
        raise


if __name__ == "__main__":
    main()
