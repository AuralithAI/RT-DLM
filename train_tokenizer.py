"""
Train the multi-modal tokenizer for RT-DLM AGI
"""
import argparse
import logging
from pathlib import Path

from data_processing.data_processor import DataProcessor
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
    return parser.parse_args()

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