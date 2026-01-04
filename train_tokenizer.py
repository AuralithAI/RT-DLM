"""
Train the multi-modal tokenizer for RT-DLM AGI
"""
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

def main():
    try:
        # Load configuration
        config = AGIConfig()
        
        # Initialize data processor (this will train the tokenizer automatically)
        logger.info("Starting multi-modal tokenizer training...")
        processor = DataProcessor(config)
        
        logger.info("Multi-modal tokenizer training complete!")
        logger.info(f"Total vocabulary size: {processor.get_vocab_size():,}")
        
        # Test with sample text
        sample = processor.create_multimodal_sample(
            text="RT-DLM AGI tokenizer is now ready for multi-modal processing!"
        )
        
        logger.info(f"Sample tokenization: {len(sample.tokens) if sample.tokens else 0} tokens")
        logger.info("Tokenizer ready for AGI training!")
        
    except Exception as e:
        logger.error(f"Tokenizer training failed: {e}")
        raise

if __name__ == "__main__":
    main()