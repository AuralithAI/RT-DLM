"""
Train the multi-modal tokenizer for RT-DLM AGI
"""
from data_processing.data_processor import DataProcessor
from agi_config import AGIConfig

def main():
    # Load configuration
    config = AGIConfig()
    
    # Initialize data processor (this will train the tokenizer automatically)
    processor = DataProcessor(config)
    
    print("Multi-modal tokenizer training complete!")
    print(f"Total vocabulary size: {processor.get_vocab_size():,}")
    
    # Test with sample text
    sample = processor.create_multimodal_sample(
        text="RT-DLM AGI tokenizer is now ready for multi-modal processing!"
    )
    
    print(f"Sample tokenization: {len(sample.tokens) if sample.tokens else 0} tokens")
    print("🚀 Tokenizer ready for AGI training!")

if __name__ == "__main__":
    main()