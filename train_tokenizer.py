"""
Train the multi-modal tokenizer for RT-DLM AGI
"""
from data_processing.advanced_data_processor import AdvancedDataProcessor
from agi_config import AdvancedAGIConfig

def main():
    # Load configuration
    config = AdvancedAGIConfig()
    
    # Initialize data processor (this will train the tokenizer automatically)
    processor = AdvancedDataProcessor(config)
    
    print("✅ Multi-modal tokenizer training complete!")
    print(f"Total vocabulary size: {processor.get_vocab_size():,}")
    
    # Test with sample text
    sample = processor.create_multimodal_sample(
        text="RT-DLM AGI tokenizer is now ready for multi-modal processing!"
    )
    
    print(f"Sample tokenization: {len(sample.tokens) if sample.tokens else 0} tokens")
    print("🚀 Tokenizer ready for AGI training!")

if __name__ == "__main__":
    main()