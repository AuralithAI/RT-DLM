from data_utils import DataProcessor
from train_config import TrainConfig

# Load configuration
config = TrainConfig()

processor = DataProcessor(vocab_size=config.vocab_size)
processor.train_tokenizer("data/train_data.txt")