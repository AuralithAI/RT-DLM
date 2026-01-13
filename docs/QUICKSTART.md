# RT-DLM Quick Start Guide

Get started with RT-DLM model training and inference.

> **Note**: This guide covers model training and inference. For data collection and processing, see [Auralith-Data-Pipeline](https://github.com/AuralithAI/Auralith-Data-Pipeline).

## Installation

### Prerequisites

- Python 3.10+
- pip or conda
- CUDA (optional, for GPU acceleration)

### Install Dependencies

```bash
git clone https://github.com/AuralithAI/RT-DLM.git
cd RT-DLM
pip install -r requirements.txt
```

Or use the installer script:

```bash
python install_dependencies.py
```

## Training

### Prepare Training Data

Training data should be prepared using [Auralith-Data-Pipeline](https://github.com/AuralithAI/Auralith-Data-Pipeline) and placed in a `data/` directory:

```
data/
├── train_data.txt       # Training corpus
├── validation_data.txt  # Validation corpus
└── rt_dlm_sp.model      # Trained tokenizer (generated)
```

### Train the Tokenizer

Train a SentencePiece tokenizer on your data:

```bash
python train_tokenizer.py --input data/train_data.txt --vocab-size 50000
```

This creates `data/rt_dlm_sp.model` and `data/rt_dlm_sp.vocab`.

### Train the Model

Train the model:

```bash
python train.py
```

Training configuration is in `config/train_config.py`.

## Inference

### Run the Inference Demo

```bash
python inference.py
```

This starts an interactive session with the trained model.

### Using in Code

```python
from inference import RT_DLM_AGI_Assistant

# Initialize
assistant = RT_DLM_AGI_Assistant()

# Generate response
response = assistant.generate_response(
    "What is quantum computing?",
    temperature=0.7,
    top_k=50
)
print(response)

# Step-by-step reasoning
reasoning = assistant.think_step_by_step(
    "Why is the sky blue?"
)
print(reasoning)

# Creative generation
creative = assistant.creative_generation(
    "Write a poem about AI"
)
print(creative)
```

## Running Tests

### Full Test Suite

```bash
# Using pytest
pytest tests/

# Using test runner
python tests/test_runner.py
```

### Specific Tests

```bash
# Run specific test file
pytest tests/test_framework.py

# Run specific test
pytest tests/test_framework.py::test_function_name

# Run with verbose output
pytest tests/ -v
```

## Configuration

Model configuration is in `config/agi_config.py`:

```python
from config.agi_config import AGIConfig

config = AGIConfig(
    vocab_size=50000,
    hidden_dim=512,
    num_heads=8,
    num_layers=6,
    max_seq_len=1024,
    # ... more options
)
```

Training configuration is in `config/train_config.py`.

## Next Steps

- Read the [Architecture Overview](ARCHITECTURE.md) for system design
- Check the [Sampling Guide](SAMPLING.md) for generation control
- Run tests with `pytest tests/` to validate your setup
- For data preparation, see [Auralith-Data-Pipeline](https://github.com/AuralithAI/Auralith-Data-Pipeline)
