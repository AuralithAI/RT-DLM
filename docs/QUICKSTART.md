# RT-DLM Quick Start Guide

Get started with RT-DLM in minutes.

## Installation

### Prerequisites

- Python 3.10+
- pip or conda

### Install Dependencies

```bash
cd RT-DLM
pip install -r requirements.txt
```

Or use the installer script:

```bash
python install_dependencies.py
```

## Training

### Train the Tokenizer

First, train the SentencePiece tokenizer on your data:

```bash
python train_tokenizer.py
```

This creates `data/rt_dlm_sp.model` and `data/rt_dlm_sp.vocab`.

### Train the Model

Train the full AGI model:

```bash
python train.py
```

Training configuration is in `config/agi_config.py`.

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

## Project Structure

```
RT-DLM/
├── rtdlm.py          # Main AGI model
├── train.py          # Training script
├── inference.py      # Inference script
├── config/           # Configuration
├── core/             # Core components
├── modules/          # Feature modules
├── data/             # Data files
├── apps/             # Applications
├── tests/            # Test suite
└── docs/             # Documentation
```

## Configuration

The main configuration is in `config/agi_config.py`:

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

## Next Steps

- Read the [Architecture Overview](ARCHITECTURE.md) for system design
- Check the [Sampling Guide](SAMPLING.md) for generation control
- Explore `apps/` for downstream applications
- Run tests to validate your setup
