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

Training data (pre-tokenized SafeTensor shards) should be prepared using [Auralith-Data-Pipeline](https://github.com/AuralithAI/Auralith-Data-Pipeline):

```bash
# Using Auralith-Data-Pipeline
auralith-pipeline process --input ./raw_data --output ./shards
```

The shards directory should contain `.safetensors` files with `input_ids` tensors.

### Train the Model

Train the model with pre-tokenized data:

```bash
python train.py --data-dir /path/to/shards
```

#### Training Options

```bash
# Train with custom hyperparameters
python train.py --data-dir ./shards --epochs 50 --batch-size 32 --lr 1e-4

# Specify model architecture
python train.py --data-dir ./shards --d-model 768 --num-layers 24 --num-heads 12

# Resume from a checkpoint
python train.py --data-dir ./shards --resume checkpoints/rtdlm_epoch_10.safetensors
```

Training configuration is in `config/agi_config.py`.

## Distributed Training

For multi-GPU training, use `core.scalable_training`:

- `recommend_parallelism()` - Get optimal strategy for your hardware
- `estimate_model_memory()` - Estimate GPU memory requirements  
- `ScalableMesh` - Configure data and tensor parallelism

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed distributed training documentation.

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

# Run distributed training tests
pytest tests/distributed/ -v
```

## Configuration

Model configuration is in `config/agi_config.py`. Use `AGIConfig` with parameters like `vocab_size`, `hidden_dim`, `num_heads`, `num_layers`, and `max_seq_len`.

### Quantum Simulation (Optional)

Quantum-inspired layers are optional and add computational overhead. Use `estimate_quantum_overhead()` from `core.quantum` to check memory requirements. Set `quantum_layers=0` to disable.

## Next Steps

- Read the [Architecture Overview](ARCHITECTURE.md) for system design
- Check the [Sampling Guide](SAMPLING.md) for generation control
- Run tests with `pytest tests/` to validate your setup
- For data preparation and tokenization, see [Auralith-Data-Pipeline](https://github.com/AuralithAI/Auralith-Data-Pipeline)
