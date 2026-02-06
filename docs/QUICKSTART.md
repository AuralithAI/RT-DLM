# RT-DLM Quick Start Guide

Get started with RT-DLM model training and inference.

> **Note**: This guide covers model training and inference. For data collection and processing, see [Auralith-Data-Pipeline](https://github.com/AuralithAI/Auralith-Data-Pipeline).

## Installation

### Prerequisites

- Python 3.10+
- pip or conda
- CUDA (optional, for GPU acceleration)

### Install Dependencies

Clone the repository and install with `pip install -r requirements.txt` or use `python install_dependencies.py`.

## Training

### Prepare Training Data

Training data (pre-tokenized SafeTensor shards) should be prepared using [Auralith-Data-Pipeline](https://github.com/AuralithAI/Auralith-Data-Pipeline). The shards directory should contain `.safetensors` files with `input_ids` tensors.

### Train the Model

Run `python train.py --data-dir /path/to/shards` to train the model.

#### Training Options

- `--epochs`, `--batch-size`, `--lr` for hyperparameters
- `--d-model`, `--num-layers`, `--num-heads` for model architecture
- `--resume` to continue from a checkpoint

Training configuration is in `config/agi_config.py`.

## Distributed Training

For multi-GPU training, use `core.scalable_training`:

- `recommend_parallelism()` - Get optimal strategy for your hardware
- `estimate_model_memory()` - Estimate GPU memory requirements  
- `ScalableMesh` - Configure data and tensor parallelism

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed distributed training documentation.

## Running Tests

### Full Test Suite

Use `pytest tests/` or `python tests/test_runner.py` to run all tests.

### Specific Tests

Run specific files with `pytest tests/test_framework.py`, specific tests with `pytest tests/test_framework.py::test_function_name`, or verbose output with `pytest tests/ -v`. Distributed tests are in `tests/distributed/`.

## Configuration

Model configuration is in `config/agi_config.py`. Use `AGIConfig` with parameters like `vocab_size`, `hidden_dim`, `num_heads`, `num_layers`, and `max_seq_len`.

### Quantum Simulation (Optional)

Quantum-inspired layers are optional and add computational overhead. Use `estimate_quantum_overhead()` from `core.quantum` to check memory requirements. Set `quantum_layers=0` to disable.

## Next Steps

- Read the [Architecture Overview](ARCHITECTURE.md) for system design
- Check the [Sampling Guide](SAMPLING.md) for generation control
- Run tests with `pytest tests/` to validate your setup
- For data preparation and tokenization, see [Auralith-Data-Pipeline](https://github.com/AuralithAI/Auralith-Data-Pipeline)
