# Claude Code Configuration

## Testing Commands
```bash
# Run unit tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_models/ -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

## Linting Commands
```bash
# Format code with black
black src/ scripts/ tests/

# Check code style
flake8 src/ scripts/ tests/

# Sort imports
isort src/ scripts/ tests/

# Type checking
mypy src/
```

## Build Commands
```bash
# Install in development mode
pip install -e .

# Build distribution
python setup.py sdist bdist_wheel

# Install from source
pip install .
```

## Training Commands
```bash
# Full pretraining pipeline
python scripts/pretrain_simclr.py --config configs/pretrain/simclr_pets.yaml

# Fine-tuning with different settings
python scripts/finetune_segmentation.py --pretrained_model pets --data_ratio 0.5 --loss_function BCE

# Evaluation
python scripts/evaluate_model.py --model_path experiments/results/model.pth
```