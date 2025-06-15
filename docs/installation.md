# Installation Guide

This guide provides detailed installation instructions for the SimCLR Semantic Segmentation project.

## Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- Git

## Installation Methods

### Method 1: Conda Environment (Recommended)

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/simclr-semantic-segmentation.git
cd simclr-semantic-segmentation
```

2. **Create conda environment**
```bash
conda env create -f environment.yml
conda activate simclr-segmentation
```

3. **Install the package in development mode**
```bash
pip install -e .
```

### Method 2: Virtual Environment with pip

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/simclr-semantic-segmentation.git
cd simclr-semantic-segmentation
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
pip install -e .
```

### Method 3: Docker (Coming Soon)

Docker support will be added in future releases.

## Verification

To verify your installation:

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import segmentation_models_pytorch as smp; print('SMP installed successfully')"
python -c "from src.models.simclr import SimCLR; print('Package installed successfully')"
```

## GPU Setup

### CUDA Installation

1. **Check CUDA availability**
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

2. **If CUDA is not available**, install PyTorch with CUDA support:
```bash
# For CUDA 11.3
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113

# For CUDA 11.7
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu117
```

### CPU-Only Installation

If you don't have a GPU, you can install the CPU-only version:

```bash
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
```

## Common Issues and Solutions

### Issue 1: ImportError for segmentation_models_pytorch

**Solution:**
```bash
pip install segmentation-models-pytorch
```

### Issue 2: CUDA out of memory

**Solution:**
- Reduce batch size in configuration files
- Use gradient accumulation
- Use mixed precision training (if supported)

### Issue 3: Dataset download fails

**Solution:**
```bash
# Make sure you have wget installed
# On macOS: brew install wget
# On Ubuntu: sudo apt-get install wget

# Manually run the download script
bash scripts/download_datasets.sh
```

### Issue 4: Permission denied for scripts

**Solution:**
```bash
chmod +x scripts/*.sh
```

## Development Setup

For development, install additional dependencies:

```bash
pip install -e ".[dev]"
```

This includes:
- pytest for testing
- black for code formatting
- flake8 for linting
- isort for import sorting

## Testing Installation

Run the test suite to ensure everything is working:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test
python -m pytest tests/test_models/test_simclr.py -v
```

## System Requirements

### Minimum Requirements
- Python 3.8+
- 8GB RAM
- 2GB disk space

### Recommended Requirements
- Python 3.9+
- 16GB RAM
- NVIDIA GPU with 8GB+ VRAM
- 10GB disk space

## Next Steps

After installation:

1. [Download datasets](../scripts/download_datasets.sh)
2. Check out the [usage guide](usage.md)
3. Try the [quick start tutorial](../README.md#quick-start)
4. Explore the [Jupyter notebooks](../notebooks/)

## Getting Help

If you encounter issues:

1. Check this installation guide
2. Look at [troubleshooting](troubleshooting.md)
3. Search existing [GitHub issues](https://github.com/yourusername/simclr-semantic-segmentation/issues)
4. Open a new issue with detailed error information