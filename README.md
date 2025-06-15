# Self-Supervised Semantic Segmentation with SimCLR

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A PyTorch implementation of self-supervised semantic segmentation using SimCLR pretraining followed by supervised fine-tuning on segmentation tasks.

## 🚀 Key Features

- **Self-supervised pretraining** with SimCLR on multiple datasets (CIFAR-10, Cats vs Dogs)
- **Segmentation fine-tuning** using U-Net decoder with pretrained ResNet-50 encoder
- **Multiple loss functions** (Dice, Focal, BCE) with comprehensive evaluation metrics
- **Configurable training pipeline** with YAML-based configuration system
- **Extensive evaluation** including IoU, F1-score, and visualization tools

## 📋 Table of Contents

- [🚀 Key Features](#key-features)
- [🛠️ Installation](#installation)
- [⚡ Quick Start](#quick-start)
- [📊 Dataset Preparation](#dataset-preparation)
- [🏋️ Training](#training)
- [📈 Evaluation](#evaluation)
- [🏆 Model Zoo](#model-zoo)
- [📊 Results](#results)
- [🔧 Configuration](#configuration)
- [🤝 Contributing](#contributing)
- [📄 License](#license)
- [🙏 Acknowledgments](#acknowledgments)

## 🛠️ Installation

### Option 1: Conda Environment (Recommended)
```bash
# Clone the repository
git clone https://github.com/RoboRabbit666/simclr-semantic-segmentation.git
cd simclr-semantic-segmentation

# Create conda environment
conda env create -f environment.yml
conda activate simclr-segmentation
```

### Option 2: Pip Installation
```bash
pip install -r requirements.txt
```

## ⚡ Quick Start

### 1. Download Pretrained Models
```bash
# Download pretrained SimCLR backbones
bash scripts/download_datasets.sh
```

### 2. Run Complete Pipeline
```bash
# Pretrain + Fine-tune + Evaluate
bash scripts/run_experiments.sh
```

### 3. Fine-tune Only (Using Pretrained Backbone)
```bash
python scripts/finetune_segmentation.py \
  --config configs/finetune/pets_segmentation.yaml \
  --pretrained experiments/pretrained_models/pets_simclr_backbone.ckpt \
  --data_ratio 0.5
```

## 📊 Dataset Preparation

### Oxford-IIIT Pet Dataset (Default)
```bash
cd experiments
mkdir -p data/{images,annotations}

# Download and extract
wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
tar -xzf images.tar.gz
tar -xzf annotations.tar.gz

# Clean dataset
python tools/data_preprocessing.py --data_dir experiments/data/
```

### Custom Dataset
Organize your data as:
```
experiments/data/
├── images/
│   ├── img1.jpg
│   └── img2.jpg
└── masks/
    ├── img1.png
    └── img2.png
```

## 🏋️ Training

### Self-Supervised Pretraining
```bash
# Pretrain on CIFAR-10
python scripts/pretrain_simclr.py --config configs/pretrain/simclr_cifar10.yaml

# Pretrain on Cats vs Dogs
python scripts/pretrain_simclr.py --config configs/pretrain/simclr_pets.yaml
```

### Segmentation Fine-tuning
```bash
# Fine-tune with different data ratios
python scripts/finetune_segmentation.py \
  --pretrained_model pets \
  --data_ratio 0.2 \
  --loss_function BCE \
  --epochs 60
```

## 📈 Evaluation

```bash
# Evaluate all trained models
python scripts/evaluate_model.py --results_dir experiments/results/

# Generate visualizations
python src/evaluation/visualize.py --model_path path/to/model.pth
```

## 🏆 Model Zoo

| Pretrain Dataset | Fine-tune Ratio | IoU Score | F1 Score | Download |
|------------------|----------------|-----------|----------|----------|
| CIFAR-10        | 80%            | 0.847     | 0.916    | [Link](experiments/pretrained_models/) |
| Cats vs Dogs    | 80%            | 0.851     | 0.919    | [Link](experiments/pretrained_models/) |
| Cats vs Dogs    | 50%            | 0.832     | 0.906    | [Link](experiments/pretrained_models/) |
| Baseline (No PT)| 80%            | 0.823     | 0.902    | [Link](experiments/pretrained_models/) |

## 📊 Results

### Performance Comparison
- **SimCLR Pretraining** improves segmentation performance by ~3-5% IoU
- **Data efficiency**: Achieves 90% of full-data performance with only 50% data
- **Robust across architectures**: Benefits observed with different backbone networks

<!-- ### Training Curves
![Training Results](docs/images/training_curves.png) -->

## 🔧 Configuration

All training parameters are configurable via YAML files in `configs/`. Key parameters:

```yaml
# Example configuration
model:
  backbone: resnet50
  decoder: unet
  pretrained_path: null

training:
  batch_size: 32
  learning_rate: 1e-4
  epochs: 60
  loss_function: BCE

data:
  dataset_ratio: 0.8
  image_size: 224
  augmentations: true
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [SimCLR](https://arxiv.org/abs/2002.05709) for the self-supervised learning framework
- [Segmentation Models PyTorch](https://github.com/qubvel/segmentation_models.pytorch) for model implementations
- Oxford-IIIT Pet Dataset for providing the benchmark dataset