# Semantic Segmentation using Self-Supervised Learning (SimCLR)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive PyTorch implementation exploring the effectiveness of **SimCLR self-supervised pretraining** for semantic segmentation tasks. This research project demonstrates how contrastive learning can improve data efficiency and performance in dense prediction tasks through systematic experimentation on the Oxford-IIIT Pet Dataset.

## Project Structure

```
simclr-semantic-segmentation/
├── configs/                     # Experiment configurations
│   ├── pretrain/               # SimCLR pretraining configs
│   │   ├── simclr_cifar10.yaml
│   │   └── simclr_pets.yaml
│   └── finetune/               # Segmentation configs
│       └── pets_segmentation.yaml
├── src/                        # Source code
│   ├── models/                 # Model implementations
│   │   ├── simclr.py          # SimCLR framework
│   │   ├── backbones.py       # ResNet architectures
│   │   └── losses.py          # Loss functions (BCE, Dice, Focal)
│   ├── data/                  # Data handling
│   │   ├── datasets.py        # Dataset classes
│   │   └── transforms.py      # Augmentation pipelines
│   └── training/              # Training utilities
├── scripts/                   # Training scripts
│   ├── pretrain_simclr.py     # Self-supervised pretraining
│   └── finetune_segmentation.py # Segmentation fine-tuning
├── experiments/               # Results and models
│   ├── pretrained_models/     # Saved checkpoints
│   └── results/              # Training logs and metrics
└── docs/                     # Documentation and reports
    └── Project_Report.pdf     # Full research report
```

## Research Overview

This project investigates two key research questions:

1. **Self-Supervised vs Supervised**: Can SimCLR pretraining improve semantic segmentation performance compared to fully supervised baselines?
2. **Dataset Similarity Impact**: How similar do pretraining and fine-tuning datasets need to be for optimal segmentation performance?

### Key Findings
- SimCLR pretraining achieves **comparable performance** (0.90 IoU) to fully supervised baselines
- **Data efficiency**: Pre-trained models maintain performance with significantly less labelled data
- **Domain similarity**: Pretraining dataset similarity has minimal impact on final segmentation performance
- **Scalability**: Model performance benefits significantly from larger fine-tuning dataset sizes

**Full Project Report**: For detailed methodology, extended analysis, and comprehensive results, see the complete project report [here](docs/Project_Report.pdf).

## Methodology

### Two-Stage Learning Pipeline

#### Stage 1: Self-Supervised Pretraining (SimCLR)
- **Backbone**: ResNet-50 (trained from scratch, no ImageNet initialisation)
- **Projection Head**: 2-layer MLP (2048 → 512 → 128) with ReLU activation
- **Data Augmentation**: Random cropping/resizing, colour jittering, Gaussian blur
- **Loss Function**: NT-Xent (Normalized Temperature-scaled Cross Entropy, τ=0.1)
- **Optimiser**: LARS with learning rate 0.3, momentum 0.9
- **Training**: 20 epochs (GPU resource constrained)

#### Stage 2: Supervised Fine-tuning
- **Architecture**: U-Net decoder with pretrained ResNet-50 encoder
- **Task**: Binary semantic segmentation (foreground/background)
- **Loss Function**: Binary Cross Entropy (BCE)
- **Optimizer**: Adam with learning rate 1e-4
- **Training**: 60 epochs with validation monitoring

## Experimental Design

### Datasets Used

#### Pretraining Datasets
- **CIFAR-10**: 60,000 diverse images across 10 classes (animals + objects)
- **Cats & Dogs**: 25,000 domain-specific animal images from HuggingFace

#### Fine-tuning Dataset  
- **Oxford-IIIT Pet Dataset**: 7,390 images with trimap annotations
- **Task**: Binary segmentation (pet vs background)
- **Preprocessing**: Trimaps converted to binary masks

### Experimental Comparisons

#### Experiment 1: Self-Supervised vs Baseline
- **Baseline**: U-Net with ResNet-50 (fully supervised)
- **SimCLR Models**: CIFAR-10 pretrained vs Cats&Dogs pretrained
- **Data Split**: 80% development / 20% test
- **Evaluation**: Same metrics across all models for fair comparison

#### Experiment 2: Data Efficiency Analysis
- **Dataset Ratios**: 80%, 50%, 20% of development data
- **Fixed Test Set**: 20% held constant across all experiments
- **Purpose**: Evaluate pretraining benefits with limited labelled data

#### Experiment 3: Domain Similarity Investigation
- **CIFAR-10 Pretraining**: General visual features across diverse object categories
- **Cats&Dogs Pretraining**: Domain-specific animal features
- **Research Question**: Impact of pretraining-finetuning domain similarity

## Results & Performance

### Model Performance Summary

| Model | Pretraining | Dev Split | IoU | F1 Score | Accuracy | Recall |
|-------|-------------|-----------|-----|----------|----------|--------|
| Baseline | None | 80% | 0.906 | 0.951 | 0.961 | 0.942 |
| SimCLR | CIFAR-10 | 80% | 0.900 | 0.947 | 0.958 | 0.937 |
| SimCLR | Cats&Dogs | 80% | 0.902 | 0.948 | 0.958 | 0.929 |
| Baseline | None | 50% | 0.828 | 0.906 | 0.925 | 0.904 |
| SimCLR | CIFAR-10 | 50% | 0.852 | 0.920 | 0.935 | 0.903 |
| SimCLR | Cats&Dogs | 50% | 0.843 | 0.915 | 0.932 | 0.909 |
| Baseline | None | 20% | 0.758 | 0.862 | 0.893 | 0.867 |
| SimCLR | CIFAR-10 | 20% | 0.754 | 0.860 | 0.890 | 0.871 |
| SimCLR | Cats&Dogs | 20% | 0.759 | 0.863 | 0.892 | 0.867 |

### Key Research Findings

1. **Comparable Performance**: SimCLR achieves similar results to fully supervised methods (0.90 IoU)
2. **Data Efficiency Benefits**: Self-supervised pretraining shows advantages with limited labeled data
3. **Minimal Domain Impact**: CIFAR-10 vs Cats&Dogs pretraining shows negligible performance difference
4. **Dataset Size Dependency**: Performance scales significantly with fine-tuning dataset size

## Installation & Setup

### System Requirements
- Python 3.8+
- CUDA-capable GPU (A100 recommended for pretraining)
- ~6GB GPU memory for pretraining
- ~4GB GPU memory for fine-tuning

### Installation
```bash
# Clone repository
git clone <repository-url>
cd simclr-semantic-segmentation

# Create environment
conda env create -f environment.yml
conda activate simclr-segmentation

# Install dependencies
pip install -r requirements.txt
```

### Core Dependencies
```
torch>=1.9.0
torchvision>=0.10.0
segmentation-models-pytorch
datasets
PyYAML
wandb
matplotlib
seaborn
```

## Usage

### Reproduce Research Experiments

#### 1. Pretrain SimCLR Models
```bash
# Pretrain on CIFAR-10
python scripts/pretrain_simclr.py --config configs/pretrain/simclr_cifar10.yaml

# Pretrain on Cats & Dogs
python scripts/pretrain_simclr.py --config configs/pretrain/simclr_pets.yaml
```

#### 2. Fine-tune for Segmentation
```bash
# Full dataset fine-tuning
python scripts/finetune_segmentation.py \
  --config configs/finetune/pets_segmentation.yaml \
  --pretrained experiments/pretrained_models/pet_simclr_backbone.ckpt \
  --data_ratio 0.8

# Data efficiency experiments
for ratio in 0.2 0.5 0.8; do
  python scripts/finetune_segmentation.py \
    --config configs/finetune/pets_segmentation.yaml \
    --pretrained experiments/pretrained_models/cifar_simclr_backbone.ckpt \
    --data_ratio $ratio
done
```

#### 3. Baseline Comparison
```bash
# Train fully supervised baseline
python scripts/finetune_segmentation.py \
  --config configs/finetune/pets_segmentation.yaml \
  --data_ratio 0.8
  # (no --pretrained flag for baseline)
```

## Configuration

### SimCLR Pretraining Configuration
```yaml
model:
  backbone: resnet50
  projection_dim: 128
  temperature: 0.1

training:
  batch_size: 120
  learning_rate: 0.3
  epochs: 20
  optimizer: lars
  weight_decay: 1e-6

data:
  dataset: cats_vs_dogs  # or cifar10
  image_size: 224
  augmentations: simclr_strong
```

### Segmentation Fine-tuning Configuration
```yaml
model:
  encoder: resnet50
  decoder: unet
  pretrained_path: experiments/pretrained_models/pet_simclr_backbone.ckpt

training:
  batch_size: 32
  learning_rate: 1e-4
  epochs: 60
  loss_function: bce

data:
  dataset_ratio: 0.8  # 80%, 50%, or 20%
  split_ratios: [0.9, 0.1]  # train/val within dev set
  test_ratio: 0.2  # fixed test set
```

## Evaluation Metrics

The project uses comprehensive evaluation metrics:
- **IoU (Intersection over Union)**: Primary segmentation metric
- **F1 Score**: Harmonic mean of precision and recall  
- **Dice Coefficient**: Alternative overlap measure
- **Focal Loss**: Handles class imbalance
- **BCE Loss**: Binary classification loss
- **Pixel Accuracy**: Simple classification accuracy
- **Recall**: True positive rate


### Key References
- [SimCLR Paper](https://arxiv.org/abs/2002.05709) - Original contrastive learning framework
- [U-Net Paper](https://arxiv.org/abs/1505.04597) - Segmentation architecture
- [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) - Benchmark dataset

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Academic Research**: UCL CV coursework
- **Computational Resources**: A100 GPU access for pretraining experiments
- **Open Source Libraries**: PyTorch, segmentation-models-pytorch, HuggingFace datasets
- **Dataset Providers**: Oxford Visual Geometry Group, CIFAR-10 contributors
