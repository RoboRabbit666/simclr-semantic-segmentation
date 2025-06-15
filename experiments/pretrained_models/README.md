# Pretrained Models

This directory contains pretrained SimCLR backbone models that can be used for segmentation fine-tuning.

## Available Models

### SimCLR Pretrained Backbones

| Model | Dataset | Architecture | Epochs | Download Size |
|-------|---------|-------------|---------|---------------|
| `pets_simclr_backbone.ckpt` | Cats vs Dogs | ResNet-50 | 20 | ~90MB |
| `cifar_simclr_backbone.ckpt` | CIFAR-10 | ResNet-50 | 20 | ~90MB |

## Model Details

### pets_simclr_backbone.ckpt
- **Pretraining Dataset**: Cats vs Dogs (HuggingFace)
- **Architecture**: ResNet-50 without final classification layer
- **Training**: SimCLR framework with NT-Xent loss
- **Augmentations**: Random crops, color jittering, Gaussian blur
- **Optimizer**: LARS with learning rate 0.3
- **Batch Size**: 120 with gradient accumulation

### cifar_simclr_backbone.ckpt
- **Pretraining Dataset**: CIFAR-10 (subset of 23,402 images)
- **Architecture**: ResNet-50 without final classification layer
- **Training**: SimCLR framework with NT-Xent loss
- **Augmentations**: Random crops, color jittering, Gaussian blur
- **Optimizer**: LARS with learning rate 0.3
- **Batch Size**: 120 with gradient accumulation

## Usage

To use these pretrained models in fine-tuning:

```python
import torch
from src.models.backbones import load_pretrained_backbone

# Load pretrained backbone
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = create_segmentation_model()  # Your segmentation model
model = load_pretrained_backbone(
    model, 
    'experiments/pretrained_models/pets_simclr_backbone.ckpt',
    device,
    strict=False
)
```

Or use the command line scripts:

```bash
# Fine-tune using pets pretrained model
python scripts/finetune_segmentation.py \
  --pretrained_model pets \
  --data_ratio 0.8 \
  --loss_function BCE

# Fine-tune using CIFAR pretrained model
python scripts/finetune_segmentation.py \
  --pretrained_model cifar \
  --data_ratio 0.8 \
  --loss_function BCE
```

## Performance

Based on segmentation fine-tuning on Oxford-IIIT Pet Dataset:

| Pretrained Model | Data Ratio | IoU Score | F1 Score | Improvement |
|------------------|------------|-----------|----------|-------------|
| pets_simclr_backbone | 80% | 0.851 | 0.919 | +2.8% IoU |
| cifar_simclr_backbone | 80% | 0.847 | 0.916 | +2.4% IoU |
| Baseline (no pretraining) | 80% | 0.823 | 0.902 | - |

## Downloading Models

If models are not included in the repository, you can:

1. Train your own using the pretraining scripts
2. Download from [model release page] (if available)
3. Contact the authors for access

## Model Format

All models are saved as PyTorch state dictionaries (`.ckpt` files) containing only the backbone network weights (ResNet-50 without the final classification layer).