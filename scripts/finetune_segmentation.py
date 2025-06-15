#!/usr/bin/env python3
"""
Segmentation Fine-tuning Script

This script handles fine-tuning pretrained SimCLR models for semantic segmentation.
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.datasets import OxfordPetsDataset
from data.transforms import get_segmentation_transforms, get_mask_transforms
from data.data_utils import data_split


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config, device):
    """Create segmentation model with optional pretrained backbone."""
    model = smp.Unet(
        encoder_name=config['model']['backbone'],
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )
    
    # Load pretrained weights if specified
    if config['model'].get('pretrained_path'):
        if os.path.exists(config['model']['pretrained_path']):
            print(f"Loading pretrained weights from {config['model']['pretrained_path']}")
            model.load_state_dict(
                torch.load(config['model']['pretrained_path'], map_location=device), 
                strict=False
            )
        else:
            print(f"Warning: Pretrained weights not found at {config['model']['pretrained_path']}")
    
    return model.to(device)


def train_segmentation_model(model, train_loader, val_loader, config, device):
    """Train segmentation model."""
    # Setup optimizer
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=config['training']['learning_rate']
    )
    
    # Setup loss function
    loss_fn_map = {
        'BCE': nn.BCEWithLogitsLoss(),
        'Dice': smp.losses.DiceLoss(mode='binary'),
        'Focal': smp.losses.FocalLoss(mode='binary')
    }
    criterion = loss_fn_map[config['training']['loss_function']]
    
    # Metrics for evaluation
    dice_loss_fn = smp.losses.DiceLoss(mode='binary')
    focal_loss_fn = smp.losses.FocalLoss(mode='binary')
    bce_loss_fn = nn.BCEWithLogitsLoss()
    
    best_iou = 0
    save_dir = os.path.join(config['experiment']['save_dir'], 'models')
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(config['training']['epochs']):
        # Training phase
        model.train()
        running_loss = 0.0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.float().to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{config["training"]["epochs"]}, '
                      f'Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}')
        
        avg_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch+1} Training Loss: {avg_loss:.4f}')
        
        # Evaluation phase
        if (epoch + 1) % config['training']['eval_frequency'] == 0:
            model.eval()
            iou_scores, f1_scores = [], []
            
            with torch.no_grad():
                for images, masks in val_loader:
                    images, masks = images.to(device), masks.float().to(device)
                    outputs = model(images)
                    
                    # Compute metrics
                    tp, fp, fn, tn = smp.metrics.get_stats(
                        masks.long(), outputs.long(), mode='binary', threshold=0.5
                    )
                    iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
                    f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
                    
                    iou_scores.append(iou_score.item())
                    f1_scores.append(f1_score.item())
            
            mean_iou = sum(iou_scores) / len(iou_scores)
            mean_f1 = sum(f1_scores) / len(f1_scores)
            
            print(f'Epoch {epoch+1} - IoU: {mean_iou:.4f}, F1: {mean_f1:.4f}')
            
            # Save best model
            if mean_iou > best_iou:
                best_iou = mean_iou
                model_path = os.path.join(save_dir, f"{config['experiment']['name']}_best.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'iou': mean_iou,
                    'f1': mean_f1
                }, model_path)
                print(f'Best model saved with IoU: {best_iou:.4f}')


def main():
    parser = argparse.ArgumentParser(description='Segmentation Fine-tuning')
    parser.add_argument('--config', type=str, 
                       help='Path to configuration file')
    parser.add_argument('--pretrained_model', type=str, choices=['pets', 'cifar', 'baseline'],
                       help='Pretrained model type')
    parser.add_argument('--data_ratio', type=float, default=0.8,
                       help='Ratio of data to use for training')
    parser.add_argument('--loss_function', type=str, default='BCE',
                       choices=['BCE', 'Dice', 'Focal'],
                       help='Loss function to use')
    parser.add_argument('--epochs', type=int, default=60,
                       help='Number of training epochs')
    
    args = parser.parse_args()
    
    # Load config or create from args
    if args.config:
        config = load_config(args.config)
    else:
        # Create config from command line args
        config = {
            'model': {
                'backbone': 'resnet50',
                'decoder': 'unet',
                'pretrained_path': None if args.pretrained_model == 'baseline' 
                                 else f'experiments/pretrained_models/{args.pretrained_model}_simclr_backbone.ckpt'
            },
            'training': {
                'batch_size': 32,
                'learning_rate': 1e-4,
                'epochs': args.epochs,
                'loss_function': args.loss_function,
                'eval_frequency': 2
            },
            'data': {
                'images_dir': 'experiments/data/images',
                'masks_dir': 'experiments/data/annotations/trimaps',
                'data_ratio': args.data_ratio,
                'image_size': 224
            },
            'experiment': {
                'name': f'{args.pretrained_model}_{args.loss_function}_{int(args.data_ratio*100)}',
                'save_dir': 'experiments/results'
            }
        }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset
    transform = get_segmentation_transforms(config['data']['image_size'])
    mask_transform = get_mask_transforms(config['data']['image_size'])
    
    dataset = OxfordPetsDataset(
        images_dir=config['data']['images_dir'],
        masks_dir=config['data']['masks_dir'],
        transform=transform,
        mask_transform=mask_transform
    )
    
    # Split data based on ratio
    ignore_ratio = 1.0 - config['data']['data_ratio']
    train_loader, test_loader, val_loader = data_split(
        dataset, ignore_size=ignore_ratio, batch_size=config['training']['batch_size']
    )
    
    # Create and train model
    model = create_model(config, device)
    train_segmentation_model(model, train_loader, val_loader, config, device)


if __name__ == '__main__':
    main()