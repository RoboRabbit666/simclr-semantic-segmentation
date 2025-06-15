#!/usr/bin/env python3
"""
SimCLR Pretraining Script

This script handles self-supervised pretraining using SimCLR framework.
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from datasets import load_dataset

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.simclr import SimCLR
from models.backbones import get_resnet_backbone
from training.optimizers import LARS
from data.datasets import SimClrData, preprocess
from data.transforms import get_simclr_transforms


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_dataset(config, device):
    """Create dataset based on configuration."""
    if config['data']['dataset'] == 'cats_vs_dogs':
        # Load Cats vs Dogs dataset from HuggingFace
        dataset = load_dataset("cats_vs_dogs")['train']
        dataset = preprocess(dataset, device=device)
        transforms_fn = get_simclr_transforms(config['data']['image_size'])
        dataset = SimClrData(dataset, transforms=transforms_fn)
    
    elif config['data']['dataset'] == 'cifar10':
        # Load CIFAR-10 dataset
        transform = transforms.Compose([
            transforms.Resize((config['data']['image_size'], config['data']['image_size'])),
            transforms.ToTensor()
        ])
        dataset = datasets.CIFAR10(root='experiments/data/', train=True, 
                                 transform=transform, download=True)
        
        # Subset if specified
        if 'subset_size' in config['data']:
            subset_size = config['data']['subset_size']
            dataset = torch.utils.data.Subset(dataset, range(subset_size))
    
    else:
        raise ValueError(f"Unsupported dataset: {config['data']['dataset']}")
    
    return dataset


def train_simclr(config):
    """Main training function for SimCLR."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset
    dataset = create_dataset(config, device)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=True, 
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=False, 
        num_workers=4
    )
    
    # Create model
    backbone = get_resnet_backbone(config['model']['backbone'], pretrained=False)
    model = SimCLR(
        backbone=backbone,
        projection_dim=config['model']['projection_dim'],
        hidden_dim=config['model']['hidden_dim'],
        temperature=config['model']['temperature']
    )
    model = model.to(device)
    
    # Create optimizer
    if config['training']['optimizer'] == 'LARS':
        optimizer = LARS(
            [params for params in model.parameters() if params.requires_grad],
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay'],
            exclude_from_weight_decay=["batch_normalization", "bias"],
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config['training']['learning_rate']
        )
    
    # Training loop
    best_loss = float('inf')
    os.makedirs(config['experiment']['save_dir'], exist_ok=True)
    
    for epoch in range(config['training']['epochs']):
        model.train()
        total_loss = 0
        
        for batch_idx, (images_1, images_2) in enumerate(train_loader):
            images_1, images_2 = images_1.to(device), images_2.to(device)
            
            optimizer.zero_grad()
            loss = model(images_1, images_2)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{config["training"]["epochs"]}, '
                      f'Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1} Average Loss: {avg_loss:.4f}')
        
        # Save checkpoint
        if (epoch + 1) % config['experiment']['save_frequency'] == 0:
            checkpoint_path = os.path.join(
                config['experiment']['save_dir'], 
                f"{config['experiment']['name']}_epoch_{epoch+1}.ckpt"
            )
            torch.save(model.backbone.state_dict(), checkpoint_path)
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(
                config['experiment']['save_dir'], 
                f"{config['experiment']['name']}_best.ckpt"
            )
            torch.save(model.backbone.state_dict(), best_path)


def main():
    parser = argparse.ArgumentParser(description='SimCLR Pretraining')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    args = parser.parse_args()
    
    config = load_config(args.config)
    train_simclr(config)


if __name__ == '__main__':
    main()