import torch
import torch.nn as nn
from torchvision.models import resnet50, resnet18, resnet34, resnet101
from typing import Optional


def get_resnet_backbone(arch: str = 'resnet50', pretrained: bool = False) -> nn.Module:
    """
    Get ResNet backbone for feature extraction.
    
    Parameters:
    - arch (str): ResNet architecture ('resnet18', 'resnet34', 'resnet50', 'resnet101')
    - pretrained (bool): Whether to use ImageNet pretrained weights
    
    Returns:
    - nn.Module: ResNet backbone without final classification layer
    """
    if arch == 'resnet18':
        model = resnet18(pretrained=pretrained)
    elif arch == 'resnet34':
        model = resnet34(pretrained=pretrained)
    elif arch == 'resnet50':
        model = resnet50(pretrained=pretrained)
    elif arch == 'resnet101':
        model = resnet101(pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported architecture: {arch}")
    
    # Remove final classification layer
    model.fc = nn.Identity()
    return model


class ResNetBackbone(nn.Module):
    """
    ResNet backbone wrapper for consistent interface.
    
    Parameters:
    - arch (str): ResNet architecture
    - pretrained (bool): Whether to use pretrained weights
    - freeze_backbone (bool): Whether to freeze backbone parameters
    """
    
    def __init__(self, arch: str = 'resnet50', pretrained: bool = False, 
                 freeze_backbone: bool = False):
        super(ResNetBackbone, self).__init__()
        self.backbone = get_resnet_backbone(arch, pretrained)
        
        if freeze_backbone:
            self._freeze_backbone()
    
    def _freeze_backbone(self):
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def load_pretrained_backbone(model: nn.Module, checkpoint_path: str, 
                           device: torch.device, strict: bool = False) -> nn.Module:
    """
    Load pretrained weights into backbone model.
    
    Parameters:
    - model (nn.Module): Model to load weights into
    - checkpoint_path (str): Path to checkpoint file
    - device (torch.device): Device to load model on
    - strict (bool): Whether to strictly enforce matching keys
    
    Returns:
    - nn.Module: Model with loaded weights
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Load weights
    model.load_state_dict(state_dict, strict=strict)
    return model