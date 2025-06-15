import torch
import torch.nn as nn
import torch.nn.functional as F
from .losses import nt_xent_loss


class SimCLR(nn.Module):
    """
    SimCLR model for self-supervised contrastive learning.
    
    Parameters:
    - backbone (nn.Module): Backbone network (e.g., ResNet)
    - projection_dim (int): Dimension of projection head output
    - hidden_dim (int): Hidden dimension in projection head
    - temperature (float): Temperature parameter for contrastive loss
    """
    
    def __init__(self, backbone: nn.Module, projection_dim: int = 128, 
                 hidden_dim: int = 512, temperature: float = 0.1):
        super(SimCLR, self).__init__()
        self.backbone = backbone
        self.temperature = temperature
        
        # Get backbone output dimension
        backbone_dim = self._get_backbone_dim()
        
        # Two-layer MLP projection head as described in the paper
        self.projection_head = nn.Sequential(
            nn.Linear(backbone_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim)
        )
        
    def _get_backbone_dim(self):
        """Get the output dimension of the backbone network."""
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            dummy_output = self.backbone(dummy_input)
            return dummy_output.shape[1]
    
    def forward(self, x_i: torch.Tensor, x_j: torch.Tensor):
        """
        Forward pass for SimCLR training.
        
        Parameters:
        - x_i (torch.Tensor): First augmented view
        - x_j (torch.Tensor): Second augmented view
        
        Returns:
        - torch.Tensor: Contrastive loss
        """
        # Extract features using backbone
        h_i = self.backbone(x_i)
        h_j = self.backbone(x_j)
        
        # Apply projection head
        z_i = self.projection_head(h_i)
        z_j = self.projection_head(h_j)
        
        # Compute contrastive loss
        loss = nt_xent_loss(z_i, z_j, self.temperature)
        return loss
    
    def get_representations(self, x: torch.Tensor):
        """
        Get backbone representations without projection head.
        
        Parameters:
        - x (torch.Tensor): Input images
        
        Returns:
        - torch.Tensor: Backbone features
        """
        return self.backbone(x)