import torch
import torch.nn as nn
import torch.nn.functional as F


def nt_xent_loss(queries: torch.Tensor, keys: torch.Tensor, 
                 temperature: float = 0.1) -> torch.Tensor:
    """
    Normalized Temperature-scaled Cross Entropy Loss for SimCLR.
    
    Parameters:
    - queries (torch.Tensor): Query representations (batch_size, embedding_dim)
    - keys (torch.Tensor): Key representations (batch_size, embedding_dim)
    - temperature (float): Temperature scaling parameter
    
    Returns:
    - torch.Tensor: NT-Xent loss value
    """
    b, device = queries.shape[0], queries.device
    n = b * 2
    
    # Concatenate queries and keys
    projs = torch.cat((queries, keys))
    
    # Compute similarity matrix
    logits = projs @ projs.t()
    
    # Remove self-similarities (diagonal)
    mask = torch.eye(n, device=device).bool()
    logits = logits[~mask].reshape(n, n - 1)
    logits /= temperature
    
    # Create labels: each query should be similar to its corresponding key
    labels = torch.cat((
        (torch.arange(b, device=device) + b - 1), 
        torch.arange(b, device=device)
    ), dim=0)
    
    # Compute cross entropy loss
    loss = F.cross_entropy(logits, labels, reduction='sum')
    loss /= n
    
    return loss


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation tasks.
    
    Parameters:
    - smooth (float): Smoothing factor to avoid division by zero
    - reduction (str): Reduction method ('mean', 'sum', or 'none')
    """
    
    def __init__(self, smooth: float = 1e-6, reduction: str = 'mean'):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Dice loss.
        
        Parameters:
        - predictions (torch.Tensor): Model predictions
        - targets (torch.Tensor): Ground truth targets
        
        Returns:
        - torch.Tensor: Dice loss
        """
        # Apply sigmoid to predictions if needed
        if predictions.max() > 1:
            predictions = torch.sigmoid(predictions)
        
        # Flatten tensors
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        # Compute Dice coefficient
        intersection = (predictions * targets).sum()
        dice = (2. * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)
        
        # Return Dice loss (1 - Dice coefficient)
        loss = 1 - dice
        
        if self.reduction == 'mean':
            return loss
        elif self.reduction == 'sum':
            return loss * predictions.numel()
        else:
            return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in segmentation.
    
    Parameters:
    - alpha (float): Weighting factor for rare class
    - gamma (float): Focusing parameter
    - reduction (str): Reduction method ('mean', 'sum', or 'none')
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Focal loss.
        
        Parameters:
        - predictions (torch.Tensor): Model predictions
        - targets (torch.Tensor): Ground truth targets
        
        Returns:
        - torch.Tensor: Focal loss
        """
        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(predictions, targets, reduction='none')
        
        # Compute probabilities
        p_t = torch.exp(-bce_loss)
        
        # Compute focal loss
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CombinedLoss(nn.Module):
    """
    Combined loss function for segmentation (e.g., Dice + BCE).
    
    Parameters:
    - dice_weight (float): Weight for Dice loss
    - bce_weight (float): Weight for BCE loss
    """
    
    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5):
        super(CombinedLoss, self).__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute combined loss.
        
        Parameters:
        - predictions (torch.Tensor): Model predictions
        - targets (torch.Tensor): Ground truth targets
        
        Returns:
        - torch.Tensor: Combined loss
        """
        dice = self.dice_loss(predictions, targets)
        bce = self.bce_loss(predictions, targets)
        
        return self.dice_weight * dice + self.bce_weight * bce