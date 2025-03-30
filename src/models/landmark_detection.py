import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights, EfficientNet_B3_Weights
import numpy as np

class LandmarkDetectionModel(nn.Module):
    """
    Neural network model for cephalometric landmark detection.
    Uses a pretrained backbone with custom regression head for landmark prediction.
    """
    
    def __init__(self, num_landmarks, backbone='resnet50', pretrained=True):
        """
        Initialize the model
        
        Args:
            num_landmarks (int): Number of landmarks to predict (each with x,y coordinates)
            backbone (str): Name of the backbone architecture ('resnet50' or 'efficientnet')
            pretrained (bool): Whether to use pretrained weights for the backbone
        """
        super(LandmarkDetectionModel, self).__init__()
        
        self.num_landmarks = num_landmarks
        
        # Create the backbone
        if backbone == 'resnet50':
            if pretrained:
                weights = ResNet50_Weights.DEFAULT
            else:
                weights = None
                
            self.backbone = models.resnet50(weights=weights)
            self.feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove the final FC layer
            
        elif backbone == 'efficientnet':
            if pretrained:
                weights = EfficientNet_B3_Weights.DEFAULT
            else:
                weights = None
                
            self.backbone = models.efficientnet_b3(weights=weights)
            self.feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()  # Remove the classifier
            
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Create the landmark regression head
        self.landmark_head = nn.Sequential(
            nn.Linear(self.feature_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_landmarks * 2)  # 2 coordinates (x,y) per landmark
        )
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Predicted landmarks of shape (batch_size, num_landmarks*2)
        """
        # Extract features using the backbone
        features = self.backbone(x)
        
        # Predict landmark coordinates
        landmarks = self.landmark_head(features)
        
        # Reshape to (batch_size, num_landmarks, 2)
        landmarks = landmarks.view(-1, self.num_landmarks, 2)
        
        return landmarks


def create_model(landmark_cols, backbone='resnet50', pretrained=True):
    """
    Create a landmark detection model
    
    Args:
        landmark_cols (list): List of landmark column names
        backbone (str): Name of the backbone architecture
        pretrained (bool): Whether to use pretrained weights
        
    Returns:
        LandmarkDetectionModel: The created model
    """
    # Calculate number of landmarks
    num_landmarks = len(landmark_cols) // 2  # Each landmark has x and y coordinates
    
    # Create model
    model = LandmarkDetectionModel(
        num_landmarks=num_landmarks,
        backbone=backbone,
        pretrained=pretrained
    )
    
    return model


class LandmarkLoss(nn.Module):
    """
    Custom loss function for landmark detection
    Combines MSE loss with weighted penalties for certain important landmarks
    """
    
    def __init__(self, landmark_weights=None):
        """
        Initialize the loss function
        
        Args:
            landmark_weights (dict, optional): Dictionary mapping landmark indices to weight values
        """
        super(LandmarkLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='none')
        self.landmark_weights = landmark_weights or {}
    
    def forward(self, pred, target, mask=None):
        """
        Compute the loss
        
        Args:
            pred (torch.Tensor): Predicted landmarks of shape (batch_size, num_landmarks, 2)
            target (torch.Tensor): Target landmarks of shape (batch_size, num_landmarks, 2)
            mask (torch.Tensor, optional): Mask for valid landmarks of shape (batch_size, num_landmarks)
            
        Returns:
            torch.Tensor: Computed loss value
        """
        # Compute MSE loss for each landmark coordinate
        loss = self.mse_loss(pred, target)  # Shape: (batch_size, num_landmarks, 2)
        
        # Apply landmark-specific weights
        if self.landmark_weights:
            weights = torch.ones_like(loss)
            for idx, weight in self.landmark_weights.items():
                weights[:, idx, :] = weight
            loss = loss * weights
        
        # Apply mask for missing landmarks if provided
        if mask is not None:
            # Expand mask to match loss dimensions
            mask = mask.unsqueeze(-1).expand_as(loss)
            loss = loss * mask
            
            # Compute average over valid landmarks only
            return loss.sum() / (mask.sum() + 1e-8)
        
        # Compute average over all landmarks
        return loss.mean() 