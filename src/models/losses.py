import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class AdaptiveWingLoss(nn.Module):
    """
    Adaptive Wing Loss for heatmap regression
    
    Reference:
    Feng et al. "Wing Loss for Robust Facial Landmark Localisation with 
    Convolutional Neural Networks", CVPR 2018
    
    Wang et al. "Adaptive Wing Loss for Robust Face Alignment via Heatmap 
    Regression", ICCV 2019
    """
    def __init__(self, alpha=2.1, omega=14, epsilon=1, theta=0.5):
        """
        Initialize Adaptive Wing Loss
        
        Args:
            alpha (float): Parameter to control the shape of the loss function
            omega (float): Weight parameter to adjust influence of non-linear part
            epsilon (float): Small constant to avoid numerical issues
            theta (float): Threshold to switch between linear and non-linear part
        """
        super(AdaptiveWingLoss, self).__init__()
        self.alpha = alpha
        self.omega = omega
        self.epsilon = epsilon
        self.theta = theta
        
    def forward(self, pred, target):
        """
        Forward pass of the loss function
        
        Args:
            pred (torch.Tensor): Predicted heatmaps (B, C, H, W)
            target (torch.Tensor): Ground truth heatmaps (B, C, H, W)
            
        Returns:
            torch.Tensor: Computed loss value
        """
        # Calculate the difference
        delta = torch.abs(target - pred)
        
        # Calculate A and C constants according to the paper
        A = self.omega * (1 / (1 + torch.pow(self.theta / self.epsilon, self.alpha - target))) * \
            (self.alpha - target) * torch.pow(self.theta / self.epsilon, self.alpha - target - 1) * (1 / self.epsilon)
        C = (self.theta * A - self.omega * torch.log(1 + torch.pow(self.theta / self.epsilon, self.alpha - target)))
        
        # Apply different loss functions based on the delta value
        losses = torch.where(
            delta < self.theta,
            self.omega * torch.log(1 + torch.pow(delta / self.epsilon, self.alpha - target)),
            A * delta - C
        )
        
        return torch.mean(losses)


class GaussianHeatmapGenerator:
    """
    Generates Gaussian heatmaps from landmark coordinates
    """
    def __init__(self, output_size=(64, 64), sigma=2.0):
        """
        Initialize the heatmap generator
        
        Args:
            output_size (tuple): Size of the output heatmaps (height, width)
            sigma (float): Standard deviation for the Gaussian kernel
        """
        self.output_size = output_size
        self.sigma = sigma
        
        # Pre-compute coordinate meshgrid
        x = np.arange(0, output_size[1], 1)
        y = np.arange(0, output_size[0], 1)
        self.xx, self.yy = np.meshgrid(x, y)
        
    def generate_heatmap(self, landmark_x, landmark_y):
        """
        Generate a single Gaussian heatmap for a landmark
        
        Args:
            landmark_x (float): x-coordinate of the landmark (in heatmap space)
            landmark_y (float): y-coordinate of the landmark (in heatmap space)
            
        Returns:
            np.ndarray: Generated heatmap
        """
        # Calculate the Gaussian
        heatmap = np.exp(-((self.xx - landmark_x) ** 2 + (self.yy - landmark_y) ** 2) / (2 * self.sigma ** 2))
        
        return heatmap
    
    def generate_heatmaps(self, landmarks):
        """
        Generate heatmaps for multiple landmarks
        
        Args:
            landmarks (np.ndarray or torch.Tensor): Landmark coordinates of shape (batch_size, num_landmarks, 2)
            
        Returns:
            torch.Tensor: Generated heatmaps of shape (batch_size, num_landmarks, height, width)
        """
        if isinstance(landmarks, torch.Tensor):
            landmarks = landmarks.detach().cpu().numpy()
        
        batch_size, num_landmarks, _ = landmarks.shape
        heatmaps = np.zeros((batch_size, num_landmarks, self.output_size[0], self.output_size[1]))
        
        # Scale landmarks to heatmap space
        scale_factor = self.output_size[0] / 224.0  # Assuming 224x224 input images
        scaled_landmarks = landmarks * scale_factor
        
        for i in range(batch_size):
            for j in range(num_landmarks):
                x, y = scaled_landmarks[i, j]
                heatmaps[i, j] = self.generate_heatmap(x, y)
        
        return torch.tensor(heatmaps, dtype=torch.float32)


class WingLoss(nn.Module):
    """
    Wing Loss for direct coordinate regression
    
    Reference:
    Feng et al. "Wing Loss for Robust Facial Landmark Localisation with 
    Convolutional Neural Networks", CVPR 2018
    """
    def __init__(self, omega=10, epsilon=2):
        """
        Initialize Wing Loss
        
        Args:
            omega (float): Parameter that controls the width of the non-linear part
            epsilon (float): Parameter that controls the curvature of the non-linear part
        """
        super(WingLoss, self).__init__()
        self.omega = omega
        self.epsilon = epsilon
        
        # Pre-compute constant C
        self.C = self.omega - self.omega * np.log(1 + self.omega / self.epsilon)
        
    def forward(self, pred, target, mask=None):
        """
        Forward pass of the loss function
        
        Args:
            pred (torch.Tensor): Predicted coordinates (B, N, 2) or (B, N*2)
            target (torch.Tensor): Ground truth coordinates (B, N, 2) or (B, N*2)
            mask (torch.Tensor, optional): Mask for valid landmarks of shape (B, N) or (B, N*2)
            
        Returns:
            torch.Tensor: Computed loss value
        """
        # Reshape tensors if needed to be (batch_size, num_points*2)
        if pred.dim() == 3:
            batch_size, num_points, _ = pred.size()
            pred = pred.view(batch_size, -1)
            target = target.view(batch_size, -1)
            if mask is not None and mask.dim() == 2:
                # Repeat mask for x and y coordinates
                mask = mask.repeat(1, 2)
        
        # Calculate Euclidean distance
        delta = torch.abs(target - pred)
        
        # Apply wing loss formula
        loss = torch.where(
            delta < self.omega,
            self.omega * torch.log(1 + delta / self.epsilon),
            delta - self.C
        )
        
        # Apply mask if provided
        if mask is not None:
            loss = loss * mask
            return loss.sum() / (mask.sum() + 1e-8)
        
        return torch.mean(loss)


class CombinedLoss(nn.Module):
    """
    Combined loss function for heatmap and coordinate regression
    
    Combines Adaptive Wing Loss for heatmaps with Wing Loss for coordinates
    """
    def __init__(self, heatmap_weight=1.0, coord_weight=1.0, output_size=(64, 64), image_size=(224, 224)):
        """
        Initialize combined loss
        
        Args:
            heatmap_weight (float): Weight for heatmap loss
            coord_weight (float): Weight for coordinate loss
            output_size (tuple): Size of heatmap output (height, width)
            image_size (tuple): Size of original image (height, width)
        """
        super(CombinedLoss, self).__init__()
        self.heatmap_weight = heatmap_weight
        self.coord_weight = coord_weight
        self.output_size = output_size
        self.image_size = image_size
        
        # Calculate scale factor
        self.scale_factor = image_size[0] / output_size[0]  # Assuming square aspect ratio
        
        self.heatmap_loss = AdaptiveWingLoss()
        self.coord_loss = WingLoss()
    
    def forward(self, pred_dict, target_heatmaps, target_coords, mask=None):
        """
        Forward pass of the loss function
        
        Args:
            pred_dict (dict): Dictionary containing predicted heatmaps and coordinates
            target_heatmaps (torch.Tensor): Ground truth heatmaps
            target_coords (torch.Tensor): Ground truth coordinates in image space (224x224)
            mask (torch.Tensor, optional): Mask for valid landmarks
            
        Returns:
            tuple: (total_loss, heatmap_loss, coord_loss)
        """
        # Compute heatmap loss
        heatmap_loss = self.heatmap_loss(pred_dict['heatmaps'], target_heatmaps)
        
        # Convert target coordinates from image space to heatmap space for loss computation
        target_coords_heatmap = target_coords / self.scale_factor
        
        # Compute coordinate loss if refined coordinates are available
        if 'refined_coords' in pred_dict:
            coord_loss = self.coord_loss(pred_dict['refined_coords'], target_coords_heatmap, mask)
        else:
            # Use initial coordinates if refined are not available
            coord_loss = self.coord_loss(pred_dict['initial_coords'], target_coords_heatmap, mask)
        
        # Compute total loss
        total_loss = self.heatmap_weight * heatmap_loss + self.coord_weight * coord_loss
        
        return total_loss, heatmap_loss, coord_loss


def soft_argmax(heatmaps, beta=100):
    """
    Soft-argmax function for sub-pixel accuracy
    
    Args:
        heatmaps (torch.Tensor): Heatmaps of shape (batch_size, num_landmarks, height, width)
        beta (float): Temperature parameter for softmax
        
    Returns:
        torch.Tensor: Coordinates of shape (batch_size, num_landmarks, 2)
    """
    batch_size, num_landmarks, height, width = heatmaps.shape
    
    # Create coordinate grid - important to use exactly 0 to height-1/width-1
    device = heatmaps.device
    x_range = torch.arange(0, width, dtype=heatmaps.dtype, device=device)
    y_range = torch.arange(0, height, dtype=heatmaps.dtype, device=device)
    
    # Create meshgrid
    y_grid, x_grid = torch.meshgrid(y_range, x_range, indexing='ij')
    
    # Reshape heatmaps for softmax
    flat_heatmaps = heatmaps.reshape(batch_size, num_landmarks, -1)
    
    # Apply softmax with temperature parameter
    weights = F.softmax(beta * flat_heatmaps, dim=-1)
    weights = weights.reshape(batch_size, num_landmarks, height, width)
    
    # Calculate weighted average of coordinates
    x_coords = (weights * x_grid).sum(dim=(-2, -1))
    y_coords = (weights * y_grid).sum(dim=(-2, -1))
    
    # Stack to get coordinates
    coords = torch.stack([x_coords, y_coords], dim=-1)
    
    return coords 