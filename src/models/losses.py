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
    def __init__(self, omega=10, epsilon=2,
                 target_landmark_indices=None, landmark_weights=None,
                 use_loss_normalization=False, norm_decay=0.99, norm_epsilon=1e-6):
        """
        Initialize Wing Loss
        
        Args:
            omega (float): Parameter that controls the width of the non-linear part
            epsilon (float): Parameter that controls the curvature of the non-linear part
            target_landmark_indices (list, optional): List of landmark indices to compute loss for.
            landmark_weights (torch.Tensor, optional): Tensor of weights (shape [N]) for each landmark.
            use_loss_normalization (bool): Whether loss normalization is active.
            norm_decay (float): Decay factor if normalization is used.
            norm_epsilon (float): Epsilon if normalization is used.
        """
        super(WingLoss, self).__init__()
        self.omega = omega
        self.epsilon = epsilon
        self.target_landmark_indices = target_landmark_indices
        self.landmark_weights = landmark_weights
        
        # Pre-compute constant C
        self.C = self.omega - self.omega * np.log(1 + self.omega / self.epsilon)
        
        # --- Normalization setup ---
        self.use_loss_normalization = use_loss_normalization
        self.norm_decay = norm_decay
        self.norm_epsilon = norm_epsilon
        if use_loss_normalization:
            # Use register_buffer for state that should be saved but not trained
            self.register_buffer('running_avg_loss', torch.tensor(1.0))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        # --- End Normalization setup ---

    def _apply_weights_and_normalize(self, losses):
        """ Helper to apply weights and normalize (if enabled) before mean. Assumes losses shape (B, N, 2) """
        B, N, _ = losses.shape
        
        # 1. Apply per-landmark weights
        current_weights = self.landmark_weights
        if current_weights is not None:
            # Ensure weights are on the same device as losses
            current_weights = current_weights.to(losses.device)
            # Filter weights if target indices are specified
            if self.target_landmark_indices is not None:
                 # Ensure indices are valid before indexing weights
                 num_original_landmarks = current_weights.shape[0] # Get original number of landmarks
                 valid_indices = [idx for idx in self.target_landmark_indices if 0 <= idx < num_original_landmarks]
                 if len(valid_indices) > 0:
                     current_weights = current_weights[valid_indices]
                 else: # If no valid indices, weights don't matter, but avoid error
                     current_weights = torch.ones(N, device=losses.device) # Weights for the filtered landmarks

            # Check if the filtered weights shape matches the current landmark dimension N
            if current_weights.shape[0] != N:
                 # This case shouldn't happen if filtering logic is correct in forward, but good to check
                 print(f"Warning: WingLoss weight dimension mismatch after filtering. Expected {N}, got {current_weights.shape[0]}. Using ones.")
                 current_weights = torch.ones(N, device=losses.device)
            
            # Reshape weights for broadcasting: [1, N, 1]
            weights_reshaped = current_weights.view(1, N, 1)
            losses = losses * weights_reshaped # Apply to both x and y
            
        # 2. Normalize (if enabled) - applied to the weighted loss
        if self.use_loss_normalization:
            # Calculate batch mean of weighted loss
            batch_mean_loss = torch.mean(losses)
            
            # Update running average (using EMA)
            if self.training:
                self.num_batches_tracked += 1
                self.running_avg_loss = (self.norm_decay * self.running_avg_loss + 
                                        (1 - self.norm_decay) * batch_mean_loss.detach())
            
            # Normalize losses: Divide by the running average + epsilon
            normalization_factor = self.running_avg_loss + self.norm_epsilon
            losses = losses / normalization_factor
            
        return losses

    def forward(self, pred, target, mask=None):
        """
        Forward pass of the loss function
        
        Args:
            pred (torch.Tensor): Predicted coordinates (B, N, 2)
            target (torch.Tensor): Ground truth coordinates (B, N, 2)
            mask (torch.Tensor, optional): Mask for valid landmarks (B, N). 
                                           If used, weights are applied BEFORE masking.
            
        Returns:
            torch.Tensor: Computed loss value (scalar mean)
        """
        # Ensure input shapes are (B, N, 2)
        if pred.dim() != 3 or pred.shape[-1] != 2:
            raise ValueError(f"WingLoss expects input shape (B, N, 2), but got pred: {pred.shape}")
        if target.dim() != 3 or target.shape[-1] != 2:
            raise ValueError(f"WingLoss expects input shape (B, N, 2), but got target: {target.shape}")
            
        num_landmarks_pred = pred.shape[1]

        # --- Filter predictions and targets based on indices --- 
        local_pred = pred
        local_target = target
        valid_indices = None # Keep track of indices used for filtering mask
        
        if self.target_landmark_indices is not None:
            valid_indices = [idx for idx in self.target_landmark_indices if 0 <= idx < num_landmarks_pred]
            if len(valid_indices) == 0:
                return torch.tensor(0.0, device=pred.device, requires_grad=True)
            
            local_pred = pred[:, valid_indices, :]
            local_target = target[:, valid_indices, :]
            # Note: We don't filter self.landmark_weights here, the helper _apply_weights_and_normalize does it.
        # --- End Filtering ---

        # Calculate coordinate differences (shape B, N_filtered, 2)
        delta = torch.abs(local_target - local_pred)
        
        # Apply wing loss formula element-wise (per coordinate)
        losses = torch.where(
            delta < self.omega,
            self.omega * torch.log(1 + delta / self.epsilon),
            delta - self.C
        )
        
        # --- Apply weights and normalize (BEFORE mask, if any) --- 
        losses = self._apply_weights_and_normalize(losses)
        # --- End Apply weights and normalize --- 
        
        # Apply mask if provided (AFTER weighting/normalization)
        if mask is not None:
            if mask.dim() == 2: # Mask shape (B, N_original)
                # Filter mask if needed based on valid_indices
                if valid_indices is not None:
                     # Ensure mask filtering aligns with data filtering
                     if mask.shape[1] == num_landmarks_pred:
                         mask = mask[:, valid_indices]
                     else:
                         print(f"Warning: WingLoss mask shape {mask.shape} incompatible with pred {pred.shape} after index filtering. Ignoring mask.")
                         mask = None # Ignore mask if dimensions mismatch
                
                if mask is not None: # Re-check if mask is still valid
                    mask = mask.unsqueeze(-1) # Expand to (B, N_filtered, 1) for broadcasting
                    losses = losses * mask
            else:
                 print(f"Warning: WingLoss mask shape {mask.shape} invalid, expected (B, N). Ignoring mask.")
                 mask = None # Ignore invalid mask

        # Calculate final mean loss
        if mask is not None:
             # Mean over non-masked elements
             masked_sum = torch.sum(mask) # Sum over B, N_filtered, 1
             # Sum weighted, normalized, masked losses and divide by count
             return torch.sum(losses) / (masked_sum * 2 + 1e-8) if masked_sum > 0 else torch.tensor(0.0, device=losses.device)
        else:
            # Simple mean over all elements (B, N_filtered, 2)
            return torch.mean(losses)


class CombinedLoss(nn.Module):
    """
    Combined loss function for heatmap and coordinate regression
    
    Combines Adaptive Wing Loss for heatmaps with Wing Loss for coordinates
    """
    def __init__(self, heatmap_weight=1.0, coord_weight=1.0, output_size=(64, 64), image_size=(224, 224),
                 use_loss_normalization=True, norm_decay=0.99, norm_epsilon=1e-6,
                 target_landmark_indices=None,
                 landmark_weights=None):
        """
        Initialize combined loss
        
        Args:
            heatmap_weight (float): Overall weight for heatmap loss component.
            coord_weight (float): Overall weight for coordinate loss component.
            output_size (tuple): Size of heatmap output (height, width).
            image_size (tuple): Size of original image (height, width).
            use_loss_normalization (bool): Whether to enable loss normalization *within* sub-losses.
            norm_decay (float): Decay factor for running average normalization.
            norm_epsilon (float): Epsilon for numerical stability in normalization.
            target_landmark_indices (list, optional): List of landmark indices to compute loss for.
                                                      Passed down to sub-losses.
            landmark_weights (torch.Tensor, optional): Tensor of weights for each landmark.
                                                      Passed down to sub-losses.
        """
        super(CombinedLoss, self).__init__()
        # Store overall weights (may be scheduled by trainer)
        self.heatmap_weight = heatmap_weight
        self.coord_weight = coord_weight
        self.output_size = output_size
        self.image_size = image_size
        
        # Pass down parameters to sub-losses
        self.target_landmark_indices = target_landmark_indices
        self.landmark_weights = landmark_weights
        self.use_loss_normalization = use_loss_normalization
        self.norm_decay = norm_decay
        self.norm_epsilon = norm_epsilon
        
        # Calculate scale factor
        self.scale_factor = image_size[0] / output_size[0]  # Assuming square aspect ratio
        
        # Initialize sub-losses, passing all relevant parameters
        self.heatmap_loss_fn = AdaptiveWingLoss(
            target_landmark_indices=self.target_landmark_indices,
            landmark_weights=self.landmark_weights,
            use_loss_normalization=self.use_loss_normalization,
            norm_decay=self.norm_decay,
            norm_epsilon=self.norm_epsilon
        )
        self.coord_loss_fn = WingLoss(
            target_landmark_indices=self.target_landmark_indices,
            landmark_weights=self.landmark_weights,
            use_loss_normalization=self.use_loss_normalization,
            norm_decay=self.norm_decay,
            norm_epsilon=self.norm_epsilon
        )
        
        # Remove normalization buffers from CombinedLoss itself if they exist
        # (now handled within sub-losses)
        if hasattr(self, 'running_avg_heatmap_loss'):
            del self.running_avg_heatmap_loss
        if hasattr(self, 'running_avg_coord_loss'):
            del self.running_avg_coord_loss
        if hasattr(self, 'num_batches_tracked'):
             del self.num_batches_tracked

    def forward(self, pred_dict, target_heatmaps, target_coords, mask=None):
        """
        Forward pass for combined loss.
        
        Args:
            pred_dict (dict): Dictionary containing 'heatmaps' and 'coords' predictions.
            target_heatmaps (torch.Tensor): Ground truth heatmaps (B, C, H, W).
            target_coords (torch.Tensor): Ground truth coordinates (B, N, 2).
            mask (torch.Tensor, optional): Mask for valid landmarks (B, N).
            
        Returns:
            tuple: (total_loss, heatmap_loss_scaled, coord_loss_scaled)
                   where losses are already weighted by heatmap_weight/coord_weight.
        """
        # Extract predictions
        pred_heatmaps = pred_dict['heatmaps']
        pred_coords = pred_dict['coords']
        
        # --- Calculate Heatmap Loss --- 
        # AdaptiveWingLoss now handles filtering, weighting, and normalization internally
        heatmap_loss = self.heatmap_loss_fn(pred_heatmaps, target_heatmaps)
        
        # --- Calculate Coordinate Loss --- 
        # Ensure target_coords are in the same scale as predictions (usually model output scale)
        # Note: The WingLoss implementation assumes inputs are coordinates directly.
        # Check if pred_coords are scaled (e.g., 0-224) or normalized (-1 to 1) and adjust target if needed.
        # Assuming pred_coords from RefinementMLP are offsets, we need the base coordinates.
        # If pred_coords are absolute coordinates from soft-argmax + refinement, use them directly.
        
        # Let's assume pred_coords are absolute coordinates in image space (e.g., 0-224)
        # and target_coords are also in image space.
        
        # WingLoss now handles filtering, weighting, and normalization internally
        # Pass the mask if provided
        coord_loss = self.coord_loss_fn(pred_coords, target_coords, mask=mask)
        
        # --- Combine Losses --- 
        # Apply the overall weights (potentially scheduled)
        heatmap_loss_scaled = self.heatmap_weight * heatmap_loss
        coord_loss_scaled = self.coord_weight * coord_loss
        
        total_loss = heatmap_loss_scaled + coord_loss_scaled
        
        return total_loss, heatmap_loss_scaled, coord_loss_scaled


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