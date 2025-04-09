import numpy as np
import torch


def mean_euclidean_distance(pred, target, mask=None, reduction='mean'):
    """
    Compute the mean Euclidean distance between predicted and target landmarks
    
    Args:
        pred (torch.Tensor or numpy.ndarray): Predicted landmarks of shape (batch_size, num_landmarks, 2)
        target (torch.Tensor or numpy.ndarray): Target landmarks of shape (batch_size, num_landmarks, 2)
        mask (torch.Tensor or numpy.ndarray, optional): Mask for valid landmarks of shape (batch_size, num_landmarks)
        reduction (str): Reduction method ('mean' or 'sum' or 'none')
        
    Returns:
        torch.Tensor or numpy.ndarray: Mean Euclidean distance across landmarks and samples,
                                     or per-sample distance if reduction='none'
    """
    # Track input type for consistent return type
    is_tensor = isinstance(pred, torch.Tensor)
    
    # Convert tensors to numpy if needed
    if is_tensor:
        pred_numpy = pred.detach().cpu().numpy()
        target_numpy = target.detach().cpu().numpy()
        mask_numpy = mask.detach().cpu().numpy() if mask is not None and isinstance(mask, torch.Tensor) else mask
    else:
        pred_numpy = pred
        target_numpy = target
        mask_numpy = mask
    
    # Compute Euclidean distance for each landmark: shape (batch_size, num_landmarks)
    distances = np.sqrt(np.sum((pred_numpy - target_numpy) ** 2, axis=-1))
    
    # Apply mask if provided
    if mask_numpy is not None:
        distances = distances * mask_numpy
    
    # Apply reduction
    if reduction == 'none':
        # Return per-sample mean distance: shape (batch_size,)
        if mask_numpy is not None:
            # For masked data, we need to be careful about the denominator
            sample_distances = np.sum(distances, axis=1) / (np.sum(mask_numpy, axis=1) + 1e-8)
        else:
            sample_distances = np.mean(distances, axis=1)
            
        # Convert back to tensor if input was tensor
        if is_tensor:
            return torch.tensor(sample_distances, device=pred.device)
        return sample_distances
        
    elif reduction == 'sum':
        # Return sum of all distances
        if mask_numpy is not None:
            result = np.sum(distances)
        else:
            result = np.sum(distances)
            
    else:  # reduction == 'mean'
        # Return mean distance
        if mask_numpy is not None:
            result = np.sum(distances) / (np.sum(mask_numpy) + 1e-8)
        else:
            result = np.mean(distances)
    
    # Convert back to tensor if input was tensor
    if is_tensor:
        return torch.tensor(result, device=pred.device)
    return result


def landmark_success_rate(pred, target, threshold=2.0, mask=None):
    """
    Compute the success rate of landmark detection within a threshold
    
    Args:
        pred (torch.Tensor or numpy.ndarray): Predicted landmarks of shape (batch_size, num_landmarks, 2)
        target (torch.Tensor or numpy.ndarray): Target landmarks of shape (batch_size, num_landmarks, 2)
        threshold (float): Distance threshold for a successful detection (in pixels)
        mask (torch.Tensor or numpy.ndarray, optional): Mask for valid landmarks of shape (batch_size, num_landmarks)
        
    Returns:
        float: Success rate (percentage of landmarks within the threshold)
    """
    # Convert tensors to numpy if needed
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    if mask is not None and isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    
    # Compute Euclidean distance for each landmark
    distances = np.sqrt(np.sum((pred - target) ** 2, axis=-1))  # Shape: (batch_size, num_landmarks)
    
    # Count landmarks within threshold
    success = (distances <= threshold)
    
    # Apply mask if provided
    if mask is not None:
        success = success * mask
        return np.sum(success) / (np.sum(mask) + 1e-8)
    
    # Return success rate
    return np.mean(success)


def per_landmark_metrics(pred, target, mask=None):
    """
    Compute metrics for each individual landmark
    
    Args:
        pred (torch.Tensor or numpy.ndarray): Predicted landmarks of shape (batch_size, num_landmarks, 2)
        target (torch.Tensor or numpy.ndarray): Target landmarks of shape (batch_size, num_landmarks, 2)
        mask (torch.Tensor or numpy.ndarray, optional): Mask for valid landmarks of shape (batch_size, num_landmarks)
        
    Returns:
        dict: Dictionary with per-landmark metrics
    """
    # Convert tensors to numpy if needed
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    if mask is not None and isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    
    # Compute Euclidean distance for each landmark
    distances = np.sqrt(np.sum((pred - target) ** 2, axis=-1))  # Shape: (batch_size, num_landmarks)
    
    # Compute metrics for each landmark
    num_landmarks = pred.shape[1]
    metrics = {}
    
    for i in range(num_landmarks):
        if mask is not None:
            valid_mask = mask[:, i]
            valid_distances = distances[:, i] * valid_mask
            if np.sum(valid_mask) > 0:
                mean_dist = np.sum(valid_distances) / np.sum(valid_mask)
                std_dist = np.std(valid_distances[valid_mask > 0])
                success_rate_2mm = np.sum((valid_distances <= 2.0) * valid_mask) / np.sum(valid_mask)
                success_rate_4mm = np.sum((valid_distances <= 4.0) * valid_mask) / np.sum(valid_mask)
            else:
                mean_dist = std_dist = success_rate_2mm = success_rate_4mm = float('nan')
        else:
            mean_dist = np.mean(distances[:, i])
            std_dist = np.std(distances[:, i])
            success_rate_2mm = np.mean(distances[:, i] <= 2.0)
            success_rate_4mm = np.mean(distances[:, i] <= 4.0)
        
        metrics[f'landmark_{i}'] = {
            'mean_distance': mean_dist,
            'std_distance': std_dist,
            'success_rate_2mm': success_rate_2mm,
            'success_rate_4mm': success_rate_4mm
        }
    
    return metrics 


def per_landmark_euclidean_distance(predictions, targets):
    """
    Calculate the Mean Euclidean Distance for each landmark individually.

    Args:
        predictions (torch.Tensor): Predicted coordinates (B, N, 2).
        targets (torch.Tensor): Ground truth coordinates (B, N, 2).

    Returns:
        torch.Tensor: A tensor of shape (N,) containing the MED for each landmark.
    """
    if predictions.shape != targets.shape:
        raise ValueError(f"Predictions shape {predictions.shape} must match targets shape {targets.shape}")
    if predictions.dim() != 3 or predictions.shape[-1] != 2:
        raise ValueError("Inputs must have shape (B, N, 2)")

    # Calculate squared differences: (B, N, 2)
    diff_sq = (predictions - targets) ** 2
    # Sum squared differences for x and y, then take sqrt: (B, N)
    distances = torch.sqrt(diff_sq.sum(dim=-1))
    # Average distance across the batch for each landmark: (N,)
    per_landmark_med = distances.mean(dim=0)
    
    return per_landmark_med 