import numpy as np
import torch


def mean_euclidean_distance(pred, target, mask=None):
    """
    Compute the mean Euclidean distance between predicted and target landmarks
    
    Args:
        pred (torch.Tensor or numpy.ndarray): Predicted landmarks of shape (batch_size, num_landmarks, 2)
        target (torch.Tensor or numpy.ndarray): Target landmarks of shape (batch_size, num_landmarks, 2)
        mask (torch.Tensor or numpy.ndarray, optional): Mask for valid landmarks of shape (batch_size, num_landmarks)
        
    Returns:
        float: Mean Euclidean distance across all landmarks and samples
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
    
    # Apply mask if provided
    if mask is not None:
        distances = distances * mask
        return np.sum(distances) / (np.sum(mask) + 1e-8)
    
    # Return mean distance
    return np.mean(distances)


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