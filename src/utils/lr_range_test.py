import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import math
import os

def lr_range_test(model, train_loader, criterion, optimizer, device, 
                  start_lr=1e-7, end_lr=10, num_iterations=None, 
                  smooth_window=0.05, diverge_threshold=5.0, output_dir=None):
    """
    Perform a learning rate range test.
    
    Args:
        model: The neural network model to test
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer (already initialized)
        device: Device to run the test on ('cuda', 'mps', or 'cpu')
        start_lr: Minimum learning rate to test
        end_lr: Maximum learning rate to test
        num_iterations: Number of iterations to run (default: one epoch)
        smooth_window: Window size for smoothing the loss curve (as fraction of total iterations)
        diverge_threshold: Threshold for considering the loss to have diverged
        output_dir: Directory to save the plot (None to not save)
        
    Returns:
        dict: Dictionary containing test results including suggested max_lr
    """
    # Save original model state
    model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    optimizer_state = optimizer.state_dict()
    
    # Set model to training mode
    model.train()
    
    # Determine number of iterations if not specified
    if num_iterations is None:
        num_iterations = len(train_loader)
    
    # Calculate the multiplication factor for exponential LR increase
    if num_iterations > 1:
        lr_factor = (end_lr / start_lr) ** (1 / (num_iterations - 1))
    else:
        lr_factor = 1
    
    # Initialize learning rate
    current_lr = start_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr
    
    # Containers for tracking
    learning_rates = []
    losses = []
    smoothed_losses = []
    min_loss = float('inf')
    
    # Run the test
    print(f"Running LR Range Test from {start_lr} to {end_lr}...")
    batch_counter = 0
    
    try:
        with tqdm(total=num_iterations) as pbar:
            for inputs, targets in train_loader:
                if batch_counter >= num_iterations:
                    break
                
                # Move data to device
                if isinstance(inputs, torch.Tensor):
                    inputs = inputs.to(device)
                elif isinstance(inputs, dict):
                    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                
                if isinstance(targets, torch.Tensor):
                    targets = targets.to(device)
                elif isinstance(targets, dict):
                    targets = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in targets.items()}
                
                # Forward pass
                optimizer.zero_grad()
                
                # Handle different model output formats
                outputs = model(inputs)
                
                # Handle different forms of criterion
                if isinstance(outputs, dict) and isinstance(targets, dict):
                    # For combined loss functions that take dictionaries
                    loss = criterion(outputs, targets['heatmaps'], targets['coords'])
                    if isinstance(loss, tuple):
                        # If the loss function returns a tuple (e.g., total_loss, heatmap_loss, coord_loss)
                        loss = loss[0]  # Use total loss
                else:
                    # Standard case
                    loss = criterion(outputs, targets)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Store results
                learning_rates.append(current_lr)
                losses.append(loss.item())
                
                # Update progress bar
                pbar.update(1)
                pbar.set_description(f"LR: {current_lr:.7f}, Loss: {loss.item():.4f}")
                
                # Check for divergence
                if batch_counter > 0 and batch_counter > int(num_iterations * 0.1):  # Ignore first 10% of iterations
                    if loss.item() > diverge_threshold * min_loss:
                        print(f"\nLoss diverged at LR: {current_lr:.7f} (Loss: {loss.item():.4f})")
                        break
                
                min_loss = min(min_loss, loss.item())
                
                # Update learning rate for next iteration
                current_lr *= lr_factor
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
                
                batch_counter += 1
                
                # Stop if we've reached the end learning rate
                if current_lr > end_lr:
                    break
    
    except KeyboardInterrupt:
        print("LR Range Test interrupted.")
    
    # Smooth the loss curve
    if losses:
        # Convert window from fraction to absolute number of points
        window_size = max(1, int(len(losses) * smooth_window))
        
        # Apply smoothing
        smoothed_losses = []
        for i in range(len(losses)):
            # Get window boundaries
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(losses), i + window_size // 2 + 1)
            # Calculate smoothed value
            smoothed_losses.append(np.mean(losses[start_idx:end_idx]))
    
    # Find the learning rate with steepest negative gradient (where loss is decreasing fastest)
    steepest_gradient = 0
    steepest_lr = start_lr
    
    if len(smoothed_losses) > 10:  # Need enough points to calculate meaningful gradients
        gradients = []
        
        # Calculate gradients for the smoothed loss curve
        for i in range(1, len(smoothed_losses) - 5):  # Skip first point and last 5 points
            # Calculate gradient over a 5-point window for stability
            gradient = (smoothed_losses[i+5] - smoothed_losses[i]) / (learning_rates[i+5] - learning_rates[i])
            gradients.append((gradient, learning_rates[i]))
        
        # Find the point with steepest negative gradient
        if gradients:
            steepest_gradient, steepest_lr = min(gradients, key=lambda x: x[0])
    
    # Alternative: Find point of minimum loss before divergence
    min_loss_idx = np.argmin(smoothed_losses)
    min_loss_lr = learning_rates[min_loss_idx]
    
    # Suggested max_lr should be before the steepest point or minimum loss point
    # For OneCycleLR, we often want to choose a value before the minimum loss point
    suggested_max_lr = min_loss_lr / 2  # Conservative estimate
    
    # Restore original model state
    model.load_state_dict({k: v.to(device) for k, v in model_state.items()})
    optimizer.load_state_dict(optimizer_state)
    
    # Plot results
    plt.figure(figsize=(12, 8))
    plt.subplot(121)
    plt.plot(learning_rates, losses, 'b-', alpha=0.3, label='Raw loss')
    plt.plot(learning_rates, smoothed_losses, 'r-', label='Smoothed loss')
    plt.axvline(x=steepest_lr, color='g', linestyle='--', label=f'Steepest slope: {steepest_lr:.6f}')
    plt.axvline(x=min_loss_lr, color='purple', linestyle='--', label=f'Min loss: {min_loss_lr:.6f}')
    plt.axvline(x=suggested_max_lr, color='orange', linestyle='--', label=f'Suggested max_lr: {suggested_max_lr:.6f}')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.xscale('log')
    plt.grid(True)
    plt.legend()
    plt.title('LR Range Test')
    
    # Second plot with normal x-axis scale (not log) to better visualize the curve shape
    plt.subplot(122)
    plt.plot(learning_rates, losses, 'b-', alpha=0.3, label='Raw loss')
    plt.plot(learning_rates, smoothed_losses, 'r-', label='Smoothed loss')
    plt.axvline(x=steepest_lr, color='g', linestyle='--', label=f'Steepest slope: {steepest_lr:.6f}')
    plt.axvline(x=min_loss_lr, color='purple', linestyle='--', label=f'Min loss: {min_loss_lr:.6f}')
    plt.axvline(x=suggested_max_lr, color='orange', linestyle='--', label=f'Suggested max_lr: {suggested_max_lr:.6f}')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.title('LR Range Test (Linear Scale)')
    
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'lr_range_test.png'))
    
    plt.show()
    
    # Return results
    return {
        'learning_rates': learning_rates,
        'losses': losses,
        'smoothed_losses': smoothed_losses,
        'steepest_slope_lr': steepest_lr,
        'min_loss_lr': min_loss_lr,
        'suggested_max_lr': suggested_max_lr
    } 