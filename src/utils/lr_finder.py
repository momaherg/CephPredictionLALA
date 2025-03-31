import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class LRFinder:
    """
    Learning rate range test implementation
    
    This class helps to find the optimal learning rate for training by running
    a short test that increases the learning rate exponentially and records
    the training loss at each step.
    
    Inspired by fastai's LR Finder and PyTorch Lightning's implementation.
    """
    def __init__(self, model, optimizer, criterion, device):
        """
        Initialize LR Finder
        
        Args:
            model: PyTorch model to test
            optimizer: PyTorch optimizer attached to the model
            criterion: Loss function to use for evaluation
            device: Device to use for training
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        
        # Store original learning rate to restore after test
        self.original_lr = optimizer.param_groups[0]['lr']
        
        # Results storage
        self.history = {
            'lr': [],
            'loss': [],
            'smoothed_loss': []
        }
        
        # Test settings
        self.best_loss = float('inf')
        self.smoothing = 0.05  # For exponential moving average smoothing

    def range_test(self, train_loader, start_lr=1e-7, end_lr=10, num_iter=None, 
                   step_mode='exp', beta=0.98, loss_smoothing=True, diverge_threshold=5.0):
        """
        Performs the learning rate range test
        
        Args:
            train_loader: DataLoader for training data
            start_lr: Starting learning rate
            end_lr: Maximum learning rate to try
            num_iter: Number of iterations to run (defaults to one epoch)
            step_mode: 'exp' for exponential increase, 'linear' for linear increase
            beta: For creating an exponential moving average of loss
            loss_smoothing: Whether to smooth the loss values for plotting
            diverge_threshold: Stop test after loss exceeds best loss by this factor
            
        Returns:
            Dictionary with learning rates and corresponding losses
        """
        # Reset test
        self.history = {'lr': [], 'loss': [], 'smoothed_loss': []}
        self.best_loss = float('inf')
        
        # Set model to training mode
        self.model.train()
        
        # Set learning rate to start_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = start_lr
        
        # Default num_iter to one epoch
        if num_iter is None:
            num_iter = len(train_loader)
        
        # Calculate learning rate multiplier based on step_mode
        if step_mode == 'exp':
            lr_multiplier = (end_lr / start_lr) ** (1 / num_iter)
        else:
            lr_multiplier = (end_lr - start_lr) / num_iter
        
        # Initialize smoothed loss for exponential moving average
        avg_loss = 0.
        
        print(f"Running LR Range Test over {num_iter} iterations:")
        print(f"Start LR: {start_lr}, End LR: {end_lr}, Mode: {step_mode}")
        
        # Iterate through batches
        iteration = 0
        for batch_idx, batch in enumerate(tqdm(train_loader, total=num_iter)):
            if iteration >= num_iter:
                break
                
            # Get data
            images = batch['image'].to(self.device)
            landmarks = batch['landmarks'].to(self.device)
            
            # Generate target heatmaps if needed
            if hasattr(self.criterion, 'heatmap_generator'):
                target_heatmaps = self.criterion.heatmap_generator.generate_heatmaps(landmarks).to(self.device)
            else:
                # Assuming we're working with a model that expects target heatmaps
                # This will need to be adjusted based on your specific model
                from src.models.losses import GaussianHeatmapGenerator
                heatmap_generator = GaussianHeatmapGenerator(output_size=(64, 64), sigma=2.0)
                target_heatmaps = heatmap_generator.generate_heatmaps(landmarks).to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            
            # Compute loss based on whether refinement is used
            if isinstance(outputs, dict) and 'heatmaps' in outputs and 'refined_coords' in outputs:
                loss, _, _ = self.criterion(outputs, target_heatmaps, landmarks)
            else:
                # For models without refinement
                loss = self.criterion(outputs['heatmaps'], target_heatmaps)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Initialize smoothed_loss with current loss value
            smoothed_loss = loss.item()
            
            # Compute smoothed loss using exponential moving average if requested
            if loss_smoothing:
                if iteration == 0:
                    avg_loss = loss.item()
                else:
                    avg_loss = beta * avg_loss + (1 - beta) * loss.item()
                    # Bias correction
                    smoothed_loss = avg_loss / (1 - beta ** (iteration + 1))
            
            # Record lr and loss
            self.history['lr'].append(current_lr)
            self.history['loss'].append(loss.item())
            self.history['smoothed_loss'].append(smoothed_loss)
            
            # Check if loss is getting better
            if smoothed_loss < self.best_loss:
                self.best_loss = smoothed_loss
            
            # Check for divergence
            if smoothed_loss > self.best_loss * diverge_threshold:
                print(f"\nStopping early: loss {smoothed_loss:.4f} exceeds best loss {self.best_loss:.4f} by factor of {diverge_threshold}")
                break
            
            # Update learning rate
            if step_mode == 'exp':
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= lr_multiplier
            else:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] += lr_multiplier
            
            iteration += 1
        
        # Restore original learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.original_lr
        
        print(f"LR Range Test completed with {len(self.history['lr'])} iterations")
        return self.history
    
    def plot(self, output_dir=None, skip_start=0, skip_end=0, log_lr=True, suggest_lr=True):
        """
        Plot the loss versus learning rate graph
        
        Args:
            output_dir: Directory to save the plot
            skip_start: Number of initial points to skip (often noisy)
            skip_end: Number of final points to skip (often diverging too much)
            log_lr: Whether to use log scale for learning rate axis
            suggest_lr: Whether to suggest an optimal learning rate
            
        Returns:
            Suggested learning rate if suggest_lr is True, None otherwise
        """
        # Skip points if requested
        lrs = self.history['lr'][skip_start:-skip_end] if skip_end > 0 else self.history['lr'][skip_start:]
        losses = self.history['smoothed_loss'][skip_start:-skip_end] if skip_end > 0 else self.history['smoothed_loss'][skip_start:]
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Plot loss vs learning rate
        plt.plot(lrs, losses)
        
        # Add labels and title
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        plt.title('Learning Rate Range Test')
        
        # Apply log scale to x-axis if requested
        if log_lr:
            plt.xscale('log')
        
        # Set grid
        plt.grid(True, which="both", ls="-")
        
        # Calculate rate of change (gradient) of the loss curve
        suggested_lr = None
        if suggest_lr and len(lrs) > 1:
            # Simple approach: find the point of steepest negative slope
            derivatives = [
                (losses[i+1] - losses[i]) / (lrs[i+1] - lrs[i])
                for i in range(len(lrs)-1)
            ]
            
            # Find the point where the loss starts to increase rapidly
            # We'll look for a large positive derivative
            min_idx = np.argmin(derivatives)
            
            # The suggested LR is just before the minimum
            suggested_lr = lrs[min_idx]
            
            # Alternatively, more sophisticated methods like the "steepest descent" point
            # can be implemented here for better suggestions
            
            plt.axvline(x=suggested_lr, color='r', linestyle='--', 
                       label=f'Suggested LR: {suggested_lr:.2e}')
            plt.legend()
            
            print(f"Suggested learning rate: {suggested_lr:.2e}")
        
        # Save the figure if output_dir is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, 'lr_finder.png'), dpi=300, bbox_inches='tight')
            print(f"Plot saved to {os.path.join(output_dir, 'lr_finder.png')}")
        
        plt.tight_layout()
        plt.show()
        
        return suggested_lr
        
    def reset(self):
        """
        Reset the model and optimizer to its initial state
        
        This must be called if you want to reuse the same model for actual training.
        """
        # Restore original learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.original_lr
        
        # Reset model weights if model has reset method
        if hasattr(self.model, 'reset'):
            self.model.reset() 