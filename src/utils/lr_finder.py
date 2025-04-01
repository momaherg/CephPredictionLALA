import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import copy

class LRFinder:
    """
    Learning Rate Finder helps find an optimal learning rate range for training.
    It trains the model for a few epochs while linearly or exponentially increasing
    the learning rate and records the loss at each step.
    """
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.history = {"lr": [], "loss": []}
        self._best_loss = float('inf')

        # Save initial state
        self._initial_state = {
            'model': copy.deepcopy(model.state_dict()),
            'optimizer': copy.deepcopy(optimizer.state_dict())
        }

    def reset(self):
        """Restores the model and optimizer to their initial states."""
        self.model.load_state_dict(self._initial_state['model'])
        self.optimizer.load_state_dict(self._initial_state['optimizer'])
        print("Model and optimizer reset to initial state.")

    def range_test(
        self, 
        train_loader, 
        start_lr=1e-7, 
        end_lr=1.0, 
        num_iter=100, 
        step_mode="exp", 
        smooth_f=0.05, 
        diverge_th=5
    ):
        """
        Performs the LR range test.

        Args:
            train_loader: The training DataLoader.
            start_lr: The starting learning rate.
            end_lr: The ending learning rate.
            num_iter: Number of iterations (batches) to perform the test over.
            step_mode: 'exp' or 'linear', schedule for updating learning rate.
            smooth_f: Factor for smoothing the loss (0 = no smoothing).
            diverge_th: Stop the test if loss exceeds best_loss * diverge_th.
        """
        self.history = {"lr": [], "loss": []}
        self._best_loss = float('inf')

        # Calculate LR update factor
        if step_mode.lower() == "exp":
            lr_schedule = ExponentialLR(self.optimizer, end_lr, num_iter, start_lr)
        elif step_mode.lower() == "linear":
            lr_schedule = LinearLR(self.optimizer, end_lr, num_iter, start_lr)
        else:
            raise ValueError(f"Invalid step_mode: {step_mode}")

        if smooth_f < 0 or smooth_f >= 1:
            raise ValueError("smooth_f must be >= 0 and < 1")

        self.model.train()
        batch_iter = iter(train_loader)
        smoothed_loss = 0.0
        iteration = 0

        pbar = tqdm(total=num_iter, desc="LR Range Test")

        while iteration < num_iter:
            try:
                batch = next(batch_iter)
            except StopIteration:
                batch_iter = iter(train_loader) # Reset iterator if needed
                batch = next(batch_iter)

            iteration += 1
            pbar.update(1)
            
            # Extract data and move to device
            inputs = batch['image'].to(self.device)
            # Note: Criterion might need different inputs depending on the model
            # Assuming target heatmaps are needed for the criterion
            if 'landmarks' in batch: # Need to generate heatmaps for loss
                from src.models.losses import GaussianHeatmapGenerator
                heatmap_gen = GaussianHeatmapGenerator() # Temporary generator - remove .to(self.device)
                targets = heatmap_gen.generate_heatmaps(batch['landmarks']).to(self.device)
                target_coords = batch['landmarks'].to(self.device)
            else:
                raise ValueError("LRFinder requires 'landmarks' in the batch to calculate loss.")
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            # Calculate loss - handle different output formats
            if isinstance(outputs, dict):
                if isinstance(self.criterion, CombinedLoss):
                     loss, _, _ = self.criterion(outputs, targets, target_coords)
                elif 'heatmaps' in outputs:
                     loss = self.criterion(outputs['heatmaps'], targets)
                else:
                    raise ValueError("Model output dictionary format not recognized by LRFinder.")
            else: # Assume outputs are heatmaps directly
                 loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Get current learning rate
            current_lr = lr_schedule.get_lr()[0]
            self.history["lr"].append(current_lr)

            # Update smoothed loss
            loss_val = loss.item()
            if iteration == 1:
                smoothed_loss = loss_val
            else:
                smoothed_loss = (1 - smooth_f) * smoothed_loss + smooth_f * loss_val
            self.history["loss"].append(smoothed_loss)

            # Update best loss
            if smoothed_loss < self._best_loss:
                self._best_loss = smoothed_loss

            pbar.set_postfix(lr=f"{current_lr:.2e}", loss=f"{smoothed_loss:.4f}")

            # Check if loss diverged
            if smoothed_loss > self._best_loss * diverge_th:
                print("\nLoss diverging, stopping early.")
                pbar.close()
                break

            # Update learning rate
            lr_schedule.step()
            
        pbar.close()
        print("LR Range Test completed.")

    def plot(self, skip_start=10, skip_end=5, log_lr=True, save_path=None):
        """
        Plots the learning rate range test results.

        Args:
            skip_start: Number of iterations to skip at the beginning.
            skip_end: Number of iterations to skip at the end.
            log_lr: Whether to plot the learning rate in log scale.
            save_path: Path to save the plot figure.
        """
        if skip_start < 0:
            raise ValueError("skip_start must be >= 0")
        if skip_end < 0:
            raise ValueError("skip_end must be >= 0")

        lrs = self.history["lr"]
        losses = self.history["loss"]

        if len(lrs) == 0 or len(losses) == 0:
            print("No history found. Run range_test first.")
            return

        # Skip beginning and end
        lrs = lrs[skip_start:-skip_end] if skip_end > 0 else lrs[skip_start:]
        losses = losses[skip_start:-skip_end] if skip_end > 0 else losses[skip_start:]

        if not lrs:  # Check if list is empty after slicing
            print("Not enough data points after skipping. Adjust skip_start/skip_end.")
            return

        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.plot(lrs, losses)
        ax.set_ylabel("Loss")
        ax.set_xlabel("Learning Rate")
        ax.set_title("Learning Rate Range Test")
        if log_lr:
            ax.set_xscale('log')
        ax.grid(True)
        
        # Find point of minimum gradient
        try:
            grads = np.gradient(np.array(losses))
            min_grad_idx = np.argmin(grads)
            print(f" Suggestion: Minimum loss found at LR={lrs[np.argmin(losses)]:.2e}")
            print(f" Suggestion: Steepest gradient descent found at LR={lrs[min_grad_idx]:.2e}")
            # Often, a good max_lr is one order of magnitude lower than the minimum loss point
            # Or where the loss is still clearly decreasing but before it plateaus/explodes.
            # Recommend picking a point before the minimum gradient.
            suggested_lr = lrs[min_grad_idx] / 10
            print(f" Suggested max_lr for OneCycleLR: {suggested_lr:.2e} (often 10x smaller than min gradient point)")
            ax.plot(lrs[min_grad_idx], losses[min_grad_idx], 'ro', markersize=8, label=f'Min grad @ {lrs[min_grad_idx]:.1e}')
            ax.plot(lrs[np.argmin(losses)], losses[np.argmin(losses)], 'go', markersize=8, label=f'Min loss @ {lrs[np.argmin(losses)]:.1e}')
            ax.legend()

        except Exception as e:
            print(f"Could not calculate gradient for LR suggestion: {e}")

        if save_path:
            fig.savefig(save_path)
            print(f"LR Range Test plot saved to {save_path}")
        
        plt.show()

# Utility classes for LR scheduling within the finder
class BaseLRSchedule:
    def __init__(self, optimizer, end_lr, num_iter, start_lr=None):
        self.optimizer = optimizer
        self.end_lr = end_lr
        self.num_iter = num_iter
        if start_lr is None:
            start_lr = optimizer.param_groups[0]['lr']
        self.start_lr = start_lr
        self.current_iter = 0
        self.set_lr(self.start_lr)

    def step(self):
        self.current_iter += 1
        lr = self.get_lr()
        self.set_lr(lr)

    def get_lr(self):
        raise NotImplementedError

    def set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

class ExponentialLR(BaseLRSchedule):
    def __init__(self, optimizer, end_lr, num_iter, start_lr=None):
        super().__init__(optimizer, end_lr, num_iter, start_lr)
        self.factor = (end_lr / self.start_lr) ** (1 / num_iter)

    def get_lr(self):
        # Returns the lr for the *next* step
        return self.start_lr * (self.factor ** self.current_iter)

class LinearLR(BaseLRSchedule):
    def __init__(self, optimizer, end_lr, num_iter, start_lr=None):
        super().__init__(optimizer, end_lr, num_iter, start_lr)
        self.increment = (end_lr - self.start_lr) / num_iter

    def get_lr(self):
        # Returns the lr for the *next* step
        return self.start_lr + self.increment * self.current_iter

# Helper to get CombinedLoss if available
try:
    from src.models.losses import CombinedLoss
except ImportError:
    CombinedLoss = None # Set to None if it cannot be imported 