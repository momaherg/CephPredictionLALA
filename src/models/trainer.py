import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
import platform

# Add the project root to the path if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from .hrnet import create_hrnet_model
from .losses import AdaptiveWingLoss, GaussianHeatmapGenerator, soft_argmax, CombinedLoss
from src.utils.metrics import mean_euclidean_distance, landmark_success_rate, per_landmark_metrics

class LandmarkTrainer:
    """
    Trainer for landmark detection model
    """
    def __init__(self, num_landmarks=19, 
                 learning_rate=1e-4, 
                 weight_decay=1e-5,
                 device=None,
                 output_dir='./outputs',
                 use_refinement=True,
                 heatmap_weight=1.0,
                 coord_weight=1.0,
                 use_mps=False):
        """
        Initialize trainer
        
        Args:
            num_landmarks (int): Number of landmarks to detect
            learning_rate (float): Learning rate for optimizer
            weight_decay (float): Weight decay for optimizer
            device (torch.device): Device to use for training
            output_dir (str): Directory to save outputs
            use_refinement (bool): Whether to use refinement MLP
            heatmap_weight (float): Weight for heatmap loss
            coord_weight (float): Weight for coordinate loss
            use_mps (bool): Whether to use MPS device on Mac
        """
        # Set device
        if device is not None:
            self.device = device
        else:
            # Check for MPS (Mac GPU) availability
            if use_mps and torch.backends.mps.is_available() and platform.system() == 'Darwin':
                self.device = torch.device('mps')
                print("Using MPS device (Apple Silicon GPU)")
            elif torch.cuda.is_available():
                self.device = torch.device('cuda')
                print("Using CUDA device")
            else:
                self.device = torch.device('cpu')
                print("Using CPU device")
        
        # Create model
        self.model = create_hrnet_model(num_landmarks=num_landmarks, pretrained=True, use_refinement=use_refinement)
        self.model = self.model.to(self.device)
        
        # Create optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Create loss function
        if use_refinement:
            self.criterion = CombinedLoss(
                heatmap_weight=heatmap_weight, 
                coord_weight=coord_weight,
                output_size=(64, 64),   # Heatmap size
                image_size=(224, 224)   # Original image size
            )
        else:
            self.criterion = AdaptiveWingLoss()
        
        # Create heatmap generator
        self.heatmap_generator = GaussianHeatmapGenerator(output_size=(64, 64), sigma=2.5)
        
        # Set number of landmarks
        self.num_landmarks = num_landmarks
        
        # Set output directory
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set whether to use refinement MLP
        self.use_refinement = use_refinement
        
        # Initialize training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_heatmap_loss': [],
            'val_heatmap_loss': [],
            'train_coord_loss': [],
            'val_coord_loss': [],
            'train_med': [],  # Mean Euclidean Distance
            'val_med': []     # Mean Euclidean Distance
        }
    
    def train_epoch(self, train_loader):
        """
        Train for one epoch
        
        Args:
            train_loader (DataLoader): DataLoader for training data
            
        Returns:
            tuple: (average_loss, heatmap_loss, coord_loss, med)
        """
        self.model.train()
        epoch_loss = 0.0
        epoch_heatmap_loss = 0.0
        epoch_coord_loss = 0.0
        all_predictions = []
        all_targets = []
        
        # Use tqdm for progress bar
        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, batch in enumerate(pbar):
            # Get images and landmarks
            images = batch['image'].to(self.device)
            landmarks = batch['landmarks'].to(self.device)
            
            # Generate target heatmaps
            target_heatmaps = self.heatmap_generator.generate_heatmaps(landmarks).to(self.device)
            
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)
            
            # Compute loss based on whether refinement is used
            if self.use_refinement:
                loss, heatmap_loss, coord_loss = self.criterion(outputs, target_heatmaps, landmarks)
                epoch_heatmap_loss += heatmap_loss.item()
                epoch_coord_loss += coord_loss.item()
            else:
                loss = self.criterion(outputs['heatmaps'], target_heatmaps)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Update statistics
            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            
            # Get landmark predictions for metrics
            with torch.no_grad():
                predicted_landmarks = self.model.predict_landmarks(images)
                all_predictions.append(predicted_landmarks.cpu())
                all_targets.append(landmarks.cpu())
        
        # Compute mean loss
        epoch_loss /= len(train_loader)
        
        # Compute mean heatmap and coord losses if refinement is used
        if self.use_refinement:
            epoch_heatmap_loss /= len(train_loader)
            epoch_coord_loss /= len(train_loader)
        
        # Compute mean Euclidean distance
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        med = mean_euclidean_distance(all_predictions, all_targets)
        
        return epoch_loss, epoch_heatmap_loss, epoch_coord_loss, med
    
    def validate(self, val_loader):
        """
        Validate the model
        
        Args:
            val_loader (DataLoader): DataLoader for validation data
            
        Returns:
            tuple: (average_loss, heatmap_loss, coord_loss, med)
        """
        self.model.eval()
        val_loss = 0.0
        val_heatmap_loss = 0.0
        val_coord_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                # Get images and landmarks
                images = batch['image'].to(self.device)
                landmarks = batch['landmarks'].to(self.device)
                
                # Generate target heatmaps
                target_heatmaps = self.heatmap_generator.generate_heatmaps(landmarks).to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Compute loss based on whether refinement is used
                if self.use_refinement:
                    loss, heatmap_loss, coord_loss = self.criterion(outputs, target_heatmaps, landmarks)
                    val_heatmap_loss += heatmap_loss.item()
                    val_coord_loss += coord_loss.item()
                else:
                    loss = self.criterion(outputs['heatmaps'], target_heatmaps)
                
                # Update statistics
                val_loss += loss.item()
                
                # Get landmark predictions for metrics
                predicted_landmarks = self.model.predict_landmarks(images)
                all_predictions.append(predicted_landmarks.cpu())
                all_targets.append(landmarks.cpu())
        
        # Compute mean loss
        val_loss /= len(val_loader)
        
        # Compute mean heatmap and coord losses if refinement is used
        if self.use_refinement:
            val_heatmap_loss /= len(val_loader)
            val_coord_loss /= len(val_loader)
        
        # Compute mean Euclidean distance
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        med = mean_euclidean_distance(all_predictions, all_targets)
        
        return val_loss, val_heatmap_loss, val_coord_loss, med
    
    def train(self, train_loader, val_loader, num_epochs=50, save_freq=5):
        """
        Train the model
        
        Args:
            train_loader (DataLoader): DataLoader for training data
            val_loader (DataLoader): DataLoader for validation data
            num_epochs (int): Number of epochs to train
            save_freq (int): Frequency of saving model checkpoints
            
        Returns:
            dict: Training history
        """
        # Create output directory
        os.makedirs(os.path.join(self.output_dir, 'checkpoints'), exist_ok=True)
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Train for one epoch
            train_loss, train_heatmap_loss, train_coord_loss, train_med = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_heatmap_loss, val_coord_loss, val_med = self.validate(val_loader)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            if self.use_refinement:
                self.history['train_heatmap_loss'].append(train_heatmap_loss)
                self.history['val_heatmap_loss'].append(val_heatmap_loss)
                self.history['train_coord_loss'].append(train_coord_loss)
                self.history['val_coord_loss'].append(val_coord_loss)
            
            self.history['train_med'].append(train_med)
            self.history['val_med'].append(val_med)
            
            # Print progress
            if self.use_refinement:
                print(f"Epoch {epoch+1}/{num_epochs} - "
                      f"Train Loss: {train_loss:.6f} (Heatmap: {train_heatmap_loss:.6f}, Coord: {train_coord_loss:.6f}), Train MED: {train_med:.2f} - "
                      f"Val Loss: {val_loss:.6f} (Heatmap: {val_heatmap_loss:.6f}, Coord: {val_coord_loss:.6f}), Val MED: {val_med:.2f}")
            else:
                print(f"Epoch {epoch+1}/{num_epochs} - "
                      f"Train Loss: {train_loss:.6f}, Train MED: {train_med:.2f} - "
                      f"Val Loss: {val_loss:.6f}, Val MED: {val_med:.2f}")
            
            # Save checkpoint
            if (epoch + 1) % save_freq == 0:
                checkpoint_path = os.path.join(self.output_dir, 'checkpoints', f'checkpoint_epoch_{epoch+1}.pth')
                self.save_checkpoint(checkpoint_path)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join(self.output_dir, 'best_model.pth')
                self.save_checkpoint(best_model_path)
                print(f"Saved best model with validation loss: {val_loss:.6f}")
            
            # Plot and save training curves
            self.plot_training_curves()
        
        # Save final model
        final_model_path = os.path.join(self.output_dir, 'final_model.pth')
        self.save_checkpoint(final_model_path)
        
        return self.history
    
    def save_checkpoint(self, path):
        """
        Save model checkpoint
        
        Args:
            path (str): Path to save the checkpoint
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'use_refinement': self.use_refinement
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path):
        """
        Load model checkpoint
        
        Args:
            path (str): Path to the checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        self.use_refinement = checkpoint.get('use_refinement', False)
    
    def plot_training_curves(self):
        """
        Plot and save training curves
        """
        # Create figure with more subplots if using refinement
        if self.use_refinement:
            plt.figure(figsize=(12, 12))
            
            # Plot total loss curves
            plt.subplot(3, 1, 1)
            plt.plot(self.history['train_loss'], label='Train Loss')
            plt.plot(self.history['val_loss'], label='Validation Loss')
            plt.title('Total Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            # Plot component loss curves
            plt.subplot(3, 1, 2)
            plt.plot(self.history['train_heatmap_loss'], label='Train Heatmap Loss')
            plt.plot(self.history['val_heatmap_loss'], label='Val Heatmap Loss')
            plt.plot(self.history['train_coord_loss'], label='Train Coord Loss')
            plt.plot(self.history['val_coord_loss'], label='Val Coord Loss')
            plt.title('Component Losses')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            # Plot MED curves
            plt.subplot(3, 1, 3)
            plt.plot(self.history['train_med'], label='Train MED')
            plt.plot(self.history['val_med'], label='Validation MED')
            plt.title('Mean Euclidean Distance')
            plt.xlabel('Epoch')
            plt.ylabel('Pixels')
            plt.legend()
        else:
            # Standard plot for non-refinement model
            plt.figure(figsize=(12, 8))
            
            # Plot loss curves
            plt.subplot(2, 1, 1)
            plt.plot(self.history['train_loss'], label='Train Loss')
            plt.plot(self.history['val_loss'], label='Validation Loss')
            plt.title('Loss Curves')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            # Plot MED curves
            plt.subplot(2, 1, 2)
            plt.plot(self.history['train_med'], label='Train MED')
            plt.plot(self.history['val_med'], label='Validation MED')
            plt.title('Mean Euclidean Distance')
            plt.xlabel('Epoch')
            plt.ylabel('Pixels')
            plt.legend()
        
        # Adjust layout and save figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_curves.png'))
        plt.close()
    
    def evaluate(self, test_loader, save_visualizations=True):
        """
        Evaluate the model on test data
        
        Args:
            test_loader (DataLoader): DataLoader for test data
            save_visualizations (bool): Whether to save visualization images
            
        Returns:
            dict: Evaluation results
        """
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        # Create output directory for visualizations
        if save_visualizations:
            vis_dir = os.path.join(self.output_dir, 'visualizations')
            os.makedirs(vis_dir, exist_ok=True)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                # Get images and landmarks
                images = batch['image'].to(self.device)
                landmarks = batch['landmarks'].to(self.device)
                
                # Forward pass
                predicted_landmarks = self.model.predict_landmarks(images)
                
                # Save predictions and targets
                all_predictions.append(predicted_landmarks.cpu())
                all_targets.append(landmarks.cpu())
                
                # Save visualizations for the first few batches
                if save_visualizations and batch_idx < 5:
                    self._save_visualizations(images, predicted_landmarks, landmarks, 
                                             os.path.join(vis_dir, f'batch_{batch_idx}'))
        
        # Concatenate all predictions and targets
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Compute metrics
        med = mean_euclidean_distance(all_predictions, all_targets)
        success_rate_2mm = landmark_success_rate(all_predictions, all_targets, threshold=2.0)
        success_rate_4mm = landmark_success_rate(all_predictions, all_targets, threshold=4.0)
        per_landmark_stats = per_landmark_metrics(all_predictions, all_targets)
        
        # Create results dictionary
        results = {
            'mean_euclidean_distance': med,
            'success_rate_2mm': success_rate_2mm,
            'success_rate_4mm': success_rate_4mm,
            'per_landmark_stats': per_landmark_stats
        }
        
        return results
    
    def _save_visualizations(self, images, predictions, targets, output_dir):
        """
        Save visualization images
        
        Args:
            images (torch.Tensor): Input images
            predictions (torch.Tensor): Predicted landmarks
            targets (torch.Tensor): Ground truth landmarks
            output_dir (str): Directory to save visualizations
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert tensors to numpy
        images = images.cpu().permute(0, 2, 3, 1).numpy()
        predictions = predictions.cpu().numpy()
        targets = targets.cpu().numpy()
        
        # Denormalize images if needed
        if images.max() <= 1.0:
            images = np.clip(images, 0, 1)
        else:
            images = np.clip(images / 255.0, 0, 1)
        
        # Plot each image in the batch
        batch_size = images.shape[0]
        for i in range(min(batch_size, 4)):  # Limit to 4 images per batch
            plt.figure(figsize=(10, 10))
            plt.imshow(images[i])
            
            # Plot predicted landmarks
            plt.scatter(predictions[i, :, 0], predictions[i, :, 1], 
                       c='blue', marker='o', label='Predicted')
            
            # Plot ground truth landmarks
            plt.scatter(targets[i, :, 0], targets[i, :, 1], 
                       c='red', marker='x', label='Ground Truth')
            
            plt.title(f'Landmark Detection Results')
            plt.legend()
            plt.axis('off')
            
            # Save figure
            plt.savefig(os.path.join(output_dir, f'sample_{i}.png'))
            plt.close() 