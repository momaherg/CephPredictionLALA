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
from src.utils.landmark_evaluation import generate_landmark_evaluation_report

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
                 coord_weight=0.1,
                 use_mps=False,
                 hrnet_type='w32',
                 use_weight_schedule=False,
                 initial_heatmap_weight=1.0,
                 initial_coord_weight=0.1,
                 final_heatmap_weight=0.5,
                 final_coord_weight=1.0,
                 weight_schedule_epochs=30,
                 scheduler_type=None,
                 lr_patience=5,
                 lr_factor=0.5,
                 lr_min=1e-6,
                 lr_t_max=10,
                 # OneCycleLR parameters (add defaults)
                 max_lr=1e-3, 
                 pct_start=0.3, 
                 div_factor=25.0, 
                 final_div_factor=1e4,
                 # Optimizer parameters (add defaults)
                 optimizer_type='adam',
                 momentum=0.9,
                 nesterov=True,
                 # Loss Normalization parameters (add defaults)
                 use_loss_normalization=False,
                 norm_decay=0.99,
                 norm_epsilon=1e-6,
                 # Target landmarks
                 target_landmark_indices=None,
                 # Per-landmark weights
                 landmark_weights=None):
        """
        Initialize trainer
        
        Args:
            num_landmarks (int): Number of landmarks to detect
            learning_rate (float): Learning rate for optimizer
            weight_decay (float): Weight decay for optimizer
            device (torch.device): Device to use for training
            output_dir (str): Directory to save outputs
            use_refinement (bool): Whether to use refinement MLP
            heatmap_weight (float): Weight for heatmap loss (used when use_weight_schedule=False)
            coord_weight (float): Weight for coordinate loss (used when use_weight_schedule=False)
            use_mps (bool): Whether to use MPS device on Mac
            hrnet_type (str): HRNet variant to use ('w32' or 'w48')
            use_weight_schedule (bool): Whether to use dynamic weight scheduling
            initial_heatmap_weight (float): Initial weight for heatmap loss in schedule
            initial_coord_weight (float): Initial weight for coordinate loss in schedule
            final_heatmap_weight (float): Final weight for heatmap loss in schedule
            final_coord_weight (float): Final weight for coordinate loss in schedule
            weight_schedule_epochs (int): Number of epochs to transition from initial to final weights
            scheduler_type (str): Type of learning rate scheduler to use ('cosine', 'plateau', or None)
            lr_patience (int): Patience for ReduceLROnPlateau scheduler
            lr_factor (float): Factor by which to reduce learning rate for ReduceLROnPlateau
            lr_min (float): Minimum learning rate for schedulers
            lr_t_max (int): T_max parameter for CosineAnnealingLR (usually set to num_epochs/2)
            max_lr (float): Maximum learning rate for OneCycleLR
            pct_start (float): Percentage of total steps to reach max_lr
            div_factor (float): Factor by which to divide learning rate
            final_div_factor (float): Factor by which to divide learning rate at the end of OneCycleLR
            optimizer_type (str): Type of optimizer to use ('adam', 'adamw', or 'sgd')
            momentum (float): Momentum for SGD optimizer
            nesterov (bool): Whether to use Nesterov momentum for SGD
            use_loss_normalization (bool): Whether to use loss normalization
            norm_decay (float): Decay rate for loss normalization
            norm_epsilon (float): Epsilon for loss normalization
            target_landmark_indices (list[int], optional): Indices of landmarks to focus loss on. Defaults to None (all landmarks).
            landmark_weights (list[float], optional): Weights for each landmark's loss contribution. Must match num_landmarks. Defaults to None (all 1.0).
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
        
        self.num_landmarks = num_landmarks
        self.output_dir = output_dir
        self.use_refinement = use_refinement
        self.target_landmark_indices = target_landmark_indices
        self.use_loss_normalization = use_loss_normalization

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Validate and store landmark_weights
        if landmark_weights is not None:
            if not isinstance(landmark_weights, list) or len(landmark_weights) != num_landmarks:
                raise ValueError(f"landmark_weights must be a list of length {num_landmarks}, but got {landmark_weights}")
            self.landmark_weights = torch.tensor(landmark_weights, dtype=torch.float32, device=self.device)
            print(f"Using custom landmark weights: {landmark_weights}")
        else:
            # Default to equal weights (tensor of ones)
            self.landmark_weights = torch.ones(num_landmarks, dtype=torch.float32, device=self.device)
            print("Using default landmark weights (all 1.0)")
            
        # Print target landmarks info
        if target_landmark_indices is not None:
            print(f"Targeting specific landmarks with indices: {target_landmark_indices}")
            # Ensure weights tensor only considers target landmarks if specified implicitly by loss function
            # Note: The actual filtering happens within the loss function
        
        # Create model
        self.model = create_hrnet_model(
            num_landmarks=num_landmarks, 
            pretrained=True, 
            use_refinement=use_refinement,
            hrnet_type=hrnet_type
        )
        self.model = self.model.to(self.device)
        
        # Optimizer parameters
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.nesterov = nesterov
        self.create_optimizer() # Call helper method to create optimizer

        # Learning rate scheduler parameters
        self.scheduler_type = scheduler_type
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor
        self.lr_min = lr_min
        self.lr_t_max = lr_t_max
        # Store OneCycleLR parameters separately for later initialization
        self.onecycle_params = {
            'max_lr': max_lr,
            'pct_start': pct_start,
            'div_factor': div_factor,
            'final_div_factor': final_div_factor
        }
        
        # Create scheduler (except for OneCycleLR which needs dataloader length)
        self.create_scheduler()

        # Weight scheduling parameters
        self.use_weight_schedule = use_weight_schedule
        self.initial_heatmap_weight = initial_heatmap_weight
        self.initial_coord_weight = initial_coord_weight
        self.final_heatmap_weight = final_heatmap_weight
        self.final_coord_weight = final_coord_weight
        self.weight_schedule_epochs = weight_schedule_epochs
        
        # Set current weights (will be updated if scheduling is used)
        self.current_heatmap_weight = initial_heatmap_weight if use_weight_schedule else heatmap_weight
        self.current_coord_weight = initial_coord_weight if use_weight_schedule else coord_weight
        
        # Create loss function
        if use_refinement:
            self.criterion = CombinedLoss(
                heatmap_weight=self.current_heatmap_weight, 
                coord_weight=self.current_coord_weight,
                output_size=(64, 64),   # Heatmap size
                image_size=(224, 224),   # Original image size
                use_loss_normalization=use_loss_normalization,
                norm_decay=norm_decay,
                norm_epsilon=norm_epsilon,
                target_landmark_indices=target_landmark_indices,
                landmark_weights=self.landmark_weights # Pass the weights tensor
            )
        else:
            # AdaptiveWingLoss also needs weights and target indices if used standalone
            self.criterion = AdaptiveWingLoss(
                use_loss_normalization=use_loss_normalization,
                norm_decay=norm_decay,
                norm_epsilon=norm_epsilon,
                target_landmark_indices=target_landmark_indices,
                landmark_weights=self.landmark_weights # Pass the weights tensor
            )
        
        # Create heatmap generator
        self.heatmap_generator = GaussianHeatmapGenerator(output_size=(64, 64), sigma=2.5)
        
        # Initialize training history (add sella_med tracking)
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_heatmap_loss': [], 'val_heatmap_loss': [],
            'train_coord_loss': [], 'val_coord_loss': [],
            'train_med': [], 'val_med': [],
            'train_sella_med': [], 'val_sella_med': [], # Specific MED for Sella (index 0)
            'heatmap_weight': [], 'coord_weight': [],
            'learning_rate': []
        }
    
    def create_optimizer(self):
        """Creates the optimizer based on the specified type."""
        if self.optimizer_type.lower() == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(), 
                lr=self.learning_rate, 
                weight_decay=self.weight_decay
            )
            print(f"Using Adam optimizer with LR={self.learning_rate}, WeightDecay={self.weight_decay}")
        elif self.optimizer_type.lower() == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(), 
                lr=self.learning_rate, 
                weight_decay=self.weight_decay
            )
            print(f"Using AdamW optimizer with LR={self.learning_rate}, WeightDecay={self.weight_decay}")
        elif self.optimizer_type.lower() == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(), 
                lr=self.learning_rate, 
                momentum=self.momentum, 
                weight_decay=self.weight_decay,
                nesterov=self.nesterov
            )
            print(f"Using SGD optimizer with LR={self.learning_rate}, Momentum={self.momentum}, Nesterov={self.nesterov}, WeightDecay={self.weight_decay}")
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")
    
    def create_scheduler(self):
        """Creates the learning rate scheduler based on the specified type."""
        if self.scheduler_type is None or self.scheduler_type.lower() == 'none':
            self.scheduler = None
            print("No learning rate scheduler will be used")
            return
            
        if self.scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.lr_t_max, eta_min=self.lr_min
            )
            print(f"Using CosineAnnealingLR scheduler with T_max={self.lr_t_max}, min_lr={self.lr_min}")
        elif self.scheduler_type == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=self.lr_factor, 
                patience=self.lr_patience, verbose=True, min_lr=self.lr_min
            )
            print(f"Using ReduceLROnPlateau scheduler with patience={self.lr_patience}, factor={self.lr_factor}, min_lr={self.lr_min}")
        elif self.scheduler_type == 'onecycle':
            # For OneCycleLR, we need to know the number of steps in an epoch
            # This will be called during train when we know the size of the dataloader
            print(f"OneCycleLR scheduler parameters ready (will be initialized at training start)")
            # OneCycleLR will be initialized in train() when we have train_loader
            self.scheduler = None
        else:
            print(f"Warning: Unknown scheduler type {self.scheduler_type}, no scheduler will be used")
            self.scheduler = None
    
    def train_epoch(self, train_loader):
        """
        Train for one epoch
        
        Args:
            train_loader (DataLoader): DataLoader for training data
            
        Returns:
            tuple: (average_loss, heatmap_loss, coord_loss, med, sella_med)
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
                epoch_heatmap_loss += loss.item()  # When not using refinement, all loss is heatmap loss
                epoch_coord_loss += 0.0  # Set coordinate loss to 0
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Step OneCycleLR scheduler if used
            if self.scheduler_type == 'onecycle':
                self.scheduler.step()
                # Update progress bar to show current learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                pbar.set_postfix(loss=loss.item(), lr=f"{current_lr:.2e}")
            else:
                pbar.set_postfix(loss=loss.item())
            
            # Update statistics
            epoch_loss += loss.item()
            
            # Get landmark predictions for metrics
            with torch.no_grad():
                predicted_landmarks = self.model.predict_landmarks(images)
                all_predictions.append(predicted_landmarks.cpu())
                all_targets.append(landmarks.cpu())
        
        # Compute mean loss
        epoch_loss /= len(train_loader)
        
        # Compute mean heatmap and coord losses
        epoch_heatmap_loss /= len(train_loader)
        epoch_coord_loss /= len(train_loader)
        
        # Compute mean Euclidean distance
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        med = mean_euclidean_distance(all_predictions, all_targets)
        
        # Compute MED for Sella (index 0) if training on all landmarks or targeting Sella
        sella_med = float('nan')  # Default to NaN if not computed
        if all_predictions.shape[1] > 0:  # Check if there are landmarks
            if self.target_landmark_indices is None or 0 in self.target_landmark_indices:
                sella_med = mean_euclidean_distance(all_predictions[:, 0:1, :], all_targets[:, 0:1, :])
        
        return epoch_loss, epoch_heatmap_loss, epoch_coord_loss, med, sella_med
    
    def validate(self, val_loader):
        """
        Validate the model
        
        Args:
            val_loader (DataLoader): DataLoader for validation data
            
        Returns:
            tuple: (average_loss, heatmap_loss, coord_loss, med, sella_med)
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
                    val_heatmap_loss += loss.item()  # When not using refinement, all loss is heatmap loss
                    val_coord_loss += 0.0  # Set coordinate loss to 0
                
                # Update statistics
                val_loss += loss.item()
                
                # Get landmark predictions for metrics
                predicted_landmarks = self.model.predict_landmarks(images)
                all_predictions.append(predicted_landmarks.cpu())
                all_targets.append(landmarks.cpu())
        
        # Compute mean loss
        val_loss /= len(val_loader)
        
        # Compute mean heatmap and coord losses
        val_heatmap_loss /= len(val_loader)
        val_coord_loss /= len(val_loader)
        
        # Compute mean Euclidean distance
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        med = mean_euclidean_distance(all_predictions, all_targets)
        
        # Compute MED for Sella (index 0) if validating on all landmarks or targeting Sella
        sella_med = float('nan')  # Default to NaN if not computed
        if all_predictions.shape[1] > 0:  # Check if there are landmarks
            if self.target_landmark_indices is None or 0 in self.target_landmark_indices:
                sella_med = mean_euclidean_distance(all_predictions[:, 0:1, :], all_targets[:, 0:1, :])
        
        return val_loss, val_heatmap_loss, val_coord_loss, med, sella_med
    
    def train(self, train_loader, val_loader, num_epochs=50, save_freq=5):
        """
        Train the model
        
        Args:
            train_loader (DataLoader): DataLoader for training data
            val_loader (DataLoader): DataLoader for validation data
            num_epochs (int): Number of epochs to train
            save_freq (int): How often to save model checkpoints (in epochs)
            
        Returns:
            dict: Training history
        """
        print(f"Starting training for {num_epochs} epochs...")
        
        best_val_loss = float('inf')
        best_val_med = float('inf')
        start_time = time.time()
        
        # Initialize OneCycleLR if selected (needs dataloader length)
        if self.scheduler_type == 'onecycle':
            # Get max_lr from parameters (or default to 10x base lr)
            max_lr = self.onecycle_params['max_lr']
            if max_lr is None:
                max_lr = 10 * self.learning_rate
                
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=max_lr,
                steps_per_epoch=len(train_loader),
                epochs=num_epochs,
                pct_start=self.onecycle_params['pct_start'],
                div_factor=self.onecycle_params['div_factor'],
                final_div_factor=self.onecycle_params['final_div_factor']
            )
            print(f"Initialized OneCycleLR scheduler with max_lr={max_lr}, steps_per_epoch={len(train_loader)}")
        
        # Create figures directory for plots
        figures_dir = os.path.join(self.output_dir, 'figures')
        os.makedirs(figures_dir, exist_ok=True)
        
        # Display initial weights if using weight scheduling
        if self.use_weight_schedule:
            print(f"Initial weights: heatmap={self.current_heatmap_weight:.2f}, coord={self.current_coord_weight:.2f}")
        
        # Create log file for metrics
        log_file_path = os.path.join(self.output_dir, 'training_log.csv')
        with open(log_file_path, 'w') as f:
            # Write header
            f.write("epoch,train_loss,train_heatmap_loss,train_coord_loss,train_med,train_sella_med,val_loss,val_heatmap_loss,val_coord_loss,val_med,val_sella_med,learning_rate\n")
        
        for epoch in range(num_epochs):
            # Update loss weights if using weight schedule
            if self.use_weight_schedule and self.use_refinement:
                if epoch < self.weight_schedule_epochs:
                    # Linear interpolation between initial and final weights
                    progress = epoch / self.weight_schedule_epochs
                    self.current_heatmap_weight = self.initial_heatmap_weight + progress * (self.final_heatmap_weight - self.initial_heatmap_weight)
                    self.current_coord_weight = self.initial_coord_weight + progress * (self.final_coord_weight - self.initial_coord_weight)
                else:
                    # Use final weights after transition period
                    self.current_heatmap_weight = self.final_heatmap_weight
                    self.current_coord_weight = self.final_coord_weight
                
                # Update the loss function weights
                self.criterion.heatmap_weight = self.current_heatmap_weight
                self.criterion.coord_weight = self.current_coord_weight
                
                print(f"Epoch {epoch+1}/{num_epochs}: Updated weights - heatmap={self.current_heatmap_weight:.2f}, coord={self.current_coord_weight:.2f}")
            
            # Train one epoch
            train_loss, train_heatmap_loss, train_coord_loss, train_med, train_sella_med = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_heatmap_loss, val_coord_loss, val_med, val_sella_med = self.validate(val_loader)
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_heatmap_loss'].append(train_heatmap_loss)
            self.history['val_heatmap_loss'].append(val_heatmap_loss)
            self.history['train_coord_loss'].append(train_coord_loss)
            self.history['val_coord_loss'].append(val_coord_loss)
            self.history['train_med'].append(train_med)
            self.history['val_med'].append(val_med)
            self.history['train_sella_med'].append(train_sella_med)
            self.history['val_sella_med'].append(val_sella_med)
            self.history['heatmap_weight'].append(self.current_heatmap_weight)
            self.history['coord_weight'].append(self.current_coord_weight)
            self.history['learning_rate'].append(current_lr)
            
            # Update learning rate if scheduler is used
            if self.scheduler_type == 'cosine':
                # Cosine scheduler steps every epoch
                if self.scheduler is not None:
                    self.scheduler.step()
                    current_lr = self.scheduler.get_last_lr()[0]
            elif self.scheduler_type == 'plateau':
                # For ReduceLROnPlateau, we use validation MED as the metric to monitor
                if self.scheduler is not None:
                    self.scheduler.step(val_med)
                    current_lr = self.optimizer.param_groups[0]['lr']
            elif self.scheduler_type == 'onecycle':
                # OneCycleLR already steps every batch in train_epoch
                if self.scheduler is not None:
                    current_lr = self.scheduler.get_last_lr()[0]
            else:
                # If no scheduler is used, learning rate doesn't change
                current_lr = self.optimizer.param_groups[0]['lr']
                
            # Track learning rate
            self.history['learning_rate'].append(current_lr)
            
            # Write metrics to log file
            with open(log_file_path, 'a') as f:
                f.write(f"{epoch+1},{train_loss:.6f},{train_heatmap_loss:.6f},{train_coord_loss:.6f},{train_med:.6f},{train_sella_med:.6f},"
                        f"{val_loss:.6f},{val_heatmap_loss:.6f},{val_coord_loss:.6f},{val_med:.6f},{val_sella_med:.6f},{current_lr:.8f}\n")
            
            # Print detailed progress with all metrics
            elapsed_time = time.time() - start_time
            
            # More detailed progress log including all metrics
            print(f"Epoch {epoch+1}/{num_epochs} [{elapsed_time:.2f}s]")
            print(f"  Train: Loss={train_loss:.4f} (Heatmap={self.current_heatmap_weight:.1f}×{train_heatmap_loss:.4f}, Coord={self.current_coord_weight:.1f}×{train_coord_loss:.4f}), MED={train_med:.2f}px (Sella={train_sella_med:.2f}px)")
            print(f"  Valid: Loss={val_loss:.4f} (Heatmap={val_heatmap_loss:.4f}, Coord={val_coord_loss:.4f}), MED={val_med:.2f}px (Sella={val_sella_med:.2f}px)")
            print(f"  LR: {current_lr:.2e}")
            
            # Save if best validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(os.path.join(self.output_dir, 'best_model_loss.pth'))
                print(f"  Saved best model (by loss) with validation loss: {val_loss:.4f}")
            
            # Also save if best validation MED (this might be a better metric for landmark detection)
            if val_med < best_val_med:
                best_val_med = val_med
                self.save_checkpoint(os.path.join(self.output_dir, 'best_model_med.pth'))
                print(f"  Saved best model (by MED) with validation MED: {val_med:.2f} pixels")
            
            # Save checkpoint periodically
            if (epoch + 1) % save_freq == 0:
                self.save_checkpoint(os.path.join(self.output_dir, f'checkpoint_epoch_{epoch+1}.pth'))
                print(f"  Saved checkpoint at epoch {epoch+1}")
                
                # Plot training curves at checkpoint epochs
                self.plot_training_curves()
        
        # Save final model
        self.save_checkpoint(os.path.join(self.output_dir, 'final_model.pth'))
        print(f"Saved final model checkpoint after {num_epochs} epochs")
        
        # Plot final training curves
        self.plot_training_curves()
        
        print(f"Training completed in {time.time() - start_time:.2f} seconds")
    
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
        # Create figures directory
        figures_dir = os.path.join(self.output_dir, 'figures')
        os.makedirs(figures_dir, exist_ok=True)
        
        # Plot loss curves
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(figures_dir, 'loss_curves.png'))
        plt.close()
        
        # Plot heatmap loss curves
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['train_heatmap_loss'], label='Train Heatmap Loss')
        plt.plot(self.history['val_heatmap_loss'], label='Validation Heatmap Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Heatmap Loss')
        plt.title('Heatmap Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(figures_dir, 'heatmap_loss_curves.png'))
        plt.close()
        
        # Plot coordinate loss curves
        if self.use_refinement:
            plt.figure(figsize=(10, 6))
            plt.plot(self.history['train_coord_loss'], label='Train Coordinate Loss')
            plt.plot(self.history['val_coord_loss'], label='Validation Coordinate Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Coordinate Loss')
            plt.title('Coordinate Loss')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(figures_dir, 'coord_loss_curves.png'))
            plt.close()
        
        # Plot MED curves
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['train_med'], label='Train MED')
        plt.plot(self.history['val_med'], label='Validation MED')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Euclidean Distance (pixels)')
        plt.title('Mean Euclidean Distance (MED)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(figures_dir, 'med_curves.png'))
        plt.close()
        
        # Plot weight schedule if using it
        if self.use_weight_schedule and len(self.history['heatmap_weight']) > 0:
            plt.figure(figsize=(10, 6))
            plt.plot(self.history['heatmap_weight'], label='Heatmap Weight')
            plt.plot(self.history['coord_weight'], label='Coordinate Weight')
            plt.xlabel('Epoch')
            plt.ylabel('Weight Value')
            plt.title('Loss Weight Schedule')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(figures_dir, 'weight_schedule.png'))
            plt.close()
        
        # Plot learning rate if available
        if len(self.history['learning_rate']) > 0:
            plt.figure(figsize=(10, 6))
            plt.plot(self.history['learning_rate'])
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.title(f'Learning Rate Schedule ({self.scheduler_type if self.scheduler_type else "None"})')
            plt.yscale('log')  # Use log scale to better visualize the changes
            plt.grid(True)
            plt.savefig(os.path.join(figures_dir, 'learning_rate_schedule.png'))
            plt.close()
            
        # Create a combined loss components plot
        epochs = range(1, len(self.history['train_loss']) + 1)
        plt.figure(figsize=(12, 8))
        
        # Plot combined loss components in one figure for easier comparison
        plt.subplot(2, 1, 1)
        plt.plot(epochs, self.history['train_loss'], 'b-', label='Total Loss')
        plt.plot(epochs, self.history['train_heatmap_loss'], 'g--', label='Heatmap Loss')
        if self.use_refinement:
            plt.plot(epochs, self.history['train_coord_loss'], 'r-.', label='Coordinate Loss')
        plt.title('Training Loss Components')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(epochs, self.history['val_loss'], 'b-', label='Total Loss')
        plt.plot(epochs, self.history['val_heatmap_loss'], 'g--', label='Heatmap Loss')
        if self.use_refinement:
            plt.plot(epochs, self.history['val_coord_loss'], 'r-.', label='Coordinate Loss')
        plt.title('Validation Loss Components')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, 'loss_components.png'))
        plt.close()
        
        # Plot Sella MED curves if available
        if len(self.history['train_sella_med']) > 0 and len(self.history['val_sella_med']) > 0:
            plt.figure(figsize=(10, 6))
            
            # First, filter out NaN values
            train_epochs = [i for i, v in enumerate(self.history['train_sella_med']) if not np.isnan(v)]
            train_values = [v for v in self.history['train_sella_med'] if not np.isnan(v)]
            val_epochs = [i for i, v in enumerate(self.history['val_sella_med']) if not np.isnan(v)]
            val_values = [v for v in self.history['val_sella_med'] if not np.isnan(v)]
            
            # Plot only if we have non-NaN values
            if len(train_values) > 0:
                plt.plot(train_epochs, train_values, label='Train Sella MED')
            if len(val_values) > 0:
                plt.plot(val_epochs, val_values, label='Validation Sella MED')
                
            plt.xlabel('Epoch')
            plt.ylabel('Mean Euclidean Distance (pixels)')
            plt.title('Sella Landmark Mean Euclidean Distance (MED)')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(figures_dir, 'sella_med_curves.png'))
            plt.close()
    
    def evaluate(self, test_loader, save_visualizations=True, landmark_names=None, landmark_cols=None):
        """
        Evaluate the model on test data
        
        Args:
            test_loader (DataLoader): DataLoader for test data
            save_visualizations (bool): Whether to save visualization images
            landmark_names (list, optional): Names of landmarks for better reporting
            landmark_cols (list, optional): List of landmark column names for skeletal classification
            
        Returns:
            dict: Evaluation results
        """
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        # Create output directory for visualizations and reports
        eval_dir = os.path.join(self.output_dir, 'evaluation')
        os.makedirs(eval_dir, exist_ok=True)
        
        if save_visualizations:
            vis_dir = os.path.join(eval_dir, 'visualizations')
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
        
        # Compute standard metrics
        med = mean_euclidean_distance(all_predictions, all_targets)
        success_rate_2mm = landmark_success_rate(all_predictions, all_targets, threshold=2.0)
        success_rate_4mm = landmark_success_rate(all_predictions, all_targets, threshold=4.0)
        
        # Compute per-landmark metrics using our new functions
        report_dir = os.path.join(eval_dir, 'reports')
        detailed_report = generate_landmark_evaluation_report(
            predictions=all_predictions,
            targets=all_targets,
            landmark_names=landmark_names,
            output_dir=report_dir,
            thresholds=[2.0, 4.0, 6.0],
            landmark_cols=landmark_cols  # Pass landmark_cols for skeletal classification
        )
        
        # Create results dictionary
        results = {
            'mean_euclidean_distance': med,
            'success_rate_2mm': success_rate_2mm,
            'success_rate_4mm': success_rate_4mm,
            'per_landmark_metrics': detailed_report
        }
        
        # Print detailed results
        print("\nDetailed Evaluation Results:")
        print(f"Overall Mean Euclidean Distance (MED): {med:.2f} pixels")
        print(f"Success Rate (2mm): {success_rate_2mm * 100:.2f}%")
        print(f"Success Rate (4mm): {success_rate_4mm * 100:.2f}%")
        
        # Print worst performing landmarks
        print("\nTop 3 Worst Performing Landmarks:")
        worst_landmarks = detailed_report['worst_landmarks']
        worst_med = detailed_report['worst_landmarks_med']
        for i, (idx, med_val) in enumerate(zip(worst_landmarks, worst_med)):
            name = landmark_names[idx] if landmark_names else f"Landmark {idx+1}"
            print(f"  {i+1}. {name}: {med_val:.2f} pixels")
        
        # Print skeletal classification results if available
        if 'classification' in detailed_report:
            class_results = detailed_report['classification']
            print("\nSkeletal Classification Results:")
            print(f"  Classification Accuracy: {class_results['accuracy']*100:.2f}%")
            print(f"  Mean ANB Angle Error: {class_results['ANB_error_mean']:.2f}°")
            print(f"  Mean SNA Angle Error: {class_results['SNA_error_mean']:.2f}°")
            print(f"  Mean SNB Angle Error: {class_results['SNB_error_mean']:.2f}°")
        
        print(f"\nDetailed report saved to: {report_dir}")
        
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