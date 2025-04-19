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
from src.utils.metrics import mean_euclidean_distance, landmark_success_rate, per_landmark_metrics, per_landmark_euclidean_distance
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
                 coord_weight=1.0,
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
                 # OneCycleLR specific parameters
                 max_lr=None,
                 pct_start=0.3,
                 div_factor=25.0,
                 final_div_factor=1e4,
                 # Optimizer parameters
                 optimizer_type='adam',
                 momentum=0.9,
                 nesterov=True,
                 # Loss normalization parameters removed
                 total_steps=None,
                 target_landmark_indices=None,
                 landmark_weights=None,
                 log_specific_landmark_indices=None): # Indices for specific MED logging
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
            scheduler_type (str): Type of learning rate scheduler to use ('cosine', 'plateau', 'onecycle', or None)
            lr_patience (int): Patience for ReduceLROnPlateau scheduler
            lr_factor (float): Factor by which to reduce learning rate for ReduceLROnPlateau
            lr_min (float): Minimum learning rate for schedulers
            lr_t_max (int): T_max parameter for CosineAnnealingLR (usually set to num_epochs/2)
            max_lr (float): Maximum learning rate for OneCycleLR (defaults to 10x learning_rate if None)
            pct_start (float): Percentage of training spent increasing learning rate for OneCycleLR
            div_factor (float): Initial learning rate division factor for OneCycleLR
            final_div_factor (float): Final learning rate division factor for OneCycleLR
            optimizer_type (str): Type of optimizer to use ('adam', 'adamw', 'sgd')
            momentum (float): Momentum factor for SGD optimizer
            nesterov (bool): Whether to use Nesterov momentum for SGD
            total_steps (int, optional): Total number of training steps, required for OneCycleLR if initialized directly.
            target_landmark_indices (list, optional): Indices of landmarks to focus on during loss calculation.
            landmark_weights (list or numpy array, optional): Weights to apply to each landmark's loss.
                                                              Must have length equal to num_landmarks.
            log_specific_landmark_indices (list, optional): Indices of landmarks to log MED for separately.
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
        self.model = create_hrnet_model(
            num_landmarks=num_landmarks, 
            pretrained=True, 
            use_refinement=use_refinement,
            hrnet_type=hrnet_type
        )
        self.model = self.model.to(self.device)
        
        # Create optimizer based on type
        optimizer_type = optimizer_type.lower()
        self.optimizer_type = optimizer_type
        
        if optimizer_type == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(), 
                lr=learning_rate, 
                weight_decay=weight_decay
            )
            print(f"Using Adam optimizer with lr={learning_rate}, weight_decay={weight_decay}")
        
        elif optimizer_type == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(), 
                lr=learning_rate, 
                weight_decay=weight_decay
            )
            print(f"Using AdamW optimizer with lr={learning_rate}, weight_decay={weight_decay}")
        
        elif optimizer_type == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(), 
                lr=learning_rate, 
                momentum=momentum,
                weight_decay=weight_decay,
                nesterov=nesterov
            )
            print(f"Using SGD optimizer with lr={learning_rate}, momentum={momentum}, "
                  f"nesterov={nesterov}, weight_decay={weight_decay}")
        
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}. "
                            "Choose from 'adam', 'adamw', or 'sgd'.")
        
        # Learning rate scheduler parameters
        self.scheduler_type = scheduler_type
        self.scheduler = None
        if scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=lr_t_max, eta_min=lr_min
            )
            print(f"Using CosineAnnealingLR scheduler with T_max={lr_t_max}, min_lr={lr_min}")
        
        elif scheduler_type == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=lr_factor, 
                patience=lr_patience, verbose=True, min_lr=lr_min
            )
            print(f"Using ReduceLROnPlateau scheduler with patience={lr_patience}, factor={lr_factor}, min_lr={lr_min}")
            
        elif scheduler_type == 'onecycle':
            # If max_lr not specified, use 10x the base learning rate
            if max_lr is None:
                max_lr = learning_rate * 10
            
            # If total_steps provided (e.g., from external calculation), initialize now
            if total_steps:
                self.scheduler = optim.lr_scheduler.OneCycleLR(
                    self.optimizer, max_lr=max_lr, total_steps=total_steps,
                    pct_start=pct_start, div_factor=div_factor, 
                    final_div_factor=final_div_factor
                )
                print(f"Using OneCycleLR scheduler initialized directly with total_steps={total_steps}, max_lr={max_lr}")
            else:
                # Store parameters for later initialization in train() when total_steps is known
                self.onecycle_params = {
                    'max_lr': max_lr,
                    'pct_start': pct_start,
                    'div_factor': div_factor,
                    'final_div_factor': final_div_factor
                }
                print(f"OneCycleLR scheduler will be initialized during training with max_lr={max_lr}")
        
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
        
        # Convert landmark_weights list/array to a tensor if provided
        self.landmark_weights_tensor = None
        if landmark_weights is not None:
            if len(landmark_weights) == num_landmarks:
                self.landmark_weights_tensor = torch.tensor(landmark_weights, dtype=torch.float32)
            else:
                print(f"Warning: landmark_weights length ({len(landmark_weights)}) does not match num_landmarks ({num_landmarks}). Ignoring weights.")
        self.target_landmark_indices = target_landmark_indices
        
        # Store indices for specific logging
        self.log_specific_landmark_indices = log_specific_landmark_indices
        
        # Create loss function
        if use_refinement:
            self.criterion = CombinedLoss(
                heatmap_weight=self.current_heatmap_weight, 
                coord_weight=self.current_coord_weight,
                output_size=(64, 64),   # Heatmap size
                image_size=(224, 224),  # Original image size
                target_landmark_indices=self.target_landmark_indices,
                landmark_weights=self.landmark_weights_tensor
            )
        else:
            # Pass weights/indices also to AdaptiveWingLoss if refinement is off
            self.criterion = AdaptiveWingLoss(
                 target_landmark_indices=self.target_landmark_indices,
                 landmark_weights=self.landmark_weights_tensor,
            )
        
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
            'val_med': [],    # Mean Euclidean Distance
            'heatmap_weight': [],
            'coord_weight': [],
            'learning_rate': [],  # Track learning rate changes
            # History for specific landmarks MED - now stores dict {idx: [med1, med2, ...]}
            'train_med_specific': {},
            'val_med_specific': {}
        }
    
    def train_epoch(self, train_loader):
        """
        Train for one epoch
        
        Args:
            train_loader (DataLoader): DataLoader for training data
            
        Returns:
            tuple: (average_loss, heatmap_loss, coord_loss, med, med_specific)
                   med_specific will be None if log_specific_landmark_indices is not set.
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
        
        # Compute specific MED if requested
        med_specific = None
        if self.log_specific_landmark_indices is not None:
            try:
                # Ensure indices are valid
                num_landmarks = all_predictions.shape[1]
                valid_indices = [idx for idx in self.log_specific_landmark_indices if 0 <= idx < num_landmarks]
                if valid_indices:
                    # Calculate per-landmark MED for all landmarks first
                    per_lm_meds = per_landmark_euclidean_distance(
                        all_predictions, all_targets
                    )
                    # Create a dictionary mapping the requested index to its MED
                    med_specific = {idx: per_lm_meds[idx].item() for idx in valid_indices}
                else:
                     print("Warning: No valid indices provided for specific MED logging.")
            except Exception as e:
                print(f"Warning: Could not compute specific MED - {e}")
        
        return epoch_loss, epoch_heatmap_loss, epoch_coord_loss, med, med_specific
    
    def validate(self, val_loader):
        """
        Validate the model
        
        Args:
            val_loader (DataLoader): DataLoader for validation data
            
        Returns:
            tuple: (average_loss, heatmap_loss, coord_loss, med, med_specific)
                   med_specific will be None if log_specific_landmark_indices is not set.
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
        
        # Compute specific MED if requested
        med_specific = None
        if self.log_specific_landmark_indices is not None:
            try:
                # Ensure indices are valid
                num_landmarks = all_predictions.shape[1]
                valid_indices = [idx for idx in self.log_specific_landmark_indices if 0 <= idx < num_landmarks]
                if valid_indices:
                    # Calculate per-landmark MED for all landmarks first
                    per_lm_meds = per_landmark_euclidean_distance(
                        all_predictions, all_targets
                    )
                    # Create a dictionary mapping the requested index to its MED
                    med_specific = {idx: per_lm_meds[idx].item() for idx in valid_indices}
                else:
                     print("Warning: No valid indices provided for specific MED logging.")
            except Exception as e:
                print(f"Warning: Could not compute specific MED - {e}")
        
        return val_loss, val_heatmap_loss, val_coord_loss, med, med_specific
    
    def train(self, train_loader, val_loader, num_epochs=50, save_freq=5):
        """
        Train the model
        
        Args:
            train_loader (DataLoader): DataLoader for training data
            val_loader (DataLoader): DataLoader for validation data
            num_epochs (int): Number of epochs to train
            save_freq (int): Frequency of saving model checkpoints
        """
        best_val_loss = float('inf')
        best_val_med = float('inf')  # Also track best validation MED
        start_time = time.time()
        
        # Initialize OneCycleLR scheduler if needed (requires knowing steps_per_epoch)
        if self.scheduler_type == 'onecycle' and self.scheduler is None:
            steps_per_epoch = len(train_loader)
            total_steps = steps_per_epoch * num_epochs
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer, 
                max_lr=self.onecycle_params['max_lr'],
                total_steps=total_steps,
                pct_start=self.onecycle_params['pct_start'],
                div_factor=self.onecycle_params['div_factor'],
                final_div_factor=self.onecycle_params['final_div_factor']
            )
            print(f"OneCycleLR scheduler initialized with total_steps={total_steps}")
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Initial weights: heatmap={self.current_heatmap_weight:.2f}, coord={self.current_coord_weight:.2f}")
        
        # Create a log file for detailed metrics
        log_dir = os.path.join(self.output_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'training_log.csv')
        
        # Write header to log file
        with open(log_file, 'w') as f:
            f.write("epoch,train_loss,train_heatmap_loss,train_coord_loss,train_med,train_med_specific,val_loss,val_heatmap_loss,val_coord_loss,val_med,val_med_specific,learning_rate\n")
        
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
            train_loss, train_heatmap_loss, train_coord_loss, train_med, train_med_specific = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_heatmap_loss, val_coord_loss, val_med, val_med_specific = self.validate(val_loader)
            
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
            
            # Update specific MED history
            if self.log_specific_landmark_indices:
                # Train specific MED
                if train_med_specific is not None:
                    for idx, med_val in train_med_specific.items():
                        if idx not in self.history['train_med_specific']:
                            self.history['train_med_specific'][idx] = []
                        self.history['train_med_specific'][idx].append(med_val)
                # Validation specific MED
                if val_med_specific is not None:
                    for idx, med_val in val_med_specific.items():
                        if idx not in self.history['val_med_specific']:
                            self.history['val_med_specific'][idx] = []
                        self.history['val_med_specific'][idx].append(med_val)
            
            self.history['heatmap_weight'].append(self.current_heatmap_weight)
            self.history['coord_weight'].append(self.current_coord_weight)
            self.history['learning_rate'].append(current_lr)
            
            # Update learning rate scheduler (except OneCycleLR which is updated per batch)
            if self.scheduler_type == 'cosine':
                self.scheduler.step()
                print(f"Learning rate updated to {self.optimizer.param_groups[0]['lr']:.2e}")
            elif self.scheduler_type == 'plateau':
                # For ReduceLROnPlateau, we use validation MED as the metric to monitor
                self.scheduler.step(val_med)
            
            # Write metrics to log file
            with open(log_file, 'a') as f:
                # Prepare specific MED strings (handle missing data)
                train_med_specific_str_parts = []
                val_med_specific_str_parts = []
                if self.log_specific_landmark_indices:
                    for idx in self.log_specific_landmark_indices:
                        train_med_val = self.history['train_med_specific'].get(idx, [])[-1] if self.history['train_med_specific'].get(idx) else np.nan
                        val_med_val = self.history['val_med_specific'].get(idx, [])[-1] if self.history['val_med_specific'].get(idx) else np.nan
                        train_med_specific_str_parts.append(f"{train_med_val:.6f}")
                        val_med_specific_str_parts.append(f"{val_med_val:.6f}")
                
                # Join parts or use empty string if no specific logging
                train_med_specific_str = ",".join(train_med_specific_str_parts) if train_med_specific_str_parts else ""
                val_med_specific_str = ",".join(val_med_specific_str_parts) if val_med_specific_str_parts else ""
                
                # Update header if first epoch and specific logging is enabled
                if epoch == 0 and self.log_specific_landmark_indices:
                    specific_headers = ",".join([f"train_med_{i}" for i in self.log_specific_landmark_indices]) + "," + \
                                       ",".join([f"val_med_{i}" for i in self.log_specific_landmark_indices])
                    # Re-read header, remove newline, append new headers, write back
                    with open(log_file, 'r+') as log_f_rw:
                        header = log_f_rw.readline().strip()
                        new_header = f"{header.replace(',learning_rate', '')},{specific_headers},learning_rate\n"
                        log_f_rw.seek(0)
                        log_f_rw.write(new_header)
                        # Need to rewrite the first line of data since we overwrote it
                        log_f_rw.write(f"{epoch+1},{train_loss:.6f},{train_heatmap_loss:.6f},{train_coord_loss:.6f},{train_med:.6f},,"
                                       f"{val_loss:.6f},{val_heatmap_loss:.6f},{val_coord_loss:.6f},{val_med:.6f},,{current_lr:.8f}\n")
                        # Now write the actual first line data with specific MEDs
                        f.write(f"{epoch+1},{train_loss:.6f},{train_heatmap_loss:.6f},{train_coord_loss:.6f},{train_med:.6f},{train_med_specific_str},"
                                f"{val_loss:.6f},{val_heatmap_loss:.6f},{val_coord_loss:.6f},{val_med:.6f},{val_med_specific_str},{current_lr:.8f}\n")
                else:
                     # Write data for subsequent epochs
                    f.write(f"{epoch+1},{train_loss:.6f},{train_heatmap_loss:.6f},{train_coord_loss:.6f},{train_med:.6f},{train_med_specific_str},"
                            f"{val_loss:.6f},{val_heatmap_loss:.6f},{val_coord_loss:.6f},{val_med:.6f},{val_med_specific_str},{current_lr:.8f}\n")

            # Print detailed progress with all metrics
            elapsed_time = time.time() - start_time
            
            # More detailed progress log including all metrics
            print(f"Epoch {epoch+1}/{num_epochs} [{elapsed_time:.2f}s]")
            print(f"  Train: Loss={train_loss:.4f} (Heatmap={self.current_heatmap_weight:.1f}×{train_heatmap_loss:.4f}, Coord={self.current_coord_weight:.1f}×{train_coord_loss:.4f}), MED={train_med:.2f}px")
            # Print specific train MED if available
            if train_med_specific is not None:
                med_str = ", ".join([f"L{idx}: {med:.2f}" for idx, med in train_med_specific.items()])
                print(f"         Specific Train MEDs: [{med_str}] px")
            print(f"  Valid: Loss={val_loss:.4f} (Heatmap={val_heatmap_loss:.4f}, Coord={val_coord_loss:.4f}), MED={val_med:.2f}px")
            # Print specific validation MED if available
            if val_med_specific is not None:
                med_str = ", ".join([f"L{idx}: {med:.2f}" for idx, med in val_med_specific.items()])
                print(f"         Specific Valid MEDs: [{med_str}] px")
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
            'optimizer_type': self.optimizer_type,
            'history': self.history,
            'use_refinement': self.use_refinement,
            'current_heatmap_weight': self.current_heatmap_weight,
            'current_coord_weight': self.current_coord_weight,
            'scheduler_type': self.scheduler_type
        }
        
        # Save scheduler state if it exists
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path):
        """
        Load model checkpoint
        
        Args:
            path (str): Path to the checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Check if optimizers match, warn if they don't
        saved_optimizer_type = checkpoint.get('optimizer_type', 'adam')  # Default to 'adam' if not specified in older checkpoints
        if saved_optimizer_type != self.optimizer_type:
            print(f"Warning: Current optimizer type ({self.optimizer_type}) differs from saved checkpoint ({saved_optimizer_type}).")
            print("This may lead to suboptimal training. Consider using the same optimizer type.")
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load history and model settings
        self.history = checkpoint['history']
        self.use_refinement = checkpoint.get('use_refinement', False)
        
        # Load loss weights
        self.current_heatmap_weight = checkpoint.get('current_heatmap_weight', 1.0)
        self.current_coord_weight = checkpoint.get('current_coord_weight', 1.0)
        
        # Update loss function weights if using refinement
        if self.use_refinement:
            self.criterion.heatmap_weight = self.current_heatmap_weight
            self.criterion.coord_weight = self.current_coord_weight
        
        # Load scheduler state if it exists and matches current config
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            # Check if scheduler types match
            if checkpoint.get('scheduler_type') == self.scheduler_type:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print(f"Loaded scheduler state with current learning rate: {self.optimizer.param_groups[0]['lr']:.2e}")
            else:
                print(f"Warning: Current scheduler type ({self.scheduler_type}) differs from saved checkpoint ({checkpoint.get('scheduler_type')}).")
                print("Scheduler state not loaded.")
    
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
            
        # Plot specific MED if available
        if self.log_specific_landmark_indices and len(self.history['train_med_specific']) > 0:
            plt.figure(figsize=(12, 7))
            num_plots = len(self.log_specific_landmark_indices)
            # Create subplots dynamically - consider layout if many landmarks
            cols = min(3, num_plots) # Max 3 columns
            rows = (num_plots + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
            axes = axes.flatten()
            
            plot_idx = 0
            for idx in self.log_specific_landmark_indices:
                if idx in self.history['train_med_specific'] and idx in self.history['val_med_specific']:
                    ax = axes[plot_idx]
                    ax.plot(self.history['train_med_specific'][idx], label=f'Train MED')
                    ax.plot(self.history['val_med_specific'][idx], label=f'Valid MED')
                    ax.set_title(f'Landmark {idx} MED')
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('MED (pixels)')
                    ax.legend()
                    ax.grid(True)
                    plot_idx += 1
            
            # Hide unused subplots
            for i in range(plot_idx, len(axes)):
                fig.delaxes(axes[i])
                
            fig.suptitle('Mean Euclidean Distance for Specific Landmarks', fontsize=16)
            fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
            plt.savefig(os.path.join(figures_dir, 'med_specific_curves.png'))
            plt.close(fig)
            
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