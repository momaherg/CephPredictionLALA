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
    def __init__(self, num_landmarks=19, learning_rate=1e-4, weight_decay=1e-5, 
                 device=None, output_dir='./outputs', use_refinement=True,
                 use_depth=False, depth_channels=64,
                 heatmap_weight=1.0, coord_weight=0.1, use_mps=False, hrnet_type='w32',
                 # Weight scheduling params
                 use_weight_schedule=False, initial_heatmap_weight=1.0, initial_coord_weight=0.1,
                 final_heatmap_weight=0.5, final_coord_weight=1.0, weight_schedule_epochs=30,
                 # Learning rate scheduler params
                 scheduler_type=None, lr_patience=5, lr_factor=0.5, lr_min=1e-6, lr_t_max=25,
                 max_lr=1e-3, pct_start=0.3, div_factor=25.0, final_div_factor=1e4,
                 # Optimizer params
                 optimizer_type='adam', momentum=0.9, nesterov=True,
                 # Loss normalization params
                 use_loss_normalization=False, norm_decay=0.99, norm_epsilon=1e-6,
                 # Per-Landmark Weighting/Focusing
                 target_landmark_indices=None, landmark_weights=None,
                 # Specific MED Logging
                 log_specific_landmark_indices=None,
                 # Image and heatmap parameters
                 output_size=(64, 64), image_size=(224, 224)):
        """
        Initialize the landmark trainer
        
        Args:
            num_landmarks (int): Number of landmarks to detect
            learning_rate (float): Learning rate for the optimizer
            weight_decay (float): Weight decay for the optimizer
            device (torch.device): Device to use for training
            output_dir (str): Directory to save outputs
            use_refinement (bool): Whether to use refinement MLP
            use_depth (bool): Whether to use depth features
            depth_channels (int): Number of channels for depth features
            heatmap_weight (float): Weight for heatmap loss
            coord_weight (float): Weight for coordinate loss
            use_mps (bool): Whether to use Metal Performance Shaders (MPS) for Mac
            hrnet_type (str): HRNet variant to use ('w32' or 'w48')
            use_weight_schedule (bool): Whether to use dynamic weight scheduling
            initial_heatmap_weight (float): Initial weight for heatmap loss
            initial_coord_weight (float): Initial weight for coordinate loss
            final_heatmap_weight (float): Final weight for heatmap loss after schedule
            final_coord_weight (float): Final weight for coordinate loss after schedule
            weight_schedule_epochs (int): Number of epochs for weight scheduling
            scheduler_type (str): Type of LR scheduler ('plateau', 'cosine', 'onecycle')
            lr_patience (int): Patience for ReduceLROnPlateau scheduler
            lr_factor (float): Factor for ReduceLROnPlateau scheduler
            lr_min (float): Minimum learning rate for schedulers
            lr_t_max (int): T_max for CosineAnnealingLR scheduler
            max_lr (float): Maximum learning rate for OneCycleLR
            pct_start (float): Percentage of cycle to increase LR for OneCycleLR
            div_factor (float): Initial LR division factor for OneCycleLR
            final_div_factor (float): Final LR division factor for OneCycleLR
            optimizer_type (str): Type of optimizer ('adam', 'adamw', 'sgd')
            momentum (float): Momentum for SGD optimizer
            nesterov (bool): Whether to use Nesterov momentum for SGD
            use_loss_normalization (bool): Whether to normalize loss values
            norm_decay (float): Decay factor for running average in normalization
            norm_epsilon (float): Epsilon for numerical stability in normalization
            target_landmark_indices (list): Indices of landmarks to focus on
            landmark_weights (torch.Tensor): Per-landmark weights for loss calculation
            log_specific_landmark_indices (list): Indices of landmarks to log specific metrics for
            output_size (tuple): Size of heatmap output (height, width)
            image_size (tuple): Size of input images (height, width)
        """        
        # Store parameters
        self.num_landmarks = num_landmarks
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.output_dir = output_dir
        self.use_refinement = use_refinement
        self.use_depth = use_depth
        self.depth_channels = depth_channels
        self.hrnet_type = hrnet_type
        
        # Image and heatmap parameters
        self.output_size = output_size
        self.image_size = image_size
        # Scale factor to convert between heatmap coordinates and image coordinates 
        self.coord_scale_factor = self.image_size[0] / self.output_size[0]
        
        # Loss weighting
        self.heatmap_weight = heatmap_weight
        self.coord_weight = coord_weight
        
        # Weight scheduling
        self.use_weight_schedule = use_weight_schedule
        self.initial_heatmap_weight = initial_heatmap_weight
        self.initial_coord_weight = initial_coord_weight
        self.final_heatmap_weight = final_heatmap_weight
        self.final_coord_weight = final_coord_weight
        self.weight_schedule_epochs = weight_schedule_epochs
        
        # Initialize current weights (fixes missing attribute error)
        if use_weight_schedule:
            self.current_heatmap_weight = initial_heatmap_weight
            self.current_coord_weight = initial_coord_weight
        else:
            self.current_heatmap_weight = heatmap_weight
            self.current_coord_weight = coord_weight
        
        # Loss normalization
        self.use_loss_normalization = use_loss_normalization
        self.norm_decay = norm_decay
        self.norm_epsilon = norm_epsilon
        self.running_heatmap_loss = 1.0
        self.running_coord_loss = 1.0
        
        # Target landmark indices
        self.target_landmark_indices = target_landmark_indices
        self.landmark_weights = landmark_weights
        
        # Specific MED logging
        self.log_specific_landmark_indices = log_specific_landmark_indices
        
        # Set device
        if device is None:
            if use_mps and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            elif torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create HRNet model
        self.model = create_hrnet_model(
            num_landmarks=num_landmarks,
            pretrained=True,
            use_refinement=use_refinement,
            hrnet_type=hrnet_type,
            use_depth=use_depth,
            depth_channels=depth_channels
        )
        self.model = self.model.to(self.device)
        
        # Define loss function
        self.criterion = CombinedLoss(
            num_landmarks=num_landmarks,
            use_refinement=use_refinement,
            heatmap_weight=initial_heatmap_weight if use_weight_schedule else heatmap_weight,
            coord_weight=initial_coord_weight if use_weight_schedule else coord_weight,
            output_size=self.output_size,
            image_size=self.image_size,
            use_loss_normalization=self.use_loss_normalization,
            norm_decay=self.norm_decay,
            norm_epsilon=self.norm_epsilon,
            target_landmark_indices=target_landmark_indices,
            landmark_weights=landmark_weights
        )
        
        # Create optimizer
        if optimizer_type.lower() == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_type.lower() == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_type.lower() == 'sgd':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                momentum=momentum,
                nesterov=nesterov,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        
        # Create learning rate scheduler
        self.scheduler = None
        self.scheduler_type = scheduler_type
        
        if scheduler_type == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=lr_factor,
                patience=lr_patience,
                verbose=True,
                min_lr=lr_min
            )
        elif scheduler_type == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=lr_t_max,
                eta_min=lr_min
            )
        elif scheduler_type == 'onecycle':
            # OneCycle starts from lr = max_lr / div_factor, 
            # peaks at max_lr, and ends at max_lr / (div_factor * final_div_factor)
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=max_lr,
                total_steps=weight_schedule_epochs,  # Will be updated in train()
                pct_start=pct_start,
                div_factor=div_factor,
                final_div_factor=final_div_factor
            )
        
        # Initialize history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_heatmap_loss': [],
            'val_heatmap_loss': [],
            'train_coord_loss': [],
            'val_coord_loss': [],
            'train_med': [],
            'val_med': [],
            'train_med_specific': {},  # Per-landmark MED for training
            'val_med_specific': {},    # Per-landmark MED for validation
            'learning_rate': [],
            'heatmap_weight': [],
            'coord_weight': []
        }
        
        # Initialize specific landmark MEDs if requested
        if log_specific_landmark_indices:
            for idx in log_specific_landmark_indices:
                if idx < num_landmarks:
                    self.history['train_med_specific'][idx] = []
                    self.history['val_med_specific'][idx] = []
    
    def train_epoch(self, train_loader):
        """
        Train the model for one epoch
        
        Args:
            train_loader (torch.utils.data.DataLoader): Training data loader
            
        Returns:
            dict: Training metrics for the epoch
        """
        self.model.train()
        epoch_loss = 0.0
        epoch_heatmap_loss = 0.0
        epoch_coord_loss = 0.0
        
        # Track the number of batches for averaging
        num_batches = 0
        
        # Track per-landmark distances and overall distances
        all_pred_landmarks = []
        all_gt_landmarks = []
        
        # Track per-landmark distances if specific logging is enabled
        specific_distances = {}
        if self.log_specific_landmark_indices:
            for idx in self.log_specific_landmark_indices:
                specific_distances[idx] = []
        
        # Training loop
        for batch in train_loader:
            # Get inputs and targets
            images = batch['image'].to(self.device)
            landmarks = batch['landmarks'].to(self.device)
            
            # Get depth maps if available and model uses them
            depth = None
            if self.use_depth and 'depth' in batch:
                depth = batch['depth'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images, depth)
            
            # Compute loss
            loss_dict = self.criterion(outputs, landmarks)
            loss = loss_dict['loss']
            
            # Backward pass
            loss.backward()
            
            # Update weights
            self.optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            
            if self.use_refinement:
                # Track component losses if using refinement
                epoch_heatmap_loss += loss_dict['heatmap_loss'].item()
                epoch_coord_loss += loss_dict['coord_loss'].item()
                
                # Get final coordinates for MED calculation
                pred_coords = outputs['refined_coords']
            else:
                # Get initial coordinates for MED calculation
                pred_coords = outputs['initial_coords']
            
            # Scale predicted coordinates from heatmap space to image space before calculating MED
            # Use the scale factor from the trainer instance for consistency
            image_space_pred_coords = pred_coords * self.coord_scale_factor
            
            # Store predictions and ground truth for overall MED calculation
            all_pred_landmarks.append(image_space_pred_coords.detach().cpu())
            all_gt_landmarks.append(landmarks.detach().cpu())
            
            # Calculate per-landmark MEDs if specified
            if self.log_specific_landmark_indices:
                per_landmark_med = per_landmark_euclidean_distance(image_space_pred_coords, landmarks)
                for idx in self.log_specific_landmark_indices:
                    # Fix: per_landmark_med is a 1D tensor of shape (num_landmarks,)
                    specific_distances[idx].append(per_landmark_med[idx].item())
            
            num_batches += 1
        
        # Calculate average loss metrics
        avg_loss = epoch_loss / num_batches
        avg_heatmap_loss = epoch_heatmap_loss / num_batches if self.use_refinement else 0.0
        avg_coord_loss = epoch_coord_loss / num_batches if self.use_refinement else 0.0
        
        # Concatenate all predictions and ground truth
        all_pred_landmarks = torch.cat(all_pred_landmarks, dim=0)
        all_gt_landmarks = torch.cat(all_gt_landmarks, dim=0)
        
        # Calculate overall MED
        avg_med = mean_euclidean_distance(all_pred_landmarks, all_gt_landmarks).item()
        
        # Also calculate the mean of per-landmark MEDs for validation/debugging
        all_landmark_meds = per_landmark_euclidean_distance(all_pred_landmarks, all_gt_landmarks)
        avg_of_per_landmark_meds = torch.mean(all_landmark_meds).item()
        
        # Log diagnostic info to help debug MED discrepancies
        print(f"Train Diagnostic - Overall MED: {avg_med:.2f}px, Mean of per-landmark MEDs: {avg_of_per_landmark_meds:.2f}px")
        
        # Calculate specific landmark MEDs if requested
        specific_meds = {}
        if self.log_specific_landmark_indices:
            for idx in self.log_specific_landmark_indices:
                if specific_distances[idx]:  # Check if we have data
                    specific_meds[idx] = sum(specific_distances[idx]) / len(specific_distances[idx])
        
        train_metrics = {
            'loss': avg_loss,
            'heatmap_loss': avg_heatmap_loss,
            'coord_loss': avg_coord_loss,
            'med': avg_med,
            'med_specific': specific_meds
        }
        
        return train_metrics
    
    def validate(self, val_loader):
        """
        Validate the model on the validation set
        
        Args:
            val_loader (torch.utils.data.DataLoader): Validation data loader
            
        Returns:
            dict: Validation metrics
        """
        self.model.eval()
        epoch_loss = 0.0
        epoch_heatmap_loss = 0.0
        epoch_coord_loss = 0.0
        
        # Track the number of batches for averaging
        num_batches = 0
        
        # Track per-landmark distances and overall distances
        all_pred_landmarks = []
        all_gt_landmarks = []
        
        # Track per-landmark distances if specific logging is enabled
        specific_distances = {}
        if self.log_specific_landmark_indices:
            for idx in self.log_specific_landmark_indices:
                specific_distances[idx] = []
        
        # Validation loop
        with torch.no_grad():
            for batch in val_loader:
                # Get inputs and targets
                images = batch['image'].to(self.device)
                landmarks = batch['landmarks'].to(self.device)
                
                # Get depth maps if available and model uses them
                depth = None
                if self.use_depth and 'depth' in batch:
                    depth = batch['depth'].to(self.device)
                
                # Forward pass
                outputs = self.model(images, depth)
                
                # Compute loss
                loss_dict = self.criterion(outputs, landmarks)
                loss = loss_dict['loss']
                
                # Update metrics
                epoch_loss += loss.item()
                
                if self.use_refinement:
                    # Track component losses if using refinement
                    epoch_heatmap_loss += loss_dict['heatmap_loss'].item()
                    epoch_coord_loss += loss_dict['coord_loss'].item()
                    
                    # Get final coordinates for MED calculation
                    pred_coords = outputs['refined_coords']
                else:
                    # Get initial coordinates for MED calculation
                    pred_coords = outputs['initial_coords']
                
                # Scale predicted coordinates from heatmap space to image space before calculating MED
                # Use the scale factor from the trainer instance for consistency
                image_space_pred_coords = pred_coords * self.coord_scale_factor
                
                # Store predictions and ground truth for overall MED calculation
                all_pred_landmarks.append(image_space_pred_coords.detach().cpu())
                all_gt_landmarks.append(landmarks.detach().cpu())
                
                # Calculate per-landmark MEDs if specified
                if self.log_specific_landmark_indices:
                    per_landmark_med = per_landmark_euclidean_distance(image_space_pred_coords, landmarks)
                    for idx in self.log_specific_landmark_indices:
                        # Fix: per_landmark_med is a 1D tensor of shape (num_landmarks,)
                        specific_distances[idx].append(per_landmark_med[idx].item())
                
                num_batches += 1
        
        # Calculate average loss metrics
        avg_loss = epoch_loss / num_batches
        avg_heatmap_loss = epoch_heatmap_loss / num_batches if self.use_refinement else 0.0
        avg_coord_loss = epoch_coord_loss / num_batches if self.use_refinement else 0.0
        
        # Concatenate all predictions and ground truth
        all_pred_landmarks = torch.cat(all_pred_landmarks, dim=0)
        all_gt_landmarks = torch.cat(all_gt_landmarks, dim=0)
        
        # Calculate overall MED
        avg_med = mean_euclidean_distance(all_pred_landmarks, all_gt_landmarks).item()
        
        # Also calculate the mean of per-landmark MEDs for validation/debugging
        all_landmark_meds = per_landmark_euclidean_distance(all_pred_landmarks, all_gt_landmarks)
        avg_of_per_landmark_meds = torch.mean(all_landmark_meds).item()
        
        # Log diagnostic info to help debug MED discrepancies
        print(f"Valid Diagnostic - Overall MED: {avg_med:.2f}px, Mean of per-landmark MEDs: {avg_of_per_landmark_meds:.2f}px")
        
        # Calculate specific landmark MEDs if requested
        specific_meds = {}
        if self.log_specific_landmark_indices:
            for idx in self.log_specific_landmark_indices:
                if specific_distances[idx]:  # Check if we have data
                    specific_meds[idx] = sum(specific_distances[idx]) / len(specific_distances[idx])
        
        val_metrics = {
            'loss': avg_loss,
            'heatmap_loss': avg_heatmap_loss,
            'coord_loss': avg_coord_loss,
            'med': avg_med,
            'med_specific': specific_meds
        }
        
        return val_metrics
    
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
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_heatmap_loss'].append(train_metrics['heatmap_loss'])
            self.history['val_heatmap_loss'].append(val_metrics['heatmap_loss'])
            self.history['train_coord_loss'].append(train_metrics['coord_loss'])
            self.history['val_coord_loss'].append(val_metrics['coord_loss'])
            self.history['train_med'].append(train_metrics['med'])
            self.history['val_med'].append(val_metrics['med'])
            
            # Update specific MED history
            if self.log_specific_landmark_indices:
                # Train specific MED
                if train_metrics['med_specific'] is not None:
                    for idx, med_val in train_metrics['med_specific'].items():
                        if idx not in self.history['train_med_specific']:
                            self.history['train_med_specific'][idx] = []
                        self.history['train_med_specific'][idx].append(med_val)
                # Validation specific MED
                if val_metrics['med_specific'] is not None:
                    for idx, med_val in val_metrics['med_specific'].items():
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
                self.scheduler.step(val_metrics['med'])
            
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
                        log_f_rw.write(f"{epoch+1},{train_metrics['loss']:.6f},{train_metrics['heatmap_loss']:.6f},{train_metrics['coord_loss']:.6f},{train_metrics['med']:.6f},,"
                                       f"{val_metrics['loss']:.6f},{val_metrics['heatmap_loss']:.6f},{val_metrics['coord_loss']:.6f},{val_metrics['med']:.6f},,{current_lr:.8f}\n")
                        # Now write the actual first line data with specific MEDs
                        f.write(f"{epoch+1},{train_metrics['loss']:.6f},{train_metrics['heatmap_loss']:.6f},{train_metrics['coord_loss']:.6f},{train_metrics['med']:.6f},{train_med_specific_str},"
                                f"{val_metrics['loss']:.6f},{val_metrics['heatmap_loss']:.6f},{val_metrics['coord_loss']:.6f},{val_metrics['med']:.6f},{val_med_specific_str},{current_lr:.8f}\n")
                else:
                     # Write data for subsequent epochs
                    f.write(f"{epoch+1},{train_metrics['loss']:.6f},{train_metrics['heatmap_loss']:.6f},{train_metrics['coord_loss']:.6f},{train_metrics['med']:.6f},{train_med_specific_str},"
                            f"{val_metrics['loss']:.6f},{val_metrics['heatmap_loss']:.6f},{val_metrics['coord_loss']:.6f},{val_metrics['med']:.6f},{val_med_specific_str},{current_lr:.8f}\n")

            # Print detailed progress with all metrics
            elapsed_time = time.time() - start_time
            
            # More detailed progress log including all metrics
            print(f"Epoch {epoch+1}/{num_epochs} [{elapsed_time:.2f}s]")
            print(f"  Train: Loss={train_metrics['loss']:.4f} (Heatmap={self.current_heatmap_weight:.1f}×{train_metrics['heatmap_loss']:.4f}, Coord={self.current_coord_weight:.1f}×{train_metrics['coord_loss']:.4f}), MED={train_metrics['med']:.2f}px")
            # Print specific train MED if available
            if train_metrics['med_specific'] is not None:
                med_str = ", ".join([f"L{idx}: {med:.2f}" for idx, med in train_metrics['med_specific'].items()])
                print(f"         Specific Train MEDs: [{med_str}] px")
            print(f"  Valid: Loss={val_metrics['loss']:.4f} (Heatmap={val_metrics['heatmap_loss']:.4f}, Coord={val_metrics['coord_loss']:.4f}), MED={val_metrics['med']:.2f}px")
            # Print specific validation MED if available
            if val_metrics['med_specific'] is not None:
                med_str = ", ".join([f"L{idx}: {med:.2f}" for idx, med in val_metrics['med_specific'].items()])
                print(f"         Specific Valid MEDs: [{med_str}] px")
            print(f"  LR: {current_lr:.2e}")
            
            # Save if best validation loss
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                self.save_checkpoint(os.path.join(self.output_dir, 'best_model_loss.pth'))
                print(f"  Saved best model (by loss) with validation loss: {val_metrics['loss']:.4f}")
            
            # Also save if best validation MED (this might be a better metric for landmark detection)
            if val_metrics['med'] < best_val_med:
                best_val_med = val_metrics['med']
                self.save_checkpoint(os.path.join(self.output_dir, 'best_model_med.pth'))
                print(f"  Saved best model (by MED) with validation MED: {val_metrics['med']:.2f} pixels")
            
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
    
    def save_model_info(self, file_path):
        """
        Save model information to a file
        
        Args:
            file_path (str): Path to save the model info
        """
        info = {
            'num_landmarks': self.num_landmarks,
            'use_refinement': self.use_refinement,
            'use_depth': self.use_depth,
            'depth_channels': self.depth_channels if self.use_depth else None,
            'hrnet_type': self.hrnet_type,
            'heatmap_weight': self.heatmap_weight,
            'coord_weight': self.coord_weight,
            'use_weight_schedule': self.use_weight_schedule,
            'output_size': self.output_size,
            'image_size': self.image_size,
            'date': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save as plain text
        with open(file_path, 'w') as f:
            for key, value in info.items():
                f.write(f"{key}: {value}\n")
    
    def save_checkpoint(self, file_path):
        """
        Save model checkpoint
        
        Args:
            file_path (str): Path to save the checkpoint
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'num_landmarks': self.num_landmarks,
            'use_refinement': self.use_refinement,
            'use_depth': self.use_depth,
            'depth_channels': self.depth_channels if self.use_depth else None,
            'hrnet_type': self.hrnet_type,
            'output_size': self.output_size,
            'image_size': self.image_size
        }
        
        # Add scheduler state if exists
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, file_path)
    
    def load_checkpoint(self, file_path):
        """
        Load model checkpoint
        
        Args:
            file_path (str): Path to the checkpoint file
        """
        checkpoint = torch.load(file_path, map_location=self.device)
        
        # Load model parameters
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if exists
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Verify model parameters
        if 'num_landmarks' in checkpoint and checkpoint['num_landmarks'] != self.num_landmarks:
            print(f"Warning: Checkpoint was trained with {checkpoint['num_landmarks']} landmarks, but model is configured for {self.num_landmarks} landmarks.")
        
        if 'use_refinement' in checkpoint and checkpoint['use_refinement'] != self.use_refinement:
            print(f"Warning: Checkpoint was trained with use_refinement={checkpoint['use_refinement']}, but model is configured with use_refinement={self.use_refinement}.")
        
        if 'use_depth' in checkpoint and checkpoint['use_depth'] != self.use_depth:
            print(f"Warning: Checkpoint was trained with use_depth={checkpoint['use_depth']}, but model is configured with use_depth={self.use_depth}.")
        
        print(f"Loaded checkpoint from {file_path}")
    
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
    
    def evaluate(self, test_loader, save_visualizations=False, save_predictions=False):
        """
        Evaluate the model on the test set
        
        Args:
            test_loader (torch.utils.data.DataLoader): Test data loader
            save_visualizations (bool): Whether to save visualizations of predictions
            save_predictions (bool): Whether to save detailed predictions including patient_id and distances.
            
        Returns:
            dict: Evaluation metrics, optionally includes 'predictions' list if save_predictions=True.
        """
        self.model.eval()
        
        # Lists to store predictions and ground truth for overall metrics
        all_pred_landmarks_agg = []
        all_gt_landmarks_agg = []
        # List to store detailed prediction info if save_predictions=True
        detailed_predictions = []
        
        # Evaluation loop
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                # Get inputs and targets
                images = batch['image'].to(self.device)
                landmarks = batch['landmarks'].to(self.device)
                patient_ids = batch.get('patient_id', [None] * images.size(0)) # Get patient IDs if available
                
                # Get depth maps if available and model uses them
                depth = None
                if self.use_depth and 'depth' in batch:
                    depth = batch['depth'].to(self.device)
                
                # Forward pass - use the trainer's scale factor for consistent scaling
                pred_landmarks = self.model.predict_landmarks(images, depth, scale_factor=self.coord_scale_factor)
                
                # Move results to CPU for aggregation and further processing
                pred_landmarks_cpu = pred_landmarks.cpu()
                landmarks_cpu = landmarks.cpu()
                
                # Store predictions and ground truth for overall metrics calculation
                all_pred_landmarks_agg.append(pred_landmarks_cpu)
                all_gt_landmarks_agg.append(landmarks_cpu)
                
                # Store detailed info if requested
                if save_predictions:
                    # Calculate per-landmark distances for this batch
                    batch_distances = torch.sqrt(torch.sum((pred_landmarks_cpu - landmarks_cpu) ** 2, dim=2))
                    
                    # Iterate through samples in the batch
                    for i in range(images.size(0)):
                        detailed_predictions.append({
                            'patient_id': patient_ids[i], # Store patient ID
                            'pred_landmarks': pred_landmarks_cpu[i], # Shape (num_landmarks, 2)
                            'gt_landmarks': landmarks_cpu[i],       # Shape (num_landmarks, 2)
                            'distances': batch_distances[i].tolist() # Per-landmark distances (list)
                        })
                
                # Save visualization of the first few batches (using pixel values)
                if save_visualizations and batch_idx < 5:
                    vis_dir = os.path.join(self.output_dir, 'visualizations')
                    os.makedirs(vis_dir, exist_ok=True)
                    
                    # Visualize up to 4 images per batch
                    num_vis = min(4, images.size(0))
                    for i in range(num_vis):
                        # Get image from batch (denormalize)
                        img = images[i].cpu().permute(1, 2, 0).numpy()
                        mean = np.array([0.485, 0.456, 0.406])
                        std = np.array([0.229, 0.224, 0.225])
                        img = std * img + mean
                        img = np.clip(img, 0, 1)
                        
                        # Get ground truth and predicted landmarks (pixel values)
                        gt_lm = landmarks_cpu[i].numpy()
                        pred_lm = pred_landmarks_cpu[i].numpy()
                        
                        # Create visualization
                        try:
                            # Use the correct method name if it exists
                            if hasattr(self, '_save_visualization'):
                                self._save_visualization(
                                    img, gt_lm, pred_lm,
                                    save_path=os.path.join(vis_dir, f'batch{batch_idx}_sample{i}.png')
                                )
                            elif hasattr(self, '_visualize_landmarks'): # Keep fallback just in case
                                self._visualize_landmarks(
                                    img, gt_lm, pred_lm,
                                    save_path=os.path.join(vis_dir, f'batch{batch_idx}_sample{i}.png')
                                )
                            else:
                                if batch_idx == 0 and i == 0: # Print warning only once
                                     print("Warning: Visualization method not found in trainer.")
                        except Exception as e:
                             if batch_idx == 0 and i == 0: # Print warning only once
                                 print(f"Warning: Error during visualization: {e}")
        
        # Concatenate all predictions and ground truth for overall metrics
        all_pred_landmarks_agg = torch.cat(all_pred_landmarks_agg, dim=0)
        all_gt_landmarks_agg = torch.cat(all_gt_landmarks_agg, dim=0)
        
        # Calculate Overall Mean Euclidean Distance (MED) in pixels
        med = mean_euclidean_distance(all_pred_landmarks_agg, all_gt_landmarks_agg, reduction='mean').item()
        
        # Calculate overall success rates at different thresholds (in pixels)
        all_distances_flat = torch.sqrt(torch.sum((all_pred_landmarks_agg - all_gt_landmarks_agg) ** 2, dim=2)).view(-1)
        # Convert thresholds to pixels (assuming 2mm and 4mm are physical distances)
        # NOTE: Success rates here are based on *pixel* distances. Calibration happens later.
        success_rate_2px = (all_distances_flat < 2.0).float().mean().item()
        success_rate_4px = (all_distances_flat < 4.0).float().mean().item()
        
        # Calculate per-landmark metrics (in pixels)
        per_landmark_stats = []
        for i in range(self.num_landmarks):
            # Extract landmarks at index i
            pred_lm_i = all_pred_landmarks_agg[:, i, :]
            gt_lm_i = all_gt_landmarks_agg[:, i, :]
            
            # Calculate MED for this landmark
            distances_i = torch.sqrt(torch.sum((pred_lm_i - gt_lm_i) ** 2, dim=1))
            med_i = distances_i.mean().item()
            
            # Calculate success rates for this landmark (pixel thresholds)
            success_2px_i = (distances_i < 2.0).float().mean().item()
            success_4px_i = (distances_i < 4.0).float().mean().item()
            
            per_landmark_stats.append({
                'index': i,
                'med': med_i,           # Pixel MED
                'success_rate_2mm': success_2px_i, # Note: Name kept for consistency, but it's 2px
                'success_rate_4mm': success_4px_i  # Note: Name kept for consistency, but it's 4px
            })
        
        # Prepare results dictionary
        results = {
            'mean_euclidean_distance': med, # Overall pixel MED
            'success_rate_2mm': success_rate_2px, # Overall 2px success rate
            'success_rate_4mm': success_rate_4px, # Overall 4px success rate
            'per_landmark_stats': per_landmark_stats # Per-landmark pixel stats
        }
        
        # Add detailed predictions if requested
        if save_predictions:
            results['predictions'] = detailed_predictions
            # Optionally save to file here if needed, but calibration needs it in memory
            # pred_file = os.path.join(self.output_dir, 'predictions_detailed.pt')
            # torch.save(detailed_predictions, pred_file)
            # print(f"Saved detailed predictions to {pred_file}")
        
        # Return evaluation metrics
        return results 