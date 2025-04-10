#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# Cephalometric Landmark Detection Training Notebook

This notebook demonstrates how to train a cephalometric landmark detection model
using HRNet with coordinate refinement MLP.
"""

# %% [markdown]
# ## 1. Setup and Imports

# %%
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import time
from tqdm.notebook import tqdm
import platform
from torchvision import transforms

# Add the src directory to the path for imports
import sys
sys.path.append('./src')

# Import model and training components
from src.models.hrnet import create_hrnet_model
from src.models.trainer import LandmarkTrainer
from src.data.dataset import CephalometricDataset, ToTensor, Normalize
from src.data.data_augmentation import get_train_transforms
from src.data.data_processor import DataProcessor
from src.utils.lr_finder import LRFinder

# Set random seed for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# %% [markdown]
# ## 2. Configure Training Parameters
# 
# Here we define all the customizable hyperparameters for our training.
# 
# Key parameters:
# - `NUM_LANDMARKS`: Number of landmarks to detect (19 by default)
# - `USE_REFINEMENT`: Whether to use coordinate refinement MLP
# - `HRNET_TYPE`: HRNet backbone variant ('w32' or 'w48'). W48 is larger and may be more accurate but requires more memory.
# - `BALANCE_CLASSES`: Whether to balance training data based on skeletal classification
# - `BALANCE_METHOD`: Method to balance classes ('upsample' or 'downsample')
# - `LEARNING_RATE`, `BATCH_SIZE`, etc.: Standard training parameters

# %%
# Data parameters
DATA_PATH = "data/train_data.csv"  # Path to your dataset
OUTPUT_DIR = "./outputs"
APPLY_CLAHE = True  # Whether to apply Contrast Limited Adaptive Histogram Equalization
USE_DEPTH_FEATURES = False  # Whether to generate and use depth features
DEPTH_FEATURES_PATH = None  # Path to pre-generated depth features (if available)

# Model parameters
NUM_LANDMARKS = 19
PRETRAINED = True
USE_REFINEMENT = True  # Whether to use refinement MLP
MAX_DELTA = 2.0  # Maximum allowed delta for refinement
HRNET_TYPE = 'w32'  # HRNet variant: 'w32' (default) or 'w48' (larger model)

# Depth feature parameters
DEPTH_CHANNELS = 64  # Number of channels for depth features in the depth CNN

# Loss parameters
HEATMAP_WEIGHT = 1.0  # Weight for heatmap loss (used when not using weight scheduling)
COORD_WEIGHT = 0.1  # Weight for coordinate loss (used when not using weight scheduling)

# Weight scheduling parameters
USE_WEIGHT_SCHEDULE = True  # Whether to use dynamic weight scheduling
INITIAL_HEATMAP_WEIGHT = 1.0  # Initial weight for heatmap loss in schedule
INITIAL_COORD_WEIGHT = 0.1  # Initial weight for coordinate loss in schedule
FINAL_HEATMAP_WEIGHT = 0.5  # Final weight for heatmap loss in schedule
FINAL_COORD_WEIGHT = 1.0  # Final weight for coordinate loss in schedule
WEIGHT_SCHEDULE_EPOCHS = 30  # Number of epochs to transition from initial to final weights

# Class balancing parameters
BALANCE_CLASSES = True  # Whether to balance training data based on skeletal classes
BALANCE_METHOD = 'upsample'  # 'upsample' or 'downsample'

# Training parameters
BATCH_SIZE = 16
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
SAVE_FREQ = 5  # Frequency of saving checkpoints

# Hardware parameters
USE_MPS = (platform.system() == 'Darwin')  # Automatically use MPS for Mac
NUM_WORKERS = 0  # Use 0 for single process (more stable)

# Learning rate scheduler parameters
SCHEDULER_TYPE = 'cosine'  # 'cosine', 'plateau', 'onecycle', or None
LR_PATIENCE = 5  # Patience for ReduceLROnPlateau
LR_FACTOR = 0.5  # Factor to reduce learning rate for ReduceLROnPlateau
LR_MIN = 1e-6  # Minimum learning rate for schedulers
LR_T_MAX = 25  # T_max parameter for CosineAnnealingLR (half of total epochs)

# OneCycleLR specific parameters
MAX_LR = 1e-3  # Maximum learning rate for OneCycleLR (typically 3-10x base learning rate)
PCT_START = 0.3  # Percentage of training to increase learning rate (30% is typical)
DIV_FACTOR = 25.0  # Initial learning rate division factor (initial_lr = max_lr/div_factor)
FINAL_DIV_FACTOR = 1e4  # Final learning rate division factor (final_lr = initial_lr/final_div_factor)

# Optimizer parameters
OPTIMIZER_TYPE = 'adam'  # 'adam', 'adamw', or 'sgd'
MOMENTUM = 0.9  # Momentum factor for SGD optimizer
NESTEROV = True  # Whether to use Nesterov momentum for SGD optimizer

# Loss Normalization parameters
USE_LOSS_NORMALIZATION = True  # Normalize losses before weighting
NORM_DECAY = 0.99  # Decay factor for running average
NORM_EPSILON = 1e-6  # Epsilon for numerical stability

# Per-Landmark Weighting/Focusing Parameters
TARGET_LANDMARK_INDICES = None  # e.g., [0, 1, 10] to focus on Sella, Nasion, Gonion
LANDMARK_WEIGHTS = None # e.g., [2.0, 1.0, ..., 1.0] (list/array of length NUM_LANDMARKS)
                       # If None, all landmarks have weight 1.0

# Specific MED Logging
LOG_SPECIFIC_LANDMARK_INDICES = None # e.g., [0, 1, 10] to log MED for Sella, Nasion, Gonion

# LR Range Test parameters
RUN_LR_FINDER = True  # Set to True to run the LR range test before training
LR_FINDER_START_LR = 1e-7
LR_FINDER_END_LR = 0.1
LR_FINDER_NUM_ITER = 200  # Increase for smoother curve
LR_FINDER_STEP_MODE = 'exp' # 'exp' or 'linear'
LR_FINDER_SMOOTH_F = 0.05
LR_FINDER_DIVERGE_TH = 5

# %% [markdown]
# ## 3. Set Up Device

# %%
# Determine the device to use
if USE_MPS and torch.backends.mps.is_available() and platform.system() == 'Darwin':
    device = torch.device('mps')
    print("Using MPS (Metal Performance Shaders) for Mac GPU acceleration")
elif torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device('cpu')
    print("Using CPU")

# %% [markdown]
# ## 4. Setup Dataset and DataLoaders

# %%
# Define landmark columns - adjust based on your specific dataset
landmark_cols = ['sella_x', 'sella_y', 'nasion_x', 'nasion_y', 'A point_x', 'A point_y',
                'B point_x', 'B point_y', 'upper 1 tip_x', 'upper 1 tip_y',
                'upper 1 apex_x', 'upper 1 apex_y', 'lower 1 tip_x', 'lower 1 tip_y',
                'lower 1 apex_x', 'lower 1 apex_y', 'ANS_x', 'ANS_y', 'PNS_x', 'PNS_y',
                'Gonion _x', 'Gonion _y', 'Menton_x', 'Menton_y', 'ST Nasion_x',
                'ST Nasion_y', 'Tip of the nose_x', 'Tip of the nose_y', 'Subnasal_x',
                'Subnasal_y', 'Upper lip_x', 'Upper lip_y', 'Lower lip_x',
                'Lower lip_y', 'ST Pogonion_x', 'ST Pogonion_y', 'gnathion_x',
                'gnathion_y']

# Create data processor
data_processor = DataProcessor(
    data_path=DATA_PATH,
    landmark_cols=landmark_cols,
    image_size=(224, 224),
    apply_clahe=APPLY_CLAHE,
    generate_depth=USE_DEPTH_FEATURES,  # Generate depth features if needed
    depth_features_path=DEPTH_FEATURES_PATH  # Path to pre-generated depth features
)

# Load and preprocess data
df = data_processor.load_data()
df = data_processor.preprocess_data(balance_classes=BALANCE_CLASSES)

# Print data statistics
data_stats = data_processor.get_data_stats()
print(f"Dataset statistics:")
print(f"  Total samples: {data_stats['total_samples']}")

if 'depth_feature_count' in data_stats:
    print(f"  Depth features: {data_stats['depth_feature_count']} samples")
    if data_stats['depth_feature_count'] > 0:
        print(f"  Depth shape: {data_stats['depth_feature_shape']}")
        print(f"  Depth range: [{data_stats['depth_feature_min']:.3f}, {data_stats['depth_feature_max']:.3f}]")

# Create data loaders
train_loader, val_loader, test_loader = data_processor.create_data_loaders(
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    balance_classes=BALANCE_CLASSES,
    use_depth=USE_DEPTH_FEATURES  # Pass the flag to use depth features
)

print(f"Created data loaders:")
print(f"  Train: {len(train_loader.dataset)} samples")
print(f"  Validation: {len(val_loader.dataset)} samples")
print(f"  Test: {len(test_loader.dataset)} samples")

# Optionally visualize a few samples to verify data loading
def visualize_sample(loader, idx=0, with_depth=False):
    """Visualize a sample from a data loader"""
    # Get a batch
    batch = next(iter(loader))
    
    # Get image and landmarks
    image = batch['image'][idx].permute(1, 2, 0).cpu().numpy()
    landmarks = batch['landmarks'][idx].cpu().numpy()
    
    # Denormalize image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)
    
    # Plot
    fig, axes = plt.subplots(1, 2 if with_depth else 1, figsize=(12, 6))
    if not with_depth:
        axes = [axes]
    
    # Plot image with landmarks
    axes[0].imshow(image)
    axes[0].scatter(landmarks[:, 0], landmarks[:, 1], c='red', marker='x')
    axes[0].set_title('Image with Landmarks')
    
    # If depth is available, plot it
    if with_depth and 'depth' in batch:
        depth = batch['depth'][idx].permute(1, 2, 0).cpu().numpy()
        # If depth is single channel, convert to 3 channel grayscale
        if depth.shape[2] == 1:
            depth = np.repeat(depth, 3, axis=2)
        axes[1].imshow(depth)
        axes[1].set_title('Depth Map')
    
    plt.tight_layout()
    plt.show()

# Visualize a sample (uncomment to run)
# visualize_sample(train_loader, with_depth=USE_DEPTH_FEATURES)

# %% [markdown]
# ## 5. Create and Train the Model

# %%
# Create trainer
trainer = LandmarkTrainer(
    num_landmarks=NUM_LANDMARKS,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    device=device,
    output_dir=OUTPUT_DIR,
    use_refinement=USE_REFINEMENT,
    heatmap_weight=HEATMAP_WEIGHT,
    coord_weight=COORD_WEIGHT,
    use_mps=USE_MPS,
    hrnet_type=HRNET_TYPE,
    # Depth feature parameters
    use_depth=USE_DEPTH_FEATURES,
    depth_channels=DEPTH_CHANNELS,
    # Weight scheduling parameters
    use_weight_schedule=USE_WEIGHT_SCHEDULE,
    initial_heatmap_weight=INITIAL_HEATMAP_WEIGHT,
    initial_coord_weight=INITIAL_COORD_WEIGHT,
    final_heatmap_weight=FINAL_HEATMAP_WEIGHT,
    final_coord_weight=FINAL_COORD_WEIGHT,
    weight_schedule_epochs=WEIGHT_SCHEDULE_EPOCHS,
    # Learning rate scheduler parameters
    scheduler_type=SCHEDULER_TYPE,
    lr_patience=LR_PATIENCE,
    lr_factor=LR_FACTOR,
    lr_min=LR_MIN,
    lr_t_max=LR_T_MAX,
    # OneCycleLR parameters
    max_lr=MAX_LR,
    pct_start=PCT_START,
    div_factor=DIV_FACTOR,
    final_div_factor=FINAL_DIV_FACTOR,
    # Optimizer parameters
    optimizer_type=OPTIMIZER_TYPE,
    momentum=MOMENTUM,
    nesterov=NESTEROV,
    # Loss normalization parameters
    use_loss_normalization=USE_LOSS_NORMALIZATION,
    norm_decay=NORM_DECAY,
    norm_epsilon=NORM_EPSILON,
    # Per-Landmark Weighting/Focusing
    target_landmark_indices=TARGET_LANDMARK_INDICES,
    landmark_weights=LANDMARK_WEIGHTS,
    # Specific MED Logging
    log_specific_landmark_indices=LOG_SPECIFIC_LANDMARK_INDICES
)

# Custom max_delta setting for the refinement MLP if needed
if USE_REFINEMENT and hasattr(trainer.model, 'refinement_mlp'):
    trainer.model.refinement_mlp.max_delta = MAX_DELTA
    print(f"Set refinement MLP max_delta to {MAX_DELTA}")

# Print model summary
total_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
print(f"Model created with {total_params:,} trainable parameters")
print(f"HRNet type: {HRNET_TYPE.upper()}")
print(f"Refinement MLP: {'Enabled' if USE_REFINEMENT else 'Disabled'}")

# Print optimizer info
print(f"Optimizer: {OPTIMIZER_TYPE.upper()}")
if OPTIMIZER_TYPE == 'sgd':
    print(f"  Momentum: {MOMENTUM}")
    print(f"  Nesterov: {NESTEROV}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Weight decay: {WEIGHT_DECAY}")

# Print learning rate scheduler info
if USE_WEIGHT_SCHEDULE:
    print(f"Loss weight scheduling: Enabled")
    print(f"  Initial weights - Heatmap: {INITIAL_HEATMAP_WEIGHT}, Coordinate: {INITIAL_COORD_WEIGHT}")
    print(f"  Final weights   - Heatmap: {FINAL_HEATMAP_WEIGHT}, Coordinate: {FINAL_COORD_WEIGHT}")
    print(f"  Transition period: {WEIGHT_SCHEDULE_EPOCHS} epochs")
else:
    print(f"Loss weight scheduling: Disabled")
    print(f"  Fixed weights - Heatmap: {HEATMAP_WEIGHT}, Coordinate: {COORD_WEIGHT}")

if BALANCE_CLASSES:
    print(f"Class balancing: Enabled (method: {BALANCE_METHOD})")
else:
    print(f"Class balancing: Disabled")

# %% [markdown]
# ## 7. (Optional) Learning Rate Range Test
# 
# If `RUN_LR_FINDER` is set to `True`, we perform an LR range test
# to find a good maximum learning rate for the OneCycleLR scheduler.
# The test trains the model for a few iterations, gradually increasing
# the learning rate, and plots the loss against the LR.

# %%
if RUN_LR_FINDER:
    print("\nRunning Learning Rate Range Test...")
    
    # Create a temporary trainer instance JUST for the LR finder
    # Use a base learning rate, it will be overridden by the finder
    temp_trainer_for_finder = LandmarkTrainer(
        num_landmarks=NUM_LANDMARKS,
        learning_rate=LR_FINDER_START_LR, # Start LR for the finder
        weight_decay=WEIGHT_DECAY,
        device=device,
        output_dir=os.path.join(OUTPUT_DIR, 'lr_finder_temp'), # Use a temp output dir
        use_refinement=USE_REFINEMENT,
        heatmap_weight=HEATMAP_WEIGHT, # Use initial weights
        coord_weight=COORD_WEIGHT,
        use_mps=USE_MPS,
        hrnet_type=HRNET_TYPE,
        optimizer_type=OPTIMIZER_TYPE,
        momentum=MOMENTUM,
        nesterov=NESTEROV
        # Schedulers and weight scheduling are not needed for the finder itself
    )
    
    # Initialize the LRFinder
    # Note: The LRFinder requires the model, optimizer, criterion, and device.
    # It needs a criterion instance, which is created inside the trainer.
    lr_finder = LRFinder(
        temp_trainer_for_finder.model, 
        temp_trainer_for_finder.optimizer, 
        temp_trainer_for_finder.criterion, 
        device
    )
    
    # Run the range test
    lr_finder.range_test(
        train_loader, 
        start_lr=LR_FINDER_START_LR, 
        end_lr=LR_FINDER_END_LR, 
        num_iter=LR_FINDER_NUM_ITER, 
        step_mode=LR_FINDER_STEP_MODE, 
        smooth_f=LR_FINDER_SMOOTH_F,
        diverge_th=LR_FINDER_DIVERGE_TH
    )
    
    # Plot the results
    lr_finder_plot_path = os.path.join(OUTPUT_DIR, 'lr_finder_plot.png')
    lr_finder.plot(skip_start=10, skip_end=10, log_lr=True, save_path=lr_finder_plot_path)
    
    # Reset the state of the main trainer's model and optimizer
    # This is crucial because the LR finder modifies the model state.
    # We create the *real* trainer after this step to ensure it starts fresh.
    print("LR Range Test finished. Proceeding to create the final trainer and start training.")
    # It's generally safer to re-create the trainer after the LR test to ensure clean state
    # rather than trying to reset the temp_trainer_for_finder.
else:
    print("Skipping LR Range Test.")

# %% [markdown]
# ## 8. Train the Model

# %%
# Train the model
print("Starting training...")
history = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=NUM_EPOCHS,
    save_freq=SAVE_FREQ
)

# %% [markdown]
# ## 9. Evaluate the Model

# %%
# Evaluate the model on the test set
print("Evaluating model on test set...")
results = trainer.evaluate(test_loader, save_visualizations=True)

# Print evaluation results
print("\nEvaluation Results:")
print(f"Mean Euclidean Distance: {results['mean_euclidean_distance']:.2f} pixels")
print(f"Success Rate (2mm): {results['success_rate_2mm'] * 100:.2f}%")
print(f"Success Rate (4mm): {results['success_rate_4mm'] * 100:.2f}%")

# %% [markdown]
# ## 10. Visualize Results

# %%
# Plot training history
plt.figure(figsize=(15, 10))

# Plot loss
plt.subplot(2, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot MED
plt.subplot(2, 2, 2)
plt.plot(history['train_med'], label='Train MED')
plt.plot(history['val_med'], label='Val MED')
plt.title('Mean Euclidean Distance')
plt.xlabel('Epoch')
plt.ylabel('Pixels')
plt.legend()

if USE_REFINEMENT:
    # Plot component losses
    plt.subplot(2, 2, 3)
    plt.plot(history['train_heatmap_loss'], label='Train Heatmap Loss')
    plt.plot(history['val_heatmap_loss'], label='Val Heatmap Loss')
    plt.title('Heatmap Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.plot(history['train_coord_loss'], label='Train Coord Loss')
    plt.plot(history['val_coord_loss'], label='Val Coord Loss')
    plt.title('Coordinate Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Add a figure for weight schedule visualization if using weight scheduling
    if USE_WEIGHT_SCHEDULE and 'heatmap_weight' in history and 'coord_weight' in history:
        plt.figure(figsize=(10, 6))
        plt.plot(history['heatmap_weight'], label='Heatmap Weight')
        plt.plot(history['coord_weight'], label='Coordinate Weight')
        plt.title('Loss Weight Schedule')
        plt.xlabel('Epoch')
        plt.ylabel('Weight Value')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(OUTPUT_DIR, 'weight_schedule.png'))
        plt.show()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'training_results.png'))
plt.show()

# %% [markdown]
# ## 11. Visualize Predictions
# 
# Let's visualize some predictions on the test set.

# %%
def visualize_sample_predictions(trainer, test_loader, num_samples=3):
    """Visualize sample predictions from the test set"""
    trainer.model.eval()
    
    # Get samples from test loader
    data_iter = iter(test_loader)
    samples = next(data_iter)
    
    images = samples['image'].to(trainer.device)
    gt_landmarks = samples['landmarks'].cpu().numpy()
    
    # Make predictions
    with torch.no_grad():
        pred_landmarks = trainer.model.predict_landmarks(images).cpu().numpy()
    
    # Convert images to numpy
    images = images.cpu().permute(0, 2, 3, 1).numpy()
    
    # Normalize images for display
    if images.max() <= 1.0:
        images = np.clip(images, 0, 1)
    else:
        images = np.clip(images / 255.0, 0, 1)
    
    # Plot predictions vs ground truth
    fig, axes = plt.subplots(1, num_samples, figsize=(5*num_samples, 5))
    
    for i in range(min(num_samples, len(images))):
        ax = axes[i] if num_samples > 1 else axes
        ax.imshow(images[i])
        
        # Plot ground truth landmarks
        ax.scatter(gt_landmarks[i, :, 0], gt_landmarks[i, :, 1], 
                  c='green', marker='x', s=50, label='Ground Truth')
        
        # Plot predicted landmarks
        ax.scatter(pred_landmarks[i, :, 0], pred_landmarks[i, :, 1], 
                  c='red', marker='o', s=30, label='Prediction')
        
        ax.set_title(f'Sample {i+1}')
        ax.axis('off')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'sample_predictions.png'))
    plt.show()

# Visualize some predictions
try:
    visualize_sample_predictions(trainer, test_loader)
except Exception as e:
    print(f"Error visualizing predictions: {str(e)}")

# %% [markdown]
# ## 12. Save a Custom Model Configuration

# %%
# Function to save a custom model configuration
def save_model_config(output_dir, config):
    """Save model configuration to a file"""
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, 'model_config.txt')
    
    with open(config_path, 'w') as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Model configuration saved to {config_path}")

# Create and save model configuration
model_config = {
    'num_landmarks': NUM_LANDMARKS,
    'use_refinement': USE_REFINEMENT,
    'max_delta': MAX_DELTA,
    'hrnet_type': HRNET_TYPE,
    'heatmap_weight': HEATMAP_WEIGHT,
    'coord_weight': COORD_WEIGHT,
    'use_weight_schedule': USE_WEIGHT_SCHEDULE,
    'initial_heatmap_weight': INITIAL_HEATMAP_WEIGHT if USE_WEIGHT_SCHEDULE else None,
    'initial_coord_weight': INITIAL_COORD_WEIGHT if USE_WEIGHT_SCHEDULE else None,
    'final_heatmap_weight': FINAL_HEATMAP_WEIGHT if USE_WEIGHT_SCHEDULE else None,
    'final_coord_weight': FINAL_COORD_WEIGHT if USE_WEIGHT_SCHEDULE else None,
    'weight_schedule_epochs': WEIGHT_SCHEDULE_EPOCHS if USE_WEIGHT_SCHEDULE else None,
    'learning_rate': LEARNING_RATE,
    'weight_decay': WEIGHT_DECAY,
    'batch_size': BATCH_SIZE,
    'num_epochs': NUM_EPOCHS,
    'output_size': (64, 64),
    'image_size': (224, 224),
    'balance_classes': BALANCE_CLASSES,
    'balance_method': BALANCE_METHOD,
    'train_samples': len(train_loader.dataset),
    'val_samples': len(val_loader.dataset),
    'test_samples': len(test_loader.dataset),
    'final_train_loss': history['train_loss'][-1] if len(history['train_loss']) > 0 else None,
    'final_val_loss': history['val_loss'][-1] if len(history['val_loss']) > 0 else None,
    'final_train_med': history['train_med'][-1] if len(history['train_med']) > 0 else None,
    'final_val_med': history['val_med'][-1] if len(history['val_med']) > 0 else None,
    'test_med': results['mean_euclidean_distance'],
    'test_success_rate_2mm': results['success_rate_2mm'],
    'test_success_rate_4mm': results['success_rate_4mm'],
    'device': str(device),
    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
    'scheduler_type': SCHEDULER_TYPE,
    'lr_patience': LR_PATIENCE,
    'lr_factor': LR_FACTOR,
    'lr_min': LR_MIN,
    'lr_t_max': LR_T_MAX,
    'max_lr': MAX_LR,
    'pct_start': PCT_START,
    'div_factor': DIV_FACTOR,
    'final_div_factor': FINAL_DIV_FACTOR,
    'optimizer_type': OPTIMIZER_TYPE,
    'momentum': MOMENTUM,
    'nesterov': NESTEROV,
    'use_loss_normalization': USE_LOSS_NORMALIZATION,
    'norm_decay': NORM_DECAY,
    'norm_epsilon': NORM_EPSILON,
    'target_landmark_indices': TARGET_LANDMARK_INDICES,
    'landmark_weights': LANDMARK_WEIGHTS,
    'log_specific_landmark_indices': LOG_SPECIFIC_LANDMARK_INDICES
}

save_model_config(OUTPUT_DIR, model_config)

print("\nTraining completed successfully!")

# %% [markdown]
# ## 13. Additional Analysis: Per-Landmark Performance

# %%
# Plot per-landmark performance
if 'per_landmark_stats' in results:
    per_landmark_stats = results['per_landmark_stats']
    
    # Extract per landmark metrics
    landmark_meds = [stats['med'] for stats in per_landmark_stats]
    landmark_success_rates_2mm = [stats['success_rate_2mm'] for stats in per_landmark_stats]
    landmark_success_rates_4mm = [stats['success_rate_4mm'] for stats in per_landmark_stats]
    
    # Plot per-landmark MED
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(landmark_meds)), landmark_meds)
    plt.title('Mean Euclidean Distance Per Landmark')
    plt.xlabel('Landmark Index')
    plt.ylabel('MED (pixels)')
    plt.axhline(y=np.mean(landmark_meds), color='r', linestyle='-', label=f'Average: {np.mean(landmark_meds):.2f}')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, 'per_landmark_med.png'))
    plt.show()
    
    # Plot per-landmark success rates
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(landmark_success_rates_2mm)), landmark_success_rates_2mm, label='2mm')
    plt.bar(range(len(landmark_success_rates_4mm)), landmark_success_rates_4mm, alpha=0.5, label='4mm')
    plt.title('Success Rate Per Landmark')
    plt.xlabel('Landmark Index')
    plt.ylabel('Success Rate')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, 'per_landmark_success_rate.png'))
    plt.show()
else:
    print("Per-landmark statistics not available")

# %% [markdown]
# ## 14. Model Evaluation from Checkpoint
# 
# We can load a specific model checkpoint (e.g., best model by MED) and evaluate it on the test set to see detailed metrics.

# %%
def evaluate_checkpoint(checkpoint_path, data_processor=None, test_loader=None):
    """
    Load a specific checkpoint and evaluate it on the test set.
    
    Args:
        checkpoint_path (str): Path to the model checkpoint file.
        data_processor (DataProcessor, optional): Data processor to create test loader if not provided.
        test_loader (DataLoader, optional): Test data loader. If None, will be created from data_processor.
        
    Returns:
        tuple: (evaluation_results, trainer) - Evaluation results dictionary and the trainer instance
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    
    # Create a new trainer instance with the same parameters as the training run
    # You can customize these parameters if they differ from the training run
    eval_trainer = LandmarkTrainer(
        num_landmarks=NUM_LANDMARKS,
        learning_rate=LEARNING_RATE,
        device=device,
        output_dir=OUTPUT_DIR,
        use_refinement=USE_REFINEMENT,
        use_depth=USE_DEPTH_FEATURES,
        depth_channels=DEPTH_CHANNELS,
        hrnet_type=HRNET_TYPE,
        output_size=OUTPUT_SIZE if 'OUTPUT_SIZE' in globals() else (64, 64),
        image_size=IMAGE_SIZE if 'IMAGE_SIZE' in globals() else (224, 224),
        use_loss_normalization=USE_LOSS_NORMALIZATION
    )
    
    # Set the model to evaluation mode BEFORE the dummy forward pass
    # This prevents batch normalization issues with batch size 1
    eval_trainer.model.eval()
    
    # Run a dummy forward pass to initialize all model components
    # This ensures the heatmap_layer is created before loading the checkpoint
    dummy_input = torch.zeros((1, 3, 224, 224), device=device)
    dummy_depth = None
    if USE_DEPTH_FEATURES:
        dummy_depth = torch.zeros((1, 1, 224, 224), device=device)
    
    print("Initializing model components with dummy forward pass...")
    with torch.no_grad():
        _ = eval_trainer.model(dummy_input, dummy_depth)
    print("Model components initialized")
    
    # Now load the checkpoint
    eval_trainer.load_checkpoint(checkpoint_path)
    print("Model checkpoint loaded successfully")
    
    # Create test loader if not provided
    if test_loader is None:
        if data_processor is None:
            raise ValueError("Either test_loader or data_processor must be provided")
        
        _, _, test_loader = data_processor.create_data_loaders(
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            balance_classes=False,  # No need to balance for test set
            use_depth=USE_DEPTH_FEATURES
        )
        print(f"Created test loader with {len(test_loader.dataset)} samples")
    
    # Evaluate on test set with detailed metrics
    print("\nEvaluating model on test set and gathering raw pixel predictions...")
    
    # Set save_visualizations=False to avoid the _visualize_landmarks error
    # We need the raw predictions to apply patient-specific calibration
    model_results = eval_trainer.evaluate(test_loader, save_visualizations=False, save_predictions=True)
    
    # IMPROVED CALIBRATION APPROACH
    # Check if the test dataset has ruler point columns and if we have saved predictions
    test_df = data_processor.df.copy()
    if (all(col in test_df.columns for col in ['ruler_point_up_x', 'ruler_point_up_y', 
                                              'ruler_point_down_x', 'ruler_point_down_y']) and
        'patient_id' in test_df.columns and
        'predictions' in model_results):
        
        print("\nApplying patient-specific calibration from pixels to millimeters...")
        
        # Patients with 20mm rulers instead of 10mm
        patients_with_20mm_ruler = ["16", "17", "811"]
        
        # Create a dictionary to store patient_id -> calibration factor
        patient_calibration_factors = {}
        
        # Calculate calibration factor for each patient with valid ruler points
        for _, row in test_df.iterrows():
            patient_id = str(row['patient_id'])
            
            if (pd.notna(row['ruler_point_up_x']) and pd.notna(row['ruler_point_up_y']) and 
                pd.notna(row['ruler_point_down_x']) and pd.notna(row['ruler_point_down_y'])):
                
                # Calculate Euclidean distance between ruler points
                ruler_dist_px = np.sqrt(
                    (row['ruler_point_up_x'] - row['ruler_point_down_x'])**2 + 
                    (row['ruler_point_up_y'] - row['ruler_point_down_y'])**2
                )
                
                # Determine ruler size based on patient ID
                ruler_size_mm = 20.0 if patient_id in patients_with_20mm_ruler else 10.0
                
                # Calculate mm/pixel calibration factor for this patient
                if ruler_dist_px > 0:  # Avoid division by zero
                    calibration_factor = ruler_size_mm / ruler_dist_px
                    patient_calibration_factors[patient_id] = calibration_factor
                    
                    if patient_id in patients_with_20mm_ruler:
                        print(f"Patient {patient_id}: 20mm ruler = {ruler_dist_px:.2f}px, factor = {calibration_factor:.5f} mm/px")
                    
        print(f"Calculated calibration factors for {len(patient_calibration_factors)} patients")
        
        # Now apply patient-specific calibration to saved predictions and recalculate metrics
        if patient_calibration_factors and 'predictions' in model_results:
            predictions = model_results['predictions']  # List of prediction dicts
            
            # Lists to store calibrated distances
            all_calibrated_distances = []
            per_landmark_calibrated_distances = [[] for _ in range(NUM_LANDMARKS)]
            
            # Apply calibration to each prediction
            for pred in predictions:
                patient_id = str(pred['patient_id']) if 'patient_id' in pred else None
                
                if patient_id in patient_calibration_factors:
                    # Get patient-specific calibration factor
                    calib_factor = patient_calibration_factors[patient_id]
                    
                    # Apply calibration to distances
                    if 'distances' in pred:
                        # These are per-landmark distances in pixels
                        pixel_distances = pred['distances']
                        
                        # Convert to mm using patient-specific factor
                        mm_distances = [dist * calib_factor for dist in pixel_distances]
                        
                        # Store calibrated distances for overall and per-landmark statistics
                        all_calibrated_distances.extend(mm_distances)
                        
                        # Store distances by landmark index
                        for i, dist_mm in enumerate(mm_distances):
                            if i < len(per_landmark_calibrated_distances):
                                per_landmark_calibrated_distances[i].append(dist_mm)
            
            # Calculate overall MED in mm (mean of all calibrated distances)
            if all_calibrated_distances:
                calibrated_med_mm = np.mean(all_calibrated_distances)
                model_results['mean_euclidean_distance_mm'] = calibrated_med_mm
                
                # Calculate success rates in mm (using the same 2mm and 4mm thresholds)
                success_rate_2mm = np.mean([dist <= 2.0 for dist in all_calibrated_distances])
                success_rate_4mm = np.mean([dist <= 4.0 for dist in all_calibrated_distances])
                
                model_results['success_rate_2mm_mm'] = success_rate_2mm
                model_results['success_rate_4mm_mm'] = success_rate_4mm
                
                # Calculate per-landmark MED and success rates in mm
                model_results['per_landmark_stats_mm'] = []
                
                for i, landmark_distances in enumerate(per_landmark_calibrated_distances):
                    if landmark_distances:
                        med_mm = np.mean(landmark_distances)
                        sr_2mm = np.mean([dist <= 2.0 for dist in landmark_distances])
                        sr_4mm = np.mean([dist <= 4.0 for dist in landmark_distances])
                        
                        model_results['per_landmark_stats_mm'].append({
                            'landmark_idx': i,
                            'med_mm': med_mm,
                            'success_rate_2mm_mm': sr_2mm,
                            'success_rate_4mm_mm': sr_4mm
                        })
                
                # Print overall metrics in millimeters (patient-specific calibration)
                print("\nOverall Results (patient-specific calibration to mm):")
                print(f"Mean Euclidean Distance (MED): {calibrated_med_mm:.2f} mm")
                print(f"Success Rate (2mm): {success_rate_2mm * 100:.2f}%")
                print(f"Success Rate (4mm): {success_rate_4mm * 100:.2f}%")
                
                # Print per-landmark metrics in millimeters
                print("\nPer-Landmark Results (patient-specific calibration to mm):")
                print("{:<4} {:<15} {:<15} {:<15}".format("ID", "MED (mm)", "SR 2mm (%)", "SR 4mm (%)"))
                print("-" * 50)
                
                for stats in model_results['per_landmark_stats_mm']:
                    print("{:<4} {:<15.2f} {:<15.2f} {:<15.2f}".format(
                        stats['landmark_idx'], 
                        stats['med_mm'], 
                        stats['success_rate_2mm_mm'] * 100,
                        stats['success_rate_4mm_mm'] * 100
                    ))
                
                # Create visualization of per-landmark MEDs in mm
                plt.figure(figsize=(12, 6))
                landmark_indices = [i for i in range(len(model_results['per_landmark_stats_mm']))]
                med_values_mm = [stats['med_mm'] for stats in model_results['per_landmark_stats_mm']]
                
                plt.bar(landmark_indices, med_values_mm)
                plt.axhline(y=calibrated_med_mm, color='r', linestyle='-', 
                            label=f'Overall MED: {calibrated_med_mm:.2f} mm')
                
                plt.xlabel('Landmark Index')
                plt.ylabel('MED (mm)')
                plt.title('Mean Euclidean Distance per Landmark (patient-specific calibration, mm)')
                plt.xticks(landmark_indices)
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(OUTPUT_DIR, 'patient_specific_calibration_med_mm.png'))
                plt.show()
            else:
                print("Warning: No calibrated distances calculated. Using original pixel measurements.")
                # Fall back to using pixel measurements
                print_pixel_results(model_results)
        else:
            print("Warning: No calibration factors or predictions available. Using original pixel measurements.")
            # Fall back to using pixel measurements
            print_pixel_results(model_results)
    else:
        print("\nRuler point columns or patient IDs not found in test dataset. Reporting results in pixels only.")
        # Print results in pixels
        print_pixel_results(model_results)
    
    # Return both the evaluation results and the trainer for additional use
    return model_results, eval_trainer

def print_pixel_results(detailed_results):
    """Helper function to print evaluation results in pixels"""
    # Print overall metrics in pixels
    print("\nOverall Results (in pixels):")
    print(f"Mean Euclidean Distance (MED): {detailed_results['mean_euclidean_distance']:.2f} pixels")
    print(f"Success Rate (2mm): {detailed_results['success_rate_2mm'] * 100:.2f}%")
    print(f"Success Rate (4mm): {detailed_results['success_rate_4mm'] * 100:.2f}%")
    
    # Print per-landmark metrics in pixels
    print("\nPer-Landmark Results (in pixels):")
    print("{:<4} {:<15} {:<15} {:<15}".format("ID", "MED (pixels)", "SR 2mm (%)", "SR 4mm (%)"))
    print("-" * 50)
    
    for i, stats in enumerate(detailed_results['per_landmark_stats']):
        print("{:<4} {:<15.2f} {:<15.2f} {:<15.2f}".format(
            i, 
            stats['med'], 
            stats['success_rate_2mm'] * 100,
            stats['success_rate_4mm'] * 100
        ))
    
    # Create visualization of per-landmark MEDs in pixels
    plt.figure(figsize=(12, 6))
    landmark_indices = [i for i in range(len(detailed_results['per_landmark_stats']))]
    med_values = [stats['med'] for stats in detailed_results['per_landmark_stats']]
    
    plt.bar(landmark_indices, med_values)
    plt.axhline(y=detailed_results['mean_euclidean_distance'], color='r', linestyle='-', 
                label=f'Overall MED: {detailed_results["mean_euclidean_distance"]:.2f} px')
    
    plt.xlabel('Landmark Index')
    plt.ylabel('MED (pixels)')
    plt.title('Mean Euclidean Distance per Landmark (pixels)')
    plt.xticks(landmark_indices)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, 'pixel_med.png'))
    plt.show()

# %%
# Configuration
CHECKPOINT_TO_EVALUATE = os.path.join(OUTPUT_DIR, 'best_model_med.pth')  # Path to the checkpoint you want to evaluate
SAVE_VISUALIZATIONS = True  # Whether to save visualizations of predictions

# Evaluate the checkpoint
evaluation_results, eval_trainer = evaluate_checkpoint(
    checkpoint_path=CHECKPOINT_TO_EVALUATE,
    data_processor=data_processor,
    test_loader=test_loader
)

# %% [markdown]
# ## 15. Visualize Individual Landmark Predictions
# 
# We can visualize predictions for specific landmarks or samples to better understand model performance.

# %%
def visualize_landmark_predictions(trainer, test_loader, num_samples=3, landmark_indices=None):
    """
    Visualize predictions for specific landmarks or samples
    
    Args:
        trainer (LandmarkTrainer): Trained model trainer
        test_loader (DataLoader): Test data loader
        num_samples (int): Number of samples to visualize
        landmark_indices (list, optional): List of specific landmark indices to highlight
    """
    trainer.model.eval()
    
    # Get a batch of data
    batch = next(iter(test_loader))
    images = batch['image'].to(trainer.device)
    gt_landmarks = batch['landmarks']
    
    # Get depth maps if available and model uses them
    depth = None
    if trainer.use_depth and 'depth' in batch:
        depth = batch['depth'].to(trainer.device)
    
    # Make predictions
    with torch.no_grad():
        pred_landmarks = trainer.model.predict_landmarks(images, depth, scale_factor=trainer.coord_scale_factor)
    
    # Create visualization output directory
    vis_dir = os.path.join(OUTPUT_DIR, 'landmark_visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Visualize predictions for each sample
    for i in range(min(num_samples, len(images))):
        # Convert tensors to numpy
        img = images[i].cpu().permute(1, 2, 0).numpy()
        gt_lm = gt_landmarks[i].cpu().numpy()
        pred_lm = pred_landmarks[i].cpu().numpy()
        
        # Denormalize image if needed
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        # Calculate error for each landmark
        errors = np.sqrt(np.sum((gt_lm - pred_lm)**2, axis=1))
        
        # Create figure
        plt.figure(figsize=(12, 10))
        plt.imshow(img)
        
        # Plot all ground truth landmarks
        plt.scatter(gt_lm[:, 0], gt_lm[:, 1], c='green', marker='x', s=100, alpha=0.7, label='Ground Truth')
        
        # Plot all predicted landmarks
        plt.scatter(pred_lm[:, 0], pred_lm[:, 1], c='red', marker='o', s=80, alpha=0.7, label='Prediction')
        
        # Highlight specific landmarks if requested
        if landmark_indices is not None:
            # Plot connecting lines between GT and predictions for selected landmarks
            for idx in landmark_indices:
                if idx < len(gt_lm):
                    plt.plot([gt_lm[idx, 0], pred_lm[idx, 0]], 
                             [gt_lm[idx, 1], pred_lm[idx, 1]], 
                             'b-', alpha=0.5)
                    
                    # Add landmark index and error labels
                    plt.annotate(f"{idx}: {errors[idx]:.1f}px", 
                                 (pred_lm[idx, 0], pred_lm[idx, 1]),
                                 xytext=(5, 5), textcoords='offset points',
                                 color='blue', fontsize=9, fontweight='bold')
        
        plt.title(f'Landmark Predictions - Sample {i+1}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f'landmark_predictions_sample_{i+1}.png'))
        plt.show()

# %%
# Configure which landmarks to highlight
LANDMARK_INDICES_TO_HIGHLIGHT = [0, 1, 2, 3]  # Customize these indices based on your specific landmarks

# Visualize predictions with highlighted landmarks
visualize_landmark_predictions(
    trainer=eval_trainer,  # Use the trainer from the checkpoint evaluation
    test_loader=test_loader,
    num_samples=3,
    landmark_indices=LANDMARK_INDICES_TO_HIGHLIGHT
) 