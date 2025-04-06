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

# Model parameters
NUM_LANDMARKS = 19
PRETRAINED = True
USE_REFINEMENT = True  # Whether to use refinement MLP
MAX_DELTA = 2.0  # Maximum allowed delta for refinement
HRNET_TYPE = 'w32'  # HRNet variant: 'w32' (default) or 'w48' (larger model)

# Depth feature parameters
USE_DEPTH = False  # Whether to use depth features
DEPTH_FUSION_METHOD = 'concat'  # Method to fuse depth features: 'concat', 'add', 'attention'
DEPTH_CACHE_DIR = './outputs/depth_cache'  # Directory to cache depth features

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

# Initialize data processor
try:
    data_processor = DataProcessor(
        data_path=DATA_PATH,
        landmark_cols=landmark_cols,
        image_size=(224, 224),
        apply_clahe=APPLY_CLAHE
    )
    
    # Load and preprocess data
    df = data_processor.preprocess_data()
    
    print(f"Dataset loaded with {len(df)} samples")
except Exception as e:
    print(f"Error loading dataset: {str(e)}")
    # Create a small synthetic dataset for demonstration purposes
    print("Creating synthetic dataset for demonstration...")
    num_samples = 100
    synthetic_data = []
    
    for i in range(num_samples):
        # Create sample with required columns for CephalometricDataset
        patient_id = f'patient_{i:03d}'
        sample = {
            'image_path': f'synthetic_image_{i}.jpg',
            'set': 'train' if i < 70 else ('dev' if i < 85 else 'test'),
            'patient': patient_id,  # Add patient ID column
            'patient_id': patient_id  # Also add patient_id for consistency
        }
        # Add random landmark coordinates
        for col in landmark_cols:
            sample[col] = np.random.randint(0, 224)
        synthetic_data.append(sample)
    
    df = pd.DataFrame(synthetic_data)
    print(f"Created synthetic dataset with {len(df)} samples")

# %% [markdown]
# ## 5. Create DataLoaders with Augmentation
# 
# In this section, we:
# 1. Define the data transformations for image preprocessing
# 2. Split the dataset into train/validation/test sets
# 3. Optionally balance the training data using skeletal classification
# 4. Create PyTorch DataLoaders for model training and evaluation

# %%
# Define a class for training transforms that can be pickled
class TrainTransform:
    def __init__(self, train_augmentations, base_transforms):
        self.train_augmentations = train_augmentations
        self.base_transforms = base_transforms
        
    def __call__(self, sample):
        # First apply augmentation
        augmented = self.train_augmentations(sample)
        # Then apply base transforms (ToTensor, Normalize)
        return self.base_transforms(augmented)

# Define data transformations
base_transforms = transforms.Compose([
    ToTensor(),
    Normalize()
])

# Get training transformations with augmentations
train_augmentations = get_train_transforms(include_horizontal_flip=False)

# Create custom transform for training
train_transform = TrainTransform(train_augmentations, base_transforms)

# Split data into train, val, test
train_df = df[df['set'] == 'train'] if 'set' in df.columns else df.sample(frac=0.7)
val_df = df[df['set'] == 'dev'] if 'set' in df.columns else df[~df.index.isin(train_df.index)].sample(frac=0.5)
test_df = df[df['set'] == 'test'] if 'set' in df.columns else df[~df.index.isin(train_df.index) & ~df.index.isin(val_df.index)]

# Optional: Balance training data classes
if BALANCE_CLASSES and 'skeletal_class' not in train_df.columns:
    print("Computing skeletal classifications for the training set...")
    from src.data.patient_classifier import PatientClassifier
    classifier = PatientClassifier(landmark_cols)
    train_df = classifier.classify_patients(train_df)

if BALANCE_CLASSES and 'skeletal_class' in train_df.columns:
    print("Balancing training data using skeletal classification...")
    from src.data.patient_classifier import PatientClassifier
    classifier = PatientClassifier(landmark_cols)
    
    # Show original class distribution
    train_class_counts = train_df['skeletal_class'].value_counts().sort_index()
    print("Original training class distribution:")
    for label, count in train_class_counts.items():
        class_name = {1: "Class I", 2: "Class II", 3: "Class III"}.get(label, f"Class {label}")
        print(f"  {class_name}: {count} samples ({count/len(train_df)*100:.1f}%)")
    
    # Balance classes using the specified method
    train_df = classifier.balance_classes(train_df, class_column='skeletal_class', balance_method=BALANCE_METHOD)
    print(f"Balanced training data using {BALANCE_METHOD}: {len(train_df)} samples")
    
    # Show balanced distribution
    train_class_counts = train_df['skeletal_class'].value_counts().sort_index()
    print("Balanced training class distribution:")
    for label, count in train_class_counts.items():
        class_name = {1: "Class I", 2: "Class II", 3: "Class III"}.get(label, f"Class {label}")
        print(f"  {class_name}: {count} samples ({count/len(train_df)*100:.1f}%)")

# Create datasets
if USE_DEPTH:
    print("\nInitializing depth feature extraction...")
    from src.utils.depth_features import DepthFeatureExtractor
    
    # Create depth feature cache directory
    os.makedirs(DEPTH_CACHE_DIR, exist_ok=True)
    
    # Initialize depth extractor
    depth_extractor = DepthFeatureExtractor(
        cache_dir=DEPTH_CACHE_DIR,
        device=device
    )
    
    # Create temporary datasets for feature extraction
    temp_transform = transforms.Compose([ToTensor()])
    
    temp_train_dataset = CephalometricDataset(
        train_df, transform=temp_transform, 
        landmark_cols=landmark_cols, apply_clahe=APPLY_CLAHE
    )
    
    temp_val_dataset = CephalometricDataset(
        val_df, transform=temp_transform, 
        landmark_cols=landmark_cols, apply_clahe=APPLY_CLAHE
    )
    
    temp_test_dataset = CephalometricDataset(
        test_df, transform=temp_transform, 
        landmark_cols=landmark_cols, apply_clahe=APPLY_CLAHE
    )
    
    print("Extracting depth features for training set...")
    train_features = depth_extractor.process_and_cache_dataset(temp_train_dataset, cache_suffix="train")
    
    print("Extracting depth features for validation set...")
    val_features = depth_extractor.process_and_cache_dataset(temp_val_dataset, cache_suffix="val")
    
    print("Extracting depth features for test set...")
    test_features = depth_extractor.process_and_cache_dataset(temp_test_dataset, cache_suffix="test")
    
    # Combine all features into a global index mapping
    depth_features = {}
    
    # For train set
    for i, global_idx in enumerate(train_df.index):
        if i in train_features:
            depth_features[global_idx] = train_features[i]
            
    # For validation set
    for i, global_idx in enumerate(val_df.index):
        if i in val_features:
            depth_features[global_idx] = val_features[i]
            
    # For test set
    for i, global_idx in enumerate(test_df.index):
        if i in test_features:
            depth_features[global_idx] = test_features[i]
    
    print(f"Extracted depth features for {len(depth_features)} images")
    
    # Save sample depth visualizations
    vis_dir = os.path.join(OUTPUT_DIR, 'depth_samples')
    os.makedirs(vis_dir, exist_ok=True)
    
    for i in range(min(5, len(temp_train_dataset))):
        sample = temp_train_dataset[i]
        image = sample['image']
        # Convert from tensor to numpy for visualization
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).numpy()
            
        save_path = os.path.join(vis_dir, f'sample_{i}.png')
        depth_extractor.save_depth_visualization(image, save_path)
else:
    depth_features = None

# Create datasets
train_dataset = CephalometricDataset(
    train_df, root_dir=None, transform=train_transform, 
    landmark_cols=landmark_cols, train=True, apply_clahe=APPLY_CLAHE,
    depth_features=train_features if USE_DEPTH else None, use_depth=USE_DEPTH
)

val_dataset = CephalometricDataset(
    val_df, root_dir=None, transform=base_transforms, 
    landmark_cols=landmark_cols, train=False, apply_clahe=APPLY_CLAHE,
    depth_features=val_features if USE_DEPTH else None, use_depth=USE_DEPTH
)

test_dataset = CephalometricDataset(
    test_df, root_dir=None, transform=base_transforms, 
    landmark_cols=landmark_cols, train=False, apply_clahe=APPLY_CLAHE,
    depth_features=test_features if USE_DEPTH else None, use_depth=USE_DEPTH
)

# Create dataloaders
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
    num_workers=NUM_WORKERS, pin_memory=True
)

val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
    num_workers=NUM_WORKERS, pin_memory=True
)

test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
    num_workers=NUM_WORKERS, pin_memory=True
)

print(f"Created data loaders:")
print(f"  Training samples: {len(train_dataset)}")
print(f"  Validation samples: {len(val_dataset)}")
print(f"  Test samples: {len(test_dataset)}")

# %% [markdown]
# ## 6. Create and Train the Model

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
    log_specific_landmark_indices=LOG_SPECIFIC_LANDMARK_INDICES,
    # Depth feature parameters
    use_depth=USE_DEPTH,
    depth_fusion_method=DEPTH_FUSION_METHOD
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

# Print depth feature info
print(f"Depth features: {'Enabled' if USE_DEPTH else 'Disabled'}")
if USE_DEPTH:
    print(f"  Fusion method: {DEPTH_FUSION_METHOD}")
    print(f"  Cache directory: {DEPTH_CACHE_DIR}")

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
        nesterov=NESTEROV,
        use_depth=USE_DEPTH,
        depth_fusion_method=DEPTH_FUSION_METHOD
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
    """
    Visualize sample predictions from the test set
    
    Args:
        trainer (LandmarkTrainer): Trained model
        test_loader (DataLoader): Test data loader
        num_samples (int): Number of samples to visualize
    """
    trainer.model.eval()
    
    # Get a batch of data
    for batch_idx, batch in enumerate(test_loader):
        if batch_idx > 0:  # Only use the first batch
            break
        
        # Get images and landmarks
        images = batch['image'].to(trainer.device)
        true_landmarks = batch['landmarks'].to(trainer.device)
        
        # Get depth maps if available
        depth_maps = None
        if USE_DEPTH and 'depth' in batch:
            depth_maps = batch['depth'].to(trainer.device)
            if depth_maps.dim() == 3:  # [B, H, W]
                depth_maps = depth_maps.unsqueeze(1)  # [B, 1, H, W]
        
        # Get predictions
        with torch.no_grad():
            if USE_DEPTH and depth_maps is not None:
                predictions = trainer.model.predict_landmarks(images, depth_maps)
            else:
                predictions = trainer.model.predict_landmarks(images)
        
        # Convert to numpy for plotting
        images_np = images.cpu().permute(0, 2, 3, 1).numpy()
        true_landmarks_np = true_landmarks.cpu().numpy()
        predictions_np = predictions.cpu().numpy()
        
        if USE_DEPTH and depth_maps is not None:
            depth_maps_np = depth_maps.cpu().squeeze(1).numpy()  # Remove channel dimension
        
        # Denormalize images
        if images_np.max() <= 1.0:
            # Using ImageNet normalization
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            
            # Denormalize: x = x*std + mean for each channel
            for i in range(len(images_np)):
                images_np[i] = images_np[i] * std + mean
            
            # Clip to valid range [0, 1]
            images_np = np.clip(images_np, 0, 1)
        
        # Create figure
        for i in range(min(num_samples, len(images_np))):
            if USE_DEPTH and depth_maps is not None:
                fig, axs = plt.subplots(1, 3, figsize=(18, 6))
                
                # Plot original image with landmarks
                axs[0].imshow(images_np[i])
                axs[0].scatter(predictions_np[i, :, 0], predictions_np[i, :, 1], 
                              c='blue', marker='o', label='Predicted')
                axs[0].scatter(true_landmarks_np[i, :, 0], true_landmarks_np[i, :, 1], 
                              c='red', marker='x', label='Ground Truth')
                axs[0].set_title('RGB Image with Landmarks')
                axs[0].legend()
                axs[0].axis('off')
                
                # Plot depth map
                axs[1].imshow(depth_maps_np[i], cmap='plasma')
                axs[1].set_title('Depth Map')
                axs[1].axis('off')
                
                # Plot depth map with landmarks overlay
                axs[2].imshow(depth_maps_np[i], cmap='plasma')
                axs[2].scatter(predictions_np[i, :, 0], predictions_np[i, :, 1], 
                              c='blue', marker='o', label='Predicted')
                axs[2].scatter(true_landmarks_np[i, :, 0], true_landmarks_np[i, :, 1], 
                              c='red', marker='x', label='Ground Truth')
                axs[2].set_title('Depth Map with Landmarks')
                axs[2].legend()
                axs[2].axis('off')
            else:
                # Regular visualization without depth
                plt.figure(figsize=(10, 10))
                plt.imshow(images_np[i])
                plt.scatter(predictions_np[i, :, 0], predictions_np[i, :, 1], 
                          c='blue', marker='o', label='Predicted')
                plt.scatter(true_landmarks_np[i, :, 0], true_landmarks_np[i, :, 1], 
                          c='red', marker='x', label='Ground Truth')
                plt.title('Landmark Predictions')
                plt.legend()
                plt.axis('off')
            
            # Calculate MED for this sample
            sample_pred = predictions_np[i:i+1]
            sample_true = true_landmarks_np[i:i+1]
            med = np.mean(np.sqrt(np.sum((sample_pred - sample_true) ** 2, axis=2)))
            
            plt.suptitle(f'Mean Euclidean Distance: {med:.2f} pixels')
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f'prediction_sample_{i}.png'))
            plt.close()
    
    print(f"Prediction visualizations saved to {OUTPUT_DIR}")

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
    """Save the model configuration to a file"""
    config_path = os.path.join(output_dir, 'model_config.txt')
    with open(config_path, 'w') as f:
        # General info
        f.write("# Model Configuration\n\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"HRNet Type: {config['hrnet_type'].upper()}\n")
        f.write(f"Number of Landmarks: {config['num_landmarks']}\n")
        f.write(f"Refinement MLP: {'Enabled' if config['use_refinement'] else 'Disabled'}\n")
        f.write(f"Max Delta: {config.get('max_delta', 'Not specified')}\n\n")
        
        # Depth feature parameters
        f.write("# Depth Feature Parameters\n")
        f.write(f"Use Depth Features: {'Enabled' if config['use_depth'] else 'Disabled'}\n")
        if config['use_depth']:
            f.write(f"Depth Fusion Method: {config['depth_fusion_method']}\n")
            f.write(f"Depth Cache Directory: {DEPTH_CACHE_DIR}\n")
        f.write("\n")
        
        # Training parameters
        f.write("# Training Parameters\n")
        f.write(f"Batch Size: {config['batch_size']}\n")
        f.write(f"Number of Epochs: {config['num_epochs']}\n")
        f.write(f"Learning Rate: {config['learning_rate']}\n")
        f.write(f"Weight Decay: {config['weight_decay']}\n\n")
        
        # Optimizer parameters
        f.write("# Optimizer Parameters\n")
        f.write(f"Optimizer: {config['optimizer_type'].upper()}\n")
        if config['optimizer_type'] == 'sgd':
            f.write(f"Momentum: {config['momentum']}\n")
            f.write(f"Nesterov: {config['nesterov']}\n")
        f.write("\n")
        
        # LR scheduler parameters
        f.write("# Learning Rate Scheduler Parameters\n")
        f.write(f"Scheduler Type: {config['scheduler_type'] if config['scheduler_type'] else 'None'}\n")
        if config['scheduler_type']:
            if config['scheduler_type'] == 'plateau':
                f.write(f"Patience: {config['lr_patience']}\n")
                f.write(f"Factor: {config['lr_factor']}\n")
            elif config['scheduler_type'] == 'cosine':
                f.write(f"T Max: {config['lr_t_max']}\n")
            elif config['scheduler_type'] == 'onecycle':
                f.write(f"Max LR: {config['max_lr']}\n")
                f.write(f"Pct Start: {config['pct_start']}\n")
                f.write(f"Div Factor: {config['div_factor']}\n")
                f.write(f"Final Div Factor: {config['final_div_factor']}\n")
            f.write(f"Min LR: {config['lr_min']}\n")
        f.write("\n")
        
        # Loss parameters
        f.write("# Loss Parameters\n")
        f.write(f"Loss Weight Scheduling: {'Enabled' if config['use_weight_schedule'] else 'Disabled'}\n")
        if config['use_weight_schedule']:
            f.write(f"Initial Heatmap Weight: {config['initial_heatmap_weight']}\n")
            f.write(f"Initial Coordinate Weight: {config['initial_coord_weight']}\n")
            f.write(f"Final Heatmap Weight: {config['final_heatmap_weight']}\n")
            f.write(f"Final Coordinate Weight: {config['final_coord_weight']}\n")
            f.write(f"Weight Schedule Epochs: {config['weight_schedule_epochs']}\n")
        else:
            f.write(f"Heatmap Weight: {config['heatmap_weight']}\n")
            f.write(f"Coordinate Weight: {config['coord_weight']}\n")
        f.write("\n")
        
        # Loss normalization parameters
        f.write("# Loss Normalization Parameters\n")
        f.write(f"Use Loss Normalization: {'Enabled' if config['use_loss_normalization'] else 'Disabled'}\n")
        if config['use_loss_normalization']:
            f.write(f"Normalization Decay: {config['norm_decay']}\n")
            f.write(f"Normalization Epsilon: {config['norm_epsilon']}\n")
        f.write("\n")
        
        # Per-landmark weighting
        f.write("# Per-Landmark Weighting Parameters\n")
        if config['target_landmark_indices']:
            f.write(f"Target Landmark Indices: {config['target_landmark_indices']}\n")
        else:
            f.write("Target Landmark Indices: None (all landmarks used)\n")
        
        if config['landmark_weights']:
            f.write(f"Landmark Weights: {config['landmark_weights']}\n")
        else:
            f.write("Landmark Weights: None (equal weighting)\n")
        f.write("\n")
        
        # Specific MED logging
        f.write("# Specific MED Logging\n")
        if config['log_specific_landmark_indices']:
            f.write(f"Log Specific Landmark Indices: {config['log_specific_landmark_indices']}\n")
        else:
            f.write("Log Specific Landmark Indices: None\n")
        f.write("\n")
        
        # Data parameters
        f.write("# Data Parameters\n")
        f.write(f"Data Path: {config['data_path']}\n")
        f.write(f"Apply CLAHE: {config['apply_clahe']}\n")
        f.write(f"Balance Classes: {'Enabled' if config['balance_classes'] else 'Disabled'}\n")
        if config['balance_classes']:
            f.write(f"Balance Method: {config['balance_method']}\n")
        f.write("\n")
        
        # Hardware parameters
        f.write("# Hardware Parameters\n")
        f.write(f"Device: {device}\n")
        f.write(f"Use MPS: {config['use_mps']}\n")
        f.write(f"Number of Workers: {config['num_workers']}\n")
        
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
    'train_samples': len(train_dataset),
    'val_samples': len(val_dataset),
    'test_samples': len(test_dataset),
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
    'log_specific_landmark_indices': LOG_SPECIFIC_LANDMARK_INDICES,
    'use_depth': USE_DEPTH,
    'depth_fusion_method': DEPTH_FUSION_METHOD,
    'data_path': DATA_PATH,
    'apply_clahe': APPLY_CLAHE,
    'use_mps': USE_MPS,
    'num_workers': NUM_WORKERS
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