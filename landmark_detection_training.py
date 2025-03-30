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

# Add the src directory to the path for imports
import sys
sys.path.append('./src')

# Import model and training components
from src.models.hrnet import create_hrnet_model
from src.models.trainer import LandmarkTrainer
from src.data.dataset import CephalometricDataset, ToTensor, Normalize
from src.data.data_augmentation import get_train_transforms
from src.data.data_processor import DataProcessor

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

# Loss parameters
HEATMAP_WEIGHT = 1.0  # Weight for heatmap loss
COORD_WEIGHT = 0.1  # Weight for coordinate loss

# Training parameters
BATCH_SIZE = 16
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
SAVE_FREQ = 5  # Frequency of saving checkpoints

# Hardware parameters
USE_MPS = (platform.system() == 'Darwin')  # Automatically use MPS for Mac
NUM_WORKERS = 0  # Use 0 for single process (more stable)

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
        sample = {'image_path': f'synthetic_image_{i}.jpg', 'set': 'train' if i < 70 else ('dev' if i < 85 else 'test')}
        # Add random landmark coordinates
        for col in landmark_cols:
            sample[col] = np.random.randint(0, 224)
        synthetic_data.append(sample)
    
    df = pd.DataFrame(synthetic_data)
    print(f"Created synthetic dataset with {len(df)} samples")

# %% [markdown]
# ## 5. Create DataLoaders with Augmentation

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
base_transforms = torch.nn.Sequential(
    ToTensor(),
    Normalize()
)

# Get training transformations with augmentations
train_augmentations = get_train_transforms(include_horizontal_flip=False)

# Create custom transform for training
train_transform = TrainTransform(train_augmentations, base_transforms)

# Split data into train, val, test
train_df = df[df['set'] == 'train'] if 'set' in df.columns else df.sample(frac=0.7)
val_df = df[df['set'] == 'dev'] if 'set' in df.columns else df[~df.index.isin(train_df.index)].sample(frac=0.5)
test_df = df[df['set'] == 'test'] if 'set' in df.columns else df[~df.index.isin(train_df.index) & ~df.index.isin(val_df.index)]

# Create datasets
train_dataset = CephalometricDataset(
    train_df, root_dir=None, transform=train_transform, 
    landmark_cols=landmark_cols, train=True, apply_clahe=APPLY_CLAHE
)

val_dataset = CephalometricDataset(
    val_df, root_dir=None, transform=base_transforms, 
    landmark_cols=landmark_cols, train=False, apply_clahe=APPLY_CLAHE
)

test_dataset = CephalometricDataset(
    test_df, root_dir=None, transform=base_transforms, 
    landmark_cols=landmark_cols, train=False, apply_clahe=APPLY_CLAHE
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
    use_mps=USE_MPS
)

# Custom max_delta setting for the refinement MLP if needed
if USE_REFINEMENT and hasattr(trainer.model, 'refinement_mlp'):
    trainer.model.refinement_mlp.max_delta = MAX_DELTA
    print(f"Set refinement MLP max_delta to {MAX_DELTA}")

# Print model summary
total_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
print(f"Model created with {total_params:,} trainable parameters")
print(f"Refinement MLP: {'Enabled' if USE_REFINEMENT else 'Disabled'}")

# %% [markdown]
# ## 7. Train the Model

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
# ## 8. Evaluate the Model

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
# ## 9. Visualize Results

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

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'training_results.png'))
plt.show()

# %% [markdown]
# ## 10. Visualize Predictions
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
# ## 11. Save a Custom Model Configuration

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
    'heatmap_weight': HEATMAP_WEIGHT,
    'coord_weight': COORD_WEIGHT,
    'learning_rate': LEARNING_RATE,
    'weight_decay': WEIGHT_DECAY,
    'batch_size': BATCH_SIZE,
    'num_epochs': NUM_EPOCHS,
    'output_size': (64, 64),
    'image_size': (224, 224),
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
    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
}

save_model_config(OUTPUT_DIR, model_config)

print("\nTraining completed successfully!")

# %% [markdown]
# ## 12. Additional Analysis: Per-Landmark Performance

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