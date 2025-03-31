#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cephalometric Landmark Detection Training Script

This script provides a command-line interface for training
landmark detection models on cephalometric X-ray images.
"""

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import platform
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm
from torchvision import transforms

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path for local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import model and training components
from src.models.trainer import LandmarkTrainer
from src.data.dataset import CephalometricDataset, ToTensor, Normalize
from src.data.data_augmentation import get_train_transforms
from src.data.data_processor import DataProcessor
from src.data.patient_classifier import PatientClassifier

# Define TrainTransform as a top-level class so it can be pickled
class TrainTransform:
    def __init__(self, train_augmentations, base_transforms):
        self.train_augmentations = train_augmentations
        self.base_transforms = base_transforms
        
    def __call__(self, sample):
        # First apply augmentation
        augmented = self.train_augmentations(sample)
        # Then apply base transforms (ToTensor, Normalize)
        return self.base_transforms(augmented)

def parse_args():
    parser = argparse.ArgumentParser(description='Train a landmark detection model')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset file')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Directory to save outputs')
    parser.add_argument('--apply_clahe', action='store_true', help='Apply CLAHE for histogram equalization')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--save_freq', type=int, default=5, help='Frequency of saving model checkpoints')
    
    # Model arguments
    parser.add_argument('--num_landmarks', type=int, default=19, help='Number of landmarks to detect')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained HRNet backbone')
    parser.add_argument('--use_refinement', action='store_true', help='Use refinement MLP for coordinate regression')
    parser.add_argument('--heatmap_weight', type=float, default=1.0, help='Weight for heatmap loss')
    parser.add_argument('--coord_weight', type=float, default=0.1, help='Weight for coordinate loss')
    
    # Data balancing arguments
    parser.add_argument('--balance_classes', action='store_true', 
                       help='Balance training data based on skeletal classification (preserves validation and test distributions)')
    parser.add_argument('--balance_method', type=str, choices=['upsample', 'downsample'], default='upsample', 
                        help='Method to balance training data classes (upsample: duplicate minority classes, downsample: reduce majority class)')
    
    # Device arguments
    parser.add_argument('--use_mps', action='store_true', help='Use Metal Performance Shaders (MPS) for Mac GPU acceleration')
    parser.add_argument('--force_cpu', action='store_true', help='Force CPU usage even if GPU is available')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of worker threads for dataloader (0 for single process)')
    
    # Learning rate scheduler arguments
    parser.add_argument('--scheduler', type=str, choices=['cosine', 'plateau', 'onecycle', 'none'], default='none', 
                        help='Learning rate scheduler type')
    parser.add_argument('--lr_patience', type=int, default=5, help='Patience for ReduceLROnPlateau scheduler')
    parser.add_argument('--lr_factor', type=float, default=0.5, help='Factor to reduce learning rate for ReduceLROnPlateau')
    parser.add_argument('--lr_min', type=float, default=1e-6, help='Minimum learning rate for schedulers')
    parser.add_argument('--lr_t_max', type=int, default=10, help='T_max parameter for CosineAnnealingLR (defaults to num_epochs/2 if not specified)')
    # OneCycleLR specific parameters
    parser.add_argument('--max_lr', type=float, default=None, help='Maximum learning rate for OneCycleLR (defaults to 10x learning_rate if None)')
    parser.add_argument('--pct_start', type=float, default=0.3, help='Percentage of training to increase learning rate for OneCycleLR')
    parser.add_argument('--div_factor', type=float, default=25.0, help='Initial learning rate division factor for OneCycleLR')
    parser.add_argument('--final_div_factor', type=float, default=1e4, help='Final learning rate division factor for OneCycleLR')
    
    # LR Range Test parameters
    parser.add_argument('--run_lr_range_test', action='store_true', help='Run learning rate range test before training')
    parser.add_argument('--lr_test_start', type=float, default=1e-7, help='Starting learning rate for range test')
    parser.add_argument('--lr_test_end', type=float, default=1.0, help='End learning rate for range test')
    parser.add_argument('--lr_test_iterations', type=int, default=100, help='Number of iterations for range test')
    
    # Optimizer arguments
    parser.add_argument('--optimizer', type=str, choices=['adam', 'adamw', 'sgd'], default='adam',
                        help='Optimizer type to use for training')
    parser.add_argument('--momentum', type=float, default=0.9, 
                        help='Momentum factor for SGD optimizer')
    parser.add_argument('--nesterov', action='store_true',
                        help='Enable Nesterov momentum for SGD optimizer')
    
    return parser.parse_args()

def set_seed(seed):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_dataloader_with_augmentations(df, landmark_cols, batch_size=16, 
                                        train_ratio=0.8, val_ratio=0.1, 
                                        apply_clahe=True, root_dir=None, 
                                        num_workers=4, balance_classes=False):
    """
    Create train, validation, and test DataLoaders with proper augmentations
    
    Args:
        df (pandas.DataFrame): DataFrame containing the dataset
        landmark_cols (list): List of column names containing landmark coordinates
        batch_size (int): Batch size for dataloaders
        train_ratio (float): Ratio of data to use for training
        val_ratio (float): Ratio of data to use for validation
        apply_clahe (bool): Whether to apply CLAHE for histogram equalization
        root_dir (str): Directory containing images (if images are stored as files)
        num_workers (int): Number of worker threads for dataloader
        balance_classes (bool): Whether to balance training data using skeletal classification
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # If 'set' column is already present, use it for splitting
    if 'set' in df.columns:
        train_df = df[df['set'] == 'train'].copy()
        val_df = df[df['set'] == 'dev'].copy()
        test_df = df[df['set'] == 'test'].copy()
    else:
        # Randomly split the data
        n = len(df)
        indices = np.random.permutation(n)
        
        train_size = int(train_ratio * n)
        val_size = int(val_ratio * n)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        train_df = df.iloc[train_indices].copy()
        val_df = df.iloc[val_indices].copy()
        test_df = df.iloc[test_indices].copy()
    
    # Display original class distribution before balancing
    if balance_classes and 'skeletal_class' in df.columns:
        # Count classes in original training data
        if 'skeletal_class' in train_df.columns:
            train_class_counts = train_df['skeletal_class'].value_counts().sort_index()
            print("Original training class distribution:")
            for label, count in train_class_counts.items():
                class_name = {1: "Class I", 2: "Class II", 3: "Class III"}.get(label, f"Class {label}")
                print(f"  {class_name}: {count} samples ({count/len(train_df)*100:.1f}%)")
    
    # Balance ONLY the training data if requested
    if balance_classes:
        # First, make sure we have skeletal class information
        if 'skeletal_class' not in train_df.columns:
            print("Computing skeletal classifications for the training set...")
            classifier = PatientClassifier(landmark_cols)
            train_df = classifier.classify_patients(train_df)
        
        # Now balance the training data
        print("Balancing training data using skeletal classification...")
        classifier = PatientClassifier(landmark_cols)
        train_df = classifier.balance_classes(train_df, class_column='skeletal_class', balance_method='upsample')
        print(f"Balanced training data: {len(train_df)} samples")
        
        # Show balanced distribution
        if 'skeletal_class' in train_df.columns:
            train_class_counts = train_df['skeletal_class'].value_counts().sort_index()
            print("Balanced training class distribution:")
            for label, count in train_class_counts.items():
                class_name = {1: "Class I", 2: "Class II", 3: "Class III"}.get(label, f"Class {label}")
                print(f"  {class_name}: {count} samples ({count/len(train_df)*100:.1f}%)")
    
    # Define data transformations (non-augmentation)
    base_transforms = transforms.Compose([
        ToTensor(),
        Normalize()
    ])
    
    # Training transformations with augmentations
    train_augmentations = get_train_transforms(include_horizontal_flip=False)
    
    # Create custom transform that applies augmentations before tensor conversion
    train_transform = TrainTransform(train_augmentations, base_transforms)
    
    # Create datasets
    train_dataset = CephalometricDataset(
        train_df, root_dir=root_dir, transform=train_transform, 
        landmark_cols=landmark_cols, train=True, apply_clahe=apply_clahe
    )
    
    val_dataset = CephalometricDataset(
        val_df, root_dir=root_dir, transform=base_transforms, 
        landmark_cols=landmark_cols, train=False, apply_clahe=apply_clahe
    )
    
    test_dataset = CephalometricDataset(
        test_df, root_dir=root_dir, transform=base_transforms, 
        landmark_cols=landmark_cols, train=False, apply_clahe=apply_clahe
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define landmark columns - fix Gonion column names
    landmark_cols = ['sella_x', 'sella_y', 'nasion_x', 'nasion_y', 'A point_x', 'A point_y',
       'B point_x', 'B point_y', 'upper 1 tip_x', 'upper 1 tip_y',
       'upper 1 apex_x', 'upper 1 apex_y', 'lower 1 tip_x', 'lower 1 tip_y',
       'lower 1 apex_x', 'lower 1 apex_y', 'ANS_x', 'ANS_y', 'PNS_x', 'PNS_y',
       'Gonion _x', 'Gonion _y', 'Menton_x', 'Menton_y', 'ST Nasion_x',
       'ST Nasion_y', 'Tip of the nose_x', 'Tip of the nose_y', 'Subnasal_x',
       'Subnasal_y', 'Upper lip_x', 'Upper lip_y', 'Lower lip_x',
       'Lower lip_y', 'ST Pogonion_x', 'ST Pogonion_y', 'gnathion_x',
       'gnathion_y']
    
    # Extract landmark names for evaluation
    landmark_names = []
    for i in range(0, len(landmark_cols), 2):
        # Extract name from column (removing _x or _y suffix)
        name = landmark_cols[i].replace('_x', '')
        landmark_names.append(name)
    
    # Initialize data processor
    data_processor = DataProcessor(
        data_path=args.data_path,
        landmark_cols=landmark_cols,
        image_size=(224, 224),
        apply_clahe=args.apply_clahe
    )
    
    # Load and preprocess data - but DON'T balance here!
    # We only want to balance the training set, not all data
    df = data_processor.preprocess_data(balance_classes=False)
    
    # Compute skeletal classifications for reporting, if class balancing is requested
    if args.balance_classes and df is not None:
        print("Computing skeletal classifications for reporting purposes...")
        df = data_processor.compute_patient_classes()
    
    # Get dataset statistics
    stats = data_processor.get_data_stats()
    print("Dataset Statistics:")
    print(f"Total samples: {stats['total_samples']}")
    
    if 'set_counts' in stats:
        print("Set counts:")
        for set_name, count in stats['set_counts'].items():
            print(f"  {set_name}: {count}")
    
    if 'class_counts' in stats:
        print("Class counts:")
        for class_name, count in stats['class_counts'].items():
            print(f"  {class_name}: {count}")
    
    if 'skeletal_class_counts' in stats:
        print("Overall Skeletal Class Distribution:")
        for class_label, count in stats['skeletal_class_counts'].items():
            class_name = stats['skeletal_class_names'].get(class_label, str(class_label))
            print(f"  {class_name}: {count} patients ({count/stats['total_samples']*100:.1f}%)")
    
    # Check MPS availability for Mac users
    if args.use_mps and platform.system() == 'Darwin':
        if torch.backends.mps.is_available():
            print("MPS (Metal Performance Shaders) is available for Mac GPU acceleration.")
        else:
            print("WARNING: MPS requested but not available. Falling back to CPU.")
            args.use_mps = False
    
    # Create data loaders with augmentation (only balance the training set)
    print("Creating dataloaders with augmentation for training set...")
    train_loader, val_loader, test_loader = create_dataloader_with_augmentations(
        df=df,
        landmark_cols=landmark_cols,
        batch_size=args.batch_size,
        train_ratio=0.8,
        val_ratio=0.1,
        apply_clahe=args.apply_clahe,
        root_dir=None,
        num_workers=args.num_workers,
        balance_classes=args.balance_classes  # Only the training set will be balanced
    )
    
    print(f"Created data loaders:")
    print(f"  Training samples: {len(train_loader.dataset)} {'(balanced)' if args.balance_classes else ''}")
    print(f"  Validation samples: {len(val_loader.dataset)} (original distribution)")
    print(f"  Test samples: {len(test_loader.dataset)} (original distribution)")
    
    # Determine device
    device = None
    if args.force_cpu:
        device = torch.device('cpu')
        print("Using CPU as requested with --force_cpu")
    
    # Log training configuration
    print("\nTraining Configuration:")
    print(f"  Model: HRNet-W32 with{'' if args.use_refinement else 'out'} refinement MLP")
    
    # Log optimizer info
    print(f"  Optimizer: {args.optimizer.upper()}")
    if args.optimizer == 'sgd':
        print(f"    Momentum: {args.momentum}")
        print(f"    Nesterov: {args.nesterov}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Weight decay: {args.weight_decay}")
    
    print(f"  Batch size: {args.batch_size}")
    print(f"  Number of epochs: {args.num_epochs}")
    if args.use_refinement:
        print(f"  Heatmap loss weight: {args.heatmap_weight}")
        print(f"  Coordinate loss weight: {args.coord_weight}")
    if args.balance_classes:
        print(f"  Training data balance method: {args.balance_method} (validation and test sets maintain original distribution)")
    print(f"  Device: {'MPS (Mac GPU)' if args.use_mps else 'CUDA' if torch.cuda.is_available() and not args.force_cpu else 'CPU'}")
    print(f"  Output directory: {args.output_dir}")
    
    # Run LR Range Test if requested (for OneCycleLR only)
    if args.run_lr_range_test and args.scheduler == 'onecycle':
        from src.utils.lr_range_test import lr_range_test
        from src.models.losses import CombinedLoss
        from src.models.hrnet import create_hrnet_model
        
        print("\nRunning Learning Rate Range Test to find optimal max_lr for OneCycleLR...")
        
        # Create a temporary model for the test (same architecture as for training)
        temp_model = create_hrnet_model(
            num_landmarks=args.num_landmarks,
            pretrained=args.pretrained,
            use_refinement=args.use_refinement
        )
        
        # Create loss function
        test_criterion = CombinedLoss(
            heatmap_weight=args.heatmap_weight,
            coord_weight=args.coord_weight,
            output_size=(64, 64),   # Heatmap size
            image_size=(224, 224)   # Original image size
        )
        
        # Create optimizer
        if args.optimizer == 'adam':
            test_optimizer = torch.optim.Adam(
                temp_model.parameters(), 
                lr=args.learning_rate, 
                weight_decay=args.weight_decay
            )
        elif args.optimizer == 'adamw':
            test_optimizer = torch.optim.AdamW(
                temp_model.parameters(), 
                lr=args.learning_rate, 
                weight_decay=args.weight_decay
            )
        elif args.optimizer == 'sgd':
            test_optimizer = torch.optim.SGD(
                temp_model.parameters(), 
                lr=args.learning_rate, 
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                nesterov=args.nesterov
            )
        
        # Move model to device
        if args.use_mps and platform.system() == 'Darwin' and torch.backends.mps.is_available():
            device = torch.device('mps')
        elif torch.cuda.is_available() and not args.force_cpu:
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        
        temp_model.to(device)
        
        # Run the LR Range Test
        lr_test_results = lr_range_test(
            model=temp_model,
            train_loader=train_loader,
            criterion=test_criterion,
            optimizer=test_optimizer,
            device=device,
            start_lr=args.lr_test_start,
            end_lr=args.lr_test_end,
            num_iterations=args.lr_test_iterations,
            smooth_window=0.05,
            diverge_threshold=5.0,
            output_dir=args.output_dir
        )
        
        # Update max_lr based on the test results
        old_max_lr = args.max_lr if args.max_lr else (args.learning_rate * 10)
        args.max_lr = lr_test_results['suggested_max_lr']
        
        print(f"\nLR Range Test completed. Results:")
        print(f"  - Steepest gradient LR: {lr_test_results['steepest_slope_lr']:.6f}")
        print(f"  - Min loss LR: {lr_test_results['min_loss_lr']:.6f}")
        print(f"  - Suggested max_lr: {args.max_lr:.6f} (previous: {old_max_lr})")
        
        # Clean up memory
        del temp_model, test_optimizer, test_criterion
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Create trainer
    trainer = LandmarkTrainer(
        num_landmarks=args.num_landmarks,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=device,  # Auto-select or forced CPU
        output_dir=args.output_dir,
        use_refinement=args.use_refinement,
        heatmap_weight=args.heatmap_weight,
        coord_weight=args.coord_weight,
        use_mps=args.use_mps,
        # Weight scheduling parameters
        use_weight_schedule=args.use_weight_schedule,
        initial_heatmap_weight=args.initial_heatmap_weight,
        initial_coord_weight=args.initial_coord_weight,
        final_heatmap_weight=args.final_heatmap_weight,
        final_coord_weight=args.final_coord_weight,
        weight_schedule_epochs=args.weight_schedule_epochs,
        # Learning rate scheduler parameters
        scheduler_type=None if args.scheduler == 'none' else args.scheduler,
        lr_patience=args.lr_patience,
        lr_factor=args.lr_factor,
        lr_min=args.lr_min,
        lr_t_max=args.lr_t_max if args.lr_t_max > 0 else args.num_epochs // 2,
        # OneCycleLR parameters
        max_lr=args.max_lr,
        pct_start=args.pct_start,
        div_factor=args.div_factor,
        final_div_factor=args.final_div_factor,
        # Optimizer parameters
        optimizer_type=args.optimizer,
        momentum=args.momentum,
        nesterov=args.nesterov
    )
    
    # Train model
    print("\nStarting training...")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        save_freq=args.save_freq
    )
    
    # Evaluate model on test set
    print("Evaluating model on test set...")
    results = trainer.evaluate(
        test_loader, 
        save_visualizations=True,
        landmark_names=landmark_names,
        landmark_cols=landmark_cols  # Pass landmark columns for skeletal classification
    )
    
    # Print evaluation results
    print("\nEvaluation Results:")
    print(f"Mean Euclidean Distance: {results['mean_euclidean_distance']:.2f} pixels")
    print(f"Success Rate (2mm): {results['success_rate_2mm'] * 100:.2f}%")
    print(f"Success Rate (4mm): {results['success_rate_4mm'] * 100:.2f}%")
    
    print("\nDetailed per-landmark metrics have been saved to:")
    print(f"  {os.path.join(args.output_dir, 'evaluation', 'reports')}")
    
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main() 