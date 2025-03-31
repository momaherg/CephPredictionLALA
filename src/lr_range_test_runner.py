#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Learning Rate Range Test Runner

This script runs a standalone Learning Rate Range Test to help find the optimal
maximum learning rate for the OneCycleLR scheduler.
"""

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import platform
from torch.utils.data import DataLoader
from torchvision import transforms

# Add parent directory to path for local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import model and training components
from src.utils.lr_range_test import lr_range_test
from src.models.hrnet import create_hrnet_model
from src.models.losses import CombinedLoss
from src.data.dataset import CephalometricDataset, ToTensor, Normalize
from src.data.data_augmentation import get_train_transforms
from src.data.data_processor import DataProcessor


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


def parse_args():
    parser = argparse.ArgumentParser(description='Run a Learning Rate Range Test')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset file')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Directory to save outputs')
    parser.add_argument('--apply_clahe', action='store_true', help='Apply CLAHE for histogram equalization')
    
    # Model arguments
    parser.add_argument('--num_landmarks', type=int, default=19, help='Number of landmarks to detect')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained HRNet backbone')
    parser.add_argument('--use_refinement', action='store_true', help='Use refinement MLP for coordinate regression')
    parser.add_argument('--hrnet_type', type=str, default='w32', choices=['w32', 'w48'], help='HRNet backbone type')
    parser.add_argument('--heatmap_weight', type=float, default=1.0, help='Weight for heatmap loss')
    parser.add_argument('--coord_weight', type=float, default=0.1, help='Weight for coordinate loss')
    
    # LR Range Test parameters
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for the test')
    parser.add_argument('--start_lr', type=float, default=1e-7, help='Starting learning rate for range test')
    parser.add_argument('--end_lr', type=float, default=1.0, help='End learning rate for range test')
    parser.add_argument('--num_iterations', type=int, default=100, help='Number of iterations for range test')
    parser.add_argument('--smooth_window', type=float, default=0.05, help='Window size for smoothing (fraction of iterations)')
    parser.add_argument('--diverge_threshold', type=float, default=5.0, help='Threshold for considering the loss diverged')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay to use for the optimizer')
    
    # Optimizer arguments
    parser.add_argument('--optimizer', type=str, choices=['adam', 'adamw', 'sgd'], default='adam',
                        help='Optimizer type to use for the test')
    parser.add_argument('--momentum', type=float, default=0.9, 
                        help='Momentum factor for SGD optimizer')
    parser.add_argument('--nesterov', action='store_true',
                        help='Enable Nesterov momentum for SGD optimizer')
    
    # Device arguments
    parser.add_argument('--use_mps', action='store_true', help='Use Metal Performance Shaders (MPS) for Mac GPU acceleration')
    parser.add_argument('--force_cpu', action='store_true', help='Force CPU usage even if GPU is available')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of worker threads for dataloader (0 for single process)')
    
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


def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    try:
        data_processor = DataProcessor(
            data_path=args.data_path,
            landmark_cols=None,  # Auto-detect landmark columns
            image_size=(224, 224),
            apply_clahe=args.apply_clahe
        )
        
        # Load and preprocess data
        df = data_processor.preprocess_data()
        landmark_cols = data_processor.landmark_cols
        
        print(f"Dataset loaded with {len(df)} samples")
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return
    
    # Define data transforms
    base_transforms = transforms.Compose([
        ToTensor(),
        Normalize()
    ])
    
    # Get training transformations with augmentations
    train_augmentations = get_train_transforms(include_horizontal_flip=False)
    
    # Create custom transform for training
    train_transform = TrainTransform(train_augmentations, base_transforms)
    
    # Use only training data for the LR Range Test
    train_df = df[df['set'] == 'train'] if 'set' in df.columns else df.sample(frac=0.7)
    
    # Create dataset
    train_dataset = CephalometricDataset(
        train_df, root_dir=None, transform=train_transform,
        landmark_cols=landmark_cols, train=True, apply_clahe=args.apply_clahe
    )
    
    # Create dataloader
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    
    print(f"Created dataloader with {len(train_dataset)} training samples")
    
    # Determine device
    if args.force_cpu:
        device = torch.device('cpu')
        print("Using CPU as requested")
    elif args.use_mps and platform.system() == 'Darwin' and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS (Metal Performance Shaders) for Mac GPU acceleration")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Create model
    model = create_hrnet_model(
        num_landmarks=args.num_landmarks,
        pretrained=args.pretrained,
        use_refinement=args.use_refinement,
        hrnet_type=args.hrnet_type
    )
    model.to(device)
    
    # Create loss function
    criterion = CombinedLoss(
        heatmap_weight=args.heatmap_weight,
        coord_weight=args.coord_weight,
        output_size=(64, 64),   # Heatmap size
        image_size=(224, 224)   # Original image size
    )
    
    # Create optimizer
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.start_lr,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.start_lr,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.start_lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov
        )
    
    print(f"\nRunning Learning Rate Range Test with:")
    print(f"  Model: HRNet-{args.hrnet_type.upper()} with{'' if args.use_refinement else 'out'} refinement MLP")
    print(f"  Optimizer: {args.optimizer.upper()}")
    print(f"  Learning rate range: {args.start_lr} to {args.end_lr}")
    print(f"  Number of iterations: {args.num_iterations}")
    print(f"  Batch size: {args.batch_size}")
    
    # Run the LR Range Test
    results = lr_range_test(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        start_lr=args.start_lr,
        end_lr=args.end_lr,
        num_iterations=args.num_iterations,
        smooth_window=args.smooth_window,
        diverge_threshold=args.diverge_threshold,
        output_dir=args.output_dir
    )
    
    # Print results
    print(f"\nLR Range Test completed. Results:")
    print(f"  - Steepest gradient LR: {results['steepest_slope_lr']:.6f}")
    print(f"  - Min loss LR: {results['min_loss_lr']:.6f}")
    print(f"  - Suggested max_lr (for OneCycleLR): {results['suggested_max_lr']:.6f}")
    
    # Save results to a text file
    results_file = os.path.join(args.output_dir, 'lr_range_test_results.txt')
    with open(results_file, 'w') as f:
        f.write(f"LR Range Test Results\n")
        f.write(f"=====================\n\n")
        f.write(f"Test Parameters:\n")
        f.write(f"  - Model: HRNet-{args.hrnet_type.upper()} with{'' if args.use_refinement else 'out'} refinement MLP\n")
        f.write(f"  - Optimizer: {args.optimizer.upper()}\n")
        f.write(f"  - Learning rate range: {args.start_lr} to {args.end_lr}\n")
        f.write(f"  - Number of iterations: {args.num_iterations}\n")
        f.write(f"  - Batch size: {args.batch_size}\n\n")
        f.write(f"Results:\n")
        f.write(f"  - Steepest gradient LR: {results['steepest_slope_lr']:.6f}\n")
        f.write(f"  - Min loss LR: {results['min_loss_lr']:.6f}\n")
        f.write(f"  - Suggested max_lr (for OneCycleLR): {results['suggested_max_lr']:.6f}\n\n")
        f.write(f"Recommendation:\n")
        f.write(f"  Use max_lr={results['suggested_max_lr']:.6f} for OneCycleLR scheduler.\n")
        f.write(f"  This is a conservative estimate (min_loss_lr / 2).\n")
        f.write(f"  You may try values between {results['suggested_max_lr']:.6f} and {results['min_loss_lr']:.6f}.\n")
    
    print(f"Results saved to {results_file}")
    print(f"Plot saved to {os.path.join(args.output_dir, 'lr_range_test.png')}")


if __name__ == "__main__":
    main() 