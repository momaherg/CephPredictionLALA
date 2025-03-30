import os
import argparse
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
import platform

from data.data_processor import DataProcessor
from data.dataset import CephalometricDataset, ToTensor, Normalize
from data.data_augmentation import get_train_transforms
from models.trainer import LandmarkTrainer

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

def create_dataloader_with_augmentations(df, landmark_cols, batch_size=16, 
                                        train_ratio=0.8, val_ratio=0.1, 
                                        apply_clahe=True, root_dir=None, 
                                        num_workers=4):
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
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # If 'set' column is already present, use it for splitting
    if 'set' in df.columns:
        train_df = df[df['set'] == 'train']
        val_df = df[df['set'] == 'dev']
        test_df = df[df['set'] == 'test']
    else:
        # Randomly split the data
        n = len(df)
        indices = np.random.permutation(n)
        
        train_size = int(train_ratio * n)
        val_size = int(val_ratio * n)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        train_df = df.iloc[train_indices]
        val_df = df.iloc[val_indices]
        test_df = df.iloc[test_indices]
    
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
    
    # Initialize data processor
    data_processor = DataProcessor(
        data_path=args.data_path,
        landmark_cols=landmark_cols,
        image_size=(224, 224),
        apply_clahe=args.apply_clahe
    )
    
    # Load and preprocess data
    df = data_processor.preprocess_data()
    
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
    
    # Check MPS availability for Mac users
    if args.use_mps and platform.system() == 'Darwin':
        if torch.backends.mps.is_available():
            print("MPS (Metal Performance Shaders) is available for Mac GPU acceleration.")
        else:
            print("WARNING: MPS requested but not available. Falling back to CPU.")
            args.use_mps = False
    
    # Create data loaders with augmentation (only for training set)
    print("Creating dataloaders with augmentation for training set...")
    train_loader, val_loader, test_loader = create_dataloader_with_augmentations(
        df=df,
        landmark_cols=landmark_cols,
        batch_size=args.batch_size,
        train_ratio=0.8,
        val_ratio=0.1,
        apply_clahe=args.apply_clahe,
        root_dir=None,
        num_workers=args.num_workers
    )
    
    print(f"Created data loaders:")
    print(f"  Training samples: {len(train_loader.dataset)}")
    print(f"  Validation samples: {len(val_loader.dataset)}")
    print(f"  Test samples: {len(test_loader.dataset)}")
    
    # Determine device
    device = None
    if args.force_cpu:
        device = torch.device('cpu')
        print("Using CPU as requested with --force_cpu")
    
    # Log training configuration
    print("\nTraining Configuration:")
    print(f"  Model: HRNet-W32 with{'' if args.use_refinement else 'out'} refinement MLP")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Number of epochs: {args.num_epochs}")
    if args.use_refinement:
        print(f"  Heatmap loss weight: {args.heatmap_weight}")
        print(f"  Coordinate loss weight: {args.coord_weight}")
    print(f"  Device: {'MPS (Mac GPU)' if args.use_mps else 'CUDA' if torch.cuda.is_available() and not args.force_cpu else 'CPU'}")
    print(f"  Output directory: {args.output_dir}")
    
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
        use_mps=args.use_mps
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
    results = trainer.evaluate(test_loader, save_visualizations=True)
    
    # Print evaluation results
    print("\nEvaluation Results:")
    print(f"Mean Euclidean Distance: {results['mean_euclidean_distance']:.2f} pixels")
    print(f"Success Rate (2mm): {results['success_rate_2mm'] * 100:.2f}%")
    print(f"Success Rate (4mm): {results['success_rate_4mm'] * 100:.2f}%")
    
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main() 