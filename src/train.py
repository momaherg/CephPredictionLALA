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
    
    # Optimizer arguments
    parser.add_argument('--optimizer', type=str, choices=['adam', 'adamw', 'sgd'], default='adam',
                        help='Optimizer type to use for training')
    parser.add_argument('--momentum', type=float, default=0.9, 
                        help='Momentum factor for SGD optimizer')
    parser.add_argument('--nesterov', action='store_true',
                        help='Enable Nesterov momentum for SGD optimizer')
    
    # Loss Normalization arguments
    parser.add_argument('--no_loss_norm', action='store_true',
                        help='Disable running average loss normalization in CombinedLoss')
    parser.add_argument('--loss_norm_decay', type=float, default=0.99,
                        help='Decay factor for loss normalization running average')
    parser.add_argument('--loss_norm_epsilon', type=float, default=1e-6,
                        help='Epsilon for loss normalization stability')
    
    # Per-Landmark Weighting/Focusing
    parser.add_argument('--target_indices', type=int, nargs='*', default=None,
                        help='List of landmark indices (0-based) to focus loss calculation on. If None, uses all landmarks.')
    parser.add_argument('--landmark_weights', type=float, nargs='*', default=None,
                        help='List of weights (one per landmark) to apply to the loss calculation. Must match num_landmarks.')
    
    # Specific MED Logging
    parser.add_argument('--log_specific_med', type=int, nargs='*', default=None,
                        help='List of landmark indices (0-based) to log MED for separately during training.')
    
    # Depth Feature Arguments
    parser.add_argument('--use_depth', action='store_true',
                        help='Use depth features for training')
    parser.add_argument('--depth_fusion_method', type=str, choices=['concat', 'add', 'attention'], default='concat',
                        help='Method to fuse depth features with RGB features')
    parser.add_argument('--depth_cache_dir', type=str, default=None,
                        help='Directory to cache depth features (if not provided, features will be extracted on-the-fly)')
    
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
            from data.patient_classifier import PatientClassifier
            classifier = PatientClassifier(landmark_cols)
            train_df = classifier.classify_patients(train_df)
        
        # Now balance the training data
        print("Balancing training data using skeletal classification...")
        from data.patient_classifier import PatientClassifier
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
       'ST Nasion_y', 'ST A point_x', 'ST A point_y', 'ST B point_x',
       'ST B point_y', 'ST Upper lip_x', 'ST Upper lip_y', 'ST Lower lip_x',
       'ST Lower lip_y', 'ST Menton_x', 'ST Menton_y', 'ST Pogonion_x',
       'ST Pogonion_y']
    
    # Fix inconsistent column names
    landmark_cols = [col.replace('Gonion _', 'Gonion_') for col in landmark_cols]
    
    # Load data
    print(f"Loading data from {args.data_path}")
    try:
        if args.data_path.endswith('.csv'):
            df = pd.read_csv(args.data_path)
        elif args.data_path.endswith('.parquet'):
            df = pd.read_parquet(args.data_path)
        elif args.data_path.endswith('.h5') or args.data_path.endswith('.hdf5'):
            df = pd.read_hdf(args.data_path)
        else:
            raise ValueError(f"Unsupported file format: {args.data_path}")
        
        print(f"Loaded data: {len(df)} samples")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Determine device
    if args.force_cpu:
        device = torch.device("cpu")
        print("Forcing CPU usage")
    elif args.use_mps and torch.backends.mps.is_available() and platform.system() == 'Darwin':
        device = torch.device("mps")
        print("Using MPS device (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    
    # Process data
    processor = DataProcessor(df, landmark_cols=landmark_cols)
    df = processor.process_data()
    
    # Extract and cache depth features if enabled
    depth_features = None
    if args.use_depth:
        from utils.depth_features import DepthFeatureExtractor
        
        print("Initializing depth feature extraction...")
        cache_dir = args.depth_cache_dir or os.path.join(args.output_dir, 'depth_cache')
        
        depth_extractor = DepthFeatureExtractor(cache_dir=cache_dir, device=device)
        
        # Create temporary dataset to extract features
        from data.dataset import CephalometricDataset
        temp_transform = transforms.Compose([ToTensor()])
        
        # Split the dataframe if not already split
        if 'set' not in df.columns:
            print("Splitting data into train/val/test sets")
            np.random.seed(args.seed)
            n = len(df)
            indices = np.random.permutation(n)
            
            train_ratio = 0.8
            val_ratio = 0.1
            
            train_size = int(train_ratio * n)
            val_size = int(val_ratio * n)
            
            train_indices = indices[:train_size]
            val_indices = indices[train_size:train_size + val_size]
            test_indices = indices[train_size + val_size:]
            
            df.loc[:, 'set'] = 'test'  # Default to test
            df.loc[train_indices, 'set'] = 'train'
            df.loc[val_indices, 'set'] = 'dev'
        
        # Create temporary datasets for feature extraction
        train_df = df[df['set'] == 'train'].copy()
        val_df = df[df['set'] == 'dev'].copy()
        test_df = df[df['set'] == 'test'].copy()
        
        temp_train_dataset = CephalometricDataset(
            train_df, transform=temp_transform, 
            landmark_cols=landmark_cols, apply_clahe=args.apply_clahe
        )
        
        temp_val_dataset = CephalometricDataset(
            val_df, transform=temp_transform, 
            landmark_cols=landmark_cols, apply_clahe=args.apply_clahe
        )
        
        temp_test_dataset = CephalometricDataset(
            test_df, transform=temp_transform, 
            landmark_cols=landmark_cols, apply_clahe=args.apply_clahe
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
        os.makedirs(os.path.join(args.output_dir, 'depth_samples'), exist_ok=True)
        
        # Get a few samples from the training set
        for i in range(min(5, len(temp_train_dataset))):
            sample = temp_train_dataset[i]
            image = sample['image']
            # Convert from tensor to numpy for visualization
            if isinstance(image, torch.Tensor):
                image = image.permute(1, 2, 0).numpy()
                
            save_path = os.path.join(args.output_dir, 'depth_samples', f'sample_{i}.png')
            depth_extractor.save_depth_visualization(image, save_path)
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        df, landmark_cols, batch_size=args.batch_size, 
        apply_clahe=args.apply_clahe, num_workers=args.num_workers,
        balance_classes=args.balance_classes, depth_features=depth_features,
        use_depth=args.use_depth
    )
    print(f"Created dataloaders: train={len(train_loader.dataset)} samples, val={len(val_loader.dataset)} samples, test={len(test_loader.dataset)} samples")
    
    # Calculate total steps for OneCycleLR if needed
    total_steps = None
    if args.scheduler == 'onecycle':
        steps_per_epoch = len(train_loader)
        total_steps = steps_per_epoch * args.num_epochs
    
    # Set scheduler params - default to num_epochs/2 for T_max if not specified
    if args.scheduler == 'cosine' and args.lr_t_max == 10:
        lr_t_max = args.num_epochs // 2
    else:
        lr_t_max = args.lr_t_max
    
    # Create trainer
    print("Creating landmark trainer...")
    trainer = LandmarkTrainer(
        num_landmarks=args.num_landmarks // 2,  # Each landmark has x,y coordinates
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=device,
        output_dir=args.output_dir,
        use_refinement=args.use_refinement,
        heatmap_weight=args.heatmap_weight,
        coord_weight=args.coord_weight,
        use_mps=args.use_mps,
        hrnet_type='w32',
        scheduler_type=args.scheduler if args.scheduler != 'none' else None,
        lr_patience=args.lr_patience,
        lr_factor=args.lr_factor,
        lr_min=args.lr_min,
        lr_t_max=lr_t_max,
        max_lr=args.max_lr,
        pct_start=args.pct_start,
        div_factor=args.div_factor,
        final_div_factor=args.final_div_factor,
        optimizer_type=args.optimizer,
        momentum=args.momentum,
        nesterov=args.nesterov,
        use_loss_normalization=not args.no_loss_norm,
        norm_decay=args.loss_norm_decay,
        norm_epsilon=args.loss_norm_epsilon,
        total_steps=total_steps,
        target_landmark_indices=args.target_indices,
        landmark_weights=args.landmark_weights,
        log_specific_landmark_indices=args.log_specific_med,
        use_depth=args.use_depth,
        depth_fusion_method=args.depth_fusion_method
    )
    
    # Train the model
    print("Starting training...")
    trainer.train(train_loader, val_loader, num_epochs=args.num_epochs, save_freq=args.save_freq)
    
    # Evaluate the final model
    print("\nEvaluating the final model on test set...")
    trainer.evaluate(test_loader, save_visualizations=True)
    
    print("Training and evaluation completed successfully!")

if __name__ == "__main__":
    main() 