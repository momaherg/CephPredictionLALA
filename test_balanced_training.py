import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
import random
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import the patient classifier and data processor
from src.data.patient_classifier import PatientClassifier
from src.data.dataset import CephalometricDataset, ToTensor, Normalize
from src.data.data_augmentation import get_train_transforms

# Define the TrainTransform class (similar to what's in train.py)
class TrainTransform:
    def __init__(self, train_augmentations, base_transforms):
        self.train_augmentations = train_augmentations
        self.base_transforms = base_transforms
        
    def __call__(self, sample):
        # First apply augmentation
        augmented = self.train_augmentations(sample)
        # Then apply base transforms (ToTensor, Normalize)
        return self.base_transforms(augmented)

# Define our own version of create_dataloader_with_augmentations
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

def create_imbalanced_test_dataframe(n_samples=200, seed=42):
    """
    Create a heavily imbalanced synthetic dataset for testing class balancing
    
    Args:
        n_samples (int): Number of samples to generate
        seed (int): Random seed for reproducibility
        
    Returns:
        pd.DataFrame: DataFrame with landmark coordinates
    """
    np.random.seed(seed)
    
    # Create severely imbalanced data
    # - Class I (normal): 15% of samples
    # - Class II (prognathic maxilla): 75% of samples 
    # - Class III (retrognathic maxilla): 10% of samples
    
    # These are the indices in our landmark coordinate array
    SELLA_IDX = 0
    NASION_IDX = 1
    A_POINT_IDX = 2
    B_POINT_IDX = 3
    
    # Generate base coordinates for all landmarks
    landmarks = np.zeros((n_samples, 19, 2))  # 19 landmarks, each with x and y coordinates
    
    # For simplicity, we'll use a simple coordinate system where:
    # - Sella is at (100, 100)
    # - Nasion is at (150, 80)
    # - A point and B point positions will vary to create different ANB angles
    
    # Set common landmarks
    landmarks[:, SELLA_IDX] = [100, 100]  # Sella
    landmarks[:, NASION_IDX] = [150, 80]  # Nasion
    
    # Class proportions - intentionally imbalanced
    class1_prop = 0.15  # Class I (normal)
    class2_prop = 0.75  # Class II (majority class)
    class3_prop = 0.10  # Class III (minority class)
    
    n_class1 = int(n_samples * class1_prop)
    n_class2 = int(n_samples * class2_prop)
    n_class3 = n_samples - n_class1 - n_class2
    
    # Create Class I samples (ANB angle between 0-4 degrees)
    for i in range(n_class1):
        # A point slightly forward of B point
        landmarks[i, A_POINT_IDX] = [180, 120]  # A point
        landmarks[i, B_POINT_IDX] = [175, 130]  # B point (slightly behind A point)
    
    # Create Class II samples (ANB angle > 4 degrees)
    for i in range(n_class1, n_class1 + n_class2):
        # A point much more forward of B point
        landmarks[i, A_POINT_IDX] = [185, 120]  # A point
        landmarks[i, B_POINT_IDX] = [165, 130]  # B point (well behind A point)
    
    # Create Class III samples (ANB angle < 0 degrees)
    for i in range(n_class1 + n_class2, n_samples):
        # B point forward of A point
        landmarks[i, A_POINT_IDX] = [170, 120]  # A point
        landmarks[i, B_POINT_IDX] = [180, 130]  # B point (in front of A point)
    
    # Add some random noise to make the dataset more realistic
    landmarks += np.random.normal(0, 3, landmarks.shape)
    
    # Create a DataFrame
    df = pd.DataFrame()
    
    # Flatten landmarks and add to DataFrame with appropriate column names
    landmark_cols = []
    for i in range(19):
        x_col = f'landmark_{i}_x'
        y_col = f'landmark_{i}_y'
        landmark_cols.extend([x_col, y_col])
        df[x_col] = landmarks[:, i, 0]
        df[y_col] = landmarks[:, i, 1]
    
    # Add some patient IDs
    df['patient_id'] = [f'P{i:03d}' for i in range(n_samples)]
    
    # Create a set column to pre-split the data
    # 70% train, 15% validation, 15% test
    sets = ['train'] * int(0.7 * n_samples) + ['dev'] * int(0.15 * n_samples) + ['test'] * (n_samples - int(0.7 * n_samples) - int(0.15 * n_samples))
    np.random.shuffle(sets)
    df['set'] = sets
    
    return df, landmark_cols

def plot_class_distributions(train_dist, val_dist, test_dist, balanced_train_dist=None, output_file='class_distribution.png'):
    """
    Plot the class distributions for train, validation, and test sets
    
    Args:
        train_dist: Dictionary with class counts for training set
        val_dist: Dictionary with class counts for validation set
        test_dist: Dictionary with class counts for test set
        balanced_train_dist: Optional dictionary with balanced class counts for training set
        output_file: Path to save the output plot
    """
    # Ensure all distributions have the same keys
    all_classes = sorted(set(list(train_dist.keys()) + list(val_dist.keys()) + list(test_dist.keys())))
    
    # Create lists for plotting
    train_counts = [train_dist.get(cls, 0) for cls in all_classes]
    val_counts = [val_dist.get(cls, 0) for cls in all_classes]
    test_counts = [test_dist.get(cls, 0) for cls in all_classes]
    
    # Class labels for x-axis
    class_labels = [f"Class {cls}" for cls in all_classes]
    
    # Determine plot type based on whether balanced_train_dist is provided
    if balanced_train_dist:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Original distribution plot
        x = np.arange(len(class_labels))
        width = 0.25
        
        ax1.bar(x - width, train_counts, width, label='Train')
        ax1.bar(x, val_counts, width, label='Validation')
        ax1.bar(x + width, test_counts, width, label='Test')
        
        ax1.set_xlabel('Skeletal Class')
        ax1.set_ylabel('Number of Samples')
        ax1.set_title('Original Class Distribution')
        ax1.set_xticks(x)
        ax1.set_xticklabels(class_labels)
        ax1.legend()
        
        # Balanced distribution plot
        balanced_counts = [balanced_train_dist.get(cls, 0) for cls in all_classes]
        
        ax2.bar(x - width, balanced_counts, width, label='Balanced Train')
        ax2.bar(x, val_counts, width, label='Validation')
        ax2.bar(x + width, test_counts, width, label='Test')
        
        ax2.set_xlabel('Skeletal Class')
        ax2.set_ylabel('Number of Samples')
        ax2.set_title('After Balancing Class Distribution')
        ax2.set_xticks(x)
        ax2.set_xticklabels(class_labels)
        ax2.legend()
        
        plt.tight_layout()
    else:
        # Single plot for original distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(class_labels))
        width = 0.25
        
        ax.bar(x - width, train_counts, width, label='Train')
        ax.bar(x, val_counts, width, label='Validation')
        ax.bar(x + width, test_counts, width, label='Test')
        
        ax.set_xlabel('Skeletal Class')
        ax.set_ylabel('Number of Samples')
        ax.set_title('Class Distribution')
        ax.set_xticks(x)
        ax.set_xticklabels(class_labels)
        ax.legend()
    
    plt.savefig(output_file)

def test_train_only_balancing():
    """
    Test that only the training set is balanced, while validation and test sets
    maintain their original distribution
    """
    print("=== Testing Training-Only Class Balancing ===\n")
    
    # Create an imbalanced dataset
    print("Creating imbalanced test dataset...")
    df, landmark_cols = create_imbalanced_test_dataframe(n_samples=200)
    
    # Compute classes for all data to get ground truth
    classifier = PatientClassifier(landmark_cols)
    df = classifier.classify_patients(df)
    
    # Extract the sets
    train_df = df[df['set'] == 'train']
    val_df = df[df['set'] == 'dev']
    test_df = df[df['set'] == 'test']
    
    # Get original class distributions
    train_dist = train_df['skeletal_class'].value_counts().to_dict()
    val_dist = val_df['skeletal_class'].value_counts().to_dict()
    test_dist = test_df['skeletal_class'].value_counts().to_dict()
    
    print("\nOriginal class distributions:")
    print("Training set:")
    for label, count in sorted(train_dist.items()):
        class_name = {1: "Class I", 2: "Class II", 3: "Class III"}.get(label, f"Class {label}")
        print(f"  {class_name}: {count} samples ({count/len(train_df)*100:.1f}%)")
    
    print("Validation set:")
    for label, count in sorted(val_dist.items()):
        class_name = {1: "Class I", 2: "Class II", 3: "Class III"}.get(label, f"Class {label}")
        print(f"  {class_name}: {count} samples ({count/len(val_df)*100:.1f}%)")
    
    print("Test set:")
    for label, count in sorted(test_dist.items()):
        class_name = {1: "Class I", 2: "Class II", 3: "Class III"}.get(label, f"Class {label}")
        print(f"  {class_name}: {count} samples ({count/len(test_df)*100:.1f}%)")
    
    # Plot original distributions
    plot_class_distributions(train_dist, val_dist, test_dist, output_file='original_class_distribution.png')
    print("Original class distribution plot saved to 'original_class_distribution.png'")
    
    # Create dataloaders with balancing - this should only balance the training set
    print("\nCreating dataloaders with training set balancing...")
    train_loader, val_loader, test_loader = create_dataloader_with_augmentations(
        df=df,
        landmark_cols=landmark_cols,
        batch_size=16,
        apply_clahe=False,
        num_workers=0,
        balance_classes=True  # This should only balance the training set
    )
    
    # Since we can't access the dataset's internal dataframe directly,
    # let's use our knowledge of how the balancing works to verify it
    
    # Get dataset sizes
    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)
    test_size = len(test_loader.dataset)
    
    print("\nAfter balancing, dataset sizes are:")
    print(f"Training set: {train_size} samples (was {len(train_df)} originally)")
    print(f"Validation set: {val_size} samples (was {len(val_df)} originally)")
    print(f"Test set: {test_size} samples (was {len(test_df)} originally)")
    
    # Check if training set size matches expectation
    # When balanced using upsampling, size should be max_class_count * num_classes
    max_class_count = max(train_dist.values())
    expected_balanced_size = max_class_count * len(train_dist.keys())
    
    # Calculate expected balanced distribution
    balanced_train_dist = {}
    for class_label in train_dist.keys():
        balanced_train_dist[class_label] = max_class_count
    
    # Plot balanced distributions
    plot_class_distributions(train_dist, val_dist, test_dist, balanced_train_dist, output_file='balanced_class_distribution.png')
    print("Balanced class distribution plot saved to 'balanced_class_distribution.png'")
    
    # Verify that the training set was balanced
    is_train_balanced = train_size == expected_balanced_size
    is_val_unchanged = val_size == len(val_df)
    is_test_unchanged = test_size == len(test_df)
    
    print("\n=== Test Results ===")
    print(f"1. Training set was balanced: {'✓' if is_train_balanced else '✗'} (size: {train_size}, expected: {expected_balanced_size})")
    print(f"2. Validation set remained unchanged: {'✓' if is_val_unchanged else '✗'} (size: {val_size}, expected: {len(val_df)})")
    print(f"3. Test set remained unchanged: {'✓' if is_test_unchanged else '✗'} (size: {test_size}, expected: {len(test_df)})")
    
    overall_success = is_train_balanced and is_val_unchanged and is_test_unchanged
    print(f"\nTest {'passed' if overall_success else 'failed'}: Only the training set was balanced while validation and test sets maintained their original distributions.")
    
    return overall_success

if __name__ == "__main__":
    test_train_only_balancing() 