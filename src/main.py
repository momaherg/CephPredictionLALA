import os
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from data.data_processor import DataProcessor
from data.data_augmentation import (
    get_train_transforms, get_val_transforms, 
    apply_augmentation, RandomRotation, RandomScaling, 
    RandomShift, RandomBrightness, RandomContrast,
    RandomGaussianNoise, RandomBlur
)
from data.dataset import ToTensor, Normalize

def parse_args():
    parser = argparse.ArgumentParser(description='Cephalometric Landmark Detection Pipeline')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset file')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Directory to save outputs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--apply_clahe', action='store_true', help='Apply CLAHE for histogram equalization')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--visualize_samples', action='store_true', help='Visualize sample data')
    parser.add_argument('--visualize_augmentations', action='store_true', help='Visualize augmentation examples')
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

def visualize_sample(sample, output_path=None):
    """Visualize a sample with landmarks"""
    image = sample['image']
    landmarks = sample['landmarks']
    
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    
    # Plot landmarks
    if landmarks.size > 0:
        plt.scatter(landmarks[:, 0], landmarks[:, 1], s=30, marker='o', c='r')
        
        # Connect specific landmarks if needed
        # For example, connect points that form the jaw line
        # plt.plot(landmarks[[0, 1, 2, 3, 4], 0], landmarks[[0, 1, 2, 3, 4], 1], 'g-', linewidth=2)
    
    plt.title('Image with Landmarks')
    plt.axis('off')
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

def visualize_augmentations(sample, augmentations=None, n_examples=5, output_dir=None):
    """
    Visualize different augmentations of a sample
    
    Args:
        sample (dict): Sample containing 'image' and 'landmarks'
        augmentations (list): List of augmentation transforms to visualize individually
        n_examples (int): Number of examples to generate per augmentation type
        output_dir (str): Directory to save visualizations
    """
    # Get the original sample
    orig_image = sample['image'].copy()
    orig_landmarks = sample['landmarks'].copy()
    
    # If no specific augmentations provided, use all available ones
    if augmentations is None:
        augmentations = [
            ('Original', None),
            ('Rotation', RandomRotation(max_angle=10.0, p=1.0)),
            ('Scaling', RandomScaling(scale_factor=(0.9, 1.1), p=1.0)),
            ('Shift', RandomShift(max_pixels=10, p=1.0)),
            ('Brightness', RandomBrightness(brightness_factor=(0.8, 1.2), p=1.0)),
            ('Contrast', RandomContrast(contrast_factor=(0.8, 1.2), p=1.0)),
            ('Noise', RandomGaussianNoise(std_range=(0.001, 0.01), p=1.0)),
            ('Blur', RandomBlur(blur_limit=(0, 2), p=1.0)),
            ('Combined', get_train_transforms(include_horizontal_flip=False))
        ]
    
    # Create a directory for each augmentation type
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Visualize each type of augmentation
    for aug_name, transform in augmentations:
        # Create a figure for this augmentation type
        plt.figure(figsize=(15, 3 * (n_examples + 1)))
        
        # Plot the original sample
        plt.subplot(n_examples + 1, 1, 1)
        plt.imshow(orig_image)
        plt.scatter(orig_landmarks[:, 0], orig_landmarks[:, 1], s=30, marker='o', c='r')
        plt.title(f'Original Image')
        plt.axis('off')
        
        # Generate multiple examples of this augmentation
        for i in range(n_examples):
            if transform is None:
                # Just duplicate the original for the "Original" category
                augmented = {'image': orig_image.copy(), 'landmarks': orig_landmarks.copy()}
            else:
                # Apply the specific augmentation
                augmented = transform({'image': orig_image.copy(), 'landmarks': orig_landmarks.copy()})
            
            # Plot augmented sample
            plt.subplot(n_examples + 1, 1, i + 2)
            plt.imshow(augmented['image'])
            plt.scatter(augmented['landmarks'][:, 0], augmented['landmarks'][:, 1], s=30, marker='o', c='r')
            plt.title(f'{aug_name} Example {i+1}')
            plt.axis('off')
        
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, f'augmentation_{aug_name.lower()}.png'))
            plt.close()
        else:
            plt.show()

def create_dataloader_with_augmentations(df, landmark_cols, batch_size=32, 
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
    from data.dataset import CephalometricDataset, ToTensor, Normalize
    from torch.utils.data import DataLoader
    
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
    class TrainTransform:
        def __call__(self, sample):
            # First apply augmentation
            augmented = train_augmentations(sample)
            # Then apply base transforms (ToTensor, Normalize)
            return base_transforms(augmented)
    
    # Create datasets
    train_dataset = CephalometricDataset(
        train_df, root_dir=root_dir, transform=TrainTransform(), 
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
    
    # Define landmark columns
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
        num_workers=4
    )
    
    print(f"Created data loaders:")
    print(f"  Training samples: {len(train_loader.dataset)}")
    print(f"  Validation samples: {len(val_loader.dataset)}")
    print(f"  Test samples: {len(test_loader.dataset)}")
    
    # Visualize sample data if requested
    if args.visualize_samples:
        # Get a sample from the training set
        sample_idx = np.random.randint(0, len(train_loader.dataset))
        sample = train_loader.dataset[sample_idx]
        
        # Convert tensor to numpy for visualization
        if isinstance(sample['image'], torch.Tensor):
            # If normalized, denormalize
            image = sample['image'].permute(1, 2, 0).numpy()
            
            # Denormalize if needed
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = std * image + mean
            
            # Clip values to [0, 1]
            image = np.clip(image, 0, 1)
            
            # Convert to uint8
            image = (image * 255).astype(np.uint8)
        else:
            image = sample['image']
        
        # Convert landmarks tensor to numpy if needed
        if isinstance(sample['landmarks'], torch.Tensor):
            landmarks = sample['landmarks'].numpy()
        else:
            landmarks = sample['landmarks']
        
        # Create a sample for visualization
        vis_sample = {
            'image': image,
            'landmarks': landmarks
        }
        
        # Visualize the sample
        visualize_sample(
            vis_sample, 
            output_path=os.path.join(args.output_dir, 'sample.png')
        )
    
    # Visualize augmentations if requested
    if args.visualize_augmentations:
        # Get a sample from the dataset (preferably before augmentation)
        sample_idx = np.random.randint(0, len(df))
        raw_sample = df.iloc[sample_idx]
        
        # Get image data
        if 'Image' in df.columns:
            img_data = raw_sample['Image']
            if isinstance(img_data, list):
                img_array = np.array(img_data).reshape(224, 224, 3)
            else:
                img_array = img_data
                
            # Ensure image is uint8
            if img_array.dtype != np.uint8:
                if img_array.max() <= 1.0:
                    img_array = (img_array * 255).astype(np.uint8)
                else:
                    img_array = img_array.astype(np.uint8)
        else:
            # If images are stored as files - adapt this part if needed
            img_name = os.path.join(root_dir, raw_sample['patient'] + '.jpg')
            img_array = np.array(Image.open(img_name))
        
        # Get landmarks
        if all(col in df.columns for col in landmark_cols):
            landmarks = raw_sample[landmark_cols].values.astype('float32')
            landmarks = landmarks.reshape(-1, 2)
        else:
            landmarks = np.array([])
        
        vis_sample = {
            'image': img_array,
            'landmarks': landmarks
        }
        
        # Visualize different augmentations
        visualize_augmentations(
            vis_sample,
            augmentations=None,  # Use all available augmentations
            n_examples=3,
            output_dir=os.path.join(args.output_dir, 'augmentations')
        )
    
    print("Data preparation and preprocessing complete.")

if __name__ == "__main__":
    main() 