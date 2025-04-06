import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image
from torchvision import transforms
import io
from skimage import color

class CephalometricDataset(Dataset):
    """Cephalometric X-ray dataset with landmark coordinates"""

    def __init__(self, dataframe, root_dir=None, transform=None, landmark_cols=None, 
                 train=True, apply_clahe=False, clahe_clip_limit=2.0, 
                 clahe_grid_size=(8, 8), use_depth=False):
        """
        Args:
            dataframe (pandas.DataFrame): DataFrame with annotations and image data.
            root_dir (string, optional): Directory with all the images (if image paths are provided).
            transform (callable, optional): Optional transform to be applied on a sample.
            landmark_cols (list, optional): List of column names for landmark coordinates.
            train (bool): Whether this is the training set (for applying train-specific transforms).
            apply_clahe (bool): Apply CLAHE for histogram equalization.
            clahe_clip_limit (float): Clip limit for CLAHE.
            clahe_grid_size (tuple): Grid size for CLAHE.
            use_depth (bool): Whether to load and include the depth map.
        """
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform
        self.landmark_cols = landmark_cols
        self.train = train
        self.apply_clahe = apply_clahe
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_grid_size = clahe_grid_size
        self.use_depth = use_depth
        
        # Check if depth column exists if use_depth is True
        if self.use_depth and 'depth_map' not in self.dataframe.columns:
            raise ValueError("use_depth is True, but 'depth_map' column is missing from the DataFrame.")

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        '''Return a single sample from the dataset'''
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get row from dataframe
        row = self.dataframe.iloc[idx]
        img_data = row['Image']
        
        # Process image data
        if isinstance(img_data, list):
            # Check if it's RGB or grayscale based on list length
            list_len = len(img_data)
            expected_gray_len = 224 * 224  # Assuming fixed 224x224 size
            expected_rgb_len = expected_gray_len * 3
            
            if list_len == expected_gray_len:
                # Grayscale image
                img = np.array(img_data).reshape((224, 224)).astype(np.float32)
            elif list_len == expected_rgb_len:
                # RGB image
                img = np.array(img_data).reshape((224, 224, 3)).astype(np.float32)
            else:
                raise ValueError(f"Cannot reshape list of length {list_len}, expected {expected_gray_len} or {expected_rgb_len}")
        elif isinstance(img_data, np.ndarray):
            img = img_data.astype(np.float32)
        else:
            raise TypeError(f"Unexpected image data type: {type(img_data)}")
            
        # Convert to grayscale ONLY if not using depth features - keep RGB if using depth
        if not self.use_depth:
            # Ensure image is grayscale (single channel)
            if img.ndim == 3:
                if img.shape[-1] == 3: # HWC -> HW
                    img = color.rgb2gray(img)
                elif img.shape[0] == 3: # CHW -> HW
                    img = img[0, :, :]
                elif img.shape[-1] == 1: # HW1 -> HW
                    img = img.squeeze(-1)
                else:
                    raise ValueError(f"Unexpected image shape: {img.shape}")
            elif img.ndim != 2:
                raise ValueError(f"Unexpected image shape: {img.shape}")
        else:
            # When using depth, ensure HWC format for RGB or grayscale
            if img.ndim == 2:  # If grayscale, add channel dim
                img = img[:, :, np.newaxis]
            elif img.ndim == 3 and img.shape[0] == 3:  # If CHW, convert to HWC
                img = img.transpose(1, 2, 0)
            # Now img should be (H, W, 1) for grayscale or (H, W, 3) for RGB

        # Apply CLAHE if requested (only to grayscale or first channel of RGB)
        if self.apply_clahe:
            # Create CLAHE on demand (no more instance attribute)
            clahe = cv2.createCLAHE(clipLimit=self.clahe_clip_limit, tileGridSize=self.clahe_grid_size)
            
            if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
                # Grayscale image or single channel
                img_to_process = img if img.ndim == 2 else img[:, :, 0]
                # Convert to uint8 for CLAHE
                img_uint8 = (img_to_process * 255).astype(np.uint8) if img_to_process.max() <= 1.0 else img_to_process.astype(np.uint8)
                processed = clahe.apply(img_uint8).astype(np.float32)
                
                if img.ndim == 2:
                    img = processed
                else:
                    img[:, :, 0] = processed
            elif img.ndim == 3 and img.shape[2] == 3:
                # RGB image - apply CLAHE to each channel separately
                for c in range(3):
                    channel = img[:, :, c]
                    img_uint8 = (channel * 255).astype(np.uint8) if channel.max() <= 1.0 else channel.astype(np.uint8)
                    img[:, :, c] = clahe.apply(img_uint8).astype(np.float32)

        # Extract landmarks
        landmarks = np.array([0.0, 0.0]) # Default placeholder
        if self.landmark_cols:
            landmarks = row[self.landmark_cols].values
            landmarks = landmarks.astype('float').reshape(-1, 2)

        # Create base sample dictionary
        sample = {'image': img, 'landmarks': landmarks}
        
        # Load and add depth map if requested
        if self.use_depth:
            depth_map = row['depth_map']
            # Ensure depth map is loaded correctly (should be (224, 224) shape)
            if isinstance(depth_map, np.ndarray) and depth_map.ndim == 2:
                sample['depth'] = depth_map.astype(np.float32)
            else:
                # Handle potential errors or provide a default depth map
                print(f"Warning: Invalid or missing depth map for index {idx}. Using zeros.")
                # Create zero depth map of appropriate shape
                sample['depth'] = np.zeros((224, 224), dtype=np.float32)
                 
        # Apply transforms
        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor:
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        depth = sample.get('depth') # Get depth if exists

        # Handle different image formats with and without depth
        if depth is not None:
            # Case 1: Image with depth
            
            # Ensure image is in HWC format
            if image.ndim == 3 and image.shape[0] == 3:  # CHW format
                image = image.transpose((1, 2, 0))
            elif image.ndim == 2:  # HW format (grayscale)
                image = image[:, :, np.newaxis]  # Add channel dim -> HWC
                
            # Ensure depth is in HW format
            if depth.ndim == 3:
                depth = depth.squeeze()  # Remove any singleton dimensions
            
            # Add channel dimension to depth map
            depth = depth[:, :, np.newaxis]
            
            # RGB+depth case: stack RGB channels and depth channel
            if image.shape[2] == 3:
                # Concatenate R,G,B,D channels -> RGBD (4 channels)
                combined = np.concatenate([image, depth], axis=2)
            else:
                # Concatenate grayscale and depth -> 2 channels
                combined = np.concatenate([image, depth], axis=2)
                
            # Convert to PyTorch tensor with channels first (C,H,W)
            combined = combined.transpose((2, 0, 1))
            image_tensor = torch.from_numpy(combined)
            
        else:
            # Case 2: Image without depth
            
            # For grayscale: ensure single channel format (C,H,W)
            if image.ndim == 2:  # HW -> CHW
                image = image[np.newaxis, :, :]  # Add channel dim at front
            elif image.ndim == 3:
                if image.shape[2] == 3 or image.shape[2] == 1:  # HWC -> CHW
                    image = image.transpose((2, 0, 1))
                # else assume already CHW format
                
            image_tensor = torch.from_numpy(image)
            
        # Convert landmarks to tensor
        landmarks_tensor = torch.from_numpy(landmarks)

        return {'image': image_tensor, 'landmarks': landmarks_tensor}


class Normalize:
    """Normalize image tensor (and optionally depth channel)."""
    def __init__(self, mean=(0.485,), std=(0.229,)):
        """
        Initialize normalizer with appropriate mean/std values.
        
        Args:
            mean: Mean values for RGB or grayscale channels
            std: Std values for RGB or grayscale channels
        """
        # Ensure mean and std are tuples or lists
        self.mean = mean if isinstance(mean, (list, tuple)) else (mean,)
        self.std = std if isinstance(std, (list, tuple)) else (std,)
        
        # Default values for different channel configurations
        self.rgb_mean = [0.485, 0.456, 0.406]  # ImageNet RGB means
        self.rgb_std = [0.229, 0.224, 0.225]   # ImageNet RGB stds
        self.depth_mean = [0.5]                # Centered depth mean
        self.depth_std = [0.5]                 # Centered depth std

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        num_channels = image.shape[0]
        
        # Prepare mean and std tensors based on number of channels
        if num_channels == 1:  # Grayscale only (1 channel)
            mean = torch.tensor(self.mean[:1]).view(-1, 1, 1)
            std = torch.tensor(self.std[:1]).view(-1, 1, 1)
        elif num_channels == 2:  # Grayscale + Depth (2 channels)
            # First channel: grayscale, Second channel: depth
            mean = torch.tensor([self.mean[0]] + self.depth_mean).view(-1, 1, 1)
            std = torch.tensor([self.std[0]] + self.depth_std).view(-1, 1, 1)
        elif num_channels == 3:  # RGB only (3 channels)
            # Use RGB normalization values
            mean = torch.tensor(self.rgb_mean).view(-1, 1, 1)
            std = torch.tensor(self.rgb_std).view(-1, 1, 1)
        elif num_channels == 4:  # RGB + Depth (4 channels)
            # First 3 channels: RGB, Fourth channel: depth
            mean = torch.tensor(self.rgb_mean + self.depth_mean).view(-1, 1, 1)
            std = torch.tensor(self.rgb_std + self.depth_std).view(-1, 1, 1)
        else:
            raise ValueError(f"Normalize expects 1-4 channels, but got {num_channels}")
            
        # Ensure tensors are on the same device and dtype
        mean = mean.to(image.device).type(image.dtype)
        std = std.to(image.device).type(image.dtype)

        # Apply normalization
        image = (image - mean) / (std + 1e-8)  # Add epsilon for stability

        return {'image': image, 'landmarks': landmarks}


def create_dataloaders(df, landmark_cols, batch_size=32, train_ratio=0.8, val_ratio=0.1, 
                       apply_clahe=True, root_dir=None, num_workers=4):
    """
    Create train, validation and test DataLoaders from a DataFrame
    
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
    
    # Define transformations
    train_transform = transforms.Compose([
        ToTensor(),
        Normalize()
    ])
    
    val_transform = transforms.Compose([
        ToTensor(),
        Normalize()
    ])
    
    # Create datasets
    train_dataset = CephalometricDataset(
        train_df, root_dir=root_dir, transform=train_transform, 
        landmark_cols=landmark_cols, train=True, apply_clahe=apply_clahe
    )
    
    val_dataset = CephalometricDataset(
        val_df, root_dir=root_dir, transform=val_transform, 
        landmark_cols=landmark_cols, train=False, apply_clahe=apply_clahe
    )
    
    test_dataset = CephalometricDataset(
        test_df, root_dir=root_dir, transform=val_transform, 
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