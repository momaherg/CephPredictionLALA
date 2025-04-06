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
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.dataframe.iloc[idx]
        
        # --- Image Loading --- 
        img_data = row['Image']
        if isinstance(img_data, np.ndarray):
            img = img_data.astype(np.float32)
        else:
             # This should ideally not happen if DataProcessor ran correctly
             # You might want to log a more specific error or handle list conversion again
             # For now, raise an error if it's not an ndarray as expected.
             raise TypeError(f"Expected 'Image' column to contain numpy arrays, but got {type(img_data)} at index {idx}")
             
        # --- Landmark Loading ---
        landmarks = row[self.landmark_cols].values.astype('float32').reshape(-1, 2)
        
        # --- Depth Map Loading (if applicable) ---
        depth_map = None
        if self.use_depth:
            depth_data = row.get('depth_map', None) # Use .get for safety
            if isinstance(depth_data, np.ndarray):
                 # Ensure depth is float32 and has shape (H, W) - add channel dim later if needed by transforms
                 depth_map = depth_data.astype(np.float32)
                 if depth_map.ndim == 3: # Handle cases where it might have an unnecessary channel dim
                     depth_map = depth_map.squeeze()
                 if depth_map.shape != (img.shape[0], img.shape[1]): # Check shape consistency with image H,W
                      raise ValueError(f"Depth map at index {idx} has shape {depth_map.shape}, expected {(img.shape[0], img.shape[1])}")
            elif depth_data is not None:
                 raise TypeError(f"Expected 'depth_map' column to contain numpy arrays, but got {type(depth_data)} at index {idx}")
            # If depth_data is None, depth_map remains None

        # --- Apply CLAHE if requested (only to image) ---
        if self.apply_clahe:
             # Ensure image is uint8 for CLAHE
             if img.dtype != np.uint8:
                 # Scale if float 0-1 range
                 if img.max() <= 1.0 and img.min() >= 0.0:
                      img = (img * 255).astype(np.uint8)
                 else: # Assume 0-255 range otherwise
                      img = img.astype(np.uint8)
                      
             # Apply CLAHE to grayscale version if RGB
             if img.ndim == 3 and img.shape[-1] == 3:
                 img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                 clahe_img = self._create_clahe().apply(img_gray)
                 img = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2RGB) # Convert back to 3 channels
             elif img.ndim == 2: # Apply directly if grayscale
                 img = self._create_clahe().apply(img)
             else:
                 print(f"Warning: Skipping CLAHE for unexpected image shape {img.shape} at index {idx}")
                 # Ensure img is float32 again if CLAHE wasn't applied as expected
                 img = img.astype(np.float32) / 255.0 if img.dtype == np.uint8 else img.astype(np.float32)
                 
             # Convert back to float32 (0-1 range) after CLAHE
             img = img.astype(np.float32) / 255.0
        
        # --- Prepare Sample Dictionary ---
        sample = {'image': img, 'landmarks': landmarks}
        if self.use_depth and depth_map is not None:
             sample['depth'] = depth_map # Add depth map if available
        
        # --- Apply Transforms --- 
        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor:
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        depth = sample.get('depth') # Get depth if exists

        # Handle image:
        # Ensure image is HWC if it has 3 dims (e.g., after some OpenCV ops)
        if image.ndim == 3 and image.shape[0] == 3:
            image = image.transpose((1, 2, 0)) # CHW to HWC
        elif image.ndim == 2:
            # Add channel dimension for grayscale image
            image = image[:, :, np.newaxis]
            
        # Handle depth map:
        if depth is not None:
             if depth.ndim == 2:
                 depth = depth[:, :, np.newaxis] # Add channel dim
             # Concatenate image and depth along the channel axis (C)
             # Ensure image and depth are compatible (e.g., grayscale image + depth)
             if image.shape[-1] == 1 and depth.shape[-1] == 1:
                 image = np.concatenate((image, depth), axis=2) # Now image is H, W, 2
             else:
                 print(f"Warning: Image shape {image.shape} or depth shape {depth.shape} not suitable for concatenation. Using image only.")
                 # Fallback: only use image if concatenation is not straightforward
                 if image.shape[-1] != 1:
                      print("Warning: Image has more than 1 channel, cannot combine with depth. Using first channel.")
                      image = image[:,:,0:1]
        elif image.shape[-1] != 1: # Ensure image has only 1 channel if no depth
             print(f"Warning: No depth map, but image has shape {image.shape}. Using first channel.")
             image = image[:,:,0:1]

        # Transpose C x H x W for PyTorch
        # Image shape is now (H, W, C) where C=1 (grayscale) or C=2 (grayscale+depth)
        image = image.transpose((2, 0, 1))
        image_tensor = torch.from_numpy(image)
        landmarks_tensor = torch.from_numpy(landmarks)

        return {'image': image_tensor, 'landmarks': landmarks_tensor}


class Normalize:
    """Normalize image tensor (and optionally depth channel)."""
    # Default mean/std for grayscale images. Add values for depth channel if present.
    def __init__(self, mean=(0.485,), std=(0.229,)): # Default for 1 channel (grayscale)
        # Ensure mean and std are tuples or lists
        self.mean = mean if isinstance(mean, (list, tuple)) else (mean,)
        self.std = std if isinstance(std, (list, tuple)) else (std,)
        
        # Expected mean/std for depth (normalized 0-1)
        self.depth_mean = [0.5] 
        self.depth_std = [0.5]  

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        num_channels = image.shape[0]
        
        # Prepare mean and std tensors based on number of channels
        if num_channels == 1: # Grayscale only
            mean = torch.tensor(self.mean).view(-1, 1, 1)
            std = torch.tensor(self.std).view(-1, 1, 1)
        elif num_channels == 2: # Grayscale + Depth
            mean = torch.tensor(list(self.mean) + self.depth_mean).view(-1, 1, 1)
            std = torch.tensor(list(self.std) + self.depth_std).view(-1, 1, 1)
        # Add case for 4 channels if RGB + Depth is used later
        # elif num_channels == 4: # RGB + Depth
        #     mean = torch.tensor(list(self.mean) + self.depth_mean).view(-1, 1, 1)
        #     std = torch.tensor(list(self.std) + self.depth_std).view(-1, 1, 1)
        else:
            raise ValueError(f"Normalize expects 1 or 2 channels, but got {num_channels}")
            
        # Ensure tensors are on the same device and dtype
        mean = mean.to(image.device).type(image.dtype)
        std = std.to(image.device).type(image.dtype)

        # Apply normalization
        image = (image - mean) / (std + 1e-8) # Add epsilon for stability

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