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

        # Load image
        img_data = self.dataframe.iloc[idx]['Image']
        depth_data = self.dataframe.iloc[idx]['depth_map'] if self.use_depth and 'depth_map' in self.dataframe.columns else None

        # Handle image data loading (from list or numpy array)
        img = None
        if isinstance(img_data, list):
            try:
                np_array_flat = np.array(img_data)
                list_len = np_array_flat.size
                expected_len_gray = 224 * 224 # Assuming fixed size for now
                expected_len_rgb = expected_len_gray * 3

                if list_len == expected_len_gray:
                    img = np_array_flat.reshape((224, 224)).astype(np.float32)
                elif list_len == expected_len_rgb:
                    # Reshape and potentially convert to grayscale if needed later, or handle 3 channels
                    img = np_array_flat.reshape((224, 224, 3)).astype(np.float32)
                    # If grayscale is always expected later, convert here:
                    # img = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.float32)
                else:
                     raise ValueError(f"List length {list_len} doesn't match expected gray {expected_len_gray} or RGB {expected_len_rgb}")
            except ValueError as e:
                 print(f"Error reshaping image list at index {idx}: {e}")
                 # Handle error: return None or raise, depending on desired behavior
                 # For now, let's return None to indicate failure for this sample
                 return None # Or raise an exception
        elif isinstance(img_data, np.ndarray):
            img = img_data.astype(np.float32)
        else:
            # Handle unexpected data type
            print(f"Warning: Unexpected image data type at index {idx}: {type(img_data)}")
            return None # Indicate failure

        # If the loaded image is RGB but we expect Grayscale later, convert it
        # (Common if transforms expect single channel input)
        if img is not None and img.ndim == 3 and img.shape[2] == 3:
            # Example: Convert RGB to Grayscale using standard weights
            img = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140]) 
            # Ensure it's back to (H, W) shape
            img = img.reshape((224, 224)).astype(np.float32)

        # Ensure image is 2D (H, W) before applying CLAHE/transforms expecting grayscale
        if img is not None and img.ndim != 2:
             print(f"Warning: Image at index {idx} is not 2D (shape: {img.shape}) after initial load/conversion. Skipping.")
             return None # Indicate failure

        # Apply CLAHE if requested
        if self.apply_clahe:
            # Convert to uint8 for CLAHE
            img_uint8 = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
            clahe = cv2.createCLAHE(clipLimit=self.clahe_clip_limit, tileGridSize=self.clahe_grid_size)
            img = clahe.apply(img_uint8)
            # Convert back to float32 (0-255 range is fine for normalization later)
            img = img.astype(np.float32)

        # Extract landmarks
        landmarks = np.array([0.0, 0.0]) # Default placeholder
        if self.landmark_cols:
            landmarks = self.dataframe.iloc[idx][self.landmark_cols].values
            landmarks = landmarks.astype('float').reshape(-1, 2)

        # Create base sample dictionary
        sample = {'image': img, 'landmarks': landmarks}
        
        # Load and add depth map if requested
        if self.use_depth:
             depth_map = depth_data
             # Ensure depth map is loaded correctly and has the right shape (H, W)
             if isinstance(depth_map, np.ndarray) and depth_map.shape == img.shape:
                 sample['depth'] = depth_map.astype(np.float32)
             else:
                 # Handle potential errors or provide a default depth map
                 print(f"Warning: Invalid or missing depth map for index {idx}. Using zeros.")
                 sample['depth'] = np.zeros_like(img, dtype=np.float32)
                 
        # Apply transforms
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