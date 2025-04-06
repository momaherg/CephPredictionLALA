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
        if 'Image' in self.dataframe.columns:
            # Image data is directly in the DataFrame (e.g., as numpy array)
            img_data = self.dataframe.iloc[idx]['Image']
            
            # Handle image data (list or numpy array)
            if isinstance(img_data, list):
                list_len = len(img_data)
                expected_len_gray = self.image_size[0] * self.image_size[1]
                expected_len_rgb = expected_len_gray * 3
                
                if list_len == expected_len_gray:
                    # Reshape grayscale list
                    img = np.array(img_data).reshape(self.image_size) # Shape (H, W)
                elif list_len == expected_len_rgb:
                    # Reshape RGB list and convert to grayscale
                    img_rgb = np.array(img_data).reshape((self.image_size[0], self.image_size[1], 3))
                    # Convert HWC to grayscale (using simple mean, could use weighted avg) 
                    img = img_rgb.mean(axis=2) # Shape (H, W)
                else:
                    raise ValueError(f"Row {idx}: List length {list_len} does not match expected grayscale ({expected_len_gray}) or RGB ({expected_len_rgb})")
            
            elif isinstance(img_data, np.ndarray):
                img = img_data
                # Ensure grayscale if numpy array is provided
                if img.ndim == 3:
                    if img.shape[0] == 3: # CHW -> HWC -> Gray
                        img = img.transpose(1, 2, 0).mean(axis=2)
                    elif img.shape[-1] == 3: # HWC -> Gray
                        img = img.mean(axis=2)
                    elif img.shape[-1] == 1: # HW1 -> HW
                        img = img.squeeze(axis=-1)
                # If img.ndim == 2, it's already grayscale HW
            else:
                raise TypeError(f"Row {idx}: Unexpected image data type: {type(img_data)}")

            # Ensure img is 2D after processing
            if img.ndim != 2:
                 raise ValueError(f"Row {idx}: Image processing failed, expected 2D array, got shape {img.shape}")
                 
            img = img.astype(np.float32) # Ensure float32 for transforms
        elif self.root_dir and 'image_path' in self.dataframe.columns:
            # Load image from file path
            img_name = os.path.join(self.root_dir, self.dataframe.iloc[idx, 0]) # Assuming first column is path
            img = io.imread(img_name)
            if img.ndim == 3: # Convert RGB to grayscale if necessary
                img = color.rgb2gray(img)
            img = img.astype(np.float32)
        else:
            raise ValueError("DataFrame must contain either 'Image' column or 'image_path' column with root_dir specified.")
            
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
             depth_map = self.dataframe.iloc[idx]['depth_map']
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