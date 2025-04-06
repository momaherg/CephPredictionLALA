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

        # Load image - with proper handling of RGB images
        if 'Image' in self.dataframe.columns:
            # Image data is directly in the DataFrame (e.g., as numpy array)
            img_data = self.dataframe.iloc[idx]['Image']
            img = None  # Initialize image variable
            
            # Handle different types of image data
            if isinstance(img_data, list):
                img_data_np = np.array(img_data)
                list_len = len(img_data)
                
                # Determine if grayscale or RGB based on size
                if list_len == 224 * 224:  # Grayscale
                    img = img_data_np.reshape((224, 224)).astype(np.float32)
                elif list_len == 224 * 224 * 3:  # RGB
                    # Reshape as RGB, then convert to grayscale
                    rgb_img = img_data_np.reshape((224, 224, 3)).astype(np.float32)
                    img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
                else:
                    raise ValueError(f"Cannot determine image shape from list length {list_len}")
                    
            elif isinstance(img_data, np.ndarray):
                # Handle numpy arrays based on shape
                if img_data.ndim == 2:  # Already grayscale
                    img = img_data.astype(np.float32)
                elif img_data.ndim == 3:
                    # Determine format and convert to grayscale
                    if img_data.shape[-1] == 3:  # HWC format (e.g., (224, 224, 3))
                        img = cv2.cvtColor(img_data.astype(np.float32), cv2.COLOR_RGB2GRAY)
                    elif img_data.shape[0] == 3:  # CHW format (e.g., (3, 224, 224))
                        img = cv2.cvtColor(img_data.transpose(1, 2, 0).astype(np.float32), cv2.COLOR_RGB2GRAY)
                    elif img_data.shape[-1] == 1:  # HW1 format
                        img = img_data.squeeze(-1).astype(np.float32)
                    else:
                        raise ValueError(f"Unsupported array shape: {img_data.shape}")
                else:
                    raise ValueError(f"Unsupported array dimensions: {img_data.ndim}")
            else:
                raise TypeError(f"Unsupported image data type: {type(img_data)}")
                
        elif self.root_dir and 'image_path' in self.dataframe.columns:
            # Load image from file path
            img_path = os.path.join(self.root_dir, self.dataframe.iloc[idx]['image_path'])
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
            if img is None:
                raise ValueError(f"Failed to load image from path: {img_path}")
        else:
            raise ValueError("DataFrame must contain either 'Image' column or 'image_path' column with root_dir")
            
        # Ensure img is a valid 2D grayscale image at this point
        if img is None or img.ndim != 2:
            raise ValueError(f"Failed to create a valid grayscale image. Shape: {None if img is None else img.shape}")

        # Apply CLAHE if requested
        if self.apply_clahe:
            # Convert to uint8 for CLAHE
            img_uint8 = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
            clahe = cv2.createCLAHE(clipLimit=self.clahe_clip_limit, tileGridSize=self.clahe_grid_size)
            img = clahe.apply(img_uint8).astype(np.float32)

        # Extract landmarks
        landmarks = np.array([0.0, 0.0])  # Default placeholder
        if self.landmark_cols:
            landmarks = self.dataframe.iloc[idx][self.landmark_cols].values
            landmarks = landmarks.astype('float').reshape(-1, 2)

        # Create base sample dictionary
        sample = {'image': img, 'landmarks': landmarks}
        
        # Load and add depth map if requested
        if self.use_depth:
            depth_map = self.dataframe.iloc[idx]['depth_map']
            
            # Ensure depth map is valid
            if isinstance(depth_map, np.ndarray) and depth_map.shape == (224, 224):
                sample['depth'] = depth_map.astype(np.float32)
            else:
                # Debug output for depth map issue
                if isinstance(depth_map, np.ndarray):
                    shape_info = f"shape={depth_map.shape}, dtype={depth_map.dtype}"
                else:
                    shape_info = f"type={type(depth_map)}"
                print(f"Warning: Invalid depth map for index {idx}: {shape_info}. Using zeros.")
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
        
        # Step 1: Ensure image is 2D grayscale (this should always be the case after __getitem__)
        if image.ndim != 2:
            raise ValueError(f"Expected 2D grayscale image, got shape {image.shape}")
        
        # Step 2: Add channel dimension to grayscale image
        image_tensor = torch.from_numpy(image[np.newaxis, ...])  # Add channel dim: (1, H, W)
        
        # Step 3: Handle depth if present
        if depth is not None:
            # Ensure depth is 2D
            if depth.ndim != 2:
                raise ValueError(f"Expected 2D depth map, got shape {depth.shape}")
                
            # Convert depth to tensor with channel dim
            depth_tensor = torch.from_numpy(depth[np.newaxis, ...])  # Shape: (1, H, W)
            
            # Concatenate image and depth along the channel dimension
            combined_tensor = torch.cat([image_tensor, depth_tensor], dim=0)  # Shape: (2, H, W)
            
            # Use the combined tensor as our image tensor
            image_tensor = combined_tensor
        
        # Convert landmarks to tensor
        landmarks_tensor = torch.from_numpy(landmarks)

        return {'image': image_tensor, 'landmarks': landmarks_tensor}


class Normalize:
    """Normalize image tensor (and optionally depth channel)."""
    def __init__(self, 
                grayscale_mean=0.485, grayscale_std=0.229,
                depth_mean=0.5, depth_std=0.5):
        """
        Initialize the normalizer with parameters for different channels.
        
        Args:
            grayscale_mean: Mean for grayscale image normalization
            grayscale_std: Std for grayscale image normalization
            depth_mean: Mean for depth channel normalization
            depth_std: Std for depth channel normalization
        """
        self.grayscale_mean = grayscale_mean
        self.grayscale_std = grayscale_std
        self.depth_mean = depth_mean
        self.depth_std = depth_std

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        
        # Get the number of channels
        num_channels = image.shape[0]
        
        if num_channels == 1:  # Grayscale only
            # Create parameters for 1-channel normalization
            mean = torch.tensor([self.grayscale_mean], dtype=torch.float32).view(-1, 1, 1)
            std = torch.tensor([self.grayscale_std], dtype=torch.float32).view(-1, 1, 1)
            
        elif num_channels == 2:  # Grayscale + Depth
            # Create parameters for 2-channel normalization
            mean = torch.tensor([self.grayscale_mean, self.depth_mean], dtype=torch.float32).view(-1, 1, 1)
            std = torch.tensor([self.grayscale_std, self.depth_std], dtype=torch.float32).view(-1, 1, 1)
            
        else:
            raise ValueError(f"Normalize expects 1 or 2 channels, but got {num_channels}")
        
        # Ensure tensors are on the same device and dtype as the image
        mean = mean.to(image.device).type(image.dtype)
        std = std.to(image.device).type(image.dtype)
        
        # Apply normalization
        normalized_image = (image - mean) / (std + 1e-8)  # Add epsilon for numerical stability
        
        return {'image': normalized_image, 'landmarks': landmarks}


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