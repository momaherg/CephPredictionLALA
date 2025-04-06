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

        row_data = self.dataframe.iloc[idx]

        # Load image - Assume it's already a numpy array
        if 'Image' in row_data and isinstance(row_data['Image'], np.ndarray):
            img = row_data['Image'].astype(np.float32) # Ensure float32

            # Convert RGB to grayscale if needed (using cv2 for consistency)
            if img.ndim == 3 and img.shape[-1] == 3: # HWC format
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            elif img.ndim == 3 and img.shape[0] == 3: # CHW format
                 img_hwc = img.transpose(1, 2, 0)
                 img = cv2.cvtColor(img_hwc, cv2.COLOR_RGB2GRAY)
            # Add check for HW1 format
            elif img.ndim == 3 and img.shape[-1] == 1:
                 img = img.squeeze(-1) # HW1 -> HW

            # Final check: ensure img is 2D (HW)
            if img.ndim != 2:
                 # Handle error: maybe log, return None, or raise?
                 print(f"Warning: Image at index {idx} has unexpected shape {img.shape} after processing. Returning None sample.")
                 # Returning None might cause issues in DataLoader collation. Consider raising an error or returning a placeholder.
                 # For now, let's raise an error to stop immediately.
                 raise ValueError(f"Image at index {idx} has unexpected shape {img.shape} after processing.")

        elif self.root_dir and 'image_path' in row_data:
            # Load image from file path (keep existing logic)
            img_name = os.path.join(self.root_dir, row_data['image_path'])
            # Use cv2.imread to handle different formats, read as grayscale directly
            img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
            if img is None:
                 raise FileNotFoundError(f"Could not read image file: {img_name}")
            img = img.astype(np.float32)
        else:
            raise ValueError(f"Row {idx} must contain either a valid 'Image' numpy array or an 'image_path' with root_dir specified.")

        # Apply CLAHE if requested (on the 2D grayscale image)
        if self.apply_clahe:
            # Ensure uint8 for CLAHE
            img_uint8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            clahe = cv2.createCLAHE(clipLimit=self.clahe_clip_limit, tileGridSize=self.clahe_grid_size)
            img_clahe = clahe.apply(img_uint8)
            # Convert back to float32 (range 0-255)
            img = img_clahe.astype(np.float32)

        # Extract landmarks
        landmarks = np.array([0.0, 0.0]) # Default placeholder
        if self.landmark_cols:
            # Check if all landmark columns exist
            missing_cols = [col for col in self.landmark_cols if col not in row_data]
            if missing_cols:
                raise KeyError(f"Missing landmark columns for index {idx}: {missing_cols}")
            landmarks = row_data[self.landmark_cols].values.astype('float').reshape(-1, 2)

        # Create base sample dictionary
        sample = {'image': img, 'landmarks': landmarks} # img is now guaranteed 2D (HW) float32

        # Load and add depth map if requested
        if self.use_depth:
             depth_map_data = row_data.get('depth_map') # Use .get for safety
             if isinstance(depth_map_data, np.ndarray) and depth_map_data.ndim == 2 and depth_map_data.shape == img.shape:
                 sample['depth'] = depth_map_data.astype(np.float32) # Ensure float32
             else:
                 # Handle potential errors or provide a default depth map
                 if depth_map_data is None:
                      print(f"Warning: Depth map missing for index {idx}. Using zeros.")
                 else:
                      print(f"Warning: Invalid depth map (type: {type(depth_map_data)}, shape: {getattr(depth_map_data, 'shape', 'N/A')}) for index {idx}. Using zeros.")
                 sample['depth'] = np.zeros_like(img, dtype=np.float32)

        # Apply transforms
        if self.transform:
            sample = self.transform(sample) # Transform expects HW image and HW depth

        return sample


class ToTensor:
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        depth = sample.get('depth') # Get depth if exists

        # Handle image - ensure it's in right format for tensor conversion
        if image.ndim == 3 and image.shape[0] == 3:  # CHW format already
            pass
        elif image.ndim == 2:  # Add channel dimension for grayscale
            image = image[:, :, np.newaxis]  # HW -> HWC
        elif image.ndim == 3 and image.shape[-1] == 3:  # HWC format
            image = image.transpose(2, 0, 1)  # HWC -> CHW
            
        # If we still have HWC format at this point, convert to CHW
        if image.ndim == 3 and image.shape[-1] in [1, 3]:
            image = image.transpose(2, 0, 1)
        
        # Now handle depth map if present
        if depth is not None:
            # Ensure depth is 2D or has a single channel
            if depth.ndim == 3 and depth.shape[-1] == 1:
                depth = depth.squeeze(-1)  # HW1 -> HW
            elif depth.ndim == 3 and depth.shape[0] == 1:
                depth = depth.squeeze(0)  # 1HW -> HW
                
            # Add channel dimension if it's 2D
            if depth.ndim == 2:
                depth = depth[np.newaxis, :, :]  # HW -> CHW directly
            
            # Stack image and depth along channel dimension
            # At this point, image should be CHW and depth should be 1HW
            if image.shape[1:] == depth.shape[1:]:  # Check spatial dimensions match
                # Combine them - image is CHW, depth is 1HW
                combined = np.concatenate([image, depth], axis=0)
                image_tensor = torch.from_numpy(combined)
            else:
                print(f"Warning: Image shape {image.shape} and depth shape {depth.shape} don't match spatially. Using image only.")
                image_tensor = torch.from_numpy(image)
        else:
            # No depth, just convert image tensor
            image_tensor = torch.from_numpy(image)
            
        # Convert landmarks to tensor
        landmarks_tensor = torch.from_numpy(landmarks)

        return {'image': image_tensor, 'landmarks': landmarks_tensor}


class Normalize:
    """Normalize image tensor (and optionally depth channel)."""
    def __init__(self, 
                mean_gray=(0.485,), std_gray=(0.229,),
                mean_rgb=(0.485, 0.456, 0.406), std_rgb=(0.229, 0.224, 0.225),
                depth_mean=(0.5,), depth_std=(0.5,)):
        """
        Initialize normalizer with different normalization parameters for different input types.
        
        Args:
            mean_gray: Mean for grayscale image normalization
            std_gray: Std for grayscale image normalization
            mean_rgb: Mean for RGB image normalization (used if RGB inputs detected)
            std_rgb: Std for RGB image normalization
            depth_mean: Mean for depth channel normalization
            depth_std: Std for depth channel normalization
        """
        self.mean_gray = mean_gray
        self.std_gray = std_gray
        self.mean_rgb = mean_rgb
        self.std_rgb = std_rgb
        self.depth_mean = depth_mean
        self.depth_std = depth_std

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        num_channels = image.shape[0]

        # Choose appropriate normalization parameters based on channel count
        if num_channels == 1:  # Grayscale only
            mean = torch.tensor(self.mean_gray).view(-1, 1, 1)
            std = torch.tensor(self.std_gray).view(-1, 1, 1)
        elif num_channels == 2:  # Grayscale + Depth
            mean = torch.tensor(list(self.mean_gray) + list(self.depth_mean)).view(-1, 1, 1)
            std = torch.tensor(list(self.std_gray) + list(self.depth_std)).view(-1, 1, 1)
        elif num_channels == 3:  # RGB only
            mean = torch.tensor(self.mean_rgb).view(-1, 1, 1)
            std = torch.tensor(self.std_rgb).view(-1, 1, 1)
        elif num_channels == 4:  # RGB + Depth
            mean = torch.tensor(list(self.mean_rgb) + list(self.depth_mean)).view(-1, 1, 1)
            std = torch.tensor(list(self.std_rgb) + list(self.depth_std)).view(-1, 1, 1)
        else:
            raise ValueError(f"Normalize expects 1-4 channels, but got {num_channels}")
            
        # Ensure tensors are on the same device and dtype
        mean = mean.to(image.device).type(image.dtype)
        std = std.to(image.device).type(image.dtype)

        # Apply normalization
        image = (image - mean) / (std + 1e-8)  # Add epsilon for stability

        return {'image': image, 'landmarks': landmarks}


def create_dataloaders(df, landmark_cols, batch_size=32, train_ratio=0.8, val_ratio=0.1, 
                       apply_clahe=True, root_dir=None, num_workers=4, use_depth=False):
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
        use_depth (bool): Whether to include depth maps in the input features
        
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
    
    # Create dataset objects
    train_dataset = CephalometricDataset(
        train_df, 
        root_dir=root_dir, 
        transform=train_transform,
        landmark_cols=landmark_cols,
        train=True,
        apply_clahe=apply_clahe,
        use_depth=use_depth  # Pass use_depth parameter
    )
    
    val_dataset = CephalometricDataset(
        val_df, 
        root_dir=root_dir, 
        transform=val_transform,
        landmark_cols=landmark_cols,
        train=False,
        apply_clahe=apply_clahe,
        use_depth=use_depth  # Pass use_depth parameter
    )
    
    test_dataset = CephalometricDataset(
        test_df, 
        root_dir=root_dir, 
        transform=val_transform,
        landmark_cols=landmark_cols,
        train=False,
        apply_clahe=apply_clahe,
        use_depth=use_depth  # Pass use_depth parameter
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