import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image
from torchvision import transforms

class CephalometricDataset(Dataset):
    def __init__(self, data_frame, root_dir=None, transform=None, 
                 landmark_cols=None, train=True, apply_clahe=True,
                 clahe_clip_limit=2.0, clahe_grid_size=(8,8),
                 use_depth=False):
        """
        Args:
            data_frame (pandas.DataFrame): DataFrame containing image paths and landmarks
            root_dir (string): Directory with all the images
            transform (callable, optional): Optional transform to be applied on a sample
            landmark_cols (list): Columns containing landmark coordinates
            train (bool): Whether this is training set or not
            apply_clahe (bool): Whether to apply CLAHE for histogram equalization
            clahe_clip_limit (float): Clip limit for CLAHE
            clahe_grid_size (tuple): Grid size for CLAHE
            use_depth (bool): Whether to include depth features in the sample
        """
        self.data_frame = data_frame
        self.root_dir = root_dir
        self.transform = transform
        self.landmark_cols = landmark_cols
        self.train = train
        self.apply_clahe = apply_clahe
        
        # Store CLAHE parameters instead of the object itself
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_grid_size = clahe_grid_size
        
        # Whether to include depth features
        self.use_depth = use_depth
    
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get image data
        if 'Image' in self.data_frame.columns:
            # Assuming 'Image' column contains pixel array data
            img_data = self.data_frame.iloc[idx]['Image']
            # Convert from list of RGB tuples to numpy array if needed
            if isinstance(img_data, list):
                img_array = np.array(img_data).reshape(224, 224, 3)
            else:
                img_array = img_data
        else:
            # For synthetic data when no actual images are available, create a blank image
            if self.root_dir is None or 'image_path' in self.data_frame.columns:
                # Create a synthetic image for demonstration
                img_array = np.zeros((224, 224, 3), dtype=np.uint8)
                # Add some random noise to make it more realistic
                img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            else:
                # If images are stored as files
                img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx]['patient'] + '.jpg')
                img_array = np.array(Image.open(img_name))
        
        # Ensure image is in the right format (uint8) for OpenCV operations
        if img_array.dtype != np.uint8:
            # If float in range [0,1], convert to [0,255]
            if img_array.dtype == np.float32 or img_array.dtype == np.float64:
                if img_array.max() <= 1.0:
                    img_array = (img_array * 255).astype(np.uint8)
                else:
                    img_array = img_array.astype(np.uint8)
            else:
                # For other types, simply convert to uint8
                img_array = img_array.astype(np.uint8)
        
        # Apply CLAHE for histogram equalization
        if self.apply_clahe and len(img_array.shape) == 3:
            try:
                # Create CLAHE object on-demand (no longer stored as instance attribute)
                clahe = cv2.createCLAHE(clipLimit=self.clahe_clip_limit, tileGridSize=self.clahe_grid_size)
                
                # Convert to LAB color space
                lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                
                # Apply CLAHE to L channel
                cl = clahe.apply(l)
                
                # Merge the CLAHE enhanced L channel with the a and b channels
                limg = cv2.merge((cl, a, b))
                
                # Convert back to RGB color space
                img_array = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
            except cv2.error as e:
                # If color conversion fails, use original image
                print(f"Warning: CLAHE could not be applied, using original image. Error: {e}")
                # You might want to log this instead of printing
        
        # Get landmarks if available
        if self.landmark_cols and all(col in self.data_frame.columns for col in self.landmark_cols):
            landmarks = self.data_frame.iloc[idx][self.landmark_cols].values.astype('float32')
            # Reshape to (num_landmarks, 2) for easier processing
            landmarks = landmarks.reshape(-1, 2)
        else:
            landmarks = np.array([])
        
        # Get depth features if available and requested
        depth_feature = None
        if self.use_depth and 'depth_feature' in self.data_frame.columns:
            depth_feature = self.data_frame.iloc[idx]['depth_feature']
            
            # If depth feature is None or NaN, create an empty array
            if depth_feature is None or (hasattr(depth_feature, 'size') and depth_feature.size == 0):
                # Create an empty depth map of the same size as the image (single channel)
                depth_feature = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=np.float32)
        
        # Create sample
        sample = {'image': img_array, 'landmarks': landmarks}
        
        # Add depth feature if available
        if self.use_depth:
            sample['depth'] = depth_feature
        
        # Apply transformations
        if self.transform:
            sample = self.transform(sample)
        
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        
        # Convert image to tensor
        # Transpose image dimensions: (H x W x C) -> (C X H X W)
        image = image.transpose((2, 0, 1))
        
        result = {
            'image': torch.from_numpy(image).float() / 255.0,
            'landmarks': torch.from_numpy(landmarks).float(),
        }
        
        # Convert depth to tensor if available
        if 'depth' in sample and sample['depth'] is not None:
            depth = sample['depth']
            # Add channel dimension if needed
            if len(depth.shape) == 2:
                depth = np.expand_dims(depth, axis=0)  # (H, W) -> (1, H, W)
            elif len(depth.shape) == 3 and depth.shape[2] == 1:
                depth = depth.transpose((2, 0, 1))  # (H, W, 1) -> (1, H, W)
            elif len(depth.shape) == 3:
                depth = depth.transpose((2, 0, 1))  # (H, W, C) -> (C, H, W)
                
            # Normalize depth to [0, 1] if needed
            if depth.max() > 1.0:
                depth = depth.astype(np.float32) / 255.0
                
            result['depth'] = torch.from_numpy(depth).float()
        
        return result


class Normalize(object):
    """Normalize the image tensors."""
    
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std
    
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        
        # Normalize image
        for t, m, s in zip(image, self.mean, self.std):
            t.sub_(m).div_(s)
            
        result = {'image': image, 'landmarks': landmarks}
        
        # Pass through depth without normalizing (already normalized)
        if 'depth' in sample:
            result['depth'] = sample['depth']
            
        return result


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
        use_depth (bool): Whether to include depth features in the sample
        
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
        landmark_cols=landmark_cols, train=True, apply_clahe=apply_clahe,
        use_depth=use_depth
    )
    
    val_dataset = CephalometricDataset(
        val_df, root_dir=root_dir, transform=val_transform, 
        landmark_cols=landmark_cols, train=False, apply_clahe=apply_clahe,
        use_depth=use_depth
    )
    
    test_dataset = CephalometricDataset(
        test_df, root_dir=root_dir, transform=val_transform, 
        landmark_cols=landmark_cols, train=False, apply_clahe=apply_clahe,
        use_depth=use_depth
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