import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image
from torchvision import transforms
import ast

class CephalometricDataset(Dataset):
    """Cephalometric X-ray Dataset"""

    def __init__(self, dataframe, root_dir=None, transform=None, 
                 all_landmark_cols=None, target_indices=None, 
                 train=False, apply_clahe=True, 
                 clahe_clip_limit=2.0, clahe_grid_size=(8, 8),
                 image_size=(224, 224)):
        """
        Args:
            dataframe (pandas.DataFrame): DataFrame with image paths and landmarks.
            root_dir (string, optional): Directory with all the images. If None, uses paths in dataframe.
            transform (callable, optional): Optional transform to be applied on a sample.
            all_landmark_cols (list): List of ALL column names containing landmark coordinates.
            target_indices (list, optional): List of 0-based indices of the landmarks to extract.
                                            If None, all landmarks are used.
            train (bool): Flag indicating if this is the training dataset.
            apply_clahe (bool): Whether to apply CLAHE preprocessing.
            clahe_clip_limit (float): Clip limit for CLAHE.
            clahe_grid_size (tuple): Grid size for CLAHE.
            image_size (tuple): Expected size of the images (height, width).
        """
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.apply_clahe = apply_clahe
        self.image_size = image_size
        
        self.all_landmark_cols = all_landmark_cols
        self.target_indices = target_indices
        
        if self.target_indices is not None:
            self.landmark_cols = []
            max_original_index = (len(self.all_landmark_cols) // 2) - 1
            for idx in self.target_indices:
                if 0 <= idx <= max_original_index:
                    self.landmark_cols.extend(self.all_landmark_cols[2*idx : 2*idx+2])
                else:
                    raise IndexError(f"Target landmark index {idx} is out of bounds. " 
                                     f"Available indices: 0 to {max_original_index}")
            self.num_landmarks = len(self.target_indices)
        else:
            self.landmark_cols = self.all_landmark_cols
            self.num_landmarks = len(self.landmark_cols) // 2
            
        if self.apply_clahe:
            self.clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_grid_size)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_info = self.dataframe.iloc[idx]
        
        if 'image_path' in img_info and pd.notna(img_info['image_path']):
            img_path = img_info['image_path']
            if self.root_dir:
                img_path = os.path.join(self.root_dir, img_path)
            
            try:
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                     raise FileNotFoundError(f"Image not found or unable to read: {img_path}")
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                raise
                
        elif 'Image' in img_info:
            img_data = img_info['Image']
            # Check if the data is actually missing (None or NaN) before proceeding
            # pd.isna works correctly on scalars
            if pd.isna(img_data):
                 # If 'Image' column exists but value is NaN/None, treat as invalid
                 raise ValueError(f"DataFrame contains 'Image' column but value is missing at index {idx}")

            # --- If we reach here, img_data is not None/NaN --- 
            
            # Handle string representation or direct list/array
            if isinstance(img_data, str):
                try:
                    # Safely evaluate the string representation into a list
                    img_data = ast.literal_eval(img_data)
                except (ValueError, SyntaxError) as e:
                     raise ValueError(f"Could not parse 'Image' column data at index {idx}: {e}. Data: {img_data[:100]}...")
            
            # Ensure img_data is now a list or numpy array after potential parsing
            if not isinstance(img_data, (list, np.ndarray)):
                # This might be redundant now but adds safety
                raise TypeError(f"Expected 'Image' column data to be a list or numpy array at index {idx}, got {type(img_data)}")
                
            # Validate image data size before reshaping
            expected_size = self.image_size[0] * self.image_size[1]
            if len(img_data) != expected_size:
                raise ValueError(f"Image data length mismatch at index {idx}. " 
                                 f"Expected {expected_size} ({self.image_size[0]}x{self.image_size[1]}), " 
                                 f"but got {len(img_data)}.")
                                 
            # Reshape the flattened data into HxW format
            try:
                image = np.array(img_data, dtype=np.uint8).reshape(self.image_size[0], self.image_size[1])
            except Exception as e:
                raise ValueError(f"Error reshaping image data at index {idx}: {e}")

        else:
            # This else now covers cases where neither 'image_path' nor 'Image' columns exist
            raise ValueError(f"DataFrame must contain either 'image_path' or 'Image' column with valid data at index {idx}")
        
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif len(image.shape) == 3 and image.shape[2] == 4:
             image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        elif len(image.shape) != 2:
            raise ValueError(f"Unexpected image shape {image.shape} at index {idx} after loading.")
            
        if self.apply_clahe:
            image = self.clahe.apply(image)
            
        landmarks = img_info[self.landmark_cols].values.astype('float')
        landmarks = landmarks.reshape(-1, 2)
        
        sample = {'image': image, 'landmarks': landmarks}
        
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
        
        return {
            'image': torch.from_numpy(image).float() / 255.0,
            'landmarks': torch.from_numpy(landmarks).float(),
        }


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
        all_landmark_cols=landmark_cols, target_indices=None,
        train=True, apply_clahe=apply_clahe
    )
    
    val_dataset = CephalometricDataset(
        val_df, root_dir=root_dir, transform=val_transform, 
        all_landmark_cols=landmark_cols, target_indices=None,
        train=False, apply_clahe=apply_clahe
    )
    
    test_dataset = CephalometricDataset(
        test_df, root_dir=root_dir, transform=val_transform, 
        all_landmark_cols=landmark_cols, target_indices=None,
        train=False, apply_clahe=apply_clahe
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