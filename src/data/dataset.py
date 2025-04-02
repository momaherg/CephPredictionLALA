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
                 clahe_clip_limit=2.0, clahe_grid_size=(8,8)):
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
        
        # Create sample
        sample = {'image': img_array, 'landmarks': landmarks}
        
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