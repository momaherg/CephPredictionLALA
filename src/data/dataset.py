import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image
from torchvision import transforms

class CephalometricDataset(Dataset):
    """Cephalometric landmark dataset."""

    def __init__(self, dataframe, root_dir=None, transform=None, 
                 all_landmark_cols=None, target_indices=None,
                 train=False, apply_clahe=True, clahe_clip_limit=2.0, 
                 clahe_grid_size=(8, 8)):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame with image info and landmarks.
            root_dir (string, optional): Directory with all the images (if paths are relative).
            transform (callable, optional): Optional transform to be applied on a sample.
            all_landmark_cols (list): List of ALL column names for landmark coordinates (x, y).
            target_indices (list, optional): List of 0-based indices of landmarks to use. 
                                            If None, all landmarks are used.
            train (bool): Indicates if this is the training dataset (for augmentation control).
            apply_clahe (bool): Apply CLAHE preprocessing.
            clahe_clip_limit (float): Clip limit for CLAHE.
            clahe_grid_size (tuple): Grid size for CLAHE.
        """
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.apply_clahe = apply_clahe
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_grid_size = clahe_grid_size
        
        self.all_landmark_cols = all_landmark_cols
        self.target_indices = target_indices
        
        # Determine the final landmark columns to be extracted
        if self.target_indices is not None:
            self.landmark_cols = []
            for idx in self.target_indices:
                 if 2*idx + 1 < len(self.all_landmark_cols):
                     self.landmark_cols.extend(self.all_landmark_cols[2*idx : 2*idx+2])
                 else:
                     # This should ideally be caught earlier, but double-check
                     raise IndexError(f"Landmark index {idx} is out of bounds.")
            self.num_landmarks = len(self.target_indices)
        else:
            self.landmark_cols = self.all_landmark_cols
            self.num_landmarks = len(self.landmark_cols) // 2

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get image data
        if 'image_path' in self.dataframe.columns:
            img_name = self.dataframe.iloc[idx, self.dataframe.columns.get_loc('image_path')]
            if self.root_dir:
                img_name = os.path.join(self.root_dir, img_name)
            try:
                image = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    raise FileNotFoundError(f"Image not found or could not be read: {img_name}")
            except Exception as e:
                print(f"Error loading image {img_name}: {e}")
                # Return a dummy sample or raise an error
                return None # Or raise an exception if preferred
        elif 'Image' in self.dataframe.columns:
            # Assumes image data is stored directly in the DataFrame (e.g., flattened list)
            img_data = self.dataframe.iloc[idx]['Image']
            # Infer image size (assuming square for now)
            side_len = int(np.sqrt(len(img_data)))
            image = np.array(img_data, dtype=np.uint8).reshape(side_len, side_len)
        else:
            raise ValueError("DataFrame must contain either 'image_path' or 'Image' column")

        # Apply CLAHE if requested
        if self.apply_clahe:
            clahe = cv2.createCLAHE(clipLimit=self.clahe_clip_limit, tileGridSize=self.clahe_grid_size)
            image = clahe.apply(image)
        
        # Ensure image is 2D (grayscale)
        if len(image.shape) > 2:
             image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Convert if accidentally loaded as BGR

        # Get landmark coordinates (only the ones specified by self.landmark_cols)
        landmarks = self.dataframe.iloc[idx][self.landmark_cols].values.astype('float')
        landmarks = landmarks.reshape(-1, 2) # Reshape to (num_landmarks, 2)

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