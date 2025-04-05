import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image
from torchvision import transforms
from skimage import io, transform
import platform

# Conditional import for Depth Model
try:
    from transformers import DepthProImageProcessorFast, DepthProForDepthEstimation
    from requests.exceptions import RequestException
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

class CephalometricDataset(Dataset):
    """Cephalometric landmark dataset."""

    def __init__(self, dataframe, root_dir=None, transform=None, 
                 landmark_cols=None, train=True, apply_clahe=False,
                 use_depth_features=False, image_size=(224, 224)):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame with image paths and landmarks.
            root_dir (string, optional): Directory with all the images. If None, assumes images are in dataframe.
            transform (callable, optional): Optional transform to be applied on a sample.
            landmark_cols (list): List of landmark column names.
            train (bool): Indicates if the dataset is for training (used for CLAHE).
            apply_clahe (bool): Whether to apply CLAHE.
            use_depth_features (bool): Whether to predict and add depth as a feature.
            image_size (tuple): Target size for images and depth maps.
        """
        self.landmarks_frame = dataframe
        self.root_dir = root_dir
        self.transform = transform
        self.landmark_cols = landmark_cols
        self.train = train
        self.apply_clahe = apply_clahe
        self.use_depth_features = use_depth_features
        self.image_size = image_size
        
        # Initialize CLAHE object here if needed, but create on demand in getitem
        self.clahe_clip_limit = 2.0
        self.clahe_grid_size = (8, 8)
        
        # Initialize Depth model if requested and available
        self.depth_model = None
        self.depth_processor = None
        if self.use_depth_features:
            if TRANSFORMERS_AVAILABLE:
                try:
                    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if platform.system() == 'Darwin' and torch.backends.mps.is_available() else "cpu")
                    self.depth_processor = DepthProImageProcessorFast.from_pretrained("apple/DepthPro-hf")
                    self.depth_model = DepthProForDepthEstimation.from_pretrained("apple/DepthPro-hf").to(device)
                    self.depth_model.eval() # Set to evaluation mode
                    print(f"Depth model apple/DepthPro-hf loaded successfully on device: {device}")
                except (OSError, RequestException, ImportError) as e:
                    print(f"Warning: Failed to load depth model apple/DepthPro-hf: {e}. Depth features will not be used.")
                    self.use_depth_features = False # Disable if loading fails
            else:
                print("Warning: 'transformers' package not found. Cannot use depth features. Install with 'pip install transformers'")
                self.use_depth_features = False

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load image
        image = None
        if 'image_path' in self.landmarks_frame.columns and self.root_dir:
            img_name = os.path.join(self.root_dir, self.landmarks_frame.iloc[idx, self.landmarks_frame.columns.get_loc('image_path')])
            try:
                image = io.imread(img_name)
            except FileNotFoundError:
                print(f"Warning: Image file not found: {img_name}")
                # Return a placeholder or handle error appropriately
                return None 
        elif 'Image' in self.landmarks_frame.columns:
             # Assuming image data is stored directly in the dataframe as numpy array or list
            img_data = self.landmarks_frame.iloc[idx, self.landmarks_frame.columns.get_loc('Image')]
            if isinstance(img_data, (np.ndarray, list)):
                try:
                    # --- FIX: Assume 3 channels (H, W, C) if reshaping from flat array ---
                    expected_elements = self.image_size[0] * self.image_size[1] * 3
                    if np.array(img_data).size == expected_elements:
                        image = np.array(img_data).reshape(self.image_size[0], self.image_size[1], 3).astype(np.uint8)
                    # --- Fallback for grayscale (if needed, though unlikely given the error) ---
                    elif np.array(img_data).size == self.image_size[0] * self.image_size[1]:
                         image = np.array(img_data).reshape(self.image_size).astype(np.uint8)
                         print(f"Debug idx {idx}: Reshaped image from DataFrame as grayscale {image.shape}") # Debug print
                    else:
                         raise ValueError(f"Unexpected number of elements {np.array(img_data).size} for image size {self.image_size}")
                    # --- End Fix ---
                except ValueError as e:
                     print(f"Warning: Error reshaping image data at index {idx}: {e}")
                     return None # Return None or handle error appropriately
            else:
                print(f"Warning: Invalid image data format at index {idx}")
                return None
        else:
            raise KeyError("Dataset needs either 'image_path' and 'root_dir' or an 'Image' column.")
            
        # Handle grayscale images - convert to 3 channels for consistency
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.ndim == 3 and image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.ndim == 3 and image.shape[2] == 4:
            # Handle RGBA images (convert to RGB)
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif image.ndim != 3 or image.shape[2] != 3:
             print(f"Warning: Unexpected image shape {image.shape} at index {idx}. Attempting to process.")
             # Try to force into 3 channels if possible, otherwise error might occur later
             if image.ndim == 2: image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
             # Add more specific handling if needed

        # Ensure image is HWC format and uint8
        image = image.astype(np.uint8)
        original_h, original_w = image.shape[:2]
        print(f"Debug idx {idx}: Loaded image shape {image.shape}, type {image.dtype}") # Debug print

        # Apply CLAHE if requested
        if self.apply_clahe:
            # Create CLAHE object on demand
            clahe = cv2.createCLAHE(clipLimit=self.clahe_clip_limit, tileGridSize=self.clahe_grid_size)
            # Convert to LAB color space, apply CLAHE to L channel, convert back to RGB
            lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab_image)
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            image = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
            print(f"Debug idx {idx}: Image shape after CLAHE {image.shape}") # Debug print

        # Resize image
        print(f"Debug idx {idx}: Resizing image from {image.shape[:2]} to {self.image_size}") # Debug print
        image = transform.resize(image, self.image_size, anti_aliasing=True)
        # Convert image back to uint8 after resize (skimage converts to float)
        image = (image * 255).astype(np.uint8)
        print(f"Debug idx {idx}: Image shape after resize {image.shape}") # Debug print

        # Load landmarks
        landmarks = self.landmarks_frame.iloc[idx][self.landmark_cols].values
        landmarks = landmarks.astype('float').reshape(-1, 2)

        # Adjust landmarks for resize
        scale_y = self.image_size[0] / original_h
        scale_x = self.image_size[1] / original_w
        landmarks = landmarks * [scale_x, scale_y]

        # --- Predict and Add Depth Feature --- 
        depth_map = None
        if self.use_depth_features and self.depth_model is not None:
            try:
                # Prepare image for depth model (needs PIL format)
                pil_image = Image.fromarray(image) # Image is already resized uint8 RGB
                
                # Process image and predict depth
                with torch.no_grad():
                    inputs = self.depth_processor(images=pil_image, return_tensors="pt").to(self.depth_model.device)
                    outputs = self.depth_model(**inputs)
                
                # Post-process depth map
                post_processed = self.depth_processor.post_process_depth_estimation(
                    outputs, target_sizes=[self.image_size]
                )
                depth_map_tensor = post_processed[0]["predicted_depth"]
                print(f"Debug idx {idx}: Raw depth map shape {depth_map_tensor.shape}, device {depth_map_tensor.device}") # Debug print
                
                # Normalize depth map (e.g., min-max to 0-1)
                min_d = torch.min(depth_map_tensor)
                max_d = torch.max(depth_map_tensor)
                if max_d > min_d:
                    depth_map = (depth_map_tensor - min_d) / (max_d - min_d)
                else:
                    depth_map = torch.zeros_like(depth_map_tensor) # Handle case of flat depth
                
                # Add channel dimension and move to CPU if needed for concatenation
                depth_map = depth_map.unsqueeze(0).cpu() # Shape: (1, H, W)
                print(f"Debug idx {idx}: Processed depth map shape {depth_map.shape}") # Debug print

            except Exception as e:
                print(f"Warning: Error predicting depth for index {idx}: {e}. Skipping depth feature.")
                depth_map = None # Ensure depth_map is None if error occurs
        # --- End Depth Feature --- 

        sample = {'image': image, 'landmarks': landmarks}

        # Apply standard data augmentations if transform is provided
        if self.transform:
            sample = self.transform(sample)
            # After augmentation, sample['image'] might be tensor or numpy array
        
        # --- Convert image to tensor and concatenate depth --- 
        # Ensure image is a tensor (some transforms might return numpy)
        final_image = sample['image']
        if isinstance(final_image, np.ndarray):
            # Convert HWC numpy array to CHW tensor
            final_image = torch.from_numpy(final_image.transpose((2, 0, 1))).float() / 255.0
        elif not isinstance(final_image, torch.Tensor):
            raise TypeError(f"Unsupported image type after transform: {type(final_image)}")
        print(f"Debug idx {idx}: Image tensor shape before depth concat {final_image.shape}") # Debug print
            
        # If depth map exists, concatenate it
        if depth_map is not None and isinstance(final_image, torch.Tensor) and final_image.dim() == 3:
            if depth_map.shape[1:] == final_image.shape[1:]:
                # Concatenate along channel dimension (C, H, W)
                final_image = torch.cat((final_image, depth_map), dim=0) # Now has 4 channels
                print(f"Debug idx {idx}: Final image tensor shape after depth concat {final_image.shape}") # Debug print
            else:
                print(f"Warning: Depth map size {depth_map.shape[1:]} mismatch with image size {final_image.shape[1:]} at index {idx}. Skipping depth.")
        elif depth_map is not None:
             print(f"Debug idx {idx}: Depth map existed but was not concatenated (Image type: {type(final_image)}, Image dims: {final_image.dim()})") # Debug print

        sample['image'] = final_image
        # Note: Landmarks are kept as numpy array until ToTensor or similar in DataLoader transform
        # --- End Concatenate --- 
        
        # --- Save first sample for debugging ---
        if idx == 0:
            try:
                save_path = 'debug_sample.pt'
                torch.save({'image': sample['image'].clone(), # Clone to avoid issues if tensor is modified later
                           'landmarks': sample['landmarks'].copy()}, # Copy numpy array
                           save_path)
                print(f"Debug: Saved first processed sample (image tensor & landmarks) to {save_path}")
            except Exception as e:
                print(f"Debug: Failed to save debug sample: {e}")
        # --- End Save Sample ---

        return sample


# --- Standard Transforms (needed when depth is not used) ---
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # Convert image to tensor if it's a numpy array
        # Transpose image dimensions: (H x W x C) -> (C X H X W)
        if isinstance(image, np.ndarray):
            # Assuming image is HWC uint8
            image = image.transpose((2, 0, 1))
            image = torch.from_numpy(image).float() / 255.0
        elif isinstance(image, torch.Tensor):
            # If already a tensor, ensure correct shape and type (e.g., CHW, float)
            if image.dim() == 3 and image.shape[0] != 3: # Check if channel is first dim
                 if image.shape[2] == 3: # If HWC, permute
                     image = image.permute(2, 0, 1)
            if image.dtype != torch.float32:
                image = image.float() / 255.0 if image.max() > 1.0 else image.float()
        else:
            raise TypeError(f"ToTensor received unexpected type: {type(image)}")

        # Convert landmarks to tensor
        landmarks = torch.from_numpy(landmarks).float()

        return {'image': image, 'landmarks': landmarks}

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation."""
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        # Ensure mean and std are tensors for broadcasting
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        
        if not isinstance(image, torch.Tensor):
            raise TypeError("Normalize expects a torch Tensor image.")
        if image.shape[0] != 3:
            raise ValueError(f"Normalize expects a 3-channel image, but got {image.shape[0]} channels.")
            
        # Normalize image (ensure tensors are on the same device implicitly)
        image = (image - self.mean.to(image.device)) / self.std.to(image.device)
        
        return {'image': image, 'landmarks': landmarks}

# --- Custom Transforms for handling 4 channels --- 
class ToTensor4CH:
    """Convert ndarrays in sample to Tensors, handle 3 or 4 channel image."""
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # numpy image: H x W x C
        # torch image: C X H X W
        if isinstance(image, np.ndarray):
            image = image.transpose((2, 0, 1))
            image = torch.from_numpy(image).float() / 255.0 # Normalize uint8
        elif not isinstance(image, torch.Tensor):
             raise TypeError(f"ToTensor4CH expects numpy array or torch Tensor, got {type(image)}")
             
        # Convert landmarks to tensor
        landmarks = torch.from_numpy(landmarks).float()

        return {'image': image, 'landmarks': landmarks}

class Normalize4CH:
    """Normalize a tensor image with mean and standard deviation, handling 4 channels.
       Normalizes first 3 channels (RGB) with given stats, leaves 4th channel (depth) as is (or applies specific norm)."""
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), normalize_depth=False, depth_mean=0.5, depth_std=0.5):
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
        self.normalize_depth = normalize_depth
        self.depth_mean = depth_mean
        self.depth_std = depth_std

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        
        if not isinstance(image, torch.Tensor):
             raise TypeError("Normalize4CH expects a torch Tensor image.")

        # Normalize RGB channels
        image[:3, :, :] = (image[:3, :, :] - self.mean) / self.std
        
        # Normalize depth channel if requested and present
        if self.normalize_depth and image.shape[0] == 4:
             image[3, :, :] = (image[3, :, :] - self.depth_mean) / self.depth_std
        # If normalize_depth is False, the 4th channel (already scaled 0-1) is passed through.
        
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
        ToTensor4CH(),
        Normalize4CH()
    ])
    
    val_transform = transforms.Compose([
        ToTensor4CH(),
        Normalize4CH()
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