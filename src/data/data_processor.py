import os
import pandas as pd
import numpy as np
from tqdm import tqdm # Add tqdm for progress bar
import torch # Add torch
from PIL import Image # Add PIL

from .dataset import create_dataloaders
from .patient_classifier import PatientClassifier

# Try importing transformers for depth prediction
try:
    from transformers import DepthProImageProcessorFast, DepthProForDepthEstimation
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: Transformers library not found. Depth feature generation will be unavailable.")
    print("Install it with: pip install transformers accelerate")

class DataProcessor:
    def __init__(self, data_path, landmark_cols=None, image_size=(224, 224), apply_clahe=True):
        """
        Initialize the data processor
        
        Args:
            data_path (str): Path to the CSV file or directory containing the data
            landmark_cols (list): List of column names containing landmark coordinates
            image_size (tuple): Size of images (height, width)
            apply_clahe (bool): Whether to apply CLAHE for histogram equalization
        """
        self.data_path = data_path
        self.landmark_cols = landmark_cols
        self.image_size = image_size
        self.apply_clahe = apply_clahe
        self.df = None
        
        # Create classifier if landmark columns are provided
        if landmark_cols:
            self.classifier = PatientClassifier(landmark_cols)
        else:
            self.classifier = None
        
        # Device for depth model inference
        self.inference_device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"DataProcessor using device: {self.inference_device} for potential depth inference.")
        self.depth_model = None
        self.depth_processor = None
    
    def _load_depth_model(self):
        """Loads the DepthPro model and processor if not already loaded."""
        if not TRANSFORMERS_AVAILABLE:
            print("Error: Transformers library not available. Cannot load depth model.")
            return False
        if self.depth_model is None or self.depth_processor is None:
            try:
                print("Loading DepthPro model (apple/DepthPro-hf)... This might take a moment.")
                self.depth_processor = DepthProImageProcessorFast.from_pretrained("apple/DepthPro-hf")
                self.depth_model = DepthProForDepthEstimation.from_pretrained("apple/DepthPro-hf").to(self.inference_device)
                print("DepthPro model loaded successfully.")
                return True
            except Exception as e:
                print(f"Error loading DepthPro model: {e}")
                self.depth_model = None
                self.depth_processor = None
                return False
        return True

    def _predict_depth(self, image_array):
        """Predicts depth for a single image array (H, W, C)."""
        if self.depth_model is None or self.depth_processor is None:
            if not self._load_depth_model():
                return None # Return None if model couldn't be loaded
                
        try:
            # Convert NumPy array (H, W, C) to PIL Image
            # Ensure it's uint8
            if image_array.dtype != np.uint8:
                 image_array = image_array.astype(np.uint8)
            image_pil = Image.fromarray(image_array)
            
            # Prepare inputs
            inputs = self.depth_processor(images=image_pil, return_tensors="pt").to(self.inference_device)

            # Predict depth
            with torch.no_grad():
                outputs = self.depth_model(**inputs)

            # Post-process
            post_processed_output = self.depth_processor.post_process_depth_estimation(
                outputs, target_sizes=[(image_pil.height, image_pil.width)],
            )
            
            # Extract, normalize (0-1), and convert depth map
            depth_map = post_processed_output[0]["predicted_depth"]
            depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-6) # Normalize 0-1
            depth_map = depth_map.squeeze().detach().cpu().numpy() # (H, W)
            
            return depth_map.astype(np.float32) # Store as float32

        except Exception as e:
            print(f"Error during depth prediction: {e}")
            return None

    def add_depth_features(self, df):
        """Generates and adds depth maps to the DataFrame."""
        if 'Image' not in df.columns:
            print("Error: DataFrame must contain an 'Image' column with image arrays.")
            return df
            
        if not self._load_depth_model(): # Ensure model is loaded
             print("Skipping depth feature generation due to model loading error.")
             return df

        print("Generating depth features for dataset... This may take a while.")
        depth_maps = []
        saved_depth_count = 0
        max_depth_saves = 2 # Number of depth maps to save for preview
        depth_save_dir = "./outputs/depth_previews"
        os.makedirs(depth_save_dir, exist_ok=True) # Create directory if it doesn't exist
        
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Generating Depth Maps"):
            image_data = row['Image']
            image_array = None # Initialize image_array for this iteration
            
            # --- Start: Image Data Conversion --- 
            try:
                if isinstance(image_data, list):
                    # --- Debugging Start ---
                    # print(f"\n[Debug Index {index}] Detected list input.")
                    try:
                        np_array_flat = np.array(image_data) # Convert list to numpy array first
                    except Exception as e:
                         # print(f"[Debug Index {index}] Error during np.array(list): {e}")
                         raise TypeError("Failed to convert list to np array")
                         
                    list_len = np_array_flat.size
                    dtype = np_array_flat.dtype
                    expected_len_gray = self.image_size[0] * self.image_size[1]
                    expected_len_rgb = expected_len_gray * 3
                    # print(f"[Debug Index {index}] List Length: {list_len}, Dtype: {dtype}")
                    # print(f"[Debug Index {index}] Expected Gray Length: {expected_len_gray}, Expected RGB Length: {expected_len_rgb}")
                    # --- Debugging End ---
                    
                    if list_len == expected_len_gray:
                        # print(f"[Debug Index {index}] Matched Gray Length. Reshaping to {self.image_size}.")
                        image_array = np_array_flat.reshape(self.image_size) # Reshape grayscale
                    elif list_len == expected_len_rgb:
                        # print(f"[Debug Index {index}] Matched RGB Length. Reshaping to {(self.image_size[0], self.image_size[1], 3)}.")
                        image_array = np_array_flat.reshape((self.image_size[0], self.image_size[1], 3)) # Reshape RGB
                    else:
                        raise ValueError(f"List length {list_len} does not match expected grayscale ({expected_len_gray}) or RGB ({expected_len_rgb})")
                
                elif isinstance(image_data, np.ndarray):
                    # print(f"\n[Debug Index {index}] Detected numpy array input. Shape: {image_data.shape}, Dtype: {image_data.dtype}")
                    image_array = image_data # Already a numpy array
                
                else:
                    raise TypeError(f"Unexpected image data type: {type(image_data)}")
                    
            except (ValueError, TypeError) as e:
                print(f"Warning: Skipping depth prediction for row index {index} due to image data conversion error: {e}")
                depth_maps.append(None)
                continue # Skip to next row
            # --- End: Image Data Conversion ---
            
            # --- Start: Image Preparation for Depth Model ---
            try: 
                # Ensure image is in HWC uint8 format
                if image_array.ndim == 2: # Grayscale -> HWC
                     image_array_hwc = np.stack([image_array]*3, axis=-1)
                elif image_array.ndim == 3 and image_array.shape[0] == 3: # CHW -> HWC
                     image_array_hwc = image_array.transpose(1, 2, 0)
                elif image_array.ndim == 3 and image_array.shape[-1] == 1: # HW1 -> HWC
                    image_array_hwc = np.concatenate([image_array]*3, axis=-1)
                elif image_array.ndim == 3 and image_array.shape[-1] == 3: # Already HWC
                    image_array_hwc = image_array
                else:
                    raise ValueError(f"Unexpected image shape {image_array.shape} after conversion")
                
                # Ensure uint8, scale if necessary (assuming input range 0-1 or 0-255)
                if image_array_hwc.dtype != np.uint8:
                    if image_array_hwc.max() <= 1.0 and image_array_hwc.min() >= 0.0:
                        image_array_hwc = (image_array_hwc * 255).astype(np.uint8)
                    elif image_array_hwc.max() > 1.0: # Assume already 0-255 range if max > 1
                         image_array_hwc = image_array_hwc.astype(np.uint8)
                    # Handle other cases or raise error if range is unexpected
                    else:
                        raise ValueError(f"Cannot reliably convert dtype {image_array_hwc.dtype} with range [{image_array_hwc.min()}, {image_array_hwc.max()}] to uint8")
                
                # --- Pre-Prediction Validation ---
                if not isinstance(image_array_hwc, np.ndarray):
                     raise TypeError(f"image_array_hwc is not a numpy array (type: {type(image_array_hwc)})")
                if image_array_hwc.shape != (self.image_size[0], self.image_size[1], 3):
                    raise ValueError(f"image_array_hwc has incorrect shape {image_array_hwc.shape}. Expected {(self.image_size[0], self.image_size[1], 3)}")
                if image_array_hwc.dtype != np.uint8:
                    raise ValueError(f"image_array_hwc has incorrect dtype {image_array_hwc.dtype}. Expected np.uint8")
                # --- End Pre-Prediction Validation ---
                
            except (ValueError, TypeError) as e:
                 print(f"Warning: Skipping depth prediction for row index {index} due to image preparation error: {e}")
                 depth_maps.append(None)
                 continue
            # --- End: Image Preparation ---

            # Predict depth    
            depth_map = self._predict_depth(image_array_hwc)
            depth_maps.append(depth_map)
            
            # Save preview if prediction was successful and limit not reached
            if depth_map is not None and saved_depth_count < max_depth_saves:
                try:
                    # Save Depth Map
                    depth_img_array = (depth_map * 255).astype(np.uint8) # Scale 0-1 float to 0-255 uint8
                    depth_pil = Image.fromarray(depth_img_array)
                    depth_save_path = os.path.join(depth_save_dir, f"depth_preview_idx_{index}.png")
                    depth_pil.save(depth_save_path)
                    
                    # Save Corresponding Input Image
                    input_img_pil = Image.fromarray(image_array_hwc) # Use the prepared HWC uint8 image
                    input_save_path = os.path.join(depth_save_dir, f"input_preview_idx_{index}.png")
                    input_img_pil.save(input_save_path)
                    
                    print(f"Saved depth map preview to: {depth_save_path}")
                    print(f"Saved corresponding input preview to: {input_save_path}")
                    saved_depth_count += 1
                except Exception as e:
                     print(f"Warning: Could not save depth/input preview for index {index}: {e}")
        
        df['depth_map'] = depth_maps
        # Optional: Drop rows where depth prediction failed
        initial_len = len(df)
        df = df.dropna(subset=['depth_map'])
        if len(df) < initial_len:
            print(f"Warning: Dropped {initial_len - len(df)} rows due to depth prediction errors.")
            
        print("Depth features generated and added.")
        return df

    def load_data(self):
        """
        Load the dataset from a CSV file or pandas pickle file
        
        Returns:
            pandas.DataFrame: The loaded dataset
        """
        if self.data_path.endswith('.csv'):
            self.df = pd.read_csv(self.data_path)
        elif self.data_path.endswith('.pkl') or self.data_path.endswith('.pickle'):
            self.df = pd.read_pickle(self.data_path)
        else:
            raise ValueError(f"Unsupported file format: {self.data_path}")
        
        print(f"Loaded dataset with {len(self.df)} records and {len(self.df.columns)} columns")
        return self.df
    
    def preprocess_data(self, balance_classes=False, add_depth=False):
        """
        Preprocess the loaded dataset
        
        Args:
            balance_classes (bool): Whether to balance classes based on skeletal classification
            add_depth (bool): Whether to generate and add depth features.
            
        Returns:
            pandas.DataFrame: The preprocessed dataset
        """
        if self.df is None:
            self.load_data()
        
        # Check for missing values in landmark columns
        if self.landmark_cols:
            missing_landmarks = self.df[self.landmark_cols].isnull().sum().sum()
            if missing_landmarks > 0:
                print(f"Warning: Found {missing_landmarks} missing values in landmark columns")
                
                # Fill missing values (optional, depending on your strategy)
                # For demonstration, we'll drop rows with missing landmarks
                self.df = self.df.dropna(subset=self.landmark_cols)
                print(f"Dropped rows with missing landmarks. Remaining records: {len(self.df)}")
        
        # Validate image data
        if 'Image' in self.df.columns:
            # Check that all images have the expected format
            valid_images = []
            for idx, row in self.df.iterrows():
                img_data = row['Image']
                if isinstance(img_data, list) and len(img_data) == self.image_size[0] * self.image_size[1]:
                    valid_images.append(True)
                elif isinstance(img_data, np.ndarray) and img_data.shape[0] * img_data.shape[1] == self.image_size[0] * self.image_size[1]:
                    valid_images.append(True)
                else:
                    valid_images.append(False)
            
            invalid_count = valid_images.count(False)
            if invalid_count > 0:
                print(f"Warning: Found {invalid_count} records with invalid image data")
                
                # Keep only valid images
                self.df = self.df[valid_images]
                print(f"Removed invalid images. Remaining records: {len(self.df)}")
        
        # Generate depth features if requested
        if add_depth:
            self.df = self.add_depth_features(self.df)
        
        # Compute patient classes and balance if requested (AFTER potentially adding depth)
        if balance_classes and self.classifier is not None:
            print("Computing skeletal classifications for patients...")
            self.df = self.classifier.classify_patients(self.df)
            
            print("Balancing dataset by upsampling minority classes...")
            self.df = self.classifier.balance_classes(
                self.df, 
                class_column='skeletal_class', 
                balance_method='upsample'
            )
        
        return self.df
    
    def create_data_loaders(self, batch_size=32, train_ratio=0.8, val_ratio=0.1, num_workers=4, root_dir=None, balance_classes=False):
        """
        Create data loaders for training, validation, and testing
        
        Args:
            batch_size (int): Batch size for the data loaders
            train_ratio (float): Ratio of data to use for training (if 'set' column is not present)
            val_ratio (float): Ratio of data to use for validation (if 'set' column is not present)
            num_workers (int): Number of worker threads for data loading
            root_dir (str): Directory containing image files (if images are stored as files)
            balance_classes (bool): Whether to balance classes based on skeletal classification
            
        Returns:
            tuple: (train_loader, val_loader, test_loader)
        """
        if self.df is None or balance_classes:  # Re-preprocess if balancing requested
            self.preprocess_data(balance_classes=balance_classes)
        
        # Create data loaders
        train_loader, val_loader, test_loader = create_dataloaders(
            df=self.df,
            landmark_cols=self.landmark_cols,
            batch_size=batch_size,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            apply_clahe=self.apply_clahe,
            root_dir=root_dir,
            num_workers=num_workers
        )
        
        return train_loader, val_loader, test_loader
    
    def compute_patient_classes(self):
        """
        Compute skeletal class for each patient based on ANB angle
        
        Returns:
            pandas.DataFrame: DataFrame with added skeletal classification
        """
        if self.df is None:
            self.load_data()
        
        if self.classifier is None:
            if self.landmark_cols:
                self.classifier = PatientClassifier(self.landmark_cols)
            else:
                raise ValueError("Landmark columns must be provided to compute patient classes")
        
        self.df = self.classifier.classify_patients(self.df)
        return self.df
    
    def balance_dataset(self, method='upsample', class_column='skeletal_class'):
        """
        Balance the dataset based on a specified class column
        
        Args:
            method (str): Method to balance the dataset ('upsample' or 'downsample')
            class_column (str): Column to balance by
            
        Returns:
            pandas.DataFrame: Balanced DataFrame
        """
        if self.df is None:
            self.load_data()
        
        if class_column not in self.df.columns and class_column == 'skeletal_class':
            print("Computing skeletal classifications before balancing...")
            self.compute_patient_classes()
            
        if self.classifier is None:
            raise ValueError("Patient classifier is required for dataset balancing")
        
        self.df = self.classifier.balance_classes(
            self.df,
            class_column=class_column, 
            balance_method=method
        )
        
        return self.df
    
    def get_data_stats(self):
        """
        Get statistics about the dataset
        
        Returns:
            dict: Dictionary containing dataset statistics
        """
        if self.df is None:
            self.load_data()
        
        stats = {
            'total_samples': len(self.df),
            'columns': list(self.df.columns)
        }
        
        # Count samples by set if available
        if 'set' in self.df.columns:
            set_counts = self.df['set'].value_counts().to_dict()
            stats['set_counts'] = set_counts
        
        # Count samples by class if available
        if 'class' in self.df.columns:
            class_counts = self.df['class'].value_counts().to_dict()
            stats['class_counts'] = class_counts
        
        # Count samples by skeletal class if available
        if 'skeletal_class' in self.df.columns:
            skeletal_class_counts = self.df['skeletal_class'].value_counts().to_dict()
            stats['skeletal_class_counts'] = skeletal_class_counts
            
            # Add skeletal class names for clarity
            class_names = {
                1: "Class I (Normal)",
                2: "Class II (Prognathic maxilla)",
                3: "Class III (Retrognathic maxilla)"
            }
            
            stats['skeletal_class_names'] = {
                k: class_names.get(k, f"Unknown ({k})") 
                for k in skeletal_class_counts.keys()
            }
        
        # Get landmark statistics if available
        if self.landmark_cols:
            landmark_stats = {}
            for col in self.landmark_cols:
                landmark_stats[col] = {
                    'mean': self.df[col].mean(),
                    'std': self.df[col].std(),
                    'min': self.df[col].min(),
                    'max': self.df[col].max()
                }
            stats['landmark_stats'] = landmark_stats
        
        return stats 