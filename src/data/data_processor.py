import os
import pandas as pd
import numpy as np
from .dataset import create_dataloaders
from .patient_classifier import PatientClassifier
import torch
from PIL import Image
import cv2
import io
import matplotlib.pyplot as plt
from tqdm import tqdm

class DataProcessor:
    def __init__(self, data_path, landmark_cols=None, image_size=(224, 224), apply_clahe=True, generate_depth=False):
        """
        Initialize the data processor
        
        Args:
            data_path (str): Path to the CSV file or directory containing the data
            landmark_cols (list): List of column names containing landmark coordinates
            image_size (tuple): Size of images (height, width)
            apply_clahe (bool): Whether to apply CLAHE for histogram equalization
            generate_depth (bool): Whether to generate depth features from images
        """
        self.data_path = data_path
        self.landmark_cols = landmark_cols
        self.image_size = image_size
        self.apply_clahe = apply_clahe
        self.generate_depth = generate_depth
        self.df = None
        self.depth_model = None
        self.depth_processor = None
        
        # Create classifier if landmark columns are provided
        if landmark_cols:
            self.classifier = PatientClassifier(landmark_cols)
        else:
            self.classifier = None
    
    def _initialize_depth_model(self):
        """
        Initialize the depth prediction model
        """
        if self.depth_model is None and self.generate_depth:
            try:
                print("\nInitializing depth prediction model...")
                try:
                    from transformers import DepthProImageProcessorFast, DepthProForDepthEstimation
                except ImportError:
                    print("Error: transformers package not found. Installing required packages...")
                    import subprocess
                    try:
                        subprocess.check_call(["pip", "install", "transformers"])
                        print("Successfully installed transformers package. Continuing with initialization...")
                        from transformers import DepthProImageProcessorFast, DepthProForDepthEstimation
                    except Exception as e:
                        print(f"Failed to install transformers package: {str(e)}")
                        print("Please manually install the required packages with:")
                        print("pip install transformers")
                        self.generate_depth = False
                        return
                
                # Check for torch
                try:
                    import torch
                    print(f"Using PyTorch version: {torch.__version__}")
                except ImportError:
                    print("Error: PyTorch not found. Please install PyTorch.")
                    self.generate_depth = False
                    return
                
                # Determine device
                device = torch.device("cuda" if torch.cuda.is_available() else 
                                     ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"))
                print(f"Using device: {device}")
                
                # Print debug info about available GPU
                if device.type == "cuda":
                    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
                    print(f"Available memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
                elif device.type == "mps":
                    print("Using Apple Metal Performance Shaders (MPS) backend")
                
                print("Downloading depth prediction model from apple/DepthPro-hf...")
                try:
                    # First try to load the processor
                    self.depth_processor = DepthProImageProcessorFast.from_pretrained("apple/DepthPro-hf")
                    print("✓ Downloaded processor successfully")
                except Exception as e:
                    print(f"Error downloading processor: {str(e)}")
                    self.generate_depth = False
                    return
                
                try:
                    # Then load the model
                    self.depth_model = DepthProForDepthEstimation.from_pretrained("apple/DepthPro-hf").to(device)
                    print("✓ Downloaded model successfully")
                except Exception as e:
                    print(f"Error downloading model: {str(e)}")
                    self.generate_depth = False
                    return
                
                self.device = device
                print("Depth prediction model initialized successfully.")
                
                # Test the model with a small random image
                try:
                    print("Testing depth model with a sample image...")
                    sample_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                    sample_pil = Image.fromarray(sample_img)
                    
                    # Process image for depth prediction
                    inputs = self.depth_processor(images=sample_pil, return_tensors="pt").to(device)
                    
                    # Predict depth
                    with torch.no_grad():
                        outputs = self.depth_model(**inputs)
                    
                    # Post-process depth prediction
                    post_processed_output = self.depth_processor.post_process_depth_estimation(
                        outputs, target_sizes=[(sample_pil.height, sample_pil.width)],
                    )
                    
                    # Get depth map
                    depth = post_processed_output[0]["predicted_depth"]
                    print("✓ Model test successful")
                except Exception as e:
                    print(f"Error testing model: {str(e)}")
                    print("Model initialization completed, but model test failed. Proceeding anyway.")
            except ImportError as e:
                print(f"Error importing required packages: {str(e)}")
                print("Please install required packages with:")
                print("pip install transformers torch")
                self.generate_depth = False
            except Exception as e:
                print(f"Error initializing depth prediction model: {str(e)}")
                self.generate_depth = False
    
    def _predict_depth(self, image):
        """
        Predict depth from an image
        
        Args:
            image: Input image (PIL Image or numpy array)
            
        Returns:
            numpy.ndarray: Normalized depth map
        """
        if not self.generate_depth or self.depth_model is None:
            return None
        
        try:
            # Convert numpy array to PIL Image if needed
            if isinstance(image, np.ndarray):
                # Ensure image is uint8 for PIL
                if image.dtype != np.uint8:
                    if image.max() <= 1.0:
                        image = (image * 255).astype(np.uint8)
                    else:
                        image = image.astype(np.uint8)
                pil_img = Image.fromarray(image)
            else:
                pil_img = image
            
            # Process image for depth prediction
            inputs = self.depth_processor(images=pil_img, return_tensors="pt").to(self.device)
            
            # Predict depth
            with torch.no_grad():
                outputs = self.depth_model(**inputs)
            
            # Post-process depth prediction
            post_processed_output = self.depth_processor.post_process_depth_estimation(
                outputs, target_sizes=[(pil_img.height, pil_img.width)],
            )
            
            # Get depth map
            depth = post_processed_output[0]["predicted_depth"]
            
            # Normalize depth map to [0, 1]
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
            
            # Convert to numpy array
            depth_np = depth.detach().cpu().numpy()
            
            # Resize depth map to match image size if needed
            if depth_np.shape != self.image_size:
                depth_np = cv2.resize(depth_np, (self.image_size[1], self.image_size[0]))
            
            return depth_np
        except Exception as e:
            print(f"Error predicting depth: {str(e)}")
            return None
    
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
    
    def preprocess_data(self, balance_classes=False):
        """
        Preprocess the loaded dataset
        
        Args:
            balance_classes (bool): Whether to balance classes based on skeletal classification
            
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
        if self.generate_depth:
            self._initialize_depth_model()
            
            if self.depth_model is not None:
                print("Generating depth features for all images...")
                
                # Check if depth features already exist
                if 'depth_feature' in self.df.columns:
                    print("Depth features already exist in the dataset. Skipping depth generation.")
                else:
                    # Initialize depth feature column
                    self.df['depth_feature'] = None
                    
                    # Process images
                    if 'Image' in self.df.columns:
                        # Process in-memory images
                        image_types_found = {}
                        depth_features_generated = 0
                        
                        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Generating depth features"):
                            img_data = row['Image']
                            
                            # Track types of image data for debugging
                            data_type = type(img_data).__name__
                            if data_type not in image_types_found:
                                image_types_found[data_type] = 1
                            else:
                                image_types_found[data_type] += 1
                            
                            # Handle different image formats
                            try:
                                if isinstance(img_data, np.ndarray):
                                    # Direct numpy array
                                    image_to_process = img_data
                                elif isinstance(img_data, list):
                                    # Convert list to numpy array
                                    if all(isinstance(p, list) for p in img_data):
                                        # Nested list structure
                                        image_to_process = np.array(img_data)
                                    else:
                                        # Flat list - try to reshape based on expected dimensions
                                        image_to_process = np.array(img_data).reshape(self.image_size[0], self.image_size[1], -1)
                                elif hasattr(img_data, 'numpy'):
                                    # PyTorch tensor or similar
                                    image_to_process = img_data.numpy()
                                else:
                                    # Unknown format - skip
                                    continue
                                
                                # Make sure image has correct dimensions (H, W, C)
                                if len(image_to_process.shape) == 2:
                                    # Single channel image - add channel dimension
                                    image_to_process = np.expand_dims(image_to_process, axis=2)
                                    # Convert to 3 channels if needed
                                    image_to_process = np.repeat(image_to_process, 3, axis=2)
                                elif image_to_process.shape[2] == 1:
                                    # Single channel image - convert to 3 channels
                                    image_to_process = np.repeat(image_to_process, 3, axis=2)
                                
                                # Ensure image shape is as expected
                                if image_to_process.shape[0] != self.image_size[0] or image_to_process.shape[1] != self.image_size[1]:
                                    # Resize to expected dimensions
                                    image_to_process = cv2.resize(image_to_process, (self.image_size[1], self.image_size[0]))
                                
                                # Actually predict depth
                                depth_map = self._predict_depth(image_to_process)
                                if depth_map is not None:
                                    self.df.at[idx, 'depth_feature'] = depth_map
                                    depth_features_generated += 1
                                    
                                    # Log progress occasionally
                                    if depth_features_generated % 10 == 0:
                                        print(f"Generated {depth_features_generated} depth features so far...")
                            
                            except Exception as e:
                                print(f"Error processing image at index {idx}: {str(e)}")
                        
                        # Print debug information
                        print("\nImage data types found in dataset:")
                        for data_type, count in image_types_found.items():
                            print(f"  {data_type}: {count} images")
                        
                    elif 'image_path' in self.df.columns:
                        # Process images from file paths
                        depth_features_generated = 0
                        
                        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Generating depth features"):
                            img_path = row['image_path']
                            try:
                                img = Image.open(img_path)
                                depth_map = self._predict_depth(img)
                                if depth_map is not None:
                                    self.df.at[idx, 'depth_feature'] = depth_map
                                    depth_features_generated += 1
                                    
                                    # Log progress occasionally
                                    if depth_features_generated % 10 == 0:
                                        print(f"Generated {depth_features_generated} depth features so far...")
                            except Exception as e:
                                print(f"Error processing image {img_path}: {str(e)}")
                    
                    # Check how many depth features were generated
                    depth_count = self.df['depth_feature'].notna().sum()
                    print(f"Generated depth features for {depth_count}/{len(self.df)} images.")
                    
                    # Save preprocessed data with depth features
                    if depth_count > 0:
                        output_path = self.data_path.replace('.csv', '_with_depth.pkl').replace('.pkl', '_with_depth.pkl')
                        self.df.to_pickle(output_path)
                        print(f"Saved preprocessed data with depth features to {output_path}")
                    else:
                        print("\nNo depth features were generated. Please check the following:")
                        print("1. Make sure the transformers package is installed: pip install transformers")
                        print("2. Verify that you have internet access to download the model")
                        print("3. Check that your images are in a supported format")
                        print("4. Try running with a small test dataset to debug")
        
        # Compute patient classes and balance if requested
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
    
    def create_data_loaders(self, batch_size=32, train_ratio=0.8, val_ratio=0.1, num_workers=4, root_dir=None, balance_classes=False, use_depth=False):
        """
        Create data loaders for training, validation, and testing
        
        Args:
            batch_size (int): Batch size for the data loaders
            train_ratio (float): Ratio of data to use for training (if 'set' column is not present)
            val_ratio (float): Ratio of data to use for validation (if 'set' column is not present)
            num_workers (int): Number of worker threads for data loading
            root_dir (str): Directory containing image files (if images are stored as files)
            balance_classes (bool): Whether to balance classes based on skeletal classification
            use_depth (bool): Whether to include depth features in the sample
            
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
            num_workers=num_workers,
            use_depth=use_depth
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
        
        # Get depth feature statistics if available
        if 'depth_feature' in self.df.columns:
            depth_count = self.df['depth_feature'].notna().sum()
            stats['depth_feature_count'] = depth_count
            
            if depth_count > 0:
                # Get sample depth map
                sample_depth = next(item for item in self.df['depth_feature'] if item is not None)
                if isinstance(sample_depth, np.ndarray):
                    stats['depth_feature_shape'] = sample_depth.shape
                    stats['depth_feature_min'] = float(sample_depth.min())
                    stats['depth_feature_max'] = float(sample_depth.max())
                    stats['depth_feature_mean'] = float(sample_depth.mean())
        
        return stats 