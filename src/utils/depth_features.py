import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import DepthProImageProcessorFast, DepthProForDepthEstimation

class DepthFeatureExtractor:
    """
    Extracts depth features from images using the DepthPro model
    """
    def __init__(self, cache_dir=None, model_name="apple/DepthPro-hf", device=None):
        """
        Initialize the depth feature extractor
        
        Args:
            cache_dir (str, optional): Directory to cache depth features
            model_name (str): The pretrained model name to use
            device (torch.device, optional): Device to run the model on
        """
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            
        # Set device
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"DepthFeatureExtractor using device: {self.device}")
        
        # Load model and processor
        try:
            self.image_processor = DepthProImageProcessorFast.from_pretrained(model_name)
            self.model = DepthProForDepthEstimation.from_pretrained(model_name).to(self.device)
            print(f"Successfully loaded {model_name} model")
        except Exception as e:
            print(f"Error loading depth model: {e}")
            print("Please install required packages: pip install transformers")
            raise
        
    def extract_depth(self, image):
        """
        Extract depth map from a single image
        
        Args:
            image (PIL.Image.Image or numpy.ndarray): Input image
            
        Returns:
            numpy.ndarray: Normalized depth map (values 0-255)
        """
        # Convert numpy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image.astype("uint8"))
        
        # Process image
        inputs = self.image_processor(images=image, return_tensors="pt").to(self.device)
        
        # Generate depth map
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process depth map
        post_processed_output = self.image_processor.post_process_depth_estimation(
            outputs, target_sizes=[(image.height, image.width)],
        )
        
        # Get depth map
        depth = post_processed_output[0]["predicted_depth"]
        
        # Normalize depth to 0-255 range
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        depth = depth * 255.
        depth = depth.detach().cpu().numpy()
        
        return depth
    
    def extract_batch_depth(self, images_batch):
        """
        Extract depth maps for a batch of images
        
        Args:
            images_batch (torch.Tensor): Batch of images (B, C, H, W)
            
        Returns:
            torch.Tensor: Batch of depth maps (B, 1, H, W)
        """
        batch_size = images_batch.shape[0]
        device_images = images_batch.to(self.device)
        
        # Denormalize images if they are in normalized format
        if device_images.max() <= 1.0:
            # Assuming images are normalized using ImageNet stats
            mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
            device_images = device_images * std + mean
            device_images = torch.clamp(device_images, 0, 1)
            
        # Process images
        depth_maps = []
        for i in range(batch_size):
            # Convert tensor to PIL image
            img = device_images[i].permute(1, 2, 0).cpu().numpy()
            img = (img * 255).astype(np.uint8)
            pil_img = Image.fromarray(img)
            
            # Extract depth map
            depth = self.extract_depth(pil_img)
            depth_maps.append(torch.from_numpy(depth).float())
        
        # Stack depth maps and add channel dimension
        depth_tensor = torch.stack(depth_maps).unsqueeze(1) / 255.0  # Normalize to [0,1]
        return depth_tensor
    
    def process_and_cache_dataset(self, dataset, cache_suffix=""):
        """
        Process all images in a dataset and cache the depth features
        
        Args:
            dataset: Dataset containing images
            cache_suffix (str): Suffix for cache file names to differentiate train/val/test
            
        Returns:
            dict: Mapping of image indices to depth features
        """
        if not self.cache_dir:
            raise ValueError("Cache directory must be set to use caching functionality")
        
        cache_file = os.path.join(self.cache_dir, f"depth_features_{cache_suffix}.pt")
        
        # Check if cache file exists
        if os.path.exists(cache_file):
            print(f"Loading cached depth features from {cache_file}")
            return torch.load(cache_file)
        
        print(f"Generating depth features for {len(dataset)} images...")
        depth_features = {}
        
        for idx in range(len(dataset)):
            sample = dataset[idx]
            image = sample['image']
            
            # Convert to numpy array if it's a tensor
            if isinstance(image, torch.Tensor):
                image = image.permute(1, 2, 0).numpy()
            
            # Extract depth map
            depth = self.extract_depth(image)
            depth_features[idx] = torch.from_numpy(depth).float()
            
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{len(dataset)} images")
                
        # Save to cache file
        torch.save(depth_features, cache_file)
        print(f"Saved depth features to {cache_file}")
        
        return depth_features
    
    def save_depth_visualization(self, image, save_path):
        """
        Generate and save a visualization of the depth map
        
        Args:
            image (PIL.Image or numpy.ndarray): Input image
            save_path (str): Path to save the visualization
        """
        depth = self.extract_depth(image)
        
        plt.figure(figsize=(12, 5))
        
        # Display original image
        plt.subplot(1, 2, 1)
        if isinstance(image, np.ndarray):
            plt.imshow(image)
        else:
            plt.imshow(image)
        plt.title("Original Image")
        
        # Display depth map
        plt.subplot(1, 2, 2)
        plt.imshow(depth, cmap='plasma')
        plt.title("Depth Map")
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        print(f"Saved depth visualization to {save_path}")