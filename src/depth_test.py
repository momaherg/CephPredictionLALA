import os
import argparse
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import requests
from io import BytesIO

from utils.depth_features import DepthFeatureExtractor

def parse_args():
    parser = argparse.ArgumentParser(description='Test depth feature extraction')
    parser.add_argument('--output_dir', type=str, default='./outputs/depth_test', 
                        help='Directory to save outputs')
    parser.add_argument('--use_sample_images', action='store_true', 
                        help='Use sample images from URLs')
    parser.add_argument('--image_dir', type=str, default=None,
                        help='Directory with test images')
    parser.add_argument('--use_mps', action='store_true', 
                        help='Use MPS device on Mac with Apple Silicon')
    return parser.parse_args()

def load_sample_images():
    """Load sample images from URLs"""
    urls = [
        'https://huggingface.co/datasets/mishig/sample_images/resolve/main/tiger.jpg',
        'https://github.com/pytorch/ios-demo-app/raw/master/PyTorchDemo/PyTorchDemoApp/images/dog.jpg'
    ]
    
    images = []
    for url in urls:
        try:
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                image = Image.open(BytesIO(response.content))
                images.append(image)
                print(f"Successfully loaded image from {url}")
            else:
                print(f"Failed to load image from {url}")
        except Exception as e:
            print(f"Error loading image from {url}: {e}")
    
    return images

def load_images_from_dir(image_dir):
    """Load images from a directory"""
    images = []
    for file_name in os.listdir(image_dir):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_dir, file_name)
            try:
                image = Image.open(image_path)
                images.append(image)
                print(f"Loaded image: {file_name}")
            except Exception as e:
                print(f"Error loading {file_name}: {e}")
    
    return images

def main():
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    if args.use_mps and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS device")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using CUDA device")
    else:
        device = torch.device('cpu')
        print("Using CPU device")
    
    # Initialize depth feature extractor
    depth_extractor = DepthFeatureExtractor(
        cache_dir=args.output_dir,
        device=device
    )
    
    # Load images
    if args.use_sample_images:
        images = load_sample_images()
    elif args.image_dir:
        images = load_images_from_dir(args.image_dir)
    else:
        print("No images specified. Use --use_sample_images or --image_dir")
        return
    
    # Process each image and save visualization
    for i, image in enumerate(images):
        try:
            print(f"Processing image {i+1}/{len(images)}")
            save_path = os.path.join(args.output_dir, f'depth_visualization_{i+1}.png')
            depth_extractor.save_depth_visualization(image, save_path)
        except Exception as e:
            print(f"Error processing image {i+1}: {e}")
    
    print(f"Depth visualizations saved to {args.output_dir}")

if __name__ == '__main__':
    main() 