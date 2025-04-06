#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to preprocess a cephalometric dataset and generate depth features.
This should be run once before training to generate depth features, which are saved for later use.
"""

import os
import argparse
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add the src directory to the path for imports
import sys
sys.path.append('./src')

from src.data.data_processor import DataProcessor
from src.data.dataset import CephalometricDataset

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess cephalometric dataset with depth features')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset CSV file')
    parser.add_argument('--output_dir', type=str, default='./data', help='Directory to save the processed dataset')
    parser.add_argument('--apply_clahe', action='store_true', help='Apply CLAHE for contrast enhancement')
    parser.add_argument('--balance_classes', action='store_true', help='Balance skeletal classes')
    parser.add_argument('--balance_method', type=str, default='upsample', choices=['upsample', 'downsample'], 
                        help='Method to balance classes')
    parser.add_argument('--visualize', action='store_true', help='Visualize sample depth prediction')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Define landmark columns - you might need to adjust these based on your dataset
    landmark_cols = [
        'sella_x', 'sella_y', 'nasion_x', 'nasion_y', 'A point_x', 'A point_y',
        'B point_x', 'B point_y', 'upper 1 tip_x', 'upper 1 tip_y',
        'upper 1 apex_x', 'upper 1 apex_y', 'lower 1 tip_x', 'lower 1 tip_y',
        'lower 1 apex_x', 'lower 1 apex_y', 'ANS_x', 'ANS_y', 'PNS_x', 'PNS_y',
        'Gonion _x', 'Gonion _y', 'Menton_x', 'Menton_y', 'ST Nasion_x',
        'ST Nasion_y', 'Tip of the nose_x', 'Tip of the nose_y', 'Subnasal_x',
        'Subnasal_y', 'Upper lip_x', 'Upper lip_y', 'Lower lip_x',
        'Lower lip_y', 'ST Pogonion_x', 'ST Pogonion_y', 'gnathion_x',
        'gnathion_y'
    ]
    
    print(f"Processing dataset: {args.data_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create data processor with depth feature generation enabled
    data_processor = DataProcessor(
        data_path=args.data_path,
        landmark_cols=landmark_cols,
        image_size=(224, 224),
        apply_clahe=args.apply_clahe,
        generate_depth=True  # Always generate depth features in this script
    )
    
    # Load and preprocess data
    print("Loading dataset...")
    df = data_processor.load_data()
    
    print("Preprocessing dataset with depth features...")
    df = data_processor.preprocess_data(balance_classes=args.balance_classes)
    
    # Print data statistics
    data_stats = data_processor.get_data_stats()
    print(f"\nDataset statistics:")
    print(f"  Total samples: {data_stats['total_samples']}")
    
    if 'depth_feature_count' in data_stats:
        print(f"  Depth features: {data_stats['depth_feature_count']} samples")
        if data_stats['depth_feature_count'] > 0:
            print(f"  Depth shape: {data_stats['depth_feature_shape']}")
            print(f"  Depth range: [{data_stats['depth_feature_min']:.3f}, {data_stats['depth_feature_max']:.3f}]")
    
    # Generate file name for the processed dataset
    output_filename = os.path.basename(args.data_path).replace('.csv', '_with_depth.pkl').replace('.pkl', '_with_depth.pkl')
    if args.balance_classes:
        output_filename = output_filename.replace('_with_depth.pkl', f'_{args.balance_method}_with_depth.pkl')
    output_path = os.path.join(args.output_dir, output_filename)
    
    # Save the processed dataset
    df.to_pickle(output_path)
    print(f"\nSaved processed dataset to: {output_path}")
    
    # Visualize sample depth prediction if requested
    if args.visualize and 'depth_feature' in df.columns and df['depth_feature'].notna().sum() > 0:
        print("\nVisualizing sample depth predictions...")
        
        # Find samples with depth features
        samples_with_depth = df[df['depth_feature'].notna()]
        if len(samples_with_depth) == 0:
            print("No samples with depth features found.")
            return
        
        # Sample up to 3 random samples
        num_samples = min(3, len(samples_with_depth))
        sample_indices = np.random.choice(samples_with_depth.index, num_samples, replace=False)
        
        # Create a figure with 2 columns (image and depth) and num_samples rows
        fig, axes = plt.subplots(num_samples, 2, figsize=(12, 5 * num_samples))
        
        for i, idx in enumerate(sample_indices):
            row = samples_with_depth.loc[idx]
            
            # Get image
            if 'Image' in row:
                img_data = row['Image']
                if isinstance(img_data, list):
                    img_array = np.array(img_data).reshape(224, 224, 3)
                else:
                    img_array = img_data
            elif 'image_path' in row:
                from PIL import Image
                img_array = np.array(Image.open(row['image_path']))
            else:
                print(f"Warning: No image data found for sample {idx}")
                continue
            
            # Get depth
            depth_array = row['depth_feature']
            
            # Get landmarks if available
            landmarks = None
            if all(col in row for col in landmark_cols):
                landmarks = row[landmark_cols].values.reshape(-1, 2).astype('float32')
            
            # Plot image
            ax_img = axes[i, 0] if num_samples > 1 else axes[0]
            ax_img.imshow(img_array)
            ax_img.set_title(f"Sample {idx} - Original Image")
            
            # Add landmarks if available
            if landmarks is not None:
                ax_img.scatter(landmarks[:, 0], landmarks[:, 1], c='red', marker='x')
            
            # Plot depth
            ax_depth = axes[i, 1] if num_samples > 1 else axes[1]
            
            # If depth is grayscale, use a colormap
            if len(depth_array.shape) == 2 or depth_array.shape[2] == 1:
                im = ax_depth.imshow(depth_array, cmap='plasma')
                fig.colorbar(im, ax=ax_depth, shrink=0.8)
            else:
                ax_depth.imshow(depth_array)
                
            ax_depth.set_title(f"Sample {idx} - Depth Map")
        
        plt.tight_layout()
        
        # Save the visualization
        vis_path = os.path.join(args.output_dir, 'depth_visualization.png')
        plt.savefig(vis_path)
        print(f"Saved visualization to: {vis_path}")
        plt.close()

if __name__ == "__main__":
    main() 