#!/usr/bin/env python3
"""
Test script to verify that coordinate refinement works correctly
"""
import os
import sys
import torch
import platform

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.hrnet import create_hrnet_model

def test_coordinate_refinement():
    """Test the coordinate refinement functionality with random data"""
    print("Testing coordinate refinement MLP...")
    
    # Check for MPS (Mac GPU) availability
    use_mps = False
    if platform.system() == 'Darwin' and torch.backends.mps.is_available():
        print("MPS (Metal Performance Shaders) is available for Mac GPU acceleration.")
        device = torch.device('mps')
        use_mps = True
    elif torch.cuda.is_available():
        print("CUDA is available for GPU acceleration.")
        device = torch.device('cuda')
    else:
        print("Using CPU for computations.")
        device = torch.device('cpu')
    
    # Create model with refinement
    print("Creating HRNet model with refinement MLP...")
    model = create_hrnet_model(num_landmarks=19, pretrained=False, use_refinement=True)
    model = model.to(device)
    
    # Put model in eval mode to avoid batch norm issues
    model.eval()
    
    # Create a random input tensor
    print("Creating random input...")
    batch_size = 2
    input_tensor = torch.randn(batch_size, 3, 224, 224, device=device)
    
    print("Running forward pass to test model...")
    with torch.no_grad():
        # Run forward pass
        output = model(input_tensor)
    
    # Check if coordinates are constrained properly
    refined_coords = output['refined_coords']
    h, w = model.output_size
    
    # Check if coordinates are within bounds
    in_bounds = (refined_coords[..., 0] >= 0).all() and (refined_coords[..., 0] <= w-1).all() and \
                (refined_coords[..., 1] >= 0).all() and (refined_coords[..., 1] <= h-1).all()
    
    # Print results
    print("\nRefinement Test Results:")
    print(f"Output shape: {refined_coords.shape}")
    print(f"Min x: {refined_coords[..., 0].min().item():.2f}, Max x: {refined_coords[..., 0].max().item():.2f}")
    print(f"Min y: {refined_coords[..., 1].min().item():.2f}, Max y: {refined_coords[..., 1].max().item():.2f}")
    print(f"Coordinates in bounds: {in_bounds}")
    
    print("\nTest completed successfully!")
    return in_bounds

if __name__ == "__main__":
    test_coordinate_refinement() 