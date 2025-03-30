#!/usr/bin/env python3
"""
Test script to verify that the HRNet model loads correctly with timm
"""
import os
import sys
import torch
import platform

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models.hrnet import create_hrnet_model

def test_hrnet_model():
    print("Testing HRNet model loading...")
    
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
    model = create_hrnet_model(num_landmarks=19, pretrained=True, use_refinement=True)
    model = model.to(device)
    print(f"Model moved to {device} device")
    
    # Create a random input tensor
    print("Creating random input...")
    batch_size = 2
    input_tensor = torch.randn(batch_size, 3, 224, 224, device=device)
    
    print("Running forward pass to test model...")
    with torch.no_grad():
        # Run forward pass
        output = model(input_tensor)
    
    # Check output shape and type
    print("\nOutput check:")
    print(f"Output type: {type(output)}")
    
    for key, value in output.items():
        print(f"  {key}: shape {value.shape}, device {value.device}")
    
    print("\nHRNet model test completed successfully!")
    print("You can now run the training script without errors.")

if __name__ == "__main__":
    test_hrnet_model() 