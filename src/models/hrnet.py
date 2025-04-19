import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import warnings
import logging
from .losses import soft_argmax

# Try to import timm - if not available, provide instructions
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    warnings.warn(
        "The 'timm' package is not installed. To use pretrained models, "
        "please install it with: pip install timm"
    )

class HighResolutionModule(nn.Module):
    """
    High Resolution Module for HRNet
    
    Multi-branch and multi-resolution module that processes features at different resolutions
    """
    def __init__(self, num_branches, blocks, num_blocks, num_channels,
                 multi_scale_output=True, fuse_method='SUM'):
        super(HighResolutionModule, self).__init__()
        self.num_branches = num_branches
        self.fuse_method = fuse_method
        
        self.branches = self._make_branches(num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers(num_branches, num_channels, multi_scale_output)
        self.relu = nn.ReLU(inplace=True)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        # Implementation depends on HRNet specifics
        # Simplified placeholder for the branches
        branches = nn.ModuleList()
        for i in range(num_branches):
            layers = []
            for j in range(num_blocks[i]):
                layers.append(nn.Conv2d(num_channels[i], num_channels[i], 3, 1, 1))
                layers.append(nn.BatchNorm2d(num_channels[i]))
                layers.append(nn.ReLU(inplace=True))
            branches.append(nn.Sequential(*layers))
        return branches

    def _make_fuse_layers(self, num_branches, num_channels, multi_scale_output=True):
        # Implementation depends on HRNet specifics
        # Simplified placeholder for the fuse layers
        if num_branches == 1:
            return None

        fuse_layers = nn.ModuleList()
        for i in range(num_branches if multi_scale_output else 1):
            fuse_layer = nn.ModuleList()
            for j in range(num_branches):
                if j > i:
                    # Upsample to the i-th branch's resolution
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_channels[j], num_channels[i], 1, 1, 0, bias=False),
                        nn.BatchNorm2d(num_channels[i]),
                        nn.Upsample(scale_factor=2**(j-i), mode='bilinear', align_corners=False)
                    ))
                elif j == i:
                    # Identity mapping
                    fuse_layer.append(nn.Identity())
                else:
                    # Downsample to the i-th branch's resolution
                    ops = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            ops.extend([
                                nn.Conv2d(num_channels[j], num_channels[i], 3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_channels[i])
                            ])
                        else:
                            ops.extend([
                                nn.Conv2d(num_channels[j], num_channels[j], 3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_channels[j]),
                                nn.ReLU(inplace=True)
                            ])
                    fuse_layer.append(nn.Sequential(*ops))
            fuse_layers.append(fuse_layer)
        
        return fuse_layers

    def forward(self, x):
        # Forward pass through branches
        branch_outputs = []
        for i in range(self.num_branches):
            branch_outputs.append(self.branches[i](x[i] if isinstance(x, list) else x))

        # Return directly if no fusion needed
        if self.fuse_layers is None:
            return branch_outputs
        
        # Fuse branches
        out = []
        for i, fuse_layer in enumerate(self.fuse_layers):
            y = fuse_layer[0](branch_outputs[0])
            for j in range(1, self.num_branches):
                y = y + fuse_layer[j](branch_outputs[j])
            out.append(self.relu(y))
            
        return out


class HRNet(nn.Module):
    """
    High-Resolution Network (HRNet) for landmark detection
    
    Supports different HRNet variants (W32, W48, etc.)
    """
    def __init__(self, pretrained=True, hrnet_type='w32'):
        super(HRNet, self).__init__()
        
        # Load pretrained HRNet model
        if pretrained:
            try:
                if not TIMM_AVAILABLE:
                    raise ImportError("timm package is not installed")
                
                # Use timm to load pretrained HRNet
                model_name = f'hrnet_{hrnet_type}'
                self.backbone = timm.create_model(model_name, pretrained=True, features_only=True)
                print(f"Successfully loaded pretrained {model_name} using timm.")
            except Exception as e:
                # Fallback to a simplified backbone if pretrained model is not available
                logging.warning(f"Error loading pretrained HRNet-{hrnet_type.upper()}: {str(e)}")
                print(f"Warning: Pretrained HRNet-{hrnet_type.upper()} not available. Using simplified backbone.")
                print("To fix this issue, install timm: pip install timm")
                self.backbone = self._create_simplified_backbone()
        else:
            # Use simplified backbone if pretrained is not required
            self.backbone = self._create_simplified_backbone()
    
    def _create_simplified_backbone(self):
        # Simplified backbone as a placeholder
        # In practice, this would be a full implementation of HRNet
        backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        return backbone
    
    def forward(self, x):
        # Just return whatever the backbone outputs - we'll handle it in LandmarkHeatmapNet
        return self.backbone(x)


class RefinementMLP(nn.Module):
    """
    Refinement MLP for landmark coordinates
    
    Takes initial landmark coordinates and outputs deltas for refinement
    """
    def __init__(self, num_landmarks, max_delta=2.0):
        """
        Initialize the refinement MLP
        
        Args:
            num_landmarks (int): Number of landmarks to refine
            max_delta (float): Maximum allowed delta in each direction (in heatmap coordinates)
        """
        super(RefinementMLP, self).__init__()
        
        self.num_landmarks = num_landmarks
        self.max_delta = max_delta
        input_dim = num_landmarks * 2  # x, y for each landmark
        hidden_dim = 256
        
        # Simple input normalization using mean/std instead of InstanceNorm
        self.input_norm = False  # Flag to control whether normalization is applied
        
        # Two hidden layers with 256 units each
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)  # Output deltas for each coordinate
        
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()  # Use tanh to constrain the delta range
    
    def forward(self, x):
        """
        Forward pass through the MLP
        
        Args:
            x (torch.Tensor): Initial landmark coordinates of shape (batch_size, num_landmarks*2)
            
        Returns:
            torch.Tensor: Coordinate deltas of shape (batch_size, num_landmarks*2)
        """
        # Flatten if necessary (if x is in format batch_size, num_landmarks, 2)
        if x.dim() == 3:
            x = x.view(x.size(0), -1)

        # Simple normalization (standardize each batch separately)
        if self.input_norm:
            # Compute mean and std along batch dimension
            mean = x.mean(dim=1, keepdim=True)
            std = x.std(dim=1, keepdim=True) + 1e-6  # Avoid division by zero
            x_norm = (x - mean) / std
        else:
            x_norm = x
                        
        # Forward through the network
        # Forward through the network with normalized input
        h1 = self.relu(self.bn1(self.fc1(x_norm)))
        h2 = self.relu(self.bn2(self.fc2(h1)))
        deltas = self.fc3(h2)
        
        
        # # Output deltas with tanh activation and scaling
        # # tanh outputs values between -1 and 1, we scale by max_delta
        # deltas = self.tanh(self.fc3(h2)) * self.max_delta

        return deltas


class LandmarkHeatmapNet(nn.Module):
    """
    Landmark detection network using heatmap regression with refinement MLP
    
    Architecture:
    1. HRNet backbone (W32 or W48)
    2. 1x1 convolution to output heatmaps (one per landmark)
    3. Coordinate extraction from heatmaps
    4. Refinement MLP to improve coordinate predictions
    """
    def __init__(self, num_landmarks=19, output_size=(64, 64), pretrained=True, use_refinement=True, hrnet_type='w32'):
        super(LandmarkHeatmapNet, self).__init__()
        
        self.num_landmarks = num_landmarks
        self.output_size = output_size
        self.use_refinement = use_refinement
        self.hrnet_type = hrnet_type
        
        # HRNet backbone
        self.hrnet = HRNet(pretrained=pretrained, hrnet_type=hrnet_type)
        
        # Determine backbone output channels for heatmap layer
        # Check if the backbone has feature_info (common in timm models)
        if hasattr(self.hrnet.backbone, 'feature_info'):
            # Get the number of channels from the last feature map (highest resolution)
            in_channels = self.hrnet.backbone.feature_info[-1]['num_chs']
        else:
            # Fallback: Perform a dummy forward pass to get the shape
            # This is less ideal but works if feature_info is not available
            print("Warning: Backbone does not have feature_info. Performing dummy forward pass to determine heatmap layer input channels.")
            with torch.no_grad():
                # Create a dummy input tensor on CPU first
                dummy_input = torch.randn(1, 3, 224, 224)
                
                # Handle device placement safely
                # 1. Get the device from the backbone's first parameter if available
                try:
                    model_params = list(self.hrnet.parameters())
                    if model_params:
                        model_device = model_params[0].device
                        # Move both the dummy input and model to the same device if needed
                        dummy_input = dummy_input.to(model_device)
                    else:
                        # No parameters yet, default to CPU and move model there explicitly
                        model_device = torch.device('cpu')
                        dummy_input = dummy_input.to(model_device)
                        self.hrnet = self.hrnet.to(model_device)
                except Exception as e:
                    print(f"Warning: Error determining device for dummy forward pass: {str(e)}. Using CPU.")
                    model_device = torch.device('cpu')
                    dummy_input = dummy_input.to(model_device)
                    # Try to ensure model is on the same device
                    try:
                        self.hrnet = self.hrnet.to(model_device)
                    except:
                        pass  # If this fails, we'll let the forward pass itself handle errors
                
                # Perform dummy forward pass
                try:
                    features = self.hrnet(dummy_input)
                    if isinstance(features, list):
                        features = features[-1]
                    in_channels = features.shape[1]
                except Exception as e:
                    # If forward pass fails, fall back to a reasonable default
                    print(f"Warning: Dummy forward pass failed: {str(e)}. Using default channel count of 256.")
                    in_channels = 256  # Reasonable default for HRNet
        
        # We'll create the heatmap layer after we know the channel size
        # Create the heatmap layer now
        self.heatmap_layer = nn.Conv2d(in_channels, self.num_landmarks, kernel_size=1, stride=1, padding=0)
        print(f"Created heatmap layer with {in_channels} input channels")
        
        # Refinement MLP
        if use_refinement:
            self.refinement_mlp = RefinementMLP(num_landmarks)
    
    def forward(self, x):
        """
        Forward pass to generate heatmaps
        
        Args:
            x (torch.Tensor): Input images of shape (batch_size, channels, height, width)
            
        Returns:
            dict: Dictionary containing heatmaps and optionally refined coordinates
        """
        # Forward pass through HRNet backbone
        features = self.hrnet(x)
        
        # Handle case where features is a list (from timm model with features_only=True)
        if isinstance(features, list):
            # Use the highest resolution features (last in the list)
            features = features[-1]
        
        # Create the heatmap layer if it doesn't exist or if the channel dimension doesn't match
        if self.heatmap_layer is None or self.heatmap_layer.in_channels != features.shape[1]:
            in_channels = features.shape[1]
            self.heatmap_layer = nn.Conv2d(in_channels, self.num_landmarks, kernel_size=1, stride=1, padding=0)
            # Move to the same device as the features
            self.heatmap_layer = self.heatmap_layer.to(features.device)
            print(f"Created heatmap layer with {in_channels} input channels")
        
        # Resize features to desired output size if needed
        if features.shape[2:] != self.output_size:
            features = F.interpolate(features, size=self.output_size, mode='bilinear', align_corners=False)
        
        # Generate heatmaps
        heatmaps = self.heatmap_layer(features)
        
        # Get initial landmark coordinates using soft-argmax
        initial_coords = self._extract_coordinates(heatmaps)
        
        # If using refinement, pass through MLP
        if self.use_refinement:
            # Flatten coordinates for MLP
            batch_size = initial_coords.size(0)
            flat_coords = initial_coords.view(batch_size, -1)
            
            # Get coordinate deltas from MLP
            deltas = self.refinement_mlp(flat_coords)
            
            # Add deltas to initial coordinates
            refined_flat_coords = flat_coords + deltas
            
            # Reshape back to (batch_size, num_landmarks, 2)
            refined_coords = refined_flat_coords.view(batch_size, self.num_landmarks, 2)
            
            # Constrain refined coordinates to be within a valid range
            # Heatmap coordinates should be within [0, width] and [0, height]
            h, w = self.output_size
            
            # Create min and max tensors with the same shape as refined_coords
            min_values = torch.zeros_like(refined_coords)
            max_values = torch.ones_like(refined_coords)
            # Set different max values for x and y coordinates
            max_values[:, :, 0] = w - 1  # Max x
            max_values[:, :, 1] = h - 1  # Max y
            
            # Clamp with tensor arguments
            refined_coords = torch.clamp(refined_coords, min=min_values, max=max_values)
            
            # Return both heatmaps and refined coordinates
            return {
                'heatmaps': heatmaps,
                'initial_coords': initial_coords,
                'refined_coords': refined_coords
            }
        
        # If not using refinement, just return heatmaps and initial coordinates
        return {
            'heatmaps': heatmaps,
            'initial_coords': initial_coords
        }
    
    def _extract_coordinates(self, heatmaps, beta=100):
        """
        Extract landmark coordinates from heatmaps using soft-argmax
        
        Args:
            heatmaps (torch.Tensor): Heatmaps of shape (batch_size, num_landmarks, height, width)
            beta (float): Temperature parameter for softmax
            
        Returns:
            torch.Tensor: Coordinates of shape (batch_size, num_landmarks, 2)
        """
        return soft_argmax(heatmaps, beta=beta)
    
    def predict_landmarks(self, x, use_soft_argmax=True, beta=100):
        """
        Predict landmark coordinates from input images
        
        Args:
            x (torch.Tensor): Input images of shape (batch_size, channels, height, width)
            use_soft_argmax (bool): Whether to use soft-argmax for sub-pixel accuracy
            beta (float): Temperature parameter for softmax (only used if use_soft_argmax=True)
            
        Returns:
            torch.Tensor: Predicted landmark coordinates of shape (batch_size, num_landmarks, 2)
        """
        # Get model output
        output = self(x)
        
        # If using refinement, return refined coordinates
        if self.use_refinement and 'refined_coords' in output:
            coords = output['refined_coords']
        else:
            # Otherwise, use initial coordinates
            coords = output['initial_coords']
        
        # Scale coordinates to the original image size (224x224)
        scale_factor = 224.0 / self.output_size[0]  # Assuming square output
        coords = coords * scale_factor
        
        return coords


def create_hrnet_model(num_landmarks=19, pretrained=True, use_refinement=True, hrnet_type='w32'):
    """
    Create a HRNet-based landmark detection model
    
    Args:
        num_landmarks (int): Number of landmarks to detect
        pretrained (bool): Whether to use pretrained weights for the backbone
        use_refinement (bool): Whether to use refinement MLP
        hrnet_type (str): HRNet variant to use ('w32' or 'w48')
        
    Returns:
        LandmarkHeatmapNet: The created model
    """
    model = LandmarkHeatmapNet(
        num_landmarks=num_landmarks,
        output_size=(64, 64),
        pretrained=pretrained,
        use_refinement=use_refinement,
        hrnet_type=hrnet_type
    )
    return model 