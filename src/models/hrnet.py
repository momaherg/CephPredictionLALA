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


class DepthCNN(nn.Module):
    """
    Lightweight CNN for processing depth features.
    
    Takes a single-channel depth map and extracts features that can be fused
    with RGB features from the HRNet backbone.
    """
    def __init__(self, output_channels=128, output_size=(64, 64)):
        """
        Initialize the Depth CNN
        
        Args:
            output_channels (int): Number of output channels for the depth features
            output_size (tuple): Target output spatial dimensions (height, width)
        """
        super(DepthCNN, self).__init__()
        
        self.output_channels = output_channels
        self.output_size = output_size
        
        # Define a lightweight CNN for depth processing
        # Using a ResNet-style architecture with fewer parameters
        self.conv1 = nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet-style blocks with fewer channels
        self.layer1 = self._make_layer(32, 32, 2, stride=1)
        self.layer2 = self._make_layer(32, 64, 2, stride=2)
        self.layer3 = self._make_layer(64, output_channels, 2, stride=2)
        
        # Final projection to ensure proper channel count
        self.final_conv = nn.Conv2d(output_channels, output_channels, kernel_size=1)
        self.final_bn = nn.BatchNorm2d(output_channels)
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        """Create a sequence of ResNet-style blocks"""
        layers = []
        
        # First block handles downsampling if needed
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        # Add first block with potential downsampling
        layers.append(self._resnet_block(in_channels, out_channels, stride, downsample))
        
        # Add remaining blocks
        for _ in range(1, blocks):
            layers.append(self._resnet_block(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _resnet_block(self, in_channels, out_channels, stride=1, downsample=None):
        """Create a single ResNet block"""
        class ResBlock(nn.Module):
            def __init__(self, in_channels, out_channels, stride, downsample):
                super(ResBlock, self).__init__()
                self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
                self.bn1 = nn.BatchNorm2d(out_channels)
                self.relu = nn.ReLU(inplace=True)
                self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
                self.bn2 = nn.BatchNorm2d(out_channels)
                self.downsample = downsample
            
            def forward(self, x):
                residual = x
                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)
                out = self.conv2(out)
                out = self.bn2(out)
                
                if self.downsample is not None:
                    residual = self.downsample(x)
                
                out += residual
                out = self.relu(out)
                return out
        
        return ResBlock(in_channels, out_channels, stride, downsample)
    
    def forward(self, x):
        """
        Forward pass through depth CNN
        
        Args:
            x (torch.Tensor): Input depth map tensor of shape (batch_size, 1, height, width)
        
        Returns:
            torch.Tensor: Depth features of shape (batch_size, output_channels, output_size[0], output_size[1])
        """
        # Basic feature extraction
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # ResNet-style blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # Final projection
        x = self.final_conv(x)
        x = self.final_bn(x)
        x = self.relu(x)
        
        # Resize to match desired output dimensions if needed
        if x.shape[2:] != self.output_size:
            x = F.interpolate(x, size=self.output_size, mode='bilinear', align_corners=False)
        
        return x


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
        # deltas = self.fc3(h2)
        
        
        # # Output deltas with tanh activation and scaling
        # # tanh outputs values between -1 and 1, we scale by max_delta
        deltas = self.tanh(self.fc3(h2)) * self.max_delta

        return deltas


class LandmarkHeatmapNet(nn.Module):
    """
    Landmark detection network using heatmap regression with refinement MLP and optional depth features.
    
    Architecture:
    1. HRNet backbone (W32 or W48) for RGB image
    2. Optional DepthCNN for depth features
    3. Feature fusion (concatenation)
    4. 1x1 convolution to output heatmaps (one per landmark)
    5. Coordinate extraction from heatmaps
    6. Refinement MLP to improve coordinate predictions
    """
    def __init__(self, num_landmarks=19, output_size=(64, 64), pretrained=True, 
                 use_refinement=True, hrnet_type='w32', use_depth=False, depth_channels=64):
        super(LandmarkHeatmapNet, self).__init__()
        
        self.num_landmarks = num_landmarks
        self.output_size = output_size
        self.use_refinement = use_refinement
        self.hrnet_type = hrnet_type
        self.use_depth = use_depth
        self.depth_channels = depth_channels
        
        # HRNet backbone for RGB
        self.hrnet = HRNet(pretrained=pretrained, hrnet_type=hrnet_type)
        
        # Depth CNN for depth features (if enabled)
        if use_depth:
            self.depth_cnn = DepthCNN(output_channels=depth_channels, output_size=output_size)
        
        # We'll create the heatmap layer after we know the channel size in forward()
        self.heatmap_layer = None
        
        # Refinement MLP
        if use_refinement:
            self.refinement_mlp = RefinementMLP(num_landmarks)
    
    def forward(self, x, depth=None):
        """
        Forward pass to generate heatmaps
        
        Args:
            x (torch.Tensor): Input RGB images of shape (batch_size, 3, height, width)
            depth (torch.Tensor, optional): Input depth maps of shape (batch_size, 1, height, width)
            
        Returns:
            dict: Dictionary containing heatmaps and optionally refined coordinates
        """
        # Forward pass through HRNet backbone
        features = self.hrnet(x)
        
        # Handle case where features is a list (from timm model with features_only=True)
        if isinstance(features, list):
            # Use the highest resolution features (last in the list)
            features = features[-1]
        
        # Process depth features if available and enabled
        depth_features = None
        if self.use_depth and depth is not None:
            depth_features = self.depth_cnn(depth)
            
            # Ensure HRNet features match depth features' spatial dimensions before concatenation
            if features.shape[2:] != depth_features.shape[2:]:
                print(f"Resizing HRNet features from {features.shape[2:]} to {depth_features.shape[2:]}")
                features = F.interpolate(features, size=depth_features.shape[2:], mode='bilinear', align_corners=False)
            
            # Concatenate RGB and depth features along channel dimension
            features = torch.cat([features, depth_features], dim=1)
        
        # Create or update the heatmap layer if needed
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
    
    def predict_landmarks(self, x, depth=None, use_soft_argmax=True, beta=100):
        """
        Predict landmark coordinates from input images
        
        Args:
            x (torch.Tensor): Input RGB images of shape (batch_size, channels, height, width)
            depth (torch.Tensor, optional): Input depth maps of shape (batch_size, 1, height, width)
            use_soft_argmax (bool): Whether to use soft-argmax for sub-pixel accuracy
            beta (float): Temperature parameter for softmax (only used if use_soft_argmax=True)
            
        Returns:
            torch.Tensor: Predicted landmark coordinates of shape (batch_size, num_landmarks, 2)
        """
        # Get model output
        output = self(x, depth)
        
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


def create_hrnet_model(num_landmarks=19, pretrained=True, use_refinement=True, 
                       hrnet_type='w32', use_depth=False, depth_channels=64):
    """
    Create a HRNet-based landmark detection model
    
    Args:
        num_landmarks (int): Number of landmarks to detect
        pretrained (bool): Whether to use pretrained weights for the backbone
        use_refinement (bool): Whether to use refinement MLP
        hrnet_type (str): HRNet variant to use ('w32' or 'w48')
        use_depth (bool): Whether to use depth features
        depth_channels (int): Number of channels for depth features
        
    Returns:
        LandmarkHeatmapNet: The created model
    """
    model = LandmarkHeatmapNet(
        num_landmarks=num_landmarks,
        output_size=(64, 64),
        pretrained=pretrained,
        use_refinement=use_refinement,
        hrnet_type=hrnet_type,
        use_depth=use_depth,
        depth_channels=depth_channels
    )
    return model 