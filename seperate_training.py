import os
import sys
import torch
import torch.nn as nn
import numpy as np
import logging
import time
import json
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from datetime import datetime

# Add src to path for imports
import sys
sys.path.append('/content/CephPredictionLALA/')

# Import model components
from src.models.hrnet import HRNet, DepthHRNet, RefinementMLP, LandmarkHeatmapNet, create_hrnet_model
from src.models.losses import AdaptiveWingLoss, WingLoss, CombinedLoss
from src.models.trainer import LandmarkTrainer

# Setup logging
def setup_logging(log_file='phased_training.log'):
    """Set up logging configuration with all logging disabled"""
    # Create logger but set level to a very high value to effectively disable logging
    logger = logging.getLogger('phased_training')
    logger.setLevel(logging.CRITICAL + 100)  # Set to level higher than CRITICAL to disable all logs
    
    # Create a null handler to avoid "No handler found" warnings
    null_handler = logging.NullHandler()
    logger.addHandler(null_handler)
    
    # Prevent log propagation to root logger
    logger.propagate = False
    
    return logger

# Helper function for silent logging - use this instead of logger.info to completely disable output
def silent_log(message):
    """Do nothing - silent logging function to replace logger.info calls"""
    pass

# Set seed for reproducibility
def set_seed(seed=42):
    """Set seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Save model configuration
def save_model_config(output_dir, config):
    """Save model configuration to JSON file"""
    config_path = os.path.join(output_dir, 'model_config.json')
    
    # Convert non-serializable types to strings
    serializable_config = {}
    for key, value in config.items():
        if key == 'device':
            serializable_config[key] = str(value)
        else:
            serializable_config[key] = value
    
    with open(config_path, 'w') as f:
        json.dump(serializable_config, f, indent=4)
    return config_path

# Visualize sample
def visualize_sample(loader, idx=0, with_depth=False):
    """Visualize a sample from the dataset"""
    # Get a sample
    for i, data in enumerate(loader):
        if i == idx:
            if with_depth:
                image, depth, landmarks, mask = data
                depth = depth.numpy()[0, 0]  # Take first batch item and first channel
            else:
                image, landmarks, mask = data
                depth = None
            
            image = image.numpy()[0].transpose(1, 2, 0)  # CHW -> HWC
            landmarks = landmarks.numpy()[0]
            mask = mask.numpy()[0] if mask is not None else None
            
            # Normalize image for visualization
            image = np.clip(image, 0, 1)
            
            # Plot
            fig, ax = plt.subplots(1, 3 if with_depth else 2, figsize=(15, 5))
            
            # Plot image
            ax[0].imshow(image)
            ax[0].set_title('RGB Image')
            
            # Plot landmarks on image
            for i, (x, y) in enumerate(landmarks):
                if mask is None or mask[i]:
                    ax[0].plot(x, y, 'ro', markersize=3)
                    ax[0].text(x, y, str(i), fontsize=8)
            
            # Plot depth if available
            if with_depth:
                ax[1].imshow(depth, cmap='viridis')
                ax[1].set_title('Depth Map')
                
                # Plot landmarks on depth
                for i, (x, y) in enumerate(landmarks):
                    if mask is None or mask[i]:
                        ax[1].plot(x, y, 'ro', markersize=3)
                        ax[1].text(x, y, str(i), fontsize=8)
                
                # Plot landmarks separately
                ax[2].scatter(landmarks[:, 0], landmarks[:, 1], c='red', s=10)
                ax[2].invert_yaxis()  # Invert y-axis to match image coordinates
                ax[2].set_title('Landmarks')
            else:
                # Plot landmarks separately
                ax[1].scatter(landmarks[:, 0], landmarks[:, 1], c='red', s=10)
                ax[1].invert_yaxis()  # Invert y-axis to match image coordinates
                ax[1].set_title('Landmarks')
            
            plt.tight_layout()
            return fig
    
    return None

# Phase 1: Train RGB Stream
def train_rgb_stream(data_processor, train_loader, val_loader, config):
    """Train the RGB stream of the landmark detection model"""
    logger = logging.getLogger('phased_training')
    silent_log("=== Phase 1: Training RGB Stream ===")
    
    # Create output directory
    phase1_dir = os.path.join(config['output_dir'], 'phase1_rgb')
    os.makedirs(phase1_dir, exist_ok=True)
    
    # Create RGB-only model
    rgb_model = create_hrnet_model(
        num_landmarks=config['num_landmarks'],
        pretrained=config['pretrained'],
        use_refinement=False,  # No refinement during separate training
        hrnet_type=config['hrnet_type'],
        use_rgb_features=True,
        use_depth=False  # RGB-only
    )
    
    # Move model to the correct device
    rgb_model = rgb_model.to(config['device'])
    
    # Create trainer
    rgb_trainer = LandmarkTrainer(
        num_landmarks=config['num_landmarks'],
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        device=config['device'],
        output_dir=phase1_dir,
        use_rgb_features=True,
        use_depth=False,  # RGB-only
        use_refinement=False,
        heatmap_weight=config['heatmap_weight'],
        coord_weight=config['coord_weight'],
        rgb_weight=1.0,
        depth_weight=0.0,  # No depth
        hrnet_type=config['hrnet_type'],
        scheduler_type=config['scheduler_type'],
        target_landmark_indices=config.get('target_landmark_indices')
    )
    
    # Set model
    rgb_trainer.model = rgb_model
    
    # Train
    rgb_trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['phase1_epochs'],
        save_freq=config['save_freq']
    )
    
    # Save final model
    rgb_trainer.save_checkpoint(os.path.join(phase1_dir, 'final_rgb_model.pth'))
    
    # Plot training curves
    rgb_trainer.plot_training_curves()
    
    # Save figure
    plt.savefig(os.path.join(phase1_dir, 'rgb_training_curves.png'))
    plt.close()
    
    return rgb_trainer, rgb_model

# Phase 2: Train Depth Stream
def train_depth_stream(data_processor, train_loader, val_loader, config):
    """Train the Depth stream of the landmark detection model"""
    logger = logging.getLogger('phased_training')
    silent_log("=== Phase 2: Training Depth Stream ===")
    
    # Create output directory
    phase2_dir = os.path.join(config['output_dir'], 'phase2_depth')
    os.makedirs(phase2_dir, exist_ok=True)
    
    # Create Depth-only model
    depth_model = create_hrnet_model(
        num_landmarks=config['num_landmarks'],
        pretrained=config['pretrained'],
        use_refinement=False,  # No refinement during separate training
        hrnet_type=config['hrnet_type'],
        use_rgb_features=False,  # Depth-only
        use_depth=True
    )
    
    # Move model to the correct device
    depth_model = depth_model.to(config['device'])
    
    # Create trainer
    depth_trainer = LandmarkTrainer(
        num_landmarks=config['num_landmarks'],
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        device=config['device'],
        output_dir=phase2_dir,
        use_rgb_features=False,  # Depth-only
        use_depth=True,
        use_refinement=False,
        heatmap_weight=config['heatmap_weight'],
        coord_weight=config['coord_weight'],
        rgb_weight=0.0,  # No RGB
        depth_weight=1.0,
        hrnet_type=config['hrnet_type'],
        scheduler_type=config['scheduler_type'],
        target_landmark_indices=config.get('target_landmark_indices')
    )
    
    # Set model
    depth_trainer.model = depth_model
    
    # Train
    depth_trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['phase2_epochs'],
        save_freq=config['save_freq']
    )
    
    # Save final model
    depth_trainer.save_checkpoint(os.path.join(phase2_dir, 'final_depth_model.pth'))
    
    # Plot training curves
    depth_trainer.plot_training_curves()
    
    # Save figure
    plt.savefig(os.path.join(phase2_dir, 'depth_training_curves.png'))
    plt.close()
    
    return depth_trainer, depth_model

# Phase 3: Train Fusion MLP (with frozen streams)
def train_fusion_mlp(data_processor, train_loader, val_loader, rgb_model, depth_model, config):
    """Train the fusion MLP while keeping the RGB and Depth streams frozen"""
    logger = logging.getLogger('phased_training')
    silent_log("=== Phase 3: Training Fusion MLP (with frozen streams) ===")
    
    # Create output directory
    phase3_dir = os.path.join(config['output_dir'], 'phase3_fusion')
    os.makedirs(phase3_dir, exist_ok=True)
    
    # Create combined model with pretrained streams
    combined_model = LandmarkHeatmapNet(
        num_landmarks=config['num_landmarks'],
        output_size=(64, 64),
        pretrained=False,  # We'll load pretrained weights manually
        use_refinement=True,  # Use refinement MLP
        hrnet_type=config['hrnet_type'],
        use_rgb_features=True,
        use_depth=True
    )
    
    # Move model to the correct device
    combined_model = combined_model.to(config['device'])
    
    # Load pretrained weights for RGB and Depth streams
    # Copy HRNet weights from rgb_model
    if combined_model.hrnet is not None and rgb_model.hrnet is not None:
        combined_model.hrnet.load_state_dict(rgb_model.hrnet.state_dict())
        silent_log("Loaded RGB HRNet weights")
    
    # Copy RGB heatmap layer weights
    if hasattr(rgb_model, 'rgb_heatmap_layer') and rgb_model.rgb_heatmap_layer is not None:
        if combined_model.rgb_heatmap_layer is None:
            # We need to create the layer first with matching dimensions
            in_features = rgb_model.rgb_heatmap_layer.in_channels
            combined_model.rgb_heatmap_layer = nn.Conv2d(
                in_features, config['num_landmarks'], 
                kernel_size=1, stride=1, padding=0
            ).to(config['device'])
        # Now copy weights
        combined_model.rgb_heatmap_layer.load_state_dict(rgb_model.rgb_heatmap_layer.state_dict())
        silent_log("Loaded RGB heatmap layer weights")
    
    # Copy DepthHRNet weights from depth_model
    if combined_model.depth_hrnet is not None and depth_model.depth_hrnet is not None:
        combined_model.depth_hrnet.load_state_dict(depth_model.depth_hrnet.state_dict())
        silent_log("Loaded Depth HRNet weights")
    
    # Copy Depth heatmap layer weights
    if hasattr(depth_model, 'depth_heatmap_layer') and depth_model.depth_heatmap_layer is not None:
        if combined_model.depth_heatmap_layer is None:
            # We need to create the layer first with matching dimensions
            in_features = depth_model.depth_heatmap_layer.in_channels
            combined_model.depth_heatmap_layer = nn.Conv2d(
                in_features, config['num_landmarks'], 
                kernel_size=1, stride=1, padding=0
            ).to(config['device'])
        # Now copy weights
        combined_model.depth_heatmap_layer.load_state_dict(depth_model.depth_heatmap_layer.state_dict())
        silent_log("Loaded Depth heatmap layer weights")
    
    # Freeze RGB and Depth streams
    for param in combined_model.hrnet.parameters():
        param.requires_grad = False
    
    for param in combined_model.depth_hrnet.parameters():
        param.requires_grad = False
    
    if combined_model.rgb_heatmap_layer is not None:
        for param in combined_model.rgb_heatmap_layer.parameters():
            param.requires_grad = False
    
    if combined_model.depth_heatmap_layer is not None:
        for param in combined_model.depth_heatmap_layer.parameters():
            param.requires_grad = False
    
    silent_log("Frozen RGB and Depth streams")
    
    # Only train the refinement MLP
    trainable_params = []
    for name, param in combined_model.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)
    
    # Create trainer
    fusion_trainer = LandmarkTrainer(
        num_landmarks=config['num_landmarks'],
        learning_rate=config['fusion_learning_rate'],  # Potentially different LR for MLP
        weight_decay=config['weight_decay'],
        device=config['device'],
        output_dir=phase3_dir,
        use_rgb_features=True,
        use_depth=True,
        use_refinement=True,
        heatmap_weight=0.0,  # Don't train on heatmaps since streams are frozen
        coord_weight=1.0,    # Only train on coordinate regression
        rgb_weight=0.0,      # No RGB heatmap loss
        depth_weight=0.0,    # No Depth heatmap loss
        hrnet_type=config['hrnet_type'],
        scheduler_type=config['scheduler_type'],
        target_landmark_indices=config.get('target_landmark_indices')
    )
    
    # Set model
    fusion_trainer.model = combined_model
    
    # Override optimizer to only train refinement MLP
    fusion_trainer.optimizer = torch.optim.Adam(
        trainable_params,
        lr=config['fusion_learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Train
    fusion_trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['phase3_epochs'],
        save_freq=config['save_freq']
    )
    
    # Save final model
    fusion_trainer.save_checkpoint(os.path.join(phase3_dir, 'final_fusion_model.pth'))
    
    # Plot training curves
    fusion_trainer.plot_training_curves()
    
    # Save figure
    plt.savefig(os.path.join(phase3_dir, 'fusion_training_curves.png'))
    plt.close()
    
    return fusion_trainer, combined_model

# Phase 4: Finetune Combined Model
def finetune_combined_model(data_processor, train_loader, val_loader, combined_model, config):
    """Finetune the entire combined model"""
    logger = logging.getLogger('phased_training')
    silent_log("=== Phase 4: Finetuning Combined Model ===")
    
    # Create output directory
    phase4_dir = os.path.join(config['output_dir'], 'phase4_finetune')
    os.makedirs(phase4_dir, exist_ok=True)
    
    # Unfreeze all parameters
    for param in combined_model.parameters():
        param.requires_grad = True
    
    silent_log("Unfrozen all model parameters")
    
    # Create trainer with lower learning rate for finetuning
    finetune_trainer = LandmarkTrainer(
        num_landmarks=config['num_landmarks'],
        learning_rate=config['finetune_learning_rate'],  # Lower LR for finetuning
        weight_decay=config['weight_decay'],
        device=config['device'],
        output_dir=phase4_dir,
        use_rgb_features=True,
        use_depth=True,
        use_refinement=True,
        heatmap_weight=config['heatmap_weight'],
        coord_weight=config['coord_weight'],
        rgb_weight=config['rgb_weight'],
        depth_weight=config['depth_weight'],
        hrnet_type=config['hrnet_type'],
        scheduler_type=config['scheduler_type'],
        target_landmark_indices=config.get('target_landmark_indices')
    )
    
    # Set model
    finetune_trainer.model = combined_model
    
    # Train
    finetune_trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['phase4_epochs'],
        save_freq=config['save_freq']
    )
    
    # Save final model
    finetune_trainer.save_checkpoint(os.path.join(phase4_dir, 'final_finetune_model.pth'))
    
    # Evaluate on validation set
    val_results = finetune_trainer.evaluate(val_loader, save_visualizations=True)
    
    # Save validation results
    with open(os.path.join(phase4_dir, 'validation_results.json'), 'w') as f:
        json.dump(val_results, f, indent=4)
    
    # Plot training curves
    finetune_trainer.plot_training_curves()
    
    # Save figure
    plt.savefig(os.path.join(phase4_dir, 'finetune_training_curves.png'))
    plt.close()
    
    return finetune_trainer

# Main function for phased training
def run_phased_training(config=None, data_processor=None, train_loader=None, val_loader=None, test_loader=None):
    """
    Run phased training with the given configuration
    
    Args:
        config (dict): Configuration dictionary with training parameters
        data_processor: Data processor object
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
    
    Returns:
        dict: Dictionary with trainer objects for each phase
    """
    # Use default configuration if none is provided
    if config is None:
        config = get_default_config()
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(config['output_dir'], f'phased_training_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    config['output_dir'] = output_dir
    
    # Setup logging
    logger = setup_logging(os.path.join(output_dir, 'phased_training.log'))
    silent_log(f"Starting phased training at {timestamp}")
    
    # Set seed
    set_seed(config['seed'])
    silent_log(f"Set seed to {config['seed']}")
    
    # Set device
    if 'device' not in config:
        config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    silent_log(f"Using device: {config['device']}")
    
    # Save configuration
    save_model_config(output_dir, config)
    silent_log(f"Saved model configuration to {output_dir}/model_config.json")
    
    # Store trainers for return
    trainers = {}
    
    # Phase 1: Train RGB Stream
    if config['run_phase1']:
        silent_log("Starting Phase 1: Training RGB Stream")
        rgb_trainer, rgb_model = train_rgb_stream(data_processor, train_loader, val_loader, config)
        trainers['phase1'] = rgb_trainer
    else:
        silent_log("Skipping Phase 1: Loading pretrained RGB model")
        # Load pretrained RGB model
        rgb_model = create_hrnet_model(
            num_landmarks=config['num_landmarks'],
            pretrained=config['pretrained'],
            use_refinement=False,
            hrnet_type=config['hrnet_type'],
            use_rgb_features=True,
            use_depth=False
        )
        
        # Move model to specified device
        rgb_model = rgb_model.to(config['device'])
        
        # Load pretrained weights if path is provided
        if config.get('rgb_model_path'):
            checkpoint = torch.load(config['rgb_model_path'], map_location=config['device'])
            if 'model_state_dict' in checkpoint:
                rgb_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                rgb_model.load_state_dict(checkpoint)
            silent_log(f"Loaded RGB model from {config['rgb_model_path']}")
        
        trainers['phase1'] = None
    
    # Phase 2: Train Depth Stream
    if config['run_phase2']:
        silent_log("Starting Phase 2: Training Depth Stream")
        depth_trainer, depth_model = train_depth_stream(data_processor, train_loader, val_loader, config)
        trainers['phase2'] = depth_trainer
    else:
        silent_log("Skipping Phase 2: Loading pretrained Depth model")
        # Load pretrained Depth model
        depth_model = create_hrnet_model(
            num_landmarks=config['num_landmarks'],
            pretrained=config['pretrained'],
            use_refinement=False,
            hrnet_type=config['hrnet_type'],
            use_rgb_features=False,
            use_depth=True
        )
        
        # Move model to specified device
        depth_model = depth_model.to(config['device'])
        
        # Load pretrained weights if path is provided
        if config.get('depth_model_path'):
            checkpoint = torch.load(config['depth_model_path'], map_location=config['device'])
            if 'model_state_dict' in checkpoint:
                depth_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                depth_model.load_state_dict(checkpoint)
            silent_log(f"Loaded depth model from {config['depth_model_path']}")
        
        trainers['phase2'] = None
    
    # Phase 3: Train Fusion MLP
    if config['run_phase3']:
        silent_log("Starting Phase 3: Training Fusion MLP")
        fusion_trainer, combined_model = train_fusion_mlp(data_processor, train_loader, val_loader, rgb_model, depth_model, config)
        trainers['phase3'] = fusion_trainer
    else:
        silent_log("Skipping Phase 3: Loading pretrained Fusion model")
        # Load pretrained Fusion model
        combined_model = LandmarkHeatmapNet(
            num_landmarks=config['num_landmarks'],
            output_size=(64, 64),
            pretrained=False,
            use_refinement=True,
            hrnet_type=config['hrnet_type'],
            use_rgb_features=True,
            use_depth=True
        )
        
        # Move model to specified device
        combined_model = combined_model.to(config['device'])
        
        # Load pretrained weights if path is provided
        if config.get('fusion_model_path'):
            checkpoint = torch.load(config['fusion_model_path'], map_location=config['device'])
            if 'model_state_dict' in checkpoint:
                combined_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                combined_model.load_state_dict(checkpoint)
            silent_log(f"Loaded fusion model from {config['fusion_model_path']}")
        
        trainers['phase3'] = None
    
    # Phase 4: Finetune Combined Model
    if config['run_phase4']:
        silent_log("Starting Phase 4: Finetuning Combined Model")
        finetune_trainer = finetune_combined_model(data_processor, train_loader, val_loader, combined_model, config)
        trainers['phase4'] = finetune_trainer
    else:
        silent_log("Skipping Phase 4: Using model from Phase 3")
        trainers['phase4'] = trainers['phase3']
    
    # Evaluate on test set if available
    if config['evaluate'] and test_loader is not None and trainers['phase4'] is not None:
        silent_log("Evaluating on test set")
        test_results = trainers['phase4'].evaluate(test_loader, save_visualizations=True)
        
        # Save test results
        with open(os.path.join(output_dir, 'test_results.json'), 'w') as f:
            json.dump(test_results, f, indent=4)
        
        # Visualize sample predictions
        visualize_sample_predictions(trainers['phase4'], test_loader, num_samples=config['vis_samples'])
    
    silent_log("Phased training completed")
    
    return trainers

# Default configuration function
def get_default_config():
    """Get default configuration for phased training"""
    return {
        # General parameters
        'output_dir': './outputs',
        'seed': 42,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        
        # Model parameters
        'num_landmarks': 19,
        'pretrained': True,
        'hrnet_type': 'w32',
        'target_landmark_indices': None,
        
        # Training parameters
        'batch_size': 32,
        'learning_rate': 1e-4,
        'fusion_learning_rate': 1e-3,
        'finetune_learning_rate': 1e-5,
        'weight_decay': 1e-5,
        'heatmap_weight': 1.0,
        'coord_weight': 0.1,
        'rgb_weight': 1.0,
        'depth_weight': 1.0,
        'scheduler_type': 'plateau',
        
        # Phase control
        'run_phase1': True,
        'run_phase2': True,
        'run_phase3': True,
        'run_phase4': True,
        
        # Phase-specific parameters
        'phase1_epochs': 50,
        'phase2_epochs': 50,
        'phase3_epochs': 30,
        'phase4_epochs': 20,
        'save_freq': 5,
        
        # Evaluation parameters
        'evaluate': True,
        'vis_samples': 5,
        
        # Model paths for loading pretrained models (optional)
        'rgb_model_path': None,
        'depth_model_path': None,
        'fusion_model_path': None
    }

# Example notebook usage
if __name__ == "__main__":
    # This section won't run in a notebook, but shows how to use the code
    
    # Get default configuration
    config = get_default_config()
    
    # Modify configuration as needed
    config['num_landmarks'] = 19
    config['phase1_epochs'] = 30
    config['phase2_epochs'] = 30
    config['phase3_epochs'] = 20
    config['phase4_epochs'] = 10
    
    # Run phased training
    trainers = run_phased_training(config) 