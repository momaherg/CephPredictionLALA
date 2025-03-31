import os
import torch
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.data.dataset import CephalometricDataset, create_dataloader_with_augmentations
from src.models.model import LandmarkDetectionModel
from src.models.losses import CombinedLoss
from src.utils.lr_finder import LRFinder

def parse_args():
    parser = argparse.ArgumentParser(description='Run Learning Rate Range Test')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset file')
    parser.add_argument('--output_dir', type=str, default='./outputs/lr_finder', help='Directory to save outputs')
    parser.add_argument('--apply_clahe', action='store_true', help='Apply CLAHE for histogram equalization')
    
    # Model arguments
    parser.add_argument('--num_landmarks', type=int, default=19, help='Number of landmarks to detect')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained HRNet backbone')
    parser.add_argument('--use_refinement', action='store_true', help='Use refinement MLP for coordinate regression')
    parser.add_argument('--heatmap_weight', type=float, default=1.0, help='Weight for heatmap loss')
    parser.add_argument('--coord_weight', type=float, default=0.1, help='Weight for coordinate loss')
    
    # LR Range Test arguments
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--start_lr', type=float, default=1e-7, help='Starting learning rate')
    parser.add_argument('--end_lr', type=float, default=10.0, help='Ending learning rate')
    parser.add_argument('--num_iter', type=int, default=100, help='Number of iterations to run test (default: entire dataset)')
    parser.add_argument('--step_mode', type=str, choices=['exp', 'linear'], default='exp', help='Mode of increasing learning rate')
    parser.add_argument('--optimizer', type=str, choices=['adam', 'adamw', 'sgd'], default='adam', help='Optimizer to use')
    parser.add_argument('--skip_start', type=int, default=5, help='Number of initial points to skip in plot')
    parser.add_argument('--skip_end', type=int, default=5, help='Number of final points to skip in plot')
    
    # Device arguments
    parser.add_argument('--use_mps', action='store_true', help='Use Metal Performance Shaders (MPS) for Mac GPU acceleration')
    parser.add_argument('--force_cpu', action='store_true', help='Force CPU usage even if GPU is available')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    if args.force_cpu:
        device = torch.device('cpu')
        print("Using CPU as requested")
    elif args.use_mps and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS (Metal Performance Shaders) for Mac GPU acceleration")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Load dataset
    print(f"Loading dataset from {args.data_path}...")
    import pandas as pd
    df = pd.read_csv(args.data_path)
    
    # Extract landmark columns from dataframe
    landmark_cols = [col for col in df.columns if col.endswith('_x') or col.endswith('_y')]
    
    # Create dataloaders with augmentations - we only need the training loader for LR Range Test
    print("Creating dataloaders...")
    train_loader, _, _ = create_dataloader_with_augmentations(
        df=df,
        landmark_cols=landmark_cols,
        batch_size=args.batch_size,
        train_ratio=0.8,
        val_ratio=0.1,
        apply_clahe=args.apply_clahe,
        root_dir=None,
        num_workers=0
    )
    
    # Create model
    print("Creating model...")
    model = LandmarkDetectionModel(
        num_landmarks=args.num_landmarks,
        use_pretrained=args.pretrained,
        use_refinement=args.use_refinement
    ).to(device)
    
    # Create loss function
    criterion = CombinedLoss(
        heatmap_weight=args.heatmap_weight,
        coord_weight=args.coord_weight
    )
    
    # Create optimizer
    print(f"Creating {args.optimizer.upper()} optimizer...")
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, 
                                   weight_decay=1e-5, nesterov=True)
    
    # Create LR Finder
    lr_finder = LRFinder(model, optimizer, criterion, device)
    
    # Run LR Range Test
    print("Running Learning Rate Range Test...")
    history = lr_finder.range_test(
        train_loader=train_loader,
        start_lr=args.start_lr,
        end_lr=args.end_lr,
        num_iter=args.num_iter if args.num_iter > 0 else None,
        step_mode=args.step_mode
    )
    
    # Plot results
    print("Plotting results...")
    suggested_lr = lr_finder.plot(
        output_dir=args.output_dir,
        skip_start=args.skip_start,
        skip_end=args.skip_end,
        log_lr=True,
        suggest_lr=True
    )
    
    # Save suggested learning rate to a file
    if suggested_lr is not None:
        with open(os.path.join(args.output_dir, 'suggested_lr.txt'), 'w') as f:
            f.write(f"Suggested learning rate: {suggested_lr:.8e}\n")
            f.write(f"Conservative max_lr (1/10 of suggested): {suggested_lr/10:.8e}\n")
            f.write(f"Aggressive max_lr (suggested): {suggested_lr:.8e}\n")
        print(f"Suggested learning rate saved to {os.path.join(args.output_dir, 'suggested_lr.txt')}")
    
    print("LR Range Test completed successfully!")

if __name__ == "__main__":
    main() 