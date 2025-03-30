import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import our evaluation utilities
from src.utils.landmark_evaluation import (
    mean_euclidean_distance_per_landmark,
    success_rate_per_landmark,
    plot_landmark_metrics,
    generate_landmark_evaluation_report
)

def create_sample_data(batch_size=20, num_landmarks=19, prediction_error_level=5.0, error_bias=None):
    """
    Create sample ground truth and prediction data with controlled error levels
    
    Args:
        batch_size (int): Number of samples to generate
        num_landmarks (int): Number of landmarks per sample
        prediction_error_level (float): Average error level in pixels
        error_bias (list, optional): List of landmark indices that should have higher error levels
        
    Returns:
        tuple: (ground_truth, predictions)
    """
    # Create random ground truth landmarks within a 224x224 image
    ground_truth = np.random.uniform(20, 200, size=(batch_size, num_landmarks, 2))
    
    # Create predictions with controlled error
    # Base error - normal distribution around ground truth
    predictions = ground_truth + np.random.normal(0, prediction_error_level/2, size=ground_truth.shape)
    
    # Add bias to specific landmarks if requested
    if error_bias is not None:
        for idx in error_bias:
            # Add more error to biased landmarks
            predictions[:, idx, :] += np.random.normal(0, prediction_error_level*2, size=(batch_size, 2))
    
    # Convert to torch tensors (our functions should handle both numpy and torch)
    ground_truth_tensor = torch.tensor(ground_truth, dtype=torch.float32)
    predictions_tensor = torch.tensor(predictions, dtype=torch.float32)
    
    return ground_truth_tensor, predictions_tensor

def main():
    print("=== Testing Per-Landmark Evaluation Metrics ===\n")
    
    # Create output directory
    output_dir = Path("landmark_evaluation_test")
    output_dir.mkdir(exist_ok=True)
    
    # Define landmark names for better visualization
    landmark_names = [
        "Sella",
        "Nasion",
        "A Point",
        "B Point",
        "Upper 1 Tip",
        "Upper 1 Apex",
        "Lower 1 Tip",
        "Lower 1 Apex",
        "ANS",
        "PNS",
        "Gonion",
        "Menton",
        "ST Nasion",
        "Tip of nose",
        "Subnasal",
        "Upper Lip",
        "Lower Lip",
        "ST Pogonion",
        "Gnathion"
    ]
    
    # Create sample data with controlled error levels
    # Make specific landmarks more challenging to detect
    error_bias = [3, 10, 14]  # B Point, Gonion, Subnasal
    print(f"Creating sample data with error bias on: {[landmark_names[i] for i in error_bias]}")
    
    ground_truth, predictions = create_sample_data(
        batch_size=30,
        num_landmarks=19,
        prediction_error_level=4.0,
        error_bias=error_bias
    )
    
    # Calculate per-landmark MED
    med_per_landmark = mean_euclidean_distance_per_landmark(predictions, ground_truth)
    
    # Print results
    print("\nMean Euclidean Distance (MED) per landmark:")
    for i, (name, med) in enumerate(zip(landmark_names, med_per_landmark)):
        print(f"  {name}: {med:.2f} pixels")
    
    # Calculate overall MED
    overall_med = med_per_landmark.mean()
    print(f"\nOverall MED: {overall_med:.2f} pixels")
    
    # Calculate success rate at different thresholds
    thresholds = [2.0, 4.0, 6.0]
    success_rates = {}
    for threshold in thresholds:
        success_rates[threshold] = success_rate_per_landmark(predictions, ground_truth, threshold)
        
        print(f"\nSuccess Rate at {threshold}mm threshold:")
        for i, (name, rate) in enumerate(zip(landmark_names, success_rates[threshold])):
            print(f"  {name}: {rate*100:.2f}%")
    
    # Generate comprehensive evaluation report
    print("\nGenerating comprehensive evaluation report...")
    report = generate_landmark_evaluation_report(
        predictions=predictions, 
        targets=ground_truth,
        landmark_names=landmark_names,
        output_dir=output_dir,
        thresholds=thresholds
    )
    
    # Print worst landmarks from the report
    print("\nWorst performing landmarks:")
    worst_landmarks = report['worst_landmarks']
    worst_med = report['worst_landmarks_med']
    for i, (idx, med_val) in enumerate(zip(worst_landmarks, worst_med)):
        print(f"  {i+1}. {landmark_names[idx]}: {med_val:.2f} pixels")
    
    print(f"\nEvaluation report and visualizations saved to: {output_dir}")
    print("Test completed successfully!")

if __name__ == "__main__":
    main() 