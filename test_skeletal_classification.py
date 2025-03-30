import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import our evaluation utilities
from src.utils.landmark_evaluation import (
    evaluate_skeletal_classification,
    generate_landmark_evaluation_report
)
from src.data.patient_classifier import PatientClassifier

def create_sample_data(batch_size=30, error_level=5.0, class_distribution=None):
    """
    Create sample landmark data for skeletal classification testing
    
    Args:
        batch_size (int): Number of samples to generate
        error_level (float): Average error level in pixels
        class_distribution (list, optional): Distribution of skeletal classes [class1, class2, class3]
        
    Returns:
        tuple: (ground_truth, predictions, landmark_cols)
    """
    # Define landmark column names
    landmark_cols = ['sella_x', 'sella_y', 'nasion_x', 'nasion_y', 'A point_x', 'A point_y',
                     'B point_x', 'B point_y']
    
    # Create a base landmark configuration for each class
    # Class 1 (normal): ANB angle around 2 degrees
    # Class 2 (prognathic maxilla): ANB angle around 6 degrees
    # Class 3 (retrognathic maxilla): ANB angle around -2 degrees
    
    # Base configuration - landmarks for a "normal" class 1 patient
    # Format: [sella_x, sella_y, nasion_x, nasion_y, A_x, A_y, B_x, B_y]
    base_config = np.array([
        [100, 100],  # Sella
        [100, 150],  # Nasion
        [120, 180],  # A point
        [110, 190],  # B point
    ])
    
    # Class 1 variations (minimal modification from base)
    class1_config = base_config.copy()
    
    # Class 2 variations (move A point forward relative to B point)
    class2_config = base_config.copy()
    class2_config[2, 0] += 15  # Move A point forward (x+15)
    
    # Class 3 variations (move B point forward relative to A point)
    class3_config = base_config.copy()
    class3_config[3, 0] += 15  # Move B point forward (x+15)
    
    # Define class distribution if not provided
    if class_distribution is None:
        class_distribution = [0.5, 0.3, 0.2]  # 50% Class 1, 30% Class 2, 20% Class 3
    
    # Calculate number of samples per class
    samples_per_class = [int(batch_size * p) for p in class_distribution]
    # Adjust for rounding errors
    samples_per_class[0] += batch_size - sum(samples_per_class)
    
    # Create ground truth data
    ground_truth = np.zeros((batch_size, 4, 2))  # 4 landmarks, 2 coordinates
    sample_idx = 0
    
    # Class 1 samples
    for i in range(samples_per_class[0]):
        # Add small random variations to base configuration
        ground_truth[sample_idx] = class1_config + np.random.normal(0, 3, size=(4, 2))
        sample_idx += 1
    
    # Class 2 samples
    for i in range(samples_per_class[1]):
        ground_truth[sample_idx] = class2_config + np.random.normal(0, 3, size=(4, 2))
        sample_idx += 1
    
    # Class 3 samples
    for i in range(samples_per_class[2]):
        ground_truth[sample_idx] = class3_config + np.random.normal(0, 3, size=(4, 2))
        sample_idx += 1
    
    # Create predictions with controlled error
    predictions = ground_truth.copy() + np.random.normal(0, error_level, size=ground_truth.shape)
    
    # Convert to torch tensors
    ground_truth_tensor = torch.tensor(ground_truth, dtype=torch.float32)
    predictions_tensor = torch.tensor(predictions, dtype=torch.float32)
    
    return ground_truth_tensor, predictions_tensor, landmark_cols

def test_classification_accuracy(error_level=5.0):
    """
    Test the accuracy of skeletal classification at different error levels
    
    Args:
        error_level (float): Baseline error level in pixels
    """
    output_dir = Path("skeletal_classification_test")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Test with different error levels
    error_levels = [1.0, 3.0, 5.0, 10.0, 15.0]
    accuracies = []
    anb_errors = []
    
    for error in error_levels:
        print(f"\nTesting with error level: {error:.1f} pixels")
        ground_truth, predictions, landmark_cols = create_sample_data(
            batch_size=100, 
            error_level=error,
            class_distribution=[0.4, 0.3, 0.3]  # More balanced distribution for testing
        )
        
        # Evaluate classification
        current_output_dir = output_dir / f"error_{error:.1f}"
        results = evaluate_skeletal_classification(
            predictions, 
            ground_truth, 
            landmark_cols,
            output_dir=current_output_dir
        )
        
        # Print results
        print(f"Classification accuracy: {results['accuracy']*100:.2f}%")
        print(f"Mean ANB angle error: {results['ANB_error_mean']:.2f}°")
        
        # Store results for plotting
        accuracies.append(results['accuracy'] * 100)
        anb_errors.append(results['ANB_error_mean'])
    
    # Plot accuracy vs error level
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(error_levels, accuracies, 'o-', linewidth=2)
    plt.xlabel('Landmark Error Level (pixels)')
    plt.ylabel('Classification Accuracy (%)')
    plt.title('Classification Accuracy vs. Landmark Error')
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(error_levels, anb_errors, 'o-', linewidth=2, color='orange')
    plt.xlabel('Landmark Error Level (pixels)')
    plt.ylabel('ANB Angle Error (degrees)')
    plt.title('ANB Angle Error vs. Landmark Error')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'error_impact_on_classification.png')
    plt.close()
    
    print(f"\nResults saved to {output_dir}")

def main():
    print("=== Testing Skeletal Classification Evaluation ===\n")
    
    # Create output directory
    output_dir = Path("skeletal_classification_test")
    output_dir.mkdir(exist_ok=True)
    
    # Generate sample data
    print("Generating sample data with controlled skeletal class distribution...")
    ground_truth, predictions, landmark_cols = create_sample_data(
        batch_size=50,
        error_level=3.0,  # Moderate error level
        class_distribution=[0.5, 0.3, 0.2]  # Class distribution
    )
    
    print(f"Generated {len(ground_truth)} samples")
    
    # Check that our ground truth represents the expected classes
    classifier = PatientClassifier(landmark_cols)
    
    # Convert to numpy for classification
    ground_truth_np = ground_truth.numpy()
    predictions_np = predictions.numpy()
    
    # Calculate angles from ground truth
    SNA_true = classifier.calculate_SNA_angle(ground_truth_np)
    SNB_true = classifier.calculate_SNB_angle(ground_truth_np)
    ANB_true = classifier.calculate_ANB_angle(SNA_true, SNB_true)
    classes_true = classifier.classify_ANB(ANB_true)
    
    # Print class distribution
    class_counts = np.bincount(classes_true, minlength=4)[1:]  # Skip class 0
    print("\nTrue class distribution:")
    for cls, count in enumerate(class_counts, 1):
        print(f"  Class {cls}: {count} samples ({count/len(ground_truth_np)*100:.1f}%)")
    
    print("\nRunning skeletal classification evaluation...")
    results = evaluate_skeletal_classification(
        predictions, 
        ground_truth, 
        landmark_cols,
        output_dir=output_dir
    )
    
    # Print detailed results
    print("\nClassification Metrics:")
    print(f"  Classification Accuracy: {results['accuracy']*100:.2f}%")
    print(f"  Mean SNA Angle Error: {results['SNA_error_mean']:.2f}°")
    print(f"  Mean SNB Angle Error: {results['SNB_error_mean']:.2f}°")
    print(f"  Mean ANB Angle Error: {results['ANB_error_mean']:.2f}°")
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    conf_matrix = results['confusion_matrix']
    print("  Predicted →")
    print("  ↓ True    Class 1  Class 2  Class 3")
    for i, row in enumerate(conf_matrix):
        print(f"  Class {i+1}    {row[0]:7d}  {row[1]:7d}  {row[2]:7d}")
    
    # Test impact of different error levels on classification accuracy
    print("\nTesting impact of landmark error on classification accuracy...")
    test_classification_accuracy()
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main() 