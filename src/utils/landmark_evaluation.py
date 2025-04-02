import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import seaborn as sns
import os
import warnings

def mean_euclidean_distance_per_landmark(predictions, targets):
    """
    Calculate Mean Euclidean Distance for each landmark individually
    
    Args:
        predictions (torch.Tensor or np.ndarray): Predicted landmark coordinates of shape (batch_size, num_landmarks, 2)
        targets (torch.Tensor or np.ndarray): Ground truth landmark coordinates of shape (batch_size, num_landmarks, 2)
        
    Returns:
        np.ndarray: Mean Euclidean Distance for each landmark of shape (num_landmarks,)
    """
    # Convert to numpy if needed
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    # Calculate Euclidean distance for each landmark
    squared_diff = (predictions - targets) ** 2
    # Sum over the x and y dimensions
    squared_dist = np.sum(squared_diff, axis=2)
    # Take the square root to get Euclidean distance
    distances = np.sqrt(squared_dist)
    
    # Calculate mean distance for each landmark over the batch
    med_per_landmark = np.mean(distances, axis=0)
    
    return med_per_landmark

def success_rate_per_landmark(predictions, targets, threshold=2.0):
    """
    Calculate the success rate for each landmark at a given threshold
    
    Args:
        predictions (torch.Tensor or np.ndarray): Predicted landmark coordinates of shape (batch_size, num_landmarks, 2)
        targets (torch.Tensor or np.ndarray): Ground truth landmark coordinates of shape (batch_size, num_landmarks, 2)
        threshold (float): Distance threshold for successful detection (in pixels)
        
    Returns:
        np.ndarray: Success rate for each landmark of shape (num_landmarks,)
    """
    # Convert to numpy if needed
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    # Calculate Euclidean distance for each landmark
    squared_diff = (predictions - targets) ** 2
    squared_dist = np.sum(squared_diff, axis=2)
    distances = np.sqrt(squared_dist)
    
    # Calculate success rate for each landmark (percentage of predictions within threshold)
    success = distances <= threshold
    success_rate_per_landmark = np.mean(success, axis=0)
    
    return success_rate_per_landmark

def plot_landmark_metrics(med_per_landmark, success_rate_per_landmark=None, landmark_names=None, output_dir=None, thresholds=None):
    """
    Create visualizations for per-landmark metrics
    
    Args:
        med_per_landmark (np.ndarray): Mean Euclidean Distance for each landmark
        success_rate_per_landmark (dict, optional): Success rate for each landmark at different thresholds
        landmark_names (list, optional): Names of the landmarks for better visualization
        output_dir (str, optional): Directory to save the plots
        thresholds (list, optional): List of thresholds used for success rates
    """
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
    
    # If landmark names are not provided, create generic names
    if landmark_names is None:
        landmark_names = [f"Landmark {i+1}" for i in range(len(med_per_landmark))]
    
    # Create a DataFrame for better plotting with seaborn
    df = pd.DataFrame({
        'Landmark': landmark_names,
        'MED (pixels)': med_per_landmark
    })
    
    # Sort by MED for better visualization
    df = df.sort_values('MED (pixels)', ascending=False)
    
    # Plot 1: Mean Euclidean Distance per landmark
    plt.figure(figsize=(14, 8))
    ax = sns.barplot(x='Landmark', y='MED (pixels)', data=df)
    plt.title('Mean Euclidean Distance (MED) per Landmark', fontsize=16)
    plt.xlabel('Landmark Name', fontsize=14)
    plt.ylabel('Mean Euclidean Distance (pixels)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add values on top of bars
    for i, p in enumerate(ax.patches):
        ax.annotate(f'{p.get_height():.2f}', 
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', va='bottom', fontsize=10, rotation=0)
    
    plt.tight_layout()
    if output_dir:
        plt.savefig(output_path / 'med_per_landmark.png')
    plt.close()
    
    # Plot 2: Success rate per landmark (if provided)
    if success_rate_per_landmark and thresholds:
        # Create a DataFrame for success rates
        success_df = pd.DataFrame({
            'Landmark': np.repeat(landmark_names, len(thresholds)),
            'Success Rate': np.concatenate([success_rate_per_landmark[t] for t in thresholds]),
            'Threshold': np.concatenate([[f"{t}mm"] * len(landmark_names) for t in thresholds])
        })
        
        plt.figure(figsize=(14, 8))
        ax = sns.barplot(x='Landmark', y='Success Rate', hue='Threshold', data=success_df)
        plt.title('Success Rate per Landmark at Different Thresholds', fontsize=16)
        plt.xlabel('Landmark Name', fontsize=14)
        plt.ylabel('Success Rate (%)', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.legend(title='Threshold')
        
        plt.tight_layout()
        if output_dir:
            plt.savefig(output_path / 'success_rate_per_landmark.png')
        plt.close()
    
    # Plot 3: Heatmap of MED values
    plt.figure(figsize=(12, 10))
    # Reshape the MED values to a matrix for heatmap
    num_landmarks = len(med_per_landmark)
    heatmap_size = int(np.ceil(np.sqrt(num_landmarks)))
    
    # Create a square matrix filled with NaN
    heatmap_data = np.full((heatmap_size, heatmap_size), np.nan)
    
    # Fill the matrix with MED values
    for i in range(num_landmarks):
        row = i // heatmap_size
        col = i % heatmap_size
        heatmap_data[row, col] = med_per_landmark[i]
    
    # Create a mask for NaN values
    mask = np.isnan(heatmap_data)
    
    # Create heatmap
    ax = sns.heatmap(heatmap_data, annot=True, fmt='.2f', mask=mask, cmap='YlOrRd', cbar_kws={'label': 'MED (pixels)'})
    plt.title('Mean Euclidean Distance (MED) Heatmap', fontsize=16)
    
    # Add landmark names as annotations
    for i in range(num_landmarks):
        row = i // heatmap_size
        col = i % heatmap_size
        plt.text(col + 0.5, row + 0.8, landmark_names[i], 
                 ha='center', va='center', fontsize=8, 
                 color='black' if heatmap_data[row, col] < np.nanmean(heatmap_data) else 'white')
    
    plt.tight_layout()
    if output_dir:
        plt.savefig(output_path / 'med_heatmap.png')
    plt.close()

def generate_landmark_evaluation_report(predictions, targets, landmark_names=None, 
                                          output_dir='./evaluation_reports', 
                                          thresholds=[2.0, 4.0, 6.0],
                                          landmark_cols=None,
                                          target_indices=None):
    """
    Generates a detailed evaluation report for landmark predictions.

    Args:
        predictions (torch.Tensor): Predicted landmarks (N, num_landmarks, 2).
        targets (torch.Tensor): Ground truth landmarks (N, num_landmarks, 2).
        landmark_names (list, optional): Names of the landmarks being evaluated.
        output_dir (str): Directory to save the report and plots.
        thresholds (list): List of thresholds (in pixels) for success rate calculation.
        landmark_cols (list, optional): List of ALL original landmark column names for skeletal classification.
        target_indices (list, optional): List of 0-based indices of the landmarks that were targeted.

    Returns:
        dict: Dictionary containing overall and per-landmark metrics.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Ensure tensors are numpy arrays on CPU
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
        
    num_samples, num_landmarks, _ = predictions.shape
    
    if landmark_names is None:
        landmark_names = [f'Landmark_{i+1}' for i in range(num_landmarks)]
    elif len(landmark_names) != num_landmarks:
        warnings.warn(f"Number of landmark names ({len(landmark_names)}) does not match number of predicted landmarks ({num_landmarks}). Using generic names.")
        landmark_names = [f'Landmark_{i+1}' for i in range(num_landmarks)]

    # 1. Calculate Euclidean Distances
    distances = np.sqrt(np.sum((predictions - targets)**2, axis=2))
    
    # 2. Overall Metrics
    overall_med = np.mean(distances)
    overall_std = np.std(distances)
    overall_success_rates = {f'{t}mm': np.mean(distances < t) for t in thresholds}
    
    # 3. Per-Landmark Metrics
    per_landmark_stats = []
    for i in range(num_landmarks):
        landmark_distances = distances[:, i]
        stats = {
            'name': landmark_names[i],
            'index_relative': i, # Index within the evaluated subset
            'index_original': target_indices[i] if target_indices is not None else i, # Original index if provided
            'med': np.mean(landmark_distances),
            'std': np.std(landmark_distances),
            'min': np.min(landmark_distances),
            'max': np.max(landmark_distances)
        }
        for t in thresholds:
            stats[f'success_rate_{t}mm'] = np.mean(landmark_distances < t)
        per_landmark_stats.append(stats)
        
    # Sort landmarks by MED (worst first)
    per_landmark_stats.sort(key=lambda x: x['med'], reverse=True)
    
    # Identify worst performing landmarks (relative indices)
    worst_landmarks_indices_relative = [stats['index_relative'] for stats in per_landmark_stats[:5]]
    worst_landmarks_med = [stats['med'] for stats in per_landmark_stats[:5]]
    
    # 4. Skeletal Classification (Optional)
    classification_report = None
    if landmark_cols is not None:
        try:
            from ..data.patient_classifier import PatientClassifier # Relative import
            classifier = PatientClassifier(landmark_cols) # Use all columns here
            
            # We need to map the predicted subset back to the full landmark array structure
            # required by the classifier. Assume classifier needs all landmarks.
            
            # Create placeholder arrays for full predictions and targets
            num_all_landmarks = len(landmark_cols) // 2
            full_predictions = np.full((num_samples, num_all_landmarks, 2), np.nan)
            full_targets = np.full((num_samples, num_all_landmarks, 2), np.nan)
            
            # Fill in the values for the landmarks that were predicted
            if target_indices is not None:
                for i, original_idx in enumerate(target_indices):
                    full_predictions[:, original_idx, :] = predictions[:, i, :]
                    full_targets[:, original_idx, :] = targets[:, i, :]
            else: # If target_indices is None, assume all were predicted
                full_predictions = predictions
                full_targets = targets
            
            # Create temporary dataframes for classification
            pred_data = {landmark_cols[j]: full_predictions[:, j // 2, j % 2] for j in range(len(landmark_cols))}
            target_data = {landmark_cols[j]: full_targets[:, j // 2, j % 2] for j in range(len(landmark_cols))}
            pred_df = pd.DataFrame(pred_data)
            target_df = pd.DataFrame(target_data)
            
            # Classify based on predictions and ground truth
            # Need to handle potential NaNs if required landmarks were not predicted
            pred_class_df = classifier.classify_patients(pred_df.dropna(subset=classifier.required_cols))
            target_class_df = classifier.classify_patients(target_df.dropna(subset=classifier.required_cols))
            
            # Calculate classification metrics if classification was possible
            if 'skeletal_class' in pred_class_df.columns and 'skeletal_class' in target_class_df.columns:
                classification_report = classifier.evaluate_classification(
                    pred_class_df, 
                    target_class_df
                )
            else:
                warnings.warn("Could not perform skeletal classification due to missing required landmarks in predictions or targets.")

        except ImportError:
            warnings.warn("PatientClassifier not found. Skipping skeletal classification evaluation.")
        except Exception as e:
            warnings.warn(f"Error during skeletal classification: {e}")

    # 5. Generate Report File
    report_path = os.path.join(output_dir, 'evaluation_report.txt')
    with open(report_path, 'w') as f:
        f.write("Cephalometric Landmark Detection Evaluation Report\n")
        f.write("====================================================\n\n")
        f.write(f"Total Samples Evaluated: {num_samples}\n")
        f.write(f"Number of Landmarks Evaluated: {num_landmarks}\n")
        if target_indices:
            f.write(f"Target Landmark Indices (original): {target_indices}\n")
            f.write(f"Target Landmark Names: {landmark_names}\n\n")
        else:
            f.write("Evaluated All Landmarks\n\n")
            
        f.write("Overall Metrics:\n")
        f.write(f"  Mean Euclidean Distance (MED): {overall_med:.4f} pixels\n")
        f.write(f"  Standard Deviation (SD): {overall_std:.4f} pixels\n")
        for t in thresholds:
            f.write(f"  Success Rate ({t}mm threshold): {overall_success_rates[f'{t}mm'] * 100:.2f}%\n")
        f.write("\n")
        
        if classification_report:
            f.write("Skeletal Classification Metrics:\n")
            f.write(f"  Classification Accuracy: {classification_report['accuracy'] * 100:.2f}%\n")
            f.write(f"  Mean ANB Angle Error: {classification_report['ANB_error_mean']:.2f}° (SD: {classification_report['ANB_error_std']:.2f}°)\n")
            f.write(f"  Mean SNA Angle Error: {classification_report['SNA_error_mean']:.2f}° (SD: {classification_report['SNA_error_std']:.2f}°)\n")
            f.write(f"  Mean SNB Angle Error: {classification_report['SNB_error_mean']:.2f}° (SD: {classification_report['SNB_error_std']:.2f}°)\n")
            # Include confusion matrix if desired
            # f.write("\nConfusion Matrix:\n")
            # f.write(str(classification_report['confusion_matrix']))
            f.write("\n")
            
        f.write("Per-Landmark Metrics (Sorted by MED, worst first):\n")
        header = "{:<5} {:<20} {:<10} {:<10} {:<10} {:<10}".format(
            "Idx", "Name", "MED", "StdDev", f"SR@{thresholds[0]}mm", f"SR@{thresholds[1]}mm"
        )
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for stats in per_landmark_stats:
            line = "{:<5} {:<20} {:<10.4f} {:<10.4f} {:<10.2f} {:<10.2f}".format(
                stats['index_original'], 
                stats['name'], 
                stats['med'], 
                stats['std'], 
                stats[f'success_rate_{thresholds[0]}mm'] * 100, 
                stats[f'success_rate_{thresholds[1]}mm'] * 100
            )
            f.write(line + "\n")
            
    print(f"Evaluation report saved to {report_path}")
    
    # 6. Generate Plots
    # Plot MED distribution
    plt.figure(figsize=(10, 6))
    plt.hist(distances.flatten(), bins=50, density=True)
    plt.title('Distribution of Euclidean Distances')
    plt.xlabel('Distance (pixels)')
    plt.ylabel('Density')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'distance_distribution.png'))
    plt.close()
    
    # Plot per-landmark MED
    landmark_indices_orig = [stats['index_original'] for stats in per_landmark_stats]
    med_values = [stats['med'] for stats in per_landmark_stats]
    plt.figure(figsize=(max(10, num_landmarks * 0.5), 6))
    plt.bar(range(num_landmarks), med_values)
    plt.xticks(range(num_landmarks), [landmark_names[stats['index_relative']] for stats in per_landmark_stats], rotation=90)
    plt.ylabel('MED (pixels)')
    plt.title('Mean Euclidean Distance per Landmark (Sorted)')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_landmark_med.png'))
    plt.close()

    # 7. Return results dictionary
    report_dict = {
        'overall_med': overall_med,
        'overall_std': overall_std,
        'overall_success_rates': overall_success_rates,
        'per_landmark_stats': per_landmark_stats,
        'worst_landmarks_indices_relative': worst_landmarks_indices_relative,
        'worst_landmarks_med': worst_landmarks_med,
        'classification': classification_report
    }
    
    return report_dict 