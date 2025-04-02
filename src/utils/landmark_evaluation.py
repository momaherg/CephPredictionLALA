import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import sys
import os

# Add the project root to the path to import patient_classifier
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.data.patient_classifier import PatientClassifier

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

def evaluate_skeletal_classification(predictions, targets, landmark_cols, output_dir=None):
    """
    Evaluate the accuracy of skeletal classification based on predicted vs. true landmarks
    
    Args:
        predictions (torch.Tensor or np.ndarray): Predicted landmark coordinates of shape (batch_size, num_landmarks, 2)
        targets (torch.Tensor or np.ndarray): Ground truth landmark coordinates of shape (batch_size, num_landmarks, 2)
        landmark_cols (list): List of landmark column names
        output_dir (str, optional): Directory to save the evaluation plots
        
    Returns:
        dict: Dictionary containing classification metrics
    """
    # Convert to numpy if needed
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    # Create patient classifier instance
    classifier = PatientClassifier(landmark_cols)
    
    # Calculate cephalometric angles and classification from ground truth
    SNA_true = classifier.calculate_SNA_angle(targets)
    SNB_true = classifier.calculate_SNB_angle(targets)
    ANB_true = classifier.calculate_ANB_angle(SNA_true, SNB_true)
    class_true = classifier.classify_ANB(ANB_true)
    
    # Calculate cephalometric angles and classification from predictions
    SNA_pred = classifier.calculate_SNA_angle(predictions)
    SNB_pred = classifier.calculate_SNB_angle(predictions)
    ANB_pred = classifier.calculate_ANB_angle(SNA_pred, SNB_pred)
    class_pred = classifier.classify_ANB(ANB_pred)
    
    # Calculate classification accuracy
    accuracy = accuracy_score(class_true, class_pred)
    conf_matrix = confusion_matrix(class_true, class_pred)
    class_report = classification_report(class_true, class_pred, output_dict=True)
    
    # Calculate angle errors
    SNA_error = np.abs(SNA_true - SNA_pred)
    SNB_error = np.abs(SNB_true - SNB_pred)
    ANB_error = np.abs(ANB_true - ANB_pred)
    
    # Create result dictionary
    results = {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report,
        'SNA_error_mean': np.mean(SNA_error),
        'SNB_error_mean': np.mean(SNB_error),
        'ANB_error_mean': np.mean(ANB_error),
        'class_true': class_true,
        'class_pred': class_pred,
        'ANB_true': ANB_true,
        'ANB_pred': ANB_pred
    }
    
    # Create plots if output_dir is provided
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Class I', 'Class II', 'Class III'],
                   yticklabels=['Class I', 'Class II', 'Class III'])
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        plt.title('Skeletal Classification Confusion Matrix')
        plt.tight_layout()
        plt.savefig(output_path / 'class_confusion_matrix.png')
        plt.close()
        
        # Plot angle error distribution
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 3, 1)
        sns.histplot(SNA_error, kde=True)
        plt.title(f'SNA Angle Error (Mean: {np.mean(SNA_error):.2f}°)')
        plt.xlabel('Error (degrees)')
        
        plt.subplot(1, 3, 2)
        sns.histplot(SNB_error, kde=True)
        plt.title(f'SNB Angle Error (Mean: {np.mean(SNB_error):.2f}°)')
        plt.xlabel('Error (degrees)')
        
        plt.subplot(1, 3, 3)
        sns.histplot(ANB_error, kde=True)
        plt.title(f'ANB Angle Error (Mean: {np.mean(ANB_error):.2f}°)')
        plt.xlabel('Error (degrees)')
        
        plt.tight_layout()
        plt.savefig(output_path / 'angle_error_distribution.png')
        plt.close()
        
        # Plot true vs predicted ANB angles
        plt.figure(figsize=(10, 6))
        plt.scatter(ANB_true, ANB_pred, alpha=0.6)
        
        # Add diagonal perfect prediction line
        min_val = min(ANB_true.min(), ANB_pred.min()) - 1
        max_val = max(ANB_true.max(), ANB_pred.max()) + 1
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)
        
        # Add class boundary lines
        plt.axhline(y=0, color='g', linestyle='--', alpha=0.5)
        plt.axhline(y=4, color='g', linestyle='--', alpha=0.5)
        plt.axvline(x=0, color='g', linestyle='--', alpha=0.5)
        plt.axvline(x=4, color='g', linestyle='--', alpha=0.5)
        
        # Add annotations for class regions
        plt.text(min_val, (0+min_val)/2, 'Class III', fontsize=10, ha='left', va='center')
        plt.text(min_val, 2, 'Class I', fontsize=10, ha='left', va='center')
        plt.text(min_val, (4+max_val)/2, 'Class II', fontsize=10, ha='left', va='center')
        
        plt.xlabel('True ANB Angle (degrees)')
        plt.ylabel('Predicted ANB Angle (degrees)')
        plt.title('True vs. Predicted ANB Angle')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / 'anb_angle_comparison.png')
        plt.close()
        
        # Save numerical results to CSV
        results_df = pd.DataFrame({
            'SNA_true': SNA_true,
            'SNA_pred': SNA_pred,
            'SNA_error': SNA_error,
            'SNB_true': SNB_true,
            'SNB_pred': SNB_pred,
            'SNB_error': SNB_error,
            'ANB_true': ANB_true,
            'ANB_pred': ANB_pred,
            'ANB_error': ANB_error,
            'true_class': class_true,
            'predicted_class': class_pred,
            'correct_classification': class_true == class_pred
        })
        
        results_df.to_csv(output_path / 'skeletal_classification_results.csv', index=False)
    
    return results

def generate_landmark_evaluation_report(predictions, targets, landmark_names=None, output_dir=None, thresholds=None, landmark_cols=None):
    """
    Generate a comprehensive evaluation report for landmark detection
    
    Args:
        predictions (torch.Tensor or np.ndarray): Predicted landmark coordinates of shape (batch_size, num_landmarks, 2)
        targets (torch.Tensor or np.ndarray): Ground truth landmark coordinates of shape (batch_size, num_landmarks, 2)
        landmark_names (list, optional): Names of the landmarks
        output_dir (str, optional): Directory to save the report and plots
        thresholds (list, optional): List of thresholds for success rate calculation
        landmark_cols (list, optional): List of landmark column names (needed for skeletal classification)
        
    Returns:
        dict: Dictionary containing all evaluation metrics
    """
    if thresholds is None:
        thresholds = [2.0, 4.0, 6.0]  # Default thresholds in pixels
    
    # Calculate MED for each landmark
    med_per_landmark = mean_euclidean_distance_per_landmark(predictions, targets)
    
    # Calculate success rate for each landmark at different thresholds
    success_rates = {}
    for threshold in thresholds:
        success_rates[threshold] = success_rate_per_landmark(predictions, targets, threshold=threshold)
    
    # Calculate overall metrics
    overall_med = med_per_landmark.mean()
    overall_success_rates = {t: rate.mean() for t, rate in success_rates.items()}
    
    # Identify problematic landmarks (highest MED)
    sort_indices = np.argsort(med_per_landmark)[::-1]  # Sort in descending order
    worst_landmarks = sort_indices[:3]  # Top 3 worst landmarks
    
    # Create evaluation report
    report = {
        'overall_med': overall_med,
        'overall_success_rates': overall_success_rates,
        'med_per_landmark': med_per_landmark,
        'success_rates_per_landmark': success_rates,
        'worst_landmarks': worst_landmarks,
        'worst_landmarks_med': med_per_landmark[worst_landmarks]
    }
    
    # Add skeletal classification evaluation if landmark_cols is provided
    if landmark_cols is not None:
        try:
            classification_dir = None
            if output_dir:
                classification_dir = os.path.join(output_dir, 'classification')
            
            classification_results = evaluate_skeletal_classification(
                predictions, 
                targets, 
                landmark_cols, 
                output_dir=classification_dir
            )
            
            report['classification'] = classification_results
        except Exception as e:
            print(f"Warning: Could not evaluate skeletal classification: {e}")
    
    # Generate plots
    if output_dir:
        plot_landmark_metrics(
            med_per_landmark=med_per_landmark,
            success_rate_per_landmark=success_rates,
            landmark_names=landmark_names,
            output_dir=output_dir,
            thresholds=thresholds
        )
        
        # Save report as CSV
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Create a DataFrame with all metrics
        if landmark_names is None:
            landmark_names = [f"Landmark {i+1}" for i in range(len(med_per_landmark))]
        
        report_df = pd.DataFrame({
            'Landmark': landmark_names,
            'MED (pixels)': med_per_landmark
        })
        
        # Add success rates for each threshold
        for t in thresholds:
            report_df[f'Success Rate ({t}mm)'] = success_rates[t] * 100
        
        # Save to CSV
        report_df.to_csv(output_path / 'landmark_evaluation_report.csv', index=False)
        
        # Save summary as text
        with open(output_path / 'summary.txt', 'w') as f:
            f.write(f"Overall Mean Euclidean Distance (MED): {overall_med:.2f} pixels\n\n")
            f.write("Success Rates:\n")
            for t, rate in overall_success_rates.items():
                f.write(f"  {t}mm threshold: {rate*100:.2f}%\n")
            
            f.write("\nWorst Performing Landmarks:\n")
            for i, idx in enumerate(worst_landmarks):
                name = landmark_names[idx] if landmark_names else f"Landmark {idx+1}"
                f.write(f"  {i+1}. {name}: {med_per_landmark[idx]:.2f} pixels\n")
            
            # Add classification results if available
            if 'classification' in report:
                f.write("\nSkeletal Classification Results:\n")
                f.write(f"  Accuracy: {report['classification']['accuracy']*100:.2f}%\n")
                f.write(f"  Mean ANB Angle Error: {report['classification']['ANB_error_mean']:.2f}°\n")
    
    return report 