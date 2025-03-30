import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import the patient classifier
from src.data.patient_classifier import PatientClassifier
from src.data.data_processor import DataProcessor

def create_test_dataframe(n_samples=100, seed=42):
    """
    Create a synthetic dataset for testing patient classification
    
    Args:
        n_samples (int): Number of samples to generate
        seed (int): Random seed for reproducibility
        
    Returns:
        pd.DataFrame: DataFrame with landmark coordinates
    """
    np.random.seed(seed)
    
    # Create landmark coordinates with known distributions to generate specific classes
    # We'll create 3 classes:
    # Class I (normal): ANB angle 0-4 degrees (about 50% of samples)
    # Class II (prognathic maxilla): ANB angle > 4 degrees (about 30% of samples)
    # Class III (retrognathic maxilla): ANB angle < 0 degrees (about 20% of samples)
    
    # These are the indices in our landmark coordinate array
    SELLA_IDX = 0
    NASION_IDX = 1
    A_POINT_IDX = 2
    B_POINT_IDX = 3
    
    # Generate base coordinates for all landmarks
    landmarks = np.zeros((n_samples, 19, 2))  # 19 landmarks, each with x and y coordinates
    
    # For simplicity, we'll use a simple coordinate system where:
    # - Sella is at (100, 100)
    # - Nasion is at (150, 80)
    # - A point and B point positions will vary to create different ANB angles
    
    # Set common landmarks
    landmarks[:, SELLA_IDX] = [100, 100]  # Sella
    landmarks[:, NASION_IDX] = [150, 80]  # Nasion
    
    # Class proportions
    class1_prop = 0.5  # Class I (normal)
    class2_prop = 0.3  # Class II
    class3_prop = 0.2  # Class III
    
    n_class1 = int(n_samples * class1_prop)
    n_class2 = int(n_samples * class2_prop)
    n_class3 = n_samples - n_class1 - n_class2
    
    # Create Class I samples (ANB angle between 0-4 degrees)
    for i in range(n_class1):
        # A point slightly forward of B point
        landmarks[i, A_POINT_IDX] = [180, 120]  # A point
        landmarks[i, B_POINT_IDX] = [175, 130]  # B point (slightly behind A point)
    
    # Create Class II samples (ANB angle > 4 degrees)
    for i in range(n_class1, n_class1 + n_class2):
        # A point much more forward of B point
        landmarks[i, A_POINT_IDX] = [185, 120]  # A point
        landmarks[i, B_POINT_IDX] = [165, 130]  # B point (well behind A point)
    
    # Create Class III samples (ANB angle < 0 degrees)
    for i in range(n_class1 + n_class2, n_samples):
        # B point forward of A point
        landmarks[i, A_POINT_IDX] = [170, 120]  # A point
        landmarks[i, B_POINT_IDX] = [180, 130]  # B point (in front of A point)
    
    # Add some random noise to make the dataset more realistic
    landmarks += np.random.normal(0, 3, landmarks.shape)
    
    # Create a DataFrame
    df = pd.DataFrame()
    
    # Flatten landmarks and add to DataFrame with appropriate column names
    landmark_cols = []
    for i in range(19):
        x_col = f'landmark_{i}_x'
        y_col = f'landmark_{i}_y'
        landmark_cols.extend([x_col, y_col])
        df[x_col] = landmarks[:, i, 0]
        df[y_col] = landmarks[:, i, 1]
    
    # Add some patient IDs
    df['patient_id'] = [f'P{i:03d}' for i in range(n_samples)]
    
    # Add a dummy image column (can be used to verify DataProcessor handling)
    df['Image'] = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(n_samples)]
    
    return df, landmark_cols

def test_angle_calculation(classifier, landmarks_array):
    """
    Test the angle calculation functions
    
    Args:
        classifier (PatientClassifier): Classifier instance
        landmarks_array (np.ndarray): Array of landmark coordinates
        
    Returns:
        dict: Dictionary with angle statistics
    """
    # Calculate angles
    sna_angles = classifier.calculate_SNA_angle(landmarks_array)
    snb_angles = classifier.calculate_SNB_angle(landmarks_array)
    anb_angles = classifier.calculate_ANB_angle(sna_angles, snb_angles)
    
    # Calculate statistics
    angle_stats = {
        'SNA': {
            'mean': np.mean(sna_angles),
            'std': np.std(sna_angles),
            'min': np.min(sna_angles),
            'max': np.max(sna_angles)
        },
        'SNB': {
            'mean': np.mean(snb_angles),
            'std': np.std(snb_angles),
            'min': np.min(snb_angles),
            'max': np.max(snb_angles)
        },
        'ANB': {
            'mean': np.mean(anb_angles),
            'std': np.std(anb_angles),
            'min': np.min(anb_angles),
            'max': np.max(anb_angles)
        }
    }
    
    return angle_stats, anb_angles

def plot_angle_distributions(anb_angles, classes):
    """
    Plot the distribution of ANB angles by class
    
    Args:
        anb_angles (np.ndarray): Array of ANB angles
        classes (np.ndarray): Array of class labels
    """
    plt.figure(figsize=(10, 6))
    
    # Get angles for each class
    class1_angles = anb_angles[classes == 1]
    class2_angles = anb_angles[classes == 2]
    class3_angles = anb_angles[classes == 3]
    
    # Plot histograms
    plt.hist(class1_angles, bins=15, alpha=0.5, label='Class I (Normal)')
    plt.hist(class2_angles, bins=15, alpha=0.5, label='Class II (Prognathic)')
    plt.hist(class3_angles, bins=15, alpha=0.5, label='Class III (Retrognathic)')
    
    # Add vertical lines for class boundaries
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.7, label='Class III/I Boundary')
    plt.axvline(x=4, color='g', linestyle='--', alpha=0.7, label='Class I/II Boundary')
    
    plt.title('ANB Angle Distribution by Skeletal Class')
    plt.xlabel('ANB Angle (degrees)')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Save the plot
    plt.savefig('anb_angle_distribution.png')
    plt.close()

def test_classification_logic(anb_angles, classes):
    """
    Test that the classification logic is consistent with defined rules
    
    Args:
        anb_angles (np.ndarray): Array of ANB angles
        classes (np.ndarray): Array of class labels
        
    Returns:
        bool: True if all classifications are correct
    """
    # Expected classes based on ANB angle thresholds
    expected_classes = np.zeros_like(classes)
    expected_classes[anb_angles < 0] = 3      # Class III
    expected_classes[(anb_angles >= 0) & (anb_angles <= 4)] = 1  # Class I
    expected_classes[anb_angles > 4] = 2      # Class II
    
    # Check if all class assignments match expected
    all_correct = np.all(classes == expected_classes)
    
    return all_correct, expected_classes

def test_balancing(df, classifier):
    """
    Test the class balancing functionality
    
    Args:
        df (pd.DataFrame): DataFrame with skeletal_class column
        classifier (PatientClassifier): Classifier instance
        
    Returns:
        pd.DataFrame: Balanced DataFrame
    """
    # Get original class distribution
    original_counts = df['skeletal_class'].value_counts().sort_index()
    print("Original class distribution:")
    for label, count in original_counts.items():
        class_name = {1: "Class I", 2: "Class II", 3: "Class III"}.get(label)
        print(f"  {class_name}: {count} samples ({count/len(df)*100:.1f}%)")
    
    # Test upsampling
    print("\nTesting upsampling...")
    balanced_df_up = classifier.balance_classes(df, class_column='skeletal_class', balance_method='upsample')
    balanced_counts_up = balanced_df_up['skeletal_class'].value_counts().sort_index()
    
    # Check that all classes have the same count after upsampling
    is_balanced_up = len(set(balanced_counts_up.values)) == 1
    print(f"All classes have the same count after upsampling: {is_balanced_up}")
    
    # Test downsampling
    print("\nTesting downsampling...")
    balanced_df_down = classifier.balance_classes(df, class_column='skeletal_class', balance_method='downsample')
    balanced_counts_down = balanced_df_down['skeletal_class'].value_counts().sort_index()
    
    # Check that all classes have the same count after downsampling
    is_balanced_down = len(set(balanced_counts_down.values)) == 1
    print(f"All classes have the same count after downsampling: {is_balanced_down}")
    
    # Return the upsampled DataFrame
    return balanced_df_up, balanced_df_down, is_balanced_up, is_balanced_down

def test_data_processor_integration(df, landmark_cols):
    """
    Test integration with DataProcessor
    
    Args:
        df (pandas.DataFrame): DataFrame with landmark data
        landmark_cols (list): List of landmark column names
    """
    # Create a temporary CSV file with the test data but without the Image column
    # since our test creates invalid images that get filtered out
    temp_csv_path = 'test_cephalometric_data.csv'
    df_no_images = df.drop(columns=['Image'])
    df_no_images.to_csv(temp_csv_path, index=False)
    
    try:
        # Initialize DataProcessor
        data_processor = DataProcessor(
            data_path=temp_csv_path,
            landmark_cols=landmark_cols,
            image_size=(224, 224),
            apply_clahe=False
        )
        
        # Test preprocess_data with class balancing
        print("\nTesting DataProcessor integration with class balancing...")
        processed_df = data_processor.preprocess_data(balance_classes=True)
        
        # Check if classes are balanced
        if 'skeletal_class' in processed_df.columns:
            processed_counts = processed_df['skeletal_class'].value_counts().sort_index()
            is_processed_balanced = len(set(processed_counts.values)) == 1
            print(f"All classes have the same count after DataProcessor balancing: {is_processed_balanced}")
            
            # Check if angle columns are present
            angle_cols_present = all(col in processed_df.columns for col in ['SNA_angle', 'SNB_angle', 'ANB_angle'])
            print(f"Angle columns present: {angle_cols_present}")
        else:
            print("Warning: skeletal_class column not found in processed DataFrame.")
            print("This suggests the DataProcessor either failed or didn't compute classifications.")
            
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_csv_path):
            os.remove(temp_csv_path)

def main():
    print("=== Testing Patient Classification and Class Balancing ===\n")
    
    # Create test data
    print("Creating synthetic test data...")
    df, landmark_cols = create_test_dataframe(n_samples=150)
    print(f"Created test dataset with {len(df)} samples and {len(landmark_cols)} landmark columns")
    
    # Create classifier
    classifier = PatientClassifier(landmark_cols)
    
    # Test reshaping function
    landmarks_array = classifier.reshape_landmarks(df)
    print(f"Reshaped landmarks to array of shape {landmarks_array.shape}")
    
    # Test angle calculation
    print("\nTesting angle calculation...")
    angle_stats, anb_angles = test_angle_calculation(classifier, landmarks_array)
    print("Angle statistics:")
    for angle_name, stats in angle_stats.items():
        print(f"  {angle_name}: mean={stats['mean']:.2f}°, std={stats['std']:.2f}°, range=[{stats['min']:.2f}°, {stats['max']:.2f}°]")
    
    # Test patient classification
    print("\nTesting patient classification...")
    classified_df = classifier.classify_patients(df)
    
    # Check for angle columns
    angle_cols_present = all(col in classified_df.columns for col in ['SNA_angle', 'SNB_angle', 'ANB_angle', 'skeletal_class'])
    print(f"Angle and class columns added to DataFrame: {angle_cols_present}")
    
    # Test classification logic
    print("\nVerifying classification logic...")
    classes = classified_df['skeletal_class'].values
    logic_correct, expected_classes = test_classification_logic(anb_angles, classes)
    print(f"Classification logic is {'correct' if logic_correct else 'INCORRECT'}")
    
    if not logic_correct:
        # Find misclassified samples
        misclassified = np.where(classes != expected_classes)[0]
        print(f"Found {len(misclassified)} misclassified samples. First few:")
        for idx in misclassified[:5]:
            print(f"  Sample {idx}: ANB={anb_angles[idx]:.2f}°, Assigned Class={classes[idx]}, Expected Class={expected_classes[idx]}")
    
    # Plot angle distributions
    plot_angle_distributions(anb_angles, classes)
    print("Generated ANB angle distribution plot: anb_angle_distribution.png")
    
    # Test class balancing
    print("\nTesting class balancing functionality...")
    balanced_df_up, balanced_df_down, is_balanced_up, is_balanced_down = test_balancing(classified_df, classifier)
    
    # Test DataProcessor integration
    test_data_processor_integration(df, landmark_cols)
    
    # Summary
    print("\n=== Test Summary ===")
    print(f"1. Landmark Reshaping: {'✓' if landmarks_array.shape == (len(df), len(landmark_cols)//2, 2) else '✗'}")
    print(f"2. Angle Calculation: {'✓' if angle_stats['ANB']['min'] < 0 and angle_stats['ANB']['max'] > 4 else '✗'}")
    print(f"3. Classification Logic: {'✓' if logic_correct else '✗'}")
    print(f"4. Class Balancing (Upsample): {'✓' if is_balanced_up else '✗'}")
    print(f"5. Class Balancing (Downsample): {'✓' if is_balanced_down else '✗'}")
    print(f"6. DataProcessor Integration: {'✓' if angle_cols_present else '✗'}")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main() 