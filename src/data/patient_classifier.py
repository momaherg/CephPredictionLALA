import numpy as np
import pandas as pd
from sklearn.utils import resample

class PatientClassifier:
    """
    Classifier for cephalometric patient data based on skeletal measurements
    """
    def __init__(self, landmark_cols):
        """
        Initialize the patient classifier
        
        Args:
            landmark_cols (list): List of column names containing landmark coordinates
        """
        self.landmark_cols = landmark_cols
        
    def reshape_landmarks(self, df):
        """
        Reshape landmarks from DataFrame columns to 3D array
        
        Args:
            df (pandas.DataFrame): DataFrame containing landmark columns
            
        Returns:
            numpy.ndarray: Array of shape (n_samples, n_landmarks, 2)
        """
        # Extract landmark coordinates
        landmarks_df = df[self.landmark_cols].copy()
        
        # Number of landmarks
        num_landmarks = len(self.landmark_cols) // 2
        
        # Reshape landmarks to (n_samples, n_landmarks, 2)
        landmarks_array = np.zeros((len(df), num_landmarks, 2))
        for i in range(num_landmarks):
            x_col = self.landmark_cols[i*2]
            y_col = self.landmark_cols[i*2 + 1]
            landmarks_array[:, i, 0] = landmarks_df[x_col].values
            landmarks_array[:, i, 1] = landmarks_df[y_col].values
            
        return landmarks_array
    
    def calculate_SNA_angle(self, landmarks_array):
        """
        Calculate SNA angle from landmarks
        
        Args:
            landmarks_array (numpy.ndarray): Array of landmark coordinates of shape (batch_size, num_landmarks, 2)
            
        Returns:
            numpy.ndarray: SNA angles in degrees
        """
        sella = landmarks_array[:, 0, :]       # Index 0 (sella)
        nasion = landmarks_array[:, 1, :]      # Index 1 (nasion) 
        A_point = landmarks_array[:, 2, :]     # Index 2 (A point)

        NS_vector = nasion - sella
        NA_vector = A_point - nasion

        dot_product = np.sum(NS_vector * NA_vector, axis=1)
        NS_norm = np.linalg.norm(NS_vector, axis=1)
        NA_norm = np.linalg.norm(NA_vector, axis=1)
        cos_theta = dot_product / (NS_norm * NA_norm)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Prevent numerical errors
        angles = np.arccos(cos_theta) * 180 / np.pi

        return 180 - angles
    
    def calculate_SNB_angle(self, landmarks_array):
        """
        Calculate SNB angle from landmarks
        
        Args:
            landmarks_array (numpy.ndarray): Array of landmark coordinates of shape (batch_size, num_landmarks, 2)
            
        Returns:
            numpy.ndarray: SNB angles in degrees
        """
        sella = landmarks_array[:, 0, :]       # Index 0 (sella)
        nasion = landmarks_array[:, 1, :]      # Index 1 (nasion)
        B_point = landmarks_array[:, 3, :]     # Index 3 (B point)

        NS_vector = nasion - sella
        NB_vector = B_point - nasion

        dot_product = np.sum(NS_vector * NB_vector, axis=1)
        NS_norm = np.linalg.norm(NS_vector, axis=1)
        NB_norm = np.linalg.norm(NB_vector, axis=1)
        cos_theta = dot_product / (NS_norm * NB_norm)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Prevent numerical errors
        angles = np.arccos(cos_theta) * 180 / np.pi

        return 180 - angles
    
    def calculate_ANB_angle(self, SNA_angles, SNB_angles):
        """
        Calculate ANB angle from SNA and SNB angles
        
        Args:
            SNA_angles (numpy.ndarray): SNA angles in degrees
            SNB_angles (numpy.ndarray): SNB angles in degrees
            
        Returns:
            numpy.ndarray: ANB angles in degrees
        """
        return SNA_angles - SNB_angles
    
    def classify_ANB(self, ANB_angles):
        """
        Classify patients based on ANB angle
        
        Args:
            ANB_angles (numpy.ndarray): ANB angles in degrees
            
        Returns:
            numpy.ndarray: Class labels (1 for Class I, 2 for Class II, 3 for Class III)
        """
        classes = np.zeros_like(ANB_angles, dtype=int)
        classes[ANB_angles < 0] = 3      # Class III (retrognathic maxilla)
        classes[(ANB_angles >= 0) & (ANB_angles <= 4)] = 1  # Class I (normal)
        classes[ANB_angles > 4] = 2      # Class II (prognathic maxilla)
        return classes
    
    def classify_patients(self, df):
        """
        Add skeletal classification to DataFrame based on cephalometric measurements
        
        Args:
            df (pandas.DataFrame): DataFrame containing landmark data
            
        Returns:
            pandas.DataFrame: DataFrame with added skeletal classification columns
        """
        # Create a copy of the input DataFrame
        classified_df = df.copy()
        
        # Reshape landmarks
        landmarks_array = self.reshape_landmarks(classified_df)
        
        # Calculate angles
        SNA_angles = self.calculate_SNA_angle(landmarks_array)
        SNB_angles = self.calculate_SNB_angle(landmarks_array)
        ANB_angles = self.calculate_ANB_angle(SNA_angles, SNB_angles)
        
        # Classify patients
        skeletal_classes = self.classify_ANB(ANB_angles)
        
        # Add calculated values to DataFrame
        classified_df['SNA_angle'] = SNA_angles
        classified_df['SNB_angle'] = SNB_angles
        classified_df['ANB_angle'] = ANB_angles
        classified_df['skeletal_class'] = skeletal_classes
        
        # Print class distribution
        class_counts = classified_df['skeletal_class'].value_counts().sort_index()
        print("Skeletal Class Distribution:")
        for class_label, count in class_counts.items():
            class_name = {1: "Class I", 2: "Class II", 3: "Class III"}.get(class_label, f"Unknown ({class_label})")
            print(f"  {class_name}: {count} patients ({count/len(classified_df)*100:.1f}%)")
        
        return classified_df
    
    def balance_classes(self, df, class_column='skeletal_class', balance_method='upsample', stratify_column=None):
        """
        Balance the dataset based on skeletal class through upsampling or downsampling
        
        Args:
            df (pandas.DataFrame): DataFrame to balance
            class_column (str): Column containing class labels
            balance_method (str): Method to balance the dataset ('upsample' or 'downsample')
            stratify_column (str): Optional column to stratify within each class
            
        Returns:
            pandas.DataFrame: Balanced DataFrame
        """
        # Verify that the class column exists
        if class_column not in df.columns:
            raise ValueError(f"Column '{class_column}' not found in DataFrame")
        
        # Get unique classes and their counts
        class_counts = df[class_column].value_counts()
        print(f"Original class distribution: {class_counts.to_dict()}")
        
        # Determine target sample size
        if balance_method == 'upsample':
            # Use the size of the largest class
            target_size = class_counts.max()
        elif balance_method == 'downsample':
            # Use the size of the smallest class
            target_size = class_counts.min()
        else:
            raise ValueError(f"Unknown balance method: {balance_method}")
        
        # Balance the dataset
        balanced_dfs = []
        for class_label in class_counts.index:
            class_df = df[df[class_column] == class_label]
            
            if len(class_df) < target_size:
                # Upsample
                resampled_df = resample(
                    class_df,
                    replace=True,
                    n_samples=target_size,
                    random_state=42
                )
                balanced_dfs.append(resampled_df)
            elif len(class_df) > target_size and balance_method == 'downsample':
                # Downsample if that method was chosen
                resampled_df = resample(
                    class_df,
                    replace=False,
                    n_samples=target_size,
                    random_state=42
                )
                balanced_dfs.append(resampled_df)
            else:
                # Keep as is if using upsample and already at or above target size
                balanced_dfs.append(class_df)
        
        # Combine balanced classes
        balanced_df = pd.concat(balanced_dfs).reset_index(drop=True)
        
        # Verify new distribution
        new_class_counts = balanced_df[class_column].value_counts()
        print(f"Balanced class distribution: {new_class_counts.to_dict()}")
        
        return balanced_df 