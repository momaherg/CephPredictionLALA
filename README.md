# Cephalometric Landmark Detection

A deep learning project for detecting cephalometric landmarks from lateral photographs of patients. This pipeline uses pretrained models to predict anatomical landmarks for orthodontic diagnosis without the need for X-rays.

## Dataset

The dataset comprises 1501 patient records with 47 columns, each capturing essential cephalometric landmarks and auxiliary information for orthodontic diagnosis. Images are lateral photographs of patients, and the model is designed to predict the coordinates of various anatomical landmarks used in orthodontic assessment.

Key landmarks include:
- Sella
- Nasion
- A point and B point
- Upper and lower incisor tips and apices
- ANS (Anterior Nasal Spine) and PNS (Posterior Nasal Spine)
- Gonion, Menton, Gnathion
- Soft tissue landmarks (ST Nasion, tip of the nose, subnasal, etc.)

## Project Structure

```
.
├── src
│   ├── data
│   │   ├── __init__.py
│   │   ├── dataset.py           # Dataset and DataLoader implementations
│   │   ├── data_processor.py    # Data loading and preprocessing
│   │   └── data_augmentation.py # Data augmentation techniques
│   ├── models
│   │   ├── __init__.py
│   │   └── landmark_detection.py # Model architecture
│   ├── utils
│   │   ├── __init__.py
│   │   └── metrics.py           # Evaluation metrics
│   ├── config
│   └── main.py                  # Main script
├── README.md
└── requirements.txt
```

## Features

- **Data Preprocessing**:
  - Image standardization using CLAHE (Contrast Limited Adaptive Histogram Equalization)
  - Proper dataset splits for training, validation, and testing
  - Image normalization and transformation

- **Data Augmentation**:
  - Random rotation with landmark adjustment
  - Random shifts
  - Brightness and contrast adjustments
  - Random blur
  - Optional horizontal flipping (use with caution for anatomical data)

- **Model Architecture**:
  - Pretrained backbone (ResNet50 or EfficientNet)
  - Custom regression head for landmark prediction
  - Support for different backbones

- **Evaluation Metrics**:
  - Mean Euclidean distance
  - Landmark success rate at different thresholds
  - Per-landmark performance analysis

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- OpenCV
- NumPy, Pandas, Matplotlib

### Installation

```bash
pip install -r requirements.txt
```

### Data Preparation

Prepare your dataset in either CSV or pickle format with the following columns:
- Patient identifiers (`patient`, `patient_id`)
- Image data (`Image` column containing pixel arrays) or paths to image files
- Landmark coordinates (e.g., `sella_x`, `sella_y`, `nasion_x`, `nasion_y`, etc.)
- Optional diagnostic class (`class`)

### Running the Pipeline

```bash
python src/main.py --data_path path/to/dataset.csv --output_dir ./outputs --batch_size 32 --apply_clahe --visualize_samples
```

Arguments:
- `--data_path`: Path to the dataset file (CSV or pickle)
- `--output_dir`: Directory to save outputs (default: ./outputs)
- `--batch_size`: Batch size for training (default: 32)
- `--apply_clahe`: Apply CLAHE for histogram equalization
- `--seed`: Random seed for reproducibility (default: 42)
- `--visualize_samples`: Visualize sample data and augmentations

## Extending the Project

### Adding New Backbones

To add a new backbone architecture, modify the `LandmarkDetectionModel` class in `src/models/landmark_detection.py` to include your desired backbone.

### Custom Augmentations

Create new augmentation classes in `src/data/data_augmentation.py` following the existing pattern.

### Training Script

Extend the main script to include the training loop, validation, and model saving functionality.

## License

This project is licensed under the MIT License.

## Acknowledgments

- [PyTorch](https://pytorch.org/) for the deep learning framework
- [torchvision](https://pytorch.org/vision/stable/index.html) for pretrained models
- [OpenCV](https://opencv.org/) for image processing 