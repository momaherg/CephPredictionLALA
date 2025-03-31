# Learning Rate Range Test for Landmark Detection Model

This document explains the Learning Rate Range Test implementation and how to use it to find the optimal learning rate for the landmark detection model.

## What is a Learning Rate Range Test?

The Learning Rate Range Test (LR Range Test) is a technique proposed by Leslie Smith in the paper ["Cyclical Learning Rates for Training Neural Networks"](https://arxiv.org/abs/1506.01186). It helps to find the optimal learning rate for training neural networks by:

1. Starting with a very small learning rate (e.g., 1e-7)
2. Gradually increasing it (either exponentially or linearly) during a single training run
3. Plotting the loss against the learning rate
4. Identifying the optimal learning rate range based on this plot

This is particularly useful for one-cycle learning rate schedulers like `OneCycleLR` where setting the maximum learning rate parameter (`max_lr`) correctly is critical for optimal performance.

## Implementation

We've implemented the LR Range Test in two ways:

1. As a standalone Python script (`lr_range_test.py`)
2. Integrated within the main training scripts (`landmark_detection_training.py` and `src/train.py`)

### Using the Standalone Script

The standalone script provides more options for fine-tuning the test:

```bash
python lr_range_test.py --data_path /path/to/dataset.csv --output_dir ./outputs/lr_finder \
    --batch_size 16 --num_landmarks 19 --use_refinement \
    --start_lr 1e-7 --end_lr 1.0 --num_iter 100 --step_mode exp \
    --optimizer adam --skip_start 5 --skip_end 5
```

### Command-line Arguments

| Argument | Description |
|----------|-------------|
| `--data_path` | Path to the dataset CSV file (required) |
| `--output_dir` | Directory to save test results (default: ./outputs/lr_finder) |
| `--batch_size` | Batch size for the test (default: 16) |
| `--start_lr` | Starting learning rate (default: 1e-7) |
| `--end_lr` | Ending learning rate (default: 10.0) |
| `--num_iter` | Number of iterations to run (default: 100, 0 for full epoch) |
| `--step_mode` | Mode of increasing LR: 'exp' (exponential) or 'linear' (default: exp) |
| `--optimizer` | Optimizer to use: 'adam', 'adamw', or 'sgd' (default: adam) |
| `--skip_start` | Number of initial points to skip in plot (default: 5) |
| `--skip_end` | Number of final points to skip in plot (default: 5) |
| `--use_refinement` | Use refinement MLP in the model (flag) |
| `--num_landmarks` | Number of landmarks to detect (default: 19) |

### Using the Integrated Version

#### In the Jupyter Notebook (`landmark_detection_training.py`)

Set the following parameters at the top of the notebook:

```python
# LR Range Test parameters
RUN_LR_RANGE_TEST = True  # Enable the test
LR_TEST_START = 1e-7      # Starting learning rate
LR_TEST_END = 1.0         # Ending learning rate
LR_TEST_NUM_ITER = 100    # Number of iterations (0 for full epoch)
LR_TEST_STEP_MODE = 'exp' # 'exp' or 'linear'
```

#### Using the Command-line Interface (`src/train.py`)

```bash
python src/train.py --data_path /path/to/dataset.csv --output_dir ./outputs \
    --run_lr_test --lr_test_start 1e-7 --lr_test_end 1.0 --lr_test_iter 100 --lr_test_mode exp \
    --scheduler onecycle --num_epochs 50 --batch_size 16
```

## Interpreting the Results

The LR Range Test will generate a plot of the loss versus learning rate. The optimal learning rate is typically:

1. The point where the loss decreases the fastest (steepest negative slope)
2. Just before the loss starts to increase rapidly

Our implementation automatically suggests a learning rate based on finding the point of steepest descent. For the `OneCycleLR` scheduler, this value will be automatically used as the `max_lr` parameter if the test is run before training.

## Output Files

The test generates the following output:

1. `lr_finder.png` - A plot of the loss vs learning rate
2. `suggested_lr.txt` - A text file containing the suggested learning rate

## Best Practices

1. **Start conservatively**: If the suggested learning rate seems very high, you might want to use a fraction of it (e.g., half or a tenth) to be safe.
2. **Adjust batch size**: The optimal learning rate depends on batch size - larger batch sizes typically allow for larger learning rates.
3. **Short runs**: The test is designed to be quick - usually just a fraction of an epoch is sufficient to get good results.
4. **Check for divergence**: If the loss diverges too quickly, try a lower `end_lr` value. 