# YOLOv11n Object Detection Model Fine-tuning

This repository demonstrates how to fine-tune YOLOv11n on multiple fire detection datasets. It provides a complete pipeline for combining multiple datasets from Roboflow, training a unified model, and evaluating its performance.

## Features
- Multiple dataset combination and preprocessing
- YOLOv11n model fine-tuning
- Comprehensive evaluation metrics
- Performance visualization tools
- Support for various image formats

## Prerequisites

### 1. Clone the Repository
```bash
git clone https://github.com/prakash-aryan/yolov11n-fire-detection.git
cd yolov11n-fire-detection
```

### 2. Set Up Python Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Required Packages
```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install packages
pip install ultralytics pandas matplotlib seaborn tqdm
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118  # For CUDA 11.8
# OR for CPU only
# pip install torch torchvision
```

### 4. Verify Installation
```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
```

Note: Always activate the virtual environment before working on the project:
```bash
# Windows: venv\Scripts\activate
# Linux/macOS: source venv/bin/activate
```

## Dataset Collection

### 1. Getting Roboflow Datasets
1. Create an account on [Roboflow](https://roboflow.com)
2. Create or find fire detection datasets
3. For each dataset:
   - Go to export page
   - Select "YOLOv11" format (this format is compatible with YOLOv11n)
   - Get the download URL:
   ```
   https://universe.roboflow.com/ds/XXXXX?key=YYYYY
   ```

### 2. Configure Dataset Downloads
1. Edit `prepare_datasets.sh`
2. Replace placeholder URLs with your Roboflow URLs:
```bash
download_and_extract "YOUR_ROBOFLOW_URL_1" "roboflow1.zip" 1
download_and_extract "YOUR_ROBOFLOW_URL_2" "roboflow2.zip" 2
```


## Project Structure

```
├── prepare_datasets.sh    # Dataset download and organization
├── prepare_dataset.py     # Combines dataset configurations
├── train_model.py         # YOLOv11n training script
├── evaluate_model.py      # Model evaluation
├── visualize_results.py   # Results visualization
└── datasets/             # Downloaded and processed datasets
    ├── train/
    ├── valid/
    ├── test/
    └── combined_data.yaml
```

## Usage Pipeline

### 1. Download and Prepare Datasets
```bash
# Make script executable
chmod +x prepare_datasets.sh

# Run dataset preparation (after adding your URLs)
./prepare_datasets.sh
```

### 2. Combine Dataset Configurations
```bash
python prepare_dataset.py
```
This creates a unified `combined_data.yaml` with:
- Combined class names
- Updated paths
- Merged dataset configurations

### 3. Train YOLOv11n Model
```bash
python train_model.py
```

Key training parameters:
```python
parameters = {
    'model': 'yolo11n.pt',  # Base YOLOv11n model
    'epochs': 100,
    'batch_size': 64,
    'imgsz': 640,
    'lr0': 0.001,
    'lrf': 0.01,
    'weight_decay': 0.0005,
    'warmup_epochs': 3,
    'device': 0  # GPU device id
}
```

### 4. Evaluate Model
```bash
python evaluate_model.py
```

This generates:
- Precision, recall metrics
- Confusion matrix
- mAP50 and mAP50-95 scores
- Training curves

### 5. Visualize Results
```bash
python visualize_results.py
```

## Model Architecture

We use YOLOv11n as our base model, which offers:
- Efficient architecture for edge devices
- Good balance of speed and accuracy
- Suitable for real-time fire detection

## Training Details

The training process includes:
1. Loading pretrained YOLOv11n weights
2. Fine-tuning on combined fire detection datasets
3. Automatic learning rate scheduling
4. Multi-scale training
5. Regular checkpointing

## Performance Metrics

The evaluation provides:
- Precision and Recall curves
- Confusion matrix
- mAP (mean Average Precision)
- Inference speed metrics
- Per-class performance analysis

