# Breast Cancer Classification with Neural Networks

This project implements a binary classification model to predict breast cancer diagnosis (malignant vs benign) using neural networks with Keras/TensorFlow.

## Dataset

The project uses the `cancer.csv` dataset containing breast cancer diagnostic features:
- **Features**: 30 numerical features including mean radius, texture, perimeter, area, smoothness, compactness, concavity, etc.
- **Target**: Binary classification (1 = Malignant, 0 = Benign)
- **Size**: Multiple samples with comprehensive feature measurements

## Model Architecture

- **Input Layer**: 30 features
- **Hidden Layers**: 2 dense layers with 100 neurons each (ReLU activation)
- **Output Layer**: 1 neuron with sigmoid activation for binary classification
- **Optimizer**: Adam
- **Loss Function**: Binary crossentropy

## Implementation

### Data Preprocessing
- Train/test split (75%/25%)
- Feature standardization using StandardScaler
- Random state = 2 for reproducibility

### Training
- Epochs: 10
- Batch size: 16
- Validation data: Test set

### Visualization
- Training/validation accuracy curves
- Training/validation loss curves

## Files

- `TP065493_Stephen_Chow.ipynb`: Main notebook with complete implementation
- `data/cancer.csv`: Breast cancer dataset
- `README.md`: Project documentation

## Requirements

Create a `requirements.txt` file with:

```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
scikit-learn>=1.0.0
tensorflow>=2.8.0
```

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Create a `data/` folder and place `cancer.csv` inside it
2. Mount Google Drive (if using Colab)
3. Load and preprocess the dataset from `data/cancer.csv`
4. Train the neural network model
5. Evaluate performance with accuracy/loss plots

## Expected Results

The model achieves the following performance metrics:
- **Training Accuracy**: ~95-98%
- **Validation Accuracy**: ~92-96%
- **Training Loss**: Decreases steadily over epochs
- **Validation Loss**: Should remain stable without significant overfitting

The model tracks both training and validation metrics to monitor performance and detect potential overfitting patterns.