# Chromatographic Column Separation Prediction System QGeoGNN

## Project Overview
This project builds a chromatographic column separation behavior prediction system based on graph neural networks and traditional machine learning methods, comprising two core modules:

1. **Molecular Characterization Module** (QGeoGNN.py)
2. **Data Processing & Feature Engineering Module** (utils.py)

## System Architecture

### 1. Molecular Graph Neural Network Module (QGeoGNN.py)
```python
class GINGraphPooling(nn.Module):
    def __init__(self, num_tasks=2, ...):
        # Implements multi-task graph attention pooling
        # Integrates molecular geometric features with experimental parameters

class GINConv(MessagePassing):
    def forward(self, x, edge_index, edge_attr):
        # Graph Isomorphism Network convolution
        # Processes atom-bond-angle features
```

#### Core Features:
- **Molecular Graph Construction**: Converts SMILES to 3D molecular graphs
- **Geometric Feature Extraction**: 3D information including atomic coordinates, bond lengths, angles
- **Multimodal Fusion**: Molecular fingerprints + experimental parameters (flow rate, eluent ratio, etc.)

### 2. Data Processing Module (utils.py)
```python
def get_descriptor(smiles, ratio):
    # Calculates 6D molecular descriptors:
    # Molecular weight, polar surface area, rotatable bonds count, etc.

def convert_eluent(eluent):
    # Eluent feature engineering:
    # Converts PE/EA ratios to weighted sum of molecular descriptors

def read_data_CC():
    # Data loading & preprocessing:
    # Data cleaning, unit conversion, feature combination
```

#### Supported Data Types:
- Normal-phase chromatography (Silica-CS)
- Reverse-phase chromatography (C18/NH2/CN)
- Various column specifications (4g/8g/25g/40g)
- Multiple eluent systems (PE/EA, DCM/MeOH, etc.)

## Workflow

1. **Data Preparation Phase**
   - Load raw experimental data from Excel
   - Calculate molecular descriptors (Mordred 1826D)
   - Generate 3D molecular conformations

2. **Model Training Phase**
```python
# Main program logic (main.py)
if mode == 'QGeoGNN':
    # Invoke molecular graph neural network
    QGeoGNN(Data_CC, MODEL='Test')

elif mode == 'Train_XGB':
    # Traditional machine learning mode
    Model = Model_ML(config)
    model = Model.train(X_train, y_train)
```

3. **Prediction & Evaluation**
   - Multi-task output prediction (t1, t2 retention times)
   - Visualization of prediction results
   - Calculate MSE/RMSE/MAE/R² metrics

## Key Features

1. **Transfer Learning Framework**
```python
def QGeoGNN_transfer_C18(data, MODEL):
    # Cross-column migration prediction
    # Adapt pretrained models to new column types

def Construct_dataset_C18(...):
    # Dedicated builder for reverse-phase data
    # Handles MeOH/H2O eluent systems
```

2. **Separation Probability Calculation**
```python
def calculate_separation_probability_3compounds(...):
    # Based on predicted retention times
    # Calculate multi-component separation probabilities
```

## Data Interface

Input Data Requirements (Excel format):
| Column | Description | Example |
|--------|-------------|---------|
| smiles | Compound SMILES | CCO |
| t1/t2 | Retention times | 1.25 |
| PE/EA | Eluent ratio | 8/2 |
| Flow rate | Mobile phase speed | 20 |

Output Results:
- CSV prediction files
- PNG format prediction plots
- Model performance metrics logs

## Dependencies
- RDKit >= 2022.03
- PyTorch Geometric
- XGBoost/LightGBM
- Mordred 1.2.0

The system provides complete workflow for chromatographic prediction including molecular descriptor calculation, 3D conformation generation, and graph neural network training. Suitable for retention time prediction in pharmaceutical separation/purification and analytical chemistry applications.
```
