# TorchResearchCV
#### PyTorch cross validation framework designed for research in small dataset with binary classification.

When dataset is small and you have to sacriface some of data for test dataset, sometimes it is to high cost to divide train dataset into train and validation. 
In such situation `TorchResearchCV` is very helpful to find the best architecture and the optimal number of epochs, allowing you to eventually train the final model on 100% of your data.

---

## Classes

### **CrossValidator**
Main class to run Stratified K-Fold and track metrics.

| Parameter | Description | Default |
| :--- | :--- | :--- |
| `n_splits` | Number of folds for Stratified K-Fold | 5 |
| `random_state` | Seed for reproducibility | 42 |
| `shuffle` | Whether to shuffle data before splitting | True |

**Method: `run()`**

| Parameter | Description | Default |
| :--- | :--- | :--- |
| `X`, `y` | Input features and target labels | (required) |
| `model_fn` | Function that returns a new PyTorch model | (required) |
| `optimizer_fn` | Function that returns an optimizer for the model | (required) |
| `criterion_fn` | Function that returns the Loss Function | (required) |
| `pipeline_fn` | Function that returns a Scikit-learn Pipeline | None |
| `metrics` | List of metrics to track (e.g., `['accuracy', 'f1']`) | None |
| `epochs` | Maximum number of epochs per fold | 100 |
| `patience` | Epochs to wait for improvement (Early Stopping) | 10 |
| `monitor_metric` | Metric used to determine Early Stopping | 'loss' |
| `monitor_mode` | Mode for monitor ('min' or 'max') | 'min' |

**Method: `plot_history()`**

<img width="1308" height="648" alt="metrics" src="https://github.com/user-attachments/assets/0da7ab4d-f0bc-4c22-80f1-c3f65a0d9f30" />

---

### **FinalModel**
Training on 100% of data using results from Cross-Validation.

| Parameter | Description | Default |
| :--- | :--- | :--- |
| `device` | Device to train on ('cuda' or 'cpu') | None (Auto) |

**Method: `fit()`**

| Parameter | Description | Default |
| :--- | :--- | :--- |
| `X`, `y` | Full dataset features and labels | (required) |
| `epochs` | Number of epochs (suggested: `mean(stopped_epochs)`) | (required) |
| `model_init_fn` | Function that returns the model | (required) |
| `optimizer_init_fn` | Function that returns the optimizer | (required) |
| `criterion_fn` | Function that returns the Loss Function | (required) |

---

### **SimplePyTorch**
Simple PyTorch neural network.

| Parameter | Description | Default |
| :--- | :--- | :--- |
| `input_dim` | Number of input features | (required) |
| `hidden_layers` | List of neurons per hidden layer (e.g., `[64, 32]`) | (required) |
| `output_dim` | Number of output units | 1 |
| `dropout_p` | Dropout probability | 0.1 |
| `activation_fn` | PyTorch activation function class | nn.ReLU |

---

## Example Usage

```python
import torch.nn as nn
import torch.optim as optim
import numpy as np
from TorchResearchCV import CrossValidator, FinalModel, SimplePyTorch

def get_model():
    return SimplePyTorch(input_dim=20, hidden_layers=[64, 32], output_dim=1)

def get_optimizer(model):
    return optim.Adam(model.parameters(), lr=0.001)

cv = CrossValidator(n_splits=5)
history = cv.run(
    X, y, 
    model_fn=get_model, 
    optimizer_fn=get_optimizer,
    criterion_fn=nn.BCEWithLogitsLoss,
    metrics=['accuracy', 'f1'],
    epochs=50
)

cv.plot_history()

optimal_epochs = int(np.mean([fold['stopped_epoch'] for fold in history]))
fm = FinalModel()
final_model = fm.fit(X, y, get_model, get_optimizer, nn.BCEWithLogitsLoss, epochs=optimal_epochs)
```

---

## Download

```bash
git clone https://github.com/Michau-H/torch-cv
cd torch-cv
cd TorchResearchCV
pip install -e .
```
---

## Requirements

This project requires the following Python packages:

- **Python 3.8+**
- **PyTorch**
- **Scikit-learn**
- **NumPy**
- **Matplotlib**

You can install the required packages using pip:

```bash
pip install torch scikit-learn numpy matplotlib
```
or 
```bash
pip install -r requirements.txt
```

---

### License
MIT License © 2026 Michau-H
