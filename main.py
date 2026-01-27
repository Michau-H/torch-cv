import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from TorchResearchCV import CrossValidator, SimplePyTorch, FinalModel

# example data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# init functions
def get_pipeline():
    return Pipeline([
        ('scaler', StandardScaler())
    ])
def get_model():
    return SimplePyTorch(input_dim=20, hidden_layers=[64, 32], output_dim=1)
def get_optimizer(model):
    return optim.Adam(model.parameters(), lr=0.001)
def get_criterion():
    return nn.BCEWithLogitsLoss()

# cross-validation
cv = CrossValidator(n_splits=5, random_state=42)

history = cv.run(
    X=X, y=y,
    model_fn=get_model,
    optimizer_fn=get_optimizer,
    criterion_fn=get_criterion,
    pipeline_fn=get_pipeline,
    metrics=['accuracy', 'precision'],
    monitor_metric='accuracy',
    monitor_mode='max',
    epochs=50,
    patience=5
)

# plot
# cv.plot_history()

stopped_epochs = [fold['stopped_epoch'] for fold in history]
optimal_epochs = int(np.mean(stopped_epochs))

fm = FinalModel()

final_model = fm.fit(
    X, y,
    model_init_fn=get_model,
    optimizer_init_fn=get_optimizer,
    criterion_fn=get_criterion,
    pipeline_fn=get_pipeline,
    epochs=optimal_epochs,
    metrics=['accuracy']
)

# save
# fm.save_model("final_model.pt")




