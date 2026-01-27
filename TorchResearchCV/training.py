import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# avaliable metrics
AVAILABLE_METRICS = {
    'accuracy': accuracy_score,
    'precision': lambda y, p: precision_score(y, p, zero_division=0),
    'recall': lambda y, p: recall_score(y, p, zero_division=0),
    'f1': lambda y, p: f1_score(y, p, zero_division=0)
}

class PyTorchTraining:
    def __init__(self, model, optimizer, criterion, device=None, metrics=None):
        """For training and evaluation loop"""
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.metrics = metrics if metrics else {}

        self.metrics_fn = {}
        if metrics:
            for name in metrics:
                if name in AVAILABLE_METRICS:
                    self.metrics_fn[name] = AVAILABLE_METRICS[name]
                else:
                    print(f"Warning: Metric '{name}' is not supported.")
        
        # auto-detect device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        self.model.to(self.device)

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        for X_batch, y_batch in train_loader:

            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(X_batch)
            
            # dimensions of y
            loss = self.criterion(outputs, y_batch.unsqueeze(1))

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            
            # predictions for metrics
            preds = self._predictions(outputs)
            all_preds.append(preds.detach().cpu().numpy())
            all_targets.append(y_batch.detach().cpu().numpy())

        # metrics for epoch
        metrics_results = self._compute_metrics(np.concatenate(all_targets), np.concatenate(all_preds))
        metrics_results['loss'] = total_loss / len(train_loader)
        
        return metrics_results

    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                outputs = self.model(X_batch)
                
                loss = self.criterion(outputs, y_batch.unsqueeze(1))

                total_loss += loss.item()
                
                preds = self._predictions(outputs)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())

        metrics_results = self._compute_metrics(np.concatenate(all_targets), np.concatenate(all_preds))
        
        final_results = {}
        final_results['val_loss'] = total_loss / len(val_loader)

        for k, v in metrics_results.items():
            final_results[f"val_{k}"] = v
        
        return final_results

    def _predictions(self, outputs):
        probs = torch.sigmoid(outputs)
        return (probs >= 0.5).float().squeeze()

    def _compute_metrics(self, y_true, y_pred):
        results = {}
        for name, func in self.metrics_fn.items():
            results[name] = func(y_true, y_pred)
        return results