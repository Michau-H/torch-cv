import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt


from .training import PyTorchTraining
from .early_stopping import EarlyStopping

class CrossValidator:
    def __init__(self, n_splits=5, random_state=42, shuffle=True):
        """ Stratified K-Fold Cross Validation for PyTorch models  """
        self.n_splits = n_splits
        self.random_state = random_state
        self.shuffle = shuffle
        self.skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        self.history = []
        self.fold_scores = []

    def run(self, 
            X, y, 
            model_fn, 
            optimizer_fn, 
            criterion_fn, 
            pipeline_fn, 
            metrics=None,
            epochs=100, 
            batch_size=32, 
            patience=10,
            monitor_metric='loss',
            monitor_mode='min',
            device=None):
        """ Cross-validation loop """

        if metrics is None: metrics = []
        
        if monitor_metric != 'loss' and monitor_metric not in metrics:
             raise ValueError(
                f"No '{monitor_metric}' on metrics list: {metrics}."
            )
        
        # handling numpy and pandas
        if hasattr(X, 'values'): X = X.values
        if hasattr(y, 'values'): y = y.values
        
        self.history = []
        
        print(f"\nStarting {self.n_splits} Fold Cross-Validation\n")

        # loop on folds
        for fold, (train_ids, val_ids) in enumerate(self.skf.split(X, y)):

            # prepare data
            X_train, X_val = X[train_ids], X[val_ids]
            y_train, y_val = y[train_ids], y[val_ids]

            if pipeline_fn:
                pipeline = pipeline_fn()
                X_train = pipeline.fit_transform(X_train)
                X_val = pipeline.transform(X_val)

            train_loader = self._create_dataloader(X_train, y_train, batch_size, shuffle=True)
            val_loader = self._create_dataloader(X_val, y_val, batch_size, shuffle=False)

            # functions
            model = model_fn()
            optimizer = optimizer_fn(model)
            criterion = criterion_fn()
            
            # init trainer
            trainer = PyTorchTraining(
                model=model, 
                optimizer=optimizer, 
                criterion=criterion, 
                device=device,
                metrics=metrics
            )

            # init early stoppping
            stopper = EarlyStopping(
                patience=patience, 
                min_delta=0.0, 
                mode=monitor_mode
            )

            # history (fold)
            fold_history = {'train_loss': [], 'val_loss': [], 'stopped_epoch': epochs}
            for m in metrics:
                fold_history[f'train_{m}'] = []
                fold_history[f'val_{m}'] = []

            # loop on epochs
            for epoch in range(epochs):

                train_res = trainer.train_epoch(train_loader)
                val_res = trainer.evaluate(val_loader)

                # metrics
                fold_history['train_loss'].append(train_res['loss'])
                fold_history['val_loss'].append(val_res['val_loss'])
                
                for k in metrics:
                    fold_history[f'train_{k}'].append(train_res.get(k, 0))
                    fold_history[f'val_{k}'].append(val_res.get(f"val_{k}", 0))

                # early stopping
                if monitor_metric == 'loss':
                    current_score = val_res['val_loss']
                else:
                    current_score = val_res.get(f"val_{monitor_metric}")

                stopper(current_score, model)

                if stopper.early_stop:
                    fold_history['stopped_epoch'] = epoch + 1
                    print(f"Early stopping: {epoch+1} epochs.")
                    break
            
            # best weights
            stopper.restore_best_weights(model)
            final_res = trainer.evaluate(val_loader)
            self.fold_scores.append(final_res)
            print(f"Fold {fold+1}/{self.n_splits} || Loss={final_res['val_loss']:.4f}", end=" ")
            for k in metrics:
                print(f"| {k}={final_res[f"val_{k}"]:.4f}", end=" ")
            print("\n")

            self.history.append(fold_history)
        self._print_summary(metrics)
        return self.history

    def _create_dataloader(self, X, y, batch_size, shuffle):
        """Create TensorDataset and DataLoader."""
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32) 
        dataset = TensorDataset(X_t, y_t)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def _print_summary(self, metrics_names):
        """Prints summary of metrics """
        print("\n" + "="*35)
        print("            CV results")
        print("="*35)

        # epochs
        epochs = [item['stopped_epoch'] for item in self.history]
        print(f"Average epochs: {np.mean(epochs):.4f} ± {np.std(epochs):.4f}")

        # loss
        losses = [item['val_loss'] for item in self.fold_scores]
        print(f"Average Loss: {np.mean(losses):.4f} ± {np.std(losses):.4f}")
        
        # other metrics
        for m in metrics_names:
            key = f"val_{m}"
            scores = [item.get(key, 0.0) for item in self.fold_scores]
            print(f"Average {m}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
        print("="*35)

    def plot_history(self):
        available_metrics = [k.replace('train_', '') for k in self.history[0].keys() if k.startswith('train_')]
        if 'loss' in available_metrics:
            available_metrics.remove('loss')
            available_metrics.insert(0, 'loss')

        n_metrics = len(available_metrics)
        n_folds = len(self.history)

        fig, axes = plt.subplots(
                nrows=n_metrics, 
                ncols=n_folds, 
                figsize=(4 * n_folds, 4 * n_metrics), 
                sharey='row', 
                squeeze=False 
            )

        for row_idx, metric in enumerate(available_metrics):
            for col_idx, fold_hist in enumerate(self.history):
                ax = axes[row_idx][col_idx]
                
                train_data = fold_hist.get(f'train_{metric}', [])
                val_data = fold_hist.get(f'val_{metric}', [])
                
                ax.plot(train_data, label='Train')
                ax.plot(val_data, label='Val', linestyle='--')
                if row_idx == 0:
                    ax.set_title(f'Fold {col_idx + 1}', fontsize=12)
                if col_idx == 0:
                    ax.set_ylabel(metric.capitalize(), fontsize=12)
                ax.grid(True, alpha=0.3)

                if col_idx == 0:
                    ax.legend()

        plt.tight_layout()
        plt.show()