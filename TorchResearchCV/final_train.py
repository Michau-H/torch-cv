import torch
from torch.utils.data import DataLoader, TensorDataset
from .training import PyTorchTraining

class FinalModel:
    def __init__(self, device=None):
        """Class for training without validation dataset.
           Use only after CV with mean number of epochs"""
        self.device = device
        self.model = None
        self.pipeline = None

    def fit(self, 
            X, y, 
            model_init_fn, 
            optimizer_init_fn, 
            criterion_fn, 
            epochs, 
            pipeline_fn=None,
            batch_size=32,
            metrics=None,
            show_progress=True):
        """ Train model on 100% data"""
        
        # prepare data
        if hasattr(X, 'values'): X = X.values
        if hasattr(y, 'values'): y = y.values

        if pipeline_fn:
            self.pipeline = pipeline_fn()
            X = self.pipeline.fit_transform(X)
        
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        dataset = TensorDataset(X_t, y_t)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # functions
        self.model = model_init_fn()
        optimizer = optimizer_init_fn(self.model)
        criterion = criterion_fn()

        trainer = PyTorchTraining(
            model=self.model,
            optimizer=optimizer,
            criterion=criterion,
            device=self.device,
            metrics=metrics
        )

        # Train
        print(f"\nStarting Final Training for {epochs} epochs on {len(X)} samples")
        
        for epoch in range(epochs):
            train_res = trainer.train_epoch(train_loader)
            
            if show_progress:
                msg = f"Epoch {epoch+1}/{epochs} | Loss: {train_res['loss']:.4f}"
                if metrics:
                    for m in metrics:
                        msg += f" | {m}: {train_res.get(m, 0):.4f}"
                print(msg)

        return self.model

    def save_model(self, path):
        checkpoint = {
            'model': self.model.state_dict(),
        }
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")