import copy
import numpy as np

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, mode='min'):
        """ Early stops the training"""
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.early_stop = False
        self.best_model = None
        
        # init best_score based on mode
        if self.mode == 'min':
            self.best_score = float('inf')
        else:
            self.best_score = -float('inf')

    def __call__(self, current_score, model):
        
        is_improvement = False
        
        if self.mode == 'min':
            if current_score < (self.best_score - self.min_delta):
                is_improvement = True
        else: # mode == 'max'
            if current_score > (self.best_score + self.min_delta):
                is_improvement = True

        if is_improvement:
            self.best_score = current_score
            self.best_model = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def restore_best_weights(self, model):
        """ Model weights from the best epoch"""
        if self.best_model is not None:
            model.load_state_dict(self.best_model)
        return model