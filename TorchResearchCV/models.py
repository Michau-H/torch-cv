import torch
import torch.nn as nn

class SimplePyTorch(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim=1, dropout_p=0.1, activation_fn=nn.ReLU):
        """ A simple PyTorch baseline model """
        super(SimplePyTorch, self).__init__()

        layers = []
        current_dim = input_dim

        for neurons in hidden_layers:
            layers.append(nn.Linear(current_dim, neurons))
            layers.append(activation_fn())
            
            if dropout_p > 0:
                layers.append(nn.Dropout(dropout_p))
            
            current_dim = neurons

        # Last layer
        layers.append(nn.Linear(current_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)