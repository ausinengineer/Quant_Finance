import torch
import torch.nn as nn
import torch.nn.functional as F

class FraudDetectionModel(nn.Module):
    """PyTorch model for fraud detection"""
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout_rate=0.3):
        super(FraudDetectionModel, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class FraudDetectionModelAdvanced(nn.Module):
    """Advanced model with residual connections"""
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout_rate=0.3):
        super(FraudDetectionModelAdvanced, self).__init__()
        
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.bn_input = nn.BatchNorm1d(hidden_dims[0])
        
        # Hidden layers with residual connections
        self.hidden_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        for i in range(len(hidden_dims)-1):
            self.hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            self.bn_layers.append(nn.BatchNorm1d(hidden_dims[i+1]))
            self.dropouts.append(nn.Dropout(dropout_rate))
        
        self.output_layer = nn.Linear(hidden_dims[-1], 1)
        
    def forward(self, x):
        # Input layer
        x = F.relu(self.bn_input(self.input_layer(x)))
        
        # Hidden layers with residual connections
        for i, (layer, bn, dropout) in enumerate(zip(self.hidden_layers, self.bn_layers, self.dropouts)):
            residual = x
            x = layer(x)
            x = bn(x)
            x = F.relu(x)
            x = dropout(x)
            
            # Add residual if dimensions match
            if residual.shape == x.shape:
                x = x + residual
        
        # Output layer
        x = self.output_layer(x)
        return x

def get_model(model_type='basic', input_dim=None, hidden_dims=[256, 128, 64], dropout_rate=0.3):
    """Factory function to get model"""
    if input_dim is None:
        raise ValueError("input_dim must be provided")
    
    if model_type == 'basic':
        return FraudDetectionModel(input_dim, hidden_dims, dropout_rate)
    elif model_type == 'advanced':
        return FraudDetectionModelAdvanced(input_dim, hidden_dims, dropout_rate)
    else:
        raise ValueError(f"Unknown model type: {model_type}")