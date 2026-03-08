import torch
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def save_model(model, path, metadata=None):
    """Save model with metadata"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    save_dict = {
        'model_state_dict': model.state_dict(),
        'metadata': metadata or {}
    }
    
    torch.save(save_dict, path)
    print(f"Model saved to {path}")

def load_model(model_class, path, input_dim=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Load model from checkpoint"""
    checkpoint = torch.load(path, map_location=device)
    
    if input_dim is None and 'metadata' in checkpoint and 'input_dim' in checkpoint['metadata']:
        input_dim = checkpoint['metadata']['input_dim']
    
    model = model_class(input_dim=input_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded from {path}")
    return model

def save_metrics(metrics, path='graphs/metrics.json'):
    """Save evaluation metrics to JSON"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Convert numpy types to Python types
    serializable_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, np.floating):
            serializable_metrics[key] = float(value)
        elif isinstance(value, np.integer):
            serializable_metrics[key] = int(value)
        else:
            serializable_metrics[key] = value
    
    with open(path, 'w') as f:
        json.dump(serializable_metrics, f, indent=2)
    
    print(f"Metrics saved to {path}")

def load_metrics(path='graphs/metrics.json'):
    """Load evaluation metrics from JSON"""
    with open(path, 'r') as f:
        metrics = json.load(f)
    
    return metrics

def print_model_summary(model, input_dim):
    """Print model summary"""
    print("\n" + "="*60)
    print("MODEL SUMMARY")
    print("="*60)
    print(f"Model Type: {model.__class__.__name__}")
    print(f"Input Dimension: {input_dim}")
    print("\nLayers:")
    
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_params = param.numel()
            total_params += num_params
            print(f"  {name}: {tuple(param.shape)} ({num_params:,} parameters)")
    
    print(f"\nTotal trainable parameters: {total_params:,}")
    print("="*60)

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # For reproducibility on GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Random seed set to {seed}")

def get_timestamp():
    """Get current timestamp for file naming"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")