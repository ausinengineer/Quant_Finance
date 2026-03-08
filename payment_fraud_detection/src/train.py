import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from tqdm import tqdm

class Trainer:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=None)
        self.optimizer = None
        self.scheduler = None
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_val_loss = float('inf')
        
    def set_optimizer(self, optimizer_type='adam', lr=0.001, weight_decay=1e-5):
        """Set optimizer"""
        if optimizer_type == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        elif optimizer_type == 'adamw':
            self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
        
        # Set scheduler
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    def set_class_weights(self, train_loader):
        """Set class weights for imbalanced data"""
        # Count samples per class
        n_samples = 0
        n_pos = 0
        for _, labels in train_loader:
            n_samples += len(labels)
            n_pos += labels.sum().item()
        
        n_neg = n_samples - n_pos
        pos_weight = torch.tensor([n_neg / n_pos]).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"Class weight (positive): {pos_weight.item():.2f}")
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for features, labels in tqdm(train_loader, desc="Training", leave=False):
            features = features.to(self.device)
            labels = labels.to(self.device).unsqueeze(1)
            
            # Forward pass
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Store predictions for accuracy
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            all_preds.extend(preds.cpu().detach().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return avg_loss, accuracy
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(self.device)
                labels = labels.to(self.device).unsqueeze(1)
                
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        # Calculate additional metrics
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        
        try:
            roc_auc = roc_auc_score(all_labels, all_probs)
        except:
            roc_auc = 0.5
        
        return avg_loss, accuracy, precision, recall, f1, roc_auc, all_labels, all_preds, all_probs
    
    def train(self, train_loader, val_loader, epochs=50, early_stopping_patience=10, 
              save_path='models/best_model.pth'):
        """Main training loop"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        early_stopping_counter = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            # Validate
            val_loss, val_acc, val_precision, val_recall, val_f1, val_roc_auc, _, _, _ = self.validate(val_loader)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step(val_loss)
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}")
            print(f"Val F1: {val_f1:.4f}, Val ROC-AUC: {val_roc_auc:.4f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_accuracy': val_acc,
                    'val_f1': val_f1,
                    'val_roc_auc': val_roc_auc
                }, save_path)
                print(f"Model saved to {save_path}")
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= early_stopping_patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        return self.train_losses, self.val_losses
    
    def plot_training_history(self, save_path='graphs/training_history.png'):
        """Plot training and validation losses"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot losses
        axes[0].plot(self.train_losses, label='Train Loss')
        axes[0].plot(self.val_losses, label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot accuracies
        axes[1].plot(self.train_accuracies, label='Train Accuracy')
        axes[1].plot(self.val_accuracies, label='Val Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.show()
        print(f"Training history plot saved to {save_path}")