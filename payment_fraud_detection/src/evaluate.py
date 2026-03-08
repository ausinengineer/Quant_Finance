import torch
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           roc_auc_score, confusion_matrix, classification_report,
                           precision_recall_curve, average_precision_score)
import matplotlib.pyplot as plt
import seaborn as sns
import os

class Evaluator:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
    
    def evaluate(self, test_loader, threshold=0.5):
        """Evaluate model on test data"""
        self.model.eval()
        all_preds = []
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for features, labels in test_loader:
                features = features.to(self.device)
                labels = labels.to(self.device).unsqueeze(1)
                
                outputs = self.model(features)
                probs = torch.sigmoid(outputs)
                preds = (probs > threshold).float()
                
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        all_labels = np.array(all_labels).flatten()
        all_preds = np.array(all_preds).flatten()
        all_probs = np.array(all_probs).flatten()
        
        return all_labels, all_preds, all_probs
    
    def calculate_metrics(self, y_true, y_pred, y_probs):
        """Calculate various metrics"""
        metrics = {}
        
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
        
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_probs)
        except:
            metrics['roc_auc'] = 0.5
        
        metrics['average_precision'] = average_precision_score(y_true, y_probs)
        
        return metrics
    
    def find_optimal_threshold(self, y_true, y_probs):
        """Find optimal threshold using precision-recall curve"""
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
        
        # F1 score for each threshold
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        
        optimal_idx = np.argmax(f1_scores[:-1])  # Exclude last threshold
        optimal_threshold = thresholds[optimal_idx]
        optimal_f1 = f1_scores[optimal_idx]
        
        print(f"Optimal threshold: {optimal_threshold:.4f} (F1: {optimal_f1:.4f})")
        
        return optimal_threshold, optimal_f1, precisions, recalls
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path='graphs/confusion_matrix.png'):
        """Plot confusion matrix"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Legitimate', 'Fraud'],
                    yticklabels=['Legitimate', 'Fraud'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.show()
        print(f"Confusion matrix saved to {save_path}")
    
    def plot_roc_curve(self, y_true, y_probs, save_path='graphs/roc_curve.png'):
        """Plot ROC curve"""
        from sklearn.metrics import roc_curve
        
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        roc_auc = roc_auc_score(y_true, y_probs)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.show()
        print(f"ROC curve saved to {save_path}")
    
    def plot_precision_recall_curve(self, y_true, y_probs, save_path='graphs/precision_recall_curve.png'):
        """Plot Precision-Recall curve"""
        precisions, recalls, _ = precision_recall_curve(y_true, y_probs)
        avg_precision = average_precision_score(y_true, y_probs)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recalls, precisions, color='green', lw=2, 
                label=f'PR curve (AP = {avg_precision:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.show()
        print(f"Precision-Recall curve saved to {save_path}")
    
    def plot_feature_importance(self, model, feature_names, save_path='graphs/feature_importance.png'):
        """Plot feature importance (for simple models)"""
        # This is a placeholder - feature importance depends on model architecture
        pass
    
    def print_classification_report(self, y_true, y_pred):
        """Print detailed classification report"""
        print("\n" + "="*60)
        print("CLASSIFICATION REPORT")
        print("="*60)
        print(classification_report(y_true, y_pred, target_names=['Legitimate', 'Fraud']))
        print("="*60)