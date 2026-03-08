import torch
import numpy as np
import pandas as pd

class Predictor:
    def __init__(self, model, preprocessor, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.preprocessor = preprocessor
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def predict_single(self, transaction_data):
        """Predict fraud for a single transaction"""
        # Convert to DataFrame
        if isinstance(transaction_data, dict):
            df = pd.DataFrame([transaction_data])
        else:
            df = transaction_data
        
        # Preprocess
        X, _ = self.preprocessor.prepare_data(df, fit_scaler=False)
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probs = torch.sigmoid(outputs).cpu().numpy()
        
        return {
            'fraud_probability': float(probs[0][0]),
            'is_fraud': probs[0][0] > 0.5,
            'risk_level': self._get_risk_level(probs[0][0])
        }
    
    def predict_batch(self, transactions_df):
        """Predict fraud for multiple transactions"""
        # Preprocess
        X, _ = self.preprocessor.prepare_data(transactions_df, fit_scaler=False)
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probs = torch.sigmoid(outputs).cpu().numpy()
        
        results = transactions_df.copy()
        results['fraud_probability'] = probs
        results['predicted_fraud'] = (probs > 0.5).astype(int)
        results['risk_level'] = results['fraud_probability'].apply(self._get_risk_level)
        
        return results
    
    def _get_risk_level(self, probability):
        """Get risk level based on probability"""
        if probability < 0.3:
            return 'Low'
        elif probability < 0.7:
            return 'Medium'
        else:
            return 'High'
    
    def explain_prediction(self, transaction_data):
        """Simple explanation of prediction (SHAP not included)"""
        result = self.predict_single(transaction_data)
        
        explanation = f"""
        Fraud Detection Result:
        -----------------------
        Fraud Probability: {result['fraud_probability']:.2%}
        Risk Level: {result['risk_level']}
        Prediction: {'FRAUD' if result['is_fraud'] else 'LEGITIMATE'}
        
        Key Factors (based on model):
        - Amount: {transaction_data.get('amount', 'N/A')}
        - Category: {transaction_data.get('category', 'N/A')}
        - Merchant: {transaction_data.get('merchant', 'N/A')}
        """
        
        return explanation