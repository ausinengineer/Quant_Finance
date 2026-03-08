import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import torch
from torch.utils.data import Dataset, DataLoader
import joblib
import os

class FraudDataset(Dataset):
    """Custom Dataset for fraud detection"""
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class DataPreprocessor:
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.categorical_cols = ['customer', 'merchant', 'category', 'gender', 'zipcodeOri', 'zipMerchant']
        self.numerical_cols = ['amount']
        self.age_mapping = {'U': 0, 'E': 0, '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6}
        
    def load_data(self, file_path=None):
        """Load data from CSV file"""
        if file_path:
            self.data_path = file_path
        if not self.data_path or not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        df = pd.read_csv(self.data_path)
        print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"Fraud cases: {df['fraud'].sum()} ({df['fraud'].mean()*100:.2f}%)")
        return df
    
    def clean_age_column(self, df):
        """Clean and map age column to numeric values"""
        df['age_clean'] = df['age'].map(self.age_mapping).fillna(0).astype(int)
        return df
    
    def encode_categorical(self, df, columns, fit=True):
        """Encode categorical columns"""
        for col in columns:
            if col in df.columns:
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    # Handle unseen categories
                    df[f'{col}_encoded'] = df[col].astype(str).map(
                        lambda x: self.label_encoders[col].transform([x])[0] 
                        if x in self.label_encoders[col].classes_ else -1
                    )
        return df
    
    def create_features(self, df):
        """Create additional features"""
        # Customer transaction frequency features
        customer_stats = df.groupby('customer_encoded').agg({
            'amount': ['count', 'mean', 'std']
        }).fillna(0)
        customer_stats.columns = ['customer_tx_count', 'customer_avg_amount', 'customer_std_amount']
        
        df = df.merge(customer_stats, left_on='customer_encoded', right_index=True, how='left')
        
        # Merchant features
        merchant_stats = df.groupby('merchant_encoded').agg({
            'amount': ['mean', 'std']
        }).fillna(0)
        merchant_stats.columns = ['merchant_avg_amount', 'merchant_std_amount']
        
        df = df.merge(merchant_stats, left_on='merchant_encoded', right_index=True, how='left')
        
        # Category features
        category_stats = df.groupby('category_encoded').agg({
            'amount': ['mean', 'std']
        }).fillna(0)
        category_stats.columns = ['category_avg_amount', 'category_std_amount']
        
        df = df.merge(category_stats, left_on='category_encoded', right_index=True, how='left')
        
        # Amount deviation features
        df['amount_dev_from_customer_avg'] = df['amount'] - df['customer_avg_amount']
        df['amount_dev_from_merchant_avg'] = df['amount'] - df['merchant_avg_amount']
        df['amount_dev_from_category_avg'] = df['amount'] - df['category_avg_amount']
        
        # Location match feature
        df['location_match'] = (df['zipcodeOri'] == df['zipMerchant']).astype(int)
        
        return df
    
    def prepare_data(self, df, fit_scaler=True):
        """Prepare data for training"""
        # Clean age column
        df = self.clean_age_column(df)
        
        # Encode categorical variables
        categorical_features = ['customer', 'merchant', 'category', 'gender']
        df = self.encode_categorical(df, categorical_features, fit=fit_scaler)
        
        # Create additional features
        df = self.create_features(df)
        
        # Define feature columns
        feature_cols = [
            'age_clean',
            'amount',
            'customer_tx_count',
            'customer_avg_amount',
            'customer_std_amount',
            'merchant_avg_amount',
            'merchant_std_amount',
            'category_avg_amount',
            'category_std_amount',
            'amount_dev_from_customer_avg',
            'amount_dev_from_merchant_avg',
            'amount_dev_from_category_avg',
            'location_match'
        ]
        
        # Add encoded categorical features
        for col in categorical_features:
            feature_cols.append(f'{col}_encoded')
        
        self.feature_columns = feature_cols
        
        # Extract features and target
        X = df[feature_cols].values
        y = df['fraud'].values
        
        # Handle NaN values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale features
        if fit_scaler:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2, val_size=0.1, random_state=42):
        """Split data into train, validation, and test sets"""
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Second split: train vs val
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
        )
        
        print(f"Train set: {X_train.shape[0]} samples, Fraud: {y_train.mean()*100:.2f}%")
        print(f"Validation set: {X_val.shape[0]} samples, Fraud: {y_val.mean()*100:.2f}%")
        print(f"Test set: {X_test.shape[0]} samples, Fraud: {y_test.mean()*100:.2f}%")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def apply_smote(self, X_train, y_train, random_state=42):
        """Apply SMOTE to handle class imbalance"""
        smote = SMOTE(random_state=random_state)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        print(f"After SMOTE: {X_train_resampled.shape[0]} samples, Fraud: {y_train_resampled.mean()*100:.2f}%")
        return X_train_resampled, y_train_resampled
    
    def create_dataloaders(self, X_train, y_train, X_val, y_val, X_test, y_test, 
                          batch_size=64, use_smote=True):
        """Create PyTorch DataLoaders"""
        # Apply SMOTE to training data if requested
        if use_smote:
            X_train, y_train = self.apply_smote(X_train, y_train)
        
        # Create datasets
        train_dataset = FraudDataset(X_train, y_train)
        val_dataset = FraudDataset(X_val, y_val)
        test_dataset = FraudDataset(X_test, y_test)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    def save_preprocessor(self, path='models/preprocessor.pkl'):
        """Save preprocessor objects"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'age_mapping': self.age_mapping
        }, path)
        print(f"Preprocessor saved to {path}")
    
    def load_preprocessor(self, path='models/preprocessor.pkl'):
        """Load preprocessor objects"""
        data = joblib.load(path)
        self.label_encoders = data['label_encoders']
        self.scaler = data['scaler']
        self.feature_columns = data['feature_columns']
        self.age_mapping = data['age_mapping']
        print(f"Preprocessor loaded from {path}")