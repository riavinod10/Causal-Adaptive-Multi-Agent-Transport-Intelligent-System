"""
Stage 1: Data Foundation
Load and prepare the engineered dataset
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from camatis.config import *

class DataLoader:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self):
        """Load train and test datasets"""
        print("Loading datasets...")
        train_df = pd.read_csv(TRAIN_FILE)
        test_df = pd.read_csv(TEST_FILE)
        
        print(f"Train shape: {train_df.shape}")
        print(f"Test shape: {test_df.shape}")
        
        return train_df, test_df
    
    def prepare_features(self, train_df, test_df):
        """Prepare feature matrices and target variables"""
        # Encode categorical variables
        for col in ['bus_id', 'utilization_status']:
            if col in train_df.columns:
                le = LabelEncoder()
                train_df[f'{col}_encoded'] = le.fit_transform(train_df[col].astype(str))
                test_df[f'{col}_encoded'] = le.transform(test_df[col].astype(str))
                self.label_encoders[col] = le
        
        # Select all numeric features
        feature_cols = (CAUSAL_FEATURES + TEMPORAL_FEATURES + 
                       SPATIAL_FEATURES + OPERATIONAL_FEATURES + 
                       ['bus_id_encoded'])
        
        # Remove duplicates
        feature_cols = list(dict.fromkeys(feature_cols))
        
        X_train = train_df[feature_cols].values
        X_test = test_df[feature_cols].values
        
        # Extract targets
        y_train = {
            'passenger_demand': train_df['passenger_demand'].values,
            'load_factor': train_df['load_factor'].values,
            'utilization_encoded': train_df['utilization_encoded'].values
        }
        
        y_test = {
            'passenger_demand': test_df['passenger_demand'].values,
            'load_factor': test_df['load_factor'].values,
            'utilization_encoded': test_df['utilization_encoded'].values
        }
        
        # Normalize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Feature matrix shape: {X_train_scaled.shape}")
        print(f"Number of features: {len(feature_cols)}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols
    
    def save_preprocessor(self):
        """Save scaler and encoders"""
        joblib.dump(self.scaler, f"{MODELS_DIR}/scaler.pkl")
        joblib.dump(self.label_encoders, f"{MODELS_DIR}/label_encoders.pkl")
        print("Preprocessors saved.")
