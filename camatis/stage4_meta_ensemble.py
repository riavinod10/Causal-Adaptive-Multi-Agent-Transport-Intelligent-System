"""
Stage 4: Meta-Ensemble Stabilization
Combines deep learning with gradient boosting models
"""

import numpy as np
import lightgbm as lgb
from catboost import CatBoostRegressor, CatBoostClassifier
import xgboost as xgb
from sklearn.metrics import mean_squared_error, accuracy_score
import joblib
from camatis.config import *

class MetaEnsemble:
    def __init__(self):
        self.lgb_models = {}
        self.catboost_models = {}
        self.meta_models = {}
        
    def train_base_models(self, X_train, y_train):
        """Train LightGBM and CatBoost models"""
        print("Training LightGBM models...")
        
        # LightGBM for demand
        self.lgb_models['demand'] = lgb.LGBMRegressor(**LIGHTGBM_CONFIG, random_state=RANDOM_SEED)
        self.lgb_models['demand'].fit(X_train, y_train['passenger_demand'])
        
        # LightGBM for load factor
        self.lgb_models['load'] = lgb.LGBMRegressor(**LIGHTGBM_CONFIG, random_state=RANDOM_SEED)
        self.lgb_models['load'].fit(X_train, y_train['load_factor'])
        
        print("Training CatBoost models...")
        
        # CatBoost for demand
        self.catboost_models['demand'] = CatBoostRegressor(**CATBOOST_CONFIG, random_state=RANDOM_SEED)
        self.catboost_models['demand'].fit(X_train, y_train['passenger_demand'])
        
        # CatBoost for load factor
        self.catboost_models['load'] = CatBoostRegressor(**CATBOOST_CONFIG, random_state=RANDOM_SEED)
        self.catboost_models['load'].fit(X_train, y_train['load_factor'])
        
        # CatBoost for utilization classification
        self.catboost_models['utilization'] = CatBoostClassifier(**CATBOOST_CONFIG, random_state=RANDOM_SEED)
        self.catboost_models['utilization'].fit(X_train, y_train['utilization_encoded'])
        
        print("Base models trained!")
        
    def predict_base_models(self, X):
        """Get predictions from all base models"""
        predictions = {
            'lgb_demand': self.lgb_models['demand'].predict(X),
            'lgb_load': self.lgb_models['load'].predict(X),
            'catboost_demand': self.catboost_models['demand'].predict(X),
            'catboost_load': self.catboost_models['load'].predict(X),
            'catboost_utilization': self.catboost_models['utilization'].predict(X)
        }
        return predictions
    
    def train_meta_learner(self, X_train, y_train, dl_predictions):
        """
        Train XGBoost meta-learner that learns when DL vs ML is correct
        """
        print("Training meta-learner...")
        
        # Get base model predictions
        base_preds = self.predict_base_models(X_train)
        
        # Stack features: original features + base predictions + DL predictions
        meta_features_demand = np.column_stack([
            X_train,
            base_preds['lgb_demand'],
            base_preds['catboost_demand'],
            dl_predictions['passenger_demand']
        ])
        
        meta_features_load = np.column_stack([
            X_train,
            base_preds['lgb_load'],
            base_preds['catboost_load'],
            dl_predictions['load_factor']
        ])
        
        # Train meta-models
        self.meta_models['demand'] = xgb.XGBRegressor(**XGBOOST_META_CONFIG, random_state=RANDOM_SEED)
        self.meta_models['demand'].fit(meta_features_demand, y_train['passenger_demand'])
        
        self.meta_models['load'] = xgb.XGBRegressor(**XGBOOST_META_CONFIG, random_state=RANDOM_SEED)
        self.meta_models['load'].fit(meta_features_load, y_train['load_factor'])
        
        print("Meta-learner trained!")
        
    def predict_meta(self, X, dl_predictions):
        """Make final predictions using meta-learner"""
        base_preds = self.predict_base_models(X)
        
        # Stack features
        meta_features_demand = np.column_stack([
            X,
            base_preds['lgb_demand'],
            base_preds['catboost_demand'],
            dl_predictions['passenger_demand']
        ])
        
        meta_features_load = np.column_stack([
            X,
            base_preds['lgb_load'],
            base_preds['catboost_load'],
            dl_predictions['load_factor']
        ])
        
        # Meta predictions
        final_predictions = {
            'passenger_demand': self.meta_models['demand'].predict(meta_features_demand),
            'load_factor': self.meta_models['load'].predict(meta_features_load),
            'utilization_encoded': base_preds['catboost_utilization']
        }
        
        return final_predictions
    
    def save_models(self):
        """Save all ensemble models"""
        joblib.dump(self.lgb_models, f"{MODELS_DIR}/lgb_models.pkl")
        joblib.dump(self.catboost_models, f"{MODELS_DIR}/catboost_models.pkl")
        joblib.dump(self.meta_models, f"{MODELS_DIR}/meta_models.pkl")
        print("Ensemble models saved!")
    
    def evaluate(self, y_true, y_pred):
        """Evaluate predictions"""
        metrics = {
            'demand_rmse': np.sqrt(mean_squared_error(y_true['passenger_demand'], y_pred['passenger_demand'])),
            'load_rmse': np.sqrt(mean_squared_error(y_true['load_factor'], y_pred['load_factor'])),
            'utilization_accuracy': accuracy_score(y_true['utilization_encoded'], y_pred['utilization_encoded'])
        }
        return metrics
