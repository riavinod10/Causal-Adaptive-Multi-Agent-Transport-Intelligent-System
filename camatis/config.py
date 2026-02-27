"""
CAMATIS Configuration
Causal-Adaptive Multi-Agent Transport Intelligence System
"""

import os

# Paths
DATA_DIR = "data"
MODELS_DIR = "camatis/models_saved"
RESULTS_DIR = "camatis/results"
LOGS_DIR = "camatis/logs"

# Create directories
for dir_path in [MODELS_DIR, RESULTS_DIR, LOGS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Data Configuration
TRAIN_FILE = os.path.join(DATA_DIR, "train_engineered.csv")
TEST_FILE = os.path.join(DATA_DIR, "test_engineered.csv")

# Feature Groups
CAUSAL_FEATURES = ['congestion_level', 'speed', 'is_peak', 'capacity']
TEMPORAL_FEATURES = ['hour', 'day_of_week', 'month', 'is_weekend', 'is_peak']
SPATIAL_FEATURES = ['route_id', 'route_avg_speed', 'route_avg_demand', 'route_avg_congestion']
OPERATIONAL_FEATURES = ['SRI', 'demand_capacity_ratio', 'speed_congestion_ratio', 'utilization_score']

# Target Variables (Multi-Task)
TARGET_VARIABLES = ['passenger_demand', 'load_factor', 'utilization_encoded']

# Model Hyperparameters
DEEP_LEARNING_CONFIG = {
    'hidden_dim': 128,
    'num_layers': 3,
    'num_heads': 4,
    'dropout': 0.2,
    'learning_rate': 0.001,
    'batch_size': 256,
    'epochs': 50,
    'patience': 10
}

LIGHTGBM_CONFIG = {
    'num_leaves': 31,
    'learning_rate': 0.05,
    'n_estimators': 200,
    'max_depth': -1,
    'min_child_samples': 20
}

CATBOOST_CONFIG = {
    'iterations': 200,
    'learning_rate': 0.05,
    'depth': 6,
    'l2_leaf_reg': 3,
    'verbose': False
}

XGBOOST_META_CONFIG = {
    'n_estimators': 100,
    'learning_rate': 0.05,
    'max_depth': 5,
    'subsample': 0.8
}

# Uncertainty Configuration
MC_DROPOUT_SAMPLES = 50
CONFIDENCE_LEVEL = 0.95

# Multi-Objective Optimization
OPTIMIZATION_OBJECTIVES = ['waiting_time', 'fuel_cost', 'utilization', 'fairness']
POPULATION_SIZE = 100
N_GENERATIONS = 50

# Simulation Configuration
SIMULATION_SCENARIOS = ['normal', 'festival_surge', 'congestion_spike', 'fleet_breakdown']
SIMULATION_DURATION = 24  # hours

# Random Seed
RANDOM_SEED = 42
