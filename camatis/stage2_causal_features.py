"""
Stage 2: Causal Feature Intelligence
Build causal graph and generate counterfactual features
"""

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.linear_model import LinearRegression

class CausalFeatureEngine:
    def __init__(self):
        self.causal_graph = None
        self.causal_models = {}
        
    def build_causal_graph(self):
        """
        Construct causal dependency graph:
        Congestion → Speed → Demand → Load → Utilization
        """
        G = nx.DiGraph()
        
        # Define causal relationships
        edges = [
            ('congestion_level', 'speed'),
            ('speed', 'passenger_demand'),
            ('is_peak', 'passenger_demand'),
            ('passenger_demand', 'load_factor'),
            ('capacity', 'load_factor'),
            ('load_factor', 'utilization_encoded')
        ]
        
        G.add_edges_from(edges)
        self.causal_graph = G
        
        print("Causal graph constructed with nodes:", G.nodes())
        print("Causal edges:", G.edges())
        
        return G
    
    def learn_causal_relationships(self, train_df):
        """Learn causal effect models"""
        print("Learning causal relationships...")
        
        # Congestion → Speed
        X = train_df[['congestion_level']].values
        y = train_df['speed'].values
        model_cong_speed = LinearRegression().fit(X, y)
        self.causal_models['congestion_to_speed'] = model_cong_speed
        
        # Speed + Peak → Demand
        X = train_df[['speed', 'is_peak']].values
        y = train_df['passenger_demand'].values
        model_speed_demand = LinearRegression().fit(X, y)
        self.causal_models['speed_to_demand'] = model_speed_demand
        
        # Demand + Capacity → Load Factor
        X = train_df[['passenger_demand', 'capacity']].values
        y = train_df['load_factor'].values
        model_demand_load = LinearRegression().fit(X, y)
        self.causal_models['demand_to_load'] = model_demand_load
        
        print("Causal models learned.")
    
    def generate_counterfactual_features(self, df):
        """
        Generate counterfactual scenarios:
        'What if congestion was low instead of high?'
        """
        print("Generating counterfactual features...")
        
        df_cf = df.copy()
        
        # Counterfactual: Low congestion scenario
        cf_congestion = np.where(df['congestion_level'] > 1.0, 0.5, df['congestion_level'])
        cf_speed = self.causal_models['congestion_to_speed'].predict(cf_congestion.reshape(-1, 1))
        
        df_cf['cf_speed_low_congestion'] = cf_speed
        df_cf['cf_speed_diff'] = df['speed'] - cf_speed
        
        # Counterfactual: High capacity scenario
        cf_capacity = df['capacity'] * 1.2
        X_cf = np.column_stack([df['passenger_demand'], cf_capacity])
        cf_load = self.causal_models['demand_to_load'].predict(X_cf)
        
        df_cf['cf_load_high_capacity'] = cf_load
        df_cf['cf_load_diff'] = df['load_factor'] - cf_load
        
        print(f"Added {len([c for c in df_cf.columns if c.startswith('cf_')])} counterfactual features")
        
        return df_cf
    
    def extract_causal_features(self, X, feature_names):
        """Extract causal features from feature matrix"""
        causal_indices = [i for i, name in enumerate(feature_names) 
                         if name in CAUSAL_FEATURES]
        return X[:, causal_indices]
