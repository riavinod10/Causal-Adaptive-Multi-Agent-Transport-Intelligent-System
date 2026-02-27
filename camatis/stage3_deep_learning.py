"""
Stage 3: Causal Dual-Attention Graph Transformer (CDAGT)
Core deep learning model with graph and temporal attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from camatis.config import *

class CausalGraphAttention(nn.Module):
    """Graph attention layer for route relationships"""
    def __init__(self, in_features, out_features, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.out_features = out_features
        
        self.W = nn.Linear(in_features, out_features * num_heads)
        self.a = nn.Parameter(torch.randn(num_heads, 2 * out_features))
        
    def forward(self, x, adj_matrix=None):
        batch_size = x.size(0)
        h = self.W(x).view(batch_size, self.num_heads, self.out_features)
        
        # Self-attention
        h_concat = torch.cat([h, h], dim=-1)  # [batch, num_heads, 2*out_features]
        e = torch.matmul(h_concat, self.a.t())  # [batch, num_heads, num_heads]
        attention = F.softmax(e, dim=-1)  # Softmax over last dimension
        
        # Apply attention: [batch, num_heads, num_heads] @ [batch, num_heads, out_features]
        h_prime = torch.bmm(attention, h)  # [batch, num_heads, out_features]
        h_prime = h_prime.mean(dim=1)  # Average over heads: [batch, out_features]
        return h_prime

class TemporalTransformer(nn.Module):
    """Temporal transformer for time-series patterns"""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        
    def forward(self, x):
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x

class CDAGT(nn.Module):
    """Causal Dual-Attention Graph Transformer"""
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, num_heads=4, dropout=0.2):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Graph attention layers
        self.graph_attention = CausalGraphAttention(hidden_dim, hidden_dim, num_heads)
        
        # Temporal transformer layers
        self.temporal_layers = nn.ModuleList([
            TemporalTransformer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Multi-task heads
        self.demand_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.load_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.utilization_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3)  # 3 classes
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mc_dropout=False):
        # Input projection
        h = self.input_projection(x)
        h = F.relu(h)
        
        # Graph attention
        h = self.graph_attention(h)
        h = self.dropout(h)
        
        # Temporal transformers
        for layer in self.temporal_layers:
            h = layer(h.unsqueeze(0)).squeeze(0)
            if mc_dropout or self.training:
                h = self.dropout(h)
        
        # Multi-task predictions
        demand_pred = self.demand_head(h).squeeze(-1)
        load_pred = self.load_head(h).squeeze(-1)
        util_pred = self.utilization_head(h)
        
        return demand_pred, load_pred, util_pred

class DeepLearningTrainer:
    def __init__(self, input_dim):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CDAGT(
            input_dim=input_dim,
            hidden_dim=DEEP_LEARNING_CONFIG['hidden_dim'],
            num_layers=DEEP_LEARNING_CONFIG['num_layers'],
            num_heads=DEEP_LEARNING_CONFIG['num_heads'],
            dropout=DEEP_LEARNING_CONFIG['dropout']
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=DEEP_LEARNING_CONFIG['learning_rate']
        )
        
        self.criterion_regression = nn.MSELoss()
        self.criterion_classification = nn.CrossEntropyLoss()
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the deep learning model"""
        print(f"Training on device: {self.device}")
        
        # Prepare data
        X_tensor = torch.FloatTensor(X_train).to(self.device)
        y_demand = torch.FloatTensor(y_train['passenger_demand']).to(self.device)
        y_load = torch.FloatTensor(y_train['load_factor']).to(self.device)
        y_util = torch.LongTensor(y_train['utilization_encoded']).to(self.device)
        
        dataset = TensorDataset(X_tensor, y_demand, y_load, y_util)
        dataloader = DataLoader(dataset, batch_size=DEEP_LEARNING_CONFIG['batch_size'], shuffle=True)
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(DEEP_LEARNING_CONFIG['epochs']):
            self.model.train()
            total_loss = 0
            
            for batch_X, batch_demand, batch_load, batch_util in dataloader:
                self.optimizer.zero_grad()
                
                # Forward pass
                pred_demand, pred_load, pred_util = self.model(batch_X)
                
                # Multi-task loss
                loss_demand = self.criterion_regression(pred_demand, batch_demand)
                loss_load = self.criterion_regression(pred_load, batch_load)
                loss_util = self.criterion_classification(pred_util, batch_util)
                
                loss = loss_demand + loss_load + loss_util
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{DEEP_LEARNING_CONFIG['epochs']}, Loss: {avg_loss:.4f}")
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), f"{MODELS_DIR}/cdagt_best.pth")
            else:
                patience_counter += 1
                if patience_counter >= DEEP_LEARNING_CONFIG['patience']:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        print("Training completed!")
        
    def predict(self, X):
        """Make predictions"""
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            pred_demand, pred_load, pred_util = self.model(X_tensor)
            pred_util_class = torch.argmax(pred_util, dim=1)
        
        return {
            'passenger_demand': pred_demand.cpu().numpy(),
            'load_factor': pred_load.cpu().numpy(),
            'utilization_encoded': pred_util_class.cpu().numpy()
        }
    
    def predict_with_uncertainty(self, X, n_samples=MC_DROPOUT_SAMPLES):
        """Predict with uncertainty using MC Dropout"""
        self.model.train()  # Enable dropout
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        demand_samples = []
        load_samples = []
        
        for _ in range(n_samples):
            with torch.no_grad():
                pred_demand, pred_load, _ = self.model(X_tensor, mc_dropout=True)
                demand_samples.append(pred_demand.cpu().numpy())
                load_samples.append(pred_load.cpu().numpy())
        
        demand_samples = np.array(demand_samples)
        load_samples = np.array(load_samples)
        
        return {
            'demand_mean': demand_samples.mean(axis=0),
            'demand_std': demand_samples.std(axis=0),
            'load_mean': load_samples.mean(axis=0),
            'load_std': load_samples.std(axis=0)
        }
