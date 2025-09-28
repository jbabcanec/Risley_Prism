#!/usr/bin/env python3
"""
TRANSFORMER NEURAL NETWORK - Next-generation pattern analysis

Revolutionary architecture using:
- Multi-head self-attention for pattern understanding
- Positional encoding for temporal dynamics
- Cross-attention between pattern and physics
- Hierarchical feature extraction
- Uncertainty quantification
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math

class PositionalEncoding(nn.Module):
    """Positional encoding for pattern sequences."""
    
    def __init__(self, d_model: int, max_length: int = 200):
        super().__init__()
        
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MultiHeadCrossAttention(nn.Module):
    """Cross-attention between pattern and physics features."""
    
    def __init__(self, d_model: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention
        attention_output = self._attention(Q, K, V, mask)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        return self.w_o(attention_output)
    
    def _attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        return torch.matmul(attention_weights, V)

class TransformerEncoderLayer(nn.Module):
    """Enhanced transformer encoder layer."""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.self_attention = nn.MultiheadAttention(d_model, num_heads, dropout, batch_first=True)
        self.cross_attention = MultiHeadCrossAttention(d_model, num_heads)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, physics_features=None):
        # Self-attention
        attn_output, _ = self.self_attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Cross-attention with physics (if available)
        if physics_features is not None:
            cross_output = self.cross_attention(x, physics_features, physics_features)
            x = self.norm2(x + self.dropout(cross_output))
        
        # Feed forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x

class PhysicsEmbedding(nn.Module):
    """Physics-aware embedding layer."""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        # Separate embeddings for different physics aspects
        self.position_embedding = nn.Linear(2, d_model // 4)  # x, y coordinates
        self.velocity_embedding = nn.Linear(2, d_model // 4)  # vx, vy
        self.acceleration_embedding = nn.Linear(2, d_model // 4)  # ax, ay
        self.frequency_embedding = nn.Linear(32, d_model // 4)  # FFT features
        
        self.projection = nn.Linear(d_model, d_model)
    
    def forward(self, pattern: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = pattern.shape
        
        # Extract physics features
        positions = pattern  # x, y positions
        
        # Calculate velocities (finite differences)
        velocities = torch.zeros_like(positions)
        velocities[:, 1:] = positions[:, 1:] - positions[:, :-1]
        
        # Calculate accelerations
        accelerations = torch.zeros_like(velocities)
        accelerations[:, 1:] = velocities[:, 1:] - velocities[:, :-1]
        
        # FFT features (simplified)
        fft_features = torch.zeros(batch_size, seq_len, 32, device=pattern.device)
        for i in range(batch_size):
            for j in range(2):  # x and y
                signal = pattern[i, :, j]
                fft = torch.fft.fft(signal)
                fft_features[i, :, j*16:(j+1)*16] = torch.abs(fft[:16]).unsqueeze(0).repeat(seq_len, 1)
        
        # Embed each physics aspect
        pos_emb = self.position_embedding(positions)
        vel_emb = self.velocity_embedding(velocities)
        acc_emb = self.acceleration_embedding(accelerations)
        freq_emb = self.frequency_embedding(fft_features)
        
        # Concatenate and project
        combined = torch.cat([pos_emb, vel_emb, acc_emb, freq_emb], dim=-1)
        return self.projection(combined)

class UncertaintyHead(nn.Module):
    """Uncertainty quantification head."""
    
    def __init__(self, d_model: int, output_size: int):
        super().__init__()
        self.mean_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, output_size)
        )
        
        self.variance_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, output_size),
            nn.Softplus()  # Ensure positive variance
        )
    
    def forward(self, x):
        mean = self.mean_head(x)
        variance = self.variance_head(x)
        return mean, variance

class TransformerPatternNet(nn.Module):
    """State-of-the-art Transformer network for pattern analysis."""
    
    def __init__(self, 
                 d_model: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 d_ff: int = 1024,
                 max_length: int = 100,
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        
        # Physics-aware embedding
        self.physics_embedding = PhysicsEmbedding(d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_length)
        
        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Global feature aggregation
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        # Classification head (wedge count)
        self.wedge_classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 6)  # 1-6 wedges
        )
        
        # Regression head with uncertainty
        self.parameter_regressor = UncertaintyHead(d_model * 2, 36)
        
        # Confidence estimator
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, pattern: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = pattern.shape
        
        # Physics-aware embedding
        x = self.physics_embedding(pattern)
        x = self.positional_encoding(x)
        
        # Transformer encoding
        physics_features = x.clone()  # Use as physics context
        
        for layer in self.encoder_layers:
            x = layer(x, physics_features)
        
        # Global feature aggregation
        x_avg = self.global_pool(x.transpose(1, 2)).squeeze(-1)
        x_max = self.global_max_pool(x.transpose(1, 2)).squeeze(-1)
        global_features = torch.cat([x_avg, x_max], dim=-1)
        
        # Multi-task outputs
        wedge_logits = self.wedge_classifier(global_features)
        param_mean, param_variance = self.parameter_regressor(global_features)
        confidence = self.confidence_head(global_features)
        
        return wedge_logits, param_mean, param_variance, confidence

class HierarchicalPatternAnalyzer(nn.Module):
    """Hierarchical analysis at multiple scales."""
    
    def __init__(self, scales: List[int] = [25, 50, 100]):
        super().__init__()
        self.scales = scales
        
        # Multi-scale transformers
        self.scale_networks = nn.ModuleList([
            TransformerPatternNet(d_model=128, num_layers=3, max_length=scale)
            for scale in scales
        ])
        
        # Fusion network
        self.fusion = nn.Sequential(
            nn.Linear(128 * 2 * len(scales), 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256)
        )
        
        # Final heads
        self.final_wedge_classifier = nn.Linear(256, 6)
        self.final_parameter_regressor = UncertaintyHead(256, 36)
        self.final_confidence = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, pattern: torch.Tensor):
        scale_features = []
        
        for i, (scale, network) in enumerate(zip(self.scales, self.scale_networks)):
            # Resample pattern to current scale
            if pattern.size(1) != scale:
                # Simple resampling (could use more sophisticated interpolation)
                indices = torch.linspace(0, pattern.size(1) - 1, scale, device=pattern.device).long()
                scaled_pattern = pattern[:, indices, :]
            else:
                scaled_pattern = pattern
            
            # Extract features at this scale
            wedge_logits, param_mean, param_var, confidence = network(scaled_pattern)
            
            # Combine predictions as features
            scale_feature = torch.cat([
                torch.softmax(wedge_logits, dim=-1),  # Wedge probabilities
                param_mean,  # Parameter predictions
                confidence   # Confidence scores
            ], dim=-1)
            
            scale_features.append(scale_feature)
        
        # Fuse multi-scale features
        fused_features = self.fusion(torch.cat(scale_features, dim=-1))
        
        # Final predictions
        final_wedge_logits = self.final_wedge_classifier(fused_features)
        final_param_mean, final_param_var = self.final_parameter_regressor(fused_features)
        final_confidence = self.final_confidence(fused_features)
        
        return final_wedge_logits, final_param_mean, final_param_var, final_confidence

class TransformerNeuralPredictor:
    """Transformer-based neural predictor with advanced capabilities."""
    
    def __init__(self, model_path: str = "weights/transformer_predictor.pth"):
        self.model_path = model_path
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Use hierarchical architecture for multi-scale analysis
        self.use_hierarchical = True
        
    def load(self) -> bool:
        """Load trained transformer model."""
        if not os.path.exists(self.model_path):
            return False
        
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            if self.use_hierarchical:
                self.model = HierarchicalPatternAnalyzer()
            else:
                self.model = TransformerPatternNet()
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            return True
        except Exception as e:
            print(f"❌ Failed to load transformer model: {e}")
            return False
    
    def predict(self, pattern: np.ndarray) -> Dict:
        """Predict with uncertainty quantification."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load() first.")
        
        # Prepare pattern
        if len(pattern.shape) == 1:
            pattern = pattern.reshape(-1, 2)
        
        # Convert to tensor
        pattern_tensor = torch.tensor(pattern, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            self.model.eval()
            
            if self.use_hierarchical:
                wedge_logits, param_mean, param_var, confidence = self.model(pattern_tensor)
            else:
                wedge_logits, param_mean, param_var, confidence = self.model(pattern_tensor)
            
            # Process predictions
            wedge_probs = F.softmax(wedge_logits, dim=-1)
            wedge_pred = torch.argmax(wedge_probs, dim=-1).item() + 1
            wedge_confidence = wedge_probs[0, wedge_pred - 1].item()
            
            # Parameter predictions with uncertainty
            param_predictions = param_mean[0].cpu().numpy()
            param_uncertainties = param_var[0].cpu().numpy()
            
            # Overall confidence
            overall_confidence = confidence[0].item()
        
        # Convert to parameter dictionary
        params = self._vector_to_parameters(param_predictions, wedge_pred)
        
        # Add uncertainty information
        params['prediction_confidence'] = {
            'wedge_count': wedge_confidence,
            'overall_confidence': overall_confidence,
            'parameter_uncertainties': param_uncertainties.tolist(),
            'uncertainty_score': float(np.mean(param_uncertainties))
        }
        
        return params
    
    def _vector_to_parameters(self, vector: np.ndarray, wedge_count: int) -> Dict:
        """Convert parameter vector to dictionary."""
        # Enhanced parameter extraction with bounds checking
        max_wedges = 6
        
        # Extract parameters for actual wedge count
        rotation_speeds = []
        phi_x = []
        phi_y = []
        
        for i in range(wedge_count):
            if i < max_wedges:
                # Apply proper scaling and bounds
                speed = np.tanh(vector[i]) * 5.0  # ±5 range
                px = np.tanh(vector[max_wedges + i]) * 20.0  # ±20 range
                py = np.tanh(vector[2*max_wedges + i]) * 20.0  # ±20 range
                
                rotation_speeds.append(float(speed))
                phi_x.append(float(px))
                phi_y.append(float(py))
        
        # Distances and refractive indices
        distances = [1.0]
        refractive_indices = [1.0]
        
        for i in range(wedge_count):
            dist = 5.0 + np.tanh(vector[3*max_wedges + i]) * 3.0  # 2-8 range
            ri = 1.5 + np.tanh(vector[4*max_wedges + i]) * 0.2   # 1.3-1.7 range
            
            distances.append(float(dist))
            refractive_indices.append(float(ri))
        
        refractive_indices.append(1.0)  # Final medium
        
        return {
            'rotation_speeds': rotation_speeds,
            'phi_x': phi_x,
            'phi_y': phi_y,
            'distances': distances,
            'refractive_indices': refractive_indices,
            'wedgenum': wedge_count
        }
    
    def get_attention_weights(self, pattern: np.ndarray) -> Dict:
        """Extract attention weights for interpretability."""
        if self.model is None or self.use_hierarchical:
            return {}
        
        # This would require modifying the forward pass to return attention weights
        # Implementation would depend on specific interpretability needs
        return {'attention_available': False}
    
    def get_training_info(self) -> Dict:
        """Get transformer model information."""
        if not os.path.exists(self.model_path):
            return {'trained': False}
        
        try:
            checkpoint = torch.load(self.model_path, map_location='cpu')
            return {
                'trained': True,
                'model_path': self.model_path,
                'architecture': 'Transformer with Multi-scale Analysis',
                'features': [
                    'Self-attention mechanisms',
                    'Physics-aware embeddings', 
                    'Hierarchical pattern analysis',
                    'Uncertainty quantification',
                    'Multi-objective optimization'
                ],
                'device': str(self.device),
                'hierarchical': self.use_hierarchical
            }
        except:
            return {'trained': False, 'error': 'Could not load model info'}