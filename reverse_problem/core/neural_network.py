#!/usr/bin/env python3
"""
Neural Network for Risley Prism Reverse Problem

High-speed pattern-to-parameter prediction using deep learning.
Provides intelligent starting points for genetic algorithm refinement.
"""

import numpy as np
import pickle
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

@dataclass
class TrainingConfig:
    """Neural network training configuration."""
    batch_size: int = 64
    learning_rate: float = 0.001
    epochs: int = 200
    validation_split: float = 0.2
    early_stopping_patience: int = 20
    hidden_sizes: List[int] = None
    dropout_rate: float = 0.2
    
    def __post_init__(self):
        if self.hidden_sizes is None:
            self.hidden_sizes = [512, 256, 128, 64]

class PatternDataset(Dataset):
    """Dataset for pattern-to-parameter mapping."""
    
    def __init__(self, patterns: List[np.ndarray], parameters: List[Dict], 
                 pattern_length: int = 60):
        self.patterns = []
        self.parameters = []
        self.pattern_length = pattern_length
        
        for pattern, params in zip(patterns, parameters):
            # Normalize and pad/truncate patterns to fixed length
            normalized_pattern = self._normalize_pattern(pattern)
            self.patterns.append(normalized_pattern)
            
            # Convert parameters to normalized vector (FULL PARAMETERS)
            param_vector = self._parameters_to_vector(params)
            self.parameters.append(param_vector)
        
        self.patterns = torch.FloatTensor(np.array(self.patterns))
        self.parameters = torch.FloatTensor(np.array(self.parameters))
    
    def _normalize_pattern(self, pattern: np.ndarray) -> np.ndarray:
        """Normalize pattern to fixed length and scale."""
        # Ensure pattern is 2D (x, y coordinates)
        if len(pattern.shape) == 1:
            pattern = pattern.reshape(-1, 2)
        
        # Resize to fixed length
        if len(pattern) != self.pattern_length:
            # Interpolate to fixed length
            old_indices = np.linspace(0, len(pattern)-1, len(pattern))
            new_indices = np.linspace(0, len(pattern)-1, self.pattern_length)
            
            new_pattern = np.zeros((self.pattern_length, 2))
            new_pattern[:, 0] = np.interp(new_indices, old_indices, pattern[:, 0])
            new_pattern[:, 1] = np.interp(new_indices, old_indices, pattern[:, 1])
            pattern = new_pattern
        
        # Normalize coordinates
        # Center on mean
        pattern = pattern - np.mean(pattern, axis=0)
        
        # Scale by standard deviation
        std = np.std(pattern)
        if std > 0:
            pattern = pattern / std
        
        # Flatten for neural network input
        return pattern.flatten()
    
    def _parameters_to_vector(self, params: Dict) -> np.ndarray:
        """Convert parameter dictionary to normalized vector."""
        wedge_count = params.get('wedgenum', 1)
        
        # Fixed-size parameter vector (max 6 wedges)
        max_wedges = 6
        vector = np.zeros(max_wedges * 5 + 1)  # rotation_speeds + phi_x + phi_y + distances + refractive_indices + wedgenum
        
        # Rotation speeds (max 6)
        rotation_speeds = params.get('rotation_speeds', [])
        for i, speed in enumerate(rotation_speeds[:max_wedges]):
            vector[i] = speed / 10.0  # Normalize to [-1, 1]
        
        # Phi X (max 6)
        phi_x = params.get('phi_x', [])
        for i, angle in enumerate(phi_x[:max_wedges]):
            vector[max_wedges + i] = angle / 30.0  # Normalize to [-1, 1]
        
        # Phi Y (max 6)
        phi_y = params.get('phi_y', [])
        for i, angle in enumerate(phi_y[:max_wedges]):
            vector[2*max_wedges + i] = angle / 30.0  # Normalize to [-1, 1]
        
        # Distances (max 7: initial + 6 wedges)
        distances = params.get('distances', [1.0])
        for i, dist in enumerate(distances[:max_wedges+1]):
            if i == 0:
                continue  # Skip first distance (always 1.0)
            vector[3*max_wedges + i-1] = (dist - 5.0) / 5.0  # Normalize around 5.0
        
        # Refractive indices (skip first and last, always 1.0)
        refractive_indices = params.get('refractive_indices', [1.0, 1.5, 1.0])
        if len(refractive_indices) > 2:
            for i, ri in enumerate(refractive_indices[1:-1][:max_wedges]):
                vector[4*max_wedges + i] = (ri - 1.5) / 0.3  # Normalize around 1.5
        
        # Wedge count (normalized)
        vector[-1] = (wedge_count - 3.5) / 2.5  # Normalize to [-1, 1] for range 1-6
        
        return vector
    
    def __len__(self):
        return len(self.patterns)
    
    def __getitem__(self, idx):
        return self.patterns[idx], self.parameters[idx]

class PatternToParameterNet(nn.Module):
    """Neural network for pattern-to-parameter prediction."""
    
    def __init__(self, input_size: int = 120, output_size: int = 31, 
                 hidden_sizes: List[int] = None, dropout_rate: float = 0.2):
        super().__init__()
        
        if hidden_sizes is None:
            hidden_sizes = [512, 256, 128, 64]
        
        self.input_size = input_size
        self.output_size = output_size
        
        # Build network layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        return self.network(x)

class NeuralPredictor:
    """High-level neural network predictor interface."""
    
    def __init__(self, model_path: str = "weights/pattern_predictor.pth"):
        self.model_path = model_path
        self.model = None
        self.scaler_stats = None
        self.config = TrainingConfig()
        self.pattern_length = 60
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available. Install with: pip install torch")
    
    def train(self, training_data: List[Dict], validation_split: float = 0.2):
        """Train neural network on training data."""
        print("🧠 Training neural network predictor...")
        
        # Prepare data
        patterns = [np.array(sample['pattern']) for sample in training_data]
        parameters = [sample['parameters'] for sample in training_data]
        
        # Create dataset
        dataset = PatternDataset(patterns, parameters, self.pattern_length)
        
        # Split train/validation
        val_size = int(len(dataset) * validation_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
        
        # Initialize model
        input_size = self.pattern_length * 2  # x, y coordinates
        output_size = 31  # Full parameter vector (6 wedges * 5 params + wedgenum)
        
        self.model = PatternToParameterNet(
            input_size=input_size,
            output_size=output_size,
            hidden_sizes=self.config.hidden_sizes,
            dropout_rate=self.config.dropout_rate
        )
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        print(f"Training on {train_size} samples, validating on {val_size} samples")
        
        for epoch in range(self.config.epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_patterns, batch_params in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_patterns)
                loss = criterion(outputs, batch_params)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_patterns, batch_params in val_loader:
                    outputs = self.model(batch_patterns)
                    loss = criterion(outputs, batch_params)
                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'config': self.config,
                    'pattern_length': self.pattern_length,
                    'train_losses': train_losses,
                    'val_losses': val_losses
                }, self.model_path)
            else:
                patience_counter += 1
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
            
            # Early stopping
            if patience_counter >= self.config.early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        print(f"✅ Training complete! Best validation loss: {best_val_loss:.6f}")
        print(f"💾 Model saved to: {self.model_path}")
        
        return {
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'best_val_loss': best_val_loss,
            'epochs_trained': len(train_losses),
            'train_losses': train_losses,
            'val_losses': val_losses
        }
    
    def load(self) -> bool:
        """Load trained model."""
        if not os.path.exists(self.model_path):
            return False
        
        try:
            checkpoint = torch.load(self.model_path, map_location='cpu')
            
            # Reconstruct model
            input_size = self.pattern_length * 2
            output_size = 31
            
            self.model = PatternToParameterNet(
                input_size=input_size,
                output_size=output_size,
                hidden_sizes=checkpoint.get('config', self.config).hidden_sizes,
                dropout_rate=checkpoint.get('config', self.config).dropout_rate
            )
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False
    
    def predict(self, pattern: np.ndarray) -> Dict:
        """Predict parameters from pattern."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load() first.")
        
        # Prepare pattern
        dataset = PatternDataset([pattern], [{}], self.pattern_length)
        pattern_tensor = dataset.patterns[0:1]  # Batch of 1
        
        # Predict
        with torch.no_grad():
            self.model.eval()
            output = self.model(pattern_tensor)
            param_vector = output[0].numpy()
        
        # Convert back to full parameter dictionary
        params = self._vector_to_parameters(param_vector)
        
        return params
    
    def _vector_to_parameters(self, vector: np.ndarray) -> Dict:
        """Convert normalized vector back to parameter dictionary."""
        max_wedges = 6
        
        # Extract wedge count first
        wedge_count = int(np.round(vector[-1] * 2.5 + 3.5))
        wedge_count = np.clip(wedge_count, 1, 6)
        
        # Extract parameters
        rotation_speeds = [float(vector[i] * 10.0) for i in range(wedge_count)]
        phi_x = [float(vector[max_wedges + i] * 30.0) for i in range(wedge_count)]
        phi_y = [float(vector[2*max_wedges + i] * 30.0) for i in range(wedge_count)]
        
        # Distances
        distances = [1.0]  # First distance always 1.0
        for i in range(wedge_count):
            dist = vector[3*max_wedges + i] * 5.0 + 5.0
            distances.append(float(np.clip(dist, 1.0, 10.0)))
        
        # Refractive indices
        refractive_indices = [1.0]  # First always 1.0
        for i in range(wedge_count):
            ri = vector[4*max_wedges + i] * 0.3 + 1.5
            refractive_indices.append(float(np.clip(ri, 1.3, 1.8)))
        refractive_indices.append(1.0)  # Last always 1.0
        
        return {
            'rotation_speeds': rotation_speeds,
            'phi_x': phi_x,
            'phi_y': phi_y,
            'distances': distances,
            'refractive_indices': refractive_indices,
            'wedgenum': wedge_count
        }
    
    def get_training_info(self) -> Dict:
        """Get information about trained model."""
        if not os.path.exists(self.model_path):
            return {'trained': False}
        
        try:
            checkpoint = torch.load(self.model_path, map_location='cpu')
            return {
                'trained': True,
                'model_path': self.model_path,
                'train_losses': checkpoint.get('train_losses', []),
                'val_losses': checkpoint.get('val_losses', []),
                'epochs_trained': len(checkpoint.get('train_losses', [])),
                'final_train_loss': checkpoint.get('train_losses', [0])[-1] if checkpoint.get('train_losses') else 0,
                'final_val_loss': checkpoint.get('val_losses', [0])[-1] if checkpoint.get('val_losses') else 0,
            }
        except:
            return {'trained': False, 'error': 'Could not load training info'}