#!/usr/bin/env python3
"""
SUPER-POWERED Neural Network for Risley Prism Reverse Problem

State-of-the-art deep learning architecture with:
- Advanced feature extraction
- Multi-scale pattern analysis
- Attention mechanisms
- Ensemble predictions
- Sophisticated data augmentation
"""

import numpy as np
import pickle
import os
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json
import time

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

@dataclass
class SuperTrainingConfig:
    """Enhanced training configuration for super-powered model."""
    batch_size: int = 32
    initial_lr: float = 0.001
    min_lr: float = 0.00001
    epochs: int = 500
    validation_split: float = 0.2
    early_stopping_patience: int = 50
    dropout_rate: float = 0.3
    weight_decay: float = 0.0001
    gradient_clip: float = 1.0
    warmup_epochs: int = 10
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.2
    
    # Architecture settings
    use_attention: bool = True
    use_residual: bool = True
    use_ensemble: bool = True
    ensemble_size: int = 3

class AdvancedPatternFeatureExtractor:
    """Extract comprehensive features from patterns."""
    
    @staticmethod
    def extract_features(pattern: np.ndarray) -> np.ndarray:
        """Extract rich feature set from pattern."""
        if len(pattern.shape) == 1:
            pattern = pattern.reshape(-1, 2)
        
        features = []
        
        # Basic statistics
        features.extend([
            np.mean(pattern[:, 0]),
            np.mean(pattern[:, 1]),
            np.std(pattern[:, 0]),
            np.std(pattern[:, 1]),
            np.min(pattern[:, 0]),
            np.max(pattern[:, 0]),
            np.min(pattern[:, 1]),
            np.max(pattern[:, 1])
        ])
        
        # Trajectory features
        if len(pattern) > 1:
            velocities = np.diff(pattern, axis=0)
            speeds = np.linalg.norm(velocities, axis=1)
            features.extend([
                np.mean(speeds),
                np.std(speeds),
                np.max(speeds),
                np.min(speeds)
            ])
            
            if len(pattern) > 2:
                accelerations = np.diff(velocities, axis=0)
                acc_magnitudes = np.linalg.norm(accelerations, axis=1)
                features.extend([
                    np.mean(acc_magnitudes),
                    np.std(acc_magnitudes),
                    np.max(acc_magnitudes)
                ])
            else:
                features.extend([0, 0, 0])
        else:
            features.extend([0, 0, 0, 0, 0, 0, 0])
        
        # Frequency domain features
        if len(pattern) > 3:
            fft_x = np.abs(np.fft.fft(pattern[:, 0]))[:len(pattern)//2]
            fft_y = np.abs(np.fft.fft(pattern[:, 1]))[:len(pattern)//2]
            
            # Find dominant frequencies
            top_freqs_x = np.argsort(fft_x)[-3:]
            top_freqs_y = np.argsort(fft_y)[-3:]
            
            features.extend([
                np.max(fft_x),
                np.mean(fft_x),
                float(top_freqs_x[-1]),
                float(top_freqs_x[-2] if len(top_freqs_x) > 1 else 0),
                np.max(fft_y),
                np.mean(fft_y),
                float(top_freqs_y[-1]),
                float(top_freqs_y[-2] if len(top_freqs_y) > 1 else 0)
            ])
        else:
            features.extend([0] * 8)
        
        # Shape features
        if len(pattern) > 4:
            # Curvature approximation
            curvatures = []
            for i in range(1, len(pattern) - 1):
                v1 = pattern[i] - pattern[i-1]
                v2 = pattern[i+1] - pattern[i]
                if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    curvatures.append(np.arccos(np.clip(cos_angle, -1, 1)))
            
            if curvatures:
                features.extend([
                    np.mean(curvatures),
                    np.std(curvatures),
                    np.max(curvatures)
                ])
            else:
                features.extend([0, 0, 0])
                
            # Convex hull area
            try:
                from scipy.spatial import ConvexHull
                hull = ConvexHull(pattern)
                features.append(hull.volume)  # Area in 2D
            except:
                features.append(0)
                
            # Pattern complexity (entropy-like measure)
            distances = np.linalg.norm(np.diff(pattern, axis=0), axis=1)
            if len(distances) > 0:
                hist, _ = np.histogram(distances, bins=10)
                hist = hist / hist.sum() + 1e-10
                entropy = -np.sum(hist * np.log(hist))
                features.append(entropy)
            else:
                features.append(0)
        else:
            features.extend([0, 0, 0, 0, 0])
        
        return np.array(features, dtype=np.float32)

class SuperPatternDataset(Dataset):
    """Enhanced dataset with data augmentation and advanced preprocessing."""
    
    def __init__(self, patterns: List[np.ndarray], parameters: List[Dict], 
                 pattern_length: int = 100, augment: bool = False):
        self.patterns = []
        self.features = []
        self.parameters = []
        self.pattern_length = pattern_length
        self.augment = augment
        
        # Feature extractor
        self.feature_extractor = AdvancedPatternFeatureExtractor()
        
        for pattern, params in zip(patterns, parameters):
            # Store both normalized pattern and extracted features
            normalized_pattern = self._normalize_pattern(pattern)
            features = self.feature_extractor.extract_features(pattern)
            
            self.patterns.append(normalized_pattern)
            self.features.append(features)
            
            # Enhanced parameter vector
            param_vector = self._parameters_to_enhanced_vector(params)
            self.parameters.append(param_vector)
        
        self.patterns = torch.FloatTensor(np.array(self.patterns))
        self.features = torch.FloatTensor(np.array(self.features))
        self.parameters = torch.FloatTensor(np.array(self.parameters))
    
    def _normalize_pattern(self, pattern: np.ndarray) -> np.ndarray:
        """Advanced pattern normalization."""
        if len(pattern.shape) == 1:
            pattern = pattern.reshape(-1, 2)
        
        # Resize to fixed length
        if len(pattern) != self.pattern_length:
            # Use numpy interp for simpler implementation
            old_indices = np.linspace(0, len(pattern)-1, len(pattern))
            new_indices = np.linspace(0, len(pattern)-1, self.pattern_length)
            
            new_pattern = np.zeros((self.pattern_length, 2))
            new_pattern[:, 0] = np.interp(new_indices, old_indices, pattern[:, 0])
            new_pattern[:, 1] = np.interp(new_indices, old_indices, pattern[:, 1])
            pattern = new_pattern
        
        # Robust normalization
        # Center pattern
        center = np.median(pattern, axis=0)
        pattern = pattern - center
        
        # Scale using robust statistics (less sensitive to outliers)
        scale = np.percentile(np.abs(pattern), 95)
        if scale > 0:
            pattern = pattern / scale
        
        # Clip extreme values
        pattern = np.clip(pattern, -3, 3)
        
        return pattern.flatten()
    
    def _parameters_to_enhanced_vector(self, params: Dict) -> np.ndarray:
        """Convert parameters with better encoding."""
        max_wedges = 6
        
        # Create one-hot encoding for wedge count
        wedge_count = params.get('wedgenum', 1)
        wedge_one_hot = np.zeros(max_wedges)
        wedge_one_hot[wedge_count - 1] = 1.0
        
        # Parameter vectors with better normalization
        vector = []
        
        # Add one-hot wedge encoding
        vector.extend(wedge_one_hot)
        
        # Rotation speeds (normalized with tanh-like scaling)
        rotation_speeds = params.get('rotation_speeds', [])
        for i in range(max_wedges):
            if i < len(rotation_speeds):
                # Use tanh-like normalization for better gradient flow
                normalized = np.tanh(rotation_speeds[i] / 5.0)
                vector.append(normalized)
            else:
                vector.append(0.0)
        
        # Phi X (normalized)
        phi_x = params.get('phi_x', [])
        for i in range(max_wedges):
            if i < len(phi_x):
                normalized = np.tanh(phi_x[i] / 20.0)
                vector.append(normalized)
            else:
                vector.append(0.0)
        
        # Phi Y (normalized)
        phi_y = params.get('phi_y', [])
        for i in range(max_wedges):
            if i < len(phi_y):
                normalized = np.tanh(phi_y[i] / 20.0)
                vector.append(normalized)
            else:
                vector.append(0.0)
        
        # Distances (normalized)
        distances = params.get('distances', [1.0])
        for i in range(max_wedges + 1):
            if i < len(distances):
                # Skip first distance (always 1.0)
                if i == 0:
                    continue
                normalized = (distances[i] - 5.0) / 3.0
                vector.append(np.tanh(normalized))
            else:
                vector.append(0.0)
        
        # Refractive indices
        refractive_indices = params.get('refractive_indices', [1.0, 1.5, 1.0])
        for i in range(1, max_wedges + 1):  # Skip first and last
            if i < len(refractive_indices) - 1:
                normalized = (refractive_indices[i] - 1.5) / 0.2
                vector.append(np.tanh(normalized))
            else:
                vector.append(0.0)
        
        return np.array(vector, dtype=np.float32)
    
    def __len__(self):
        return len(self.patterns)
    
    def __getitem__(self, idx):
        pattern = self.patterns[idx]
        features = self.features[idx]
        params = self.parameters[idx]
        
        # Data augmentation during training
        if self.augment and np.random.random() > 0.5:
            # Add noise
            noise_level = np.random.uniform(0, 0.05)
            pattern = pattern + torch.randn_like(pattern) * noise_level
            
            # Random scaling
            scale = np.random.uniform(0.9, 1.1)
            pattern = pattern * scale
        
        return pattern, features, params

class AttentionBlock(nn.Module):
    """Self-attention mechanism for pattern analysis."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        # Reshape for attention (batch, seq, features)
        batch_size = x.size(0)
        x_reshaped = x.view(batch_size, -1, x.size(-1))
        
        # Apply attention
        attn_out, _ = self.attention(x_reshaped, x_reshaped, x_reshaped)
        attn_out = self.dropout(attn_out)
        
        # Residual connection and normalization
        out = self.norm(x_reshaped + attn_out)
        
        # Reshape back
        return out.view(batch_size, -1)

class ResidualBlock(nn.Module):
    """Residual block for deep network training."""
    
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        # Skip connection
        self.skip = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()
    
    def forward(self, x):
        residual = self.skip(x)
        
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.bn2(out)
        
        out = out + residual
        out = self.relu(out)
        
        return out

class SuperPatternToParameterNet(nn.Module):
    """State-of-the-art neural network with attention and residual connections."""
    
    def __init__(self, pattern_size: int = 200, feature_size: int = 40, 
                 output_size: int = 36, dropout_rate: float = 0.3):
        super().__init__()
        
        # Pattern encoder with residual blocks
        self.pattern_encoder = nn.Sequential(
            nn.Linear(pattern_size, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            ResidualBlock(512, 512, dropout_rate),
            ResidualBlock(512, 256, dropout_rate),
            ResidualBlock(256, 256, dropout_rate),
            ResidualBlock(256, 128, dropout_rate)
        )
        
        # Feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_size, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64)
        )
        
        # Attention mechanism
        self.attention = AttentionBlock(128)
        
        # Fusion layer
        fusion_size = 128 + 64  # pattern + features
        self.fusion = nn.Sequential(
            nn.Linear(fusion_size, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate),
            ResidualBlock(256, 256, dropout_rate),
            ResidualBlock(256, 128, dropout_rate)
        )
        
        # Multi-task outputs
        # Wedge count classifier (6 classes)
        self.wedge_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 6)
        )
        
        # Parameter regressor
        self.parameter_regressor = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate),
            nn.Linear(256, output_size)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Improved weight initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, pattern, features):
        # Encode pattern
        pattern_encoded = self.pattern_encoder(pattern)
        
        # Apply attention
        pattern_attended = self.attention(pattern_encoded)
        
        # Encode features
        features_encoded = self.feature_encoder(features)
        
        # Fuse representations
        fused = torch.cat([pattern_attended, features_encoded], dim=1)
        fused = self.fusion(fused)
        
        # Multi-task outputs
        wedge_logits = self.wedge_classifier(fused)
        parameters = self.parameter_regressor(fused)
        
        return wedge_logits, parameters

class EnsembleModel(nn.Module):
    """Ensemble of models for improved predictions."""
    
    def __init__(self, num_models: int = 3, **model_kwargs):
        super().__init__()
        self.models = nn.ModuleList([
            SuperPatternToParameterNet(**model_kwargs)
            for _ in range(num_models)
        ])
    
    def forward(self, pattern, features):
        wedge_logits_list = []
        parameters_list = []
        
        for model in self.models:
            wedge_logits, parameters = model(pattern, features)
            wedge_logits_list.append(wedge_logits)
            parameters_list.append(parameters)
        
        # Average predictions
        wedge_logits = torch.stack(wedge_logits_list).mean(dim=0)
        parameters = torch.stack(parameters_list).mean(dim=0)
        
        return wedge_logits, parameters

class SuperNeuralPredictor:
    """Super-powered neural network predictor with advanced training."""
    
    def __init__(self, model_path: str = "weights/super_pattern_predictor.pth"):
        self.model_path = model_path
        self.model = None
        self.config = SuperTrainingConfig()
        self.pattern_length = 100
        self.feature_extractor = AdvancedPatternFeatureExtractor()
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available. Install with: pip install torch")
        
        # Training history
        self.training_history = {
            'train_losses': [],
            'val_losses': [],
            'train_accuracies': [],
            'val_accuracies': [],
            'learning_rates': []
        }
    
    def train(self, training_data: List[Dict], validation_split: float = 0.2):
        """Train super-powered neural network."""
        print("🚀 Training SUPER-POWERED neural network predictor...")
        print(f"   Architecture: Residual + Attention + Ensemble")
        print(f"   Features: Advanced pattern analysis + Multi-task learning")
        
        # Prepare data
        patterns = [np.array(sample['pattern']) for sample in training_data]
        parameters = [sample['parameters'] for sample in training_data]
        
        # Create datasets
        full_dataset = SuperPatternDataset(patterns, parameters, self.pattern_length, augment=True)
        
        # Split train/validation
        val_size = int(len(full_dataset) * validation_split)
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, 
                                shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, 
                              shuffle=False, num_workers=0)
        
        # Initialize model
        pattern_size = self.pattern_length * 2
        feature_size = 28  # Actual number of extracted features
        output_size = 36  # Enhanced parameter vector size
        
        if self.config.use_ensemble:
            self.model = EnsembleModel(
                num_models=self.config.ensemble_size,
                pattern_size=pattern_size,
                feature_size=feature_size,
                output_size=output_size,
                dropout_rate=self.config.dropout_rate
            )
        else:
            self.model = SuperPatternToParameterNet(
                pattern_size=pattern_size,
                feature_size=feature_size,
                output_size=output_size,
                dropout_rate=self.config.dropout_rate
            )
        
        # Loss functions
        wedge_criterion = nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)
        param_criterion = nn.MSELoss()
        
        # Optimizer with weight decay
        optimizer = optim.AdamW(self.model.parameters(), 
                               lr=self.config.initial_lr,
                               weight_decay=self.config.weight_decay)
        
        # Learning rate schedulers
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=self.config.warmup_epochs
        )
        main_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=50, T_mult=2, eta_min=self.config.min_lr
        )
        
        # Training loop
        best_val_loss = float('inf')
        best_val_acc = 0
        patience_counter = 0
        
        print(f"Training on {train_size} samples, validating on {val_size} samples")
        print(f"Batch size: {self.config.batch_size}, Initial LR: {self.config.initial_lr}")
        
        for epoch in range(self.config.epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_wedge_correct = 0
            train_total = 0
            
            for batch_patterns, batch_features, batch_params in train_loader:
                # Extract wedge labels from parameters (one-hot encoded in first 6 positions)
                wedge_labels = torch.argmax(batch_params[:, :6], dim=1)
                
                optimizer.zero_grad()
                
                # Forward pass
                wedge_logits, param_preds = self.model(batch_patterns, batch_features)
                
                # Multi-task loss
                wedge_loss = wedge_criterion(wedge_logits, wedge_labels)
                param_loss = param_criterion(param_preds, batch_params)
                
                # Combined loss with weighting
                total_loss = wedge_loss + 0.5 * param_loss
                
                # Backward pass with gradient clipping
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                optimizer.step()
                
                train_loss += total_loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(wedge_logits, 1)
                train_wedge_correct += (predicted == wedge_labels).sum().item()
                train_total += wedge_labels.size(0)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_wedge_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_patterns, batch_features, batch_params in val_loader:
                    wedge_labels = torch.argmax(batch_params[:, :6], dim=1)
                    
                    wedge_logits, param_preds = self.model(batch_patterns, batch_features)
                    
                    wedge_loss = wedge_criterion(wedge_logits, wedge_labels)
                    param_loss = param_criterion(param_preds, batch_params)
                    total_loss = wedge_loss + 0.5 * param_loss
                    
                    val_loss += total_loss.item()
                    
                    _, predicted = torch.max(wedge_logits, 1)
                    val_wedge_correct += (predicted == wedge_labels).sum().item()
                    val_total += wedge_labels.size(0)
            
            # Calculate metrics
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            train_acc = train_wedge_correct / train_total
            val_acc = val_wedge_correct / val_total
            
            # Store history
            self.training_history['train_losses'].append(train_loss)
            self.training_history['val_losses'].append(val_loss)
            self.training_history['train_accuracies'].append(train_acc)
            self.training_history['val_accuracies'].append(val_acc)
            self.training_history['learning_rates'].append(optimizer.param_groups[0]['lr'])
            
            # Learning rate scheduling
            if epoch < self.config.warmup_epochs:
                warmup_scheduler.step()
            else:
                main_scheduler.step()
            
            # Model selection based on validation accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'config': self.config,
                    'pattern_length': self.pattern_length,
                    'training_history': self.training_history,
                    'best_val_acc': best_val_acc,
                    'best_val_loss': best_val_loss
                }, self.model_path)
            else:
                patience_counter += 1
            
            # Logging
            if epoch % 10 == 0 or epoch == self.config.epochs - 1:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch:3d}: "
                      f"Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.1%} | "
                      f"Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.1%} | "
                      f"LR = {current_lr:.6f}")
            
            # Early stopping
            if patience_counter >= self.config.early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        print(f"✅ Training complete!")
        print(f"🏆 Best validation accuracy: {best_val_acc:.1%}")
        print(f"📉 Best validation loss: {best_val_loss:.4f}")
        print(f"💾 Model saved to: {self.model_path}")
        
        return {
            'final_train_loss': self.training_history['train_losses'][-1],
            'final_val_loss': self.training_history['val_losses'][-1],
            'final_train_acc': self.training_history['train_accuracies'][-1],
            'final_val_acc': self.training_history['val_accuracies'][-1],
            'best_val_acc': best_val_acc,
            'best_val_loss': best_val_loss,
            'epochs_trained': len(self.training_history['train_losses']),
            'training_history': self.training_history
        }
    
    def load(self) -> bool:
        """Load trained model."""
        if not os.path.exists(self.model_path):
            return False
        
        try:
            checkpoint = torch.load(self.model_path, map_location='cpu')
            
            # Reconstruct model
            pattern_size = self.pattern_length * 2
            feature_size = 28  # Actual number of extracted features
            output_size = 36
            
            if self.config.use_ensemble:
                self.model = EnsembleModel(
                    num_models=self.config.ensemble_size,
                    pattern_size=pattern_size,
                    feature_size=feature_size,
                    output_size=output_size
                )
            else:
                self.model = SuperPatternToParameterNet(
                    pattern_size=pattern_size,
                    feature_size=feature_size,
                    output_size=output_size
                )
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            # Load training history
            if 'training_history' in checkpoint:
                self.training_history = checkpoint['training_history']
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False
    
    def predict(self, pattern: np.ndarray) -> Dict:
        """Predict parameters from pattern with confidence scores."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load() first.")
        
        # Prepare pattern and features
        dataset = SuperPatternDataset([pattern], [{}], self.pattern_length, augment=False)
        pattern_tensor = dataset.patterns[0:1]
        features_tensor = dataset.features[0:1]
        
        # Predict
        with torch.no_grad():
            self.model.eval()
            wedge_logits, param_preds = self.model(pattern_tensor, features_tensor)
            
            # Get wedge count prediction with confidence
            wedge_probs = F.softmax(wedge_logits, dim=1)
            wedge_pred = torch.argmax(wedge_probs, dim=1).item() + 1  # Add 1 for 1-indexed
            wedge_confidence = wedge_probs[0, wedge_pred - 1].item()
            
            # Get parameter predictions
            param_vector = param_preds[0].numpy()
        
        # Convert back to parameter dictionary
        params = self._vector_to_parameters(param_vector, wedge_pred)
        
        # Add confidence scores
        params['prediction_confidence'] = {
            'wedge_count': wedge_confidence,
            'wedge_probabilities': wedge_probs[0].numpy().tolist()
        }
        
        return params
    
    def _vector_to_parameters(self, vector: np.ndarray, wedge_count: int) -> Dict:
        """Convert enhanced vector back to parameter dictionary."""
        max_wedges = 6
        idx = 6  # Skip one-hot encoding
        
        # Extract parameters based on predicted wedge count
        rotation_speeds = []
        for i in range(wedge_count):
            # Inverse tanh normalization
            normalized = vector[idx + i]
            speed = np.arctanh(np.clip(normalized, -0.99, 0.99)) * 5.0
            rotation_speeds.append(float(speed))
        idx += max_wedges
        
        phi_x = []
        for i in range(wedge_count):
            normalized = vector[idx + i]
            angle = np.arctanh(np.clip(normalized, -0.99, 0.99)) * 20.0
            phi_x.append(float(angle))
        idx += max_wedges
        
        phi_y = []
        for i in range(wedge_count):
            normalized = vector[idx + i]
            angle = np.arctanh(np.clip(normalized, -0.99, 0.99)) * 20.0
            phi_y.append(float(angle))
        idx += max_wedges
        
        distances = [1.0]
        for i in range(wedge_count):
            normalized = vector[idx + i]
            dist = np.arctanh(np.clip(normalized, -0.99, 0.99)) * 3.0 + 5.0
            distances.append(float(np.clip(dist, 1.0, 10.0)))
        idx += max_wedges
        
        refractive_indices = [1.0]
        for i in range(wedge_count):
            normalized = vector[idx + i] if idx + i < len(vector) else 0
            ri = np.arctanh(np.clip(normalized, -0.99, 0.99)) * 0.2 + 1.5
            refractive_indices.append(float(np.clip(ri, 1.3, 1.8)))
        refractive_indices.append(1.0)
        
        return {
            'rotation_speeds': rotation_speeds,
            'phi_x': phi_x,
            'phi_y': phi_y,
            'distances': distances,
            'refractive_indices': refractive_indices,
            'wedgenum': wedge_count
        }
    
    def get_training_info(self) -> Dict:
        """Get comprehensive training information."""
        if not os.path.exists(self.model_path):
            return {'trained': False}
        
        try:
            checkpoint = torch.load(self.model_path, map_location='cpu')
            history = checkpoint.get('training_history', {})
            
            return {
                'trained': True,
                'model_path': self.model_path,
                'architecture': 'SuperNet with Attention + Residual + Ensemble',
                'best_val_accuracy': checkpoint.get('best_val_acc', 0),
                'best_val_loss': checkpoint.get('best_val_loss', 0),
                'epochs_trained': len(history.get('train_losses', [])),
                'final_train_acc': history.get('train_accuracies', [0])[-1] if history.get('train_accuracies') else 0,
                'final_val_acc': history.get('val_accuracies', [0])[-1] if history.get('val_accuracies') else 0,
                'training_history': history
            }
        except Exception as e:
            return {'trained': False, 'error': str(e)}