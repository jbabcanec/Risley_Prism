#!/usr/bin/env python3
"""
Neural Network for Initial Parameter Estimation
Uses physical features extracted from patterns to estimate Risley parameters.
"""

import numpy as np
import pickle
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


class NeuralNetwork:
    """Simple feedforward neural network for pattern-to-parameter mapping."""

    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        """
        Initialize network architecture.

        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer sizes
            output_dim: Number of output parameters
        """
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        # Initialize weights
        self.weights = []
        self.biases = []

        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            # Xavier initialization
            w = np.random.randn(dims[i], dims[i+1]) * np.sqrt(2.0 / dims[i])
            b = np.zeros((1, dims[i+1]))
            self.weights.append(w)
            self.biases.append(b)

    def relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation function."""
        return np.maximum(0, x)

    def relu_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of ReLU."""
        return (x > 0).astype(float)

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation for output layer."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Forward pass through the network.

        Returns:
            output: Network predictions
            activations: List of activations at each layer (for backprop)
        """
        activations = [X]
        current = X

        # Hidden layers with ReLU
        for i in range(len(self.weights) - 1):
            z = current @ self.weights[i] + self.biases[i]
            current = self.relu(z)
            activations.append(current)

        # Output layer (no activation for regression)
        output = current @ self.weights[-1] + self.biases[-1]
        activations.append(output)

        return output, activations

    def backward(self, X: np.ndarray, y: np.ndarray,
                 learning_rate: float = 0.001) -> float:
        """
        Backward pass with gradient descent.

        Returns:
            loss: Mean squared error
        """
        batch_size = X.shape[0]

        # Forward pass
        predictions, activations = self.forward(X)

        # Calculate loss
        loss = np.mean((predictions - y) ** 2)

        # Backward pass
        delta = 2 * (predictions - y) / batch_size

        for i in range(len(self.weights) - 1, -1, -1):
            # Gradient for weights and biases
            dW = activations[i].T @ delta
            db = np.sum(delta, axis=0, keepdims=True)

            # Update weights
            self.weights[i] -= learning_rate * dW
            self.biases[i] -= learning_rate * db

            # Propagate error backward
            if i > 0:
                delta = delta @ self.weights[i].T
                delta *= self.relu_derivative(activations[i])

        return loss

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              epochs: int = 100, batch_size: int = 32,
              learning_rate: float = 0.001, verbose: bool = True) -> Dict:
        """
        Train the neural network.

        Returns:
            history: Training history
        """
        history = {'train_loss': [], 'val_loss': []}
        n_samples = X_train.shape[0]

        for epoch in range(epochs):
            # Shuffle training data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            # Mini-batch training
            epoch_losses = []
            for i in range(0, n_samples, batch_size):
                batch_X = X_shuffled[i:i+batch_size]
                batch_y = y_shuffled[i:i+batch_size]
                loss = self.backward(batch_X, batch_y, learning_rate)
                epoch_losses.append(loss)

            train_loss = np.mean(epoch_losses)
            history['train_loss'].append(train_loss)

            # Validation
            if X_val is not None and y_val is not None:
                val_pred, _ = self.forward(X_val)
                val_loss = np.mean((val_pred - y_val) ** 2)
                history['val_loss'].append(val_loss)

            # Progress
            if verbose and (epoch + 1) % 10 == 0:
                msg = f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.6f}"
                if X_val is not None:
                    msg += f" - Val Loss: {val_loss:.6f}"
                print(msg)

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        predictions, _ = self.forward(X)
        return predictions

    def save(self, filepath: str):
        """Save network weights."""
        model_data = {
            'architecture': {
                'input_dim': self.input_dim,
                'hidden_dims': self.hidden_dims,
                'output_dim': self.output_dim
            },
            'weights': self.weights,
            'biases': self.biases
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

    def load(self, filepath: str):
        """Load network weights."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.input_dim = model_data['architecture']['input_dim']
        self.hidden_dims = model_data['architecture']['hidden_dims']
        self.output_dim = model_data['architecture']['output_dim']
        self.weights = model_data['weights']
        self.biases = model_data['biases']


class RisleyNeuralPredictor:
    """
    Neural predictor specifically for Risley prism parameters.
    Handles feature normalization and parameter denormalization.
    """

    def __init__(self):
        self.network = None
        self.feature_scaler = None
        self.param_scaler = None
        self.feature_names = None

    def prepare_features(self, features_list: List[Dict]) -> np.ndarray:
        """Convert feature dictionaries to numpy array."""
        if not features_list:
            return np.array([])

        # Get feature names from first dict
        if self.feature_names is None:
            self.feature_names = sorted(features_list[0].keys())

        # Convert to array
        X = np.array([[f[name] for name in self.feature_names]
                      for f in features_list])

        return X

    def prepare_parameters(self, params_list: List) -> np.ndarray:
        """
        Convert RisleyParameters to array format.
        Format: [wedge_count, speeds..., phi_x..., phi_y...]
        """
        max_wedges = 6
        y = []

        for params in params_list:
            row = [params.wedge_count]

            # Pad speeds
            speeds = params.rotation_speeds + [0] * (max_wedges - len(params.rotation_speeds))
            row.extend(speeds)

            # Pad angles
            phi_x = params.phi_x + [0] * (max_wedges - len(params.phi_x))
            phi_y = params.phi_y + [0] * (max_wedges - len(params.phi_y))
            row.extend(phi_x)
            row.extend(phi_y)

            y.append(row)

        return np.array(y)

    def normalize_features(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Normalize features to [0, 1] range."""
        if fit or self.feature_scaler is None:
            self.feature_scaler = {
                'min': np.min(X, axis=0),
                'max': np.max(X, axis=0)
            }
            # Avoid division by zero
            range_vals = self.feature_scaler['max'] - self.feature_scaler['min']
            range_vals[range_vals == 0] = 1
            self.feature_scaler['range'] = range_vals

        X_norm = (X - self.feature_scaler['min']) / self.feature_scaler['range']
        return X_norm

    def normalize_parameters(self, y: np.ndarray, fit: bool = False) -> np.ndarray:
        """Normalize parameters for training."""
        if fit or self.param_scaler is None:
            # Different ranges for different parameters
            self.param_scaler = {
                'wedge_range': [1, 6],
                'speed_range': [-10, 10],
                'angle_range': [-45, 45]
            }

        y_norm = np.zeros_like(y)

        # Normalize wedge count (index 0)
        y_norm[:, 0] = (y[:, 0] - self.param_scaler['wedge_range'][0]) / \
                       (self.param_scaler['wedge_range'][1] - self.param_scaler['wedge_range'][0])

        # Normalize speeds (indices 1-6)
        y_norm[:, 1:7] = (y[:, 1:7] - self.param_scaler['speed_range'][0]) / \
                          (self.param_scaler['speed_range'][1] - self.param_scaler['speed_range'][0])

        # Normalize angles (indices 7-19)
        y_norm[:, 7:] = (y[:, 7:] - self.param_scaler['angle_range'][0]) / \
                        (self.param_scaler['angle_range'][1] - self.param_scaler['angle_range'][0])

        return y_norm

    def denormalize_parameters(self, y_norm: np.ndarray) -> np.ndarray:
        """Denormalize parameters after prediction."""
        y = np.zeros_like(y_norm)

        # Denormalize wedge count
        y[:, 0] = y_norm[:, 0] * (self.param_scaler['wedge_range'][1] -
                                  self.param_scaler['wedge_range'][0]) + \
                  self.param_scaler['wedge_range'][0]
        y[:, 0] = np.round(np.clip(y[:, 0], 1, 6)).astype(int)

        # Denormalize speeds
        y[:, 1:7] = y_norm[:, 1:7] * (self.param_scaler['speed_range'][1] -
                                       self.param_scaler['speed_range'][0]) + \
                    self.param_scaler['speed_range'][0]

        # Denormalize angles
        y[:, 7:] = y_norm[:, 7:] * (self.param_scaler['angle_range'][1] -
                                     self.param_scaler['angle_range'][0]) + \
                   self.param_scaler['angle_range'][0]

        return y

    def train(self, features_list: List[Dict], params_list: List,
              validation_split: float = 0.2, epochs: int = 200) -> Dict:
        """
        Train the neural network on pattern features.
        """
        # Prepare data
        X = self.prepare_features(features_list)
        y = self.prepare_parameters(params_list)

        # Normalize
        X = self.normalize_features(X, fit=True)
        y = self.normalize_parameters(y, fit=True)

        # Split data
        n_samples = X.shape[0]
        n_train = int(n_samples * (1 - validation_split))
        indices = np.random.permutation(n_samples)

        X_train = X[indices[:n_train]]
        y_train = y[indices[:n_train]]
        X_val = X[indices[n_train:]]
        y_val = y[indices[n_train:]]

        # Create network
        input_dim = X.shape[1]
        output_dim = y.shape[1]
        hidden_dims = [128, 64, 32]  # Architecture tuned for this problem

        self.network = NeuralNetwork(input_dim, hidden_dims, output_dim)

        # Train
        print(f"Training neural network on {n_train} samples...")
        history = self.network.train(
            X_train, y_train, X_val, y_val,
            epochs=epochs, batch_size=32,
            learning_rate=0.001, verbose=True
        )

        return history

    def predict(self, features: Dict) -> Dict:
        """
        Predict parameters from pattern features.

        Returns:
            Dictionary with predicted parameters
        """
        if self.network is None:
            raise ValueError("Network not trained yet")

        # Prepare features
        X = self.prepare_features([features])
        X = self.normalize_features(X)

        # Predict
        y_norm = self.network.predict(X)
        y = self.denormalize_parameters(y_norm)

        # Extract parameters
        wedge_count = int(y[0, 0])
        speeds = y[0, 1:1+wedge_count].tolist()
        phi_x = y[0, 7:7+wedge_count].tolist()
        phi_y = y[0, 13:13+wedge_count].tolist()

        return {
            'wedge_count': wedge_count,
            'rotation_speeds': speeds,
            'phi_x': phi_x,
            'phi_y': phi_y
        }

    def save(self, directory: str):
        """Save the trained model."""
        os.makedirs(directory, exist_ok=True)

        # Save network
        self.network.save(os.path.join(directory, 'network.pkl'))

        # Save scalers and metadata
        metadata = {
            'feature_scaler': self.feature_scaler,
            'param_scaler': self.param_scaler,
            'feature_names': self.feature_names
        }
        with open(os.path.join(directory, 'metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)

    def load(self, directory: str):
        """Load a trained model."""
        # Load network
        self.network = NeuralNetwork(0, [], 0)  # Dummy init
        self.network.load(os.path.join(directory, 'network.pkl'))

        # Load metadata
        with open(os.path.join(directory, 'metadata.pkl'), 'rb') as f:
            metadata = pickle.load(f)

        self.feature_scaler = metadata['feature_scaler']
        self.param_scaler = metadata['param_scaler']
        self.feature_names = metadata['feature_names']