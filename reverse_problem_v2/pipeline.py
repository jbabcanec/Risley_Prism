#!/usr/bin/env python3
"""
Complete Pipeline for Reverse Risley Prism Problem
Generates data, trains neural network, evaluates performance
"""

import numpy as np
import sys
import os
import pickle
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import inputs
import model
from core import RisleyParameters, ForwardModel, PatternAnalyzer
from neural_network import RisleyNeuralPredictor


class Pipeline:
    """Single pipeline to run everything."""

    def __init__(self, n_samples: int = 50000, epochs: int = 300):
        """
        Initialize pipeline.

        Args:
            n_samples: Number of training samples to generate
            epochs: Number of training epochs
        """
        self.n_samples = n_samples
        self.epochs = epochs
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Components
        self.forward_model = ForwardModel()
        self.analyzer = PatternAnalyzer()
        self.predictor = RisleyNeuralPredictor()

        # Paths
        self.dataset_path = f'dataset_{self.timestamp}.pkl'
        self.model_path = f'model_{self.timestamp}'
        self.results_path = f'results_{self.timestamp}.json'


    def generate_dataset(self):
        """Generate training dataset using REAL PHYSICS forward model."""
        print(f"\n{'='*60}")
        print(f"GENERATING DATASET: {self.n_samples} samples")
        print(f"Using REAL PHYSICS from verified forward model")
        print(f"{'='*60}")

        samples = []
        samples_per_wedge = self.n_samples // 6

        for wedge_count in range(1, 7):
            n = samples_per_wedge + (1 if wedge_count <= self.n_samples % 6 else 0)
            print(f"Generating {n} samples for {wedge_count} wedges...")

            for i in range(n):
                # Random parameters
                speeds = np.random.uniform(-3, 3, wedge_count).tolist()
                phi_x = np.random.uniform(-15, 15, wedge_count).tolist()
                phi_y = np.random.uniform(-15, 15, wedge_count).tolist()

                params = RisleyParameters(wedge_count, speeds, phi_x, phi_y)

                # Generate pattern using REAL PHYSICS
                try:
                    pattern = self.forward_model.simulate(params, time_points=30)
                except Exception as e:
                    # Skip if simulation fails
                    continue

                # Extract features
                features = self.analyzer.extract_features(pattern)

                samples.append({
                    'params': params,
                    'pattern': pattern,
                    'features': features
                })

                if (i + 1) % 1000 == 0:
                    print(f"  {i + 1}/{n}")

        # Save dataset
        with open(self.dataset_path, 'wb') as f:
            pickle.dump(samples, f)

        print(f"Dataset saved to {self.dataset_path}")
        return samples

    def train(self, samples: List[Dict]):
        """Train neural network."""
        print(f"\n{'='*60}")
        print(f"TRAINING NEURAL NETWORK")
        print(f"{'='*60}")

        # Split data
        np.random.shuffle(samples)
        n_train = int(len(samples) * 0.8)
        train_data = samples[:n_train]
        test_data = samples[n_train:]

        print(f"Training samples: {len(train_data)}")
        print(f"Test samples: {len(test_data)}")

        # Prepare training data
        features_train = [s['features'] for s in train_data]
        params_train = [s['params'] for s in train_data]

        # Train
        print(f"Training for {self.epochs} epochs...")
        history = self.predictor.train(
            features_train,
            params_train,
            validation_split=0.2,
            epochs=self.epochs
        )

        print(f"Final training loss: {history['train_loss'][-1]:.6f}")
        print(f"Final validation loss: {history['val_loss'][-1]:.6f}")

        # Save model
        self.predictor.save(self.model_path)
        print(f"Model saved to {self.model_path}")

        return test_data

    def evaluate(self, test_data: List[Dict]) -> Dict:
        """Evaluate model performance."""
        print(f"\n{'='*60}")
        print(f"EVALUATING PERFORMANCE")
        print(f"{'='*60}")

        # Classification metrics
        confusion = np.zeros((6, 6), dtype=int)
        correct = 0
        total = len(test_data)

        # Regression metrics
        speed_errors = []
        angle_errors = []

        for sample in test_data:
            true_params = sample['params']
            pred = self.predictor.predict(sample['features'])

            true_w = true_params.wedge_count
            pred_w = pred['wedge_count']

            confusion[true_w-1][pred_w-1] += 1

            if pred_w == true_w:
                correct += 1

                # Calculate parameter errors
                for i in range(true_w):
                    speed_err = abs(pred['rotation_speeds'][i] - true_params.rotation_speeds[i])
                    angle_err = np.mean([
                        abs(pred['phi_x'][i] - true_params.phi_x[i]),
                        abs(pred['phi_y'][i] - true_params.phi_y[i])
                    ])
                    speed_errors.append(speed_err)
                    angle_errors.append(angle_err)

        # Calculate metrics
        accuracy = correct / total
        speed_mae = np.mean(speed_errors) if speed_errors else 0
        angle_mae = np.mean(angle_errors) if angle_errors else 0

        results = {
            'accuracy': accuracy,
            'speed_mae': speed_mae,
            'angle_mae': angle_mae,
            'confusion_matrix': confusion.tolist(),
            'total_samples': total,
            'correct_samples': correct
        }

        # Print results
        print(f"\nRESULTS:")
        print(f"  Wedge Classification Accuracy: {accuracy:.1%}")
        print(f"  Speed Error (MAE): {speed_mae:.3f} Hz")
        print(f"  Angle Error (MAE): {angle_mae:.2f} degrees")

        print(f"\nConfusion Matrix:")
        print("True\\Pred", end="")
        for i in range(1, 7):
            print(f"\t{i}", end="")
        print()
        for i in range(6):
            print(f"{i+1}\t", end="")
            for j in range(6):
                print(f"{confusion[i][j]}\t", end="")
            print()

        # Save results
        with open(self.results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to {self.results_path}")

        return results

    def run(self):
        """Run complete pipeline."""
        start_time = time.time()

        print(f"\n{'='*60}")
        print(f"REVERSE RISLEY PRISM PIPELINE")
        print(f"{'='*60}")
        print(f"Samples: {self.n_samples}")
        print(f"Epochs: {self.epochs}")
        print(f"Timestamp: {self.timestamp}")

        # Generate dataset
        samples = self.generate_dataset()

        # Train model
        test_data = self.train(samples)

        # Evaluate
        results = self.evaluate(test_data)

        # Summary
        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"PIPELINE COMPLETE")
        print(f"{'='*60}")
        print(f"Total time: {elapsed/60:.1f} minutes")
        print(f"Dataset: {self.dataset_path}")
        print(f"Model: {self.model_path}")
        print(f"Results: {self.results_path}")
        print(f"\nFinal Accuracy: {results['accuracy']:.1%}")
        print(f"{'='*60}")

        return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run pipeline for reverse Risley prism problem')
    parser.add_argument('--samples', type=int, default=50000, help='Number of training samples')
    parser.add_argument('--epochs', type=int, default=300, help='Training epochs')

    args = parser.parse_args()

    # Run pipeline
    pipeline = Pipeline(n_samples=args.samples, epochs=args.epochs)
    results = pipeline.run()