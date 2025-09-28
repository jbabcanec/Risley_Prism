#!/usr/bin/env python3
"""
Clean Neural Network Training Pipeline
Systematically trains on generated data from forward model
"""

import numpy as np
import sys
import os
import pickle
import json
from datetime import datetime
from typing import Dict, List, Tuple
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import RisleyParameters, ForwardModel, PatternAnalyzer, ReverseRisleySolver
from neural_network import RisleyNeuralPredictor


class TrainingPipeline:
    """Clean training pipeline for neural network."""

    def __init__(self):
        """Initialize pipeline components."""
        self.analyzer = PatternAnalyzer()
        self.predictor = RisleyNeuralPredictor()
        self.forward_model = ForwardModel()

    def load_dataset(self, dataset_path: str) -> Tuple[List, List]:
        """
        Load dataset from file.

        Args:
            dataset_path: Path to dataset pickle file

        Returns:
            Tuple of (training_data, test_data)
        """
        print(f"Loading dataset from {dataset_path}...")

        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)

        # Handle different dataset formats
        if isinstance(dataset, dict) and 'samples' in dataset:
            samples = dataset['samples']
            metadata = dataset.get('metadata', {})
            print(f"  Loaded {len(samples)} samples")
            if metadata:
                print(f"  Dataset ID: {metadata.get('dataset_id', 'Unknown')}")
                print(f"  Wedge distribution:")
                for w in range(1, 7):
                    if 'wedge_distribution' in metadata:
                        count = metadata['wedge_distribution'].get(w, 0)
                        print(f"    {w} wedge(s): {count} samples")
        elif isinstance(dataset, list):
            samples = dataset
            print(f"  Loaded {len(samples)} samples")
        else:
            # Handle chunked datasets from generate_dataset.py
            samples = []
            for chunk in dataset:
                if isinstance(chunk, list):
                    samples.extend(chunk)
            print(f"  Loaded {len(samples)} samples from chunks")

        # Split into train/test
        np.random.shuffle(samples)
        n_train = int(len(samples) * 0.8)

        train_data = samples[:n_train]
        test_data = samples[n_train:]

        print(f"  Training: {len(train_data)} samples")
        print(f"  Testing: {len(test_data)} samples")

        return train_data, test_data

    def prepare_features(self, samples: List[Dict]) -> Tuple[List[Dict], List[RisleyParameters]]:
        """
        Extract features from patterns and prepare parameters.

        Args:
            samples: List of sample dictionaries

        Returns:
            Tuple of (features_list, parameters_list)
        """
        features_list = []
        parameters_list = []

        print("Extracting features from patterns...")

        for i, sample in enumerate(samples):
            # Get pattern and parameters
            pattern = sample.get('pattern')
            params_dict = sample.get('parameters')

            if pattern is None or params_dict is None:
                continue

            # Extract features from pattern
            try:
                features = self.analyzer.extract_features(pattern)
                features_list.append(features)

                # Create RisleyParameters object
                params = RisleyParameters(
                    wedge_count=params_dict['wedge_count'],
                    rotation_speeds=params_dict['rotation_speeds'],
                    phi_x=params_dict['phi_x'],
                    phi_y=params_dict['phi_y']
                )
                parameters_list.append(params)

            except Exception as e:
                print(f"    Warning: Failed to process sample {i}: {e}")

            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(samples)} samples")

        print(f"  Successfully processed {len(features_list)} samples")

        return features_list, parameters_list

    def train(self, train_data: List, validation_split: float = 0.2,
              epochs: int = 150) -> Dict:
        """
        Train neural network on prepared data.

        Args:
            train_data: Training samples
            validation_split: Fraction for validation
            epochs: Number of training epochs

        Returns:
            Training history dictionary
        """
        # Prepare features and parameters
        features_train, params_train = self.prepare_features(train_data)

        if len(features_train) == 0:
            raise ValueError("No valid training samples found!")

        print(f"\nTraining neural network...")
        print(f"  Training samples: {len(features_train)}")
        print(f"  Validation split: {validation_split}")
        print(f"  Epochs: {epochs}")

        # Train the model
        history = self.predictor.train(
            features_train,
            params_train,
            validation_split=validation_split,
            epochs=epochs
        )

        print(f"\nTraining complete!")
        print(f"  Final training loss: {history['train_loss'][-1]:.6f}")
        print(f"  Final validation loss: {history['val_loss'][-1]:.6f}")

        return history

    def evaluate(self, test_data: List) -> Dict:
        """
        Evaluate model on test data.

        Args:
            test_data: Test samples

        Returns:
            Evaluation metrics dictionary
        """
        print("\nEvaluating on test set...")

        # Prepare test features
        features_test, params_test = self.prepare_features(test_data)

        if len(features_test) == 0:
            print("  No valid test samples found!")
            return {}

        # Initialize metrics
        confusion_matrix = np.zeros((6, 6), dtype=int)
        wedge_results = {i: {
            'total': 0,
            'correct': 0,
            'speed_errors': [],
            'angle_x_errors': [],
            'angle_y_errors': []
        } for i in range(1, 7)}

        # Evaluate each sample
        for i, (features, true_params) in enumerate(zip(features_test, params_test)):
            # Get prediction
            pred = self.predictor.predict(features)

            true_w = true_params.wedge_count
            pred_w = pred['wedge_count']

            # Update confusion matrix
            confusion_matrix[true_w-1][pred_w-1] += 1
            wedge_results[true_w]['total'] += 1

            if pred_w == true_w:
                wedge_results[true_w]['correct'] += 1

                # Calculate parameter errors
                for j in range(true_w):
                    speed_err = abs(pred['rotation_speeds'][j] - true_params.rotation_speeds[j])
                    angle_x_err = abs(pred['phi_x'][j] - true_params.phi_x[j])
                    angle_y_err = abs(pred['phi_y'][j] - true_params.phi_y[j])

                    wedge_results[true_w]['speed_errors'].append(speed_err)
                    wedge_results[true_w]['angle_x_errors'].append(angle_x_err)
                    wedge_results[true_w]['angle_y_errors'].append(angle_y_err)

            if (i + 1) % 50 == 0:
                print(f"  Evaluated {i + 1}/{len(features_test)} samples")

        # Calculate metrics
        total_correct = sum(r['correct'] for r in wedge_results.values())
        total = sum(r['total'] for r in wedge_results.values())
        overall_accuracy = total_correct / total * 100 if total > 0 else 0

        return {
            'overall_accuracy': overall_accuracy,
            'total_correct': total_correct,
            'total_samples': total,
            'confusion_matrix': confusion_matrix,
            'wedge_results': wedge_results
        }

    def print_results(self, metrics: Dict):
        """
        Print evaluation results in clean format.

        Args:
            metrics: Evaluation metrics dictionary
        """
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)

        # Overall accuracy
        print(f"\nOVERALL WEDGE CLASSIFICATION ACCURACY:")
        print(f"  {metrics['overall_accuracy']:.1f}% ({metrics['total_correct']}/{metrics['total_samples']})")

        # Per-wedge breakdown
        print("\nPER-WEDGE ACCURACY:")
        wedge_results = metrics['wedge_results']
        for w in range(1, 7):
            r = wedge_results[w]
            if r['total'] > 0:
                acc = r['correct'] / r['total'] * 100
                print(f"\n  {w} wedge(s): {acc:.1f}% ({r['correct']}/{r['total']})")

                if r['speed_errors']:
                    avg_speed = np.mean(r['speed_errors'])
                    avg_angle_x = np.mean(r['angle_x_errors'])
                    avg_angle_y = np.mean(r['angle_y_errors'])
                    print(f"    When correct:")
                    print(f"      Speed MAE: {avg_speed:.3f} Hz")
                    print(f"      Phi_x MAE: {avg_angle_x:.2f} degrees")
                    print(f"      Phi_y MAE: {avg_angle_y:.2f} degrees")

        # Confusion matrix
        print("\nCONFUSION MATRIX:")
        print("True\\Pred", end="")
        for i in range(1, 7):
            print(f"\t{i}", end="")
        print()

        confusion = metrics['confusion_matrix']
        for true_idx in range(6):
            print(f"{true_idx+1}\t", end="")
            for pred_idx in range(6):
                count = confusion[true_idx][pred_idx]
                print(f"{count}\t", end="")
            print()

        print("=" * 60)

    def save_model(self, model_path: str = 'trained_model'):
        """
        Save trained model to disk.

        Args:
            model_path: Path to save model
        """
        self.predictor.save(model_path)
        print(f"\nModel saved to {model_path}/")

    def validate_with_physics(self, n_samples: int = 30):
        """
        Validate predictions using real forward physics.

        Args:
            n_samples: Number of validation samples to generate
        """
        print("\n" + "=" * 60)
        print("PHYSICS VALIDATION")
        print("=" * 60)
        print(f"Generating {n_samples} test cases with real physics...")

        correct = 0
        total = 0

        for wedge_count in range(1, 7):
            n_per_wedge = n_samples // 6

            print(f"\nTesting {wedge_count}-wedge systems:")

            for i in range(n_per_wedge):
                # Generate random parameters
                speeds = np.random.uniform(-3, 3, wedge_count).tolist()
                phi_x = np.random.uniform(-15, 15, wedge_count).tolist()
                phi_y = np.random.uniform(-15, 15, wedge_count).tolist()

                true_params = RisleyParameters(
                    wedge_count=wedge_count,
                    rotation_speeds=speeds,
                    phi_x=phi_x,
                    phi_y=phi_y
                )

                try:
                    # Generate real pattern
                    pattern = self.forward_model.simulate(true_params, time_points=30)
                    features = self.analyzer.extract_features(pattern)

                    # Predict
                    pred = self.predictor.predict(features)

                    # Check accuracy
                    if pred['wedge_count'] == wedge_count:
                        correct += 1

                    total += 1

                    if i == 0:  # Show first example
                        print(f"  Example: True W={wedge_count}, Pred W={pred['wedge_count']} - "
                              f"{'✓' if pred['wedge_count'] == wedge_count else '✗'}")

                except Exception as e:
                    print(f"  Warning: Physics validation failed: {e}")

        if total > 0:
            physics_accuracy = correct / total * 100
            print(f"\nPhysics validation accuracy: {physics_accuracy:.1f}% ({correct}/{total})")

        print("=" * 60)


def main():
    """Main training pipeline execution."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Neural network training pipeline for Risley prism reverse problem'
    )
    parser.add_argument(
        '--dataset', type=str, required=True,
        help='Path to dataset file (pickle format)'
    )
    parser.add_argument(
        '--epochs', type=int, default=150,
        help='Number of training epochs (default: 150)'
    )
    parser.add_argument(
        '--model-path', type=str, default='trained_nn_model',
        help='Path to save trained model (default: trained_nn_model)'
    )
    parser.add_argument(
        '--validate-physics', action='store_true',
        help='Run physics validation after training'
    )

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = TrainingPipeline()

    # Load dataset
    train_data, test_data = pipeline.load_dataset(args.dataset)

    # Train model
    history = pipeline.train(train_data, epochs=args.epochs)

    # Evaluate on test set
    metrics = pipeline.evaluate(test_data)

    # Print results
    pipeline.print_results(metrics)

    # Save model
    pipeline.save_model(args.model_path)

    # Optional physics validation
    if args.validate_physics:
        pipeline.validate_with_physics(n_samples=30)

    print("\n" + "=" * 60)
    print("TRAINING PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Model saved to: {args.model_path}/")
    print(f"Overall accuracy: {metrics['overall_accuracy']:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()