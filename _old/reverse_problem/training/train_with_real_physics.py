#!/usr/bin/env python3
"""
Train neural network with REAL physics data.

This training pipeline:
1. Generates data using improved physics simulation
2. Properly normalizes patterns (preserving shape)
3. Trains with validation monitoring
4. Saves the best model
"""

import sys
import os
import numpy as np
import json
import time
from datetime import datetime

# Add parent to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solver import StateOfTheArtSolver
from core.super_neural_network import SuperNeuralPredictor


def generate_training_data(num_samples=5000):
    """Generate training data with improved physics."""

    print(f"Generating {num_samples} samples with improved physics...")
    print("=" * 60)

    solver = StateOfTheArtSolver(use_super_nn=False)  # Don't use existing NN
    samples = []
    samples_per_wedge = num_samples // 6

    for wedge_count in range(1, 7):
        print(f"Generating {samples_per_wedge} samples for {wedge_count} wedge(s)...")

        for i in range(samples_per_wedge):
            # Generate diverse parameters
            params = {
                'wedgenum': wedge_count,
                'rotation_speeds': np.random.uniform(-3.0, 3.0, wedge_count).tolist(),
                'phi_x': np.random.uniform(-20.0, 20.0, wedge_count).tolist(),
                'phi_y': np.random.uniform(-20.0, 20.0, wedge_count).tolist(),
                'distances': [1.0] + [5.0] * wedge_count + [100.0],
                'refractive_indices': [1.0] + [1.5] * wedge_count + [1.0]
            }

            # Add some edge cases (20% of samples)
            if np.random.random() < 0.2:
                choice = np.random.choice(['slow', 'fast', 'large_angle', 'small_angle'])
                if choice == 'slow':
                    params['rotation_speeds'] = np.random.uniform(-0.5, 0.5, wedge_count).tolist()
                elif choice == 'fast':
                    params['rotation_speeds'] = np.random.uniform(-5.0, 5.0, wedge_count).tolist()
                elif choice == 'large_angle':
                    params['phi_x'] = np.random.uniform(-30.0, 30.0, wedge_count).tolist()
                    params['phi_y'] = np.random.uniform(-30.0, 30.0, wedge_count).tolist()
                else:  # small_angle
                    params['phi_x'] = np.random.uniform(-5.0, 5.0, wedge_count).tolist()
                    params['phi_y'] = np.random.uniform(-5.0, 5.0, wedge_count).tolist()

            # Generate pattern
            pattern = solver.forward_simulate(params)

            # Store sample
            sample = {
                'id': len(samples),
                'wedge_count': wedge_count,
                'parameters': params,
                'pattern': pattern.tolist(),
                'complexity': solver.calculate_pattern_complexity(pattern)
            }
            samples.append(sample)

            if len(samples) % 500 == 0:
                print(f"  Total: {len(samples)}/{num_samples}")

    print(f"\nGenerated {len(samples)} training samples")
    return samples


def train_neural_network(samples, epochs=100, batch_size=64):
    """Train neural network with proper validation."""

    print("\nTraining Neural Network")
    print("=" * 60)

    # Initialize predictor
    predictor = SuperNeuralPredictor()
    predictor.config.epochs = epochs
    predictor.config.batch_size = batch_size
    predictor.config.early_stopping_patience = 20
    predictor.config.learning_rate = 0.001

    # Train
    start_time = time.time()
    results = predictor.train(samples, validation_split=0.2)
    training_time = time.time() - start_time

    print("\nTraining Complete!")
    print(f"  Time: {training_time:.1f}s")
    print(f"  Best validation accuracy: {results['best_val_acc']:.1%}")
    print(f"  Best validation loss: {results['best_val_loss']:.4f}")

    return results, training_time


def validate_model():
    """Validate the trained model."""

    print("\nValidating Trained Model")
    print("=" * 60)

    solver = StateOfTheArtSolver()  # Will load the trained model

    # Generate test samples
    test_samples = 60  # 10 per wedge count
    correct = 0
    results = []

    for wedge_count in range(1, 7):
        for _ in range(10):
            # Generate test parameters
            params = solver.generate_parameters(wedge_count)
            pattern = solver.forward_simulate(params)

            # Predict
            predicted_wedges, cost, recovered_params, _ = solver.intelligent_wedge_selection(
                pattern, verbose=False
            )

            is_correct = (predicted_wedges == wedge_count)
            if is_correct:
                correct += 1

            # Calculate parameter errors if correct
            param_errors = {}
            if is_correct and recovered_params:
                if 'rotation_speeds' in recovered_params:
                    speed_error = np.mean([
                        abs(params['rotation_speeds'][i] - recovered_params['rotation_speeds'][i])
                        for i in range(min(len(params['rotation_speeds']),
                                         len(recovered_params['rotation_speeds'])))
                    ])
                    param_errors['speed_error'] = speed_error

                if 'phi_x' in recovered_params:
                    phi_x_error = np.mean([
                        abs(params['phi_x'][i] - recovered_params['phi_x'][i])
                        for i in range(min(len(params['phi_x']),
                                         len(recovered_params['phi_x'])))
                    ])
                    param_errors['phi_x_error'] = phi_x_error

            results.append({
                'true_wedges': wedge_count,
                'predicted_wedges': predicted_wedges,
                'correct': is_correct,
                'cost': cost,
                'param_errors': param_errors
            })

    # Calculate statistics
    accuracy = correct / test_samples
    print(f"\nValidation Results:")
    print(f"  Overall accuracy: {accuracy:.1%}")

    # Accuracy by wedge count
    for w in range(1, 7):
        w_results = [r for r in results if r['true_wedges'] == w]
        w_accuracy = sum(r['correct'] for r in w_results) / len(w_results)
        print(f"  {w}-wedge accuracy: {w_accuracy:.1%}")

    # Parameter recovery accuracy (for correct predictions)
    correct_results = [r for r in results if r['correct'] and r['param_errors']]
    if correct_results:
        avg_speed_error = np.mean([r['param_errors'].get('speed_error', 0)
                                  for r in correct_results if 'speed_error' in r['param_errors']])
        avg_phi_error = np.mean([r['param_errors'].get('phi_x_error', 0)
                                for r in correct_results if 'phi_x_error' in r['param_errors']])
        print(f"\nParameter Recovery (correct predictions only):")
        print(f"  Avg speed error: {avg_speed_error:.3f} Hz")
        print(f"  Avg angle error: {avg_phi_error:.3f}°")

    return results


def main():
    """Main training pipeline."""

    print("\n" + "=" * 80)
    print("TRAINING WITH IMPROVED PHYSICS")
    print("=" * 80)

    # Configuration
    num_samples = 5000  # Increase for better accuracy
    epochs = 100
    batch_size = 64

    print(f"\nConfiguration:")
    print(f"  Training samples: {num_samples}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")

    # Generate training data
    samples = generate_training_data(num_samples)

    # Save training data
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    data_file = f'training/training_data_{timestamp}.json'
    with open(data_file, 'w') as f:
        json.dump({
            'metadata': {
                'timestamp': timestamp,
                'num_samples': len(samples),
                'physics_type': 'improved_fallback'
            },
            'samples': samples[:100]  # Save subset for inspection
        }, f, indent=2)
    print(f"\nTraining data saved to: {data_file}")

    # Train neural network
    results, training_time = train_neural_network(samples, epochs, batch_size)

    # Validate model
    validation_results = validate_model()

    # Save results
    results_file = f'training/training_results_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump({
            'training_config': {
                'num_samples': num_samples,
                'epochs': epochs,
                'batch_size': batch_size,
                'training_time': training_time
            },
            'training_results': results,
            'validation_results': validation_results[:10]  # Sample
        }, f, indent=2)

    print(f"\nResults saved to: {results_file}")
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()