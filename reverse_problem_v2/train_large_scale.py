#!/usr/bin/env python3
"""
Large-scale training script for multi-wedge Risley prism reverse solver.
Trains on thousands of examples across 1-6 wedges.
"""

import numpy as np
import sys
import os
import time
import pickle
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import RisleyParameters, ForwardModel, PatternAnalyzer, ReverseRisleySolver


def generate_large_training_set(n_samples=3000, save_to_file=True):
    """
    Generate a large, balanced training set across all wedge counts.

    Args:
        n_samples: Total number of samples (will be divided across 1-6 wedges)
        save_to_file: Whether to save the dataset to disk

    Returns:
        List of (pattern, params) tuples
    """
    print("="*60)
    print(f"GENERATING LARGE TRAINING SET ({n_samples} samples)")
    print("="*60)

    forward_model = ForwardModel()
    analyzer = PatternAnalyzer()

    # Divide samples across wedge counts
    samples_per_wedge = n_samples // 6
    training_data = []

    for wedge_count in range(1, 7):
        print(f"\nGenerating {samples_per_wedge} samples for {wedge_count} wedge(s)...")
        wedge_data = []
        attempts = 0

        while len(wedge_data) < samples_per_wedge and attempts < samples_per_wedge * 2:
            attempts += 1

            # Generate parameters with good variety
            if wedge_count == 1:
                # Single wedge - simpler parameter ranges
                speeds = [np.random.uniform(-3, 3)]
                phi_x = [np.random.uniform(-20, 20)]
                phi_y = [np.random.uniform(-20, 20)]
            elif wedge_count == 2:
                # Two wedges - often counter-rotating
                if np.random.rand() > 0.5:
                    # Counter-rotating
                    speed1 = np.random.uniform(0.5, 3)
                    speeds = [speed1, -speed1 * np.random.uniform(0.8, 1.2)]
                else:
                    speeds = [np.random.uniform(-3, 3) for _ in range(2)]
                phi_x = [np.random.uniform(-20, 20) for _ in range(2)]
                phi_y = [np.random.uniform(-20, 20) for _ in range(2)]
            else:
                # Multiple wedges - varied speeds
                speeds = np.random.uniform(-2, 2, wedge_count).tolist()
                phi_x = np.random.uniform(-15, 15, wedge_count).tolist()
                phi_y = np.random.uniform(-15, 15, wedge_count).tolist()

            params = RisleyParameters(
                wedge_count=wedge_count,
                rotation_speeds=speeds,
                phi_x=phi_x,
                phi_y=phi_y
            )

            # Validate parameters
            if not params.validate():
                continue

            try:
                # Generate pattern (use 40 time points for efficiency)
                pattern = forward_model.simulate(params, time_points=40)

                # Extract features for validation
                features = analyzer.extract_features(pattern)

                # Basic quality check - pattern should have reasonable extent
                if features['pattern_area'] > 0.1 and features['max_radius'] > 0.1:
                    wedge_data.append((pattern, params, features))

                    if len(wedge_data) % 100 == 0:
                        print(f"  Generated {len(wedge_data)}/{samples_per_wedge}")

            except Exception as e:
                if attempts % 100 == 0:
                    print(f"  Warning: Failed attempts: {attempts}")

        training_data.extend(wedge_data)
        print(f"  Total for {wedge_count} wedges: {len(wedge_data)}")

    print(f"\nTotal training samples generated: {len(training_data)}")

    # Save to file
    if save_to_file:
        filename = f"training_data_{len(training_data)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(training_data, f)
        print(f"Saved training data to: {filename}")

    return training_data


def train_on_large_dataset(training_data, epochs=150):
    """
    Train the neural network on a large dataset.

    Args:
        training_data: List of (pattern, params, features) tuples
        epochs: Number of training epochs

    Returns:
        Trained solver
    """
    print("\n" + "="*60)
    print(f"TRAINING NEURAL NETWORK ({len(training_data)} samples)")
    print("="*60)

    solver = ReverseRisleySolver()

    # Prepare data for training
    features_list = []
    params_list = []

    print("\nPreparing training data...")
    for pattern, params, features in training_data:
        features_list.append(features)
        params_list.append(params)

    # Train the network
    print(f"\nTraining for {epochs} epochs...")
    start_time = time.time()

    history = solver.neural_predictor.train(
        features_list,
        params_list,
        validation_split=0.2,
        epochs=epochs
    )

    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.1f} seconds")
    print(f"Final training loss: {history['train_loss'][-1]:.6f}")
    print(f"Final validation loss: {history['val_loss'][-1]:.6f}")

    # Save the trained model
    solver.neural_predictor.save('trained_model_large')
    print("Model saved to: trained_model_large/")

    return solver


def validate_multi_wedge(solver, n_test_samples=600):
    """
    Comprehensive validation across all wedge counts.

    Args:
        solver: Trained solver
        n_test_samples: Number of test samples

    Returns:
        Detailed validation results
    """
    print("\n" + "="*60)
    print(f"VALIDATION ON {n_test_samples} TEST SAMPLES")
    print("="*60)

    forward_model = ForwardModel()
    analyzer = PatternAnalyzer()

    # Generate test data (100 samples per wedge count)
    samples_per_wedge = n_test_samples // 6

    results = {
        'by_wedge': {i: {
            'total': 0,
            'wedge_correct': 0,
            'speed_errors': [],
            'angle_errors': [],
            'pattern_mse': [],
            'pattern_mse_ga': []
        } for i in range(1, 7)},
        'overall': {
            'wedge_accuracy': 0,
            'confusion_matrix': np.zeros((6, 6), dtype=int)  # true vs predicted
        }
    }

    print("\nGenerating test samples and evaluating...")

    for true_wedge_count in range(1, 7):
        print(f"\nTesting {true_wedge_count}-wedge systems:")

        for i in range(samples_per_wedge):
            # Generate test parameters
            if true_wedge_count == 1:
                speeds = [np.random.uniform(-2.5, 2.5)]
                phi_x = [np.random.uniform(-18, 18)]
                phi_y = [np.random.uniform(-18, 18)]
            else:
                speeds = np.random.uniform(-2, 2, true_wedge_count).tolist()
                phi_x = np.random.uniform(-15, 15, true_wedge_count).tolist()
                phi_y = np.random.uniform(-15, 15, true_wedge_count).tolist()

            true_params = RisleyParameters(
                wedge_count=true_wedge_count,
                rotation_speeds=speeds,
                phi_x=phi_x,
                phi_y=phi_y
            )

            if not true_params.validate():
                continue

            try:
                # Generate pattern
                pattern = forward_model.simulate(true_params, time_points=40)

                # Get NN prediction
                features = analyzer.extract_features(pattern)
                nn_pred = solver.neural_predictor.predict(features)
                pred_wedge_count = nn_pred['wedge_count']

                # Update results
                results['by_wedge'][true_wedge_count]['total'] += 1
                results['overall']['confusion_matrix'][true_wedge_count-1][pred_wedge_count-1] += 1

                if pred_wedge_count == true_wedge_count:
                    results['by_wedge'][true_wedge_count]['wedge_correct'] += 1

                    # Calculate parameter errors
                    speed_error = np.mean([
                        (nn_pred['rotation_speeds'][j] - true_params.rotation_speeds[j])**2
                        for j in range(true_wedge_count)
                    ])
                    angle_x_error = np.mean([
                        (nn_pred['phi_x'][j] - true_params.phi_x[j])**2
                        for j in range(true_wedge_count)
                    ])
                    angle_y_error = np.mean([
                        (nn_pred['phi_y'][j] - true_params.phi_y[j])**2
                        for j in range(true_wedge_count)
                    ])

                    results['by_wedge'][true_wedge_count]['speed_errors'].append(np.sqrt(speed_error))
                    results['by_wedge'][true_wedge_count]['angle_errors'].append(
                        np.sqrt((angle_x_error + angle_y_error) / 2)
                    )

                    # Calculate pattern reconstruction error
                    nn_params = RisleyParameters(
                        wedge_count=pred_wedge_count,
                        rotation_speeds=nn_pred['rotation_speeds'][:pred_wedge_count],
                        phi_x=nn_pred['phi_x'][:pred_wedge_count],
                        phi_y=nn_pred['phi_y'][:pred_wedge_count]
                    )

                    nn_pattern = forward_model.simulate(nn_params, time_points=40)
                    pattern_mse = np.mean((pattern - nn_pattern)**2)
                    results['by_wedge'][true_wedge_count]['pattern_mse'].append(pattern_mse)

                # Progress update
                if (i + 1) % 20 == 0:
                    correct = results['by_wedge'][true_wedge_count]['wedge_correct']
                    total = results['by_wedge'][true_wedge_count]['total']
                    acc = correct / total * 100 if total > 0 else 0
                    print(f"  Processed {i+1}/{samples_per_wedge} - Wedge accuracy: {acc:.1f}%")

            except Exception as e:
                print(f"  Test failed: {e}")

    return results


def print_validation_results(results):
    """Print comprehensive validation results."""

    print("\n" + "="*60)
    print("VALIDATION RESULTS")
    print("="*60)

    # Overall wedge accuracy
    total_correct = sum(r['wedge_correct'] for r in results['by_wedge'].values())
    total_samples = sum(r['total'] for r in results['by_wedge'].values())
    overall_accuracy = total_correct / total_samples * 100 if total_samples > 0 else 0

    print(f"\nOVERALL WEDGE COUNT ACCURACY: {overall_accuracy:.1f}% ({total_correct}/{total_samples})")

    # Per-wedge results
    print("\nPER-WEDGE BREAKDOWN:")
    print("-" * 40)

    for wedge_count in range(1, 7):
        data = results['by_wedge'][wedge_count]
        if data['total'] > 0:
            wedge_acc = data['wedge_correct'] / data['total'] * 100
            print(f"\n{wedge_count} Wedge(s):")
            print(f"  Wedge accuracy: {wedge_acc:.1f}% ({data['wedge_correct']}/{data['total']})")

            if data['speed_errors']:
                avg_speed_rmse = np.mean(data['speed_errors'])
                avg_angle_rmse = np.mean(data['angle_errors'])
                avg_pattern_mse = np.mean(data['pattern_mse'])

                print(f"  When wedge count correct:")
                print(f"    Speed RMSE: {avg_speed_rmse:.3f} Hz")
                print(f"    Angle RMSE: {avg_angle_rmse:.2f} degrees")
                print(f"    Pattern MSE: {avg_pattern_mse:.3f}")

    # Confusion matrix
    print("\nCONFUSION MATRIX (True vs Predicted):")
    print("-" * 40)
    print("True\\Pred", end="")
    for i in range(1, 7):
        print(f"\t{i}", end="")
    print()

    for true_idx in range(6):
        print(f"{true_idx+1}\t", end="")
        for pred_idx in range(6):
            count = results['overall']['confusion_matrix'][true_idx][pred_idx]
            print(f"{count}\t", end="")
        print()

    # Summary statistics
    print("\nSUMMARY STATISTICS:")
    print("-" * 40)

    all_speed_errors = []
    all_angle_errors = []
    all_pattern_mse = []

    for data in results['by_wedge'].values():
        all_speed_errors.extend(data['speed_errors'])
        all_angle_errors.extend(data['angle_errors'])
        all_pattern_mse.extend(data['pattern_mse'])

    if all_speed_errors:
        print(f"Overall Speed RMSE (when wedge correct): {np.mean(all_speed_errors):.3f} Hz")
        print(f"Overall Angle RMSE (when wedge correct): {np.mean(all_angle_errors):.2f} degrees")
        print(f"Overall Pattern MSE (when wedge correct): {np.mean(all_pattern_mse):.3f}")
        print(f"Median Pattern MSE: {np.median(all_pattern_mse):.3f}")


def test_with_ga_refinement(solver, n_examples=3):
    """
    Test a few examples with GA refinement to show improvement.
    """
    print("\n" + "="*60)
    print("GA REFINEMENT EXAMPLES")
    print("="*60)

    forward_model = ForwardModel()
    analyzer = PatternAnalyzer()

    test_cases = [
        {'wedges': 1, 'speeds': [2.0], 'phi_x': [15.0], 'phi_y': [10.0]},
        {'wedges': 2, 'speeds': [1.5, -1.5], 'phi_x': [10.0, 10.0], 'phi_y': [5.0, -5.0]},
        {'wedges': 3, 'speeds': [1.0, -0.5, 1.5], 'phi_x': [8.0, -12.0, 6.0], 'phi_y': [5.0, 8.0, -10.0]}
    ]

    for i, test in enumerate(test_cases[:n_examples]):
        print(f"\nExample {i+1}: {test['wedges']} wedge(s)")
        print("-" * 30)

        true_params = RisleyParameters(
            wedge_count=test['wedges'],
            rotation_speeds=test['speeds'],
            phi_x=test['phi_x'],
            phi_y=test['phi_y']
        )

        # Generate pattern
        pattern = forward_model.simulate(true_params, time_points=50)

        # NN prediction
        features = analyzer.extract_features(pattern)
        nn_pred = solver.neural_predictor.predict(features)

        print(f"True: S={test['speeds']}, X={test['phi_x']}, Y={test['phi_y']}")
        print(f"NN:   S={[f'{s:.2f}' for s in nn_pred['rotation_speeds'][:nn_pred['wedge_count']]]}")

        # Calculate NN error
        if nn_pred['wedge_count'] == test['wedges']:
            nn_params = RisleyParameters(
                wedge_count=nn_pred['wedge_count'],
                rotation_speeds=nn_pred['rotation_speeds'][:nn_pred['wedge_count']],
                phi_x=nn_pred['phi_x'][:nn_pred['wedge_count']],
                phi_y=nn_pred['phi_y'][:nn_pred['wedge_count']]
            )
            nn_pattern = forward_model.simulate(nn_params, time_points=50)
            nn_mse = np.mean((pattern - nn_pattern)**2)
            print(f"NN Pattern MSE: {nn_mse:.4f}")

            # GA refinement (quick version)
            print("Running GA refinement (20 generations)...")
            from genetic_algorithm import GAConfig, GeneticAlgorithm

            ga_config = GAConfig(
                population_size=30,
                generations=20,
                mutation_rate=0.15
            )

            ga = GeneticAlgorithm(ga_config, forward_model)
            refined, fitness, _ = ga.optimize(
                target_pattern=pattern,
                wedge_count=nn_pred['wedge_count'],
                initial_params=nn_pred,
                verbose=False
            )

            print(f"GA:   S={[f'{s:.2f}' for s in refined['rotation_speeds']]}")
            print(f"GA Pattern MSE: {fitness:.4f}")
            print(f"Improvement: {(nn_mse - fitness) / nn_mse * 100:.1f}%")
        else:
            print(f"Wrong wedge count predicted: {nn_pred['wedge_count']}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train and validate multi-wedge Risley solver')
    parser.add_argument('--train-samples', type=int, default=3000,
                       help='Number of training samples (default: 3000)')
    parser.add_argument('--test-samples', type=int, default=600,
                       help='Number of test samples (default: 600)')
    parser.add_argument('--epochs', type=int, default=150,
                       help='Training epochs (default: 150)')
    parser.add_argument('--load-data', type=str, default=None,
                       help='Load training data from file')
    parser.add_argument('--ga-test', action='store_true',
                       help='Test GA refinement examples')

    args = parser.parse_args()

    print("LARGE-SCALE MULTI-WEDGE TRAINING AND VALIDATION")
    print("="*60)
    print(f"Configuration:")
    print(f"  Training samples: {args.train_samples}")
    print(f"  Test samples: {args.test_samples}")
    print(f"  Epochs: {args.epochs}")
    print("="*60)

    # Generate or load training data
    if args.load_data:
        print(f"\nLoading training data from: {args.load_data}")
        with open(args.load_data, 'rb') as f:
            training_data = pickle.load(f)
        print(f"Loaded {len(training_data)} samples")
    else:
        training_data = generate_large_training_set(args.train_samples)

    # Train the model
    solver = train_on_large_dataset(training_data, epochs=args.epochs)

    # Validate
    results = validate_multi_wedge(solver, n_test_samples=args.test_samples)
    print_validation_results(results)

    # Test with GA refinement
    if args.ga_test:
        test_with_ga_refinement(solver, n_examples=3)

    print("\n" + "="*60)
    print("TRAINING AND VALIDATION COMPLETE")
    print("="*60)