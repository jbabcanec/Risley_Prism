#!/usr/bin/env python3
"""
Train the system on thousands of examples and evaluate accuracy.
"""

import numpy as np
import sys
import os
import time
import pickle

# Add parent for model access
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import RisleyParameters, ForwardModel, PatternAnalyzer, ReverseRisleySolver


def evaluate_accuracy(solver, test_data):
    """
    Evaluate accuracy metrics on test data.

    Returns detailed accuracy statistics.
    """
    results = {
        'wedge_count_accuracy': [],
        'speed_mse': [],
        'angle_mse': [],
        'pattern_mse': [],
        'perfect_wedge_matches': 0,
        'close_matches': 0  # MSE < 1.0
    }

    print("\nEvaluating on test set...")

    for i, (pattern, true_params) in enumerate(test_data):
        print(f"  Testing sample {i+1}/{len(test_data)}...", end='\r')

        # Solve without GA refinement for speed (initial NN prediction only)
        recovered = solver.solve(pattern, use_ga_refinement=False)

        # Wedge count accuracy
        wedge_match = (recovered.wedge_count == true_params.wedge_count)
        results['wedge_count_accuracy'].append(wedge_match)
        if wedge_match:
            results['perfect_wedge_matches'] += 1

        # Parameter errors (only if wedge counts match)
        if wedge_match:
            speed_error = np.mean((np.array(recovered.rotation_speeds) -
                                   np.array(true_params.rotation_speeds))**2)
            angle_x_error = np.mean((np.array(recovered.phi_x) -
                                    np.array(true_params.phi_x))**2)
            angle_y_error = np.mean((np.array(recovered.phi_y) -
                                    np.array(true_params.phi_y))**2)

            results['speed_mse'].append(speed_error)
            results['angle_mse'].append((angle_x_error + angle_y_error) / 2)

        # Pattern reconstruction error
        try:
            reconstructed = solver.forward_model.simulate(recovered, len(pattern))
            pattern_error = np.mean((pattern - reconstructed)**2)
            results['pattern_mse'].append(pattern_error)

            if pattern_error < 1.0:
                results['close_matches'] += 1
        except:
            results['pattern_mse'].append(float('inf'))

    print()  # New line after progress
    return results


def train_full_system(n_train=1000, n_test=100):
    """
    Train the system with a large dataset and evaluate accuracy.
    """

    print("="*60)
    print("FULL SYSTEM TRAINING AND EVALUATION")
    print("="*60)

    solver = ReverseRisleySolver()

    # Generate training data
    print(f"\n1. Generating {n_train} training samples...")
    start_time = time.time()
    train_data = solver.generate_training_data(n_samples=n_train)
    print(f"   Time taken: {time.time() - start_time:.1f} seconds")

    # Train neural network
    print(f"\n2. Training neural network...")
    start_time = time.time()
    history = solver.train_neural_network(n_samples=0, epochs=100)  # Use already generated data
    print(f"   Final training loss: {history['train_loss'][-1]:.6f}")
    if history.get('val_loss'):
        print(f"   Final validation loss: {history['val_loss'][-1]:.6f}")
    print(f"   Time taken: {time.time() - start_time:.1f} seconds")

    # Generate test data
    print(f"\n3. Generating {n_test} test samples...")
    test_data = []
    samples_per_wedge = n_test // 6

    for wedge_count in range(1, 7):
        for _ in range(samples_per_wedge):
            params = RisleyParameters(
                wedge_count=wedge_count,
                rotation_speeds=[np.random.uniform(-3, 3) for _ in range(wedge_count)],
                phi_x=[np.random.uniform(-20, 20) for _ in range(wedge_count)],
                phi_y=[np.random.uniform(-20, 20) for _ in range(wedge_count)]
            )
            try:
                pattern = solver.forward_model.simulate(params, time_points=40)
                test_data.append((pattern, params))
            except:
                pass

    print(f"   Generated {len(test_data)} test samples")

    # Evaluate accuracy
    print(f"\n4. Evaluating accuracy...")
    results = evaluate_accuracy(solver, test_data)

    # Print results
    print("\n" + "="*60)
    print("ACCURACY RESULTS")
    print("="*60)

    print(f"\nWedge Count Prediction:")
    wedge_acc = np.mean(results['wedge_count_accuracy']) * 100
    print(f"  Correct wedge count: {results['perfect_wedge_matches']}/{len(test_data)} ({wedge_acc:.1f}%)")

    if results['speed_mse']:
        print(f"\nParameter Errors (when wedge count correct):")
        print(f"  Speed MSE: {np.mean(results['speed_mse']):.4f} (Hz²)")
        print(f"  Speed RMSE: {np.sqrt(np.mean(results['speed_mse'])):.4f} Hz")
        print(f"  Angle MSE: {np.mean(results['angle_mse']):.4f} (deg²)")
        print(f"  Angle RMSE: {np.sqrt(np.mean(results['angle_mse'])):.4f} deg")

    print(f"\nPattern Reconstruction:")
    valid_pattern_mse = [x for x in results['pattern_mse'] if x != float('inf')]
    if valid_pattern_mse:
        print(f"  Mean pattern MSE: {np.mean(valid_pattern_mse):.4f}")
        print(f"  Median pattern MSE: {np.median(valid_pattern_mse):.4f}")
        print(f"  Close matches (MSE < 1.0): {results['close_matches']}/{len(test_data)} ({results['close_matches']/len(test_data)*100:.1f}%)")

    # Save the trained model
    print(f"\n5. Saving trained model...")
    solver.neural_predictor.save('trained_model')
    print(f"   Model saved to 'trained_model/' directory")

    return solver, results


def test_specific_cases(solver):
    """
    Test on specific challenging cases.
    """
    print("\n" + "="*60)
    print("TESTING SPECIFIC CASES")
    print("="*60)

    test_cases = [
        {
            'name': 'Single slow wedge',
            'params': RisleyParameters(
                wedge_count=1,
                rotation_speeds=[0.5],
                phi_x=[10.0],
                phi_y=[5.0]
            )
        },
        {
            'name': 'Two counter-rotating wedges',
            'params': RisleyParameters(
                wedge_count=2,
                rotation_speeds=[2.0, -2.0],
                phi_x=[15.0, 15.0],
                phi_y=[0.0, 0.0]
            )
        },
        {
            'name': 'Three complex wedges',
            'params': RisleyParameters(
                wedge_count=3,
                rotation_speeds=[1.0, -1.5, 0.8],
                phi_x=[10.0, -5.0, 8.0],
                phi_y=[5.0, 10.0, -3.0]
            )
        }
    ]

    for test in test_cases:
        print(f"\nTest: {test['name']}")
        print(f"  True params: W={test['params'].wedge_count}, "
              f"S={test['params'].rotation_speeds}, "
              f"X={test['params'].phi_x}, "
              f"Y={test['params'].phi_y}")

        # Generate pattern
        pattern = solver.forward_model.simulate(test['params'], time_points=50)

        # Solve with NN only (fast)
        recovered_nn = solver.solve(pattern, use_ga_refinement=False)
        print(f"  NN prediction: W={recovered_nn.wedge_count}, "
              f"S={recovered_nn.rotation_speeds}, "
              f"X={recovered_nn.phi_x}, "
              f"Y={recovered_nn.phi_y}")

        # Calculate errors
        pattern_nn = solver.forward_model.simulate(recovered_nn, time_points=50)
        mse_nn = np.mean((pattern - pattern_nn)**2)
        print(f"  NN MSE: {mse_nn:.4f}")

        # Try with GA refinement on one example
        if test['name'] == 'Two counter-rotating wedges':
            print(f"  Running GA refinement...")
            recovered_ga = solver.solve(pattern, use_ga_refinement=True)
            print(f"  GA refined: W={recovered_ga.wedge_count}, "
                  f"S={recovered_ga.rotation_speeds}, "
                  f"X={recovered_ga.phi_x}, "
                  f"Y={recovered_ga.phi_y}")

            pattern_ga = solver.forward_model.simulate(recovered_ga, time_points=50)
            mse_ga = np.mean((pattern - pattern_ga)**2)
            print(f"  GA MSE: {mse_ga:.4f} (improvement: {(mse_nn-mse_ga)/mse_nn*100:.1f}%)")


if __name__ == "__main__":
    # Run with different training sizes to see scaling
    print("Testing accuracy with different training set sizes...\n")

    # Quick test with small dataset
    print("QUICK TEST (100 training samples)")
    print("-"*40)
    solver, results = train_full_system(n_train=100, n_test=30)

    # Test specific cases
    test_specific_cases(solver)

    print("\n\n" + "="*60)
    print("Would you like to train on more data? (This will take longer)")
    print("Recommended: 1000+ samples for good accuracy")

    # Uncomment for full training:
    # solver, results = train_full_system(n_train=1000, n_test=100)