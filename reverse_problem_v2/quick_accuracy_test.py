#!/usr/bin/env python3
"""
Quick accuracy test with smaller dataset.
"""

import numpy as np
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import RisleyParameters, ForwardModel, PatternAnalyzer, ReverseRisleySolver


def quick_accuracy_test():
    """Quick test to get baseline accuracy metrics."""

    print("="*60)
    print("QUICK ACCURACY TEST")
    print("="*60)

    solver = ReverseRisleySolver()

    # Train on small dataset
    print("\n1. Training on 30 samples (5 per wedge count)...")
    train_data = []

    for wedge_count in range(1, 7):
        print(f"  Generating {wedge_count}-wedge samples...")
        for i in range(5):
            params = RisleyParameters(
                wedge_count=wedge_count,
                rotation_speeds=[np.random.uniform(-2, 2) for _ in range(wedge_count)],
                phi_x=[np.random.uniform(-15, 15) for _ in range(wedge_count)],
                phi_y=[np.random.uniform(-15, 15) for _ in range(wedge_count)]
            )
            try:
                pattern = solver.forward_model.simulate(params, time_points=30)
                train_data.append((pattern, params))
            except Exception as e:
                print(f"    Warning: Failed sample: {e}")

    solver.training_data = train_data
    print(f"  Generated {len(train_data)} training samples")

    # Train neural network
    print("\n2. Training neural network (50 epochs)...")
    history = solver.train_neural_network(n_samples=0, epochs=50)
    print(f"  Final loss: {history['train_loss'][-1]:.6f}")

    # Test on new samples
    print("\n3. Testing accuracy on 12 new samples...")
    test_results = []

    for wedge_count in range(1, 7):
        # Generate 2 test samples per wedge count
        for _ in range(2):
            # Generate test pattern
            true_params = RisleyParameters(
                wedge_count=wedge_count,
                rotation_speeds=[np.random.uniform(-2, 2) for _ in range(wedge_count)],
                phi_x=[np.random.uniform(-15, 15) for _ in range(wedge_count)],
                phi_y=[np.random.uniform(-15, 15) for _ in range(wedge_count)]
            )

            try:
                pattern = solver.forward_model.simulate(true_params, time_points=30)

                # Get NN prediction (no GA for speed)
                features = solver.analyzer.extract_features(pattern)
                predicted_dict = solver.neural_predictor.predict(features)

                # Compare
                wedge_correct = (predicted_dict['wedge_count'] == true_params.wedge_count)

                # Calculate errors
                if wedge_correct:
                    speed_error = np.mean([
                        (predicted_dict['rotation_speeds'][i] - true_params.rotation_speeds[i])**2
                        for i in range(wedge_count)
                    ])
                    angle_error = np.mean([
                        (predicted_dict['phi_x'][i] - true_params.phi_x[i])**2 +
                        (predicted_dict['phi_y'][i] - true_params.phi_y[i])**2
                        for i in range(wedge_count)
                    ]) / 2
                else:
                    speed_error = float('inf')
                    angle_error = float('inf')

                test_results.append({
                    'wedge_count': wedge_count,
                    'wedge_correct': wedge_correct,
                    'speed_error': speed_error,
                    'angle_error': angle_error,
                    'true_speeds': true_params.rotation_speeds,
                    'pred_speeds': predicted_dict['rotation_speeds'][:predicted_dict['wedge_count']],
                    'true_angles_x': true_params.phi_x,
                    'pred_angles_x': predicted_dict['phi_x'][:predicted_dict['wedge_count']]
                })

            except Exception as e:
                print(f"    Test failed for {wedge_count} wedges: {e}")

    # Print results
    print("\n" + "="*60)
    print("ACCURACY RESULTS")
    print("="*60)

    # Wedge count accuracy
    wedge_correct = sum(1 for r in test_results if r['wedge_correct'])
    print(f"\nWedge Count Accuracy: {wedge_correct}/{len(test_results)} ({wedge_correct/len(test_results)*100:.0f}%)")

    # Per-wedge analysis
    print("\nPer-wedge breakdown:")
    for w in range(1, 7):
        wedge_results = [r for r in test_results if r['wedge_count'] == w]
        if wedge_results:
            correct = sum(1 for r in wedge_results if r['wedge_correct'])
            print(f"  {w} wedge(s): {correct}/{len(wedge_results)} correct")

    # Parameter errors (when wedge count is correct)
    correct_results = [r for r in test_results if r['wedge_correct']]
    if correct_results:
        avg_speed_rmse = np.sqrt(np.mean([r['speed_error'] for r in correct_results]))
        avg_angle_rmse = np.sqrt(np.mean([r['angle_error'] for r in correct_results]))

        print(f"\nParameter Errors (when wedge count correct):")
        print(f"  Avg Speed RMSE: {avg_speed_rmse:.3f} Hz")
        print(f"  Avg Angle RMSE: {avg_angle_rmse:.3f} degrees")

    # Show a few examples
    print("\nExample predictions:")
    for i, r in enumerate(test_results[:3]):
        print(f"\n  Example {i+1} ({r['wedge_count']} wedges):")
        print(f"    Wedge count: {'✓' if r['wedge_correct'] else '✗'}")
        if r['wedge_correct']:
            print(f"    True speeds: {[f'{s:.2f}' for s in r['true_speeds']]}")
            print(f"    Pred speeds: {[f'{s:.2f}' for s in r['pred_speeds']]}")
            print(f"    Speed RMSE: {np.sqrt(r['speed_error']):.3f} Hz")

    return test_results


def test_with_ga_refinement():
    """Test one example with GA refinement to show improvement."""

    print("\n" + "="*60)
    print("GA REFINEMENT EXAMPLE")
    print("="*60)

    solver = ReverseRisleySolver()

    # Use pre-trained network if available
    try:
        solver.neural_predictor.load('trained_model')
        print("Loaded pre-trained model")
    except:
        print("Training quick model...")
        # Quick training
        train_data = []
        for wedge_count in [1, 2]:
            for _ in range(5):
                params = RisleyParameters(
                    wedge_count=wedge_count,
                    rotation_speeds=[np.random.uniform(-2, 2) for _ in range(wedge_count)],
                    phi_x=[np.random.uniform(-15, 15) for _ in range(wedge_count)],
                    phi_y=[np.random.uniform(-10, 10) for _ in range(wedge_count)]
                )
                pattern = solver.forward_model.simulate(params, time_points=30)
                train_data.append((pattern, params))

        solver.training_data = train_data
        solver.train_neural_network(n_samples=0, epochs=30)

    # Test case
    true_params = RisleyParameters(
        wedge_count=2,
        rotation_speeds=[1.5, -1.0],
        phi_x=[10.0, -5.0],
        phi_y=[5.0, 8.0]
    )

    print("\nTrue parameters:")
    print(f"  Wedges: {true_params.wedge_count}")
    print(f"  Speeds: {true_params.rotation_speeds}")
    print(f"  Angles X: {true_params.phi_x}")
    print(f"  Angles Y: {true_params.phi_y}")

    # Generate pattern
    pattern = solver.forward_model.simulate(true_params, time_points=40)

    # NN prediction
    print("\nNeural Network prediction:")
    features = solver.analyzer.extract_features(pattern)
    nn_pred = solver.neural_predictor.predict(features)
    print(f"  Wedges: {nn_pred['wedge_count']}")
    print(f"  Speeds: {[f'{s:.2f}' for s in nn_pred['rotation_speeds'][:nn_pred['wedge_count']]]}")
    print(f"  Angles X: {[f'{a:.2f}' for a in nn_pred['phi_x'][:nn_pred['wedge_count']]]}")

    # Calculate NN error
    if nn_pred['wedge_count'] == true_params.wedge_count:
        nn_params = RisleyParameters(
            wedge_count=nn_pred['wedge_count'],
            rotation_speeds=nn_pred['rotation_speeds'][:nn_pred['wedge_count']],
            phi_x=nn_pred['phi_x'][:nn_pred['wedge_count']],
            phi_y=nn_pred['phi_y'][:nn_pred['wedge_count']]
        )
        nn_pattern = solver.forward_model.simulate(nn_params, time_points=40)
        nn_mse = np.mean((pattern - nn_pattern)**2)
        print(f"  Pattern MSE: {nn_mse:.4f}")

        # GA refinement
        print("\nRunning GA refinement (10 generations)...")
        from genetic_algorithm import GAConfig, GeneticAlgorithm

        ga_config = GAConfig(
            population_size=20,
            generations=10,
            mutation_rate=0.15,
            mutation_scale=0.2
        )

        ga = GeneticAlgorithm(ga_config, solver.forward_model)
        refined, fitness, _ = ga.optimize(
            target_pattern=pattern,
            wedge_count=nn_pred['wedge_count'],
            initial_params=nn_pred,
            verbose=False
        )

        print("\nGA refined parameters:")
        print(f"  Speeds: {[f'{s:.2f}' for s in refined['rotation_speeds']]}")
        print(f"  Angles X: {[f'{a:.2f}' for a in refined['phi_x']]}")
        print(f"  Final fitness: {fitness:.4f}")

        # Improvement
        if fitness < nn_mse:
            improvement = (nn_mse - fitness) / nn_mse * 100
            print(f"\n✓ GA improved MSE by {improvement:.1f}%")
        else:
            print(f"\n✗ GA did not improve (may need more generations)")


if __name__ == "__main__":
    # Run quick test
    results = quick_accuracy_test()

    # Test GA refinement
    test_with_ga_refinement()