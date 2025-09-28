#!/usr/bin/env python3
"""
Test raw predictions to understand what the system is actually outputting.
This will help verify accuracy by comparing true parameters vs predictions.
"""

import sys
import os
import numpy as np
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solver import StateOfTheArtSolver


def test_raw_prediction_accuracy():
    """Test and display raw predictions vs ground truth."""

    print("=" * 80)
    print("RAW PREDICTION ACCURACY TEST")
    print("=" * 80)

    solver = StateOfTheArtSolver()

    # Test Case 1: Simple 1-wedge system
    print("\n TEST CASE 1: Single Wedge")
    print("-" * 40)

    true_params = {
        'wedgenum': 1,
        'rotation_speeds': [2.0],
        'phi_x': [15.0],
        'phi_y': [5.0],
        'distances': [1.0, 5.0],
        'refractive_indices': [1.0, 1.5, 1.0]
    }

    print("TRUE PARAMETERS:")
    print(f"  Wedges: {true_params['wedgenum']}")
    print(f"  Speed: {true_params['rotation_speeds']}")
    print(f"  Phi_x: {true_params['phi_x']}")
    print(f"  Phi_y: {true_params['phi_y']}")

    # Generate pattern
    pattern = solver.forward_simulate(true_params)
    print(f"\nGenerated pattern shape: {pattern.shape}")

    # Get prediction
    predicted_wedges, cost, recovered_params, info = solver.intelligent_wedge_selection(
        pattern, verbose=False
    )

    print("\nPREDICTED PARAMETERS:")
    print(f"  Wedges: {predicted_wedges} {'✓' if predicted_wedges == true_params['wedgenum'] else '✗'}")
    print(f"  Speed: {recovered_params.get('rotation_speeds', 'N/A')}")
    print(f"  Phi_x: {recovered_params.get('phi_x', 'N/A')}")
    print(f"  Phi_y: {recovered_params.get('phi_y', 'N/A')}")
    print(f"  Cost: {cost:.4f}")

    # Verify by reconstruction
    print("\nVERIFICATION:")
    if recovered_params:
        if 'wedgenum' not in recovered_params:
            recovered_params['wedgenum'] = predicted_wedges
        reconstructed = solver.forward_simulate(recovered_params)
    else:
        # If no params recovered, generate new ones for the predicted wedge count
        recovered_params = solver.generate_parameters(predicted_wedges)
        reconstructed = solver.forward_simulate(recovered_params)
    mse = np.mean((pattern - reconstructed)**2)
    print(f"  Reconstruction MSE: {mse:.6f}")

    # Test Case 2: Complex multi-wedge
    print("\n TEST CASE 2: Three Wedges")
    print("-" * 40)

    true_params = {
        'wedgenum': 3,
        'rotation_speeds': [1.0, -1.5, 2.0],
        'phi_x': [10.0, -12.0, 8.0],
        'phi_y': [5.0, 7.0, -5.0],
        'distances': [1.0, 5.0, 5.0, 5.0],
        'refractive_indices': [1.0, 1.5, 1.5, 1.5, 1.0]
    }

    print("TRUE PARAMETERS:")
    print(f"  Wedges: {true_params['wedgenum']}")
    print(f"  Speed: {true_params['rotation_speeds']}")
    print(f"  Phi_x: {true_params['phi_x']}")
    print(f"  Phi_y: {true_params['phi_y']}")

    pattern = solver.forward_simulate(true_params)
    predicted_wedges, cost, recovered_params, info = solver.intelligent_wedge_selection(
        pattern, verbose=False
    )

    print("\nPREDICTED PARAMETERS:")
    print(f"  Wedges: {predicted_wedges} {'✓' if predicted_wedges == true_params['wedgenum'] else '✗'}")
    print(f"  Speed: {recovered_params.get('rotation_speeds', 'N/A')}")
    print(f"  Phi_x: {recovered_params.get('phi_x', 'N/A')}")
    print(f"  Phi_y: {recovered_params.get('phi_y', 'N/A')}")
    print(f"  Cost: {cost:.4f}")

    print("\nVERIFICATION:")
    if recovered_params:
        if 'wedgenum' not in recovered_params:
            recovered_params['wedgenum'] = predicted_wedges
        reconstructed = solver.forward_simulate(recovered_params)
    else:
        # If no params recovered, generate new ones for the predicted wedge count
        recovered_params = solver.generate_parameters(predicted_wedges)
        reconstructed = solver.forward_simulate(recovered_params)
    mse = np.mean((pattern - reconstructed)**2)
    print(f"  Reconstruction MSE: {mse:.6f}")

    return predicted_wedges == true_params['wedgenum']


def test_batch_accuracy():
    """Test accuracy across multiple random cases."""

    print("\n" + "=" * 80)
    print("BATCH ACCURACY TEST")
    print("=" * 80)

    solver = StateOfTheArtSolver()
    results = []

    samples_per_wedge = 5

    for wedge_count in range(1, 7):
        print(f"\nTesting {wedge_count} wedge(s):")

        for i in range(samples_per_wedge):
            # Generate random parameters
            params = solver.generate_parameters(wedge_count)
            pattern = solver.forward_simulate(params)

            # Predict
            predicted_wedges, cost, recovered_params, _ = solver.intelligent_wedge_selection(
                pattern, verbose=False
            )

            correct = (predicted_wedges == wedge_count)
            results.append({
                'true': wedge_count,
                'predicted': predicted_wedges,
                'correct': correct,
                'cost': cost
            })

            status = '✓' if correct else '✗'
            print(f"  Sample {i+1}: True={wedge_count}, Pred={predicted_wedges} {status}, Cost={cost:.3f}")

    # Calculate statistics
    print("\n" + "-" * 40)
    print("STATISTICS:")

    overall_accuracy = sum(r['correct'] for r in results) / len(results)
    print(f"  Overall Accuracy: {overall_accuracy:.1%}")

    for w in range(1, 7):
        w_results = [r for r in results if r['true'] == w]
        if w_results:
            w_accuracy = sum(r['correct'] for r in w_results) / len(w_results)
            print(f"  {w}-wedge Accuracy: {w_accuracy:.1%}")

    return results


if __name__ == "__main__":
    print("\n🔬 TESTING RAW PREDICTIONS\n")

    # Test specific cases with raw output
    test_raw_prediction_accuracy()

    # Test batch accuracy
    results = test_batch_accuracy()

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)