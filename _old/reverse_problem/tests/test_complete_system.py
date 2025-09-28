#!/usr/bin/env python3
"""
Complete system test with verification of parameter recovery accuracy.

This test:
1. Generates known parameters
2. Creates patterns
3. Recovers parameters
4. Verifies accuracy by comparing patterns
"""

import sys
import os
import numpy as np
import json

# Add parent to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solver import StateOfTheArtSolver


def test_parameter_recovery():
    """Test complete parameter recovery with verification."""

    print("=" * 80)
    print("COMPLETE SYSTEM TEST - PARAMETER RECOVERY")
    print("=" * 80)

    solver = StateOfTheArtSolver()
    results = []

    # Test each wedge configuration
    for wedge_count in range(1, 7):
        print(f"\n{'=' * 40}")
        print(f"Testing {wedge_count} Wedge(s)")
        print('=' * 40)

        # Generate known parameters
        true_params = {
            'wedgenum': wedge_count,
            'rotation_speeds': np.random.uniform(-2.0, 2.0, wedge_count).tolist(),
            'phi_x': np.random.uniform(-15.0, 15.0, wedge_count).tolist(),
            'phi_y': np.random.uniform(-15.0, 15.0, wedge_count).tolist(),
            'distances': [1.0] + [5.0] * wedge_count + [100.0],
            'refractive_indices': [1.0] + [1.5] * wedge_count + [1.0]
        }

        print("\nTRUE PARAMETERS:")
        print(f"  Speeds: {[f'{s:.2f}' for s in true_params['rotation_speeds']]}")
        print(f"  Phi_x:  {[f'{p:.1f}°' for p in true_params['phi_x']]}")
        print(f"  Phi_y:  {[f'{p:.1f}°' for p in true_params['phi_y']]}")

        # Generate pattern
        true_pattern = solver.forward_simulate(true_params)

        # Recover parameters
        predicted_wedges, cost, recovered_params, info = solver.intelligent_wedge_selection(
            true_pattern, verbose=False
        )

        print("\nPREDICTED:")
        print(f"  Wedge count: {predicted_wedges} {'✓' if predicted_wedges == wedge_count else '✗'}")

        if recovered_params and predicted_wedges == wedge_count:
            print(f"  Speeds: {[f'{s:.2f}' for s in recovered_params.get('rotation_speeds', [])]}")
            print(f"  Phi_x:  {[f'{p:.1f}°' for p in recovered_params.get('phi_x', [])]}")
            print(f"  Phi_y:  {[f'{p:.1f}°' for p in recovered_params.get('phi_y', [])]}")

            # Calculate parameter errors
            speed_errors = []
            phi_x_errors = []
            phi_y_errors = []

            for i in range(wedge_count):
                if i < len(recovered_params.get('rotation_speeds', [])):
                    speed_errors.append(abs(true_params['rotation_speeds'][i] -
                                          recovered_params['rotation_speeds'][i]))
                if i < len(recovered_params.get('phi_x', [])):
                    phi_x_errors.append(abs(true_params['phi_x'][i] -
                                          recovered_params['phi_x'][i]))
                if i < len(recovered_params.get('phi_y', [])):
                    phi_y_errors.append(abs(true_params['phi_y'][i] -
                                          recovered_params['phi_y'][i]))

            if speed_errors:
                print(f"\nERRORS:")
                print(f"  Speed error: {np.mean(speed_errors):.3f} Hz (max: {np.max(speed_errors):.3f})")
            if phi_x_errors:
                print(f"  Phi_x error: {np.mean(phi_x_errors):.2f}° (max: {np.max(phi_x_errors):.2f})")
            if phi_y_errors:
                print(f"  Phi_y error: {np.mean(phi_y_errors):.2f}° (max: {np.max(phi_y_errors):.2f})")

        # Verify by reconstruction
        print("\nVERIFICATION BY RECONSTRUCTION:")
        if recovered_params:
            if 'wedgenum' not in recovered_params:
                recovered_params['wedgenum'] = predicted_wedges
            recovered_pattern = solver.forward_simulate(recovered_params)

            # Calculate pattern similarity
            if len(recovered_pattern) == len(true_pattern):
                mse = np.mean((true_pattern - recovered_pattern) ** 2)
                correlation_x = np.corrcoef(true_pattern[:, 0], recovered_pattern[:, 0])[0, 1]
                correlation_y = np.corrcoef(true_pattern[:, 1], recovered_pattern[:, 1])[0, 1]

                print(f"  Pattern MSE: {mse:.6f}")
                print(f"  Correlation: X={correlation_x:.3f}, Y={correlation_y:.3f}")

                # Judge quality
                if mse < 0.01:
                    quality = "EXCELLENT"
                elif mse < 0.1:
                    quality = "GOOD"
                elif mse < 1.0:
                    quality = "FAIR"
                else:
                    quality = "POOR"
                print(f"  Recovery Quality: {quality}")
            else:
                print(f"  Pattern length mismatch!")

        results.append({
            'wedge_count': wedge_count,
            'predicted': predicted_wedges,
            'correct': predicted_wedges == wedge_count,
            'cost': cost
        })

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    overall_accuracy = sum(r['correct'] for r in results) / len(results)
    print(f"\nOverall Wedge Count Accuracy: {overall_accuracy:.1%}")

    for w in range(1, 7):
        w_results = [r for r in results if r['wedge_count'] == w]
        if w_results:
            correct = w_results[0]['correct']
            status = '✓' if correct else '✗'
            print(f"  {w} wedges: {status}")

    avg_cost = np.mean([r['cost'] for r in results])
    print(f"\nAverage Cost: {avg_cost:.3f}")

    return results


def test_speed():
    """Test system processing speed."""

    print("\n" + "=" * 80)
    print("SPEED TEST")
    print("=" * 80)

    solver = StateOfTheArtSolver()
    import time

    # Test different wedge counts
    for wedge_count in [1, 3, 6]:
        params = solver.generate_parameters(wedge_count)
        pattern = solver.forward_simulate(params)

        start = time.time()
        predicted_wedges, cost, recovered_params, _ = solver.intelligent_wedge_selection(
            pattern, verbose=False
        )
        elapsed = time.time() - start

        print(f"\n{wedge_count} wedges: {elapsed:.3f}s ({1/elapsed:.1f} samples/sec)")


if __name__ == "__main__":
    print("\n🔬 COMPLETE SYSTEM TEST\n")

    # Run parameter recovery test
    results = test_parameter_recovery()

    # Run speed test
    test_speed()

    print("\n" + "=" * 80)
    print("COMPLETE SYSTEM TEST FINISHED")
    print("=" * 80)