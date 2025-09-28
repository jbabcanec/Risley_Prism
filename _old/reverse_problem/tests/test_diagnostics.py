#!/usr/bin/env python3
"""
Diagnostic test to understand the complete data flow and identify issues.
"""

import sys
import os
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solver import StateOfTheArtSolver
from core.genetic_algorithm import solve_reverse_problem


def diagnose_pipeline():
    """Diagnose the complete prediction pipeline."""

    print("=" * 80)
    print("PIPELINE DIAGNOSTICS")
    print("=" * 80)

    solver = StateOfTheArtSolver()

    # Step 1: Generate known parameters
    print("\n1. PARAMETER GENERATION")
    print("-" * 40)

    true_params = {
        'wedgenum': 2,
        'rotation_speeds': [1.0, -1.5],
        'phi_x': [10.0, -15.0],
        'phi_y': [5.0, 8.0],
        'distances': [1.0, 5.0, 5.0],
        'refractive_indices': [1.0, 1.5, 1.5, 1.0]
    }

    print("Original parameters:")
    for key in ['wedgenum', 'rotation_speeds', 'phi_x', 'phi_y']:
        print(f"  {key}: {true_params[key]}")

    # Step 2: Forward simulate (current fake physics)
    print("\n2. FORWARD SIMULATION")
    print("-" * 40)

    pattern = solver.forward_simulate(true_params)
    print(f"Pattern shape: {pattern.shape}")
    print(f"Pattern sample (first 5 points):")
    for i in range(5):
        print(f"  [{i}]: x={pattern[i,0]:.4f}, y={pattern[i,1]:.4f}")

    # Analyze pattern characteristics
    print(f"\nPattern statistics:")
    print(f"  X: mean={np.mean(pattern[:,0]):.3f}, std={np.std(pattern[:,0]):.3f}, range={np.ptp(pattern[:,0]):.3f}")
    print(f"  Y: mean={np.mean(pattern[:,1]):.3f}, std={np.std(pattern[:,1]):.3f}, range={np.ptp(pattern[:,1]):.3f}")

    # Step 3: Neural Network Prediction
    print("\n3. NEURAL NETWORK PREDICTION")
    print("-" * 40)

    if solver.neural_predictor:
        nn_pred = solver.neural_predictor.predict(pattern)
        print(f"NN prediction available: {nn_pred is not None}")
        if nn_pred:
            print(f"  Predicted wedges: {nn_pred.get('wedgenum', 'N/A')}")
            print(f"  Confidence: {nn_pred.get('confidence', 'N/A')}")
            if 'parameters' in nn_pred:
                print(f"  Has parameters: Yes")
            else:
                print(f"  Has parameters: No")
    else:
        print("Neural network not loaded!")

    # Step 4: Direct GA Test
    print("\n4. DIRECT GENETIC ALGORITHM TEST")
    print("-" * 40)

    # Convert pattern to GA format (with time)
    time_vals = np.linspace(0, 2.0, len(pattern))
    target_pattern_with_time = [(pattern[i,0], pattern[i,1], time_vals[i])
                                for i in range(len(pattern))]

    print("Testing GA directly...")
    recovered_params, cost, info = solve_reverse_problem(
        target_pattern=target_pattern_with_time,
        wedge_count=2,
        population_size=50,
        generations=20,
        verbose=False
    )

    print(f"GA returned parameters: {recovered_params is not None}")
    if recovered_params:
        print(f"  Keys in recovered_params: {list(recovered_params.keys())}")
        print(f"  Rotation speeds: {recovered_params.get('rotation_speeds', 'N/A')}")
        print(f"  Cost: {cost:.4f}")

    # Step 5: Full Pipeline Test
    print("\n5. FULL PIPELINE TEST (intelligent_wedge_selection)")
    print("-" * 40)

    predicted_wedges, final_cost, params, opt_info = solver.intelligent_wedge_selection(
        pattern, verbose=True
    )

    print(f"\nResults:")
    print(f"  Predicted wedges: {predicted_wedges}")
    print(f"  Final cost: {final_cost:.4f}")
    print(f"  Params returned: {params is not None}")
    if params:
        print(f"  Param keys: {list(params.keys())}")
    else:
        print(f"  Params is: {params}")

    # Step 6: Verification
    print("\n6. VERIFICATION")
    print("-" * 40)

    if params:
        # Try to reconstruct
        if 'wedgenum' not in params:
            params['wedgenum'] = predicted_wedges
        try:
            reconstructed = solver.forward_simulate(params)
            mse = np.mean((pattern - reconstructed)**2)
            print(f"Reconstruction successful, MSE: {mse:.6f}")
        except Exception as e:
            print(f"Reconstruction failed: {e}")
    else:
        print("No parameters to verify!")

    # Step 7: Pattern complexity check
    print("\n7. PATTERN COMPLEXITY ANALYSIS")
    print("-" * 40)

    complexity = solver.calculate_pattern_complexity(pattern)
    print(f"Pattern complexity: {complexity:.3f}")

    # Check if pattern is too simple (might be degenerate)
    fft_x = np.abs(np.fft.fft(pattern[:, 0]))
    fft_y = np.abs(np.fft.fft(pattern[:, 1]))

    dominant_freq_x = np.argmax(fft_x[1:len(fft_x)//2]) + 1
    dominant_freq_y = np.argmax(fft_y[1:len(fft_y)//2]) + 1

    print(f"Dominant frequencies: X={dominant_freq_x}, Y={dominant_freq_y}")
    print(f"Expected frequencies from rotation speeds: ~{[abs(s)*30 for s in true_params['rotation_speeds']]}")


if __name__ == "__main__":
    print("\n🔬 RUNNING PIPELINE DIAGNOSTICS\n")
    diagnose_pipeline()
    print("\n" + "=" * 80)
    print("DIAGNOSTICS COMPLETE")
    print("=" * 80)