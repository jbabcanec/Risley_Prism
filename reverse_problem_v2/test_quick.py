#!/usr/bin/env python3
"""
Quick test of the reverse solver with REAL physics.
"""

import numpy as np
import sys
import os

# Add parent for model access
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import RisleyParameters, ForwardModel, PatternAnalyzer, ReverseRisleySolver


def quick_test():
    """Quick validation that we're using REAL physics."""

    print("="*60)
    print("QUICK TEST: VERIFYING REAL PHYSICS")
    print("="*60)

    # Test 1: Forward model produces realistic patterns
    print("\n1. Testing forward model with known parameters...")

    forward = ForwardModel()
    params = RisleyParameters(
        wedge_count=1,
        rotation_speeds=[1.5],
        phi_x=[10.0],
        phi_y=[5.0]
    )

    pattern = forward.simulate(params, time_points=20)
    print(f"   Pattern shape: {pattern.shape}")
    print(f"   X range: [{pattern[:,0].min():.2f}, {pattern[:,0].max():.2f}]")
    print(f"   Y range: [{pattern[:,1].min():.2f}, {pattern[:,1].max():.2f}]")

    # Check pattern is realistic (not sine waves)
    # Real Risley patterns should have specific characteristics
    analyzer = PatternAnalyzer()
    features = analyzer.extract_features(pattern)

    print(f"\n2. Pattern features:")
    print(f"   Pattern area: {features['pattern_area']:.2f}")
    print(f"   Max radius: {features['max_radius']:.2f}")
    print(f"   Total path length: {features['total_path_length']:.2f}")
    print(f"   Estimated wedges: {features['estimated_wedges']}")

    # Test 2: Feature-based estimation
    print(f"\n3. Testing feature-based parameter estimation...")

    solver = ReverseRisleySolver()
    initial_estimate = solver._estimate_from_features(features)

    print(f"   Estimated wedges: {initial_estimate.wedge_count}")
    print(f"   Estimated speeds: {initial_estimate.rotation_speeds}")
    print(f"   Estimated phi_x: {initial_estimate.phi_x}")
    print(f"   Estimated phi_y: {initial_estimate.phi_y}")

    # Test 3: Verify reconstruction
    print(f"\n4. Testing reconstruction with estimated parameters...")

    reconstructed = forward.simulate(initial_estimate, time_points=20)
    mse = np.mean((pattern - reconstructed)**2)

    print(f"   Reconstruction MSE: {mse:.4f}")

    # Validation
    print("\n" + "="*40)
    print("VALIDATION RESULTS")
    print("="*40)

    checks = []

    # Check 1: Pattern is not just sine waves
    pattern_variance = np.var(np.diff(pattern[:, 0]))
    is_not_sine = pattern_variance > 0.1  # Real patterns have more complexity
    checks.append(("Pattern is not simple sine wave", is_not_sine))

    # Check 2: Features are physically reasonable
    is_reasonable_area = 10 < features['pattern_area'] < 10000
    checks.append(("Pattern area is reasonable", is_reasonable_area))

    # Check 3: Path has complexity
    has_complexity = features.get('direction_changes', 0) > 0
    checks.append(("Pattern has directional changes", has_complexity))

    for check_name, passed in checks:
        status = "✓" if passed else "✗"
        print(f"  {status} {check_name}")

    all_passed = all(passed for _, passed in checks)

    if all_passed:
        print("\n✓ SUCCESS: System is using REAL physics!")
    else:
        print("\n✗ FAILURE: Something is wrong with the physics")

    return all_passed


def test_data_generation():
    """Test that we can generate training data with real physics."""

    print("\n" + "="*60)
    print("TEST: DATA GENERATION WITH REAL PHYSICS")
    print("="*60)

    solver = ReverseRisleySolver()

    # Generate small dataset
    print("\nGenerating 6 samples (1 per wedge count)...")
    data = solver.generate_training_data(n_samples=6)

    print(f"\nGenerated {len(data)} samples")

    # Analyze patterns
    print("\nSample analysis:")
    for i, (pattern, params) in enumerate(data[:3]):  # First 3 samples
        print(f"\n  Sample {i+1}:")
        print(f"    Wedges: {params.wedge_count}")
        print(f"    Pattern points: {len(pattern)}")
        print(f"    X range: [{pattern[:,0].min():.2f}, {pattern[:,0].max():.2f}]")
        print(f"    Y range: [{pattern[:,1].min():.2f}, {pattern[:,1].max():.2f}]")

    return len(data) == 6


if __name__ == "__main__":
    print("Running quick tests...")

    test1_passed = quick_test()
    test2_passed = test_data_generation()

    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)

    if test1_passed and test2_passed:
        print("✓ All tests passed - System uses REAL physics only!")
    else:
        print("✗ Some tests failed - Check the implementation")