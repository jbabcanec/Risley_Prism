#!/usr/bin/env python3
"""
FIXED Solver - Uses ONLY real physics, NO FALLBACKS
"""

import os
import sys
import numpy as np
import pickle

# Add path for main model
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.genetic_algorithm import solve_reverse_problem
from core.neural_network import NeuralPredictor


class FixedSolver:
    """Solver that uses ONLY real physics - no fallbacks."""

    def __init__(self):
        self.neural_predictor = None
        self._load_neural_predictor()

    def _load_neural_predictor(self):
        """Load neural network if available."""
        try:
            predictor = NeuralPredictor()
            if predictor.load():
                self.neural_predictor = predictor
                print("Neural network loaded")
        except:
            print("No neural network available")

    def forward_simulate_real(self, params):
        """
        Use the ACTUAL forward model - no fallbacks, no fake physics.

        This calls your verified model.py directly.
        """
        import model
        import inputs

        # Save original state
        original_state = {
            'WEDGENUM': inputs.WEDGENUM,
            'N': inputs.N[:],
            'STARTPHIX': inputs.STARTPHIX[:],
            'STARTPHIY': inputs.STARTPHIY[:],
            'int_dist': inputs.int_dist[:],
            'ref_ind': inputs.ref_ind[:],
            'TIMELIM': inputs.TIMELIM,
            'INC': inputs.INC,
            'plotit': inputs.plotit,
            'printit': inputs.printit
        }

        try:
            # Configure inputs for simulation
            wedge_count = params['wedgenum']
            inputs.WEDGENUM = wedge_count
            inputs.N = params['rotation_speeds'][:wedge_count]
            inputs.STARTPHIX = params['phi_x'][:wedge_count]
            inputs.STARTPHIY = params['phi_y'][:wedge_count]

            # Set distances and refractive indices correctly
            inputs.int_dist = [6.0] * wedge_count + [100.0]
            inputs.ref_ind = [1.0] + [1.5] * wedge_count

            # Time settings
            inputs.TIMELIM = 2.0
            inputs.INC = 60  # Number of points
            inputs.plotit = 'off'
            inputs.printit = 'off'

            # Run the actual model
            model.main()

            # Get the output from the latest simulation
            import os
            latest_dir = sorted([d for d in os.listdir('output/examples')
                               if 'simulation' in d])[-1]

            # Load the workpiece projections (what we actually care about)
            projections_file = f'output/examples/{latest_dir}/workpiece_projections.csv'

            # Read CSV and extract x,y positions
            pattern = []
            with open(projections_file, 'r') as f:
                lines = f.readlines()[1:]  # Skip header
                for line in lines[:60]:  # Take first 60 points
                    parts = line.strip().split(',')
                    x = float(parts[1])
                    y = float(parts[2])
                    pattern.append([x, y])

            return np.array(pattern)

        finally:
            # Restore original state
            for key, value in original_state.items():
                setattr(inputs, key, value)

    def solve_reverse(self, pattern, true_wedge_count=None):
        """
        Solve reverse problem for a pattern.
        NO FALLBACKS - if it fails, it fails.
        """

        # Try different wedge counts
        best_result = None
        best_cost = float('inf')

        for wedge_count in range(1, 7):
            print(f"Testing {wedge_count} wedges...")

            # Convert pattern to format expected by GA
            time_vals = np.linspace(0, 2.0, len(pattern))
            target_pattern = [(pattern[i,0], pattern[i,1], time_vals[i])
                            for i in range(len(pattern))]

            # Run GA
            recovered_params, cost, info = solve_reverse_problem(
                target_pattern=target_pattern,
                wedge_count=wedge_count,
                population_size=50,
                generations=30,
                parallel=False,
                verbose=False
            )

            print(f"  Cost: {cost:.4f}")

            if cost < best_cost:
                best_cost = cost
                best_result = {
                    'wedge_count': wedge_count,
                    'parameters': recovered_params,
                    'cost': cost
                }

        return best_result


def test_real_physics():
    """Test with ONLY real physics."""

    print("=" * 60)
    print("TESTING WITH REAL PHYSICS ONLY - NO FALLBACKS")
    print("=" * 60)

    solver = FixedSolver()

    # Test case
    true_params = {
        'wedgenum': 1,
        'rotation_speeds': [1.5],
        'phi_x': [10.0],
        'phi_y': [5.0]
    }

    print("\nGenerating pattern with REAL physics...")
    pattern = solver.forward_simulate_real(true_params)
    print(f"Pattern shape: {pattern.shape}")
    print(f"First 3 points:")
    for i in range(3):
        print(f"  [{i}]: ({pattern[i,0]:.4f}, {pattern[i,1]:.4f})")

    print("\nSolving reverse problem...")
    result = solver.solve_reverse(pattern, true_wedge_count=1)

    print(f"\nResult:")
    print(f"  Predicted wedges: {result['wedge_count']}")
    print(f"  Cost: {result['cost']:.4f}")
    if result['parameters']:
        print(f"  Recovered speeds: {result['parameters']['rotation_speeds']}")
        print(f"  Recovered phi_x: {result['parameters']['phi_x']}")
        print(f"  Recovered phi_y: {result['parameters']['phi_y']}")


if __name__ == "__main__":
    test_real_physics()