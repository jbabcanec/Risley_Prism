#!/usr/bin/env python3
"""
CLEAN REVERSE RISLEY PRISM SOLVER
Uses real physics, no fallbacks, principled approach.
"""

import numpy as np
import sys
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import csv

# Add parent for model access
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from neural_network import RisleyNeuralPredictor
from genetic_algorithm import HybridGARefiner


@dataclass
class RisleyParameters:
    """Parameters defining a Risley prism system."""
    wedge_count: int
    rotation_speeds: List[float]  # Hz
    phi_x: List[float]  # degrees
    phi_y: List[float]  # degrees

    def validate(self) -> bool:
        """Check if parameters are physically valid."""
        if self.wedge_count < 1 or self.wedge_count > 6:
            return False
        if len(self.rotation_speeds) != self.wedge_count:
            return False
        if len(self.phi_x) != self.wedge_count:
            return False
        if len(self.phi_y) != self.wedge_count:
            return False
        # Physical limits
        if any(abs(s) > 10 for s in self.rotation_speeds):
            return False
        if any(abs(a) > 45 for a in self.phi_x + self.phi_y):
            return False
        return True


class PatternAnalyzer:
    """Extracts physically meaningful features from beam patterns."""

    @staticmethod
    def extract_features(pattern: np.ndarray) -> Dict:
        """Extract features that correlate with physical parameters."""

        features = {}

        # Basic statistics
        features['mean_x'] = np.mean(pattern[:, 0])
        features['mean_y'] = np.mean(pattern[:, 1])
        features['std_x'] = np.std(pattern[:, 0])
        features['std_y'] = np.std(pattern[:, 1])
        features['range_x'] = np.ptp(pattern[:, 0])
        features['range_y'] = np.ptp(pattern[:, 1])

        # Pattern extent (related to wedge angles)
        features['pattern_area'] = features['range_x'] * features['range_y']
        features['max_radius'] = np.max(np.sqrt(
            (pattern[:, 0] - features['mean_x'])**2 +
            (pattern[:, 1] - features['mean_y'])**2
        ))

        # Frequency analysis (reveals rotation speeds)
        if len(pattern) > 10:
            fft_x = np.abs(np.fft.fft(pattern[:, 0] - features['mean_x']))
            fft_y = np.abs(np.fft.fft(pattern[:, 1] - features['mean_y']))

            # Find dominant frequencies
            freq_bins = np.fft.fftfreq(len(pattern))
            positive_freqs = freq_bins[:len(freq_bins)//2]

            # Top 3 frequencies in each axis
            top_freq_idx_x = np.argsort(fft_x[:len(fft_x)//2])[-3:]
            top_freq_idx_y = np.argsort(fft_y[:len(fft_y)//2])[-3:]

            features['dom_freq_x_1'] = positive_freqs[top_freq_idx_x[-1]] if len(top_freq_idx_x) > 0 else 0
            features['dom_freq_x_2'] = positive_freqs[top_freq_idx_x[-2]] if len(top_freq_idx_x) > 1 else 0
            features['dom_freq_y_1'] = positive_freqs[top_freq_idx_y[-1]] if len(top_freq_idx_y) > 0 else 0
            features['dom_freq_y_2'] = positive_freqs[top_freq_idx_y[-2]] if len(top_freq_idx_y) > 1 else 0

            # Frequency spread (indicates multiple wedges)
            features['freq_spread'] = np.std(fft_x[:len(fft_x)//2]) + np.std(fft_y[:len(fft_y)//2])

        # Trajectory complexity
        if len(pattern) > 2:
            # Calculate path length
            path_segments = np.sqrt(np.sum(np.diff(pattern, axis=0)**2, axis=1))
            features['total_path_length'] = np.sum(path_segments)
            features['path_smoothness'] = np.std(path_segments)

            # Count direction changes (indicates pattern complexity)
            dx = np.diff(pattern[:, 0])
            dy = np.diff(pattern[:, 1])
            direction_changes_x = np.sum(np.diff(np.sign(dx)) != 0)
            direction_changes_y = np.sum(np.diff(np.sign(dy)) != 0)
            features['direction_changes'] = direction_changes_x + direction_changes_y

            # Curvature analysis
            if len(pattern) > 3:
                curvatures = []
                for i in range(1, len(pattern) - 1):
                    v1 = pattern[i] - pattern[i-1]
                    v2 = pattern[i+1] - pattern[i]
                    if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                        curvatures.append(np.arccos(np.clip(cos_angle, -1, 1)))

                features['mean_curvature'] = np.mean(curvatures) if curvatures else 0
                features['max_curvature'] = np.max(curvatures) if curvatures else 0

        # Pattern closure (closed loops indicate full rotations)
        start_end_distance = np.linalg.norm(pattern[-1] - pattern[0])
        features['closure_ratio'] = start_end_distance / features['max_radius'] if features['max_radius'] > 0 else 0

        # Estimate wedge count based on pattern complexity
        complexity_score = (
            features.get('direction_changes', 0) / 10 +
            features.get('freq_spread', 0) +
            features.get('mean_curvature', 0)
        )

        if complexity_score < 2:
            features['estimated_wedges'] = 1
        elif complexity_score < 5:
            features['estimated_wedges'] = 2
        elif complexity_score < 10:
            features['estimated_wedges'] = 3
        else:
            features['estimated_wedges'] = min(6, int(complexity_score / 3))

        return features


class ForwardModel:
    """Interface to the real Risley prism forward model."""

    @staticmethod
    def simulate(params: RisleyParameters, time_points: int = 60) -> np.ndarray:
        """
        Generate pattern using the REAL verified forward model.

        Returns:
            Array of (x, y) positions at workpiece
        """
        import model
        import inputs

        # Save original state
        original_state = {
            'WEDGENUM': inputs.WEDGENUM,
            'N': inputs.N[:] if hasattr(inputs.N, '__len__') else [inputs.N],
            'STARTPHIX': inputs.STARTPHIX[:] if hasattr(inputs.STARTPHIX, '__len__') else [inputs.STARTPHIX],
            'STARTPHIY': inputs.STARTPHIY[:] if hasattr(inputs.STARTPHIY, '__len__') else [inputs.STARTPHIY],
            'int_dist': inputs.int_dist[:] if hasattr(inputs.int_dist, '__len__') else [inputs.int_dist],
            'ref_ind': inputs.ref_ind[:] if hasattr(inputs.ref_ind, '__len__') else [inputs.ref_ind],
            'TIMELIM': inputs.TIMELIM,
            'INC': inputs.INC,
            'plotit': inputs.plotit,
            'printit': inputs.printit
        }

        try:
            # Configure for simulation
            inputs.WEDGENUM = params.wedge_count
            inputs.N = params.rotation_speeds
            inputs.STARTPHIX = params.phi_x
            inputs.STARTPHIY = params.phi_y

            # Proper array sizes
            inputs.int_dist = [6.0] * params.wedge_count + [100.0]
            inputs.ref_ind = [1.0] + [1.5] * params.wedge_count

            # Time settings
            inputs.TIMELIM = 2.0  # seconds
            inputs.INC = time_points
            inputs.plotit = 'off'
            inputs.printit = 'off'

            # Run simulation
            model.main()

            # Read output
            import csv
            latest_dir = sorted([d for d in os.listdir('output/examples')
                               if 'simulation' in d])[-1]
            csv_file = f'output/examples/{latest_dir}/workpiece_projections.csv'

            pattern = []
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    x = float(row['X_Position'])
                    y = float(row['Y_Position'])
                    pattern.append([x, y])

            return np.array(pattern[:time_points])  # Ensure correct length

        finally:
            # Restore state
            for key, value in original_state.items():
                setattr(inputs, key, value)


class ReverseRisleySolver:
    """Main solver combining NN and GA with real physics."""

    def __init__(self):
        self.forward_model = ForwardModel()
        self.analyzer = PatternAnalyzer()
        self.neural_predictor = RisleyNeuralPredictor()
        self.ga_refiner = HybridGARefiner(self.forward_model)
        self.training_data = None

    def generate_training_data(self, n_samples: int = 100) -> List[Tuple[np.ndarray, RisleyParameters]]:
        """Generate training data using real forward model."""

        print(f"Generating {n_samples} training samples with REAL physics...")
        data = []
        samples_per_wedge = n_samples // 6

        for wedge_count in range(1, 7):
            print(f"  Generating {samples_per_wedge} samples for {wedge_count} wedge(s)")

            for i in range(samples_per_wedge):
                # Generate random but realistic parameters
                params = RisleyParameters(
                    wedge_count=wedge_count,
                    rotation_speeds=[np.random.uniform(-3, 3) for _ in range(wedge_count)],
                    phi_x=[np.random.uniform(-20, 20) for _ in range(wedge_count)],
                    phi_y=[np.random.uniform(-20, 20) for _ in range(wedge_count)]
                )

                # Generate pattern
                try:
                    pattern = self.forward_model.simulate(params)
                    data.append((pattern, params))

                    if (i + 1) % 5 == 0:
                        print(f"    Generated {i+1}/{samples_per_wedge} for {wedge_count} wedges")
                except Exception as e:
                    print(f"    Warning: Failed to generate sample: {e}")

        print(f"  Total samples generated: {len(data)}")
        self.training_data = data
        return data

    def train_neural_network(self, n_samples: int = 100, epochs: int = 200):
        """Train neural network on generated data."""

        # Generate training data if not already done
        if self.training_data is None:
            self.generate_training_data(n_samples)

        # Extract features and parameters
        print("\nExtracting features from patterns...")
        features_list = []
        params_list = []

        for pattern, params in self.training_data:
            features = self.analyzer.extract_features(pattern)
            features_list.append(features)
            params_list.append(params)

        # Train neural network
        print("\nTraining neural network...")
        history = self.neural_predictor.train(features_list, params_list, epochs=epochs)

        return history

    def solve(self, pattern: np.ndarray, use_ga_refinement: bool = True) -> RisleyParameters:
        """
        Solve the reverse problem for a given pattern.

        Strategy:
        1. Extract features from pattern
        2. Use NN to get initial parameter guess
        3. Use GA to refine parameters with REAL physics
        4. Verify with forward model
        """

        # Extract features
        features = self.analyzer.extract_features(pattern)
        estimated_wedges = features['estimated_wedges']

        print(f"\nPattern analysis:")
        print(f"  Estimated wedges: {estimated_wedges}")
        print(f"  Pattern area: {features['pattern_area']:.2f}")
        print(f"  Dominant frequency: {features.get('dom_freq_x_1', 0):.2f}")

        # Get initial estimate
        if self.neural_predictor.network is not None:
            print(f"\nUsing neural network for initial estimate...")
            initial_dict = self.neural_predictor.predict(features)
        else:
            print(f"\nUsing feature-based initial estimate...")
            initial_params = self._estimate_from_features(features)
            initial_dict = {
                'wedge_count': initial_params.wedge_count,
                'rotation_speeds': initial_params.rotation_speeds,
                'phi_x': initial_params.phi_x,
                'phi_y': initial_params.phi_y
            }

        print(f"\nInitial estimate:")
        print(f"  Wedges: {initial_dict['wedge_count']}")
        print(f"  Speeds: {initial_dict['rotation_speeds']}")
        print(f"  Phi_x: {initial_dict['phi_x']}")
        print(f"  Phi_y: {initial_dict['phi_y']}")

        # GA refinement with REAL physics
        if use_ga_refinement:
            print(f"\nRefining with GA using REAL physics...")
            refined_dict = self.ga_refiner.refine(
                initial_dict, pattern, max_iterations=2, verbose=True
            )
        else:
            refined_dict = initial_dict

        # Convert to RisleyParameters
        return RisleyParameters(
            wedge_count=refined_dict['wedge_count'],
            rotation_speeds=refined_dict['rotation_speeds'],
            phi_x=refined_dict['phi_x'],
            phi_y=refined_dict['phi_y']
        )

    def _estimate_from_features(self, features: Dict) -> RisleyParameters:
        """Simple feature-based parameter estimation."""

        wedge_count = features['estimated_wedges']

        # Estimate rotation speeds from dominant frequencies
        speeds = []
        if features.get('dom_freq_x_1', 0) > 0:
            speeds.append(features['dom_freq_x_1'] * 2)  # Convert to Hz
        if features.get('dom_freq_x_2', 0) > 0 and wedge_count > 1:
            speeds.append(features['dom_freq_x_2'] * 2)

        # Fill remaining speeds
        while len(speeds) < wedge_count:
            speeds.append(np.random.uniform(0.5, 2))

        # Estimate angles from pattern extent
        angle_scale = min(features['pattern_area'] / 100, 20)  # Rough scaling
        phi_x = [np.random.uniform(-angle_scale, angle_scale) for _ in range(wedge_count)]
        phi_y = [np.random.uniform(-angle_scale/2, angle_scale/2) for _ in range(wedge_count)]

        return RisleyParameters(
            wedge_count=wedge_count,
            rotation_speeds=speeds[:wedge_count],
            phi_x=phi_x,
            phi_y=phi_y
        )


def test_solver():
    """Test the clean solver with REAL physics only."""

    print("="*60)
    print("TESTING CLEAN REVERSE RISLEY SOLVER WITH REAL PHYSICS")
    print("="*60)

    solver = ReverseRisleySolver()

    # Test Case 1: Simple single wedge
    print("\n" + "="*40)
    print("TEST 1: Single Wedge")
    print("="*40)

    true_params_1 = RisleyParameters(
        wedge_count=1,
        rotation_speeds=[2.0],
        phi_x=[15.0],
        phi_y=[10.0]
    )

    print(f"\nTrue parameters:")
    print(f"  Wedges: {true_params_1.wedge_count}")
    print(f"  Speeds: {true_params_1.rotation_speeds}")
    print(f"  Phi_x: {true_params_1.phi_x}")
    print(f"  Phi_y: {true_params_1.phi_y}")

    print(f"\nGenerating pattern with REAL physics...")
    pattern_1 = solver.forward_model.simulate(true_params_1, time_points=30)
    print(f"  Pattern shape: {pattern_1.shape}")
    print(f"  First 3 points:")
    for i in range(min(3, len(pattern_1))):
        print(f"    [{i}]: ({pattern_1[i,0]:.4f}, {pattern_1[i,1]:.4f})")

    # Solve without training (feature-based)
    print(f"\nSolving reverse problem (without NN training)...")
    recovered_1 = solver.solve(pattern_1, use_ga_refinement=False)

    print(f"\nRecovered parameters (initial estimate):")
    print(f"  Wedges: {recovered_1.wedge_count}")
    print(f"  Speeds: {recovered_1.rotation_speeds}")
    print(f"  Phi_x: {recovered_1.phi_x}")
    print(f"  Phi_y: {recovered_1.phi_y}")

    # Verify
    print(f"\nVerifying by reconstruction...")
    reconstructed_1 = solver.forward_model.simulate(recovered_1, time_points=30)
    mse_1 = np.mean((pattern_1 - reconstructed_1[:len(pattern_1)])**2)
    print(f"  Reconstruction MSE: {mse_1:.6f}")

    # Test Case 2: Two wedges after training
    print("\n" + "="*40)
    print("TEST 2: Training NN and Testing Two Wedges")
    print("="*40)

    # Train on small dataset
    print("\nTraining neural network on small dataset...")
    solver.train_neural_network(n_samples=30, epochs=50)

    true_params_2 = RisleyParameters(
        wedge_count=2,
        rotation_speeds=[1.0, -1.5],
        phi_x=[10.0, -15.0],
        phi_y=[5.0, 8.0]
    )

    print(f"\nTrue parameters:")
    print(f"  Wedges: {true_params_2.wedge_count}")
    print(f"  Speeds: {true_params_2.rotation_speeds}")
    print(f"  Phi_x: {true_params_2.phi_x}")
    print(f"  Phi_y: {true_params_2.phi_y}")

    print(f"\nGenerating pattern with REAL physics...")
    pattern_2 = solver.forward_model.simulate(true_params_2, time_points=40)
    print(f"  Pattern shape: {pattern_2.shape}")

    print(f"\nSolving with NN + GA refinement...")
    recovered_2 = solver.solve(pattern_2, use_ga_refinement=True)

    print(f"\nFinal recovered parameters:")
    print(f"  Wedges: {recovered_2.wedge_count}")
    print(f"  Speeds: {recovered_2.rotation_speeds}")
    print(f"  Phi_x: {recovered_2.phi_x}")
    print(f"  Phi_y: {recovered_2.phi_y}")

    # Final verification
    print(f"\nFinal verification...")
    reconstructed_2 = solver.forward_model.simulate(recovered_2, time_points=40)
    mse_2 = np.mean((pattern_2 - reconstructed_2[:len(pattern_2)])**2)
    print(f"  Final Reconstruction MSE: {mse_2:.6f}")

    # Success criteria
    print("\n" + "="*40)
    print("RESULTS SUMMARY")
    print("="*40)
    print(f"Test 1 (Single wedge) MSE: {mse_1:.6f}")
    print(f"Test 2 (Two wedges) MSE: {mse_2:.6f}")

    if mse_1 < 1.0 and mse_2 < 1.0:
        print("\n✓ SUCCESS: System working with REAL physics!")
    else:
        print("\n⚠ System needs more training or tuning")


if __name__ == "__main__":
    test_solver()