#!/usr/bin/env python3
"""
Fast Dataset Generation Pipeline
Generates datasets quickly with optimizations for speed
"""

import numpy as np
import sys
import os
import pickle
import json
from datetime import datetime
from typing import Dict, List, Tuple
import time

# Add parent for model access
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import inputs
import model


class FastDatasetGenerator:
    """Optimized dataset generator for rapid data creation."""

    def __init__(self):
        """Initialize with parameter ranges from inputs.py."""

        # Parameter ranges - focused for faster convergence
        self.ranges = {
            'wedge_count': (1, 6),  # 1-6 wedges
            'rotation_speed': (-5, 5),  # Hz (N parameter) - reduced range
            'theta_x': (5, 25),  # Initial laser angle x degrees
            'theta_y': (5, 25),  # Initial laser angle y degrees
            'phi_x': (-20, 20),  # Wedge angle x, focused range
            'phi_y': (-20, 20),  # Wedge angle y, focused range
            'ref_index': (1.4, 1.6),  # Typical optical glass range
            'distance': (7, 9),  # Inter-wedge distance
            'workpiece_distance': (90, 110)  # Distance to workpiece
        }

        # Fixed parameters optimized for speed
        self.fixed = {
            'diameter_x': 10,  # DX from inputs.py
            'diameter_y': 10,  # DY from inputs.py
            'time_limit': 0.5,  # Seconds of simulation (fast)
            'time_points': 20,  # Number of points (minimum needed)
            'rx': 0,  # Initial laser position x
            'ry': 0  # Initial laser position y
        }

    def generate_random_parameters(self, wedge_count: int = None) -> Dict:
        """
        Generate random parameters within valid ranges.

        Args:
            wedge_count: Number of wedges (if None, randomly chosen)

        Returns:
            Dictionary of parameters ready for simulation
        """

        # Randomly choose wedge count if not specified
        if wedge_count is None:
            wedge_count = np.random.randint(
                self.ranges['wedge_count'][0],
                self.ranges['wedge_count'][1] + 1
            )

        # Generate parameters
        params = {
            'wedge_count': wedge_count,
            'rotation_speeds': np.random.uniform(
                self.ranges['rotation_speed'][0],
                self.ranges['rotation_speed'][1],
                wedge_count
            ).tolist(),
            'theta_x': np.random.uniform(
                self.ranges['theta_x'][0],
                self.ranges['theta_x'][1]
            ),
            'theta_y': np.random.uniform(
                self.ranges['theta_y'][0],
                self.ranges['theta_y'][1]
            ),
            'phi_x': np.random.uniform(
                self.ranges['phi_x'][0],
                self.ranges['phi_x'][1],
                wedge_count
            ).tolist(),
            'phi_y': np.random.uniform(
                self.ranges['phi_y'][0],
                self.ranges['phi_y'][1],
                wedge_count
            ).tolist(),
            'ref_indices': np.random.uniform(
                self.ranges['ref_index'][0],
                self.ranges['ref_index'][1],
                wedge_count
            ).tolist(),
            'distances': np.random.uniform(
                self.ranges['distance'][0],
                self.ranges['distance'][1],
                wedge_count
            ).tolist()
        }

        # Add workpiece distance
        params['workpiece_distance'] = np.random.uniform(
            self.ranges['workpiece_distance'][0],
            self.ranges['workpiece_distance'][1]
        )

        return params

    def simulate_forward_fast(self, params: Dict) -> np.ndarray:
        """
        Run forward simulation quickly with minimal output.

        Args:
            params: Dictionary of parameters

        Returns:
            Array of (x, y) positions at workpiece
        """

        # Save current state
        original_state = self._save_inputs_state()

        try:
            # Configure inputs for simulation
            inputs.WEDGENUM = params['wedge_count']
            inputs.N = params['rotation_speeds']
            inputs.STARTPHIX = params['phi_x']
            inputs.STARTPHIY = params['phi_y']

            # Set initial laser angles
            inputs.THETAX = params['theta_x']
            inputs.THETAY = params['theta_y']

            # Set distances and refractive indices
            inputs.int_dist = params['distances'] + [params['workpiece_distance']]
            inputs.ref_ind = [1.0] + params['ref_indices']  # Air + wedges

            # Time and output settings
            inputs.TIMELIM = self.fixed['time_limit']
            inputs.INC = self.fixed['time_points']
            inputs.plotit = 'off'
            inputs.printit = 'off'  # Critical for speed

            # Run simulation
            model.main()

            # Read output
            pattern = self._read_latest_output()

            return pattern

        except Exception as e:
            # Return None on failure
            return None

        finally:
            # Restore original state
            self._restore_inputs_state(original_state)

    def _save_inputs_state(self) -> Dict:
        """Save current inputs.py state."""
        return {
            'WEDGENUM': inputs.WEDGENUM,
            'N': inputs.N[:] if hasattr(inputs.N, '__len__') else [inputs.N],
            'STARTPHIX': inputs.STARTPHIX[:] if hasattr(inputs.STARTPHIX, '__len__') else [inputs.STARTPHIX],
            'STARTPHIY': inputs.STARTPHIY[:] if hasattr(inputs.STARTPHIY, '__len__') else [inputs.STARTPHIY],
            'THETAX': inputs.THETAX if hasattr(inputs, 'THETAX') else 10,
            'THETAY': inputs.THETAY if hasattr(inputs, 'THETAY') else 5,
            'int_dist': inputs.int_dist[:] if hasattr(inputs.int_dist, '__len__') else [inputs.int_dist],
            'ref_ind': inputs.ref_ind[:] if hasattr(inputs.ref_ind, '__len__') else [inputs.ref_ind],
            'TIMELIM': inputs.TIMELIM,
            'INC': inputs.INC,
            'plotit': inputs.plotit,
            'printit': inputs.printit
        }

    def _restore_inputs_state(self, state: Dict):
        """Restore inputs.py state."""
        for key, value in state.items():
            setattr(inputs, key, value)

    def _read_latest_output(self) -> np.ndarray:
        """Read the latest simulation output quickly."""
        import csv

        # Find latest simulation directory
        output_dir = 'output/examples'
        dirs = [d for d in os.listdir(output_dir) if 'simulation' in d]
        if not dirs:
            return None
        latest_dir = sorted(dirs)[-1]

        # Read workpiece projections
        csv_file = f'{output_dir}/{latest_dir}/workpiece_projections.csv'

        pattern = []
        try:
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    x = float(row['X_Position'])
                    y = float(row['Y_Position'])
                    pattern.append([x, y])
        except:
            return None

        return np.array(pattern[:self.fixed['time_points']])

    def generate_dataset_batch(self, n_samples: int,
                              wedge_count: int = None) -> List[Dict]:
        """
        Generate a batch of samples quickly.

        Args:
            n_samples: Number of samples to generate
            wedge_count: Specific wedge count (None for random)

        Returns:
            List of dictionaries with parameters and patterns
        """

        samples = []
        attempts = 0
        max_attempts = n_samples * 3  # Allow for some failures

        while len(samples) < n_samples and attempts < max_attempts:
            attempts += 1

            # Generate parameters
            params = self.generate_random_parameters(wedge_count)

            # Run simulation
            pattern = self.simulate_forward_fast(params)

            if pattern is not None and len(pattern) > 0:
                # Store sample
                sample = {
                    'parameters': params,
                    'pattern': pattern,
                    'timestamp': datetime.now().isoformat()
                }
                samples.append(sample)

                # Progress update every 10 samples
                if len(samples) % 10 == 0:
                    print(f"    Generated {len(samples)}/{n_samples} samples")

        return samples

    def generate_fast_dataset(self, total_samples: int,
                             save_dir: str = 'datasets') -> str:
        """
        Generate dataset quickly with smart sampling.

        Args:
            total_samples: Total number of samples to generate
            save_dir: Directory to save dataset

        Returns:
            Path to saved dataset file
        """

        # Create save directory
        os.makedirs(save_dir, exist_ok=True)

        # Create unique dataset ID
        dataset_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        dataset_file = os.path.join(save_dir, f'fast_dataset_{dataset_id}.pkl')

        print(f"\nFAST DATASET GENERATION")
        print(f"Target samples: {total_samples}")
        print(f"Output: {dataset_file}")
        print("-" * 60)

        # Distribute samples across wedge counts
        samples_per_wedge = total_samples // 6
        remainder = total_samples % 6

        all_samples = []
        start_time = time.time()

        for wedge_count in range(1, 7):
            n_samples = samples_per_wedge
            if wedge_count <= remainder:
                n_samples += 1

            print(f"\nGenerating {n_samples} samples for {wedge_count} wedge(s)...")

            # Generate batch
            batch = self.generate_dataset_batch(n_samples, wedge_count)
            all_samples.extend(batch)

            elapsed = time.time() - start_time
            rate = len(all_samples) / elapsed if elapsed > 0 else 0
            eta = (total_samples - len(all_samples)) / rate if rate > 0 else 0

            print(f"  Progress: {len(all_samples)}/{total_samples} samples")
            print(f"  Rate: {rate:.1f} samples/sec")
            print(f"  ETA: {eta/60:.1f} minutes")

        # Save dataset
        print(f"\nSaving dataset to {dataset_file}...")

        dataset = {
            'samples': all_samples,
            'metadata': {
                'dataset_id': dataset_id,
                'total_samples': len(all_samples),
                'creation_date': datetime.now().isoformat(),
                'parameter_ranges': self.ranges,
                'fixed_parameters': self.fixed,
                'generation_time': time.time() - start_time,
                'wedge_distribution': {
                    i: len([s for s in all_samples if s['parameters']['wedge_count'] == i])
                    for i in range(1, 7)
                }
            }
        }

        with open(dataset_file, 'wb') as f:
            pickle.dump(dataset, f)

        # Print summary
        print("\n" + "=" * 60)
        print("DATASET GENERATION COMPLETE")
        print("=" * 60)
        print(f"Total samples: {len(all_samples)}")
        print(f"Generation time: {time.time() - start_time:.1f} seconds")
        print(f"Dataset saved to: {dataset_file}")
        print("\nWedge distribution:")
        for w in range(1, 7):
            count = dataset['metadata']['wedge_distribution'][w]
            print(f"  {w} wedge(s): {count} samples")
        print("=" * 60)

        return dataset_file


def main():
    """Main function to generate dataset."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Fast dataset generation for Risley prism reverse problem'
    )
    parser.add_argument(
        '--samples', type=int, default=1000,
        help='Number of samples to generate (default: 1000)'
    )

    args = parser.parse_args()

    # Generate dataset
    generator = FastDatasetGenerator()
    dataset_file = generator.generate_fast_dataset(args.samples)

    print(f"\nDataset ready for training: {dataset_file}")

    return dataset_file


if __name__ == "__main__":
    main()