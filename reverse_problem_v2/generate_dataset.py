#!/usr/bin/env python3
"""
Dataset Generation Pipeline
Generates tens of thousands of forward simulations with parameters
sampled from ranges specified in inputs.py
"""

import numpy as np
import sys
import os
import pickle
import json
from datetime import datetime
from typing import Dict, List, Tuple
import multiprocessing as mp
from functools import partial

# Add parent for model access
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import inputs
import model


class DatasetGenerator:
    """Clean dataset generator using real forward model."""

    def __init__(self):
        """Initialize with parameter ranges from inputs.py."""

        # Parameter ranges based on inputs.py documentation
        self.ranges = {
            'wedge_count': (1, 6),  # 1-6 wedges
            'rotation_speed': (-10, 10),  # Hz (N parameter)
            'theta_x': (0, 30),  # Initial laser angle x degrees (reduced range)
            'theta_y': (0, 30),  # Initial laser angle y degrees (reduced range)
            'phi_x': (-45, 45),  # Wedge angle x, reasonable range
            'phi_y': (-45, 45),  # Wedge angle y, reasonable range
            'ref_index': (1.3, 1.7),  # Typical optical glass range
            'distance': (6, 10),  # Inter-wedge distance
            'workpiece_distance': (80, 120)  # Distance to workpiece
        }

        # Fixed parameters
        self.fixed = {
            'diameter_x': 10,  # DX from inputs.py
            'diameter_y': 10,  # DY from inputs.py
            'time_limit': 1,  # Seconds of simulation (reduced for speed)
            'time_points': 30,  # Number of points to sample (reduced for speed)
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

    def simulate_forward(self, params: Dict) -> np.ndarray:
        """
        Run forward simulation with given parameters.

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

            # Set distances and refractive indices
            inputs.int_dist = params['distances'] + [params['workpiece_distance']]
            inputs.ref_ind = [1.0] + params['ref_indices']  # Air + wedges

            # Set initial angles
            inputs.THETAX = np.random.uniform(0, 30)  # Initial laser angle x
            inputs.THETAY = np.random.uniform(0, 30)  # Initial laser angle y

            # Time and output settings
            inputs.TIMELIM = self.fixed['time_limit']
            inputs.INC = self.fixed['time_points']
            inputs.plotit = 'off'
            inputs.printit = 'off'

            # Run simulation
            model.main()

            # Read output
            pattern = self._read_latest_output()

            return pattern

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
        """Read the latest simulation output."""
        import csv

        # Find latest simulation directory
        output_dir = 'output/examples'
        dirs = [d for d in os.listdir(output_dir) if 'simulation' in d]
        latest_dir = sorted(dirs)[-1]

        # Read workpiece projections
        csv_file = f'{output_dir}/{latest_dir}/workpiece_projections.csv'

        pattern = []
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                x = float(row['X_Position'])
                y = float(row['Y_Position'])
                pattern.append([x, y])

        return np.array(pattern[:self.fixed['time_points']])

    def generate_dataset_batch(self, n_samples: int,
                               wedge_count: int = None,
                               batch_id: int = 0) -> List[Dict]:
        """
        Generate a batch of samples.

        Args:
            n_samples: Number of samples to generate
            wedge_count: Specific wedge count (None for random)
            batch_id: Batch identifier for progress tracking

        Returns:
            List of dictionaries with parameters and patterns
        """

        samples = []

        for i in range(n_samples):
            # Generate parameters
            params = self.generate_random_parameters(wedge_count)

            try:
                # Run simulation
                pattern = self.simulate_forward(params)

                # Store sample
                sample = {
                    'parameters': params,
                    'pattern': pattern,
                    'timestamp': datetime.now().isoformat()
                }

                samples.append(sample)

                # Progress update
                if (i + 1) % 10 == 0:
                    print(f"  Batch {batch_id}: Generated {i + 1}/{n_samples} samples")

            except Exception as e:
                print(f"  Warning: Simulation failed for sample {i}: {e}")

        return samples

    def generate_large_dataset(self, total_samples: int,
                              save_dir: str = 'datasets',
                              chunk_size: int = 1000) -> str:
        """
        Generate large dataset with automatic saving in chunks.

        Args:
            total_samples: Total number of samples to generate
            save_dir: Directory to save dataset chunks
            chunk_size: Samples per saved chunk

        Returns:
            Path to dataset manifest file
        """

        # Create save directory
        os.makedirs(save_dir, exist_ok=True)

        # Create unique dataset ID
        dataset_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        dataset_dir = os.path.join(save_dir, f'dataset_{dataset_id}')
        os.makedirs(dataset_dir, exist_ok=True)

        print(f"\nGenerating dataset with {total_samples} samples")
        print(f"Saving to: {dataset_dir}")
        print("-" * 60)

        # Distribute samples across wedge counts
        samples_per_wedge = total_samples // 6
        remainder = total_samples % 6

        all_chunks = []
        sample_count = 0

        for wedge_count in range(1, 7):
            n_samples = samples_per_wedge
            if wedge_count <= remainder:
                n_samples += 1

            print(f"\nGenerating {n_samples} samples for {wedge_count} wedge(s)...")

            # Generate in chunks
            for chunk_idx in range(0, n_samples, chunk_size):
                chunk_samples = min(chunk_size, n_samples - chunk_idx)

                # Generate batch
                batch = self.generate_dataset_batch(
                    chunk_samples,
                    wedge_count,
                    batch_id=chunk_idx // chunk_size
                )

                # Save chunk
                chunk_filename = f'chunk_{sample_count:06d}_{sample_count + len(batch):06d}.pkl'
                chunk_path = os.path.join(dataset_dir, chunk_filename)

                with open(chunk_path, 'wb') as f:
                    pickle.dump(batch, f)

                all_chunks.append({
                    'filename': chunk_filename,
                    'wedge_count': wedge_count,
                    'n_samples': len(batch),
                    'sample_range': (sample_count, sample_count + len(batch))
                })

                sample_count += len(batch)
                print(f"  Saved chunk: {chunk_filename}")

        # Save manifest
        manifest = {
            'dataset_id': dataset_id,
            'total_samples': sample_count,
            'creation_date': datetime.now().isoformat(),
            'parameter_ranges': self.ranges,
            'fixed_parameters': self.fixed,
            'chunks': all_chunks,
            'wedge_distribution': {
                i: len([c for c in all_chunks if c['wedge_count'] == i])
                for i in range(1, 7)
            }
        }

        manifest_path = os.path.join(dataset_dir, 'manifest.json')
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        print("\n" + "=" * 60)
        print(f"Dataset generation complete!")
        print(f"Total samples: {sample_count}")
        print(f"Dataset directory: {dataset_dir}")
        print(f"Manifest: {manifest_path}")
        print("=" * 60)

        return manifest_path


def generate_dataset(n_samples: int = 10000):
    """
    Main function to generate dataset.

    Args:
        n_samples: Number of samples to generate
    """

    generator = DatasetGenerator()
    manifest_path = generator.generate_large_dataset(
        total_samples=n_samples,
        save_dir='datasets',
        chunk_size=500  # Save every 500 samples
    )

    return manifest_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate large dataset for Risley prism reverse problem'
    )
    parser.add_argument(
        '--samples', type=int, default=10000,
        help='Number of samples to generate (default: 10000)'
    )

    args = parser.parse_args()

    # Generate dataset
    manifest = generate_dataset(args.samples)

    print(f"\nDataset ready for training!")
    print(f"Use this manifest for training: {manifest}")