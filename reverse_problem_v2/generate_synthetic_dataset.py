#!/usr/bin/env python3
"""
Synthetic Dataset Generator
Generates approximate patterns quickly for initial training
"""

import numpy as np
import pickle
from datetime import datetime
from typing import Dict, List
import time


def generate_synthetic_pattern(wedge_count: int, rotation_speeds: List[float],
                              phi_x: List[float], phi_y: List[float],
                              n_points: int = 30) -> np.ndarray:
    """
    Generate synthetic pattern that approximates real physics.
    Much faster than full simulation.
    """
    t = np.linspace(0, 1.0, n_points)
    pattern = np.zeros((n_points, 2))

    # Base deflection from wedge angles (simplified physics)
    for i in range(wedge_count):
        # Each wedge contributes to deflection
        # Approximate refraction effect
        deflection_x = phi_x[i] * 0.6  # Simplified refraction coefficient
        deflection_y = phi_y[i] * 0.6

        # Rotation creates circular/elliptical patterns
        phase = 2 * np.pi * rotation_speeds[i] * t

        # Add rotation effect
        pattern[:, 0] += deflection_x * np.cos(phase + i * np.pi/4)
        pattern[:, 1] += deflection_y * np.sin(phase + i * np.pi/4)

    # Add interaction effects for multiple wedges
    if wedge_count > 1:
        # Beat frequencies from speed differences
        for i in range(wedge_count - 1):
            for j in range(i + 1, wedge_count):
                beat_freq = abs(rotation_speeds[i] - rotation_speeds[j])
                if beat_freq > 0:
                    amplitude = min(abs(phi_x[i] + phi_x[j]) * 0.1, 2.0)
                    pattern[:, 0] += amplitude * np.sin(2 * np.pi * beat_freq * t * 0.5)
                    pattern[:, 1] += amplitude * np.cos(2 * np.pi * beat_freq * t * 0.5)

    # Add base offset from combined deflection
    base_x = np.mean(phi_x) * 0.3
    base_y = np.mean(phi_y) * 0.3
    pattern[:, 0] += base_x
    pattern[:, 1] += base_y

    # Add noise for realism (very small)
    noise_level = 0.05
    pattern += np.random.normal(0, noise_level, pattern.shape)

    return pattern


def generate_synthetic_dataset(n_samples: int = 10000) -> str:
    """
    Generate large synthetic dataset quickly.

    Args:
        n_samples: Number of samples to generate

    Returns:
        Path to saved dataset file
    """
    print("=" * 60)
    print("SYNTHETIC DATASET GENERATION")
    print("=" * 60)
    print(f"Generating {n_samples} synthetic samples...")

    # Parameter ranges (conservative for good patterns)
    ranges = {
        'rotation_speed': (-3, 3),  # Hz
        'phi_x': (-15, 15),  # degrees
        'phi_y': (-15, 15),  # degrees
    }

    samples = []
    samples_per_wedge = n_samples // 6
    remainder = n_samples % 6

    start_time = time.time()

    for wedge_count in range(1, 7):
        n_wedge_samples = samples_per_wedge
        if wedge_count <= remainder:
            n_wedge_samples += 1

        print(f"\nGenerating {n_wedge_samples} samples for {wedge_count} wedge(s)...")

        for i in range(n_wedge_samples):
            # Generate random parameters
            rotation_speeds = np.random.uniform(
                ranges['rotation_speed'][0],
                ranges['rotation_speed'][1],
                wedge_count
            ).tolist()

            phi_x = np.random.uniform(
                ranges['phi_x'][0],
                ranges['phi_x'][1],
                wedge_count
            ).tolist()

            phi_y = np.random.uniform(
                ranges['phi_y'][0],
                ranges['phi_y'][1],
                wedge_count
            ).tolist()

            # Generate synthetic pattern
            pattern = generate_synthetic_pattern(
                wedge_count, rotation_speeds, phi_x, phi_y, n_points=30
            )

            # Store sample
            sample = {
                'parameters': {
                    'wedge_count': wedge_count,
                    'rotation_speeds': rotation_speeds,
                    'phi_x': phi_x,
                    'phi_y': phi_y
                },
                'pattern': pattern,
                'timestamp': datetime.now().isoformat()
            }

            samples.append(sample)

            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{n_wedge_samples} completed")

    # Save dataset
    dataset_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    dataset_file = f'synthetic_dataset_{dataset_id}.pkl'

    dataset = {
        'samples': samples,
        'metadata': {
            'dataset_id': dataset_id,
            'total_samples': len(samples),
            'creation_date': datetime.now().isoformat(),
            'parameter_ranges': ranges,
            'generation_time': time.time() - start_time,
            'dataset_type': 'synthetic',
            'wedge_distribution': {
                i: len([s for s in samples if s['parameters']['wedge_count'] == i])
                for i in range(1, 7)
            }
        }
    }

    with open(dataset_file, 'wb') as f:
        pickle.dump(dataset, f)

    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("DATASET GENERATION COMPLETE")
    print("=" * 60)
    print(f"Total samples: {len(samples)}")
    print(f"Generation time: {elapsed:.1f} seconds")
    print(f"Rate: {len(samples)/elapsed:.0f} samples/second")
    print(f"Dataset saved to: {dataset_file}")
    print("\nWedge distribution:")
    for w in range(1, 7):
        count = dataset['metadata']['wedge_distribution'][w]
        print(f"  {w} wedge(s): {count} samples")
    print("=" * 60)

    return dataset_file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate synthetic dataset for Risley prism training'
    )
    parser.add_argument(
        '--samples', type=int, default=10000,
        help='Number of samples to generate (default: 10000)'
    )

    args = parser.parse_args()

    # Generate dataset
    dataset_file = generate_synthetic_dataset(args.samples)

    print(f"\nUse this dataset for training:")
    print(f"  python3 train_nn_pipeline.py --dataset {dataset_file}")