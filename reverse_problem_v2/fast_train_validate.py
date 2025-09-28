#!/usr/bin/env python3
"""
Fast training and validation script that uses cached data or smaller time points.
"""

import numpy as np
import sys
import os
import time
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import RisleyParameters, ForwardModel, PatternAnalyzer, ReverseRisleySolver


def generate_fast_dataset(n_samples=1200, time_points=20):
    """
    Generate dataset quickly using fewer time points.
    """
    print("="*60)
    print(f"FAST DATASET GENERATION ({n_samples} samples)")
    print("="*60)

    forward_model = ForwardModel()
    analyzer = PatternAnalyzer()
    samples_per_wedge = n_samples // 6

    dataset = []

    for wedge_count in range(1, 7):
        print(f"\nGenerating {samples_per_wedge} samples for {wedge_count} wedges...")
        count = 0

        while count < samples_per_wedge:
            # Generate parameters
            speeds = np.random.uniform(-2.5, 2.5, wedge_count).tolist()
            phi_x = np.random.uniform(-15, 15, wedge_count).tolist()
            phi_y = np.random.uniform(-15, 15, wedge_count).tolist()

            params = RisleyParameters(
                wedge_count=wedge_count,
                rotation_speeds=speeds,
                phi_x=phi_x,
                phi_y=phi_y
            )

            if params.validate():
                try:
                    # Use fewer time points for speed
                    pattern = forward_model.simulate(params, time_points=time_points)
                    features = analyzer.extract_features(pattern)

                    dataset.append({
                        'pattern': pattern,
                        'params': params,
                        'features': features
                    })
                    count += 1

                    if count % 50 == 0:
                        print(f"  {count}/{samples_per_wedge}")
                except:
                    pass

    print(f"\nGenerated {len(dataset)} samples total")
    return dataset


def quick_train_and_validate():
    """
    Quick training and validation run.
    """
    print("\n" + "="*60)
    print("QUICK MULTI-WEDGE TRAINING & VALIDATION")
    print("="*60)

    # Generate or load dataset
    cache_file = 'quick_dataset_1200.pkl'

    if os.path.exists(cache_file):
        print(f"\nLoading cached dataset from {cache_file}...")
        with open(cache_file, 'rb') as f:
            dataset = pickle.load(f)
    else:
        print("\nGenerating new dataset...")
        dataset = generate_fast_dataset(1200, time_points=20)
        with open(cache_file, 'wb') as f:
            pickle.dump(dataset, f)
        print(f"Saved to {cache_file}")

    # Split into train/test
    n_train = int(len(dataset) * 0.8)
    np.random.shuffle(dataset)
    train_data = dataset[:n_train]
    test_data = dataset[n_train:]

    print(f"\nTraining samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")

    # Train
    print("\n" + "-"*40)
    print("TRAINING NEURAL NETWORK")
    print("-"*40)

    solver = ReverseRisleySolver()

    features_train = [d['features'] for d in train_data]
    params_train = [d['params'] for d in train_data]

    history = solver.neural_predictor.train(
        features_train,
        params_train,
        validation_split=0.2,
        epochs=100
    )

    print(f"\nFinal training loss: {history['train_loss'][-1]:.6f}")
    print(f"Final validation loss: {history['val_loss'][-1]:.6f}")

    # Save model
    solver.neural_predictor.save('trained_model_multi')

    # Validate
    print("\n" + "-"*40)
    print("VALIDATION RESULTS")
    print("-"*40)

    # Initialize results tracking
    confusion_matrix = np.zeros((6, 6), dtype=int)
    results_by_wedge = {i: {
        'total': 0,
        'correct': 0,
        'speed_errors': [],
        'angle_errors': []
    } for i in range(1, 7)}

    print("\nEvaluating on test set...")

    for i, test_sample in enumerate(test_data):
        true_params = test_sample['params']
        features = test_sample['features']

        # Get prediction
        pred = solver.neural_predictor.predict(features)

        true_wedges = true_params.wedge_count
        pred_wedges = pred['wedge_count']

        # Update confusion matrix
        confusion_matrix[true_wedges-1][pred_wedges-1] += 1
        results_by_wedge[true_wedges]['total'] += 1

        if pred_wedges == true_wedges:
            results_by_wedge[true_wedges]['correct'] += 1

            # Calculate errors
            speed_err = np.sqrt(np.mean([
                (pred['rotation_speeds'][j] - true_params.rotation_speeds[j])**2
                for j in range(true_wedges)
            ]))

            angle_x_err = np.mean([
                (pred['phi_x'][j] - true_params.phi_x[j])**2
                for j in range(true_wedges)
            ])
            angle_y_err = np.mean([
                (pred['phi_y'][j] - true_params.phi_y[j])**2
                for j in range(true_wedges)
            ])
            angle_err = np.sqrt((angle_x_err + angle_y_err) / 2)

            results_by_wedge[true_wedges]['speed_errors'].append(speed_err)
            results_by_wedge[true_wedges]['angle_errors'].append(angle_err)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(test_data)}")

    # Print results
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)

    # Overall accuracy
    total_correct = sum(r['correct'] for r in results_by_wedge.values())
    total = sum(r['total'] for r in results_by_wedge.values())
    overall_acc = total_correct / total * 100

    print(f"\nOVERALL WEDGE ACCURACY: {overall_acc:.1f}% ({total_correct}/{total})")

    # Per-wedge breakdown
    print("\nPER-WEDGE ACCURACY:")
    for wedges in range(1, 7):
        r = results_by_wedge[wedges]
        if r['total'] > 0:
            acc = r['correct'] / r['total'] * 100
            print(f"  {wedges} wedge(s): {acc:.1f}% ({r['correct']}/{r['total']})")

            if r['speed_errors']:
                avg_speed = np.mean(r['speed_errors'])
                avg_angle = np.mean(r['angle_errors'])
                print(f"    Speed RMSE: {avg_speed:.3f} Hz")
                print(f"    Angle RMSE: {avg_angle:.2f} deg")

    # Confusion matrix
    print("\nCONFUSION MATRIX:")
    print("True\\Pred", end="")
    for i in range(1, 7):
        print(f"\t{i}", end="")
    print()

    for true_idx in range(6):
        print(f"{true_idx+1}\t", end="")
        for pred_idx in range(6):
            print(f"{confusion_matrix[true_idx][pred_idx]}\t", end="")
        print()

    # Summary stats
    all_speed_errors = []
    all_angle_errors = []

    for r in results_by_wedge.values():
        all_speed_errors.extend(r['speed_errors'])
        all_angle_errors.extend(r['angle_errors'])

    if all_speed_errors:
        print(f"\nWHEN WEDGE COUNT CORRECT:")
        print(f"  Overall Speed RMSE: {np.mean(all_speed_errors):.3f} Hz")
        print(f"  Overall Angle RMSE: {np.mean(all_angle_errors):.2f} degrees")

    return solver, results_by_wedge


def test_specific_examples(solver):
    """
    Test on specific examples to show system working.
    """
    print("\n" + "="*60)
    print("SPECIFIC TEST EXAMPLES")
    print("="*60)

    forward_model = ForwardModel()
    analyzer = PatternAnalyzer()

    test_cases = [
        {
            'name': '1 wedge - slow rotation',
            'params': RisleyParameters(1, [0.5], [10.0], [5.0])
        },
        {
            'name': '2 wedges - counter-rotating',
            'params': RisleyParameters(2, [1.5, -1.5], [10.0, 10.0], [0.0, 0.0])
        },
        {
            'name': '3 wedges - complex',
            'params': RisleyParameters(3, [1.0, -0.5, 2.0], [5.0, -10.0, 8.0], [8.0, -5.0, 3.0])
        },
        {
            'name': '4 wedges',
            'params': RisleyParameters(4, [0.8, -1.2, 0.5, 1.5],
                                      [8.0, -6.0, 10.0, -5.0],
                                      [5.0, 8.0, -3.0, 6.0])
        }
    ]

    for test in test_cases:
        print(f"\n{test['name']}:")
        print(f"  True: W={test['params'].wedge_count}, "
              f"S={test['params'].rotation_speeds}")

        # Generate pattern
        pattern = forward_model.simulate(test['params'], time_points=30)
        features = analyzer.extract_features(pattern)

        # Predict
        pred = solver.neural_predictor.predict(features)

        print(f"  Pred: W={pred['wedge_count']}, "
              f"S={[f'{s:.2f}' for s in pred['rotation_speeds'][:pred['wedge_count']]]}")

        if pred['wedge_count'] == test['params'].wedge_count:
            # Calculate errors
            speed_err = np.sqrt(np.mean([
                (pred['rotation_speeds'][i] - test['params'].rotation_speeds[i])**2
                for i in range(test['params'].wedge_count)
            ]))
            print(f"  Speed RMSE: {speed_err:.3f} Hz ✓")
        else:
            print(f"  Wrong wedge count! ✗")


if __name__ == "__main__":
    # Run quick training and validation
    solver, results = quick_train_and_validate()

    # Test specific examples
    test_specific_examples(solver)

    print("\n" + "="*60)
    print("COMPLETE - Multi-wedge system trained and validated")
    print("="*60)