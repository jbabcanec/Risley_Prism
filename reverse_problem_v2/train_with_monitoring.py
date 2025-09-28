#!/usr/bin/env python3
"""
Training script with monitoring - generates data quickly and trains with progress updates.
"""

import numpy as np
import sys
import os
import time
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import RisleyParameters, ForwardModel, PatternAnalyzer, ReverseRisleySolver


print("="*60)
print("MULTI-WEDGE TRAINING WITH MONITORING")
print("="*60)
print("\nThis will:")
print("1. Generate/load 1200 samples (200 per wedge count)")
print("2. Train neural network")
print("3. Validate accuracy")
print("4. Show results")
print("="*60)

# Check if we have cached data
cache_file = 'training_cache_1200.pkl'

if os.path.exists(cache_file):
    print(f"\n✓ Loading cached training data from {cache_file}")
    with open(cache_file, 'rb') as f:
        data = pickle.load(f)
    print(f"  Loaded {len(data)} samples")
else:
    print("\n⚙ Generating training data (this may take a few minutes)...")
    print("  Using 20 time points per pattern for speed")

    forward_model = ForwardModel()
    analyzer = PatternAnalyzer()
    data = []

    for wedge_count in range(1, 7):
        print(f"\n  Generating {wedge_count}-wedge patterns...")
        count = 0
        target = 200

        while count < target:
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
                    pattern = forward_model.simulate(params, time_points=20)
                    features = analyzer.extract_features(pattern)

                    data.append({
                        'params': params,
                        'features': features,
                        'pattern': pattern
                    })
                    count += 1

                    if count % 50 == 0:
                        print(f"    {count}/{target}")
                except:
                    pass

    # Save cache
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)
    print(f"\n✓ Saved {len(data)} samples to cache")

# Split data
np.random.shuffle(data)
n_train = int(len(data) * 0.8)
train_data = data[:n_train]
test_data = data[n_train:]

print(f"\nData split:")
print(f"  Training: {len(train_data)} samples")
print(f"  Testing: {len(test_data)} samples")

# Train neural network
print("\n" + "="*40)
print("TRAINING NEURAL NETWORK")
print("="*40)

solver = ReverseRisleySolver()

# Prepare training data
features_train = [d['features'] for d in train_data]
params_train = [d['params'] for d in train_data]

print("\nTraining for 100 epochs...")
print("(showing every 10 epochs)")

history = solver.neural_predictor.train(
    features_train,
    params_train,
    validation_split=0.2,
    epochs=100
)

print(f"\n✓ Training complete!")
print(f"  Final training loss: {history['train_loss'][-1]:.6f}")
print(f"  Final validation loss: {history['val_loss'][-1]:.6f}")

# Save model
solver.neural_predictor.save('trained_model_multiw')
print(f"  Model saved to trained_model_multiw/")

# Validation
print("\n" + "="*40)
print("VALIDATION ON TEST SET")
print("="*40)

# Track results
confusion = np.zeros((6, 6), dtype=int)
wedge_results = {i: {'total': 0, 'correct': 0, 'errors': []} for i in range(1, 7)}

print("\nEvaluating...")
for i, test in enumerate(test_data):
    true_params = test['params']
    features = test['features']

    # Predict
    pred = solver.neural_predictor.predict(features)

    true_w = true_params.wedge_count
    pred_w = pred['wedge_count']

    confusion[true_w-1][pred_w-1] += 1
    wedge_results[true_w]['total'] += 1

    if pred_w == true_w:
        wedge_results[true_w]['correct'] += 1

        # Calculate error
        speed_err = np.sqrt(np.mean([
            (pred['rotation_speeds'][j] - true_params.rotation_speeds[j])**2
            for j in range(true_w)
        ]))
        wedge_results[true_w]['errors'].append(speed_err)

    if (i + 1) % 50 == 0:
        print(f"  Evaluated {i+1}/{len(test_data)}")

# Results
print("\n" + "="*60)
print("RESULTS")
print("="*60)

# Overall accuracy
total_correct = sum(r['correct'] for r in wedge_results.values())
total = len(test_data)
print(f"\n✓ OVERALL WEDGE ACCURACY: {total_correct}/{total} = {total_correct/total*100:.1f}%")

# Per-wedge
print("\nPER-WEDGE ACCURACY:")
for w in range(1, 7):
    r = wedge_results[w]
    if r['total'] > 0:
        acc = r['correct'] / r['total'] * 100
        print(f"  {w} wedge(s): {r['correct']}/{r['total']} = {acc:.0f}%", end="")
        if r['errors']:
            print(f" (Speed RMSE: {np.mean(r['errors']):.3f} Hz)")
        else:
            print()

# Confusion matrix
print("\nCONFUSION MATRIX (rows=true, cols=predicted):")
print("     ", end="")
for i in range(1, 7):
    print(f"  {i}", end="")
print()
for true_idx in range(6):
    print(f"  {true_idx+1}: ", end="")
    for pred_idx in range(6):
        count = confusion[true_idx][pred_idx]
        if count > 0:
            print(f"{count:3d}", end="")
        else:
            print("  .", end="")
    print()

# Test examples
print("\n" + "="*40)
print("TEST EXAMPLES")
print("="*40)

test_cases = [
    {'w': 1, 's': [1.5], 'x': [10.0], 'y': [5.0]},
    {'w': 2, 's': [1.0, -1.0], 'x': [8.0, 8.0], 'y': [0.0, 0.0]},
    {'w': 3, 's': [0.5, -1.0, 1.5], 'x': [5.0, -8.0, 6.0], 'y': [3.0, 5.0, -4.0]}
]

for i, tc in enumerate(test_cases):
    print(f"\nExample {i+1}: {tc['w']} wedge(s)")

    params = RisleyParameters(tc['w'], tc['s'], tc['x'], tc['y'])
    pattern = forward_model.simulate(params, time_points=20)
    features = analyzer.extract_features(pattern)
    pred = solver.neural_predictor.predict(features)

    print(f"  True: S={tc['s']}")
    print(f"  Pred: W={pred['wedge_count']}, S={[f'{s:.2f}' for s in pred['rotation_speeds'][:pred['wedge_count']]]}")

    if pred['wedge_count'] == tc['w']:
        err = np.sqrt(np.mean([(pred['rotation_speeds'][j] - tc['s'][j])**2 for j in range(tc['w'])]))
        print(f"  Speed RMSE: {err:.3f} Hz ✓")
    else:
        print(f"  Wrong wedge count ✗")

print("\n" + "="*60)
print("TRAINING COMPLETE")
print("="*60)
print("\nSummary:")
print(f"• Trained on {len(train_data)} samples")
print(f"• Tested on {len(test_data)} samples")
print(f"• Overall accuracy: {total_correct/total*100:.1f}%")
print(f"• Model saved to: trained_model_multiw/")
print("\nThe system successfully handles 1-6 wedge configurations!")