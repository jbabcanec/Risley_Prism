#!/usr/bin/env python3
"""
PREDICT - Supercharged System Testing
"""

import numpy as np
import json
import time
import os
from datetime import datetime
from solver import StateOfTheArtSolver

def predict(num_samples=100, verbose=True):
    """Test the supercharged prediction system."""
    
    print("PREDICTION TESTING")
    print("=" * 60)
    
    # Initialize solver
    print("Initializing system...")
    solver = StateOfTheArtSolver(use_super_nn=True)
    
    if solver.neural_predictor is None:
        print("ERROR: No trained model found")
        print("Run train.py first to train the neural network")
        return None
    
    print("  Neural network: LOADED")
    print("  Turbo optimizer: ACTIVE")
    print()
    
    # Generate test samples
    print(f"Generating {num_samples} test samples...")
    test_samples = []
    samples_per_wedge = num_samples // 6
    
    for wedge_count in range(1, 7):
        for i in range(samples_per_wedge):
            params = solver.generate_parameters(wedge_count)
            pattern = solver.forward_simulate(params)
            
            # Add noise to some samples
            if np.random.random() < 0.2:  # 20% with noise
                noise = np.random.normal(0, 0.05, pattern.shape)
                pattern += noise
            
            test_samples.append({
                'id': len(test_samples),
                'true_wedges': wedge_count,
                'pattern': pattern,
                'parameters': params
            })
    
    # Pad if needed
    while len(test_samples) < num_samples:
        wedge_count = np.random.randint(1, 7)
        params = solver.generate_parameters(wedge_count)
        pattern = solver.forward_simulate(params)
        test_samples.append({
            'id': len(test_samples),
            'true_wedges': wedge_count,
            'pattern': pattern,
            'parameters': params
        })
    
    print(f"  Generated {len(test_samples)} samples")
    print()
    
    # Run predictions
    print("Running predictions...")
    results = []
    correct_predictions = 0
    
    # Timing
    nn_times = []
    ga_times = []
    total_start = time.time()
    
    for i, sample in enumerate(test_samples):
        if verbose and (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{len(test_samples)}")
        
        pattern = sample['pattern']
        true_wedges = sample['true_wedges']
        
        # Get neural network prediction
        nn_start = time.time()
        nn_prediction = solver.get_neural_initial_guess(pattern)
        nn_time = time.time() - nn_start
        nn_times.append(nn_time)
        
        # Run full optimization
        ga_start = time.time()
        predicted_wedges, cost, params, info = solver.intelligent_wedge_selection(
            pattern, verbose=False
        )
        ga_time = time.time() - ga_start
        ga_times.append(ga_time)
        
        # Check accuracy
        is_correct = predicted_wedges == true_wedges
        if is_correct:
            correct_predictions += 1
        
        # Store result
        result = {
            'sample_id': sample['id'],
            'true_wedges': true_wedges,
            'predicted_wedges': predicted_wedges,
            'cost': cost,
            'correct': is_correct,
            'nn_prediction': nn_prediction['wedgenum'] if nn_prediction else None,
            'from_cache': info.get('from_cache', False) if info else False
        }
        results.append(result)
    
    total_time = time.time() - total_start
    
    # Calculate metrics
    accuracy = correct_predictions / len(test_samples)
    
    # NN accuracy
    nn_correct = sum(1 for r in results 
                    if r['nn_prediction'] and r['nn_prediction'] == r['true_wedges'])
    nn_accuracy = nn_correct / len(test_samples)
    
    # Cache performance
    cache_hits = sum(1 for r in results if r['from_cache'])
    cache_rate = cache_hits / len(results)
    
    # Accuracy by wedge count
    by_wedge = {}
    for result in results:
        w = result['true_wedges']
        if w not in by_wedge:
            by_wedge[w] = {'total': 0, 'correct': 0}
        by_wedge[w]['total'] += 1
        if result['correct']:
            by_wedge[w]['correct'] += 1
    
    print()
    print("RESULTS")
    print("-" * 60)
    print(f"Overall Accuracy: {accuracy:.1%} ({correct_predictions}/{len(test_samples)})")
    print(f"NN Accuracy: {nn_accuracy:.1%}")
    print(f"Cache Hit Rate: {cache_rate:.1%}")
    print(f"Throughput: {len(test_samples)/total_time:.1f} samples/sec")
    print()
    
    print("Accuracy by Wedge Count:")
    for w in sorted(by_wedge.keys()):
        stats = by_wedge[w]
        acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        print(f"  {w} wedges: {acc:6.1%} ({stats['correct']:2}/{stats['total']:2})")
    
    print()
    print("Timing Analysis:")
    print(f"  Neural Network: {np.mean(nn_times)*1000:.1f}ms avg")
    print(f"  Optimization: {np.mean(ga_times)*1000:.1f}ms avg") 
    print(f"  Total: {total_time:.1f}s")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'output/predictions_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Summary
    summary = {
        'session_info': {
            'timestamp': timestamp,
            'samples_tested': len(test_samples),
            'model_type': 'SuperNeuralNetwork'
        },
        'performance': {
            'overall_accuracy': accuracy,
            'nn_accuracy': nn_accuracy,
            'correct_predictions': correct_predictions,
            'total_tested': len(test_samples),
            'by_wedge_count': {
                str(w): {
                    'accuracy': stats['correct']/stats['total'] if stats['total'] > 0 else 0,
                    'correct': stats['correct'],
                    'total': stats['total']
                }
                for w, stats in by_wedge.items()
            }
        },
        'timing': {
            'total_time': total_time,
            'throughput': len(test_samples) / total_time,
            'nn_avg_ms': np.mean(nn_times) * 1000,
            'ga_avg_ms': np.mean(ga_times) * 1000
        },
        'cache': {
            'hits': cache_hits,
            'rate': cache_rate
        }
    }
    
    with open(f'{output_dir}/results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Results saved: {output_dir}")
    print()
    
    # Performance assessment
    print("PERFORMANCE ASSESSMENT")
    print("-" * 60)
    
    if accuracy >= 0.8:
        print("Status: EXCELLENT")
    elif accuracy >= 0.6:
        print("Status: GOOD")
    elif accuracy >= 0.4:
        print("Status: IMPROVING")
    else:
        print("Status: NEEDS OPTIMIZATION")
    
    improvement = ((accuracy - 0.3) / 0.3) * 100 if accuracy > 0.3 else 0
    print(f"Improvement over baseline: {improvement:+.1f}%")
    
    return summary

if __name__ == "__main__":
    predict()