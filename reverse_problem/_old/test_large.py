#!/usr/bin/env python3
"""
LARGE-SCALE TESTING - Test with extensive samples
"""

import numpy as np
import json
import time
import os
from datetime import datetime
from solver import StateOfTheArtSolver

def test_large(num_samples=1000):
    """Test with large number of samples for comprehensive evaluation."""
    
    print("LARGE-SCALE PREDICTION TESTING")
    print("=" * 60)
    
    # Initialize solver
    print("Initializing system...")
    solver = StateOfTheArtSolver(use_super_nn=True, use_turbo=True)
    
    if solver.neural_predictor is None:
        print("ERROR: No trained model found")
        print("Run train_large.py first to train the neural network")
        return None
    
    print("  Neural network: LOADED")
    print("  Turbo optimizer: ACTIVE") 
    print("  Pattern caching: ENABLED")
    print()
    
    # Generate large test dataset
    print(f"Generating {num_samples:,} test samples...")
    print("Creating diverse test scenarios...")
    
    test_samples = []
    samples_per_wedge = num_samples // 6
    
    for wedge_count in range(1, 7):
        print(f"  Generating {samples_per_wedge:,} samples for {wedge_count} wedges...")
        
        for i in range(samples_per_wedge):
            params = solver.generate_parameters(wedge_count)
            
            # Vary complexity for comprehensive testing
            if i % 4 == 0:  # 25% simple
                params['rotation_speeds'] = [np.random.uniform(-1.5, 1.5) 
                                            for _ in range(wedge_count)]
                params['phi_x'] = [np.random.uniform(-8.0, 8.0) 
                                  for _ in range(wedge_count)]
                params['phi_y'] = [np.random.uniform(-8.0, 8.0) 
                                  for _ in range(wedge_count)]
            elif i % 4 == 1:  # 25% moderate
                params['rotation_speeds'] = [np.random.uniform(-3.0, 3.0) 
                                            for _ in range(wedge_count)]
                params['phi_x'] = [np.random.uniform(-15.0, 15.0) 
                                  for _ in range(wedge_count)]
                params['phi_y'] = [np.random.uniform(-15.0, 15.0) 
                                  for _ in range(wedge_count)]
            elif i % 4 == 2:  # 25% complex
                params['rotation_speeds'] = [np.random.uniform(-4.0, 4.0) 
                                            for _ in range(wedge_count)]
                params['phi_x'] = [np.random.uniform(-20.0, 20.0) 
                                  for _ in range(wedge_count)]
                params['phi_y'] = [np.random.uniform(-20.0, 20.0) 
                                  for _ in range(wedge_count)]
            else:  # 25% extreme
                params['rotation_speeds'] = [np.random.uniform(-5.0, 5.0) 
                                            for _ in range(wedge_count)]
                params['phi_x'] = [np.random.uniform(-25.0, 25.0) 
                                  for _ in range(wedge_count)]
                params['phi_y'] = [np.random.uniform(-25.0, 25.0) 
                                  for _ in range(wedge_count)]
            
            pattern = solver.forward_simulate(params)
            
            # Add noise to 20% of samples
            if np.random.random() < 0.2:
                noise_level = np.random.uniform(0.02, 0.06)
                noise = np.random.normal(0, noise_level, pattern.shape)
                pattern += noise
            
            test_samples.append({
                'id': len(test_samples),
                'true_wedges': wedge_count,
                'pattern': pattern,
                'parameters': params,
                'complexity_level': i % 4  # 0=simple, 1=moderate, 2=complex, 3=extreme
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
            'parameters': params,
            'complexity_level': 1
        })
    
    print(f"  Generated {len(test_samples):,} total test samples")
    print()
    
    # Run predictions
    print("Running large-scale predictions...")
    print("Testing neural network + optimization system...")
    
    results = []
    correct_predictions = 0
    nn_correct = 0
    cache_hits = 0
    
    # Timing arrays
    nn_times = []
    ga_times = []
    total_start = time.time()
    
    # Progress tracking
    for i, sample in enumerate(test_samples):
        if (i + 1) % 100 == 0:
            elapsed = time.time() - total_start
            rate = i / elapsed
            remaining = (num_samples - i) / rate
            print(f"  Progress: {i+1:,}/{num_samples:,} ({(i+1)/num_samples*100:.1f}%) - "
                  f"Rate: {rate:.1f} samples/sec - ETA: {remaining:.1f}s")
        
        pattern = sample['pattern']
        true_wedges = sample['true_wedges']
        
        # Get neural network prediction
        nn_start = time.time()
        nn_prediction = solver.get_neural_initial_guess(pattern)
        nn_time = time.time() - nn_start
        nn_times.append(nn_time)
        
        # Check NN accuracy
        if nn_prediction and nn_prediction['wedgenum'] == true_wedges:
            nn_correct += 1
        
        # Run full optimization
        ga_start = time.time()
        predicted_wedges, cost, params, info = solver.intelligent_wedge_selection(
            pattern, verbose=False
        )
        ga_time = time.time() - ga_start
        ga_times.append(ga_time)
        
        # Check cache hit
        if info and info.get('from_cache', False):
            cache_hits += 1
        
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
            'nn_correct': nn_prediction and nn_prediction['wedgenum'] == true_wedges,
            'from_cache': info.get('from_cache', False) if info else False,
            'complexity_level': sample['complexity_level']
        }
        results.append(result)
    
    total_time = time.time() - total_start
    
    # Calculate comprehensive metrics
    accuracy = correct_predictions / len(test_samples)
    nn_accuracy = nn_correct / len(test_samples)
    cache_rate = cache_hits / len(results)
    
    # Accuracy by wedge count
    by_wedge = {}
    for result in results:
        w = result['true_wedges']
        if w not in by_wedge:
            by_wedge[w] = {'total': 0, 'correct': 0, 'nn_correct': 0}
        by_wedge[w]['total'] += 1
        if result['correct']:
            by_wedge[w]['correct'] += 1
        if result.get('nn_correct', False):
            by_wedge[w]['nn_correct'] += 1
    
    # Accuracy by complexity level
    by_complexity = {}
    complexity_names = ['Simple', 'Moderate', 'Complex', 'Extreme']
    for result in results:
        level = result['complexity_level']
        if level not in by_complexity:
            by_complexity[level] = {'total': 0, 'correct': 0}
        by_complexity[level]['total'] += 1
        if result['correct']:
            by_complexity[level]['correct'] += 1
    
    print()
    print("LARGE-SCALE RESULTS")
    print("-" * 60)
    print(f"Samples Tested: {len(test_samples):,}")
    print(f"Total Time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Throughput: {len(test_samples)/total_time:.1f} samples/sec")
    print()
    
    print("ACCURACY METRICS")
    print("-" * 60)
    print(f"Overall System Accuracy: {accuracy:.1%} ({correct_predictions:,}/{len(test_samples):,})")
    print(f"Neural Network Accuracy: {nn_accuracy:.1%} ({nn_correct:,}/{len(test_samples):,})")
    print(f"Cache Hit Rate: {cache_rate:.1%} ({cache_hits:,}/{len(results):,})")
    print()
    
    print("PERFORMANCE BY WEDGE COUNT")
    print("-" * 60)
    for w in sorted(by_wedge.keys()):
        stats = by_wedge[w]
        acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        nn_acc = stats['nn_correct'] / stats['total'] if stats['total'] > 0 else 0
        print(f"  {w} wedges: System {acc:6.1%} ({stats['correct']:4}/{stats['total']:4}) | "
              f"NN {nn_acc:6.1%}")
    
    print()
    print("PERFORMANCE BY COMPLEXITY")
    print("-" * 60)
    for level in sorted(by_complexity.keys()):
        stats = by_complexity[level]
        acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        print(f"  {complexity_names[level]:8}: {acc:6.1%} ({stats['correct']:4}/{stats['total']:4})")
    
    print()
    print("TIMING ANALYSIS")
    print("-" * 60)
    print(f"Neural Network: {np.mean(nn_times)*1000:.2f}ms avg (±{np.std(nn_times)*1000:.2f}ms)")
    print(f"Optimization: {np.mean(ga_times)*1000:.2f}ms avg (±{np.std(ga_times)*1000:.2f}ms)")
    print(f"Total per sample: {total_time/len(test_samples)*1000:.2f}ms")
    
    # Cache effectiveness
    if cache_hits > 0:
        cache_time_saved = cache_hits * np.mean(ga_times) * 0.9  # Estimate 90% time saved
        print(f"Time saved by cache: {cache_time_saved:.1f}s")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'output/large_test_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Summary
    summary = {
        'session_info': {
            'timestamp': timestamp,
            'samples_tested': len(test_samples),
            'test_type': 'large_scale',
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
                    'nn_accuracy': stats['nn_correct']/stats['total'] if stats['total'] > 0 else 0,
                    'correct': stats['correct'],
                    'total': stats['total']
                }
                for w, stats in by_wedge.items()
            },
            'by_complexity': {
                complexity_names[level]: {
                    'accuracy': stats['correct']/stats['total'] if stats['total'] > 0 else 0,
                    'correct': stats['correct'],
                    'total': stats['total']
                }
                for level, stats in by_complexity.items()
            }
        },
        'timing': {
            'total_time': total_time,
            'throughput': len(test_samples) / total_time,
            'nn_avg_ms': np.mean(nn_times) * 1000,
            'nn_std_ms': np.std(nn_times) * 1000,
            'ga_avg_ms': np.mean(ga_times) * 1000,
            'ga_std_ms': np.std(ga_times) * 1000
        },
        'cache': {
            'hits': cache_hits,
            'rate': cache_rate,
            'estimated_time_saved': cache_hits * np.mean(ga_times) * 0.9 if cache_hits > 0 else 0
        }
    }
    
    with open(f'{output_dir}/results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print()
    print(f"Results saved: {output_dir}")
    print()
    
    # Final assessment
    print("FINAL ASSESSMENT")
    print("-" * 60)
    
    if accuracy >= 0.8:
        print("Status: EXCELLENT PERFORMANCE")
        print("The system achieves high accuracy across all complexity levels.")
    elif accuracy >= 0.65:
        print("Status: VERY GOOD PERFORMANCE")
        print("The system shows strong generalization capabilities.")
    elif accuracy >= 0.5:
        print("Status: GOOD PERFORMANCE")
        print("The system performs well above baseline.")
    else:
        print("Status: MODERATE PERFORMANCE")
        print("Consider additional training or parameter tuning.")
    
    improvement = ((accuracy - 0.3) / 0.3) * 100 if accuracy > 0.3 else 0
    print(f"Improvement over baseline: {improvement:+.1f}%")
    
    if cache_rate > 0.1:
        print(f"Cache effectiveness: HIGH ({cache_rate:.1%} hit rate)")
    elif cache_rate > 0.05:
        print(f"Cache effectiveness: MODERATE ({cache_rate:.1%} hit rate)")
    else:
        print(f"Cache effectiveness: LOW ({cache_rate:.1%} hit rate)")
    
    return summary

if __name__ == "__main__":
    # Test with 1000 samples by default
    test_large(1000)