#!/usr/bin/env python3
"""
PREDICT - Make predictions using hybrid neural network + genetic algorithm

Usage: python3 predict.py [test_fraction]
Input: Latest training session from input/ + trained weights from weights/
Output: output/predictions_TIMESTAMP/ with results and speed comparison
"""

import sys
import os
import json
import glob
import numpy as np
from datetime import datetime
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from solver import StateOfTheArtSolver
from core.neural_network import NeuralPredictor

def load_latest_training():
    """Load latest training session."""
    training_dirs = glob.glob("input/training_*")
    if not training_dirs:
        print("❌ No training data found. Run train.py first.")
        return None, None
    
    latest_dir = max(training_dirs, key=os.path.getmtime)
    
    # Load metadata
    with open(f"{latest_dir}/metadata.json", 'r') as f:
        metadata = json.load(f)
    
    # Load all samples
    sample_files = glob.glob(f"{latest_dir}/sample_*.json")
    samples = []
    for file in sorted(sample_files):
        with open(file, 'r') as f:
            samples.append(json.load(f))
    
    return samples, metadata, latest_dir

def process_batch(batch_data):
    """Process a batch of samples using hybrid NN+GA approach."""
    batch_samples, batch_start, use_hybrid = batch_data
    solver = StateOfTheArtSolver()
    results = []
    
    # Try to load neural network predictor
    predictor = None
    if use_hybrid:
        try:
            predictor = NeuralPredictor()
            if predictor.load():
                pass  # Successfully loaded
            else:
                predictor = None
        except:
            predictor = None
    
    for i, sample in enumerate(batch_samples):
        pattern = np.array(sample['pattern'])
        
        nn_time = 0
        ga_time = 0
        nn_prediction = None
        
        if predictor is not None:
            # Neural network prediction (fast initial guess)
            nn_start = time.time()
            try:
                nn_prediction = predictor.predict(pattern)
                nn_time = time.time() - nn_start
            except:
                nn_prediction = None
                nn_time = 0
        
        # Genetic algorithm refinement
        ga_start = time.time()
        recovery = solver.test_recovery(pattern, sample['wedge_count'], verbose=False)
        ga_time = time.time() - ga_start
        
        result = {
            'sample_id': sample['id'],
            'true_wedges': sample['wedge_count'],
            'predicted_wedges': recovery['predicted_wedge_count'],
            'cost': recovery['final_cost'],
            'correct': recovery['predicted_wedge_count'] == sample['wedge_count'],
            'timing': {
                'nn_time': nn_time,
                'ga_time': ga_time,
                'total_time': nn_time + ga_time,
                'speedup': ga_time / (nn_time + ga_time) if (nn_time + ga_time) > 0 else 1.0
            },
            'nn_prediction': nn_prediction
        }
        results.append(result)
    
    return results

def main():
    """Run predictions on training data."""
    # Get test fraction
    test_fraction = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3
    
    # Check if neural network weights exist
    weights_exist = os.path.exists("weights/pattern_predictor.pth")
    use_hybrid = weights_exist
    
    if use_hybrid:
        print(f"🔮 PREDICTING: Testing {test_fraction:.0%} with hybrid NN+GA approach")
        print(f"🧠 Using trained neural network for initial predictions")
    else:
        print(f"🔮 PREDICTING: Testing {test_fraction:.0%} with GA-only approach")
        print(f"⚠️  No trained weights found. Run train.py first for hybrid approach.")
    
    # Load training data
    samples, metadata, training_dir = load_latest_training()
    if not samples:
        return
    
    print(f"📊 Loaded {len(samples)} samples from {os.path.basename(training_dir)}")
    
    # Select test samples
    test_count = max(10, int(len(samples) * test_fraction))
    selected_indices = np.random.choice(len(samples), test_count, replace=False)
    test_samples = [samples[i] for i in selected_indices]
    
    # Setup parallel processing
    num_cores = mp.cpu_count()
    batch_size = max(1, len(test_samples) // (num_cores * 2))
    batches = []
    for i in range(0, len(test_samples), batch_size):
        batch = test_samples[i:i+batch_size]
        batches.append((batch, i, use_hybrid))
    
    print(f"🧠 Testing {len(test_samples)} samples with {num_cores} cores...")
    
    # Process in parallel
    start_time = time.time()
    all_results = []
    
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        future_to_batch = {executor.submit(process_batch, batch): i for i, batch in enumerate(batches)}
        
        completed = 0
        for future in as_completed(future_to_batch):
            batch_results = future.result()
            all_results.extend(batch_results)
            completed += 1
            
            progress = (completed / len(batches)) * 100
            elapsed = time.time() - start_time
            rate = len(all_results) / elapsed if elapsed > 0 else 0
            print(f"\r⚡ Progress: {progress:.1f}% | {rate:.1f} samples/sec", end="", flush=True)
    
    print()  # New line
    
    # Calculate results
    correct = sum(1 for r in all_results if r['correct'])
    accuracy = correct / len(all_results)
    
    # Timing analysis
    if use_hybrid and all_results:
        total_nn_time = sum(r['timing']['nn_time'] for r in all_results)
        total_ga_time = sum(r['timing']['ga_time'] for r in all_results)
        total_time = sum(r['timing']['total_time'] for r in all_results)
        avg_speedup = np.mean([r['timing']['speedup'] for r in all_results])
    else:
        total_nn_time = total_ga_time = total_time = avg_speedup = 0
    
    # Group by wedge count
    by_wedge = {}
    for result in all_results:
        w = result['true_wedges']
        if w not in by_wedge:
            by_wedge[w] = {'total': 0, 'correct': 0}
        by_wedge[w]['total'] += 1
        if result['correct']:
            by_wedge[w]['correct'] += 1
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"output/predictions_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results
    results_data = {
        'session_info': {
            'timestamp': timestamp,
            'source_training': os.path.basename(training_dir),
            'test_fraction': test_fraction,
            'samples_tested': len(all_results)
        },
        'performance': {
            'overall_accuracy': accuracy,
            'total_tested': len(all_results),
            'correct_predictions': correct,
            'by_wedge_count': {str(k): {
                'total': v['total'],
                'correct': v['correct'],
                'accuracy': v['correct'] / v['total']
            } for k, v in by_wedge.items()}
        },
        'timing': {
            'use_hybrid': use_hybrid,
            'total_nn_time': total_nn_time,
            'total_ga_time': total_ga_time,
            'total_time': total_time,
            'average_speedup': avg_speedup,
            'nn_percentage': total_nn_time / total_time * 100 if total_time > 0 else 0,
            'ga_percentage': total_ga_time / total_time * 100 if total_time > 0 else 100
        },
        'detailed_results': all_results
    }
    
    # Convert numpy types to python types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        return obj
    
    with open(f"{output_dir}/results.json", 'w') as f:
        json.dump(convert_numpy_types(results_data), f, indent=2)
    
    # Print summary
    elapsed = time.time() - start_time
    print(f"⚡ Completed in {elapsed:.1f}s ({len(all_results)/elapsed:.1f} samples/sec)")
    print(f"🎯 Overall Accuracy: {accuracy:.1%} ({correct}/{len(all_results)})")
    print(f"📁 Results saved: {output_dir}")
    
    # Show timing results for hybrid approach
    if use_hybrid and total_time > 0:
        print(f"\n⏱️ Hybrid Performance:")
        print(f"   Neural Network: {total_nn_time:.2f}s ({total_nn_time/total_time*100:.1f}%)")
        print(f"   Genetic Algorithm: {total_ga_time:.2f}s ({total_ga_time/total_time*100:.1f}%)")
        print(f"   Average speedup: {avg_speedup:.1f}x")
    
    # Show per-wedge results
    print(f"\n📊 Accuracy by wedge count:")
    for w in sorted(by_wedge.keys()):
        stats = by_wedge[w]
        acc = stats['correct'] / stats['total']
        print(f"   {w} wedges: {acc:.1%} ({stats['correct']}/{stats['total']})")

if __name__ == "__main__":
    main()