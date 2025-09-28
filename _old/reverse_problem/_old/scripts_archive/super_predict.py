#!/usr/bin/env python3
"""
SUPER PREDICT - Test super-powered neural network on validation data

Usage: python3 super_predict.py [training_session]
Output: output/super_predictions_TIMESTAMP/ with detailed results
"""

import sys
import os
import json
import glob
import numpy as np
import time
from datetime import datetime
from solver import StateOfTheArtSolver

def main():
    """Test super-powered neural network predictions."""
    # Find latest training session or use specified
    if len(sys.argv) > 1:
        training_pattern = sys.argv[1]
        training_dirs = glob.glob(f"input/*{training_pattern}*")
    else:
        # Find super training sessions first, then regular ones
        training_dirs = glob.glob("input/super_training_*")
        if not training_dirs:
            training_dirs = glob.glob("input/training_*")
    
    if not training_dirs:
        print("❌ No training data found. Run super_train.py first.")
        return
    
    # Use latest session
    latest_training = max(training_dirs, key=os.path.getmtime)
    print(f"📁 Using training data: {latest_training}")
    
    # Load sample files
    sample_files = sorted(glob.glob(f"{latest_training}/sample_*.json"))
    if not sample_files:
        print("❌ No sample files found in training directory.")
        return
    
    # Determine test fraction
    test_fraction = 0.3
    if len(sys.argv) > 2:
        test_fraction = float(sys.argv[2])
    
    # Load samples
    print(f"📊 Loading samples...")
    samples = []
    for file in sample_files[:5000]:  # Limit to 5000 for memory
        with open(file, 'r') as f:
            samples.append(json.load(f))
    
    # Select test samples
    num_test = min(1500, int(len(samples) * test_fraction))
    test_indices = np.random.choice(len(samples), num_test, replace=False)
    test_samples = [samples[i] for i in test_indices]
    
    print(f"🧪 Testing on {num_test} samples from {len(samples)} available")
    
    # Initialize solver with super neural network
    print(f"\n🚀 Initializing solver with SUPER-POWERED neural network...")
    solver = StateOfTheArtSolver(use_super_nn=True)
    
    if solver.neural_predictor is None:
        print("❌ No neural network model found. Train one first with super_train.py")
        return
    
    # Get model info
    model_type = "SuperNeuralNetwork" if hasattr(solver.neural_predictor, 'feature_extractor') else "StandardNN"
    print(f"   Model type: {model_type}")
    
    # Test predictions
    print(f"\n🔬 Running predictions...")
    
    results = []
    correct_predictions = 0
    
    # Timing statistics
    total_nn_time = 0
    total_ga_time = 0
    start_time = time.time()
    
    for i, sample in enumerate(test_samples):
        if i % 100 == 0 and i > 0:
            elapsed = time.time() - start_time
            rate = i / elapsed
            eta = (num_test - i) / rate
            print(f"   Progress: {i}/{num_test} ({i/num_test*100:.1f}%) | "
                  f"Rate: {rate:.1f} samples/s | ETA: {eta:.0f}s")
        
        pattern = np.array(sample['pattern'])
        true_wedges = sample['wedge_count']
        
        # Time neural network prediction
        nn_start = time.time()
        nn_prediction = solver.get_neural_initial_guess(pattern)
        nn_time = time.time() - nn_start
        total_nn_time += nn_time
        
        # Time genetic algorithm optimization
        ga_start = time.time()
        predicted_wedges, cost, params, info = solver.intelligent_wedge_selection(
            pattern, verbose=False
        )
        ga_time = time.time() - ga_start
        total_ga_time += ga_time
        
        # Check if prediction is correct
        is_correct = predicted_wedges == true_wedges
        if is_correct:
            correct_predictions += 1
        
        # Store detailed result
        result = {
            'sample_id': sample['id'],
            'true_wedges': true_wedges,
            'predicted_wedges': predicted_wedges,
            'cost': cost,
            'correct': is_correct,
            'timing': {
                'nn_time': nn_time,
                'ga_time': ga_time,
                'total_time': nn_time + ga_time,
                'speedup': ga_time / (nn_time + ga_time) if (nn_time + ga_time) > 0 else 1.0
            },
            'nn_prediction': nn_prediction
        }
        
        # Add confidence scores if available
        if nn_prediction and 'prediction_confidence' in nn_prediction:
            result['nn_confidence'] = nn_prediction['prediction_confidence']
        
        results.append(result)
    
    total_time = time.time() - start_time
    
    # Calculate comprehensive metrics
    accuracy = correct_predictions / num_test
    
    # Accuracy by wedge count
    by_wedge = {}
    for result in results:
        w = result['true_wedges']
        if w not in by_wedge:
            by_wedge[w] = {'total': 0, 'correct': 0}
        by_wedge[w]['total'] += 1
        if result['correct']:
            by_wedge[w]['correct'] += 1
    
    # Calculate accuracy for each wedge count
    for w in by_wedge:
        by_wedge[w]['accuracy'] = by_wedge[w]['correct'] / by_wedge[w]['total']
    
    # Neural network accuracy (wedge count only)
    nn_correct = 0
    for result in results:
        if result['nn_prediction'] and 'wedgenum' in result['nn_prediction']:
            if result['nn_prediction']['wedgenum'] == result['true_wedges']:
                nn_correct += 1
    nn_accuracy = nn_correct / num_test if num_test > 0 else 0
    
    # Print results
    print(f"\n" + "=" * 60)
    print(f"🏆 SUPER-POWERED PREDICTION RESULTS")
    print(f"=" * 60)
    print(f"Overall Accuracy: {accuracy:.1%} ({correct_predictions}/{num_test})")
    print(f"Neural Network Accuracy: {nn_accuracy:.1%}")
    print(f"Total Time: {total_time:.1f}s")
    print(f"Throughput: {num_test/total_time:.2f} samples/sec")
    
    print(f"\n📊 Accuracy by Wedge Count:")
    for w in sorted(by_wedge.keys()):
        stats = by_wedge[w]
        print(f"   {w} wedges: {stats['accuracy']:.1%} ({stats['correct']}/{stats['total']})")
    
    print(f"\n⏱️  Timing Analysis:")
    print(f"   Neural Network Time: {total_nn_time:.2f}s ({total_nn_time/total_time*100:.1f}%)")
    print(f"   Genetic Algorithm Time: {total_ga_time:.2f}s ({total_ga_time/total_time*100:.1f}%)")
    print(f"   Average NN time per sample: {total_nn_time/num_test*1000:.2f}ms")
    print(f"   Average GA time per sample: {total_ga_time/num_test*1000:.2f}ms")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"output/super_predictions_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare comprehensive results
    full_results = {
        'session_info': {
            'timestamp': timestamp,
            'source_training': os.path.basename(latest_training),
            'test_fraction': test_fraction,
            'samples_tested': num_test,
            'model_type': model_type
        },
        'performance': {
            'overall_accuracy': accuracy,
            'nn_wedge_accuracy': nn_accuracy,
            'total_tested': num_test,
            'correct_predictions': correct_predictions,
            'by_wedge_count': by_wedge
        },
        'timing': {
            'use_hybrid': True,
            'total_nn_time': total_nn_time,
            'total_ga_time': total_ga_time,
            'total_time': total_time,
            'average_speedup': np.mean([r['timing']['speedup'] for r in results]),
            'nn_percentage': (total_nn_time / total_time) * 100,
            'ga_percentage': (total_ga_time / total_time) * 100,
            'throughput_samples_per_sec': num_test / total_time
        },
        'detailed_results': results
    }
    
    # Save results
    with open(f"{output_dir}/results.json", 'w') as f:
        json.dump(full_results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_dir}")
    
    # Performance comparison with standard NN
    print(f"\n📈 Performance Analysis:")
    if nn_accuracy > 0.5:
        print(f"   ✅ Excellent NN performance (>{50}% accuracy)")
        print(f"   • Neural network is providing strong guidance")
        print(f"   • GA can focus on refinement rather than exploration")
    elif nn_accuracy > 0.3:
        print(f"   🟡 Good NN performance ({30}-{50}% accuracy)")
        print(f"   • Neural network is helpful but could be better")
        print(f"   • Consider more training data or longer training")
    else:
        print(f"   🔴 Poor NN performance (<{30}% accuracy)")
        print(f"   • Neural network needs improvement")
        print(f"   • Check training data quality and distribution")
    
    # Compare with previous results if available
    previous_results = glob.glob("output/predictions_*/results.json")
    if previous_results:
        latest_previous = max(previous_results, key=os.path.getmtime)
        with open(latest_previous, 'r') as f:
            prev_data = json.load(f)
        
        prev_acc = prev_data['performance']['overall_accuracy']
        improvement = (accuracy - prev_acc) / prev_acc * 100
        
        print(f"\n🔄 Comparison with Previous Results:")
        print(f"   Previous accuracy: {prev_acc:.1%}")
        print(f"   Current accuracy: {accuracy:.1%}")
        print(f"   Improvement: {improvement:+.1f}%")
        
        if improvement > 20:
            print(f"   🚀 MASSIVE IMPROVEMENT! Super NN is working excellently!")
        elif improvement > 10:
            print(f"   ✅ Significant improvement with Super NN")
        elif improvement > 0:
            print(f"   📈 Modest improvement with Super NN")
        else:
            print(f"   ⚠️  No improvement - check training quality")
    
    print(f"\n✅ SUPER PREDICTION COMPLETE!")

if __name__ == "__main__":
    main()