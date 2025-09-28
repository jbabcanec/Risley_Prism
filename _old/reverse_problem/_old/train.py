#!/usr/bin/env python3
"""
TRAIN - Generate training data and train neural network for Risley prism reverse problem

Usage: python3 train.py [num_samples]
Output: input/training_TIMESTAMP/ with samples and metadata
        weights/pattern_predictor.pth with trained neural network
"""

import sys
import os
import json
import numpy as np
from datetime import datetime
from solver import StateOfTheArtSolver
from core.neural_network import NeuralPredictor

def main():
    """Generate training data."""
    # Get number of samples
    if len(sys.argv) > 1:
        num_samples = int(sys.argv[1])
    else:
        try:
            num_samples = int(input("Number of training samples: "))
        except (ValueError, KeyboardInterrupt):
            num_samples = 1000
    
    print(f"🏋️ TRAINING: Generating {num_samples} samples")
    
    # Create timestamped training session
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    session_dir = f"input/training_{timestamp}"
    os.makedirs(session_dir, exist_ok=True)
    
    # Generate samples
    solver = StateOfTheArtSolver()
    samples = []
    
    print("Generating samples...")
    for i in range(num_samples):
        # Random wedge configuration
        wedge_count = np.random.randint(1, 7)
        params = solver.generate_parameters(wedge_count)
        pattern = solver.forward_simulate(params)
        
        sample = {
            'id': i,
            'wedge_count': wedge_count,
            'parameters': params,
            'pattern': pattern.tolist(),
            'complexity': solver.calculate_pattern_complexity(pattern)
        }
        samples.append(sample)
        
        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{num_samples}")
    
    # Save samples as individual files
    for sample in samples:
        with open(f"{session_dir}/sample_{sample['id']:05d}.json", 'w') as f:
            json.dump(sample, f, indent=2)
    
    # Save session metadata
    metadata = {
        'session_id': timestamp,
        'total_samples': num_samples,
        'wedge_distribution': {str(i): sum(1 for s in samples if s['wedge_count'] == i) for i in range(1, 7)},
        'generation_time': datetime.now().isoformat()
    }
    
    with open(f"{session_dir}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Training data complete!")
    print(f"📁 Data saved: {session_dir}")
    print(f"📊 Distribution: {metadata['wedge_distribution']}")
    
    # Train neural network
    print(f"\n🧠 Training neural network on {num_samples} samples...")
    
    # Ensure weights directory exists
    os.makedirs("weights", exist_ok=True)
    
    # Initialize neural network predictor
    predictor = NeuralPredictor()
    
    # Train on generated samples
    training_results = predictor.train(samples)
    
    print(f"✅ Neural network training complete!")
    print(f"💾 Model weights saved to: weights/pattern_predictor.pth")
    print(f"📊 Final validation loss: {training_results['best_val_loss']:.6f}")
    print(f"🔄 Epochs trained: {training_results['epochs_trained']}")
    
    # Save training results
    training_info = {
        'session_id': timestamp,
        'training_samples': num_samples,
        'neural_network_results': training_results,
        'training_complete': datetime.now().isoformat()
    }
    
    with open(f"{session_dir}/training_results.json", 'w') as f:
        json.dump(training_info, f, indent=2)

if __name__ == "__main__":
    main()