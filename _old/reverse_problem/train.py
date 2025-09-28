#!/usr/bin/env python3
"""
TRAIN - Supercharged Neural Network Training
"""

import numpy as np
import json
import time
import os
from datetime import datetime
from solver import StateOfTheArtSolver
from core.super_neural_network import SuperNeuralPredictor

def train():
    """Train the supercharged neural network system."""
    
    print("NEURAL NETWORK TRAINING")
    print("=" * 60)
    
    # Configuration
    num_samples = 3000  # Balanced across wedge counts
    validation_split = 0.2
    epochs = 50
    batch_size = 64
    
    print(f"Configuration:")
    print(f"  Samples: {num_samples}")
    print(f"  Validation: {int(validation_split*100)}%")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print()
    
    # Initialize solver
    print("Initializing system...")
    solver = StateOfTheArtSolver(use_super_nn=False)
    
    # Generate training data
    print("Generating training data...")
    samples = []
    samples_per_wedge = num_samples // 6
    
    for wedge_count in range(1, 7):
        for i in range(samples_per_wedge):
            # Generate parameters with variation
            params = solver.generate_parameters(wedge_count)
            
            # Add variation for robustness
            if np.random.random() < 0.3:  # 30% with extreme values
                params['rotation_speeds'] = [np.random.uniform(-5.0, 5.0) 
                                            for _ in range(wedge_count)]
                params['phi_x'] = [np.random.uniform(-25.0, 25.0) 
                                  for _ in range(wedge_count)]
                params['phi_y'] = [np.random.uniform(-25.0, 25.0) 
                                  for _ in range(wedge_count)]
            
            # Simulate pattern
            pattern = solver.forward_simulate(params)
            
            # Store sample
            sample = {
                'id': len(samples),
                'wedge_count': wedge_count,
                'parameters': params,
                'pattern': pattern.tolist(),
                'complexity': solver.calculate_pattern_complexity(pattern)
            }
            samples.append(sample)
            
            if (len(samples) % 500) == 0:
                print(f"  Generated {len(samples)}/{num_samples} samples...")
    
    print(f"  Generated {len(samples)} total samples")
    print()
    
    # Train neural network
    print("Training neural network...")
    predictor = SuperNeuralPredictor()
    predictor.config.epochs = epochs
    predictor.config.early_stopping_patience = 15
    predictor.config.batch_size = batch_size
    
    start_time = time.time()
    results = predictor.train(samples, validation_split=validation_split)
    training_time = time.time() - start_time
    
    print()
    print("TRAINING COMPLETE")
    print("-" * 60)
    print(f"Time: {training_time:.1f}s")
    print(f"Best accuracy: {results['best_val_acc']:.1%}")
    print(f"Best loss: {results['best_val_loss']:.4f}")
    print(f"Model saved: weights/super_pattern_predictor.pth")
    
    # Save training session
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    session_dir = f'input/training_{timestamp}'
    os.makedirs(session_dir, exist_ok=True)
    
    # Save metadata
    metadata = {
        'session_id': timestamp,
        'total_samples': len(samples),
        'training_time': training_time,
        'model_type': 'SuperNeuralNetwork',
        'epochs_trained': results.get('epochs_trained', epochs),
        'best_val_accuracy': results['best_val_acc'],
        'best_val_loss': results['best_val_loss'],
        'configuration': {
            'epochs': epochs,
            'batch_size': batch_size,
            'validation_split': validation_split
        },
        'wedge_distribution': {
            str(i): sum(1 for s in samples if s['wedge_count'] == i) 
            for i in range(1, 7)
        }
    }
    
    with open(f'{session_dir}/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save sample subset for validation
    sample_indices = np.random.choice(len(samples), 
                                    min(100, len(samples)), 
                                    replace=False)
    for idx in sample_indices:
        with open(f'{session_dir}/sample_{idx:05d}.json', 'w') as f:
            json.dump(samples[idx], f, indent=2)
    
    print(f"Session saved: {session_dir}")
    print()
    
    # Performance summary
    print("PERFORMANCE SUMMARY")
    print("-" * 60)
    
    if results['best_val_acc'] >= 0.7:
        print("Status: EXCELLENT - High accuracy achieved")
    elif results['best_val_acc'] >= 0.5:
        print("Status: GOOD - Significant learning achieved")
    else:
        print("Status: BASELINE - Further optimization needed")
    
    print(f"Recommendation: {'Ready for deployment' if results['best_val_acc'] >= 0.6 else 'Continue training with more data'}")
    
    return metadata

if __name__ == "__main__":
    train()