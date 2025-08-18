#!/usr/bin/env python3
"""
LARGE-SCALE TRAINING - Train with extensive dataset
"""

import numpy as np
import json
import time
import os
from datetime import datetime
from solver import StateOfTheArtSolver
from core.super_neural_network import SuperNeuralPredictor

def train_large():
    """Train with large dataset for maximum performance."""
    
    print("LARGE-SCALE NEURAL NETWORK TRAINING")
    print("=" * 60)
    
    # Large-scale configuration
    num_samples = 10000  # Large balanced dataset
    validation_split = 0.2
    epochs = 100  # More epochs for convergence
    batch_size = 128  # Larger batch for efficiency
    
    print(f"Configuration:")
    print(f"  Samples: {num_samples:,}")
    print(f"  Validation: {int(validation_split*100)}%")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print()
    
    # Initialize solver
    print("Initializing system...")
    solver = StateOfTheArtSolver(use_super_nn=False)
    
    # Generate large training dataset
    print("Generating large training dataset...")
    samples = []
    samples_per_wedge = num_samples // 6
    
    for wedge_count in range(1, 7):
        print(f"  Generating {samples_per_wedge:,} samples for {wedge_count} wedges...")
        
        for i in range(samples_per_wedge):
            # Generate parameters with various complexity levels
            params = solver.generate_parameters(wedge_count)
            
            # Add different variations for robustness
            variation_type = np.random.choice(['normal', 'extreme', 'moderate', 'subtle'])
            
            if variation_type == 'extreme':  # 25% extreme
                params['rotation_speeds'] = [np.random.uniform(-5.0, 5.0) 
                                            for _ in range(wedge_count)]
                params['phi_x'] = [np.random.uniform(-30.0, 30.0) 
                                  for _ in range(wedge_count)]
                params['phi_y'] = [np.random.uniform(-30.0, 30.0) 
                                  for _ in range(wedge_count)]
            elif variation_type == 'moderate':  # 25% moderate
                params['rotation_speeds'] = [np.random.uniform(-3.5, 3.5) 
                                            for _ in range(wedge_count)]
                params['phi_x'] = [np.random.uniform(-20.0, 20.0) 
                                  for _ in range(wedge_count)]
                params['phi_y'] = [np.random.uniform(-20.0, 20.0) 
                                  for _ in range(wedge_count)]
            elif variation_type == 'subtle':  # 25% subtle
                params['rotation_speeds'] = [np.random.uniform(-1.5, 1.5) 
                                            for _ in range(wedge_count)]
                params['phi_x'] = [np.random.uniform(-8.0, 8.0) 
                                  for _ in range(wedge_count)]
                params['phi_y'] = [np.random.uniform(-8.0, 8.0) 
                                  for _ in range(wedge_count)]
            # else: 25% normal (no change)
            
            # Simulate pattern
            pattern = solver.forward_simulate(params)
            
            # Add noise to some patterns for robustness
            if np.random.random() < 0.15:  # 15% with noise
                noise_level = np.random.uniform(0.01, 0.08)
                noise = np.random.normal(0, noise_level, pattern.shape)
                pattern += noise
            
            # Store sample
            sample = {
                'id': len(samples),
                'wedge_count': wedge_count,
                'parameters': params,
                'pattern': pattern.tolist(),
                'complexity': solver.calculate_pattern_complexity(pattern),
                'variation': variation_type
            }
            samples.append(sample)
            
            if (len(samples) % 1000) == 0:
                print(f"    Progress: {len(samples):,}/{num_samples:,} samples...")
    
    # Pad to exact number if needed
    while len(samples) < num_samples:
        wedge_count = np.random.randint(1, 7)
        params = solver.generate_parameters(wedge_count)
        pattern = solver.forward_simulate(params)
        samples.append({
            'id': len(samples),
            'wedge_count': wedge_count,
            'parameters': params,
            'pattern': pattern.tolist(),
            'complexity': solver.calculate_pattern_complexity(pattern),
            'variation': 'normal'
        })
    
    print(f"  Generated {len(samples):,} total samples")
    print()
    
    # Train neural network with large dataset
    print("Training neural network on large dataset...")
    print("This may take several minutes...")
    
    predictor = SuperNeuralPredictor()
    predictor.config.epochs = epochs
    predictor.config.early_stopping_patience = 20  # More patience for large dataset
    predictor.config.batch_size = batch_size
    predictor.config.learning_rate = 0.001
    predictor.config.min_learning_rate = 1e-6
    
    start_time = time.time()
    results = predictor.train(samples, validation_split=validation_split)
    training_time = time.time() - start_time
    
    print()
    print("TRAINING COMPLETE")
    print("-" * 60)
    print(f"Time: {training_time:.1f}s ({training_time/60:.1f} minutes)")
    print(f"Best accuracy: {results['best_val_acc']:.1%}")
    print(f"Best loss: {results['best_val_loss']:.4f}")
    print(f"Final epoch: {results.get('epochs_trained', epochs)}")
    print(f"Model saved: weights/super_pattern_predictor.pth")
    
    # Save training session
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    session_dir = f'input/large_training_{timestamp}'
    os.makedirs(session_dir, exist_ok=True)
    
    # Save metadata
    metadata = {
        'session_id': timestamp,
        'total_samples': len(samples),
        'training_time': training_time,
        'model_type': 'SuperNeuralNetwork',
        'training_type': 'large_scale',
        'epochs_trained': results.get('epochs_trained', epochs),
        'best_val_accuracy': results['best_val_acc'],
        'best_val_loss': results['best_val_loss'],
        'configuration': {
            'epochs': epochs,
            'batch_size': batch_size,
            'validation_split': validation_split,
            'learning_rate': 0.001
        },
        'wedge_distribution': {
            str(i): sum(1 for s in samples if s['wedge_count'] == i) 
            for i in range(1, 7)
        },
        'variation_distribution': {
            var: sum(1 for s in samples if s.get('variation') == var)
            for var in ['normal', 'extreme', 'moderate', 'subtle']
        }
    }
    
    with open(f'{session_dir}/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save sample subset for validation (don't save all 10k)
    sample_indices = np.random.choice(len(samples), min(200, len(samples)), replace=False)
    for idx in sample_indices:
        with open(f'{session_dir}/sample_{idx:05d}.json', 'w') as f:
            json.dump(samples[idx], f, indent=2)
    
    print(f"Session saved: {session_dir}")
    print()
    
    # Performance summary
    print("PERFORMANCE SUMMARY")
    print("-" * 60)
    
    if results['best_val_acc'] >= 0.8:
        print("Status: EXCELLENT - High accuracy achieved!")
        print("The model is well-trained and ready for deployment.")
    elif results['best_val_acc'] >= 0.65:
        print("Status: VERY GOOD - Strong performance achieved!")
        print("The model shows good generalization.")
    elif results['best_val_acc'] >= 0.5:
        print("Status: GOOD - Solid baseline established!")
        print("The model has learned meaningful patterns.")
    else:
        print("Status: TRAINING IN PROGRESS")
        print("Consider adjusting hyperparameters or architecture.")
    
    print()
    print(f"Samples per second: {len(samples)/training_time:.1f}")
    print(f"Training efficiency: {training_time/epochs:.2f}s per epoch")
    
    return metadata

if __name__ == "__main__":
    train_large()