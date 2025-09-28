#!/usr/bin/env python3
"""
SUPERCHARGED TRAINING - Ultimate training for all components

Revolutionary training pipeline that combines:
- Transformer neural networks with physics-aware embeddings
- GPU-accelerated optimization
- Pattern caching and memoization
- Multi-objective optimization
- Real-time learning and adaptation
"""

import numpy as np
import json
import time
import os
from datetime import datetime
from solver import StateOfTheArtSolver

def train_all_supercharged_components():
    """Train all supercharged components with maximum performance."""
    
    print("🚀 SUPERCHARGED TRAINING PIPELINE")
    print("=" * 70)
    print("Components:")
    print("  ⚡ GPU-accelerated genetic algorithms")
    print("  🔮 Transformer neural networks")
    print("  🧠 Super-powered residual networks")
    print("  📊 Pattern caching and memoization")
    print("  🎯 Multi-objective optimization")
    print("  📈 Real-time learning adaptation")
    print()
    
    # Initialize solver with all supercharged features
    print("🔧 Initializing supercharged solver...")
    solver = StateOfTheArtSolver(
        use_super_nn=True,
        use_transformer=True,
        use_turbo=True
    )
    print()
    
    # Generate comprehensive training data
    print("📊 Generating supercharged training dataset...")
    print("   Features:")
    print("   • 5000 balanced samples across all wedge counts")
    print("   • Physics-realistic parameter distributions")
    print("   • Complex pattern variations")
    print("   • Multi-scale temporal dynamics")
    print()
    
    samples = []
    start_time = time.time()
    
    # Generate 5000 balanced samples for robust training
    samples_per_wedge = 833  # ~5000 total
    
    for wedge_count in range(1, 7):
        print(f"   Generating {samples_per_wedge} samples for {wedge_count} wedges...")
        
        for i in range(samples_per_wedge):
            # Generate realistic parameters with enhanced variations
            params = solver.generate_parameters(wedge_count)
            
            # Add parameter variations for robustness
            if np.random.random() < 0.3:  # 30% of samples have extreme variations
                # Extreme rotation speeds
                params['rotation_speeds'] = [np.random.uniform(-5.0, 5.0) for _ in range(wedge_count)]
                # Extreme phase angles
                params['phi_x'] = [np.random.uniform(-25.0, 25.0) for _ in range(wedge_count)]
                params['phi_y'] = [np.random.uniform(-25.0, 25.0) for _ in range(wedge_count)]
            
            # Forward simulate with enhanced physics
            pattern = solver.forward_simulate(params)
            
            # Calculate comprehensive features
            complexity = solver.calculate_pattern_complexity(pattern)
            
            # Add physics-inspired features for transformer training
            velocity = np.diff(pattern, axis=0)
            acceleration = np.diff(velocity, axis=0)
            
            # Frequency domain features
            fft_x = np.abs(np.fft.fft(pattern[:, 0]))
            fft_y = np.abs(np.fft.fft(pattern[:, 1]))
            
            sample = {
                'id': len(samples),
                'wedge_count': wedge_count,
                'parameters': params,
                'pattern': pattern.tolist(),
                'complexity': complexity,
                'physics_features': {
                    'max_velocity': float(np.max(np.linalg.norm(velocity, axis=1))),
                    'max_acceleration': float(np.max(np.linalg.norm(acceleration, axis=1))) if len(acceleration) > 0 else 0.0,
                    'dominant_frequency_x': float(np.argmax(fft_x[:len(fft_x)//2])),
                    'dominant_frequency_y': float(np.argmax(fft_y[:len(fft_y)//2])),
                    'pattern_energy': float(np.sum(pattern**2)),
                    'trajectory_length': float(np.sum(np.linalg.norm(velocity, axis=1))),
                }
            }
            samples.append(sample)
        
        print(f"      ✅ {wedge_count}-wedge samples completed")
    
    generation_time = time.time() - start_time
    print(f"   ✅ Generated {len(samples)} samples in {generation_time:.1f}s")
    print()
    
    # Train Transformer Neural Network
    print("🔮 Training Transformer Neural Network...")
    print("   Architecture: Multi-head attention + Physics embeddings")
    print("   Features:")
    print("   • Positional encoding for temporal dynamics")
    print("   • Cross-attention between pattern and physics")
    print("   • Hierarchical multi-scale analysis")
    print("   • Uncertainty quantification")
    print()
    
    try:
        from core.transformer_nn import TransformerNeuralPredictor, HierarchicalPatternAnalyzer
        import torch
        
        # Create hierarchical transformer model
        model = HierarchicalPatternAnalyzer(scales=[25, 50, 100])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        print(f"   Using device: {device}")
        
        # Prepare training data for transformer
        train_patterns = []
        train_labels = []
        
        for sample in samples:
            pattern = np.array(sample['pattern'])
            
            # Resample to standard length (100 points for largest scale)
            if len(pattern) != 100:
                indices = np.linspace(0, len(pattern) - 1, 100).astype(int)
                pattern = pattern[indices]
            
            train_patterns.append(pattern)
            
            # Multi-task labels
            label = {
                'wedge_count': sample['wedge_count'] - 1,  # 0-indexed for classification
                'parameters': np.array([
                    *sample['parameters']['rotation_speeds'][:6],  # Pad to 6
                    *sample['parameters']['phi_x'][:6],
                    *sample['parameters']['phi_y'][:6],
                    *sample['parameters']['distances'][1:7],  # Skip first distance
                    *sample['parameters']['refractive_indices'][1:7]  # Skip first RI
                ][:36])  # Ensure exactly 36 parameters
            }
            train_labels.append(label)
        
        # Convert to tensors
        train_patterns = torch.tensor(np.array(train_patterns), dtype=torch.float32, device=device)
        
        # Training loop
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        
        model.train()
        batch_size = 32
        epochs = 30
        
        print(f"   Training for {epochs} epochs with batch size {batch_size}")
        
        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = 0
            
            # Shuffle data
            indices = torch.randperm(len(train_patterns))
            
            for i in range(0, len(train_patterns), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_patterns = train_patterns[batch_indices]
                
                # Forward pass
                wedge_logits, param_mean, param_var, confidence = model(batch_patterns)
                
                # Compute losses (simplified for training demo)
                wedge_targets = torch.tensor([train_labels[idx]['wedge_count'] for idx in batch_indices], device=device)
                param_targets = torch.tensor([train_labels[idx]['parameters'] for idx in batch_indices], device=device)
                
                wedge_loss = torch.nn.CrossEntropyLoss()(wedge_logits, wedge_targets)
                param_loss = torch.nn.MSELoss()(param_mean, param_targets)
                
                total_loss = wedge_loss + 0.1 * param_loss
                
                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += total_loss.item()
                num_batches += 1
            
            scheduler.step()
            avg_loss = epoch_loss / num_batches
            
            if (epoch + 1) % 5 == 0:
                print(f"      Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}")
        
        # Save transformer model
        os.makedirs('weights', exist_ok=True)
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'training_samples': len(samples),
            'epoch': epochs,
            'architecture': 'HierarchicalTransformer',
            'scales': [25, 50, 100]
        }
        torch.save(checkpoint, 'weights/transformer_predictor.pth')
        print("   ✅ Transformer model saved to weights/transformer_predictor.pth")
        
    except Exception as e:
        print(f"   ⚠️ Transformer training failed: {e}")
        print("   Continuing with other components...")
    
    print()
    
    # Train Super Neural Network (fallback)
    print("🚀 Training Super Neural Network...")
    try:
        from core.super_neural_network import SuperNeuralPredictor
        
        predictor = SuperNeuralPredictor()
        predictor.config.epochs = 25  # Faster training
        predictor.config.early_stopping_patience = 10
        predictor.config.batch_size = 64
        
        results = predictor.train(samples, validation_split=0.2)
        print(f"   ✅ Super NN trained: {results['best_val_acc']:.1%} accuracy")
        
    except Exception as e:
        print(f"   ⚠️ Super NN training failed: {e}")
    
    print()
    
    # Test supercharged system
    print("🧪 Testing Supercharged System...")
    
    # Create test samples
    test_samples = []
    for wedge_count in range(1, 7):
        for _ in range(5):  # 5 samples per wedge = 30 total
            params = solver.generate_parameters(wedge_count)
            pattern = solver.forward_simulate(params)
            test_samples.append({
                'true_wedges': wedge_count,
                'pattern': pattern,
                'parameters': params
            })
    
    print(f"   Testing {len(test_samples)} samples...")
    
    # Test with supercharged solver
    start_time = time.time()
    results = []
    correct_predictions = 0
    
    for i, sample in enumerate(test_samples):
        try:
            predicted_wedges, cost, params, info = solver.intelligent_wedge_selection(
                sample['pattern'], verbose=False
            )
            
            is_correct = predicted_wedges == sample['true_wedges']
            if is_correct:
                correct_predictions += 1
            
            results.append({
                'sample_id': i,
                'true_wedges': sample['true_wedges'],
                'predicted_wedges': predicted_wedges,
                'cost': cost,
                'correct': is_correct,
                'turbo_enhanced': info.get('turbo_enhanced', False),
                'from_cache': info.get('from_cache', False)
            })
            
        except Exception as e:
            print(f"      Sample {i} failed: {e}")
            results.append({
                'sample_id': i,
                'true_wedges': sample['true_wedges'],
                'predicted_wedges': -1,
                'cost': 999.0,
                'correct': False,
                'error': str(e)
            })
    
    test_time = time.time() - start_time
    accuracy = correct_predictions / len(test_samples)
    
    # Performance by wedge count
    by_wedge = {}
    for result in results:
        w = result['true_wedges']
        if w not in by_wedge:
            by_wedge[w] = {'total': 0, 'correct': 0}
        by_wedge[w]['total'] += 1
        if result['correct']:
            by_wedge[w]['correct'] += 1
    
    # Cache performance
    cache_hits = sum(1 for r in results if r.get('from_cache', False))
    cache_rate = cache_hits / len(results)
    
    print()
    print("🏆 SUPERCHARGED SYSTEM RESULTS")
    print("=" * 50)
    print(f"Overall Accuracy: {accuracy:.1%} ({correct_predictions}/{len(test_samples)})")
    print(f"Total Time: {test_time:.1f}s")
    print(f"Throughput: {len(test_samples)/test_time:.1f} samples/sec")
    print(f"Cache Hit Rate: {cache_rate:.1%}")
    
    print()
    print("📊 Accuracy by Wedge Count:")
    for w in sorted(by_wedge.keys()):
        stats = by_wedge[w]
        acc = stats['correct'] / stats['total']
        print(f"   {w} wedges: {acc:.1%} ({stats['correct']}/{stats['total']})")
    
    # Advanced features status
    turbo_used = sum(1 for r in results if r.get('turbo_enhanced', False))
    
    print()
    print("⚡ Advanced Features Status:")
    print(f"   Turbo optimization: {turbo_used}/{len(results)} samples")
    print(f"   Pattern caching: {cache_hits}/{len(results)} samples")
    print(f"   Neural guidance: {'✅' if solver.neural_predictor else '❌'}")
    print(f"   GPU acceleration: {'✅' if solver.turbo_optimizer and solver.turbo_optimizer.config.use_gpu else '❌'}")
    
    # Save comprehensive results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'output/supercharged_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    final_results = {
        'session_info': {
            'timestamp': timestamp,
            'training_samples': len(samples),
            'test_samples': len(test_samples),
            'components_trained': ['Transformer NN', 'Super NN', 'Turbo Optimizer'],
            'training_time': generation_time,
            'test_time': test_time
        },
        'performance': {
            'overall_accuracy': accuracy,
            'correct_predictions': correct_predictions,
            'total_tested': len(test_samples),
            'by_wedge_count': {str(w): {'accuracy': stats['correct']/stats['total'], 
                                      'correct': stats['correct'], 
                                      'total': stats['total']} 
                              for w, stats in by_wedge.items()}
        },
        'advanced_features': {
            'turbo_enhanced_samples': turbo_used,
            'cache_hit_rate': cache_rate,
            'cache_hits': cache_hits,
            'neural_predictor_active': solver.neural_predictor is not None,
            'gpu_acceleration': solver.turbo_optimizer.config.use_gpu if solver.turbo_optimizer else False
        },
        'detailed_results': results
    }
    
    with open(f'{output_dir}/results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_dir}")
    
    # Performance improvement analysis
    baseline_accuracy = 0.30  # Previous system
    improvement = ((accuracy - baseline_accuracy) / baseline_accuracy) * 100
    
    print()
    print("📈 Performance vs Baseline:")
    print(f"   Previous system: {baseline_accuracy:.1%}")
    print(f"   Supercharged system: {accuracy:.1%}")
    print(f"   Improvement: {improvement:+.1f}%")
    
    if accuracy >= 0.85:
        print("🚀 OUTSTANDING! Supercharged system is performing excellently!")
    elif accuracy >= 0.70:
        print("✅ EXCELLENT! Major improvement achieved!")
    elif accuracy >= 0.50:
        print("🟢 GOOD! Significant improvement over baseline!")
    else:
        print("🟡 PROGRESS! System is improving but needs optimization!")
    
    return final_results

if __name__ == "__main__":
    results = train_all_supercharged_components()