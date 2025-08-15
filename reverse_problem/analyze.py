#!/usr/bin/env python3
"""
ANALYZE - Comprehensive analysis of hybrid NN+GA system performance

Usage: python3 analyze.py
Input: Latest prediction results from output/ + training data from input/
Output: Detailed analysis with improvement tracking and optimization suggestions
"""

import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict

def load_latest_predictions():
    """Load latest prediction results."""
    prediction_dirs = glob.glob("output/predictions_*")
    if not prediction_dirs:
        print("❌ No prediction results found. Run predict.py first.")
        return None, None
    
    latest_dir = max(prediction_dirs, key=os.path.getmtime)
    
    with open(f"{latest_dir}/results.json", 'r') as f:
        data = json.load(f)
    
    return data, latest_dir

def load_training_info():
    """Load training data and neural network info."""
    training_dirs = glob.glob("input/training_*")
    if not training_dirs:
        return None, None
    
    latest_training = max(training_dirs, key=os.path.getmtime)
    
    # Load training metadata
    metadata_file = f"{latest_training}/metadata.json"
    training_results_file = f"{latest_training}/training_results.json"
    
    metadata = None
    training_results = None
    
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    
    if os.path.exists(training_results_file):
        with open(training_results_file, 'r') as f:
            training_results = json.load(f)
    
    return metadata, training_results

def load_neural_network_info():
    """Load neural network training information."""
    try:
        from core.neural_network import NeuralPredictor
        predictor = NeuralPredictor()
        return predictor.get_training_info()
    except:
        return {'trained': False}

def analyze_neural_network_accuracy(data):
    """Analyze neural network prediction accuracy vs GA results."""
    detailed_results = data.get('detailed_results', [])
    
    nn_accuracy_stats = {
        'wedge_count_accuracy': 0,
        'parameter_correlation': {},
        'nn_vs_ga_comparison': []
    }
    
    if not detailed_results:
        return nn_accuracy_stats
    
    wedge_correct = 0
    total_samples = 0
    
    for result in detailed_results:
        if 'nn_prediction' in result and result['nn_prediction']:
            total_samples += 1
            
            # Check wedge count accuracy
            true_wedges = result['true_wedges']
            nn_prediction = result['nn_prediction']
            
            if 'wedgenum' in nn_prediction:
                predicted_wedges = nn_prediction['wedgenum']
                if predicted_wedges == true_wedges:
                    wedge_correct += 1
                
                # Store comparison data
                nn_vs_ga_comparison = {
                    'true_wedges': true_wedges,
                    'nn_predicted_wedges': predicted_wedges,
                    'ga_final_wedges': result['predicted_wedges'],
                    'nn_correct': predicted_wedges == true_wedges,
                    'ga_correct': result['correct'],
                    'final_cost': result['cost']
                }
                nn_accuracy_stats['nn_vs_ga_comparison'].append(nn_vs_ga_comparison)
    
    if total_samples > 0:
        nn_accuracy_stats['wedge_count_accuracy'] = wedge_correct / total_samples
    
    return nn_accuracy_stats

def analyze_cost_patterns(data):
    """Analyze cost patterns and convergence."""
    detailed_results = data.get('detailed_results', [])
    
    cost_analysis = {
        'by_wedge_count': defaultdict(list),
        'by_complexity': defaultdict(list),
        'convergence_patterns': []
    }
    
    for result in detailed_results:
        wedges = result['true_wedges']
        cost = result['cost']
        correct = result['correct']
        
        cost_analysis['by_wedge_count'][wedges].append({
            'cost': cost,
            'correct': correct
        })
    
    return cost_analysis

def print_comprehensive_analysis(data, training_metadata, training_results, nn_info):
    """Print detailed analysis report."""
    session = data['session_info']
    performance = data['performance']
    timing = data.get('timing', {})
    
    print("🎯 COMPREHENSIVE HYBRID NN+GA SYSTEM ANALYSIS")
    print("=" * 80)
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Session: {session['timestamp']}")
    print(f"Source Training: {session['source_training']}")
    print()
    
    # =============================================================================
    # TRAINING DATA ANALYSIS
    # =============================================================================
    print("📚 TRAINING DATA ANALYSIS")
    print("-" * 50)
    
    if training_metadata:
        total_training = training_metadata['total_samples']
        wedge_dist = training_metadata['wedge_distribution']
        
        print(f"Training Samples: {total_training:,}")
        print(f"Wedge Distribution:")
        for wedges, count in sorted(wedge_dist.items(), key=lambda x: int(x[0])):
            pct = (count / total_training) * 100
            print(f"  {wedges} wedges: {count:4d} samples ({pct:5.1f}%)")
        
        # Check for imbalance
        counts = list(wedge_dist.values())
        balance_ratio = max(counts) / min(counts) if min(counts) > 0 else float('inf')
        if balance_ratio > 2.0:
            print(f"⚠️  DATA IMBALANCE: Ratio {balance_ratio:.1f}:1 - consider rebalancing")
        else:
            print(f"✅ WELL BALANCED: Ratio {balance_ratio:.1f}:1")
    
    print()
    
    # =============================================================================
    # NEURAL NETWORK ANALYSIS
    # =============================================================================
    print("🧠 NEURAL NETWORK ANALYSIS")
    print("-" * 50)
    
    if nn_info.get('trained', False):
        epochs = nn_info.get('epochs_trained', 0)
        final_train_loss = nn_info.get('final_train_loss', 0)
        final_val_loss = nn_info.get('final_val_loss', 0)
        
        print(f"Training Status: ✅ TRAINED")
        print(f"Epochs Trained: {epochs}")
        print(f"Final Training Loss: {final_train_loss:.6f}")
        print(f"Final Validation Loss: {final_val_loss:.6f}")
        
        # Loss analysis
        if final_val_loss < 0.05:
            print(f"Loss Quality: 🟢 EXCELLENT (< 0.05)")
        elif final_val_loss < 0.1:
            print(f"Loss Quality: 🟡 GOOD (< 0.1)")
        else:
            print(f"Loss Quality: 🔴 POOR (≥ 0.1)")
        
        # Overfitting check
        if final_train_loss > 0 and final_val_loss > 0:
            overfitting_ratio = final_val_loss / final_train_loss
            if overfitting_ratio > 1.5:
                print(f"⚠️  OVERFITTING DETECTED: Val/Train ratio = {overfitting_ratio:.2f}")
            else:
                print(f"✅ NO OVERFITTING: Val/Train ratio = {overfitting_ratio:.2f}")
    else:
        print(f"Training Status: ❌ NOT TRAINED")
        print(f"Recommendation: Run train.py first")
    
    # Neural network accuracy analysis
    nn_analysis = analyze_neural_network_accuracy(data)
    nn_wedge_acc = nn_analysis['wedge_count_accuracy']
    
    if nn_wedge_acc > 0:
        print(f"NN Wedge Count Accuracy: {nn_wedge_acc:.1%}")
        if nn_wedge_acc >= 0.8:
            print(f"NN Performance: 🟢 EXCELLENT")
        elif nn_wedge_acc >= 0.6:
            print(f"NN Performance: 🟡 GOOD")
        elif nn_wedge_acc >= 0.4:
            print(f"NN Performance: 🟠 FAIR")
        else:
            print(f"NN Performance: 🔴 POOR")
    
    print()
    
    # =============================================================================
    # HYBRID SYSTEM PERFORMANCE
    # =============================================================================
    print("⚡ HYBRID SYSTEM PERFORMANCE")
    print("-" * 50)
    
    if timing.get('use_hybrid', False):
        total_time = timing.get('total_time', 0)
        nn_time = timing.get('total_nn_time', 0)
        ga_time = timing.get('total_ga_time', 0)
        avg_speedup = timing.get('average_speedup', 1.0)
        
        samples_tested = performance['total_tested']
        samples_per_sec = samples_tested / total_time if total_time > 0 else 0
        
        print(f"Hybrid Mode: ✅ ACTIVE")
        print(f"Total Runtime: {total_time:.1f}s")
        print(f"Neural Network Time: {nn_time:.3f}s ({nn_time/total_time*100:.1f}%)")
        print(f"Genetic Algorithm Time: {ga_time:.1f}s ({ga_time/total_time*100:.1f}%)")
        print(f"Throughput: {samples_per_sec:.2f} samples/sec")
        print(f"Average per Sample: {total_time/samples_tested:.1f}s")
        
        # Performance classification
        if samples_per_sec >= 1.0:
            print(f"Speed Rating: 🟢 FAST")
        elif samples_per_sec >= 0.5:
            print(f"Speed Rating: 🟡 MODERATE")
        else:
            print(f"Speed Rating: 🔴 SLOW")
        
        # NN vs GA comparison
        nn_vs_ga = nn_analysis['nn_vs_ga_comparison']
        if nn_vs_ga:
            nn_correct = sum(1 for x in nn_vs_ga if x['nn_correct'])
            ga_correct = sum(1 for x in nn_vs_ga if x['ga_correct'])
            nn_helped = sum(1 for x in nn_vs_ga if x['nn_correct'] and x['ga_correct'])
            
            print(f"NN Standalone Accuracy: {nn_correct/len(nn_vs_ga):.1%}")
            print(f"GA Final Accuracy: {ga_correct/len(nn_vs_ga):.1%}")
            print(f"NN Helped GA: {nn_helped/len(nn_vs_ga):.1%}")
    else:
        print(f"Hybrid Mode: ❌ DISABLED")
        print(f"Recommendation: Train neural network with train.py")
    
    print()
    
    # =============================================================================
    # DETAILED ACCURACY ANALYSIS
    # =============================================================================
    print("🎯 DETAILED ACCURACY ANALYSIS")
    print("-" * 50)
    
    overall_acc = performance['overall_accuracy'] * 100
    total = performance['total_tested']
    correct = performance['correct_predictions']
    
    print(f"Overall Accuracy: {overall_acc:.1f}% ({correct}/{total})")
    
    # Performance by wedge count with detailed stats
    by_wedge = performance['by_wedge_count']
    print(f"\nPerformance by Wedge Count:")
    
    cost_analysis = analyze_cost_patterns(data)
    
    for wedge_count in sorted(by_wedge.keys(), key=int):
        stats = by_wedge[wedge_count]
        acc = stats['accuracy'] * 100
        total_w = stats['total']
        correct_w = stats['correct']
        
        costs = cost_analysis['by_wedge_count'][int(wedge_count)]
        if costs:
            avg_cost = np.mean([c['cost'] for c in costs])
            correct_costs = [c['cost'] for c in costs if c['correct']]
            incorrect_costs = [c['cost'] for c in costs if not c['correct']]
            
            print(f"  {wedge_count} wedges: {acc:5.1f}% ({correct_w:2d}/{total_w:2d}) | "
                  f"Avg Cost: {avg_cost:.3f}")
            
            if correct_costs and incorrect_costs:
                print(f"    Correct samples avg cost: {np.mean(correct_costs):.3f}")
                print(f"    Incorrect samples avg cost: {np.mean(incorrect_costs):.3f}")
    
    print()
    
    # =============================================================================
    # OPTIMIZATION INSIGHTS
    # =============================================================================
    print("🔧 OPTIMIZATION INSIGHTS & RECOMMENDATIONS")
    print("-" * 50)
    
    # Identify bottlenecks
    if timing.get('use_hybrid', False):
        ga_percentage = timing.get('ga_percentage', 100)
        if ga_percentage > 95:
            print("🔍 BOTTLENECK: Genetic Algorithm dominates runtime (>95%)")
            print("   → Reduce GA population sizes or generations")
            print("   → Improve neural network to need less GA refinement")
        
        if nn_wedge_acc < 0.5:
            print("🔍 ISSUE: Neural network wedge prediction accuracy < 50%")
            print("   → Increase training data size")
            print("   → Try different network architecture")
            print("   → Check data quality and balance")
        
        # Cost analysis insights
        all_costs = [r['cost'] for r in data.get('detailed_results', [])]
        if all_costs:
            cost_std = np.std(all_costs)
            cost_mean = np.mean(all_costs)
            
            if cost_std / cost_mean > 0.5:
                print("🔍 ISSUE: High cost variance suggests inconsistent optimization")
                print("   → Check GA parameter settings")
                print("   → Verify pattern complexity normalization")
    
    # Training recommendations
    if training_metadata:
        training_size = training_metadata['total_samples']
        if training_size < 1000:
            print("🔍 RECOMMENDATION: Small training set")
            print(f"   → Current: {training_size} samples, try 2000+ for better NN performance")
        elif training_size < 5000:
            print("🔍 RECOMMENDATION: Medium training set")
            print(f"   → Current: {training_size} samples, try 5000+ for excellent NN performance")
    
    # Performance targets
    print(f"\n📊 PERFORMANCE TARGETS:")
    print(f"   Current Overall Accuracy: {overall_acc:.1f}%")
    
    if overall_acc < 30:
        print(f"   → Target: 50% (Focus on neural network training)")
    elif overall_acc < 50:
        print(f"   → Target: 70% (Optimize hybrid system balance)")
    elif overall_acc < 70:
        print(f"   → Target: 80% (Fine-tune parameters)")
    else:
        print(f"   → Target: 90% (System optimization)")
    
    print()

def create_comprehensive_plots(data, training_metadata, nn_info):
    """Create detailed performance visualization."""
    performance = data['performance']
    by_wedge = performance['by_wedge_count']
    timing = data.get('timing', {})
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    fig.suptitle('Comprehensive Hybrid NN+GA System Analysis', fontsize=16, fontweight='bold')
    
    # 1. Overall accuracy gauge
    ax1 = fig.add_subplot(gs[0, 0])
    overall_acc = performance['overall_accuracy'] * 100
    color = 'green' if overall_acc >= 70 else 'orange' if overall_acc >= 50 else 'red'
    ax1.pie([overall_acc, 100-overall_acc], colors=[color, 'lightgray'], 
            startangle=90, counterclock=False)
    ax1.add_patch(plt.Circle((0,0), 0.7, color='white'))
    ax1.text(0, 0, f'{overall_acc:.1f}%', ha='center', va='center', 
             fontsize=16, fontweight='bold')
    ax1.set_title('Overall Accuracy')
    
    # 2. Accuracy by wedge count
    ax2 = fig.add_subplot(gs[0, 1:3])
    wedge_counts = sorted(by_wedge.keys(), key=int)
    accuracies = [by_wedge[w]['accuracy'] * 100 for w in wedge_counts]
    sample_counts = [by_wedge[w]['total'] for w in wedge_counts]
    
    colors = ['green' if acc >= 70 else 'orange' if acc >= 50 else 'red' for acc in accuracies]
    bars = ax2.bar(wedge_counts, accuracies, color=colors, alpha=0.7)
    
    # Add sample count labels
    for bar, acc, count in zip(bars, accuracies, sample_counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 2, 
                f'{acc:.0f}%\n({count})', ha='center', va='bottom', fontsize=10)
    
    ax2.set_xlabel('Wedge Count')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy by Wedge Count')
    ax2.set_ylim(0, 110)
    
    # 3. Training data distribution
    ax3 = fig.add_subplot(gs[0, 3])
    if training_metadata:
        wedge_dist = training_metadata['wedge_distribution']
        labels = [f'{w}w' for w in sorted(wedge_dist.keys(), key=int)]
        sizes = [wedge_dist[w] for w in sorted(wedge_dist.keys(), key=int)]
        ax3.pie(sizes, labels=labels, autopct='%1.0f%%', startangle=90)
    ax3.set_title('Training Data\nDistribution')
    
    # 4. Neural network training progress
    ax4 = fig.add_subplot(gs[1, 0:2])
    if nn_info.get('trained', False):
        train_losses = nn_info.get('train_losses', [])
        val_losses = nn_info.get('val_losses', [])
        
        if train_losses and val_losses:
            epochs = range(1, len(train_losses) + 1)
            ax4.plot(epochs, train_losses, 'b-', label='Training Loss', alpha=0.7)
            ax4.plot(epochs, val_losses, 'r-', label='Validation Loss', alpha=0.7)
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Loss')
            ax4.set_title('Neural Network Training Progress')
            ax4.legend()
            ax4.set_yscale('log')
        else:
            ax4.text(0.5, 0.5, 'No training\nhistory available', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('NN Training Progress')
    else:
        ax4.text(0.5, 0.5, 'Neural Network\nNot Trained', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=14)
        ax4.set_title('NN Training Status')
    
    # 5. Timing breakdown
    ax5 = fig.add_subplot(gs[1, 2])
    if timing.get('use_hybrid', False):
        nn_time = timing.get('total_nn_time', 0)
        ga_time = timing.get('total_ga_time', 0)
        
        if nn_time + ga_time > 0:
            ax5.pie([nn_time, ga_time], labels=['NN', 'GA'], 
                   colors=['lightblue', 'lightcoral'], autopct='%1.1f%%', startangle=90)
        ax5.set_title('Time Distribution')
    else:
        ax5.text(0.5, 0.5, 'Hybrid Mode\nDisabled', 
                ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Timing Analysis')
    
    # 6. Performance trends (if multiple sessions available)
    ax6 = fig.add_subplot(gs[1, 3])
    # Placeholder for trend analysis
    ax6.text(0.5, 0.5, 'Performance\nTrends\n(Future)', 
            ha='center', va='center', transform=ax6.transAxes)
    ax6.set_title('Improvement Trends')
    
    # 7. Cost distribution analysis
    ax7 = fig.add_subplot(gs[2, 0:2])
    detailed_results = data.get('detailed_results', [])
    if detailed_results:
        costs = [r['cost'] for r in detailed_results]
        correct_costs = [r['cost'] for r in detailed_results if r['correct']]
        incorrect_costs = [r['cost'] for r in detailed_results if not r['correct']]
        
        ax7.hist(incorrect_costs, bins=20, alpha=0.5, color='red', label='Incorrect')
        ax7.hist(correct_costs, bins=20, alpha=0.5, color='green', label='Correct')
        ax7.set_xlabel('Cost')
        ax7.set_ylabel('Frequency')
        ax7.set_title('Cost Distribution: Correct vs Incorrect')
        ax7.legend()
    
    # 8. Neural network vs GA accuracy comparison
    ax8 = fig.add_subplot(gs[2, 2:4])
    nn_analysis = analyze_neural_network_accuracy(data)
    nn_vs_ga = nn_analysis['nn_vs_ga_comparison']
    
    if nn_vs_ga:
        nn_correct = [1 if x['nn_correct'] else 0 for x in nn_vs_ga]
        ga_correct = [1 if x['ga_correct'] else 0 for x in nn_vs_ga]
        
        # Create scatter plot
        sample_indices = range(len(nn_vs_ga))
        ax8.scatter(sample_indices, nn_correct, alpha=0.6, color='blue', label='NN Prediction', s=30)
        ax8.scatter(sample_indices, [x + 0.05 for x in ga_correct], alpha=0.6, color='red', label='GA Final', s=30)
        ax8.set_xlabel('Sample Index')
        ax8.set_ylabel('Correct (1) / Incorrect (0)')
        ax8.set_title('NN vs GA Prediction Accuracy')
        ax8.set_ylim(-0.1, 1.2)
        ax8.legend()
    
    plt.tight_layout()
    return fig

def main():
    """Main comprehensive analysis function."""
    print("🔍 COMPREHENSIVE RISLEY PRISM ANALYSIS")
    print("=" * 60)
    
    # Load all data
    data, prediction_dir = load_latest_predictions()
    if not data:
        return
    
    training_metadata, training_results = load_training_info()
    nn_info = load_neural_network_info()
    
    print(f"📁 Analyzing: {os.path.basename(prediction_dir)}")
    print()
    
    # Create results directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f"results/analysis_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Print comprehensive analysis
    print_comprehensive_analysis(data, training_metadata, training_results, nn_info)
    
    # Create comprehensive visualization dashboard
    print("📊 GENERATING COMPREHENSIVE DASHBOARD...")
    fig = create_comprehensive_plots(data, training_metadata, nn_info)
    
    # Save detailed dashboard to results
    dashboard_file = f"{results_dir}/dashboard.png"
    fig.savefig(dashboard_file, dpi=300, bbox_inches='tight')
    
    print(f"✅ Dashboard saved: {dashboard_file}")
    plt.close(fig)
    
    # Save comprehensive analysis report
    analysis_data = {
        'timestamp': datetime.now().isoformat(),
        'session_analyzed': os.path.basename(prediction_dir),
        'overall_accuracy': data['performance']['overall_accuracy'],
        'neural_network_info': nn_info,
        'training_info': training_metadata,
        'timing_info': data.get('timing', {}),
        'detailed_metrics': analyze_neural_network_accuracy(data),
        'performance_by_wedge': data['performance']['by_wedge_count'],
        'cost_analysis': analyze_cost_patterns(data)
    }
    
    # Save analysis data
    analysis_file = f"{results_dir}/analysis_report.json"
    with open(analysis_file, 'w') as f:
        json.dump(analysis_data, f, indent=2)
    
    # Save human-readable report
    report_file = f"{results_dir}/performance_report.txt"
    with open(report_file, 'w') as f:
        # Redirect print output to file
        import sys
        original_stdout = sys.stdout
        sys.stdout = f
        
        print_comprehensive_analysis(data, training_metadata, training_results, nn_info)
        
        sys.stdout = original_stdout
    
    # Create summary stats file
    summary_stats = {
        'analysis_timestamp': datetime.now().isoformat(),
        'overall_accuracy_percent': round(data['performance']['overall_accuracy'] * 100, 1),
        'total_samples_tested': data['performance']['total_tested'],
        'correct_predictions': data['performance']['correct_predictions'],
        'neural_network_trained': nn_info.get('trained', False),
        'neural_network_epochs': nn_info.get('epochs_trained', 0),
        'neural_network_validation_loss': nn_info.get('final_val_loss', 0),
        'neural_network_wedge_accuracy_percent': round(analyze_neural_network_accuracy(data)['wedge_count_accuracy'] * 100, 1),
        'hybrid_mode_active': data.get('timing', {}).get('use_hybrid', False),
        'throughput_samples_per_sec': round(data['performance']['total_tested'] / data.get('timing', {}).get('total_time', 1), 2),
        'training_samples_used': training_metadata.get('total_samples', 0) if training_metadata else 0,
        'best_performing_wedge_count': max(data['performance']['by_wedge_count'].items(), key=lambda x: x[1]['accuracy'])[0],
        'worst_performing_wedge_count': min(data['performance']['by_wedge_count'].items(), key=lambda x: x[1]['accuracy'])[0]
    }
    
    stats_file = f"{results_dir}/summary_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    print(f"📊 Analysis report saved: {analysis_file}")
    print(f"📄 Performance report saved: {report_file}")
    print(f"📈 Summary stats saved: {stats_file}")
    print(f"📁 All results saved to: {results_dir}")
    print()
    print("🎯 ANALYSIS COMPLETE!")

if __name__ == "__main__":
    main()