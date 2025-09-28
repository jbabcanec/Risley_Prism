#!/usr/bin/env python3
"""
Real-time Metrics Dashboard for Reverse Problem Solver

Modern dashboard that tracks:
- Neural network performance evolution
- Hybrid system accuracy trends
- Training progress metrics
- Performance by wedge count
- Cumulative improvement tracking
"""

import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
from collections import defaultdict

class MetricsDashboard:
    """Comprehensive metrics dashboard for the reverse problem solver."""
    
    def __init__(self):
        self.metrics_history = []
        self.training_sessions = []
        self.prediction_sessions = []
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def load_all_data(self):
        """Load all historical data from sessions."""
        print("📊 Loading historical data...")
        
        # Load training sessions
        training_dirs = sorted(glob.glob("input/super_training_*"))
        for train_dir in training_dirs:
            metadata_file = f"{train_dir}/training_results.json"
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    data = json.load(f)
                    data['type'] = 'training'
                    data['timestamp'] = os.path.basename(train_dir).replace('super_training_', '')
                    self.training_sessions.append(data)
        
        # Load prediction sessions
        prediction_dirs = sorted(glob.glob("output/super_predictions_*"))
        for pred_dir in prediction_dirs:
            results_file = f"{pred_dir}/results.json"
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    data = json.load(f)
                    data['type'] = 'prediction'
                    data['timestamp'] = os.path.basename(pred_dir).replace('super_predictions_', '')
                    self.prediction_sessions.append(data)
        
        print(f"   Loaded {len(self.training_sessions)} training sessions")
        print(f"   Loaded {len(self.prediction_sessions)} prediction sessions")
    
    def create_comprehensive_dashboard(self):
        """Create the main dashboard with all key metrics."""
        self.load_all_data()
        
        # Create large dashboard figure
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 5, hspace=0.3, wspace=0.3)
        
        # Main title
        fig.suptitle('🚀 Reverse Problem Solver - Super Neural Network Dashboard', 
                    fontsize=24, fontweight='bold', y=0.98)
        
        # 1. Performance Evolution (2x2)
        self._plot_performance_evolution(fig, gs[0:2, 0:2])
        
        # 2. Neural Network Training Progress (1x2)
        self._plot_training_progress(fig, gs[0, 2:4])
        
        # 3. Accuracy by Wedge Count (1x2)
        self._plot_wedge_accuracy(fig, gs[1, 2:4])
        
        # 4. Current Performance Summary (2x1)
        self._plot_performance_summary(fig, gs[0:2, 4])
        
        # 5. System Timing Analysis (1x2)
        self._plot_timing_analysis(fig, gs[2, 0:2])
        
        # 6. Confidence Distribution (1x2)
        self._plot_confidence_distribution(fig, gs[2, 2:4])
        
        # 7. Training Data Quality (1x1)
        self._plot_data_quality(fig, gs[2, 4])
        
        # 8. Improvement Trajectory (1x2)
        self._plot_improvement_trajectory(fig, gs[3, 0:2])
        
        # 9. Model Architecture Summary (1x2)
        self._plot_architecture_info(fig, gs[3, 2:4])
        
        # 10. Next Steps & Recommendations (1x1)
        self._plot_recommendations(fig, gs[3, 4])
        
        return fig
    
    def _plot_performance_evolution(self, fig, gs):
        """Plot overall performance evolution over time."""
        ax = fig.add_subplot(gs)
        
        if not self.prediction_sessions:
            ax.text(0.5, 0.5, 'No prediction data available\nRun super_predict.py', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title('🎯 Performance Evolution Over Time')
            return
        
        # Extract performance data
        timestamps = []
        overall_acc = []
        nn_acc = []
        
        for session in self.prediction_sessions:
            timestamps.append(session['timestamp'])
            overall_acc.append(session['performance']['overall_accuracy'] * 100)
            nn_acc.append(session['performance'].get('nn_wedge_accuracy', 0) * 100)
        
        x = range(len(timestamps))
        
        # Plot lines
        ax.plot(x, overall_acc, 'o-', linewidth=3, markersize=8, 
               label='Hybrid System Accuracy', color='red')
        ax.plot(x, nn_acc, 's-', linewidth=3, markersize=8, 
               label='Neural Network Accuracy', color='blue')
        
        # Add baseline references
        ax.axhline(y=30, color='gray', linestyle='--', alpha=0.7, label='Old System (30%)')
        ax.axhline(y=28, color='gray', linestyle=':', alpha=0.7, label='Old NN (28%)')
        
        ax.set_xlabel('Session')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('🎯 Performance Evolution Over Time', fontsize=16, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
        
        # Add current performance annotation
        if overall_acc:
            latest_acc = overall_acc[-1]
            ax.annotate(f'Latest: {latest_acc:.1f}%', 
                       xy=(len(x)-1, latest_acc), xytext=(len(x)-1, latest_acc+10),
                       arrowprops=dict(arrowstyle='->', color='red'),
                       fontsize=12, fontweight='bold', color='red')
    
    def _plot_training_progress(self, fig, gs):
        """Plot neural network training progress."""
        ax = fig.add_subplot(gs)
        
        if not self.training_sessions:
            ax.text(0.5, 0.5, 'No training data available\nRun super_train.py', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('🧠 Neural Network Training Progress')
            return
        
        # Get latest training session
        latest_session = self.training_sessions[-1]
        results = latest_session.get('results', {})
        history = results.get('training_history', {})
        
        if 'train_losses' in history and 'val_losses' in history:
            epochs = range(1, len(history['train_losses']) + 1)
            
            ax.plot(epochs, history['train_losses'], 'b-', label='Training Loss', alpha=0.7)
            ax.plot(epochs, history['val_losses'], 'r-', label='Validation Loss', alpha=0.7)
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('🧠 Neural Network Training Progress', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
        else:
            ax.text(0.5, 0.5, 'No training history available', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def _plot_wedge_accuracy(self, fig, gs):
        """Plot accuracy by wedge count."""
        ax = fig.add_subplot(gs)
        
        if not self.prediction_sessions:
            ax.text(0.5, 0.5, 'No prediction data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('📊 Accuracy by Wedge Count')
            return
        
        # Get latest prediction session
        latest_session = self.prediction_sessions[-1]
        by_wedge = latest_session['performance']['by_wedge_count']
        
        wedge_counts = sorted([int(k) for k in by_wedge.keys()])
        accuracies = [by_wedge[str(w)]['accuracy'] * 100 for w in wedge_counts]
        sample_counts = [by_wedge[str(w)]['total'] for w in wedge_counts]
        
        # Color bars based on performance
        colors = ['red' if acc < 50 else 'orange' if acc < 70 else 'green' 
                 for acc in accuracies]
        
        bars = ax.bar(wedge_counts, accuracies, color=colors, alpha=0.7)
        
        # Add sample count labels
        for bar, acc, count in zip(bars, accuracies, sample_counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 2, 
                   f'{acc:.0f}%\n({count})', ha='center', va='bottom', fontsize=10)
        
        ax.set_xlabel('Wedge Count')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('📊 Accuracy by Wedge Count', fontweight='bold')
        ax.set_ylim(0, 110)
        ax.grid(True, alpha=0.3)
    
    def _plot_performance_summary(self, fig, gs):
        """Plot current performance summary."""
        ax = fig.add_subplot(gs)
        ax.axis('off')
        
        if not self.prediction_sessions:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
            return
        
        # Get latest metrics
        latest = self.prediction_sessions[-1]
        perf = latest['performance']
        timing = latest.get('timing', {})
        
        # Create performance summary
        summary = f"""
🏆 CURRENT PERFORMANCE

Overall Accuracy: {perf['overall_accuracy']:.1%}
NN Accuracy: {perf.get('nn_wedge_accuracy', 0):.1%}
Total Tested: {perf['total_tested']:,}
Correct: {perf['correct_predictions']:,}

⏱️ TIMING METRICS

Throughput: {timing.get('throughput_samples_per_sec', 0):.2f} samples/s
Avg Time/Sample: {1/timing.get('throughput_samples_per_sec', 1):.1f}s
NN Time %: {timing.get('nn_percentage', 0):.1f}%
GA Time %: {timing.get('ga_percentage', 100):.1f}%

🚀 IMPROVEMENTS

vs Old System: {((perf['overall_accuracy'] - 0.30) / 0.30 * 100):+.1f}%
vs Old NN: {((perf.get('nn_wedge_accuracy', 0) - 0.28) / 0.28 * 100):+.1f}%

📊 MODEL STATUS

Type: Super Neural Network
Architecture: Attention + Residual + Ensemble
Status: {'🟢 Excellent' if perf['overall_accuracy'] > 0.7 else '🟡 Good' if perf['overall_accuracy'] > 0.5 else '🔴 Needs Work'}
        """
        
        ax.text(0.05, 0.95, summary, transform=ax.transAxes, 
               fontsize=11, fontfamily='monospace', verticalalignment='top')
    
    def _plot_timing_analysis(self, fig, gs):
        """Plot system timing analysis."""
        ax = fig.add_subplot(gs)
        
        if not self.prediction_sessions:
            ax.text(0.5, 0.5, 'No timing data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('⏱️ System Timing Analysis')
            return
        
        # Get timing data from latest session
        latest = self.prediction_sessions[-1]
        timing = latest.get('timing', {})
        
        # Create pie chart of time distribution
        if 'nn_percentage' in timing and 'ga_percentage' in timing:
            sizes = [timing['nn_percentage'], timing['ga_percentage']]
            labels = ['Neural Network', 'Genetic Algorithm']
            colors = ['lightblue', 'lightcoral']
            
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, 
                                            autopct='%1.1f%%', startangle=90)
            
            ax.set_title('⏱️ System Timing Distribution', fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No timing breakdown available', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def _plot_confidence_distribution(self, fig, gs):
        """Plot neural network confidence distribution."""
        ax = fig.add_subplot(gs)
        
        if not self.prediction_sessions:
            ax.text(0.5, 0.5, 'No confidence data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('🎯 Prediction Confidence Distribution')
            return
        
        # Extract confidence scores from detailed results
        latest = self.prediction_sessions[-1]
        detailed = latest.get('detailed_results', [])
        
        confidences = []
        for result in detailed:
            if 'nn_confidence' in result:
                conf = result['nn_confidence'].get('wedge_count', 0)
                confidences.append(conf)
        
        if confidences:
            ax.hist(confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax.axvline(np.mean(confidences), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(confidences):.2f}')
            ax.set_xlabel('Confidence Score')
            ax.set_ylabel('Frequency')
            ax.set_title('🎯 Prediction Confidence Distribution', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No confidence scores available', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def _plot_data_quality(self, fig, gs):
        """Plot training data quality metrics."""
        ax = fig.add_subplot(gs)
        
        if not self.training_sessions:
            ax.text(0.5, 0.5, 'No training\ndata', ha='center', va='center')
            ax.set_title('📈 Data Quality')
            return
        
        # Get data distribution from latest training
        latest = self.training_sessions[-1]
        dist = latest.get('wedge_distribution', {})
        
        if dist:
            wedges = sorted([int(k) for k in dist.keys()])
            counts = [dist[str(w)] for w in wedges]
            
            ax.bar(wedges, counts, alpha=0.7, color='lightgreen')
            ax.set_xlabel('Wedge Count')
            ax.set_ylabel('Samples')
            ax.set_title('📈 Training Data\nDistribution', fontweight='bold')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No distribution\ndata', ha='center', va='center')
    
    def _plot_improvement_trajectory(self, fig, gs):
        """Plot improvement trajectory prediction."""
        ax = fig.add_subplot(gs)
        
        # Theoretical improvement curve
        samples = np.array([300, 1000, 5000, 10000, 20000, 50000])
        predicted_acc = np.array([55, 65, 75, 82, 87, 90])
        
        ax.plot(samples, predicted_acc, 'g--', linewidth=2, 
               label='Predicted Trajectory', alpha=0.7)
        
        # Add current performance points if available
        if self.training_sessions and self.prediction_sessions:
            current_samples = self.training_sessions[-1].get('total_samples', 0)
            current_acc = self.prediction_sessions[-1]['performance']['overall_accuracy'] * 100
            
            ax.plot(current_samples, current_acc, 'ro', markersize=10, 
                   label=f'Current ({current_samples:,} samples)')
        
        ax.set_xscale('log')
        ax.set_xlabel('Training Samples')
        ax.set_ylabel('Expected Accuracy (%)')
        ax.set_title('📈 Improvement Trajectory', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(40, 95)
    
    def _plot_architecture_info(self, fig, gs):
        """Plot model architecture information."""
        ax = fig.add_subplot(gs)
        ax.axis('off')
        
        arch_info = """
🏗️ SUPER NEURAL NETWORK ARCHITECTURE

🔹 Pattern Encoder:
  • Residual Blocks (512→256→128)
  • Batch Normalization
  • Dropout Regularization

🔹 Feature Extraction:
  • 28 Advanced Features
  • Statistical, Kinematic, Frequency
  • Shape & Complexity Analysis

🔹 Attention Mechanism:
  • Multi-head Self-Attention
  • 4 Attention Heads
  • Pattern Focus Enhancement

🔹 Multi-task Learning:
  • Wedge Classification (6 classes)
  • Parameter Regression (36 outputs)
  • Joint Optimization

🔹 Ensemble Prediction:
  • Multiple Model Voting
  • Confidence Scoring
  • Robust Predictions
        """
        
        ax.text(0.05, 0.95, arch_info, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top')
    
    def _plot_recommendations(self, fig, gs):
        """Plot next steps and recommendations."""
        ax = fig.add_subplot(gs)
        ax.axis('off')
        
        # Generate recommendations based on current performance
        recommendations = "🎯 RECOMMENDATIONS\n\n"
        
        if self.prediction_sessions:
            latest_acc = self.prediction_sessions[-1]['performance']['overall_accuracy']
            
            if latest_acc < 0.5:
                recommendations += "🔴 PRIORITY ACTIONS:\n"
                recommendations += "• Increase training data\n"
                recommendations += "• Check data quality\n"
                recommendations += "• Tune hyperparameters\n"
            elif latest_acc < 0.7:
                recommendations += "🟡 OPTIMIZATION:\n"
                recommendations += "• Add more training samples\n"
                recommendations += "• Try ensemble size = 5\n"
                recommendations += "• Longer training epochs\n"
            else:
                recommendations += "🟢 EXCELLENT PERFORMANCE:\n"
                recommendations += "• System is production-ready\n"
                recommendations += "• Consider deployment\n"
                recommendations += "• Monitor in production\n"
        
        recommendations += "\n📊 NEXT STEPS:\n"
        recommendations += "• Run super_train.py 10000\n"
        recommendations += "• Test on real data\n"
        recommendations += "• Monitor performance\n"
        
        ax.text(0.05, 0.95, recommendations, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top')
    
    def save_dashboard(self, filename="dashboard/current_metrics.png"):
        """Save the dashboard to file."""
        fig = self.create_comprehensive_dashboard()
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        fig.savefig(filename, dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        
        print(f"📊 Dashboard saved to: {filename}")
        return filename
    
    def show_dashboard(self):
        """Display the dashboard."""
        fig = self.create_comprehensive_dashboard()
        plt.show()
        return fig

def main():
    """Generate and display the metrics dashboard."""
    print("🚀 Generating Comprehensive Metrics Dashboard")
    print("=" * 60)
    
    dashboard = MetricsDashboard()
    
    # Save dashboard
    dashboard_file = dashboard.save_dashboard()
    
    print("✅ Dashboard generation complete!")
    print(f"📁 Saved to: {dashboard_file}")
    
    return dashboard

if __name__ == "__main__":
    main()