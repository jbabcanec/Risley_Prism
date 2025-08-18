#!/usr/bin/env python3
"""
Clean Performance Dashboard
"""

import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime

class CleanDashboard:
    """Clean, professional dashboard for metrics visualization."""
    
    def __init__(self):
        self.training_data = []
        self.prediction_data = []
        
        # Professional color scheme
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'success': '#77B28C',
            'warning': '#F18F01',
            'info': '#C73E1D',
            'dark': '#2D3142',
            'light': '#F5F5F5'
        }
        
        # Set clean style
        plt.style.use('seaborn-v0_8-whitegrid')
    
    def load_data(self):
        """Load all available data."""
        
        # Load training sessions
        for path in glob.glob("input/training_*") + glob.glob("input/super_training_*"):
            metadata_file = os.path.join(path, "metadata.json")
            training_file = os.path.join(path, "training_results.json")
            
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    self.training_data.append(json.load(f))
            elif os.path.exists(training_file):
                with open(training_file, 'r') as f:
                    self.training_data.append(json.load(f))
        
        # Load prediction sessions
        for path in glob.glob("output/predictions_*") + glob.glob("output/super_predictions_*"):
            results_file = os.path.join(path, "results.json")
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    self.prediction_data.append(json.load(f))
    
    def create_dashboard(self, output_file='dashboard/performance_dashboard.png'):
        """Create the main performance dashboard."""
        self.load_data()
        
        # Create figure
        fig = plt.figure(figsize=(18, 12))
        fig.patch.set_facecolor('white')
        
        # Create grid
        gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3,
                              left=0.06, right=0.96, top=0.93, bottom=0.07)
        
        # Title
        fig.text(0.5, 0.97, 'NEURAL NETWORK PERFORMANCE DASHBOARD', 
                fontsize=16, fontweight='bold', ha='center')
        
        # Subtitle with timestamp
        fig.text(0.5, 0.95, f'Analysis Date: {datetime.now().strftime("%Y-%m-%d %H:%M")}',
                fontsize=10, ha='center', color='gray')
        
        # 1. Overall Accuracy Trend (2x2)
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        self._plot_accuracy_trend(ax1)
        
        # 2. Wedge Count Performance (1x2)
        ax2 = fig.add_subplot(gs[0, 2:4])
        self._plot_wedge_performance(ax2)
        
        # 3. Model Metrics (1x2)
        ax3 = fig.add_subplot(gs[1, 2:4])
        self._plot_model_metrics(ax3)
        
        # 4. Training Progress (1x2)
        ax4 = fig.add_subplot(gs[2, 0:2])
        self._plot_training_progress(ax4)
        
        # 5. System Performance (1x2)
        ax5 = fig.add_subplot(gs[2, 2:4])
        self._plot_system_performance(ax5)
        
        # Save
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Dashboard saved: {output_file}")
        
        return fig
    
    def _plot_accuracy_trend(self, ax):
        """Plot accuracy evolution over time."""
        ax.set_title('ACCURACY EVOLUTION', fontsize=12, fontweight='bold', pad=15)
        
        if not self.prediction_data:
            ax.text(0.5, 0.5, 'No Data Available', ha='center', va='center')
            ax.set_xticks([])
            ax.set_yticks([])
            return
        
        # Extract accuracy data
        sessions = []
        accuracies = []
        nn_accuracies = []
        
        for data in self.prediction_data:
            perf = data.get('performance', {})
            sessions.append(len(sessions) + 1)
            accuracies.append(perf.get('overall_accuracy', 0) * 100)
            nn_accuracies.append(perf.get('nn_accuracy', perf.get('nn_wedge_accuracy', 0)) * 100)
        
        # Plot lines
        ax.plot(sessions, accuracies, 'o-', color=self.colors['primary'], 
                linewidth=2.5, markersize=8, label='Overall System')
        ax.plot(sessions, nn_accuracies, 's--', color=self.colors['secondary'], 
                linewidth=2, markersize=6, label='Neural Network')
        
        # Baseline
        ax.axhline(y=30, color='gray', linestyle=':', alpha=0.5, label='Baseline')
        
        # Target
        ax.axhline(y=85, color=self.colors['success'], linestyle='--', 
                  alpha=0.5, label='Target')
        
        # Labels
        ax.set_xlabel('Session', fontsize=10)
        ax.set_ylabel('Accuracy (%)', fontsize=10)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.2)
        ax.legend(loc='upper left', fontsize=9)
        
        # Add latest value annotation
        if accuracies:
            latest = accuracies[-1]
            ax.annotate(f'{latest:.1f}%', 
                       xy=(sessions[-1], latest),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=10, fontweight='bold',
                       color=self.colors['primary'])
    
    def _plot_wedge_performance(self, ax):
        """Plot performance by wedge count."""
        ax.set_title('ACCURACY BY WEDGE COUNT', fontsize=12, fontweight='bold', pad=15)
        
        if not self.prediction_data:
            ax.text(0.5, 0.5, 'No Data Available', ha='center', va='center')
            ax.set_xticks([])
            ax.set_yticks([])
            return
        
        # Get latest prediction data
        latest = self.prediction_data[-1]
        by_wedge = latest.get('performance', {}).get('by_wedge_count', {})
        
        if not by_wedge:
            ax.text(0.5, 0.5, 'No Wedge Data', ha='center', va='center')
            return
        
        wedges = []
        accuracies = []
        counts = []
        
        for w in range(1, 7):
            w_str = str(w)
            if w_str in by_wedge:
                wedges.append(w)
                acc = by_wedge[w_str].get('accuracy', 0) * 100
                accuracies.append(acc)
                counts.append(by_wedge[w_str].get('total', 0))
        
        # Create bars
        bars = ax.bar(wedges, accuracies, color=self.colors['primary'], alpha=0.7)
        
        # Color bars by performance
        for i, (bar, acc) in enumerate(zip(bars, accuracies)):
            if acc >= 80:
                bar.set_color(self.colors['success'])
            elif acc >= 50:
                bar.set_color(self.colors['primary'])
            else:
                bar.set_color(self.colors['warning'])
            
            # Add value on top
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{acc:.0f}%', ha='center', fontsize=9)
        
        ax.set_xlabel('Wedge Count', fontsize=10)
        ax.set_ylabel('Accuracy (%)', fontsize=10)
        ax.set_xticks(wedges)
        ax.set_ylim(0, 110)
        ax.grid(True, axis='y', alpha=0.2)
    
    def _plot_model_metrics(self, ax):
        """Plot key model metrics."""
        ax.set_title('MODEL METRICS', fontsize=12, fontweight='bold', pad=15)
        
        if not self.prediction_data:
            ax.text(0.5, 0.5, 'No Data Available', ha='center', va='center')
            ax.set_xticks([])
            ax.set_yticks([])
            return
        
        latest = self.prediction_data[-1]
        perf = latest.get('performance', {})
        timing = latest.get('timing', {})
        
        # Metrics to display
        metrics = {
            'Overall\nAccuracy': perf.get('overall_accuracy', 0) * 100,
            'NN\nAccuracy': perf.get('nn_accuracy', 0) * 100,
            'Throughput\n(samples/s)': timing.get('throughput', 0),
            'Cache Hit\nRate': latest.get('cache', {}).get('rate', 0) * 100
        }
        
        x_pos = np.arange(len(metrics))
        values = list(metrics.values())
        
        # Normalize for display (different scales)
        display_values = []
        labels = []
        for key, val in metrics.items():
            if 'Throughput' in key:
                display_values.append(min(val / 2, 100))  # Scale throughput
                labels.append(f'{key}\n{val:.1f}')
            else:
                display_values.append(val)
                labels.append(f'{key}\n{val:.1f}%')
        
        bars = ax.bar(x_pos, display_values, color=self.colors['info'], alpha=0.7)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0, 110)
        ax.set_ylabel('Performance', fontsize=10)
        ax.grid(True, axis='y', alpha=0.2)
    
    def _plot_training_progress(self, ax):
        """Plot training history."""
        ax.set_title('TRAINING HISTORY', fontsize=12, fontweight='bold', pad=15)
        
        if not self.training_data:
            ax.text(0.5, 0.5, 'No Training Data', ha='center', va='center')
            ax.set_xticks([])
            ax.set_yticks([])
            return
        
        sessions = []
        val_accs = []
        train_times = []
        
        for i, data in enumerate(self.training_data):
            sessions.append(i + 1)
            
            # Try different keys for accuracy
            val_acc = 0
            if 'results' in data and 'best_val_acc' in data['results']:
                val_acc = data['results']['best_val_acc'] * 100
            elif 'best_val_accuracy' in data:
                val_acc = data['best_val_accuracy'] * 100
            val_accs.append(val_acc)
            
            train_times.append(data.get('training_time', 0))
        
        # Dual axis plot
        ax2 = ax.twinx()
        
        # Accuracy line
        line1 = ax.plot(sessions, val_accs, 'o-', color=self.colors['primary'],
                       linewidth=2, markersize=7, label='Validation Accuracy')
        
        # Training time bars
        bars = ax2.bar(sessions, train_times, alpha=0.3, color=self.colors['warning'],
                      label='Training Time')
        
        ax.set_xlabel('Training Session', fontsize=10)
        ax.set_ylabel('Validation Accuracy (%)', fontsize=10, color=self.colors['primary'])
        ax2.set_ylabel('Training Time (s)', fontsize=10, color=self.colors['warning'])
        
        ax.tick_params(axis='y', labelcolor=self.colors['primary'])
        ax2.tick_params(axis='y', labelcolor=self.colors['warning'])
        
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.2)
        
        # Combined legend
        lines = line1 + [bars]
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left', fontsize=9)
    
    def _plot_system_performance(self, ax):
        """Plot system performance metrics."""
        ax.set_title('SYSTEM PERFORMANCE', fontsize=12, fontweight='bold', pad=15)
        
        if not self.prediction_data:
            ax.text(0.5, 0.5, 'No Data Available', ha='center', va='center')
            ax.set_xticks([])
            ax.set_yticks([])
            return
        
        latest = self.prediction_data[-1]
        timing = latest.get('timing', {})
        
        # Performance breakdown
        nn_time = timing.get('nn_avg_ms', 0)
        ga_time = timing.get('ga_avg_ms', 0)
        total_time = nn_time + ga_time
        
        if total_time > 0:
            sizes = [nn_time/total_time * 100, ga_time/total_time * 100]
            labels = [f'Neural Network\n{nn_time:.1f}ms ({sizes[0]:.1f}%)',
                     f'Optimization\n{ga_time:.1f}ms ({sizes[1]:.1f}%)']
            colors = [self.colors['primary'], self.colors['secondary']]
            
            wedgeprops = {'edgecolor': 'white', 'linewidth': 2}
            ax.pie(sizes, labels=labels, colors=colors, autopct='',
                  startangle=90, wedgeprops=wedgeprops)
            
            # Add center text
            ax.text(0, 0, f'Total\n{total_time:.1f}ms', 
                   ha='center', va='center', fontsize=11, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No Timing Data', ha='center', va='center')
    
    def generate_report(self, output_file='dashboard/report.txt'):
        """Generate text report of current performance."""
        self.load_data()
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w') as f:
            f.write("NEURAL NETWORK PERFORMANCE REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            if self.prediction_data:
                latest = self.prediction_data[-1]
                perf = latest.get('performance', {})
                
                f.write("LATEST RESULTS\n")
                f.write("-" * 40 + "\n")
                f.write(f"Overall Accuracy: {perf.get('overall_accuracy', 0)*100:.1f}%\n")
                f.write(f"Neural Network Accuracy: {perf.get('nn_accuracy', 0)*100:.1f}%\n")
                f.write(f"Samples Tested: {perf.get('total_tested', 0)}\n")
                f.write(f"Correct Predictions: {perf.get('correct_predictions', 0)}\n\n")
                
                f.write("PERFORMANCE BY WEDGE COUNT\n")
                f.write("-" * 40 + "\n")
                by_wedge = perf.get('by_wedge_count', {})
                for w in range(1, 7):
                    if str(w) in by_wedge:
                        data = by_wedge[str(w)]
                        acc = data.get('accuracy', 0) * 100
                        f.write(f"{w} wedges: {acc:6.1f}% ({data.get('correct')}/{data.get('total')})\n")
            
            if self.training_data:
                f.write("\nTRAINING SUMMARY\n")
                f.write("-" * 40 + "\n")
                f.write(f"Total Training Sessions: {len(self.training_data)}\n")
                
                latest_train = self.training_data[-1]
                f.write(f"Latest Training:\n")
                f.write(f"  Samples: {latest_train.get('total_samples', 0)}\n")
                f.write(f"  Time: {latest_train.get('training_time', 0):.1f}s\n")
                
                if 'results' in latest_train:
                    results = latest_train['results']
                    f.write(f"  Best Validation Accuracy: {results.get('best_val_acc', 0)*100:.1f}%\n")
            
            f.write("\n" + "=" * 60 + "\n")
            f.write("END OF REPORT\n")
        
        print(f"Report saved: {output_file}")

def create_clean_dashboard():
    """Convenience function to create dashboard."""
    dashboard = CleanDashboard()
    dashboard.create_dashboard()
    dashboard.generate_report()
    return dashboard

if __name__ == "__main__":
    create_clean_dashboard()