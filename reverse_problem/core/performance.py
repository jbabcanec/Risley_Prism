#!/usr/bin/env python3
"""
Performance Monitoring and Optimization Metrics

Advanced analytics for tracking optimization performance and convergence.
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque

@dataclass
class OptimizationMetrics:
    """Comprehensive optimization performance metrics."""
    start_time: float = field(default_factory=time.time)
    generation_times: List[float] = field(default_factory=list)
    best_fitness_history: List[float] = field(default_factory=list)
    mean_fitness_history: List[float] = field(default_factory=list)
    diversity_history: List[float] = field(default_factory=list)
    convergence_history: List[float] = field(default_factory=list)
    constraint_violations: List[int] = field(default_factory=list)
    evaluations_per_second: List[float] = field(default_factory=list)
    
    # Convergence detection
    stagnation_window: int = 10
    convergence_threshold: float = 1e-6
    
    def update(self, generation: int, population_fitness: List[float], 
               diversity: float, violations: int, evaluations: int):
        """Update metrics for current generation."""
        current_time = time.time()
        
        if len(self.generation_times) > 0:
            gen_time = current_time - self.generation_times[-1]
        else:
            gen_time = current_time - self.start_time
        
        self.generation_times.append(current_time)
        
        # Fitness statistics
        best_fitness = min(population_fitness)
        mean_fitness = np.mean(population_fitness)
        
        self.best_fitness_history.append(best_fitness)
        self.mean_fitness_history.append(mean_fitness)
        self.diversity_history.append(diversity)
        self.constraint_violations.append(violations)
        
        # Performance metrics
        if gen_time > 0:
            eval_rate = evaluations / gen_time
            self.evaluations_per_second.append(eval_rate)
        
        # Convergence detection
        convergence_metric = self._calculate_convergence_metric()
        self.convergence_history.append(convergence_metric)
    
    def _calculate_convergence_metric(self) -> float:
        """Calculate convergence metric based on fitness improvement rate."""
        if len(self.best_fitness_history) < self.stagnation_window:
            return 1.0  # Not enough data
        
        recent_fitness = self.best_fitness_history[-self.stagnation_window:]
        improvement_rate = (recent_fitness[0] - recent_fitness[-1]) / self.stagnation_window
        
        return max(0.0, improvement_rate)
    
    def is_converged(self) -> bool:
        """Check if optimization has converged."""
        if len(self.convergence_history) < self.stagnation_window:
            return False
        
        recent_convergence = self.convergence_history[-self.stagnation_window:]
        return all(c < self.convergence_threshold for c in recent_convergence)
    
    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary."""
        total_time = time.time() - self.start_time
        total_generations = len(self.best_fitness_history)
        
        summary = {
            'total_time': total_time,
            'total_generations': total_generations,
            'generations_per_second': total_generations / total_time if total_time > 0 else 0,
            'best_fitness': min(self.best_fitness_history) if self.best_fitness_history else float('inf'),
            'final_fitness': self.best_fitness_history[-1] if self.best_fitness_history else float('inf'),
            'total_improvement': (self.best_fitness_history[0] - self.best_fitness_history[-1]) if len(self.best_fitness_history) > 1 else 0,
            'convergence_rate': np.mean(self.convergence_history) if self.convergence_history else 0,
            'final_diversity': self.diversity_history[-1] if self.diversity_history else 0,
            'mean_evaluations_per_second': np.mean(self.evaluations_per_second) if self.evaluations_per_second else 0,
            'total_constraint_violations': sum(self.constraint_violations),
            'converged': self.is_converged()
        }
        
        return summary

class AdaptiveParameters:
    """Adaptive parameter management for optimization algorithms."""
    
    def __init__(self, initial_params: Dict):
        self.initial_params = initial_params.copy()
        self.current_params = initial_params.copy()
        self.adaptation_history = []
        
        # Adaptation strategies
        self.strategies = {
            'mutation_rate': self._adapt_mutation_rate,
            'crossover_rate': self._adapt_crossover_rate,
            'population_size': self._adapt_population_size,
            'selection_pressure': self._adapt_selection_pressure
        }
        
        # Performance tracking
        self.performance_window = deque(maxlen=10)
        self.diversity_window = deque(maxlen=10)
    
    def update_performance(self, metrics: OptimizationMetrics):
        """Update adaptation based on current performance metrics."""
        if len(metrics.best_fitness_history) > 0:
            self.performance_window.append(metrics.best_fitness_history[-1])
        
        if len(metrics.diversity_history) > 0:
            self.diversity_window.append(metrics.diversity_history[-1])
        
        # Apply adaptation strategies
        for param_name, strategy in self.strategies.items():
            if param_name in self.current_params:
                old_value = self.current_params[param_name]
                new_value = strategy(metrics)
                
                if new_value != old_value:
                    self.current_params[param_name] = new_value
                    self.adaptation_history.append({
                        'generation': len(metrics.best_fitness_history),
                        'parameter': param_name,
                        'old_value': old_value,
                        'new_value': new_value,
                        'reason': self._get_adaptation_reason(param_name, metrics)
                    })
    
    def _adapt_mutation_rate(self, metrics: OptimizationMetrics) -> float:
        """Adapt mutation rate based on convergence and diversity."""
        current_rate = self.current_params['mutation_rate']
        
        # Increase mutation if stagnating
        if metrics.is_converged():
            return min(current_rate * 1.5, 0.5)
        
        # Decrease mutation if too much diversity
        if len(self.diversity_window) > 0 and np.mean(self.diversity_window) > 0.8:
            return max(current_rate * 0.8, 0.01)
        
        # Increase mutation if low diversity
        if len(self.diversity_window) > 0 and np.mean(self.diversity_window) < 0.2:
            return min(current_rate * 1.2, 0.5)
        
        return current_rate
    
    def _adapt_crossover_rate(self, metrics: OptimizationMetrics) -> float:
        """Adapt crossover rate based on population performance."""
        current_rate = self.current_params['crossover_rate']
        
        # Increase crossover if good diversity
        if len(self.diversity_window) > 0 and np.mean(self.diversity_window) > 0.6:
            return min(current_rate * 1.1, 0.95)
        
        # Decrease crossover if low diversity
        if len(self.diversity_window) > 0 and np.mean(self.diversity_window) < 0.3:
            return max(current_rate * 0.9, 0.5)
        
        return current_rate
    
    def _adapt_population_size(self, metrics: OptimizationMetrics) -> int:
        """Adapt population size based on problem complexity."""
        current_size = self.current_params['population_size']
        
        # This is typically fixed during run, but could be adapted for restart strategies
        return current_size
    
    def _adapt_selection_pressure(self, metrics: OptimizationMetrics) -> float:
        """Adapt selection pressure based on convergence."""
        current_pressure = self.current_params.get('selection_pressure', 1.0)
        
        # Increase selection pressure if converging too slowly
        if len(self.performance_window) >= 5:
            recent_improvement = self.performance_window[0] - self.performance_window[-1]
            if recent_improvement < 0.01:
                return min(current_pressure * 1.1, 2.0)
        
        return current_pressure
    
    def _get_adaptation_reason(self, param_name: str, metrics: OptimizationMetrics) -> str:
        """Get human-readable reason for parameter adaptation."""
        if param_name == 'mutation_rate':
            if metrics.is_converged():
                return "Increased due to convergence stagnation"
            elif len(self.diversity_window) > 0:
                diversity = np.mean(self.diversity_window)
                if diversity > 0.8:
                    return "Decreased due to high diversity"
                elif diversity < 0.2:
                    return "Increased due to low diversity"
        
        return "Adaptive adjustment based on performance metrics"

class PerformanceProfiler:
    """Detailed performance profiling for optimization algorithms."""
    
    def __init__(self):
        self.operation_times = {}
        self.operation_counts = {}
        self.memory_usage = []
        self.start_times = {}
    
    def start_operation(self, operation_name: str):
        """Start timing an operation."""
        self.start_times[operation_name] = time.time()
    
    def end_operation(self, operation_name: str):
        """End timing an operation."""
        if operation_name in self.start_times:
            elapsed = time.time() - self.start_times[operation_name]
            
            if operation_name not in self.operation_times:
                self.operation_times[operation_name] = []
                self.operation_counts[operation_name] = 0
            
            self.operation_times[operation_name].append(elapsed)
            self.operation_counts[operation_name] += 1
            
            del self.start_times[operation_name]
    
    def get_profile_summary(self) -> Dict:
        """Get detailed profiling summary."""
        summary = {}
        
        for operation, times in self.operation_times.items():
            summary[operation] = {
                'total_time': sum(times),
                'average_time': np.mean(times),
                'min_time': min(times),
                'max_time': max(times),
                'std_time': np.std(times),
                'call_count': self.operation_counts[operation],
                'time_percentage': 0.0  # Will be calculated below
            }
        
        # Calculate time percentages
        total_time = sum(summary[op]['total_time'] for op in summary)
        if total_time > 0:
            for operation in summary:
                summary[operation]['time_percentage'] = (summary[operation]['total_time'] / total_time) * 100
        
        return summary
    
    def get_bottlenecks(self, top_n: int = 5) -> List[Tuple[str, float]]:
        """Identify performance bottlenecks."""
        summary = self.get_profile_summary()
        
        # Sort by total time
        bottlenecks = [(op, data['total_time']) for op, data in summary.items()]
        bottlenecks.sort(key=lambda x: x[1], reverse=True)
        
        return bottlenecks[:top_n]