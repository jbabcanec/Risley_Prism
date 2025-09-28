"""
Core algorithms for Risley Prism Reverse Problem

High-performance optimization algorithms with sophisticated evolutionary strategies,
advanced constraint handling, and comprehensive performance monitoring.
"""

from .genetic_algorithm import solve_reverse_problem, GeneticAlgorithm, Individual
from .constraints import PhysicsConstraints, ConstraintViolation
from .performance import OptimizationMetrics, AdaptiveParameters, PerformanceProfiler
from .neural_network import NeuralPredictor, PatternToParameterNet, TrainingConfig

__all__ = [
    'solve_reverse_problem', 'GeneticAlgorithm', 'Individual',
    'PhysicsConstraints', 'ConstraintViolation',
    'OptimizationMetrics', 'AdaptiveParameters', 'PerformanceProfiler',
    'NeuralPredictor', 'PatternToParameterNet', 'TrainingConfig'
]