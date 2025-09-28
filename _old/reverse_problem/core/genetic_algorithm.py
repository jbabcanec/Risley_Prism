#!/usr/bin/env python3
"""
High-Performance Genetic Algorithm for Risley Prism Reverse Problem

Sophisticated evolutionary optimization with:
- Tournament selection
- Adaptive mutation
- Crossover operators
- Constraint handling
- Parallel evaluation
"""

import numpy as np
import random
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import time

@dataclass
class Individual:
    """Individual in the genetic algorithm population."""
    genes: Dict[str, np.ndarray]
    fitness: float = float('inf')
    age: int = 0
    constraint_violations: int = 0
    
    def __post_init__(self):
        """Ensure genes are numpy arrays."""
        for key, value in self.genes.items():
            if not isinstance(value, np.ndarray):
                self.genes[key] = np.array(value)

class GeneticAlgorithm:
    """High-performance genetic algorithm for reverse Risley prism problem."""
    
    def __init__(self, wedge_count: int, population_size: int = 100, 
                 elite_ratio: float = 0.1, mutation_rate: float = 0.15,
                 crossover_rate: float = 0.8, tournament_size: int = 3):
        """
        Initialize sophisticated GA.
        
        Args:
            wedge_count: Number of wedges in system
            population_size: Population size
            elite_ratio: Fraction of population to keep as elites
            mutation_rate: Base mutation probability
            crossover_rate: Crossover probability
            tournament_size: Tournament selection size
        """
        self.wedge_count = wedge_count
        self.population_size = population_size
        self.elite_size = max(1, int(population_size * elite_ratio))
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        
        # Parameter bounds
        self.bounds = {
            'rotation_speeds': (-5.0, 5.0),
            'phi_x': (-20.0, 20.0),
            'phi_y': (-20.0, 20.0),
            'distances': (1.0, 10.0),
            'refractive_indices': (1.4, 1.6)
        }
        
        # Adaptive parameters
        self.generation = 0
        self.best_fitness_history = []
        self.diversity_history = []
        self.stagnation_counter = 0
        
    def create_individual(self) -> Individual:
        """Create a random individual with proper constraints."""
        genes = {
            'rotation_speeds': np.random.uniform(*self.bounds['rotation_speeds'], self.wedge_count),
            'phi_x': np.random.uniform(*self.bounds['phi_x'], self.wedge_count),
            'phi_y': np.random.uniform(*self.bounds['phi_y'], self.wedge_count),
            'distances': np.concatenate([[1.0], np.random.uniform(*self.bounds['distances'], self.wedge_count)]),
            'refractive_indices': np.concatenate([[1.0], 
                                               np.random.uniform(*self.bounds['refractive_indices'], self.wedge_count),
                                               [1.0]]),
            'wedgenum': np.array([self.wedge_count])
        }
        
        return Individual(genes=genes)
    
    def initialize_population(self) -> List[Individual]:
        """Initialize population with diversity."""
        population = []
        
        # Create diverse initial population
        for i in range(self.population_size):
            individual = self.create_individual()
            
            # Add some structure to initial population
            if i < self.population_size // 4:
                # Quarter with small rotation speeds (simple patterns)
                individual.genes['rotation_speeds'] *= 0.3
            elif i < self.population_size // 2:
                # Quarter with large rotation speeds (complex patterns)
                individual.genes['rotation_speeds'] *= 2.0
            elif i < 3 * self.population_size // 4:
                # Quarter with small angles
                individual.genes['phi_x'] *= 0.5
                individual.genes['phi_y'] *= 0.5
            
            population.append(individual)
        
        return population
    
    def evaluate_fitness(self, individual: Individual, target_pattern: List[Tuple]) -> float:
        """
        Evaluate fitness of individual by simulating forward model.
        Lower fitness = better solution.
        """
        try:
            # Convert to solver format
            params = {
                'rotation_speeds': individual.genes['rotation_speeds'].tolist(),
                'phi_x': individual.genes['phi_x'].tolist(),
                'phi_y': individual.genes['phi_y'].tolist(),
                'distances': individual.genes['distances'].tolist(),
                'refractive_indices': individual.genes['refractive_indices'].tolist(),
                'wedgenum': int(individual.genes['wedgenum'][0])
            }
            
            # Simulate pattern (placeholder - replace with actual forward model)
            simulated_pattern = self._simulate_pattern(params)
            
            # Calculate pattern matching cost
            cost = self._pattern_distance(simulated_pattern, target_pattern)
            
            # Add constraint penalties
            penalty = self._constraint_penalty(individual)
            
            return cost + penalty
            
        except Exception as e:
            # Heavy penalty for invalid individuals
            return 1000.0
    
    def _simulate_pattern(self, params: Dict) -> np.ndarray:
        """Simplified pattern simulation (replace with actual forward model)."""
        # Generate pattern based on parameters
        wedge_count = params['wedgenum']
        rotation_speeds = params['rotation_speeds']
        phi_x = params['phi_x']
        phi_y = params['phi_y']
        
        # Time series
        t = np.linspace(0, 2.0, 60)
        
        # Simulate combined effect of all wedges
        x_pattern = np.zeros_like(t)
        y_pattern = np.zeros_like(t)
        
        for i in range(wedge_count):
            speed = rotation_speeds[i] if i < len(rotation_speeds) else 1.0
            px = phi_x[i] if i < len(phi_x) else 0.0
            py = phi_y[i] if i < len(phi_y) else 0.0
            
            phase = speed * t + i * 0.1
            amplitude_x = abs(px) / 20.0
            amplitude_y = abs(py) / 20.0
            
            x_pattern += amplitude_x * np.cos(phase) + amplitude_y * np.sin(phase * 1.2)
            y_pattern += amplitude_y * np.sin(phase) + amplitude_x * np.cos(phase * 0.8)
        
        return np.column_stack([x_pattern, y_pattern])
    
    def _pattern_distance(self, pattern1: np.ndarray, target_pattern: List[Tuple]) -> float:
        """Calculate distance between patterns."""
        # Convert target to array
        target_array = np.array([(p[0], p[1]) for p in target_pattern])
        
        # Resize to same length
        min_len = min(len(pattern1), len(target_array))
        p1 = pattern1[:min_len]
        p2 = target_array[:min_len]
        
        # Calculate RMS distance
        diff = p1 - p2
        rms_distance = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
        
        return rms_distance
    
    def _constraint_penalty(self, individual: Individual) -> float:
        """Calculate constraint violation penalty."""
        penalty = 0.0
        violations = 0
        
        # Check parameter bounds
        for param, bounds in self.bounds.items():
            if param in individual.genes:
                values = individual.genes[param]
                # Skip first/last elements for distances and refractive_indices (boundary conditions)
                if param in ['distances', 'refractive_indices']:
                    if param == 'distances':
                        check_values = values[1:]  # Skip first element (fixed at 1.0)
                    else:  # refractive_indices
                        check_values = values[1:-1]  # Skip first and last (fixed at 1.0)
                else:
                    check_values = values
                
                under = np.sum(check_values < bounds[0])
                over = np.sum(check_values > bounds[1])
                violations += under + over
                
                # Exponential penalty for constraint violations
                if under > 0:
                    penalty += under * 10.0 * abs(bounds[0] - np.min(check_values[check_values < bounds[0]]))
                if over > 0:
                    penalty += over * 10.0 * abs(np.max(check_values[check_values > bounds[1]]) - bounds[1])
        
        individual.constraint_violations = violations
        return penalty
    
    def tournament_selection(self, population: List[Individual]) -> Individual:
        """Tournament selection with diversity consideration."""
        tournament = random.sample(population, min(self.tournament_size, len(population)))
        
        # Primary selection by fitness
        tournament.sort(key=lambda x: x.fitness)
        
        # Add diversity pressure (avoid selecting very similar individuals)
        if len(tournament) > 1 and random.random() < 0.3:
            return tournament[1]  # Second best for diversity
        
        return tournament[0]  # Best fitness
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Advanced crossover with parameter-aware operators."""
        if random.random() > self.crossover_rate:
            return parent1, parent2
        
        child1_genes = {}
        child2_genes = {}
        
        for param in parent1.genes:
            p1_genes = parent1.genes[param].copy()
            p2_genes = parent2.genes[param].copy()
            
            if len(p1_genes.shape) == 0:  # Scalar
                # Simple swap for scalars
                if random.random() < 0.5:
                    child1_genes[param] = p2_genes.copy()
                    child2_genes[param] = p1_genes.copy()
                else:
                    child1_genes[param] = p1_genes.copy()
                    child2_genes[param] = p2_genes.copy()
            else:  # Array
                # Uniform crossover for arrays
                mask = np.random.random(p1_genes.shape) < 0.5
                
                child1_genes[param] = np.where(mask, p1_genes, p2_genes)
                child2_genes[param] = np.where(mask, p2_genes, p1_genes)
                
                # Blend crossover for 20% of elements
                blend_mask = np.random.random(p1_genes.shape) < 0.2
                alpha = 0.5
                
                if np.any(blend_mask):
                    blend1 = alpha * p1_genes + (1 - alpha) * p2_genes
                    blend2 = alpha * p2_genes + (1 - alpha) * p1_genes
                    
                    child1_genes[param] = np.where(blend_mask, blend1, child1_genes[param])
                    child2_genes[param] = np.where(blend_mask, blend2, child2_genes[param])
        
        return Individual(genes=child1_genes), Individual(genes=child2_genes)
    
    def mutate(self, individual: Individual) -> Individual:
        """Adaptive mutation with parameter-specific strategies."""
        mutation_strength = self._adaptive_mutation_strength()
        
        for param in individual.genes:
            if random.random() < self.mutation_rate:
                genes = individual.genes[param].copy()
                
                if len(genes.shape) == 0:  # Scalar
                    continue  # Don't mutate wedgenum
                
                # Parameter-specific mutation
                if param == 'rotation_speeds':
                    # Gaussian mutation for rotation speeds
                    mutation = np.random.normal(0, mutation_strength * 0.5, genes.shape)
                    genes += mutation
                elif param in ['phi_x', 'phi_y']:
                    # Gaussian mutation for angles
                    mutation = np.random.normal(0, mutation_strength * 2.0, genes.shape)
                    genes += mutation
                elif param == 'distances':
                    # Only mutate non-fixed distances
                    if len(genes) > 1:
                        mutation = np.random.normal(0, mutation_strength * 0.3, genes[1:].shape)
                        genes[1:] += mutation
                elif param == 'refractive_indices':
                    # Only mutate wedge refractive indices
                    if len(genes) > 2:
                        mutation = np.random.normal(0, mutation_strength * 0.02, genes[1:-1].shape)
                        genes[1:-1] += mutation
                
                # Apply bounds
                if param in self.bounds:
                    bounds = self.bounds[param]
                    if param in ['distances', 'refractive_indices']:
                        if param == 'distances' and len(genes) > 1:
                            genes[1:] = np.clip(genes[1:], bounds[0], bounds[1])
                        elif param == 'refractive_indices' and len(genes) > 2:
                            genes[1:-1] = np.clip(genes[1:-1], bounds[0], bounds[1])
                    else:
                        genes = np.clip(genes, bounds[0], bounds[1])
                
                individual.genes[param] = genes
        
        return individual
    
    def _adaptive_mutation_strength(self) -> float:
        """Adapt mutation strength based on population diversity and stagnation."""
        base_strength = 1.0
        
        # Increase mutation if population is stagnating
        if self.stagnation_counter > 10:
            base_strength *= 2.0
        elif self.stagnation_counter > 5:
            base_strength *= 1.5
        
        # Decrease mutation if population is too diverse
        if len(self.diversity_history) > 0 and self.diversity_history[-1] > 0.8:
            base_strength *= 0.7
        
        return base_strength
    
    def calculate_diversity(self, population: List[Individual]) -> float:
        """Calculate population diversity."""
        if len(population) < 2:
            return 0.0
        
        # Calculate average pairwise distance
        total_distance = 0.0
        comparisons = 0
        
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                distance = self._individual_distance(population[i], population[j])
                total_distance += distance
                comparisons += 1
        
        return total_distance / comparisons if comparisons > 0 else 0.0
    
    def _individual_distance(self, ind1: Individual, ind2: Individual) -> float:
        """Calculate distance between two individuals."""
        total_distance = 0.0
        
        for param in ind1.genes:
            if param == 'wedgenum':
                continue
            
            genes1 = ind1.genes[param].flatten()
            genes2 = ind2.genes[param].flatten()
            
            # Normalize by parameter range
            if param in self.bounds:
                bounds = self.bounds[param]
                param_range = bounds[1] - bounds[0]
                distance = np.mean(np.abs(genes1 - genes2)) / param_range
            else:
                distance = np.mean(np.abs(genes1 - genes2))
            
            total_distance += distance
        
        return total_distance
    
    def evolve_generation(self, population: List[Individual], target_pattern: List[Tuple]) -> List[Individual]:
        """Evolve one generation."""
        # Evaluate fitness for new individuals
        for individual in population:
            if individual.fitness == float('inf'):
                individual.fitness = self.evaluate_fitness(individual, target_pattern)
        
        # Sort by fitness
        population.sort(key=lambda x: x.fitness)
        
        # Track best fitness and diversity
        best_fitness = population[0].fitness
        diversity = self.calculate_diversity(population)
        
        self.best_fitness_history.append(best_fitness)
        self.diversity_history.append(diversity)
        
        # Check for stagnation
        if len(self.best_fitness_history) > 1:
            if abs(self.best_fitness_history[-1] - self.best_fitness_history[-2]) < 1e-6:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0
        
        # Create next generation
        next_generation = []
        
        # Elitism - keep best individuals
        elites = population[:self.elite_size]
        next_generation.extend([Individual(genes={k: v.copy() for k, v in elite.genes.items()}, 
                                         fitness=elite.fitness) for elite in elites])
        
        # Generate offspring
        while len(next_generation) < self.population_size:
            parent1 = self.tournament_selection(population)
            parent2 = self.tournament_selection(population)
            
            child1, child2 = self.crossover(parent1, parent2)
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            
            # Reset fitness for evaluation
            child1.fitness = float('inf')
            child2.fitness = float('inf')
            
            next_generation.extend([child1, child2])
        
        # Trim to population size
        next_generation = next_generation[:self.population_size]
        
        self.generation += 1
        
        return next_generation

def solve_reverse_problem(target_pattern: List[Tuple], wedge_count: int, 
                         population_size: int = 100, generations: int = 50,
                         parallel: bool = False, verbose: bool = False) -> Tuple[Dict, float, Dict]:
    """
    Solve reverse problem using sophisticated genetic algorithm.
    
    Args:
        target_pattern: List of (x, y, t) tuples
        wedge_count: Number of wedges
        population_size: GA population size
        generations: Number of generations
        parallel: Use parallel evaluation (not implemented yet)
        verbose: Print progress
        
    Returns:
        best_params: Dict with optimized parameters
        best_cost: Float cost of best solution
        info: Dict with optimization info
    """
    
    # Create and run GA
    ga = GeneticAlgorithm(wedge_count, population_size)
    population = ga.initialize_population()
    
    best_fitness = float('inf')
    best_individual = None
    
    for gen in range(generations):
        population = ga.evolve_generation(population, target_pattern)
        
        current_best = min(population, key=lambda x: x.fitness)
        if current_best.fitness < best_fitness:
            best_fitness = current_best.fitness
            best_individual = current_best
        
        if verbose and gen % 10 == 0:
            diversity = ga.diversity_history[-1] if ga.diversity_history else 0.0
            print(f"Gen {gen}: Best={best_fitness:.4f}, Diversity={diversity:.3f}")
    
    # Convert best individual to result format
    if best_individual:
        best_params = {
            'rotation_speeds': best_individual.genes['rotation_speeds'].tolist(),
            'phi_x': best_individual.genes['phi_x'].tolist(),
            'phi_y': best_individual.genes['phi_y'].tolist(),
            'distances': best_individual.genes['distances'].tolist(),
            'refractive_indices': best_individual.genes['refractive_indices'].tolist(),
            'wedgenum': int(best_individual.genes['wedgenum'][0])
        }
    else:
        # Fallback
        best_params = {
            'rotation_speeds': [0.0] * wedge_count,
            'phi_x': [0.0] * wedge_count,
            'phi_y': [0.0] * wedge_count,
            'distances': [1.0] + [2.0] * wedge_count,
            'refractive_indices': [1.0] + [1.5] * wedge_count + [1.0],
            'wedgenum': wedge_count
        }
    
    info = {
        'generations': generations,
        'population_size': population_size,
        'final_diversity': ga.diversity_history[-1] if ga.diversity_history else 0.0,
        'stagnation_count': ga.stagnation_counter,
        'best_fitness_history': ga.best_fitness_history,
        'converged': ga.stagnation_counter < 5
    }
    
    return best_params, best_fitness, info