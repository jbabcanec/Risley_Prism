#!/usr/bin/env python3
"""
Genetic Algorithm for Parameter Refinement
Uses REAL forward model for fitness evaluation - NO FAKE PHYSICS.
"""

import numpy as np
from typing import Dict, List, Tuple, Callable
from dataclasses import dataclass
import sys
import os

# Add parent for model access
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class GAConfig:
    """Configuration for genetic algorithm."""
    population_size: int = 50
    generations: int = 100
    elite_size: int = 5
    mutation_rate: float = 0.1
    mutation_scale: float = 0.2
    crossover_rate: float = 0.7
    tournament_size: int = 3


class GeneticAlgorithm:
    """
    Genetic algorithm for refining Risley prism parameters.
    Uses REAL physics for evaluation.
    """

    def __init__(self, config: GAConfig, forward_model: Callable):
        """
        Initialize GA with configuration and forward model.

        Args:
            config: GA configuration
            forward_model: Function that simulates pattern from parameters
        """
        self.config = config
        self.forward_model = forward_model

    def create_individual(self, wedge_count: int,
                          initial_params: Dict = None) -> Dict:
        """
        Create a random individual or mutate from initial parameters.

        Args:
            wedge_count: Number of wedges
            initial_params: Optional initial parameters to start from
        """
        if initial_params:
            # Start from initial guess with small perturbation
            speeds = np.array(initial_params['rotation_speeds'])
            phi_x = np.array(initial_params['phi_x'])
            phi_y = np.array(initial_params['phi_y'])

            # Add small random perturbation
            speeds += np.random.randn(wedge_count) * 0.5
            phi_x += np.random.randn(wedge_count) * 5.0
            phi_y += np.random.randn(wedge_count) * 5.0
        else:
            # Random initialization
            speeds = np.random.uniform(-5, 5, wedge_count)
            phi_x = np.random.uniform(-30, 30, wedge_count)
            phi_y = np.random.uniform(-30, 30, wedge_count)

        # Apply physical constraints
        speeds = np.clip(speeds, -10, 10)
        phi_x = np.clip(phi_x, -45, 45)
        phi_y = np.clip(phi_y, -45, 45)

        return {
            'wedge_count': wedge_count,
            'rotation_speeds': speeds.tolist(),
            'phi_x': phi_x.tolist(),
            'phi_y': phi_y.tolist()
        }

    def evaluate_fitness(self, individual: Dict,
                         target_pattern: np.ndarray) -> float:
        """
        Evaluate fitness using REAL forward model.

        Lower is better (minimizing error).

        Args:
            individual: Parameters to evaluate
            target_pattern: Target pattern to match

        Returns:
            fitness: Mean squared error between patterns
        """
        try:
            # Import here to ensure we use the real model
            from core import RisleyParameters, ForwardModel

            # Create parameters object
            params = RisleyParameters(
                wedge_count=individual['wedge_count'],
                rotation_speeds=individual['rotation_speeds'],
                phi_x=individual['phi_x'],
                phi_y=individual['phi_y']
            )

            # Validate physical constraints
            if not params.validate():
                return float('inf')  # Invalid parameters

            # Simulate using REAL physics
            simulated = self.forward_model.simulate(params, len(target_pattern))

            # Ensure same length
            min_len = min(len(simulated), len(target_pattern))
            simulated = simulated[:min_len]
            target = target_pattern[:min_len]

            # Calculate pattern error
            mse = np.mean((simulated - target) ** 2)

            # Add penalty for extreme parameters (soft constraint)
            speed_penalty = np.sum(np.abs(params.rotation_speeds)) * 0.01
            angle_penalty = (np.sum(np.abs(params.phi_x)) +
                           np.sum(np.abs(params.phi_y))) * 0.001

            fitness = mse + speed_penalty + angle_penalty

            return fitness

        except Exception as e:
            # If simulation fails, return worst fitness
            print(f"Simulation failed: {e}")
            return float('inf')

    def crossover(self, parent1: Dict, parent2: Dict) -> Tuple[Dict, Dict]:
        """
        Uniform crossover between two parents.

        Physical constraint: maintain consistency in wedge count.
        """
        child1 = parent1.copy()
        child2 = parent2.copy()

        if np.random.rand() < self.config.crossover_rate:
            # Uniform crossover for each parameter
            for key in ['rotation_speeds', 'phi_x', 'phi_y']:
                mask = np.random.rand(len(parent1[key])) > 0.5

                child1_vals = np.array(parent1[key])
                child2_vals = np.array(parent2[key])

                temp = child1_vals[mask].copy()
                child1_vals[mask] = child2_vals[mask]
                child2_vals[mask] = temp

                child1[key] = child1_vals.tolist()
                child2[key] = child2_vals.tolist()

        return child1, child2

    def mutate(self, individual: Dict) -> Dict:
        """
        Mutate individual with physical constraints.

        Mutations respect physical limits and continuity.
        """
        mutated = individual.copy()

        # Mutate rotation speeds
        if np.random.rand() < self.config.mutation_rate:
            speeds = np.array(mutated['rotation_speeds'])
            mutation = np.random.randn(len(speeds)) * self.config.mutation_scale * 2
            speeds += mutation
            speeds = np.clip(speeds, -10, 10)
            mutated['rotation_speeds'] = speeds.tolist()

        # Mutate phi_x angles
        if np.random.rand() < self.config.mutation_rate:
            phi_x = np.array(mutated['phi_x'])
            mutation = np.random.randn(len(phi_x)) * self.config.mutation_scale * 10
            phi_x += mutation
            phi_x = np.clip(phi_x, -45, 45)
            mutated['phi_x'] = phi_x.tolist()

        # Mutate phi_y angles
        if np.random.rand() < self.config.mutation_rate:
            phi_y = np.array(mutated['phi_y'])
            mutation = np.random.randn(len(phi_y)) * self.config.mutation_scale * 10
            phi_y += mutation
            phi_y = np.clip(phi_y, -45, 45)
            mutated['phi_y'] = phi_y.tolist()

        return mutated

    def tournament_selection(self, population: List[Dict],
                            fitnesses: List[float]) -> Dict:
        """
        Tournament selection for parent selection.
        """
        tournament_idx = np.random.choice(
            len(population),
            size=self.config.tournament_size,
            replace=False
        )

        tournament_fitness = [fitnesses[i] for i in tournament_idx]
        winner_idx = tournament_idx[np.argmin(tournament_fitness)]

        return population[winner_idx].copy()

    def optimize(self, target_pattern: np.ndarray,
                 wedge_count: int,
                 initial_params: Dict = None,
                 verbose: bool = True) -> Tuple[Dict, float, Dict]:
        """
        Run genetic algorithm optimization.

        Args:
            target_pattern: Pattern to match
            wedge_count: Number of wedges
            initial_params: Optional initial parameters from NN
            verbose: Print progress

        Returns:
            best_params: Best parameters found
            best_fitness: Best fitness achieved
            info: Additional information about optimization
        """
        # Initialize population
        population = []

        # Add initial guess if provided
        if initial_params:
            population.append(initial_params)
            # Create variations of initial guess
            for _ in range(self.config.population_size // 2):
                population.append(
                    self.create_individual(wedge_count, initial_params)
                )

        # Fill rest with random individuals
        while len(population) < self.config.population_size:
            population.append(self.create_individual(wedge_count))

        # Evolution loop
        best_fitness = float('inf')
        best_params = None
        history = []

        for generation in range(self.config.generations):
            # Evaluate fitness
            fitnesses = [
                self.evaluate_fitness(ind, target_pattern)
                for ind in population
            ]

            # Track best
            gen_best_idx = np.argmin(fitnesses)
            gen_best_fitness = fitnesses[gen_best_idx]

            if gen_best_fitness < best_fitness:
                best_fitness = gen_best_fitness
                best_params = population[gen_best_idx].copy()

            history.append({
                'generation': generation,
                'best_fitness': best_fitness,
                'mean_fitness': np.mean([f for f in fitnesses if f != float('inf')])
            })

            if verbose and generation % 10 == 0:
                print(f"  GA Gen {generation}: Best fitness = {best_fitness:.6f}")

            # Check for convergence
            if best_fitness < 0.001:  # Very good match
                print(f"  Converged at generation {generation}")
                break

            # Create next generation
            new_population = []

            # Elitism - keep best individuals
            sorted_idx = np.argsort(fitnesses)
            for i in range(self.config.elite_size):
                new_population.append(population[sorted_idx[i]].copy())

            # Create offspring
            while len(new_population) < self.config.population_size:
                # Select parents
                parent1 = self.tournament_selection(population, fitnesses)
                parent2 = self.tournament_selection(population, fitnesses)

                # Crossover
                child1, child2 = self.crossover(parent1, parent2)

                # Mutation
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)

                new_population.extend([child1, child2])

            # Trim to population size
            population = new_population[:self.config.population_size]

        # Return results
        info = {
            'generations_run': len(history),
            'history': history,
            'converged': best_fitness < 0.001
        }

        return best_params, best_fitness, info


class HybridGARefiner:
    """
    Combines multiple GA strategies for robust refinement.
    """

    def __init__(self, forward_model: Callable):
        self.forward_model = forward_model

    def refine(self, initial_params: Dict, target_pattern: np.ndarray,
               max_iterations: int = 3, verbose: bool = True) -> Dict:
        """
        Iteratively refine parameters using GA with different strategies.

        Args:
            initial_params: Initial parameter estimate (from NN)
            target_pattern: Target pattern to match
            max_iterations: Maximum refinement iterations
            verbose: Print progress

        Returns:
            refined_params: Best refined parameters
        """
        current_params = initial_params.copy()

        for iteration in range(max_iterations):
            if verbose:
                print(f"\nRefinement iteration {iteration + 1}/{max_iterations}")

            # Adjust GA config based on iteration
            if iteration == 0:
                # First pass: broad search
                config = GAConfig(
                    population_size=50,
                    generations=50,
                    mutation_rate=0.2,
                    mutation_scale=0.3
                )
            else:
                # Later passes: fine-tuning
                config = GAConfig(
                    population_size=30,
                    generations=30,
                    mutation_rate=0.1,
                    mutation_scale=0.1
                )

            # Run GA
            ga = GeneticAlgorithm(config, self.forward_model)
            refined_params, fitness, info = ga.optimize(
                target_pattern=target_pattern,
                wedge_count=current_params['wedge_count'],
                initial_params=current_params,
                verbose=verbose
            )

            if verbose:
                print(f"  Iteration {iteration + 1} fitness: {fitness:.6f}")

            # Check for convergence
            if fitness < 0.001:
                print(f"  Excellent match achieved!")
                return refined_params

            current_params = refined_params

        return current_params