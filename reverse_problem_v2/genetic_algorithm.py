#!/usr/bin/env python3
"""
Genetic Algorithm for parameter refinement using the fast forward model.
Includes multi-start strategy and progressive fine-tuning.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

from core import RisleyParameters, fast_forward


@dataclass
class GAConfig:
    population_size: int = 80
    generations: int = 100
    elite_size: int = 5
    mutation_rate: float = 0.15
    mutation_scale: float = 0.25
    crossover_rate: float = 0.7
    tournament_size: int = 3


class GeneticAlgorithm:

    def __init__(self, config: GAConfig, n_points: int = 200,
                 time_limit: float = 10.0):
        self.cfg = config
        self.n_points = n_points
        self.time_limit = time_limit

    def _individual(self, wc: int, seed: Optional[Dict] = None,
                    scale: float = 1.0) -> Dict:
        if seed:
            s  = np.array(seed['rotation_speeds']) + np.random.randn(wc) * 0.5 * scale
            px = np.array(seed['phi_x'])  + np.random.randn(wc) * 3.0 * scale
            py = np.array(seed['phi_y'])  + np.random.randn(wc) * 3.0 * scale
        else:
            s  = np.random.uniform(-4, 4, wc)
            px = np.random.uniform(-18, 18, wc)
            py = np.random.uniform(-18, 18, wc)
        return {
            'wedge_count': wc,
            'rotation_speeds': np.clip(s, -10, 10).tolist(),
            'phi_x': np.clip(px, -45, 45).tolist(),
            'phi_y': np.clip(py, -45, 45).tolist(),
        }

    def _evaluate(self, ind: Dict, target: np.ndarray) -> float:
        try:
            p = RisleyParameters(**ind)
            if not p.validate():
                return float('inf')
            sim = fast_forward(p, self.n_points, self.time_limit)
            L = min(len(sim), len(target))
            return float(np.mean((sim[:L] - target[:L]) ** 2))
        except Exception:
            return float('inf')

    def _crossover(self, a: Dict, b: Dict) -> Tuple[Dict, Dict]:
        c1, c2 = dict(a), dict(b)
        if np.random.rand() < self.cfg.crossover_rate:
            for key in ('rotation_speeds', 'phi_x', 'phi_y'):
                va, vb = np.array(a[key]), np.array(b[key])
                mask = np.random.rand(len(va)) > 0.5
                ca, cb = va.copy(), vb.copy()
                ca[mask], cb[mask] = vb[mask], va[mask]
                c1[key], c2[key] = ca.tolist(), cb.tolist()
        return c1, c2

    def _mutate(self, ind: Dict) -> Dict:
        m = dict(ind)
        ms = self.cfg.mutation_scale
        if np.random.rand() < self.cfg.mutation_rate:
            s = np.array(m['rotation_speeds'])
            s += np.random.randn(len(s)) * ms * 2
            m['rotation_speeds'] = np.clip(s, -10, 10).tolist()
        if np.random.rand() < self.cfg.mutation_rate:
            px = np.array(m['phi_x'])
            px += np.random.randn(len(px)) * ms * 10
            m['phi_x'] = np.clip(px, -45, 45).tolist()
        if np.random.rand() < self.cfg.mutation_rate:
            py = np.array(m['phi_y'])
            py += np.random.randn(len(py)) * ms * 10
            m['phi_y'] = np.clip(py, -45, 45).tolist()
        return m

    def _select(self, pop, fits):
        ti = np.random.choice(len(pop), self.cfg.tournament_size, replace=False)
        return dict(pop[ti[np.argmin([fits[j] for j in ti])]])

    def optimise(self, target: np.ndarray, wc: int,
                 seed: Optional[Dict] = None,
                 verbose: bool = False) -> Tuple[Dict, float]:

        pop = []
        if seed:
            pop.append(dict(seed))
            # Dense cluster near seed
            for _ in range(self.cfg.population_size // 3):
                pop.append(self._individual(wc, seed, scale=0.5))
            # Wider exploration near seed
            for _ in range(self.cfg.population_size // 3):
                pop.append(self._individual(wc, seed, scale=2.0))
        # Random individuals for diversity
        while len(pop) < self.cfg.population_size:
            pop.append(self._individual(wc))

        best, best_f = pop[0], float('inf')
        stale = 0

        for gen in range(self.cfg.generations):
            fits = [self._evaluate(ind, target) for ind in pop]
            gi = int(np.argmin(fits))
            if fits[gi] < best_f:
                best_f, best = fits[gi], dict(pop[gi])
                stale = 0
            else:
                stale += 1

            if verbose and gen % 20 == 0:
                print(f"    GA gen {gen:3d}: best={best_f:.6f}")
            if best_f < 0.001:
                break
            if stale > 30:
                break

            elite_idx = np.argsort(fits)[:self.cfg.elite_size]
            nxt = [dict(pop[i]) for i in elite_idx]
            while len(nxt) < self.cfg.population_size:
                p1 = self._select(pop, fits)
                p2 = self._select(pop, fits)
                c1, c2 = self._crossover(p1, p2)
                nxt.extend([self._mutate(c1), self._mutate(c2)])
            pop = nxt[:self.cfg.population_size]

        return best, best_f


class HybridGARefiner:
    """Multi-stage GA: broad → medium → fine, with multi-start."""

    def __init__(self, n_points: int = 200, time_limit: float = 10.0):
        self.n_points = n_points
        self.time_limit = time_limit

    def refine(self, initial: Dict, target: np.ndarray,
               iterations: int = 3, verbose: bool = True) -> Tuple[Dict, float]:
        current = dict(initial)
        best_fitness = float('inf')

        stages = [
            GAConfig(population_size=100, generations=120, elite_size=6,
                     mutation_rate=0.25, mutation_scale=0.4, crossover_rate=0.7),
            GAConfig(population_size=60, generations=80, elite_size=5,
                     mutation_rate=0.15, mutation_scale=0.15, crossover_rate=0.7),
            GAConfig(population_size=40, generations=60, elite_size=4,
                     mutation_rate=0.10, mutation_scale=0.05, crossover_rate=0.8),
        ]

        for it in range(min(iterations, len(stages))):
            if verbose:
                print(f"  Refinement round {it + 1}/{iterations}")
            cfg = stages[it]
            ga = GeneticAlgorithm(cfg, self.n_points, self.time_limit)
            result, fitness = ga.optimise(
                target, current['wedge_count'], seed=current, verbose=verbose)
            if fitness < best_fitness:
                best_fitness = fitness
                current = result
            if verbose:
                print(f"    Round {it+1} fitness: {best_fitness:.6f}")
            if best_fitness < 0.001:
                break

        return current, best_fitness

    def refine_multistart(self, initial: Dict, target: np.ndarray,
                          n_starts: int = 3,
                          verbose: bool = True) -> Tuple[Dict, float]:
        """Run refinement multiple times with different random seeds."""
        best, best_f = None, float('inf')

        for s in range(n_starts):
            if verbose:
                print(f"\n  === Start {s+1}/{n_starts} ===")
            result, fitness = self.refine(initial, target, iterations=3,
                                          verbose=verbose)
            if fitness < best_f:
                best_f = fitness
                best = result
            if best_f < 0.001:
                break

        return best, best_f
