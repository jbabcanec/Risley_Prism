#!/usr/bin/env python3
"""
Pipeline for the inverse Risley prism problem with physical prism model.

Each prism = 2 interfaces (entry face + exit face) rotating in lockstep.
  2 prisms → 4 interfaces
  3 prisms → 6 interfaces
"""

import numpy as np
import os, sys, time as _time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from itertools import permutations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import PrismParameters, SystemGeometry, DEFAULT_GEOMETRY, fast_forward
from neural_network import PrismPredictor


# ═════════════════════════════════════════════════════════════════════════
# Scipy refinement
# ═════════════════════════════════════════════════════════════════════════

def refine_scipy(initial: Dict, target: np.ndarray,
                 n_points: int = 200, time_limit: float = 10.0,
                 geometry: SystemGeometry = None,
                 recover_glass: bool = False,
                 verbose: bool = False) -> Tuple[Dict, float]:
    """
    Multi-restart DE + Nelder-Mead + permutation search.

    INPUT (the "picture"):
      target — (T, 2) array of observed (x, y) workpiece positions

    USER-SPECIFIED (known hardware):
      geometry — physical layout (distances, beam angles)
      n_prisms — number of physical prisms

    RECOVERED (the answer):
      rotation_speeds  — N_i (Hz) per prism
      wedge_angles_x   — α_{x,i} (deg) per prism
      wedge_angles_y   — α_{y,i} (deg) per prism
      glass_indices    — n_{g,i} per prism (if recover_glass=True)
    """
    from scipy.optimize import differential_evolution, minimize

    if geometry is None:
        geometry = DEFAULT_GEOMETRY

    np_ = initial['n_prisms']
    init_glass = initial.get('glass_indices', [1.5 + 0.05*i for i in range(np_)])

    def objective(x):
        try:
            speeds = x[:np_].tolist()
            ax = x[np_:2*np_].tolist()
            ay = x[2*np_:3*np_].tolist()
            gi = x[3*np_:4*np_].tolist() if recover_glass else init_glass
            p = PrismParameters(np_, speeds, ax, ay,
                                glass_indices=gi, geometry=geometry)
            return float(np.mean((fast_forward(p, n_points, time_limit) - target)**2))
        except Exception:
            return 1e6

    # Bounds
    bounds = ([(-3.5, 3.5)] * np_ +        # speeds
              [(-18.0, 18.0)] * np_ +       # angles_x
              [(-18.0, 18.0)] * np_)        # angles_y
    if recover_glass:
        bounds += [(1.3, 2.0)] * np_        # glass indices

    # Initial point from NN
    x0_parts = [initial['rotation_speeds'], initial['wedge_angles_x'],
                initial['wedge_angles_y']]
    if recover_glass:
        x0_parts.append(init_glass)
    x0_nn = np.concatenate(x0_parts)
    x0_nn = np.clip(x0_nn, [b[0] for b in bounds], [b[1] for b in bounds])

    ndim = len(bounds)
    best_x, best_f = x0_nn.copy(), objective(x0_nn)
    n_restarts = 4

    if verbose:
        mode = f"{ndim}D ({'with' if recover_glass else 'without'} glass recovery)"
        print(f"    Search: {mode}")

    for r in range(n_restarts):
        seed = 42 + r * 13
        x0 = x0_nn if r == 0 else None

        de = differential_evolution(
            objective, bounds, seed=seed, maxiter=250, popsize=25,
            tol=1e-9, mutation=(0.5, 1.5), recombination=0.8,
            polish=False, x0=x0, disp=False)

        nm = minimize(objective, de.x, method='Nelder-Mead',
                      options={'maxiter': 15000, 'xatol': 1e-5, 'fatol': 1e-11})

        if nm.fun < best_f:
            best_f, best_x = nm.fun, nm.x

        if verbose:
            print(f"    Restart {r+1}/{n_restarts}: MSE={nm.fun:.2e}")
        if best_f < 1e-8:
            break

    # Permutation search
    if np_ <= 3:
        for perm in permutations(range(np_)):
            parts = [
                np.array([best_x[perm[j]] for j in range(np_)]),
                np.array([best_x[np_ + perm[j]] for j in range(np_)]),
                np.array([best_x[2*np_ + perm[j]] for j in range(np_)]),
            ]
            if recover_glass:
                parts.append(np.array([best_x[3*np_ + perm[j]] for j in range(np_)]))
            xp = np.concatenate(parts)
            nm = minimize(objective, xp, method='Nelder-Mead',
                          options={'maxiter': 8000, 'fatol': 1e-11})
            if nm.fun < best_f:
                best_f, best_x = nm.fun, nm.x

    if verbose:
        print(f"    Final MSE: {best_f:.2e}")

    result = {
        'n_prisms': np_,
        'rotation_speeds': best_x[:np_].tolist(),
        'wedge_angles_x': best_x[np_:2*np_].tolist(),
        'wedge_angles_y': best_x[2*np_:3*np_].tolist(),
    }
    if recover_glass:
        result['glass_indices'] = best_x[3*np_:4*np_].tolist()

    return result, float(best_f)


# ═════════════════════════════════════════════════════════════════════════
# Pipeline
# ═════════════════════════════════════════════════════════════════════════

class Pipeline:
    N_POINTS = 200
    TIME_LIMIT = 10.0

    def __init__(self, prism_counts=None, samples_per_pc=100000, epochs=200):
        if prism_counts is None:
            prism_counts = [2, 3]
        self.prism_counts = prism_counts
        self.samples_per_pc = samples_per_pc
        self.epochs = epochs
        self.predictor = PrismPredictor()
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.out_dir = os.path.join(os.path.dirname(__file__), 'runs', self.timestamp)
        os.makedirs(self.out_dir, exist_ok=True)

    def generate_dataset(self):
        total = self.samples_per_pc * len(self.prism_counts)
        print(f"\n{'='*60}")
        print(f"GENERATING {total} SAMPLES  ({self.samples_per_pc}/prism count)")
        n_if = [2*pc for pc in self.prism_counts]
        print(f"Prism counts: {self.prism_counts}  (interfaces: {n_if})")
        print(f"{'='*60}")

        patterns, pc_arr, params_all = [], [], []
        for pc in self.prism_counts:
            n = self.samples_per_pc
            print(f"  {pc} prisms ({2*pc} interfaces): {n} samples ", end='', flush=True)
            t0 = _time.time()
            ok = 0
            for _ in range(n):
                p = PrismParameters(
                    n_prisms=pc,
                    rotation_speeds=np.random.uniform(-3, 3, pc).tolist(),
                    wedge_angles_x=np.random.uniform(-15, 15, pc).tolist(),
                    wedge_angles_y=np.random.uniform(-15, 15, pc).tolist(),
                )
                try:
                    pat = fast_forward(p, self.N_POINTS, self.TIME_LIMIT)
                    if np.std(pat) < 1e-8 or not np.all(np.isfinite(pat)):
                        continue
                    patterns.append(pat)
                    pc_arr.append(pc)
                    params_all.append(p.to_flat())
                    ok += 1
                except Exception:
                    continue
            print(f" {ok}/{n} valid  ({_time.time()-t0:.1f}s)")

        data = {
            'patterns': np.array(patterns, dtype=np.float32),
            'prism_counts': np.array(pc_arr, dtype=np.int32),
            'params': np.array(params_all, dtype=np.float32),
        }
        np.savez_compressed(os.path.join(self.out_dir, 'dataset.npz'), **data)
        print(f"  Total: {len(patterns)} samples saved")
        return data

    def train(self, data):
        print(f"\n{'='*60}")
        print(f"TRAINING  ({self.epochs} epochs)")
        print(f"{'='*60}")
        history = self.predictor.train(
            data['patterns'], data['prism_counts'], data['params'],
            prism_counts=self.prism_counts, epochs=self.epochs, batch_size=512)
        self.predictor.save(os.path.join(self.out_dir, 'model'))
        return history

    def evaluate(self, data, n_test=3000):
        print(f"\n{'='*60}")
        print(f"EVALUATING")
        print(f"{'='*60}")

        N = len(data['patterns'])
        n_test = min(n_test, N)
        idx = np.random.permutation(N)[:n_test]
        pats = data['patterns'][idx]
        true_pc = data['prism_counts'][idx]
        true_p = data['params'][idx]

        pred_pc, pred_p = self.predictor.predict_batch(pats)

        acc = (pred_pc == true_pc).mean()
        print(f"\n  Prism-count classification: {acc:.1%}")
        for pc in self.prism_counts:
            m = true_pc == pc
            if m.any():
                print(f"    {pc} prisms ({2*pc} if): {(pred_pc[m]==pc).mean():.1%}  ({m.sum()} samples)")

        for pc in self.prism_counts:
            m = (true_pc == pc) & (pred_pc == pc)
            if not m.any(): continue
            print(f"\n  {pc} prisms -- NN regression MAE (per prism):")
            for w in range(pc):
                sp = np.abs(pred_p[m, w] - true_p[m, w]).mean()
                ax = np.abs(pred_p[m, 3+w] - true_p[m, 3+w]).mean()
                ay = np.abs(pred_p[m, 6+w] - true_p[m, 6+w]).mean()
                print(f"    Prism {w+1}: speed={sp:.3f} Hz  angle_x={ax:.2f} deg  angle_y={ay:.2f} deg")

        # Reconstruction MSE (NN only)
        n_recon = min(300, n_test)
        mses = []
        for k in range(n_recon):
            pc = int(pred_pc[k])
            try:
                rp = PrismParameters(pc, pred_p[k,:pc].tolist(),
                                     pred_p[k,3:3+pc].tolist(), pred_p[k,6:6+pc].tolist())
                recon = fast_forward(rp, self.N_POINTS, self.TIME_LIMIT)
                mses.append(float(np.mean((recon - pats[k])**2)))
            except: pass
        if mses:
            print(f"\n  Reconstruction MSE (NN only, n={len(mses)}):")
            print(f"    Mean={np.mean(mses):.2f}  Median={np.median(mses):.2f}  "
                  f"<1.0: {sum(m<1 for m in mses)/len(mses):.0%}  "
                  f"<0.1: {sum(m<0.1 for m in mses)/len(mses):.0%}")
        print(f"{'='*60}")
        return {'accuracy': float(acc)}

    def solve(self, pattern, n_prisms=None, geometry=None,
              recover_glass=False, refine=True, verbose=True):
        """
        THE MAIN ENTRY POINT: pattern in → prism parameters out.

        INPUT:
          pattern  — (T, 2) array: the observed scan on the workpiece
          n_prisms — int: how many physical prisms (if known; else auto-detected)
          geometry — SystemGeometry: your hardware layout (distances, beam angles)

        OUTPUT dict:
          rotation_speeds  — how fast each prism spins (Hz)
          wedge_angles_x   — exit-face tilt x per prism (degrees)
          wedge_angles_y   — exit-face tilt y per prism (degrees)
          glass_indices    — refractive index per prism (if recover_glass=True)
          mse              — reconstruction quality (lower = better)
        """
        if geometry is None:
            geometry = DEFAULT_GEOMETRY

        nn = self.predictor.predict(pattern, n_prisms=n_prisms)
        np_ = nn['n_prisms']
        if verbose:
            print(f"\n  NN estimate: {np_} prisms ({2*np_} interfaces)")
            print(f"    speeds   = {[f'{s:.4f}' for s in nn['rotation_speeds']]}")
            print(f"    angles_x = {[f'{a:.3f}' for a in nn['wedge_angles_x']]}")
            print(f"    angles_y = {[f'{a:.3f}' for a in nn['wedge_angles_y']]}")

        result = nn
        if refine:
            if verbose:
                print(f"  Refining ({'with' if recover_glass else 'without'} glass recovery)...")
            refined, fitness = refine_scipy(
                nn, pattern, self.N_POINTS, self.TIME_LIMIT,
                geometry=geometry, recover_glass=recover_glass, verbose=verbose)
            result = refined
            result['fitness'] = fitness

        # Reconstruction check
        try:
            gi = result.get('glass_indices')
            rp = PrismParameters(result['n_prisms'], result['rotation_speeds'],
                                 result['wedge_angles_x'], result['wedge_angles_y'],
                                 glass_indices=gi, geometry=geometry)
            recon = fast_forward(rp, self.N_POINTS, self.TIME_LIMIT)
            result['mse'] = float(np.mean((recon - pattern)**2))
        except:
            result['mse'] = None

        if verbose:
            print(f"\n  === RECOVERED PARAMETERS ===")
            print(f"  Prisms: {result['n_prisms']}  ({2*result['n_prisms']} interfaces)")
            print(f"  Speeds:   {[f'{s:.6f}' for s in result['rotation_speeds']]} Hz")
            print(f"  Angles_x: {[f'{a:.4f}' for a in result['wedge_angles_x']]} deg")
            print(f"  Angles_y: {[f'{a:.4f}' for a in result['wedge_angles_y']]} deg")
            if 'glass_indices' in result:
                print(f"  Glass n:  {[f'{n:.4f}' for n in result['glass_indices']]}")
            if result.get('mse') is not None:
                print(f"  Reconstruction MSE: {result['mse']:.2e}")
        return result

    def run(self):
        t0 = _time.time()
        n_if = [2*pc for pc in self.prism_counts]
        print(f"\n{'='*60}")
        print(f"  INVERSE RISLEY PRISM -- PHYSICAL PRISM MODEL")
        print(f"  Prisms: {self.prism_counts}  (interfaces: {n_if})")
        print(f"  Samples/pc: {self.samples_per_pc}  Epochs: {self.epochs}")
        print(f"{'='*60}")

        data = self.generate_dataset()
        history = self.train(data)
        results = self.evaluate(data)

        elapsed = _time.time() - t0
        print(f"\n  Pipeline: {elapsed/60:.1f} min")

        # ── Demos with full error reporting ──
        demos = []
        for pc in self.prism_counts:
            speeds = np.random.uniform(-2.5, 2.5, pc).round(2).tolist()
            ax = np.random.uniform(-12, 12, pc).round(1).tolist()
            ay = np.random.uniform(-12, 12, pc).round(1).tolist()
            demos.append(PrismParameters(pc, speeds, ax, ay))

        for true_p in demos:
            np_ = true_p.n_prisms
            print(f"\n{'='*60}")
            print(f"  DEMO: {np_} prisms ({2*np_} interfaces)")
            print(f"{'='*60}")
            print(f"  True: N={true_p.rotation_speeds}  "
                  f"ax={true_p.wedge_angles_x}  ay={true_p.wedge_angles_y}")

            pat = fast_forward(true_p, self.N_POINTS, self.TIME_LIMIT)
            rec = self.solve(pat, n_prisms=np_, refine=True)

            if rec.get('mse') is not None:
                print(f"\n  Error report:")
                for w in range(np_):
                    ds = abs(rec['rotation_speeds'][w] - true_p.rotation_speeds[w])
                    dx = abs(rec['wedge_angles_x'][w] - true_p.wedge_angles_x[w])
                    dy = abs(rec['wedge_angles_y'][w] - true_p.wedge_angles_y[w])
                    print(f"    Prism {w+1}: |dN|={ds:.2e} Hz  "
                          f"|d_ax|={dx:.2e} deg  |d_ay|={dy:.2e} deg")
                print(f"    Reconstruction MSE = {rec['mse']:.2e}")

        return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Inverse Risley Prism -- Physical Prism Model')
    parser.add_argument('--prisms', type=int, nargs='+', default=[2, 3],
                        help='Number of physical prisms (default: 2 3)')
    parser.add_argument('--samples', type=int, default=100000,
                        help='Samples per prism count (default: 100000)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Training epochs (default: 200)')
    args = parser.parse_args()
    Pipeline(prism_counts=args.prisms, samples_per_pc=args.samples, epochs=args.epochs).run()
