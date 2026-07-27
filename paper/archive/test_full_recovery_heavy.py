#!/usr/bin/env python3
"""
Heavy-budget full recovery: retry configs B-F with much more compute.

Config A (9-D) already passed. The others need more restarts and
longer DE runs. Based on the 14-D case that took 48 min with 7 restarts
and 2.1M evals to succeed.

Uses 4 workers for DE parallelism (gentle on thermals).
"""
import sys, os, time, json, io
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'reverse_problem_v2'))

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from core import PrismParameters, SystemGeometry, fast_forward
from scipy.optimize import differential_evolution, minimize
from itertools import permutations


# True system
TRUE_SPEEDS = [1.5, -1.0, 2.0]
TRUE_AX = [12.0, -8.0, 5.0]
TRUE_AY = [3.0, 10.0, -6.0]
TRUE_GLASS = [1.50, 1.55, 1.60]
TRUE_GEO = SystemGeometry(
    source_distance=6.0, prism_thickness=3.0, inter_prism_gap=6.0,
    workpiece_distance=100.0, beam_angle_x=10.0, beam_angle_y=5.0,
    beam_pos_x=0.0, beam_pos_y=0.0,
)
TRUE_P = PrismParameters(3, TRUE_SPEEDS, TRUE_AX, TRUE_AY,
                          glass_indices=TRUE_GLASS, geometry=TRUE_GEO)
TARGET = fast_forward(TRUE_P, 200, 10.0)

N_WORKERS = 4


class Objective:
    def __init__(self, build_fn):
        self.build_fn = build_fn
    def __call__(self, x):
        try:
            p = self.build_fn(x)
            return float(np.mean((fast_forward(p, 200, 10.0) - TARGET)**2))
        except:
            return 1e6


# Build functions (module-level for pickling)
def build_B(x):
    geo = SystemGeometry(beam_angle_x=x[9], beam_angle_y=x[10],
                         beam_pos_x=x[11], beam_pos_y=x[12])
    return PrismParameters(3, x[:3].tolist(), x[3:6].tolist(), x[6:9].tolist(),
                           glass_indices=TRUE_GLASS, geometry=geo)

def build_C(x):
    return PrismParameters(3, x[:3].tolist(), x[3:6].tolist(), x[6:9].tolist(),
                           glass_indices=x[9:12].tolist(), geometry=TRUE_GEO)

def build_D(x):
    geo = SystemGeometry(workpiece_distance=x[12], inter_prism_gap=x[13])
    return PrismParameters(3, x[:3].tolist(), x[3:6].tolist(), x[6:9].tolist(),
                           glass_indices=x[9:12].tolist(), geometry=geo)

def build_E(x):
    geo = SystemGeometry(workpiece_distance=x[12], inter_prism_gap=x[13],
                         beam_angle_x=x[14], beam_angle_y=x[15])
    return PrismParameters(3, x[:3].tolist(), x[3:6].tolist(), x[6:9].tolist(),
                           glass_indices=x[9:12].tolist(), geometry=geo)

def build_F(x):
    geo = SystemGeometry(workpiece_distance=x[12], inter_prism_gap=x[13],
                         beam_angle_x=x[14], beam_angle_y=x[15],
                         beam_pos_x=x[16], beam_pos_y=x[17])
    return PrismParameters(3, x[:3].tolist(), x[3:6].tolist(), x[6:9].tolist(),
                           glass_indices=x[9:12].tolist(), geometry=geo)


# Bounds
bounds_base = [(-3.5, 3.5)]*3 + [(-18, 18)]*6
bounds_B = bounds_base + [(-25, 25)]*2 + [(-5, 5)]*2
bounds_C = bounds_base + [(1.3, 1.8)]*3
bounds_D = bounds_base + [(1.3, 1.8)]*3 + [(50, 200)] + [(2, 15)]
bounds_E = bounds_D + [(-25, 25)]*2
bounds_F = bounds_E + [(-5, 5)]*2

# True vectors
true_B = np.array(TRUE_SPEEDS + TRUE_AX + TRUE_AY +
                  [10.0, 5.0, 0.0, 0.0])
true_C = np.array(TRUE_SPEEDS + TRUE_AX + TRUE_AY + TRUE_GLASS)
true_D = np.array(TRUE_SPEEDS + TRUE_AX + TRUE_AY + TRUE_GLASS + [100.0, 6.0])
true_E = np.array(TRUE_SPEEDS + TRUE_AX + TRUE_AY + TRUE_GLASS + [100.0, 6.0, 10.0, 5.0])
true_F = np.array(TRUE_SPEEDS + TRUE_AX + TRUE_AY + TRUE_GLASS + [100.0, 6.0, 10.0, 5.0, 0.0, 0.0])


def run_heavy(label, build_fn, bounds, true_vec, n_restarts, de_maxiter,
              de_popsize, nm_maxiter=30000):
    """Heavy-budget solver with progress reporting."""
    obj = Objective(build_fn)
    dim = len(bounds)
    t0 = time.time()
    best_x, best_mse = None, float('inf')

    for r in range(n_restarts):
        rt0 = time.time()
        res = differential_evolution(obj, bounds, seed=42 + r,
                                     maxiter=de_maxiter, popsize=de_popsize,
                                     tol=1e-14, polish=False,
                                     mutation=(0.5, 1.5), recombination=0.8,
                                     workers=N_WORKERS, updating='deferred')
        dt = time.time() - rt0
        improved = ""
        if res.fun < best_mse:
            best_x, best_mse = res.x.copy(), res.fun
            improved = " *"
        print(f"    restart {r+1}/{n_restarts}: MSE={res.fun:.2e} ({dt:.0f}s){improved}",
              flush=True)

    # Heavy NM polish
    if best_x is not None:
        print(f"    NM polishing (maxiter={nm_maxiter})...", flush=True)
        res2 = minimize(obj, best_x, method='Nelder-Mead',
                        options={'maxiter': nm_maxiter, 'fatol': 1e-15, 'xatol': 1e-15})
        if res2.fun < best_mse:
            best_x, best_mse = res2.x.copy(), res2.fun
        print(f"    NM result: MSE={best_mse:.2e}", flush=True)

    # Permutation search with glass permutation
    if best_x is not None:
        print(f"    Permutation search (6 orderings)...", flush=True)
        sp = best_x[:3]; ax = best_x[3:6]; ay = best_x[6:9]
        extra = best_x[9:]
        for perm in permutations(range(3)):
            if list(perm) == [0, 1, 2]:
                continue
            # Permute prism params
            sp_p = sp[list(perm)]; ax_p = ax[list(perm)]; ay_p = ay[list(perm)]
            # Also permute glass if in extra (first 3 of extra)
            if len(extra) >= 3:
                glass_p = extra[:3][list(perm)]
                rest = extra[3:]
                x_perm = np.concatenate([sp_p, ax_p, ay_p, glass_p, rest])
            else:
                x_perm = np.concatenate([sp_p, ax_p, ay_p, extra])
            rp = minimize(obj, x_perm, method='Nelder-Mead',
                          options={'maxiter': 15000, 'fatol': 1e-15})
            if rp.fun < best_mse:
                best_x, best_mse = rp.x.copy(), rp.fun
                print(f"    perm improved: MSE={best_mse:.2e}", flush=True)

    wt = time.time() - t0
    errors = np.abs(best_x - true_vec) if best_x is not None else None
    return best_x, best_mse, errors, wt


if __name__ == '__main__':
    results = []

    # Heavy configs — much more budget than the light run
    configs = [
        ("B) 13-D: +beam params",      build_B, bounds_B, true_B,  10, 350, 30),
        ("C) 12-D: +glass",            build_C, bounds_C, true_C,  8,  300, 30),
        ("D) 14-D: +glass+distances",  build_D, bounds_D, true_D,  10, 350, 30),
        ("E) 17-D: +glass+dist+beam",  build_E, bounds_E, true_E,  12, 400, 35),
        ("F) 19-D: everything",         build_F, bounds_F, true_F,  15, 500, 40),
    ]

    print("=" * 70)
    print("HEAVY-BUDGET FULL RECOVERY TEST (P=3)")
    print("=" * 70)

    for label, build_fn, bounds, true_vec, nr, mi, ps in configs:
        dim = len(bounds)
        print(f"\n{label} ({dim} free parameters)")
        print(f"  Budget: {nr} restarts, DE maxiter={mi}, popsize={ps}")

        best_x, mse, errors, wt = run_heavy(label, build_fn, bounds, true_vec, nr, mi, ps)

        if best_x is not None:
            max_err = np.max(errors)
            print(f"  RESULT: MSE={mse:.2e}, max_err={max_err:.2e} ({wt:.0f}s)")

            # Print named errors
            names = ['N1','N2','N3','ax1','ax2','ax3','ay1','ay2','ay3']
            if dim >= 12: names += ['ng1','ng2','ng3']
            if dim >= 13 and 'beam' in label and 'glass' not in label:
                names = ['N1','N2','N3','ax1','ax2','ax3','ay1','ay2','ay3',
                         'beam_ax','beam_ay','beam_px','beam_py']
            if dim >= 14 and 'glass' in label:
                names += ['d_W','gap']
            if dim >= 16: names += ['beam_ax','beam_ay']
            if dim >= 18: names += ['beam_px','beam_py']

            for i, (n, e) in enumerate(zip(names, errors)):
                status = "OK" if e < 0.01 else "CLOSE" if e < 1.0 else "BAD"
                print(f"    {n:10s}: err={e:.2e}  [{status}]")

            row = {
                'config': label, 'dim': dim, 'mse': float(mse),
                'max_error': float(max_err),
                'errors': errors.tolist(),
                'recovered': best_x.tolist(),
                'true': true_vec.tolist(),
                'time_s': round(wt, 1),
            }
        else:
            print(f"  FAILED")
            row = {'config': label, 'dim': dim, 'failed': True}

        results.append(row)

    out_path = os.path.join(os.path.dirname(__file__), 'full_recovery_heavy_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
