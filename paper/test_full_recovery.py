#!/usr/bin/env python3
"""
Full unknown recovery: recover ALL parameters from pattern alone.

For P=3 prisms, recovers up to 17 unknowns:
  - 3 rotation speeds
  - 3 wedge angles x
  - 3 wedge angles y
  - 3 glass indices
  - workpiece distance
  - inter-prism gap
  - beam angle x
  - beam angle y
  (beam positions are less identifiable and omitted here —
   position shifts are partially degenerate with angle changes)

Test configurations:
  A) 9-D:  speeds + angles only (baseline, known to work)
  B) 13-D: + 4 beam params (beam angles + positions)
  C) 12-D: + 3 glass indices
  D) 14-D: + glass + workpiece dist + gap (previously tested)
  E) 17-D: + glass + distances + beam angles (the full monty minus positions)
  F) 19-D: everything including beam positions

Run: python test_full_recovery.py
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
    source_distance=6.0,
    prism_thickness=3.0,
    inter_prism_gap=6.0,
    workpiece_distance=100.0,
    beam_angle_x=10.0,
    beam_angle_y=5.0,
    beam_pos_x=0.0,
    beam_pos_y=0.0,
)

TRUE_P = PrismParameters(3, TRUE_SPEEDS, TRUE_AX, TRUE_AY,
                          glass_indices=TRUE_GLASS, geometry=TRUE_GEO)
TARGET = fast_forward(TRUE_P, 200, 10.0)


def run_config(label, build_params_fn, bounds, true_vec, n_restarts=6,
               de_maxiter=300, de_popsize=30):
    """Run a recovery configuration."""
    n_evals = [0]

    def objective(x):
        n_evals[0] += 1
        try:
            p = build_params_fn(x)
            return float(np.mean((fast_forward(p, 200, 10.0) - TARGET)**2))
        except:
            return 1e6

    t0 = time.time()
    best_x, best_mse = None, float('inf')

    for r in range(n_restarts):
        res = differential_evolution(objective, bounds, seed=42 + r,
                                     maxiter=de_maxiter, popsize=de_popsize,
                                     tol=1e-14, polish=False,
                                     mutation=(0.5, 1.5), recombination=0.8)
        if res.fun < best_mse:
            best_x, best_mse = res.x.copy(), res.fun

    # NM polish
    if best_x is not None:
        res2 = minimize(objective, best_x, method='Nelder-Mead',
                        options={'maxiter': 25000, 'fatol': 1e-14, 'xatol': 1e-14})
        if res2.fun < best_mse:
            best_x, best_mse = res2.x.copy(), res2.fun

    # Permutation search (on prism params only — first 9 entries)
    if best_x is not None:
        sp = best_x[:3]; ax = best_x[3:6]; ay = best_x[6:9]
        extra = best_x[9:]
        for perm in permutations(range(3)):
            if list(perm) == [0, 1, 2]:
                continue
            x_perm = np.concatenate([sp[list(perm)], ax[list(perm)],
                                      ay[list(perm)], extra])
            # Also permute glass if present in extra
            if len(extra) >= 3:
                glass_part = extra[:3]
                rest = extra[3:]
                x_perm_g = np.concatenate([sp[list(perm)], ax[list(perm)],
                                            ay[list(perm)],
                                            glass_part[list(perm)], rest])
                rp = minimize(objective, x_perm_g, method='Nelder-Mead',
                              options={'maxiter': 10000, 'fatol': 1e-14})
                if rp.fun < best_mse:
                    best_x, best_mse = rp.x.copy(), rp.fun
            else:
                rp = minimize(objective, x_perm, method='Nelder-Mead',
                              options={'maxiter': 10000, 'fatol': 1e-14})
                if rp.fun < best_mse:
                    best_x, best_mse = rp.x.copy(), rp.fun

    wt = time.time() - t0
    errors = np.abs(best_x - true_vec) if best_x is not None else None

    return best_x, best_mse, errors, n_evals[0], wt


if __name__ == '__main__':
    results = []

    # ═══════════════════════════════════════════════════
    # A) 9-D: speeds + angles (baseline)
    # ═══════════════════════════════════════════════════
    def build_A(x):
        return PrismParameters(3, x[:3].tolist(), x[3:6].tolist(), x[6:9].tolist(),
                               glass_indices=TRUE_GLASS, geometry=TRUE_GEO)
    bounds_A = [(-3.5, 3.5)]*3 + [(-18, 18)]*6
    true_A = np.array(TRUE_SPEEDS + TRUE_AX + TRUE_AY)

    # ═══════════════════════════════════════════════════
    # B) 13-D: + beam angles and positions
    # ═══════════════════════════════════════════════════
    def build_B(x):
        geo = SystemGeometry(beam_angle_x=x[9], beam_angle_y=x[10],
                             beam_pos_x=x[11], beam_pos_y=x[12])
        return PrismParameters(3, x[:3].tolist(), x[3:6].tolist(), x[6:9].tolist(),
                               glass_indices=TRUE_GLASS, geometry=geo)
    bounds_B = bounds_A + [(-25, 25)]*2 + [(-5, 5)]*2
    true_B = np.array(TRUE_SPEEDS + TRUE_AX + TRUE_AY +
                      [TRUE_GEO.beam_angle_x, TRUE_GEO.beam_angle_y,
                       TRUE_GEO.beam_pos_x, TRUE_GEO.beam_pos_y])

    # ═══════════════════════════════════════════════════
    # C) 12-D: + glass indices
    # ═══════════════════════════════════════════════════
    def build_C(x):
        return PrismParameters(3, x[:3].tolist(), x[3:6].tolist(), x[6:9].tolist(),
                               glass_indices=x[9:12].tolist(), geometry=TRUE_GEO)
    bounds_C = bounds_A + [(1.3, 1.8)]*3
    true_C = np.array(TRUE_SPEEDS + TRUE_AX + TRUE_AY + TRUE_GLASS)

    # ═══════════════════════════════════════════════════
    # D) 14-D: + glass + workpiece dist + gap
    # ═══════════════════════════════════════════════════
    def build_D(x):
        geo = SystemGeometry(workpiece_distance=x[12], inter_prism_gap=x[13])
        return PrismParameters(3, x[:3].tolist(), x[3:6].tolist(), x[6:9].tolist(),
                               glass_indices=x[9:12].tolist(), geometry=geo)
    bounds_D = bounds_A + [(1.3, 1.8)]*3 + [(50, 200)] + [(2, 15)]
    true_D = np.array(TRUE_SPEEDS + TRUE_AX + TRUE_AY + TRUE_GLASS +
                      [TRUE_GEO.workpiece_distance, TRUE_GEO.inter_prism_gap])

    # ═══════════════════════════════════════════════════
    # E) 17-D: + glass + distances + beam angles
    # ═══════════════════════════════════════════════════
    def build_E(x):
        geo = SystemGeometry(workpiece_distance=x[12], inter_prism_gap=x[13],
                             beam_angle_x=x[14], beam_angle_y=x[15])
        return PrismParameters(3, x[:3].tolist(), x[3:6].tolist(), x[6:9].tolist(),
                               glass_indices=x[9:12].tolist(), geometry=geo)
    bounds_E = bounds_D + [(-25, 25)]*2
    true_E = np.array(TRUE_SPEEDS + TRUE_AX + TRUE_AY + TRUE_GLASS +
                      [TRUE_GEO.workpiece_distance, TRUE_GEO.inter_prism_gap,
                       TRUE_GEO.beam_angle_x, TRUE_GEO.beam_angle_y])

    # ═══════════════════════════════════════════════════
    # F) 19-D: everything
    # ═══════════════════════════════════════════════════
    def build_F(x):
        geo = SystemGeometry(workpiece_distance=x[12], inter_prism_gap=x[13],
                             beam_angle_x=x[14], beam_angle_y=x[15],
                             beam_pos_x=x[16], beam_pos_y=x[17])
        return PrismParameters(3, x[:3].tolist(), x[3:6].tolist(), x[6:9].tolist(),
                               glass_indices=x[9:12].tolist(), geometry=geo)
    bounds_F = bounds_E + [(-5, 5)]*2
    true_F = np.array(TRUE_SPEEDS + TRUE_AX + TRUE_AY + TRUE_GLASS +
                      [TRUE_GEO.workpiece_distance, TRUE_GEO.inter_prism_gap,
                       TRUE_GEO.beam_angle_x, TRUE_GEO.beam_angle_y,
                       TRUE_GEO.beam_pos_x, TRUE_GEO.beam_pos_y])

    configs = [
        ("A) 9-D: speeds+angles",     build_A, bounds_A, true_A, 4, 200, 25),
        ("B) 13-D: +beam params",      build_B, bounds_B, true_B, 5, 250, 25),
        ("C) 12-D: +glass",            build_C, bounds_C, true_C, 5, 250, 25),
        ("D) 14-D: +glass+distances",  build_D, bounds_D, true_D, 7, 300, 30),
        ("E) 17-D: +glass+dist+beam",  build_E, bounds_E, true_E, 8, 350, 30),
        ("F) 19-D: everything",         build_F, bounds_F, true_F, 10, 400, 35),
    ]

    print("=" * 70)
    print("FULL UNKNOWN RECOVERY TEST (P=3)")
    print("=" * 70)

    for label, build_fn, bounds, true_vec, nr, mi, ps in configs:
        dim = len(bounds)
        print(f"\n{label} ({dim} free parameters)")
        print(f"  Restarts={nr}, DE maxiter={mi}, popsize={ps}")

        best_x, mse, errors, n_ev, wt = run_config(
            label, build_fn, bounds, true_vec, nr, mi, ps)

        if best_x is not None:
            max_err = np.max(errors)
            print(f"  MSE = {mse:.2e}  ({wt:.0f}s, {n_ev} evals)")
            print(f"  Max error across all params: {max_err:.2e}")
            print(f"  Per-param errors: {[f'{e:.2e}' for e in errors]}")

            row = {
                'config': label, 'dim': dim, 'mse': float(mse),
                'max_error': float(max_err),
                'errors': errors.tolist(),
                'recovered': best_x.tolist(),
                'true': true_vec.tolist(),
                'evals': n_ev, 'time_s': round(wt, 1),
            }
        else:
            print(f"  FAILED")
            row = {'config': label, 'dim': dim, 'failed': True}

        results.append(row)

    out_path = os.path.join(os.path.dirname(__file__), 'full_recovery_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
