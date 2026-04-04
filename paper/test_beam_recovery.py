#!/usr/bin/env python3
"""
Test recovery of beam source parameters (position + angles) as free parameters.

Currently the solver assumes beam_pos_x, beam_pos_y, beam_angle_x, beam_angle_y
are known. This script tests whether they can be recovered along with prism params.

Search dimensions:
  - Base (P=2): 6 prism params + 4 beam params = 10-D
  - Base (P=3): 9 prism params + 4 beam params = 13-D

Run: python test_beam_recovery.py
"""
import sys, os, time, json, io
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'reverse_problem_v2'))

# Force UTF-8 on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from core import PrismParameters, SystemGeometry, fast_forward
from scipy.optimize import differential_evolution, minimize
from itertools import permutations


def solve_with_beam(target, n_prisms, true_geo, n_restarts=5,
                    de_maxiter=250, de_popsize=25):
    """
    Recover prism params + beam source params simultaneously.

    Search vector: [N_1..N_P, ax_1..ax_P, ay_1..ay_P, beam_ax, beam_ay, beam_px, beam_py]
    Dimensions: 3P + 4
    """
    T = len(target)
    n = n_prisms
    n_evals = [0]

    def objective(x):
        n_evals[0] += 1
        try:
            speeds = x[:n].tolist()
            ax = x[n:2*n].tolist()
            ay = x[2*n:3*n].tolist()
            beam_ax = x[3*n]
            beam_ay = x[3*n + 1]
            beam_px = x[3*n + 2]
            beam_py = x[3*n + 3]

            geo = SystemGeometry(
                source_distance=true_geo.source_distance,
                prism_thickness=true_geo.prism_thickness,
                inter_prism_gap=true_geo.inter_prism_gap,
                workpiece_distance=true_geo.workpiece_distance,
                beam_angle_x=beam_ax,
                beam_angle_y=beam_ay,
                beam_pos_x=beam_px,
                beam_pos_y=beam_py,
            )
            p = PrismParameters(n, speeds, ax, ay, geometry=geo)
            return float(np.mean((fast_forward(p, T, 10.0) - target)**2))
        except:
            return 1e6

    # Bounds: prism params + beam params
    bounds = (
        [(-3.5, 3.5)] * n +      # speeds
        [(-18, 18)] * n +         # angles_x
        [(-18, 18)] * n +         # angles_y
        [(-25, 25)] +             # beam_angle_x (degrees)
        [(-25, 25)] +             # beam_angle_y (degrees)
        [(-5, 5)] +               # beam_pos_x
        [(-5, 5)]                 # beam_pos_y
    )

    t0 = time.time()
    best_x, best_mse = None, float('inf')

    for r in range(n_restarts):
        try:
            res = differential_evolution(objective, bounds, seed=42 + r,
                                         maxiter=de_maxiter, popsize=de_popsize,
                                         tol=1e-12, polish=False,
                                         mutation=(0.5, 1.5), recombination=0.8)
            if res.fun < best_mse:
                best_x, best_mse = res.x.copy(), res.fun
        except:
            pass

    # NM polish
    if best_x is not None:
        res_nm = minimize(objective, best_x, method='Nelder-Mead',
                          options={'maxiter': 20000, 'fatol': 1e-12, 'xatol': 1e-12})
        if res_nm.fun < best_mse:
            best_x, best_mse = res_nm.x.copy(), res_nm.fun

    # Permutation search on prism params only
    if best_x is not None and n <= 4:
        sp = best_x[:n]
        ax = best_x[n:2*n]
        ay = best_x[2*n:3*n]
        beam = best_x[3*n:]  # keep beam params fixed
        for perm in permutations(range(n)):
            if list(perm) == list(range(n)):
                continue
            x_perm = np.concatenate([sp[list(perm)], ax[list(perm)],
                                      ay[list(perm)], beam])
            rp = minimize(objective, x_perm, method='Nelder-Mead',
                          options={'maxiter': 10000, 'fatol': 1e-12})
            if rp.fun < best_mse:
                best_x, best_mse = rp.x.copy(), rp.fun

    wall_time = time.time() - t0
    return best_x, best_mse, n_evals[0], wall_time


if __name__ == '__main__':
    results = []

    cases = [
        # (label, n_prisms, speeds, angles_x, angles_y, beam_ax, beam_ay, beam_px, beam_py)
        ("2P default beam",
         2, [1.5, -1.0], [12.0, -8.0], [3.0, 10.0],
         10.0, 5.0, 0.0, 0.0),

        ("2P offset beam",
         2, [1.5, -1.0], [12.0, -8.0], [3.0, 10.0],
         12.0, 7.0, 1.5, -0.8),

        ("3P default beam",
         3, [1.5, -1.0, 2.0], [12.0, -8.0, 5.0], [3.0, 10.0, -6.0],
         10.0, 5.0, 0.0, 0.0),

        ("3P offset beam",
         3, [1.5, -1.0, 2.0], [12.0, -8.0, 5.0], [3.0, 10.0, -6.0],
         15.0, 8.0, 2.0, -1.0),

        ("2P small beam angle",
         2, [1.5, -1.0], [12.0, -8.0], [3.0, 10.0],
         2.0, 1.0, 0.5, 0.3),
    ]

    print("=" * 70)
    print("BEAM SOURCE PARAMETER RECOVERY TEST")
    print("=" * 70)

    for label, n_prisms, speeds, ax, ay, bax, bay, bpx, bpy in cases:
        geo = SystemGeometry(beam_angle_x=bax, beam_angle_y=bay,
                             beam_pos_x=bpx, beam_pos_y=bpy)
        true_p = PrismParameters(n_prisms, speeds, ax, ay, geometry=geo)
        pattern = fast_forward(true_p, 200, 10.0)

        dim = 3 * n_prisms + 4
        print(f"\n{label} ({dim}-D search)")
        print(f"  True beam: ax={bax}, ay={bay}, px={bpx}, py={bpy}")
        print(f"  True prisms: N={speeds}, ax={ax}, ay={ay}")

        best_x, mse, n_ev, wt = solve_with_beam(pattern, n_prisms, geo,
                                                  n_restarts=5, de_maxiter=250)

        if best_x is not None:
            n = n_prisms
            rec_speeds = best_x[:n]
            rec_ax = best_x[n:2*n]
            rec_ay = best_x[2*n:3*n]
            rec_bax = best_x[3*n]
            rec_bay = best_x[3*n + 1]
            rec_bpx = best_x[3*n + 2]
            rec_bpy = best_x[3*n + 3]

            speed_err = max(abs(rec_speeds[i] - speeds[i]) for i in range(n))
            angx_err = max(abs(rec_ax[i] - ax[i]) for i in range(n))
            angy_err = max(abs(rec_ay[i] - ay[i]) for i in range(n))
            bax_err = abs(rec_bax - bax)
            bay_err = abs(rec_bay - bay)
            bpx_err = abs(rec_bpx - bpx)
            bpy_err = abs(rec_bpy - bpy)

            print(f"  MSE = {mse:.2e}  ({wt:.0f}s, {n_ev} evals)")
            print(f"  Speed err:    {speed_err:.2e} Hz")
            print(f"  Angle_x err:  {angx_err:.2e} deg")
            print(f"  Angle_y err:  {angy_err:.2e} deg")
            print(f"  Beam ax err:  {bax_err:.2e} deg")
            print(f"  Beam ay err:  {bay_err:.2e} deg")
            print(f"  Beam px err:  {bpx_err:.2e}")
            print(f"  Beam py err:  {bpy_err:.2e}")

            row = {
                'case': label, 'dim': dim, 'mse': float(mse),
                'speed_err': float(speed_err), 'angx_err': float(angx_err),
                'angy_err': float(angy_err),
                'beam_ax_err': float(bax_err), 'beam_ay_err': float(bay_err),
                'beam_px_err': float(bpx_err), 'beam_py_err': float(bpy_err),
                'evals': n_ev, 'time_s': round(wt, 1),
            }
        else:
            print(f"  FAILED")
            row = {'case': label, 'failed': True}

        results.append(row)

    # Save results
    out_path = os.path.join(os.path.dirname(__file__), 'beam_recovery_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
