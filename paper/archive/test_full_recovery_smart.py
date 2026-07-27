#!/usr/bin/env python3
"""
Smart full recovery: extract hints from the pattern before optimizing.

Pre-processing (no assumptions, just measurement):
  1. Centroid → estimate beam angles
  2. FFT peaks → estimate speeds
  3. RMS radius → estimate d_W * alpha product
  4. Use these to seed DE's first restart + tighten bounds

Then run DE+NM+permutation with informed initialization.
"""
import sys, os, time, json, io
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'reverse_problem_v2'))

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from core import PrismParameters, SystemGeometry, fast_forward
from scipy.optimize import differential_evolution, minimize
from itertools import permutations


# True system (same as other tests)
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
N_PRISMS = 3
T_OBS = 10.0


def extract_hints(pattern, n_prisms, t_obs):
    """Extract geometric hints from the pattern. No assumptions made."""
    T = len(pattern)
    dt = t_obs / T

    # 1. Centroid → beam angle estimate
    #    In paraxial limit: centroid ≈ d_total * tan(beam_angle)
    #    We don't know d_total, but centroid gives a rough angle
    centroid_x = np.mean(pattern[:, 0])
    centroid_y = np.mean(pattern[:, 1])

    # 2. FFT → speed estimates
    #    Find the n_prisms strongest frequency peaks
    freqs = np.fft.rfftfreq(T, d=dt)
    fx = np.abs(np.fft.rfft(pattern[:, 0] - centroid_x))
    fy = np.abs(np.fft.rfft(pattern[:, 1] - centroid_y))
    power = fx + fy
    power[0] = 0  # kill DC

    speed_estimates = []
    power_copy = power.copy()
    for _ in range(n_prisms):
        idx = np.argmax(power_copy)
        speed_estimates.append(freqs[idx])
        # Zero out neighborhood
        lo = max(1, idx - 3)
        hi = min(len(power_copy), idx + 4)
        power_copy[lo:hi] = 0

    speed_estimates.sort(key=lambda x: abs(x))

    # 3. RMS radius → amplitude scale
    rms_x = np.std(pattern[:, 0])
    rms_y = np.std(pattern[:, 1])
    rms_total = np.sqrt(rms_x**2 + rms_y**2)

    # 4. Estimate beam angles from centroid
    #    Assume d_total ~ 120 (reasonable for default-ish geometry)
    #    beam_angle ≈ arctan(centroid / d_total) in degrees
    d_guess = 120.0
    beam_ax_est = np.degrees(np.arctan2(centroid_x, d_guess))
    beam_ay_est = np.degrees(np.arctan2(centroid_y, d_guess))

    # 5. Estimate workpiece distance from pattern scale
    #    pattern_rms ≈ (n_g - 1) * alpha_rms * d_W
    #    Assume alpha_rms ~ 8 deg, n_g ~ 1.5
    alpha_guess_rad = np.radians(8.0)
    ng_guess = 1.5
    d_w_est = rms_total / ((ng_guess - 1) * alpha_guess_rad)

    return {
        'speeds': speed_estimates,
        'centroid': (centroid_x, centroid_y),
        'beam_ax': beam_ax_est,
        'beam_ay': beam_ay_est,
        'rms': rms_total,
        'd_w_est': d_w_est,
    }


class Objective:
    def __init__(self, build_fn, target):
        self.build_fn = build_fn
        self.target = target
    def __call__(self, x):
        try:
            p = self.build_fn(x)
            return float(np.mean((fast_forward(p, 200, T_OBS) - self.target)**2))
        except:
            return 1e6


def build_full(x):
    """19-D: everything."""
    geo = SystemGeometry(workpiece_distance=x[12], inter_prism_gap=x[13],
                         beam_angle_x=x[14], beam_angle_y=x[15],
                         beam_pos_x=x[16], beam_pos_y=x[17])
    return PrismParameters(3, x[:3].tolist(), x[3:6].tolist(), x[6:9].tolist(),
                           glass_indices=x[9:12].tolist(), geometry=geo)


def build_17d(x):
    """17-D: no beam positions."""
    geo = SystemGeometry(workpiece_distance=x[12], inter_prism_gap=x[13],
                         beam_angle_x=x[14], beam_angle_y=x[15])
    return PrismParameters(3, x[:3].tolist(), x[3:6].tolist(), x[6:9].tolist(),
                           glass_indices=x[9:12].tolist(), geometry=geo)


def run_smart(label, build_fn, bounds, true_vec, hints, n_restarts=12,
              de_maxiter=400, de_popsize=35):
    """Smart solver: use hints to seed first restart."""
    obj = Objective(build_fn, TARGET)
    dim = len(bounds)
    t0 = time.time()
    best_x, best_mse = None, float('inf')

    # Build seed from hints
    seed_x = np.zeros(dim)
    # Speeds from FFT
    for i, s in enumerate(hints['speeds'][:3]):
        seed_x[i] = s
    # Angles: start at moderate values
    seed_x[3:6] = [8.0, -5.0, 3.0]  # rough guesses
    seed_x[6:9] = [2.0, 5.0, -3.0]
    # Glass: start near typical
    if dim >= 12:
        seed_x[9:12] = [1.50, 1.55, 1.60]
    # Distances from hint
    if dim >= 14:
        seed_x[12] = hints['d_w_est']
        seed_x[13] = 6.0  # default gap guess
    # Beam angles from centroid
    if dim >= 16:
        seed_x[14] = hints['beam_ax']
        seed_x[15] = hints['beam_ay']
    # Beam positions: start at zero
    if dim >= 18:
        seed_x[16] = 0.0
        seed_x[17] = 0.0

    # Clip seed to bounds
    for i in range(dim):
        seed_x[i] = np.clip(seed_x[i], bounds[i][0], bounds[i][1])

    print(f"  Hints: speeds={[f'{s:.2f}' for s in hints['speeds']]}, "
          f"beam_ax={hints['beam_ax']:.1f}, beam_ay={hints['beam_ay']:.1f}, "
          f"d_W_est={hints['d_w_est']:.0f}", flush=True)

    for r in range(n_restarts):
        x0 = seed_x if r == 0 else None
        rt0 = time.time()
        res = differential_evolution(obj, bounds, seed=42 + r,
                                     maxiter=de_maxiter, popsize=de_popsize,
                                     tol=1e-14, polish=False,
                                     mutation=(0.5, 1.5), recombination=0.8,
                                     x0=x0,
                                     workers=N_WORKERS, updating='deferred')
        dt = time.time() - rt0
        tag = " *" if res.fun < best_mse else ""
        if res.fun < best_mse:
            best_x, best_mse = res.x.copy(), res.fun
        print(f"    restart {r+1}/{n_restarts}: MSE={res.fun:.2e} ({dt:.0f}s){tag}",
              flush=True)

    # Heavy NM polish
    if best_x is not None:
        print(f"    NM polishing...", flush=True)
        res2 = minimize(obj, best_x, method='Nelder-Mead',
                        options={'maxiter': 30000, 'fatol': 1e-15, 'xatol': 1e-15})
        if res2.fun < best_mse:
            best_x, best_mse = res2.x.copy(), res2.fun
        print(f"    NM: MSE={best_mse:.2e}", flush=True)

    # Permutation search with glass
    if best_x is not None:
        print(f"    Permutation search...", flush=True)
        sp = best_x[:3]; ax = best_x[3:6]; ay = best_x[6:9]
        extra = best_x[9:]
        for perm in permutations(range(3)):
            if list(perm) == [0, 1, 2]:
                continue
            sp_p = sp[list(perm)]; ax_p = ax[list(perm)]; ay_p = ay[list(perm)]
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
    # Extract hints from the pattern
    print("=" * 70)
    print("SMART FULL RECOVERY (P=3, hints from pattern)")
    print("=" * 70)

    hints = extract_hints(TARGET, N_PRISMS, T_OBS)
    print(f"\nExtracted hints:")
    print(f"  FFT speeds: {hints['speeds']}")
    print(f"  True speeds: {TRUE_SPEEDS}")
    print(f"  Centroid: ({hints['centroid'][0]:.2f}, {hints['centroid'][1]:.2f})")
    print(f"  Beam angle est: ({hints['beam_ax']:.2f}, {hints['beam_ay']:.2f}) deg")
    print(f"  True beam angles: (10.0, 5.0) deg")
    print(f"  d_W estimate: {hints['d_w_est']:.0f}")
    print(f"  True d_W: 100.0")

    bounds_base = [(-3.5, 3.5)]*3 + [(-18, 18)]*6

    # 17-D: glass + distances + beam angles (no beam positions)
    bounds_17 = bounds_base + [(1.3, 1.8)]*3 + [(50, 200)] + [(2, 15)] + [(-25, 25)]*2
    true_17 = np.array(TRUE_SPEEDS + TRUE_AX + TRUE_AY + TRUE_GLASS +
                       [100.0, 6.0, 10.0, 5.0])

    # 19-D: everything
    bounds_19 = bounds_17 + [(-5, 5)]*2
    true_19 = np.array(TRUE_SPEEDS + TRUE_AX + TRUE_AY + TRUE_GLASS +
                       [100.0, 6.0, 10.0, 5.0, 0.0, 0.0])

    results = []

    for label, build_fn, bounds, true_vec, nr, mi, ps in [
        ("17-D: glass+dist+beam_angles", build_17d, bounds_17, true_17, 12, 400, 35),
        ("19-D: everything",              build_full, bounds_19, true_19, 15, 500, 40),
    ]:
        dim = len(bounds)
        print(f"\n{label} ({dim} free parameters)")

        best_x, mse, errors, wt = run_smart(label, build_fn, bounds, true_vec, hints,
                                             nr, mi, ps)

        if best_x is not None:
            max_err = np.max(errors)
            print(f"\n  RESULT: MSE={mse:.2e}, max_err={max_err:.2e} ({wt:.0f}s)")

            names = ['N1','N2','N3','ax1','ax2','ax3','ay1','ay2','ay3',
                     'ng1','ng2','ng3','d_W','gap','beam_ax','beam_ay']
            if dim >= 18:
                names += ['beam_px','beam_py']

            for i, (n, e) in enumerate(zip(names, errors)):
                status = "OK" if e < 0.01 else "CLOSE" if e < 1.0 else "BAD"
                print(f"    {n:10s}: err={e:.2e}  true={true_vec[i]:.4f}  "
                      f"rec={best_x[i]:.4f}  [{status}]")

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

    out_path = os.path.join(os.path.dirname(__file__), 'full_recovery_smart_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
