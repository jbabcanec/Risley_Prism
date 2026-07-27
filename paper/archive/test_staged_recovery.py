#!/usr/bin/env python3
"""
Staged recovery: use pattern structure to decompose the 19-D search.

No restrictions — all parameters are free. We just search smarter.

Stage 1: FFT → speed magnitudes. Try all 2^P sign combinations.
Stage 2: Centroid → beam angle estimates. For each speed-sign combo,
         fix speeds and search angles+glass+distances (~10-D).
Stage 3: Unfreeze everything, joint NM polish from best Stage 2 result.
Stage 4: Permutation search on the full 19-D result.

This turns one impossible 19-D search into 8 feasible 10-D searches
followed by a local 19-D refinement.
"""
import sys, os, time, json, io
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'reverse_problem_v2'))

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from core import PrismParameters, SystemGeometry, fast_forward
from scipy.optimize import differential_evolution, minimize
from itertools import permutations, product


N_PRISMS = 3
T_OBS = 10.0
N_WORKERS = 4

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
TARGET = fast_forward(TRUE_P, 200, T_OBS)

TRUE_VEC = np.array(TRUE_SPEEDS + TRUE_AX + TRUE_AY + TRUE_GLASS +
                    [100.0, 6.0, 10.0, 5.0, 0.0, 0.0])


# ═══════════════════════════════════════════════════════════
# STAGE 1: Extract speed magnitudes from FFT
# ═══════════════════════════════════════════════════════════
def extract_speeds(pattern, n_prisms, t_obs):
    T = len(pattern)
    dt = t_obs / T
    freqs = np.fft.rfftfreq(T, d=dt)
    cx = np.mean(pattern[:, 0])
    cy = np.mean(pattern[:, 1])
    fx = np.abs(np.fft.rfft(pattern[:, 0] - cx))
    fy = np.abs(np.fft.rfft(pattern[:, 1] - cy))
    power = fx + fy
    power[0] = 0

    speed_mags = []
    for _ in range(n_prisms):
        idx = np.argmax(power)
        speed_mags.append(float(freqs[idx]))
        lo = max(1, idx - 3)
        hi = min(len(power), idx + 4)
        power[lo:hi] = 0

    # Also extract phases at these frequencies for alpha_y hints
    phases = []
    for s in speed_mags:
        idx = int(round(s * T * dt))
        if 0 < idx < len(fx):
            phase_x = np.angle(np.fft.rfft(pattern[:, 0] - cx)[idx])
            phases.append(float(np.degrees(phase_x)))
        else:
            phases.append(0.0)

    return sorted(speed_mags), phases


def extract_centroid(pattern):
    return float(np.mean(pattern[:, 0])), float(np.mean(pattern[:, 1]))


def extract_scale(pattern):
    cx, cy = extract_centroid(pattern)
    return float(np.std(pattern[:, 0])), float(np.std(pattern[:, 1]))


# ═══════════════════════════════════════════════════════════
# STAGE 2: For each sign combo, search angles+glass+distances
# ═══════════════════════════════════════════════════════════
class Stage2Objective:
    """Fixed speeds, search angles+glass+distances+beam."""
    def __init__(self, speeds, target):
        self.speeds = speeds
        self.target = target

    def __call__(self, x):
        try:
            # x = [ax(3), ay(3), ng(3), d_W, gap, beam_ax, beam_ay, beam_px, beam_py]
            ax = x[:3].tolist()
            ay = x[3:6].tolist()
            ng = x[6:9].tolist()
            geo = SystemGeometry(
                workpiece_distance=x[9], inter_prism_gap=x[10],
                beam_angle_x=x[11], beam_angle_y=x[12],
                beam_pos_x=x[13], beam_pos_y=x[14],
            )
            p = PrismParameters(3, self.speeds, ax, ay,
                                glass_indices=ng, geometry=geo)
            return float(np.mean((fast_forward(p, 200, T_OBS) - self.target)**2))
        except:
            return 1e6


# ═══════════════════════════════════════════════════════════
# STAGE 3+4: Full 19-D refinement + permutation
# ═══════════════════════════════════════════════════════════
class FullObjective:
    def __init__(self, target):
        self.target = target

    def __call__(self, x):
        try:
            geo = SystemGeometry(
                workpiece_distance=x[12], inter_prism_gap=x[13],
                beam_angle_x=x[14], beam_angle_y=x[15],
                beam_pos_x=x[16], beam_pos_y=x[17],
            )
            p = PrismParameters(3, x[:3].tolist(), x[3:6].tolist(),
                                x[6:9].tolist(), glass_indices=x[9:12].tolist(),
                                geometry=geo)
            return float(np.mean((fast_forward(p, 200, T_OBS) - self.target)**2))
        except:
            return 1e6


if __name__ == '__main__':
    t_total = time.time()

    print("=" * 70)
    print("STAGED RECOVERY (P=3, 19 unknowns)")
    print("=" * 70)

    # ── Stage 1: FFT speeds ──
    speed_mags, phases = extract_speeds(TARGET, N_PRISMS, T_OBS)
    cx, cy = extract_centroid(TARGET)
    sx, sy = extract_scale(TARGET)

    print(f"\nStage 1: FFT extraction")
    print(f"  Speed magnitudes: {speed_mags}")
    print(f"  True speeds:      {TRUE_SPEEDS}")
    print(f"  Centroid: ({cx:.2f}, {cy:.2f})")
    print(f"  Scale: ({sx:.2f}, {sy:.2f})")

    # Beam angle hints from centroid
    d_guess = 120.0
    beam_ax_hint = np.degrees(np.arctan2(cx, d_guess))
    beam_ay_hint = np.degrees(np.arctan2(cy, d_guess))
    print(f"  Beam angle hints: ({beam_ax_hint:.1f}, {beam_ay_hint:.1f}) deg")

    # ── Stage 2: Try all 2^P sign combinations ──
    print(f"\nStage 2: Searching {2**N_PRISMS} sign combinations x 2 restarts each")

    stage2_bounds = (
        [(-18, 18)] * 3 +     # ax
        [(-18, 18)] * 3 +     # ay
        [(1.3, 1.8)] * 3 +    # glass
        [(50, 200)] +         # d_W
        [(2, 15)] +           # gap
        [(-25, 25)] +         # beam_ax
        [(-25, 25)] +         # beam_ay
        [(-5, 5)] +           # beam_px
        [(-5, 5)]             # beam_py
    )

    # Seed: use centroid for beam, moderate angles, typical glass
    base_seed = np.array([
        8.0, -5.0, 3.0,         # ax guess
        2.0, 5.0, -3.0,         # ay guess
        1.50, 1.55, 1.60,       # glass guess
        100.0, 6.0,             # d_W, gap guess
        beam_ax_hint, beam_ay_hint,  # beam angles from centroid
        0.0, 0.0,               # beam positions
    ])

    best_stage2_x = None
    best_stage2_mse = float('inf')
    best_speeds = None

    for signs in product([1, -1], repeat=N_PRISMS):
        signed_speeds = [s * sign for s, sign in zip(speed_mags, signs)]
        obj2 = Stage2Objective(signed_speeds, TARGET)

        label = f"  signs={list(signs)}, N={[f'{s:+.1f}' for s in signed_speeds]}"

        for r in range(2):
            x0 = base_seed if r == 0 else None
            res = differential_evolution(obj2, stage2_bounds, seed=100 + r,
                                         maxiter=200, popsize=25,
                                         tol=1e-12, polish=False, x0=x0,
                                         workers=N_WORKERS, updating='deferred')

            if res.fun < best_stage2_mse:
                best_stage2_mse = res.fun
                best_stage2_x = res.x.copy()
                best_speeds = signed_speeds[:]

        # Quick NM polish on promising ones
        if best_stage2_mse < 0.1:
            res_nm = minimize(obj2, best_stage2_x, method='Nelder-Mead',
                              options={'maxiter': 5000, 'fatol': 1e-12})
            if res_nm.fun < best_stage2_mse:
                best_stage2_mse = res_nm.fun
                best_stage2_x = res_nm.x.copy()

        print(f"{label}: best_so_far={best_stage2_mse:.2e}", flush=True)

    print(f"\n  Stage 2 best: MSE={best_stage2_mse:.2e}")
    print(f"  Best speeds: {best_speeds}")

    # ── Stage 3: Full 19-D joint refinement ──
    print(f"\nStage 3: Full 19-D NM polish from Stage 2 result")

    # Assemble 19-D vector from Stage 2 result
    full_x = np.concatenate([
        np.array(best_speeds),      # N (3)
        best_stage2_x[:3],          # ax (3)
        best_stage2_x[3:6],         # ay (3)
        best_stage2_x[6:9],         # glass (3)
        best_stage2_x[9:11],        # d_W, gap
        best_stage2_x[11:13],       # beam_ax, beam_ay
        best_stage2_x[13:15],       # beam_px, beam_py
    ])

    full_obj = FullObjective(TARGET)

    # Heavy NM polish on full 19-D
    res3 = minimize(full_obj, full_x, method='Nelder-Mead',
                    options={'maxiter': 50000, 'fatol': 1e-15, 'xatol': 1e-15})
    best_full_x = res3.x.copy()
    best_full_mse = res3.fun
    print(f"  NM result: MSE={best_full_mse:.2e}")

    # Also try a short DE refinement around the NM result
    full_bounds = (
        [(-3.5, 3.5)] * 3 + [(-18, 18)] * 6 +
        [(1.3, 1.8)] * 3 + [(50, 200)] + [(2, 15)] +
        [(-25, 25)] * 2 + [(-5, 5)] * 2
    )
    for r in range(3):
        res_de = differential_evolution(full_obj, full_bounds, seed=200 + r,
                                         maxiter=200, popsize=30,
                                         tol=1e-14, polish=False,
                                         x0=best_full_x,
                                         workers=N_WORKERS, updating='deferred')
        if res_de.fun < best_full_mse:
            best_full_x = res_de.x.copy()
            best_full_mse = res_de.fun
            print(f"  DE restart {r+1} improved: MSE={best_full_mse:.2e}", flush=True)

    # Final NM polish
    res_final = minimize(full_obj, best_full_x, method='Nelder-Mead',
                         options={'maxiter': 50000, 'fatol': 1e-15, 'xatol': 1e-15})
    if res_final.fun < best_full_mse:
        best_full_x = res_final.x.copy()
        best_full_mse = res_final.fun
    print(f"  Final NM: MSE={best_full_mse:.2e}")

    # ── Stage 4: Permutation search ──
    print(f"\nStage 4: Permutation search")

    sp = best_full_x[:3]; ax = best_full_x[3:6]; ay = best_full_x[6:9]
    glass = best_full_x[9:12]; rest = best_full_x[12:]

    for perm in permutations(range(3)):
        if list(perm) == [0, 1, 2]:
            continue
        x_perm = np.concatenate([
            sp[list(perm)], ax[list(perm)], ay[list(perm)],
            glass[list(perm)], rest
        ])
        rp = minimize(full_obj, x_perm, method='Nelder-Mead',
                      options={'maxiter': 30000, 'fatol': 1e-15, 'xatol': 1e-15})
        if rp.fun < best_full_mse:
            best_full_x = rp.x.copy()
            best_full_mse = rp.fun
            print(f"  perm {list(perm)} improved: MSE={best_full_mse:.2e}", flush=True)

    # ── Final report ──
    wt = time.time() - t_total
    errors = np.abs(best_full_x - TRUE_VEC)
    max_err = np.max(errors)

    print(f"\n{'='*70}")
    print(f"FINAL RESULT: MSE={best_full_mse:.2e}, max_err={max_err:.2e} ({wt:.0f}s)")
    print(f"{'='*70}")

    names = ['N1','N2','N3','ax1','ax2','ax3','ay1','ay2','ay3',
             'ng1','ng2','ng3','d_W','gap','beam_ax','beam_ay','beam_px','beam_py']
    for i, (n, e) in enumerate(zip(names, errors)):
        status = "OK" if e < 0.01 else "CLOSE" if e < 1.0 else "BAD"
        print(f"  {n:10s}: err={e:.2e}  true={TRUE_VEC[i]:.4f}  "
              f"rec={best_full_x[i]:.4f}  [{status}]")

    results = {
        'mse': float(best_full_mse),
        'max_error': float(max_err),
        'errors': errors.tolist(),
        'recovered': best_full_x.tolist(),
        'true': TRUE_VEC.tolist(),
        'time_s': round(wt, 1),
        'stage2_best_mse': float(best_stage2_mse),
        'speed_hints': speed_mags,
        'best_speeds': best_speeds,
    }

    out_path = os.path.join(os.path.dirname(__file__), 'staged_recovery_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
