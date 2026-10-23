#!/usr/bin/env python3
"""
THE GUN: One pipeline, no patchwork.

Physics determines what physics can:
  - FFT → speed magnitudes
  - Harmonic LS → αᵧ (analytical, two options per prism via 180° flip)

Optimization handles the rest:
  - 3-D DE on (αₓ₁, αₓ₂, αₓ₃) with αᵧ fixed from harmonic LS
  - 3-D is 46000× smaller search space than 6-D

Precision:
  - TRF on full 9-D from best DE result → machine precision

Total: 8 speed signs × 8 αᵧ flips × 3-D DE = 64 DE runs.
Each 3-D DE: ~2000 evals × 1ms = 2s. Total: ~130s + TRF.
"""
import sys, os, time
import numpy as np
from scipy.optimize import least_squares, differential_evolution
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'reverse_problem_v2'))
sys.path.insert(0, os.path.dirname(__file__))

from ml_staged_solver import (
    N_PAR, LO, HI, RG, canon, P,
    extract_speeds_and_peaks, T_PTS, T_OBS, SRC, THK,
)
from solve_preconditioned import vec2pat


def harmonic_phases(pattern, speeds):
    """Harmonic LS → αᵧ from phase at each speed frequency."""
    T = T_PTS
    t = np.arange(T) * (T_OBS / T)
    abs_f = np.abs(speeds)

    X = np.zeros((T, 2*P + 1))
    for i in range(P):
        X[:, 2*i] = np.cos(2*np.pi*abs_f[i]*t)
        X[:, 2*i+1] = np.sin(2*np.pi*abs_f[i]*t)
    X[:, -1] = 1.0

    bx, _, _, _ = np.linalg.lstsq(X, pattern[:, 0], rcond=None)

    phases = []
    for i in range(P):
        phase = np.arctan2(-bx[2*i+1], bx[2*i])
        # Adjust for speed sign
        if speeds[i] < 0:
            ay = -np.degrees(phase)
        else:
            ay = np.degrees(phase)
        phases.append(ay)
    return phases


def solve_gun(pattern, fixed, pf):
    """The gun: FFT speeds → harmonic αᵧ → 3-D DE on αₓ → TRF polish."""
    target_flat = pattern.reshape(-1)
    lo9 = LO[:9].astype(np.float64)
    hi9 = HI[:9].astype(np.float64)
    lo_ax = [float(LO[3]), float(LO[4]), float(LO[5])]
    hi_ax = [float(HI[3]), float(HI[4]), float(HI[5])]
    bounds_3d = list(zip(lo_ax, hi_ax))

    best_mse = 1e30
    best_x = None
    total_nfev = 0

    speed_mags = np.sort(pf)[::-1].astype(np.float64)

    for sign_bits in range(8):
        signs = np.array([(1.0 if (sign_bits>>j)&1==0 else -1.0)
                          for j in range(P)], np.float64)
        speeds = signs * speed_mags

        # Harmonic LS → αᵧ
        ay_base = harmonic_phases(pattern, speeds)

        for flip_bits in range(8):
            # Apply 180° flips
            ay = []
            for j in range(P):
                if (flip_bits >> j) & 1:
                    v = ay_base[j] + 180.0
                    if v > 180: v -= 360
                    ay.append(v)
                else:
                    ay.append(ay_base[j])

            # Clip αᵧ to bounds
            ay_clipped = [np.clip(a, float(LO[6+j]), float(HI[6+j]))
                          for j, a in enumerate(ay)]

            # 3-D DE on αₓ only
            def objective(ax_3d):
                theta = np.zeros(9)
                theta[:3] = speeds
                theta[3:6] = ax_3d
                theta[6:9] = ay_clipped
                full = np.concatenate([theta, fixed])
                try:
                    return float(np.mean((vec2pat(full).reshape(-1) - target_flat)**2))
                except:
                    return 1e6

            try:
                result = differential_evolution(
                    objective, bounds_3d,
                    maxiter=300, popsize=15, tol=1e-12,
                    seed=42 + sign_bits*8 + flip_bits,
                    polish=False, disp=False,
                    mutation=(0.5, 1.5), recombination=0.9,
                )
                total_nfev += result.nfev

                if result.fun < best_mse:
                    # Build full 9-D and TRF
                    theta = np.zeros(9)
                    theta[:3] = speeds
                    theta[3:6] = result.x
                    theta[6:9] = ay_clipped

                    def residual(x9):
                        return vec2pat(np.concatenate([x9, fixed])).reshape(-1) - target_flat

                    try:
                        res = least_squares(residual, theta, jac='2-point',
                            bounds=(lo9, hi9), method='trf',
                            ftol=1e-15, xtol=1e-15, gtol=1e-15, max_nfev=3000)
                        mse = float(np.mean(res.fun**2))
                        if mse < best_mse:
                            best_mse = mse
                            best_x = res.x.copy()
                    except:
                        if result.fun < best_mse:
                            best_mse = result.fun
                            best_x = theta.copy()
            except:
                pass

            if best_mse < 1e-15:
                return best_x, best_mse, total_nfev

    return best_x, best_mse, total_nfev


# Generate 30 cases
rng = np.random.default_rng(2026)
cases = []
for i in range(30):
    v = (LO + RG * rng.random(N_PAR).astype(np.float32)).astype(np.float64)
    for j in range(3):
        if abs(v[j]) < 0.15: v[j] = np.copysign(0.15, v[j])
    cases.append(canon(v))

print(f"{'='*80}")
print("THE GUN: FFT → Harmonic αᵧ → 3-D DE αₓ → TRF (30 cases)")
print(f"{'='*80}\n")

perfect = 0
t0 = time.time()

for ci, tc in enumerate(cases):
    pat = vec2pat(tc)
    pf, _ = extract_speeds_and_peaks(pat)
    fixed = tc[9:].copy()

    t_case = time.time()
    x, mse, nfev = solve_gun(pat, fixed, pf)
    dt = time.time() - t_case

    err = float(np.max(np.abs(x - tc[:9]))) if x is not None else 999
    ok = err < 1e-3
    if ok: perfect += 1

    tag = 'PERFECT' if ok else 'FAIL'
    print(f"  {ci+1:2d}: {tag}  err={err:.1e}  MSE={mse:.1e}  "
          f"nfev={nfev:,}  {dt:.0f}s", flush=True)

elapsed = time.time() - t0
print(f"\n{'='*80}")
print(f"  PERFECT: {perfect}/30")
print(f"  Time: {elapsed:.0f}s ({elapsed/30:.0f}s/case)")
print(f"{'='*80}")
