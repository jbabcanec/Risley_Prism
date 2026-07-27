#!/usr/bin/env python3
"""
THE GUN: FFT speeds + DE on 6-D angle space + TRF polish.

No ML. No Adam. No screening. No continuation. No basin hopping.
Just physics (FFT) + global optimization (DE) + precision (TRF).

For each case:
  1. FFT → speed magnitudes (top-3 peaks)
  2. For each of 8 sign combos:
     - Fix speeds, run scipy.optimize.differential_evolution on 6 angles
     - DE is a GLOBAL optimizer — finds the right basin regardless of init
  3. TRF polish from best DE result → machine precision
"""
import sys, os, time
import numpy as np
from scipy.optimize import least_squares, differential_evolution
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'reverse_problem_v2'))
sys.path.insert(0, os.path.dirname(__file__))

from ml_staged_solver import (
    N_PAR, LO, HI, RG, canon, P,
    extract_speeds_and_peaks, T_PTS, T_OBS,
)
from solve_preconditioned import vec2pat

rng_gen = np.random.default_rng(2026)
cases = []
for i in range(30):
    v = (LO + RG * rng_gen.random(N_PAR).astype(np.float32)).astype(np.float64)
    for j in range(3):
        if abs(v[j]) < 0.15: v[j] = np.copysign(0.15, v[j])
    cases.append(canon(v))


def solve_de(speeds, target_flat, fixed, maxiter=1000, popsize=25, tol=1e-12):
    """DE on 6 angle params (αₓ₁,αₓ₂,αₓ₃,αᵧ₁,αᵧ₂,αᵧ₃), speeds fixed."""
    bounds_6d = [(float(LO[3+i]), float(HI[3+i])) for i in range(3)] + \
                [(float(LO[6+i]), float(HI[6+i])) for i in range(3)]

    def objective(angles_6d):
        theta = np.zeros(9)
        theta[:3] = speeds
        theta[3:6] = angles_6d[:3]
        theta[6:9] = angles_6d[3:]
        full = np.concatenate([theta, fixed])
        try:
            return float(np.mean((vec2pat(full).reshape(-1) - target_flat)**2))
        except:
            return 1e6

    result = differential_evolution(
        objective, bounds_6d,
        maxiter=maxiter, popsize=popsize, tol=tol,
        seed=42, polish=False, disp=False,
        mutation=(0.5, 1.5), recombination=0.9,
        strategy='best1bin',
    )
    best_angles = result.x
    best_mse = result.fun

    # TRF polish (9-D: speeds + angles)
    theta_init = np.zeros(9)
    theta_init[:3] = speeds
    theta_init[3:6] = best_angles[:3]
    theta_init[6:9] = best_angles[3:]
    lo9 = LO[:9].astype(np.float64)
    hi9 = HI[:9].astype(np.float64)

    def residual(x9):
        return vec2pat(np.concatenate([x9, fixed])).reshape(-1) - target_flat

    try:
        res = least_squares(residual, theta_init, jac='2-point',
            bounds=(lo9, hi9), method='trf',
            ftol=1e-15, xtol=1e-15, gtol=1e-15, max_nfev=3000)
        return res.x, float(np.mean(res.fun**2)), result.nfev
    except:
        return theta_init, best_mse, result.nfev


print(f"{'='*80}")
print("DE ON 6-D ANGLES + TRF POLISH — 9-D (30 cases)")
print(f"{'='*80}\n")

perfect = 0
t0 = time.time()

for ci, tc in enumerate(cases):
    pat = vec2pat(tc)
    tf = pat.reshape(-1)
    pf, _ = extract_speeds_and_peaks(pat)
    fixed = tc[9:].copy()

    best_mse = 1e30; best_x = None; total_nfev = 0

    t_case = time.time()
    for bits in range(8):
        signs = np.array([(1.0 if (bits>>j)&1==0 else -1.0) for j in range(P)], np.float64)
        speeds = signs * np.sort(pf)[::-1].astype(np.float64)

        x, mse, nfev = solve_de(speeds, tf, fixed, maxiter=800, popsize=20)
        total_nfev += nfev
        if mse < best_mse:
            best_mse = mse; best_x = x.copy()
        if best_mse < 1e-15:
            break

    dt = time.time() - t_case
    err = float(np.max(np.abs(best_x - tc[:9]))) if best_x is not None else 999
    ok = err < 1e-3
    if ok: perfect += 1

    tag = 'PERFECT' if ok else 'FAIL'
    print(f"  {ci+1:2d}: {tag}  err={err:.1e}  MSE={best_mse:.1e}  "
          f"nfev={total_nfev}  {dt:.0f}s", flush=True)

elapsed = time.time() - t0
print(f"\n{'='*80}")
print(f"  PERFECT: {perfect}/30")
print(f"  Time: {elapsed:.0f}s ({elapsed/30:.0f}s/case)")
print(f"{'='*80}")
