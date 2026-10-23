#!/usr/bin/env python3
"""
Random restart baseline for 9-D: how many starts does it take to find the solution?
Tests 5 representative cases with up to 500 random starts each.
This establishes the ceiling: if random restarts solve it, the problem is init-limited.
"""
import sys, os, time
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'reverse_problem_v2'))
sys.path.insert(0, os.path.dirname(__file__))

from ml_staged_solver import N_PAR, LO, HI, RG, canon, P, T_PTS, T_OBS
from solve_preconditioned import vec2pat

rng_gen = np.random.default_rng(2026)
cases = []
for i in range(30):
    v = (LO + RG * rng_gen.random(N_PAR).astype(np.float32)).astype(np.float64)
    for j in range(3):
        if abs(v[j]) < 0.15: v[j] = np.copysign(0.15, v[j])
    cases.append(canon(v))

print(f"{'='*80}")
print("RANDOM RESTART CEILING — 9-D (500 starts per case, 5 cases)")
print(f"{'='*80}\n")

# Test cases: pick some that ML failed on
test_indices = [2, 3, 5, 6, 9]  # 0-indexed

for ci in test_indices:
    tc = cases[ci]
    pat = vec2pat(tc)
    target = pat.reshape(-1)
    fixed = tc[9:].copy()
    lo9 = LO[:9].astype(np.float64)
    hi9 = HI[:9].astype(np.float64)

    def make_res(fix):
        def residual(x9):
            return vec2pat(np.concatenate([x9, fix])).reshape(-1) - target
        return residual
    res_fn = make_res(fixed)

    rng = np.random.default_rng(42 + ci)
    best_mse = 1e30
    best_x = None
    solved_at = None

    t0 = time.time()
    for trial in range(500):
        # Random init: speeds from true ±0.5, angles random
        x0 = np.zeros(9)
        x0[:3] = tc[:3] + rng.uniform(-0.3, 0.3, 3)  # near-true speeds
        x0[3:9] = LO[3:9] + RG[3:9] * rng.random(6).astype(np.float64)  # random angles
        x0 = np.clip(x0, lo9, hi9)

        try:
            res = least_squares(res_fn, x0, jac='2-point',
                bounds=(lo9, hi9), method='trf',
                ftol=1e-15, xtol=1e-15, gtol=1e-15, max_nfev=1000)
            mse = float(np.mean(res.fun**2))
            if mse < best_mse:
                best_mse = mse
                best_x = res.x.copy()
            if mse < 1e-15 and solved_at is None:
                solved_at = trial + 1
        except Exception:
            pass

        if trial in [9, 49, 99, 199, 499]:
            err = float(np.max(np.abs(best_x - tc[:9]))) if best_x is not None else 999
            tag = "PERFECT" if err < 1e-3 else f"err={err:.2e}"
            elapsed = time.time() - t0
            print(f"  Case {ci+1}, {trial+1:3d} starts: MSE={best_mse:.2e}, {tag}, "
                  f"{elapsed:.0f}s", flush=True)
            if err < 1e-3:
                break

    dt = time.time() - t0
    err = float(np.max(np.abs(best_x - tc[:9]))) if best_x is not None else 999
    tag = "PERFECT" if err < 1e-3 else "FAIL"
    print(f"  Case {ci+1} FINAL: {tag}, err={err:.2e}, MSE={best_mse:.2e}, "
          f"solved_at={'N/A' if solved_at is None else solved_at}, {dt:.0f}s\n", flush=True)

print(f"{'='*80}")
