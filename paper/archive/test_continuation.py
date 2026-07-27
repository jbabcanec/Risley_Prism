#!/usr/bin/env python3
"""
Homotopy continuation for 9-D Risley inverse problem.

Key idea: instead of solving F(θ) = target directly (hard — narrow basin),
deform the target smoothly:

  target_λ = (1-λ) * F(θ₀) + λ * target_real

At λ=0: θ₀ is the exact solution (trivially).
At λ=1: we need the real solution.

By taking small steps in λ, TRF converges at each step because
the target changes by a small amount.

This is the Davidenko continuation method — no brute force,
purely algorithmic, and mathematically guaranteed to work
if the path doesn't hit a bifurcation.
"""
import sys, os, time
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'reverse_problem_v2'))
sys.path.insert(0, os.path.dirname(__file__))

import torch
from ml_staged_solver import (
    AngleNet, RemainNet, ANG_MODEL, REM_MODEL,
    DEVICE, N_PAR, LO, HI, RG, canon,
    extract_speeds_and_peaks, _build_peak_feats_single, P,
    T_PTS, T_OBS,
)
from solve_preconditioned import vec2pat, ml_init


def continuation_solve_9d(init_9d, target_pat, fixed_params, n_steps=50):
    """
    Homotopy continuation for 9-D subproblem.

    Smoothly deforms from target_λ=0 = F(init) to target_λ=1 = target.
    """
    target_flat = target_pat.reshape(-1)
    lo9 = LO[:9].astype(np.float64)
    hi9 = HI[:9].astype(np.float64)

    # Evaluate F at initial guess
    theta0_full = np.concatenate([init_9d, fixed_params])
    f_init = vec2pat(theta0_full).reshape(-1)

    current = init_9d.copy()
    lambdas = np.linspace(0, 1, n_steps + 1)[1:]  # skip λ=0

    for lam in lambdas:
        current_target = (1 - lam) * f_init + lam * target_flat

        def residual(x9):
            theta = np.concatenate([x9, fixed_params])
            return vec2pat(theta).reshape(-1) - current_target

        try:
            res = least_squares(
                residual, current, jac='2-point',
                bounds=(lo9, hi9), method='trf',
                ftol=1e-15, xtol=1e-15, gtol=1e-15, max_nfev=200)
            current = res.x.copy()
        except Exception:
            break

    # Final polish at λ=1
    def residual_final(x9):
        theta = np.concatenate([x9, fixed_params])
        return vec2pat(theta).reshape(-1) - target_flat

    try:
        res = least_squares(
            residual_final, current, jac='2-point',
            bounds=(lo9, hi9), method='trf',
            ftol=1e-15, xtol=1e-15, gtol=1e-15, max_nfev=3000)
        return res.x, float(np.mean(res.fun**2))
    except Exception:
        return current, float(np.mean((vec2pat(np.concatenate([current, fixed_params])).reshape(-1) - target_flat)**2))


# Load models
print("Loading models...", flush=True)
ang = AngleNet(); rem = RemainNet()
ang.load_state_dict(torch.load(ANG_MODEL, map_location=DEVICE, weights_only=True))
rem.load_state_dict(torch.load(REM_MODEL, map_location=DEVICE, weights_only=True))
ang.to(DEVICE); rem.to(DEVICE)

# Generate 30 cases
rng = np.random.default_rng(2026)
cases = []
for i in range(30):
    v = (LO + RG * rng.random(N_PAR).astype(np.float32)).astype(np.float64)
    for j in range(3):
        if abs(v[j]) < 0.15: v[j] = np.copysign(0.15, v[j])
    cases.append(canon(v))

print(f"\n{'='*80}")
print("ML-ONLY vs HOMOTOPY CONTINUATION — 9-D TRF (30 cases)")
print(f"{'='*80}")
print("Steps: 20, 50, 100\n")

for n_steps in [20, 50, 100]:
    ml_perfect = 0
    cont_perfect = 0
    either = 0

    t0_all = time.time()

    for ci, tc in enumerate(cases):
        pat = vec2pat(tc)
        target = pat.reshape(-1)
        pf, pi_info = extract_speeds_and_peaks(pat)
        pk = _build_peak_feats_single(pat, pf, pi_info)
        fixed = tc[9:].copy()
        lo9 = LO[:9].astype(np.float64)
        hi9 = HI[:9].astype(np.float64)

        def make_res(fix):
            def residual(x9):
                return vec2pat(np.concatenate([x9, fix])).reshape(-1) - target
            return residual
        res_fn = make_res(fixed)

        ml_best = 1e30; ml_x = None
        cont_best = 1e30; cont_x = None

        for bits in range(8):
            signs = np.array([(1.0 if (bits>>j)&1==0 else -1.0) for j in range(P)], np.float64)
            speeds = signs * np.sort(pf)[::-1].astype(np.float64)
            ml = ml_init(ang, rem, speeds.astype(np.float32), pk, pat)

            # ML-only TRF
            try:
                res = least_squares(res_fn, ml[:9], jac='2-point',
                    bounds=(lo9, hi9), method='trf',
                    ftol=1e-15, xtol=1e-15, gtol=1e-15, max_nfev=2000)
                mse = float(np.mean(res.fun**2))
                if mse < ml_best: ml_best = mse; ml_x = res.x.copy()
            except Exception: pass

            # Continuation from ML init
            cx, cmse = continuation_solve_9d(ml[:9], pat, fixed, n_steps=n_steps)
            if cmse < cont_best: cont_best = cmse; cont_x = cx.copy()

        ml_err = float(np.max(np.abs(ml_x - tc[:9]))) if ml_x is not None else 999
        cont_err = float(np.max(np.abs(cont_x - tc[:9]))) if cont_x is not None else 999
        ml_ok = ml_err < 1e-3
        cont_ok = cont_err < 1e-3

        if ml_ok: ml_perfect += 1
        if cont_ok: cont_perfect += 1
        if ml_ok or cont_ok: either += 1

        tag = ""
        if cont_ok and not ml_ok: tag = " ***CONT SAVED***"

        sys.stdout.write(".")
        if tag: sys.stdout.write(f" (case {ci+1}{tag})")
        sys.stdout.flush()

    elapsed = time.time() - t0_all
    print(flush=True)
    print(f"  Steps={n_steps:3d}: ML={ml_perfect}/30  CONT={cont_perfect}/30  "
          f"EITHER={either}/30  time={elapsed:.0f}s ({elapsed/30:.1f}s/case)")

print(f"\n{'='*80}")
