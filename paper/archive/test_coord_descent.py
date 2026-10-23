#!/usr/bin/env python3
"""
Coordinate descent: solve for each prism's (αₓ, αᵧ) one at a time.

Key insight: A 2-D basin (one prism) is MUCH wider than a 9-D basin (all prisms).
Even if the paraxial init is off by 5°, a 2-D search can find the correct local minimum.

Algorithm:
  1. FFT → speeds (with sign combos)
  2. Harmonic LS → initial (αₓ, αᵧ) for each prism
  3. Coordinate descent: cycle through prisms, optimize 2-D per prism
     - For each prism: 2-D grid search (coarse) → TRF (fine)
     - Other prisms fixed at current best estimates
  4. Final joint 9-D TRF polish
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
    T_PTS, T_OBS, SRC, THK,
)
from solve_preconditioned import vec2pat, ml_init


def harmonic_init(speeds, pattern, fixed_params):
    """Get paraxial (αₓ, αᵧ) from harmonic LS for given signed speeds."""
    T = T_PTS
    t = np.arange(T) * (T_OBS / T)
    ng = fixed_params[:3]
    d_W = fixed_params[3]
    gap = fixed_params[4]

    X = np.zeros((T, 2*P + 1))
    for i in range(P):
        X[:, 2*i] = np.cos(2*np.pi*speeds[i]*t)
        X[:, 2*i+1] = np.sin(2*np.pi*speeds[i]*t)
    X[:, -1] = 1.0

    bx, _, _, _ = np.linalg.lstsq(X, pattern[:, 0], rcond=None)
    by, _, _, _ = np.linalg.lstsq(X, pattern[:, 1], rcond=None)

    theta = np.zeros(9)
    theta[:3] = speeds

    for i in range(P):
        d_eff = (P - 1 - i) * (THK + gap) + d_W
        amp = np.sqrt(bx[2*i]**2 + bx[2*i+1]**2 + by[2*i]**2 + by[2*i+1]**2)
        denom = d_eff * (ng[i] - 1.0)
        ax = np.degrees(np.arctan(amp / denom)) if denom > 1e-10 else 5.0

        phase = np.arctan2(-bx[2*i+1], bx[2*i])
        ay = np.degrees(phase)

        # Determine αₓ sign from the phase/amplitude relationship
        if bx[2*i] < 0:
            ax = -ax
            ay = ay + 180.0
            if ay > 180: ay -= 360

        theta[3+i] = np.clip(ax, float(LO[3+i]), float(HI[3+i]))
        theta[6+i] = np.clip(ay, float(LO[6+i]), float(HI[6+i]))

    return theta


def coord_descent_solve(init_9d, target, fixed_params, n_cycles=3):
    """
    Coordinate descent: optimize 2-D (αₓ, αᵧ) per prism, cycling.
    """
    target_flat = target.reshape(-1)
    lo9 = LO[:9].astype(np.float64)
    hi9 = HI[:9].astype(np.float64)
    current = init_9d.copy()

    for cycle in range(n_cycles):
        # Optimize each prism in order of amplitude (strongest first)
        amps = [abs(current[3+i]) for i in range(P)]
        order = sorted(range(P), key=lambda i: -amps[i])

        for pi in order:
            # 2-D optimization: αₓ[pi] and αᵧ[pi], everything else fixed
            def residual_2d(x2):
                theta = current.copy()
                theta[3+pi] = x2[0]  # αₓ
                theta[6+pi] = x2[1]  # αᵧ
                full = np.concatenate([theta, fixed_params])
                return vec2pat(full).reshape(-1) - target_flat

            lo2 = np.array([float(LO[3+pi]), float(LO[6+pi])])
            hi2 = np.array([float(HI[3+pi]), float(HI[6+pi])])
            x0_2d = np.array([current[3+pi], current[6+pi]])

            try:
                res = least_squares(
                    residual_2d, x0_2d, jac='2-point',
                    bounds=(lo2, hi2), method='trf',
                    ftol=1e-14, xtol=1e-14, gtol=1e-14, max_nfev=500)
                current[3+pi] = res.x[0]
                current[6+pi] = res.x[1]
            except Exception:
                pass

    # Final joint 9-D TRF
    def residual_full(x9):
        return vec2pat(np.concatenate([x9, fixed_params])).reshape(-1) - target_flat

    try:
        res = least_squares(
            residual_full, current, jac='2-point',
            bounds=(lo9, hi9), method='trf',
            ftol=1e-15, xtol=1e-15, gtol=1e-15, max_nfev=3000)
        return res.x, float(np.mean(res.fun**2))
    except Exception:
        r = vec2pat(np.concatenate([current, fixed_params])).reshape(-1) - target_flat
        return current, float(np.mean(r**2))


def coord_descent_grid(init_9d, target, fixed_params, n_cycles=2, grid_pts=19):
    """
    Coordinate descent with 2-D grid search per prism (coarse→fine).

    For each prism: scan (αₓ, αᵧ) on a coarse grid, then TRF from best.
    """
    target_flat = target.reshape(-1)
    lo9 = LO[:9].astype(np.float64)
    hi9 = HI[:9].astype(np.float64)
    current = init_9d.copy()

    for cycle in range(n_cycles):
        for pi in range(P):
            # Grid search: αₓ full range, αᵧ ±90° around current
            ax_grid = np.linspace(float(LO[3+pi]), float(HI[3+pi]), grid_pts)
            ay_center = current[6+pi]
            ay_grid = np.linspace(max(float(LO[6+pi]), ay_center - 90),
                                  min(float(HI[6+pi]), ay_center + 90), grid_pts)

            best_mse = 1e30
            best_ax, best_ay = current[3+pi], current[6+pi]

            for ax in ax_grid:
                for ay in ay_grid:
                    theta = current.copy()
                    theta[3+pi] = ax
                    theta[6+pi] = ay
                    try:
                        full = np.concatenate([theta, fixed_params])
                        r = vec2pat(full).reshape(-1) - target_flat
                        mse = float(np.mean(r**2))
                        if mse < best_mse:
                            best_mse = mse
                            best_ax, best_ay = ax, ay
                    except Exception:
                        pass

            # TRF from best grid point (2-D)
            current[3+pi] = best_ax
            current[6+pi] = best_ay

            def residual_2d(x2):
                theta = current.copy()
                theta[3+pi] = x2[0]
                theta[6+pi] = x2[1]
                return vec2pat(np.concatenate([theta, fixed_params])).reshape(-1) - target_flat

            lo2 = np.array([float(LO[3+pi]), float(LO[6+pi])])
            hi2 = np.array([float(HI[3+pi]), float(HI[6+pi])])
            try:
                res = least_squares(
                    residual_2d, np.array([best_ax, best_ay]), jac='2-point',
                    bounds=(lo2, hi2), method='trf',
                    ftol=1e-14, xtol=1e-14, gtol=1e-14, max_nfev=500)
                current[3+pi] = res.x[0]
                current[6+pi] = res.x[1]
            except Exception:
                pass

    # Final joint 9-D TRF
    def residual_full(x9):
        return vec2pat(np.concatenate([x9, fixed_params])).reshape(-1) - target_flat

    try:
        res = least_squares(
            residual_full, current, jac='2-point',
            bounds=(lo9, hi9), method='trf',
            ftol=1e-15, xtol=1e-15, gtol=1e-15, max_nfev=3000)
        return res.x, float(np.mean(res.fun**2))
    except Exception:
        r = vec2pat(np.concatenate([current, fixed_params])).reshape(-1) - target_flat
        return current, float(np.mean(r**2))


# ── Load and generate ──
print("Loading models...", flush=True)
ang = AngleNet(); rem = RemainNet()
ang.load_state_dict(torch.load(ANG_MODEL, map_location=DEVICE, weights_only=True))
rem.load_state_dict(torch.load(REM_MODEL, map_location=DEVICE, weights_only=True))
ang.to(DEVICE); rem.to(DEVICE)

rng = np.random.default_rng(2026)
cases = []
for i in range(30):
    v = (LO + RG * rng.random(N_PAR).astype(np.float32)).astype(np.float64)
    for j in range(3):
        if abs(v[j]) < 0.15: v[j] = np.copysign(0.15, v[j])
    cases.append(canon(v))

print(f"\n{'='*80}")
print("COORDINATE DESCENT: 2-D per prism (30 cases, 9-D)")
print(f"{'='*80}\n")

ml_perfect = 0
cd_perfect = 0
cdg_perfect = 0
t_total = time.time()

for ci, tc in enumerate(cases):
    pat = vec2pat(tc)
    target_flat = pat.reshape(-1)
    pf, pi_info = extract_speeds_and_peaks(pat)
    pk = _build_peak_feats_single(pat, pf, pi_info)
    fixed = tc[9:].copy()

    lo9 = LO[:9].astype(np.float64)
    hi9 = HI[:9].astype(np.float64)

    def make_res(fix):
        def residual(x9):
            return vec2pat(np.concatenate([x9, fix])).reshape(-1) - target_flat
        return residual
    res_fn = make_res(fixed)

    ml_best = 1e30; ml_x = None
    cd_best = 1e30; cd_x = None
    cdg_best = 1e30; cdg_x = None

    for bits in range(8):
        signs = np.array([(1.0 if (bits>>j)&1==0 else -1.0) for j in range(P)], np.float64)
        speeds = signs * np.sort(pf)[::-1].astype(np.float64)

        # ML init + TRF (baseline)
        ml = ml_init(ang, rem, speeds.astype(np.float32), pk, pat)
        try:
            res = least_squares(res_fn, ml[:9], jac='2-point',
                bounds=(lo9, hi9), method='trf',
                ftol=1e-15, xtol=1e-15, gtol=1e-15, max_nfev=2000)
            mse = float(np.mean(res.fun**2))
            if mse < ml_best: ml_best = mse; ml_x = res.x.copy()
        except Exception: pass

        # Harmonic init → coordinate descent (TRF per prism)
        hi_v = harmonic_init(speeds, pat, fixed)
        x, mse = coord_descent_solve(hi_v, pat, fixed, n_cycles=3)
        if mse < cd_best: cd_best = mse; cd_x = x.copy()

        # ML init → coordinate descent with grid (best of both worlds)
        x2, mse2 = coord_descent_grid(ml[:9], pat, fixed, n_cycles=2, grid_pts=13)
        if mse2 < cdg_best: cdg_best = mse2; cdg_x = x2.copy()

    ml_err = float(np.max(np.abs(ml_x - tc[:9]))) if ml_x is not None else 999
    cd_err = float(np.max(np.abs(cd_x - tc[:9]))) if cd_x is not None else 999
    cdg_err = float(np.max(np.abs(cdg_x - tc[:9]))) if cdg_x is not None else 999
    ml_ok = ml_err < 1e-3
    cd_ok = cd_err < 1e-3
    cdg_ok = cdg_err < 1e-3

    if ml_ok: ml_perfect += 1
    if cd_ok: cd_perfect += 1
    if cdg_ok: cdg_perfect += 1

    tags = []
    if cd_ok and not ml_ok: tags.append("CD_SAVED")
    if cdg_ok and not ml_ok: tags.append("GRID_SAVED")
    tag = " ***" + "+".join(tags) + "***" if tags else ""

    dt = time.time() - t_total
    print(f"  {ci+1:2d}: ML={'P' if ml_ok else 'F'} "
          f"CD={'P' if cd_ok else 'F'} "
          f"GRID={'P' if cdg_ok else 'F'}{tag}  [{dt:.0f}s]", flush=True)

elapsed = time.time() - t_total
print(f"\n{'='*80}")
print(f"  ML only:          {ml_perfect}/30")
print(f"  Harmonic+CD:      {cd_perfect}/30")
print(f"  ML+Grid-CD:       {cdg_perfect}/30")
print(f"  Time: {elapsed:.0f}s ({elapsed/30:.1f}s/case)")
print(f"{'='*80}")
