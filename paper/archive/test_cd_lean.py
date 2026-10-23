#!/usr/bin/env python3
"""
Lean coordinate descent: harmonic init → 2-D TRF per prism → joint 9-D TRF.
No grid search (too slow). Tests whether breaking 9-D into 2-D helps the basin.
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
    """Paraxial init from harmonic LS."""
    T = T_PTS
    t = np.arange(T) * (T_OBS / T)
    ng = fixed_params[:3]; d_W = fixed_params[3]; gap = fixed_params[4]

    X = np.zeros((T, 2*P + 1))
    for i in range(P):
        X[:, 2*i] = np.cos(2*np.pi*speeds[i]*t)
        X[:, 2*i+1] = np.sin(2*np.pi*speeds[i]*t)
    X[:, -1] = 1.0

    bx, _, _, _ = np.linalg.lstsq(X, pattern[:, 0], rcond=None)

    theta = np.zeros(9); theta[:3] = speeds
    for i in range(P):
        d_eff = (P-1-i)*(THK+gap)+d_W
        amp = np.sqrt(bx[2*i]**2+bx[2*i+1]**2)
        denom = d_eff*(ng[i]-1.0)
        ax = np.degrees(np.arctan(amp/denom)) if denom > 1e-10 else 5.0
        phase = np.arctan2(-bx[2*i+1], bx[2*i])
        ay = np.degrees(phase)
        if bx[2*i] < 0: ax = -ax; ay += 180.0; ay = ay if ay <= 180 else ay-360
        theta[3+i] = np.clip(ax, float(LO[3+i]), float(HI[3+i]))
        theta[6+i] = np.clip(ay, float(LO[6+i]), float(HI[6+i]))
    return theta


def coord_descent(init_9d, target_flat, fixed, n_cycles=4):
    """Coordinate descent: 2-D TRF per prism, then joint 9-D TRF."""
    lo9 = LO[:9].astype(np.float64); hi9 = HI[:9].astype(np.float64)
    current = init_9d.copy()

    for _ in range(n_cycles):
        for pi in range(P):
            def res2d(x2):
                theta = current.copy()
                theta[3+pi] = x2[0]; theta[6+pi] = x2[1]
                return vec2pat(np.concatenate([theta, fixed])).reshape(-1) - target_flat

            lo2 = np.array([float(LO[3+pi]), float(LO[6+pi])])
            hi2 = np.array([float(HI[3+pi]), float(HI[6+pi])])
            try:
                res = least_squares(res2d, np.array([current[3+pi], current[6+pi]]),
                    jac='2-point', bounds=(lo2, hi2), method='trf',
                    ftol=1e-14, xtol=1e-14, gtol=1e-14, max_nfev=500)
                current[3+pi] = res.x[0]; current[6+pi] = res.x[1]
            except Exception: pass

    # Joint 9-D polish
    def res_full(x9):
        return vec2pat(np.concatenate([x9, fixed])).reshape(-1) - target_flat
    try:
        res = least_squares(res_full, current, jac='2-point',
            bounds=(lo9, hi9), method='trf',
            ftol=1e-15, xtol=1e-15, gtol=1e-15, max_nfev=3000)
        return res.x, float(np.mean(res.fun**2))
    except Exception:
        r = vec2pat(np.concatenate([current, fixed])).reshape(-1) - target_flat
        return current, float(np.mean(r**2))


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
print("ML vs HARMONIC+CD vs ML+CD — 9-D (30 cases)")
print(f"{'='*80}\n")

ml_p = 0; hcd_p = 0; mcd_p = 0; any_p = 0
t0 = time.time()

for ci, tc in enumerate(cases):
    pat = vec2pat(tc)
    tf = pat.reshape(-1)
    pf, pi_info = extract_speeds_and_peaks(pat)
    pk = _build_peak_feats_single(pat, pf, pi_info)
    fixed = tc[9:].copy()
    lo9 = LO[:9].astype(np.float64); hi9 = HI[:9].astype(np.float64)

    def make_res(fix):
        def residual(x9):
            return vec2pat(np.concatenate([x9, fix])).reshape(-1) - tf
        return residual
    res_fn = make_res(fixed)

    ml_b = 1e30; ml_x = None
    hcd_b = 1e30; hcd_x = None
    mcd_b = 1e30; mcd_x = None

    for bits in range(8):
        signs = np.array([(1.0 if (bits>>j)&1==0 else -1.0) for j in range(P)], np.float64)
        speeds = signs * np.sort(pf)[::-1].astype(np.float64)

        # (A) ML → direct TRF
        ml = ml_init(ang, rem, speeds.astype(np.float32), pk, pat)
        try:
            res = least_squares(res_fn, ml[:9], jac='2-point',
                bounds=(lo9, hi9), method='trf',
                ftol=1e-15, xtol=1e-15, gtol=1e-15, max_nfev=2000)
            mse = float(np.mean(res.fun**2))
            if mse < ml_b: ml_b = mse; ml_x = res.x.copy()
        except Exception: pass

        # (B) Harmonic init → coord descent
        hi_v = harmonic_init(speeds, pat, fixed)
        x, mse = coord_descent(hi_v, tf, fixed, n_cycles=4)
        if mse < hcd_b: hcd_b = mse; hcd_x = x.copy()

        # (C) ML init → coord descent (uses ML's better αₓ, refines via CD)
        x2, mse2 = coord_descent(ml[:9], tf, fixed, n_cycles=4)
        if mse2 < mcd_b: mcd_b = mse2; mcd_x = x2.copy()

    ml_err = float(np.max(np.abs(ml_x - tc[:9]))) if ml_x is not None else 999
    hcd_err = float(np.max(np.abs(hcd_x - tc[:9]))) if hcd_x is not None else 999
    mcd_err = float(np.max(np.abs(mcd_x - tc[:9]))) if mcd_x is not None else 999

    ml_ok = ml_err < 1e-3; hcd_ok = hcd_err < 1e-3; mcd_ok = mcd_err < 1e-3
    if ml_ok: ml_p += 1
    if hcd_ok: hcd_p += 1
    if mcd_ok: mcd_p += 1
    if ml_ok or hcd_ok or mcd_ok: any_p += 1

    tags = []
    if hcd_ok and not ml_ok: tags.append("HCD")
    if mcd_ok and not ml_ok: tags.append("MCD")
    tag = " ***" + "+".join(tags) + " SAVED***" if tags else ""

    print(f"  {ci+1:2d}: ML={'P' if ml_ok else 'F'} HCD={'P' if hcd_ok else 'F'} "
          f"MCD={'P' if mcd_ok else 'F'}{tag}  [{time.time()-t0:.0f}s]", flush=True)

print(f"\n{'='*80}")
print(f"  ML only:     {ml_p}/30")
print(f"  Harmonic+CD: {hcd_p}/30")
print(f"  ML+CD:       {mcd_p}/30")
print(f"  Any:         {any_p}/30")
print(f"  Time: {time.time()-t0:.0f}s ({(time.time()-t0)/30:.1f}s/case)")
print(f"{'='*80}")
