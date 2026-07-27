#!/usr/bin/env python3
"""
Model-based homotopy: deform from paraxial model to full non-paraxial model.

At λ=0: solve paraxial model P(θ) = target → exact solution via harmonic LS
At λ=1: solve full model F(θ) = target → the real problem

Continuation:
  R_λ(θ) = (1-λ) P(θ) + λ F(θ) - target = 0
  At each step, solve this with TRF using the previous θ as init.

Why this works:
  1. The paraxial solution is EXACT (no init error — computed analytically)
  2. Each λ step is a small model perturbation → TRF converges
  3. The solution path is smooth for small deformations

This is the Davidenko method applied to a model deformation homotopy.
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


def paraxial_forward(theta_9d, fixed_params):
    """
    Paraxial forward model: linear superposition of prism deflections.

    p_x(t) = Σᵢ d_eff_i * (nᵢ-1) * tan(αₓᵢ) * cos(2πNᵢt + αᵧᵢ_rad) + offset_x
    p_y(t) = Σᵢ d_eff_i * (nᵢ-1) * tan(αₓᵢ) * sin(2πNᵢt + αᵧᵢ_rad) + offset_y

    theta_9d: [N1,N2,N3, αₓ1,αₓ2,αₓ3, αᵧ1,αᵧ2,αᵧ3]
    fixed_params: [ng1,ng2,ng3, d_W, gap, bm_ax, bm_ay, bm_px, bm_py]
    """
    speeds = theta_9d[:3]
    ax_deg = theta_9d[3:6]
    ay_deg = theta_9d[6:9]

    ng = fixed_params[:3]
    d_W = fixed_params[3]
    gap = fixed_params[4]
    bm_ax = fixed_params[5]
    bm_ay = fixed_params[6]
    # bm_px, bm_py not used in paraxial (beam position shifts are secondary)

    t = np.arange(T_PTS) * (T_OBS / T_PTS)
    px = np.zeros(T_PTS)
    py = np.zeros(T_PTS)

    for i in range(P):
        # Effective distance: prism i to workpiece
        # Prism i is at position SRC + i*(THK + gap) + THK (exit face)
        # Remaining distance: (P-1-i)*(THK+gap) + d_W
        d_eff = (P - 1 - i) * (THK + gap) + d_W

        # Deflection amplitude
        amp = d_eff * (ng[i] - 1.0) * np.tan(np.radians(ax_deg[i]))

        # Phase
        phase = np.radians(ay_deg[i])

        px += amp * np.cos(2 * np.pi * speeds[i] * t + phase)
        py += amp * np.sin(2 * np.pi * speeds[i] * t + phase)

    # Add beam angle offset (paraxial: shifts pattern by d_total * tan(beam_angle))
    d_total = SRC + P * THK + (P - 1) * gap + d_W
    px += d_total * np.tan(np.radians(bm_ax))
    py += d_total * np.tan(np.radians(bm_ay))

    return np.column_stack([px, py])


def paraxial_solve(target, fixed_params, speed_magnitudes):
    """
    Exact paraxial solution via harmonic least-squares.

    For each speed sign combo, decompose the target into harmonic coefficients,
    convert to (αₓ, αᵧ), and evaluate MSE.
    """
    ng = fixed_params[:3]
    d_W = fixed_params[3]
    gap = fixed_params[4]
    bm_ax = fixed_params[5]
    bm_ay = fixed_params[6]

    T = T_PTS
    t = np.arange(T) * (T_OBS / T)

    # Remove beam offset (paraxial estimate)
    d_total = SRC + P * THK + (P - 1) * gap + d_W
    offset_x = d_total * np.tan(np.radians(bm_ax))
    offset_y = d_total * np.tan(np.radians(bm_ay))
    centered_x = target[:, 0] - offset_x
    centered_y = target[:, 1] - offset_y

    best_mse = 1e30
    best_theta = None

    for bits in range(8):
        signs = np.array([(1.0 if (bits>>j)&1==0 else -1.0) for j in range(P)], np.float64)
        speeds = signs * speed_magnitudes

        # Build design matrix for harmonic LS
        X = np.zeros((T, 2*P + 1))
        for i in range(P):
            X[:, 2*i] = np.cos(2*np.pi*speeds[i]*t)
            X[:, 2*i+1] = np.sin(2*np.pi*speeds[i]*t)
        X[:, -1] = 1.0  # residual DC

        bx, _, _, _ = np.linalg.lstsq(X, centered_x, rcond=None)
        by, _, _, _ = np.linalg.lstsq(X, centered_y, rcond=None)

        theta = np.zeros(9)
        theta[:3] = speeds

        for i in range(P):
            d_eff = (P - 1 - i) * (THK + gap) + d_W

            ax_cos = bx[2*i]
            ax_sin = bx[2*i+1]
            ay_cos = by[2*i]
            ay_sin = by[2*i+1]

            # Combined amplitude and phase
            # p_x = amp*cos(phase)*cos(ωt) - amp*sin(phase)*sin(ωt) = amp*cos(ωt+phase)
            # So ax_cos = amp*cos(phase), ax_sin = -amp*sin(phase)
            amp_x = np.sqrt(ax_cos**2 + ax_sin**2)
            amp_y = np.sqrt(ay_cos**2 + ay_sin**2)
            amp = np.sqrt(amp_x**2 + amp_y**2)

            # αₓ from amplitude
            denominator = d_eff * (ng[i] - 1.0)
            if denominator > 1e-10:
                alpha_x_rad = np.arctan(amp / denominator)
            else:
                alpha_x_rad = np.radians(5.0)
            alpha_x = np.degrees(alpha_x_rad)

            # αᵧ from phase of x-component
            phase_x = np.arctan2(-ax_sin, ax_cos)
            alpha_y = np.degrees(phase_x)

            # Need to determine sign of αₓ
            # In paraxial: amp_x = d_eff * (n-1) * tan(αₓ) * cos(αᵧ_rad)
            # If cos(αᵧ_rad)*tan(αₓ) should match the sign of amp_x coefficient
            expected_sign = np.sign(ax_cos) if abs(ax_cos) > abs(ax_sin) else np.sign(ay_cos)
            if expected_sign < 0:
                alpha_x = -alpha_x

            theta[3+i] = np.clip(alpha_x, float(LO[3+i]), float(HI[3+i]))
            theta[6+i] = np.clip(alpha_y, float(LO[6+i]), float(HI[6+i]))

        # Evaluate paraxial MSE
        pred = paraxial_forward(theta, fixed_params)
        mse = float(np.mean((pred - target)**2))

        if mse < best_mse:
            best_mse = mse
            best_theta = theta.copy()

        # Also try the 180° flipped version of αᵧ
        theta_flip = theta.copy()
        for i in range(P):
            theta_flip[3+i] = -theta[3+i]
            theta_flip[6+i] = theta[6+i] + 180.0
            if theta_flip[6+i] > 180: theta_flip[6+i] -= 360

        theta_flip[3:6] = np.clip(theta_flip[3:6], LO[3:6], HI[3:6])
        theta_flip[6:9] = np.clip(theta_flip[6:9], LO[6:9], HI[6:9])

        pred_flip = paraxial_forward(theta_flip, fixed_params)
        mse_flip = float(np.mean((pred_flip - target)**2))
        if mse_flip < best_mse:
            best_mse = mse_flip
            best_theta = theta_flip.copy()

    return best_theta, best_mse


def model_homotopy_solve(paraxial_theta, target, fixed_params, n_steps=30):
    """
    Corrected homotopy from paraxial to full model.

    H(θ, λ) = (1-λ) [P(θ) - P(θ₀)] + λ [F(θ) - target] = 0

    At λ=0: P(θ) = P(θ₀) → θ = θ₀ (EXACT, by construction)
    At λ=1: F(θ) = target → the real problem

    This formulation guarantees the starting point is an exact solution.
    """
    lo9 = LO[:9].astype(np.float64)
    hi9 = HI[:9].astype(np.float64)
    target_flat = target.reshape(-1)

    # Compute paraxial prediction at θ₀ (fixed reference)
    par_at_theta0 = paraxial_forward(paraxial_theta, fixed_params).reshape(-1)

    current = paraxial_theta.copy()
    lambdas = np.linspace(0, 1, n_steps + 1)[1:]

    for lam in lambdas:
        def residual(x9, _lam=lam):
            par = paraxial_forward(x9, fixed_params).reshape(-1)
            full_theta = np.concatenate([x9, fixed_params])
            full = vec2pat(full_theta).reshape(-1)
            return (1 - _lam) * (par - par_at_theta0) + _lam * (full - target_flat)

        try:
            res = least_squares(
                residual, current, jac='2-point',
                bounds=(lo9, hi9), method='trf',
                ftol=1e-14, xtol=1e-14, gtol=1e-14, max_nfev=300)
            current = res.x.copy()
        except Exception:
            break

    # Final TRF at λ=1 (full model only)
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

# ── Test ──
print(f"\n{'='*80}")
print("MODEL HOMOTOPY: Paraxial → Full (30 cases, 9-D)")
print(f"{'='*80}\n")

ml_perfect = 0
homo_perfect = 0
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
    homo_best = 1e30; homo_x = None

    # ML-only baseline
    for bits in range(8):
        signs = np.array([(1.0 if (bits>>j)&1==0 else -1.0) for j in range(P)], np.float64)
        speeds = signs * np.sort(pf)[::-1].astype(np.float64)
        ml = ml_init(ang, rem, speeds.astype(np.float32), pk, pat)
        try:
            res = least_squares(res_fn, ml[:9], jac='2-point',
                bounds=(lo9, hi9), method='trf',
                ftol=1e-15, xtol=1e-15, gtol=1e-15, max_nfev=2000)
            mse = float(np.mean(res.fun**2))
            if mse < ml_best: ml_best = mse; ml_x = res.x.copy()
        except Exception: pass

    # Model homotopy: try multiple paraxial inits → continuation → full
    from itertools import combinations
    all_pf = pi_info.get('all_peak_freqs', pf)

    # Generate frequency triples from top-K peaks
    triples = [tuple(np.sort(pf)[::-1])]
    for combo in combinations(range(len(all_pf)), P):
        triple = tuple(sorted([all_pf[i] for i in combo], reverse=True))
        if triple not in triples and all(f > 0.05 for f in triple):
            triples.append(triple)

    for triple in triples[:10]:  # limit to top-10 triples for speed
        speed_mags = np.array(triple, dtype=np.float64)
        par_theta, par_mse = paraxial_solve(pat, fixed, speed_mags)

        if par_theta is not None:
            x, mse = model_homotopy_solve(par_theta, pat, fixed, n_steps=50)
            if mse < homo_best: homo_best = mse; homo_x = x.copy()
            if homo_best < 1e-15: break  # already perfect

    ml_err = float(np.max(np.abs(ml_x - tc[:9]))) if ml_x is not None else 999
    homo_err = float(np.max(np.abs(homo_x - tc[:9]))) if homo_x is not None else 999
    ml_ok = ml_err < 1e-3
    homo_ok = homo_err < 1e-3

    if ml_ok: ml_perfect += 1
    if homo_ok: homo_perfect += 1

    tag = ""
    if homo_ok and not ml_ok: tag = " ***HOMO SAVED***"
    elif ml_ok and not homo_ok: tag = " ***ML ONLY***"

    print(f"  {ci+1:2d}: ML={'P' if ml_ok else 'F'}(e={ml_err:.1e}) "
          f"HOMO={'P' if homo_ok else 'F'}(e={homo_err:.1e}){tag}", flush=True)

elapsed = time.time() - t_total
print(f"\n{'='*80}")
print(f"  ML only:     {ml_perfect}/30 PERFECT")
print(f"  Homotopy:    {homo_perfect}/30 PERFECT")
print(f"  Time: {elapsed:.0f}s ({elapsed/30:.1f}s/case)")
print(f"{'='*80}")
