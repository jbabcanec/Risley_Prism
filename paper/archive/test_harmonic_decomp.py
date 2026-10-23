#!/usr/bin/env python3
"""
Harmonic least-squares decomposition for spectral parameter extraction.

Instead of reading FFT bins (corrupted by discretization and cross-terms),
fit a linear harmonic model:
  p_x(t) = Σᵢ [aᵢ cos(2πNᵢt) + bᵢ sin(2πNᵢt)] + c
  p_y(t) = Σᵢ [aᵢ' cos(2πNᵢt) + bᵢ' sin(2πNᵢt)] + c'

This gives exact amplitudes and phases at the specified frequencies,
handling cross-term contamination because all prisms are fit jointly.

Then: Aᵢ = sqrt(aᵢ² + bᵢ²), φᵢ = atan2(-bᵢ, aᵢ) → αᵧ
      αₓ from Aᵢ via paraxial formula
"""
import sys, os, time, json
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'reverse_problem_v2'))
sys.path.insert(0, os.path.dirname(__file__))

from ml_staged_solver import (
    N_PAR, LO, HI, RG, canon, T_PTS, T_OBS, P, SRC, THK,
    extract_speeds_and_peaks, _build_peak_feats_single,
    AngleNet, RemainNet, ANG_MODEL, REM_MODEL, DEVICE,
)
from solve_preconditioned import vec2pat, ml_init
import torch


def harmonic_decompose(pattern, freqs):
    """
    Decompose pattern into harmonic components at given frequencies.

    Returns per-frequency: (amp_x, amp_y, phase_x, phase_y)
    where p_x(t) ≈ Σ amp_x_i * cos(2π f_i t + phase_x_i) + offset
    """
    T = len(pattern)
    t = np.arange(T) * (T_OBS / T)

    # Build design matrix: [cos(2πf₁t), sin(2πf₁t), cos(2πf₂t), ..., 1]
    n_freq = len(freqs)
    X = np.zeros((T, 2 * n_freq + 1))
    for i, f in enumerate(freqs):
        X[:, 2*i]     = np.cos(2 * np.pi * f * t)
        X[:, 2*i + 1] = np.sin(2 * np.pi * f * t)
    X[:, -1] = 1.0  # DC offset

    # Solve for x and y components
    beta_x, _, _, _ = np.linalg.lstsq(X, pattern[:, 0], rcond=None)
    beta_y, _, _, _ = np.linalg.lstsq(X, pattern[:, 1], rcond=None)

    results = []
    for i in range(n_freq):
        ax_cos = beta_x[2*i]
        ax_sin = beta_x[2*i + 1]
        ay_cos = beta_y[2*i]
        ay_sin = beta_y[2*i + 1]

        # p_x = a*cos(ωt) + b*sin(ωt) = A*cos(ωt + φ) where
        # A = sqrt(a² + b²), φ = atan2(-b, a)
        amp_x = np.sqrt(ax_cos**2 + ax_sin**2)
        amp_y = np.sqrt(ay_cos**2 + ay_sin**2)
        phase_x = np.arctan2(-ax_sin, ax_cos)
        phase_y = np.arctan2(-ay_sin, ay_cos)

        results.append({
            'freq': freqs[i],
            'amp_x': amp_x, 'amp_y': amp_y,
            'phase_x': phase_x, 'phase_y': phase_y,
            'offset_x': beta_x[-1], 'offset_y': beta_y[-1],
        })

    return results


def spectral_init_harmonic(speeds, pattern, nom_dw=100.0, nom_ng=1.5, nom_gap=8.0):
    """
    Build 18-D init from harmonic decomposition at given speeds.

    Uses absolute frequencies for decomposition (FFT is symmetric).
    """
    abs_freqs = np.abs(speeds)
    decomp = harmonic_decompose(pattern, abs_freqs)

    v = np.zeros(N_PAR, dtype=np.float64)
    v[:3] = speeds

    for i, d in enumerate(decomp):
        # d_eff: remaining optical path for prism i
        d_eff = (P - 1 - i) * (THK + nom_gap) + nom_dw

        # αₓ from amplitude: A_x ≈ d_eff * (n-1) * tan(αₓ) * |cos(αᵧ)|
        # A_y ≈ d_eff * (n-1) * tan(αₓ) * |sin(αᵧ)|
        # Total: A = sqrt(A_x² + A_y²) ≈ d_eff * (n-1) * tan(αₓ)
        amp_total = np.sqrt(d['amp_x']**2 + d['amp_y']**2)
        alpha_x_rad = np.arctan(amp_total / (d_eff * (nom_ng - 1.0) + 1e-10))
        alpha_x = np.degrees(alpha_x_rad)
        alpha_x = max(alpha_x, 0.5)

        # αᵧ from the x-phase directly
        # In paraxial: p_x ≈ A * cos(2π|N|t + αᵧ) (for N > 0)
        #              p_x ≈ A * cos(2π|N|t - αᵧ) (for N < 0)
        # So phase at |N| = ±αᵧ
        phase = d['phase_x']
        if speeds[i] < 0:
            alpha_y = -np.degrees(phase)
        else:
            alpha_y = np.degrees(phase)

        v[3 + i] = alpha_x
        v[6 + i] = alpha_y

    v[9:12] = nom_ng
    v[12] = nom_dw
    v[13] = nom_gap
    v[14:18] = 0.0

    return np.clip(v, LO, HI)


# ── Generate 30 cases ──
rng = np.random.default_rng(2026)
cases = []
for i in range(30):
    v = (LO + RG * rng.random(N_PAR).astype(np.float32)).astype(np.float64)
    for j in range(3):
        if abs(v[j]) < 0.15: v[j] = np.copysign(0.15, v[j])
    cases.append(canon(v))


# ── Phase 1: Test accuracy of harmonic decomposition ──
print(f"{'='*90}")
print("HARMONIC DECOMPOSITION — Accuracy Test (true frequencies)")
print(f"{'='*90}\n")

phase_errs = []
ax_errs = []

for ci, tc in enumerate(cases[:10]):
    pat = vec2pat(tc)
    true_speeds = tc[:3]
    abs_freqs = np.abs(true_speeds)

    decomp = harmonic_decompose(pat, abs_freqs)

    print(f"Case {ci+1}: speeds={true_speeds}")
    for i, d in enumerate(decomp):
        # Compare phases
        phase = d['phase_x']
        if true_speeds[i] < 0:
            ay_est = -np.degrees(phase)
        else:
            ay_est = np.degrees(phase)
        ay_true = tc[6+i]
        ay_err = min(abs(ay_est - ay_true), abs(ay_est - ay_true + 360),
                     abs(ay_est - ay_true - 360))
        phase_errs.append(ay_err)

        # Compare αₓ (using TRUE geometry for fairness)
        d_eff = (P-1-i) * (THK + tc[13]) + tc[12]
        ng = tc[9+i]
        amp_total = np.sqrt(d['amp_x']**2 + d['amp_y']**2)
        ax_est = np.degrees(np.arctan(amp_total / (d_eff * (ng - 1.0))))
        ax_true = abs(tc[3+i])
        ax_err = abs(ax_est - ax_true)
        ax_errs.append(ax_err)

        print(f"  P{i+1}: αᵧ_est={ay_est:7.1f} true={ay_true:7.1f} err={ay_err:5.1f}°  "
              f"|αₓ|_est={ax_est:5.2f} true={ax_true:5.2f} err={ax_err:.2f}°")

# Rest silently
for ci in range(10, 30):
    tc = cases[ci]
    pat = vec2pat(tc)
    decomp = harmonic_decompose(pat, np.abs(tc[:3]))
    for i, d in enumerate(decomp):
        phase = d['phase_x']
        ay_est = -np.degrees(phase) if tc[i] < 0 else np.degrees(phase)
        ay_true = tc[6+i]
        ay_err = min(abs(ay_est-ay_true), abs(ay_est-ay_true+360), abs(ay_est-ay_true-360))
        phase_errs.append(ay_err)
        d_eff = (P-1-i)*(THK+tc[13])+tc[12]
        ng = tc[9+i]
        amp_total = np.sqrt(d['amp_x']**2 + d['amp_y']**2)
        ax_est = np.degrees(np.arctan(amp_total/(d_eff*(ng-1.0))))
        ax_errs.append(abs(ax_est - abs(tc[3+i])))

phase_errs = np.array(phase_errs)
ax_errs = np.array(ax_errs)
print(f"\nWith TRUE frequencies + TRUE geometry:")
print(f"  αᵧ error: median={np.median(phase_errs):.1f}°, "
      f"<5°: {np.sum(phase_errs<5)}/90, <10°: {np.sum(phase_errs<10)}/90, "
      f"<20°: {np.sum(phase_errs<20)}/90")
print(f"  |αₓ| error: median={np.median(ax_errs):.2f}°, "
      f"<1°: {np.sum(ax_errs<1)}/90, <2°: {np.sum(ax_errs<2)}/90")


# ── Phase 2: Use FFT frequencies (realistic) and run TRF ──
print(f"\n{'='*90}")
print("HARMONIC INIT vs ML INIT — 9-D TRF Recovery (30 cases)")
print(f"{'='*90}\n")

# Load ML models
ang = AngleNet(); rem = RemainNet()
ang.load_state_dict(torch.load(ANG_MODEL, map_location=DEVICE, weights_only=True))
rem.load_state_dict(torch.load(REM_MODEL, map_location=DEVICE, weights_only=True))
ang.to(DEVICE); rem.to(DEVICE)

ml_perfect = 0
harm_perfect = 0
either_perfect = 0

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
    harm_best = 1e30; harm_x = None

    for bits in range(8):
        signs = np.array([(1.0 if (bits>>j)&1==0 else -1.0) for j in range(P)], np.float64)
        speeds = signs * np.sort(pf)[::-1].astype(np.float64)

        # ML init
        ml = ml_init(ang, rem, speeds.astype(np.float32), pk, pat)
        try:
            res = least_squares(res_fn, ml[:9], jac='2-point',
                bounds=(lo9, hi9), method='trf',
                ftol=1e-15, xtol=1e-15, gtol=1e-15, max_nfev=2000)
            mse = float(np.mean(res.fun**2))
            if mse < ml_best: ml_best = mse; ml_x = res.x.copy()
        except Exception: pass

        # Harmonic init
        hi_v = spectral_init_harmonic(speeds, pat)
        try:
            res = least_squares(res_fn, hi_v[:9], jac='2-point',
                bounds=(lo9, hi9), method='trf',
                ftol=1e-15, xtol=1e-15, gtol=1e-15, max_nfev=2000)
            mse = float(np.mean(res.fun**2))
            if mse < harm_best: harm_best = mse; harm_x = res.x.copy()
        except Exception: pass

    ml_err = float(np.max(np.abs(ml_x - tc[:9]))) if ml_x is not None else 999
    harm_err = float(np.max(np.abs(harm_x - tc[:9]))) if harm_x is not None else 999
    ml_ok = ml_err < 1e-3
    harm_ok = harm_err < 1e-3

    if ml_ok: ml_perfect += 1
    if harm_ok: harm_perfect += 1
    if ml_ok or harm_ok: either_perfect += 1

    tag = ""
    if ml_ok != harm_ok:
        tag = " ***DIFFER***"
    sys.stdout.write(f"  {ci+1:2d}: ML={'P' if ml_ok else 'F'}(e={ml_err:.1e}) "
                     f"HARM={'P' if harm_ok else 'F'}(e={harm_err:.1e}){tag}\n")
    sys.stdout.flush()

print(f"\n{'='*90}")
print(f"  ML init:       {ml_perfect}/30 PERFECT")
print(f"  Harmonic init: {harm_perfect}/30 PERFECT")
print(f"  Either (union):{either_perfect}/30")
print(f"{'='*90}")
