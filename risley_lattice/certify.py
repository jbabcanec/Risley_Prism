"""Certificates: tight error bounds on success, quantitative impossibility
on failure.

SUCCESS   per-parameter bound = 3 sqrt(diag s^2 (J^T J)^-1) + |Newton gap|,
          J the exact-model numerical Jacobian at the solution,
          s^2 = ||r||^2/(m-18). NO covariance truncation: rank(J) = 18 is a
          theorem, and the weakest (d_W-gap) eigenvalue -- near 1e-15 of the
          largest -- is genuine information carrying the dominant bound.
          Battery: 26/26 coverage, median tightness 5x.

FAILURE   Fisher information of the fitted lattice model (merged design, no
          ridge) gives sigma(N_i), sigma(amp_i); certificates:
            close-pair   |dN| < 3 sigma  -> need T >= T (3sigma/dN)^(2/3)
            weak-prism   amp < 5 sigma   -> min detectable |ax| stated
            relation     |k.N - N_j| < 3 sigma -> subspace degenerate at T
            glitch-floor TIR mask + residual floor inflate all sigmas
          Confirmed empirically: every certified T_req rescues its case
          (assumption test A8).
"""
from itertools import product as iproduct

import numpy as np

from .model import N_PAR, T_OBS, DT, vec2pat
from .lattice import lattice_fit, MERGE_C
from .angles import calibrate_ax, nominal_rest

MERGE_SAFETY = 2.0        # c in T_req = max(stat term, c * MERGE_C / gap)


def t_required(T, thr, gap):
    """Corrected observation-time prescription, Eq. (treq) of the paper:
    the statistical T^{-3/2} extrapolation AND the estimator's resolution
    floor (lines closer than MERGE_C/T are merged by construction)."""
    g = max(gap, 1e-12)
    t_stat = T * (thr / g) ** (2.0 / 3.0)
    t_floor = MERGE_SAFETY * MERGE_C / g
    return max(t_stat, t_floor)


def certify_success(x18, pattern, rmask):
    """3-sigma + optimality-gap per-parameter bounds at the solution."""
    target = pattern.reshape(-1).astype(np.float64)

    def resid(x):
        return (vec2pat(x).reshape(-1) - target)[rmask]

    r0 = resid(x18)
    m = len(r0)
    J = np.empty((m, N_PAR))
    for i in range(N_PAR):
        h = max(1e-7, 1e-7 * abs(x18[i]))
        xp = x18.copy(); xp[i] += h
        xm = x18.copy(); xm[i] -= h
        J[:, i] = (resid(xp) - resid(xm)) / (2 * h)
    s2 = float(r0 @ r0) / max(m - N_PAR, 1)
    JTJi = np.linalg.pinv(J.T @ J, rcond=1e-18)
    cov = s2 * JTJi
    gap = np.abs(JTJi @ (J.T @ r0))
    return 3.0 * np.sqrt(np.maximum(np.diag(cov), 0.0)) + gap


def spectral_fisher(pattern, N_est, mask, dt=DT):
    """Joint Fisher covariance of (generators, line amplitudes) at the
    fitted lattice model. Merged design, NO ridge (honest bounds)."""
    t = np.arange(len(pattern)) * dt
    z = pattern[:, 0] + 1j * pattern[:, 1]
    g, K, c, res = lattice_fit(z, t, mask, list(N_est), B=3)
    zm, tm = z[mask], t[mask]
    E = np.exp(2j * np.pi * np.outer(tm, K @ g))
    r = zm - E @ c
    Jg = 2j * np.pi * tm[:, None] * (E * c[None, :]) @ K
    X = np.block([[Jg.real, E.real, -E.imag],
                  [Jg.imag, E.imag, E.real]])
    mr, ncol = X.shape
    rms2 = (np.linalg.norm(r) ** 2) / max(mr - ncol, 1)
    F = X.T @ X
    cov = rms2 * np.linalg.pinv(F, rcond=1e-12)
    n = len(g)
    sig_g = np.sqrt(np.maximum(np.diag(cov)[:n], 0.0))
    amp, sig_amp = np.zeros(n), np.zeros(n)
    nk = len(K)
    for i in range(n):
        row = np.zeros(n); row[i] = 1.0
        j = int(np.argmin(np.abs(K - row).sum(1)))
        amp[i] = abs(c[j])
        sig_amp[i] = np.sqrt(max(cov[n + j, n + j], 0.0) +
                             max(cov[n + nk + j, n + nk + j], 0.0))
    return g, sig_g, amp, sig_amp, float(res)


def spectral_certificate(pattern, N_est, info, dt=DT, n_target=3):
    """Quantitative failure certificate. Returns (reasons, sigma_N).
    N_est may have fewer than n_target entries (rank-deficient front end):
    the Fisher analysis then runs on the partial basis and additionally
    reports the detectability floor for the missing prism(s)."""
    mask = info.get('mask', np.ones(len(pattern), bool))
    N_est = [x for x in np.atleast_1d(N_est)]
    if not N_est:
        return (['rank-deficient: no generators detectable '
                 '(front-end failure)'], np.array([]))
    g, sig_g, amp, sig_amp, res = spectral_fisher(pattern, N_est, mask, dt)
    T = T_OBS
    n = len(g)
    reasons = []
    if n < n_target:
        # detectability floor for a missing prism: 5-sigma on the amplitude
        # of a hypothetical extra line, through the nominal cubic gain
        sig_extra = float(np.median(sig_amp)) if len(sig_amp) else 0.0
        try:
            gain = calibrate_ax(list(g) + [1.0] * (3 - n),
                                nominal_rest(pattern))[0][0]
        except Exception:
            gain = 20.0
        ax_min = np.degrees(np.arctan(5.0 * sig_extra / max(gain, 1e-6)))
        reasons.append(
            f"rank-deficient: only {n} of {n_target} generators "
            f"detectable; any remaining prism with |ax| < {ax_min:.2f} deg "
            f"is invisible at T={T:.0f} s")

    for i in range(n):
        for j in range(i + 1, n):
            d = min(abs(g[i] - g[j]), abs(g[i] + g[j]))
            thr = 3.0 * np.sqrt(sig_g[i] ** 2 + sig_g[j] ** 2)
            if d < thr:
                T_req = t_required(T, thr, d)
                reasons.append(
                    f"close-pair N{i+1}/N{j+1}: sep {d:.4f} Hz < 3-sigma "
                    f"{thr:.4f} Hz -> need T >= {min(T_req, 9999):.0f} s")

    try:
        coefs = calibrate_ax(g, nominal_rest(pattern))
    except Exception:
        coefs = [(20.0, 0.0)] * n
    for i in range(n):
        if amp[i] < 5.0 * sig_amp[i]:
            a_lin = max(coefs[i][0], 1e-6)
            ax_min = np.degrees(np.arctan(5.0 * sig_amp[i] / a_lin))
            reasons.append(
                f"weak-prism {i+1}: amp {amp[i]:.3f} < 5-sigma "
                f"{5*sig_amp[i]:.3f} -> any |ax{i+1}| < {ax_min:.2f} deg "
                f"is invisible at T={T:.0f} s")

    KS = np.array([k for k in iproduct(range(-3, 4), repeat=n)
                   if 0 < sum(abs(x) for x in k) <= 4])
    for j in range(n):
        ej = np.zeros(n); ej[j] = 1.0
        for k in KS:
            if np.array_equal(k, ej) or np.array_equal(k, -ej):
                continue
            gap = abs(k @ g - g[j])
            thr = 3.0 * (np.abs(k) @ sig_g + sig_g[j])
            if gap < thr:
                T_req = t_required(T, thr, gap)
                reasons.append(
                    f"relation N{j+1} ~ {tuple(int(x) for x in k)}.N: gap "
                    f"{gap:.4f} Hz < {thr:.4f} -> degenerate; need "
                    f"T >= {min(T_req, 9999):.0f} s")
                break

    nmask = int((~mask).sum())
    if nmask > 0 or res > 5e-2:
        reasons.append(
            f"glitch-floor: {nmask} TIR-masked samples, lattice residual "
            f"{res:.1e} inflates all sigmas by ~{max(res/2e-3, 1):.0f}x")

    if not reasons:
        reasons.append("basin: spectral stage certified "
                       f"(sigma_N max {sig_g.max():.1e} Hz); failure is "
                       "TRF initialization, not identifiability")
    return reasons, sig_g
