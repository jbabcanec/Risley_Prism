#!/usr/bin/env python3
"""
test_alphax_grid.py — JOINT 3-D grid over the three alpha_x.

Hypothesis (from the failure diagnostics): the 9-D angle-recovery bottleneck is
ENTIRELY the joint alpha_x basin. The basin is narrow (~5 deg) in the three
alpha_x directions and WIDE in the three alpha_y directions. Therefore:

  - 1-init-per-sign + TRF (ML/harmonic) lands outside the basin ~70% of the time.
  - 6-D DE over all angles is a needle search (basin tiny in 6-D volume) -> ~1/30.
  - 2-D-per-prism grid + coordinate descent fails: gridding one prism's alpha_x
    while the OTHER prisms are wrong picks a compensating (wrong) value -> +0.

The untried middle: grid the THREE alpha_x JOINTLY (the only narrow directions),
set alpha_y to the wide-basin midpoint (0), screen every node with the batched
differentiable model, then run ONE joint 9-D TRF from each of the top-K nodes.
A full 3-D grid guarantees a node where all three alpha_x are simultaneously
within ~2 deg of truth; the joint TRF (wide alpha_y basin) finishes to machine
precision.

9-D protocol (identical to test_diagnose_failures.py / test_harmonic_decomp.py):
glass/geometry/beam fixed to truth; recover speeds+angles; success = max|delta|<1e-3.
Baselines on this exact harness: ML=9/30, harmonic=1/30, coord-descent +0, DE-6D ~1/30.

Run: python paper/test_alphax_grid.py
"""
import sys, os, time, io
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'reverse_problem_v2'))
sys.path.insert(0, os.path.dirname(__file__))

import torch
from ml_staged_solver import (
    AngleNet, RemainNet, ANG_MODEL, REM_MODEL, DiffFwd,
    DEVICE, N_PAR, LO, HI, RG, canon, P,
    extract_speeds_and_peaks, _build_peak_feats_single,
)
from solve_preconditioned import vec2pat, ml_init

# ---- grid config ----
N_GRID      = 9     # alpha_x grid points per prism over [-18,18] -> 4.5 deg spacing
N_SCREEN    = 24    # nodes kept from the screen for tier-1 TRF
N_FULL      = 2     # nodes given a full-precision TRF
MULTI_PEAKS = 8     # escalation: C(8,3)=56 triples from the top-8 FFT peaks

_ax_axis = np.linspace(float(LO[3]), float(HI[3]), N_GRID)
AX_GRID = np.stack(np.meshgrid(_ax_axis, _ax_axis, _ax_axis, indexing='ij'),
                   axis=-1).reshape(-1, 3)            # (N_GRID^3, 3)
_FWD = DiffFwd().to(DEVICE)
_LO9 = LO[:9].astype(np.float64)
_HI9 = HI[:9].astype(np.float64)


def _screen(cand, target_t, chunk=16000):
    """Batched MSE of every candidate via the differentiable model (float32)."""
    out = []
    with torch.no_grad():
        for i in range(0, len(cand), chunk):
            c = torch.tensor(cand[i:i+chunk], dtype=torch.float32, device=DEVICE)
            p = _FWD(c, high_precision=False)
            out.append(((p - target_t) ** 2).mean(dim=(1, 2)).cpu().numpy())
    return np.concatenate(out)


def _candidates(freq_triples, fixed9):
    """For each freq triple x 8 sign combos x grid node: an 18-vec (alpha_y=0)."""
    n = AX_GRID.shape[0]
    blocks = []
    for tri in freq_triples:
        mags = np.array(tri, dtype=np.float64)
        for bits in range(8):
            signs = np.array([(1.0 if (bits >> j) & 1 == 0 else -1.0)
                              for j in range(P)], np.float64)
            blk = np.zeros((n, N_PAR), dtype=np.float64)
            blk[:, :3] = signs * mags
            blk[:, 3:6] = AX_GRID
            blk[:, 9:] = fixed9
            blocks.append(blk)
    return np.concatenate(blocks, axis=0)


def _attempt(freq_triples, fixed9, target_t, target_flat):
    """Screen all (triples x signs x grid) nodes; TRF the top screened nodes
    PLUS a zero-angle init for the best distinct speed-signs (insurance for
    large-angle cases whose in-basin origin init screens at high MSE)."""
    cand = _candidates(freq_triples, fixed9)
    order = np.argsort(_screen(cand, target_t))

    def residual(x9):
        return vec2pat(np.concatenate([x9, fixed9])).reshape(-1) - target_flat

    # TRF init set: top screened grid nodes ...
    inits = [cand[i, :9].copy() for i in order[:N_SCREEN]]
    # ... plus a zero-angle init for the few best distinct speed-sign vectors.
    seen = set()
    for i in order:
        sp = tuple(np.round(cand[i, :3], 4))
        if sp not in seen:
            seen.add(sp)
            z = np.zeros(9); z[:3] = cand[i, :3]
            inits.append(z)
        if len(seen) >= 4:
            break

    # tier 1: cheap descent to re-rank
    tier1 = []
    for x0 in inits:
        try:
            r = least_squares(residual, x0, jac='2-point', bounds=(_LO9, _HI9),
                              method='trf', ftol=1e-12, xtol=1e-12, gtol=1e-12,
                              max_nfev=200)
            tier1.append((float(np.mean(r.fun ** 2)), r.x))
        except Exception:
            pass
    tier1.sort(key=lambda t: t[0])

    # tier 2: full precision on the best few
    best_x, best_mse = None, 1e30
    for _, x9 in tier1[:N_FULL]:
        try:
            r = least_squares(residual, x9, jac='2-point', bounds=(_LO9, _HI9),
                              method='trf', ftol=1e-15, xtol=1e-15, gtol=1e-15,
                              max_nfev=2000)
            m = float(np.mean(r.fun ** 2))
            if m < best_mse:
                best_mse, best_x = m, r.x.copy()
        except Exception:
            pass
        if best_mse < 1e-18:
            break
    return best_x, best_mse


def grid_solve(pattern, fixed9):
    """Adaptive: fast single-triple path; escalate to multi-triple-with-
    replacement (top-MULTI_PEAKS FFT peaks) only when the fast path stalls.
    Replacement lets a single merged FFT peak seed two near-equal speeds,
    which TRF then splits (close-speed cases)."""
    target_flat = pattern.reshape(-1).astype(np.float64)
    target_t = torch.tensor(pattern, dtype=torch.float32, device=DEVICE)
    pf, info = extract_speeds_and_peaks(pattern)

    top3 = tuple(np.sort(pf)[::-1])
    bx, bm = _attempt([top3], fixed9, target_t, target_flat)
    if bm < 1e-12:
        return bx, bm

    # Escalate: combinations-with-replacement of the strongest peaks.
    from itertools import combinations_with_replacement
    peaks = np.sort(info['all_peak_freqs'])[::-1][:MULTI_PEAKS]
    triples = []
    for combo in combinations_with_replacement(range(len(peaks)), P):
        tri = tuple(sorted((float(peaks[i]) for i in combo), reverse=True))
        if tri != top3 and all(f > 0.05 for f in tri):
            triples.append(tri)
    if triples:
        bx2, bm2 = _attempt(triples, fixed9, target_t, target_flat)
        if bm2 < bm:
            bx, bm = bx2, bm2
    return bx, bm


def ml_solve(pattern, fixed9, ang, rem):
    """ML init + 8-sign TRF baseline (reproduces the diagnostic's 9/30)."""
    target_flat = pattern.reshape(-1).astype(np.float64)
    pf, pinfo = extract_speeds_and_peaks(pattern)
    pk = _build_peak_feats_single(pattern, pf, pinfo)
    speed_mags = np.sort(pf)[::-1].astype(np.float64)

    def residual(x9):
        return vec2pat(np.concatenate([x9, fixed9])).reshape(-1) - target_flat

    best_x, best_mse = None, 1e30
    for bits in range(8):
        signs = np.array([(1.0 if (bits >> j) & 1 == 0 else -1.0)
                          for j in range(P)], np.float64)
        ml = ml_init(ang, rem, (signs * speed_mags).astype(np.float32), pk, pattern)
        try:
            res = least_squares(residual, ml[:9], jac='2-point',
                                bounds=(_LO9, _HI9), method='trf',
                                ftol=1e-15, xtol=1e-15, gtol=1e-15, max_nfev=1200)
            m = float(np.mean(res.fun ** 2))
            if m < best_mse:
                best_mse, best_x = m, res.x.copy()
        except Exception:
            pass
        if best_mse < 1e-12:
            break
    return best_x, best_mse


if __name__ == '__main__':
    print("Loading ML models (for baseline)...", flush=True)
    ang, rem = AngleNet(), RemainNet()
    ang.load_state_dict(torch.load(ANG_MODEL, map_location=DEVICE, weights_only=True))
    rem.load_state_dict(torch.load(REM_MODEL, map_location=DEVICE, weights_only=True))
    ang.to(DEVICE); rem.to(DEVICE)

    rng = np.random.default_rng(2026)
    cases = []
    for _ in range(30):
        v = (LO + RG * rng.random(N_PAR).astype(np.float32)).astype(np.float64)
        for j in range(3):
            if abs(v[j]) < 0.15:
                v[j] = np.copysign(0.15, v[j])
        cases.append(canon(v))

    print(f"\n{'='*82}")
    print(f"JOINT ALPHA_X GRID  (N_GRID={N_GRID} -> {N_GRID**3} nodes x 8 signs, "
          f"screen-{N_SCREEN}/full-{N_FULL}, multi-triple top-{MULTI_PEAKS})  vs  ML")
    print(f"{'='*82}\n")

    ml_ok_n = grid_ok_n = either_n = 0
    t0 = time.time()
    for ci, tc in enumerate(cases):
        pat = vec2pat(tc)
        fixed9 = tc[9:].copy()

        tg = time.time()
        gx, gmse = grid_solve(pat, fixed9)
        tg = time.time() - tg

        mx, mmse = ml_solve(pat, fixed9, ang, rem)

        g_err = float(np.max(np.abs(gx - tc[:9]))) if gx is not None else 999
        m_err = float(np.max(np.abs(mx - tc[:9]))) if mx is not None else 999
        g_ok, m_ok = g_err < 1e-3, m_err < 1e-3
        ml_ok_n += m_ok; grid_ok_n += g_ok; either_n += (g_ok or m_ok)

        tag = ""
        if g_ok and not m_ok: tag = "  <<< GRID SAVED"
        elif m_ok and not g_ok: tag = "  (ML only)"
        print(f"  {ci+1:2d}: GRID={'P' if g_ok else 'F'}(e={g_err:.1e}) "
              f"ML={'P' if m_ok else 'F'}(e={m_err:.1e})  "
              f"[{tg:.0f}s]{tag}", flush=True)

    dt = time.time() - t0
    print(f"\n{'='*82}")
    print(f"  GRID (joint alpha_x):  {grid_ok_n}/30")
    print(f"  ML (baseline):         {ml_ok_n}/30")
    print(f"  Either:                {either_n}/30")
    print(f"  Total time: {dt:.0f}s ({dt/30:.1f}s/case)")
    print(f"{'='*82}")
