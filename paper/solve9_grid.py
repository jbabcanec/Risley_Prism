#!/usr/bin/env python3
"""
solve9_grid.py -- 9-D Risley angle+speed recovery by speed-triple selection
followed by a joint alpha_x grid.

Two empirical facts drive the design (see test_trueseed_angles.py / test_basin_probe.py):
  1. GIVEN the speeds, recovering the 6 angles is easy: a joint 3-D grid over the
     three alpha_x (alpha_y=0), screened by the differentiable model and finished
     with one joint TRF, recovers 29/30; adding a zero-angle init makes it 30/30.
     The angle basin is WIDE in alpha_y and the grid covers the narrow alpha_x.
  2. The real bottleneck is SPEED extraction. The FFT top-3 peaks are often
     harmonics/cross-terms, not fundamentals.

So: enumerate candidate speed triples from the top-K FFT peaks, RANK them by a
cheap coarse-grid forward screen (per triple, so the right one is never diluted),
then run the full fine alpha_x grid + TRF on only the best few triples.

Protocol matches the diagnostics: glass/geometry/beam fixed to truth; recover
speeds+angles; success = max|delta| over the 9 < 1e-3.

Run: python paper/solve9_grid.py            # GRID only, 30 cases
     python paper/solve9_grid.py --ml        # also run the ML baseline
"""
import sys, os, time, argparse
from itertools import combinations
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'reverse_problem_v2'))
sys.path.insert(0, os.path.dirname(__file__))

import torch
from ml_staged_solver import (N_PAR, LO, HI, RG, canon, P, T_PTS, T_OBS, DT,
                              FREQS, DiffFwd, DEVICE)
from solve_preconditioned import vec2pat

# ---- config ----
K_PEAKS   = 12                 # FFT peaks to draw speed triples from
N_TRIPLES = 5                  # triples (after ranking) given a full angle solve
N_GRID    = 9                  # fine alpha_x grid points per prism
N_COARSE  = 5                  # coarse alpha_x points per prism (triple ranking)
N_SCREEN  = 16                 # fine nodes given a full TRF (per triple)

_LO9 = LO[:9].astype(np.float64); _HI9 = HI[:9].astype(np.float64)
_fine = np.linspace(float(LO[3]), float(HI[3]), N_GRID)
_coarse = np.linspace(float(LO[3]), float(HI[3]), N_COARSE)
FINE_AX = np.stack(np.meshgrid(_fine, _fine, _fine, indexing='ij'), -1).reshape(-1, 3)
COARSE_AX = np.stack(np.meshgrid(_coarse, _coarse, _coarse, indexing='ij'), -1).reshape(-1, 3)
_FWD = DiffFwd().to(DEVICE)
_LO9_T = torch.tensor(_LO9, device=DEVICE).float()
_HI9_T = torch.tensor(_HI9, device=DEVICE).float()
_SIGNS = np.array([[1.0 if (b >> j) & 1 == 0 else -1.0 for j in range(P)]
                   for b in range(8)], np.float64)


def top_peaks(pattern, K=K_PEAKS):
    """Greedy top-K spectral peaks (|FFT_x|+|FFT_y|), with +/-2 bin suppression."""
    fx = np.fft.rfft(pattern[:, 0]); fy = np.fft.rfft(pattern[:, 1])
    pw = (np.abs(fx) + np.abs(fy)); pw[0] = 0.0
    out = []
    for _ in range(K):
        i = int(np.argmax(pw))
        if pw[i] <= 0:
            break
        out.append(float(FREQS[i]))
        pw[max(1, i-2):i+3] = 0.0
    return out


def _screen(cand, target_t, chunk=20000):
    out = []
    with torch.no_grad():
        for i in range(0, len(cand), chunk):
            c = torch.tensor(cand[i:i+chunk], dtype=torch.float32, device=DEVICE)
            p = _FWD(c, high_precision=False)
            out.append(((p - target_t) ** 2).mean(dim=(1, 2)).cpu().numpy())
    return np.concatenate(out)


def _nodes(mags, grid, fixed9):
    """All 8 sign combos x grid nodes for one freq-magnitude triple."""
    n = grid.shape[0]
    blocks = []
    for s in _SIGNS:
        blk = np.zeros((n, N_PAR))
        blk[:, :3] = s * mags
        blk[:, 3:6] = grid
        blk[:, 9:] = fixed9
        blocks.append(blk)
    return np.concatenate(blocks, 0)


def _batched_adam(inits9, fixed9, target_t, steps=250, lr=0.03):
    """Refine MANY 9-vec inits (speeds+angles) in PARALLEL through DiffFwd, with
    glass/geo/beam held at fixed9. One tensor op per step descends every node at
    once -- far cheaper than per-node scipy TRF. Returns refined (M,9) + MSEs."""
    M = inits9.shape[0]
    par = torch.tensor(inits9, dtype=torch.float32, device=DEVICE, requires_grad=True)
    fix = torch.tensor(np.broadcast_to(fixed9, (M, 9)).copy(),
                       dtype=torch.float32, device=DEVICE)
    tgt = target_t.unsqueeze(0)
    opt = torch.optim.Adam([par], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        pred = _FWD(torch.cat([par, fix], 1), high_precision=False)
        ((pred - tgt) ** 2).mean(dim=(1, 2)).sum().backward()
        opt.step()
        with torch.no_grad():
            par.clamp_(_LO9_T, _HI9_T)
    with torch.no_grad():
        pred = _FWD(torch.cat([par, fix], 1), high_precision=False)
        mses = ((pred - tgt) ** 2).mean(dim=(1, 2)).cpu().numpy()
    return par.detach().cpu().numpy().astype(np.float64), mses


def _angle_solve(mags, fixed9, target_t, target_flat, n_adam=48, steps=250):
    """One freq triple: screen the 8x729 alpha_x grid, parallel-Adam-refine the
    top nodes (+ zero-angle inits), then scipy-TRF the best few to precision."""
    cand = _nodes(mags, FINE_AX, fixed9)
    order = np.argsort(_screen(cand, target_t))
    inits = cand[order[:n_adam], :9].copy()
    seen, extra = set(), []
    for i in order:                              # zero-angle insurance per sign
        sp = tuple(np.round(cand[i, :3], 4))
        if sp not in seen:
            seen.add(sp); z = np.zeros(9); z[:3] = cand[i, :3]; extra.append(z)
        if len(seen) >= 4:
            break
    if extra:
        inits = np.vstack([inits, np.array(extra)])

    refined, amse = _batched_adam(inits, fixed9, target_t, steps=steps)

    def residual(x9):
        return vec2pat(np.concatenate([x9, fixed9])).reshape(-1) - target_flat

    best_x, best_mse = None, 1e30
    for idx in np.argsort(amse)[:4]:
        try:
            r = least_squares(residual, refined[idx], jac='2-point',
                              bounds=(_LO9, _HI9), method='trf',
                              ftol=1e-15, xtol=1e-15, gtol=1e-15, max_nfev=2000)
            m = float(np.mean(r.fun ** 2))
            if m < best_mse:
                best_mse, best_x = m, r.x.copy()
        except Exception:
            pass
        if best_mse < 1e-18:
            break
    return best_x, best_mse


def solve9(pattern, fixed9):
    target_flat = pattern.reshape(-1).astype(np.float64)
    target_t = torch.tensor(pattern, dtype=torch.float32, device=DEVICE)
    peaks = top_peaks(pattern)
    top3 = tuple(sorted(peaks[:3], reverse=True))

    # Fast path: the top-3 peaks.
    bx, bm = _angle_solve(np.array(top3), fixed9, target_t, target_flat)
    if bm < 1e-12:
        return bx, bm

    # Build distinct triples from the top-K peaks and RANK them by a cheap
    # per-triple coarse-grid screen (best node MSE) -- no global dilution.
    triples = []
    for c in combinations(range(len(peaks)), P):
        tri = tuple(sorted((peaks[i] for i in c), reverse=True))
        if tri != top3 and all(f > 0.05 for f in tri):
            triples.append(tri)
    scored = []
    for tri in triples:
        cand = _nodes(np.array(tri), COARSE_AX, fixed9)
        scored.append((float(_screen(cand, target_t).min()), tri))
    scored.sort(key=lambda t: t[0])

    for _, tri in scored[:N_TRIPLES]:
        x, m = _angle_solve(np.array(tri), fixed9, target_t, target_flat)
        if m < bm:
            bx, bm = x, m
        if bm < 1e-12:
            break
    return bx, bm


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ml', action='store_true', help='also run ML baseline')
    args = ap.parse_args()

    ang = rem = None
    if args.ml:
        from ml_staged_solver import AngleNet, RemainNet, ANG_MODEL, REM_MODEL
        from test_alphax_grid import ml_solve
        ang, rem = AngleNet(), RemainNet()
        ang.load_state_dict(torch.load(ANG_MODEL, map_location=DEVICE, weights_only=True))
        rem.load_state_dict(torch.load(REM_MODEL, map_location=DEVICE, weights_only=True))
        ang.to(DEVICE); rem.to(DEVICE)

    rng = np.random.default_rng(2026)
    cases = []
    for _ in range(30):
        v = (LO + RG * rng.random(N_PAR).astype(np.float32)).astype(np.float64)
        for j in range(3):
            if abs(v[j]) < 0.15: v[j] = np.copysign(0.15, v[j])
        cases.append(canon(v))

    print(f"\n{'='*86}")
    print(f"SPEED-TRIPLE SELECTION + JOINT ALPHA_X GRID  (9-D, glass/geo/beam fixed)")
    print(f"{'='*86}")
    print(f"{'#':>3} {'min_cyc':>7} {'sep':>6}  result")

    g_ok = m_ok = 0
    t0 = time.time()
    for ci, tc in enumerate(cases):
        pat = vec2pat(tc); fixed9 = tc[9:].copy()
        sp = tc[:3]
        sep = min(abs(abs(sp[i]) - abs(sp[j])) for i in range(3) for j in range(i+1, 3))
        cyc = float(min(abs(sp)) * T_OBS)

        tg = time.time()
        gx, gm = solve9(pat, fixed9)
        tg = time.time() - tg
        gerr = float(np.max(np.abs(gx - tc[:9]))) if gx is not None else 999
        gok = gerr < 1e-3; g_ok += gok

        ml_str = ""
        if args.ml:
            mx, mm = ml_solve(pat, fixed9, ang, rem)
            merr = float(np.max(np.abs(mx - tc[:9]))) if mx is not None else 999
            mok = merr < 1e-3; m_ok += mok
            saved = "  <<< GRID SAVED" if gok and not mok else (
                    "  (ML only)" if mok and not gok else "")
            ml_str = f"  ML={'P' if mok else 'F'}(e={merr:.0e}){saved}"

        flag = "" if gok else ("  [close-speed]" if sep < 0.08
                               else "  [low-cycle]" if cyc < 2.5 else "  [**]")
        print(f"{ci+1:>3} {cyc:>7.1f} {sep:>6.3f}  GRID={'P' if gok else 'F'}"
              f"(e={gerr:.0e}) [{tg:.0f}s]{ml_str}{flag if not gok else ''}",
              flush=True)

    dt = time.time() - t0
    print(f"\n{'='*86}")
    print(f"  GRID (triple-select + alpha_x grid): {g_ok}/30")
    if args.ml:
        print(f"  ML baseline:                         {m_ok}/30")
    print(f"  Total {dt:.0f}s ({dt/30:.0f}s/case)")
    print(f"{'='*86}")
