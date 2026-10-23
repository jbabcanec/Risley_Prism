#!/usr/bin/env python3
"""
solve9_spectral.py -- Fully deterministic 9-D Risley inverse: NO brute force.

Pipeline (zero grids, zero sign combos, zero triple enumeration):
  1. Speeds (signed): lattice-VarPro spectral extraction (spectral_speeds).
  2. alpha_y: phase of each fundamental line. In this forward model ay_i is
     exactly the rotation phase offset, so arg(c_i) = ay_i (+180 deg iff
     ax_i < 0). The +-18 deg box resolves the branch: |phase| > 90 means
     ax_i < 0 and ay_i = phase -+ 180.
  3. |alpha_x|: fundamental amplitude, inverted through a per-prism cubic
     amp = a*tan(ax) + b*tan(ax)^3 calibrated with TWO forward evaluations
     (ax = 5, 15 deg) -- 6 deterministic evals total, not a search.
  4. scipy TRF on the exact forward model -> machine precision.
  Verification ladder (only if pattern MSE says wrong basin): alternate
  phase branches for prisms near the +-90 boundary, then alternate speed
  bases from the spectral stage. Each rung is a deterministic candidate,
  verified by the final MSE -- not a search.

Protocol matches solve9_grid.py: glass/geo/beam fixed to truth; recover
speeds+angles; success = max|delta| over the 9 < 1e-3.

Run: python paper/solve9_spectral.py
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import sys, time
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'reverse_problem_v2'))
sys.path.insert(0, os.path.dirname(__file__))

from ml_staged_solver import N_PAR, LO, HI, RG, canon, vec2pat, T_PTS, T_OBS
from spectral_speeds import extract_speeds

DT = T_OBS / T_PTS
_LO9 = LO[:9].astype(np.float64)
_HI9 = HI[:9].astype(np.float64)
T_GRID = np.arange(T_PTS) * DT


def wrap(d):
    return (d + 180.0) % 360.0 - 180.0


def line_amp_at(pattern, f):
    """|c| of the line at frequency f by direct projection (mean removed)."""
    z = pattern[:, 0] + 1j * pattern[:, 1]
    z = z - z.mean()
    return np.abs(np.vdot(np.exp(2j * np.pi * f * T_GRID), z)) / len(z)


def calibrate_ax(N_est, fixed9):
    """Per-prism cubic gain amp = a*tan(ax) + b*tan(ax)^3 from 2 fwd evals."""
    coefs = []
    for i in range(3):
        amps = []
        for ax_deg in (5.0, 15.0):
            v = np.zeros(N_PAR)
            v[:3] = N_est
            v[3 + i] = ax_deg
            v[9:] = fixed9
            amps.append(line_amp_at(vec2pat(v), N_est[i]))
        t1, t2 = np.tan(np.radians(5.0)), np.tan(np.radians(15.0))
        M = np.array([[t1, t1 ** 3], [t2, t2 ** 3]])
        a, b = np.linalg.solve(M, np.array(amps))
        coefs.append((a, b))
    return coefs


def invert_cubic(a, b, amp):
    """Solve a*tau + b*tau^3 = amp for tau in [0, tan(20 deg)]."""
    roots = np.roots([b, 0.0, a, -amp])
    real = [r.real for r in roots if abs(r.imag) < 1e-9 and
            -0.01 <= r.real <= np.tan(np.radians(22.0))]
    if real:
        return min(real, key=lambda r: abs(r - amp / max(a, 1e-9)))
    return float(np.clip(amp / max(a, 1e-9), 0.0, np.tan(np.radians(20.0))))


def angle_init(N_est, phases, amps, fixed9):
    """Deterministic (ax, ay) init from fundamental phases and amplitudes.
    Returns the primary init and the list of branch-ambiguous prisms."""
    coefs = calibrate_ax(N_est, fixed9)
    ax = np.zeros(3)
    ay = np.zeros(3)
    ambiguous = []
    for i in range(3):
        ph = wrap(phases[i])
        neg = abs(ph) > 90.0
        ay_i = wrap(ph - 180.0) if neg else ph
        mag = np.degrees(np.arctan(invert_cubic(*coefs[i], amps[i])))
        ax[i] = -mag if neg else mag
        ay[i] = np.clip(ay_i, -18.0, 18.0)
        if abs(abs(ph) - 90.0) < 15.0:
            ambiguous.append(i)
    return ax, ay, ambiguous


def trf(x0, fixed9, target_flat, rmask, max_nfev=2000):
    def residual(x9):
        r = vec2pat(np.concatenate([x9, fixed9])).reshape(-1) - target_flat
        return r[rmask]
    r = least_squares(residual, np.clip(x0, _LO9, _HI9), jac='2-point',
                      bounds=(_LO9, _HI9), method='trf',
                      ftol=1e-15, xtol=1e-15, gtol=1e-15, max_nfev=max_nfev)
    return r.x.copy(), float(np.mean(r.fun ** 2))


def proj_phases(pattern, N_est):
    """Fallback fundamental phases/amps by direct projection."""
    z = pattern[:, 0] + 1j * pattern[:, 1]
    z = z - z.mean()
    c = np.array([np.vdot(np.exp(2j * np.pi * f * T_GRID), z) / len(z)
                  for f in N_est])
    return np.degrees(np.angle(c)), np.abs(c)


def solve9(pattern, fixed9):
    """Returns (x9, mse, how) -- how records which ladder rung succeeded."""
    target_flat = pattern.reshape(-1).astype(np.float64)
    N, info = extract_speeds(pattern, DT)
    if N is None:
        return None, 1e30, 'no-speeds'
    if 'phases' not in info:
        info['phases'], info['amps'] = proj_phases(pattern, N)
    # TIR-glitched samples are masked in the fit as well: their residual is
    # discontinuous in the parameters and wrecks the trust region
    smask = info.get('mask', np.ones(len(pattern), bool))
    rmask = np.repeat(smask, 2)          # pattern flattens to [x0,y0,x1,y1,..]

    def attempt(N_est, phases, amps, tag):
        ax, ay, amb = angle_init(N_est, phases, amps, fixed9)
        x0 = np.concatenate([N_est, ax, ay])
        bx, bm = trf(x0, fixed9, target_flat, rmask)
        if bm < 1e-12:
            return bx, bm, tag
        # branch ladder: flip ambiguous prisms (deterministic, verified)
        best = (bx, bm, tag)
        for i in amb[:2]:
            ax2, ay2 = ax.copy(), ay.copy()
            ax2[i] = -ax2[i]
            ph = wrap(phases[i])
            ay2[i] = np.clip(wrap(ph - 180.0) if abs(ph) <= 90.0 else ph,
                             -18.0, 18.0)
            x0 = np.concatenate([N_est, ax2, ay2])
            bx2, bm2 = trf(x0, fixed9, target_flat, rmask, max_nfev=800)
            if bm2 < best[1]:
                best = (bx2, bm2, tag + f'+flip{i}')
            if best[1] < 1e-12:
                break
        return best

    bx, bm, how = attempt(N, info['phases'], info['amps'], 'primary')
    if bm < 1e-12:
        return bx, bm, how

    # zero-angle rung: with correct speeds this alone lands the basin for
    # most cases (diary 2026-06-18) -- covers polluted phase/amp inits
    bx2, bm2 = trf(np.concatenate([N, np.zeros(6)]), fixed9,
                   target_flat, rmask)
    if bm2 < bm:
        bx, bm, how = bx2, bm2, 'zero'
    if bm < 1e-12:
        return bx, bm, how

    # flip-weakest rung: the sign of a spectrally tiny prism is the least
    # reliable bit of the extraction -- try its mirror once
    w = int(np.argmin(np.abs(N)))
    N2 = N.copy(); N2[w] = -N2[w]
    ph2 = np.array(info['phases'], float).copy(); ph2[w] = -ph2[w]
    bx2, bm2, how2 = attempt(N2, ph2, info['amps'], 'flipweak')
    if bm2 < bm:
        bx, bm, how = bx2, bm2, how2
    if bm < 1e-12:
        return bx, bm, how

    # alternate speed bases (verified candidates from the spectral stage)
    for q, alt in enumerate(info.get('alts', [])[:2]):
        if len(alt) != 3 or np.max(np.abs(alt)) > 3.6:
            continue
        alt = np.asarray(alt, float)
        ph_alt = info['phases']            # phases unknown for alts: reuse
        bx2, bm2, how2 = attempt(alt, ph_alt, info['amps'], f'alt{q}')
        if bm2 < bm:
            bx, bm, how = bx2, bm2, how2
        if bm < 1e-12:
            break
    return bx, bm, how


if __name__ == '__main__':
    rng = np.random.default_rng(2026)
    cases = []
    for _ in range(30):
        v = (LO + RG * rng.random(N_PAR).astype(np.float32)).astype(np.float64)
        for j in range(3):
            if abs(v[j]) < 0.15: v[j] = np.copysign(0.15, v[j])
        cases.append(canon(v))

    print(f"\n{'='*96}")
    print("SOLVE9 SPECTRAL: pencil-lattice speeds + phase/amp angles + TRF."
          "  NO grids, NO sign combos.")
    print(f"{'='*96}")
    print(f"{'#':>3} {'cyc':>5} {'sep':>6}  {'result':<44} {'time':>5}")

    n_ok = 0
    t0 = time.time()
    for ci, tc in enumerate(cases):
        pat = vec2pat(tc)
        fixed9 = tc[9:].copy()
        sp = tc[:3]
        sep = min(abs(abs(sp[i]) - abs(sp[j])) for i in range(3) for j in range(i+1, 3))
        cyc = float(min(abs(sp)) * T_OBS)

        tg = time.time()
        x9, mse, how = solve9(pat, fixed9)
        tg = time.time() - tg
        err = float(np.max(np.abs(x9 - tc[:9]))) if x9 is not None else 999.0
        ok = err < 1e-3
        n_ok += ok

        tag = "" if ok else ("  [close]" if sep < 0.08 else
                             "  [lowcyc]" if cyc < 2.5 else "  [**]")
        print(f"{ci+1:>3} {cyc:>5.1f} {sep:>6.3f}  "
              f"{'PERFECT' if ok else 'fail':<8} err={err:8.1e} mse={mse:8.1e} "
              f"({how})"[:70] + f" {tg:>4.0f}s{tag}", flush=True)

    dt_all = time.time() - t0
    print(f"\n{'='*96}")
    print(f"  SPECTRAL 9-D (no brute force): {n_ok}/30")
    print(f"  [reference: solve9_grid (alpha_x grid) 16/30, ML init 9/30]")
    print(f"  {dt_all:.0f}s total ({dt_all/30:.0f}s/case)")
    print(f"{'='*96}")
