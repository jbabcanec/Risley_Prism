#!/usr/bin/env python3
"""
solve18_spectral.py -- FULL 18-parameter Risley inverse from one scan pattern.
No brute force, nothing assumed known.

  [N1,N2,N3, ax1..3, ay1..3, ng1..3, d_W, gap, bm_ax, bm_ay, bm_px, bm_py]

Pipeline:
  1. Speeds + fundamental phases/amps: lattice VarPro (spectral_speeds).
  2. ay_i = arg(c_i) (+180 iff ax_i<0, box-resolved); |ax_i| from the
     fundamental amplitude through a cubic gain calibrated at NOMINAL
     glass/geometry (2 forward evals per prism -- deterministic).
  3. Beam angles from the pattern DC: first-order rotating deflections
     average out, so the DC is the static ray -> bm_a = atan(DC/L_nom).
  4. 18-D scipy TRF (masked for TIR glitches) from
     x0 = [N, ax, ay, glass=1.55, d_W=125, gap=8.5, bm from DC, bm_p=0].
  5. Verified ladder (only on failure): gain recalibration with the fitted
     geometry, phase-branch flips, zero angles, alternate spectral bases.

Success = pattern MSE < 1e-12  (empirically -> all 18 params ~1e-9..1e-11).

Run: python paper/solve18_spectral.py
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

from ml_staged_solver import N_PAR, LO, HI, RG, canon, vec2pat, T_PTS, T_OBS, SRC, THK
from spectral_speeds import extract_speeds

DT = T_OBS / T_PTS
_LO = LO.astype(np.float64)
_HI = HI.astype(np.float64)
T_GRID = np.arange(T_PTS) * DT
NG_MID, DW_MID, GAP_MID = 1.55, 125.0, 8.5
L_NOM = SRC + 3 * THK + 2 * GAP_MID + DW_MID


def wrap(d):
    return (d + 180.0) % 360.0 - 180.0


def line_amp_at(pattern, f):
    z = pattern[:, 0] + 1j * pattern[:, 1]
    z = z - z.mean()
    return np.abs(np.vdot(np.exp(2j * np.pi * f * T_GRID), z)) / len(z)


def proj_phases(pattern, N_est):
    z = pattern[:, 0] + 1j * pattern[:, 1]
    z = z - z.mean()
    c = np.array([np.vdot(np.exp(2j * np.pi * f * T_GRID), z) / len(z)
                  for f in N_est])
    return np.degrees(np.angle(c)), np.abs(c)


def calibrate_ax(N_est, fixed9):
    """Per-prism cubic gain amp = a*tan(ax)+b*tan(ax)^3 (2 fwd evals each)."""
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
    roots = np.roots([b, 0.0, a, -amp])
    real = [r.real for r in roots if abs(r.imag) < 1e-9 and
            -0.01 <= r.real <= np.tan(np.radians(22.0))]
    if real:
        return min(real, key=lambda r: abs(r - amp / max(a, 1e-9)))
    return float(np.clip(amp / max(a, 1e-9), 0.0, np.tan(np.radians(20.0))))


def angle_init(N_est, phases, amps, fixed9):
    coefs = calibrate_ax(N_est, fixed9)
    ax, ay, ambiguous = np.zeros(3), np.zeros(3), []
    for i in range(3):
        ph = wrap(phases[i])
        neg = abs(ph) > 90.0
        ay_i = wrap(ph - 180.0) if neg else ph
        mag = np.degrees(np.arctan(invert_cubic(*coefs[i], amps[i])))
        mag = min(mag, 17.5)
        ax[i] = -mag if neg else mag
        ay[i] = np.clip(ay_i, -18.0, 18.0)
        if abs(abs(ph) - 90.0) < 15.0:
            ambiguous.append(i)
    return ax, ay, ambiguous


def nominal_rest(pattern):
    """[ng x3, d_W, gap, bm_ax, bm_ay, bm_px, bm_py] init; beam from DC."""
    dc = pattern.mean(0)
    bm_ax = float(np.clip(np.degrees(np.arctan(dc[0] / L_NOM)), -24.0, 24.0))
    bm_ay = float(np.clip(np.degrees(np.arctan(dc[1] / L_NOM)), -24.0, 24.0))
    return np.array([NG_MID, NG_MID, NG_MID, DW_MID, GAP_MID,
                     bm_ax, bm_ay, 0.0, 0.0])


def trf18(x0, target_flat, rmask, max_nfev=4000):
    def residual(x):
        return (vec2pat(x).reshape(-1) - target_flat)[rmask]
    r = least_squares(residual, np.clip(x0, _LO, _HI), jac='2-point',
                      bounds=(_LO, _HI), method='trf',
                      ftol=1e-15, xtol=1e-15, gtol=1e-15, max_nfev=max_nfev)
    return r.x.copy(), float(np.mean(r.fun ** 2))


def solve18(pattern):
    target_flat = pattern.reshape(-1).astype(np.float64)
    N, info = extract_speeds(pattern, DT)
    if N is None:
        return None, 1e30, 'no-speeds', info
    if 'phases' not in info:
        info['phases'], info['amps'] = proj_phases(pattern, N)
    info['N'] = N.copy()
    smask = info.get('mask', np.ones(len(pattern), bool))
    rmask = np.repeat(smask, 2)
    rest0 = nominal_rest(pattern)

    def attempt(N_est, phases, amps, rest, tag, nfev=4000):
        ax, ay, amb = angle_init(N_est, phases, amps, rest)
        x0 = np.concatenate([N_est, ax, ay, rest])
        bx, bm = trf18(x0, target_flat, rmask, max_nfev=nfev)
        best = (bx, bm, tag)
        if bm < 1e-12:
            return best
        for i in amb[:2]:
            ax2, ay2 = ax.copy(), ay.copy()
            ax2[i] = -ax2[i]
            ph = wrap(phases[i])
            ay2[i] = np.clip(wrap(ph - 180.0) if abs(ph) <= 90.0 else ph,
                             -18.0, 18.0)
            x0 = np.concatenate([N_est, ax2, ay2, rest])
            bx2, bm2 = trf18(x0, target_flat, rmask, max_nfev=1200)
            if bm2 < best[1]:
                best = (bx2, bm2, tag + f'+flip{i}')
            if best[1] < 1e-12:
                return best
        return best

    bx, bm, how = attempt(N, info['phases'], info['amps'], rest0, 'primary')
    if bm < 1e-12:
        return bx, bm, how, info

    # recalibration rung: the first fit's geometry (even off-basin it gets
    # the d_W scale roughly right) -> better amplitude gains -> re-init
    bx2, bm2, how2 = attempt(N, info['phases'], info['amps'],
                             bx[9:].copy(), 'recal', nfev=2000)
    if bm2 < bm:
        bx, bm, how = bx2, bm2, how2
    if bm < 1e-12:
        return bx, bm, how, info

    # zero-angle rung
    x0 = np.concatenate([N, np.zeros(6), rest0])
    bx2, bm2 = trf18(x0, target_flat, rmask, max_nfev=1500)
    if bm2 < bm:
        bx, bm, how = bx2, bm2, 'zero'
    if bm < 1e-12:
        return bx, bm, how, info

    # flip-weakest rung
    w = int(np.argmin(np.abs(N)))
    N2 = N.copy(); N2[w] = -N2[w]
    ph2 = np.array(info['phases'], float).copy(); ph2[w] = -ph2[w]
    bx2, bm2, how2 = attempt(N2, ph2, info['amps'], rest0, 'flipweak',
                             nfev=1500)
    if bm2 < bm:
        bx, bm, how = bx2, bm2, how2
    if bm < 1e-12:
        return bx, bm, how, info

    # alternate spectral bases
    for q, alt in enumerate(info.get('alts', [])[:2]):
        if len(alt) != 3 or np.max(np.abs(alt)) > 3.6:
            continue
        alt = np.asarray(alt, float)
        ph_a, am_a = proj_phases(pattern, alt)
        bx2, bm2, how2 = attempt(alt, ph_a, am_a, rest0, f'alt{q}', nfev=1200)
        if bm2 < bm:
            bx, bm, how = bx2, bm2, how2
        if bm < 1e-12:
            break
    return bx, bm, how, info


if __name__ == '__main__':
    rng = np.random.default_rng(2026)
    cases = []
    for _ in range(30):
        v = (LO + RG * rng.random(N_PAR).astype(np.float32)).astype(np.float64)
        for j in range(3):
            if abs(v[j]) < 0.15: v[j] = np.copysign(0.15, v[j])
        cases.append(canon(v))

    print(f"\n{'='*96}")
    print("SOLVE18 SPECTRAL: ALL 18 parameters from one pattern."
          "  NO brute force, NOTHING assumed known.")
    print(f"{'='*96}")
    print(f"{'#':>3} {'cyc':>5} {'sep':>6}  {'result':<52} {'time':>5}")

    n_ok = n_near = 0
    t0 = time.time()
    for ci, tc in enumerate(cases):
        pat = vec2pat(tc)
        sp = tc[:3]
        sep = min(abs(abs(sp[i]) - abs(sp[j])) for i in range(3) for j in range(i+1, 3))
        cyc = float(min(abs(sp)) * T_OBS)

        tg = time.time()
        x18, mse, how, info = solve18(pat)
        tg = time.time() - tg
        err = float(np.max(np.abs(x18 - tc))) if x18 is not None else 999.0
        ok = err < 1e-3
        near = err < 1e-2
        n_ok += ok; n_near += near

        tag = "" if ok else ("  [close]" if sep < 0.08 else
                             "  [lowcyc]" if cyc < 2.5 else "  [**]")
        print(f"{ci+1:>3} {cyc:>5.1f} {sep:>6.3f}  "
              f"{'PERFECT' if ok else 'near' if near else 'fail':<8}"
              f" err={err:8.1e} mse={mse:8.1e} ({how})"[:74] +
              f" {tg:>4.0f}s{tag}", flush=True)

    dt_all = time.time() - t0
    print(f"\n{'='*96}")
    print(f"  FULL 18-D (no brute force, nothing known): {n_ok}/30 PERFECT"
          f"   ({n_near}/30 within 1e-2)")
    print(f"  {dt_all:.0f}s total ({dt_all/30:.0f}s/case)")
    print(f"{'='*96}")
