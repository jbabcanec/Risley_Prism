#!/usr/bin/env python3
"""
Probe WHY grid cases fail. For selected cases, with glass/geo/beam fixed to truth
and TRUE speeds (isolates the angle question from FFT), run TRF from controlled
inits and see which recovers:

  A  true ax, true ay           -> sanity: must hit ~0 (else degenerate physics)
  B  true ax, ay=0              -> is correct-ax + ay=0 enough? (grid's premise)
  C  nearest-grid ax, ay=0      -> does grid spacing + ay=0 work?
  D  true ax, harmonic ay       -> does harmonic ay help?
  E  ax=0,   ay=0               -> basin width from the origin

Also reports min rotation cycles (min|N|*T_obs) and speed separation.
"""
import sys, os
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'reverse_problem_v2'))
sys.path.insert(0, os.path.dirname(__file__))

from ml_staged_solver import (N_PAR, LO, HI, RG, canon, P, T_PTS, T_OBS,
                              extract_speeds_and_peaks)
from solve_preconditioned import vec2pat

_LO9 = LO[:9].astype(np.float64); _HI9 = HI[:9].astype(np.float64)
_axg = np.linspace(float(LO[3]), float(HI[3]), 9)


def harmonic_ay(speeds, pattern):
    t = np.arange(T_PTS) * (T_OBS / T_PTS)
    X = np.zeros((T_PTS, 2 * P + 1))
    for i in range(P):
        X[:, 2*i] = np.cos(2*np.pi*abs(speeds[i])*t)
        X[:, 2*i+1] = np.sin(2*np.pi*abs(speeds[i])*t)
    X[:, -1] = 1.0
    bx, *_ = np.linalg.lstsq(X, pattern[:, 0], rcond=None)
    ay = np.zeros(P)
    for i in range(P):
        ph = np.degrees(np.arctan2(-bx[2*i+1], bx[2*i]))
        ay[i] = np.clip(ph if speeds[i] >= 0 else -ph, -18, 18)
    return ay


def trf(init9, fixed9, target_flat):
    def res(x9):
        return vec2pat(np.concatenate([x9, fixed9])).reshape(-1) - target_flat
    r = least_squares(res, init9, jac='2-point', bounds=(_LO9, _HI9), method='trf',
                      ftol=1e-15, xtol=1e-15, gtol=1e-15, max_nfev=4000)
    return r.x, float(np.mean(r.fun ** 2))


rng = np.random.default_rng(2026)
cases = []
for _ in range(30):
    v = (LO + RG * rng.random(N_PAR).astype(np.float32)).astype(np.float64)
    for j in range(3):
        if abs(v[j]) < 0.15: v[j] = np.copysign(0.15, v[j])
    cases.append(canon(v))

for ci in [0, 7, 13, 2, 9, 17]:   # 1=solved, 8/14/3/10/18 = failed (0-indexed)
    tc = cases[ci]
    pat = vec2pat(tc); tf = pat.reshape(-1)
    fixed9 = tc[9:].copy()
    sp, ax, ay = tc[:3], tc[3:6], tc[6:9]
    sep = min(abs(abs(sp[i]) - abs(sp[j])) for i in range(3) for j in range(i+1, 3))
    cyc = min(abs(sp)) * T_OBS
    pf, _ = extract_speeds_and_peaks(pat)
    tmag = np.sort(np.abs(sp))[::-1]
    match = all(min(abs(np.sort(pf)[::-1] - m)) < 0.1 for m in tmag)

    ax_grid = np.array([_axg[np.argmin(np.abs(_axg - a))] for a in ax])
    hay = harmonic_ay(sp, pat)

    def err(x):  # max abs error over the 9 recovered params
        return float(np.max(np.abs(x - tc[:9])))

    xa, ma = trf(np.concatenate([sp, ax, ay]),       fixed9, tf)
    xb, mb = trf(np.concatenate([sp, ax, np.zeros(3)]), fixed9, tf)
    xc, mc = trf(np.concatenate([sp, ax_grid, np.zeros(3)]), fixed9, tf)
    xd, md = trf(np.concatenate([sp, ax, hay]),      fixed9, tf)
    xe, me = trf(np.concatenate([sp, np.zeros(3), np.zeros(3)]), fixed9, tf)

    print(f"\nCASE {ci+1}: |N|={np.round(tmag,3)} sep={sep:.3f} min_cyc={cyc:.1f} "
          f"FFTmatch={match}")
    print(f"   true ax={np.round(ax,1)}  ay={np.round(ay,1)}  hay={np.round(hay,1)}")
    print(f"   A true-ax/true-ay : err={err(xa):.1e}  {'OK' if err(xa)<1e-3 else 'DEGEN'}")
    print(f"   B true-ax/ay=0    : err={err(xb):.1e}  {'OK' if err(xb)<1e-3 else 'FAIL'}")
    print(f"   C grid-ax/ay=0    : err={err(xc):.1e}  {'OK' if err(xc)<1e-3 else 'FAIL'}")
    print(f"   D true-ax/harm-ay : err={err(xd):.1e}  {'OK' if err(xd)<1e-3 else 'FAIL'}")
    print(f"   E ax=0/ay=0       : err={err(xe):.1e}  {'OK' if err(xe)<1e-3 else 'FAIL'}")
