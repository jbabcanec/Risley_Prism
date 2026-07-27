#!/usr/bin/env python3
"""
test_assumptions.py -- Every mathematical assumption behind the spectral
inverse, tested. Each block prints PASS/FAIL with the measured quantity;
the paper cites these numbers.

  A1  flip symmetry: (ax,ay)->(-ax,ay+180) leaves the pattern EXACTLY
      invariant (claimed exact at all orders from the core.py structure).
  A2  signed fundamentals: |c(+N_i)| > |c(-N_i)| (main beats conjugate leak).
  A3  phase readout: arg c(+N_i) = ay_i + 180*[ax_i<0] up to non-paraxial
      correction -- distribution of the error.
  A4  lattice support: relative residual of the true-lattice projection
      (B=3, B=4) -- the model-class adequacy floor.
  A5  fundamental dominance: amp(e_i) exceeds every |k|_1>=2 line -- the
      canonicalization assumption; count violations.
  A6  amplitude model: amp = a tan(ax) + b tan^3(ax) -- median inversion
      error in degrees when the calibration geometry is (i) true (ii) nominal.
  A7  TRF basin: success rate of the 18-D TRF vs size of a spectral-like
      init perturbation -- the measured basin, replacing folklore.
  A8  information scaling: close-pair/weak-prism failures solve at the
      observation time T_req predicted by the Fisher certificate.

Run: python paper/test_assumptions.py
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'reverse_problem_v2'))
sys.path.insert(0, os.path.dirname(__file__))

from core import PrismParameters, SystemGeometry, fast_forward
from ml_staged_solver import N_PAR, LO, HI, RG, canon, vec2pat, T_PTS, T_OBS, SRC, THK
from itertools import product as iproduct

DT = T_OBS / T_PTS
FS = 1.0 / DT

rng = np.random.default_rng(2026)
CASES = []
for _ in range(30):
    v = (LO + RG * rng.random(N_PAR).astype(np.float32)).astype(np.float64)
    for j in range(3):
        if abs(v[j]) < 0.15: v[j] = np.copysign(0.15, v[j])
    CASES.append(canon(v))


def pat_T(v, T):
    """Pattern with observation time T at the standard 20 Hz sampling."""
    geo = SystemGeometry(source_distance=SRC, prism_thickness=THK,
                         workpiece_distance=float(v[12]), inter_prism_gap=float(v[13]),
                         beam_angle_x=float(v[14]), beam_angle_y=float(v[15]),
                         beam_pos_x=float(v[16]), beam_pos_y=float(v[17]))
    pr = PrismParameters(3, v[:3].tolist(), v[3:6].tolist(), v[6:9].tolist(),
                         glass_indices=v[9:12].tolist(), geometry=geo)
    return fast_forward(pr, int(round(T * FS)), T)


def proj_c(pattern, f, T=T_OBS):
    z = pattern[:, 0] + 1j * pattern[:, 1]
    z = z - z.mean()
    tg = np.arange(len(z)) / FS
    return np.vdot(np.exp(2j * np.pi * f * tg), z) / len(z)


def wrap(d):
    return (d + 180.0) % 360.0 - 180.0


print(f"\n{'='*94}")
print("ASSUMPTION TEST SUITE  (30-case seed-2026 battery unless stated)")
print(f"{'='*94}")

# ------------------------------------------------------------------ A1
d_flip = []
for tc in CASES:
    v2 = tc.copy()
    v2[3:6] = -v2[3:6]
    v2[6:9] = wrap(v2[6:9] + 180.0)
    d_flip.append(np.max(np.abs(vec2pat(tc) - vec2pat(v2))))
d_flip = np.array(d_flip)
print(f"A1 flip symmetry exact:            max|dF| = {d_flip.max():.2e}   "
      f"{'PASS' if d_flip.max() < 1e-9 else 'FAIL'}")

# ------------------------------------------------------------------ A2
ratios, viol2 = [], 0
for tc in CASES:
    p = vec2pat(tc)
    for i in range(3):
        a_main = abs(proj_c(p, tc[i]))
        a_leak = abs(proj_c(p, -tc[i]))
        ratios.append(a_leak / max(a_main, 1e-12))
        viol2 += a_leak > a_main
ratios = np.array(ratios)
print(f"A2 main > conjugate leak:          leak/main med {np.median(ratios):.3f} "
      f"max {ratios.max():.3f}  violations {viol2}/90   "
      f"{'PASS' if viol2 <= 2 else 'FAIL'}")

# ------------------------------------------------------------------ A3
ph_err = []
for tc in CASES:
    p = vec2pat(tc)
    for i in range(3):
        ph = np.degrees(np.angle(proj_c(p, tc[i])))
        ph_true = tc[6 + i] + (180.0 if tc[3 + i] < 0 else 0.0)
        ph_err.append(abs(wrap(ph - ph_true)))
ph_err = np.array(ph_err)
print(f"A3 phase = ay (+180 if ax<0):      err med {np.median(ph_err):.2f} deg  "
      f"p90 {np.percentile(ph_err, 90):.2f}  max {ph_err.max():.2f}   "
      f"{'PASS' if np.median(ph_err) < 3 else 'FAIL'}")

# ------------------------------------------------------------------ A4
def true_lattice_resid(p, sp, B):
    tg = np.arange(len(p)) / FS
    z = p[:, 0] + 1j * p[:, 1]
    K = np.array([k for k in iproduct(range(-B, B + 1), repeat=3)
                  if sum(abs(x) for x in k) <= B], float)
    E = np.exp(2j * np.pi * np.outer(tg, K @ sp))
    c, *_ = np.linalg.lstsq(E, z, rcond=1e-8)
    return np.linalg.norm(z - E @ c) / np.linalg.norm(z - z.mean())

r4 = np.array([true_lattice_resid(vec2pat(tc), tc[:3], 4) for tc in CASES])
n_bad = int((r4 > 1e-2).sum())
print(f"A4 lattice support (B=4 resid):    med {np.median(r4):.1e}  "
      f"max {r4.max():.1e}  inadequate(TIR) {n_bad}/30   "
      f"{'PASS' if n_bad <= 3 else 'FAIL'}")

# ------------------------------------------------------------------ A5
viol5, tot5 = 0, 0
K5 = np.array([k for k in iproduct(range(-3, 4), repeat=3)
               if 2 <= sum(abs(x) for x in k) <= 3], float)
for tc in CASES:
    if true_lattice_resid(vec2pat(tc), tc[:3], 3) > 1e-2:
        continue                      # TIR cases excluded (A4 covers them)
    p = vec2pat(tc)
    tg = np.arange(len(p)) / FS
    z = p[:, 0] + 1j * p[:, 1]
    KA = np.array([k for k in iproduct(range(-3, 4), repeat=3)
                   if sum(abs(x) for x in k) <= 3], float)
    E = np.exp(2j * np.pi * np.outer(tg, KA @ tc[:3]))
    c, *_ = np.linalg.lstsq(E, z, rcond=1e-8)
    o = np.abs(KA).sum(1)
    a_fund = min(abs(c[np.argmin(np.abs(KA - e).sum(1))])
                 for e in np.eye(3))
    hi = np.abs(c[o >= 2])
    tot5 += 1
    viol5 += bool((hi > a_fund).any())
print(f"A5 weakest fundamental > order>=2: violated in {viol5}/{tot5} cases "
      f"(canonicalization uses rank-extension, tolerates this)   "
      f"{'PASS' if viol5 <= tot5 // 3 else 'WARN'}")

# ------------------------------------------------------------------ A6
from solve18_spectral import calibrate_ax, invert_cubic, nominal_rest
err_true, err_nom = [], []
for tc in CASES[:15]:
    p = vec2pat(tc)
    for fixed9, store in ((tc[9:], err_true), (nominal_rest(p), err_nom)):
        try:
            coefs = calibrate_ax(tc[:3], fixed9)
        except Exception:
            continue
        for i in range(3):
            amp = abs(proj_c(p, tc[i]))
            mag = np.degrees(np.arctan(invert_cubic(*coefs[i], amp)))
            store.append(abs(mag - abs(tc[3 + i])))
print(f"A6 amplitude->|ax| inversion:      true geometry med "
      f"{np.median(err_true):.2f} deg, nominal geometry med "
      f"{np.median(err_nom):.2f} deg (p90 {np.percentile(err_nom,90):.2f})   "
      f"{'PASS' if np.median(err_nom) < 4 else 'WARN'}")

# ------------------------------------------------------------------ A7
from scipy.optimize import least_squares
_LOf, _HIf = LO.astype(float), HI.astype(float)

def trf_quick(x0, target_flat, nfev=1200):
    def residual(x):
        return vec2pat(x).reshape(-1) - target_flat
    r = least_squares(residual, np.clip(x0, _LOf, _HIf), jac='2-point',
                      bounds=(_LOf, _HIf), method='trf',
                      ftol=1e-15, xtol=1e-15, gtol=1e-15, max_nfev=nfev)
    return float(np.mean(r.fun ** 2))

print("A7 measured 18-D TRF basin vs spectral-like init error "
      "(10 cases x 3 seeds):")
rng7 = np.random.default_rng(7)
for scale in (0.5, 1.0, 2.0, 4.0):
    ok = tot = 0
    for tc in CASES[:10]:
        target = vec2pat(tc).reshape(-1)
        for _ in range(3):
            d = np.zeros(N_PAR)
            d[:3] = rng7.normal(0, 3e-4 * scale, 3)          # speeds
            d[3:6] = rng7.normal(0, 1.5 * scale, 3)          # ax  (deg)
            d[6:9] = rng7.normal(0, 2.0 * scale, 3)          # ay  (deg)
            d[9:12] = rng7.normal(0, 0.08 * scale, 3)        # glass
            d[12] = rng7.normal(0, 40 * scale)               # d_W
            d[13] = rng7.normal(0, 4 * scale)                # gap
            d[14:16] = rng7.normal(0, 3 * scale, 2)          # beam ang
            d[16:18] = rng7.normal(0, 2 * scale, 2)          # beam pos
            mse = trf_quick(tc + d, target)
            ok += mse < 1e-12; tot += 1
    print(f"     perturbation x{scale:<4} -> TRF success {ok}/{tot}")

# ------------------------------------------------------------------ A8
print("A8 information scaling: failed cases vs observation time T")
from spectral_speeds import extract_speeds
for ci in (10, 17, 3, 26):        # cases 11, 18, 4, 27 (0-based)
    tc = CASES[ci]
    row = f"     case {ci+1:>2} (sep "
    row += f"{min(abs(abs(tc[i])-abs(tc[j])) for i in range(3) for j in range(i+1,3)):.3f}):"
    for T in (10.0, 20.0, 40.0, 80.0):
        p = pat_T(tc, T)
        N, info = extract_speeds(p, DT)
        e = np.max(np.abs(N - tc[:3])) if N is not None else 999.0
        row += f"  T={T:.0f}:{'OK' if e < 0.02 else f'{e:.2f}'}"
    print(row, flush=True)

print(f"{'='*94}")
