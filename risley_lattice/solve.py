"""Completion stage: trust-region solve from the spectral init, with the
verified candidate ladder. Every rung is deterministic and accepted only by
the pattern-MSE certificate -- no search.

solve9(pattern, fixed9): 9-D protocol (glass/geometry/beam known).
solve18(pattern):        full 18-D, nothing assumed known.
"""
import itertools
import os

import numpy as np
from scipy.optimize import least_squares

from .model import N_PAR, LO, HI, DT, vec2pat
from .spectral import extract_speeds
from .angles import angle_init, proj_phases, nominal_rest, wrap

_LO = LO.copy()
_HI = HI.copy()
_LO9 = LO[:9].copy()
_HI9 = HI[:9].copy()
MSE_OK = 1e-12
# Final verified rung (2026-10): flip the sign of ax for prism pairs / the
# triple / singletons, warm-started from the ladder's best answer. Runs only
# when every other rung failed, so it cannot change a case that already
# solves. Disable with RISLEY_SIGN_SWEEP=0.
SIGN_SWEEP = os.environ.get('RISLEY_SIGN_SWEEP', '1') != '0'
_SS_NFEV = {10.0: 1200, 20.0: 1000, 40.0: 600, 80.0: 400}


def trf(x0, target_flat, rmask, lo, hi, fixed9=None, max_nfev=2000,
        n_pts=None, time_limit=None):
    """Masked TRF on the exact model. fixed9 given -> 9-D protocol.
    n_pts/time_limit default to the standard 200-sample, 10 s recording."""
    if n_pts is None:
        n_pts = len(rmask) // 2
    if time_limit is None:
        time_limit = n_pts * DT
    if fixed9 is None:
        def residual(x):
            return (vec2pat(x, n_pts, time_limit).reshape(-1)
                    - target_flat)[rmask]
    else:
        def residual(x):
            v = np.concatenate([x, fixed9])
            return (vec2pat(v, n_pts, time_limit).reshape(-1)
                    - target_flat)[rmask]
    r = least_squares(residual, np.clip(x0, lo, hi), jac='2-point',
                      bounds=(lo, hi), method='trf',
                      ftol=1e-15, xtol=1e-15, gtol=1e-15, max_nfev=max_nfev)
    return r.x.copy(), float(np.mean(r.fun ** 2))


def _prep(pattern):
    target_flat = pattern.reshape(-1).astype(np.float64)
    N, info = extract_speeds(pattern, DT)
    if N is None:
        return None, None, None, None, info
    if 'phases' not in info:
        info['phases'], info['amps'] = proj_phases(pattern, N)
    smask = info.get('mask', np.ones(len(pattern), bool))
    rmask = np.repeat(smask, 2)          # pattern flattens [x0,y0,x1,y1,..]
    return target_flat, N, smask, rmask, info


def _branch_inits(N_est, phases, amps, rest):
    """Primary angle init + the branch flips for ambiguous prisms."""
    ax, ay, amb = angle_init(N_est, phases, amps, rest)
    yield np.concatenate([N_est, ax, ay]), ''
    for i in amb[:2]:
        ax2, ay2 = ax.copy(), ay.copy()
        ax2[i] = -ax2[i]
        ph = wrap(phases[i])
        ay2[i] = np.clip(wrap(ph - 180.0) if abs(ph) <= 90.0 else ph,
                         -18.0, 18.0)
        yield np.concatenate([N_est, ax2, ay2]), f'+flip{i}'


def solve9(pattern, fixed9):
    """9-D protocol. Returns (x9, mse, how)."""
    target_flat, N, smask, rmask, info = _prep(pattern)
    if N is None:
        return None, 1e30, 'no-speeds'

    def attempt(N_est, phases, amps, tag, nfev=2000):
        best = (None, np.inf, tag)
        for x0, sub in _branch_inits(N_est, phases, amps, fixed9):
            bx, bm = trf(x0, target_flat, rmask, _LO9, _HI9,
                         fixed9=fixed9, max_nfev=nfev if not sub else 800)
            if bm < best[1]:
                best = (bx, bm, tag + sub)
            if best[1] < MSE_OK:
                break
        return best

    bx, bm, how = attempt(N, info['phases'], info['amps'], 'primary')
    if bm < MSE_OK:
        return bx, bm, how
    bx2, bm2 = trf(np.concatenate([N, np.zeros(6)]), target_flat, rmask,
                   _LO9, _HI9, fixed9=fixed9)
    if bm2 < bm:
        bx, bm, how = bx2, bm2, 'zero'
    if bm < MSE_OK:
        return bx, bm, how
    w = int(np.argmin(np.abs(N)))
    N2 = N.copy(); N2[w] = -N2[w]
    ph2 = np.array(info['phases'], float).copy(); ph2[w] = -ph2[w]
    bx2, bm2, how2 = attempt(N2, ph2, info['amps'], 'flipweak')
    if bm2 < bm:
        bx, bm, how = bx2, bm2, how2
    if bm < MSE_OK:
        return bx, bm, how
    for q, alt in enumerate(info.get('alts', [])[:2]):
        if len(alt) != 3 or np.max(np.abs(alt)) > 3.6:
            continue
        bx2, bm2, how2 = attempt(np.asarray(alt, float), info['phases'],
                                 info['amps'], f'alt{q}')
        if bm2 < bm:
            bx, bm, how = bx2, bm2, how2
        if bm < MSE_OK:
            break
    return bx, bm, how


def _sign_sweep_candidates(ax_hat):
    """Pairs and the triple first (the cases the single-flip rungs never
    try), then singletons, all with glass/geometry/beam kept; then
    singletons with glass/geometry/beam reset to the nominal start.
    At most 10 TRFs."""
    order = [int(i) for i in np.argsort(np.abs(ax_hat))]   # weakest first
    multi = [S for k in (2, 3) for S in itertools.combinations(order, k)]
    single = [(i,) for i in order]
    return ([(S, 'keep') for S in multi] + [(S, 'keep') for S in single]
            + [(S, 'reset') for S in single])


def sign_sweep(bx, target_flat, rmask, rest_nominal):
    """Truth-blind sign-sweep rung. Returns (x, mse, tag) of the best
    candidate; acceptance is the usual MSE < MSE_OK certificate."""
    n_pts = len(rmask) // 2
    T = round(n_pts * DT, 6)
    nfev = _SS_NFEV.get(T, 400)
    best = (bx, np.inf, 'signsweep')
    for S, variant in _sign_sweep_candidates(bx[3:6]):
        x0 = bx.copy()
        for i in S:
            x0[3 + i] = -x0[3 + i]
        if variant == 'reset':
            x0[9:] = rest_nominal
        x, mse = trf(x0, target_flat, rmask, _LO, _HI, max_nfev=nfev)
        if mse < best[1]:
            best = (x, mse, f'signsweep:{variant}:{"".join(map(str, S))}')
        if mse < MSE_OK:
            break
    if best[1] < MSE_OK:
        # the capped-nfev TRF can stop just inside the acceptance threshold
        # with ~1e-3 parameter error (seen once in the local G1 run at
        # T = 80 s); one uncapped polish from the accepted point fixes that
        x, mse = trf(best[0], target_flat, rmask, _LO, _HI, max_nfev=2000)
        if mse < best[1]:
            best = (x, mse, best[2] + '+polish')
    return best


def solve18(pattern):
    """Full 18-D, nothing known. Returns (x18, mse, how, info)."""
    target_flat, N, smask, rmask, info = _prep(pattern)
    if N is None:
        return None, 1e30, 'no-speeds', info
    rest0 = nominal_rest(pattern)

    def attempt(N_est, phases, amps, rest, tag, nfev=4000):
        best = (None, np.inf, tag)
        for x0a, sub in _branch_inits(N_est, phases, amps, rest):
            x0 = np.concatenate([x0a, rest])
            bx, bm = trf(x0, target_flat, rmask, _LO, _HI,
                         max_nfev=nfev if not sub else 1200)
            if bm < best[1]:
                best = (bx, bm, tag + sub)
            if best[1] < MSE_OK:
                break
        return best

    bx, bm, how = attempt(N, info['phases'], info['amps'], rest0, 'primary')
    if bm < MSE_OK:
        return bx, bm, how, info
    # recalibration rung: the first fit's geometry improves the gains
    bx2, bm2, how2 = attempt(N, info['phases'], info['amps'],
                             bx[9:].copy(), 'recal', nfev=2000)
    if bm2 < bm:
        bx, bm, how = bx2, bm2, how2
    if bm < MSE_OK:
        return bx, bm, how, info
    x0 = np.concatenate([N, np.zeros(6), rest0])
    bx2, bm2 = trf(x0, target_flat, rmask, _LO, _HI, max_nfev=1500)
    if bm2 < bm:
        bx, bm, how = bx2, bm2, 'zero'
    if bm < MSE_OK:
        return bx, bm, how, info
    w = int(np.argmin(np.abs(N)))
    N2 = N.copy(); N2[w] = -N2[w]
    ph2 = np.array(info['phases'], float).copy(); ph2[w] = -ph2[w]
    bx2, bm2, how2 = attempt(N2, ph2, info['amps'], rest0, 'flipweak',
                             nfev=1500)
    if bm2 < bm:
        bx, bm, how = bx2, bm2, how2
    if bm < MSE_OK:
        return bx, bm, how, info
    for q, alt in enumerate(info.get('alts', [])[:2]):
        if len(alt) != 3 or np.max(np.abs(alt)) > 3.6:
            continue
        alt = np.asarray(alt, float)
        ph_a, am_a = proj_phases(pattern, alt)
        bx2, bm2, how2 = attempt(alt, ph_a, am_a, rest0, f'alt{q}', nfev=1200)
        if bm2 < bm:
            bx, bm, how = bx2, bm2, how2
        if bm < MSE_OK:
            break
    if bm >= MSE_OK and SIGN_SWEEP and bx is not None:
        bx2, bm2, how2 = sign_sweep(bx, target_flat, rmask, rest0)
        if bm2 < bm:
            bx, bm, how = bx2, bm2, how2
    return bx, bm, how, info
