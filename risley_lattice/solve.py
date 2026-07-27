"""Completion stage: trust-region solve from the spectral init, with the
verified candidate ladder. Every rung is deterministic and accepted only by
the pattern-MSE certificate -- no search.

solve9(pattern, fixed9): 9-D protocol (glass/geometry/beam known).
solve18(pattern):        full 18-D, nothing assumed known.
"""
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
    return bx, bm, how, info
