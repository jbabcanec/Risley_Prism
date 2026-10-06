# Adapted 2026-10-02 from risley_lattice/poc2.py (2026-09-29).
# Main changes: P=3/full18, shared two-gap propagation, all six orders,
# 14 nonlinear variables with four bounded affine geometry variables.
# This bounded numerical candidate procedure is NOT globally complete.
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
sys.path.insert(0,r'C:\Users\josep\Dropbox\Babcanec Works\Mathematics\Wedge')

"""Experimental blind three-prism recovery, not a global error certificate.

The exact Snell transfer is affine in (d, gap, px, py). Bounded variable
projection removes these four coordinates, and complex-step differentiation
supplies the reduced Jacobian. Spectral proposals and all six physical orders
are tried under a finite work budget. No truth parameter enters solve18.
"""
from dataclasses import dataclass
from itertools import permutations
import time

import numpy as np
from scipy.optimize import least_squares, lsq_linear, linprog

from risley_lattice.model import pat_P
from risley_lattice.spectral import extract_speeds
from risley_lattice.lattice import kset

from risley_lattice.model import NAMES, LO, HI
LINEAR = np.array([12,13,16,17])
NONLINEAR = np.array([i for i in range(18) if i not in LINEAR])
SPAN = HI-LO



def canonical(theta, count=200, dt=.05):
    """Independent canonical core evaluation, preserving physical order."""
    v = np.asarray(theta, float)
    return pat_P(v[:3], v[3:6], v[6:9], v[9:12], v[12:],
                 n_points=count, time_limit=count*dt)


def affine(theta, count=200, dt=.05, *, safe_trials=True):
    """Return base, design, guard values, optimization penalties.

    Supports batches for complex-step derivatives. Safe branches keep trial
    residuals finite; physical candidates must separately pass all guards.
    The four design columns are d, gap, px, py.
    """
    source = np.asarray(theta)
    single = source.ndim == 1
    v = np.atleast_2d(source)
    dtype = np.result_type(v.dtype, float)
    batch = len(v)
    times = np.arange(count)*dt
    gamma = 2*np.pi*v[:, :3, None]*times + np.pi/180*v[:, 6:9, None]
    wedge = np.sin(np.pi/180*v[:, 3:6, None])
    tilts = (wedge*np.cos(gamma), wedge*np.sin(gamma))
    base = np.empty((batch, count, 2), dtype=dtype)
    design = np.zeros((batch, count, 2, 4), dtype=dtype)
    guards, penalties = [], []

    def positive(x, floor=1e-8):
        return np.where(x.real > floor, x, floor) if safe_trials else x

    for axis in range(2):
        tangent = np.tan(np.pi/180*v[:, 14+axis, None])
        offset = np.broadcast_to(6*tangent, (batch, count)).copy()
        coeff = np.zeros((batch, count, 4), dtype=dtype)
        coeff[:, :, 2+axis] = 1
        for i in range(3):
            incoming = tangent/np.sqrt(1+tangent*tangent)
            n = v[:, 9+i, None]
            B = np.sqrt(n*n-incoming*incoming)
            s = tilts[axis][:, i]
            w = np.sqrt(1-s*s)
            q = incoming*w+B*s
            rad = 1-q*q
            R = np.sqrt(positive(rad))
            outgoing = q*w-R*s
            vertical = R*w+q*s
            face = B*w-incoming*s
            guards.extend((rad.real, vertical.real, face.real))
            for guard in (rad, vertical, face):
                penalties.append(1000*np.where(guard.real < 1e-6,
                                               guard-1e-6, 0))
            tangent = outgoing/positive(vertical)
            ell = B*R/(positive(vertical)*positive(face))
            offset = ell*(offset+3*incoming/B)
            coeff = ell[:, :, None]*coeff
            coeff[:, :, 1 if i < 2 else 0] += tangent
        base[:, :, axis] = offset
        design[:, :, axis] = coeff
    base = base.reshape(batch, -1)
    design = design.reshape(batch, -1, 4)
    guard_values = np.stack(guards, axis=1)
    penalty = np.concatenate(penalties, axis=1)
    if single:
        return base[0], design[0], guard_values[0], penalty[0]
    return base, design, guard_values, penalty


def forward(theta, count=200, dt=.05):
    """Smooth mathematical model; reject nonphysical supplied parameters."""
    v = np.asarray(theta, float)
    with np.errstate(invalid='ignore', divide='ignore'):
        base, design, guards, _ = affine(v, count, dt, safe_trials=False)
    if not np.isfinite(guards).all() or np.min(guards) <= 0:
        raise ValueError('nonphysical transmission or face intersection')
    return (base+design@v[LINEAR]).reshape(count, 2)


class WorkLimit(RuntimeError):
    pass


class Projected:
    def __init__(self, seed, pattern, dt, deadline=np.inf):
        self.seed = np.array(seed, float)
        self.target = np.asarray(pattern).ravel()
        self.count, self.dt = len(pattern), dt
        self.deadline = deadline
        self.cached = None

    def fun(self, q):
        if time.perf_counter() > self.deadline:
            raise WorkLimit('completion time budget exhausted')
        if self.cached is not None and np.array_equal(q, self.cached):
            return self.residual
        v = self.seed.copy()
        v[NONLINEAR] = q
        base, design, guards, penalty = affine(v, self.count, self.dt)
        scaled = design*SPAN[LINEAR]
        rhs = self.target-base-design@LO[LINEAR]
        linear = np.linalg.lstsq(scaled, rhs, rcond=None)[0]
        active = np.zeros(4, dtype=int)
        if np.any(linear < 0) or np.any(linear > 1):
            fit = lsq_linear(scaled, rhs, bounds=(0., 1.),
                             method='bvls', tol=1e-12)
            linear, active = fit.x, fit.active_mask
        v[LINEAR] = LO[LINEAR]+SPAN[LINEAR]*linear
        self.cached, self.v, self.design = q.copy(), v, scaled
        self.free = active == 0
        self.data_residual = scaled@linear-rhs
        self.residual = np.concatenate((self.data_residual, penalty))
        self.guards = guards
        return self.residual

    def jac(self, q):
        self.fun(q)
        step = 1e-25
        vv = np.broadcast_to(self.v, (len(NONLINEAR), 18)).astype(complex).copy()
        vv[np.arange(len(NONLINEAR)), NONLINEAR] += 1j*step
        base, design, _, penalty = affine(vv, self.count, self.dt)
        db, dA = base.imag/step, design.imag/step
        derivative = (db+dA@self.v[LINEAR]).T
        if self.free.any():
            af = self.design[:, self.free]
            # SVD pseudoinverses also handle an almost-flat prism without
            # an unstable inverse of a nearly singular triangular factor.
            inverse = np.linalg.pinv(af, rcond=1e-13)
            dAf = dA[:, :, self.free]*SPAN[LINEAR][self.free]
            correction = np.einsum('kmi,m->ik', dAf, self.data_residual)
            derivative = (derivative-af@(inverse@derivative)
                          -inverse.T@correction)
        return np.vstack((derivative, penalty.imag.T/step))


def spectral_state(pattern, speeds, dt):
    """Joint Fourier projection at proposed speeds; no truth information."""
    z = pattern[:, 0]+1j*pattern[:, 1]
    t = np.arange(len(pattern))*dt
    modes = kset(3, 3)
    E = np.exp(2j*np.pi*np.outer(t, modes@speeds))
    c = np.linalg.lstsq(E, z, rcond=1e-10)[0]
    c1 = np.array([c[np.flatnonzero(np.all(modes == unit, axis=1))[0]]
                   for unit in np.eye(3)])
    phase = np.degrees(np.angle(c1))
    negative = np.abs(phase) > 90
    phase = (phase-180*negative+180) % 360-180
    return np.clip(phase, -17.99, 17.99), np.abs(c1), np.where(negative, -1., 1.)


def initializations(pattern, speeds, dt):
    phases, amps, signs = spectral_state(pattern, speeds, dt)
    for order in permutations(range(3)):
        order = np.array(order)
        for distance, glass in ((125., 1.55), (80., 1.4), (180., 1.7)):
            v = (LO+HI)/2
            v[:3] = speeds[order]
            v[6:9] = phases[order]
            v[9:12] = glass
            v[12:14] = distance, 8.5
            lengths = np.array([distance+17+6/glass, distance+8.5+3/glass, distance])
            magnitudes = np.degrees(amps[order]/((glass-1)*lengths))
            v[3:6] = signs[order]*np.clip(magnitudes, 1e-5, 17.8)
            total = 6+distance+17+9/glass
            v[14:16] = np.clip(np.degrees(np.arctan(pattern.mean(0)/total)), -24, 24)
            v[16:18] = 0
            yield v, f'order={tuple(order)},d0={distance:g},n0={glass:g}'


@dataclass
class FitConfig:
    max_seconds: float = 45.
    screen_evaluations: int = 24
    polish_evaluations: int = 700
    finalists: int = 6
    alternate_bases: int = 1
    frontend: str = 'pencil'


def minimax_refine(theta, pattern, dt, eta, deadline=np.inf, iterations=10):
    """Sequential linear minimax fits, checked on the nonlinear model.

    This makes the candidate objective match the per-sample hard bands.
    It is a numerical candidate refinement, not an interval certificate.
    """
    best = np.asarray(theta, float).copy()
    target = pattern.ravel()
    count = len(pattern)
    error = float(np.max(abs(canonical(best, count, dt).ravel()-target)))
    history = [error]
    radius = np.full(18, .05)
    radius[:3] = .02/(count*dt*SPAN[:3])
    for _ in range(iterations):
        if time.perf_counter() > deadline or error <= eta*(1-1e-4):
            break
        base, design, _, _ = affine(best, count, dt)
        residual = base+design@best[LINEAR]-target
        vv = np.broadcast_to(best, (18, 18)).astype(complex).copy()
        vv[np.arange(18), np.arange(18)] += 1e-25j
        bb, aa, _, _ = affine(vv, count, dt)
        J = (bb+np.einsum('kmi,ki->km', aa, vv[:, LINEAR])).imag.T/1e-25
        matrix = J*SPAN/eta
        ones = np.ones((len(residual), 1))
        constraints = np.vstack((np.hstack((matrix, -ones)),
                                 np.hstack((-matrix, -ones))))
        rhs = np.concatenate((-residual/eta, residual/eta))
        lower = np.maximum((LO-best)/SPAN, -radius)
        upper = np.minimum((HI-best)/SPAN, radius)
        objective = np.zeros(19)
        objective[-1] = 1
        lp = linprog(objective, A_ub=constraints, b_ub=rhs,
                     bounds=list(zip(lower, upper))+[(0., None)], method='highs',
                     options={'primal_feasibility_tolerance': 1e-9,
                              'dual_feasibility_tolerance': 1e-9})
        if not lp.success:
            break
        step = SPAN*lp.x[:18]
        improved = False
        for fraction in (1., .5, .25, .125):
            trial = np.clip(best+fraction*step, LO, HI)
            _, _, guards, _ = affine(trial, count, dt)
            if np.min(guards) <= 0:
                continue
            trial_error = float(np.max(abs(canonical(trial, count, dt).ravel()-target)))
            if trial_error < error:
                best, error, improved = trial, trial_error, True
                history.append(error)
                break
        if not improved:
            radius *= .25
    return best, history


def solve18(pattern, dt=.05, eta=0., config=None):
    """Blind candidate fit from an ordered uniform (K,2) scan.

    No success/certification label is inferred from low scan error. A returned
    compatible candidate can still have inaccurate or ambiguous parameters.
    max_seconds bounds completion after the spectral front end returns.
    """
    config = FitConfig() if config is None else config
    y = np.asarray(pattern, float)
    if y.ndim != 2 or y.shape[1] != 2 or len(y) < 100 or not np.isfinite(y).all():
        raise ValueError('provide at least 100 finite ordered (x,y) samples')
    if not np.isfinite(dt) or dt <= 0 or not np.isfinite(eta) or eta < 0:
        raise ValueError('dt must be positive and eta nonnegative')
    if abs(dt-.05) > 1e-12:
        raise ValueError('this POC uses the existing 20 Hz spectral prior; dt must be .05')
    if not np.isfinite(config.max_seconds) or config.max_seconds <= 0 or config.finalists < 1:
        raise ValueError('positive completion budget and at least one finalist required')
    if min(config.screen_evaluations, config.polish_evaluations) < 1 or config.alternate_bases < 0:
        raise ValueError('positive evaluation limits and nonnegative alternate_bases required')
    start = time.perf_counter()
    N, info = extract_speeds(y, dt, n_gen=3, frontend=config.frontend)
    spectral_seconds = time.perf_counter()-start
    result = dict(status='unresolved', certified=False, theta=None,
                  spectral_seconds=spectral_seconds, attempts=[],
                  spectral_speeds=None if N is None else N.tolist(),
                  sampling_dt=dt, sample_count=len(y), eta=eta)
    if N is None:
        result.update(reason='spectral front end did not find three rotors',
                      seconds=time.perf_counter()-start)
        return result
    deadline = time.perf_counter()+config.max_seconds
    bases = [np.asarray(N)]
    for candidate in info.get('alts', []):
        if len(bases) >= 1+config.alternate_bases:
            break
        candidate = np.asarray(candidate)
        if (candidate.shape == (3,) and np.max(abs(candidate)) < 3.5 and
                all(np.max(abs(candidate-b)) > .002 for b in bases)):
            bases.append(candidate)
        if len(bases) >= 1+config.alternate_bases:
            break
    candidates = []
    stop = False

    def attempt(seed, tag, evaluations):
        projected = Projected(seed, y, dt, deadline)
        try:
            fit = least_squares(projected.fun, seed[NONLINEAR], jac=projected.jac,
                                bounds=(LO[NONLINEAR], HI[NONLINEAR]),
                                x_scale=SPAN[NONLINEAR], ftol=1e-14,
                                xtol=1e-14, gtol=None, max_nfev=evaluations)
            projected.fun(fit.x)
        except WorkLimit:
            if projected.cached is None:
                raise
        v = projected.v.copy()
        mse = float(np.mean(projected.data_residual**2))
        with np.errstate(invalid='ignore', divide='ignore'):
            _, _, real_guards, _ = affine(v, len(y), dt, safe_trials=False)
        valid = bool(np.isfinite(real_guards).all() and np.min(real_guards) > 0)
        result['attempts'].append(dict(tag=tag, mse=mse, physical=valid))
        if valid:
            candidates.append((mse, v, tag))
        return mse, v, tag

    try:
        for base_index, speeds in enumerate(bases):
            screened = []
            for seed, tag in initializations(y, speeds, dt):
                seed = np.clip(seed, LO+1e-10, HI-1e-10)
                screened.append(attempt(seed, f'basis={base_index}/{tag}',
                                         config.screen_evaluations))
            screened.sort(key=lambda x: x[0])
            for _, seed, tag in screened[:config.finalists]:
                mse, _, _ = attempt(seed, 'polish/'+tag, config.polish_evaluations)
                if mse < max(1e-23, eta*eta/4):
                    stop = True
                    break
            if stop:
                break
    except (WorkLimit, np.linalg.LinAlgError, ValueError) as exc:
        result['completion_note'] = str(exc)
    if not candidates:
        result.update(reason='no physical candidate within work budget',
                      seconds=time.perf_counter()-start)
        return result
    candidates.sort(key=lambda x: x[0])
    _, best, tag = candidates[0]
    if eta > 0:
        best, history = minimax_refine(best, y, dt, eta, deadline)
        result['minimax_max_residuals'] = history
        if len(history) > 1:
            tag = 'minimax/'+tag
    prediction = canonical(best, len(y), dt)
    smooth = forward(best, len(y), dt)
    residual = prediction-y
    max_residual = float(np.max(abs(residual)))
    # This is a stated numerical fit tolerance, not an added hard-noise
    # certificate. Report exact eta compatibility separately.
    numerical_tolerance = 1e-8
    result.update(theta=best.tolist(), method=tag, seconds=time.perf_counter()-start,
                  rms=float(np.sqrt(np.mean(residual**2))), max_residual=max_residual,
                  hard_band_compatible=max_residual <= eta,
                  numerical_fit_tolerance=numerical_tolerance,
                  model_parity=float(np.max(abs(smooth-prediction))),
                  status='candidate_fit' if max_residual <= max(eta, numerical_tolerance)
                  else 'unresolved')
    result['compatibility_verification'] = {'verified': False, 'reason': 'candidate solver only; run full18 interval verification separately'}
    different = [(m, v, label) for m, v, label in candidates
                 if np.max(abs(v-best)) > 1e-3]
    result['alternative'] = None
    if different:
        mse, alt, label = min(different, key=lambda x: x[0])
        alt_residual = float(np.max(abs(canonical(alt, len(y), dt)-y)))
        result['alternative'] = dict(theta=alt.tolist(), smooth_rms=float(np.sqrt(mse)),
                                     max_residual=alt_residual,
                                     hard_band_compatible=alt_residual <= eta,
                                     numerical_fit=alt_residual <= max(eta, numerical_tolerance),
                                     method=label)
    result['seconds'] = time.perf_counter()-start
    return result
