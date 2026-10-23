#!/usr/bin/env python3
"""How non-unique is the inverse? Four measurements, all truth-scored.

eps    Local epsilon-equivalence. For a pattern tolerance eps (RMS per
       coordinate), the set of parameter vectors whose patterns stay within
       eps of the truth is, to first order, the ellipsoid
           { d : |J d|^2 / m <= eps^2 },
       with per-parameter half-width  eps*sqrt(m)*sqrt[(J^T J)^-1]_jj  and
       axes along the right singular vectors of J. Recorded at truth.
twin   Discrete near-twins. Flip the sign of one wedge (ax_i -> -ax_i, the
       half-turn image with ay kept), refit all 18 parameters by TRF from
       there, and record how close the best twin pattern gets (RMS) and how
       far it sits in parameter space.
ms     Global multistart. 24 uniform random starts per case, TRF each;
       collect every exact minimum (MSE < 1e-12) and count distinct ones.
degen  The continuous-family mechanism: the same case with |N2| forced
       equal to |N1|; the Jacobian loses rank (sigma_min -> 0).

  python3 nonunique.py run --shard k --budget 150
  python3 nonunique.py report
"""
import os
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[k] = '1'
import sys, json, time, glob, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from risley_lattice.model import (case_at, vec2pat, LO, HI, RG, NAMES, canon)
from risley_lattice.solve import trf, _LO, _HI

WORK = os.environ.get('NU_WORK', os.path.expanduser('~/work/nu'))
SEED = 20261010
N_EPS, N_TWIN, N_MS, N_START, N_DEG = 150, 60, 12, 24, 20


def jac(v):
    m = 400
    J = np.empty((m, 18))
    for i in range(18):
        h = 1e-6 * max(1.0, abs(v[i]))
        a, b = v.copy(), v.copy(); a[i] += h; b[i] -= h
        J[:, i] = (vec2pat(a).reshape(-1) - vec2pat(b).reshape(-1)) / (2 * h)
    return J


def u_eps(i):
    v = case_at(i, SEED)
    J = jac(v)
    U, s, Vt = np.linalg.svd(J, full_matrices=False)
    C = np.linalg.pinv(J.T @ J, rcond=1e-20)
    hw = np.sqrt(len(J)) * np.sqrt(np.maximum(np.diag(C), 0))   # per eps=1
    # relative half-width: as a fraction of each parameter's box range
    return dict(sv=s.tolist(), hw1=hw.tolist(), hw_rel=(hw / RG).tolist(),
                weak_dir=Vt[-1].tolist(), axmin=float(np.abs(v[3:6]).min()))


def u_twin(i, p):
    v = case_at(i, SEED)
    tgt = vec2pat(v).reshape(-1)
    x0 = v.copy(); x0[3 + p] = -x0[3 + p]
    rm = np.ones(400, bool)
    x, mse = trf(x0, tgt, rm, _LO, _HI, max_nfev=1500)
    d = (x - v) / RG
    return dict(p=p, ax=float(v[3 + p]), rms=float(np.sqrt(mse)),
                rms_start=float(np.sqrt(np.mean((vec2pat(x0).reshape(-1) - tgt) ** 2))),
                dist=float(np.max(np.abs(d))), back_to_truth=bool(np.max(np.abs(x - v)) < 1e-6))


def u_ms(i, s):
    v = case_at(i, SEED)
    tgt = vec2pat(v).reshape(-1)
    rng = np.random.default_rng([SEED, i, s])
    x0 = LO + RG * rng.random(18)
    x0[:3] = v[:3] + rng.normal(0, 0.002, 3)   # speeds: from spectrum (exact
    #   to ~1e-4 Hz in the pipeline); the search is over everything else
    x, mse = trf(x0, tgt, np.ones(400, bool), _LO, _HI, max_nfev=2000)
    return dict(s=s, mse=float(mse), x=x.tolist(),
                err=float(np.max(np.abs(canon(x) - v))))


def u_degen(i):
    v = case_at(i, SEED)
    w = v.copy(); w[1] = np.copysign(abs(w[0]), w[0])
    out = {}
    for tag, q in (('generic', v), ('equal_speed', w)):
        s = np.linalg.svd(jac(q), compute_uv=False)
        out[tag] = s.tolist()
    return out


def units():
    for i in range(N_EPS):
        yield ('eps', i, 0)
    for i in range(N_TWIN):
        for p in range(3):
            yield ('twin', i, p)
    for i in range(N_DEG):
        yield ('degen', i, 0)
    for i in range(N_MS):
        for s in range(N_START):
            yield ('ms', i, s)


def load():
    d = {}
    for fp in glob.glob(os.path.join(WORK, 'nu_*.jsonl')):
        for line in open(fp):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            d[(r['u'], r['i'], r['j'])] = r
    return d


def run(shard, budget, nshard=4):
    t0 = time.time()
    os.makedirs(WORK, exist_ok=True)
    have = load()
    with open(os.path.join(WORK, f'nu_{shard}.jsonl'), 'a') as f:
        for n, (u, i, j) in enumerate(units()):
            if n % nshard != shard or (u, i, j) in have:
                continue
            if time.time() - t0 > budget - (25 if u in ('twin', 'ms') else 6):
                return
            ts = time.time()
            r = {'eps': lambda: u_eps(i), 'twin': lambda: u_twin(i, j),
                 'ms': lambda: u_ms(i, j), 'degen': lambda: u_degen(i)}[u]()
            r.update(u=u, i=i, j=j, secs=round(time.time() - ts, 2))
            f.write(json.dumps(r) + '\n'); f.flush()


def progress():
    have = load()
    tot = list(units())
    print('nonunique units', sum(k in have for k in tot), '/', len(tot))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd'); ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--budget', type=float, default=150)
    ap.add_argument('--nshard', type=int, default=4)
    a = ap.parse_args()
    run(a.shard, a.budget, a.nshard) if a.cmd == 'run' else progress()
