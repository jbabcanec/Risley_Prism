#!/usr/bin/env python3
"""Unit-granular, IPC-free driver for the sign-sweep rung prototype.

Run 4 shards in parallel (no multiprocessing.Pool -- it deadlocked on
terminate in the sandbox):
    for i in 0 1 2 3; do SS_SHARD=$i SS_BUDGET=120 python3 sign_sweep_units.py & done; wait
Each shard walks its cases in order; per case it runs the candidate ladder
(sign_sweep_rung.candidates) one TRF per unit, stopping at the first unit with
MSE < 1e-12.  A unit is only started if it is expected to finish inside the
budget.  Units are appended to /tmp/ss_units_<shard>.jsonl; case-level
results are assembled afterwards by summarize().
"""
import os
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[k] = '1'
import sys, json, time, pickle, glob
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sign_sweep_rung as S
from risley_lattice import vec2pat, case_at, FS
from risley_lattice.solve import trf, _prep, _LO, _HI, MSE_OK
from risley_lattice.angles import nominal_rest

NCAND = 10
EST = {10.0: 8, 20.0: 14, 40.0: 25, 80.0: 50}
key = lambda j: (j['atlas'], j['case'], j['T'])


def load_units():
    units = defaultdict(list)
    for fp in glob.glob('/tmp/ss_units*.jsonl'):
        for line in open(fp):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            units[key(r)].append(r)
    return units


def case_done(us):
    return any(u['mse'] < MSE_OK for u in us) or len(us) >= NCAND


def run_unit(job, k, cache):
    t0 = time.time()
    if key(job) not in cache:
        tc = case_at(job['case'], job['seed0'])
        T = job['T']; n = int(round(T * FS))
        pat = vec2pat(tc, n, T)
        tf, N, sm, rm, info = _prep(pat)
        if rm is None:
            rm = np.ones(2 * n, bool); tf = pat.reshape(-1)
        cache.clear()
        cache[key(job)] = (tc, n, pat, tf, rm)
    tc, n, pat, tf, rm = cache[key(job)]
    T = job['T']
    x_hat = np.array(job['x18'])
    Sset, variant = S.candidates(x_hat[3:6])[k]
    x0 = x_hat.copy()
    for i in Sset:
        x0[3 + i] = -x0[3 + i]
    if variant == 'reset':
        x0[9:] = nominal_rest(pat)
    x, mse = trf(x0, tf, rm, _LO, _HI, max_nfev=S.NFEV.get(T, 400),
                 n_pts=n, time_limit=T)
    return dict(atlas=job['atlas'], case=job['case'], T=T, k=k,
                mse=float(mse), err=float(np.max(np.abs(x - tc))),
                tag=f'{variant}:{"".join(map(str, Sset))}',
                secs=round(time.time() - t0, 2))


def summarize():
    jobs = pickle.load(open('/tmp/ss_jobs.pkl', 'rb'))
    units = load_units()
    old = {key(r): r for r in map(json.loads, open('/tmp/sign_sweep.jsonl'))}
    rows = []
    for j in jobs:
        if key(j) in old:
            rows.append(old[key(j)]); continue
        us = units.get(key(j), [])
        if not us or not case_done(us):
            continue
        best = min(us, key=lambda u: u['mse'])
        tc = case_at(j['case'], j['seed0'])
        rows.append(dict(group=j['group'], atlas=j['atlas'], case=j['case'],
                         T=j['T'], rescued=bool(best['mse'] < MSE_OK),
                         mse=best['mse'], err=best['err'],
                         correct=bool(best['err'] < 1e-3), tag=best['tag'],
                         n_trf=len(us), secs=round(sum(u['secs'] for u in us), 2),
                         base_err=j['base_err'], axt=np.abs(tc[3:6]).tolist()))
    return rows, jobs


if __name__ == '__main__':
    shard = int(os.environ.get('SS_SHARD', 0))
    budget = float(os.environ.get('SS_BUDGET', 120))
    t_start = time.time()
    jobs = pickle.load(open('/tmp/ss_jobs.pkl', 'rb'))
    old = {key(r) for r in map(json.loads, open('/tmp/sign_sweep.jsonl'))}
    mine = [j for j in jobs if key(j) not in old and (shard < 0 or j['case'] % 4 == shard)]
    mine.sort(key=lambda j: (j['T'], j['group'], j['case']))
    units = load_units()
    cache = {}
    with open(f'/tmp/ss_units_{shard}.jsonl', 'a') as f:
        for j in mine:
            while not case_done(units[key(j)]):
                if time.time() - t_start + EST.get(j['T'], 50) > budget:
                    sys.exit(0)
                r = run_unit(j, len(units[key(j)]), cache)
                units[key(j)].append(r)
                f.write(json.dumps(r) + '\n'); f.flush()
