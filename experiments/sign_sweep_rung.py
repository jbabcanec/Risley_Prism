#!/usr/bin/env python3
"""Prototype: a truth-blind 'sign-sweep' rung appended to the solve18 ladder.

When the ladder ends without MSE < 1e-12, take the solver's OWN best answer
x_hat and, for every non-empty subset S of prisms (singletons first, weakest
|ax_hat| first), flip ax_i -> -ax_i for i in S and re-run the masked TRF.
Two variants per subset: (a) everything else kept, (b) glass/geometry/beam
reset to the nominal spectral init.  Accept the first candidate whose pattern
MSE < 1e-12 (same certificate as every other rung).  Ground truth is used
ONLY afterwards, to score the outcome.

Target groups come from angle_ambiguity.py (/tmp/angle_amb.pkl):
  G1 unsolved at T=80, speeds right, only ax signs wrong      (all)
  G2 early failed trial of a later-rescued case, same defect  (all)
  G3 unsolved at T=80, speeds right, other angle error        (sample 300)
"""
import os
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[k] = '1'
import sys, json, time, glob, pickle, itertools, random
from multiprocessing import Pool
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from risley_lattice import vec2pat, case_at, FS
from risley_lattice.solve import trf, _prep, _LO, _HI, MSE_OK
from risley_lattice.angles import nominal_rest

RES = os.path.join(ROOT, 'results')
OUT = sys.argv[1] if len(sys.argv) > 1 else '/tmp/sign_sweep.jsonl'


NFEV = {10.0: 1200, 20.0: 1000, 40.0: 600, 80.0: 400}


def candidates(ax_hat):
    """(subset, variant) ladder: singletons weakest-first, pairs, triple with
    other params kept; then singletons again with glass/geo/beam reset.
    10 TRFs worst case."""
    order = [int(i) for i in np.argsort(np.abs(ax_hat))]   # weakest first
    keep = [S for k in (1, 2, 3) for S in itertools.combinations(order, k)]
    return [(S, 'keep') for S in keep] + [((i,), 'reset') for i in order]


def run_one(job):
    t0 = time.time()
    tc = case_at(job['case'], job['seed0'])
    T = job['T']
    n = int(round(T * FS))
    pat = vec2pat(tc, n, T)
    target_flat, N, smask, rmask, info = _prep(pat)
    if rmask is None:
        rmask = np.ones(2 * n, bool)
    x_hat = np.array(job['x18'])
    rest_nom = nominal_rest(pat)
    n_trf = 0
    best = (np.inf, None, None)
    for S, variant in candidates(x_hat[3:6]):
        x0 = x_hat.copy()
        for i in S:
            x0[3 + i] = -x0[3 + i]
        if variant == 'reset':
            x0[9:] = rest_nom
        x, mse = trf(x0, target_flat, rmask, _LO, _HI,
                     max_nfev=NFEV.get(T, 400), n_pts=n, time_limit=T)
        n_trf += 1
        if mse < best[0]:
            best = (mse, x, f'{variant}:{"".join(map(str, S))}')
        if mse < MSE_OK:
            break
    mse, x, tag = best
    err = float(np.max(np.abs(x - tc)))
    return dict(group=job['group'], atlas=job['atlas'], case=job['case'],
                T=T, rescued=bool(mse < MSE_OK), mse=float(mse), err=err,
                correct=bool(err < 1e-3), tag=tag, n_trf=n_trf,
                secs=round(time.time() - t0, 2),
                base_err=job['base_err'], axt=np.abs(tc[3:6]).tolist())


def build_jobs():
    d = pickle.load(open('/tmp/angle_amb.pkl', 'rb'))
    pure = lambda r: (r['spd_ok'] and 'sign-flip' in r['cls']
                      and 'angle-off' not in r['cls'])
    other = lambda r: r['spd_ok'] and 'angle-off' in r['cls']
    want = {}
    for r in d['last']:
        if pure(r):
            want[(r['atlas'], r['case'], r['T'])] = 'G1'
    for r in d['rescued']:
        if pure(r):
            want[(r['atlas'], r['case'], r['T'])] = 'G2'
    g3 = [r for r in d['last'] if other(r)]
    random.Random(0).shuffle(g3)
    for r in g3[:300]:
        want[(r['atlas'], r['case'], r['T'])] = 'G3'
    jobs = []
    for atlas in sorted({k[0] for k in want}):
        for fp in sorted(glob.glob(f'{RES}/{atlas}/*.jsonl')):
            for line in open(fp):
                rec = json.loads(line)
                for t in rec['trail']:
                    key = (atlas, rec['case'], t['T'])
                    if key in want and t['x18'] is not None:
                        jobs.append(dict(group=want.pop(key), atlas=atlas,
                                         case=rec['case'], seed0=rec['seed0'],
                                         T=t['T'], x18=t['x18'],
                                         base_err=t['err']))
    return jobs


if __name__ == '__main__':
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            r = json.loads(line)
            done.add((r['atlas'], r['case'], r['T']))
    # Chunked driver (the sandbox caps each command at ~3 min): jobs are cached
    # once, and a chunk only submits a job if it can finish before BUDGET s.
    JOBS = '/tmp/ss_jobs.pkl'
    if os.path.exists(JOBS):
        alljobs = pickle.load(open(JOBS, 'rb'))
    else:
        alljobs = build_jobs(); pickle.dump(alljobs, open(JOBS, 'wb'))
    jobs = [j for j in alljobs if (j['atlas'], j['case'], j['T']) not in done]
    jobs.sort(key=lambda j: (j['T'], j['group'], j['case']))
    print(f'{len(jobs)} jobs remaining', flush=True)
    BUDGET = float(os.environ.get('SS_BUDGET', 1e9))
    EST = {10.0: 45, 20.0: 70, 40.0: 110, 80.0: 120}
    t_start = time.time()
    with Pool(4) as pool, open(OUT, 'a') as f:
        pending = []
        q = list(jobs)
        while q or pending:
            while q and len(pending) < 4 and \
                    time.time() - t_start + EST.get(q[0]['T'], 135) < BUDGET:
                pending.append(pool.apply_async(run_one, (q.pop(0),)))
            if not pending:
                break
            time.sleep(0.5)
            for p in [p for p in pending if p.ready()]:
                pending.remove(p)
                f.write(json.dumps(p.get()) + '\n'); f.flush()
    print('chunk done', flush=True)
