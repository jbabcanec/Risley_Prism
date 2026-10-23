#!/usr/bin/env python3
"""Sign-sweep rung, larger local run (no Modal). Same groups and candidate
ladder as experiments/sign_sweep_rung.py, rebuilt from the deduplicated
ready dataset, executed one TRF ('unit') at a time so it survives the
~3-minute command limit.

  python3 sign_sweep_local.py build  --work DIR [--g3 300]
  python3 sign_sweep_local.py run    --work DIR --shard k --budget 160
  python3 sign_sweep_local.py report --work DIR
"""
import os
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[k] = '1'
import sys, json, time, glob, random, argparse
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(EXP)); sys.path.insert(0, EXP)
import sign_sweep_rung as S
from risley_lattice import vec2pat, case_at, FS
from risley_lattice.solve import trf, _prep, _LO, _HI, MSE_OK
from risley_lattice.angles import nominal_rest

READY = os.path.join(EXP, 'results', 'sweep_adaptive_ready_0000000.jsonl')
NCAND = 10
key = lambda j: (j['case'], j['T'])


def wrap(d):
    return (d + 180.0) % 360.0 - 180.0


def classify(x, tc):
    cls = []
    for i in range(3):
        at, ah = tc[3 + i], x[3 + i]
        if np.sign(ah) != np.sign(at) and abs(at) > 0.05:
            cls.append('sign-flip')
        elif abs(wrap(x[6 + i] - tc[6 + i])) > 5 or abs(ah - at) > 1:
            cls.append('angle-off')
        else:
            cls.append('ok')
    return cls, bool(np.max(np.abs(x[:3] - tc[:3])) < 0.02)


def build(work, n_g3, n_g2):
    g1, g2, g3 = [], [], []
    for line in open(READY):
        r = json.loads(line)
        solved = r['T_solved'] is not None
        tr = r['trail']
        cand = tr if solved else tr  # every failed trial
        for t in cand:
            if t['err'] < 1e-3 or t['x18'] is None:
                continue
            last = (t is tr[-1]) and not solved
            if not last and not solved:
                continue
            tc = case_at(r['case'], r['seed0'])
            x = np.array(t['x18'])
            cls, ok = classify(x, tc)
            pure = ok and 'sign-flip' in cls and 'angle-off' not in cls
            other = ok and 'angle-off' in cls
            job = dict(case=r['case'], seed0=r['seed0'], T=t['T'],
                       x18=t['x18'], base_err=t['err'])
            if last and pure:
                g1.append(dict(group='G1', **job))
            elif last and other:
                g3.append(dict(group='G3', **job))
            elif solved and pure:
                g2.append(dict(group='G2', **job))
    pop = dict(G1=len(g1), G2=len(g2), G3=len(g3))
    random.Random(0).shuffle(g3); random.Random(1).shuffle(g2)
    jobs = g1 + g3[:n_g3] + g2[:n_g2]
    json.dump(dict(population=pop, jobs=jobs),
              open(os.path.join(work, 'ss_jobs.json'), 'w'))
    print('population', pop, 'jobs', len(jobs))


def load_units(work):
    units = defaultdict(list)
    for fp in glob.glob(os.path.join(work, 'ss_units_*.jsonl')):
        for line in open(fp):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            units[(r['case'], r['T'])].append(r)
    return units


def done(us):
    return any(u['mse'] < MSE_OK for u in us) or len(us) >= NCAND


def run(work, shard, nshard, budget, sub=0, nsub=1):
    t0 = time.time()
    jobs = json.load(open(os.path.join(work, 'ss_jobs.json')))['jobs']
    mine = [j for j in jobs if j['case'] % nshard == shard
            and (j['case'] // nshard) % nsub == sub]
    mine.sort(key=lambda j: ({'G1': 0, 'G3': 1, 'G2': 2}[j['group']], j['T'],
                             j['case']))
    units = load_units(work)
    cache = {}
    est = defaultdict(lambda: 30.0)
    for u in [u for us in units.values() for u in us]:
        est[u['T']] = max(est[u['T']] * 0.8, u['secs'] * 1.3)
    with open(os.path.join(work, f'ss_units_{shard}_{sub}.jsonl'), 'a') as f:
        for j in mine:
            k = key(j)
            while not done(units[k]):
                if time.time() - t0 + est[j['T']] > budget:
                    return
                kk = len(units[k])
                ts = time.time()
                if k not in cache:
                    tc = case_at(j['case'], j['seed0'])
                    T = j['T']; n = int(round(T * FS))
                    pat = vec2pat(tc, n, T)
                    tf, N, sm, rm, info = _prep(pat)
                    if rm is None:
                        rm = np.ones(2 * n, bool); tf = pat.reshape(-1)
                    cache.clear(); cache[k] = (tc, n, pat, tf, rm)
                tc, n, pat, tf, rm = cache[k]
                x_hat = np.array(j['x18'])
                Sset, variant = S.candidates(x_hat[3:6])[kk]
                x0 = x_hat.copy()
                for i in Sset:
                    x0[3 + i] = -x0[3 + i]
                if variant == 'reset':
                    x0[9:] = nominal_rest(pat)
                x, mse = trf(x0, tf, rm, _LO, _HI,
                             max_nfev=S.NFEV.get(j['T'], 400),
                             n_pts=n, time_limit=j['T'])
                r = dict(group=j['group'], case=j['case'], T=j['T'], k=kk,
                         mse=float(mse), err=float(np.max(np.abs(x - tc))),
                         tag=f'{variant}:{"".join(map(str, Sset))}',
                         secs=round(time.time() - ts, 2),
                         axt=np.abs(tc[3:6]).tolist())
                units[k].append(r)
                est[j['T']] = max(est[j['T']] * 0.8, r['secs'] * 1.3)
                f.write(json.dumps(r) + '\n'); f.flush()


def report(work):
    meta = json.load(open(os.path.join(work, 'ss_jobs.json')))
    units = load_units(work)
    rows = []
    for j in meta['jobs']:
        us = units.get(key(j), [])
        if not us or not done(us):
            continue
        best = min(us, key=lambda u: u['mse'])
        rows.append(dict(group=j['group'], case=j['case'], T=j['T'],
                         rescued=bool(best['mse'] < MSE_OK),
                         correct=bool(best['err'] < 1e-3), tag=best['tag'],
                         n_trf=len(us), secs=sum(u['secs'] for u in us),
                         axmin=min(us[0]['axt'])))
    out = dict(population=meta['population'], groups={})
    for g in ('G1', 'G2', 'G3'):
        rr = [r for r in rows if r['group'] == g]
        nj = sum(1 for j in meta['jobs'] if j['group'] == g)
        k = sum(r['rescued'] for r in rr)
        wrong = sum(r['rescued'] and not r['correct'] for r in rr)
        pair = sum(r['rescued'] and len(r['tag'].split(':')[1]) >= 2 for r in rr)
        out['groups'][g] = dict(jobs=nj, tested=len(rr), rescued=k,
                                wrong_accepted=wrong, multi_flip_wins=pair,
                                median_secs=float(np.median([r['secs'] for r in rr])) if rr else None)
    json.dump(dict(out, rows=rows), open(os.path.join(HERE, 'out', 'sign_sweep_local.json'), 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd'); ap.add_argument('--work', required=True)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshard', type=int, default=4)
    ap.add_argument('--budget', type=float, default=160)
    ap.add_argument('--sub', type=int, default=0)
    ap.add_argument('--nsub', type=int, default=1)
    ap.add_argument('--g3', type=int, default=300)
    ap.add_argument('--g2', type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    {'build': lambda: build(a.work, a.g3, a.g2),
     'run': lambda: run(a.work, a.shard, a.nshard, a.budget, a.sub, a.nsub),
     'report': lambda: report(a.work)}[a.cmd]()
