#!/usr/bin/env python3
"""Does telling the solver the glass type remove the lookalike answers?

The full blind solver (solve18) is run from scratch twice on the same
pattern: once as usual (glass unknown) and once with the three glass indices
fixed at their true values (as they would be for real hardware whose glass
is known). Nothing in risley_lattice is changed: the glass bounds and the
nominal starting glass are set for the duration of one solve only.

Groups (from the September campaign):
  look   machines that produced a lookalike answer (MSE < 1e-12 but a
         parameter off by >= 1e-3), at the viewing time where it first
         happened; sample of 120 of the 403.
  flat   control: machines whose flattest wedge is under 0.5 deg, solved
         at T = 10 s; sample of 100 (checks that fixing glass does not
         break anything and measures the rate of lookalikes there).

  python3 glassknown.py build
  python3 glassknown.py run --shard k --nshard 4 --budget 160
  python3 glassknown.py report
"""
import os
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[k] = '1'
import sys, json, time, glob, random, argparse
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, EXP); sys.path.insert(0, os.path.dirname(EXP))
from risley_lattice import case_at
import risley_lattice.solve as S
from risley_lattice.model import NAMES
from sweep import make_pattern

WORK = os.path.expanduser('~/work/glass')
OUT = os.path.join(HERE, 'out')
_LO0, _HI0 = S._LO.copy(), S._HI.copy()
_NOM0 = S.nominal_rest


def solve(tc, T, glass_known):
    S._LO[:] = _LO0; S._HI[:] = _HI0; S.nominal_rest = _NOM0
    if glass_known:
        g = tc[9:12].copy()
        S._LO[9:12] = g - 1e-12; S._HI[9:12] = g + 1e-12

        def nom(pattern, _g=g):
            r = _NOM0(pattern).copy(); r[0:3] = _g
            return r
        S.nominal_rest = nom
    try:
        x, mse, how, info = S.solve18(make_pattern(tc, T))
    finally:
        S._LO[:] = _LO0; S._HI[:] = _HI0; S.nominal_rest = _NOM0
    if x is None:
        return dict(mse=float(mse), err=None, worst=None, how=how)
    e = np.abs(x - tc)
    return dict(mse=float(mse), err=float(e.max()), worst=NAMES[int(e.argmax())], how=how)


def build(n_look=120, n_flat=100):
    aw = json.load(open(os.path.join(OUT, 'mse_accepted_wrong.json')))
    first = {}
    for case, T, *_ in sorted(aw, key=lambda a: (a[0], a[1])):
        first.setdefault(case, T)
    look = [dict(group='look', case=c, T=T) for c, T in first.items()]
    camp = np.load(os.path.expanduser('~/work/camp/campaign.npz'))
    flat_ids = [int(c) for c in camp['case'][camp['ax_min'] < 0.5] if int(c) not in first]
    random.Random(11).shuffle(look); random.Random(12).shuffle(flat_ids)
    jobs = look[:n_look] + [dict(group='flat', case=c, T=10.0) for c in flat_ids[:n_flat]]
    os.makedirs(WORK, exist_ok=True)
    json.dump(dict(population=dict(look=len(look), flat=int((camp['ax_min'] < 0.5).sum())),
                   jobs=jobs), open(os.path.join(WORK, 'jobs.json'), 'w'))
    print('jobs', len(jobs))


def load():
    d = {}
    for fp in glob.glob(os.path.join(WORK, 'u_*.jsonl')):
        for line in open(fp):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            d[(r['case'], r['glass_known'])] = r
    return d


def run(shard, nshard, budget):
    t0 = time.time()
    jobs = json.load(open(os.path.join(WORK, 'jobs.json')))['jobs']
    done = load()
    lp = os.path.join(WORK, 'long.json')
    skip = set(json.load(open(lp))) if os.path.exists(lp) and not os.environ.get('GK_ALL') else set()
    with open(os.path.join(WORK, f'u_{shard}.jsonl'), 'a') as f:
        for n, j in enumerate(jobs):
            if n % nshard != shard or j['case'] in skip:
                continue
            for gk in (True, False):
                if (j['case'], gk) in done:
                    continue
                if time.time() - t0 + 25 > budget:
                    return
                sp = os.path.join(WORK, f'start_{j["case"]}_{int(gk)}')
                starts = int(open(sp).read()) if os.path.exists(sp) else 0
                if starts >= 2 and not os.environ.get('GK_ALL'):
                    continue          # too slow for a 3-minute chunk; run in Terminal
                open(sp, 'w').write(str(starts + 1))
                ts = time.time()
                tc = case_at(j['case'], 424242)
                r = solve(tc, j['T'], gk)
                r.update(group=j['group'], case=j['case'], T=j['T'], glass_known=gk,
                         axmin=float(np.min(np.abs(tc[3:6]))), secs=round(time.time() - ts, 1))
                done[(j['case'], gk)] = r
                f.write(json.dumps(r) + '\n'); f.flush()


def classify(r):
    if r['err'] is None or r['mse'] >= S.MSE_OK:
        return 'unsolved'
    return 'correct' if r['err'] < 1e-3 else 'lookalike'


def report():
    meta = json.load(open(os.path.join(WORK, 'jobs.json')))
    done = load()
    out = dict(population=meta['population'], groups={})
    for g in ('look', 'flat'):
        jobs = [j for j in meta['jobs'] if j['group'] == g]
        both = [j for j in jobs if (j['case'], True) in done and (j['case'], False) in done]
        tab = {}
        for gk in (False, True):
            cnt = defaultdict(int)
            for j in both:
                cnt[classify(done[(j['case'], gk)])] += 1
            tab['glass_known' if gk else 'glass_unknown'] = dict(cnt)
        trans = defaultdict(int)
        for j in both:
            trans[f"{classify(done[(j['case'], False)])} -> {classify(done[(j['case'], True)])}"] += 1
        worst = defaultdict(int)
        for j in both:
            r = done[(j['case'], True)]
            if classify(r) == 'lookalike':
                worst[r['worst']] += 1
        out['groups'][g] = dict(jobs=len(jobs), finished=len(both), **tab,
                                transitions=dict(trans), remaining_lookalike_worst=dict(worst))
    json.dump(out, open(os.path.join(OUT, 'glassknown.json'), 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd'); ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshard', type=int, default=4); ap.add_argument('--budget', type=float, default=160)
    a = ap.parse_args()
    {'build': build, 'run': lambda: run(a.shard, a.nshard, a.budget), 'report': report}[a.cmd]()
