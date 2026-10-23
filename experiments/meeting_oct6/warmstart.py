#!/usr/bin/env python3
"""Warm start: reuse the answer from a shorter recording as the starting
guess for the next, longer one (instead of starting over from scratch).

Two tests, both local Python:

ben87   Ben-regime machine 87: solve18 at T = 80 s, then a TRF on the 160 s
        pattern started from that answer, then the same 160 -> 320 s.
        python3 warmstart.py ben87 --step 80|160|320

camp    Unsolved configurations from the September campaign. For each failed
        trial at T_k, start a TRF on the T_(k+1) pattern from the stored
        answer x18(T_k). Compare with what the campaign got at T_(k+1)
        starting from scratch. Sample: 150 'basin' failures + 100 other.
        python3 warmstart.py build
        python3 warmstart.py run --shard k --nshard 4 --budget 160
        python3 warmstart.py report

Acceptance is the solver's own test (pattern MSE < 1e-12); ground truth is
used only afterwards, to score.
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
from risley_lattice.solve import trf, _prep, _LO, _HI, MSE_OK
from risley_lattice.model import NAMES
from sweep import make_pattern

WORK = os.path.expanduser('~/work/warm')
READY = os.path.join(EXP, 'results', 'sweep_adaptive_ready_0000000.jsonl')
LADDER = [10., 20., 40., 80.]
NFEV = 2000


def warm_trf(x0, tc, T):
    pat = make_pattern(tc, T)
    tf, N, sm, rm, info = _prep(pat)
    if rm is None:
        rm = np.ones(pat.size, bool); tf = pat.reshape(-1)
    x, mse = trf(np.asarray(x0, float), tf, rm, _LO, _HI, max_nfev=NFEV)
    err = np.abs(x - tc)
    return x, float(mse), float(err.max()), NAMES[int(err.argmax())]


# ------------------------------------------------------------------ ben87
def ben87(step):
    sys.argv = sys.argv[:1]
    import local_campaign as L
    from risley_lattice import solve18
    os.makedirs(WORK, exist_ok=True)
    tc = L.gen('ben', 87)
    f = os.path.join(WORK, 'ben87.json')
    st = json.load(open(f)) if os.path.exists(f) else {}
    t0 = time.time()
    if step == 80:
        x, mse, how, info = solve18(make_pattern(tc, 80.0))
        err = np.abs(x - tc)
        st['80'] = dict(x=x.tolist(), mse=float(mse), err=float(err.max()),
                        worst=NAMES[int(err.argmax())], how=how, mode='from scratch')
    else:
        prev = st[str(step // 2)]
        x, mse, err, worst = warm_trf(prev['x'], tc, float(step))
        st[str(step)] = dict(x=x.tolist(), mse=mse, err=err, worst=worst,
                             mode=f'warm start from {step // 2} s answer')
    st[str(step)]['secs'] = round(time.time() - t0, 1)
    json.dump(st, open(f, 'w'))
    r = st[str(step)]
    print(step, r['mode'], 'mse %.1e' % r['mse'], 'err %.1e' % r['err'], r['worst'], r['secs'], 's')


# ------------------------------------------------------------------ campaign
def build(n_basin=150, n_other=100):
    basin, other = [], []
    for line in open(READY):
        r = json.loads(line)
        if r['T_solved'] is not None:
            continue
        tr = {t['T']: t for t in r['trail']}
        if not all(T in tr and tr[T].get('x18') is not None for T in LADDER):
            continue
        job = dict(case=r['case'], seed0=r['seed0'],
                   steps=[dict(T_from=LADDER[k], T_to=LADDER[k + 1],
                               x0=tr[LADDER[k]]['x18'],
                               scratch_err=tr[LADDER[k + 1]]['err'],
                               scratch_mse=tr[LADDER[k + 1]]['mse'])
                          for k in range(3)])
        cert = tr[80.0].get('cert') or []
        (basin if any(c.startswith('basin') for c in cert) else other).append(job)
    random.Random(7).shuffle(basin); random.Random(8).shuffle(other)
    pop = dict(basin=len(basin), other=len(other))
    jobs = [dict(group='basin', **j) for j in basin[:n_basin]] + \
           [dict(group='other', **j) for j in other[:n_other]]
    os.makedirs(WORK, exist_ok=True)
    json.dump(dict(population=pop, jobs=jobs), open(os.path.join(WORK, 'jobs.json'), 'w'))
    print('population', pop, 'jobs', len(jobs))


def load_done():
    d = defaultdict(dict)
    for fp in glob.glob(os.path.join(WORK, 'u_*.jsonl')):
        for line in open(fp):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            d[r['case']][r['T_to']] = r
    return d


def case_finished(job, done):
    rs = done.get(job['case'], {})
    if any(r['mse'] < MSE_OK for r in rs.values()):
        return True
    return all(s['T_to'] in rs for s in job['steps'])


def run(shard, nshard, budget):
    t0 = time.time()
    jobs = json.load(open(os.path.join(WORK, 'jobs.json')))['jobs']
    done = load_done()
    est = {20.0: 8, 40.0: 15, 80.0: 30}
    with open(os.path.join(WORK, f'u_{shard}.jsonl'), 'a') as f:
        for j in jobs:
            if j['case'] % nshard != shard:
                continue
            for s in j['steps']:
                if case_finished(j, done):
                    break
                if s['T_to'] in done[j['case']]:
                    continue
                if time.time() - t0 + est[s['T_to']] > budget:
                    return
                ts = time.time()
                tc = case_at(j['case'], j['seed0'])
                x, mse, err, worst = warm_trf(s['x0'], tc, s['T_to'])
                r = dict(group=j['group'], case=j['case'], T_from=s['T_from'],
                         T_to=s['T_to'], mse=mse, err=err, worst=worst,
                         scratch_err=s['scratch_err'], scratch_mse=s['scratch_mse'],
                         x0_err=float(np.max(np.abs(np.array(s['x0']) - tc))),
                         axmin=float(np.min(np.abs(tc[3:6]))),
                         secs=round(time.time() - ts, 1))
                done[j['case']][s['T_to']] = r
                f.write(json.dumps(r) + '\n'); f.flush()
                est[s['T_to']] = max(est[s['T_to']] * 0.8, r['secs'] * 1.3)


def report():
    meta = json.load(open(os.path.join(WORK, 'jobs.json')))
    done = load_done()
    out = dict(population=meta['population'], groups={})
    for g in ('basin', 'other'):
        jobs = [j for j in meta['jobs'] if j['group'] == g]
        fin = [j for j in jobs if case_finished(j, done)]
        resc, wrong, by_step, near_resc, near_n = 0, 0, defaultdict(int), 0, 0
        for j in fin:
            rs = done[j['case']]
            acc = [r for r in rs.values() if r['mse'] < MSE_OK]
            if acc:
                r = min(acc, key=lambda r: r['T_to'])
                if r['err'] < 1e-3:
                    resc += 1; by_step[f"{r['T_from']:g}->{r['T_to']:g}"] += 1
                else:
                    wrong += 1
            # cases whose previous answer was already close (< 1 in every parameter)
            if any(r['x0_err'] < 1.0 for r in rs.values()):
                near_n += 1
                near_resc += int(any(r['mse'] < MSE_OK and r['err'] < 1e-3 for r in rs.values()))
        out['groups'][g] = dict(jobs=len(jobs), finished=len(fin), rescued=resc,
                                wrong_accepted=wrong, by_step=dict(by_step),
                                prev_close=near_n, prev_close_rescued=near_resc)
    json.dump(out, open(os.path.join(HERE, 'out', 'warmstart.json'), 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd'); ap.add_argument('--step', type=int, default=80)
    ap.add_argument('--shard', type=int, default=0); ap.add_argument('--nshard', type=int, default=4)
    ap.add_argument('--budget', type=float, default=160)
    a = ap.parse_args()
    {'ben87': lambda: ben87(a.step), 'build': build,
     'run': lambda: run(a.shard, a.nshard, a.budget), 'report': report}[a.cmd]()
