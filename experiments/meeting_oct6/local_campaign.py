#!/usr/bin/env python3
"""Small local campaigns (Python on this machine only), checkpointed per
trial so they survive the ~3-minute command limit.

Experiments
  box10   uniform box with every |ax| <= 10 deg (fresh seed): confirms the
          sub-population estimate from the 669,574-case campaign.
  ben     Ben's hardware regime: one strong wedge (6 or 12 deg) + two
          free wedges with |ax| in [0.3, 1.0] deg. Ladder up to 320 s.
  ben066  same, both free wedges exactly 0.66 deg (Ben's recollection).
  calib   standard box, T = 10 s, with per-parameter certified bounds
          (fills paper placeholder E2: bound coverage / tightness).

  python3 local_campaign.py run --exp ben --n 300 --shard k --budget 160
  python3 local_campaign.py report --exp ben
"""
import os
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[k] = '1'
import sys, json, time, glob, argparse
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(EXP)); sys.path.insert(0, EXP)
from risley_lattice.model import LO, RG, N_PAR, canon, truth_margins, case_at
from sweep import solve_once
from tir import tir_margin18

WORK = os.environ.get('LC_WORK', os.path.expanduser('~/work/lc'))
LADDER = {'box10': [10., 20., 40., 80.], 'ben': [10., 20., 40., 80., 160., 320.],
          'ben066': [10., 20., 40., 80., 160., 320.], 'calib': [10.]}
SEED = {'box10': 20261006, 'ben': 20261007, 'ben066': 20261009,
        'calib': 20261008}


def gen(exp, i):
    if exp == 'calib':
        return case_at(i, SEED[exp])
    rng = np.random.default_rng([SEED[exp], i])
    v = LO + RG * rng.random(N_PAR)
    for j in range(3):
        if abs(v[j]) < 0.15:
            v[j] = np.copysign(0.15, v[j])
    if exp == 'box10':
        v[3:6] = rng.uniform(-10, 10, 3)
    else:
        s = int(rng.integers(3))
        sg = rng.choice([-1.0, 1.0], 3)
        mag = rng.uniform(0.3, 1.0, 3) if exp == 'ben' else np.full(3, 0.66)
        mag[s] = rng.choice([6.0, 12.0])
        v[3:6] = sg * mag
    return canon(v)


def detect_floor(tc, T=10.0):
    """Oracle-speed detectability: smallest |ax| the 5-sigma amplitude test
    can see at T for each prism (certify.py's weak-prism formula)."""
    from risley_lattice.model import FS, vec2pat
    from risley_lattice.certify import spectral_fisher
    from risley_lattice.angles import calibrate_ax, nominal_rest
    pat = vec2pat(tc, int(round(T * FS)), T)
    g, sg, amp, samp, res = spectral_fisher(pat, list(tc[:3]),
                                            np.ones(len(pat), bool))
    coefs = calibrate_ax(g, nominal_rest(pat))
    return [float(np.degrees(np.arctan(5 * samp[i] / max(coefs[i][0], 1e-6))))
            for i in range(3)], amp.tolist(), samp.tolist()


def load(exp):
    rows = defaultdict(list)
    for fp in glob.glob(os.path.join(WORK, f'{exp}_*.jsonl')):
        for line in open(fp):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows[r['i']].append(r)
    return rows


def load_one(exp, i):
    out = []
    for fp in glob.glob(os.path.join(WORK, f'{exp}_*.jsonl')):
        for line in open(fp):
            if f'"i": {i},' not in line and f'"i": {i}}}' not in line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r['i'] == i:
                out.append(r)
    return out


def status(recs, ladder):
    """-> (finished?, next T or None). Trials with 2 unfinished starts are
    recorded as timeouts and skipped."""
    res = {r['T']: r for r in recs if r['kind'] == 'trial'}
    starts = defaultdict(int)
    for r in recs:
        if r['kind'] == 'start':
            starts[r['T']] += 1
    for T in ladder:
        if T in res:
            if res[T]['err'] < 1e-3:
                return True, None
            continue
        if starts[T] >= 2:
            continue
        return False, T
    return True, None


def run(exp, n, shard, nshard, budget):
    t0 = time.time()
    rows = load(exp)
    hist = defaultdict(list)
    for recs in rows.values():
        for r in recs:
            if r['kind'] == 'trial':
                hist[r['T']].append(r['secs'])
    # typical (not worst-case) duration; overruns are caught by `timeout`
    # in round.sh and the start markers
    est = defaultdict(lambda: 10.0,
                      {T: float(np.median(v)) * 1.5 for T, v in hist.items()})
    path = os.path.join(WORK, f'{exp}_{shard}.jsonl')
    with open(path, 'a') as f:
        order = list(range(n))
        order = order[shard::nshard] + [i for i in order if i % nshard != shard]
        cdir = os.path.join(WORK, 'claims'); os.makedirs(cdir, exist_ok=True)
        for i in order:
            while True:
                rows[i] = load_one(exp, i) if i % nshard != shard else rows[i]
                fin, T = status(rows[i], LADDER[exp])
                if fin:
                    break
                el = time.time() - t0
                killed = sum(1 for r in rows[i]
                             if r['kind'] == 'start' and r['T'] == T)
                if killed and el > 5:
                    break          # retry a killed trial only at round start
                if el + min(est[T], 150) > budget:
                    return
                try:               # work stealing: one worker per trial
                    os.close(os.open(os.path.join(
                        cdir, f'{exp}_{i}_{T}_{killed}'),
                        os.O_CREAT | os.O_EXCL | os.O_WRONLY))
                except FileExistsError:
                    break
                tc = gen(exp, i)
                if not rows[i]:
                    meta = dict(kind='meta', i=i, exp=exp, T=0,
                                truth=tc.tolist(),
                                tir_margin=float(tir_margin18(tc)[0]),
                                **truth_margins(tc))
                    if exp.startswith('ben'):
                        try:
                            meta['detect_ax10'], meta['amp10'], \
                                meta['samp10'] = detect_floor(tc)
                        except Exception as ex:
                            meta['detect_err'] = repr(ex)
                    f.write(json.dumps(meta) + '\n'); rows[i].append(meta)
                st = dict(kind='start', i=i, T=T)
                f.write(json.dumps(st) + '\n'); f.flush(); rows[i].append(st)
                row, err = solve_once(tc, T, None, None,
                                      with_bounds=(exp == 'calib'))
                row.pop('x18', None)
                row.update(kind='trial', i=i)
                f.write(json.dumps(row) + '\n'); f.flush()
                rows[i].append(row)


def summarize(exp, n):
    rows = load(exp)
    out = []
    for i in range(n):
        recs = rows.get(i, [])
        fin, _ = status(recs, LADDER[exp]) if recs else (False, None)
        if not fin:
            continue
        meta = next(r for r in recs if r['kind'] == 'meta')
        tri = sorted([r for r in recs if r['kind'] == 'trial'],
                     key=lambda r: r['T'])
        to = [T for T in LADDER[exp] if T not in {r['T'] for r in tri}]
        Ts = next((r['T'] for r in tri if r['err'] < 1e-3), None)
        out.append(dict(meta=meta, trials=tri, T_solved=Ts, timeouts=to))
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd'); ap.add_argument('--exp', required=True)
    ap.add_argument('--n', type=int, default=300)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshard', type=int, default=4)
    ap.add_argument('--budget', type=float, default=160)
    a = ap.parse_args()
    os.makedirs(WORK, exist_ok=True)
    if a.cmd == 'run':
        run(a.exp, a.n, a.shard, a.nshard, a.budget)
    else:
        s = summarize(a.exp, a.n)
        print(a.exp, 'finished cases', len(s))
