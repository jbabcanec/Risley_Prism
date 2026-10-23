"""
angle_ambiguity.py
Does wedge-angle ambiguity cause failures?  For every trial that did NOT reach
machine precision, regenerate the ground truth (case_at(i, seed0)) and classify
what went wrong per prism:
  sign-flip  : sign(ax_hat) != sign(ax_true)             (the near-symmetry ax -> -ax)
  phase-slip : sign right but |ay_hat - ay_true| > 5 deg
  ok-angles  : both angles within tolerance (failure is elsewhere: speeds/glass/geo)
Also: how often a flip rung rescued a case, binned by the true min |ax|.
"""
import json, glob, pickle
import numpy as np
from collections import Counter, defaultdict

RES = '/sessions/vibrant-youthful-curie/mnt/Ben/Risley_Prism-main/results'
LO = np.array([-3.5]*3 + [-18.]*6 + [1.3]*3 + [50., 2., -25., -25., -5., -5.])
HI = np.array([3.5]*3 + [18.]*6 + [1.8]*3 + [200., 15., 25., 25., 5., 5.])
RG = HI - LO

def canon(v):
    v = v.copy(); o = np.argsort(-np.abs(v[:3]))
    v[:3], v[3:6], v[6:9], v[9:12] = v[:3][o], v[3:6][o], v[6:9][o], v[9:12][o]
    return v

def case_at(i, seed0, min_speed=0.15):
    rng = np.random.default_rng([seed0, int(i)])
    v = LO + RG * rng.random(18)
    for j in range(3):
        if abs(v[j]) < min_speed:
            v[j] = np.copysign(min_speed, v[j])
    return canon(v)

def wrap(d): return (d + 180.0) % 360.0 - 180.0

TOL_ERR = 1e-3
bad_trials = []          # (atlas, case, T, how, per-prism classes, |ax_true|, final_solved)
flip_win = defaultdict(lambda: [0, 0])   # axmin bin -> [flip wins, solved]
fail_axmin, all_axmin = [], []
checked_truth = 0; truth_mismatch = 0

for atlas in ['atlas2', 'atlas3', 'atlas4', 'atlas5']:
    for fp in sorted(glob.glob(f'{RES}/{atlas}/*.jsonl')):
        for line in open(fp):
            r = json.loads(line)
            solved = r['T_solved'] is not None
            tc = None
            # sanity: regenerated truth must reproduce stored ax_min
            if checked_truth < 2000:
                tc = case_at(r['case'], r['seed0'])
                checked_truth += 1
                if abs(np.min(np.abs(tc[3:6])) - r['ax_min']) > 1e-6:
                    truth_mismatch += 1
            axmin = r['ax_min']
            all_axmin.append(axmin)
            if not solved:
                fail_axmin.append(axmin)
            b = min(int(axmin // 1), 17)
            if solved:
                win = next(t for t in r['trail'] if t['err'] < TOL_ERR)
                flip_win[b][1] += 1
                if 'flip' in win['how']:
                    flip_win[b][0] += 1
            for t in r['trail']:
                if t['err'] < TOL_ERR or t['x18'] is None:
                    continue
                if tc is None:
                    tc = case_at(r['case'], r['seed0'])
                x = np.array(t['x18'])
                cls = []
                for i in range(3):
                    at, ah = tc[3+i], x[3+i]
                    dy = abs(wrap(x[6+i] - tc[6+i]))
                    dax = abs(ah - at)
                    if np.sign(ah) != np.sign(at) and abs(at) > 0.05:
                        cls.append('sign-flip')
                    elif dy > 5 or dax > 1:
                        cls.append('angle-off')
                    else:
                        cls.append('ok')
                spd_ok = np.max(np.abs(x[:3] - tc[:3])) < 0.02
                bad_trials.append(dict(atlas=atlas, case=r['case'], T=t['T'],
                                       how=t['how'], cls=cls, spd_ok=spd_ok,
                                       axt=np.abs(tc[3:6]).tolist(),
                                       final=solved,
                                       cert=[c.split()[0] for c in t.get('cert', [])]))

print(f'truth regeneration check: {truth_mismatch}/{checked_truth} mismatches')
print(f'non-converged trials with x18: {len(bad_trials):,}')

# ---- classification
def summarize(rows, label):
    n = len(rows)
    spd_ok = sum(r['spd_ok'] for r in rows)
    any_flip = sum('sign-flip' in r['cls'] for r in rows)
    only_flip = sum(r['spd_ok'] and 'sign-flip' in r['cls'] and 'angle-off' not in r['cls'] for r in rows)
    angle_off = sum(r['spd_ok'] and 'angle-off' in r['cls'] for r in rows)
    print(f'\n{label}: n={n:,}')
    print(f'  speeds correct (<0.02 Hz)          : {spd_ok:,} ({100*spd_ok/n:.1f}%)')
    print(f'  any prism with ax SIGN flipped     : {any_flip:,} ({100*any_flip/n:.1f}%)')
    print(f'  speeds right, ONLY sign flips wrong: {only_flip:,} ({100*only_flip/n:.1f}%)')
    print(f'  speeds right, other angle error    : {angle_off:,} ({100*angle_off/n:.1f}%)')
    # |ax_true| of flipped prisms
    fl = [r['axt'][i] for r in rows for i in range(3) if r['cls'][i] == 'sign-flip']
    ok = [r['axt'][i] for r in rows for i in range(3) if r['cls'][i] == 'ok']
    if fl:
        print(f'  |ax_true| of flipped prisms: median {np.median(fl):.2f} deg, '
              f'p90 {np.percentile(fl,90):.2f}  (non-flipped prisms median {np.median(ok) if ok else float("nan"):.2f})')
        h = np.histogram(fl, bins=[0,0.5,1,2,3,5,10,18])[0]
        print('  flipped |ax_true| histogram [0,.5,1,2,3,5,10,18]:', h.tolist())
    return fl

final_fail = [r for r in bad_trials if not r['final']]
# keep only LAST trial per unsolved case
last = {}
for r in final_fail:
    k = (r['atlas'], r['case'])
    if k not in last or r['T'] > last[k]['T']:
        last[k] = r
last = list(last.values())
rescued = [r for r in bad_trials if r['final']]

fl_fail = summarize(last, 'UNSOLVED cases, final trial (T=80 s)')
fl_resc = summarize(rescued, 'RESCUED cases, failed early trials')

print('\nCertificates on unsolved cases whose ONLY defect is a sign flip:')
c = Counter(tuple(sorted(set(r['cert']))) for r in last
            if r['spd_ok'] and 'sign-flip' in r['cls'] and 'angle-off' not in r['cls'])
for k, v in c.most_common(8):
    print(f'  {v:6,}  {k}')

print('\nFailure rate vs true min |ax| (deg):')
fa = np.array(fail_axmin); aa = np.array(all_axmin)
edges = [0, 0.5, 1, 2, 3, 5, 10, 18]
rate_rows = []
for lo, hi in zip(edges[:-1], edges[1:]):
    n = ((aa >= lo) & (aa < hi)).sum(); f = ((fa >= lo) & (fa < hi)).sum()
    rate_rows.append((lo, hi, n, f))
    print(f'  [{lo:>4},{hi:>4})  n={n:8,}  unsolved={f:6,}  rate={100*f/max(n,1):5.2f}%')

print('\nShare of solves won by a flip rung, by min |ax| bin (1 deg bins):')
for b in sorted(flip_win):
    w, s = flip_win[b]
    print(f'  [{b:>2},{b+1:>2})  flip-wins {w:6,} / {s:8,}  = {100*w/max(s,1):5.2f}%')

pickle.dump(dict(last=last, rescued=rescued, rate_rows=rate_rows,
                 flip_win=dict(flip_win), fl_fail=fl_fail, fl_resc=fl_resc),
            open('/tmp/angle_amb.pkl', 'wb'))
