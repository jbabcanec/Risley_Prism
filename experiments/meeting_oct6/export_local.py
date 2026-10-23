#!/usr/bin/env python3
"""Summaries + figures for the local runs (box10, ben, ben066, calib,
nonunique, sign sweep). Writes out/*.json and figures/*.png."""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.argv = sys.argv[:1]
from vizstyle import setup, C, INK2, wilson, plt
import local_campaign as L
from risley_lattice.model import NAMES, RG
OUT, FIG = os.path.join(HERE, 'out'), os.path.join(HERE, 'figures')
setup()
N = dict(ben=160, ben066=60, box10=150, calib=150)
res = {}


def ladder_summary(exp):
    s = L.summarize(exp, N[exp])
    n = len(s)
    by = {}
    for T in L.LADDER[exp]:
        k = sum(1 for c in s if c['T_solved'] is not None and c['T_solved'] <= T)
        p, lo, hi = wilson(k, n)
        by[str(T)] = dict(k=k, rate=float(p), ci=[float(lo), float(hi)])
    to = sum(1 for c in s if [T for T in c['timeouts']
                                if c['T_solved'] is None or T < c['T_solved']])
    errs = [c['trials'][-1]['err'] for c in s if c['T_solved'] is not None]
    fails = [dict(i=c['meta']['i'], ax=np.round(np.abs(c['meta']['truth'][3:6]), 3).tolist(),
                  pair_sep=c['meta']['pair_sep'], rel_gap=c['meta']['rel_gap'],
                  cert=c['trials'][-1].get('cert'), timeouts=c['timeouts'])
             for c in s if c['T_solved'] is None]
    return s, dict(n=n, by_T=by, timeouts=to, median_err=float(np.median(errs)) if errs else None,
                   failures=fails)


for exp in ('box10', 'ben', 'ben066'):
    s, r = ladder_summary(exp)
    if exp.startswith('ben'):
        fl, ax_true, axerr = [], [], []
        for c in s:
            mt = c['meta']; tr = np.abs(mt['truth'][3:6])
            if 'detect_ax10' not in mt:
                continue
            for i in range(3):
                if tr[i] < 2:
                    fl.append(mt['detect_ax10'][i]); ax_true.append(tr[i])
        fl, ax_true = np.array(fl), np.array(ax_true)
        hid_cases = [c for c in s if 'detect_ax10' in c['meta'] and any(
            np.abs(c['meta']['truth'][3 + i]) < c['meta']['detect_ax10'][i]
            for i in range(3) if abs(c['meta']['truth'][3 + i]) < 2)]
        r['cases_with_hidden_wedge_T10'] = len(hid_cases)
        r['hidden_cases_solved_T10'] = sum(1 for c in hid_cases if c['T_solved'] == 10.0)
        r['hidden_cases_solved_adaptive'] = sum(1 for c in hid_cases if c['T_solved'] is not None)
        vis = [c for c in s if c not in hid_cases]
        r['visible_cases'] = len(vis)
        r['visible_cases_solved_T10'] = sum(1 for c in vis if c['T_solved'] == 10.0)
        r['weak_prisms'] = len(fl)
        r['weak_below_floor_T10'] = int(np.sum(ax_true < fl))
        r['floor_T10_median'] = float(np.median(fl))
        r['floor_T10_p90'] = float(np.percentile(fl, 90))
        # alpha_min ~ T^-1/2 (paper Prop. detect): T needed to see 0.66 deg
        treq = 10.0 * (fl / 0.66) ** 2          # only meaningful where > 10 s
        hid = fl >= 0.66
        r['n_hidden_066_T10'] = int(hid.sum())
        r['T_to_detect_066_hidden_median'] = float(np.median(treq[hid])) if hid.any() else None
        r['T_to_detect_066_hidden_p90'] = float(np.percentile(treq[hid], 90)) if hid.any() else None
        r['frac_066_visible_T10'] = float(np.mean(fl < 0.66))
        res[exp + '_floor'] = dict(floor=fl.tolist(), ax=ax_true.tolist())
    res[exp] = r
    print(exp, {k: v for k, v in r.items() if k != 'failures'}, 'fails', len(r['failures']))

# calibration rows for fill_paper.py (E2)
s = L.summarize('calib', N['calib'])
rows = []
for c in s:
    t = c['trials'][0]
    if t['err'] < 1e-3:
        rows.append(dict(i=c['meta']['i'], err=t['err'], bounds_cover=t.get('bounds_cover'),
                         bound_max=t.get('bound_max'), tightness_med=t.get('tightness_med')))
json.dump(rows, open(os.path.join(OUT, 'calib_rows.json'), 'w'))
res['calib'] = dict(n_cases=len(s), solved=len(rows),
                    covered=int(sum(1 for r in rows if r['bounds_cover'])))
print('calib', res['calib'])

# Ben figure
fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.1))
for j, exp in enumerate(('box10', 'ben', 'ben066')):
    r = res[exp]; Ts = [float(t) for t in r['by_T']]
    ax[0].plot(Ts, [100 * r['by_T'][t]['rate'] for t in r['by_T']], 'o-', color=C[j],
               label={'box10': '|ax| ≤ 10° box', 'ben': 'Ben: 1 strong + 2 wedges 0.3–1°',
                      'ben066': 'Ben: 1 strong + 2 wedges 0.66°'}[exp] + f' (n={r["n"]})')
ax[0].set_xscale('log'); ax[0].set_xlabel('observation time T (s)')
ax[0].set_ylabel('recovered by T (%)'); ax[0].legend(fontsize=7); ax[0].set_ylim(80, 101)
f = res['ben_floor']
ax[1].loglog(f['ax'], f['floor'], '.', color=C[1], ms=4)
ax[1].plot([0.2, 2], [0.2, 2], color=INK2, lw=1)
ax[1].set_xlabel('true |ax| of sub-degree wedge (deg)')
ax[1].set_ylabel('detectability floor at T = 10 s (deg)')
ax[1].set_title('below the line = visible to the spectral stage', fontsize=8)
fig.savefig(os.path.join(FIG, 'C2_ben_regime.png'))
plt.close(fig)

# nonunique
import nonunique as NU
d = NU.load()
eps = [r for k, r in d.items() if k[0] == 'eps']
hw = np.array([r['hw_rel'] for r in eps])          # per unit eps, relative to box
sv = np.array([r['sv'] for r in eps])
wd = np.abs(np.array([r['weak_dir'] for r in eps]))
nu = dict(n_eps=len(eps), cond_median=float(np.median(sv[:, 0] / sv[:, -1])),
          halfwidth_rel_at_eps_1em3={NAMES[j]: float(np.median(hw[:, j]) * 1e-3) for j in range(18)},
          weak_dir_mean_abs={NAMES[j]: float(wd[:, j].mean()) for j in range(18)})
tw = [r for k, r in d.items() if k[0] == 'twin']
twa = np.array([[abs(r['ax']), r['rms'], r['dist'], r['back_to_truth'] or r['rms'] < 1e-8] for r in tw], float)
nu['twin'] = dict(n=len(tw), back_to_truth=int(twa[:, 3].sum()) if len(tw) else 0,
                  twins_rms_lt_1em3=int(np.sum((twa[:, 1] < 1e-3) & (twa[:, 3] == 0))) if len(tw) else 0,
                  twins_rms_lt_1em2=int(np.sum((twa[:, 1] < 1e-2) & (twa[:, 3] == 0))) if len(tw) else 0,
                  twins_rms_lt_1em1=int(np.sum((twa[:, 1] < 1e-1) & (twa[:, 3] == 0))) if len(tw) else 0)
ms = [r for k, r in d.items() if k[0] == 'ms']
exact = [r for r in ms if r['mse'] < 1e-12]
nu['multistart'] = dict(starts=len(ms), exact_minima=len(exact),
                        exact_at_truth=int(sum(r['err'] < 1e-6 for r in exact)),
                        exact_elsewhere=int(sum(r['err'] >= 1e-6 for r in exact)),
                        cases=len({k[1] for k in d if k[0] == 'ms'}))
dg = [r for k, r in d.items() if k[0] == 'degen']
nu['degen'] = dict(n=len(dg),
                   sigmin_ratio_generic=float(np.median([r['generic'][-1] / r['generic'][0] for r in dg])),
                   sigmin_ratio_equal=float(np.median([r['equal_speed'][-1] / r['equal_speed'][0] for r in dg])))
res['nonunique'] = nu
print(json.dumps(nu, indent=1)[:2500])
if len(tw):
    fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.1))
    sel = twa[:, 3] == 0
    ax[0].semilogy(twa[sel, 0], twa[sel, 1], 'o', ms=4, color=C[0], label='best twin found')
    ax[0].set_xlim(0, 18)
    ax[0].set_xlabel('|ax| of the flipped wedge (deg)')
    ax[0].set_ylabel('pattern RMS of best twin')
    ax[0].set_title('How close can a sign-flipped wedge get?', fontsize=8)
    med = np.median(hw, 0)
    o = np.argsort(med)
    ax[1].barh(range(18), med[o], color=C[0])
    ax[1].set_yticks(range(18), [NAMES[j] for j in o], fontsize=7); ax[1].set_xscale('log')
    ax[1].set_xlabel('half-width at ε = 1 (fraction of range)')
    ax[1].set_title('Local ε-equivalence (median over cases)', fontsize=8)
    fig.savefig(os.path.join(FIG, 'C4_nonuniqueness.png')); plt.close(fig)

# sign sweep
import sign_sweep_local as SS
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    SS.report(os.path.expanduser('~/work/ss'))
ss = json.load(open(os.path.join(OUT, 'sign_sweep_local.json')))
res['sign_sweep'] = ss['groups']; res['sign_sweep_population'] = ss['population']
print('sign sweep', ss['groups'])
g = [k for k in ('G1', 'G3') if ss['groups'][k]['tested']]
if g:
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    for j, k in enumerate(g):
        t = ss['groups'][k]; p, lo, hi = wilson(t['rescued'], t['tested'])
        ax.bar(j, 100 * p, color=C[j], width=0.6)
        ax.errorbar(j, 100 * p, yerr=[[100 * (p - lo)], [100 * (hi - p)]], color=INK2, lw=1)
        ax.text(j, 100 * hi + 2, f'{t["rescued"]}/{t["tested"]}', ha='center', fontsize=8)
    ax.set_xticks(range(len(g)), ['G1 unsolved,\npure sign flip', 'G3 unsolved,\nother angle error'][:len(g)])
    ax.set_ylabel('rescued by sign-sweep rung (%)'); ax.set_ylim(0, 100)
    fig.savefig(os.path.join(FIG, 'C3_sign_sweep.png')); plt.close(fig)
json.dump(res, open(os.path.join(OUT, 'local_summary.json'), 'w'), indent=1)
