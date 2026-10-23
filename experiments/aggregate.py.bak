#!/usr/bin/env python3
"""Aggregate sweep JSONL into the large-scale statistics and figures.

  python experiments/aggregate.py                # stats to stdout
  python experiments/aggregate.py --figs         # + phase-diagram figures

Reads every results/sweep_*.jsonl. Stats: recovery rates per T with Wilson
intervals, certificate-class breakdown of failures, coverage/tightness if
--bounds rows exist, prescription quantiles from mint rows, P x T table
from speedsP rows. Figures: recovery probability vs the certificate
margins (pair separation x T, relation gap x T, wedge angle), the measured
phase boundaries against the derived thresholds.
"""
import os, sys, json, glob, argparse, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, c - h), min(1.0, c + h))


def load():
    rows = []
    for p in sorted(glob.glob(os.path.join(OUTDIR, 'sweep_*.jsonl'))):
        with open(p) as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows


def report(rows):
    by_exp = {}
    for r in rows:
        by_exp.setdefault(r['exp'], []).append(r)
    for exp, rr in by_exp.items():
        print(f"\n=== {exp}: {len(rr)} tasks ===")
        if exp in ('adaptive', 'noise'):
            slabs = sorted({r.get('snr') for r in rr},
                           key=lambda s: -1e9 if s is None else -s)
            for snr in slabs:
                sub = [r for r in rr if r.get('snr') == snr]
                n = len(sub)
                Ts = sorted({t['T'] for r in sub for t in r['trail']})
                lab = 'noiseless' if snr is None else f'{snr:.0f} dB'
                line = [f"  [{lab}] n={n}"]
                for T in Ts:
                    k = sum(1 for r in sub
                            if r.get('T_solved') is not None
                            and r['T_solved'] <= T)
                    p, lo, hi = wilson(k, n)
                    line.append(f"T<={T:.0f}: {100*p:.2f}%"
                                f" [{100*lo:.2f},{100*hi:.2f}]")
                print('  '.join(line))
                uns = [r for r in sub if r.get('T_solved') is None]
                cls = {}
                for r in uns:
                    key = tuple(sorted(set(
                        r['trail'][-1].get('cert', ['?']))))
                    cls[key] = cls.get(key, 0) + 1
                for key, k in sorted(cls.items(), key=lambda x: -x[1]):
                    print(f"      unsolved {k:>5}  {'; '.join(key)}")
            cov = [t for r in rr for t in r['trail']
                   if t.get('bounds_cover') is not None]
            if cov:
                k = sum(1 for t in cov if t['bounds_cover'])
                p, lo, hi = wilson(k, len(cov))
                tight = np.median([t['tightness_med'] for t in cov])
                print(f"  coverage {k}/{len(cov)} = {100*p:.2f}% "
                      f"[{100*lo:.2f},{100*hi:.2f}]  med tightness "
                      f"{tight:.1f}x")
        elif exp == 'mint':
            have = [r for r in rr if r.get('T_min') is not None]
            inf = sum(1 for r in rr if r.get('T_min') is None)
            arr = np.array([r['T_min'] for r in have])
            if len(arr):
                print(f"  minimal recovering T: median {np.median(arr):.0f}"
                      f"  p90 {np.percentile(arr, 90):.0f}"
                      f"  max {arr.max():.0f}   infeasible<=Tmax: {inf}")
        elif exp == 'speedsP':
            for P in sorted({r['P'] for r in rr}):
                sub = [r for r in rr if r['P'] == P]
                Ts = sorted({t['T'] for r in sub for t in r['trail']})
                line = [f"  P={P} n={len(sub)}"]
                for T in Ts:
                    k = sum(1 for r in sub
                            if r.get('T_solved') is not None
                            and r['T_solved'] <= T)
                    p, lo, hi = wilson(k, len(sub))
                    line.append(f"T<={T:.0f}: {100*p:.1f}%")
                print('  '.join(line))


def figures(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    BLUE, MUT, INK = '#3b6fb6', '#8a8a8a', '#333333'
    plt.rcParams.update({'font.size': 8, 'font.family': 'serif',
                         'mathtext.fontset': 'cm'})
    ad = [r for r in rows if r['exp'] == 'adaptive'
          and r.get('snr') is None]
    if not ad:
        print('no adaptive rows; skipping figures')
        return
    T0 = min(t['T'] for r in ad for t in r['trail'])
    solved0 = np.array([r.get('T_solved') == T0 for r in ad])

    def margin_curve(ax, vals, solved, nbins, xlabel, xlog=True,
                     thr=None, thr_label=None):
        vals = np.asarray(vals)
        edges = np.logspace(np.log10(max(vals.min(), 1e-5)),
                            np.log10(vals.max()), nbins + 1) if xlog \
            else np.linspace(vals.min(), vals.max(), nbins + 1)
        for a, b in zip(edges[:-1], edges[1:]):
            m = (vals >= a) & (vals < b)
            if m.sum() >= 20:
                p, lo, hi = wilson(int(solved[m].sum()), int(m.sum()))
                x = (a * b) ** 0.5 if xlog else 0.5 * (a + b)
                ax.errorbar(x, 100 * p, yerr=[[100 * (p - lo)],
                                              [100 * (hi - p)]],
                            fmt='o', ms=3.5, color=BLUE, ecolor=MUT,
                            elinewidth=0.8, capsize=0)
        if xlog:
            ax.set_xscale('log')
        if thr is not None:
            ax.axvline(thr, color='#d1495b', lw=1.0, ls='--')
            ax.text(thr, 20, ' ' + (thr_label or ''), fontsize=7,
                    color='#d1495b', rotation=90, va='bottom')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f'recovered at $T={T0:.0f}$ s (%)')
        ax.set_ylim(-3, 103)
        ax.grid(True, color='#e6e6e6', lw=0.4)

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3))
    margin_curve(axes[0], [r['pair_sep'] for r in ad], solved0, 14,
                 'min pair separation (Hz)',
                 thr=0.12 / T0, thr_label='merge floor $C_m/T$')
    margin_curve(axes[1], [r['rel_gap'] for r in ad], solved0, 14,
                 'min lattice-relation gap (Hz)',
                 thr=0.12 / T0, thr_label='merge floor $C_m/T$')
    margin_curve(axes[2], [r['ax_min'] for r in ad], solved0, 14,
                 'min wedge magnitude (deg)')
    for ax, t in zip(axes, ['(a) pair resolvability',
                            '(b) lattice degeneracy',
                            '(c) prism detectability']):
        ax.set_title(t, loc='left', fontsize=8)
    fig.tight_layout(w_pad=1.4)
    out = os.path.join(os.path.dirname(OUTDIR), '..', 'paper', 'figures',
                       'atlas.pdf')
    out = os.path.normpath(out)
    fig.savefig(out, dpi=300, bbox_inches='tight')
    fig.savefig(out.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f'wrote {out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--figs', action='store_true')
    a = ap.parse_args()
    rows = load()
    print(f'loaded {len(rows)} rows')
    report(rows)
    if a.figs:
        figures(rows)
