#!/usr/bin/env python3
"""Six analyses of the identifiability atlas, in one streaming pass.

  python3 analyze_atlas.py                          # text report + figures
  python3 analyze_atlas.py --no-figs                # text only
  python3 analyze_atlas.py --in <file> --figdir <d>

A1  prescription validation  -- does T_req PREDICT the recovering T?
A2  per-parameter recovery   -- which of the 18 parameters is hardest? (x18)
A3  2-D identifiability atlas-- recovery over (pair_sep, cyc)
A4  sigma_N scaling          -- does the Fisher sigma follow T^-3/2?
A5  certificate validity     -- do classes fire where the TRUE margin is small?
A6  the basin population     -- solver failures, not identifiability failures

Ground truth is regenerated with case_at(case, seed0), so A2/A5/A6 need no
stored truth beyond the case index.
"""
import argparse, json, math, os, sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

T_CENSOR = 9999.0        # spectral_certificate caps printed T_req at 9999

# --------------------------------------------------------------------------
# Ground-truth regeneration, replicated from risley_lattice/model.py.
#
# Importing risley_lattice pulls in reverse_problem_v2/core -> scipy, which is
# a heavy dependency this pure-analysis script does not otherwise need (it
# never runs the forward model). case_at() itself needs only numpy, so it is
# mirrored here and CHECKED against the real implementation at startup
# whenever that import happens to succeed -- so any future drift in model.py
# is caught rather than silently producing wrong errors.
# --------------------------------------------------------------------------
N_PAR = 18
NAMES = ['N1', 'N2', 'N3', 'ax1', 'ax2', 'ax3', 'ay1', 'ay2', 'ay3',
         'ng1', 'ng2', 'ng3', 'd_W', 'gap', 'bm_ax', 'bm_ay',
         'bm_px', 'bm_py']
LO = np.array([-3.5] * 3 + [-18.] * 6 + [1.3] * 3 +
              [50., 2., -25., -25., -5., -5.], dtype=np.float64)
HI = np.array([3.5] * 3 + [18.] * 6 + [1.8] * 3 +
              [200., 15., 25., 25., 5., 5.], dtype=np.float64)
RG = HI - LO


def _canon(v):
    v = v.copy()
    o = np.argsort(-np.abs(v[:3]))
    v[:3], v[3:6], v[6:9], v[9:12] = (v[:3][o], v[3:6][o],
                                      v[6:9][o], v[9:12][o])
    return v


def case_at(i, seed0=424242, min_speed=0.15):
    rng = np.random.default_rng([seed0, int(i)])
    v = LO + RG * rng.random(N_PAR)
    for j in range(3):
        if abs(v[j]) < min_speed:
            v[j] = np.copysign(min_speed, v[j])
    return _canon(v)


def _verify_case_at():
    """If the real module imports, prove the replica is bit-identical."""
    try:
        from risley_lattice import case_at as ref
        from risley_lattice.model import NAMES as refN, RG as refR
    except Exception:
        return "replica (risley_lattice not importable here)"
    for i in (0, 1, 7, 12345, 999999):
        if not np.array_equal(case_at(i), ref(i)):
            sys.exit(f"FATAL: case_at replica diverges from "
                     f"risley_lattice at case {i} -- model.py has changed")
    assert list(refN) == NAMES and np.array_equal(refR, RG), \
        "FATAL: NAMES/RG diverge from risley_lattice.model"
    return "verified identical to risley_lattice.model"


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, (c - h) / d, (c + h) / d


def classes_of(rung):
    return {c.split()[0] for c in (rung.get('cert') or []) if c}


# ----------------------------------------------------------------- pass 1
def load(path, limit=None):
    A = dict(
        # A1
        rescue=[],          # (T_req_at_failure, T_solved)
        uns_treq=[],        # T_req on the final rung of unsolved cases
        uns_treq_none=0,
        # A2
        perr=[], 
        # A3 / A5 / A6
        pair=[], relg=[], axm=[], cyc=[], solvedT=[], cls=[],
        # A4  (unsolved only -- same population at every T)
        sig=defaultdict(list),
        # misc
        how=Counter(), masked=[], overload=0, secs=defaultdict(list),
    )
    n = 0
    with open(path) as f:
        for line in f:
            if limit and n >= limit:
                break
            try:
                r = json.loads(line)
            except Exception:
                continue
            n += 1
            if n % 100000 == 0:
                print(f"    {n:,} rows...", flush=True)

            trail = r.get('trail') or []
            if not trail:
                continue
            Ts = r.get('T_solved')

            A['pair'].append(r.get('pair_sep', np.nan))
            A['relg'].append(r.get('rel_gap', np.nan))
            A['axm'].append(r.get('ax_min', np.nan))
            A['cyc'].append(r.get('cyc', np.nan))
            A['solvedT'].append(np.nan if Ts is None else float(Ts))
            A['cls'].append(classes_of(trail[-1]))

            for rung in trail:
                A['secs'][rung.get('T')].append(rung.get('secs') or 0.0)
                if rung.get('overload'):
                    A['overload'] += 1
                A['masked'].append(rung.get('masked') or 0)

            if Ts is not None:
                # A1: prescription from the FIRST failed rung vs what worked
                for rung in trail[:-1]:
                    tq = rung.get('T_req')
                    if tq:
                        A['rescue'].append((float(tq), float(Ts)))
                        break
                # A2: per-parameter error at the successful rung
                x18 = trail[-1].get('x18')
                A['how'][trail[-1].get('how')] += 1
                if x18:
                    tc = case_at(r['case'], r.get('seed0', 424242))
                    A['perr'].append(np.abs(np.asarray(x18) - tc))
            else:
                tq = trail[-1].get('T_req')
                if tq:
                    A['uns_treq'].append(float(tq))
                else:
                    A['uns_treq_none'] += 1
                for rung in trail:                 # A4
                    s = rung.get('sigma_N')
                    if s:
                        A['sig'][rung.get('T')].append(float(s))
    A['n'] = n
    for k in ('pair', 'relg', 'axm', 'cyc', 'solvedT'):
        A[k] = np.asarray(A[k], float)
    A['perr'] = (np.asarray(A['perr'], float) if A['perr']
                 else np.zeros((0, 18)))
    return A


# ------------------------------------------------------------------ A1
def a1(A, out):
    out(f"\n{'='*72}\nA1  PRESCRIPTION VALIDATION -- does T_req predict?\n{'='*72}")
    resc = A['rescue']
    if resc:
        R = np.array([x[0] for x in resc]); S = np.array([x[1] for x in resc])
        live = R < T_CENSOR
        ok = (R <= S)
        p, lo, hi = wilson(int(ok.sum()), len(ok))
        out(f"\nrescue events (failed at some T, later recovered): {len(R):,}")
        out(f"  prescription SUFFICIENT (T_req <= T_solved): "
            f"{ok.sum():,}  {100*p:.2f}%  [{100*lo:.2f},{100*hi:.2f}]")
        out(f"  over-conservative (T_req >  T_solved): {(~ok).sum():,}")
        out(f"  censored at {T_CENSOR:.0f} s: {(~live).sum():,}")
        if live.sum():
            ratio = S[live] / np.maximum(R[live], 1e-9)
            out(f"\n  T_solved / T_req  (1.0 = perfectly sharp):")
            for q in (5, 25, 50, 75, 95):
                out(f"    p{q:<3} {np.percentile(ratio, q):>8.2f}")
            out(f"  -> a median of {np.median(ratio):.2f} means the "
                f"prescription is "
                f"{'sharp' if 0.5 < np.median(ratio) < 4 else 'loose'}; "
                f"values <1 mean it under-prescribed")
    else:
        out("\n  no rescue events with a numeric T_req")

    U = np.array(A['uns_treq'])
    n_uns = len(U) + A['uns_treq_none']
    out(f"\nunsolved at T=80: {n_uns:,}")
    if len(U):
        pred = U > 80.0
        p, lo, hi = wilson(int(pred.sum()), len(U))
        out(f"  with numeric T_req      : {len(U):,}")
        out(f"    T_req >  80  PREDICTED infeasible : {pred.sum():,}  "
            f"{100*p:.2f}%  [{100*lo:.2f},{100*hi:.2f}]")
        out(f"    T_req <= 80  certificate VIOLATED : {(~pred).sum():,}"
            f"   <-- said 80 s would suffice, it did not")
        out(f"    median prescribed T_req: {np.median(U):,.0f} s"
            f"   (censored at {T_CENSOR:.0f}: {(U >= T_CENSOR).sum():,})")
    out(f"  no numeric T_req        : {A['uns_treq_none']:,}"
        f"   (weak-prism / rank-deficient / basin state a detectability"
        f" floor, not a time)")


# ------------------------------------------------------------------ A2
def a2(A, out, figdir):
    out(f"\n{'='*72}\nA2  PER-PARAMETER RECOVERY (x18 vs ground truth)\n{'='*72}")
    E = A['perr']
    if not len(E):
        out("\n  no x18 vectors found"); return
    out(f"\nrecovered cases with x18: {len(E):,}\n")
    out(f"  {'param':<8}{'median':>12}{'p90':>12}{'p99':>12}{'max':>12}"
        f"{'med/range':>12}")
    order = np.argsort(-np.median(E, axis=0) / RG)
    for i in order:
        med = np.median(E[:, i])
        out(f"  {NAMES[i]:<8}{med:>12.2e}{np.percentile(E[:,i],90):>12.2e}"
            f"{np.percentile(E[:,i],99):>12.2e}{E[:,i].max():>12.2e}"
            f"{med/RG[i]:>12.2e}")
    worst, best = NAMES[order[0]], NAMES[order[-1]]
    out(f"\n  hardest (relative to its box): {worst}"
        f"     easiest: {best}")
    out(f"  spread across parameters: "
        f"{(np.median(E,axis=0)/RG).max()/(np.median(E,axis=0)/RG).min():.1e}x")

    if figdir:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 4.2))
        rel = [E[:, i] / RG[i] for i in order]
        bp = ax.boxplot(rel, whis=(5, 95), showfliers=False, patch_artist=True)
        for b in bp['boxes']:
            b.set_facecolor('#8fb8de'); b.set_edgecolor('#31536e')
        for w in bp['medians']:
            w.set_color('#c2410c'); w.set_linewidth(1.6)
        ax.set_yscale('log')
        ax.set_xticklabels([NAMES[i] for i in order], rotation=45, ha='right')
        ax.set_ylabel('|error| / parameter range')
        ax.set_title(f'Per-parameter recovery error, n={len(E):,} '
                     f'(box 25-75%, whiskers 5-95%)')
        ax.grid(axis='y', alpha=.3, which='both')
        fig.tight_layout()
        _save(fig, figdir, 'atlas_per_parameter', out)


# ------------------------------------------------------------------ A3
def a3(A, out, figdir, nb=40):
    out(f"\n{'='*72}\nA3  2-D IDENTIFIABILITY ATLAS (pair_sep x cyc)\n{'='*72}")
    x, y, s = A['pair'], A['cyc'], A['solvedT']
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x, y, s = np.log10(x[m]), np.log10(y[m]), s[m]
    out(f"\ncases with both margins finite: {m.sum():,}")
    for Tcut in (10.0, 80.0):
        cnt, xe, ye = np.histogram2d(x, y, bins=nb)
        okc, _, _ = np.histogram2d(x[np.isfinite(s) & (s <= Tcut)],
                                   y[np.isfinite(s) & (s <= Tcut)],
                                   bins=[xe, ye])
        full = cnt >= 20
        rate = np.where(full, okc / np.maximum(cnt, 1), np.nan)
        out(f"  T<={Tcut:.0f}: {full.sum()} of {nb*nb} cells have >=20 cases;"
            f" median cell n={int(np.median(cnt[full])) if full.any() else 0}")
        if Tcut == 10.0:
            keep = (rate, xe, ye, cnt, full)
    # where is the boundary?
    rate, xe, ye, cnt, full = keep
    with np.errstate(invalid='ignore'):
        band = np.where(full & (rate > .4) & (rate < .6))
    if len(band[0]):
        out(f"\n  the 40-60% recovery band at T=10 sits near "
            f"pair_sep ~ 10^{np.median(xe[band[0]]):.2f} Hz, "
            f"cyc ~ 10^{np.median(ye[band[1]]):.2f}")

    if figdir:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
        for ax, Tcut in zip(axes, (10.0, 80.0)):
            cnt, xe, ye = np.histogram2d(x, y, bins=nb)
            okc, _, _ = np.histogram2d(x[np.isfinite(s) & (s <= Tcut)],
                                       y[np.isfinite(s) & (s <= Tcut)],
                                       bins=[xe, ye])
            rate = np.where(cnt >= 20, okc / np.maximum(cnt, 1), np.nan)
            im = ax.pcolormesh(xe, ye, rate.T, cmap='viridis',
                               vmin=0, vmax=1, shading='auto')
            ax.set_xlabel(r'$\log_{10}$ min pair separation  (Hz)')
            ax.set_ylabel(r'$\log_{10}$ min cycles at $T_{obs}$')
            ax.set_title(f'recovery rate, $T \\leq {Tcut:.0f}$ s')
            fig.colorbar(im, ax=ax, label='fraction recovered')
        fig.suptitle(f'Identifiability atlas, n={m.sum():,} '
                     f'(cells with <20 cases blank)', y=1.02)
        fig.tight_layout()
        _save(fig, figdir, 'atlas_2d_phase', out)


# ------------------------------------------------------------------ A4
def a4(A, out, figdir):
    out(f"\n{'='*72}\nA4  sigma_N SCALING -- theory says T^-3/2\n{'='*72}")
    sig = A['sig']
    Ts = sorted(t for t in sig if t and len(sig[t]) > 30)
    if len(Ts) < 2:
        out("\n  not enough T rungs with sigma_N"); return
    out(f"\n  population: unsolved cases only, so the SAME cases appear at"
        f" every T\n")
    out(f"  {'T':>8}{'n':>10}{'median sigma_N':>18}{'octave slope':>14}")
    med = []
    for i, T in enumerate(Ts):
        m = float(np.median(sig[T])); med.append(m)
        sl = ''
        if i:
            sl = f"{math.log(med[i]/med[i-1])/math.log(Ts[i]/Ts[i-1]):>14.2f}"
        out(f"  {T:>8.0f}{len(sig[T]):>10,}{m:>18.3e}{sl}")
    lt, lm = np.log(np.array(Ts)), np.log(np.array(med))
    slope, icpt = np.polyfit(lt, lm, 1)
    out(f"\n  global log-log fit slope: {slope:.3f}   (theory -1.500)")
    out(f"  {'CONSISTENT' if abs(slope+1.5) < 0.25 else 'DEVIATES'}"
        f" with the T^-3/2 Cramer-Rao prediction")

    if figdir:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4.4))
        for T in Ts:
            v = np.array(sig[T])
            ax.scatter([T]*min(len(v), 400),
                       np.random.choice(v, min(len(v), 400), replace=False),
                       s=4, alpha=.15, color='#8fb8de')
        ax.plot(Ts, med, 'o-', color='#c2410c', label='median')
        ref = med[0] * (np.array(Ts)/Ts[0])**-1.5
        ax.plot(Ts, ref, '--', color='#31536e', label=r'$T^{-3/2}$')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('T (s)'); ax.set_ylabel(r'$\sigma_N$ (Hz)')
        ax.set_title(f'Fisher uncertainty scaling (fit {slope:.2f})')
        ax.legend(); ax.grid(alpha=.3, which='both')
        fig.tight_layout()
        _save(fig, figdir, 'atlas_sigmaN_scaling', out)


# ------------------------------------------------------------------ A5
def a5(A, out):
    out(f"\n{'='*72}\nA5  CERTIFICATE VALIDITY -- does each class fire where the"
        f"\n    corresponding GROUND-TRUTH margin is actually small?\n{'='*72}")
    pairs = [('close-pair', 'pair', 'min pair separation (Hz)'),
             ('relation',   'relg', 'min lattice-relation gap (Hz)'),
             ('weak-prism', 'axm',  'min |wedge tilt| (deg)')]
    cls = A['cls']
    uns = np.array([len(c) > 0 for c in cls])
    out(f"\n  {'class':<12}{'margin':<34}{'cited':>9}"
        f"{'med(cited)':>12}{'med(all)':>12}{'ratio':>9}")
    for name, key, label in pairs:
        v = A[key]
        sel = np.array([name in c for c in cls]) & np.isfinite(v)
        allf = np.isfinite(v)
        if sel.sum() < 5:
            out(f"  {name:<12}{label:<34}{sel.sum():>9,}"
                f"{'--':>12}{'--':>12}{'--':>9}")
            continue
        mc, ma = np.median(v[sel]), np.median(v[allf])
        out(f"  {name:<12}{label:<34}{sel.sum():>9,}"
            f"{mc:>12.4f}{ma:>12.4f}{ma/max(mc,1e-12):>9.1f}x")
    out(f"\n  ratio >> 1 means the class fires exactly where the true margin"
        f" is small\n  (i.e. the certificate is diagnosing the right cause);"
        f" ratio ~ 1 is a red flag.")


# ------------------------------------------------------------------ A6
def a6(A, out):
    out(f"\n{'='*72}\nA6  THE 'basin' POPULATION -- solver, not identifiability"
        f"\n{'='*72}")
    cls = A['cls']
    basin = np.array([c == {'basin'} for c in cls])
    anyf  = np.array([len(c) > 0 for c in cls])
    nb = int(basin.sum())
    out(f"\n  cases whose ONLY certificate is 'basin': {nb:,}")
    if not nb:
        return
    out(f"  = {100*nb/max(int(anyf.sum()),1):.2f}% of failures, "
        f"{100*nb/A['n']:.4f}% of all cases")
    out(f"\n  the certificate states the spectral stage succeeded, so these"
        f"\n  are recoverable in principle and lost to TRF initialisation.\n")
    out(f"  {'margin':<24}{'basin-only':>14}{'all cases':>14}{'ratio':>9}")
    for key, label in (('pair', 'min pair sep (Hz)'),
                       ('relg', 'min relation gap (Hz)'),
                       ('axm',  'min |wedge tilt| (deg)'),
                       ('cyc',  'min cycles at T_obs')):
        v = A[key]
        s = basin & np.isfinite(v)
        if s.sum() < 5:
            continue
        mb, ma = np.median(v[s]), np.median(v[np.isfinite(v)])
        out(f"  {label:<24}{mb:>14.4f}{ma:>14.4f}{mb/max(ma,1e-12):>9.2f}x")
    out(f"\n  ratios near 1.0 across the board => basin failures are NOT"
        f" concentrated in\n  a hard corner of parameter space; they are"
        f" scattered, which points at the\n  optimiser rather than the"
        f" problem geometry.")


def _save(fig, figdir, stem, out):
    os.makedirs(figdir, exist_ok=True)
    for ext in ('pdf', 'png'):
        p = os.path.join(figdir, f'{stem}.{ext}')
        fig.savefig(p, dpi=200, bbox_inches='tight')
    out(f"\n  -> {os.path.join(figdir, stem)}.{{pdf,png}}")
    import matplotlib.pyplot as plt
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='src',
                    default='experiments/results/sweep_adaptive_ready_0000000.jsonl')
    ap.add_argument('--figdir', default=os.path.join('paper', 'figures'))
    ap.add_argument('--no-figs', action='store_true')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out', default='experiments/results/analysis.txt')
    a = ap.parse_args()
    if not os.path.exists(a.src):
        sys.exit(f"missing {a.src} -- run finalize_dataset.py first")

    lines = []
    def out(s=''):
        print(s); lines.append(s)

    print(f"ground truth: {_verify_case_at()}")
    print(f"streaming {a.src} ...")
    A = load(a.src, a.limit or None)
    out(f"\nloaded {A['n']:,} cases")
    figdir = None if a.no_figs else a.figdir

    a1(A, out)
    a2(A, out, figdir)
    a3(A, out, figdir)
    a4(A, out, figdir)
    a5(A, out)
    a6(A, out)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"\n\nreport saved -> {a.out}")


if __name__ == '__main__':
    main()
