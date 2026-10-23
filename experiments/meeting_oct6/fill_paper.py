#!/usr/bin/env python3
"""Fill paper placeholders E1-E4 from data that already exists.

Writes (main.tex is NOT modified):
  paper/campaign_E1_E4.tex       the filled tables/figure environments
  paper/main_campaign.tex        copy of main.tex with the E1-E4 PHBLOCKs
                                 replaced by \\input{campaign_E1_E4_parts/...}
  paper/campaign_E1_E4_parts/*.tex
  paper/figures/E1_atlas.pdf, E2_calibration.pdf, E3_sigmaN.pdf,
  E4_prescription.pdf (+ .png)

Data: the 669,574-case adaptive campaign (compacted to campaign.npz), plus
the local calibration run (local_campaign.py --exp calib) for E2.
Honest scope notes are written into the captions.
"""
import os, sys, re, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
from vizstyle import setup, C, INK2, wilson, plt
from risley_lattice.model import NAMES

PAPER = os.path.join(ROOT, 'paper')
PARTS = os.path.join(PAPER, 'campaign_E1_E4_parts')
FIG = os.path.join(PAPER, 'figures')
os.makedirs(PARTS, exist_ok=True)
m = np.load(os.path.expanduser(os.environ.get('CAMP', '~/work/camp/campaign.npz')))
CERTS = ['relation', 'weak-prism', 'glitch-floor', 'close-pair',
         'rank-deficient', 'basin', 'no-generators']
n = len(m['case'])
Ts = m['T_solved']; solved = np.isfinite(Ts); uns = ~solved
bits = m['cert_bits']
summary = {}


def pct(k, N):
    p, lo, hi = wilson(k, N)
    return f'{100*p:.2f}\\,\\% & [{100*lo:.2f}, {100*hi:.2f}]'


def fmt(k):
    return f'{int(k):,}'.replace(',', '{,}')


# ------------------------------------------------------------------ E1 table
rows = []
cum = 0
for T in (10., 20., 40., 80.):
    k = int(np.sum(Ts == T)); cum += k
    lab = 'recovered at $T{=}10$\\,s' if T == 10 else f'cured at $T{{=}}{int(T)}$\\,s'
    rows.append(f'{lab} & {fmt(k)} & {pct(k, n)} \\\\')
rows.append('\\midrule')
rows.append(f'\\textbf{{recovered, adaptive}} & {fmt(cum)} & {pct(cum, n)} \\\\')
nu = int(uns.sum())
rows.append(f'unsolved at $T\\le80$\\,s & {fmt(nu)} & {pct(nu, n)} \\\\')
rows.append('\\midrule')
rows.append('\\multicolumn{4}{l}{\\emph{certificate classes on the unsolved '
            '(classes co-occur; share of unsolved)}}\\\\')
names = {'relation': 'lattice relation', 'weak-prism': 'weak prism',
         'glitch-floor': 'glitch floor (TIR mask / residual)',
         'close-pair': 'close pair', 'rank-deficient': 'rank-deficient',
         'basin': 'basin (spectral stage certified)'}
for i, c in enumerate(CERTS[:6]):
    k = int(np.sum(((bits >> i) & 1)[uns] == 1))
    rows.append(f'\\quad {names[c]} & {fmt(k)} & {pct(k, nu)} \\\\')
unexpl = int(np.sum(bits[uns] == 0))
rows.append(f'\\quad \\textbf{{no certificate (unexplained)}} & {fmt(unexpl)} & {pct(unexpl, nu)} \\\\')
summary['E1'] = dict(n=n, solved=int(cum), unsolved=nu, unexplained=unexpl)

truth = m['truth']; x18 = m['x18']
err = np.abs(x18 - truth)
emax = np.nanmax(err, 1)
ok = solved & np.isfinite(emax)
med_err = float(np.median(emax[ok]))
med_t = float(np.median(m['secs_total']))
tirc = m['tir_margin'] < 0
k_tir = int(tirc.sum()); k_tir_s = int(solved[tirc].sum())
summary['E1'].update(median_err=med_err, median_secs=med_t,
                     tir_crossing=k_tir, tir_recovered=k_tir_s)
e1_table = r"""\begin{table*}[t]
\caption{E1: adaptive recovery over the full campaign of %s configurations
($P{=}3$, uniform box, noiseless, 200 samples per 10\,s). Wilson 95\,\%%
intervals. Median maximal parameter error on recoveries %.1e; median wall
time per configuration (all rungs) %.1f\,s. Configurations whose worst-case
total-internal-reflection margin is negative (%s of the box) are recovered at
%s.}
\label{tab:e1}
\centering\small
\begin{tabular}{lrrl}
\toprule
 & count & share & 95\,\%% CI\\
\midrule
%s
\bottomrule
\end{tabular}
\end{table*}
""" % (fmt(n), med_err, med_t, f'{100*k_tir/n:.2f}\\,\\%',
       pct(k_tir_s, k_tir).replace(' & ', ' '), '\n'.join(rows))

# ------------------------------------------------------------------ E1 figure
setup()
s10 = Ts == 10
fig, axs = plt.subplots(1, 3, figsize=(9.2, 2.8), sharey=True)
for ax, key, lab, lo, hi in [
        (axs[0], 'pair_sep', r'$g_{\rm pair}$ (Hz)', 1e-4, 7),
        (axs[1], 'rel_gap', r'$g_{\rm rel}$ (Hz)', 1e-5, 2),
        (axs[2], 'ax_min', r'$\min_i|\alpha_{x,i}|$ (deg)', 1e-3, 18)]:
    x = m[key]; e = np.logspace(np.log10(lo), np.log10(hi), 26)
    b = np.digitize(x, e) - 1
    xs, ps, l, h = [], [], [], []
    for j in range(len(e) - 1):
        sel = b == j
        if sel.sum() < 20:
            continue
        p, a, c = wilson(s10[sel].sum(), sel.sum())
        xs.append(np.sqrt(e[j] * e[j + 1])); ps.append(100 * p)
        l.append(100 * (p - a)); h.append(100 * (c - p))
    ax.errorbar(xs, ps, yerr=[l, h], color=C[0], marker='o', ms=3, lw=1.5,
                elinewidth=1)
    ax.set_xscale('log'); ax.set_xlabel(lab)
    if key != 'ax_min':
        ax.axvline(0.012, color=INK2, ls='--', lw=1)
        ax.text(0.013, 8, r'$C_{\rm m}/T$', color=INK2, fontsize=8)
axs[0].set_ylabel('recovered at T = 10 s (%)')
axs[0].set_ylim(0, 102)
fig.suptitle(f'Identifiability atlas, {n:,} configurations', fontsize=9)
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(FIG, f'E1_atlas.{ext}'))
plt.close(fig)
e1_fig = r"""\begin{figure*}[t]
\centering\includegraphics[width=\textwidth]{figures/E1_atlas.pdf}
\caption{The identifiability atlas: recovery probability at $T{=}10$\,s
against the three margin coordinates (Wilson 95\,\%% bars, bins with
$\ge$20 configurations; %s configurations). Dashed: the merge floor
$C_{\mathrm m}/T = 0.012$\,Hz. The derived detectability overlay
$\alpha_{\min}(T)$ is configuration-dependent and is not drawn.}
\label{fig:atlas}
\end{figure*}
""" % fmt(n)

# ------------------------------------------------------------------ E2
cal_path = os.path.join(HERE, 'out', 'calib_rows.json')
if os.path.exists(cal_path):
    cr = json.load(open(cal_path))
    cov = [r['bounds_cover'] for r in cr if r.get('bounds_cover') is not None]
    tig = [r['tightness_med'] for r in cr if r.get('tightness_med')]
    kc, nc = int(np.sum(cov)), len(cov)
    pc = pct(kc, nc)
    e2_table = r"""\begin{table*}[t]
\caption{E2 (development scale): certified bounds on %d fresh recovered
configurations (seed 20261008, $T{=}10$\,s), every parameter checked
against truth. The $10^{7}$-check campaign version of this table is still
pending; this run fixes the protocol and gives a first coverage figure.}
\label{tab:e2}
\centering\small
\begin{tabular}{lrrl}
\toprule
 & count & share & 95\,\%%~CI\\
\midrule
configurations with all 18 bounds covering & %s & %s\\
median tightness (bound / true error), median over cases & \multicolumn{3}{l}{%.1f}\\
\bottomrule
\end{tabular}
\end{table*}
""" % (nc, f'{kc}/{nc}', pc, float(np.median(tig)) if tig else float('nan'))
    xs = [r['err'] for r in cr if r.get('bound_max')]
    ys = [r['bound_max'] for r in cr if r.get('bound_max')]
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.loglog(xs, ys, 'o', ms=3.5, color=C[0], alpha=0.8)
    lo_, hi_ = min(xs + ys) / 2, max(xs + ys) * 2
    ax.plot([lo_, hi_], [lo_, hi_], color=INK2, lw=1)
    ax.set_xlabel('true max parameter error'); ax.set_ylabel('certified max bound')
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIG, f'E2_calibration.{ext}'))
    plt.close(fig)
    e2_fig = r"""\begin{figure}[t]
\centering\includegraphics[width=0.75\columnwidth]{figures/E2_calibration.pdf}
\caption{Certified maximal bound against true maximal error, %d fresh
recoveries (local development-scale run). Coverage = mass above the
identity line.}
\label{fig:cert}
\end{figure}
""" % nc
    summary['E2'] = dict(n=nc, covered=kc, tightness_median=float(np.median(tig)) if tig else None)
else:
    e2_table = e2_fig = None

# ------------------------------------------------------------------ E3 (sigma_N law only)
always = uns & np.all([np.isfinite(m[f'sigma_N{t}']) for t in (10, 20, 40, 80)], 0)
Tg = np.array([10, 20, 40, 80.])
med = np.array([np.median(m[f'sigma_N{int(t)}'][always]) for t in Tg])
rng = np.random.default_rng(0)
idx = np.where(always)[0]
boots = []
for _ in range(400):
    bsel = rng.choice(idx, len(idx))
    mm = [np.median(m[f'sigma_N{int(t)}'][bsel]) for t in Tg]
    boots.append(np.polyfit(np.log(Tg), np.log(mm), 1)[0])
slope = np.polyfit(np.log(Tg), np.log(med), 1)[0]
lo3, hi3 = np.percentile(boots, [2.5, 97.5])
s2 = np.polyfit(np.log(Tg[1:]), np.log(med[1:]), 1)[0]
summary['E3'] = dict(n=int(always.sum()), slope=float(slope), ci=[float(lo3), float(hi3)],
                     slope_20_80=float(s2), medians=med.tolist())
fig, ax = plt.subplots(figsize=(3.6, 3.0))
ax.loglog(Tg, med, 'o-', color=C[0], label='median measured')
ax.loglog(Tg, med[0] * (Tg / 10) ** -1.5, '--', color=INK2, lw=1, label=r'$T^{-3/2}$ guide')
ax.set_xlabel('T (s)'); ax.set_ylabel(r'$\sigma(\hat N)$ (Hz)'); ax.legend(fontsize=8)
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(FIG, f'E3_sigmaN.{ext}'))
plt.close(fig)
e3 = r"""\begin{table*}[t]
\caption{E3, frequency-error law only. Population: the %s configurations
unsolved at every rung, so the same cases appear at each $T$ (an
adaptive-campaign cohort, not the fixed-$T$ design; the pair-gap and
detectability laws still need the fixed-$T$ sweep).}
\label{tab:e3}
\centering\small
\begin{tabular}{lccl}
\toprule
law & derived & fitted (bootstrap 95\,\%%) & note\\
\midrule
$\sigma(\hat N)\propto T^{\gamma}$ & $-3/2$ & $%.2f$ [$%.2f$, $%.2f$] &
$T{=}20$--$80$ only: $%.2f$\\
pair-gap inflation & $-2$ & pending & needs fixed-$T$ cohorts\\
detectability & $-1/2$ & pending & needs fixed-$T$ cohorts\\
\bottomrule
\end{tabular}
\end{table*}
\begin{figure}[t]
\centering\includegraphics[width=0.75\columnwidth]{figures/E3_sigmaN.pdf}
\caption{Median certified $\sigma(\hat N)$ against $T$ for the always-unsolved
cohort, with the derived $T^{-3/2}$ guide.}
\label{fig:e3}
\end{figure}
""" % (fmt(always.sum()), slope, lo3, hi3, s2)

# ------------------------------------------------------------------ E4 (ladder-censored)
tq = m['treq10']; res = m['rescue_T']
sel = np.isfinite(tq) & np.isfinite(res) & (tq < 9999)
ratio = tq[sel] / res[sel]
ladder = np.array([10, 20, 40, 80.])
nxt = np.array([ladder[np.searchsorted(ladder, t)] if t <= 80 else np.inf for t in tq[sel]])
consistent = res[sel] <= nxt
fail80 = uns & np.isfinite(m['T_req'])
pred_inf = int(np.sum(m['T_req'][fail80] > 80))
summary['E4'] = dict(n=int(sel.sum()), ratio_q=np.percentile(ratio, [5, 25, 50, 75, 95]).tolist(),
                     consistent=int(consistent.sum()), unsolved_with_treq=int(fail80.sum()),
                     predicted_infeasible=pred_inf)
fig, ax = plt.subplots(figsize=(3.8, 3.2))
jit = res[sel] * np.exp(np.random.default_rng(1).normal(0, 0.06, sel.sum()))
ax.loglog(tq[sel], jit, '.', ms=1.2, color=C[0], alpha=0.25)
ax.plot([5, 9999], [5, 9999], color=INK2, lw=1)
ax.set_xlim(5, 1e4); ax.set_ylim(12, 110)
ax.set_xlabel(r'prescribed $T_{\rm req}$ at the failed $T{=}10$ s trial (s)')
ax.set_ylabel(r'$T$ at which recovered (s, jittered)')
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(FIG, f'E4_prescription.{ext}'))
plt.close(fig)
q = summary['E4']['ratio_q']
e4 = r"""\begin{table*}[t]
\caption{E4, ladder-censored version: %s configurations failed at
$T{=}10$\,s with a numeric prescription and were later recovered on the
ladder. $T_{\min}$ is only known to lie in (previous rung, recovering rung];
the bisection design is still pending. Among the %s configurations
unsolved at 80\,s with a numeric prescription, %s (%s) were prescribed
$T_{\mathrm{req}}>80$\,s, i.e.\ correctly predicted infeasible within the
ladder.}
\label{tab:e4}
\centering\small
\begin{tabular}{lr}
\toprule
recovered no later than the first rung $\ge T_{\mathrm{req}}$ & %s\\
$T_{\mathrm{req}}/T_{\mathrm{recovered}}$: p5 / p25 / p50 / p75 / p95 &
%.2f / %.2f / %.2f / %.2f / %.2f\\
\bottomrule
\end{tabular}
\end{table*}
\begin{figure}[t]
\centering\includegraphics[width=0.75\columnwidth]{figures/E4_prescription.pdf}
\caption{Prescribed $T_{\mathrm{req}}$ against the rung that recovered the
configuration (vertical jitter for visibility); identity line drawn.}
\label{fig:e4}
\end{figure}
""" % (fmt(sel.sum()), fmt(fail80.sum()), fmt(pred_inf),
       f'{100*pred_inf/max(fail80.sum(),1):.1f}\\,\\%',
       pct(consistent.sum(), sel.sum()).replace(' & ', ' '), *q)

parts = {'E1_table': e1_table, 'E1_figure': e1_fig, 'E2_table': e2_table,
         'E2_figure': e2_fig, 'E3': e3, 'E4': e4}
for k, v in parts.items():
    if v:
        open(os.path.join(PARTS, f'{k}.tex'), 'w').write(v)
open(os.path.join(PAPER, 'campaign_E1_E4.tex'), 'w').write(
    '%% Generated by experiments/meeting_oct6/fill_paper.py\n' +
    '\n'.join(v for v in parts.values() if v))

# main_campaign.tex: replace the six PHBLOCK groups
tex = open(os.path.join(PAPER, 'main.tex')).read()


def block_span(s, title_start):
    i = s.index('\\PHBLOCK{' + title_start)
    j = i + len('\\PHBLOCK')
    depth = 0; groups = 0
    while groups < 2:
        ch = s[j]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                groups += 1
        j += 1
    return i, j


def replace_env(s, title, repl):
    i, j = block_span(s, title)
    b = s.rfind('\\begin{figure', 0, i)
    if b != -1 and s.find('\\end{figure', b) > j:
        e = s.index('\\end{figure', j); e = s.index('\n', e) + 1
        if '\n\n' not in s[b:i]:
            return s[:b] + repl + s[e:]
    return s[:i] + repl + s[j:]


mapping = [('Table E1', 'E1_table'), ('Figure E1', 'E1_figure'),
           ('Table E2', 'E2_table'), ('Figure E2', 'E2_figure'),
           ('Figure/Table E3', 'E3'), ('Figure/Table E4', 'E4')]
for title, part in mapping:
    if parts[part]:
        tex = replace_env(tex, title, f'\\input{{campaign_E1_E4_parts/{part}}}\n')
open(os.path.join(PAPER, 'main_campaign.tex'), 'w').write(tex)
json.dump(summary, open(os.path.join(HERE, 'out', 'fill_paper_summary.json'), 'w'), indent=1)
print(json.dumps(summary, indent=1))
