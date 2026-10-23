#!/usr/bin/env python3
"""Build risley_prism_analysis_v2.docx: the September B1-B7 report plus
the follow-ups to the Oct 6 meeting (C1-C6). The original file is only
read; formatting is cloned from its own paragraphs and tables."""
import os, json, copy
import numpy as np
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
BEN = os.path.dirname(REPO)
SRC = os.path.join(BEN, 'risley_prism_analysis.docx')
DST = os.path.join(BEN, 'risley_prism_analysis_v2.docx')
OUT = os.path.join(HERE, 'out'); FIG = os.path.join(HERE, 'figures')
J = lambda f: json.load(open(os.path.join(OUT, f)))
tir = J('tir_campaign.json'); loc = J('local_summary.json')
fp = J('fill_paper_summary.json'); ov = J('atlas_overlap.json')
ssr = loc['sign_sweep']; ssp = loc['sign_sweep_population']
nu = loc['nonunique']; aw = J('mse_accepted_wrong.json')
reg = J('regress_solve18.json') if os.path.exists(os.path.join(OUT, 'regress_solve18.json')) else None

d = docx.Document(SRC)
P = d.paragraphs
TPL = dict(h1=P[4], h2=P[10], body=P[11], bullet=P[18], cap=P[14])
SYN = next(p for p in P if p.text.strip() == '9. Synthesis')
state = {'anchor': SYN}


def c(n, dg=0):
    return f'{n:,.{dg}f}'


def pc(x, dg=1):
    return f'{100*x:.{dg}f}%'


def wil(k, n, z=1.96):
    p = k / n; den = 1 + z*z/n; ce = (p + z*z/(2*n))/den
    h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    return f'{100*p:.1f}% [{100*(ce-h):.1f}, {100*(ce+h):.1f}]'


def clone(tpl, text, bold_prefix=None):
    el = copy.deepcopy(tpl._p)
    for dr in list(el.iter(qn('w:drawing'))):
        dr.getparent().remove(dr)
    rs = el.findall(qn('w:r'))
    for r in rs[1:]:
        el.remove(r)
    if not rs:
        r0 = OxmlElement('w:r'); el.append(r0)
    else:
        r0 = rs[0]
    for t in list(r0):
        if t.tag != qn('w:rPr'):
            r0.remove(t)
    if bold_prefix:
        rb = copy.deepcopy(r0)
        rpr = rb.find(qn('w:rPr'))
        if rpr is None:
            rpr = OxmlElement('w:rPr'); rb.insert(0, rpr)
        for _b in rpr.findall(qn('w:b')) + rpr.findall(qn('w:bCs')):
            rpr.remove(_b)
        rpr.insert(0, OxmlElement('w:b'))
        tb = OxmlElement('w:t'); tb.text = bold_prefix
        tb.set(qn('xml:space'), 'preserve'); rb.append(tb)
        r0.addprevious(rb)
    t = OxmlElement('w:t'); t.text = text; t.set(qn('xml:space'), 'preserve')
    r0.append(t)
    state['anchor']._p.addprevious(el)
    return el


def H1(t): clone(TPL['h1'], t)
def H2(t): clone(TPL['h2'], t)
def B(t, bp=None): clone(TPL['body'], t, bold_prefix=bp)
def L(t, bp=None): clone(TPL['bullet'], t, bold_prefix=bp)
def CAP(t): clone(TPL['cap'], t)
def EMPTY(): clone(TPL['body'], '')


def MONO(lines):
    for line in lines:
        el = clone(TPL['body'], line)
        for r in el.findall(qn('w:r')):
            rpr = r.find(qn('w:rPr'))
            if rpr is None:
                rpr = OxmlElement('w:rPr'); r.insert(0, rpr)
            f = OxmlElement('w:rFonts')
            for a in ('w:ascii', 'w:hAnsi', 'w:cs'):
                f.set(qn(a), 'Consolas')
            rpr.insert(0, f)
        ppr = el.find(qn('w:pPr'))
        if ppr is None:
            ppr = OxmlElement('w:pPr'); el.insert(0, ppr)
        ind = OxmlElement('w:ind'); ind.set(qn('w:left'), '720'); ppr.append(ind)
        sp = OxmlElement('w:spacing'); sp.set(qn('w:after'), '0'); ppr.append(sp)


def FIGURE(path, width=6.2):
    p = d.add_paragraph()
    p.alignment = 1
    p.add_run().add_picture(path, width=Inches(width))
    state['anchor']._p.addprevious(p._p)


def TABLE(header, rows, widths=None):
    t = d.add_table(rows=1 + len(rows), cols=len(header))
    tbl = t._tbl
    borders = OxmlElement('w:tblBorders')
    for side in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
        e = OxmlElement(f'w:{side}'); e.set(qn('w:val'), 'single')
        e.set(qn('w:sz'), '4'); e.set(qn('w:color'), 'auto'); borders.append(e)
    tbl.tblPr.append(borders)
    widths = widths or [9240 // len(header)] * len(header)
    for ri, vals in enumerate([header] + rows):
        for ci, v in enumerate(vals):
            cell = t.rows[ri].cells[ci]
            cell.width = Inches(widths[ci] / 1440)
            tcPr = cell._tc.get_or_add_tcPr()
            shd = OxmlElement('w:shd'); shd.set(qn('w:val'), 'clear')
            shd.set(qn('w:color'), 'auto')
            shd.set(qn('w:fill'), '1A3C6B' if ri == 0 else ('F5F7FA' if ri % 2 else 'FFFFFF'))
            tcPr.append(shd)
            para = cell.paragraphs[0]; para.alignment = 1 if ci else 0
            run = para.add_run(str(v)); run.font.size = Pt(10)
            if ri == 0:
                run.bold = True; run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    t.rows[0]._tr.get_or_add_trPr().append(OxmlElement('w:tblHeader'))
    state['anchor']._p.addprevious(tbl)
    EMPTY()


def replace_in_runs(p, old, new):
    for r in p.runs:
        if old in r.text:
            r.text = r.text.replace(old, new)
            return True
    if old in p.text:                       # spans runs: collapse into first
        full = p.text.replace(old, new)
        for i, r in enumerate(p.runs):
            r.text = full if i == 0 else ''
        return True
    return False


# --------------------------------------------------------------- header edits
replace_in_runs(P[1], P[1].text, 'Large-Scale Analysis (B1–B7) and Follow-ups to the Oct 6 Meeting (C1–C6)')
replace_in_runs(P[2], P[2].text, f'{c(ov["unique"])} distinct configurations ({c(ov["rows"])} runs across '
                'atlas2–atlas5)  ·  updated October 2026')

# correction note at the end of Section 1
state['anchor'] = next(p for p in P if p.text.startswith('2. B1'))
B(f'All four atlases were generated with the same base seed (424242), so case indices overlap: '
  f'{c(ov["multi"])} configurations were run in more than one atlas. The {c(ov["rows"])} runs therefore hold '
  f'{c(ov["unique"])} distinct configurations ({c(669574)} after dropping 15 rows with an incomplete schema; '
  'that deduplicated set is what C1–C6 use). The counts in B1–B7 are per run, so repeated configurations are '
  f'counted more than once. The overlap also works as a free reproducibility test: {c(ov["identical_outcome"])} of the '
  f'{c(ov["multi"])} repeated configurations ({pc(ov["identical_outcome"]/ov["multi"], 2)}) reached the same outcome '
  f'in every run, and the other {ov["multi"]-ov["identical_outcome"]} differ only in the observation time at which '
  'they were solved.', bp='Correction (October 2026). ')
EMPTY()
state['anchor'] = SYN

# glitch-floor description fixes
for p in P:
    replace_in_runs(p, 'too few cycles observed (glitch-floor)',
                    'TIR-masked samples or a high lattice-fit residual (glitch-floor)')
    replace_in_runs(p, 'the spectral SNR often falls below the threshold as well',
                    'the lattice fit is also more likely to leave a residual floor, the second trigger of this '
                    'class (see C1)')
    replace_in_runs(p, 'Generated: September 2026', 'Generated: September 2026, updated October 2026')
for t in d.tables:
    for row in t.rows:
        for cell in row.cells:
            for p in cell.paragraphs:
                replace_in_runs(p, 'Insufficient cycles at T_obs', 'TIR mask or residual floor')
replace_in_runs(SYN, '9. Synthesis', '15. Synthesis')

# ================================================================== C1
H1('9. C1 — Total Internal Reflection: an Exact Bound')
H2('Motivation')
B('In the Oct 6 meeting, Joseph pointed out that a bound that excludes total internal reflection (TIR) cannot depend on '
  'the wedge angle alone. It has to involve the incident angle, the ratio of refractive indices and the wedge angle '
  'together. This section derives that bound exactly from the forward model, checks it against the code, and measures '
  'what it explains in the campaign.')
H2('Derivation')
B('The forward model (reverse_problem_v2/core.py) traces each axis in its own plane. At a face with tilt φ and incoming '
  'ray angle θ, the code’s quantity cy equals −sin(θ + φ), so the angle of incidence on the face is θ + φ. TIR occurs '
  'exactly when (n_in / n_out)·|sin(θ + φ)| > 1, which only glass-to-air faces can satisfy. Each prism has a flat entry '
  'face and a tilted exit face whose projected tilt is φ_i(t) = arcsin(cos γ_i(t) · sin ax_i), so |φ_i| ≤ |ax_i|, with '
  'equality twice per revolution. The refracted direction increases with both the incoming angle and the tilt (its tilt '
  'derivative is n·cos x / cos x_out − 1 ≥ 0 when n ≥ 1). The largest angle reaching each prism therefore comes from '
  'lining up every upstream extreme, which gives the worst case over all rotation phases in closed form:')
MONO(['s₁ = |beam angle|',
      'u_i = arcsin( sin(s_i) / n_i )',
      'margin_i = arcsin(1/n_i) − (u_i + |ax_i|)',
      's_(i+1) = arcsin( n_i · sin(u_i + |ax_i|) ) − |ax_i|'])
B('A configuration is TIR-free at every rotation phase if and only if every margin is positive in both planes. The three '
  'inputs are the ones named in the meeting: incident angle (s_i), index ratio (n_i) and wedge angle (|ax_i|). Three checks '
  'confirm it. The closed-form refraction matches the code to 1×10⁻¹⁰ degrees on 10,000 random faces. No configuration '
  'with a positive margin showed TIR in 300 random configurations traced for 80 s. And 30 of 31 configurations with a '
  'negative margin did hit TIR within 320 s (with incommensurate speeds the phases eventually line up). Code: '
  'experiments/meeting_oct6/tir.py.')
H2('In Plain Language: Where the 9.30° Comes From')
B('This subsection explains the same result without assuming any background in optics.')
B('Every transparent material has a refractive index, which measures how much it slows light down. Air is 1, '
  'ordinary window glass is about 1.5, and the densest glass allowed in the simulations is 1.8. Light bends whenever it '
  'crosses from one material into another. When it leaves glass into air it bends to a steeper angle than it had inside: '
  'light that hits the inside of the surface straight on (0°) leaves straight, light at 20° leaves at about 38°, and light '
  'at 30° leaves at about 64°. It can never leave steeper than 90°, which means skimming flat along the surface. The '
  'critical angle is the inside angle that would make it leave at exactly 90°. Any steeper than that and there is no '
  'angle left for it to exit at, so it reflects back into the glass: total internal reflection.',
  bp='Why there is a limit. ')
B('The bending rule (Snell’s law) is: glass index × sin(angle inside) = sin(angle outside). At the critical point the '
  'outside angle is 90° and sin(90°) = 1, so for the densest glass:', bp='Calculating the limit. ')
MONO(['1.8 × sin(angle inside) = 1',
      'sin(angle inside) = 1 ÷ 1.8 = 0.556',
      'angle inside = 33.75°'])
B('Denser glass traps light more easily, so the limit depends on the glass:')
TABLE(['Glass index', 'Critical angle (the limit)'],
      [[f'{n} ({lab})', f'{np.degrees(np.arcsin(1/n)):.2f}°'] for n, lab in
       [(1.3, 'lightest allowed'), (1.5, 'ordinary glass'), (1.8, 'densest allowed')]], [4620, 4620])
B('The 9.30° figure is the steepest wedge that stays safe even when everything is at its worst at the same moment: '
  'the densest glass (1.8, limit 33.75°), the laser entering at the largest tilt allowed (25°), and all three wedges '
  'leaning the same way during their spin, so each one steepens the beam instead of cancelling another. At each wedge '
  'three things happen. The light enters the flat front face and bends to a shallower angle. It reaches the back face, '
  'which is tilted by the wedge angle, so the tilt adds directly to how steeply it hits that face. If that is below '
  '33.75° it escapes, bending steeper on the way out, and that steeper angle is what reaches the next wedge. Each wedge '
  'therefore receives steeper light than the one before, and the third wedge is the first to trap it.',
  bp='The worst case. ')
B('9.30° was found by guess-and-check: trace the worst-case beam through all three wedges for a trial wedge angle, raise '
  'the angle if the light escapes everywhere and lower it if it gets trapped, and narrow in on the tipping point. The '
  'table shows the angle at which the light hits each wedge’s back face; it must stay below 33.75° to escape.',
  bp='Finding the tipping point. ')
_rows = []
for _a in (5, 8, 9, 9.296, 9.5, 10):
    _s, _hits, _trap = 25.0, [], False
    for _i in range(3):
        _u = np.degrees(np.arcsin(np.sin(np.radians(_s)) / 1.8)); _h = _u + _a; _hits.append(_h)
        _x = 1.8 * np.sin(np.radians(_h))
        if _x >= 1:
            _trap = True; break
        _s = np.degrees(np.arcsin(_x)) - _a
    _hits += [None] * (3 - len(_hits))
    _verdict = ('Right at the limit' if abs(_hits[2] - 33.75) < 0.1 else
                ('Yes' if _trap else ('No, but close' if _hits[2] > 32.5 else 'No')))
    _rows.append([f'{round(_a, 1):g}°'] + [f'{h:.2f}°' if abs(_a - 9.296) < 1e-6 and h is _hits[2] else f'{h:.1f}°' for h in _hits] + [_verdict])
TABLE(['Wedge angle', '1st wedge hit', '2nd wedge hit', '3rd wedge hit', 'Trapped?'], _rows,
      [1700, 1800, 1800, 1800, 2140])
CAP('Table C1a. Worst case: glass index 1.8, beam entering at 25°, all three wedges leaning the same way.')
B('At 9° the third wedge is hit at 33.0°, just under the limit; at 9.5° it is 34.3°, just over. The crossing point is '
  '9.30°. Strictly, 10° is therefore not guaranteed safe, but only when every condition is at its extreme at once: none '
  'of the 114,554 campaign configurations with wedges up to 10° could actually reach total internal reflection. If the '
  'beam enters straight (0° tilt) instead of at 25°, the same check gives about 16.7°, and lighter glass raises the limit '
  'further, so Ben’s real hardware numbers would give a specific safe angle for his setup.')
H2('Results')
FIGURE(os.path.join(FIG, 'T1_tir_bound.png'))
CAP('Figure C1a. Left: the largest TIR-free wedge for a single prism, as a function of incident angle and glass index. '
    'Right: the largest common wedge for identical prisms at the worst phase alignment.')
rows = []
for k, lab in [('full box (|ax|<=18)', 'Current box, |ax| ≤ 18°'), ('TIR-prone (margin<0)', 'TIR reachable (margin < 0)'),
               ('TIR-free (margin>0)', 'TIR-free (margin > 0)'), ('|ax|<=14 box', '|ax| ≤ 14° box'),
               ('|ax|<=12 box', '|ax| ≤ 12° box'), ('|ax|<=10 box', '|ax| ≤ 10° box'),
               ('|ax|<=9.296 box', '|ax| ≤ 9.30° (guaranteed TIR-free)')]:
    r = tir[k]
    rows.append([lab, c(r['n']), pc(r['rate'], 2), pc(r['rate_T10'], 1), c(r['unsolved']),
                 pc(r['glitch_share_of_failures'], 0)])
TABLE(['Sub-population', 'Cases', 'Recovered (adaptive)', 'Recovered at T = 10 s', 'Unsolved', 'Glitch-floor in failures'],
      rows, [2700, 1100, 1400, 1400, 1100, 1540])
CAP('Table C1. Sub-populations of the 669,574-case campaign. A uniform box restricted to smaller wedges is itself a '
    'uniform sample of the smaller box, so each row is a restricted campaign obtained without a re-run.')
FIGURE(os.path.join(FIG, 'T2_failure_vs_margin.png'), 4.6)
CAP('Figure C1b. Share unsolved at T = 80 s by worst-case TIR margin, with Wilson 95% intervals. Orange bins can reach TIR.')
H2('Key Findings')
tp = tir['TIR-prone (margin<0)']; fb = tir['full box (|ax|<=18)']; b10 = tir['|ax|<=10 box']
L(f'{pc(tir["frac_tir_box18"],1)} of the current ±18° box can reach TIR. Those configurations are recovered '
  f'{pc(tp["rate"],1)} of the time, against {pc(tir["TIR-free (margin>0)"]["rate"],1)} for the rest, and they account for '
  f'{c(tp["unsolved"])} of the {c(fb["unsolved"])} failures ({pc(tp["unsolved"]/fb["unsolved"],1)}). The glitch-floor '
  f'certificate fires on {pc(tp["glitch_share_of_failures"],0)} of them.')
L('Every configuration with all wedges at or below 9.30° is TIR-free at every rotation phase, for any glass index in '
  '1.3–1.8 and any beam angle up to 25°, which are the box’s limits. With the beam on-axis the limit is 16.7°, and with '
  'beam angles up to 10° it is 13.6°. The ±10° soft goal from the meeting therefore coincides, to within 0.7°, with the '
  f'TIR-free region: none of the {c(b10["n"])} campaign configurations with |ax| ≤ 10° can reach TIR.')
L(f'Restricting to |ax| ≤ 10° raises adaptive recovery from {pc(fb["rate"],2)} to {pc(b10["rate"],2)} and first-window '
  f'recovery from {pc(fb["rate_T10"],1)} to {pc(b10["rate_T10"],1)}. A fresh local run (150 configurations, new seed) '
  f'agrees: {wil(loc["box10"]["by_T"]["80.0"]["k"], loc["box10"]["n"])} recovered by 80 s and '
  f'{pc(loc["box10"]["by_T"]["10.0"]["rate"],1)} at 10 s.')
L('Inside the TIR-free region, the remaining failures come from information limits (relation, weak prism), not from TIR. '
  'The glitch-floor class still fires on about 22% of them because it has a second trigger: a lattice-fit residual above '
  '0.05 with no TIR at all. Splitting it into separate “tir-mask” and “residual-floor” classes would make the failure '
  'taxonomy cleaner.')
EMPTY()

# ================================================================== C2
bn, b6 = loc['ben'], loc['ben066']
H1('10. C2 — Ben’s Hardware Regime: Sub-Degree Wedges')
H2('Motivation')
B('Ben described the earlier system as a matched pair of 6° or 12° wedges that offsets the beam, followed by '
  'free-spinning wedges under 1° (about 0.66°). Sections B5 and B7 showed that small wedges are where the solver is '
  'weakest, so this is the regime that matters for real hardware.')
H2('Setup')
B('Each configuration has one strong wedge (6° or 12°, random sign) and two free wedges with |ax| drawn from 0.3–1.0°, '
  'or fixed at 0.66° in a second set. Everything else is drawn from the standard box, and the strong wedge’s position in '
  'the speed order is random. The adaptive ladder is extended to 320 s. One limitation: in this three-prism model, one '
  'strong wedge stands in for the matched pair. Modelling the pair as two co-rotating prisms with a locked speed needs a '
  'constrained four-prism solve, which is the natural next step. Code: experiments/meeting_oct6/local_campaign.py.')
rows = []
for k, lab in [('ben', 'Two wedges 0.3–1.0°'), ('ben066', 'Two wedges 0.66°'), ('box10', 'Reference: |ax| ≤ 10° box')]:
    r = loc[k]; bt = r['by_T']
    rows.append([lab, str(r['n'])] + [pc(bt[t]['rate'], 1) if t in bt else '—' for t in ('10.0', '20.0', '40.0', '80.0')]
                + [f'{r["median_err"]:.0e}'])
TABLE(['Configuration', 'n', 'T ≤ 10 s', 'T ≤ 20 s', 'T ≤ 40 s', 'T ≤ 80 s', 'Median error'], rows,
      [2600, 700, 1150, 1150, 1150, 1150, 1340])
CAP('Table C2. Share of configurations recovered by each observation time. In both Ben sets, nothing unsolved at 80 s '
    'was rescued at 160 or 320 s.')
FIGURE(os.path.join(FIG, 'C2_ben_regime.png'))
CAP('Figure C2. Left: recovery against observation time. Right: for each sub-degree wedge, the smallest wedge the spectral '
    'stage can detect at T = 10 s (the 5σ amplitude test from certify.py, evaluated with the true speeds), plotted against '
    'its true size.')
H2('Key Findings')
L(f'Sub-degree wedges are recoverable when one strong wedge is present: {wil(bn["by_T"]["80.0"]["k"], bn["n"])} are '
  f'recovered by 80 s for 0.3–1.0° wedges, and {b6["by_T"]["40.0"]["k"]}/{b6["n"]} by 40 s for 0.66° wedges. For '
  'comparison, in the deduplicated uniform campaign 12.5% of configurations whose smallest wedge is below 0.5° stay '
  'unsolved.')
L(f'At T = 10 s, {pc(bn["frac_066_visible_T10"],0)} of the 0.3–1.0° wedges and {pc(b6["frac_066_visible_T10"],0)} of the '
  f'0.66° wedges are above the spectral detection floor. Falling below the floor slows recovery but rarely prevents '
  f'it: {bn["hidden_cases_solved_T10"]} of {bn["cases_with_hidden_wedge_T10"]} configurations with a hidden wedge were '
  f'solved at 10 s (against {bn["visible_cases_solved_T10"]} of {bn["visible_cases"]} with every wedge visible), and '
  f'{bn["hidden_cases_solved_adaptive"]} of {bn["cases_with_hidden_wedge_T10"]} were solved within the ladder. The '
  'trust-region completion often recovers wedges that the spectrum alone cannot see.')
L(f'By the T^(−1/2) detectability law, a hidden 0.66° wedge needs a median of about '
  f'{b6["T_to_detect_066_hidden_median"]:.0f} s of observation before its spectral line clears the 5σ test, with a long '
  'tail. In practice the adaptive ladder rarely needs that much time.')
L(f'Of the {len(bn["failures"])} failures, two carry a weak-prism certificate on top of an almost exactly equal speed pair, '
  'and two are basin failures. One basin case is instructive: at T = 160 s it reached pattern MSE 2.7×10⁻¹³, below the '
  'acceptance threshold, while its parameters were still 0.024 from the truth. Section C4 shows this is a real but rare '
  'property of weak wedges.')
L('Re-run of the four with the sign-sweep rung on and the full ladder to 320 s (local, Oct 13): configuration 62 is '
  'now solved at T = 10 s by a two-wedge flip. Configuration 90 is a lookalike whose error, almost all in the glass index '
  'of its flattest wedge (0.34°), falls with observation time: 0.033 at 80 s, 0.016 at 160 s, 0.0037 at 320 s. '
  'Configuration 87 is a thin-prism lookalike: the 80 s answer has ax1 = 0.498° with n1 = 1.372 against the '
  'true 0.581° with n1 = 1.319, and (n − 1)·ax is identical (0.1855) for both, so the deviation of that wedge is unchanged. '
  'Warm-started at 160 and 320 s it stays put; with the glass indices fixed to their true values it converges to the '
  'truth (error 9×10⁻⁸). Configuration 16 has a speed pair 0.00024 Hz apart; the '
  'certificate’s merge-floor term (2 × 0.12 / 0.00024) prescribes about 1,000 s, beyond the 320 s tried.',
  bp='Follow-up. ')
EMPTY()

# ================================================================== C3
H1('11. C3 — Sign-Sweep Rung: Larger Local Run, Now in solve18')
H2('What was run')
B(f'The prototype in B7 tested 36 of the 195 G1 cases and 12 of the 1,873 G3 cases. Rebuilt from the deduplicated data, '
  f'the populations are G1 = {ssp["G1"]}, G2 = {ssp["G2"]} and G3 = {c(ssp["G3"])}. This run covers G1 completely '
  f'({ssr["G1"]["tested"]} of {ssp["G1"]}) and a random {ssr["G3"]["tested"]} of G3, using the same 10-candidate ladder '
  'and the same acceptance test. Everything ran locally in Python, with nothing on Modal. Code: '
  'experiments/meeting_oct6/sign_sweep_local.py.')
rows = []
for g, lab in [('G1', 'G1 unsolved, pure sign flip'), ('G3', 'G3 unsolved, other angle error')]:
    r = ssr[g]
    if r['tested']:
        rows.append([lab, c(ssp[g]), str(r['tested']), str(r['rescued']), wil(r['rescued'], r['tested']),
                     str(r['multi_flip_wins']), str(r['wrong_accepted'])])
TABLE(['Group', 'Population', 'Tested', 'Rescued', 'Rate (95% CI)', 'Pair/triple wins', 'Accepted with error ≥ 10⁻³'],
      rows, [2500, 1100, 900, 900, 1600, 1100, 1140])
if os.path.exists(os.path.join(FIG, 'C3_sign_sweep.png')):
    FIGURE(os.path.join(FIG, 'C3_sign_sweep.png'), 4.0)
    CAP('Figure C3. Rescue rate of the sign-sweep rung by group, with Wilson 95% intervals.')
H2('Key Findings')
g1 = ssr['G1']
L(f'On the complete G1 population, the rung rescues {wil(g1["rescued"], g1["tested"])}. The prototype’s 25% '
  'estimate came from 36 cases.')
if ssr['G3']['tested']:
    g3 = ssr['G3']
    L(f'On {g3["tested"]} G3 cases the rate is {wil(g3["rescued"], g3["tested"])}, a much tighter estimate than the '
      'prototype’s, which rested on 12 cases. Scaled to the population, it would recover roughly '
      f'{c(g3["rescued"]/g3["tested"]*ssp["G3"])} of the {c(ssp["G3"])} G3 failures.')
mf = sum(ssr[g]['multi_flip_wins'] for g in ('G1', 'G3'))
rr = sum(ssr[g]['rescued'] for g in ('G1', 'G3'))
L(f'Flipping two or three wedges at once accounts for {mf} of the {rr} rescues. The existing ladder never tries these.')
L('One correction to the prototype’s claim that the rung never accepted a wrong answer: in one G1 case (smallest wedge '
  '0.27°, T = 80 s), it accepted a fit whose maximum parameter error was 3×10⁻³, just above the 10⁻³ success threshold. '
  'The fit was in the right basin but stopped early, because the TRF is capped at 400 evaluations at 80 s. The integrated '
  'version therefore adds one uncapped polish step after acceptance.')
H2('Warm start (October 13)')
_ws = J('warmstart.json'); _b, _o = _ws['groups']['basin'], _ws['groups']['other']
B(f'Each rung of the adaptive ladder currently restarts from scratch. As a test, the stored answer at T_k of an unsolved '
  f'campaign configuration was used as the starting point of a TRF on the T_(k+1) pattern. Basin failures: '
  f'{_b["rescued"]}/{_b["jobs"]} rescued ({wil(_b["rescued"], _b["jobs"])}) but {_b["wrong_accepted"]} accepted '
  f'lookalikes, almost all in the glass index of a sub-0.5° wedge. Other unsolved: {_o["rescued"]}/{_o["jobs"]} '
  f'({wil(_o["rescued"], _o["jobs"])}), none accepted wrong. The previous answer was usually far from the truth (within 1 '
  'in every parameter for only about one trial in eight), which limits what reuse can do. Code: '
  'experiments/meeting_oct6/warmstart.py.')
H2('Glass index known (October 13)')
_gk = J('glassknown.json'); _L, _F = _gk['groups']['look'], _gk['groups']['flat']
B(f'solve18 was run from scratch on {_L["finished"]} of the {_gk["population"]["look"]} lookalike configurations, at the '
  'viewing time where the lookalike first appeared, with the glass indices free and with them fixed at their true values '
  '(bounds and nominal start only; the solver code is unchanged). Lookalikes: '
  f'{_L["glass_unknown"].get("lookalike", 0)} with glass free, {_L["glass_known"].get("lookalike", 0)} with glass fixed; '
  f'correct: {_L["glass_unknown"].get("correct", 0)} and {_L["glass_known"].get("correct", 0)}; unsolved: '
  f'{_L["glass_unknown"].get("unsolved", 0)} and {_L["glass_known"].get("unsolved", 0)}. Two of the three remaining '
  'lookalikes have a wedge below 0.001°, whose phase ay is unobservable; the third is off in d_W. Control: '
  f'{_F["finished"]} configurations with a sub-0.5° wedge at T = 10 s gave {_F["glass_unknown"].get("correct", 0)} and '
  f'{_F["glass_known"].get("correct", 0)} correct, with no lookalikes either way. Median solve time fell from about 10 s to '
  'about 1 s. For hardware with known glass, fixing the indices removes almost all lookalike acceptances. Code: '
  'experiments/meeting_oct6/glassknown.py.')
H2('Integration')
B('risley_lattice/solve.py now has sign_sweep() as the last rung of solve18. It runs only after every other rung has '
  'failed, tries pairs and the triple before singletons, and polishes an accepted answer. Setting RISLEY_SIGN_SWEEP=0 '
  'turns it off. The original file is kept at experiments/meeting_oct6/backup/solve.py.orig. The local experiments in C1, '
  'C2 and C6 ran with the rung off, so they stay comparable to the campaign.')
if reg:
    txt = (f'Regression check on the seed-2026 battery: all {reg["identical_on_previously_solved"]} configurations that '
           f'solve without the rung give bit-identical answers with it ({reg["changed_on_previously_solved"]} changed). '
           f'{reg["solved_off"]}/30 are solved without the rung and {reg["solved_on"]}/30 with it.')
    if reg.get('g1_214731'):
        g = reg['g1_214731'][0]
        txt += (f' The case the prototype accepted at 3×10⁻³ is now solved by the current pipeline’s {g["how"]} path '
                f'(error {g["err"]:.1e}) without reaching the rung, so this case no longer exercises it. The current code '
                'differs from the version that generated the atlases, and some campaign failures may already be fixed.')
    B(txt)
EMPTY()

# ================================================================== C4
H1('12. C4 — How Unique Is the Answer?')
H2('Motivation')
B('Joseph noted that the inverse problem has no unique solution in general, and that converged certificates should point '
  'to the true setup rather than to something within an arbitrary tolerance. This section measures how far that '
  'non-uniqueness extends for the three-prism model. Code: experiments/meeting_oct6/nonunique.py.')
H2('Findings')
hw = nu['halfwidth_rel_at_eps_1em3']
L('The exact twin (ax, ay) → (−ax, ay + 180°) is a proved identity (see C5 and the Lean file). The ±18° box on ay admits '
  'only one member of each pair, so the twin never causes a failure.', bp='Exact twins. ')
L(f'Around the truth, the parameter vectors whose pattern stays within a tolerance ε form a thin ellipsoid (median '
  f'condition number {nu["cond_median"]:.1e} over 150 configurations). At ε = 10⁻³ RMS per coordinate, it pins the speeds '
  f'to about {np.median([hw["N1"],hw["N2"],hw["N3"]]):.0e} of their range, ay to '
  f'{np.median([hw["ay1"],hw["ay2"],hw["ay3"]])*100:.2f}%, ax to {np.median([hw["ax1"],hw["ax2"],hw["ax3"]])*100:.2f}%, '
  f'the glass indices to {np.median([hw["ng1"],hw["ng2"],hw["ng3"]])*100:.1f}%, the working distance to '
  f'{hw["d_W"]*100:.1f}% and the gap to {hw["gap"]*100:.1f}%. The long axis is the working-distance/gap trade-off (mean '
  f'|component| {nu["weak_dir_mean_abs"]["d_W"]:.2f} and {nu["weak_dir_mean_abs"]["gap"]:.2f}). So infinitely many '
  'solutions fit within any finite tolerance, but they all sit in this small, well-described region.',
  bp='Local ε-equivalence. ')
tw = nu['twin']
L(f'Flipping one wedge’s sign and refitting all 18 parameters returned to the true answer in {tw["back_to_truth"]} of '
  f'{tw["n"]} trials. The rest ended in poor local minima, with pattern RMS ≥ 0.14, and no distinct parameter set matched '
  'the pattern to within 0.1.', bp='Discrete near-twins. ')
ms = nu['multistart']
L(f'With the speeds taken from the spectrum and every other parameter started at random, {ms["exact_minima"]} of '
  f'{ms["starts"]} starts over {ms["cases"]} configurations reached an exact fit. All of them landed on the true '
  'parameters; none found an exact fit anywhere else.', bp='Global multistart. ')
cases = {a[0]: a for a in aw}
axm = np.array([a[5] for a in cases.values()])
L(f'Uniqueness does break down at the identifiability edge. In the campaign, {len(aw)} of 842,126 trials '
  f'({len(cases)} configurations) passed the MSE < 10⁻¹² acceptance test while some parameter was off by 10⁻³ or more. '
  f'{pc(np.mean(axm < 0.5),0)} of them have a wedge below 0.5°, and their median smallest wedge is {np.median(axm):.3f}°. '
  'These are genuine ε-equivalents that the pattern cannot tell apart. Reporting the certified per-parameter bounds '
  'alongside acceptance, and flagging accepted answers with a very weak wedge, would make this visible to a user.',
  bp='Where it fails. ')
_camp = np.load(os.path.expanduser(os.environ.get('CAMP', '~/work/camp/campaign.npz')))
_ax = _camp['ax_min']; _bad = np.isin(_camp['case'], np.array(list(cases.keys())))
_rows = []
for _lo, _hi, _lab in [(0, 0.01, 'Under 0.01°'), (0.01, 0.1, '0.01°–0.1°'), (0.1, 0.5, '0.1°–0.5°'),
                       (0.5, 1, '0.5°–1°'), (1, 2, '1°–2°'), (2, 5, '2°–5°'), (5, 99, 'Over 5°')]:
    _s = (_ax >= _lo) & (_ax < _hi); _n = int(_s.sum()); _k = int(_bad[_s].sum())
    _often = 'never' if _k == 0 else f'1 in {_n/_k:,.2f}'
    _rows.append([_lab, c(_n), c(_k), _often, f'{100*_k/_n:.4f}%'])
B('How often this happens depends strongly on how flat the flattest wedge is. The table groups every campaign '
  'configuration by its flattest wedge and shows how many in each group produced a lookalike answer.')
TABLE(['Flattest wedge', 'Configurations in group', 'Lookalike answers', 'How often', 'Percentage'], _rows,
      [1900, 2000, 1700, 1900, 1740])
CAP('Table C4. Configurations in the 669,574-case campaign that passed the MSE acceptance test with a parameter '
    'error of 10⁻³ or more, by the tilt of their flattest wedge. How often = configurations in the group ÷ lookalikes.')
B('None of the 674 lookalike attempts had a wedge leaning the wrong way, so the sign-sweep rung cannot help them. In '
  'about 3 out of 4 (529 of 674) the parameter furthest from the truth was the glass index of the flattest wedge; the '
  'next most common was the working distance. A nearly flat wedge barely deflects the beam, so its glass index has '
  'almost no effect on the pattern. The typical error was small (median about 0.02 in the worst parameter, under 0.1 in '
  '80% of cases). Fixing the glass index to the known hardware value would likely remove most of these cases.',
  bp='What was actually wrong. ')
L('Forcing two prisms to the same speed shrinks the weakest singular values by 2–10×, but in the exact (non-paraxial) model '
  'it does not produce an exact zero: co-rotating prisms remain weakly distinguishable through the geometry. The exact '
  'degeneracy of the paraxial theory becomes a near-degeneracy.', bp='Equal speeds. ')
if os.path.exists(os.path.join(FIG, 'C4_nonuniqueness.png')):
    FIGURE(os.path.join(FIG, 'C4_nonuniqueness.png'))
    CAP('Figure C4. Left: pattern RMS of the best twin after a one-wedge sign flip and refit (twins that returned to the '
        'truth are omitted). Right: median ε-equivalence half-width per parameter at ε = 1, as a fraction of each '
        'parameter’s range. Multiply by ε for other tolerances.')
EMPTY()

# ================================================================== C5
H1('13. C5 — Lean Formalization: a Starting Point')
B('Joseph recommended formalizing the theorems in Lean, where a proof with no “sorry” placeholders is fully checked. The '
  'Ben/lean folder holds a first file, RisleyFormal/Basic.lean, and a README written for a first-time user. The file '
  'covers three results:')
L('A complete proof that the flip (ax, ay) → (−ax, ay + 180°) leaves the exit-face normal, and therefore the pattern, '
  'unchanged.', bp='Flip symmetry. ')
L('Complete proofs that a positive worst-case margin rules out TIR at every phase, plus the inequality behind the '
  'monotone composition used in C1.', bp='TIR margin. ')
L('The pigeonhole bound on lattice relations, stated with a proof plan. It is the next target.', bp='Pigeonhole bound. ')
B('These files have not been compiled yet, because this round of work was limited to Python on this machine. Run '
  '“lake build”, or paste the file into live.lean-lang.org, before citing anything as formalized.')
EMPTY()

# ================================================================== C6
H1('14. C6 — Paper Placeholders E1–E4 Filled')
e1, e2, e3, e4 = fp['E1'], fp.get('E2'), fp['E3'], fp['E4']
B('paper/main.tex is unchanged. paper/main_campaign.tex is a copy in which the E1–E4 placeholder boxes are replaced by '
  'generated tables and figures (paper/campaign_E1_E4_parts/ and paper/figures/E1–E4). Generator: '
  'experiments/meeting_oct6/fill_paper.py.')
rows = [['E1 atlas', f'{c(e1["n"])} configurations; {pc(e1["solved"]/e1["n"],2)} recovered; '
         f'{c(e1["unsolved"])} unsolved, {e1["unexplained"]} without a certificate', 'Filled'],
        ['E2 calibration', (f'{e2["covered"]}/{e2["n"]} fresh recoveries with all 18 bounds covering; median tightness '
                            f'{e2["tightness_median"]:.1f}×') if e2 else '—', 'Development scale'],
        ['E3 exponents', f'σ(N̂) slope {e3["slope"]:.2f} (95% CI {e3["ci"][0]:.2f} to {e3["ci"][1]:.2f}) against −3/2; '
         f'{e3["slope_20_80"]:.2f} over 20–80 s', 'σ law only'],
        ['E4 prescriptions', f'{pc(e4["consistent"]/e4["n"],1)} recovered no later than the first rung ≥ T_req; '
         f'{pc(e4["predicted_infeasible"]/e4["unsolved_with_treq"],1)} of unsolved cases correctly predicted infeasible',
         'Ladder-censored']]
TABLE(['Placeholder', 'What the data shows', 'Status'], rows, [1700, 5740, 1800])
if e2:
    B(f'Still pending: the pair-gap and detectability exponents, which need fixed-T cohorts; bisected T_min values for E4; '
      f'and E5–E7 (prism count, noise, baselines), which need new runs. The calibration check passes on {e2["covered"]} of '
      f'{e2["n"]} configurations. That is consistent with the design: at 3σ each parameter misses about 0.3% of the time, '
      'so with 18 parameters about 5% of configurations (about 6 of 121) should show at least one miss. The 26/26 seen '
      'during development is also consistent, since about 1.4 misses would be expected there. A much larger run is '
      'still needed to pin down the rate before the paper quotes it. This check does not involve any of the October '
      'code changes (the sign-sweep rung was off and the bound computation is unchanged).')
EMPTY()

# ---------------------------------------------------------------- synthesis add-on
state['anchor'] = next(p for p in d.paragraphs if p.text.startswith('Figures: risley_B1'))
H2('Since the Oct 6 meeting')
L('TIR now has an exact, phase-independent bound in terms of incident angle, index ratio and wedge angle. Every wedge ≤ '
  '9.30° is TIR-free across the whole box, which matches the ±10° soft goal, and restricting to ±10° lifts adaptive '
  f'recovery from {pc(fb["rate"],1)} to {pc(b10["rate"],1)}.')
L(f'Ben’s regime, with one strong wedge plus sub-degree wedges, is recovered in {pc(bn["by_T"]["80.0"]["rate"],1)} of '
  'configurations by 80 s.')
L('The pair/triple sign-flip rung has been tested on the full G1 population and is now part of solve18, with a polish step.')
L('In the generic case the answer is unique up to a thin, quantified ε-ellipsoid. Rare near-equivalents that pass the MSE '
  'test occur only with very weak wedges.')
L('A Lean starter file proves the flip symmetry and the TIR margin lemma (not yet compiled), and placeholders E1–E4 are '
  'filled in main_campaign.tex.')
EMPTY()

d.save(DST)
print('wrote', DST)
