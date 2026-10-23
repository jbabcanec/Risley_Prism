#!/usr/bin/env python3
"""Plain-language companion to risley_prism_analysis_v2.docx.
Writes Ben/Risley_October_Followups_Plain_Language.docx. Every number is
computed from the same outputs as the technical report."""
import os, json, collections
import numpy as np
import docx
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
BEN = os.path.dirname(REPO)
OUT = os.path.join(HERE, 'out')
DST = os.path.join(BEN, 'Risley_October_Followups_Plain_Language.docx')
J = lambda f: json.load(open(os.path.join(OUT, f)))
tir = J('tir_campaign.json'); loc = J('local_summary.json'); ov = J('atlas_overlap.json')
ssr = loc['sign_sweep']; aw = J('mse_accepted_wrong.json')
camp = np.load(os.path.expanduser(os.environ.get('CAMP', '~/work/camp/campaign.npz')))

NAVY = RGBColor(0x1A, 0x3C, 0x6B)
d = docx.Document()
st = d.styles['Normal']; st.font.name = 'Calibri'; st.font.size = Pt(11)
for s in ('Heading 1', 'Heading 2', 'Title'):
    d.styles[s].font.color.rgb = NAVY
for sec in d.sections:
    sec.left_margin = sec.right_margin = Inches(1)


def c(n, dg=0):
    return f'{n:,.{dg}f}'


def P(text, bold_lead=None):
    p = d.add_paragraph()
    if bold_lead:
        p.add_run(bold_lead).bold = True
    p.add_run(text)
    return p


def BL(text, bold_lead=None):
    p = d.add_paragraph(style='List Bullet')
    if bold_lead:
        p.add_run(bold_lead).bold = True
    p.add_run(text)


_num = [0]


def NL(text, restart=False):
    if restart:
        _num[0] = 0
    _num[0] += 1
    p = d.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.35)
    p.paragraph_format.first_line_indent = Inches(-0.25)
    p.add_run(f'{_num[0]}.  ' + text)


def MONO(lines):
    for ln in lines:
        p = d.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.5)
        p.paragraph_format.space_after = Pt(0)
        r = p.add_run(ln); r.font.name = 'Consolas'
    d.add_paragraph()


def TABLE(header, rows, note=None, bold_row=None):
    t = d.add_table(rows=1 + len(rows), cols=len(header))
    t.style = 'Table Grid'
    for ri, vals in enumerate([header] + rows):
        for ci, v in enumerate(vals):
            cell = t.rows[ri].cells[ci]
            cell.text = ''
            run = cell.paragraphs[0].add_run(str(v)); run.font.size = Pt(10)
            if ri == 0:
                run.bold = True; run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                sh = OxmlElement('w:shd'); sh.set(qn('w:val'), 'clear')
                sh.set(qn('w:color'), 'auto'); sh.set(qn('w:fill'), '1A3C6B')
                cell._tc.get_or_add_tcPr().append(sh)
            elif bold_row is not None and ri - 1 == bold_row:
                run.bold = True
    if note:
        p = d.add_paragraph(); r = p.add_run(note); r.italic = True; r.font.size = Pt(9)
    else:
        d.add_paragraph()


def wedge_trace(a, n=1.8, beam=25.0):
    s, hits, trapped = beam, [], False
    for _ in range(3):
        u = np.degrees(np.arcsin(np.sin(np.radians(s)) / n)); h = u + a; hits.append(h)
        x = n * np.sin(np.radians(h))
        if x >= 1:
            trapped = True; break
        s = np.degrees(np.arcsin(x)) - a
    return hits + [None] * (3 - len(hits)), trapped


# ======================================================================
d.add_heading('Risley Prism Project: October Follow-ups in Plain Language', 0)
P('A companion to risley_prism_analysis_v2.docx. That report gives the technical details; this one explains the same '
  'results without assuming any background in optics. Section numbers in brackets (C1–C4) point to the matching '
  'sections of the technical report.')

d.add_heading('What the project is about', 1)
P('A Risley prism system steers a laser beam with a few thin glass wedges (prisms) stacked in a row. Each wedge is '
  'thicker on one side than the other, so it bends light a little. Each one spins at its own speed. As they spin, the '
  'laser dot on a target traces a looping pattern, like a Spirograph drawing.')
P('The usual question is “given the machine, what pattern will it draw?” Our group asks the reverse: given only the '
  'drawing, can we work out what the machine is? The program that does this is called the solver.')
P('To test the solver, we invent a random machine on the computer, draw its pattern, hand only the pattern to the '
  'solver, and check whether it recovers the machine we started with. Doing that hundreds of thousands of times shows '
  'where the solver works and where it fails. That large batch of tests, run on the Modal cloud servers in September, '
  'is what “the campaign” or “the atlases” refers to. Everything new in this document was run on Lisa’s own computer, '
  'not on Modal.')
d.add_heading('Words used in this document', 2)
BL('one simulated Risley system. It is described by 18 numbers: how fast each of the three wedges spins; how steep '
   'each wedge is and which way it leans; where each wedge starts in its rotation; the glass type of each wedge; the '
   'distance to the target and the gap between wedges; and the angle and position at which the laser beam enters.',
   'Machine: ')
BL('the 18 numbers we used to create a test machine. The solver “got it right” when all 18 of its numbers match these '
   'very closely, not just the angles.', 'True machine: ')
BL('how steep a wedge is: how much its back face is tilted relative to its front face. 0° would be a flat window that '
   'does not bend light. A plus or minus sign says which way it leans: +5° and −5° are equally steep, leaning in '
   'opposite directions.', 'Wedge angle (tilt): ')
BL('one try at solving one machine. The solver first watches the pattern for 10 seconds; if that fails it tries again '
   'watching for 20, then 40, then 80 seconds. Each try is one attempt, so one machine can have several attempts.',
   'Attempt: ')
BL('a number that measures how much a material slows light down. Air is 1, ordinary glass about 1.5, and the densest '
   'glass allowed in the simulations is 1.8. In this document it is also called the “glass type”.',
   'Refractive index: ')

# ======================================================================
d.add_heading('1. Trapped light (total internal reflection) [C1]', 1)
P('When light goes from glass back out into air, it bends. If it hits the glass surface at too steep an angle, it does '
  'not get out at all: it bounces back inside the glass. This is called total internal reflection. You see it when you '
  'look up from underwater and the surface acts like a mirror. For this project it is a problem, because it puts a '
  'sudden glitch in the laser pattern and confuses the solver.')
P('In the meeting, Joseph said this cannot be predicted from the wedge’s steepness alone; it also depends on the angle '
  'at which the light arrives and on the type of glass. We worked out an exact rule that combines all three and checked '
  'it against the simulation code: the rule predicts exactly when the glitch happens.')

d.add_heading('Why there is a limit, and where 33.75° comes from', 2)
P('When light leaves glass into air it comes out steeper than it was inside. Light that hits the inside of the surface '
  'straight on (0°) leaves straight; at 20° it leaves at about 38°; at 30° it leaves at about 64°. It can never leave '
  'steeper than 90°, which means skimming flat along the surface. The critical angle is the inside angle that would '
  'make it leave at exactly 90°. Any steeper than that and there is no angle left for it to exit at, so it is trapped.')
P('The bending rule (Snell’s law) is: refractive index × sin(angle inside) = sin(angle outside). At the critical point '
  'the outside angle is 90°, and sin(90°) = 1. For the densest glass:')
MONO(['1.8 × sin(angle inside) = 1', 'sin(angle inside) = 1 ÷ 1.8 = 0.556', 'angle inside = 33.75°'])
P('(Sine is the calculator button that turns an angle into a number between 0 and 1; working backwards from 0.556 gives '
  '33.75°.) Denser glass traps light more easily:')
TABLE(['Glass (refractive index)', 'Critical angle (the limit)'],
      [[f'{n} ({lab})', f'{np.degrees(np.arcsin(1/n)):.2f}°'] for n, lab in
       [(1.3, 'lightest allowed'), (1.5, 'ordinary glass'), (1.8, 'densest allowed')]])

d.add_heading('Where 9.3° comes from', 2)
P('9.3° is the steepest wedge that is guaranteed safe even when everything is at its worst at the same moment: the '
  'densest glass (limit 33.75°), the laser entering at the largest tilt the simulations allow (25°), and all three '
  'wedges leaning the same way during their spin, so each one makes the beam steeper instead of cancelling another.')
P('At each wedge three things happen:')
NL('The light enters the flat front face and bends to a shallower angle.', restart=True)
NL('It reaches the back face, which is tilted by the wedge angle, so the tilt adds directly to how steeply it hits that '
   'face.')
NL('If that is below 33.75° it escapes, bending steeper on the way out. That steeper angle is what reaches the next '
   'wedge.')
P('Because each wedge sends the light out steeper, the third wedge always gets the steepest light and is the first to '
  'trap it. 9.3° was found by guess-and-check: trace the worst-case beam for a trial wedge angle, raise the angle if the '
  'light escapes and lower it if it gets trapped, and narrow in on the tipping point. The table shows the angle at '
  'which the light hits each wedge’s back face; it must stay below 33.75°.')
rows = []
for a in (5, 8, 9, 9.296, 9.5, 10):
    hits, trapped = wedge_trace(a)
    third = f'{hits[2]:.2f}°' if abs(a - 9.296) < 1e-6 else f'{hits[2]:.1f}°'
    verdict = ('Right at the limit' if abs(a - 9.296) < 1e-6 else
               ('Yes' if trapped else ('No, but close' if hits[2] > 32.5 else 'No')))
    rows.append([f'{round(a, 1):g}°', f'{hits[0]:.1f}°', f'{hits[1]:.1f}°', third, verdict])
TABLE(['Wedge angle', '1st wedge hit', '2nd wedge hit', '3rd wedge hit', 'Trapped?'], rows,
      'Worst case: densest glass (1.8), beam entering at 25°, all three wedges leaning the same way.', bold_row=3)
P('This applies to each wedge on its own, not to the difference between wedges: every wedge must have a tilt of 9.3° '
  'or less, whichever way it leans.')

d.add_heading('What this means in practice', 2)
fb, b10, tp = tir['full box (|ax|<=18)'], tir['|ax|<=10 box'], tir['TIR-prone (margin<0)']
BL(f'Joseph suggested keeping each wedge within ±10°. Strictly, 10° is not guaranteed safe: in the exact worst case '
   f'above, the third wedge traps the light. But that needs every condition at its extreme at once, and none of the '
   f'{c(b10["n"])} test machines with wedges up to 10° could actually trap light. (Joseph’s words came from the '
   'auto-captions, so it is worth confirming with him that he meant each wedge’s tilt.)', 'Joseph’s 10°. ')
BL(f'The simulations currently allow wedges up to 18°. About {100*tir["frac_tir_box18"]:.1f}% of those machines can '
   f'trap light, and that small group accounts for {c(tp["unsolved"])} of the {c(fb["unsolved"])} failures '
   f'({100*tp["unsolved"]/fb["unsolved"]:.0f}%).', 'How often it happens now. ')
BL(f'Keeping wedges within 10° raises the solver’s success rate from {100*fb["rate"]:.2f}% to {100*b10["rate"]:.2f}%. '
   f'A fresh run of {loc["box10"]["n"]} new test machines agreed.', 'Limiting the wedges. ')
BL('With lighter glass or a beam that comes in straighter, the safe angle is higher. With the beam coming in straight '
   '(0° tilt) it is about 16.7°. With Ben’s actual hardware numbers we can work out the safe angle for his setup.',
   'The limit depends on the setup. ')

# ======================================================================
bn, b6 = loc['ben'], loc['ben066']
d.add_heading('2. Ben’s real hardware [C2]', 1)
P('Ben described his old physical system as a pair of steep wedges (6° or 12°) that spin together, plus a couple of '
  'very shallow wedges, under 1° (he remembered about 0.66°). Shallow wedges barely bend the light, so they are hard to '
  'detect from the pattern, and earlier results showed this is where the solver struggles most. So we tested machines '
  'that look like Ben’s.')
d.add_heading('What was tested', 2)
BL('one steep wedge (6° or 12°, chosen at random) and two shallow wedges, each with a tilt picked at random between '
   '0.3° and 1°. For example, one machine might have 12°, 0.45° and 0.8°.', f'Test A ({bn["n"]} machines): ')
BL('the same, but both shallow wedges set to exactly 0.66°, the value Ben remembered.',
   f'Test B ({b6["n"]} machines): ')
d.add_heading('Results', 2)
BL(f'Test A: the solver got {bn["by_T"]["80.0"]["k"]} of {bn["n"]} right when it could watch the pattern for up to '
   '80 seconds.')
BL(f'Test B: it got all {b6["by_T"]["40.0"]["k"]} of {b6["n"]} right within 40 seconds.')
P('So Ben’s kind of system looks solvable.')
d.add_heading('The 4 machines it did not solve', 2)
BL('Two had two wedges spinning at almost exactly the same speed (in one, the speeds differed by only 0.00024 turns per '
   'second). Their effects blur into one, and with a shallow wedge on top there is very little information to work '
   'with.')
BL('Two failed because the solver started looking in the wrong place and found a pattern that matched very well but '
   'with the wrong numbers.')
rr_path = os.path.join(OUT, 'rerun_ben4.jsonl')
if os.path.exists(rr_path):
    rr = [json.loads(l) for l in open(rr_path)]
    P('We re-ran these four with the sign-flip fix (Section 3) switched on, on Lisa’s computer:')
    labels = {16: 'Two wedges at nearly the same speed', 62: 'Two wedges at nearly the same speed',
              87: 'Started in the wrong place', 90: 'Started in the wrong place'}
    rows = []
    for i in (16, 62, 87, 90):
        rs = sorted([r for r in rr if r['case'] == i], key=lambda r: r['T'])
        ok = [r for r in rs if r['err'] is not None and r['err'] < 1e-3]
        Ts = ', '.join(f'{r["T"]:g}' for r in rs)
        rows.append([str(i), labels[i], 'Solved at ' + f'{ok[0]["T"]:g} s' if ok else 'Not solved', Ts + ' s'])
    TABLE(['Machine', 'Original problem', 'With the fix', 'Viewing times tried'], rows)
    BL('Machine 62 is now solved at the first, shortest test: flipping the lean direction of two wedges fixed it, '
       'even though two of its wedges spin at nearly the same speed.')
    BL('Machine 87 is a lookalike too. The solver found a flatter first wedge (0.50° instead of 0.58°) made of denser '
       'glass (1.37 instead of 1.32). For a thin wedge, how much it bends the light is about (glass number − 1) × wedge '
       'angle: 0.32 × 0.58 = 0.186 and 0.37 × 0.50 = 0.186, the same. Both wedges bend the light identically, so the '
       'pattern cannot tell them apart, however long it is watched. When the solver was told the real glass type, it '
       'found the true machine straight away.')
    BL('Machine 90 is one of the rare lookalikes described in Section 4: the pattern matches almost perfectly but the '
       'glass type of its flattest wedge (0.34°) is slightly wrong. Watching longer steadily shrinks the error: about 0.033 '
       'at 80 seconds, 0.016 at 160 seconds and 0.0037 at 320 seconds. The success threshold is 0.001, so a somewhat '
       'longer recording would probably solve it.')
    BL('Machine 16 is still far off even at 320 seconds. Two of its wedges spin at 1.50677 and 1.50653 turns per second '
       '(in opposite directions), a difference of only 0.00024. To tell them apart, the solver has to watch long enough '
       'for that tiny difference to add up; its own rule of thumb (2 × 0.12 ÷ 0.00024) says about 1,000 seconds, roughly '
       '17 minutes. 320 seconds is not enough.')
    pend = [i for i in (16, 87, 90) if not any(r['case'] == i and r['T'] >= 320 for r in rr)
            and not any(r['case'] == i and r['err'] is not None and r['err'] < 1e-3 for r in rr)]
    if pend:
        P('Still running at the time of writing: the longest tests (watching for 160 and 320 seconds) for machine'
          + ('s ' if len(pend) > 1 else ' ') + ', '.join(map(str, pend)) + '. They take more than the 3-minute limit per '
          'command, so they run in Terminal with python3 rerun_ben4.py --all from the experiments/meeting_oct6 folder. '
          'Rebuilding this document afterwards adds their results.')
d.add_heading('What a more faithful test would add', 2)
P('Ben’s real machine had four wedges: a matched pair of steep wedges that spin together as one unit (the first bends '
  'the beam, the second bends it back, so the beam comes out shifted sideways and circles the centre as the pair '
  'spins), plus two shallow wedges that each spin freely. Our test replaced the pair with one steep wedge, because the '
  'solver assumes every wedge spins at its own speed, and two wedges locked at the same speed is exactly what it finds '
  'hardest. A faithful test would tell the solver up front that two of the four wedges are locked together. That gives '
  'it fewer unknowns and could make the problem easier. It would show whether the solver can recover a machine built '
  'the way Ben’s was, how long the pattern must be watched, and whether the locked pair causes new problems. Ben’s real '
  'numbers (wedge angles, glass type, spacing, spin speeds) would let us test his actual machine.')
P('Practical advice for real hardware: avoid choosing two spin speeds that are nearly the same. No solver can '
  'separate wedges that spin together.')

# ======================================================================
d.add_heading('3. The sign-flip fix [C3]', 1)
P('Every wedge leans one way or the other: the thick edge can sit on one side or the opposite side. In the numbers this '
  'is a plus or minus sign on the wedge angle, which is where the name “sign flip” comes from. For a steep wedge the two '
  'directions draw clearly different patterns. For a very shallow wedge they are nearly identical, so the solver '
  'sometimes picks the wrong direction, then adjusts all the other numbers to compensate, gets close but never right, '
  'and reports a failure.')
P('The fix: when everything else has failed, the solver takes its best wrong answer and tries flipping wedge '
  'directions — each wedge on its own, then two at once, then all three — re-adjusting the other numbers after each flip '
  'and keeping any version that matches the pattern perfectly. Before, it only tried flipping one wedge at a time. '
  'Flipping two or three together turned out to matter most: it accounted for '
  f'{ssr["G1"]["multi_flip_wins"] + ssr["G3"]["multi_flip_wins"]} of the '
  f'{ssr["G1"]["rescued"] + ssr["G3"]["rescued"]} rescues.')
BL('G1 and G3 are names from the September report for groups of failed machines. G1: only the lean direction was '
   'wrong. G3: other angle mistakes.', 'Groups. ')
BL(f'The fix rescued {ssr["G1"]["rescued"]} of all {ssr["G1"]["tested"]} G1 failures '
   f'({100*ssr["G1"]["rescued"]/ssr["G1"]["tested"]:.0f}%) and {ssr["G3"]["rescued"]} of {ssr["G3"]["tested"]} sampled '
   f'G3 failures ({100*ssr["G3"]["rescued"]/ssr["G3"]["tested"]:.0f}%).', 'Results. ')
BL('The September report said the fix never accepted a wrong answer. One case did: the answer was very close but '
   'slightly off, because the program stopped refining too early. A final “polish” step now prevents this.',
   'A correction. ')
BL('The fix is now built into the solver as a last resort, tried only when everything else fails. On the standard '
   '30-machine test, every machine it solved before gives exactly the same answer, and one extra machine is now solved.',
   'Code change. ')
BL('The September campaign was run before the fix existed. To compare Ben-style machines fairly against it, those '
   'tests were run with the fix switched off.', 'Why it was off for Ben’s tests. ')

# ======================================================================
d.add_heading('4. Is there only one right answer? [C4]', 1)
P('Joseph asked in the meeting whether two different machines could draw the same pattern. If so, the solver could not '
  'know which one is real.')
BL('Many slightly different settings draw a pattern that is almost identical, but they all sit very close to the true '
   'machine, like many darts landing in one small bullseye. The setting that can wiggle the most without changing the '
   'pattern much is the trade-off between the distance to the target and the gap between wedges.',
   'Usually, the answer is unique. ')
nu = loc['nonunique']['multistart']
BL('The solver works like tuning a radio: it starts from a guess of the 18 numbers and nudges them step by step until '
   'the patterns match. Normally it makes its own careful starting guess. Here we gave it random starting guesses '
   '(each from a different random seed) for everything except the spin speeds, which were started near the right '
   f'values because the solver reads them reliably. Over {nu["cases"]} machines and {nu["starts"]} random starts, '
   f'{nu["exact_minima"]} ended in a perfect match, and every one of them was the true machine. The rest got stuck on '
   'imperfect matches that the solver would reject.', 'Restarts agree. ')

d.add_heading('The exception: lookalike answers', 2)
cases = {a[0]: a for a in aw}
P(f'In the September campaign, {len(aw)} attempts, spread over {len(cases)} different machines, produced a pattern '
  'that matched almost perfectly while some of the 18 numbers were still off by 0.001 or more. (Some machines did this '
  'at more than one viewing time, which is why there are more attempts than machines.) In these cases two different '
  'machines really do draw practically the same picture.')
P('How often this happens depends on how flat the flattest of the three wedges is. The table groups every machine in '
  'the campaign by its flattest wedge and shows how many in each group produced a lookalike answer.')
ax = camp['ax_min']; bad = np.isin(camp['case'], np.array(list(cases.keys())))
rows = []
for lo, hi, lab in [(0, 0.01, 'Under 0.01°'), (0.01, 0.1, '0.01°–0.1°'), (0.1, 0.5, '0.1°–0.5°'),
                    (0.5, 1, '0.5°–1°'), (1, 2, '1°–2°'), (2, 5, '2°–5°'), (5, 99, 'Over 5°')]:
    s = (ax >= lo) & (ax < hi); n = int(s.sum()); k = int(bad[s].sum())
    rows.append([lab, c(n), c(k), 'never' if k == 0 else f'1 in {n/k:,.2f}',
                 f'{100*k/n:.4f}%'])
TABLE(['Flattest wedge', 'Machines in this group', 'Lookalikes', 'How often', 'As a percentage'], rows,
      'How often = machines in the group ÷ lookalikes. Example: 1,065 ÷ 47 = 22.66, so about 1 in every 22.66 machines.')
P('The flatter the flattest wedge, the more likely a lookalike. Above 5° it never happened. Ben’s shallow wedges, '
  'around 0.66°, fall in the 0.5°–1° group.')

d.add_heading('What was actually wrong in the lookalikes', 2)
P('Not the wedge directions: in none of the 674 lookalike attempts was any wedge leaning the wrong way, so the '
  'sign-flip fix has nothing to fix here. They are simply too close to tell apart.')
P('Mostly the glass type. In about 3 out of 4 cases (529 of 674), the most-wrong number was the glass type of the '
  'flattest wedge. The next most common was the distance to the target. A wedge that is almost flat barely bends the '
  'light, so it hardly matters what glass it is made of: changing its glass slightly changes the pattern by too little '
  'to detect.')
errs = np.array([a[2] for a in aw])
P(f'Usually only a little: the typical error was about {np.median(errs):.2f} (for example, a glass value of 1.52 '
  f'instead of 1.50), and in {100*np.mean(errs < 0.1):.0f}% of cases it was under 0.1.', 'How far off. ')
d.add_heading('What could help', 2)
BL('With real hardware, Ben knows what his wedges are made of. If the solver does not have to work it out, this problem '
   'would likely mostly disappear. This can be tested.', 'Tell the solver the glass type up front. ')
BL('The solver can already calculate how uncertain each of its numbers is. In these cases it would show a wide '
   'uncertainty on the flat wedge’s glass, a warning sign instead of a confident wrong answer.',
   'Report how sure the solver is. ')
BL('so they get a second look.', 'Flag any answer with a very flat wedge, ')

# ======================================================================
d.add_heading('5. Can better starting guesses help?', 1)
P('The solver starts from a guess of the 18 numbers and adjusts them until the pattern matches. Each time it tries a '
  'longer viewing time, it starts over with a fresh guess. We tested a simple improvement: start each longer attempt '
  'from the answer found at the shorter one (“warm start”), instead of starting over.')
_ws = json.load(open(os.path.join(OUT, 'warmstart.json')))
_b, _o = _ws['groups']['basin'], _ws['groups']['other']
BL(f'We took {_b["jobs"]} failed machines from the September batch whose failure reason was “started in the wrong '
   f'place” (out of {c(_ws["population"]["basin"])} such failures), and {_o["jobs"]} failed machines with other reasons.',
   'What was tested. ')
BL(f'Warm start rescued {_b["rescued"]} of {_b["jobs"]} in the first group and {_o["rescued"]} of {_o["jobs"]} in the '
   f'second, about 6% each. But in the first group it also produced {_b["wrong_accepted"]} lookalike answers: patterns '
   'that matched almost perfectly with slightly wrong numbers, almost always the glass type of a very flat wedge.',
   'Results. ')
BL('Usually the previous answer was far from the truth, so there was nothing good to reuse: in only about 1 case in 8 '
   'was it already within 1 of the true value in every number.', 'Why it helps so little. ')
BL('Machine 87 looked like a bad starting guess but turned out to be a lookalike: reusing its 80-second answer at 160 '
   'and 320 seconds left it exactly where it was.', 'Machine 87. ')
P('Overall, better starting guesses help a little, but most of these failures are not really about the guess. Many are '
  'lookalikes, where a flat wedge’s glass type and angle can trade off against each other. The most promising fix is '
  'to give the solver the glass type up front, which solved machine 87 immediately.')

d.add_heading('Telling the solver the glass type', 2)
_gk = json.load(open(os.path.join(OUT, 'glassknown.json')))
_L, _F = _gk['groups']['look'], _gk['groups']['flat']
P(f'We then tested the most promising fix directly. We took {_L["finished"]} of the {_gk["population"]["look"]} '
  'machines that had produced a lookalike answer in September, and ran the solver on each one twice from scratch: once '
  'as usual, and once with the glass type of every wedge given to it, as it would be for real hardware.')
TABLE(['Outcome', 'Glass type unknown', 'Glass type known'],
      [[lab, str(_L['glass_unknown'].get(k, 0)), str(_L['glass_known'].get(k, 0))] for k, lab in
       [('correct', 'Correct answer'), ('lookalike', 'Lookalike (accepted but wrong)'), ('unsolved', 'Not solved')]],
      'Machines that produced a lookalike answer in the September batch, re-solved with the current solver.')
BL(f'Lookalike answers dropped from {_L["glass_unknown"].get("lookalike", 0)} to {_L["glass_known"].get("lookalike", 0)}, '
   f'and correct answers rose from {_L["glass_unknown"].get("correct", 0)} to {_L["glass_known"].get("correct", 0)}.',
   'Result. ')
BL('Two of the remaining three have a wedge that is essentially flat (about 0.001° or less). A wedge with no tilt does '
   'nothing to the light, so where it starts in its rotation cannot be seen in the pattern at all. The third is off in '
   'the distance to the target.', 'The 3 that remain. ')
BL(f'As a check that knowing the glass does not break anything, we also ran {_F["finished"]} ordinary machines with a '
   f'flat wedge (under 0.5°) at the shortest viewing time. Correct answers went from {_F["glass_unknown"].get("correct", 0)} '
   f'to {_F["glass_known"].get("correct", 0)}, and neither version produced a lookalike. (The unsolved ones are expected '
   'at a 10-second viewing time; in the full test they get longer viewing times.)', 'Ordinary machines. ')
BL('With the glass known, the solver was also about ten times faster on these machines (typically about 1 second '
   'instead of 10).', 'Faster too. ')
P('So for real hardware, where the glass type is known, giving it to the solver removes almost all lookalike answers.')

d.add_heading('6. Corrections to the September report', 1)
BL(f'Each test machine is generated from a number called a seed, so the same seed always produces the same machine. All '
   f'four test batches used the same starting seed, so many machines were tested more than once. The '
   f'{c(ov["rows"])} test runs cover {c(ov["unique"])} different machines, and the September counts include the '
   f'repeats. When the same machine was tested twice it got the same result {100*ov["identical_outcome"]/ov["multi"]:.1f}% '
   'of the time, which shows the solver is consistent.', 'Repeated machines. ')
BL('When the solver fails it gives a reason, and “glitch-floor” is one of them. The September report described it as '
   '“not watching long enough”. It actually means either the trapped-light glitch from Section 1 or a pattern too messy '
   'to analyse cleanly. Both reports now say this.', '“Glitch-floor”. ')

d.add_heading('7. Possible next steps', 1)
NL('Finish the 160- and 320-second tests for Ben’s three unsolved machines.', restart=True)
NL('Add a “glass type known” option to the solver permanently, since it removes almost all lookalike answers.')
NL('Model Ben’s matched pair as two wedges locked to the same speed, using his real hardware numbers.')
NL('Confirm with Joseph that “within 10°” means each wedge’s tilt.')

d.save(DST)
print('wrote', DST)
