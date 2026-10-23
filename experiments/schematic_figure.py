#!/usr/bin/env python3
"""The system schematic (paper Fig. 2): laser -> three rotating wedge
prisms (two interfaces each) -> workpiece.  Promoted from
paper/archive/generate_figures.py with a clean rotation glyph: a solid
circular arc with a tangent arrowhead (FancyArrowPatch, arc3), replacing
the old dashed ellipse + detached arrowhead.

Run from repo root: python experiments/schematic_figure.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'paper', 'figures')
INK = '#333333'

plt.rcParams.update({
    'font.size': 8, 'font.family': 'serif', 'mathtext.fontset': 'cm',
})

fig, ax = plt.subplots(figsize=(7.2, 3.2))
ax.set_xlim(-0.8, 13.5)
ax.set_ylim(-2.6, 2.8)
ax.set_aspect('equal')
ax.axis('off')

# Laser source
ax.add_patch(plt.Rectangle((-0.6, -0.2), 0.5, 0.4, fc='black', ec='black'))
ax.text(-0.35, -0.65, 'Laser', fontsize=8, ha='center')

# Physical wedge prisms -- exaggerated taper for visibility
prism_data = [
    (2.5, 15, '$W_1$', '$n_{g,1}$'),
    (5.8, -13, '$W_2$', '$n_{g,2}$'),
    (9.1, 10, '$W_3$', '$n_{g,3}$'),
]
colors = ['#b3d4fc', '#b3e6cc', '#fce4b3']
beam_pts = [(0.0, 0.0)]

for idx, (xc, wa, plabel, nlabel) in enumerate(prism_data):
    h = 1.4
    thick = 1.1
    taper = thick * 0.5 * np.tan(np.radians(abs(wa)))
    sign = 1 if wa > 0 else -1

    verts = [
        (xc - thick / 2, -h),
        (xc + thick / 2, -h + sign * taper),
        (xc + thick / 2,  h + sign * taper),
        (xc - thick / 2,  h),
    ]
    ax.add_patch(Polygon(verts, closed=True, facecolor=colors[idx],
                         edgecolor='black', lw=1.0, alpha=0.85, zorder=2))

    # Face labels outside the prism
    ax.text(xc - thick / 2 - 0.15, -h - 0.25, 'E', fontsize=7, ha='center',
            va='top', color=INK, weight='bold')
    ax.text(xc + thick / 2 + 0.15, -h + sign * taper - 0.25, 'X', fontsize=7,
            ha='center', va='top', color=INK, weight='bold')

    # Glass label inside, prism label below
    ax.text(xc, 0.0, nlabel, fontsize=8, ha='center', va='center', color=INK)
    ax.text(xc, -h - 0.55, plabel, fontsize=9, ha='center', va='top',
            weight='bold')

    # Rotation glyph: solid circular arc, arrowhead tangent at its end
    r = 0.30
    cy = h + 0.42
    th1, th2 = np.radians(210), np.radians(-40)   # 250-degree clockwise sweep
    th = np.linspace(th1, th2, 60)
    ax.plot(xc + r * np.cos(th), cy + r * np.sin(th), color=INK, lw=1.1,
            solid_capstyle='round', zorder=3)
    # arrowhead: short tangent step at the arc's end, clockwise direction
    xe, ye = xc + r * np.cos(th2), cy + r * np.sin(th2)
    tx, ty = np.sin(th2), -np.cos(th2)            # clockwise tangent
    ax.annotate('', xy=(xe + 0.02 * tx, ye + 0.02 * ty),
                xytext=(xe - 0.10 * tx, ye - 0.10 * ty),
                arrowprops=dict(arrowstyle='-|>', lw=1.1, color=INK,
                                mutation_scale=9, shrinkA=0, shrinkB=0))
    ax.text(xc, cy + r + 0.14, f'$N_{idx + 1}$', fontsize=7, ha='center',
            va='bottom', color=INK)

    # Beam path
    entry_x = xc - thick / 2
    exit_x = xc + thick / 2
    beam_pts.append((entry_x, beam_pts[-1][1] + 0.05 * (idx + 1) * (-1) ** idx))
    beam_pts.append((exit_x, beam_pts[-1][1] + sign * 0.18))

beam_pts.append((12.2, beam_pts[-1][1] + 0.4))
bx = [p[0] for p in beam_pts]
by = [p[1] for p in beam_pts]
ax.plot(bx, by, 'r-', lw=1.8, zorder=5)
ax.plot(bx[-1], by[-1], 'r*', markersize=12, zorder=6)

# Workpiece
ax.plot([12.0, 12.4], [-1.5, 2.0], 'k-', lw=3)
ax.text(12.2, -1.85, 'Workpiece', fontsize=8, ha='center')

# Air labels
for x in (1.25, 4.15, 7.45):
    ax.text(x, 1.8, r'$n\!=\!1$', fontsize=6, ha='center', color='gray')

ax.text(6.0, -2.45,
        'E = entry face     X = exit face     3 prisms = 6 interfaces',
        fontsize=7, ha='center', style='italic', color='#555')

fig.savefig(os.path.join(OUT, 'schematic.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(OUT, 'schematic.png'), dpi=300, bbox_inches='tight')
print('wrote figures/schematic.pdf/.png')
