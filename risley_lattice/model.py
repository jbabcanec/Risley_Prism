"""Forward-model bridge, parameter box, and the standard battery.

Single source of truth for everything the inverse method assumes about the
problem: the 18-parameter vector layout, its box, the canonical prism
ordering, the bridge to the exact forward model, and the seed-2026 random
battery used by every experiment. No torch dependency.
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, os.path.join(_ROOT, 'reverse_problem_v2')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core import PrismParameters, SystemGeometry, fast_forward  # noqa: E402

# ---------------------------------------------------------------- constants
P_DEFAULT = 3
T_PTS = 200
T_OBS = 10.0
DT = T_OBS / T_PTS
FS = 1.0 / DT
SRC = 6.0
THK = 3.0

N_PAR = 18
NAMES = ['N1', 'N2', 'N3', 'ax1', 'ax2', 'ax3', 'ay1', 'ay2', 'ay3',
         'ng1', 'ng2', 'ng3', 'd_W', 'gap', 'bm_ax', 'bm_ay', 'bm_px', 'bm_py']
LO = np.array([-3.5] * 3 + [-18.] * 6 + [1.3] * 3 +
              [50., 2., -25., -25., -5., -5.], dtype=np.float64)
HI = np.array([3.5] * 3 + [18.] * 6 + [1.8] * 3 +
              [200., 15., 25., 25., 5., 5.], dtype=np.float64)
RG = HI - LO


# ---------------------------------------------------------------- bridges
def canon(v):
    """Sort prisms by descending |speed| (canonical ordering)."""
    v = v.copy()
    o = np.argsort(-np.abs(v[:3]))
    v[:3], v[3:6], v[6:9], v[9:12] = v[:3][o], v[3:6][o], v[6:9][o], v[9:12][o]
    return v


def vec2pat(v, n_points=T_PTS, time_limit=T_OBS):
    """18-vector -> (n_points, 2) workpiece pattern via the exact model."""
    geo = SystemGeometry(
        source_distance=SRC, prism_thickness=THK,
        workpiece_distance=float(v[12]), inter_prism_gap=float(v[13]),
        beam_angle_x=float(v[14]), beam_angle_y=float(v[15]),
        beam_pos_x=float(v[16]), beam_pos_y=float(v[17]))
    pr = PrismParameters(3, v[:3].tolist(), v[3:6].tolist(), v[6:9].tolist(),
                         glass_indices=v[9:12].tolist(), geometry=geo)
    return fast_forward(pr, n_points, time_limit)


def pat_P(speeds, ax, ay, glass, geom9, n_points=T_PTS, time_limit=T_OBS):
    """Pattern for an arbitrary prism count P = len(speeds).

    geom9 = [d_W, gap, bm_ax, bm_ay, bm_px, bm_py] (last six of the 18-vec
    convention, glass excluded)."""
    geo = SystemGeometry(
        source_distance=SRC, prism_thickness=THK,
        workpiece_distance=float(geom9[0]), inter_prism_gap=float(geom9[1]),
        beam_angle_x=float(geom9[2]), beam_angle_y=float(geom9[3]),
        beam_pos_x=float(geom9[4]), beam_pos_y=float(geom9[5]))
    pr = PrismParameters(len(speeds), list(speeds), list(ax), list(ay),
                         glass_indices=list(glass), geometry=geo)
    return fast_forward(pr, n_points, time_limit)


# ---------------------------------------------------------------- batteries
def battery_cases(n=30, seed=2026, min_speed=0.15):
    """The standard random battery (seed 2026): canon-ordered 18-vectors."""
    rng = np.random.default_rng(seed)
    cases = []
    for _ in range(n):
        v = (LO + RG * rng.random(N_PAR).astype(np.float32)).astype(np.float64)
        for j in range(3):
            if abs(v[j]) < min_speed:
                v[j] = np.copysign(min_speed, v[j])
        cases.append(canon(v))
    return cases


def battery_cases_P(P, n=20, seed=2026, min_speed=0.15):
    """Random cases for arbitrary prism count P.

    Returns list of (speeds, ax, ay, glass, geom9) tuples, speeds sorted by
    descending |speed| with all per-prism arrays permuted consistently."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        sp = rng.uniform(-3.5, 3.5, P)
        for j in range(P):
            if abs(sp[j]) < min_speed:
                sp[j] = np.copysign(min_speed, sp[j])
        ax = rng.uniform(-18.0, 18.0, P)
        ay = rng.uniform(-18.0, 18.0, P)
        ng = rng.uniform(1.3, 1.8, P)
        geom9 = np.array([rng.uniform(50, 200), rng.uniform(2, 15),
                          rng.uniform(-25, 25), rng.uniform(-25, 25),
                          rng.uniform(-5, 5), rng.uniform(-5, 5)])
        o = np.argsort(-np.abs(sp))
        out.append((sp[o], ax[o], ay[o], ng[o], geom9))
    return out


def case_stats(sp):
    """(min cycles in T_OBS, min |speed|-magnitude separation) of a case."""
    sp = np.asarray(sp)
    P = len(sp)
    sep = min(abs(abs(sp[i]) - abs(sp[j]))
              for i in range(P) for j in range(i + 1, P)) if P > 1 else np.inf
    return float(np.min(np.abs(sp)) * T_OBS), float(sep)


def case_at(i, seed0=424242, min_speed=0.15):
    """Order-free, cluster-friendly case generation: the i-th case is drawn
    from an independent stream seeded by (seed0, i), so any shard can
    generate any index without generating its predecessors."""
    rng = np.random.default_rng([seed0, int(i)])
    v = LO + RG * rng.random(N_PAR)
    for j in range(3):
        if abs(v[j]) < min_speed:
            v[j] = np.copysign(min_speed, v[j])
    return canon(v)


def truth_margins(tc, kmax=4):
    """Certificate-relevant margins of a ground-truth case, for phase
    diagrams: min signed pair separation, min lattice-relation gap over
    |k|_1 <= kmax, min |wedge tilt| (deg), min cycles at the standard T."""
    from itertools import product as _iprod
    sp = np.asarray(tc[:3], float)
    pair = min(min(abs(sp[i] - sp[j]), abs(sp[i] + sp[j]))
               for i in range(3) for j in range(i + 1, 3))
    gap = np.inf
    for j in range(3):
        ej = np.zeros(3); ej[j] = 1.0
        for k in _iprod(range(-kmax, kmax + 1), repeat=3):
            ka = np.array(k, float)
            o = np.abs(ka).sum()
            if o == 0 or o > kmax or np.array_equal(ka, ej) \
                    or np.array_equal(ka, -ej):
                continue
            gap = min(gap, abs(ka @ sp - sp[j]))
    return {'pair_sep': float(pair), 'rel_gap': float(gap),
            'ax_min': float(np.min(np.abs(tc[3:6]))),
            'cyc': float(np.min(np.abs(sp)) * T_OBS)}
