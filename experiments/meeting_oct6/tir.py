#!/usr/bin/env python3
"""Total internal reflection (TIR): exact condition, worst case over all
rotation phases, and validation against the forward model.

Derivation (matches reverse_problem_v2/core.py::_trace_axis exactly)
-------------------------------------------------------------------
Each axis is traced in its own plane. At an interface with tilt phi and
incoming ray angle theta (both from the optical axis), the code computes
    cy = Nh2*Si0 - Nh0*Si2 = -sin(theta + phi),
    sq = 1 - (n_in/n_out)^2 * cy^2,
so the angle of incidence on the face is (theta + phi) and TIR occurs iff
    n_in/n_out * |sin(theta + phi)| > 1.
Only glass->air faces can do this (n_in/n_out = n_g > 1). In this model the
entry face is flat and the exit face carries the wedge, so for prism i:

    TIR  <=>  |theta_g,i + phi_i(t)| > theta_c,i = arcsin(1/n_i),
    theta_g,i = arcsin(sin(theta_air,i)/n_i)       (flat entry face),
    phi_i(t)  = arcsin(cos(gamma_i(t)) sin(ax_i))   (x-plane; sin for y),

so |phi_i| <= |ax_i|, with equality twice per revolution. The outgoing ray
angle arcsin(n sin(u + phi)) - phi increases in both u and phi (its
phi-derivative is n cos(u+phi)/cos(out) - 1 >= 0 because n >= 1), so the
largest angle reaching prism i+1 is obtained by aligning every upstream
extreme. That gives the worst case over all rotation phases in closed form:

    s_1     = |beam angle|
    u_i     = arcsin(sin(s_i)/n_i)
    margin_i = arcsin(1/n_i) - (u_i + |ax_i|)        [deg]
    s_{i+1} = arcsin(n_i sin(u_i + |ax_i|)) - |ax_i|

The system is TIR-free at every rotation phase iff min_i margin_i > 0 in both
planes. For incommensurate speeds the phases come arbitrarily close to the
worst alignment, so a negative margin means TIR WILL eventually occur; at a
finite T it occurs only if the phases get close enough during the record.
The bound depends on exactly the three quantities named in the meeting:
incident angle (s_i), index ratio (n_i) and wedge angle (|ax_i|).
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'reverse_problem_v2'))

D = np.pi / 180.0


def tir_margins(ax, ng, beam):
    """Per-prism worst-case margins (deg) for one plane. Arrays broadcast:
    ax, ng shape (..., P); beam shape (...). Returns (..., P)."""
    ax = np.abs(np.asarray(ax, float)) * D
    ng = np.asarray(ng, float)
    s = np.abs(np.asarray(beam, float)) * D
    P = ax.shape[-1]
    out = np.empty(ax.shape)
    for i in range(P):
        u = np.arcsin(np.clip(np.sin(s) / ng[..., i], -1, 1))
        tc = np.arcsin(1.0 / ng[..., i])
        out[..., i] = (tc - (u + ax[..., i])) / D
        arg = ng[..., i] * np.sin(u + ax[..., i])
        # past TIR the downstream angle is undefined; carry 90 deg (grazing)
        s = np.where(arg < 1.0,
                     np.arcsin(np.clip(arg, -1, 1)) - ax[..., i],
                     np.pi / 2)
    return out


def tir_margin18(v):
    """Worst-case TIR margin (deg) of an 18-vector (or (n,18) array):
    min over prisms and both planes. > 0 means TIR-free at every phase."""
    v = np.atleast_2d(v)
    ax, ng = v[:, 3:6], v[:, 9:12]
    mx = tir_margins(ax, ng, v[:, 14]).min(-1)
    my = tir_margins(ax, ng, v[:, 15]).min(-1)
    return np.minimum(mx, my)


def critical_wedge(theta_in, n):
    """Largest |wedge| a single prism tolerates for incoming air angle
    theta_in (deg) and index n: arcsin(1/n) - arcsin(sin(theta_in)/n)."""
    return (np.arcsin(1.0 / n) - np.arcsin(np.sin(theta_in * D) / n)) / D


def observed_tir(v, n_points, time_limit):
    """Ground truth from the forward model itself: number of samples at
    which any interface has 1 - (n_in/n_out)^2 sin^2(theta+phi) <= 0."""
    from core import PrismParameters, SystemGeometry
    from risley_lattice.model import SRC, THK
    geo = SystemGeometry(source_distance=SRC, prism_thickness=THK,
                         workpiece_distance=v[12], inter_prism_gap=v[13],
                         beam_angle_x=v[14], beam_angle_y=v[15],
                         beam_pos_x=v[16], beam_pos_y=v[17])
    pr = PrismParameters(3, v[:3].tolist(), v[3:6].tolist(),
                         v[6:9].tolist(), glass_indices=v[9:12].tolist(),
                         geometry=geo)
    speeds_if, sphix, sphiy, ref_ind, _ = pr._build_interface_model()
    t = np.arange(0, time_limit, time_limit / n_points)[:n_points]
    gam = np.radians((360.0 * np.outer(speeds_if, t)) % 360.0
                     + sphiy[:, None])
    tx = np.tan(np.radians(sphix))[:, None]
    n0, n1 = np.cos(gam) * tx, np.sin(gam) * tx
    nn = np.sqrt(n0 ** 2 + n1 ** 2 + 1.0)
    bad = np.zeros(len(t), bool)
    for comp, th0 in ((n0, v[14]), (n1, v[15])):
        phi = 90.0 - np.degrees(np.arccos(np.clip(comp / nn, -1, 1)))
        th = np.full(len(t), th0, float)
        for i in range(len(speeds_if)):
            nr = ref_ind[i] / ref_ind[i + 1]
            s = np.sin(np.radians(th + phi[i]))
            sq = 1.0 - nr ** 2 * s ** 2
            bad |= sq <= 0
            out = np.degrees(np.arcsin(np.clip(nr * s, -1, 1))) - phi[i]
            th = np.where(sq > 0, out, 90.0)
    return int(bad.sum())


if __name__ == '__main__':
    # quick self-check: margin > 0 must imply zero observed TIR
    from risley_lattice.model import case_at
    rng_bad = 0
    for i in range(300):
        v = case_at(i, 99)
        m = tir_margin18(v)[0]
        o = observed_tir(v, 8000, 80.0)
        if m > 0 and o > 0:
            rng_bad += 1
    print('violations of theorem (margin>0 but TIR seen):', rng_bad)
