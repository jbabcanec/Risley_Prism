#!/usr/bin/env python3
"""
diag_lattice.py -- Ceiling diagnostic for the lattice-VarPro approach.

For each battery case, project z(t) onto the TRUE lattice (generators = true
speeds) at order B=3 and B=4: the relative LS residual is the best any
lattice-model solver could do. Also detect non-smooth forward-model events
(TIR clipping via sq=max(0,.), near-parallel intersections) that break the
quasi-periodic model class.

Run: python paper/diag_lattice.py
"""
import sys, os
from itertools import product as iproduct
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'reverse_problem_v2'))
sys.path.insert(0, os.path.dirname(__file__))

from ml_staged_solver import N_PAR, LO, HI, RG, canon, vec2pat, T_PTS, T_OBS
from core import PrismParameters, SystemGeometry

DT = T_OBS / T_PTS


def kset(ngen, B):
    ks = [k for k in iproduct(range(-B, B + 1), repeat=ngen)
          if sum(abs(x) for x in k) <= B]
    return np.array(ks, float)


def true_lattice_resid(pat, sp, B):
    t = np.arange(len(pat)) * DT
    z = pat[:, 0] + 1j * pat[:, 1]
    K = kset(3, B)
    E = np.exp(2j * np.pi * np.outer(t, K @ sp))
    c, *_ = np.linalg.lstsq(E, z, rcond=1e-8)
    r = z - E @ c
    z0 = z - z.mean()
    return np.linalg.norm(r) / np.linalg.norm(z0)


def clip_stats(v):
    """Count TIR-clip activations (sq<=0) along the trace, replicating core."""
    geo = SystemGeometry(source_distance=6.0, prism_thickness=3.0,
                         workpiece_distance=float(v[12]), inter_prism_gap=float(v[13]),
                         beam_angle_x=float(v[14]), beam_angle_y=float(v[15]),
                         beam_pos_x=float(v[16]), beam_pos_y=float(v[17]))
    pr = PrismParameters(3, v[:3].tolist(), v[3:6].tolist(), v[6:9].tolist(),
                         glass_indices=v[9:12].tolist(), geometry=geo)
    speeds_if, sphix, sphiy, ref_ind, int_dist = pr._build_interface_model()
    times = np.arange(0, T_OBS, T_OBS / T_PTS)[:T_PTS]
    gamma = (360.0 * np.outer(speeds_if, times)) % 360.0
    gamma_rad = np.radians(gamma + sphiy[:, None])
    phix_tan = np.tan(np.radians(sphix))[:, None]
    n1_0 = np.cos(gamma_rad) * phix_tan
    n1_1 = np.sin(gamma_rad) * phix_tan
    n1_norm = np.sqrt(n1_0**2 + n1_1**2 + 1.0)
    phix_eff = 90.0 - np.degrees(np.arccos(np.clip(n1_0 / n1_norm, -1, 1)))
    phiy_eff = 90.0 - np.degrees(np.arccos(np.clip(n1_1 / n1_norm, -1, 1)))

    nclip = 0
    for phi_all, th0 in ((np.vstack([phix_eff, np.zeros((1, T_PTS))]), geo.beam_angle_x),
                         (np.vstack([phiy_eff, np.zeros((1, T_PTS))]), geo.beam_angle_y)):
        theta = np.full(T_PTS, th0, float)
        for i in range(phi_all.shape[0] - 1):
            tan_phi = np.tan(np.radians(phi_all[i]))
            Nn = np.sqrt(tan_phi**2 + 1.0)
            Nh0, Nh2 = tan_phi / Nn, -1.0 / Nn
            tan_th = np.tan(np.radians(theta))
            Sn = np.sqrt(tan_th**2 + 1.0)
            Si0, Si2 = tan_th / Sn, 1.0 / Sn
            nr = ref_ind[i] / ref_ind[i + 1]
            cy = Nh2 * Si0 - Nh0 * Si2
            sq = 1.0 - nr**2 * cy**2
            nclip += int(np.sum(sq <= 0))
            sqv = np.sqrt(np.maximum(0.0, sq))
            sf0 = nr * Nh2 * cy - Nh0 * sqv
            sf2 = -nr * Nh0 * cy - Nh2 * sqv
            sf_norm = np.sqrt(sf0**2 + sf2**2 + 1e-30)
            cos_a = np.clip(sf2 / sf_norm, -1, 1)
            theta = np.where(np.abs(sf0) < 1e-12, 0.0,
                             np.sign(sf0) * np.degrees(np.arccos(cos_a)))
    return nclip


if __name__ == '__main__':
    rng = np.random.default_rng(2026)
    cases = []
    for _ in range(30):
        v = (LO + RG * rng.random(N_PAR).astype(np.float32)).astype(np.float64)
        for j in range(3):
            if abs(v[j]) < 0.15: v[j] = np.copysign(0.15, v[j])
        cases.append(canon(v))

    print(f"\n{'#':>3} {'resid_B3':>9} {'resid_B4':>9} {'maxstep':>8} {'clip':>5} "
          f"{'pat_std':>8}   notes")
    for ci, tc in enumerate(cases):
        pat = vec2pat(tc)
        r3 = true_lattice_resid(pat, tc[:3], 3)
        r4 = true_lattice_resid(pat, tc[:3], 4)
        step = float(np.max(np.abs(np.diff(pat, axis=0))))
        std = float(np.std(pat - pat.mean(0)))
        nclip = clip_stats(tc)
        note = ""
        if nclip: note += f" TIR-CLIP x{nclip}"
        if r4 > 1e-2: note += " MODEL-INADEQUATE"
        elif r4 > 1e-3: note += " marginal"
        print(f"{ci+1:>3} {r3:>9.1e} {r4:>9.1e} {step:>8.2f} {nclip:>5} "
              f"{std:>8.2f}  {note}", flush=True)
