#!/usr/bin/env python3
"""
Reverse Risley Prism Solver - Core Module

Physical wedge prism model:
  - Each prism has TWO refractive interfaces (entry + exit) rotating in lockstep
  - Refractive indices alternate: air → glass → air → glass → air
  - Geometry (distances, thickness) is user-configurable
  - Glass indices can be known (fixed) or unknown (recovered)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SystemGeometry:
    """
    Fixed physical geometry of the Risley prism system.
    The user specifies what they know about their hardware.
    """
    source_distance: float = 6.0       # source → first prism entry face
    prism_thickness: float = 3.0       # entry face → exit face (per prism)
    inter_prism_gap: float = 6.0       # exit face → next prism entry face
    workpiece_distance: float = 100.0  # last exit face → workpiece
    beam_angle_x: float = 10.0        # initial beam angle x (degrees)
    beam_angle_y: float = 5.0         # initial beam angle y (degrees)
    beam_pos_x: float = 0.0           # initial beam position x
    beam_pos_y: float = 0.0           # initial beam position y


# Default geometry
DEFAULT_GEOMETRY = SystemGeometry()


@dataclass
class PrismParameters:
    """
    Parameters defining a system of physical wedge prisms.

    RECOVERED by the inverse solver:
      - rotation_speeds: N_i (Hz) per prism — how fast each prism spins
      - wedge_angles_x:  α_{x,i} (deg) per prism — exit-face tilt in x
      - wedge_angles_y:  α_{y,i} (deg) per prism — exit-face tilt in y
      - glass_indices:   n_{g,i} per prism — refractive index of each glass
                         (optional: can be fixed if known)

    SPECIFIED by the user:
      - n_prisms: how many physical prisms in the system
      - geometry: physical layout (distances, beam angles)
    """
    n_prisms: int
    rotation_speeds: List[float]      # Hz, one per prism
    wedge_angles_x: List[float]       # degrees, exit-face tilt per prism
    wedge_angles_y: List[float]       # degrees, exit-face tilt per prism
    glass_indices: List[float] = None # refractive index per prism glass
    geometry: SystemGeometry = None   # physical layout

    def __post_init__(self):
        if self.glass_indices is None:
            self.glass_indices = [1.5 + 0.05 * i for i in range(self.n_prisms)]
        if self.geometry is None:
            self.geometry = DEFAULT_GEOMETRY

    @property
    def n_interfaces(self) -> int:
        return 2 * self.n_prisms

    def validate(self) -> bool:
        if not (1 <= self.n_prisms <= 6):
            return False
        for arr in [self.rotation_speeds, self.wedge_angles_x, self.wedge_angles_y]:
            if len(arr) != self.n_prisms:
                return False
        if any(abs(s) > 10 for s in self.rotation_speeds):
            return False
        if any(abs(a) > 45 for a in self.wedge_angles_x + self.wedge_angles_y):
            return False
        if any(n < 1.0 or n > 3.0 for n in self.glass_indices):
            return False
        return True

    def to_flat(self, include_glass: bool = False) -> np.ndarray:
        """
        To fixed-size array.
        Without glass: [speeds(3), angles_x(3), angles_y(3)] = 9 values
        With glass:    [speeds(3), angles_x(3), angles_y(3), glass(3)] = 12 values
        """
        n = self.n_prisms
        if include_glass:
            arr = np.zeros(12)
            arr[:n] = self.rotation_speeds
            arr[3:3+n] = self.wedge_angles_x
            arr[6:6+n] = self.wedge_angles_y
            arr[9:9+n] = self.glass_indices
        else:
            arr = np.zeros(9)
            arr[:n] = self.rotation_speeds
            arr[3:3+n] = self.wedge_angles_x
            arr[6:6+n] = self.wedge_angles_y
        return arr

    @classmethod
    def from_flat(cls, n_prisms: int, arr: np.ndarray,
                  glass_indices: List[float] = None,
                  geometry: SystemGeometry = None) -> 'PrismParameters':
        n = int(np.clip(round(n_prisms), 1, 3))
        has_glass = len(arr) > 9
        gi = arr[9:9+n].tolist() if has_glass else glass_indices
        return cls(
            n_prisms=n,
            rotation_speeds=arr[:n].tolist(),
            wedge_angles_x=arr[3:3+n].tolist(),
            wedge_angles_y=arr[6:6+n].tolist(),
            glass_indices=gi,
            geometry=geometry,
        )

    def _build_interface_model(self):
        """
        Expand prism parameters into per-interface arrays.

        Each prism i → two interfaces:
          Interface 2i   : entry face (flat, φ=0), air → glass
          Interface 2i+1 : exit face  (tilted),    glass → air

        Both rotate at speed N_i (lockstep rigid body).
        """
        N_if = self.n_interfaces
        g = self.geometry

        speeds_if  = np.zeros(N_if)
        start_phix = np.zeros(N_if)
        start_phiy = np.zeros(N_if)
        ref_ind    = np.zeros(N_if + 1)

        for i in range(self.n_prisms):
            # Entry face (flat)
            speeds_if[2*i]   = self.rotation_speeds[i]
            start_phix[2*i]  = 0.0
            start_phiy[2*i]  = 0.0

            # Exit face (tilted by wedge angle)
            speeds_if[2*i+1]  = self.rotation_speeds[i]
            start_phix[2*i+1] = self.wedge_angles_x[i]
            start_phiy[2*i+1] = self.wedge_angles_y[i]

            # Refractive indices: air → glass → air
            ref_ind[2*i]     = 1.0                     # air before entry
            ref_ind[2*i + 1] = self.glass_indices[i]   # glass inside prism

        ref_ind[N_if] = 1.0  # air after last prism

        # Distances
        int_dist = []
        for i in range(self.n_prisms):
            if i == 0:
                int_dist.append(g.source_distance)
            int_dist.append(g.prism_thickness)
            if i < self.n_prisms - 1:
                int_dist.append(g.inter_prism_gap)
        int_dist.append(g.workpiece_distance)

        return (speeds_if, start_phix, start_phiy,
                np.array(ref_ind), np.array(int_dist))


def fast_forward(params: PrismParameters,
                 n_points: int = 200,
                 time_limit: float = 10.0) -> np.ndarray:
    """
    Fast vectorised forward model for physical wedge prisms.

    Returns (n_points, 2) array of workpiece (x, y) positions.
    """
    speeds_if, sphix, sphiy, ref_ind, int_dist = params._build_interface_model()
    N_if = len(speeds_if)
    cum_dist = np.cumsum(int_dist)
    g = params.geometry

    times = np.arange(0, time_limit, time_limit / n_points)[:n_points]
    T = len(times)

    # Effective phi angles at each interface
    gamma = (360.0 * np.outer(speeds_if, times)) % 360.0
    gamma_rad = np.radians(gamma + sphiy[:, None])
    phix_tan = np.tan(np.radians(sphix))[:, None]

    n1_0 = np.cos(gamma_rad) * phix_tan
    n1_1 = np.sin(gamma_rad) * phix_tan
    n1_norm = np.sqrt(n1_0**2 + n1_1**2 + 1.0)

    phix_eff = 90.0 - np.degrees(np.arccos(np.clip(n1_0 / n1_norm, -1, 1)))
    phiy_eff = 90.0 - np.degrees(np.arccos(np.clip(n1_1 / n1_norm, -1, 1)))

    phix_all = np.vstack([phix_eff, np.zeros((1, T))])
    phiy_all = np.vstack([phiy_eff, np.zeros((1, T))])

    x = _trace_axis(g.beam_pos_x, g.beam_angle_x, phix_all, cum_dist, ref_ind)
    y = _trace_axis(g.beam_pos_y, g.beam_angle_y, phiy_all, cum_dist, ref_ind)

    return np.column_stack([x, y])


# ─── Internal ray-tracing (unchanged) ────────────────────────────────────────

def _trace_axis(r0, theta0, phi_all, cum_dist, ref_ind):
    N_if, T = phi_all.shape
    tan_t0 = np.tan(np.radians(theta0))
    p3, z3, p4, z4 = _surface_endpoints(phi_all[0], cum_dist[0])
    p_cur, pz_cur = _intersect(r0, 0.0, r0 + tan_t0, 1.0, p3, z3, p4, z4)
    theta_cur = np.full(T, theta0, dtype=np.float64)

    for i in range(N_if - 1):
        phi_i = phi_all[i]
        tan_phi = np.tan(np.radians(phi_i))
        Nn = np.sqrt(tan_phi**2 + 1.0)
        Nh0, Nh2 = tan_phi / Nn, -1.0 / Nn

        tan_th = np.tan(np.radians(theta_cur))
        Sn = np.sqrt(tan_th**2 + 1.0)
        Si0, Si2 = tan_th / Sn, 1.0 / Sn

        nr = ref_ind[i] / ref_ind[i + 1]
        cy = Nh2 * Si0 - Nh0 * Si2
        sq = np.maximum(0.0, 1.0 - nr**2 * cy**2)
        sqv = np.sqrt(sq)

        sf0 = nr * Nh2 * cy - Nh0 * sqv
        sf2 = -nr * Nh0 * cy - Nh2 * sqv

        sf_norm = np.sqrt(sf0**2 + sf2**2 + 1e-30)
        cos_a = np.clip(sf2 / sf_norm, -1, 1)
        theta_cur = np.where(
            np.abs(sf0) < 1e-12, 0.0,
            np.sign(sf0) * np.degrees(np.arccos(cos_a)))

        tan_new = np.tan(np.radians(theta_cur))
        p3, z3, p4, z4 = _surface_endpoints(phi_all[i + 1], cum_dist[i + 1])
        p_cur, pz_cur = _intersect(
            p_cur, pz_cur, p_cur + tan_new, pz_cur + 1.0, p3, z3, p4, z4)

    return p_cur


def _surface_endpoints(phi, z0):
    flat = np.abs(phi) < 1e-10
    tan_p = np.tan(np.radians(np.where(flat, 1.0, phi)))
    p4 = np.where(flat, 1.0, 1.0 / tan_p)
    z4 = z0 + np.where(flat, 0.0, 1.0)
    return 0.0, z0, p4, z4


def _intersect(p1, z1, p2, z2, p3, z3, p4, z4):
    d = (p1 - p2) * (z3 - z4) - (z1 - z2) * (p3 - p4)
    d = np.where(np.abs(d) < 1e-15, np.copysign(1e-15, d + 1e-30), d)
    pn = ((p1*z2 - z1*p2) * (p3 - p4) - (p1 - p2) * (p3*z4 - z3*p4)) / d
    zn = ((p1*z2 - z1*p2) * (z3 - z4) - (z1 - z2) * (p3*z4 - z3*p4)) / d
    return pn, zn
