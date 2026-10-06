"""Reflection-tied, six-rotor-coordinate harmonic initializer primitives.

This is a finite surrogate, not the exact optical forward model. It imports
no source project and reads no observations or parameter truths on import.
No truncation-error or full18 recovery certificate is claimed by this code.
"""
from __future__ import annotations
from math import factorial
import numpy as np


def indices(degree=5):
    return np.asarray([(a,b,c) for d in range(degree+1)
                       for a in range(d+1) for b in range(d-a+1)
                       for c in [d-a-b]], dtype=int)


def dictionary(rotor, times, degree=5, axis=0, derivatives=False):
    """rotor=(N1,N2,N3,phase1_deg,phase2_deg,phase3_deg).

    axis=0 uses cos(gamma); axis=1 uses sin(gamma). The column indexed by
    h is product_i cos(h_i*(gamma_i-axis*pi/2)). Derivatives are with
    respect to native Hz and degrees. Division by cos is never used.
    """
    rotor=np.asarray(rotor, dtype=float)
    times=np.asarray(times, dtype=float)
    if rotor.shape!=(6,) or axis not in (0,1):
        raise ValueError("Expected six rotor coordinates and axis 0 or 1")
    h=indices(degree)
    gamma=2*np.pi*times[:,None]*rotor[:3]+np.deg2rad(rotor[3:])-axis*np.pi/2
    arguments=gamma[:,None,:]*h[None,:,:]
    cs=np.cos(arguments)
    phi=np.prod(cs,axis=2)
    if not derivatives:
        return phi
    deriv=np.empty((6,len(times),len(h)))
    for j in range(3):
        other=[i for i in range(3) if i!=j]
        d=-h[None,:,j]*np.sin(arguments[:,:,j])*np.prod(cs[:,:,other],axis=2)
        deriv[j]=d*(2*np.pi*times[:,None])
        deriv[j+3]=d*(np.pi/180)
    return phi,deriv


def variable_projection(rotor, times, observations, degree=5, rcond=1e-12):
    """Separate real optical coefficients; retain all 400 sample residuals.

    The Jacobian is the exact residual-dependent VarPro expression on
    full-column-rank neighborhoods. At a truncated rank change it is only a
    fixed-rank proposal derivative; diagnostics explicitly report that case.
    Rotor signs/order and physical hardware are not identified by this free
    coefficient surrogate, even if its residual vanishes.
    """
    observations=np.asarray(observations,dtype=float)
    if observations.shape!=(len(times),2):
        raise ValueError("Expected one x,y observation per supplied timestamp")
    residuals=[]; jacobians=[]; diagnostics=[]; coefficients=[]
    for axis in (0,1):
        phi,dp=dictionary(rotor,times,degree,axis,True)
        u,s,vt=np.linalg.svd(phi,full_matrices=False)
        keep=s>rcond*s[0]
        inverse=(vt[keep].T/s[keep])@u[:,keep].T
        c=inverse@observations[:,axis]
        r=phi@c-observations[:,axis]
        jp=[]
        for dphi in dp:
            action=dphi@c
            jp.append(action-phi@(inverse@action)-inverse.T@(dphi.T@r))
        residuals.append(r); jacobians.append(np.column_stack(jp)); coefficients.append(c)
        diagnostics.append(dict(axis=axis,columns=len(s),rank=int(sum(keep)),
            smallest_singular_value=float(s[-1]),largest_singular_value=float(s[0]),
            full_rank_derivative=bool(keep.all()),rcond=rcond))
    return dict(residual=np.concatenate(residuals),jacobian=np.vstack(jacobians),
                coefficients=np.asarray(coefficients),diagnostics=diagnostics)


def chebyshev_coefficient(n,r):
    """Exact integer [u**r] T_n(u); no sampled derivative approximation."""
    if n==0:
        return int(r==0)
    if r>n or (n-r)%2:
        return 0
    # Equivalent factorial formula, arranged with an even integer numerator.
    numerator=(-1)**((n-r)//2)*n*factorial((n+r)//2-1)*2**r
    denominator=2*factorial((n-r)//2)*factorial(r)
    return numerator//denominator


def jet_weights(alpha,degree=5):
    """Row converting retained Chebyshev coefficients to one Taylor jet."""
    return np.asarray([np.prod([chebyshev_coefficient(int(n),int(r))
                       for n,r in zip(h,alpha)],dtype=object)
                       for h in indices(degree)],dtype=float)


def fixed_rotor_jet_readout(rotor,times,observations,alpha,degree=5,axis=0,rcond=1e-12,
                          coefficient_bounds=None):
    """Readout plus exact hard-noise amplification in real arithmetic.

    To certify: add function-tail allowance times noise_gain and omitted
    jet-tail allowance, plus rotor uncertainty and numerical enclosure error.
    A rank-deficient or numerically truncated dictionary also needs the
    nullspace_charge. Even for full rank, validated rounding must enclose
    its computed defect. Unprovided allowances are not set to zero.
    """
    phi=dictionary(rotor,times,degree,axis)
    weights=jet_weights(alpha,degree)
    row=weights@np.linalg.pinv(phi,rcond=rcond)
    nullspace_row=weights-row@phi
    charge=None
    if coefficient_bounds is not None:
        bounds=np.asarray(coefficient_bounds,dtype=float)
        if bounds.shape!=weights.shape or np.any(bounds<0) or not np.isfinite(bounds).all():
            raise ValueError("Expected a finite nonnegative bound for each retained coefficient")
        charge=float(abs(nullspace_row)@bounds)
    return dict(value=float(row@np.asarray(observations)[:,axis]),
                noise_gain=float(np.sum(abs(row))),row=row,
                nullspace_row=nullspace_row,nullspace_charge=charge,
                certified=False,requires=["whole-torus analytic tail",
                    "omitted jet tail","rotor uncertainty","rounding enclosure",
                    "verified row reproduction or bounded nullspace charge"])
