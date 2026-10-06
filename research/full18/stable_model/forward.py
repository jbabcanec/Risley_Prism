"""Stable evaluation of the ORIGINAL independent-axis Risley optical model.

Standalone numpy implementation; no imports or edits of the Dropbox project.
This does not implement full-vector 3-D Snell optics. Each transverse axis
is refracted independently, exactly as in the repository's mathematical model.

The default time grid is constructed by the same numpy.arange expression as
core.fast_forward. Supplying times preserves the exact supplied binary64 values.
No physical parameter is fixed, estimated, sorted, clipped or canonicalized.
Strictly invalid optical branches raise rather than producing clipped rays.
"""
from __future__ import annotations
import numpy as np


class PhysicalBranchError(ValueError):
    def __init__(self, branch, axis, interface, index, value):
        self.branch, self.axis, self.interface = branch, axis, interface
        self.index, self.value = int(index), float(value)
        super().__init__(f"{branch} at axis={axis}, interface={interface}, "
                         f"sample={index}: {value}")


def default_times(n_points=200, time_limit=10.):
    """Match core.fast_forward's floating time grid, without rationalizing it."""
    if not isinstance(n_points, (int, np.integer)) or n_points < 1:
        raise ValueError("n_points must be a positive integer")
    if not np.isfinite(time_limit) or time_limit <= 0:
        raise ValueError("time_limit must be positive and finite")
    return np.arange(0, time_limit, time_limit/n_points)[:n_points]


def _evaluate(speeds, wedge, phase, glass, source_distance, thickness,
              gap, distance, beam_angle, source_pos, n_points, time_limit,
              times, return_diagnostics):
    arrays=[np.asarray(x,dtype=np.float64) for x in (speeds,wedge,phase,glass)]
    speeds,wedge,phase,glass=arrays
    pcount=len(speeds)
    if pcount < 1 or any(x.shape!=(pcount,) or not np.isfinite(x).all() for x in arrays):
        raise ValueError("Expected equal nonempty finite prism vectors")
    if np.any(glass <= 0):
        raise ValueError("Refractive indices must be positive")
    beam_angle=np.asarray(beam_angle,dtype=np.float64)
    source_pos=np.asarray(source_pos,dtype=np.float64)
    if beam_angle.shape!=(2,) or source_pos.shape!=(2,):
        raise ValueError("Expected two source angles and positions")
    geometry=np.r_[source_distance,thickness,gap,distance,beam_angle,source_pos]
    if not np.isfinite(geometry).all():
        raise ValueError("Geometry must be finite")
    if times is None:
        times=default_times(n_points,time_limit)
    else:
        times=np.asarray(times,dtype=np.float64)
        if times.ndim!=1 or not len(times) or not np.isfinite(times).all():
            raise ValueError("times must be a nonempty finite vector")

    # For native |ax|<=18 degrees, cos(ax)>0. Retaining u,w avoids introducing
    # this as an additional API restriction beyond the model's tan chart.
    angle=2*np.pi*speeds[:,None]*times + phase[:,None]*(np.pi/180.)
    tilt=np.tan(wedge[:,None]*(np.pi/180.))
    u=np.cos(angle)*tilt
    w=np.sin(angle)*tilt
    norm=np.sqrt(1+u*u+w*w)
    heights=[float(source_distance)]
    for i in range(pcount):
        heights.append(heights[-1]+float(thickness))
        if i<pcount-1:
            heights.append(heights[-1]+float(gap))
    heights.append(heights[-1]+float(distance))
    ratios=np.stack((1/glass,glass),axis=1).reshape(-1)
    result=np.empty((len(times),2))
    margins={"tir":np.inf,"fwd":np.inf,"graze":np.inf}
    smallest_transverse=dict(value=np.inf)

    def guard(values, branch, axis, face, absolute=False):
        values=np.asarray(values)
        checked=np.abs(values) if absolute else values
        idx=int(np.argmin(checked))
        value=float(checked[idx])
        if not np.isfinite(checked).all() or value<=0:
            raise PhysicalBranchError(branch,axis,face,idx,value)
        margins[branch]=min(margins[branch],value)

    for axis,(primary,secondary) in enumerate(((u,w),(w,u))):
        q=np.sqrt(1+secondary*secondary)
        sins=np.zeros((2*pcount,len(times)))
        coss=np.ones_like(sins)
        slope=np.zeros((2*pcount+1,len(times)))
        sins[1::2]=primary/norm
        coss[1::2]=q/norm
        slope[1:2*pcount:2]=primary/q
        tangent=np.tan(beam_angle[axis]*(np.pi/180.))
        incoming_norm=np.sqrt(1+tangent*tangent)
        dx=np.full(len(times),tangent/incoming_norm)
        dz=np.full(len(times),1/incoming_norm)
        position=np.full(len(times),source_pos[axis]+source_distance*tangent)
        z=np.full(len(times),heights[0])
        for face in range(2*pcount):
            # Normalize the incoming homogeneous vector, without recovering
            # an angle. At finite precision this also prevents norm drift.
            ray_norm=np.hypot(dx,dz)
            ix,iz=dx/ray_norm,dz/ray_norm
            sn,cs=sins[face],coss[face]
            nr=ratios[face]
            cy=-cs*ix-sn*iz
            rad=1-(nr*cy)**2
            guard(rad,"tir",axis,face)
            root=np.sqrt(rad)
            dx=-nr*cs*cy-sn*root
            dz=-nr*sn*cy+cs*root
            guard(dz,"fwd",axis,face)
            small=int(np.argmin(abs(dx)))
            if abs(dx[small])<smallest_transverse["value"]:
                smallest_transverse=dict(value=float(abs(dx[small])),signed=float(dx[small]),
                    axial=float(dz[small]),axis=axis,interface=face,sample=small)
            # The old normalized grazing margin is 1-m*(dx/dz). Divide only
            # for diagnostics; position propagation uses the homogeneous form.
            denominator=dz-slope[face+1]*dx
            guard(denominator/dz,"graze",axis,face,absolute=True)
            lam=(heights[face+1]+slope[face+1]*position-z)/denominator
            position=position+lam*dx
            # Evaluate the known plane equation to avoid accumulating axial
            # intersection drift across interfaces.
            z=heights[face+1]+slope[face+1]*position
        result[:,axis]=position
    if return_diagnostics:
        return result,dict(margins=margins,smallest_transverse=smallest_transverse,
                           timestamps=times.copy(),strict_physical_branch=True)
    return result


def vec2pat_stable(theta,n_points=200,time_limit=10.,*,times=None,return_diagnostics=False):
    """The unchanged native 18-vector convention; all 18 values are inputs."""
    v=np.asarray(theta,dtype=np.float64)
    if v.shape!=(18,) or not np.isfinite(v).all():
        raise ValueError("Expected one finite native 18-vector")
    return _evaluate(v[:3],v[3:6],v[6:9],v[9:12],6.,3.,v[13],v[12],
                     v[14:16],v[16:18],n_points,time_limit,times,return_diagnostics)


def fast_forward_stable(params,n_points=200,time_limit=10.,*,times=None,return_diagnostics=False):
    """Drop-in duck-typed forward API for core.PrismParameters, strict branch.

    Existing callers can use this function without changing parameters or
    geometry objects. Nonphysical clipped-core trajectories are deliberately
    refused, as in the repository's certified mathematical model.
    """
    g=params.geometry
    return _evaluate(params.rotation_speeds,params.wedge_angles_x,
        params.wedge_angles_y,params.glass_indices,g.source_distance,
        g.prism_thickness,g.inter_prism_gap,g.workpiece_distance,
        [g.beam_angle_x,g.beam_angle_y],[g.beam_pos_x,g.beam_pos_y],
        n_points,time_limit,times,return_diagnostics)
