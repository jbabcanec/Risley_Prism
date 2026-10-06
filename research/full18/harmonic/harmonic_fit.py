"""Exact-model harmonic balance continuation, optional numerical proposals.

The basis is obtained from the observed record. We transform BOTH the data and
the full nonlinear optical model, so there is no free-polynomial/Fourier-tail
model substituted for the optics. Positive complementary weights retain every
sample and all 18 original unknowns. Final ranking uses the unweighted full
canonical record. No numerical fit here certifies parameter accuracy.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'solver'))
import time
import numpy as np
from scipy.optimize import least_squares,lsq_linear
from poc3 import (affine,canonical,Projected,WorkLimit,LINEAR,NONLINEAR,
                  LO,HI,SPAN,initializations,extract_speeds)
from risley_lattice.lattice import kset


def lattice_subspace(speeds,count=200,dt=.05,degree=1):
    """Real sample subspace, including DC, from observed signed generators.

    SVD removes nearly dependent columns only in the OPTIONAL projector;
    positive complementary weights keep the complete optical fitting map.
    This is a finite-time projection, not an estimate of exact torus jets.
    """
    frequencies=kset(3,degree)@np.asarray(speeds)
    fs=1/dt
    frequencies=np.abs((frequencies+fs/2)%fs-fs/2)
    frequencies=np.unique(np.round(frequencies,12))
    frequencies=frequencies[frequencies>1e-10]
    t=np.arange(count)*dt
    argument=2*np.pi*t[:,None]*frequencies
    design=np.column_stack((np.ones(count),np.cos(argument),np.sin(argument)))
    u,s,_=np.linalg.svd(design,full_matrices=False)
    keep=s>1e-10*s[0]
    return u[:,keep],dict(degree=degree,rank=int(keep.sum()),
                         design_singular_values=s.tolist(),
                         frequencies=frequencies.tolist())


class HarmonicProjected(Projected):
    """Exact bounded affine VarPro in a fixed harmonic-weighted sample norm."""
    def __init__(self,seed,pattern,dt,basis,complement_weight,deadline=np.inf):
        super().__init__(seed,pattern,dt,deadline)
        self.basis=np.asarray(basis)
        self.rho=float(complement_weight)
        if self.basis.shape[0]!=len(pattern) or not np.isfinite(self.rho) or self.rho<=0:
            raise ValueError('basis length and positive complementary weight required')
        self.weighted_target=self.transform(self.target)

    def transform(self,array,sample_axis=0):
        arr=np.moveaxis(np.asarray(array),sample_axis,0)
        shape=arr.shape
        flat=arr.reshape(self.count,-1)
        weighted=self.rho*flat+(1-self.rho)*self.basis@(self.basis.T@flat)
        return np.moveaxis(weighted.reshape(shape),0,sample_axis)

    def fun(self,q):
        if time.perf_counter()>self.deadline:raise WorkLimit('harmonic work limit')
        if self.cached is not None and np.array_equal(q,self.cached):return self.residual
        v=self.seed.copy();v[NONLINEAR]=q
        b,A,guards,penalty=affine(v,self.count,self.dt)
        b=self.transform(b);A=self.transform(A)
        scaled=A*SPAN[LINEAR]
        rhs=self.weighted_target-b-A@LO[LINEAR]
        coefficients=np.linalg.lstsq(scaled,rhs,rcond=None)[0]
        active=np.zeros(4,dtype=int)
        if np.any(coefficients<0) or np.any(coefficients>1):
            fit=lsq_linear(scaled,rhs,bounds=(0,1),method='bvls',tol=1e-12)
            coefficients,active=fit.x,fit.active_mask
        v[LINEAR]=LO[LINEAR]+SPAN[LINEAR]*coefficients
        self.cached,self.v,self.design=q.copy(),v,scaled
        self.free=active==0
        self.data_residual=scaled@coefficients-rhs
        self.residual=np.r_[self.data_residual,penalty]
        self.guards=guards
        return self.residual

    def jac(self,q):
        self.fun(q);h=1e-25
        vv=np.broadcast_to(self.v,(len(NONLINEAR),18)).astype(complex).copy()
        vv[np.arange(len(NONLINEAR)),NONLINEAR]+=1j*h
        b,A,_,penalty=affine(vv,self.count,self.dt)
        db=self.transform(b.imag/h,sample_axis=1)
        dA=self.transform(A.imag/h,sample_axis=1)
        derivative=(db+dA@self.v[LINEAR]).T
        if self.free.any():
            af=self.design[:,self.free]
            inv=np.linalg.pinv(af,rcond=1e-13)
            dAf=dA[:,:,self.free]*SPAN[LINEAR][self.free]
            correction=np.einsum('kmi,m->ik',dAf,self.data_residual)
            derivative=derivative-af@(inv@derivative)-inv.T@correction
        return np.vstack((derivative,penalty.imag.T/h))


def complete(seed,pattern,*,evaluations=160,deadline=np.inf,basis=None,rho=1):
    model=(Projected(seed,pattern,.05,deadline) if basis is None else
           HarmonicProjected(seed,pattern,.05,basis,rho,deadline))
    try:
        fit=least_squares(model.fun,seed[NONLINEAR],jac=model.jac,
                          bounds=(LO[NONLINEAR],HI[NONLINEAR]),
                          x_scale=SPAN[NONLINEAR],ftol=1e-13,xtol=1e-13,
                          gtol=None,max_nfev=evaluations)
        model.fun(fit.x);nfev=fit.nfev
    except WorkLimit:
        if model.cached is None:raise
        nfev=None
    theta=model.v.copy()
    residual=canonical(theta,len(pattern)) - pattern
    return theta,dict(nfev=nfev,rms=float(np.sqrt(np.mean(residual**2))),
                      max_residual=float(np.max(np.abs(residual))),
                      physical=bool(np.min(model.guards)>0))


def solve_comparison(pattern,mode='direct',seconds=40):
    """Equal screen-evaluation-budget ablation; no saved estimates or truth.

    direct: 16 full-record evaluations per each of 18 common starts.
    harmonic: 8 degree1/rho.1 then 8 degree3/rho.3 evaluations, same starts.
    Both: full exact-model polish (160 evaluations) of best 4 candidates.
    This is a completion ablation, NOT identical to all v1 fallback budgets.
    """
    if mode not in ('direct','harmonic','emphasis'):raise ValueError('unknown mode')
    y=np.asarray(pattern,float);begin=time.perf_counter()
    N,info=extract_speeds(y,.05,n_gen=3,frontend='fft')
    result=dict(mode=mode,theta=None,certified=False,attempts=[],
                raw_sample_count=len(y),all18_free=True,known_hardware=False,
                noise_guarantee=False,spectral_speeds=None if N is None else N.tolist())
    if N is None:
        result.update(status='no_generators',seconds=time.perf_counter()-begin);return result
    deadline=time.perf_counter()+seconds
    basis1,binfo1=lattice_subspace(N,len(y),degree=1)
    basis3,binfo3=lattice_subspace(N,len(y),degree=3)
    result['subspaces']=[binfo1,binfo3]
    candidates=[]
    try:
        for seed,label in initializations(y,N,.05):
            seed=np.clip(seed,LO+1e-10,HI-1e-10)
            if mode=='harmonic':
                x,_=complete(seed,y,evaluations=8,deadline=deadline,basis=basis1,rho=.1)
                x,diag=complete(x,y,evaluations=8,deadline=deadline,basis=basis3,rho=.3)
            elif mode=='emphasis':
                # Declared second ablation: strengthen the optical information
                # omitted by a fundamental-only summary, then restore the
                # original objective before candidate ranking and completion.
                x,_=complete(seed,y,evaluations=8,deadline=deadline,basis=basis1,rho=3.)
                x,diag=complete(x,y,evaluations=8,deadline=deadline)
            else:x,diag=complete(seed,y,evaluations=16,deadline=deadline)
            result['attempts'].append(dict(stage='screen',label=label,**diag))
            if diag['physical']:candidates.append((diag['rms'],x,label))
        candidates.sort(key=lambda z:z[0]);screens=candidates[:4].copy()
        for _,seed,label in screens:
            x,diag=complete(seed,y,evaluations=160,deadline=deadline)
            result['attempts'].append(dict(stage='full_polish',label=label,**diag))
            if diag['physical']:candidates.append((diag['rms'],x,label))
    except (WorkLimit,ValueError,np.linalg.LinAlgError) as exc:
        result['stop_note']=str(exc)
    if candidates:
        _,theta,label=min(candidates,key=lambda z:z[0]);r=canonical(theta,len(y))-y
        result.update(theta=theta.tolist(),label=label,rms=float(np.sqrt(np.mean(r*r))),
                      max_residual=float(np.max(abs(r))),status='candidate')
    else:result['status']='no_physical_candidate'
    result['seconds']=time.perf_counter()-begin
    return result
