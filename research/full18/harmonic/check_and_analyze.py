"""Observation-only algebra checks and local harmonic-information diagnostics.

Saved blind estimates are used for Jacobian diagnostics only. No truth enters,
and no local singular value is promoted to a global noise certificate.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import sys
sys.dont_write_bytecode=True
import json
from pathlib import Path
import numpy as np
from harmonic_fit import (HarmonicProjected,Projected,lattice_subspace,affine,
                          NONLINEAR,LINEAR,SPAN,initializations,extract_speeds)

here=Path(__file__).resolve().parent;work=here.parent
observations=json.loads((work/'observations.json').read_text())
observed={r['id']:np.asarray(r['observed']) for r in observations['cases']}
checks=[];information=[]
for name in ('random_00','moderate_unsorted','wide_unsorted','weak_first'):
    y=observed[name];N,_=extract_speeds(y,.05,n_gen=3,frontend='fft')
    seed,_=next(initializations(y,N,.05));q=seed[NONLINEAR]
    basis,_=lattice_subspace(N,degree=3)
    weighted=HarmonicProjected(seed,y,.05,basis,.3)
    J=weighted.jac(q);r=weighted.fun(q).copy();free0=weighted.free.copy()
    errors=[]
    for j in range(3):
        direction=np.sin((np.arange(len(q))+1)*(j+1)*np.sqrt(2))*SPAN[NONLINEAR]
        direction[:3]*=.02;h=1e-6
        plus=weighted.fun(q+h*direction).copy();freep=weighted.free.copy()
        minus=weighted.fun(q-h*direction).copy();freem=weighted.free.copy()
        fd=(plus-minus)/(2*h);ad=J@direction
        errors.append(dict(relative_error=float(np.linalg.norm(fd-ad)/np.linalg.norm(ad)),
                           stable_active=bool(np.array_equal(freep,free0) and np.array_equal(freem,free0))))
    identity=HarmonicProjected(seed,y,.05,basis,1)
    direct=Projected(seed,y,.05)
    checks.append(dict(case=name,directions=errors,
                       rho1_residual_parity=float(np.max(abs(identity.fun(q)-direct.fun(q)))),
                       rho1_jacobian_parity=float(np.max(abs(identity.jac(q)-direct.jac(q))))))
    saved=json.loads((work/'solver'/'results_v1'/(name+'.json')).read_text())
    theta=np.array(saved['theta'])
    vv=np.broadcast_to(theta,(18,18)).astype(complex).copy()
    vv[np.arange(18),np.arange(18)]+=1e-25j
    bb,aa,_,_=affine(vv)
    physicalJ=(bb+np.einsum('kmi,ki->km',aa,vv[:,LINEAR])).imag.T/1e-25
    for degree in (1,2,3,4):
        U,diag=lattice_subspace(theta[:3],degree=degree)
        W=np.zeros((2*U.shape[1],400));W[0::2,0::2]=U.T;W[1::2,1::2]=U.T
        reduced=W@physicalJ;scaled=reduced*SPAN
        singular=np.linalg.svd(scaled,compute_uv=False)
        numerical_rank=int(np.sum(singular>singular[0]*1e-11))
        tail=y-U@(U.T@y)
        rec=dict(case=name,degree=degree,scalar_features=W.shape[0],
                 exact_dimension_deficit=max(0,18-W.shape[0]),
                 numerical_rank=numerical_rank,scaled_singular_values=singular.tolist(),
                 observed_complement_max=float(np.max(abs(tail))),
                 observed_complement_rms=float(np.sqrt(np.mean(tail*tail))),
                 projection_hard_noise_amplification=float(np.max(np.sum(abs(W),axis=1))),
                 interpretation='local linear diagnostic at saved blind estimate; no global bound')
        if numerical_rank==18:
            leftinverse=SPAN[:,None]*np.linalg.pinv(scaled,rcond=1e-13)
            per_coordinate=np.sum(abs(leftinverse@W),axis=1)
            rec['native_linear_error_per_eta']=per_coordinate.tolist()
            rec['largest_native_linear_error_per_eta']=float(per_coordinate.max())
        information.append(rec)
output=dict(checks=checks,information=information,truth_read=False)
(here/'harmonic_checks.json').write_text(json.dumps(output,indent=2))
print(json.dumps({'checks':checks,'information':[{k:r[k] for k in ('case','degree','scalar_features','numerical_rank','observed_complement_rms')} for r in information]},indent=2))
assert all(r['rho1_residual_parity']<1e-10 and r['rho1_jacobian_parity']<1e-10 for r in checks)
assert all(d['relative_error']<1e-5 for r in checks for d in r['directions'] if d['stable_active'])
