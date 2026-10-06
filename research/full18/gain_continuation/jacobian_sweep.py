"""Separate finite-step sweep and holomorphic normal-equation derivative check."""
import json,time
import numpy as np
from full_domain_gain import FullGain,LO,SPAN,NONLINEAR,LINEAR,WORK,HERE
from poc3 import affine

seed=np.array(json.loads((HERE/'result_native.json').read_text())['theta'])
y=np.array(next(c for c in json.loads((WORK/'profiles/adverse_observations_only.json').read_text())['cases'] if c['id']=='eta_1e-8_separated_speeds')['observed'])
f=FullGain(seed,y,time.perf_counter()+30,HERE/'jacobian_sweep_progress.json')
q=f.q0;J=f.jac(q)[:400]*f.span
assert f.projected.free.all()
scale=np.maximum(np.max(abs(J),axis=0),1e-8);rows=[]
for h in (3e-5,1e-5,3e-6,1e-6,3e-7,1e-7):
    FD=np.empty_like(J)
    for j in range(14):
        qp=q.copy();qm=q.copy();qp[j]+=h*f.span[j];qm[j]-=h*f.span[j]
        FD[:,j]=(f.fun(qp)[:400]-f.fun(qm)[:400])/(2*h)
    relative=np.max(abs(J-FD),axis=0)/scale
    rows.append({'h':h,'relative_error_by_column':relative.tolist(),'max':float(np.max(relative))})

def analytic_extension(qc):
    v=seed.astype(complex);p=qc.copy()
    p[3:6]=180/np.pi*np.arctan(qc[3:6]/(qc[9:12]-1))
    v[NONLINEAR]=p;b,A,_,_=affine(v)
    M=A*SPAN[LINEAR];rhs=y.ravel()-b-A@LO[LINEAR]
    coeff=np.linalg.solve(M.T@M,M.T@rhs)
    return M@coeff-rhs

JC=np.empty_like(J);step=1e-20
for j in range(14):
    qc=q.astype(complex);qc[j]+=1j*step*f.span[j]
    JC[:,j]=analytic_extension(qc).imag/step
complex_relative=np.max(abs(J-JC),axis=0)/scale
assert np.max(complex_relative)<1e-6
report={'finite_difference_steps':rows,'best_fd_max':min(x['max'] for x in rows),
        'complex_step':step,'complex_relative_error_by_column':complex_relative.tolist(),
        'complex_max_relative_error':float(np.max(complex_relative)),
        'active_affine_bounds':False,'complex_reference':'differentiate unconstrained holomorphic normal equations with transpose, not Hermitian transpose',
        'interpretation':'The earlier 1e-7-step weak-column discrepancy is central-difference cancellation, not evidence of a chain-rule error.','status':'PASS'}
(HERE/'jacobian_sweep.json').write_text(json.dumps(report,indent=2))
print(json.dumps({'best_fd_max':report['best_fd_max'],'complex_max_relative_error':report['complex_max_relative_error'],'status':report['status']}))
