"""Check full-domain coverage, native endpoints, and the reduced Jacobian."""
import json,time
from pathlib import Path
import numpy as np
from full_domain_gain import FullGain,LO,HI,NONLINEAR,TAN18,GAIN_CAP,WORK,HERE

rng=np.random.default_rng(20261003);max_roundtrip=0.;max_constraint=0.
for _ in range(200):
    x=LO+(HI-LO)*rng.random(18)
    q=x[NONLINEAR].copy();q[3:6]=(x[9:12]-1)*np.tan(np.deg2rad(x[3:6]))
    decoded,_=FullGain.expand(q)
    max_roundtrip=max(max_roundtrip,float(np.max(abs(decoded-x[NONLINEAR]))))
    max_constraint=max(max_constraint,float(np.max(abs(q[3:6])-(q[9:12]-1)*TAN18)))
assert max_roundtrip<2e-13 and max_constraint<=1e-14
edge=(LO+HI)/2;edge[3:6]=[18,-18,18];edge[9:12]=1.8
qe=edge[NONLINEAR].copy();qe[3:6]=(edge[9:12]-1)*np.tan(np.deg2rad(edge[3:6]))
decoded,_=FullGain.expand(qe)
assert np.max(abs(decoded[3:6]-edge[3:6]))<1e-13
assert np.min(abs(qe[3:6]))>.3*TAN18

seed=np.array(json.loads((HERE/'result_native.json').read_text())['theta'])
y=np.array(next(c for c in json.loads((WORK/'profiles/adverse_observations_only.json').read_text())['cases'] if c['id']=='eta_1e-8_separated_speeds')['observed'])
f=FullGain(seed,y,time.perf_counter()+30,HERE/'validation_progress.json')
q=f.q0;J=f.jac(q)[:400]*f.span
FD=np.empty_like(J)
for j in range(14):
    h=1e-7;plus=q.copy();minus=q.copy();plus[j]+=h*f.span[j];minus[j]-=h*f.span[j]
    FD[:,j]=(f.fun(plus)[:400]-f.fun(minus)[:400])/(2*h)
scale=np.maximum(np.max(abs(J),axis=0),1e-8)
relative=np.max(abs(J-FD),axis=0)/scale
assert np.max(relative)<1e-4
report={'random_native_samples':200,'max_native_roundtrip_error':max_roundtrip,'coupled_constraint_violation':max_constraint,
        'signed_native_wedge_endpoints_covered':True,'high_index_gain_outside_old_weak_rectangle_covered':True,
        'gain_cap':GAIN_CAP,'jacobian_central_difference_step_in_scaled_coordinates':1e-7,
        'relative_max_derivative_error_by_nonlinear_column':relative.tolist(),'jacobian_max_relative_error':float(np.max(relative)),
        'finite_difference_is_numerical_check_not_formal_proof':True,'status':'PASS'}
(HERE/'validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
