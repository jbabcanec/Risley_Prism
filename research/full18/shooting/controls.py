"""Model parity and directional derivative controls; no synthetic truth used."""
import json
import numpy as np
from phase_blocks import HERE,WORK,poc3,affine_blocks,BlockProjected,NONLINEAR

data=json.loads((WORK/'observations.json').read_text())
y=np.asarray(next(r['observed'] for r in data['cases'] if r['id']=='moderate_unsorted'))
N,_=poc3.extract_speeds(y,.05)
v,_=next(poc3.initializations(y,N,.05));blocks=8
a,b,guards,penalty=affine_blocks(v,np.zeros(21),blocks=blocks)
a0,b0,guards0,penalty0=poc3.affine(v)
parity=max(float(np.max(abs(a-a0))),float(np.max(abs(b-b0))),float(np.max(abs(guards-guards0))))
assert parity<1e-12,parity
obj=BlockProjected(v,y,.1,blocks=blocks)
q=np.r_[v[NONLINEAR],np.zeros(21)]
direction=np.random.default_rng(82104).normal(size=len(q));direction/=np.linalg.norm(direction)
J=obj.jac(q);h=1e-5
fd=(obj.fun(q+h*direction)-obj.fun(q-h*direction))/(2*h)
relative=float(np.linalg.norm(fd-J@direction)/max(np.linalg.norm(fd),1e-12))
assert relative<2e-4,relative
# A nonzero block phase changes only temporary block response. Dropping all
# slips restores exactly the original common-clock model checked above.
slips=np.zeros(21);slips[0]=.02
changed,changed_design,_,_=affine_blocks(v,slips,blocks=blocks)
delta=(changed+changed_design@v[poc3.LINEAR])-(a+b@v[poc3.LINEAR])
changed_rows=np.flatnonzero(abs(delta)>1e-10)//2
assert np.all((changed_rows>=25)&(changed_rows<50))
result={'zero_slip_model_parity':parity,'reduced_jacobian_relative_directional_error':relative,
        'single_block_phase_locality_pass':True,'truth_used':False,'all_controls_passed':True}
(HERE/'controls.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
