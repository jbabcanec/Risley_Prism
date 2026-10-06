"""Check the nonzero-residual bounded VarPro Jacobian on observation-only seeds."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json
import numpy as np
from poc3 import Projected, initializations, NONLINEAR, SPAN, extract_speeds

here=Path(__file__).resolve().parent
data=json.loads((here.parent/'observations.json').read_text())
rows=[]
for row in data['cases']:
    if row['id'] not in ('moderate_unsorted','wide_unsorted','random_00'):continue
    y=np.asarray(row['observed']);speeds,_=extract_speeds(y,.05,n_gen=3)
    seed,_=next(initializations(y,speeds,.05))
    model=Projected(seed,y,.05);q=seed[NONLINEAR]
    jac=model.jac(q);r=model.fun(q).copy();base_free=model.free.tolist()
    # Test independent directions, scaling each native coordinate by its prior.
    worst=0.;details=[]
    for direction in range(4):
        d=np.sin((np.arange(len(q))+1)*(direction+1)*np.sqrt(2))
        d*=SPAN[NONLINEAR];d[:3]*=.02
        h=1e-6
        plus=model.fun(q+h*d).copy();fp=model.free.tolist()
        minus=model.fun(q-h*d).copy();fm=model.free.tolist()
        fd=(plus-minus)/(2*h);ad=jac@d
        rel=float(np.linalg.norm(fd-ad)/max(np.linalg.norm(ad),1e-30))
        details.append(dict(relative_error=rel,active_set_stable=fp==fm==base_free))
        if fp==fm==base_free:worst=max(worst,rel)
    rows.append(dict(case=row['id'],seed_residual_norm=float(np.linalg.norm(r)),
                     free_affine_columns=base_free,directions=details,
                     max_stable_relative_error=worst))
(here/'projected_jacobian_checks.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(rows,indent=2))
assert all(r['max_stable_relative_error']<2e-5 for r in rows)
