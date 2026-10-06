"""Existing repeated-generator solver on observations only."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,r'C:\Users\josep\Dropbox\Babcanec Works\Mathematics\Wedge')
import json,time,numpy as np
from pathlib import Path
from risley_lattice.relations import solve18_relations
from risley_lattice.model import vec2pat
work=Path(__file__).resolve().parent
y=np.array(next(c['observed'] for c in json.loads((work.parent/'observations.json').read_text())['cases'] if c['id']=='exact_collision'))
start=time.perf_counter()
x,mse,route,info=solve18_relations(y,max_nfev=100,mse_stop=1e-24)
result=dict(seconds=time.perf_counter()-start,x18=x.tolist(),mse=mse,route=route,
            max_residual=float(np.max(abs(vec2pat(x)-y))),trials=info['trials'])
(work/'baseline_relations.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k not in ('x18','trials')}),flush=True)
