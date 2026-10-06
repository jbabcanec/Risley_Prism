"""Profile actual data around the observation-only gain continuation answer.

No truth or constructed ambiguity endpoint is input. Shifts are selected from
the requested .001 native target, rather than any generating hardware value.
"""
import os,sys,json,time,hashlib
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;WORK=HERE.parent
sys.path.insert(0,str(WORK))
from profile_observation import profile,point_check
from poc3 import NAMES

candidate=HERE/'result_native.json';observations=WORK/'profiles/adverse_observations_only.json'
v=np.array(json.loads(candidate.read_text())['theta'])
y=np.array(next(c for c in json.loads(observations.read_text())['cases'] if c['id']=='eta_1e-8_separated_speeds')['observed'])
eta=1e-8;start=time.perf_counter();rows=[]
for j,amount in ((12,.0011),(13,.0005),(11,2.5e-6)):
    for sign in (-1,1):
        result=profile(v,y,eta,j,sign*amount);rows.append(result)
        print(json.dumps({'coordinate':NAMES[j],'shift':sign*amount,'check':result['check']}),flush=True)
accepted=[r for r in rows if r['check'].get('compatible_under_ivx_contract')]
endpoints=[v]+[np.array(r['theta']) for r in accepted]
matrix=np.array(endpoints)
span=np.max(matrix,axis=0)-np.min(matrix,axis=0)
report={'input_sha256':{candidate.name:hashlib.sha256(candidate.read_bytes()).hexdigest(),observations.name:hashlib.sha256(observations.read_bytes()).hexdigest()},
        'candidate':v.tolist(),'candidate_check':point_check(v,y,eta),'eta':eta,'names':NAMES,
        'profiles':rows,'accepted_profiles':len(accepted),'observed_compatible_coordinate_spans':span.tolist(),
        'minimax_error_lower_bounds':(span/2).tolist(),'truth_inputs_used':False,
        'complete_compatible_set_enumerated':False,'scope':'found compatible points give lower uncertainty bounds, not an enclosing set or uniqueness result',
        'seconds':time.perf_counter()-start}
(HERE/'compatible_profiles.json').write_text(json.dumps(report,indent=2))
print(json.dumps({'accepted':len(accepted),'seconds':report['seconds'],'dw_span':span[12],'gap_span':span[13]}))
