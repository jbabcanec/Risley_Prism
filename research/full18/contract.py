"""Freeze a passive full18 benchmark. No solver or truth-derived initialization.

The generator samples the entire native rectangular prior without sorting,
clipping small speeds, changing wedges, or restricting glasses/source geometry.
Only failure of strict sampled transmission rejects a random draw. Each such
rejection is saved. This is a small development set, not a population estimate.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import hashlib
import json
import argparse
import numpy as np

ROOT = Path(r'C:\Users\josep\Dropbox\Babcanec Works\Mathematics\Wedge')
WORK = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from risley_lattice.model import LO, HI, NAMES, vec2pat
from risley_lattice.fmodel import forward_point

STRESS = {
 'moderate_unsorted': [1.7,-.93,.37,5,-4,3,4,-6,9,1.47,1.63,1.72,112,7,3,-2,.4,-.6],
 'wide_unsorted': [.53,2.1,-1.17,13,-16,11,3,-8,11,1.41,1.52,1.68,137,9,-4,6,-.8,1.2],
 'exact_collision': [1.2,1.2,-.71,6,-5,4,9,-7,2,1.44,1.71,1.59,91,5,2,-3,.7,-.3],
 'weak_first': [.37,1.7,-.93,.15,12,-9,9,4,-6,1.72,1.47,1.63,112,7,3,-2,.4,-.6],
}

def checked(theta):
    v = np.asarray(theta, float)
    if np.any(v<LO) or np.any(v>HI):
        raise ValueError('Outside native box')
    fm,fr,margins = forward_point(v)
    y = vec2pat(v)
    return y, {k:float(x) for k,x in margins.items()}, float(np.max(np.abs(y.ravel()-fm)+fr))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,default=20261002)
    parser.add_argument('--random-count',type=int,default=8)
    parser.add_argument('--output-stem',default='cases')
    parser.add_argument('--without-stress',action='store_true')
    args=parser.parse_args()
    rng=np.random.default_rng(args.seed)
    cases=[]; rejected=[]
    for draw in range(100):
        v=LO+(HI-LO)*rng.random(18)
        try:
            y,m,floor=checked(v)
        except Exception as exc:
            rejected.append({'draw':draw,'truth':v.tolist(),'reason':str(exc),'kind':'random'})
            continue
        cases.append({'id':f'random_{len(cases):02d}','kind':'random','draw':draw,'truth':v.tolist(),'observed':y.tolist(),'margins':m,'canonical_vs_interval_max_bound':floor})
        if len(cases)==args.random_count:break
    for name,v in ({} if args.without_stress else STRESS).items():
        try:y,m,floor=checked(v)
        except Exception as exc:
            rejected.append({'id':name,'truth':v,'reason':str(exc),'kind':'stress'})
            continue
        cases.append({'id':name,'kind':'stress','truth':v,'observed':y.tolist(),'margins':m,'canonical_vs_interval_max_bound':floor})
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'reverse_problem_v2/core.py',ROOT/'risley_lattice/model.py',ROOT/'risley_lattice/separable.py',ROOT/'risley_lattice/solve.py',ROOT/'risley_lattice/fmodel.py']}
    payload={'seed':args.seed,'names':NAMES,'lower':LO.tolist(),'upper':HI.tolist(),'timestamps':(np.arange(200)*.05).tolist(),'model':'canonical reduced per-axis; source distance 6, thickness 3; common gap; P=3','noise_contract':'hard per-coordinate maximum absolute observation error; tested as a sweep, not a user-prescribed eta','target_native_error':.001,'prior_shrink':False,'physical_order_sorted':False,'truth_only_for_generation_and_evaluation':True,'screen_protocol':'one passive plane, 200 timed positions; no interventions','source_hashes':hashes,'cases':cases,'rejected':rejected}
    WORK.mkdir(exist_ok=True)
    (WORK/(args.output_stem+'.json')).write_text(json.dumps(payload,indent=2),encoding='utf-8')
    observation_name='observations' if args.output_stem=='cases' else args.output_stem+'_observations'
    (WORK/(observation_name+'.json')).write_text(json.dumps({'timestamps':payload['timestamps'],'names':NAMES,'lower':LO.tolist(),'upper':HI.tolist(),'cases':[{'id':c['id'],'observed':c['observed']} for c in cases]},indent=2),encoding='utf-8')
    print(json.dumps({'accepted':len(cases),'rejected':len(rejected),'cases':[{'id':c['id'],'speeds':c['truth'][:3],'wedges':c['truth'][3:6],'margins':c['margins'],'data_floor_bound':c['canonical_vs_interval_max_bound']} for c in cases]},indent=2))

if __name__=='__main__':main()
