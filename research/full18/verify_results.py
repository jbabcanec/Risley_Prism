"""Independent canonical evaluator: no relabeling or truth-based selection."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys
sys.dont_write_bytecode=True
import json
import argparse
import hashlib
from pathlib import Path
import numpy as np
from contract import ROOT,WORK
from risley_lattice.model import LO,HI,NAMES,vec2pat
from risley_lattice.fmodel import forward_point

def outward_error_bound(center,radius,observed):
    # Each subtraction and subsequent addition is rounded outward. Inputs are
    # the stored binary64 endpoint center/radius and recorded observations.
    difference=np.nextafter(np.abs(center-observed),np.inf)
    return float(np.max(np.nextafter(difference+radius,np.inf)))

def evaluate(candidate, case, actual_observations=None):
    v=np.asarray(candidate,float); truth=np.asarray(case['truth']); y=np.asarray(case['observed'] if actual_observations is None else actual_observations)
    if v.shape!=(18,) or not np.all(np.isfinite(v)):raise ValueError('Expected all18 finite values')
    residual=vec2pat(v)-y
    errors=np.abs(v-truth)
    answer={'id':case['id'],'theta':v.tolist(),'native_error':dict(zip(NAMES,errors.tolist())),
        'max_native_error':float(errors.max()),'worst_coordinate':NAMES[int(errors.argmax())],
        'passes_001':bool(errors.max()<=.001),'within_original_prior':bool(np.all(v>=LO)&np.all(v<=HI)),
        'canonical_max_residual':float(np.abs(residual).max()),'canonical_axis_rms':np.sqrt(np.mean(residual**2,axis=0)).tolist(),
        'global_exclusion_proved':False,'truth_used_only_after_selection':True}
    try:
        fm,fr,margins=forward_point(v)
        answer.update(exact_model_vs_observed_interval_bound=outward_error_bound(fm,fr,y.ravel()),physical_margins=margins,strict_physical=True)
    except Exception as exc:
        answer.update(strict_physical=False,physical_error=repr(exc))
    return answer

def main():
    p=argparse.ArgumentParser();p.add_argument('--cases',default=str(WORK/'cases.json'));p.add_argument('--results',required=True);p.add_argument('--output',required=True)
    a=p.parse_args(); dataset=json.loads(Path(a.cases).read_text()); cases={c['id']:c for c in dataset['cases']}; rows=[]
    for file in sorted(Path(a.results).glob('*.json')):
        result=json.loads(file.read_text())
        if not isinstance(result,dict):continue
        name=result.get('case',result.get('id',file.stem))
        if name not in cases:continue
        v=result.get('theta',result.get('parameters',result.get('x18')))
        if v is None:continue
        try:
            actual=result.get('actual_observations')
            score=evaluate(v,cases[name],actual);score.update({'result_file':str(file),'elapsed':result.get('elapsed',result.get('seconds')),'observation_source':'result.actual_observations' if actual is not None else 'case.observed','declared_noise_amplitude':result.get('noise_amplitude')});rows.append(score)
        except Exception as exc:rows.append({'id':name,'error':repr(exc)})
    original_unchanged=all(hashlib.sha256((ROOT/k).read_bytes()).hexdigest()==v for k,v in dataset['source_hashes'].items())
    report={'original_source_hashes_unchanged':original_unchanged,'rows':rows,'passed':sum(r.get('passes_001',False) for r in rows),'evaluated':len(rows),'note':'Empirical per-coordinate verification against known synthetic truth. No global recovery guarantee.'}
    Path(a.output).write_text(json.dumps(report,indent=2));print(json.dumps({**report,'rows':[{k:v for k,v in r.items() if k in ('id','max_native_error','worst_coordinate','passes_001','canonical_max_residual','error')} for r in rows]},indent=2))

if __name__=='__main__':main()
