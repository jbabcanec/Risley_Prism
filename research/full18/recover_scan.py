"""Observation-only full18 candidate recovery on the original passive protocol.

Frozen combined policy: pencil/all-six-orders, then FFT/multi-basis on failure.
No supplied nominal hardware. Optional eta is a user-supplied hard coordinate
allowance, in native position units, never a default precision assumption.
Outputs are candidates and local diagnostics, not global accuracy certificates.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys
sys.dont_write_bytecode=True
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
WORK=Path(__file__).resolve().parent
sys.path.insert(0,str(WORK/'solver'))
sys.path.insert(0,str(WORK/'structure'))
from poc3 import solve18,FitConfig,canonical,forward,LO,HI,NAMES
from diagnostics import full_jacobian
from risley_lattice.fmodel import forward_point

def recover(y,eta=None,pencil_seconds=45.,fft_seconds=90.):
    y=np.asarray(y,float)
    if y.shape!=(200,2) or not np.isfinite(y).all():raise ValueError('Require exactly 200 finite x,y samples at t=k/20.')
    if eta is not None and (eta<0 or not np.isfinite(eta)):raise ValueError('eta must be a nonnegative finite hard position-error allowance.')
    began=time.perf_counter();stages=[]
    for label,cfg in [('pencil',FitConfig(max_seconds=pencil_seconds)),('fft',FitConfig(max_seconds=fft_seconds,frontend='fft',alternate_bases=5,screen_evaluations=16,finalists=3,polish_evaluations=700))]:
        r=solve18(y,eta=eta or 0.,config=cfg);r['policy_stage']=label;stages.append(r)
        if r.get('status')=='candidate_fit':break
    valid=[r for r in stages if r.get('theta') is not None]
    if not valid:return {'status':'unresolved','stages':stages,'global_accuracy_certified':False}
    # Only full-record residual, never parameter truth, selects an answer.
    best=min(valid,key=lambda r:r['max_residual']);theta=np.asarray(best['theta'])
    j=full_jacobian(theta);pinv=np.linalg.pinv(j);gains=np.sum(np.abs(pinv),axis=1)
    result={'status':'candidate_fit' if best['status']=='candidate_fit' else 'unresolved',
        'theta':theta.tolist(),'names':NAMES,'parameters':dict(zip(NAMES,theta.tolist())),
        'max_residual':best['max_residual'],'rms':best['rms'],'eta':eta,
        'canonical_hard_band_compatible':None if eta is None else best['max_residual']<=eta,
        'model':'reduced independent-axis model; source6 thickness3 common-gap P3',
        'times':'exact protocol k/20, k=0..199; original source uses binary64 timestamps',
        'units':{'speeds':'Hz','angles':'degrees','indices':'dimensionless','lengths':'native position units, not assumed mm'},
        'bounds_unchanged':True,'all18_unknown':True,'global_accuracy_certified':False,
        'complete_compatible_set_enumerated':False,'selected_stage':best['policy_stage'],
        'local_linf_noise_amplification':dict(zip(NAMES,gains.tolist())),
        'local_diagnostic_note':'Pseudoinverse row L1 norms only; neither finite-noise upper bounds nor global exclusion.',
        'seconds':time.perf_counter()-began,'stages':stages,
        'solver_source_sha256':hashlib.sha256((WORK/'solver'/'poc3.py').read_bytes()).hexdigest(),
        'observation_sha256':hashlib.sha256(y.tobytes()).hexdigest()}
    try:
        fm,fr,m=forward_point(theta);bound=float(np.max(np.abs(fm-y.ravel())+fr))
        result['strict_model_point_enclosure']={'physical_margins':m,'max_record_error_upper':bound,
          'hard_band_verified_under_ivx_contract':None if eta is None else bound<=eta,
          'scope':'endpoint feasibility under ivx documented arithmetic assumptions, not parameter accuracy'}
    except Exception as exc:result['strict_model_point_enclosure']={'error':repr(exc)}
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);src=p.add_mutually_exclusive_group(required=True)
    src.add_argument('--csv');src.add_argument('--observations');p.add_argument('--id');p.add_argument('--eta',type=float);p.add_argument('--out',required=True)
    a=p.parse_args()
    if a.csv:y=np.loadtxt(a.csv,delimiter=',',skiprows=1)
    else:
        payload=json.loads(Path(a.observations).read_text());row=next(r for r in payload['cases'] if r['id']==a.id);y=np.asarray(row['observed'])
    result=recover(y,a.eta);result['id']=a.id;result['actual_observations']=np.asarray(y).tolist()
    Path(a.out).parent.mkdir(parents=True,exist_ok=True);Path(a.out).write_text(json.dumps(result,indent=2))
    print(json.dumps({k:result.get(k) for k in ('id','status','parameters','max_residual','eta','canonical_hard_band_compatible','seconds','selected_stage','global_accuracy_certified')},indent=2))

if __name__=='__main__':main()
