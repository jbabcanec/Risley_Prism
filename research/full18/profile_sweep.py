"""Observation-only alternative search for each finite-noise frozen result.

No hidden truths, no interpretation of failed profiling as uniqueness.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,sys
sys.dont_write_bytecode=True
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from profile_observation import run,point_check
import numpy as np
WORK=Path(__file__).resolve().parent

def job(path):
    source=Path(path);data=json.loads(source.read_text());v=np.array(data['theta']);y=np.array(data['actual_observations']);eta=data['noise_amplitude']
    label=source.parent.name+'_'+data['id']+('_adapted' if 'minimax' in source.name else '')
    check=point_check(v,y,eta)
    if not check.get('compatible_under_ivx_contract'):
        report={'eta':eta,'id':data['id'],'candidate_check':check,'status':'optimizer_unresolved_no_compatible_start','actual_record_pair_witness':None,'complete_compatible_set_enumerated':False}
    else:
        report=run(v,y,eta);report['status']='compatible_pair_found' if report['actual_record_pair_witness'] else 'no_pair_found_local_search_only'
    report.update(source=str(source),id=data['id'],truth_inputs=False,global_coordinate_upper_bounds=None)
    target=WORK/'profile_sweep'/label;target=target.with_suffix('.json');target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(report,indent=2))
    w=report.get('actual_record_pair_witness')
    return dict(id=data['id'],eta=eta,status=report['status'],witness_coordinate=None if w is None else w['coordinate'],lower_bound=None if w is None else w['unavoidable_coordinate_error'],output=str(target))

def main():
    paths=[p for label in ('noise_1e6','noise_1e4','noise_1e3','noise_1e2') for p in sorted((WORK/'combined_validation'/label).glob('*.json'))]
    paths+=sorted((WORK/'low_frequency/noise_1e4').glob('*_minimax.json'))
    with ProcessPoolExecutor(max_workers=2) as pool:
        results=[f.result() for f in as_completed([pool.submit(job,str(p)) for p in paths])]
        for r in sorted(results,key=lambda r:(r['eta'],r['id'],r['output'])):print(json.dumps(r),flush=True)
    (WORK/'profile_sweep/SUMMARY.json').write_text(json.dumps(results,indent=2))

if __name__=='__main__':main()
