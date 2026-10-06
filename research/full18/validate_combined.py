"""Fresh ten-case validation of the combined policy, including bounded noise."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from recover_scan import recover
WORK=Path(__file__).resolve().parent

def run(job):
    row,eta=job;y=np.asarray(row['observed'])
    if eta:
        k=np.arange(200)[:,None];axis=np.arange(2)[None,:]
        y=y+eta*np.sin((k+1)*(np.sqrt(2)+axis*np.sqrt(3)))
    try:r=recover(y,eta=eta)
    except Exception as exc:r={'status':'exception','exception':repr(exc)}
    r.update(id=row['id'],noise_amplitude=eta,actual_observations=y.tolist(),input_contains_truth=False)
    label='noiseless' if eta==0 else 'noise_'+format(eta,'.0e').replace('e-0','e').replace('e-','e')
    out=WORK/'combined_validation'/label;out.mkdir(parents=True,exist_ok=True)
    (out/(row['id']+'.json')).write_text(json.dumps(r,indent=2))
    return {k:r.get(k) for k in ('id','noise_amplitude','status','selected_stage','seconds','max_residual','exception')}

def main():
    p=argparse.ArgumentParser();p.add_argument('--etas',nargs='+',type=float,default=[0.,1e-4]);p.add_argument('--workers',type=int,default=2)
    args=p.parse_args()
    rows=json.loads((WORK/'combined_holdout_observations.json').read_text())['cases']
    # Schedule every noiseless input first, then the same predefined bounded noise.
    jobs=[(row,eta) for eta in args.etas for row in rows]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for f in as_completed([pool.submit(run,j) for j in jobs]):print(json.dumps(f.result()),flush=True)

if __name__=='__main__':main()
