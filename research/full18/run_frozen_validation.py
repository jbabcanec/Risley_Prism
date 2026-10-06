"""Fixed-policy observation-only validation; truth scoring is a separate script."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys
sys.dont_write_bytecode=True
import json,hashlib,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
WORK=Path(__file__).resolve().parent
sys.path.insert(0,str(WORK/'solver'))

def worker(job):
    from poc3 import solve18,FitConfig
    import numpy as np
    group,row=job
    began=time.monotonic()
    try:
        r=solve18(np.asarray(row['observed']),eta=0.,config=FitConfig(max_seconds=45.))
        r.update(id=row['id'],group=group,wall_seconds=time.monotonic()-began,
                 solver_source_sha256=hashlib.sha256((WORK/'solver'/'poc3.py').read_bytes()).hexdigest(),
                 input_has_truth=False)
    except Exception as exc:r={'id':row['id'],'group':group,'exception':repr(exc),'wall_seconds':time.monotonic()-began}
    out=WORK/'validation'/group;out.mkdir(parents=True,exist_ok=True)
    (out/(row['id']+'.json')).write_text(json.dumps(r,indent=2))
    return {k:r.get(k) for k in ('id','group','status','seconds','max_residual','exception')}

def main():
    jobs=[]
    for group,file in [('remaining_development','observations.json'),('holdout','holdout_observations.json')]:
        for row in json.loads((WORK/file).read_text())['cases']:
            if group=='remaining_development' and row['id'] not in [f'random_{i:02d}' for i in range(3,8)]:continue
            jobs.append((group,row))
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(worker,job) for job in jobs]
        for f in as_completed(futures):print(json.dumps(f.result()),flush=True)

if __name__=='__main__':main()
