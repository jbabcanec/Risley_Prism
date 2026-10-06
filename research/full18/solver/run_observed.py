"""Run blind full18 candidates, reading observations only (no truth file)."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys
sys.dont_write_bytecode=True
import argparse
import json
from pathlib import Path
import numpy as np
from poc3 import solve18, FitConfig

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--ids',nargs='+',required=True)
    parser.add_argument('--seconds',type=float,default=45)
    parser.add_argument('--eta',type=float,default=0)
    parser.add_argument('--noise',type=float,default=0)
    parser.add_argument('--out',default='results_v1')
    parser.add_argument('--frontend',default='pencil')
    args=parser.parse_args()
    work=Path(__file__).resolve().parent
    data=json.loads((work.parent/'observations.json').read_text())
    out=work/args.out;out.mkdir(exist_ok=True)
    for row in data['cases']:
        if row['id'] not in args.ids:continue
        y=np.asarray(row['observed'],float)
        if args.noise:
            # One fixed deterministic pattern independent of truth/candidate.
            k=np.arange(len(y))[:,None]; axis=np.arange(2)[None,:]
            y=y+args.noise*np.sin((k+1)*(np.sqrt(2)+axis*np.sqrt(3)))
        try:
            result=solve18(y,eta=args.eta,config=FitConfig(max_seconds=args.seconds,frontend=args.frontend))
            result.update(id=row['id'],noise_amplitude=args.noise)
        except Exception as exc:
            result={'id':row['id'],'exception':repr(exc)}
        (out/(row['id']+'.json')).write_text(json.dumps(result,indent=2))
        print(json.dumps({k:result.get(k) for k in ('id','status','seconds','rms','max_residual','method','exception')}),flush=True)

if __name__=='__main__':main()
