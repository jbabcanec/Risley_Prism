"""Observation-only harmonic completion ablation; no truth file is read."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import sys
sys.dont_write_bytecode=True
import argparse,json
from pathlib import Path
import numpy as np
from harmonic_fit import solve_comparison

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--ids',nargs='+',default=['random_00','moderate_unsorted','wide_unsorted','weak_first'])
    parser.add_argument('--noise',type=float,default=0)
    parser.add_argument('--seconds',type=float,default=40)
    parser.add_argument('--out',default='comparison_noiseless')
    parser.add_argument('--modes',nargs='+',choices=['direct','harmonic','emphasis'],default=['direct','harmonic'])
    args=parser.parse_args();here=Path(__file__).resolve().parent
    data=json.loads((here.parent/'observations.json').read_text())
    out=here/args.out;out.mkdir(exist_ok=True)
    for record in data['cases']:
        if record['id'] not in args.ids:continue
        y=np.asarray(record['observed'])
        k=np.arange(len(y))[:,None];axis=np.arange(2)[None,:]
        y=y+args.noise*np.sin((k+1)*(np.sqrt(2)+axis*np.sqrt(3)))
        for mode in args.modes:
            result=solve_comparison(y,mode,args.seconds)
            result.update(id=record['id'],noise_amplitude=args.noise)
            (out/(record['id']+'_'+mode+'.json')).write_text(json.dumps(result,indent=2))
            print(json.dumps({k:result.get(k) for k in ('id','mode','seconds','rms','max_residual','status')}),flush=True)

if __name__=='__main__':main()
