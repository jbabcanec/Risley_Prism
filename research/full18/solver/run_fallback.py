"""Uniform FFT/multi-basis retry, selected using v1 residual status only.

Reads observation files and completed candidate outputs, never truth files.
Policy declared before execution: 90 s, FFT, five alternative bases, all six
orders, sixteen screening evaluations, three finalists per basis, 700 polish.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys
sys.dont_write_bytecode=True
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from poc3 import solve18,FitConfig

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--group',choices=['remaining_development','holdout'],required=True)
    parser.add_argument('--ids',nargs='*')
    args=parser.parse_args();here=Path(__file__).resolve().parent;work=here.parent
    source=work/('holdout_observations.json' if args.group=='holdout' else 'observations.json')
    observations=json.loads(source.read_text())
    cases={c['id']:c for c in observations['cases']}
    out=here/'fallback_fft'/args.group;out.mkdir(parents=True,exist_ok=True)
    for file in sorted((work/'validation'/args.group).glob('*.json')):
        previous=json.loads(file.read_text());name=file.stem
        if args.ids and name not in args.ids:continue
        if previous.get('status')!='unresolved':continue
        y=np.asarray(cases[name]['observed'])
        cfg=FitConfig(max_seconds=90,frontend='fft',alternate_bases=5,
                      screen_evaluations=16,finalists=3,polish_evaluations=700)
        result=solve18(y,config=cfg)
        result.update(id=name,group=args.group,previous_file=str(file),
                      previous_max_residual=previous.get('max_residual'),
                      input_sha256=hashlib.sha256(y.tobytes()).hexdigest(),
                      config=vars(cfg))
        (out/file.name).write_text(json.dumps(result,indent=2))
        print(json.dumps({k:result.get(k) for k in ('id','group','status','seconds','rms','max_residual','method')}),flush=True)

if __name__=='__main__':main()
