"""Score completed blind outputs; this evaluator does not call any solver."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse,json
from pathlib import Path
import numpy as np

def main():
    parser=argparse.ArgumentParser();parser.add_argument('directories',nargs='+')
    args=parser.parse_args();here=Path(__file__).resolve().parent
    data=json.loads((here.parent/'cases.json').read_text())
    truths={c['id']:np.array(c['truth']) for c in data['cases']}
    names=data['names'];rows=[]
    for directory in args.directories:
        for file in sorted((here/directory).glob('*.json')):
            record=json.loads(file.read_text());name=record['id']
            row={'case':name,'directory':directory,'seconds':record.get('seconds'),
                 'status':record.get('status'),'noise':record.get('noise_amplitude',0),
                 'max_residual':record.get('max_residual'),
                 'hard_band_compatible':record.get('hard_band_compatible')}
            if record.get('theta') is not None:
                errors=np.abs(np.array(record['theta'])-truths[name])
                row.update(max_native_error=float(errors.max()),
                           worst_coordinate=names[int(errors.argmax())],
                           all18_within_001=bool(np.all(errors<.001)),
                           per_coordinate_error=dict(zip(names,errors.tolist())))
            rows.append(row)
    target=here/('evaluation_'+'_'.join(args.directories)+'.json')
    target.write_text(json.dumps(rows,indent=2))
    for row in rows:
        print(json.dumps({k:v for k,v in row.items() if k!='per_coordinate_error'}))

if __name__=='__main__':main()
