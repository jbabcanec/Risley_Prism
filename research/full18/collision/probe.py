import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,r'C:\Users\josep\Dropbox\Babcanec Works\Mathematics\Wedge')
import json,numpy as np
from pathlib import Path
from risley_lattice.spectral import extract_speeds
work=Path(r'C:\Users\josep\Documents\Codex\2026-10-02\task\full18_research')
y=np.array(next(c['observed'] for c in json.loads((work/'observations.json').read_text())['cases'] if c['id']=='exact_collision'))
for front in ['fft','pencil']:
 ns,info=extract_speeds(y,.05,frontend=front)
 print(front,ns)
 print([(round(float(f),8),round(float(a),6)) for f,a in zip(info['lines'],info['line_amplitudes'])])
