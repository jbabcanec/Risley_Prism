"""Read-only forward-program checks for the saved ambiguity endpoints."""
import os,sys,json,hashlib
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
from pathlib import Path
import numpy as np
from profile_pairs import forward
root=Path(r'C:\Users\josep\Dropbox\Babcanec Works\Mathematics\Wedge')
sys.path.insert(0,str(root))
from risley_lattice.model import vec2pat
from risley_lattice.fmodel import forward_point

records=json.loads(Path('benchmark_verified_60.json').read_text())['records']
out=[]
for i,r in enumerate(records):
    for endpoint in ('base','alternative'):
        v=np.array(r[endpoint]);search=forward(v);canonical=vec2pat(v).reshape(-1)
        fm,fr,guards=forward_point(v)
        out.append({'record':i,'endpoint':endpoint,'search_vs_core_max':float(np.max(abs(search-canonical))),
                    'search_vs_interval_midpoint_max':float(np.max(abs(search-fm))),
                    'repository_interval_radius_max':float(np.max(fr)),
                    'search_inside_repository_enclosures':bool(np.all(abs(search-fm)<=fr)),
                    'canonical_inside_repository_enclosures':bool(np.all(abs(canonical-fm)<=fr)),
                    'canonical_vs_shared_observation_max':float(np.max(abs(canonical-np.array(r['common_observations'])))),
                    'repository_physical_guards':guards})
report={'scope':'numerical parity only; exact rational time differs from stored float arange timestamps','records':out,
        'source_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'risley_lattice/fmodel.py',root/'risley_lattice/ivx.py',root/'risley_lattice/model.py',root/'reverse_problem_v2/core.py')}}
Path('repository_parity.json').write_text(json.dumps(report,indent=2))
print(json.dumps({'max_search_core_difference':max(x['search_vs_core_max'] for x in out),
                  'all_search_in_enclosures':all(x['search_inside_repository_enclosures'] for x in out),
                  'all_canonical_in_enclosures':all(x['canonical_inside_repository_enclosures'] for x in out)}))
