"""Match the hard-band objective after the three observation-only repairs.

No truth file or post-scoring report is an input. All18 native coordinates move.
"""
import os,sys,json,time,hashlib
from pathlib import Path
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;WORK=HERE.parent
sys.path.insert(0,str(WORK/'solver'));sys.path.insert(0,str(WORK))
import numpy as np
from poc3 import minimax_refine,canonical,LO,HI
from risley_lattice.fmodel import forward_point

def main():
    for ident,suffix in [('random_02','profile'),('random_03','profile'),('random_07','release')]:
        seedfile=HERE/'noise_1e4'/(ident+'_'+suffix+'.json')
        obsfile=WORK/'combined_validation'/'noise_1e4'/(ident+'.json')
        seed=np.array(json.loads(seedfile.read_text())['theta']);record=json.loads(obsfile.read_text())
        y=np.array(record['actual_observations']);eta=record['noise_amplitude'];start=time.perf_counter()
        v,history=minimax_refine(seed,y,.05,eta,deadline=start+30.,iterations=10)
        fm,fr,margins=forward_point(v)
        bound=float(np.max(np.nextafter(np.nextafter(abs(fm-y.ravel()),np.inf)+fr,np.inf)))
        result={'id':ident,'theta':v.tolist(),'actual_observations':y.tolist(),'noise_amplitude':eta,
                'all18_unknown':True,'truth_inputs':False,'history':history,'seconds':time.perf_counter()-start,
                'canonical_max_residual':float(np.max(abs(canonical(v)-y))),
                'interval_error_upper':bound,'interval_compatible':bound<=eta,
                'native_bounds':bool(np.all(v>=LO)&np.all(v<=HI)),'physical_margins':margins,
                'input_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (seedfile,obsfile)}}
        (HERE/'noise_1e4'/(ident+'_minimax.json')).write_text(json.dumps(result,indent=2))
        print(json.dumps({k:val for k,val in result.items() if k not in ('theta','actual_observations','input_sha256')}),flush=True)

if __name__=='__main__':main()
