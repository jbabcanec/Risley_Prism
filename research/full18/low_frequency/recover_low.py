"""Frozen low-frequency proposal policy for all three unresolved holdouts.

Only observation files are read. Original repository and poc3 remain unchanged.
All 18 optical parameters remain unknown under their original native bounds.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
WORK=Path(__file__).resolve().parent
ROOT=Path(r'C:\Users\josep\Dropbox\Babcanec Works\Mathematics\Wedge')
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(WORK.parent/'solver'))
import argparse,hashlib,inspect,json,time
import numpy as np
import risley_lattice.spectral as spectral
import risley_lattice.lattice as lattice
from risley_lattice.model import LO,HI,vec2pat
from risley_lattice.solve import trf
from risley_lattice.fmodel import forward_point
import poc3


def spectral_diagnostic(y):
    z=y[:,0]+1j*y[:,1]
    f,c=lattice.matrix_pencil(z,.05)
    order=np.argsort(-abs(c))[:16]
    raw=[{'frequency':float(f[i]),'amplitude':float(abs(c[i]))} for i in order]
    before={}
    for front in ('pencil','fft'):
        n,info=spectral.extract_speeds(y,.05,frontend=front)
        before[front]={'speeds':None if n is None else n.tolist(),
            'lines':np.asarray(info['lines']).tolist(),
            'line_amplitudes':np.asarray(info['line_amplitudes']).tolist(),
            'residual':info.get('resid'),'fail':info.get('fail')}
    return {'raw_pencil_with_DC':raw,'original':before}


def activate_private_lowcut():
    # Only this process's module state is changed. No source file is written.
    lattice.SPEED_MIN=1e-5
    spectral.SPEED_MIN=1e-5
    lattice.MERGE_C=.001 # 0.0001005 Hz at the actual 9.95s span
    original=inspect.getsource(spectral.clean_lines)
    assert 'if abs(fj) < 0.05 or abs(fj) > 9.8:' in original
    private=original.replace('if abs(fj) < 0.05 or abs(fj) > 9.8:',
                            'if abs(fj) < 1e-5 or abs(fj) > 9.8:')
    scope=dict(spectral.__dict__)
    exec(compile(private,'<private lowcut clean_lines>','exec'),scope)
    spectral.clean_lines=scope['clean_lines']
    poc3.extract_speeds=spectral.extract_speeds
    return dict(speed_proposal_cutoff=1e-5,line_proposal_cutoff=1e-5,
        merge_C=.001,native_bounds_changed=False,
        clean_lines_source_sha256=hashlib.sha256(private.encode()).hexdigest())


def main():
    p=argparse.ArgumentParser();p.add_argument('--id',required=True)
    p.add_argument('--seconds',type=float,default=120);args=p.parse_args()
    y=np.array(next(c['observed'] for c in json.loads((WORK.parent/'combined_holdout_observations.json').read_text())['cases'] if c['id']==args.id))
    start=time.perf_counter()
    diagnostic=spectral_diagnostic(y)
    policy=activate_private_lowcut()
    output={'id':args.id,'policy':policy,'diagnostic':diagnostic,'stages':[],
        'all18_unknown':True,'truth_read_by_solver':False,'accuracy_certified':False}
    outpath=WORK/(args.id+'.json')
    outpath.write_text(json.dumps(output,indent=2))
    print(json.dumps({'id':args.id,'original':diagnostic['original'],'raw_pencil_with_DC':diagnostic['raw_pencil_with_DC'][:8]}),flush=True)
    for front in ('pencil','fft'):
        result=poc3.solve18(y,config=poc3.FitConfig(max_seconds=args.seconds/2,
            frontend=front,alternate_bases=2,finalists=6))
        result['frontend']=front
        output['stages'].append(result)
        outpath.write_text(json.dumps(output,indent=2))
        print(json.dumps({'id':args.id,'frontend':front,'new_speeds':result.get('spectral_speeds'),
            'max_residual':result.get('max_residual'),'seconds':result['seconds']}),flush=True)
    eligible=[r for r in output['stages'] if r.get('theta') is not None]
    if eligible:
        result=min(eligible,key=lambda r:r['rms'])
        best=np.array(result['theta']);old=float(np.mean((vec2pat(best)-y)**2))
        x,mse=trf(best,y.ravel(),np.ones(y.size,bool),LO,HI,max_nfev=100)
        if mse<old:best=x
        output.update(theta=best.tolist(),max_residual=float(np.max(abs(vec2pat(best)-y))),
            rms=float(np.sqrt(np.mean((vec2pat(best)-y)**2))),selected_frontend=result['frontend'])
        try:
            _,_,m=forward_point(best);output['physical_margins']={k:float(v) for k,v in m.items()}
        except Exception as exc:output['physical_check_error']=repr(exc)
    output['seconds']=time.perf_counter()-start
    outpath.write_text(json.dumps(output,indent=2))
    print(json.dumps({k:output.get(k) for k in ('id','seconds','max_residual','rms','selected_frontend','physical_check_error')}),flush=True)


if __name__=='__main__':main()
