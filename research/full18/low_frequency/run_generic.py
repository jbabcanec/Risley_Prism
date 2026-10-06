"""Generic file interface to the frozen lowcut/profile/residual proposal code.

Makes an isolated input-only run directory and rewrites only file plumbing in
private source copies. Numerical functions and candidate policies stay intact.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys
sys.dont_write_bytecode=True
import argparse,hashlib,json,subprocess,time
from pathlib import Path
HERE=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['lowcut','profile','residual'],required=True)
    p.add_argument('--observations',type=Path,required=True);p.add_argument('--id',required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--candidate',type=Path,action='append',default=[])
    p.add_argument('--seconds',type=float,default=120);args=p.parse_args()
    args.observations=args.observations.resolve();args.out=args.out.resolve()
    args.candidate=[path.resolve() for path in args.candidate]
    row=json.loads(args.observations.read_text())
    if 'actual_observations' in row:obs=row['actual_observations']
    else:obs=next(c['observed'] for c in row['cases'] if c['id']==args.id)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    sandbox=args.out.parent/(args.out.stem+'_run');sandbox.mkdir(exist_ok=True)
    data=sandbox/'observations.json';data.write_text(json.dumps({'cases':[{'id':args.id,'observed':obs}]}))
    candidate_files=[]
    for i,path in enumerate(args.candidate):
        c=json.loads(path.read_text());dest=sandbox/f'candidate_{i}.json'
        dest.write_text(json.dumps({'theta':c['theta'],'source':str(path)}));candidate_files.append(dest)
    source_name={'lowcut':'recover_low.py','profile':'profile_bases.py','residual':'residual_release.py'}[args.mode]
    original=(HERE/source_name).read_text();source=original
    # Retain imports of frozen helpers from HERE. Only run input/output paths,
    # identity and user-supplied time limit are replaced in the private driver.
    if args.mode=='lowcut':
        source=source.replace("WORK=Path(__file__).resolve().parent",'WORK=Path('+repr(str(HERE))+')')
    source=source.replace("(WORK.parent/'combined_holdout_observations.json')",'Path('+repr(str(data))+')')
    if args.mode=='profile':
        source=source.replace("diag=json.loads((WORK/(args.id+'.json')).read_text())['diagnostic']",'diag=spectral_diagnostic(y)')
        source=source.replace("out=WORK/(args.id+'_profile.json')",'out=Path('+repr(str(args.out))+')')
    elif args.mode=='lowcut':
        source=source.replace("outpath=WORK/(args.id+'.json')",'outpath=Path('+repr(str(args.out))+')')
    else:
        if not candidate_files:raise ValueError('--candidate required for residual mode')
        source=source.replace("ident='random_07';start=time.perf_counter();deadline=start+120",'ident='+repr(args.id)+';start=time.perf_counter();deadline=start+'+str(args.seconds))
        source=source.replace("sources=[WORK/(ident+'.json'),WORK/(ident+'_profile.json'),WORK.parent/'combined_validation'/'noiseless'/(ident+'.json')]",'sources=[Path(s) for s in '+repr([str(x) for x in candidate_files])+']')
        source=source.replace("diag=json.loads((WORK/(ident+'.json')).read_text())['diagnostic']",'diag=spectral_diagnostic(y)')
        source=source.replace("out=WORK/(ident+'_release.json')",'out=Path('+repr(str(args.out))+')')
        source=source.replace('if time.perf_counter()-start>65:break','if time.perf_counter()-start>'+str(args.seconds*65/120)+':break')
    # These inserted search paths only locate unchanged numerical modules.
    entry=sandbox/'driver.py'
    prefix='import sys\nsys.dont_write_bytecode=True\nsys.path.insert(0,'+repr(str(HERE))+')\n'
    entry.write_text(prefix+source)
    meta={'mode':args.mode,'id':args.id,'input':str(args.observations),'candidate_inputs':[str(p) for p in args.candidate],
        'truth_in_solver_input':False,'original_policy_source':str(HERE/source_name),
        'original_policy_sha256':hashlib.sha256(original.encode()).hexdigest(),
        'private_driver_sha256':hashlib.sha256(entry.read_bytes()).hexdigest(),
        'changes':'input/output paths, case identity, run budget; numerical proposal and fit code retained'}
    (sandbox/'manifest.json').write_text(json.dumps(meta,indent=2))
    cmd=[sys.executable,str(entry)]
    if args.mode!='residual':cmd+=['--id',args.id,'--seconds',str(args.seconds)]
    completed=subprocess.run(cmd,cwd=HERE)
    if completed.returncode:raise SystemExit(completed.returncode)


if __name__=='__main__':main()
