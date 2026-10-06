"""Frozen four-case development comparison, with external process time caps."""
import sys,subprocess,concurrent.futures,json,os,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):env[key]='1'
IDS=['moderate_unsorted','random_03','exact_collision','weak_first']

def run(item):
    case,mode=item;script='phase_blocks.py' if mode=='blocks' else 'constrained_compare.py'
    began=time.perf_counter()
    try:
        proc=subprocess.run([sys.executable,'-B',str(HERE/script),'--id',case,'--seconds','70'],
             env=env,capture_output=True,text=True,timeout=90,creationflags=subprocess.CREATE_NO_WINDOW)
        (HERE/(case+'_'+mode+'.log.txt')).write_text(proc.stdout+proc.stderr,encoding='utf-8')
        path=HERE/(case+'_'+mode+'.json')
        result=json.loads(path.read_text()) if path.exists() else {'status':'process_error','stderr':proc.stderr[-2000:]}
        row={'id':case,'mode':mode,'status':result['status'],'seconds':time.perf_counter()-began,
             'mse':None if result.get('best') is None else result['best']['mse']}
    except subprocess.TimeoutExpired:
        row={'id':case,'mode':mode,'status':'timeout','seconds':time.perf_counter()-began}
        (HERE/(case+'_'+mode+'_timeout.json')).write_text(json.dumps(row,indent=2))
    print(json.dumps(row),flush=True);return row

if __name__=='__main__':
    # Mathematical/derivative controls must pass before any optical solve.
    controls=subprocess.run([sys.executable,'-B',str(HERE/'controls.py')],env=env,capture_output=True,text=True,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    print(controls.stdout,flush=True)
    if controls.returncode:raise RuntimeError(controls.stderr)
    jobs=[(case,mode) for case in IDS for mode in ('blocks','constrained')]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        rows=[]
        for row in pool.map(run,jobs):
            rows.append(row);(HERE/'run_summary.json').write_text(json.dumps(rows,indent=2))
