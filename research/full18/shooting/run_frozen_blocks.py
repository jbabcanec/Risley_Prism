"""Frozen complementary route on two failures from a separate blind batch.

Reads observations only; full native truth remains outside this process.
"""
import os,sys,json,hashlib,subprocess,concurrent.futures,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):env[key]='1'
source=HERE/'phase_blocks.py';data=HERE.parent/'combined_holdout_observations.json'
contract={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
          'observations_sha256':hashlib.sha256(data.read_bytes()).hexdigest(),
          'ids':['random_02','random_03'],'seconds':70,'blocks':8,
          'selection':'two failed cases from a separate frozen combined-policy run, not a population sample',
          'truth_read':False,'result_prefix':'combined_'}
(HERE/'frozen_combined_contract.json').write_text(json.dumps(contract,indent=2))
def run(case):
    start=time.perf_counter()
    try:
        proc=subprocess.run([sys.executable,'-B',str(source),'--id',case,'--seconds','70','--blocks','8',
                             '--input',data.name,'--prefix','combined_'],env=env,capture_output=True,text=True,
                             timeout=90,creationflags=subprocess.CREATE_NO_WINDOW)
        (HERE/('combined_'+case+'.log.txt')).write_text(proc.stdout+proc.stderr,encoding='utf-8')
        result_file=HERE/('combined_'+case+'_blocks.json')
        result=json.loads(result_file.read_text()) if result_file.exists() else {'status':'process_error','error':proc.stderr[-2000:]}
        row={'id':case,'status':result['status'],'seconds':time.perf_counter()-start,
             'mse':None if result.get('best') is None else result['best']['mse']}
    except subprocess.TimeoutExpired:row={'id':case,'status':'timeout','seconds':time.perf_counter()-start}
    print(json.dumps(row),flush=True);return row
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    result=list(pool.map(run,contract['ids']))
assert hashlib.sha256(source.read_bytes()).hexdigest()==contract['source_sha256']
(HERE/'frozen_combined_results.json').write_text(json.dumps(result,indent=2))
