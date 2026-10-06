"""Reproducible baseline and local-noise runs; truth used only for scoring."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import json, sys, time, subprocess, concurrent.futures, hashlib, csv
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
CASES_PATH=HERE.parent/'cases.json'
ENV=os.environ.copy()

def dump(path,value):
    Path(path).write_text(json.dumps(value,indent=2),encoding='utf-8')

def execute(job, stem, cap=60):
    job_path=HERE/(stem+'.job.json'); result_path=HERE/(stem+'.result.json')
    dump(job_path,job)
    began=time.perf_counter()
    try:
        proc=subprocess.run([sys.executable,'-B',str(HERE/'worker.py'),str(job_path),str(result_path)],
                            capture_output=True,text=True,timeout=cap,env=ENV,creationflags=subprocess.CREATE_NO_WINDOW)
        (HERE/(stem+'.log.txt')).write_text(proc.stdout+proc.stderr,encoding='utf-8')
        result=json.loads(result_path.read_text(encoding='utf-8')) if result_path.exists() else {'status':'process_error','returncode':proc.returncode}
    except subprocess.TimeoutExpired as exc:
        checkpoint=result_path.with_suffix('.checkpoint.json')
        result=json.loads(checkpoint.read_text(encoding='utf-8')) if checkpoint.exists() else {}
        result.update(status='timeout',timeout_s=cap)
        dump(result_path,result)
    result.update(case_id=job['case_id'],mode=job['mode'],wall_s=time.perf_counter()-began,result_file=result_path.name)
    return result

def score(result,case):
    result['truth']=case['truth']
    if 'estimate' in result:
        error=np.asarray(result['estimate'])-np.asarray(case['truth'])
        result['native_signed_errors']=error.tolist()
        result['native_absolute_errors']=np.abs(error).tolist()
        result['native_max_error']=float(np.max(np.abs(error)))
        result['worst_coordinate']=META['names'][int(np.argmax(np.abs(error)))]
        result['recovered_below_0_001']=bool(np.max(np.abs(error))<.001 and result.get('physical_guards_pass',False))
    else:
        result['recovered_below_0_001']=False
    return result

def metadata():
    return {'cases_sha256':hashlib.sha256(CASES_PATH.read_bytes()).hexdigest(),
            'source_hashes':META.get('source_hashes'), 'names':META['names'],
            'sampling':'200 samples, 20 Hz, t=k/20; full unsorted native bounds',
            'notes':['Truth is only used for scoring, not passed to blind workers.',
                     'Every returned and checkpoint candidate is checked on all samples without masking.',
                     'Physical guard checks are point checks under existing interval-arithmetic assumptions.',
                     'Native-error successes are empirical, not global accuracy certificates.']}

def blind():
    cases=[c for c in META['cases'] if c['id'] in sys.argv[2:]] if len(sys.argv)>2 else [c for c in META['cases'] if c['id'].startswith('random')]
    prior=HERE/'baseline_summary.json'
    rows=json.loads(prior.read_text(encoding='utf-8'))['rows'] if prior.exists() else []
    rows=[row for row in rows if row['case_id'] not in {c['id'] for c in cases}]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures={pool.submit(execute,{'case_id':c['id'],'mode':'blind','observations':c['observed']},c['id']+'_blind',60):c for c in cases}
        for future in concurrent.futures.as_completed(futures):
            case=futures[future]
            row=score(future.result(),case); rows.append(row)
            dump(HERE/'baseline_summary.json',dict(metadata(),rows=sorted(rows,key=lambda r:r['case_id'])))
            print(json.dumps({k:row.get(k) for k in ('case_id','status','wall_s','native_max_error','worst_coordinate','residual_mse','recovered_below_0_001')}),flush=True)

def noises():
    baseline=json.loads((HERE/'baseline_summary.json').read_text(encoding='utf-8'))
    rows=[]; summaries=[]
    successful=[row for row in baseline['rows'] if row.get('recovered_below_0_001')]
    selected=['random_00','random_01','random_02','wide_unsorted','weak_first','moderate_unsorted']
    for case_id in selected:
        source=HERE.parent/'solver'/'results_v1'/(case_id+'.json')
        if not source.exists() or any(row['case_id']==case_id for row in successful): continue
        candidate=json.loads(source.read_text(encoding='utf-8'))
        case=next(c for c in META['cases'] if c['id']==case_id)
        if candidate.get('theta') and np.max(np.abs(np.asarray(candidate['theta'])-np.asarray(case['truth'])))<.001:
            successful.append({'case_id':case_id,'estimate':candidate['theta'],
                               'source':str(source.relative_to(HERE.parent)),
                               'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest()})
    dump(HERE/'noise_initial_estimates.json',successful)
    for saved in successful:
        case=next(c for c in META['cases'] if c['id']==saved['case_id'])
        sensitivity=execute({'case_id':case['id'],'mode':'sensitivity','observations':case['observed'],
                             'initial_estimate':saved['estimate']},case['id']+'_sensitivity',30)
        summaries.append(sensitivity)
        directions=np.asarray(sensitivity['adversarial_noise_directions'])
        gains=np.asarray(sensitivity['derivative_linf_noise_gains'])
        # One fixed-seed dense pattern and each of the three most sensitive
        # native-coordinate linear worst-case patterns. All are deterministic.
        random=np.random.default_rng([20261002,int.from_bytes(hashlib.sha256(case['id'].encode()).digest()[:4],'little')]).uniform(-1,1,(200,2))
        patterns=[('fixed_uniform',random)]
        for index in np.argsort(-gains)[:3]: patterns.append(('linear_worst_'+META['names'][int(index)],directions[index].reshape(200,2)))
        jobs=[]
        for eta in [1e-8,1e-6,1e-4,1e-3]:
            for kind,pattern in patterns:
                observations=np.asarray(case['observed'])+eta*pattern
                job={'case_id':case['id'],'mode':'local_noise','initial_estimate':saved['estimate'],
                     'observations':observations.tolist(),'eta':eta,'noise_kind':kind}
                jobs.append((job,case['id']+'_noise_'+format(eta,'.0e')+'_'+kind))
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures={pool.submit(execute,job,stem,30):(job,stem) for job,stem in jobs}
            for future in concurrent.futures.as_completed(futures):
                job,stem=futures[future]; row=score(future.result(),case)
                row.update(eta=job['eta'],noise_kind=job['noise_kind'],input_noise_max=float(np.max(np.abs(np.asarray(job['observations'])-np.asarray(case['observed'])))))
                rows.append(row)
                dump(HERE/'noise_summary.json',dict(metadata(),scope='Warm-start local perturbation of saved blind noiseless estimates, not blind noisy recovery.',sensitivity=summaries,rows=rows))
                print(json.dumps({k:row.get(k) for k in ('case_id','eta','noise_kind','status','native_max_error','worst_coordinate','residual_max_abs')}),flush=True)
    if not successful: dump(HERE/'noise_summary.json',dict(metadata(),scope='No successful noiseless blind estimate available for local-noise runs.',rows=[]))
    if rows:
        with (HERE/'noise_coordinate_errors.csv').open('w',newline='',encoding='utf-8') as handle:
            writer=csv.writer(handle);writer.writerow(['case_id','eta','noise_kind','status']+META['names'])
            for row in rows:writer.writerow([row['case_id'],row['eta'],row['noise_kind'],row['status']]+row.get('native_absolute_errors',['']*18))

META=json.loads(CASES_PATH.read_text(encoding='utf-8'))
if __name__=='__main__':
    if sys.argv[1]=='blind': blind()
    elif sys.argv[1]=='noise': noises()
    else: raise SystemExit('Expected blind or noise')
