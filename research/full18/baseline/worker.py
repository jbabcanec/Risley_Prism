"""Truth-free worker for the isolated full18 baseline/noise benchmark."""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import sys, json, time, traceback
from pathlib import Path
import numpy as np

ROOT = Path(r'C:\Users\josep\Dropbox\Babcanec Works\Mathematics\Wedge')
sys.path.insert(0, str(ROOT))
from risley_lattice.model import vec2pat, LO, HI
from risley_lattice.fmodel import forward_point, forward_iv
import risley_lattice.solve as solver

def clean(o):
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, np.generic): return o.item()
    if isinstance(o, dict): return {str(k): clean(v) for k,v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, float) and not np.isfinite(o): return str(o)
    return o

def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(clean(value), indent=2), encoding='utf-8')
    tmp.replace(path)

def evaluate(x, obs):
    residual = vec2pat(x).reshape(-1) - obs.reshape(-1)
    out = dict(estimate=x, residual_mse=float(np.mean(residual**2)),
               residual_rms=float(np.sqrt(np.mean(residual**2))),
               residual_max_abs=float(np.max(np.abs(residual))),
               residual_max_per_axis=np.max(np.abs(residual.reshape(-1,2)),axis=0),
               inside_native_bounds=bool(np.all(x>=LO) and np.all(x<=HI)))
    try:
        fm, fr, guards = forward_point(x)
        out.update(physical_guards=guards, physical_guards_pass=True,
                   exact_model_residual_upper=float(np.max(np.abs(fm-obs.reshape(-1))+fr)),
                   arithmetic_enclosure_max=float(np.max(fr)))
    except Exception as exc:
        out.update(physical_guards_pass=False, physical_guard_failure=str(exc))
    return out

def main():
    job_path, result_path = map(Path, sys.argv[1:3])
    job = json.loads(job_path.read_text(encoding='utf-8'))
    obs = np.asarray(job['observations'],float)
    assert obs.shape == (200,2)
    start=time.perf_counter()
    report={'mode':job['mode'],'case_id':job['case_id'],'truth_given_to_worker':False}
    try:
        if job['mode']=='blind':
            original=solver.trf
            calls=[]
            def monitored(*args, **kwargs):
                x,mse=original(*args, **kwargs)
                calls.append({'mse':mse,'elapsed_s':time.perf_counter()-start})
                checkpoint=dict(report,status='completed_stage_checkpoint',elapsed_s=time.perf_counter()-start,
                                solver_stage_mse=mse,stages=calls.copy(),**evaluate(x,obs))
                write(result_path.with_suffix('.checkpoint.json'),checkpoint)
                return x,mse
            solver.trf=monitored
            x,mse,how,info=solver.solve18(obs,completion='standard')
            report.update(status='returned',solver_reported_mse=mse,how=how,
                          spectral_info=info,stages=calls)
            if x is not None: report.update(evaluate(x,obs))
        elif job['mode']=='local_noise':
            # This is warm-start local perturbation, never labeled blind recovery.
            x0=np.asarray(job['initial_estimate'],float)
            x,mse=solver.trf(x0,obs.reshape(-1),np.ones(obs.size,bool),LO,HI,max_nfev=1000)
            report.update(status='returned',solver_reported_mse=mse,initial_estimate=x0,
                          warm_start='saved noiseless blind estimate',eta=job['eta'],noise_kind=job['noise_kind'],**evaluate(x,obs))
        elif job['mode']=='sensitivity':
            x=np.asarray(job['initial_estimate'],float)
            fm,fr,j,jr,guards=forward_iv(x,np.zeros(18))
            pinv=np.linalg.pinv(j,rcond=1e-14)
            singular=np.linalg.svd(j,compute_uv=False)
            report.update(status='returned',initial_estimate=x,
                          derivative_linf_noise_gains=np.sum(np.abs(pinv),axis=1),
                          jacobian_singular_values=singular,
                          jacobian_condition=float(singular[0]/singular[-1]),
                          derivative_status='local linearized sensitivities, not deterministic finite-noise guarantees',
                          physical_guards=guards,
                          adversarial_noise_directions=np.sign(pinv))
        else: raise ValueError(job['mode'])
    except Exception as exc:
        report.update(status='error',error=repr(exc),traceback=traceback.format_exc())
    report['elapsed_s']=time.perf_counter()-start
    write(result_path,report)

if __name__=='__main__': main()
