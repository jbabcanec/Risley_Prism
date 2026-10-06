"""Observation-only full-native-domain gain-coordinate continuation.

All three wedges use u=(n-1)tan(ax). The global rectangle is |u|<=.8tan18;
six explicit penalties enforce the coupled native inequalities
 +/-u <= (n-1)tan18. These describe the entire original wedge/index domain.
The four affine variables remain bounded on their original native ranges.
This is a bounded candidate search, not a complete compatible-set algorithm.
"""
import os,sys
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
HERE=Path(__file__).resolve().parent;WORK=HERE.parent
sys.path.insert(0,str(WORK/'solver'))
from poc3 import Projected,WorkLimit,LO,HI,SPAN,NONLINEAR,LINEAR,NAMES,canonical,forward,minimax_refine
from risley_lattice.fmodel import forward_point

TAN18=np.tan(np.pi/10)
GAIN_CAP=(HI[9]-1)*TAN18

class FullGain:
    def __init__(self,seed,y,deadline,output):
        self.projected=Projected(seed,y,.05,deadline)
        self.lo=LO[NONLINEAR].copy();self.hi=HI[NONLINEAR].copy()
        self.lo[3:6]=-GAIN_CAP;self.hi[3:6]=GAIN_CAP
        self.q0=seed[NONLINEAR].copy()
        self.q0[3:6]=(seed[9:12]-1)*np.tan(np.deg2rad(seed[3:6]))
        self.span=self.hi-self.lo
        self.best=seed.copy();self.best_cost=np.inf;self.best_max=np.inf
        self.calls=0;self.history=[];self.start=time.perf_counter();self.last_save=0
        self.output=output;self.lastq=None
        self.initial=seed.copy();self.y=y

    @staticmethod
    def expand(q):
        v=q.copy();D=np.eye(14)
        u=q[3:6];n=q[9:12];tau=u/(n-1)
        v[3:6]=np.rad2deg(np.arctan(tau))
        factor=180/np.pi/(1+tau*tau)
        for j in range(3):
            D[3+j,3+j]=factor[j]/(n[j]-1)
            D[3+j,9+j]=-factor[j]*u[j]/(n[j]-1)**2
        return v,D

    def fun(self,q):
        physical,_=self.expand(q)
        r=self.projected.fun(physical)
        u=q[3:6];n=q[9:12]
        constraints=np.r_[u-(n-1)*TAN18,-u-(n-1)*TAN18]
        penalties=1e3*np.maximum(constraints,0)
        self.calls+=1
        v=self.projected.v
        if np.all(v>=LO) and np.all(v<=HI) and np.min(self.projected.guards)>0:
            cost=float(np.sum(self.projected.data_residual**2))
            if cost<self.best_cost:
                self.best_cost=cost;self.best=v.copy();self.best_max=float(np.max(abs(self.projected.data_residual)))
        elapsed=time.perf_counter()-self.start
        if elapsed-self.last_save>10:
            self.history.append({'seconds':elapsed,'evaluations':self.calls,'rms':float(np.sqrt(self.best_cost/400)),
                                 'max_residual':self.best_max,'glass':self.best[9:12].tolist(),'geometry':self.best[12:14].tolist()})
            self.last_save=elapsed
            self.save('running')
            print(json.dumps(self.history[-1]),flush=True)
        return np.r_[r,penalties]

    def jac(self,q):
        physical,D=self.expand(q)
        J=self.projected.jac(physical)@D
        u=q[3:6];n=q[9:12]
        c=np.r_[u-(n-1)*TAN18,-u-(n-1)*TAN18]
        C=np.zeros((6,14))
        for j in range(3):
            C[j,3+j]=1;C[j,9+j]=-TAN18
            C[j+3,3+j]=-1;C[j+3,9+j]=-TAN18
        C*=((c>0)*1e3)[:,None]
        return np.vstack([J,C])

    def save(self,status,extra=None):
        payload={'status':status,'theta':self.best.tolist(),'names':NAMES,'strict_numerical_max_residual':self.best_max,
                 'rms':float(np.sqrt(self.best_cost/400)),'evaluations':self.calls,'seconds':time.perf_counter()-self.start,
                 'history':self.history,'initial_observation_derived_candidate':self.initial.tolist(),
                 'global_gain_bound':GAIN_CAP,'gain_inequalities':'abs(u_i) <= (n_i-1)tan(18deg)',
                 'all18_unknown':True,'affine_variables_eliminated_with_native_bounds':[NAMES[j] for j in LINEAR],
                 'native_bounds_unchanged':True,'truth_inputs_used':False,'global_accuracy_certified':False,
                 'complete_compatible_set_enumerated':False}
        if extra:payload.update(extra)
        self.output.write_text(json.dumps(payload,indent=2))
        return payload

def main():
    a=argparse.ArgumentParser();a.add_argument('--seconds',type=float,default=120);a.add_argument('--scaling',choices=['native','jac'],default='native');a.add_argument('--out',default='result_native.json')
    a.add_argument('--candidate',default=str(WORK/'profiles/blind_adverse_1e8.json'))
    a.add_argument('--observations',default=str(WORK/'profiles/adverse_observations_only.json'))
    a.add_argument('--id',default='eta_1e-8_separated_speeds');a.add_argument('--threshold',type=float,default=1e-8)
    args=a.parse_args();source=Path(args.candidate);observed=Path(args.observations)
    input_candidate=json.loads(source.read_text());seed=np.array(input_candidate['theta'])
    if input_candidate.get('actual_observations') is not None:y=np.array(input_candidate['actual_observations'])
    else:y=np.array(next(r for r in json.loads(observed.read_text())['cases'] if r['id']==args.id)['observed'])
    start=time.perf_counter();deadline=start+args.seconds
    r=FullGain(seed,y,deadline,HERE/args.out)
    r.fun(r.q0)
    reason='finished';fitinfo={}
    try:
        fit=least_squares(r.fun,r.q0,jac=r.jac,bounds=(r.lo,r.hi),x_scale=r.span if args.scaling=='native' else 'jac',
                          ftol=1e-14,xtol=1e-14,gtol=None,max_nfev=20000)
        fitinfo={'message':fit.message,'nfev':fit.nfev,'njev':fit.njev,'optimality':float(fit.optimality)}
    except WorkLimit as exc:reason=str(exc)
    v=r.best
    # Final point checks use the original native bounds and all observations.
    try:
        fm,fr,margins=forward_point(v);bound=float(np.max(abs(fm-y.ravel())+fr))
        canonical_error=float(np.max(abs(canonical(v)-y)))
        check={'physical':True,'interval_error_upper_under_ivx':bound,'canonical_max_residual':canonical_error,
               'strict_numerical_max_residual':float(np.max(abs(forward(v)-y))),'margins':margins,
               'native_bounds':bool(np.all(v>=LO)&np.all(v<=HI)),
               'numerical_residual_threshold':args.threshold,'interval_compatible':bound<=args.threshold,'canonical_compatible':canonical_error<=args.threshold}
    except Exception as exc:check={'physical':False,'error':repr(exc)}
    result=r.save('complete',{'stop_reason':reason,'scaling':args.scaling,'fit':fitinfo,'check':check,'case_id':args.id,
             'input_sha256':{source.name:hashlib.sha256(source.read_bytes()).hexdigest(),observed.name:hashlib.sha256(observed.read_bytes()).hexdigest()},
             'code_sha256':{Path(__file__).name:hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                            'poc3.py':hashlib.sha256((WORK/'solver/poc3.py').read_bytes()).hexdigest()}})
    print(json.dumps({k:result[k] for k in ('seconds','evaluations','strict_numerical_max_residual','check','stop_reason')},indent=2))

if __name__=='__main__':main()
