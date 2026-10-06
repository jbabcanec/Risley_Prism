"""Observation-only collision hypotheses, cancellation-aware starts, full18 release.

Hypothesis selection is empirical, not a global inverse or certificate. All
original native bounds remain admissible. The search reads no parameter truth.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(r'C:\Users\josep\Dropbox\Babcanec Works\Mathematics\Wedge')
WORK=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import argparse,itertools,json,time
import numpy as np
from scipy.optimize import least_squares
from risley_lattice.model import LO,HI,RG,vec2pat
from risley_lattice.spectral import extract_speeds,is_novel
from risley_lattice.angles import nominal_rest,calibrate_ax,invert_cubic
from risley_lattice.relations import RelationResidual
from risley_lattice.separable import ProjectedResidual,NONLINEAR
from risley_lattice.solve import trf
from risley_lattice.fmodel import forward_point


class WorkLimit(Exception):pass


def fit(x,y,B,nfev,deadline):
    if B is None:
        res=ProjectedResidual(x,y.ravel(),np.ones(y.size,bool),len(y),len(y)*.05)
        q0=x[NONLINEAR];lo=LO[NONLINEAR];hi=HI[NONLINEAR];scale=RG[NONLINEAR]
        base=res
    else:
        res=RelationResidual(x,y,B)
        q0=res.q0;lo=res.lo;hi=res.hi;scale=res.scale;base=res.base
    def fun(q):
        if time.perf_counter()>deadline:raise WorkLimit()
        return res.fun(q)
    result=least_squares(fun,q0,jac=res.jac,bounds=(lo,hi),x_scale=scale,
        max_nfev=nfev,ftol=1e-13,xtol=1e-13,gtol=1e-13)
    res.fun(result.x)
    x=base.v.copy();delta=vec2pat(x)-y
    return x,float(np.mean(delta*delta)),result.nfev


def starts(y,selected,lines):
    # Joint coefficients remove spectral leakage between the selected lines.
    z=y[:,0]+1j*y[:,1];times=np.arange(len(y))*.05
    fs=np.unique(np.r_[0,lines,selected])
    coef=np.linalg.lstsq(np.exp(2j*np.pi*np.outer(times,fs)),z,rcond=1e-10)[0]
    c=np.array([coef[np.argmin(abs(fs-f))] for f in selected])
    rest=nominal_rest(y)
    seeds=[]
    for repeat in range(2):
        other=1-repeat
        for singlepos in range(3):
            pair=[i for i in range(3) if i!=singlepos]
            B=np.zeros((3,2));B[pair,repeat]=1;B[singlepos,other]=1
            speeds=B@selected
            gains=calibrate_ax(speeds,rest)
            # The signed coefficient may be a difference of large rotating
            # contributions. Solve its real/imaginary parts at phase endpoints.
            phasesets=[(-16.,16.),(16.,-16.),(-8.,8.),(8.,-8.)]
            # In-phase signed split is still searched, including cancellation.
            cp=c[repeat];phase=np.degrees(np.angle(cp))
            sign=-1 if abs(phase)>90 else 1
            phase=(phase-180*(sign<0)+180)%360-180
            for share in (.25,.75,-1.,2.):
                phasesets.append((float(np.clip(phase,-18,18)),)*2+(share,))
            for ph in phasesets:
                if len(ph)==2:
                    mat=np.array([np.cos(np.radians(ph)),np.sin(np.radians(ph))])
                    amp=np.linalg.solve(mat,[cp.real,cp.imag])
                else:amp=sign*abs(cp)*np.array([ph[2],1-ph[2]])
                v=np.r_[speeds,np.zeros(6),rest]
                for j,p in enumerate(pair):
                    v[3+p]=np.sign(amp[j])*np.degrees(np.arctan(invert_cubic(*gains[p],abs(amp[j]))))
                    v[6+p]=ph[j]
                cc=c[other];phase2=np.degrees(np.angle(cc));sgn=-1 if abs(phase2)>90 else 1
                phase2=(phase2-180*(sgn<0)+180)%360-180
                v[3+singlepos]=sgn*np.degrees(np.arctan(invert_cubic(*gains[singlepos],abs(cc))))
                v[6+singlepos]=np.clip(phase2,-18,18)
                v=np.clip(v,LO+1e-10,HI-1e-10)
                seeds.append((v,B,f'repeat={repeat},singleton={singlepos},ph={ph}'))
    return seeds,c


def main():
    p=argparse.ArgumentParser();p.add_argument('--seconds',type=float,default=300)
    p.add_argument('--id',default='exact_collision');p.add_argument('--out',default='result.json')
    p.add_argument('--noise',type=float,default=0)
    args=p.parse_args();start=time.perf_counter();deadline=start+args.seconds
    data=json.loads((WORK.parent/'observations.json').read_text())
    y=np.asarray(next(c['observed'] for c in data['cases'] if c['id']==args.id))
    if args.noise:
        k=np.arange(len(y))[:,None];axis=np.arange(2)[None,:]
        y=y+args.noise*np.sin((k+1)*(np.sqrt(2)+axis*np.sqrt(3)))
    _,info=extract_speeds(y,.05,frontend='fft')
    lines=np.asarray(info['lines']);amps=np.asarray(info['line_amplitudes']);selected=[]
    for j in np.argsort(-amps):
        if 1e-7<abs(lines[j])<=3.5 and is_novel(lines[j],selected):selected.append(float(lines[j]))
        if len(selected)==2:break
    seeds,coefs=starts(y,np.array(selected),lines)
    trials=[];screens=[];best=None;best_mse=np.inf;best_tag=None
    print(json.dumps({'id':args.id,'selected_generators':selected,'coefficients':[[x.real,x.imag] for x in coefs],'starts':len(seeds)}),flush=True)
    def save(stage):
        result={'id':args.id,'noise_amplitude':args.noise,'generators':selected,'stage':stage,'seconds':time.perf_counter()-start,
            'x18':None if best is None else best.tolist(),'mse':best_mse,'route':best_tag,'trials':trials,
            'all18_unknown':True,'full_native_bounds':True,'certificate':False,
            'truth_read_by_solver':False,'known_scope':'same-signed repeats proposed; all speeds released finally',
            'researcher_disclosure':'contract.py accidentally exposed stress truth before coding; no such values enter this algorithm'}
        if best is not None:
            result['max_residual']=float(np.max(abs(vec2pat(best)-y)))
            try:
                _,_,margins=forward_point(best);result['margins']={k:float(v) for k,v in margins.items()}
            except Exception as exc:result['physical_check_error']=repr(exc)
        (WORK/args.out).write_text(json.dumps(result,indent=2))
    try:
        for i,(v,B,tag) in enumerate(seeds):
            x,mse,nfev=fit(v,y,B,24,deadline)
            screens.append((mse,x,B,tag));trials.append({'stage':'screen','route':tag,'mse':mse,'nfev':nfev})
            if mse<best_mse:best,best_mse,best_tag=x,mse,tag
            if i%8==7:print(json.dumps({'screen':i+1,'best_mse':best_mse,'seconds':time.perf_counter()-start}),flush=True);save('screen')
        screens.sort(key=lambda x:x[0])
        for rank,(_,x,B,tag) in enumerate(screens[:12]):
            x,mse,nfev=fit(x,y,B,400,deadline)
            trials.append({'stage':'relation_polish','route':tag,'mse':mse,'nfev':nfev})
            x,mse,nfev=fit(x,y,None,400,deadline)
            trials.append({'stage':'full18_release','route':tag,'mse':mse,'nfev':nfev})
            if mse<best_mse:best,best_mse,best_tag=x,mse,tag
            print(json.dumps({'polish_rank':rank,'mse':mse,'best_mse':best_mse,'seconds':time.perf_counter()-start}),flush=True);save('full18_release')
            if mse<1e-24:break
        if best is not None and time.perf_counter()<deadline:
            x,mse=trf(best,y.ravel(),np.ones(y.size,bool),LO,HI,max_nfev=100)
            if mse<best_mse:best,best_mse=x,mse
        save('complete')
    except WorkLimit:save('work_limit')


if __name__=='__main__':main()
