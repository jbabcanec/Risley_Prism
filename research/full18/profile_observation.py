"""Search alternatives to an observation-derived full18 answer.

These are LOWER witnesses of uncertainty. Failed profiling does not exclude
other solutions. No truth file is accepted. Coordinate shifts are exploratory;
all other17 unknown parameters are reoptimized against the actual observation.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys
sys.dont_write_bytecode=True
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares,linprog
WORK=Path(__file__).resolve().parent
sys.path.insert(0,str(WORK/'solver'));sys.path.insert(0,str(WORK/'structure'))
from poc3 import affine,forward,canonical,LO,HI,NAMES,SPAN,LINEAR
from diagnostics import full_jacobian
from risley_lattice.fmodel import forward_point

def point_check(v,y,eta):
    try:
        fm,fr,m=forward_point(v)
        bound=float(np.max(np.nextafter(np.nextafter(np.abs(fm-y.ravel()),np.inf)+fr,np.inf)))
        return {'physical':True,'interval_error_upper':bound,'compatible_under_ivx_contract':bound<=eta,'margins':m,'canonical_max_residual':float(np.max(np.abs(canonical(v)-y)))}
    except Exception as exc:return {'physical':False,'error':repr(exc),'compatible_under_ivx_contract':False}

def profile(base,y,eta,j,shift):
    fixed=base.copy();fixed[j]+=shift
    if not LO[j]<=fixed[j]<=HI[j]:return {'coordinate':NAMES[j],'shift':shift,'status':'outside_prior'}
    free=np.array([i for i in range(18) if i!=j]);scale=max(eta,1e-7)
    def unpack(z):
        v=fixed.copy();v[free]=base[free]+SPAN[free]*z;return v
    def fun(z):
        v=unpack(z);b,A,g,p=affine(v)
        return np.r_[(b+A@v[LINEAR]-y.ravel())/scale,p/scale]
    # Complex-step Jacobian includes trial physical penalties and data residual.
    def jac(z):
        v=unpack(z);h=1e-22;batch=np.tile(v,(17,1)).astype(complex);batch[np.arange(17),free]+=1j*h
        b,A,g,p=affine(batch)
        f=b+np.einsum('bmk,bk->bm',A,batch[:,LINEAR])
        return np.c_[f.imag/h,p.imag/h].T*SPAN[free]/scale
    opt=least_squares(fun,np.zeros(17),jac=jac,bounds=((LO[free]-base[free])/SPAN[free],(HI[free]-base[free])/SPAN[free]),max_nfev=150,ftol=1e-13,xtol=1e-13,gtol=1e-13)
    v=unpack(opt.x);history=[]
    for _ in range(6):
        try:r=forward(v).ravel()-y.ravel()
        except ValueError:break
        J=full_jacobian(v)[:,free]*SPAN[free];s=max(np.max(np.abs(r)),1e-10)
        a=np.vstack([np.c_[J/s,-np.ones(400)],np.c_[-J/s,-np.ones(400)]])
        rhs=np.r_[-r/s,r/s]
        bounds=[(max((LO[k]-v[k])/SPAN[k],-.001),min((HI[k]-v[k])/SPAN[k],.001)) for k in free]+[(0,None)]
        lp=linprog(np.r_[np.zeros(17),1.],A_ub=a,b_ub=rhs,bounds=bounds,method='highs',options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
        if not lp.success:break
        accepted=False
        for fraction in (1,.5,.25,.125):
            proposal=v.copy();proposal[free]+=fraction*SPAN[free]*lp.x[:-1]
            try:error=float(np.max(np.abs(forward(proposal)-y)))
            except ValueError:continue
            if error<np.max(np.abs(r)):v=proposal;history.append(error);accepted=True;break
        if not accepted:break
    return {'coordinate':NAMES[j],'shift':shift,'theta':v.tolist(),'native_difference_from_candidate':(v-base).tolist(),'least_squares_evaluations':opt.nfev,'minimax_history':history,'check':point_check(v,y,eta)}

def run(base,y,eta,target=.001,count=3):
    base=np.asarray(base);y=np.asarray(y);began=time.perf_counter()
    gain=np.sum(np.abs(np.linalg.pinv(full_jacobian(base))),axis=1)
    coords=np.argsort(-gain)[:count]
    rows=[profile(base,y,eta,int(j),s*2.2*target) for j in coords for s in (-1,1)]
    good=[r for r in rows if r.get('check',{}).get('compatible_under_ivx_contract')]
    endpoints=[{'theta':base.tolist(),'check':point_check(base,y,eta)}]+good
    witness=None
    for ia,a in enumerate(endpoints):
        if not a['check'].get('compatible_under_ivx_contract'):continue
        for b in endpoints[ia+1:]:
            delta=np.abs(np.asarray(a['theta'])-b['theta'])
            if delta.max()>2*target and (witness is None or delta.max()>witness['separation']):
                witness={'endpoint_a':a,'endpoint_b':b,'coordinate':NAMES[int(delta.argmax())],'separation':float(delta.max()),'unavoidable_coordinate_error':float(delta.max()/2)}
    return {'eta':eta,'point_accuracy_target':target,'candidate':base.tolist(),'actual_observations':y.tolist(),'profile_results':rows,'actual_record_pair_witness':witness,'complete_compatible_set_enumerated':False,'no_witness_does_not_prove_recovery':True,'seconds':time.perf_counter()-began}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate',required=True);p.add_argument('--observations');p.add_argument('--id');p.add_argument('--eta',required=True,type=float);p.add_argument('--out',required=True);a=p.parse_args()
    r=json.loads(Path(a.candidate).read_text());theta=r.get('theta',r.get('x18'))
    if r.get('actual_observations') is not None:y=np.asarray(r['actual_observations'])
    else:y=np.asarray(next(c for c in json.loads(Path(a.observations).read_text())['cases'] if c['id']==a.id)['observed'])
    result=run(theta,y,a.eta);Path(a.out).write_text(json.dumps(result,indent=2));print(json.dumps({'seconds':result['seconds'],'witness':result['actual_record_pair_witness'],'profiles':len(result['profile_results'])},indent=2))

if __name__=='__main__':main()
