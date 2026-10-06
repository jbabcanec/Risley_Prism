"""Block-phase multiple shooting with final exact common-clock reconstruction.

Every native coordinate is unknown. Independent per-block phase slips relax
rotor continuity only during initialization. Shared hardware remains common;
the final native model has no slips and uses all original 200 observations.
This is a finite-budget local algorithm, not a global recovery theorem.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import sys, time, json, argparse
sys.dont_write_bytecode=True
from pathlib import Path
from itertools import permutations
import numpy as np
from scipy.optimize import least_squares, lsq_linear
HERE=Path(__file__).resolve().parent
WORK=HERE.parent
sys.path.insert(0,str(WORK/'solver'))
sys.path.insert(0,str(WORK/'collision'))
import poc3
from poc3 import LO,HI,SPAN,LINEAR,NONLINEAR,Projected,WorkLimit
from risley_lattice.spectral import extract_speeds,is_novel
from risley_lattice.fmodel import forward_point


def affine_blocks(theta,slips,count=200,dt=.05,blocks=8):
    """Exact per-axis Snell transfer with a temporary phase slip per block.

    Adapted from poc3.affine; the optical equations and four affine columns
    are identical. Only gamma acquires the explicitly reported block slip.
    """
    source=np.asarray(theta);single=source.ndim==1
    v=np.atleast_2d(source);batch=len(v);dtype=np.result_type(v.dtype,np.asarray(slips).dtype,float)
    slip=np.asarray(slips).reshape(batch,blocks-1,3)
    all_slips=np.concatenate((np.zeros((batch,1,3),dtype=dtype),slip),axis=1)
    block_ids=np.minimum(np.arange(count)*blocks//count,blocks-1)
    gamma=2*np.pi*v[:,:3,None]*(np.arange(count)*dt)+np.pi/180*v[:,6:9,None]
    gamma=gamma+all_slips[:,block_ids,:].transpose(0,2,1)
    wedge=np.sin(np.pi/180*v[:,3:6,None]);tilts=(wedge*np.cos(gamma),wedge*np.sin(gamma))
    base=np.empty((batch,count,2),dtype=dtype);design=np.zeros((batch,count,2,4),dtype=dtype)
    guards=[];penalties=[]
    def positive(x):return np.where(x.real>1e-8,x,1e-8)
    for axis in range(2):
        tangent=np.tan(np.pi/180*v[:,14+axis,None])
        offset=np.broadcast_to(6*tangent,(batch,count)).astype(dtype).copy()
        coeff=np.zeros((batch,count,4),dtype=dtype);coeff[:,:,2+axis]=1
        for i in range(3):
            incoming=tangent/np.sqrt(1+tangent*tangent);n=v[:,9+i,None]
            B=np.sqrt(n*n-incoming*incoming);s=tilts[axis][:,i];w=np.sqrt(1-s*s)
            q=incoming*w+B*s;rad=1-q*q;R=np.sqrt(positive(rad))
            outgoing=q*w-R*s;vertical=R*w+q*s;face=B*w-incoming*s
            guards.extend((rad.real,vertical.real,face.real))
            for guard in (rad,vertical,face):penalties.append(1000*np.where(guard.real<1e-6,guard-1e-6,0))
            tangent=outgoing/positive(vertical);ell=B*R/(positive(vertical)*positive(face))
            offset=ell*(offset+3*incoming/B);coeff=ell[:,:,None]*coeff
            coeff[:,:,1 if i<2 else 0]+=tangent
        base[:,:,axis]=offset;design[:,:,axis]=coeff
    base=base.reshape(batch,-1);design=design.reshape(batch,-1,4)
    guard_values=np.stack(guards,axis=1);penalty=np.concatenate(penalties,axis=1)
    return (base[0],design[0],guard_values[0],penalty[0]) if single else (base,design,guard_values,penalty)


class BlockProjected:
    def __init__(self,seed,y,weight,blocks=8,deadline=np.inf):
        self.seed=np.asarray(seed,float);self.y=np.asarray(y);self.target=self.y.ravel()
        self.weight=weight;self.blocks=blocks;self.deadline=deadline;self.cached=None
        self.data_scale=max(1,float(np.std(self.y,axis=0).max()))
        self.dim=len(NONLINEAR)+3*(blocks-1)
        self.coupling=np.zeros((3*(blocks-1),self.dim))
        for b in range(blocks-1):
            for i in range(3):
                row=b*3+i;self.coupling[row,len(NONLINEAR)+b*3+i]=weight
                if b:self.coupling[row,len(NONLINEAR)+(b-1)*3+i]=-weight
    def fun(self,q):
        if time.perf_counter()>self.deadline:raise WorkLimit('block continuation budget')
        if self.cached is not None and np.array_equal(q,self.cached):return self.residual
        v=self.seed.copy();v[NONLINEAR]=q[:len(NONLINEAR)];slip=q[len(NONLINEAR):]
        base,design,guards,penalty=affine_blocks(v,slip,len(self.y),blocks=self.blocks)
        scaled=design*SPAN[LINEAR];rhs=self.target-base-design@LO[LINEAR]
        linear=np.linalg.lstsq(scaled,rhs,rcond=None)[0];active=np.zeros(4,int)
        if np.any(linear<0) or np.any(linear>1):
            fit=lsq_linear(scaled,rhs,bounds=(0,1),method='bvls',tol=1e-12)
            linear,active=fit.x,fit.active_mask
        v[LINEAR]=LO[LINEAR]+SPAN[LINEAR]*linear
        self.cached=q.copy();self.v=v;self.design=scaled;self.free=active==0
        self.data_residual=scaled@linear-rhs;self.guards=guards
        self.residual=np.r_[self.data_residual/self.data_scale,penalty/self.data_scale,self.coupling@q]
        return self.residual
    def jac(self,q):
        self.fun(q);h=1e-25;d=self.dim
        vv=np.broadcast_to(self.v,(d,18)).astype(complex).copy()
        slips=np.broadcast_to(q[len(NONLINEAR):],(d,3*(self.blocks-1))).astype(complex).copy()
        vv[np.arange(len(NONLINEAR)),NONLINEAR]+=1j*h
        slips[len(NONLINEAR)+np.arange(slips.shape[1]),np.arange(slips.shape[1])]+=1j*h
        base,design,_,penalty=affine_blocks(vv,slips,len(self.y),blocks=self.blocks)
        db=base.imag/h;dA=design.imag/h;derivative=(db+dA@self.v[LINEAR]).T
        if self.free.any():
            af=self.design[:,self.free];inverse=np.linalg.pinv(af,rcond=1e-13)
            dAf=dA[:,:,self.free]*SPAN[LINEAR][self.free]
            correction=np.einsum('kmi,m->ik',dAf,self.data_residual)
            derivative=derivative-af@(inverse@derivative)-inverse.T@correction
        return np.vstack((derivative/self.data_scale,penalty.imag.T/h/self.data_scale,self.coupling))


def evaluate(v,y):
    delta=poc3.canonical(v)-y
    result={'theta':np.asarray(v).tolist(),'mse':float(np.mean(delta*delta)),
            'rms':float(np.sqrt(np.mean(delta*delta))),'max_residual':float(np.max(abs(delta)))}
    try:
        fm,fr,guards=forward_point(v);result.update(physical=True,physical_guards=guards,
                    interval_residual_upper=float(np.max(abs(fm-y.ravel())+fr)))
    except Exception as exc:result.update(physical=False,physical_error=str(exc))
    return result


def make_seeds(y):
    seeds=[];bases=[];info_fft={}
    for frontend in ('pencil','fft'):
        try:
            N,info=extract_speeds(y,.05,frontend=frontend)
            if frontend=='fft':info_fft=info
            if N is not None and all(np.max(abs(np.sort(N)-np.sort(old)))>1e-4 for old in bases):
                bases.append(np.asarray(N));
                # Both nominal starting families cover all six physical orders.
                for v,tag in poc3.initializations(y,N,.05):
                    if 'n0=1.7' not in tag:seeds.append((v,frontend+'/'+tag))
        except Exception:pass
    lines=np.asarray(info_fft.get('lines',[]));amps=np.asarray(info_fft.get('line_amplitudes',[]));selected=[]
    for j in np.argsort(-amps):
        if 1e-7<abs(lines[j])<=3.5 and is_novel(lines[j],selected):selected.append(float(lines[j]))
        if len(selected)==2:break
    if len(selected)==2:
        # Observation-only cancellation-aware proposals; all speeds are free in
        # the block solve, including whether these proposed repeats persist.
        from recover import starts
        repeated,_=starts(y,np.array(selected),lines)
        for v,_,tag in repeated:
            if 'ph=(-16.0, 16.0)' in tag or 'ph=(16.0, -16.0)' in tag:
                seeds.append((v,'repeated/'+tag))
    return seeds,{'bases':[b.tolist() for b in bases],'two_line_proposals':selected}


def solve(y,seconds=70,blocks=8):
    start=time.perf_counter();deadline=start+seconds
    seeds,spectrum=make_seeds(y);events=[];best=None;screen=[]
    def emit(record):events.append(record);print(json.dumps(record),flush=True)
    def update(v,tag):
        nonlocal best
        candidate=evaluate(v,y);candidate['tag']=tag
        if candidate['physical'] and (best is None or candidate['mse']<best['mse']):best=candidate
    lo=np.r_[LO[NONLINEAR],np.full(3*(blocks-1),-np.pi)]
    hi=np.r_[HI[NONLINEAR],np.full(3*(blocks-1),np.pi)]
    scale=np.r_[SPAN[NONLINEAR],np.full(3*(blocks-1),2*np.pi)]
    stage='screen'
    try:
        # Reserve half the wall budget for exact-clock continuation/polishing.
        screen_deadline=min(deadline,time.perf_counter()+seconds*.45)
        for index,(seed,tag) in enumerate(seeds):
            obj=BlockProjected(seed,y,.01,blocks,screen_deadline)
            q0=np.r_[seed[NONLINEAR],np.zeros(3*(blocks-1))]
            try:
                fit=least_squares(obj.fun,q0,jac=obj.jac,bounds=(lo,hi),x_scale=scale,
                                  max_nfev=16,ftol=1e-9,xtol=1e-9,gtol=1e-9)
                obj.fun(fit.x)
            except WorkLimit:break
            score=float(np.mean(obj.data_residual**2))
            screen.append((score,obj.v.copy(),fit.x.copy(),tag))
            emit({'stage':'screen','index':index,'tag':tag,'relaxed_mse':score,
                  'max_phase_slip_rad':float(np.max(abs(fit.x[len(NONLINEAR):]))),
                  'nfev':fit.nfev,'seconds':time.perf_counter()-start})
        screen.sort(key=lambda item:item[0]);stage='continuation'
        for rank,(_,v,q,tag) in enumerate(screen[:3]):
            # Each finalist has a fair independent time share.
            local_deadline=time.perf_counter()+max(0,deadline-time.perf_counter())/max(1,min(3,len(screen))-rank)
            for weight in (.1,1.,10.,100.):
                obj=BlockProjected(v,y,weight,blocks,local_deadline)
                try:
                    fit=least_squares(obj.fun,q,jac=obj.jac,bounds=(lo,hi),x_scale=scale,
                                      max_nfev=35,ftol=1e-10,xtol=1e-10,gtol=1e-10)
                    obj.fun(fit.x);q=fit.x;v=obj.v.copy()
                except WorkLimit:break
                emit({'stage':'continuation','rank':rank,'weight':weight,'tag':tag,
                      'relaxed_mse':float(np.mean(obj.data_residual**2)),
                      'max_phase_slip_rad':float(np.max(abs(q[len(NONLINEAR):]))),
                      'nfev':fit.nfev,'seconds':time.perf_counter()-start})
            update(v,'pre_exact/'+tag)
            # Final stage removes every temporary block phase, imposes the
            # exact common clock, and releases all native nonlinear coordinates.
            obj=Projected(v,y,.05,deadline=deadline)
            try:
                fit=least_squares(obj.fun,v[NONLINEAR],jac=obj.jac,
                                  bounds=(LO[NONLINEAR],HI[NONLINEAR]),x_scale=SPAN[NONLINEAR],
                                  max_nfev=450,ftol=1e-13,xtol=1e-13,gtol=1e-13)
                obj.fun(fit.x);update(obj.v,'exact_clock/'+tag)
                emit({'stage':'exact_clock','rank':rank,'mse':best['mse'] if best else None,
                      'nfev':fit.nfev,'seconds':time.perf_counter()-start})
            except WorkLimit:pass
            if best is not None and best['max_residual']<1e-8:break
    except WorkLimit:pass
    return {'status':'candidate_fit' if best and best['max_residual']<1e-8 else 'unresolved',
            'best':best,'events':events,'spectrum':spectrum,'screened':len(screen),'seed_count':len(seeds),
            'seconds':time.perf_counter()-start,'budget_seconds':seconds,'blocks':blocks,
            'all18_unknown':True,'truth_read':False,'physical_order_not_canonicalized':True,
            'final_model':'one shared native18 vector, original200 clock, no block phase slips',
            'global_certificate':False,'method':'shared-hardware block-phase continuation + exact-clock VarPro'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--id',required=True);p.add_argument('--seconds',type=float,default=70)
    p.add_argument('--blocks',type=int,default=8);p.add_argument('--input',default='observations.json')
    p.add_argument('--prefix',default='');args=p.parse_args()
    data=json.loads((WORK/args.input).read_text())
    row=next(c for c in data['cases'] if c['id']==args.id);y=np.asarray(row['observed'])
    result=solve(y,args.seconds,args.blocks);result['id']=args.id
    result['observation_input']=args.input
    (HERE/(args.prefix+args.id+'_blocks.json')).write_text(json.dumps(result,indent=2))
    print(json.dumps({'id':args.id,'status':result['status'],'seconds':result['seconds'],
                      'best_mse':result['best']['mse'] if result['best'] else None}),flush=True)

if __name__=='__main__':main()
