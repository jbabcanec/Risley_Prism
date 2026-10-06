import sys
sys.dont_write_bytecode=True
sys.path.insert(0,'C:\\Users\\josep\\Documents\\Codex\\2026-10-02\\task\\full18_research\\low_frequency')
"""Same data-only DC/slow-frequency profile and physical completion policy.

Spectral fits always include a separate zero-frequency coefficient. Candidate
triples are proposals only; final fitting frees all18 with native bounds.
"""
from recover_low import *
from itertools import combinations
from scipy.optimize import least_squares
sys.path.insert(0,str(WORK.parent/'collision'))
from recover import fit,WorkLimit


def proposals(y,diagnostic):
    z=y[:,0]+1j*y[:,1];t=np.arange(len(y))*.05;mask=np.ones(len(y),bool)
    raw=[];mandatory=[];low=[];fast=[]
    for front in ('pencil','fft'):
        d=diagnostic['original'][front]
        pairs=sorted(zip(d['lines'],d['line_amplitudes']),key=lambda x:-x[1])
        kept=[(f,a) for f,a in pairs if 1e-5<abs(f)<3.5][:8]
        fs=[x[0] for x in kept]
        mandatory.append(np.array(fs[:3]))
        raw.extend((np.array(g),f'{front}/observed') for g in combinations(fs,3))
        low.extend(f for f,a in kept if abs(f)<.1)
        fast.extend((f,a) for f,a in kept if abs(f)>=.1)
    distinct=[]
    for f,a in sorted(fast,key=lambda x:-x[1]):
        if all(abs(f-p)>.025 for p in distinct):distinct.append(f)
        if len(distinct)>=8:break
    slow=low+[-.09,-.06,-.03,.03,.06,.09]
    for pair in combinations(distinct,2):
        for s in slow:raw.append((np.r_[pair,s],'explicit_slow_grid'))
    candidates=[]
    for v,tag in raw:
        g,K,c,rel=lattice.lattice_fit(z,t,mask,v,B=3,iters=5,sharp=False)
        if not np.isfinite(rel) or np.max(abs(g))>=3.5:continue
        # The signed dominant fundamental is data derived, not supplied.
        for j in range(3):
            row=np.eye(3)[j]
            pos=np.flatnonzero(np.all(K==row,axis=1));neg=np.flatnonzero(np.all(K==-row,axis=1))
            if len(pos) and len(neg) and abs(c[neg[0]])>abs(c[pos[0]]):g[j]*=-1
        score=float(rel*(1+.3*lattice.parsimony(K,c)))
        candidates.append((score,g,tag,float(rel)))
    chosen=[(None,g,'mandatory_top3',None) for g in mandatory]
    for row in sorted(candidates,key=lambda x:x[0]):
        g=np.array(sorted(row[1]))
        if all(np.max(abs(g-np.sort(v[1])))>.015 for v in chosen):chosen.append(row)
        if len(chosen)>=12:break
    return chosen,len(raw)


def main():
    p=argparse.ArgumentParser();p.add_argument('--id',required=True);p.add_argument('--seconds',type=float,default=120);args=p.parse_args()
    y=np.array(next(c['observed'] for c in json.loads(Path('C:\\Users\\josep\\Documents\\Codex\\2026-10-02\\task\\full18_research\\low_frequency\\noise_1e4\\random_03_profile_run\\observations.json').read_text())['cases'] if c['id']==args.id))
    diag=spectral_diagnostic(y)
    start=time.perf_counter();deadline=start+args.seconds
    # Keep the original ridge/merge behavior for an independent profile tactic;
    # the DC row in K is retained exactly, no SPEED_MIN extraction is used.
    candidates,count=proposals(y,diag)
    result={'id':args.id,'spectral_candidates':[dict(score=sc,speeds=g.tolist(),kind=tag,relative_residual=rr) for sc,g,tag,rr in candidates],
        'raw_profile_count':count,'spectral_seconds':time.perf_counter()-start,'attempts':[],
        'all18_unknown':True,'truth_read_by_solver':False,'accuracy_certified':False}
    out=Path('C:\\Users\\josep\\Documents\\Codex\\2026-10-02\\task\\full18_research\\low_frequency\\noise_1e4\\random_03_profile.json');screens=[];best=None;cost=np.inf
    def save(stage):
        result.update(stage=stage,seconds=time.perf_counter()-start,theta=None if best is None else best.tolist())
        if best is not None:
            d=vec2pat(best)-y;result.update(rms=float(np.sqrt(np.mean(d*d))),max_residual=float(np.max(abs(d))))
            try:
                _,_,m=forward_point(best);result['physical_margins']={k:float(v) for k,v in m.items()}
            except Exception as exc:result['physical_check_error']=repr(exc)
        out.write_text(json.dumps(result,indent=2))
    print(json.dumps({'id':args.id,'profile_count':count,'spectral_seconds':result['spectral_seconds'],
        'bases':result['spectral_candidates']}),flush=True)
    try:
        for j,(_,g,tag,_) in enumerate(candidates):
            for x,label in poc3.initializations(y,g,.05):
                x=np.clip(x,LO+1e-10,HI-1e-10)
                try:x,mse,nfev=fit(x,y,None,12,deadline)
                except (ValueError,FloatingPointError,np.linalg.LinAlgError):continue
                screens.append((mse,x,f'{j}/{tag}/{label}'))
                if mse<cost:best,cost=x,mse
            save('screen')
            print(json.dumps({'id':args.id,'base':j,'best_rms':float(np.sqrt(cost)),'seconds':time.perf_counter()-start}),flush=True)
        screens.sort(key=lambda r:r[0])
        for rank,(_,x,tag) in enumerate(screens[:12]):
            x,mse,nfev=fit(x,y,None,1000,deadline)
            result['attempts'].append(dict(rank=rank,route=tag,mse=mse,nfev=nfev))
            if mse<cost:best,cost=x,mse
            save('polish')
            print(json.dumps({'id':args.id,'polish':rank,'mse':mse,'seconds':time.perf_counter()-start}),flush=True)
            if mse<1e-22:break
        if best is not None and time.perf_counter()<deadline:
            x,mse=trf(best,y.ravel(),np.ones(y.size,bool),LO,HI,max_nfev=100)
            if mse<cost:best,cost=x,mse
        save('complete')
    except WorkLimit:save('work_limit')


if __name__=='__main__':main()
