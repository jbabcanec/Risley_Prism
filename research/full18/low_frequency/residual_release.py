"""Bounded observation-only residual-frequency release for unresolved 07.

No parameter truth is accepted. Every optical fit frees all18 coordinates.
"""
from profile_bases import *
from itertools import permutations
from scipy.signal import find_peaks


def main():
    ident='random_07';start=time.perf_counter();deadline=start+120
    y=np.array(next(c['observed'] for c in json.loads((WORK.parent/'combined_holdout_observations.json').read_text())['cases'] if c['id']==ident))
    sources=[WORK/(ident+'.json'),WORK/(ident+'_profile.json'),WORK.parent/'combined_validation'/'noiseless'/(ident+'.json')]
    candidates=[]
    for source in sources:
        row=json.loads(source.read_text());v=np.array(row['theta']);mse=float(np.mean((vec2pat(v)-y)**2))
        candidates.append((mse,v,str(source)))
    cost,best,source=min(candidates,key=lambda x:x[0]);base=best.copy()
    weak=int(np.argmin(abs((base[9:12]-1)*np.tan(np.radians(base[3:6])))))
    strong=[i for i in range(3) if i!=weak]
    r=y-vec2pat(base);z=r[:,0]+1j*r[:,1];t=np.arange(len(y))*.05
    ft=np.fft.fftfreq(65536,.05);sp=np.abs(np.fft.fft((z-z.mean())*np.hanning(len(z)),65536))
    peaks,_=find_peaks(sp);peaks=sorted([i for i in peaks if 1e-5<abs(ft[i])<=3.5],key=lambda i:-sp[i])[:30]
    fs=[float(ft[i]) for i in peaks]
    diag=json.loads((WORK/(ident+'.json')).read_text())['diagnostic']
    observed=[]
    for fr in ('pencil','fft'):observed.extend(diag['original'][fr]['lines'])
    for f in observed:
        for a in range(-2,3):
            for b in range(-2,3):
                if abs(a)+abs(b)<=2:
                    q=f-a*base[strong[0]]-b*base[strong[1]]
                    if 1e-5<abs(q)<=3.5:fs.append(float(q))
    scored=[]
    for f in fs:
        E=np.column_stack([np.ones(len(y)),np.exp(2j*np.pi*f*t)])
        c=np.linalg.lstsq(E,z,rcond=None)[0];rr=z-E@c
        scored.append((float(np.mean(abs(rr)**2)),f,complex(c[1])))
    chosen=[(np.inf,f,0j) for f in (-.005,.005,-.02,.02)]
    for item in sorted(scored):
        if all(abs(item[1]-j[1])>.02 for j in chosen):chosen.append(item)
        if len(chosen)>=16:break
    result={'id':ident,'source':source,'weak_prism_index_zero_based':weak,
        'initial_mse':cost,'frequency_proposals':[f for _,f,_ in chosen],
        'all18_unknown':True,'truth_read_by_solver':False,'global_accuracy_certified':False,'attempts':[]}
    out=WORK/(ident+'_release.json');screens=[]
    def save(stage):
        residual=vec2pat(best)-y
        result.update(stage=stage,seconds=time.perf_counter()-start,theta=best.tolist(),
            rms=float(np.sqrt(np.mean(residual*residual))),max_residual=float(np.max(abs(residual))))
        try:
            _,_,m=forward_point(best);result['physical_margins']={k:float(v) for k,v in m.items()}
        except Exception as exc:result['physical_check_error']=repr(exc)
        out.write_text(json.dumps(result,indent=2))
    print(json.dumps({'source':source,'weak':weak,'frequencies':result['frequency_proposals'],'rms':float(np.sqrt(cost))}),flush=True)
    try:
        # Spectral peaks first; DC-adjacent probes remain in the declared set.
        for rank,(_,f,c) in enumerate(chosen[4:]+chosen[:4]):
            for order in permutations(range(3)):
                order=np.array(order)
                for phase in (-12.,0.,12.):
                    v=base.copy();v[weak]=f;v[6+weak]=phase
                    for block in (0,3,6,9):v[block:block+3]=v[block:block+3][order]
                    v=np.clip(v,LO+1e-10,HI-1e-10)
                    try:x,mse,nfev=fit(v,y,None,12,deadline)
                    except (ValueError,FloatingPointError,np.linalg.LinAlgError):continue
                    tag=f'frequency={f}/order={order.tolist()}/phase={phase}'
                    screens.append((mse,x,tag))
                    if mse<cost:best,cost=x,mse
            save('screen')
            print(json.dumps({'screen_frequency':rank,'best_mse':cost,'seconds':time.perf_counter()-start}),flush=True)
            if time.perf_counter()-start>65:break
        for rank,(_,v,tag) in enumerate(sorted(screens,key=lambda x:x[0])[:8]):
            x,mse,nfev=fit(v,y,None,1000,deadline)
            result['attempts'].append(dict(stage='polish',rank=rank,route=tag,mse=mse,nfev=nfev))
            if mse<cost:best,cost=x,mse
            save('polish')
            print(json.dumps({'polish':rank,'mse':mse,'seconds':time.perf_counter()-start}),flush=True)
            if mse<1e-22:break
        if time.perf_counter()<deadline:
            x,mse=trf(best,y.ravel(),np.ones(y.size,bool),LO,HI,max_nfev=100)
            if mse<cost:best,cost=x,mse
        save('complete')
    except WorkLimit:save('work_limit')


if __name__=='__main__':main()
