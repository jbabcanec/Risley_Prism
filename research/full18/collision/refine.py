"""Extended full18 refinement, reading only saved candidate and observations."""
from recover import *

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',default='result.json');p.add_argument('--out',default='refined.json')
    p.add_argument('--noise',type=float,default=0);p.add_argument('--seconds',type=float,default=150)
    args=p.parse_args()
    frozen=json.loads((WORK/args.source).read_text());x=np.array(frozen['x18'])
    y=np.array(next(c['observed'] for c in json.loads((WORK.parent/'observations.json').read_text())['cases'] if c['id']==frozen['id']))
    if args.noise:
        k=np.arange(len(y))[:,None];a=np.arange(2)[None,:]
        y=y+args.noise*np.sin((k+1)*(np.sqrt(2)+a*np.sqrt(3)))
    start=time.perf_counter();deadline=start+args.seconds
    initial=x.copy()
    x,mse,nfev=fit(x,y,None,5000,deadline)
    print(json.dumps({'stage':'projected','mse':mse,'nfev':nfev,'seconds':time.perf_counter()-start}),flush=True)
    x,mse=trf(x,y.ravel(),np.ones(y.size,bool),LO,HI,max_nfev=1000)
    _,_,margins=forward_point(x)
    result={'id':frozen['id'],'source':args.source,'source_route':frozen['route'],'seconds':time.perf_counter()-start,
        'x18':x.tolist(),'mse':mse,'max_residual':float(np.max(abs(vec2pat(x)-y))),
        'margins':{k:float(v) for k,v in margins.items()},'noise_amplitude':args.noise,
        'all18_unknown':True,'truth_read_by_solver':False,'projected_nfev':nfev,
        'cumulative_seconds':frozen['seconds']+time.perf_counter()-start,
        'initial_candidate':initial.tolist()}
    (WORK/args.out).write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('x18','initial_candidate')}),flush=True)

if __name__=='__main__':main()
