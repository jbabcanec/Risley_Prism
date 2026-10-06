"""Nonuniform coarse ray-state constrained solve followed by full200 polish."""
import json,argparse,time
import numpy as np
from scipy.optimize import least_squares
from phase_blocks import HERE,WORK,evaluate,poc3,Projected,LO,HI,SPAN,NONLINEAR,WorkLimit
from risley_lattice.ray_state_initialization import initialize_ray_states
from risley_lattice.ray_state_inverse import RayStateOptions
from risley_lattice.ray_state_constrained import solve_ray_states_constrained

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--id',required=True)
    parser.add_argument('--seconds',type=float,default=70);args=parser.parse_args()
    start=time.perf_counter();data=json.loads((WORK/'observations.json').read_text())
    y=np.asarray(next(r['observed'] for r in data['cases'] if r['id']==args.id));times=np.asarray(data['timestamps'])
    # Preserve a dense initial clock section and dispersed later nodes. This
    # removes the uniform coarse-grid alias while reducing latent dimensions.
    indices=np.unique(np.r_[np.arange(8),np.linspace(8,199,32,dtype=int)])
    proposals=initialize_ray_states(y,times,starts=6,deadline=time.monotonic()+5)
    starts=[]
    for proposal in proposals:
        item=dict(proposal);item['outgoing_directions']=proposal['outgoing_directions'][indices];starts.append(item)
    events=[]
    def progress(e):
        keep={k:v for k,v in e.items() if k in ('event','start','iteration','elapsed_seconds','optimizer_optimality','optimizer_constraint_violation','rotor_max_abs','normal_closure_max_abs','inequality_max_violation','status')}
        events.append(keep)
        if e.get('event')=='start_complete':print(json.dumps(keep),flush=True)
    result=solve_ray_states_constrained(y[indices],times[indices],initializations=starts,
        options=RayStateOptions(starts=6,max_seconds=args.seconds*.65,outer_iterations=6,
                max_nfev_per_outer=60,max_residual_evaluations=20000,progress_every_evaluations=100),progress=progress)
    candidates=list(result['candidates'])
    if result['best_unresolved'] is not None:candidates.append(result['best_unresolved'])
    # Also preserve accepted states from each start, whose original selection
    # balances lifted residual and continuity rather than full-record residual.
    from risley_lattice.ray_state_inverse import _diagnostics
    for attempt in result['attempts']:
        vector=attempt.get('last_accepted_optimizer_vector')
        if vector is not None:
            candidates.append(_diagnostics(np.asarray(vector),y[indices],times[indices],np.zeros(2),RayStateOptions()))
    scored=[]
    for candidate in candidates:
        native=evaluate(candidate['parameters'],y)
        native.update(lifted_rms=np.asarray(candidate['lifted_axis_rms']).tolist(),
                      rotor_max=float(candidate['rotor_max_abs']),normal_closure=float(candidate['normal_closure_max_abs']))
        scored.append(native)
    scored.sort(key=lambda r:r['mse']);best=scored[0] if scored else None
    polishes=[];deadline=start+args.seconds
    for candidate in scored[:3]:
        obj=Projected(np.asarray(candidate['theta']),y,.05,deadline=deadline)
        try:
            fit=least_squares(obj.fun,np.asarray(candidate['theta'])[NONLINEAR],jac=obj.jac,
                 bounds=(LO[NONLINEAR],HI[NONLINEAR]),x_scale=SPAN[NONLINEAR],max_nfev=450,
                 ftol=1e-13,xtol=1e-13,gtol=1e-13)
            obj.fun(fit.x);check=evaluate(obj.v,y);polishes.append(check)
            if check['physical'] and (best is None or check['mse']<best['mse']):best=check
        except WorkLimit:break
        if best and best['max_residual']<1e-8:break
    report={'id':args.id,'method':'nonuniform40-node explicit rotor-equality trust-constr followed by full200 exact-clock VarPro',
            'status':'candidate_fit' if best and best['max_residual']<1e-8 else 'unresolved',
            'best':best,'coarse_candidates':scored,'polishes':polishes,'events':events,
            'seconds':time.perf_counter()-start,'coarse_indices':indices.tolist(),
            'coarse_stop_reason':result['stop_reason'],'coarse_evaluations':result['residual_evaluations'],
            'truth_read':False,'all18_unknown':True,'global_certificate':False,
            'initialization_orders':[list(p['diagnostics']['physical_order']) for p in proposals]}
    (HERE/(args.id+'_constrained.json')).write_text(json.dumps(report,indent=2))
    print(json.dumps({'id':args.id,'status':report['status'],'seconds':report['seconds'],
                      'best_mse':best['mse'] if best else None}),flush=True)

if __name__=='__main__':main()
