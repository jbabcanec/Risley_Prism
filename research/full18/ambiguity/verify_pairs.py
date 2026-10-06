"""Independent arbitrary-precision interval endpoint checks for ambiguity pairs.

Uses exact rational k/20 times and exact binary-rational submitted parameters.
This is directed interval numerical evidence, not a Lean formalization or a
verification of Python/mpmath itself. No optimizer or repository code imported.
"""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse, hashlib, json, math, time
from pathlib import Path
from mpmath import iv
from profile_pairs import NAMES,LO,HI

def exact(value):
    n,d=float(value).as_integer_ratio()
    return iv.mpf(n)/iv.mpf(d)

def bounds(value):
    # nextafter avoids inward rounding when interval endpoints become binary64.
    return math.nextafter(float(value.a),-math.inf),math.nextafter(float(value.b),math.inf)

def abs_upper(value):return max(abs(x) for x in bounds(value))

def optical_intervals(theta):
    """Independent scalar interval trace of the stated strict per-axis model."""
    v=[exact(x) for x in theta];one=iv.mpf(1);zero=iv.mpf(0)
    deg=iv.pi/180
    zs=[iv.mpf(6),iv.mpf(9),9+v[13],12+v[13],12+2*v[13],15+2*v[13],15+2*v[13]+v[12]]
    guards={'tir':math.inf,'fwd':math.inf,'graze':math.inf}
    result=[]
    for k in range(200):
        t=iv.mpf(k)/20
        rotor=[2*iv.pi*v[j]*t+deg*v[6+j] for j in range(3)]
        tan_wedge=[iv.tan(deg*v[3+j]) for j in range(3)]
        cs=[iv.cos(rotor[j])*tan_wedge[j] for j in range(3)]
        sn=[iv.sin(rotor[j])*tan_wedge[j] for j in range(3)]
        for axis,(u,w) in enumerate(((cs,sn),(sn,cs))):
            normal=[iv.sqrt(one+u[j]**2+w[j]**2) for j in range(3)]
            q=[iv.sqrt(one+w[j]**2) for j in range(3)]
            face_s=[zero,u[0]/normal[0],zero,u[1]/normal[1],zero,u[2]/normal[2]]
            face_c=[one,q[0]/normal[0],one,q[1]/normal[1],one,q[2]/normal[2]]
            slope=[zero,u[0]/q[0],zero,u[1]/q[1],zero,u[2]/q[2],zero]
            ratio=[one/v[9],v[9],one/v[10],v[10],one/v[11],v[11]]
            tangent=iv.tan(deg*v[14+axis]);p=v[16+axis]+6*tangent;z=iv.mpf(6)
            for j in range(6):
                norm_in=iv.sqrt(one+tangent**2)
                transverse=tangent/norm_in;vertical=one/norm_in
                cy=-face_c[j]*transverse-face_s[j]*vertical
                tir=one-(ratio[j]*cy)**2
                guards['tir']=min(guards['tir'],bounds(tir)[0])
                if bounds(tir)[0]<=0:raise ValueError('TIR/undecided branch')
                root=iv.sqrt(tir)
                out0=-ratio[j]*face_c[j]*cy-face_s[j]*root
                outz=-ratio[j]*face_s[j]*cy+face_c[j]*root
                guards['fwd']=min(guards['fwd'],bounds(outz)[0])
                if bounds(outz)[0]<=0:raise ValueError('not forward')
                tangent=out0/outz
                denominator=one-slope[j+1]*tangent
                dl,du=bounds(denominator)
                if dl<=0<=du:raise ValueError('grazing/undecided')
                guards['graze']=min(guards['graze'],min(abs(dl),abs(du)))
                step=(zs[j+1]+slope[j+1]*p-z)/denominator
                p+=step*tangent;z+=step
            result.append(p)
    return result,guards

def verify(record,dps):
    iv.dps=dps
    a,b=record['base'],record['alternative']
    assert all(LO[j]<=x<=HI[j] for j,x in enumerate(a))
    assert all(LO[j]<=x<=HI[j] for j,x in enumerate(b))
    fa,ga=optical_intervals(a);fb,gb=optical_intervals(b)
    diff=[abs_upper(x-y) for x,y in zip(fa,fb)]
    # Produce an actual shared observation vector, stored as exact binary64
    # rationals. Verify both errors to this record, including its rounding.
    y=[float((x+z).mid/2) for x,z in zip(fa,fb)]
    ea=[abs_upper(x-exact(z)) for x,z in zip(fa,y)]
    eb=[abs_upper(x-exact(z)) for x,z in zip(fb,y)]
    parameter_bounds=[bounds(abs(exact(a[j])-exact(b[j]))/2) for j in range(18)]
    return {'status':'DIRECTED_ARBITRARY_PRECISION_INTERVAL_ENDPOINT_WITNESS',
            'model':'canonical strict per-axis optical equations; no clamps',
            'timing':'exact k/20 for k=0,...,199; passive single plane',
            'arithmetic':'mpmath.iv directed interval arithmetic; not machine-checked Lean',
            'dps':dps,'inside_native_box':True,'prism_order_unchanged':True,
            'guards_base':ga,'guards_alternative':gb,'base':a,'alternative':b,
            'parameter_difference':[b[j]-a[j] for j in range(18)],
            'minimax_coordinate_lower_bound':[p[0] for p in parameter_bounds],
            'minimax_coordinate_upper_bound':[p[1] for p in parameter_bounds],
            'max_position_difference_upper':max(diff),'xy_difference_upper':[max(diff[::2]),max(diff[1::2])],
            'common_observations':y,'common_observation_error_upper':max(ea+eb),
            'base_observation_error_upper':max(ea),'alternative_observation_error_upper':max(eb),
            'interval_max_width_upper':max(bounds(x)[1]-bounds(x)[0] for x in fa+fb),
            'theorem':'For eta >= common_observation_error_upper, every estimator errs by at least |a_j-b_j|/2 on at least one of these two admissible truths, separately for each coordinate j.',
            'scope':'finite-noise obstruction; no claim of exact equality, global nonidentifiability, or algorithm failure'}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',default='profile_candidates.json');ap.add_argument('--out',default='verified_pairs.json');ap.add_argument('--dps',type=int,default=60);ap.add_argument('--indices',default='0,2,4,6,8,10');args=ap.parse_args()
    records=json.loads(Path(args.input).read_text())['records'];indices=[int(x) for x in args.indices.split(',')]
    result=[];start=time.time()
    for idx in indices:
        check=verify(records[idx],args.dps);check['source_record_index']=idx
        check['family']=records[idx].get('family');check['wedge_scale']=records[idx].get('wedge_scale')
        result.append(check)
        print(json.dumps({'index':idx,'eta':check['common_observation_error_upper'],'dw_floor':check['minimax_coordinate_lower_bound'][12],'guards':check['guards_base']}),flush=True)
        Path(args.out).write_text(json.dumps({'names':NAMES,'input_sha256':hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),'verification_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'records':result,'elapsed_seconds':time.time()-start},indent=2))

if __name__=='__main__':main()
