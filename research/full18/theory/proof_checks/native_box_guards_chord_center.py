#!/usr/bin/env python3
"""One declared original-native +/- 1/1000 box; no search or iterates.

Uses the exact Q-scaled Snell/traversal circuit of finite_wedge_point_guards.py
and the certified point derivative of finite_wedge_corrector_derivative.py.
Every interval calculation is outward mpmath interval arithmetic at 70 dps.
JSON includes outward binary endpoints AND exact internal dyadic endpoints.
"""
from pathlib import Path
import hashlib,json,math,time
import mpmath as mp
import numpy as np

iv=mp.iv; iv.dps=70
D=18
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'finite_wedge_corrector_interval.json'
OUT=ROOT/'native_box_guards_chord_center.json'

def I(x):return iv.mpf(x)
def lo(x):return math.nextafter(float(x.a),-math.inf)
def hi(x):return math.nextafter(float(x.b),math.inf)
def interval(x):
    return {'lower':lo(x),'upper':hi(x),'dyadic':[[int(y) for y in z] for z in x._mpi_]}
def vec(v):return [interval(x) for x in v]
def mat(A):return [vec(row) for row in A]
def norminf(A):return hi(max((sum(abs(x) for x in row) for row in A),key=lambda x:float(x.b)))
def zeros():return np.array([[I(0) for j in range(D)] for i in range(D)],dtype=object)
def ident():return np.array([[I(int(i==j)) for j in range(D)] for i in range(D)],dtype=object)
def binarymat(A):return np.array([[I(float(x)) for x in row] for row in A],dtype=object)
def require(name,x,where):
    if not lo(x)>0:raise ArithmeticError(f'{name} is not strictly positive at {where}: {x}')

r=I(1)/1000
pi=iv.pi
kappa=I(1)/10
w0=180/pi*iv.atan2(kappa,iv.sqrt(1-kappa**2))
beta0=180/pi*iv.atan2(I(1),I(3))
nsq0=[I(17)/8,I(233)/125,I(13)/5]
center=[I(1)/20,I(7)/20,I(49)/20]+[w0]*3+[I(0)]*3+[iv.sqrt(x) for x in nsq0]+[beta0,I(0),I(1),I(2),I(3),I(100)]
coords=['N1_Hz','N2_Hz','N3_Hz','a1_deg','a2_deg','a3_deg','phi1_deg','phi2_deg','phi3_deg','n1','n2','n3','beta_x_deg','beta_y_deg','b_x','b_y','g','d']
box=[x+iv.mpf([-1,1])*r for x in center]
priors=[(-3.5,3.5)]*3+[(-18,18)]*6+[(1.3,1.8)]*3+[(-25,25)]*2+[(-5,5)]*2+[(2,15),(50,200)]
# Exact rational decimal prior endpoints, not binary approximations to 1.3/1.8.
prior_checks=[]
for name,x,(low,high) in zip(coords,box,priors):
    a=x-I(str(low));b=I(str(high))-x
    require('prior lower',a,name);require('prior upper',b,name)
    prior_checks.append({'coordinate':name,'prior':[str(low),str(high)],'lower_slack':interval(a),'upper_slack':interval(b)})
N=box[:3];wedges=[x*pi/180 for x in box[3:6]];phi=[x*pi/180 for x in box[6:9]];n=box[9:12]
t=[iv.tan(x*pi/180) for x in box[12:14]];b=box[14:16];g,d=box[16:18]
rho=sum(x**2 for x in t);Q=1+rho;nsq=[x**2 for x in n]
# h_j = sqrt(n_j^2+(n_j^2-1)rho) retains the common native-index/slope expression.
h=[iv.sqrt(x+(x-1)*rho) for x in nsq]
eta_box=N+[iv.sin(x)/kappa for x in wedges]+phi+h+t+b+[g,d]

def physical(lam):
    """One full parameter box, all times; optional common wedge-sine lambda."""
    mins={};rows=[]
    for k in range(200):
        X=t[:];p=[b[a]+6*t[a] for a in range(2)]
        incoming_Z=I(1)
        for j in range(3):
            ang=2*pi*N[j]*I(k)/20+phi[j]
            e=lam*iv.sin(wedges[j])
            sqden=1-e**2
            require('wedge_sqrt',sqden,(k,j))
            tilt=e/iv.sqrt(sqden)
            u=[tilt*iv.cos(ang),tilt*iv.sin(ang)]
            Dj=1/sqden # exactly 1+|u|^2, uses cos^2+sin^2=1
            rad=(nsq[j]+(nsq[j]-1)*rho) if j==0 else nsq[j]*Q-sum(v**2 for v in X)
            require('glass_radicand',rad,(k,j))
            H=iv.sqrt(rad);P=H-sum(u[a]*X[a] for a in range(2))
            require('incoming_normal_scaled_numerator',P,(k,j))
            Delta=P**2-Dj*(nsq[j]-1)*Q
            require('exit_discriminant_scaled',Delta,(k,j))
            E=iv.sqrt(Delta)
            # Rationalized exact Snell coefficient; removes P-E cancellation.
            c=(nsq[j]-1)*Q/(P+E)
            Z=H-c;Xn=[X[a]+u[a]*c for a in range(2)]
            A=3+sum(u[a]*p[a] for a in range(2))
            pe=[p[a]+X[a]*A/P for a in range(2)]
            ell=g if j<2 else d
            ext=ell-sum(u[a]*pe[a] for a in range(2))
            values={
                'wedge_sqrt_radicand':sqden,
                'incoming_air_axial_scaled':incoming_Z,
                'glass_radicand_scaled':rad,
                'glass_radicand_unit_incident':rad/Q,
                'incoming_normal_scaled_numerator':P,
                'incoming_glass_unit_normal_projection':P/(n[j]*iv.sqrt(Q*Dj)),
                'exit_discriminant_scaled':Delta,
                'exit_discriminant_unit_ray_unnormalized_normal':Delta/Q,
                'exit_snell_unit_normal_radicand':Delta/(Q*Dj),
                'outgoing_axial_scaled':Z,
                'outgoing_axial_unit':Z/iv.sqrt(Q),
                'outgoing_normal_scaled_numerator':E,
                'outgoing_unit_ray_unnormalized_normal':E/iv.sqrt(Q),
                'outgoing_unit_normal_projection':E/iv.sqrt(Q*Dj),
                'internal_traversal_numerator':A,
                'internal_axial_travel':H*A/P,
                'external_axial_travel':ext,
                'exit_intersection_denominator':P/H,
            }
            for key,value in values.items():
                require(key,value,(k,j))
                if key not in mins or lo(value)<mins[key]['lower']:
                    mins[key]={'lower':lo(value),'sample':k,'prism':j+1}
            rows.append({'sample':k,'prism':j+1,'guards':{key:interval(value) for key,value in values.items()}})
            p=[pe[a]+Xn[a]*ext/Z for a in range(2)];X=Xn;incoming_Z=Z
    return {'passed':True,'lambda':interval(lam),'sample_count':200,'prism_guard_records':rows,'minima':mins}

def inverse():
    src=json.loads(SOURCE.read_text());assert src['interval'] is True and src['kappa']==0.1
    assert src['eta_coordinates']==['N1','N2','N3','e1/kappa','e2/kappa','e3/kappa','phi1','phi2','phi3','h1','h2','h3','tx','ty','bx','by','g','d']
    Qeta=np.array([[iv.mpf([a,b]) for a,b in zip(rowa,rowb)] for rowa,rowb in zip(src['DT_interval_lower'],src['DT_interval_upper'])],dtype=object)
    C=ident();J=ident();Q0=I(10)/9;t0=[I(1)/3,I(0)];h0=[I(3)/2,I(7)/5,I(5)/3];n0=[iv.sqrt(x) for x in nsq0]
    for j in range(3):
        C[3+j,3+j]=180/pi*kappa/iv.sqrt(1-kappa**2)
        J[3+j,3+j]=pi/180*iv.sqrt(1-kappa**2)/kappa
        C[6+j,6+j]=180/pi;J[6+j,6+j]=pi/180
        C[9+j,:]=[I(0)]*D;J[9+j,:]=[I(0)]*D
        C[9+j,9+j]=h0[j]/(n0[j]*Q0)
        J[9+j,9+j]=n0[j]*Q0/h0[j]
        for a in range(2):
            C[9+j,12+a]=t0[a]*(1-h0[j]**2)/(n0[j]*Q0**2)
            J[9+j,12+a]=(nsq0[j]-1)*t0[a]*(1+t0[a]**2)*pi/(180*h0[j])
    for a in range(2):
        C[12+a,12+a]=180/pi/(1+t0[a]**2)
        J[12+a,12+a]=(1+t0[a]**2)*pi/180
    Qnative=C@Qeta@J
    A=ident()-Qnative
    amid=np.array([[float(x.mid) for x in row] for row in A])
    Wfloat=np.linalg.inv(amid) # only a frozen proposal; no floating result trusted
    W=binarymat(Wfloat)
    E=ident()-W@A;e=norminf(E);wnorm=norminf(W)
    assert e<1,(e,wnorm)
    P2=(ident()+E+E@E)@W
    tail=hi(I(e)**3*I(wnorm)/(1-I(e)))
    P=P2+np.full((D,D),iv.mpf([-tail,tail]),dtype=object)
    p=hi(I(wnorm)/(1-I(e)))
    p_entry=norminf(P)
    return {
        'source_file':SOURCE.name,'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'native_coordinates':coords,'scaling':'S=(1/1000) I18; Q_scaled=Q_native since S is scalar',
        'C_dnative_deta':mat(C),'J_deta_dnative':mat(J),
        'analytic_inverse_jacobian_identity_interval_norm':norminf(C@J-ident()),
        'Q_native':mat(Qnative),'A0_I_minus_Q_native':mat(A),
        'W_binary64':Wfloat.tolist(),'W_binary64_hex':[[float(x).hex() for x in row] for row in Wfloat],
        'E_I_minus_WA0':mat(E),'defect_norm_upper':e,'W_norm_upper':wnorm,
        'neumann_degree':2,'neumann_tail_entry_bound':tail,
        'P_exact_inverse_enclosure':mat(P),'P_norm_neumann_upper':p,'P_norm_entry_enclosure_upper':p_entry,
        'P_norm_best_upper':min(p,p_entry),
        'statement':'P=(I-Q_native)^-1 is the exact automatically defined center inverse. The enclosure does not certify a native-box self-map or contraction and no iterates were tested.'
    }

if __name__=='__main__':
    start=time.time()
    out={'schema':'native-box-guards-chord-center-v1','precision_dps':iv.dps,'scope':'ONE fixed original-native +/-1/1000 box at unchanged kappa=1/10; no iterates; no parameter search','native_coordinates':coords,'native_center':vec(center),'native_box':vec(box),'original_prior_checks':prior_checks,'eta_box_outward_image':vec(eta_box),'Q_box':interval(Q)}
    out['physical_native_box']=physical(I(1));print('PHYSICAL NATIVE BOX PASS',out['physical_native_box']['minima'],flush=True)
    # The full [0,1] lambda extension is just the analytic domain used in Taylor integral remainders, not a new witness.
    try:
        out['physical_lambda_extension']=physical(iv.mpf([0,1]));print('LAMBDA ANALYTIC EXTENSION PASS',out['physical_lambda_extension']['minima'],flush=True)
    except ArithmeticError as exc:
        out['physical_lambda_extension']={'passed':False,'unproved_guard':str(exc)};print('LAMBDA ANALYTIC EXTENSION NOT CERTIFIED',exc,flush=True)
    out['chord_center_inverse']=inverse()
    print('CENTER INVERSE PASS',{k:out['chord_center_inverse'][k] for k in ['defect_norm_upper','W_norm_upper','neumann_tail_entry_bound','P_norm_best_upper']},flush=True)
    out['elapsed_seconds']=time.time()-start
    OUT.write_text(json.dumps(out,indent=2)+'\n')
    print('SAVED',OUT,flush=True)
