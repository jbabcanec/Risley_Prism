#!/usr/bin/env python3
"""Arithmetic replay and alternative unit-normal Snell cross-check.
Same original native box, same 200 times, no additional optical case.
"""
import json,math,hashlib
from pathlib import Path
import mpmath as mp
mp.mp.dps=110;mp.iv.dps=90
iv=mp.iv;I=iv.mpf
P=Path(__file__).resolve().parent
src=P/'native_box_guards_chord_center.json';x=json.loads(src.read_text())

def read(z):return I([mp.mpf(tuple(a)) for a in z['dyadic']])
def upper(z):return math.nextafter(float(z.b),math.inf)
def lower(z):return math.nextafter(float(z.a),-math.inf)
def mat(z):return [[read(v) for v in row] for row in z]
def mul(A,B):return [[sum(A[i][k]*B[k][j] for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]
def add(A,B):return [[a+b for a,b in zip(ra,rb)] for ra,rb in zip(A,B)]
def neg(A):return [[-v for v in row] for row in A]
def norm(A):return max(upper(sum(abs(z) for z in row)) for row in A)
def eye(n):return [[I(int(i==j)) for j in range(n)] for i in range(n)]
def subset(a,b):return a.a>=b.a and a.b<=b.b

z=x['chord_center_inverse'];n=18
Q=mat(z['Q_native']);A=add(eye(n),neg(Q))
W=[[I(float.fromhex(v)) for v in row] for row in z['W_binary64_hex']]
assert all(float.fromhex(a)==b for ra,rb in zip(z['W_binary64_hex'],z['W_binary64']) for a,b in zip(ra,rb))
E=add(eye(n),neg(mul(W,A)));e=norm(E)
assert e<=z['defect_norm_upper']
P2=mul(add(add(eye(n),E),mul(E,E)),W)
tail=I(z['neumann_tail_entry_bound']);Pcert=mat(z['P_exact_inverse_enclosure'])
assert all(subset(P2[i][j]+I([-tail.b,tail.b]),Pcert[i][j]) for i in range(n) for j in range(n))

# Direct Snell formula in unit incident-ray coordinates. This deliberately uses
# (K-sqrt(delta))/sqrt(D), not the rationalized Q-scaled formula of the main run.
box=[read(t) for t in x['native_box']]
t=[iv.tan(a*iv.pi/180) for a in box[12:14]];Qin=1+sum(a**2 for a in t)
sqrtQ=iv.sqrt(Qin);N=box[:3];a=[v*iv.pi/180 for v in box[3:6]];phi=[v*iv.pi/180 for v in box[6:9]];indices=box[9:12];b=box[14:16];g,d=box[16:18]
mins={};count=0
for k in range(200):
    q=[v/sqrtQ for v in t];p=[b[u]+6*t[u] for u in range(2)]
    for j in range(3):
        angle=2*iv.pi*N[j]*I(k)/20+phi[j];tilt=iv.tan(a[j]);D=1+tilt**2
        u=[tilt*iv.cos(angle),tilt*iv.sin(angle)];n=indices[j]
        rad=n**2-sum(v**2 for v in q);assert lower(rad)>0
        H=iv.sqrt(rad);Pnormal=H-sum(u[v]*q[v] for v in range(2));K=Pnormal/iv.sqrt(D)
        delta=1-n**2+K**2;assert lower(delta)>0
        cn=(K-iv.sqrt(delta))/iv.sqrt(D);qn=[q[v]+u[v]*cn for v in range(2)];zn=H-cn
        traverse=3+sum(u[v]*p[v] for v in range(2));pe=[p[v]+q[v]*traverse/Pnormal for v in range(2)]
        ext=(g if j<2 else d)-sum(u[v]*pe[v] for v in range(2))
        values={'glass_radicand':rad,'incoming_optical_normal_numerator':Pnormal,'unit_normal_snell_radicand':delta,'unit_air_axial':zn,'internal_traverse_numerator':traverse,'external_axial_flight':ext}
        for name,v in values.items():
            assert lower(v)>0,(k,j,name,v)
            mins[name]=min(mins.get(name,math.inf),lower(v))
        p=[pe[v]+qn[v]*ext/zn for v in range(2)];q=qn;count+=1
result={'source_file':src.name,'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'precision_dps':90,'endpoint_replay_pass':True,'replayed_defect_upper':e,'binary_W_hex_roundtrip_pass':True,'P2_plus_recorded_tail_contained_in_saved_P':True,'alternative_unit_normal_trace_pass':True,'sample_prism_count':count,'alternative_unit_normal_trace_minima':mins,'scope':'Arithmetic cross-check by the same author, not a separate independent review. Same fixed native box and samples; no new case or iterate.'}
(P/'native_box_guards_chord_replay.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
