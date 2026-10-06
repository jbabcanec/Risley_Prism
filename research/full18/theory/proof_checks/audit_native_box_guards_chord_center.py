#!/usr/bin/env python3
"""Independent audit of the ONE unchanged native +/-0.001 certificate.

Only reads existing certificates. Creates a NEW audit JSON. No iterates,
new physical cases, radius search, wedge changes, or damping tests.
"""
from pathlib import Path
from fractions import Fraction as F
import hashlib, json, math
import numpy as np
import mpmath as mp

ROOT = Path(__file__).resolve().parent
iv = mp.iv
iv.dps = 100

def lower(x): return math.nextafter(float(x.a), -math.inf)
def upper(x): return math.nextafter(float(x.b), math.inf)
def norm(A): return max(upper(sum(abs(x) for x in row)) for row in A)
def dyadic(t):
    sign, mantissa, exponent, bits = t
    assert mantissa >= 0 and bits == mantissa.bit_length()
    return (-1 if sign else 1) * F(mantissa) * F(2)**exponent
def exact_iv(q): return iv.mpf(q.numerator) / q.denominator
def load_iv(entry):
    ends = [exact_iv(dyadic(t)) for t in entry['dyadic']]
    return iv.mpf([ends[0].a, ends[1].b])
def matrix(entries): return np.array([[load_iv(e) for e in row] for row in entries], dtype=object)
def identity(n): return np.array([[iv.mpf(int(i == j)) for j in range(n)] for i in range(n)], dtype=object)
def contains(outer, inner): return bool(outer.a <= inner.a and outer.b >= inner.b)
def exact_norm(entries):
    return max(sum(max(abs(dyadic(e['dyadic'][0])), abs(dyadic(e['dyadic'][1]))) for e in row) for row in entries)

certificate_path = ROOT/'native_box_guards_chord_center.json'
src = json.loads(certificate_path.read_text())
out = {'scope': 'Independent audit of ONE unchanged kappa=1/10, original-native +/-1/1000 box; no corrector iterates or additional cases',
       'certificate_sha256': hashlib.sha256(certificate_path.read_bytes()).hexdigest(),
       'precision_dps': iv.dps}

count = 0
def inspect_endpoints(obj):
    global count
    if isinstance(obj, dict):
        if {'lower', 'upper', 'dyadic'} <= obj.keys():
            a, b = map(dyadic, obj['dyadic'])
            assert F(obj['lower']) <= a <= b <= F(obj['upper'])
            count += 1
        for value in obj.values(): inspect_endpoints(value)
    elif isinstance(obj, list):
        for value in obj: inspect_endpoints(value)
inspect_endpoints(src)
out['exact_dyadic_endpoint_records_checked'] = count

expected_coords = ['N1_Hz','N2_Hz','N3_Hz','a1_deg','a2_deg','a3_deg','phi1_deg','phi2_deg','phi3_deg','n1','n2','n3','beta_x_deg','beta_y_deg','b_x','b_y','g','d']
assert src['native_coordinates'] == expected_coords
# Exact native conventions, independently instantiated at higher precision.
one = iv.mpf(1); kappa = one/10; rad = one/1000; deg = iv.pi/180
a0 = iv.atan2(one, iv.sqrt(99))/deg
beta0 = iv.atan2(one, iv.mpf(3))/deg
center = [one/20,iv.mpf(7)/20,iv.mpf(49)/20]+[a0]*3+[iv.mpf(0)]*3+[iv.sqrt(one*17/8),iv.sqrt(one*233/125),iv.sqrt(one*13/5)]+[beta0,iv.mpf(0),one,one*2,one*3,one*100]
box = [c + iv.mpf([-1,1])*rad for c in center]
for e, x in zip(src['native_center'], center): assert contains(load_iv(e), x)
for e, x in zip(src['native_box'], box): assert contains(load_iv(e), x)
priors = [('-3.5','3.5')]*3+[('-18','18')]*6+[('1.3','1.8')]*3+[('-25','25')]*2+[('-5','5')]*2+[('2','15'),('50','200')]
for name, x, pr, saved in zip(expected_coords, box, priors, src['original_prior_checks']):
    assert saved['coordinate'] == name
    assert list(map(F, saved['prior'])) == list(map(F, pr))
    assert lower(x-iv.mpf(pr[0])) > 0 and lower(iv.mpf(pr[1])-x) > 0
t = [iv.tan(deg*x) for x in box[12:14]]
rho = sum(x*x for x in t) # beta_y interval crosses zero; no positivity assumption used here.
rho = sum(x**2 for x in t)
Q = 1+rho
nsq = [x**2 for x in box[9:12]]
h = [iv.sqrt(n2+(n2-1)*rho) for n2 in nsq]
eta = box[:3]+[iv.sin(deg*x)/kappa for x in box[3:6]]+[deg*x for x in box[6:9]]+h+t+box[14:]
for e, x in zip(src['eta_box_outward_image'], eta): assert contains(load_iv(e), x)
out['native_box_center_priors_and_eta_conversion'] = 'PASS: all 18 coordinates, exact native angle/index conventions'

# Inspect every saved guard and its claimed reported minimum, using exact rational
# comparisons to dyadics rather than trusting the printed decimal summaries.
for key in ['physical_native_box', 'physical_lambda_extension']:
    p = src[key]
    assert p['passed'] and p['sample_count'] == 200
    assert len(p['prism_guard_records']) == 600
    assert {(r['sample'],r['prism']) for r in p['prism_guard_records']} == {(k,j) for k in range(200) for j in range(1,4)}
    assert all(len(r['guards']) == 18 for r in p['prism_guard_records'])
    for guard, minimum in p['minima'].items():
        exact_min = min(dyadic(r['guards'][guard]['dyadic'][0]) for r in p['prism_guard_records'])
        assert F(minimum['lower']) <= exact_min and exact_min > 0
    for r in p['prism_guard_records']:
        for e in r['guards'].values(): assert dyadic(e['dyadic'][0]) > 0
out['saved_guards_exact_endpoint_audit'] = 'PASS: 2 domains x 200 timestamps x 3 prisms x 18 guards'

cc = src['chord_center_inverse']
q_source = ROOT/cc['source_file']
assert hashlib.sha256(q_source.read_bytes()).hexdigest() == cc['source_sha256']
qj = json.loads(q_source.read_text())
assert qj['interval'] and qj['kappa'] == 0.1
Qeta = np.array([[iv.mpf([a,b]) for a,b in zip(ra,rb)] for ra,rb in zip(qj['DT_interval_lower'],qj['DT_interval_upper'])],dtype=object)
C = identity(18); J = identity(18)
Q0 = one*10/9; tt=[one/3,iv.mpf(0)]; hh=[one*3/2,one*7/5,one*5/3]
nn=[iv.sqrt(one*17/8),iv.sqrt(one*233/125),iv.sqrt(one*13/5)]
for j in range(3):
    C[3+j,3+j]=kappa/(deg*iv.sqrt(1-kappa**2))
    J[3+j,3+j]=deg*iv.sqrt(1-kappa**2)/kappa
    C[6+j,6+j]=1/deg; J[6+j,6+j]=deg
    C[9+j,:]=[iv.mpf(0)]*18; J[9+j,:]=[iv.mpf(0)]*18
    C[9+j,9+j]=hh[j]/(nn[j]*Q0); J[9+j,9+j]=nn[j]*Q0/hh[j]
    for a in range(2):
        C[9+j,12+a]=tt[a]*(1-hh[j]**2)/(nn[j]*Q0**2)
        J[9+j,12+a]=(nn[j]**2-1)*tt[a]*(1+tt[a]**2)*deg/hh[j]
for a in range(2):
    C[12+a,12+a]=1/(deg*(1+tt[a]**2)); J[12+a,12+a]=deg*(1+tt[a]**2)
assert norm(C@J-identity(18)) < 1e-98
Qnative=C@Qeta@J
for actual, recorded in zip([C,J,Qnative], [matrix(cc['C_dnative_deta']),matrix(cc['J_deta_dnative']),matrix(cc['Q_native'])]):
    for a,b in zip(recorded.flat,actual.flat): assert contains(a,b)
A=identity(18)-Qnative
W=np.array([[iv.mpf(float.fromhex(x)) for x in row] for row in cc['W_binary64_hex']], dtype=object)
assert [[float.fromhex(x) for x in row] for row in cc['W_binary64_hex']] == cc['W_binary64']
E=identity(18)-W@A
assert norm(E) <= cc['defect_norm_upper'] < 1
# Exact rational audit of all reported norm/tail scalars.
e=F(cc['defect_norm_upper']); wn=F(cc['W_norm_upper']); tail=F(cc['neumann_tail_entry_bound'])
assert exact_norm(cc['E_I_minus_WA0']) <= e
assert max(sum(abs(F(x)) for x in row) for row in cc['W_binary64']) <= wn
assert e**3*wn/(1-e) <= tail
assert wn/(1-e) <= F(cc['P_norm_neumann_upper'])
assert exact_norm(cc['P_exact_inverse_enclosure']) <= F(cc['P_norm_entry_enclosure_upper'])
assert F(cc['P_norm_best_upper']) == min(F(cc['P_norm_neumann_upper']),F(cc['P_norm_entry_enclosure_upper']))
# Rebuild the Neumann polynomial from the independently transformed source matrix,
# rather than substituting a numerical inverse for the mathematical inverse.
P2=(identity(18)+E+E@E)@W
tail_iv=iv.mpf([-float(tail),float(tail)])
Psave=matrix(cc['P_exact_inverse_enclosure'])
for p,s in zip(P2.flat,Psave.flat): assert contains(s,p+tail_iv)
out['independent_center_matrix_audit'] = {'passed':True,'residual_orientation':'E=I-W A, A^-1=(I-E)^-1 W','defect_norm_upper':cc['defect_norm_upper'],'neumann_tail_entry_bound':cc['neumann_tail_entry_bound'],'P_norm_best_upper':cc['P_norm_best_upper'],'native_similarity_identity_norm':norm(C@J-identity(18))}

def unit_direction_audit(lam):
    """Independent unit-normal/unit-ray optical circuit on the SAME domain."""
    minima={}
    def guard(key,x):
        assert lower(x)>0, (key,str(x))
        minima[key]=min(minima.get(key,math.inf),lower(x))
    for k in range(200):
        q=[x/iv.sqrt(Q) for x in t]  # actual unit-air transverse directions
        z=1/iv.sqrt(Q)
        pos=[box[14+a]+6*t[a] for a in range(2)]
        for j in range(3):
            angle=2*iv.pi*box[j]*k/20+deg*box[6+j]
            es=lam*iv.sin(deg*box[3+j]); ca=iv.sqrt(1-es**2)
            normal=[-es*iv.cos(angle),-es*iv.sin(angle),ca]
            slope=[-normal[a]/ca for a in range(2)]
            H=iv.sqrt(nsq[j]-sum(x**2 for x in q))
            Cg=q[0]*normal[0]+q[1]*normal[1]+H*normal[2]
            R=1-nsq[j]+Cg**2
            guard('incoming_air_axial_unit',z)
            guard('glass_radicand_unit',H**2)
            guard('incoming_glass_unit_normal_projection',Cg/box[9+j])
            guard('exit_snell_unit_normal_radicand',R)
            Eout=iv.sqrt(R)
            correction=(nsq[j]-1)/(Cg+Eout)
            qn=[q[a]-correction*normal[a] for a in range(2)]
            zn=H-correction*normal[2]
            den=1-sum(slope[a]*q[a]/H for a in range(2))
            internal=(3+sum(slope[a]*pos[a] for a in range(2)))/den
            exitpos=[pos[a]+q[a]*internal/H for a in range(2)]
            external=(box[16] if j<2 else box[17])-sum(slope[a]*exitpos[a] for a in range(2))
            guard('outgoing_axial_unit',zn)
            guard('outgoing_unit_normal_projection',Eout)
            guard('intersection_denominator',den)
            guard('internal_axial_travel',internal)
            guard('external_axial_travel',external)
            pos=[exitpos[a]+qn[a]*external/zn for a in range(2)]
            q,z=qn,zn
    return {'passed':True,'minima':minima,'sample_prism_pairs':600}
out['independent_unit_direction_native_box']=unit_direction_audit(iv.mpf(1))
out['independent_unit_direction_lambda_extension']=unit_direction_audit(iv.mpf([0,1]))
out['verdict']='PASS: saved native-box guards and exact center inverse; NO off-image branch-domain, curvature, self-map, contraction, noise or global-recovery claim.'
(ROOT/'audit_native_box_guards_chord_center.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
