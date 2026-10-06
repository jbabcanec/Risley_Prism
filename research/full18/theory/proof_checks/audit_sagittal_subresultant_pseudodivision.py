"""Independent raw-row/two-segment/pseudo-division check of the two-chart claim.
Uses the audited outward integer enclosure and formal norm multiplication from
sagittal_two_chart_coprimality.py, but a separate position/derivative construction
and division-free polynomial pseudo-remainders. No numerical coefficient trims.
"""
import runpy, sys, json, time
from fractions import Fraction as F
sys.argv=['audit',sys.argv[1] if len(sys.argv)>1 else '2048']
base=runpy.run_path('/workspace/shared/risley_theory/proof_checks/sagittal_two_chart_coprimality.py',run_name='audit_import')
I=base['I']; dot=base['dot']; cross=base['cross']; add=base['add']; sub=base['sub']; scale=base['scale']
normal=base['anormalize']; S=base['S']; BITS=base['BITS']

def direction(q,u):
    H=(I(F(9,4))-dot(q,q)).sqrt(); P=H-dot(q,u)
    R=(P*P-(1+dot(u,u))*I(F(5,4))).sqrt()
    lam=I(F(5,4))/(P+R)
    B=add(q,scale(lam,u)); Z=H-lam
    return q,u,H,P,B,Z,R

def transport(p,opt,gap):
    q,u,H,P,B,Z,R=opt
    v=(3+dot(u,p))/P
    return add(add(p,scale(v,q)),scale((3+gap-H*v)/Z,B))

def independent_rows():
    rots=[(F(15,17),F(8,17)),(F(4,5),F(3,5)),(F(3,5),F(4,5))]
    states=[(F(1),F(0))]*3
    t=[I(F(1,3)),I(F(1,10))]; q0=scale(1/(1+dot(t,t)).sqrt(),t)
    b=[I(1),I(2)]; g=I(10); d=I(100)
    rows=[]; alphas=[]
    for k in range(6):
        q=q0[:]; opts=[]
        for c,s in states:
            opt=direction(q,[I(c/10),I(s/10)]); opts.append(opt); q=opt[4]
        def prefix(offset,gap):
            p=add(offset,scale(6,t))
            for opt in opts[:2]: p=transport(p,opt,gap)
            return p
        p=prefix(b,g); y=transport(p,opts[2],d)
        derivatives=[sub(prefix([b[0]+1,b[1]],g),p),
                     sub(prefix([b[0],b[1]+1],g),p),sub(prefix(b,g+1),p)]
        e=sub(sub(sub(p,scale(b[0],derivatives[0])),scale(b[1],derivatives[1])),scale(g,derivatives[2]))
        q,u,H,P,B,Z,R=opts[2]; delta=cross(q,u)
        # Original inverse-data augmented row: no true-geometry column operation
        # and no replacement of a last-column coefficient by H_true times r.
        const=[-cross(v,q) for v in derivatives]+[-delta,cross(sub(y,e),q)-3*delta]
        linear=[-cross(v,u) for v in derivatives]+[I(0),cross(sub(y,e),u)]
        rows.append((const,linear)); alphas.append(dot(q,q))
        states=[(c*a-s*bb,s*a+c*bb) for (c,s),(a,bb) in zip(states,rots)]
    return rows,alphas

def primitive_pseudoremainder(a,b):
    assert len(a)>=len(b) and not b[-1].contains0()
    r=a[:]
    for k in range(len(a)-len(b),-1,-1):
        top=r[-1]; lead=b[-1]
        nxt=[]
        for j in range(len(r)-1):
            value=lead*r[j]
            if j>=k: value=value-top*b[j-k]
            nxt.append(value)
        # The discarded top coefficient is symbolically lead*top-top*lead=0.
        assert nxt
        normed,_=normal({0:nxt});r=normed[0]
    return r

start=time.time(); rows,alphas=independent_rows()
a,ma=base['norm_chart'](rows,alphas,[0,1,2,3,4])
b,mb=base['norm_chart'](rows,alphas,[0,1,2,3,5])
record=[]
while len(b)>1:
    r=primitive_pseudoremainder(a,b)
    assert len(r)==len(b)-1
    assert not r[-1].contains0(),('pseudoremainder degree unresolved',len(r)-1)
    record.append({'degree':len(r)-1,'leading':[str(r[-1].lo),str(r[-1].hi)]})
    print('Certified pseudoremainder degree',len(r)-1,'leading',r[-1],flush=True)
    a,b=b,r
assert len(record)==63 and not b[0].contains0()
result={'precision_bits':BITS,'norm_metadata':[ma,mb], 'pseudoremainders':record,
        'terminal_constant':[str(b[0].lo),str(b[0].hi)],'elapsed_seconds':time.time()-start,
        'conclusion':'Original augmented rows and division-free pseudo-remainders independently certify degree64 norms and coprime exact deflated quotients.'}
path='/workspace/shared/risley_theory/proof_checks/sagittal_subresultant_independent_pseudodivision_certificate.json'
with open(path,'w') as f: json.dump(result,f,indent=2)
print('PASS',result['conclusion'],'terminal',b[0],'elapsed',result['elapsed_seconds'],'certificate',path,flush=True)
