"""One exact rational spot check of the derived cubic optical recurrence.

No observations, hardware recovery, random cases, or original project imports.
"""
from pathlib import Path
import json
import sympy as s

def main():
    n1,n2,n3,g,d=s.symbols('n1 n2 n3 g d',positive=True)
    f=s.symbols('f1 f2 f3');hardware=(n1,n2,n3,g,d)
    X1=X3=P1=P3=s.Integer(0)
    for n,u,ell in zip((n1,n2,n3),f,(g,g,d)):
        kap=n-1
        Y1=X1+kap*u
        Y3=X3+kap*u*X1**2/(2*n)+kap*u**2*X1+n*kap*u**3/2
        L2=-kap*u*X1/n-kap*u**2
        nextP1=P1+3*X1/n+ell*Y1
        nextP3=P3+L2*P1+3*X3/n+3*X1**3/(2*n**3)+3*X1*L2/n+ell*(Y3+Y1**3/2)
        X1,X3,P1,P3=Y1,Y3,s.expand(nextP1),s.expand(nextP3)
    first=s.Poly(P1,*f);third=s.Poly(P3,*f)
    gains=[first.coeff_monomial(u) for u in f]
    expected=[(n1-1)*(d+2*g+3/n2+3/n3),(n2-1)*(d+g+3/n3),(n3-1)*d]
    assert all(s.simplify(a-b)==0 for a,b in zip(gains,expected))
    indices=[(3,0,0),(0,3,0),(0,0,3),(2,1,0),(0,2,1)]
    normalized=[]
    for index in indices:
        monomial=s.prod(u**p for u,p in zip(f,index))
        normalized.append(third.coeff_monomial(monomial)/s.prod(k**p for k,p in zip(gains,index)))
    point={n1:s.Rational(3,2),n2:s.Rational(4,3),n3:s.Rational(5,3),g:s.Integer(3),d:s.Integer(100)}
    jac=s.Matrix([[s.diff(value,h).subs(point) for h in hardware] for value in normalized])
    determinant=s.factor(jac.det())
    reported=-s.Rational(2342246801318629443,280149096378993020072641029913075712000)
    assert determinant==reported
    B0=6+2*g+3/n1+3/n2+3/n3+d
    qp=-(n3-1)
    qb=-(n3-1)*(B0-d)+d*((n3-1)+s.Rational(3,2)*(n3-1)**2)
    beam=s.factor(B0*qp-qb)
    assert s.simplify(beam+d*(n3-1)*(3*n3+1)/2)==0
    result={'scope':'One exact rational derivation spot check; no inverse solve or case campaign',
      'indices':[list(x) for x in indices],'hardware_order':[str(x) for x in hardware],
      'point':{str(k):str(v) for k,v in point.items()},'first_order_gains':[str(k.subs(point)) for k in gains],
      'normalized_cubic_values':[str(x.subs(point)) for x in normalized],
      'jacobian':[[str(x) for x in row] for row in jac.tolist()],
      'determinant':str(determinant),'matches_independent_report':bool(determinant==reported),
      'beam_determinant_identity':str(beam),'beam_determinant_at_point':str(beam.subs(point)),
      'limitations':'Nonzero local-rank witness, not global uniqueness or finite-noise accuracy; exact ideal clock temporal proof is separate.'}
    (Path(__file__).parent/'cubic_rank_check.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('jacobian','normalized_cubic_values')},indent=2))

if __name__=='__main__':main()
