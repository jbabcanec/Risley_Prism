#!/usr/bin/env python3
"""ONE preselected original oblique proof point, kappa=1/10. No sweep."""
import mpmath as mp
import math
iv=mp.iv; iv.dps=60

def lo(x): return math.nextafter(float(x.a),-math.inf)
def hi(x): return math.nextafter(float(x.b),math.inf)
def audit():
    Q=iv.mpf(10)/9; nsq=[iv.mpf(17)/8,iv.mpf(233)/125,iv.mpf(13)/5]
    e=iv.mpf(1)/10; wedge=e/iv.sqrt(1-e*e)
    mins={k:None for k in ('glass_radicand','snell_radicand','P','Z','normal','internal_travel','external_travel')}
    loc={}
    for k in range(200):
        X=[iv.mpf(1)/3,iv.mpf(0)];p=[iv.mpf(3),iv.mpf(2)]
        for j,nu in enumerate([1,7,49]):
            angle=2*iv.pi*nu*k/400;u=[wedge*iv.cos(angle),wedge*iv.sin(angle)]
            rad=nsq[j]*Q-sum(x*x for x in X);H=iv.sqrt(rad)
            P=H-sum(a*b for a,b in zip(u,X));D=1+wedge*wedge
            rad2=P*P-D*(nsq[j]-1)*Q;E=iv.sqrt(rad2)
            Z=(H*D-P+E)/D
            Xn=[X[a]+u[a]*(P-E)/D for a in range(2)]
            internal=3+sum(u[a]*p[a] for a in range(2))
            pe=[p[a]+X[a]*internal/P for a in range(2)]
            external=(3 if j<2 else 100)-sum(u[a]*pe[a] for a in range(2))
            vals=[rad,rad2,P,Z,E,internal,external]
            for name,val in zip(mins,vals):
                assert lo(val)>0,(k,j,name,val)
                if mins[name] is None or lo(val)<mins[name]:mins[name]=lo(val);loc[name]=(k,j+1)
            p=[pe[a]+Xn[a]*external/Z for a in range(2)];X=Xn
    print('ONE fixed proof point: all wedges asin(1/10) radians; N=(1,7,49)/20; phases=0; t=(1/3,0); b=(1,2); h=(3/2,7/5,5/3); g=3; d=100.')
    print('Directional guards use Q-scaled vectors, Q=10/9; physical Z is scaled Z/sqrt(Q), physical Snell radicand is scaled radicand/Q.')
    print('All 200 timestamps, all three prisms: strict physical and traversal guards PASS.')
    for name in mins:print(name,': lower >',mins[name],'at sample,prism',loc[name])
    return mins
if __name__=='__main__':audit()
