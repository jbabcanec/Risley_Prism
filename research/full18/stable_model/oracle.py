"""Independent scalar interval reference on EXACT supplied binary64 inputs.

Uses mpmath.iv directed intervals, mathematical pi, and six scalar interfaces.
It imports no repository code. This is a numerical reference, not formal proof
of the Python/mpmath implementation. Time is not replaced with rational k/20.
"""
import math
from mpmath import iv


def exact(value):
    n,d=float(value).as_integer_ratio()
    return iv.mpf(n)/iv.mpf(d)


def bounds(x):
    return math.nextafter(float(x.a),-math.inf),math.nextafter(float(x.b),math.inf)


def abs_upper(x):
    lo,hi=bounds(x)
    return max(abs(lo),abs(hi))


def reference(theta,timestamps,dps=70):
    iv.dps=dps
    v=[exact(x) for x in theta]
    one,zero=iv.mpf(1),iv.mpf(0)
    deg=iv.pi/180
    zs=[iv.mpf(6),iv.mpf(9),9+v[13],12+v[13],12+2*v[13],15+2*v[13],15+2*v[13]+v[12]]
    results=[]
    margins=dict(tir=math.inf,fwd=math.inf,graze=math.inf)
    for time in timestamps:
        t=exact(time)
        angle=[2*iv.pi*v[i]*t+deg*v[6+i] for i in range(3)]
        tangent_wedge=[iv.tan(deg*v[3+i]) for i in range(3)]
        u=[iv.cos(angle[i])*tangent_wedge[i] for i in range(3)]
        w=[iv.sin(angle[i])*tangent_wedge[i] for i in range(3)]
        pair=[]
        for axis,(primary,secondary) in enumerate(((u,w),(w,u))):
            q=[iv.sqrt(one+x*x) for x in secondary]
            normal=[iv.sqrt(one+primary[i]**2+secondary[i]**2) for i in range(3)]
            s=[zero,primary[0]/normal[0],zero,primary[1]/normal[1],zero,primary[2]/normal[2]]
            c=[one,q[0]/normal[0],one,q[1]/normal[1],one,q[2]/normal[2]]
            slope=[zero,primary[0]/q[0],zero,primary[1]/q[1],zero,primary[2]/q[2],zero]
            ratio=[one/v[9],v[9],one/v[10],v[10],one/v[11],v[11]]
            tangent=iv.tan(v[14+axis]*deg)
            p=v[16+axis]+6*tangent
            z=iv.mpf(6)
            for j in range(6):
                norm=iv.sqrt(one+tangent*tangent)
                cy=-c[j]*tangent/norm-s[j]/norm
                rad=one-(ratio[j]*cy)**2
                rlo,_=bounds(rad)
                if rlo<=0:
                    raise ValueError("Nonpositive or undecided TIR margin")
                margins["tir"]=min(margins["tir"],rlo)
                root=iv.sqrt(rad)
                out0=-ratio[j]*c[j]*cy-s[j]*root
                outz=-ratio[j]*s[j]*cy+c[j]*root
                zlo,_=bounds(outz)
                if zlo<=0:
                    raise ValueError("Nonpositive forward margin")
                margins["fwd"]=min(margins["fwd"],zlo)
                tangent=out0/outz
                denominator=one-slope[j+1]*tangent
                dl,du=bounds(denominator)
                if dl<=0<=du:
                    raise ValueError("Zero or undecided grazing margin")
                margins["graze"]=min(margins["graze"],abs(dl),abs(du))
                step=(zs[j+1]+slope[j+1]*p-z)/denominator
                p+=step*tangent
                z+=step
            pair.append(p)
        results.append(pair)
    return results,margins


def error_upper(pattern,reference_pattern):
    return max(abs_upper(exact(float(x))-r)
               for pair,ref_pair in zip(pattern,reference_pattern)
               for x,r in zip(pair,ref_pair))
