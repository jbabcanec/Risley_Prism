"""Independent direct-vector Snell spot-check, not an inverse or record generator.

The only intended input is the explicitly labeled synthetic fixture's generating
rational chart point. Directions use unit surface normals and vector Snell;
positions use three-dimensional ray/plane intersections, not affine coefficients.
The shared interval kernel is independently reviewed/tested; this module supplies
an independent optical formula and integer-clock rotor implementation.
"""
from fractions import Fraction as F
from interval import I, precision


def exact_rotation(value):
    v = F(value)
    return (1-v*v)/(1+v*v), 2*v/(1+v*v)


def complex_product(left, right):
    a, b = left
    c, d = right
    return a*c-b*d, a*d+b*c


def exact_power(value, k):
    result = (F(1), F(0))
    for _ in range(k):
        result = complex_product(result, value)
    return result


def dot(left, right):
    return sum((a*b for a,b in zip(left,right)), I(0))


def trace_point(chart, *, bits=224):
    """Enclose all 200 outputs for one exact rational point at k/20.

    No input is interpreted as a floating-point number. Returns independently
    computed screen intervals and minimum strict physical lower bounds.
    """
    required = [f"{prefix}{j}" for prefix in ("r","p","v","n") for j in (1,2,3)]
    required += ["tx","ty","bx","by","g","d"]
    if set(chart) != set(required):
        raise ValueError("Expected exactly the eighteen rational chart coordinates")
    if any(isinstance(chart[n], (bool,float)) for n in required):
        raise TypeError("Point coordinates must be exact rationals")
    values = {name:F(chart[name]) for name in required}
    with precision(bits):
        t = [I(values["tx"]),I(values["ty"]),I(1)]
        incoming = [a/dot(t,t).sqrt() for a in t]
        minima = {}
        def guard(name, value):
            if value.lo <= 0:
                raise AssertionError(f"Independent reference cannot prove {name}>0")
            minima[name] = min(minima.get(name,value.lo), value.lo)
        screens=[]
        for k in range(200):
            direction = incoming[:]
            position = [I(values["bx"])+6*I(values["tx"]),
                        I(values["by"])+6*I(values["ty"]), I(6)]
            entry_z = I(6)
            for j in (1,2,3):
                rotor = complex_product(exact_rotation(values[f"p{j}"]),
                    exact_power(exact_rotation(values[f"v{j}"]),k))
                u = [I(values[f"r{j}"]*a) for a in rotor]
                index=I(values[f"n{j}"])
                gx,gy=direction[0]/index,direction[1]/index
                glass_rad=1-gx.square()-gy.square()
                guard("glass_radicand",glass_rad)
                glass=[gx,gy,glass_rad.sqrt()]
                normal_length=(1+u[0].square()+u[1].square()).sqrt()
                normal=[-u[0]/normal_length,-u[1]/normal_length,1/normal_length]
                cosine=dot(glass,normal)
                guard("incoming_normal_cosine",cosine)
                transmitted_rad=1-index.square()*(1-cosine.square())
                guard("exit_normal_radicand",transmitted_rad)
                normal_component=transmitted_rad.sqrt()
                outgoing=[index*glass[a]+(normal_component-index*cosine)*normal[a]
                          for a in range(3)]
                guard("outgoing_axial",outgoing[2])
                guard("outgoing_normal",dot(outgoing,normal))
                exit_z_base=entry_z+3
                internal_num=exit_z_base+u[0]*position[0]+u[1]*position[1]-position[2]
                internal_den=glass[2]-u[0]*glass[0]-u[1]*glass[1]
                guard("internal_numerator",internal_num)
                guard("internal_denominator",internal_den)
                internal_time=internal_num/internal_den
                exit_position=[position[a]+internal_time*glass[a] for a in range(3)]
                ell=I(values["g"] if j<3 else values["d"])
                next_z=exit_z_base+ell
                external_height=next_z-exit_position[2]
                guard("external_height",external_height)
                external_time=external_height/outgoing[2]
                position=[exit_position[a]+external_time*outgoing[a] for a in range(3)]
                direction=outgoing
                entry_z=next_z
            screens.append(position[:2])
        return {"screen":screens,"minimum_strict_lower_bounds":minima,
                "precision_bits":bits,"samples":200,"clock":"k/20"}


def independent_kernel_checks():
    """Finite exact-arithmetic review checks, not a randomized campaign."""
    from interval import IntervalDomainError, pi_interval, tan_pi_fraction
    count=0
    intervals=[(F(-7,3),F(-2,5)),(F(-3,7),F(5,9)),(F(0),F(8,11)),(F(2,7),F(17,13))]
    with precision(24):
        for left in intervals:
            a=I(*left)
            assert a.lo<=left[0]<=left[1]<=a.hi
            assert (a.square()).lo<=0 if left[0]<=0<=left[1] else True
            for right in intervals:
                b=I(*right)
                for x in (left[0], (2*left[0]+left[1])/3,left[1]):
                    for y in (right[0],(right[0]+2*right[1])/3,right[1]):
                        assert (a+b).contains(x+y)
                        assert (a-b).contains(x-y)
                        assert (a*b).contains(x*y)
                        count+=3
                        if b.hi<0 or b.lo>0:
                            assert (a/b).contains(x/y)
                            count+=1
        for q in (F(0),F(1),F(2),F(1,3),F(1001,71)):
            root=I(q).sqrt()
            assert root.lo>=0 and root.lo*root.lo<=q<=root.hi*root.hi
            count+=1
        for f in (lambda: I(-1,1).sqrt(),lambda: I(1)/I(-1,1)):
            try: f()
            except IntervalDomainError: count+=1
            else: raise AssertionError("Uncertified operation was accepted")
        for bad in (1.0,True):
            try: I(bad)
            except TypeError: count+=1
            else: raise AssertionError("Nonrational endpoint was accepted")
    with precision(224):
        five=I(5).sqrt()
        radical=(five-1)/(10+2*five).sqrt()
    tan18=tan_pi_fraction(1,10,bits=80)
    assert tan18.contains(radical)
    # A strict classical rational bracket supplies an additional pi smoke check.
    pi=pi_interval(bits=80)
    assert F(333,106)<pi.lo<pi.hi<F(355,113)
    count+=2
    return {"status":"passed","exact_scalar_containment_checks":count,
            "random_cases":0,"float_arithmetic_used":False}

