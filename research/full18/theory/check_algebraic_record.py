"""Small exact symbolic checks for algebraic_record.md; no optical cases/fits.

Run with Python -B. Imports no original research code and reads no observations
or hidden parameter vectors. These identities support the derivation; this is
not a formal verification of the full equivalence theorem or a global solver.
"""
from __future__ import annotations
import hashlib
import json
from fractions import Fraction
from pathlib import Path
import sympy as sp


def main():
    X, H, C, f, Q, R, p, gap = sp.symbols("X H C f Q R p gap", real=True)
    c, s, F, G = sp.symbols("c s F G", real=True)
    a, v = C*Q-f*R, f*Q+C*R
    D, q = H*C-X*f, C*X+f*H
    fn, gn = c*F-s*G, s*F+c*G
    checks = {}

    def zero(name, expression):
        reduced = sp.expand(expression)
        assert reduced == 0, (name, reduced)
        checks[name] = True

    zero("rotor_norm_factorization", fn**2+gn**2-(c*c+s*s)*(F*F+G*G))
    zero("outgoing_unit_norm_factorization", a*a+v*v-(C*C+f*f)*(Q*Q+R*R))
    zero("signed_face_rotation", C*v-f*a-(C*C+f*f)*R)
    zero("entry_rotation_norm", D*D+q*q-(C*C+f*f)*(H*H+X*X))
    zero("native_grazing_square_identity", (H*C)**2-(X*f)**2-
         ((H*H+X*X)*C*C-X*X*(C*C+f*f)))

    # Start with the line-plane exit point and direct propagation, multiply
    # only by the explicitly guarded positive denominators, then reduce by
    # the face-circle equation C^2+f^2=1. No numerical optics are evaluated.
    exit_position = C*(H*p+3*X)/D
    direct = exit_position+(gap-f*exit_position/C)*a/v
    collapsed = R*(H*p+3*X)/(v*D)+gap*a/v
    numerator = sp.cancel((direct-collapsed)*v*D)
    zero("position_transfer_modulo_face_circle",
         numerator-R*(H*p+3*X)*(C*C+f*f-1))

    L1, L2, L3, K1, K2, K3, T0, T1, T2, T3, d = sp.symbols(
        "L1 L2 L3 K1 K2 K3 T0 T1 T2 T3 d", real=True)
    p1 = L1*(p+6*T0)+K1+gap*T1
    p2 = L2*p1+K2+gap*T2
    p3 = L3*p2+K3+d*T3
    affine = (L3*L2*L1*p+6*T0*L3*L2*L1+K3+L3*K2+L3*L2*K1+
              gap*(L3*L2*T1+L3*T2)+d*T3)
    zero("full_three_prism_affine_expansion", p3-affine)

    # Exact isolating intervals for the algebraic native arc constants.
    x = sp.Symbol("x")
    for name, degree, lo, hi in [
        ("clock_arc_root_isolation", 10, sp.Rational(45,100), sp.Rational(46,100)),
        ("beam_arc_root_isolation", 18, sp.Rational(42,100), sp.Rational(43,100)),
    ]:
        assert sp.Poly(sp.chebyshevt(degree,x),x).count_roots(lo,hi) == 1
        checks[name] = True

    count = dict(samples=200, variables=54*200+28, equalities=54*200+10,
                 inequalities=34*200+32, max_total_polynomial_degree=3)
    assert count["variables"] == 10828 and count["equalities"] == 10810
    assert count["inequalities"] == 6832
    checks["explicit_counts"] = True
    assert Fraction(951,1000)*Fraction(83,100)-Fraction(3091,10000)>Fraction(48,100)
    checks["rational_native_D_lower_bound"] = True

    # This is a scalar clock arithmetic check, not a new optical test.
    t1 = Fraction.from_float(0.05)
    t3 = Fraction.from_float(3*0.05)
    discrepancy = t3-3*t1
    assert discrepancy == Fraction(1,2**56)
    checks["binary_clock_is_not_nominal_fixed_step"] = True

    script = Path(__file__).resolve()
    report = dict(
        purpose="Exact symbolic derivation spot checks; no optical cases or fits",
        all_passed=all(checks.values()), checks=checks, counts=count,
        binary_clock_example=dict(t1=str(t1),t3=str(t3),t3_minus_3_t1=str(discrepancy)),
        script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
        limits="Not a formal proof of the full equivalence theorem; no global solver executed",
    )
    destination = script.with_name("algebraic_record_checks.json")
    destination.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
