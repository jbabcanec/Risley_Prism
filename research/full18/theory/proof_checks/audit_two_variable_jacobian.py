#!/usr/bin/env python3
"""Independent exact audit of the two-beam-variable reversion.

This is one rational proof witness, not a parameter sweep. The observed
covariances are M_i M_i.T: arbitrary nonzero scalar wedge factors cancel
from every shape invariant, so no finite unit-wedge optical trace is used.
Requires SymPy. All checks and derivatives are exact rational arithmetic.
"""

import sympy as sp


def check_symbolic_remainder():
    lever, p, a, E, z, beta, k, c = sp.symbols(
        "lever p a E z beta k c", nonzero=True
    )
    q = p * lever**2 - a * lever + E
    f = (
        z * (k * lever + c)
        - E
        - 3 * ((p * lever)**3 - p * lever)
        - beta * (k * lever + c)**2
    )
    f1 = z*k + 3*p - 3*p*a**2 + 3*p**2*E - beta*(k**2*a/p + 2*k*c)
    f0 = z*c - E + 3*p*a*E + beta*k**2*E/p - beta*c**2
    assert sp.factor(sp.rem(f, q, lever) - f1*lever - f0) == 0
    eliminant = p*f0**2 + a*f0*f1 + E*f1**2
    assert sp.factor(eliminant - f1**2*q.subs(lever, -f0/f1)) == 0
    print("Symbolic cubic remainder and eliminant identities: exact zero defects")


def check_witness_jacobian():
    x, y = sp.symbols("x y", real=True)
    trial_t = sp.Matrix([x, y])
    trial_v = sp.Matrix([-y, x])
    rho = (trial_t.T * trial_t)[0]
    B = sp.Matrix([sp.Rational(1411, 35), 2])
    point = {x: sp.Rational(1, 3), y: 0}
    witness_t = sp.Matrix([sp.Rational(1, 3), 0])
    hs = [sp.Rational(3, 2), sp.Rational(7, 5), sp.Rational(5, 3)]
    distance, gap = sp.Rational(100), sp.Rational(3)
    alphas, betas = [], []

    for i, h_i in enumerate(hs):
        D_i = distance + (2-i)*gap
        L_i = D_i + sum(3/h for h in hs[i+1:])
        W_i = D_i + sum(3/h**3 for h in hs[i+1:])
        M_i = (h_i-1) * (
            L_i*sp.eye(2)
            + (W_i+L_i/h_i)*witness_t*witness_t.T
            - witness_t*B.T/h_i
        )
        assert M_i.det() > 0
        S_i = M_i*M_i.T
        # sqrt(det(S_i)) = det(M_i), with its positive sign verified above.
        D_shape = (trial_v.T*S_i*trial_v)[0]
        A_shape = (trial_t.T*S_i*trial_v)[0]
        Bv, Bt = (trial_v.T*B)[0], (trial_t.T*B)[0]
        betas.append(-A_shape/(D_shape*Bv))
        alphas.append(
            (rho*M_i.det()*Bv-D_shape*Bv-Bt*A_shape)/(rho*D_shape*Bv)
        )

    def value(expr):
        return sp.factor(expr.subs(point))

    def gradient(expr):
        return sp.Matrix([value(sp.diff(expr, x)), value(sp.diff(expr, y))])

    h3 = 1/(alphas[2]-1)
    d = (alphas[2]-1)/betas[2]
    E_expr = 3*(h3**(-3)-h3**(-1))
    L = sp.Rational(524, 5)
    L1 = sp.Rational(3848, 35)
    p, a, E, z, beta = map(
        value, [betas[1], alphas[1]-1, E_expr, alphas[0]-1, betas[0]]
    )
    k = 2+3*p
    c = -value(d)-3/value(h3)
    f1 = sp.factor(z*k+3*p-3*p*a**2+3*p**2*E-beta*(k**2*a/p+2*k*c))
    f0 = sp.factor(z*c-E+3*p*a*E+beta*k**2*E/p-beta*c**2)
    assert f1 == sp.Rational(399563722409659, 317911540674000)
    assert f0 == -sp.Rational(399563722409659, 3033507067500)
    assert -f0/f1 == L
    q_prime = 2*p*L-a
    assert q_prime == sp.Rational(16627, 22925)
    assert E/(p*L) == -sp.Rational(1008, 625)

    p_g, a_g, E_g, z_g, beta_g = map(
        gradient, [betas[1], alphas[1]-1, E_expr, alphas[0]-1, betas[0]]
    )
    # Differentiate the positive root of q(L)=0 without a radical expansion.
    L_g = -(p_g*L**2-a_g*L+E_g)/q_prime
    c_g = -gradient(d)+3*gradient(h3)/value(h3)**2
    L1_g = 3*p_g*L+k*L_g+c_g
    F_g = (
        z_g*L1+z*L1_g-E_g
        -9*(p*L)**2*(p_g*L+p*L_g)+3*(p_g*L+p*L_g)
        -beta_g*L1**2-2*beta*L1*L1_g
    )

    H, d_independent = sp.symbols("H d_independent", real=True)
    T, Tbar = x+sp.I*y, x-sp.I*y
    Bbar = B[0]-sp.I*B[1]
    A0 = 2*H*d_independent+d_independent*(H+1)*rho-T*Bbar
    C0 = (
        4*d_independent*H**3*Tbar
        +d_independent*H**2*(3*H-1)*rho*Tbar
        -4*H**2*(Bbar-d_independent*Tbar)
        -2*(H**2+1)*rho*(Bbar-d_independent*Tbar)
    )
    I_model = C0/(2*(H-1)*A0**2)
    optical_point = {**point, H: hs[2], d_independent: distance}
    partials = [
        sp.simplify(sp.re(sp.diff(I_model, var).subs(optical_point)))
        for var in [x, y, H, d_independent]
    ]
    G_g = sp.Matrix(partials[:2])+partials[2]*gradient(h3)+partials[3]*gradient(d)
    determinant_F = sp.factor(sp.det(sp.Matrix.vstack(F_g.T, G_g.T)))
    expected_F = -sp.Rational(
        111531835863813011759655364632788762212925993,
        697588006297615660655096744783582437500000,
    )
    assert determinant_F == expected_F
    multiplier = sp.factor(-f1*q_prime)
    determinant_Q = sp.factor(multiplier*determinant_F)
    expected_Q = sp.Rational(
        44564075504928232438851441514392898089673277813010995366387,
        305774135107420688464082035443410091384429103125000000000,
    )
    assert determinant_Q == expected_Q
    print("f1 =", f1)
    print("f0 =", f0)
    print("L2 =", L)
    print("Positive-root Jacobian determinant =", determinant_F)
    print("Eliminant gradient multiplier =", multiplier)
    print("Eliminant Jacobian determinant =", determinant_Q)
    print("Both beam Jacobians are exactly nonzero.")


if __name__ == "__main__":
    check_symbolic_remainder()
    check_witness_jacobian()
