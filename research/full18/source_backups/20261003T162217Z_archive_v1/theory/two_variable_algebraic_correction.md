# A two-variable, true 17+1 algebraic corrector

## Main result

The four-variable reversion in algebraic_defect_correction.md can be sharpened. Retain the spare first-prism shape consistency instead of discarding it, and use only one real quadrature of the physical third self-second harmonic. The result is an explicit off-model reversion A17+1 whose only multivariate nonlinear step is **two equations in the two beam-slope coordinates**.

Every iteration still updates all eighteen physical unknowns. This is an eighteen-coordinate algebraically preconditioned Picard iteration, not an exact elimination of the entire finite-angle problem to two unknowns. Nevertheless, each evaluation consists of a scalar degree-seven frequency polynomial, linear algebra, a bivariate algebraic root, and explicit rational/radical reconstruction. No fourteen- or seventeen-variable nonlinear correction is hidden in the definition.

The exact finite-angle iteration, finite native-coordinate contraction certificate, and noisy-compatible-set bound in algebraic_defect_correction.md apply with A replaced by A17+1. The one quadratic quadrature means that the second-input derivative has rank at most one. Thus this variant preserves the genuine 17+1 information split and one weak quadratic noise direction.

The bivariate branch is provably regular at the existing audited oblique witness. One additional exact rational check verifies the elimination denominator there. No new parameter campaign is used, and no full +/-18-degree or global completeness claim is made.

## 1. Inputs from the two-record construction

Use exactly the off-model record preprocessing in algebraic_defect_correction.md:

1. From the first ordinary paired record s, a full-rank stacked Hankel least-squares recurrence gives a degree-seven polynomial. Select the six rotor roots on separated disks, radially normalize conjugate pairs, and fit first-order coefficients. This recovers candidate signed speeds N_i, baseline B, and signed 2x2 coefficient matrices C_i, with all prism labels retained.
2. From the second ordinary paired record w, use the 31-column quadratic-plus-confluent design at those candidate speeds. Extract the complex positive self-second coefficient C_plus2 for the assigned third prism.
3. Put S_i=C_i C_i^T. For the complex columns a,b of C_3 let U=(a-i b)/2, and set I_obs=C_plus2/U^2.

All these operations are defined on ordinary records near a regular Taylor-model pair. Neither s nor w is presumed to lie in a finite-tone model image. The root/sign/label choices and all nonzero-denominator domains are retained explicitly.

Choose one nonzero fixed complex number chi, and keep it fixed throughout a correction branch. At the audited witness chi=1, so only Re(I_obs) is used. More generally chi is selected so the real projection Re(chi I) changes along the first-order ambiguity curve; this must be certified rather than assumed. There is no need to use both real components of I_obs.

## 2. Rational ellipse invariants in an unnormalized beam chart

Let the trial beam slope be t=(x,y), define

    v=(-y,x),       rho=x^2+y^2,
    Bv=v^T B,      Bt=t^T B,
    D_i=v^T S_i v, A_i=t^T S_i v,
    Delta_i=sqrt(det S_i)&gt;0.

The Delta_i are coefficients determined by s, not additional unknowns. On the chart rho&gt;0 and Bv!=0, define

    beta_i = -A_i/(D_i Bv),
    alpha_i = [rho Delta_i Bv-D_i Bv-Bt A_i]
              /(rho D_i Bv).                            (1)

These are rational functions of the two real unknowns x,y, with data-dependent coefficients. To verify (1), rotate to the unit beam frame:

    c_i=(S_i)_12/(S_i)_22 = A_i/D_i,
    r_i=sqrt(det S_i)/(S_i)_22 = rho Delta_i/D_i,
    T B_perp=Bv,       B_parallel/B_perp=Bt/Bv.

Substitution in the audited alpha,beta formulas gives (1). No square root of rho is needed in the rational equations.

Physical candidates require beta_i&gt;0 and alpha_3&gt;1. Then the last hardware variables are rational:

    h3 = 1/(alpha_3-1),
    d = (alpha_3-1)/beta_3,
    E = 3(h3^(-3)-h3^(-1)) &lt; 0.                         (2)

These are functions of t; h3 and d are no longer polynomial unknowns.

## 3. Eliminate the downstream positive quadratic exactly

Abbreviate the rational functions

    p = beta_2,       a = alpha_2-1,
    z = alpha_1-1,    b = beta_1,
    k = 2+3p,        c = -d-3/h3.

Here the symbols a,b,c,k are local algebraic abbreviations, not the native wedge, source, or sample index. The second-prism lever L=L2 is the unique positive root of

    Q(L)=p L^2-a L+E=0.                                 (3)

Uniqueness follows from p&gt;0 and E&lt;0: the two roots are real and have opposite signs. The downstream formulas imply

    h2=1/(pL),
    g=L-d-3/h3,
    L1=d+2g+3/h2+3/h3 = kL+c.                           (4)

Multiply the first-prism scalar consistency equation by L1. It becomes

    f(L)=z(kL+c)-E-3[(pL)^3-pL]-b(kL+c)^2=0.            (5)

Reduce this cubic modulo (3), using

    L^2=(a/p)L-E/p,
    L^3=(a^2/p^2-E/p)L-aE/p^2.

The remainder is f1 L+f0, where the explicit rational functions are

    f1 = zk+3p-3p a^2+3p^2 E-b(k^2 a/p+2kc),
    f0 = zc-E+3p aE+b k^2 E/p-bc^2.                    (6)

On the chart f1!=0, the only possible common root is

    L=-f0/f1.                                           (7)

Substitution in Q gives one rational equation in t alone:

    R(t)=p f0^2+a f0 f1+E f1^2=0.                       (8)

Equations (7)-(8), together with L&gt;0 and p&gt;0,E&lt;0, are exactly equivalent to the positive-root system (3),(5). They do not add an unphysical choice of the negative quadratic root. Denominator clearing is allowed only with rho, Bv, D_i, beta_3, alpha_3-1, p, and f1 all nonzero as already required.

This is a real reduction: the downstream lever and all five hardware variables have been explicitly eliminated before the beam solve. The physical-order consistency is retained instead of being checked only afterward.

### The exceptional elimination chart

If f1=0 but f0!=0, no L can satisfy (5) at that t. If f1=f0=0, (5) vanishes at both roots of Q; its unique positive root must still be retained. Such a point is not excluded as unphysical. Two valid alternate formulations are:

- retain L as a third algebraic unknown and solve (3),(5) together with the weak equation below;
- keep the explicit positive radical L_plus=(a+sqrt(a^2-4pE))/(2p) and solve f(L_plus(t))=0 and the weak equation as two analytic equations.

The bivariate polynomial chart (8) can become singular on f1=f0=0 even if the radical chart remains regular. This is a representation singularity and must not be used as an exclusion certificate.

## 4. One real quadratic equation completes the beam solve

For T=x+i y, H=h3(t), and d=d(t), put Rho=T conjugate(T)=rho and define

    A0 = 2Hd+d(H+1)Rho-T conjugate(B),
    C0 = 4dH^3 conjugate(T)+dH^2(3H-1)Rho conjugate(T)
         -4H^2(conjugate(B)-d conjugate(T))
         -2(H^2+1)Rho(conjugate(B)-d conjugate(T)).

The audited normalized physical self harmonic is

    I_model(t)=C0/[2(H-1)A0^2].                          (9)

Keep H&gt;1 and A0!=0. This is rational in the two real beam coordinates because H,d already are rational functions (2).

The second beam equation is

    W_chi(t)=Re(chi [I_model(t)-I_obs])=0.               (10)

Thus the only multivariate root task is

    R(x,y)=0,          W_chi(x,y)=0.                    (11)

Both equations are rational and can be converted to real polynomial numerators on the explicitly retained denominator chart. Clearing the complex denominator in (9) with its modulus squared is legitimate where A0!=0. No small claim about the expanded total degree or the number of candidate roots is made here. The compact rational circuit is usually preferable to premature expansion.

All real roots satisfying the chart conditions are potential formal candidates. Isolate the relevant simple roots, preserve every physical prism assignment, and retain other branches. A selected correction branch needs a single-valued C1 real root over its whole input domain, not just an invertible Jacobian at its center. A two-variable uniform root inclusion or discriminant-free continuation can certify that domain.

## 5. Explicit reconstruction of the remaining sixteen coordinates

For each retained beam root, use (2) and (7), then (4), and set

    h1 = 1/(beta_1 L1),
    n_i=sqrt((h_i^2+rho)/(1+rho)),
    source = B-t[6+2g+d+3 sum_i 1/h_i].                 (12)

The first-order M_i are now known. Recover wedge and phase from X_i=M_i^(-1) C_i using the conformal formulas

    x_i=((X_i)11+(X_i)22)/2,
    y_i=((X_i)21-(X_i)12)/2,
    phi_i=atan(y_i/x_i),
    e_i=x_i sqrt(1+(y_i/x_i)^2),                         (13)

with phi_i in the narrow native phase chart, and a_i^native=arcsin(e_i). The speed coordinates were already recovered from the first record. Convert beam slopes and all angles to their original native conventions and units.

In this 17+1 variant, equation (8) enforces the formerly spare ellipse consistency. Consequently, wherever all chart conditions hold, all three normalized M_i shapes match the fitted C_i shapes. With positive orientation this makes X_i exactly a signed scale times a rotation, not merely a conformal approximation. Formula (13) remains a convenient explicit analytic expression.

The first record s itself may still have a nonzero seven-tone fit residual. The map preserves its extracted seventeen first-order coordinates, not every entry of an arbitrary s. Off-model A17+1 is nevertheless well-defined on an ordinary open set of record pairs.

## 6. Left-inverse identity and one weak direction

For a matching Taylor pair (s,w)=(F1(theta),F2(theta)), preprocessing recovers the exact seventeen first-order coordinates and the assigned self harmonic. The real beam t satisfies (11) and its positive-root filter. Its simple local branch therefore recovers exactly t. Equations (2),(4),(7),(12),(13) then recover every native parameter. Thus

    A17+1(F1(theta),F2(theta))=theta.                     (14)

The second record enters A17+1 only through the one real number

    q_obs=Re(chi C_plus2/U^2).

For fixed s it is a real linear functional of w. Therefore D_w A17+1 has rank at most one wherever the selected branch is differentiable. The first input provides all seventeen leading observables. This is a true 17+1 construction, unlike the four-variable branch in the companion memo, which discards one strong consistency and uses two real quadratic components.

## 7. Regularity at the existing original-prior witness

Use the already audited rational witness

    t=(1/3,0), B=(1411/35,2),
    (h1,h2,h3,g,d)=(3/2,7/5,5/3,3,100),
    phases=0, N=(1,7,49)/20.

The actual physical indices satisfy

    (n1^2,n2^2,n3^2)=(17/8,233/125,13/5),

and source=(1,2). All these non-wedge coordinates lie inside the original bounds. The existing first-order 6x6 shape determinant is nonzero, and the existing derivative of Re(I_model) along its remaining first-order fiber is nonzero. Therefore the two analytic equations consisting of positive-root shape consistency and Re(I_model)-Re(I_obs) have an invertible two-by-two beam Jacobian at this point. This uses chi=1.

To see that replacing positive-root consistency by (8) preserves this regularity, write L_plus,L_minus for the roots of Q. The exact identity is

    R=p [f1 L_plus+f0][f1 L_minus+f0].                   (15)

At a compatible positive root, its differential relative to the unreduced consistency is multiplied by

    p f1(L_minus-L_plus) = -f1 Q'(L_plus).

One additional rational proof check gives

    f1 = 399563722409659 / 317911540674000,
    f0 = -399563722409659 / 3033507067500,
    L_plus = -f0/f1 = 524/5,
    L_minus = -1008/625,
    Q'(L_plus) = 16627/22925.

In particular f1 is nonzero, and the multiplier is

    -511042000961953861 / 560624774611650000 != 0.      (16)

An independent direct rational differentiation, holding the input S_i=M_i M_i^T and B fixed, also gives

    det D_(x,y)(R,Re I_model)
      = 44564075504928232438851441514392898089673277813010995366387
        /305774135107420688464082035443410091384429103125000000000
      != 0.

This is an unscaled nonzero-Jacobian certificate, not a conditioning constant or finite-wedge convergence margin. Multiplying either equation by cleared denominators changes its determinant value.

Thus (11) has a simple beam root at the audited witness, and a real analytic off-model branch exists around it. The separated degree-two/confluent sample nodes, nonzero Hankel tones, and strict physical small-wedge margins are those already established at this witness. The construction works on a nonempty family of actual original-prior systems; no frequency or hardware value is supplied to the algorithm from the proof witness.

## 8. Exact finite-angle correction, certification, and noise

Replace A in algebraic_defect_correction.md by A17+1:

    theta_(m+1)=A17+1(y-[F-F1](theta_m),
                     y-[F-F2](theta_m)).                (17)

The cancellation from the six confluent frequency-derivative columns remains unchanged. In a correctly scaled regular e=epsilon a chart and its coupled record tube,

    D_s A17+1=O(epsilon^-1),
    D_w A17+1=O(epsilon^-2),
    D R1=O(epsilon^2),   D R2=O(epsilon^3),

so the update derivative is O(epsilon). Only the four-variable inverse derivative in the companion proof is replaced by the inverse two-by-two Jacobian of (11). Its uniform nonsingularity is a branch-domain condition. The second-input derivative has rank at most one, as proved above.

There is also an arbitrary-order reversion consequence. For a matching exact regular family, the data-only initializer A17+1(y,y) has scaled parameter error O(epsilon). Because each correction has Lipschitz constant O(epsilon), after m corrections the scaled error is O(epsilon^(m+1)) for each fixed m. The number of variables in the algebraic root step never grows with m, and neither higher-order harmonic extraction nor observed time derivatives are required. The exact infinite iteration, where its certificate holds, converges to the compatible system; the finite-m statement is an asymptotic accuracy statement, not a global all-prior bound.

For a finite native-coordinate box D centered at theta0 with positive diagonal scale S and radius r, explicitly certify

    q &gt;= sup_D ||S^-1[(A17+1)_s D R1
                      +(A17+1)_w D R2]S|| &lt; 1,
    delta &gt;= ||S^-1(T_y(theta0)-theta0)||,
    delta+q r &lt;= r.                                     (18)

Together with uniform physical and off-model branch guards, this gives the same existence and uniqueness of a fixed point and convergence of the explicit update as before. Every exact compatible solution in D is that fixed point. A fixed point alone does not prove full-record compatibility.

For noise, retain the exact shared-error derivative

    L &gt;= sup ||S^-1[(A17+1)_s+(A17+1)_w]||,

on the simultaneous-shift domain. Every compatible physical theta_star in D lies within scaled radius L eta_obs/(1-q) of the fixed point. Intersect that tube with all 400 original observation strips and every original bound/branch/traversal constraint. The asymptotic coupled tube has first-record width O(epsilon^2) and second-record width O(epsilon^3); noise outside that width is not covered by that asymptotic proof, though a separately checked finite certificate may permit it.

## 9. Scope of the improvement

The current best constructive correction is therefore:

    unknown-frequency degree-seven algebraic extraction
    -&gt; 31-column confluent finite-record demixing
    -&gt; two beam equations
    -&gt; explicit full18 reconstruction
    -&gt; exact forward residual feedback and certification.

There is no finite linear shift lift of the exact optics, no known-frequency assumption, no hidden exact extraction of formal coefficients from finite-angle data, and no high-dimensional implicit correction solve. Formal Taylor quantities are evaluated only from current parameter trials; the remainder in (17) uses the exact finite-angle model.

What remains is substantial: a useful explicit finite-wedge basin must pass (18), all candidate root/sign/label branches must be covered or honestly left unresolved, and global physical alternatives and boundary solutions require the existing complete/exclusion machinery. The derivation does not certify 18-degree wedges uniformly or resolve arbitrary original-prior data globally. It gives a new low-dimensional algebraic step inside a rigorously defined exact-model iteration.

## 10. Audit and reproducibility

The formulas (1)-(16), the off-model interpretation, and the rank-one weak direction have been independently reviewed. The new exact rational proof check is recorded in proof_checks/two_variable_reversion.py. The independent report two_variable_defect_independent_audit.md records the algebraic identities, guarded elimination, and direct exact two-by-two Jacobian test. This is mathematical and symbolic audit, not proof-assistant formalization, a reconstruction campaign, or a practical runtime claim.
