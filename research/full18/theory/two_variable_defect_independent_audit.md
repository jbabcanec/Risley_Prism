# Independent audit of the two-variable algebraic correction variant

## Verdict

The proposed two-variable replacement for the four-variable last-prism solve is sound on a regular elimination chart. It retains all seventeen first-order directions and uses one real quadratic quadrature. The remaining nonlinear root operation is two real rational equations in the two unknown beam-slope components, or two polynomial equations after guarded denominator clearing. This is a stronger per-iteration algebraic reduction than the four-variable construction.

The exact finite-angle correction still updates eighteen coordinates by a fixed-point iteration. This is not a claim that the entire exact finite-angle inverse is globally equivalent to a two-dimensional algebraic variety. The coupled-record tube, confluent demixer, exact forward remainder, physical constraints, branch isolation, contraction certificate, and complete-record validation requirements remain necessary.

The completed `two_variable_algebraic_correction.md` was read through Section 10 and passed this audit. The algebraic identities and a nonzero two-by-two Jacobian at the existing rational witness were checked independently and exactly. Reproducible assertions are saved in `proof_checks/audit_two_variable_jacobian.py`; the script passed. There was no parameter sweep.

## 1. Shape reconstruction with an unnormalized beam frame

For trial t=(t_x,t_y), define v=Jt=(-t_y,t_x), rho=t dot t, and fixed observed positive-definite ellipse covariances S_i=C_i C_i^T. Put

    D_i=v^T S_i v,
    A_i=t^T S_i v,
    b_perp=v dot B,
    b_parallel=t dot B.

The proposed rational formulas are correct:

    beta_i = -A_i/(D_i b_perp),
    alpha_i = [rho sqrt(det S_i) b_perp
               -D_i b_perp-b_parallel A_i]
              /(rho D_i b_perp).

Indeed the orthonormal beam-frame values satisfy c_i=A_i/D_i, r_i=rho sqrt(det S_i)/D_i, T B_perp=b_perp, and B_parallel/B_perp=b_parallel/b_perp. Substitution into the existing shape inverse yields exactly these expressions.

The square roots sqrt(det S_i) are positive constants determined by the first-input coefficient fit. They do not add beam unknowns. On arbitrary nearby inputs they are smooth because S_i remains positive definite. The denominators rho, D_i, and b_perp must be excluded explicitly.

## 2. Eliminating the positive second-prism lever

Recover

    h3=1/(alpha3-1),
    d=(alpha3-1)/beta3,
    E3=3(h3^-3-h3^-1).

Write p=beta2, a=alpha2-1, E=E3, z=alpha1-1, beta=beta1, k=2+3p, and c=-d-3/h3. If L denotes L2, then

    q(L)=p L^2-a L+E=0,
    L1=k L+c.

The last equality follows directly from g=L-d-3/h3 and h2=1/(pL). The remaining first-order consistency is

    f(L)=z(kL+c)-E-3[(pL)^3-pL]-beta(kL+c)^2=0.

It is cubic in L. Direct symbolic division modulo q gives f(L)=f1 L+f0 whenever q(L)=0, with

    f1=z k+3p-3p a^2+3p^2 E
       -beta(k^2 a/p+2kc),

    f0=z c-E+3p aE+beta k^2 E/p-beta c^2.

An independent symbolic remainder calculation gives zero defect for these formulas.

On f1!=0, the first-order closure gives L=-f0/f1. Substitution into q produces the beam-only equation

    Q(t)=p f0^2+a f0 f1+E f1^2=0.

The sign on the middle term is positive. It follows from the minus sign in L=-f0/f1 and the term -aL in q.

The reconstruction must retain L>0, p>0, E<0, all original hardware bounds, and every denominator exclusion. These ensure that L is the physical positive root of q. Merely solving Q=0 without the positive-root filter can admit the negative lever branch.

## 3. The second beam equation

Let I_model(t,h3,d) be the normalized third self-harmonic from the previously audited full-vector formula, and let I_obs be the ratio extracted from the two input records by the confluent construction. Choose a fixed real quadrature ell with nonzero derivative along the first-order fiber at the witness. The already audited witness permits ell(I)=Re I.

The second equation is

    G(t)=ell(I_model(t,h3(t),d(t))-I_obs)=0.

The model invariant is rational in t_x,t_y,h3,d, and h3,d are rational functions of the trial beam coordinates. Thus G is rational in the two beam coordinates. The real part can be written rationally using the conjugate denominator. Clearing denominators yields a polynomial equation only when those denominators are retained as explicit exclusions. In particular, H, H-1, A0, conjugate(A0), beta3, alpha3-1, and the shape denominators cannot be silently restored as roots.

No second nonlinear hardware solve remains after Q=G=0: the hardware follows from the rational formulas and L=-f0/f1, and the original wedge, phase, source, and speed reconstruction is unchanged.

## 4. Exact elimination-chart check at the existing witness

Use the same physical witness:

    t=(1/3,0), B=(1411/35,2),
    h=(3/2,7/5,5/3), g=3, d=100.

The shape covariances can be taken to be S_i=M_i M_i^T because their common amplitude factors cancel. All entries are rational, and sqrt(det S_i)=det M_i>0.

Independent exact calculation gives

    L2=524/5,
    L1=3848/35,

    f1=399563722409659/317911540674000,
    f0=-399563722409659/3033507067500,

    -f0/f1=524/5,
    q_L(L2)=16627/22925,
    L_negative=-1008/625.

In particular f1 is nonzero and the positive-root elimination chart is valid. The reconstructed first-order consistency is exactly zero.

## 5. Equivalence to the positive-root closure and a direct Jacobian witness

Let L_plus(t) be the smooth positive root of q and F(t)=f(L_plus(t)). Polynomial division gives F=f1 L_plus+f0. The eliminant therefore satisfies the exact local identity

    Q(t)=-f1 q_L(L_plus) F(t)+p F(t)^2.

At a root of F, its gradient is multiplied by -f1 q_L(L_plus). Both factors are nonzero in the witness chart. Thus the eliminant has the same local first-order closure as the positive-root form; it does not create a multiple root there.

An independent rational differentiation holding the input S_i and B fixed gives

    det D_(t_x,t_y)(F,Re I_model)
      =-111531835863813011759655364632788762212925993
        /697588006297615660655096744783582437500000
      approximately -159.8821006911486.

For the eliminant itself, the determinant is

    det D_(t_x,t_y)(Q,Re I_model)
      =44564075504928232438851441514392898089673277813010995366387
        /305774135107420688464082035443410091384429103125000000000
      approximately 145.7418087022715.

The derivative multiplier is exactly

    -f1 q_L(L2)
      =-511042000961953861/560624774611650000.

These determinant values use the displayed unscaled definitions of F and Q. Multiplying either equation by an admissible nonzero denominator changes the numerical determinant, but not its nonvanishing.

The direct calculation independently verifies the expected conclusion from the prior six-by-six shape determinant and the nonzero real quadratic derivative along the one-dimensional first-order fiber: the two beam equations have a simple real root at the existing witness.

## 6. Degenerate charts and exclusions

The chart f1!=0 must not be silently extended across its zero set. At a point where f1=0 and f0!=0 there is no common root with q. When f1=f0=0, q divides the cubic closure at that point and the eliminant alone loses the lever reconstruction.

Such points can be kept using the explicit positive radical L_plus(t), giving a two-variable analytic equation involving one square root, or by retaining L as a third unknown with q=0 and f=0. These are alternate charts, not evidence that an original-prior region is empty.

Rational-to-polynomial clearing can also add components at any other discarded denominator. Every such component must be removed by the stated nonzero conditions or returned separately. No generic isolated-root count, global finite-fiber assertion, or runtime bound follows merely from using two unknowns.

## 7. Transfer of the exact correction theorem

Define the new off-image map A2(s,w) using the unchanged frequency, coefficient, and confluent extraction stages, then the isolated Q=G=0 beam root and explicit remaining reconstruction. It has the exact identity

    A2(F1(theta),F2(theta))=theta

on its matching regular branch. Its dependence on w is only through one real normalized quadratic quadrature. This restores the intended seventeen-plus-one feature choice and avoids the extra discarded first-order condition of the four-variable variant.

On a compact simple-root chart, the two-dimensional inverse Jacobian is bounded. Consequently the same scaled estimates hold:

    D_s A2=O(epsilon^-1),
    D_w A2=O(epsilon^-2).

The confluent demixer is still essential to the first bound. The exact remainder derivatives remain O(epsilon^2) and O(epsilon^3) in the scaled wedge chart, so the exact correction derivative remains O(epsilon). Every finite contraction and shared-record noise certificate from the four-variable construction transfers with A2 substituted for A.

The gain is an explicit reduction in per-step nonlinear algebra and a cleaner weak-channel choice. All statements remain branchwise and conditional on certified domains, physical margins, and full-record compatibility. No numerical finite-angle threshold or original-prior completeness is established by this audit.
