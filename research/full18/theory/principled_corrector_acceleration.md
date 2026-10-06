# Principled acceleration: reusable feasibility, not a repaired example

## Scope

The two-variable algebraic reversion and the convergence of its undamped finite-angle Picard iteration are different claims. A derivative with an eigenvalue outside the unit disk invalidates local attraction of that undamped iteration at the point; it does not invalidate the off-model algebraic left inverse, the exact fixed-point identity, or local identifiability of the physical forward map.

This note gives general reusable relaxation/preconditioning criteria. No relaxation parameter is selected for the predeclared kappa=1/10 witness, no modified iteration is numerically tested there, and no new wedge or physical case is introduced. These are mathematical feasibility statements only. Any later application needs its own declared certificate and full-record validation.

Let A be a regular bivariate two-record inverse and

    T_y(theta)=A(y-[F-F1](theta), y-[F-F2](theta)).

Work in fixed native-scaled coordinates x=S^-1(theta-theta0). Let Q be the derivative of the scaled T at the candidate. Root, alias, label, physical, and off-image branch domains are those of quantitative_algebraic_correction.md.

## 1. Positive scalar relaxation

For a fixed omega>0 define

    T_omega(theta)=(1-omega)theta+omega T_y(theta).      (1)

Its fixed points are exactly those of T_y. At a fixed point, every eigenvalue lambda of Q becomes

    mu=1-omega+omega lambda.

The exact condition |mu|<1 is

    Re(lambda)<1,
    0<omega<2[1-Re(lambda)]/|1-lambda|^2.               (2)

Therefore positive scalar relaxation can yield a locally attracting derivative if and only if every eigenvalue of Q lies strictly to the left of the vertical line Re(lambda)=1. A large negative eigenvalue does not by itself rule it out. An eigenvalue with real part at least one does rule out every positive scalar omega.

A reusable rule can be derived from a certified spectral enclosure, not by trying damping values. If it proves

    Re(1-lambda)>=a>0,       |1-lambda|<=b

for every eigenvalue, then any prescribed omega satisfying

    0<omega<2a/b^2                                       (3)

makes the spectral radius of the relaxed derivative less than one. Equation (3) is a sufficient interval bound; direct use of (2) can be sharper.

### Spectrum is not yet a finite norm certificate

For a nonnormal matrix, spectral radius below one does not imply contraction in the original infinity norm. A verified adapted norm is needed. One reusable construction is to produce a positive-definite P and certify

    A0^T P+P A0 >= 2a_P P,       A0^T P A0 <= b_P^2 P,
    A0=I-Q,        a_P>0.

Then

    ||I-omega A0||_P^2 <= 1-2omega a_P+omega^2 b_P^2<1

for 0<omega<2a_P/b_P^2. A continuous Lyapunov equation for A0 supplies a proposal when its eigenvalues have positive real part; interval Cholesky or equivalent matrix inequalities certify it. This is finite linear algebra, not a new optical nonlinear solve.

A derivative norm at one point proves only the existence of some local attraction neighborhood by continuity. A numerical positive radius requires the actual curvature and domain bounds in the scalar radii theorem. Native uncertainty must also include the conversion from the adapted P norm. A small omega can make convergence very slow; (2) is not a runtime or robustness guarantee.

## 2. Fixed Newton/chord preconditioning

Let B be a fixed nonsingular matrix in scaled coordinates. Define

    Psi_B(x,h)=x+B[Psi(x,h)-x],                         (4)

where Psi is the scaled map from quantitative_algebraic_correction.md. The native update is theta+S B S^-1[T_(y+h)(theta)-theta]. Its fixed points are unchanged because B is nonsingular.

If I-Q is invertible, the ideal center choice

    B=(I-Q)^-1                                          (5)

gives D_x Psi_B(0,0)=0. In a rigorous implementation, take an ordinary numerical inverse proposal B and certify the actual defect

    Z_B >= ||I-B(I-Q)|| <1.                             (6)

No exact matrix inverse needs to be trusted numerically. The condition (6) itself proves B and I-Q are nonsingular. B is then frozen across iterations and across the allowed data-error family. Recomputing B as a function of x or h would require additional derivative terms.

This is a reusable chord-Newton preconditioner for the fixed-point residual x-Psi(x,h). It adds an eighteen-by-eighteen linear solve (or thirteen-by-thirteen for the exact reduced lift), while the multivariate nonlinear reconstruction within each evaluation remains the two-beam-equation step. It is not a new algebraic elimination of physical unknowns, and must not be advertised as one.

### Direct transfer of the finite radius/noise theorem

Let Y,Z,L,Mxx,Mxh,Mhh be the bounds from quantitative_algebraic_correction.md. Compute the sharper center values directly:

    Y_B >= ||B Psi(0,0)||,
    Z_B >= ||I-B(I-D_x Psi(0,0))||,
    L_B >= ||B D_h Psi(0,0)||.

Safe curvature bounds are

    Mxx_B=||B||Mxx,       Mxh_B=||B||Mxh,
    Mhh_B=||B||Mhh.                                      (7)

Use these constants in the same scalar radius/noise polynomial. Domain and branch guards are unchanged; only the final self-map inequality changes. A poorly conditioned B can enlarge curvature and noise bounds enough to defeat the certificate despite its favorable center derivative. Thus the availability of (5) is not a finite-radius success claim.

## 3. Model-specific identity at an exact compatible point

At an exact compatible point the left-inverse identity gives

    A_s DF1+A_w DF2=I.

The undamped derivative consequently satisfies

    Q=I-S^-1(A_s+A_w)DF S,
    I-Q=S^-1(A_s+A_w)DF S.                                   (8)

Hence an unstable Q need not mean DF is rank deficient. It can mean that the explicit Taylor-based inverse badly scales or rotates the exact finite-angle Jacobian. Conversely, full rank of DF alone does not guarantee I-Q is invertible: its particular eighteen-row projection can still be singular.

Here A_s,A_w and DF use native output/input coordinates; the displayed S factors convert to the fixed scaled coordinates of Q. Equation (8) holds at matching exact Taylor inputs. At an off-model candidate, use the actual differentiated map; do not replace it by (8) unless its identity defect is accounted for.

This explains both the usefulness and the limit of a Newton/chord preconditioner. It can address iteration dynamics using known candidate derivatives. It cannot create information, remove genuine optical ambiguities, eliminate a weak-noise direction, or prove that a fixed point fits the entire 400-coordinate record.

## 4. General method versus case-specific repair

A principled reusable method declares in advance:

1. Which certified spectral or matrix-defect conditions trigger relaxation or chord preconditioning
2. How a fixed omega or B is selected from those bounds
3. Which adapted/native norm is used
4. The domain, curvature, and independent measurement-epsilon tests that must pass
5. What is returned when the certificate fails

Trying multiple wedge sizes, damping values, or favorable parameter boxes until one works is not such a theorem. The present note neither performs nor authorizes that search. It supplies mathematically valid options for a later algorithm, and leaves their useful finite-radius performance unproved until the complete certificate is evaluated.

All resulting candidates still require every original bound, strict optical/traversal test, and all 400 exact residual constraints. Global branch exclusion remains a separate obligation.


## 5. An operational, automatically defined certified chord rule

The rule below is reusable and symbolic. It is not numerically applied to the failed finite-wedge witness in this note.

For the nominal record and candidate, compute the exact center derivative Q0=D_x Psi(0,0) through the verified arithmetic/root circuits. In computation Q0 is enclosed, rather than assumed equal to a floating matrix. Put A0=I-Q0. Propose a numerical inverse W and certify

    E=I-W A0,          ||E||inf<=e<1.                   (9)

Then A0 is nonsingular, and the automatically defined exact chord matrix is

    P=A0^-1=(I-E)^-1 W.

No trial sequence of damping values is involved. A finite inverse enclosure is

    P_m=sum_(j=0)^m E^j W,
    ||P-P_m||inf <= e^(m+1)||W||inf/(1-e),
    ||P||inf <= p=||W||inf/(1-e).                      (10)

Matrix-ball evaluation retains dependency where practical. The Neumann tail is a rigorous remainder, not an assertion that a truncated inverse is exact. This calculation is an eighteen-dimensional linear solve, or thirteen-dimensional for the guarded sagittal lift; it is not a new nonlinear optical optimization.

Use P fixed in x and h. Define Psi_P=x+P(Psi-x). Its center x-derivative is exactly zero. Compute direct verified center products, preferably with the inverse enclosure rather than multiplying unrelated norm bounds:

    Y_P >= ||P Psi(0,0)||inf,
    L_P >= ||P D_h Psi(0,0)||inf.

Safe transformed Hessian bounds are

    Hxx=p Mxx,       Hxh=p Mxh,       Hhh=p Mhh.         (11)

Sharper componentwise verified products may replace (11). In particular, multiply the actual output Hessian arrays/circuits by P before taking absolute row bounds when possible; ||P|| times an unrelated scalar Hessian norm can lose important cancellations. This is finite arithmetic, not a new parameter optimization. The transformed radius/noise test is explicitly

    P_P(r,epsilon)=Y_P+L_P epsilon+(Hhh/2)epsilon^2
                    +Hxh r epsilon+(Hxx/2)r^2-r<=0,

    q_P(r,epsilon)=Hxx r+Hxh epsilon<1.                (12)

The original physical and A-branch domains must already be certified on that radius/noise envelope. The same shared-data correlation and all-400-residual qualifications apply.

If an implementation uses a fixed finite numerical approximation P_tilde instead of the exact matrix defined in (10), include

    Z_tilde >= ||I-P_tilde(I-Q0)||,

and use the earlier general radii polynomial with this nonzero center defect. Its curvature multipliers use a certified ||P_tilde||. Alternatively add an explicit evaluation-error allowance to each interval update. Treating a merely approximate inverse as having zero derivative defect would be unsound.

Failure of (9) means this particular chord chart is unproved; it does not prove a physical inverse is absent. Success of (9) alone is not success of (12). A large p, curvature, or noise amplification can still prevent a useful certificate.

## 6. Target the actual 0.001 native-coordinate requirement

To test a local all18 accuracy goal tau_i=0.001 in the original native units, one can set

    S=diag(tau_1,...,tau_18),        r=1.               (13)

This means exactly a 0.001 Hz speed interval, a 0.001 degree wedge/phase/beam-angle interval, a 0.001 index interval, and a 0.001 native-length source/gap/distance interval, as appropriate. It is a user-sized native box, not a wedge-dependent radius chosen after observing failure. Other specified coordinate tolerances can replace tau.

The finite local-domain and self-map tests become

    Y_P+L_P epsilon+(Hhh/2)epsilon^2+Hxh epsilon+Hxx/2 <= 1,
    Hxx+Hxh epsilon < 1.                              (14)

For constants already certified on a declared independent noise domain epsilon_bar, set

    C=1-Y_P-Hxx/2,       B=L_P+Hxh.

When C>0 and Hhh>0, a sufficient self-map allowance is

    epsilon <= 2C/[B+sqrt(B^2+2Hhh C)],                (15)

together with epsilon<=epsilon_bar, the contraction inequality in (14), and every branch/domain guard. If a fixed finite P_tilde is actually used, add its verified Z_tilde to the self-map left side and to the contraction bound in (14); use C=1-Y_tilde-Z_tilde-Hxx/2. The nominal denominator in (16) then becomes 1-Z_tilde-Hxx, with its corresponding L_tilde and curvature bounds. Linear and zero-coefficient cases are handled exactly as in the quantitative radii theorem. No relation between measurement epsilon and wedge kappa is inserted.

### Error of the delivered estimate, not just a self-map box

Let theta_hat(0) be the nominal fixed point, and let theta_m be the computed nominal iterate. Suppose a posteriori interval iteration gives

    ||S^-1(theta_m-theta_hat(0))||inf <= r_num.

For example the difference of consecutive exact-chord iterates divided by 1-Hxx gives such a bound on the unit box. For a fixed finite P_tilde use 1-Z_tilde-Hxx instead. Verified numerical evaluation errors must also be charged. The compatible-record displacement is bounded by

    r_noise(epsilon)
       =[(L_P+Hxh)epsilon+(Hhh/2)epsilon^2]/(1-Hxx).    (16)

Thus every physical theta_star in the covered chart with ||F(theta_star)-y||inf<=epsilon obeys

    |theta_m,i-theta_star,i|
       <= tau_i [r_num+r_noise(epsilon)].              (17)

A local 0.001 accuracy certificate for the delivered estimate requires r_num+r_noise<=1, not merely a small nominal update. Componentwise bounds can certify (17) more sharply.

Equations (13)-(17) can in principle be evaluated from finite arithmetic/root/matrix bounds. They have not been evaluated here. They certify only the chart covered by the parameter domain: proving the unknown true system lies there, or that every other original-prior branch has been excluded or enclosed to the same accuracy, remains necessary for an unconditional full18 recovery guarantee.

### If the thirteen-coordinate lift is used

A unit box in thirteen retained coordinates does not automatically give 0.001 errors in the five reconstructed ones. Write xi for the native retained coordinates and x=S13^-1(xi-xi0) for their scaled version. The reconstruction E_y takes xi as input and returns all eighteen native coordinates. On the certified envelope let Ex_(i,a) bound |(D_xi E_y S13)_(i,a)| and let Ey_i bound the row-one-norm of D_y E_y. Propagate

    |Delta theta_i|
       <= sum_a Ex_(i,a) |Delta x_a| + Ey_i epsilon,

plus the delivered-iterate/evaluator enclosure. Require the resulting bound to be <=0.001 separately for every one of the eighteen native coordinates. This charges subresultant and geometry-pivot sensitivity rather than hiding it in dimension reduction.

## 7. Exact inputs and calculations still needed

A full 0.001 certificate needs:

1. The actual 200 paired observations and a certified real-coordinate error allowance epsilon, including numerical-generation or timestamp/evaluator error where applicable
2. A data-generated candidate, all retained long-lag lifts/prism assignments/bivariate root charts, and a declared center; no hidden true parameters are supplied
3. Every original prior, Snell, traversal, spectral, denominator, and bivariate-branch guard on the native unit box and its corrected-record envelope
4. Enclosures for the center update, Q0, D_h Psi, and the inverse residual (9)
5. The actual transformed Y_P,L_P,Hxx,Hxh,Hhh on that same envelope, including the output-lift derivatives when thirteen coordinates are used
6. A passing test (14), a verified finite iteration error, and the full18 bounds (17)
7. Every original residual check, plus exclusion, coverage, or an explicit unresolved report for all other original-prior branches

These are finite calculations, but their success and useful numerical size are not established by the theorem alone. The rule targets the user's stated accuracy directly. It neither hides an arbitrarily tiny radius nor promises that a particular failed raw iteration will be repaired.

## Independent audit

The general spectral/chord feasibility statements and native-scaling identity were independently checked in finite_wedge_derivative_independent_audit.md. The operational inverse enclosure, transformed radius/noise rules, actual 0.001 native target, finite-precision defect, and reconstructed-coordinate error propagation passed a separate final review in certified_chord_rule_independent_audit.md. No numerical application of a modified iteration was part of either review.
