# Quantitative, record-derived certificates for the bivariate corrector

## Result and exact scope

The bivariate full18 correction admits a finite a posteriori certificate whose decisive test is an explicit scalar quadratic radius/noise inequality. Its constants come from the supplied record, a candidate parameter vector, small matrix inverses, two-variable root margins, and operation-by-operation derivative bounds. It requires no additional nonlinear optimization over fourteen or eighteen unknowns.

Two refinements materially improve its quantitative prospects:

1. Impose the known DC root in a long-lag Prony chart. The large beam baseline cancels before inversion. Every speed alias is retained and checked on all 200 original timestamps.
2. Use frozen, verified left-inverse charts for the Prony and confluent demixing stages. Their off-model extensions have explicit rational radius and derivative bounds while retaining the exact Taylor-model left-inverse identity.

A bounded interval check at the existing rational witness, without choosing a finite wedge amplitude, certifies a normalized lag-11 Hankel left inverse with infinity norm below 0.367 and smallest singular value above 20.4. It also certifies an exact third-self-harmonic extraction row with complex-input l1 norm below 1.002. These are real quantitative improvements to the spectral stages.

This memo does **not** certify the entire nonlinear iteration at a specified nonzero wedge angle or a numerical observation error. The optical-remainder, reconstruction-curvature, and branch-domain constants still have to be evaluated for the actual candidate and record. The result is more explicit than an unevaluated sup ||DT||<1, but it is not a claim that a complete finite-angle witness certificate has already passed.

Throughout this memo, kappa is a wedge-amplitude scale; epsilon is measurement uncertainty in each of the original 400 real observation coordinates. They are independent quantities. No measurement error is silently set equal to a power of kappa.

## 1. Fixed data, candidate, and native scales

Let F be the exact full18 vector-Snell record, F1 and F2 its wedge-degree-one and -two Taylor records, and Ri=F-Fi. All use the original 200 timestamps k/20 and all original parameter bounds. The Taylor records and remainders are model evaluations, not measured temporal derivatives.

Let theta0 be a candidate obtained from data without known true speeds or hardware. Let S be a specified invertible real native-coordinate scale matrix. A positive diagonal S is simplest; an explicitly reported adapted strong/weak basis is also permitted. Define

    x=S^-1(theta-theta0),       D_r={theta0+Sx: ||x||inf<=r}.

A box enclosure in native coordinate i is theta0_i +/- r sum_j |S_ij|. Degree/radian conversions and the nonlinear sine/arcsine wedge conversions remain in every derivative. An adapted scale never removes the obligation to report bounds in the original coordinates.

Fix a root/sign/label/alias chart of the two-record bivariate inverse A. It can use the quantitative refinements below. For any record perturbation h define

    T_(y+h)(theta)=A(y+h-R1(theta), y+h-R2(theta)),
    Psi(x,h)=S^-1[T_(y+h)(theta0+Sx)-theta0].             (1)

The same h enters both record arguments. Every true exact solution theta in this chart satisfies T_F(theta)(theta)=theta.

All charts and preconditioners below are frozen while x and h vary. Recomputing them adaptively during a proof without differentiating that choice would invalidate the stated bounds.

## 2. A DC-constrained long-lag Prony chart

Choose an integer lag ell with 1<=ell<=28. Let K_ell=200-7ell and stack the two axes and starts k=0,...,K_ell-1:

    H_s[(axis,k),j]=s_axis,k+j ell,        j=0,...,6,
    v_s[(axis,k)]=s_axis,k+7ell.

There are 2K_ell rows, all using original timestamps. The first-order annihilator has the exact root one, regardless of the unknown baseline. Put

    cbar=-(1/7)1,
    R in R^(7x6),       R^T R=I, R^T1=0,
    J_s=H_s R,          u_s=v_s+H_s cbar.                (2)

A convenient fixed R is the Helmert basis: column j, j=1,...,6, has its first j entries 1/sqrt(j(j+1)), its (j+1)-st entry -j/sqrt(j(j+1)), and the rest zero. The large constant baseline cancels exactly in both J_s and u_s.

For a center record s_c, require rank J_c=6. Propose a numerical left inverse P and certify P J_c invertible. Define the exact frozen left inverse

    D=(P J_c)^-1 P,       D J_c=I.                       (3)

For any nearby ordinary record s define

    a(s)=-[D J_s]^-1 D u_s,
    c(s)=cbar+R a(s),
    p_s(z)=z^7+sum_(j=0)^6 c_j(s) z^j.                 (4)

The identity p_s(1)=0 holds even off the model image. Thus speed root work can use the degree-six factor p_s(z)/(z-1). No exact recurrence of the perturbed record is assumed.

At every exact first-order record on which D J_s is invertible, the true constrained recurrence solves J_s a=-u_s; (4) returns it exactly. Therefore the left-inverse proof for A is unchanged. This chart replaces the off-model extension, not the physical model.

### Explicit finite first- and second-derivative bounds

At s_c put a_c=-D u_c. Let the two fixed linear record operators have certified norms

    mu >= || D D_s J ||_(record-inf -> matrix-inf),
    gamma >= || D D_s(u+J a_c) ||_(record-inf -> vector-inf).

They are finite arrays assembled from the lag shifts, R,D, and a_c. Their exact infinity norms can be bounded by coefficient row sums; no parameter optimization is involved. For ||s-s_c||inf<=d_s with mu d_s<1, put

    a_delta=gamma d_s/(1-mu d_s),
    a_1=(gamma+mu a_delta)/(1-mu d_s),
    a_2=2mu a_1/(1-mu d_s).                             (5)

Then

    ||a(s)-a_c||inf<=a_delta,
    ||D_s a||<=a_1,       ||D_s^2 a||<=a_2.

These follow by differentiating the six-by-six linear system D J_s a=-D u_s. In particular, D(u_c+J_c a_c)=0 even if the raw center recurrence residual is not zero. Multiplication by ||R||inf transfers derivative bounds to c. Cruder immediately available bounds are mu<=7||D||inf||R||inf and gamma<=||D||inf(1+||c_c||1), with the exact structured row sums usually sharper.

### Roots, speed lifts, and all-200 alias validation

Certify disjoint disks for the six roots of p_s/(z-1), separated from the DC root and the real axis. A fixed radius delta_z around each center root z_j gives a direct Rouche test. On its boundary,

    |p_c(z)| >= delta_z product_(l!=j)(|z_j-z_l|-delta_z),

while a coefficient perturbation bounded by delta_c in infinity norm gives

    |p_s(z)-p_c(z)| <= delta_c sum_(n=0)^6 (|z_j|+delta_z)^n.

The product runs over all seven roots of p_c, including the exact DC root 1. Require the strict inequality delta_c sum_(n=0)^6 (|z_j|+delta_z)^n < delta_z product_(l!=j)(|z_j-z_l|-delta_z), with every factor positive. These tests have finitely many scalar inputs. They certify one root per disk without assuming the perturbed roots lie on the unit circle. If center roots are represented by enclosing balls rather than exact algebraic points, subtract their certified radii from the first boundary-distance factor and every pair-distance lower bound before applying the inequality. Radially normalize each root as before.

The lag nodes are z_i^ell, not the original sampled nodes. From either chosen member w of a conjugate pair, retain every signed lift

    nu=20[Arg(w)+2pi m]/(2pi ell) in [-3.5,3.5],

use |nu| for the real cosine/sine fit, and deduplicate. Equivalently enumerate both conjugate representatives. A positive physical frequency may map to the negative-argument lag root, so retaining only positive lifts of the positive-argument representative is wrong.

For each retained lift triple, fit all 200 original samples, recover actual rotation signs from oriented ellipse determinants, and retain all physical prism assignments. On an exact nondegenerate F1 record, a different frequency set cannot also fit all 200 points: the union of two seven-tone dictionaries has at most thirteen distinct nodes, and its Vandermonde has full rank. On finite-angle or noisy inputs, a minimum residual is only an initializer; it does not certify rejection of the other lifts. Use the existing remainder/noise outer cover or keep them unresolved.

The inequality ell<=28 supplies enough rows, not rank or distinctness. Those are explicit chart guards.

## 3. Explicit root and demixer derivatives

For a simple polynomial root, a certified lower bound m_p for |p'(z)| gives

    z_1 <= P_c c_1/m_p,
    z_2 <= [P_c c_2+2P_zc c_1 z_1+P_zz z_1^2]/m_p,    (6)

where on a disk |z|<=R_z one can take

    P_c=sum_(j=0)^6 R_z^j,
    P_zc=sum_(j=1)^6 j R_z^(j-1),
    P_zz=42 R_z^5+sum_(j=2)^6 j(j-1)|c_j| R_z^(j-2).

A lower bound for m_p also follows from the product of certified separations to the other roots. If |z|>=m_z>0, the chosen original-frequency lift has derivative bounds

    N_1 <= [20/(2pi ell)] z_1/m_z,
    N_2 <= [20/(2pi ell)](z_2/m_z+z_1^2/m_z^2).        (7)

All quantities are data/candidate-derived disk bounds. The integers specifying the lift are fixed within a branch.

Use the same 31-column quadratic/confluent design V(N) as before, scaling its six confluent columns to (k/199)z_i^(+/-k). This scaling does not change the nuisance span or the extraction identity, and avoids needlessly large matrix entries.

For a fixed numerical proposal P31, define the exact local left inverse

    L(N)=[P31 V(N)]^-1 P31,      ell_3(N)=e_target^T L(N).
                                                                    (8)

Its invertibility is checked on the entire frequency domain. It need not equal the Moore-Penrose inverse; that equality is not required. Since L(N)V(N)=I, the two confluent annihilation identities and their derivative cancellation remain exact. Thus the same O(kappa) corrector argument holds. A verified center inverse followed by a Neumann radius bound is one explicit domain test; matrix-ball inversion provides another.

For a matrix inverse B=A^-1 with ||B||<=b, derivative majorants obey

    B_1<=b^2 A_1,
    B_2<=b^2 A_2+2b^3 A_1^2.                            (9)

Apply (9) to (8) and to the small coefficient-fitting systems, keeping the actual target row where possible rather than multiplying by a pessimistic full-left-inverse norm. Ordinary least squares remains an alternative off-model chart if its bounds are explicitly certified.

## 4. A bounded interval spectral certificate at the existing witness

For the original rational oblique witness and equal normalized wedge amplitudes, write its first-order record as B+kappa f. Under (2),

    J_s=kappa J0.

No finite kappa is selected in the following checks. With ell=11, the non-DC lag nodes have integer phase numerators +/-11,+/-77,+/-139 modulo400, so they are distinct and separated from DC. The finite 246-by-6 J0 has an interval-certified left inverse satisfying

    ||D0||inf < 0.367,
    ||D0||2 < 0.049,
    sigma_min(J0) > 1/0.049 > 20.4.                    (10)

Consequently D=(1/kappa)D0 is a left inverse for this formal first-order family. The reciprocal-kappa sensitivity is the expected signal scaling, without the large baseline term. This does not assume those witness frequencies or hardware values in the inverse algorithm; it certifies one admissible chart.

For the same original frequencies and all 200 timestamps, the exact normalized quadratic/confluent design admits the target row from (8) with

    sum_k |ell_3,k| < 1.002.                            (11)

This is a bound for complex-modulus infinity input. For independent real x/y errors of size epsilon, the complex error is at most sqrt(2)epsilon. For a particular real quadrature with coefficient a_k, the sharp real-coordinate row norm is sum_k(|Re a_k|+|Im a_k|). The division by U^2 in the normalized weak invariant is still present and still costs kappa^-2. Equation (11) does not remove that physical weak sensitivity.

The proof script uses floating matrices only as proposals and interval arithmetic for every assertion. If E=I-PV, ||E||inf=d<1 and the target defect row has l1 norm tau, the exact corrected target row satisfies

    ||ell||1 <= ||e_target^T P||1 + tau ||P||inf/(1-d).

For the witness the final bound is below 1.002. All final arithmetic is also outward enclosed. This verifies an exact alternative left-inverse row, not an unsupported bound on an arbitrary numerical pseudoinverse.

The script is proof_checks/certify_long_lag_spectral_margins.py. It certifies linear-algebra ingredients only, not optical remainder or bivariate curvature bounds and not a full nonlinear basin.

## 5. Finite operation bounds for the remaining maps

Every remaining ingredient is an explicit arithmetic circuit: the exact optical trace, its two Taylor records, the invariant ratios, the two beam equations, and downstream reconstruction. Values and first/second derivative norms can be enclosed by the following primitive rules. Let a have |a|<=V_a and derivative bounds a1,a2; let b have analogous bounds. Then

    (ab)1 <= V_a b1+V_b a1,
    (ab)2 <= V_a b2+V_b a2+2a1 b1.

If |a|>=m>0,

    (1/a)1 <= a1/m^2,
    (1/a)2 <= a2/m^2+2a1^2/m^3.

If a>=m>0,

    (sqrt(a))1 <= a1/(2sqrt(m)),
    (sqrt(a))2 <= a2/(2sqrt(m))+a1^2/(4m^(3/2)).

For real sin/cos inputs, the first bound is a1 and the second is a2+a1^2. Addition is componentwise. Matrix inverses use (9), and simple roots use (6). These rules also have standard componentwise versions, which are often much sharper than collapsing to a single scalar early.

All required lower bounds are explicit: Snell radicands, ray denominators, sequential traversal margins, source/hardware prior slacks, ellipse determinants, U, beam rho and B_perp, beta_i, alpha_3-1, f1, and the isolated bivariate root Jacobian. They are checked on the proposed domain, not inferred from a fitted point.

### Certifying the bivariate branch with two-dimensional work

Let K(t,u)=(R(t,u),W_chi(t,u)) be the two beam equations, where u denotes their extracted record features. At a proposed beam center t_c and feature center u_c, choose a numerical 2x2 inverse B. On a beam radius r_t and feature envelope U certify

    b_def >= ||B K(t_c,u_c)||,
    b_in >= sup_(u in U)||B[K(t_c,u)-K(t_c,u_c)]||,
    q_b >= sup_(t,u)||I-B K_t(t,u)|| < 1,
    b_def+b_in+q_b r_t <= r_t.                          (12)

The suprema here are replaced by operation bounds from the two-variable rational circuit and the known feature envelope; no high-dimensional root solve is introduced. Equation (12) supplies one real root for every input in U and keeps it on one branch. Physical and denominator guards are checked simultaneously.

Require this root box also to contain the beam coordinates of every potentially compatible theta in D_r. Then the selected A branch has its exact left-inverse identity for every compatible theta in D_r; a smooth but wrong formal beam branch would not suffice.

If m bounds ||K_t^-1|| and K_u,K_tu,K_tt,K_uu have bounded norms, implicit differentiation gives

    t_u <= m K_u,
    t_uu <= m[K_uu+2K_tu t_u+K_tt t_u^2].               (13)

Compose these with explicit downstream reconstruction derivatives. This yields bounds for A_s,A_w,A_ss,A_sw,A_ww, with native output scaling S^-1 included.

### Preserving small-wedge remainder orders without cancellation loss

Direct interval subtraction F-Fi is valid but can lose the useful kappa powers. A sharper option differentiates the exact model in an auxiliary wedge-scaling variable lambda, solely inside the evaluator:

    R1 = integral_0^1 (1-lambda) partial_lambda^2 F(lambda e) d lambda,
    R2 = integral_0^1 [(1-lambda)^2/2] partial_lambda^3 F(lambda e) d lambda.
                                                                    (14)

Differentiate these identities in the scaled native parameter directions. Thus a uniform second-lambda-derivative bound contributes a factor 1/2, and a third-lambda-derivative bound contributes 1/6, for each value/first/second parameter derivative. Each lambda derivative retains a factor e. After parameter differentiation, the advertised kappa powers additionally require the wedge rows of the native scale S to be O(kappa), or the explicit e=kappa a chart. Unscaled native wedge derivatives can lose one kappa power per wedge derivative. The integral compiler and the finite certificate remain valid for arbitrary fixed S without claiming those powers.

A univariate lambda jet of order three, with second parameter jets, computes these bounds in a constant multiple of the forward circuit cost. It does not require a full eighteen-variable fifth-order tensor. Root/denominator guards must hold along lambda in [0,1] for this integral implementation. If they cannot be certified, use direct exact-remainder bounds or leave the region unresolved. No observed temporal derivative is taken.

## 6. Explicit domain envelope and derivative constants

Fix a proposed native radius r_bar and independent record allowance epsilon_bar. First certify the physical domain D_rbar. Let

    Bi >= sup_D ||D_theta Ri S||,
    Ci >= sup_D ||D_theta^2 Ri[S.,S.]||,   i=1,2.

At the center put s0=y-R1(theta0), w0=y-R2(theta0). For all ||x||<=r_bar, ||h||<=epsilon_bar, the corrected inputs obey

    ||s-s0||<=B1 r_bar+epsilon_bar,
    ||w-w0||<=B2 r_bar+epsilon_bar.                     (15)

Certify A on this product envelope, or on a sharper coupled enclosure retaining the common x,h dependencies. The latter can be substantially less conservative. In particular, apply (5), root disks, matrix inverse guards, and (12) on the actual selected envelope.

Let as,aw,ass,asw,aww bound the corresponding derivatives of S^-1 A on that envelope in real-record infinity norms. Then explicit sufficient Hessian majorants for Psi are

    Mxx = ass B1^2+2asw B1B2+aww B2^2+as C1+aw C2,
    Mxh = (ass+asw)B1+(asw+aww)B2,
    Mhh = ass+2asw+aww.                                 (16)

Direct componentwise/coupled differentiation can replace these sums with sharper bounds. The signs in the first derivative do not affect (16). The center derivative matrices must retain their actual sums so cancellation is not discarded unnecessarily.

Compute and certify, using enclosed roots and linear solves,

    Y >= ||Psi(0,0)||inf,
    Z >= ||D_x Psi(0,0)||inf,
    L >= ||D_h Psi(0,0)||inf.                           (17)

In particular

    D_x Psi(0,0)=-S^-1[A_s DR1+A_w DR2]S,
    D_h Psi(0,0)= S^-1(A_s+A_w).

These are evaluated at the actual candidate's corrected pair. A floating value is a proposal; the certificate uses upper enclosures. This is a finite collection of arithmetic/matrix operations, not an unevaluated optimization supremum. Wide interval dependence can still make it fail.

## 7. The scalar radius/noise theorem

Assume the domain and branch checks above hold on (r_bar,epsilon_bar), so all constants (16)-(17) are fixed certified numbers. For any 0<r<=r_bar and 0<=epsilon<=epsilon_bar define

    P(r,epsilon)=Y+L epsilon+(Mhh/2)epsilon^2
                 +(Z+Mxh epsilon)r+(Mxx/2)r^2-r,

    q(r,epsilon)=Z+Mxx r+Mxh epsilon.                   (18)

If

    P(r,epsilon)<=0,       q(r,epsilon)<1,               (19)

then for every real record h with ||h||inf<=epsilon, T_(y+h) maps D_r into itself and is a contraction there. It has one fixed point theta_hat(h), and the explicit algebraic iteration converges to it from every start in D_r.

Proof: Taylor's theorem along (x,h) gives the self-map bound in P. The mean-value theorem gives ||D_x Psi(x,h)||<=q. Banach's theorem then applies. Every exact physical solution in D_r of F(theta)=y+h equals theta_hat(h), because the selected branch has the exact left-inverse identity. Existence of a fixed point alone does not imply compatibility with all 400 observations.

The theorem imposes no relation epsilon=kappa^p. The actual numerical constants decide the independent noise allowance.

### Explicit allowable epsilon

For a fixed r set

    C=r-Y-Zr-(Mxx/2)r^2,
    B=L+Mxh r.

If C>0 and Mhh>0, the self-map condition holds for

    epsilon <= 2C/[B+sqrt(B^2+2Mhh C)].                 (20)

If Mhh=0 and B>0, use epsilon<=C/B. If both are zero, the self-map test imposes no noise restriction, but the domain and contraction tests still do. Also impose epsilon<=epsilon_bar and, when Mxh>0,

    epsilon < (1-Z-Mxx r)/Mxh.

If C<0, this radius permits no noise level. If C=0, it permits only epsilon=0 unless B=Mhh=0, in which case domain and contraction still decide the allowance. The constants must already be certified on the chosen epsilon_bar domain; (20) cannot be used with constants that silently change with the resulting epsilon. Alternatively propose a pair (r,epsilon), regenerate its finite bounds, and verify (19).

For epsilon=0 and Mxx>0, an explicit radius interval exists when

    Z<1,       (1-Z)^2>2Mxx Y.

Its lower self-map root is

    r_minus=2Y/[1-Z+sqrt((1-Z)^2-2Mxx Y)].              (21)

Any positive r with r_minus<=r<(1-Z)/Mxx, inside the certified domain, passes strict contraction and the self-map test. The upper endpoint is excluded because it gives q=1. Handle Y=0 by choosing a positive radius inside that interval, and Mxx=0 by retaining Z<1 and the linear inequality Y<=(1-Z)r. No wedge or measurement scale is selected merely to force a pass.

## 8. Compatible-set noise radius and every residual

Let theta_hat(0) be the nominal fixed point. Put q0=Z+Mxx r<1. Comparing nominal and perturbed maps gives

    ||S^-1[theta_hat(h)-theta_hat(0)]||
      <= [(L+Mxh r)||h||+(Mhh/2)||h||^2]/(1-q0).        (22)

This retains the common-record correlation. Native coordinate i is bounded by row i of |S| times this scaled radius, or by sharper componentwise bounds.

Every physical theta in D_r with ||F(theta)-y||inf<=epsilon is the fixed point for h=F(theta)-y, so (22) encloses all such systems. The nominal fixed point need not itself satisfy the observation strips. Intersect the output tube with all 400 exact constraints and all original prior/branch/traversal conditions. Neither local convergence nor a small selected-feature residual excludes remote compatible branches.

If the original candidate comes from a minimum-residual frequency lift, the chart coverage requirement remains: other not-excluded lifts, prism assignments, bivariate roots, singular strata, and boundary alternatives must be retained. The radius certificate is local, not a replacement for a complete original-prior cover.

## 9. Optional exact thirteen-coordinate composition

The audited sagittal construction supplies a local exact lift E_y(x13): reconstruct n3^2=-b/a from a first subresultant and then four affine geometry coordinates from a fixed nonsingular pivot. See sagittal_corrector_composition.md. On its guarded physical chart, every exact compatible theta is E_y(pi13 theta).

Define

    Phi_y(x)=pi13 T_y(E_y(x)).                           (23)

This is an actual thirteen-coordinate iteration with a bivariate algebraic correction step. It adds subresultant-denominator and geometry-pivot guards. Its center derivatives are

    Phi_x=pi13 T_theta E_x,
    Phi_y=pi13[T_y+T_theta E_y].                         (24)

The second term in Phi_y is essential: the lift depends on the raw first-six-pair record. Reusing A_s+A_w alone would omit this sensitivity.

Its Hessians obey the explicit chain rules

    Phi_xx=pi13[T_thetatheta(E_x,E_x)+T_theta E_xx],
    Phi_xy=pi13[T_thetatheta(E_x,E_y)+T_thetay E_x+T_theta E_xy],
    Phi_yy=pi13[T_yy+2T_thetay E_y
                 +T_thetatheta(E_y,E_y)+T_theta E_yy].  (25)

In the last line of (25), 2T_thetay E_y is symmetric bilinear shorthand for T_thetay(E_y h1,h2)+T_thetay(E_y h2,h1); its norm bound has the displayed factor two. Apply the same scalar theorem to scaled norms of (24)-(25). E and its derivatives use quotient, square-root, and fixed linear-system rules, so this does not introduce another nonlinear optimization. Small subresultant or pivot margins can make these constants large; dimension reduction does not prove improved conditioning. For full18 uncertainty, also propagate the output lift: ||E_(y+h)(x_hat(h))-E_y(x_hat(0))|| <= E_x_bound ||x_hat(h)-x_hat(0)||+E_y_bound ||h|| in the stated native-scaled norms. The reduced fixed-point radius alone bounds only the retained coordinates. Off-image and noisy fixed points still require full-record validation.

## 10. What this establishes and the remaining decisive check

Established here:

- Explicit finite Prony radius/derivative formulas after DC cancellation
- Finite speed-alias handling compatible with the full original record
- A verified, well-conditioned long-lag witness chart and nearly unit-norm weak extraction row
- Arithmetic recipes for every required root, reconstruction, optical, and remainder constant
- A closed scalar radius/noise inequality and explicit independent measurement allowance
- Correct transfer to the optional exact thirteen-coordinate lift, including its extra data sensitivity

Still required before claiming a finite numerical convergence regime: evaluate the optical and bivariate derivative envelopes, full domain guards, and (18)-(19) for a candidate and a declared record/noise budget. The current spectral proof does not supply those missing values. If those bounds force an impractically small region, the correct outcome is a quantitative conditioning diagnosis, not an arbitrary tiny wedge chosen after the fact.

The next single proof certificate, if undertaken, should report the candidate-generating procedure, all retained spectral/alias/root choices, native scale S, r, independent epsilon, Y,Z,L,Mxx,Mxh,Mhh, every physical/root margin, and all 400 residual bounds. That would certify one finite-angle basin, not global completeness.

## Audit status

Independent audit passed the finite theorem, frozen inverse charts, alias handling, operation bounds, and thirteen-coordinate composition; see quantitative_radii_independent_audit.md. The bounded interval spectral script has passed its assertions; its scope is explicitly limited to the linear-algebra ingredients. No parameter sweep, hidden truth initialization, or numerical nonlinear convergence claim has been made.
