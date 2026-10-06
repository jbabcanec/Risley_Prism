# Algebraic defect correction for the exact full18 Risley inverse

## Result and limits

There is a model-specific exact correction scheme whose nonlinear candidate work is a scalar degree-seven polynomial and the existing four-variable last-prism polynomial system. It never defines a seventeen-variable strong inverse or asks an optimizer to solve the fourteen optical coordinates jointly.

The construction uses an explicit off-model map A of **two** ordinary 200-sample paired records. On its regular branches,

    A(F1(theta), F2(theta)) = theta,                         (1)

where F1 and F2 are the actual full-vector Taylor records through wedge degrees one and two, respectively. They include the baseline and all terms up to their stated degree. For a measured record y define

    R1 = F - F1,       R2 = F - F2,
    T_y(theta) = A(y - R1(theta), y - R2(theta)).             (2)

Here F is the exact finite-angle physical trace, with all intersections and all 200 original samples. Every exact physical solution is a fixed point of (2) on the corresponding branch. A finite, native-scaled contraction certificate below turns (2) into an exact inverse whenever a compatible solution lies in the certified region. All eighteen unknowns, including three speeds, are updated.

The new structural point is the construction of A, its extension away from a Taylor-model image, and a confluent demixer that makes its weak channel insensitive to first-order frequency errors. On a compact regular family with e_i = epsilon a_i and nonzero bounded a_i, the derivative of (2) in correctly scaled coordinates is O(epsilon). Thus this particular algebraic correction, not just an unspecified local inverse, converges for a nonempty original-prior regime. A direct finite certificate can also establish convergence at any particular finite wedge size where it passes.

A stronger true 17+1 variant in two_variable_algebraic_correction.md reduces the multivariate algebraic root step to two beam variables and retains only one weak quadratic direction. The present four-variable construction remains a useful independent chart.

There is **no** theorem here that the certificate passes throughout the original +/-18 degree wedge prior, no numerical non-infinitesimal wedge threshold, no global root count for the exact model, and no exclusion of remote physical branches. A fixed point of the off-model retraction need not fit all 400 observations; full-record validation remains mandatory unless an exact compatible solution in the region is independently assured. These qualifications are part of the result.

## 1. Exact model, coordinates, and Taylor records

The native bounds and timestamps are unchanged: signed speeds in [-3.5,3.5] Hz; signed wedge angles and initial phases in [-18,18] degrees; indices in [1.3,1.8]; d in [50,200], g in [2,15]; beam angles in [-25,25] degrees; source offsets in [-5,5]^2; source distance six and each prism axial reference thickness three. Samples are k/20 for k=0,...,199. Internal angular calculations use radians, with explicit conversions when differentiating native coordinates.

Write e_i = sin(a_i^native), gamma_i(k) = 2 pi N_i k/20 + phi_i, and v_i(k) = (cos gamma_i(k), sin gamma_i(k)). The exact tilted-plane slope is

    u_i(k) = e_i v_i(k) / sqrt(1-e_i^2).

For fixed non-wedge parameters and fixed e, scale every e_i by lambda and Taylor-expand the exact paired trace at lambda=0. Define

    F(lambda e) = F0 + lambda P1 + lambda^2 P2 + O(lambda^3),
    F1(theta) = F0 + P1,       F2(theta) = F0 + P1 + P2.     (3)

In particular F0=B, and the first-order coefficient matrices are the actual M_i of oblique_inverse.md. P2 includes its physical DC shift, all three self-second harmonics, and all mixed degree-two harmonics. F2 is not obtained by retaining only the selected third-prism harmonic.

### A finite arithmetic compiler for F1 and F2

No temporal derivatives or Taylor coefficients are measured. F1 and F2 are evaluated from each trial parameter vector by running the actual three-prism forward recurrence in the truncated ring R[lambda]/(lambda^3). Replace u_i by lambda e_i v_i: its omitted plane-slope correction starts at lambda^3. For a scalar ring element A=A0+A1 lambda+A2 lambda^2 with A0&gt;0,

    sqrt(A) = sqrt(A0) + A1/(2 sqrt(A0)) lambda
              + [A2/(2 sqrt(A0)) - A1^2/(8 A0^(3/2))] lambda^2,

    1/A = 1/A0 - A1/A0^2 lambda
          + [A1^2/A0^3 - A2/A0^2] lambda^2.                  (4)

Products are truncated after degree two. The zero-wedge Snell roots and position denominators are strictly positive under the original bounds. Thus the compiler is unambiguous. It preserves finite thickness, source offset, tilted-plane intersections, and the two gaps. Evaluation costs a constant multiple of one 200-sample exact trace. The coefficient identities in the audited oblique construction are exactly the coefficients of this compiler.

For finite certificate purposes R1 and R2 are simply the differences in (2); no asymptotic estimate is substituted for these exact functions.

## 2. Regular branch conditions

The following conditions describe a chart, not additional known parameters:

1. All three wedge sines are nonzero, all speeds are nonzero, and the 25 degree-at-most-two sampled nodes are distinct. A positive separation margin is retained on a compact branch.
2. The stacked first-order seven-tone Hankel matrix has rank seven. This includes a nonzero joint DC tone and nonzero joint coefficients for all six fundamental nodes.
3. The first-order ellipse matrices have nonzero determinants. Their signs determine speed orientation because det(M_i)&gt;0 on the audited original hardware/source box.
4. The beam has T=|t|&gt;0 and B_perp != 0 in the beam frame. All positive denominators in the triangular reconstruction have certified nonzero margins.
5. The selected four-variable last-prism polynomial root is real and simple. The chosen local inverse branch and its root isolation are retained explicitly.
6. Every iterate in the certified parameter region is inside the original native bounds and strict physical sampled-time branch, including sequential surface traversal. A region touching a prior face can be treated with a different shape; the box certificate below is simplest for an interior chart.

Zero wedges, zero/aliased/colliding frequencies, rank-deficient Hankel records, axial or B_perp=0 charts, exceptional polynomial fibers, and critical physical branches are not rejected as impossible systems. They are outside this correction chart and remain in the complete inverse backend or in an explicitly unresolved set.

All possible rotor-to-prism assignments and all relevant simple real last-prism roots must initially be retained. Physical order is not a permutation symmetry.

## 3. An explicit off-model frequency and first-order coefficient map

Let s be an arbitrary real paired record near a regular first-order record. It need not be a sum of seven tones.

For h in {x,y} and k=0,...,192 form

    H_s[(h,k),j] = s_h,k+j,     j=0,...,6,
    v_s[(h,k)] = s_h,k+7.

When H_s has full column rank define the rational least-squares recurrence

    c(s) = -(H_s^T H_s)^(-1) H_s^T v_s,
    p_s(z) = z^7 + sum_(j=0)^6 c_j(s) z^j.                 (5)

This is an off-model definition, not a claim that p_s annihilates s exactly. It may be evaluated by a numerically preferable QR solve, provided the exact map or its error enclosure is preserved.

At an exact regular F1 record the roots are 1 and three conjugate pairs. On a sufficiently small real record neighborhood there is one real root near 1 and three simple nonreal conjugate pairs on separated disks. Ignore the DC-near root for rotor recovery. For each root z on a selected positive-frequency disk, put

    zeta = z / |z|,
    nu = 20 Arg(zeta)/(2 pi),      0 &lt; nu &lt;= 3.5.          (6)

Radial projection and the chosen argument branch are real analytic on those disks. Root projection is deliberate: arbitrary off-model recurrence roots need not be on the unit circle. The original speed bound lies strictly below Nyquist, so no extra sampling alias is introduced by the principal arc. A projected trial outside a proposed branch's speed interval is a branch-domain failure, not by itself a global exclusion.

Fit s by ordinary real least squares on the seven-column design

    1, cos(2 pi nu_i k/20), sin(2 pi nu_i k/20), i=1,2,3.

This gives a baseline B and three 2x2 matrices E_i, with cosine and sine columns. Recover the signed speed by sigma_i=sign(det E_i), let N_i=sigma_i nu_i, and put

    C_i = E_i diag(1,sigma_i).                            (7)

For each retained physical prism assignment, C_i is now in the signed-frequency convention used in the oblique formulas.

On an exact first-order record, (5)-(7) recover the actual speeds, baseline, and all coefficient matrices. This follows from full Hankel column rank, distinct seven nodes, and exact linear independence of the seven-column design. On an ordinary perturbed record, they still define an explicit analytic map on the selected root/sign/label branch.

A complete data-derived initial cover can use the audited Hankel perturbation and Rouche bounds in oblique_record_certificate.md. A center root or a successful least-squares fit alone is not such a cover.

## 4. Confluent extraction of the weak quadratic channel

Use the signed sampled nodes z_i = exp(2 pi i N_i/20) produced by s. Let

    M2 = {m in Z^3 : |m|_1 &lt;= 2},     |M2|=25,
    z^m = z_1^(m_1) z_2^(m_2) z_3^(m_3).

Form the 200x31 complex design V(N) with columns

    (z^m)^k,                 m in M2,
    k z_i^k, k z_i^(-k),     i=1,2,3.                    (8)

The 25 nodes are distinct; the six fundamental nodes have multiplicity two. The confluent Vandermonde theorem gives full column rank on the original samples. There is ample room: 31 &lt;= 200. A real sine/cosine implementation is equivalent and has 31 real basis columns per real output.

Let L(N)=(V*V)^(-1)V*, and let ell_3(N) be its row for the m=2e_3 column. For the second arbitrary paired input record w, use its complex trace w_x+i w_y and set

    C_plus2 = ell_3(N) (w_x+i w_y).                      (9)

On w=F2(theta) at matching N, this is exactly the physical third self-second harmonic. It is uncontaminated by the physical quadratic DC and every other degree-two term.

### The cancellation that makes the correction useful

Let V1(N) be the seven fundamental/DC columns. Then

    ell_3 V1 = 0,        ell_3 partial_(N_j) V1 = 0.

The second identity follows because each frequency derivative is a scalar multiple of one of the six confluent columns. Differentiate the first identity:

    (partial_(N_j) ell_3) V1 = 0.                        (10)

Thus changing a frequency used by the demixer has **zero first-order effect** when the row is applied to a matched first-order record. This cancellation is not available with the ordinary 25-column quadratic demixer.

More quantitatively, on a compact separated-frequency domain and for an F1 record with fundamental amplitudes O(epsilon),

    |ell_3(N+Delta) F1(N)| &lt;= C epsilon |Delta|^2,
    |D_N ell_3(N+Delta) F1(N)| &lt;= C epsilon |Delta|.      (11)

These follow from (10), Taylor's theorem, and bounded first/second demixer derivatives on the separated domain. The DC contribution vanishes identically for all N. No time derivative of the observed record is used; the columns in (8) are known functions evaluated at the original timestamps.

## 5. The four-variable algebraic reconstruction and its off-image extension

Write the two complex column entries of C_3 as a,b and define the signed fundamental coefficients

    U=(a-i b)/2,       V=(a+i b)/2,
    w_ratio=V/conjugate(U),       I=C_plus2/U^2.          (12)

The symbol w_ratio is an invariant; it is distinct from the second input record w. All divisions require certified U != 0.

With B supplied by the first input s, solve the existing four real polynomial equations for (t_x,t_y,H,d). In complex notation T=t_x+i t_y and R=T conjugate(T), set

    A0 = 2 H d + d(H+1)R - T conjugate(B),
    V0 = d(H+1)T^2 - T B,
    C0 = 4 d H^3 conjugate(T) + d H^2(3H-1)R conjugate(T)
         - 4 H^2(conjugate(B)-d conjugate(T))
         - 2(H^2+1)R(conjugate(B)-d conjugate(T)).

The equations are

    Re(V0 - w_ratio conjugate(A0)) = 0,
    Im(V0 - w_ratio conjugate(A0)) = 0,
    Re(C0 - 2 I(H-1) A0^2) = 0,
    Im(C0 - 2 I(H-1) A0^2) = 0.                        (13)

Their degree bounds remain (4,4,9,9). Retain H&gt;1 and A0!=0. On the specified simple branch these four equations define a locally unique real-analytic root. For a finite input neighborhood, root existence and uniqueness must be certified uniformly, for example by a four-variable parameterized interval-Newton inclusion, or by complete initial root enumeration and discriminant-free continuation. This is a four-variable algebraic certification, not a hidden high-dimensional optical solve. Denominator components and other root branches are handled exactly as in the audited last-prism construction.

For the chosen T,H,d root, set h3=H. Rotate each S_i=C_i C_i^T and B into the inferred beam frame. With Tmag=|t|,

    c_i=(S_i)12/(S_i)22,
    r_i=sqrt(det S_i)/(S_i)22,
    beta_i=-c_i/(Tmag B_perp),
    alpha_i=[r_i-1-(B_parallel/B_perp)c_i]/Tmag^2.

Put E3=3(h3^(-3)-h3^(-1)). Reconstruct

    L2 = positive root of beta2 L2^2 - (alpha2-1)L2 + E3=0,
    h2 = 1/(beta2 L2),
    g = L2 - d - 3/h3,
    L1 = d + 2g + 3/h2 + 3/h3,
    h1 = 1/(beta1 L1).                                   (14)

At exact compatible inputs beta_i&gt;0 and E3&lt;0, so the quadratic has exactly one positive root. Its explicit radical formula or any algebraically equivalent stable evaluation suffices.

Then

    n_i=sqrt((h_i^2+|t|^2)/(1+|t|^2)),
    b=B-t[6+2g+d+3 sum_i 1/h_i].                         (15)

The first-prism scalar shape consistency equation is **not imposed in defining A away from the model image**. Nor is it asserted that every off-image C_i equals e_i M_i R(phi_i). Instead compute the actual first-order M_i at the reconstructed optical/hardware parameters and put X_i=M_i^(-1) C_i. Take its conformal projection:

    x_i = ((X_i)11+(X_i)22)/2,
    y_i = ((X_i)21-(X_i)12)/2,
    phi_i = atan(y_i/x_i) in (-pi/2,pi/2),
    e_i = x_i sqrt(1+(y_i/x_i)^2).                        (16)

The original phase interval lies inside (-pi/2,pi/2), and on an exact model x_i=e_i cos(phi_i) is nonzero. Equation (16) is consequently analytic on the selected signed-wedge branch and returns exactly the signed wedge sine and phase on the model image. Convert e_i to its principal arcsine and the beam slopes to the native beam angles, with the stated degree conversions.

Equations (5)-(16), a root/sign/label choice, and the explicit formulas define A(s,w) on an **open set of pairs of ordinary records**. Corrected records need not satisfy any exact Prony recurrence, first-order consistency, or scaled-rotation equation.

## 6. Proof of the left-inverse identity

For s=F1(theta), the full-rank exact seven-tone recurrence and coefficient fit recover N,B,C_i. For w=F2(theta), the 31-column demixer extracts exactly e_3^2 exp(2i phi_3) C_(+2), while the fundamental coefficients in (12) equal e_3 exp(i phi_3) U_+ and e_3 exp(-i phi_3) V_-. Thus the invariant ratios in (12) are exactly those of the audited last-prism algebraic map.

The matching simple root of (13) returns t,h3,d. Equations (14) return h2,g,h1; (15) returns n,b. Finally X_i=e_i R(phi_i), so (16) returns the native wedges and phases. The speeds have already been recovered. This proves (1) for all eighteen coordinates.

The identity is branchwise. If a different simple polynomial root reconstructs another formal candidate, it defines another A branch; it is not silently identified with the true branch. The redundant shape consistency and all original constraints can screen Taylor-model candidates and validate final exact candidates, but they are not assumed to hold at arbitrary intermediate records.

## 7. Exact finite-angle iteration and its algebraic cost

Given a branch candidate theta_0, perform

    r_m = y - F(theta_m),
    s_m = F1(theta_m) + r_m,
    w_m = F2(theta_m) + r_m,
    theta_(m+1) = A(s_m,w_m).                              (17)

The same full-record residual is supplied to both Taylor orders. Only one exact ray trace and one order-two Taylor trace are needed per step. The nonlinear algebraic tasks are:

- roots of one real degree-seven polynomial, with the separated disk labels retained;
- the four real last-prism equations (13), with the selected simple root branch retained;
- one scalar positive quadratic and elementary radicals/angle conversions.

All other operations are fixed-size or 200-sample linear algebra. There is no solve of a seventeen-dimensional strong system, no implicit known-frequency assumption, and no free physical coefficient pencil left to fit.

This is not a fixed-dimensional linear shift model for the exact optics. The changing exact nonlinear remainder is evaluated afresh at every step. Thus the moving-branch-point/Hankel obstruction in exact_shift_module_obstruction.md does not apply to this construction.

A practical initialization is to enumerate the regular branches of A(y,y). Their quality is established below in a perturbative regime, not assumed. A point estimate violating a prior is not alone an exclusion when its initializer has bounded truncation error; use a candidate enclosure or retain that branch as unresolved. For wider priors, the audited data-derived outer coefficient cover can be used to construct branch regions before correction.

## 8. Why the correction derivative is small: a graded theorem

Fix a compact family of non-wedge coordinates and nonzero normalized wedge amplitudes a_i, with all regular conditions in Section 2 uniformly strict. Write e_i=epsilon a_i and use the chart

    eta=(N_i,a_i,phi_i,n_i,t_x,t_y,b_x,b_y,g,d).

Angle units can be consistently converted; eta is used here only to state asymptotic powers. Let Theta_epsilon convert eta to the original eighteen native coordinates. Keep the family on one simple root/sign/label chart. Since exact optics are analytic around zero wedge with strict margins,

    ||D_eta R1|| &lt;= C1 epsilon^2,
    ||D_eta R2|| &lt;= C2 epsilon^3.                         (18)

These are not unscaled native wedge-derivative claims. An unscaled native wedge column has one less power of epsilon.

Consider the coupled input tube around a compatible reference eta_bar:

    ||s-F1(Theta_epsilon eta_bar)|| &lt;= Cs epsilon^2,
    ||w-F2(Theta_epsilon eta_bar)|| &lt;= Cw epsilon^3.       (19)

For all sufficiently small epsilon this tube lies in the chosen off-image branch. Let A_eta=Theta_epsilon^(-1) A. Uniformly on (19),

    ||D_s A_eta|| &lt;= K1/epsilon,
    ||D_w A_eta|| &lt;= K2/epsilon^2.                       (20)

Here is the required structure behind (20):

1. The six non-DC singular directions of the stacked first-order Hankel matrix are O(epsilon), while the DC direction is O(1). With uniformly separated nodes and nonzero modal amplitudes, its inverse sensitivity gives D_s N=O(epsilon^-1). The least-squares recurrence is evaluated at a residual O(epsilon^2); its additional normal-equation derivative term stays bounded and does not worsen that power.
2. First-order coefficient derivatives with respect to s are O(1): their frequency dependence is multiplied by O(epsilon) amplitudes. Ellipse shape invariants and normalized wedge amplitudes therefore cost O(epsilon^-1).
3. U is bounded below by c epsilon, and C_plus2 is O(epsilon^2). Derivatives of the normalization C_plus2/U^2 through U cost O(epsilon^-1).
4. The frequencies obtained from s differ from the reference by O(epsilon). Equations (10)-(11) show that D_N ell_3 applied to the reference first-order record is then O(epsilon^2). Applied to its quadratic part it is also O(epsilon^2); the O(epsilon^3) second-input perturbation is smaller. Consequently differentiating the normalized quadratic invariant through N(s) costs O(epsilon^-1), not O(epsilon^-2).
5. Direct perturbation of w costs O(epsilon^-2), and every remaining simple-root/triangular reconstruction derivative in normalized invariant variables is uniformly bounded on the compact regular chart.

Without the six confluent columns in (8), point 4 fails in general and D_s A_eta can be O(epsilon^-2). Then the R1 derivative in (18) would only give an O(1) update derivative. This is the substantive frequency correction in the construction.

Combining (18)-(20) yields

    ||D_eta[Theta_epsilon^(-1) T_y Theta_epsilon]||
       &lt;= (K1 C1 + K2 C2) epsilon                       (21)

whenever the evaluated input pairs remain in the coupled tube.

For an exact record y=F(Theta_epsilon eta_true), this last premise follows on any fixed bounded regular parameter neighborhood containing eta_true: for every current eta in that neighborhood,

    s(eta)-F1(eta_true)=R1(eta_true)-R1(eta)=O(epsilon^2),
    w(eta)-F2(eta_true)=R2(eta_true)-R2(eta)=O(epsilon^3).  (22)

The unknown eta_true is used only in the proof of a regime theorem; it is not supplied to the algorithm. Likewise A(y,y)=eta_true+O(epsilon) in the eta chart, by integrating (20) along the coupled input segment. Thus each uniformly interior compact regular branch has a positive epsilon threshold on which its matching algebraic initializer and iteration converge to the true system.

This is a proved nonempty original-prior regime. The already audited witness (t=(1/3,0), b=(1,2), h=(3/2,7/5,5/3), g=3,d=100, phases zero, N=(1,7,49)/20) satisfies the non-wedge regularity and the four-polynomial determinant condition; its degree-two nodes and six repeated fundamental nodes are distinct as required. Small nonzero wedge sines give an interior physical family. This is a proof witness, not initialization information. No numerical threshold or general 18-degree conclusion is extracted from the O(epsilon) statement.

## 9. A finite certificate in original native coordinates

The following certificate does not use asymptotic powers. Let S be any specified positive diagonal scale matrix in the original eighteen native coordinates, and let

    D = {theta : ||S^(-1)(theta-theta0)||_infinity &lt;= r}.

Require D to lie inside the original prior and the strict physical branch. Also certify that every pair

    (s(theta),w(theta))=(y-R1(theta),y-R2(theta)), theta in D,

belongs to the selected domain of A: all Hankel/design ranks, root disks, signs, denominators, and the simple real four-variable root branch are uniformly valid. No high-dimensional root equation is part of that A-domain check; parameterized bounds may nevertheless be expensive.

At each pair use A_s=D_s A and A_w=D_w A in original native output coordinates. Define any verified upper bound

    q &gt;= sup_(theta in D)
         ||S^(-1)[A_s D_theta R1 + A_w D_theta R2] S||_infinity,

    delta &gt;= ||S^(-1)(T_y(theta0)-theta0)||_infinity.     (23)

All chain rules, including sin of native degree wedges, arcsines, and frequency argument conversions, are included. It is permissible to bound the two summands separately; retaining their actual sum can sharpen q.

If

    q &lt; 1,             delta + q r &lt;= r,                (24)

then T_y maps D into itself and is a contraction. It has exactly one fixed point theta_hat in D, and the explicit iteration (17) converges to it from every point of D. A useful a posteriori bound is

    ||S^(-1)(theta_hat-theta_m)||_infinity
       &lt;= ||S^(-1)(theta_(m+1)-theta_m)||_infinity/(1-q). (25)

Every exact physical solution of the full 400-coordinate record inside D is a fixed point by (1), so there is at most one such solution and, if one exists, it equals theta_hat.

### Computability of the derivative bound

No numerical differentiation is required. Differentiate the exact ray and truncated-ring recurrences, the linear solves, and the elementary formulas. For a simple degree-seven root,

    dz = -[sum_(j=0)^6 z^j dc_j]/p_s'(z).

For zeta=z/|z|,

    d zeta = [dz-zeta Re(conjugate(zeta) dz)]/|z|.

For the four-polynomial root x=(t_x,t_y,H,d), differentiate (13) as

    dx = -J_x^(-1) J_input d(input).

A verified inverse or adjugate/determinant enclosure is four-dimensional. These formulas give an explicit arithmetic circuit for (23). Interval/affine/Taylor enclosures may certify it on D. This proof obligation is materially different from solving a seventeen-variable implicit strong map, although a wide-domain enclosure can still be difficult.

The certificate is allowed to fail. Failure of (24), root-branch separation, or physical margins leaves the region unresolved. It does not prove that no physical solution exists.

## 10. Shared-error noise enclosure and full-record validation

Suppose measured y has componentwise error at most eta_obs in all 400 real coordinates. Let theta_star in D satisfy

    ||F(theta_star)-y||_infinity &lt;= eta_obs.

Besides (23)-(24), require A to remain on the same branch along the simultaneous-error segments from (F1(theta_star),F2(theta_star)) to (F1(theta_star)-e,F2(theta_star)-e) for every compatible theta_star and |e|&lt;=eta_obs. A conservative box enclosing these segments suffices. Define

    L &gt;= sup ||S^(-1)(A_s+A_w)||_(infinity &lt;- infinity)  (26)

on that set. The sum is essential because the same record error enters both arguments; replacing it by the sum of the separate norms is safe but less sharp.

The left-inverse identity and the mean-value theorem give

    ||S^(-1)(T_y(theta_star)-theta_star)|| &lt;= L eta_obs.

Together with the contraction inequality,

    ||S^(-1)(theta_hat-theta_star)||
        &lt;= L eta_obs/(1-q).                             (27)

Hence the native coordinate-i error of the fixed point relative to any compatible solution in D is at most S_ii L eta_obs/(1-q). Coordinatewise noise can instead be propagated with an entrywise bound for S^(-1)(A_s+A_w). The weak-channel contribution generally scales as epsilon^-2; no stronger noise claim is made.

The O(epsilon^2)/O(epsilon^3) coupled tube used in the asymptotic proof requires noise fitting those widths; a larger finite noise budget needs the separately certified branch-domain envelope in this section. Also, this four-variable construction discards one first-order consistency and uses both real components of I. It is a 16+2 information selection and can introduce two conservative weak-noise directions; the two-variable addendum preserves 17+1.

Neither (24) nor (27) asserts that theta_hat itself is compatible with all 400 measured strips. The off-model map discards redundant information. Intersect the resulting parameter tube with every exact observation constraint and original prior/branch/traversal inequality, using the exact affine geometry profile where helpful. For noiseless data, the existence of any exact compatible solution in D identifies the fixed point as that solution; otherwise its full residual must be independently validated.

## 11. What is materially reduced, and what remains global

This construction adds a missing executable correction layer to the formal oblique initializer:

- Both unknown-frequency and optical updates are specified by finite algebraic formulas on ordinary perturbed records.
- Each step uses at most a four-variable nonlinear algebraic system rather than an implicit seventeen-variable correction.
- The actual twenty-hertz finite samples support the frequency-confluent cancellation needed to make the update contractive in a regular small-wedge regime.
- A finite native-scaled residual/derivative certificate can validate a particular finite-angle basin; a noisy compatible-set bound follows without pretending the data are exact.

The iteration still updates eighteen parameters. Its derivative certificate can require eighteen-dimensional enclosures, and finding a complete initial region cover can remain hard. The four-variable polynomial degree bound is not a small runtime guarantee. This is a reduction of the nonlinear correction solve, not a reduction of the physical parameter count or a complete all-prior solver.

For a global inverse, every original-prior region must still be covered by a validated branch, excluded by a sound exact certificate, or returned unresolved. Degenerate spectra, exceptional four-polynomial fibers, critical boundaries, and remote compatible sheets survive. The compactification and global exclusion tools already established remain relevant. No amount of successful local correction alone proves global uniqueness.

## 12. Audit status

Independent mathematical audit passed the off-model map, scaled coupled-tube argument, finite contraction certificate, and shared-noise enclosure; see algebraic_defect_independent_audit.md. It emphasized the 16+2 conditioning qualification and the difference between an eighteen-coordinate algebraically preconditioned iteration and a literal four-dimensional elimination of the exact inverse. No parameter sweep, reconstruction campaign, or numerical claim of an 18-degree convergence radius has been used. The only proposed regime witness is the existing audited rational oblique point.
