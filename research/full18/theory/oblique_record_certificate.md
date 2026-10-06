# Data-derived rotor enclosure and an exact 17+1 physical inverse

## Scope and result

The newly audited oblique physical construction has a rank-seventeen first-order record and one remaining beam-orientation fiber. A quadratic channel breaks that fiber at the audited witness. The useful next theorem is an exact strong/weak inverse with an explicit scalar Schur-complement margin, preceded by a record-derived rotor enclosure. Neither theorem assumes most physical parameters or rotor frequencies are known.

The conditions below are sufficient, checkable certificates. They are allowed to fail. A failed condition leaves the associated original-prior region unresolved; it does not authorize discarding it or replacing it with a favorable fitted neighborhood. Complete global recovery requires a cover of every compatible region, including centered, zero-amplitude, repeated-frequency and near-critical strata.

## 1. A seven-tone first stage using all 200 observations

On a uniformly guarded prior region, suppose the exact paired physical record has a certified first-order enclosure

F_h(t_k)=B_h+sum_i [C_hi cos(2pi nu_i t_k)+D_hi sin(2pi nu_i t_k)]+R_hk,

|R_hk|<=tau_hk, h=x,y, k=0,...,199.

The unknown nu_i are positive frequency magnitudes; their signs and prism assignments are not presumed. Here C,D are the general oblique ellipse coefficients. A wedge-degree-one Taylor bound is enough; its safe remainder is O(epsilon^2), without assuming normal incidence or a known beam. Let delta=max_hk(eta_hk+tau_hk), including any explicitly bounded clock or evaluator discrepancy.

Build a stacked real Hankel design H_y of size 386 by 7. Its rows, for each axis h and k=0,...,192, are

(y_hk,y_h,k+1,...,y_h,k+6).

Let v_y have corresponding entries y_h,k+7. All 200 samples are used. A true seven-tone record obeys H_0 c=-v_0, where p(z)=z^7+sum_(j=0)^6 c_j z^j has roots 1 and exp(±2pi i nu_i/20).

Choose any verified left inverse D of H_y and write gamma=||D||_(infinity<-infinity), chat=-D v_y. If

q_H=7 gamma delta < 1,

then every true H_0 compatible with the first-order enclosure has full column rank. Its recurrence coefficient vector satisfies

||c-chat||_infinity <= gamma delta(1+||chat||_1)/(1-7 gamma delta).        (1)

PROOF. H_0=H_y+E, v_0=v_y+e with ||E||_infinity<=7delta and ||e||_infinity<=delta. Multiplication by D gives (I+DE)c=chat-De. Hence (I+DE)(c-chat)=-D(e+E chat), and the Neumann bound gives (1).

A floating pseudoinverse is only a proposal. If a proposed D has defect ||I-DH_y||<=rho<1, replace it mathematically by (DH_y)^-1D and use gamma<=||D||/(1-rho), or retain the defect explicitly. The corrected left inverse must also be used in chat=-D v_y. If a floating approximation to that center is retained instead, add its certified coefficient error to Delta_c before applying Rouche. All arithmetic enclosures must be outward.

Stacking the two real axes is important. A circular complex trajectory may have only one signed exponential in x+i y, but its two real coordinates still carry both members of the conjugate node pair. The real stacked construction does not demand a spurious negative-complex-frequency coefficient. Rank can still fail when DC is zero, a rotor amplitude vanishes, or nodes collide. Such cases need their own charts or remain unresolved.

The constant root 1 can be imposed as an additional exact structural constraint. A differenced six-tone variant removes DC, at the cost of doubled worst-case noise and attenuation near zero frequency; it is a separate optional chart, not a reason to disregard rank failure.

## 2. Certified roots, signed speeds and all labels

Let Delta_c be the right-hand side of (1). Candidate disjoint disks or contours must satisfy

|phat(z)| > Delta_c sum_(j=0)^6 |z|^j

on their boundaries, with certified root counts summing to seven. Rouche's theorem then encloses every compatible first-order spectrum. Intersect the resulting disks with the unit circle, with the exact root 1 and conjugate-pair structure, and with the original native speed arc. No compatible rotor node is discarded.

Frequency magnitude intervals follow from the arguments of the three conjugate pairs. Keep every pairing and label assignment consistent with these intervals and the native box. A certified interval trigonometric design then encloses each ellipse matrix E_i=[C_i,D_i] in the two real screen coordinates. This coefficient reconstruction must charge frequency-interval uncertainty; evaluating only the midpoint frequencies is not sufficient.

For the physical first-order transfer matrix

M_i=(h_i-1)[L_i I+(W_i+L_i/h_i)t t^T-t B^T/h_i],

if det M_i>0 is certified over the candidate physical region, then a fit using the positive frequency magnitude obeys

sign N_i = sign det E_i.

Indeed E_i=e_i M_i R(phi_i) diag(1,sign N_i), and det E_i=e_i^2 det M_i sign N_i. Thus an interval for det E_i excluding zero recovers rotor direction, independently of wedge sign or phase. If det M_i or det E_i cannot be signed, retain both directions.

A useful explicit physical determinant check is

det M_i=(h_i-1)^2 L_i [L_i+(W_i+L_i/h_i)|t|^2-(t dot B)/h_i].

The observed coefficients and structural shape equations then constrain hardware and the two-component beam slope. The explicit first-order shape inverse and consistency curve must be used as a complete outer set, retaining all permitted assignments and disconnected branches. The proof-witness frequencies are not initialization data.

The seven-tone stage is more viable than demanding resolution of every cubic sideband, but it still requires a genuine uniform first-order remainder over every region it claims to cover. If a branch margin or tail bound cannot be certified on the original prior, use exact optical exclusion/subdivision there or return it unresolved. A tail bound checked only around a good fit does not establish global coverage.

## 3. Exact strong/weak formulation

Let a data-derived branch admit physical chart theta=Theta(z,psi), z in R^17, with psi the remaining beam-orientation parameter. Theta must be one-to-one on the retained chart, preserve original units/bounds, and lie on a strict optical branch. Coordinate singularities, including vanishing beam magnitude, require other charts.

Choose seventeen fixed, real finite-record features f(z,psi)=P_f F(Theta(z,psi)) and one quadratic feature g(z,psi)=P_g F(Theta(z,psi)). The feature rows are obtained from the data-derived rotor chart and then held fixed. The nineteen-or-more raw first-order coefficient observations may be compressed to seventeen independent directions by a verified selection. Frequency-tangent channels are included; there is no assumption that frequencies are fixed or known.

All functions f,g use the exact physical forward model. The first-order and quadratic expansions are used to construct the chart and propose preconditioners, not substituted for the exact equations in the final theorem.

For observed record y and componentwise allowance eta, define

a0=P_f y, c0=P_g y,
r_a=|P_f| eta, r_c=|P_g| eta.

Let A=a0+[-r_a,r_a] and C=[c0-r_c,c0+r_c]. This rectangular feature box is an outer approximation to the correlated feature-noise image. Keeping the correlations can sharpen the result but is not needed for soundness.

## 4. Uniform strong inverse: what must actually be certified

On an interval I of psi and a strong-variable domain Z, require that for every (psi,a) in I times A the exact equations

f(z,psi)=a

have exactly one solution z=Zeta(psi,a) in Z, and that this solution stays in the physical chart. This is a substantive existence-and-uniqueness condition, not merely a nonsingular Jacobian at a fitted point.

One sufficient implementation is a uniform contraction. Choose a fixed preconditioner C_f and certify

sup_(Z,I) ||I-C_f J_f,z|| < q_f < 1,

and the self-map condition z-C_f[f(z,psi)-a] in Z for every z in Z, psi in I and a in A. Compact convex Z then supplies existence and uniqueness by Banach's theorem. An interval-Newton/Krawczyk theorem with the same uniform parameter coverage, or a structural global strong inverse, can replace this sufficient test. Scaling the seventeen strong variables is encouraged, but its map back to native coordinates must be retained.

A solution at one psi, or separate numerical solves at endpoints, is insufficient. If one strong chart cannot cover a whole interval, subdivide the data-derived branch and retain a verified cover. The procedure does not assume an arbitrarily tiny chart containing the truth.

## 5. Exact scalar Schur complement

Define

G(psi,a)=g(Zeta(psi,a),psi),
J=J_f,z,
w=g_z J^-1,
s=g_psi-g_z J^-1 f_psi.

Implicit differentiation gives the exact identities

D_psi Zeta=-J^-1 f_psi, D_a Zeta=J^-1,
partial_psi G=s, D_a G=w.                                  (2)

Assume the sign of s is constant and |s|>=mu>0 throughout the strong solution family. For positive sign, a uniform endpoint bracket

sup_(a in A) G(psi_left,a) <= inf C,
inf_(a in A) G(psi_right,a) >= sup C                         (3)

ensures that for every (a,c) in A times C there is exactly one solution of G(psi,a)=c. Reverse the inequalities for negative sign. Strict inequalities give an interior-root margin.

Equations (2)-(3) reduce the exact eighteen-feature inverse to a scalar monotone root, while all eighteen original parameters remain unknown and jointly solved. Bisection or interval Newton can enclose every permitted root without blind eighteen-dimensional optimization.

If only some feature targets are bracketed, one may still enclose and exclude scalar intervals using G(I,A)-C, but may not claim existence for every target. Uniform bracketing is the clean sufficient condition for the finite perturbation theorem below.

## 6. Finite noise propagation in native coordinates

For two feature targets (a,c),(a',c') in A times C, differentiate their solution along the connecting segment. Convexity and the uniform bracket keep that entire solution path inside the certified family. Let lambda_j=sup |w_j|. Then

|Delta psi| <= [|Delta c|+sum_j lambda_j |Delta a_j|]/mu.       (4)

For physical chart coordinates, the exact differential is

d theta = U da + v d psi,
U=Theta_z J^-1,
v=Theta_psi-Theta_z J^-1 f_psi.                             (5)

Here v is the actual native tangent to the constant-strong-feature fiber. Set Ubar_ij=sup|U_ij| and vbar_i=sup|v_i| over the certified family. Combining (4)-(5) gives

|Delta theta_i| <= sum_j Ubar_ij |Delta a_j|
 + (vbar_i/mu)[|Delta c|+sum_j lambda_j|Delta a_j|].           (6)

For the diameter of the compatible feature tube, use |Delta a_j|<=2r_a,j and |Delta c|<=2r_c. For the error relative to its exact center-feature solution, use r_a,r_c instead, provided that center solution exists as certified above. Being the feature-center solution does not establish compatibility with the original full record.

The pointwise reparameterization-invariant weak-conditioning quantity is |s| divided by a native norm of v, not |s| alone. Reparameterizing psi scales s and v by the same local factor. The uniform bound mu/sup||v|| is valid but need not remain numerically identical under nonlinear reparameterization because its extrema can occur at different points. Any claim of useful hardware precision must report native coordinate factors vbar_i/mu and the strong factors Ubar, with angle and length units specified.

The row w also matters: noisy strong features move the scalar closure. Omitting w would treat jointly unknown nuisance parameters as fixed calibration data.

## 7. Relation to the 5/12/1 hierarchy and lower bounds

At the audited oblique family, use a native-conditioned adapted chart with five strong directions, twelve directions of size epsilon, and one final direction of size epsilon^2. A bounded inverse for the normalized first-order shape map is required; a rational nonzero determinant alone does not supply a useful constant.

The exact Schur complement is expected and can be certified in the form s=epsilon^2 s2+O(epsilon^3), where s2 is the nonzero quadratic derivative along the first-order fiber. Bounds on J^-1, chart derivatives, and the remainder must be uniform on the actual retained branch. They cannot be inferred just from the wedge exponent.

Under those bounds, twelve adapted modes have error of order eta/epsilon and one has error of order eta/epsilon^2. Native coordinates generally mix these modes through v. There is no assertion that only one named physical parameter has the weak rate, or that the native wedges/source offsets individually have the strong rate.

A matching one-dimensional deterministic lower bound comes from a curve along which the seventeen first-order observables remain fixed, with certified endpoint separation. One clean choice is to parameterize the curve by a monotone native coordinate theta_j=s; then two endpoints s and s+delta have native maximum-coordinate distance at least |delta|. The derivative bound must be ||D F(theta(s)) theta'(s)||_infinity<=C epsilon^2 over ALL 400 original real observation coordinates, established from the full optical derivative or a differentiated Taylor remainder along the leading-order fiber. A bound on the scalar quadratic feature g or on selected projections alone is insufficient. Under that full-record bound, their data distance is at most C epsilon^2 |delta|. Their midpoint record proves minimax native error at least (1/2)min(r,2eta/[C epsilon^2]), where r is a certified available parameter interval length. Alternatively use a certified secant lower bound ||theta(s)-theta(s')||>=c_sec|s-s'| and retain c_sec in the minimax bound. Unit native tangent norm alone is insufficient because arc length need not lower-bound endpoint separation. The exact source/beam chart conditioning is included in C and any secant factor. This lower bound and the Schur upper bound match powers, not automatically numerical constants.

## 8. Exact-model correction and every residual

For noiseless data, a successful strong inverse and scalar bracket produce at most one solution of the selected eighteen features on that branch. Verify all 400 scalar observations and every physical branch/surface-order/bound constraint before declaring it an inverse of the original record.

For noisy data, the result is a native parameter tube, not a unique system. Intersect the tube with all 400 observation strips; exact affine geometry elimination remains available to tighten it. Feature-space overapproximation may contain systems that fail the original record and must not be reported as compatible.

For global claims, every original-prior region must be excluded, covered by such a certified scalar branch, or returned unresolved. Multiple surviving branches can contain distant hardware even when each branch is individually well conditioned. The scalar theorem supplies a rigorous exact-model correction mechanism; it does not silently replace global coverage.

## 9. Minimal decisive checks

Without parameter sweeps, a useful validation would establish:

1. A certified first-order remainder on a named prior region and the observed stacked-Hankel condition q_H<1, followed by seven root enclosures;
2. Full ellipse coefficient intervals, certified direction signs where possible, and a complete first-order shape/fiber enclosure;
3. A uniform strong inverse along one entire retained scalar interval, not merely at one point;
4. A nonzero native-normalized Schur margin, endpoint bracket, and explicit coordinate uncertainty factors from (6);
5. All 400 exact residual checks, with every other prior branch either covered, excluded, or openly unresolved.

Failure at any step is a meaningful diagnosis: insufficient harmonic signal, tail too loose, unexcited/ill-conditioned shape, scalar turning point, or unresolved global ambiguity. No condition is repaired by assuming the true frequency or hardware neighborhood in advance.

## 10. Normalized quadratic observables and the audited witness

The audited oblique construction uses I3=C_(+2N3)/U_(+N3)^2 along a first-order fiber. This is useful algebraically, but division by a small fundamental is part of the noise budget.

One option is to choose a fixed nonzero reference U0 from the certified data-derived branch and define the scalar feature as Re(C_(+2N3)/U0^2), or another fixed real quadrature. This remains a linear record functional with exact noise row norm. Include the fundamental U among the strong features. Along their exact constant-feature fiber U is fixed, so the nonzero invariant derivative transfers. Variations in measured U are then handled by the strong-feature term w in (4), rather than ignored.

If instead actual noisy ratios are used, suppose |U-Uhat|<=r_U<|Uhat| and |C-Chat|<=r_C. A valid finite complex ratio bound is

|C/U^2-Chat/Uhat^2|
<= r_C/(|Uhat|-r_U)^2
 + |Chat| r_U(2|Uhat|+r_U)/[(|Uhat|-r_U)^2 |Uhat|^2].

The source proof reports a native-infinity norm of the limiting beam-orientation fiber tangent approximately 981.79 at its rational witness. The raw self-second-harmonic directional gain after native normalization is approximately 1.88 epsilon^2 per native-infinity unit. These are witness-specific values, not singular-value bounds or uniform robustness constants. They illustrate why the tangent factor in (5)-(6) cannot be omitted.

The physical matrix audit also reports a uniform positive orientation bound on the original geometric/index/source box: the transverse eigenfactor L_i is at least 50 and the parallel factor is at least 50-125/287>49.5. With that audited convention, det M_i>0 is structural; the remaining data test for signed rotation is that the interval determinant of the observed ellipse excludes zero.
