# Independent audit of algebraic defect correction

## Verdict and scope

The proposed construction supplies a meaningful local constructive improvement over a rank theorem or a generic implicit-function assertion. It explicitly defines a low-order left inverse on an open set of pairs of records, then uses that inverse to precondition an exact-model fixed-point iteration. A step requires a seven-node polynomial reconstruction, ordinary linear least squares, a specified simple root of a four-real-variable polynomial system, a scalar quadratic root, and explicit reconstruction formulas. No fourteen- or seventeen-variable nonlinear solve is hidden in those operations.

The iteration itself still has eighteen parameter coordinates. It is therefore best described as an algebraically preconditioned Picard iteration, rather than an exact four-dimensional elimination of the finite-angle inverse. It is neither a finite closed-form inverse nor an ambiguity-complete global algorithm.

The essential asymptotic claim is valid with two qualifications: derivatives must be taken in the scaled wedge chart, and the off-image inverse must be controlled on a coupled pair-of-records tube. The confluent frequency-derivative columns are essential for the claimed small contraction factor. Omitting them generally destroys the argument.

This audit covers the completed draft `algebraic_defect_correction.md` through its Section 12, together with the formulas in `oblique_inverse.md` and `oblique_record_certificate.md`. The draft incorporates the essential coupled-tube and scaled-derivative qualifications below. No blocking mathematical flaw was found. It is a mathematical audit, without parameter sweeps. It does not establish a numerical admissible wedge radius, a full eighteen-degree result, a noise tolerance on the original entire prior, or global completeness.

## 1. The off-image algebraic inverse is legitimate

Write F1 and F2 for the exact Taylor records through wedge degree one and two, respectively. They are functions of all eighteen unknown physical parameters, including all three signed frequencies and the two source coordinates. Let A(s,w) use the first input for fundamental reconstruction and the second input for the self-quadratic channel.

On a regular branch, the following stages define an actual map on an open input neighborhood, rather than only on physically compatible pairs:

1. Fit the order-seven recurrence by real stacked least squares. Its coefficients are rational functions of s wherever the Hankel matrix has full column rank.
2. Continue six designated nonreal simple roots, pair them by conjugation, and normalize them radially. The real root near one may be discarded in this chart. Root continuation, radial normalization, and a fixed argument branch are smooth operations away from their stated singularities.
3. Fit the constant and six fundamental tones by ordinary least squares. Select the locally fixed orientation signs and physical prism assignment.
4. Demix w with the degree-two harmonic dictionary augmented by six fundamental frequency-derivative columns.
5. Form the third-prism invariants using the fundamental coefficients from s, solve the four real last-prism polynomial equations on one designated uniformly isolated simple root branch, and perform the explicit triangular hardware reconstruction.
6. Recover source coordinates and conformally project each reconstructed M_i^{-1}C_i to a scalar times a rotation. The signed-wedge and narrow-phase chart is locally fixed.

All of these stages reproduce the original parameters at (s,w)=(F1(theta),F2(theta)). The conformal projection is exact there. Discarding the redundant first-prism shape condition off the physical image is permitted: a left inverse needs to reproduce compatible inputs, not enforce every compatibility equation at arbitrary inputs.

There is one conditioning cost worth making explicit. The stated construction discards one real first-order shape condition and uses both real components of the quadratic invariant. It therefore uses sixteen first-order directions plus two quadratic directions, rather than an optimally selected seventeen-plus-one system. Its full-record noise bound can consequently be conservative in two quadratic-scale directions. This does not invalidate the left-inverse identity or the contraction argument.

## 2. Required uniform regularity

The local theorem must assume a compact regular family with positive margins for:

- The physical transmitted branch, plane-intersection denominators, surface ordering, and original parameter bounds used by the chosen chart
- Nonzero wedge amplitudes in the scaled chart, nonzero and separated sampled fundamental frequencies, and distinct degree-two sampled nodes
- A nonzero leading baseline if the unconstrained seven-tone Hankel chart is used
- Full Hankel rank and full rank of both harmonic dictionaries
- The fundamental invariant denominator and the conformal projection denominator
- The trial-beam chart denominators used in the triangular reconstruction
- A uniformly isolated, single-valued C1 last-prism polynomial root branch with nonsingular four-variable Jacobian

Pointwise nonsingularity at every encountered root is not by itself a certificate that an arbitrary root-selection routine is continuous. Root labeling and an isolating neighborhood must be maintained over the whole input tube. Likewise, switching speed signs, prism assignments, wedge signs, or phase branches during iteration is outside one contraction theorem.

The chart need not cover axial illumination, vanishing source-perpendicular component, zero wedges, collisions, exceptional polynomial fibers, or critical optical branches. Such regions remain outside this local result.

## 3. Why the confluent demixer supplies the missing estimate

Let V2(nu) be the 25-column degree-two exponential dictionary and W1(nu) the six columns k exp(plus or minus i omega_i k) at the fundamental nodes. Let D(nu)=[V2(nu),W1(nu)] and let L(nu) be the row of its left inverse that extracts the designated third self-harmonic. Equivalently one can use a real sine/cosine dictionary. Separation on a compact family gives uniform bounds on D's left inverse and its first derivatives.

For every nu in the chart,

L(nu) 1 = 0,
L(nu) V_fund(nu) = 0,
L(nu) W1(nu) = 0.

Differentiate the middle identity. Each frequency derivative of a fundamental column lies in W1(nu). Therefore

(partial_nu L(nu)) V_fund(nu) = 0.

The constant column is annihilated identically as well. Thus a matched first-order record makes no contribution to the frequency derivative of the extracted quadratic coefficient. This is the crucial exact cancellation; a nonconfluent 25-column dictionary does not have it in general.

For a nearby reference frequency nu_bar, bounded second dictionary derivatives imply

||(partial_nu L(nu)) F1(theta_bar)||
  <= C epsilon ||nu-nu_bar||.

The quadratic part of F2 is O(epsilon^2), so its contribution to the same expression is O(epsilon^2). If

||s-F1(theta_bar)|| <= Cs epsilon^2,
||w-F2(theta_bar)|| <= Cw epsilon^3,

the Hankel stage gives nu(s)-nu_bar=O(epsilon), and hence

||(partial_nu L(nu(s))) w|| = O(epsilon^2).

The frequency derivative of the self-harmonic estimate is therefore O(epsilon^2), rather than the O(epsilon) value that would cause difficulty.

The same cancellation also controls the value error. The first-order leakage caused by an O(epsilon) frequency mismatch begins at its second power, giving O(epsilon^3), while moving the true quadratic coefficient produces O(epsilon^3). Thus the extracted self-harmonic differs from the reference one by O(epsilon^3).

## 4. Derivative orders and the necessary parameter scaling

Put e_i=epsilon a_i and use a coordinate vector x consisting of the nonwedge parameters and a_i. All derivative orders in this section are in x, with a_i bounded away from zero. Analyticity on a strict physical branch gives

||D_x(F-F1)|| = O(epsilon^2),
||D_x(F-F2)|| = O(epsilon^3).

These orders are not true for unscaled native wedge derivatives: the wedge columns there are O(epsilon) and O(epsilon^2), respectively.

The seven-tone Hankel matrix has six oscillatory directions of size epsilon and a nonzero constant direction of order one. Consequently its left inverse is O(epsilon^-1). Differentiating the least-squares recurrence on the stated tube retains this order: the least-squares residual is O(epsilon^2), so the extra normal-equation residual term is at most O(1). The reconstructed frequencies have s-derivative O(epsilon^-1).

The fitted baseline and raw harmonic matrices have s-derivative O(1). For this estimate it matters that the constant dictionary column is fixed: differentiating the coefficient fit with respect to frequency acts on oscillatory coefficients of size epsilon, not on an unrestricted order-one vector.

Let U be the nonzero first fundamental coefficient, with |U| bounded below by c epsilon, and C=L(nu(s))w the self-harmonic coefficient, of size O(epsilon^2). Then

D_s C = O(epsilon),
D_w C = O(1),
D_s(C/U^2) = O(epsilon^-1),
D_w(C/U^2) = O(epsilon^-2).

The first estimate uses the confluent cancellation from Section 3. Uniform invertibility of the four-variable invariant map and the explicit reconstruction formulas now yield, with A expressed in the scaled output chart,

||D_s A|| = O(epsilon^-1),
||D_w A|| = O(epsilon^-2).

Conformal phase recovery does not introduce an additional unaccounted power: its division by |e_i| is balanced by the extra epsilon in the raw conformal coefficient. The same applies to recovery of a_i=e_i/epsilon.

Composing these bounds proves an O(epsilon) derivative for the exact correction map. This establishes existence of a sufficiently small nonzero wedge regime, with constants dependent on all regularity margins. It supplies no useful numerical radius by itself.

## 5. The coupled tube is supplied by exact-compatible data

Let Rj=F-Fj and define

T_y(theta)=A(y-R1(theta),y-R2(theta)).

If y=F(theta_star), then for any current theta in a bounded regular scaled parameter neighborhood,

y-R1(theta)=F1(theta_star)+R1(theta_star)-R1(theta),
y-R2(theta)=F2(theta_star)+R2(theta_star)-R2(theta).

These lie in the O(epsilon^2)/O(epsilon^3) coupled tube. The initial pair (y,y) does as well. Thus the tube is not an assumption that the initial frequencies are already known from hidden truth. The reference truth is used to prove coverage of the data-derived initialization; the algorithm itself uses only y.

An actual self-map certificate and an isolated polynomial branch are still required to turn this asymptotic observation into a finite, usable result.

## 6. Exact finite contraction certificate

Choose a native-coordinate center theta_0, a positive diagonal scaling S, and a closed ball K={theta_0+Sx: ||x||<=r}. Require F, F1, F2, and the selected A branch to be C1 on neighborhoods of all relevant inputs and require K to lie within the intended physical chart.

At the actual corrected inputs define

q=sup_K ||S^-1 [D_s A D_theta R1 + D_w A D_theta R2] S||,
delta=||S^-1[T_y(theta_0)-theta_0]||.

If q<1 and delta+q r<=r, Banach's theorem gives one fixed point in K and convergence from every point of K. The sign omitted in the formula for q is harmless because D_theta T_y is the negative of the summed product.

S must be interpreted in the actual chosen units. If native wedge and phase angles are in degrees, the derivatives must include the exact degree-to-radian and sine factors. Asymptotic powers are not substitutes for this finite certificate.

Every exact-compatible theta in K is a fixed point because A(F1(theta),F2(theta))=theta. Therefore at most one exact-compatible solution lies in K, and if one exists, iteration recovers it. A fixed point alone does not prove that all 400 observations are matched. It is necessary to verify all exact residuals and all physical constraints before declaring an exact inverse.

For an exact noiseless existence claim, merely enclosing zero in a finite-precision interval for each residual is not a proof that the residuals vanish. The theorem's conditional compatibility statement is sound without such an existence claim. For noisy strips, outward residual bounds that fit entirely within the allowance can certify compatibility of an evaluated candidate.

This qualification is substantive: A ignores some data compatibility relations, so T_y can have a perfectly well-defined fixed point for an incompatible record.

## 7. Shared-record noise

The same observation perturbation enters both arguments of A. Its derivative is D_s A + D_w A, not two independently selectable perturbations. Let L be a certified bound for

||S^-1(D_s A+D_w A)||

over all simultaneous input segments (s+t zeta,w+t zeta) required by the allowed observation error. If theta_y is the fixed point for y and theta in K satisfies ||F(theta)-y||<=eta, then

||S^-1(theta-theta_y)|| <= L eta/(1-q).

The argument compares T_y(theta) with T_{F(theta)}(theta)=theta and uses contraction of T_y. The center theta_y need not itself be observation-compatible. The bound is an outer radius for compatible parameters; twice it bounds their diameter. Componentwise noise or native-coordinate bounds should retain the corresponding input weights and output factors S.

The simultaneous segments must stay inside the same A branch domain. In particular, the narrow asymptotic tube above directly accommodates only noise that fits its O(epsilon^3) second-input width. A larger noise allowance is possible only with a separately certified enlarged domain; it is not justified merely by the displayed rate. No guaranteed numerical noise tolerance follows from an asymptotic epsilon exponent alone.

## 8. What remains unproved

This construction does not prove:

- A useful contraction constant at any particular finite wedge in the original eighteen-degree prior
- That an unvalidated polynomial solver finds and labels every branch
- That the full prior admits a regular separated-frequency cover
- That all physically compatible records avoid exceptional algebraic or optical strata
- That noisy compatible sets are singletons
- That checking a local fixed point excludes distant exact solutions

None of those omissions makes the local construction circular. They delimit its actual contribution: an explicit low-order inverse, a structurally small exact defect-correction derivative, and a finite certificate whose success would yield a validated local finite-angle inverse.
