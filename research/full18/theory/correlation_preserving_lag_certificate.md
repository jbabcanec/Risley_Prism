# Projected, correlation-preserving lag certification

## Decision and scope

Yes: the center linear bound can be combined with a genuine second-order bound for the **six-by-six preconditioned lag matrix itself**. This avoids first enclosing 400 independent sample errors and only afterward projecting them. The useful structure is explicit:

- Every constant temporal component cancels before any absolute value.
- The quadratic optical record contributes nine conjugate pairs of rank-one matrices.
- All four source/distance geometry coordinates remain exactly affine, so their pure second derivatives vanish.
- The exact maximum norm of the center linear term requires at most 64 sign patterns per matrix row, irrespective of the eighteen-dimensional parameter box.
- The dominant quadratic contribution has explicit small coefficient and one-variable rotor-polynomial derivatives. It does not require an unspecified eighteen-dimensional Hessian calculation.

The remaining cubic-and-higher optical tail has finite curvature on the guarded compact box, but a useful numerical upper bound has not yet been certified here. Thus this theorem identifies a concrete replacement for the failed record-ball enclosure, but does not establish that the replacement passes on the unchanged +/-0.001 native box.

This is mathematical diagnosis only. No new optical evaluation, subdivision, witness change, target-box change, damping test, or numerical refinement is performed. Kappa denotes wedge amplitude; epsilon denotes independent measurement error.

## 1. The exact matrix to certify

Take the fixed lag ell, K=200-7ell, and the original lag operators

    H_f[(h,k),j]=f_h,k+j ell,     j=0,...,6,
    v_f[(h,k)]=f_h,k+7ell,

for h=x,y and k=0,...,K-1. Let R be the fixed 7x6 Helmert matrix, cbar=-(1/7)1, and L0 a frozen exact left inverse at the chosen center record s0:

    L0 H_s0 R=I6,       1^T R=0.

The record correction is s(theta)=y-R1(theta), R1=F-F1. Define

    G(theta)=L0 H_(R1(theta)) R.

Then the entire required matrix is exactly

    J(theta)=L0 H_s(theta) R
            =I6-[G(theta)-G(theta0)].                  (1)

L0, R, y, and theta0 are held fixed. Thus differentiation and projection commute. A certificate for G(theta)-G(theta0), rather than a certificate for all individual record entries, is the appropriate object.

For any temporally constant paired vector b,

    H_b R=0.

This removes both the large zero-wedge baseline and every quadratic DC shift identically. No baseline magnitude enters (1).

## 2. Exact affine geometry survives projection

Let x contain all fourteen optical coordinates and u=(b_x,b_y,g,d) the four geometry coordinates. At fixed optics one prism has the exact affine position transfer

    p_next=A_j p+3 A_j a_j+ell_j w_j,
    A_j=(I-w_j u_j^T)(I-a_j u_j^T)^(-1),

where a_j is its internal slope, w_j its outgoing external slope, u_j its exit-plane slope, and ell_j is g,g,d respectively. All these direction matrices depend only on optical coordinates. The initial position is b+6t.

Consequently the entire record has the exact representation

    F_k(x,u)=Pi_k(x) (1,b_x,b_y,g,d)^T,                 (2)

with a 2x5 coefficient block Pi_k. It can be propagated in one ray trace: multiply the whole block by A_j, add 3 A_j a_j to its constant column, and add w_j to the g or d column as appropriate. This is not five independently chosen physical cases.

Wedge Taylor coefficients and their exact remainders preserve this affine dependence. Therefore

    G(x,u)=C(x)+sum_(a=1)^4 u_a B_a(x),                 (3)

where C,B_a are six-by-six matrices. In particular,

    partial_(u_a) partial_(u_b) G=0                    (4)

for every pair of geometry coordinates. Native beam/index changes do not spoil (3): their nonlinear coordinate conversions are optical and independent of u. If the thirteen-coordinate sagittal lift is composed afterward, its geometry functions are no longer independent coordinates; its first and second chain-rule terms must be included. One must not transfer (4) unchanged to those reduced coordinates.

## 3. A nine-pair rank-one formula for the quadratic part

Split

    R1=P2+R2,       P2=F2-F1,
    G=G2+G3,
    G2=L0 H_P2 R,   G3=L0 H_R2 R.                     (5)

The actual quadratic optical record has only its DC and the eighteen nonconstant rotor multi-indices of total degree two. Choose the nine representatives

    Mplus={2e_i : i=1,2,3}
          union {e_i+e_j, e_i-e_j : 1<=i<j<=3}.

Write

    P2_k=c0+2 Re sum_(m in Mplus) c_m z_m^k,
    z_m=exp(2pi i (m dot N)/20),                       (6)

where c_m is a two-component complex coordinate coefficient. This formula holds independently of frequency separation; a nonzero multi-index whose sampled node happens to equal one is also killed by the right projection below. The inverse chart's separate rank guards are still required.

For m=2e_i, c_m has the scalar factor e_i^2 exp(2i phi_i). For m=e_i+e_j or e_i-e_j, it has e_i e_j exp(i m dot phi). The remaining two-vector coefficient is rational in the five shape variables (h1,h2,h3,t_x,t_y), and affine in u.

This last assertion is constructive: evaluate the exact optical/2x5 position-block recurrence in the lambda-degree-two Laurent ring, with

    u_i=lambda (e_i/2)
        (w_i+w_i^-1, (w_i-w_i^-1)/i).

At zero wedge the scaled glass roots are h_i and the exit roots are one. Therefore every coefficient through degree two is rational in h,t. In this scaled zero-wedge compiler its possible optical denominator factors are powers of the positive h_i; the zero-wedge exit roots, plane factors and outgoing axial denominators are one. The forward coefficients do not divide by the inverse-chart quantities U, B_perp, f1, or h_i-1. Extract the nineteen degree-two Laurent coefficients, then substitute w_i=exp(i phi_i)z_i^k. This uses no observed temporal derivatives and does not assume any physical parameter known.

Partition L0=(Lx,Ly), each block 6xK. Define

    v(z)=(1,z,...,z^(K-1))^T,
    w(z)=(1,z^ell,...,z^(6ell))^T,
    U(z)=[Lx v(z), Ly v(z)],          a 6x2 matrix,
    q(z)^T=w(z)^T R,                 a 1x6 row.

Then the exact compressed quadratic contribution is

    G2(theta)=2 Re sum_(m in Mplus)
                  [U(z_m)c_m] q(z_m)^T.                (7)

Each complex summand has rank at most one; each real conjugate pair has rank at most two. There are nine pairs, not 200 unrelated time enclosures. This is a sum-of-low-rank representation, not a claim that the final 6x6 matrix has rank below six.

The factor q(1)=0 implements baseline/DC cancellation exactly; in fact q(z)=0 whenever z^ell=1. The augmented row below has the same additional lag-periodic cancellation. For ell=11, each entry of the quadratic matrix is assembled from polynomials in a shared rotor node of degree at most K-1+6ell=188, with data-fixed coefficient arrays. The three original frequencies remain unknown shared variables; there is no fitted constant substituted for them.

## 4. Explicit derivatives before taking norms

Put z=exp(i omega), with omega=2pi(m dot N)/20. Let D_K=diag(0,...,K-1) and D_7=diag(0,...,6). The one-variable factors in (7) have exact derivatives

    U'=i[Lx D_K v, Ly D_K v],
    U''=-[Lx D_K^2 v, Ly D_K^2 v],
    q'^T=i ell (D_7 w)^T R,
    q''^T=-ell^2 (D_7^2 w)^T R.                        (8)

For a single term A=U(omega)c q(omega)^T, where c is independent of N,

    A_omegaomega=U''c q^T+2U'c q'^T+Uc q''^T,
    A_(omega,r)=U'c_r q^T+Uc_r q'^T,
    A_(r,s)=U c_rs q^T,                                (9)

for nonfrequency parameters r,s. Frequency conversion contributes the known factors (2pi/20)m_i. Amplitude and phase derivatives are explicit derivatives of the scalar factors in Section 3. The only nontrivial coefficient value/first/second derivatives in the quadratic term are rational derivatives in the five shape variables; geometry derivatives are affine as in (3).

Uniform bounds for U,U',U'',q,q',q'' require only one-variable rotor-arc polynomial bounds for each mode. Native N intervals give the omega interval exactly by a linear combination. Rational shape coefficient bounds use their known five-variable circuit and positive h denominators. Native-angle and index conversions are explicitly differentiated. In particular t_a=tan(beta_a) and h_i=sqrt(n_i^2+(n_i^2-1)|t|^2), with degree conversions included; beam differentiation changes all three h_i, and second derivatives require the full second chart chain rule. No high-dimensional optimization or subdivision is part of this representation.

Crucially, the polynomial sums Lx v and Ly v, the products in (9), and the sum over modes are formed before componentwise absolute values. Merely replacing them by sums of absolute sample weights would throw away the same correlation this construction is intended to preserve.

## 5. A true second-order certificate in shared parameters

Let the original native box be (x0+xi,u0+eta), with |xi_i|<=r_i and |eta_a|<=s_a. These radii are the prescribed native tolerances; no target shrinking is introduced. Define

    A(x)=C(x)+sum_a u0_a B_a(x).

The exact Taylor decomposition is

    G(x0+xi,u0+eta)-G0
      =DA(x0)xi+sum_a B_a(x0)eta_a
       +R_A(xi)+sum_a eta_a[B_a(x0+xi)-B_a(x0)],        (10)

with

    R_A(xi)=integral_0^1 (1-t)
              D^2 A(x0+t xi)[xi,xi] dt.

There are no geometry-geometry terms. Suppose nonnegative 6x6 arrays H_ij and K_ai bound, entrywise on the optical box,

    |partial_i partial_j A|<=H_ij,
    |partial_i B_a|<=K_ai.

Then a rigorous entrywise second-order remainder matrix is

    E2=(1/2)sum_(i,j) r_i r_j H_ij
        +sum_(a,i) s_a r_i K_ai.                       (11)

This is genuinely second order in the prescribed perturbation radii. The derivatives in H,K are derivatives of the projected matrix, not independent per-sample derivative bounds subsequently multiplied by ||L0||.

### Exact small-sign evaluation of the center linear term

Let L_n be the eighteen center derivative matrices, and let d_n be the corresponding native half-widths. The exact maximum infinity norm of their linear combination is

    alpha_lin = max_(row p=1,...,6)
                max_(sigma in {+1,-1}^6)
                sum_(n=1)^18 d_n
                   |sum_(col q=1)^6 sigma_q (L_n)_(p,q)|.          (12)

Proof: for a fixed row, its l1 norm is max_sigma sigma dot row; maximize that scalar linear form over the shared parameter box. The maximum over each parameter is its half-width times the absolute value of its coefficient. The finite maxima commute. Global sign reversal duplicates a value, so at most 32 sign choices per row are needed if desired.

The center derivative matrices in (12) must be exact or rigorously enclosed. If only intervals are available, bound each signed sum with its interval uncertainty before taking the finite maximum; a floating center matrix alone is not a certificate. Thus evaluating the exact center linear bound costs a fixed six-output sign enumeration, not 2^18 parameter corners or nonlinear optimization. The formula retains the fact that one parameter variation simultaneously changes every matrix entry.

Combining (10)-(12),

    sup_box ||G(theta)-G0||inf
       <= mu=alpha_lin+||E2||inf.                       (13)

If mu<1, equation (1) is nonsingular throughout the full original box and

    ||J(theta)^-1||inf<=1/(1-mu).                       (14)

This is the desired direct lag certificate. A weighted/componentwise version can be used, but is not needed for the theorem.

## 6. The same compression also controls the recurrence right-hand side

The root-coefficient solve needs more than J. Introduce the augmented projected operator

    Gaug(theta)=L0[H_(R1(theta)) R,
                    v_(R1(theta))+H_(R1(theta)) cbar],

of size 6x7. The rank-one formula (7) extends verbatim with right row

    qaug(z)^T=(w(z)^T R, z^(7ell)+w(z)^T cbar).         (15)

It still vanishes at z=1. The maximum polynomial degree becomes 199. The center linear norm formula uses seven column signs, at most 128 choices per row.

Write Gaug(theta)-Gaug0=[E,f], and let a0 be the center recurrence coordinate. Then

    J(theta)=I-E,
    a(theta)-a0=(I-E)^-1[f+E a0].                      (16)

The directly relevant right-hand-side perturbation is therefore g=f+E a0. In (15), contraction with (a0,1)^T gives

    z^(7ell)+w(z)^T(cbar+R a0)=p_c(z^ell),             (17)

the center annihilator evaluated at the mode's lag node. This keeps an additional exact filter cancellation. Bound g directly rather than separately bounding f and E a0 and summing their magnitudes.

If a direct projected Taylor certificate gives ||g||<=gamma and (13) gives mu<1, then

    ||a(theta)-a0||<=gamma/(1-mu).

All scalar polynomial-root, alias-lift, full-200 coefficient-fit, and bivariate root guards remain subsequent steps. Passing the lag step would not by itself establish the full native 0.001 chord certificate.

For generic independent measurement error |h|<=epsilon, the additional changes in J and g are fixed linear record operators. Let their exact real-coordinate row-norm bounds be mu_noise and gamma_noise. Then

    mu+mu_noise epsilon<1,
    ||a(theta,h)-a0||
       <=[gamma+gamma_noise epsilon]/[1-mu-mu_noise epsilon].       (18)

No correlation is falsely assigned to independent observation errors. Epsilon remains independent of kappa.

## 7. What is known explicitly, and the remaining tail obligation

Split every projected curvature coefficient in (11) into its G2 and G3 parts.

For G2, equations (7)-(9), the homogeneous amplitude/phase factors, and the affine geometry block give an explicit reusable rational/trigonometric calculation. It involves:

- nine one-variable rotor polynomials and their first two derivatives;
- five-variable rational shape coefficient derivatives;
- exact affine geometry coefficients;
- finite sums of 6x6 matrices.

Thus the dominant quadratic contribution is not merely renamed as an unknown high-dimensional Hessian. Its temporal correlations and zero terms are exposed algebraically before bounding.

For G3=L0 H_R2 R, the exact identity is

    G3=integral_0^1 [(1-lambda)^2/2]
          L0 H_(partial_lambda^3 F(lambda e)) R d lambda.          (19)

The optical parameter derivatives needed in (11) can be taken inside this integral. Geometry remains affine. The required projected tail jets are at most second optical derivatives of the third lambda derivative. A lambda-order-three jet with second optical jets supplies them without a full fifth-order eighteen-variable tensor.

However, projecting and then immediately replacing every constituent sample derivative by an independent interval can again lose the shared-frequency correlation. Equation (19) alone is not a quantitative improvement. A valid implementation must form the common-parameter projected jet, or use a further exact finite Laurent jet plus a separately certified tail. It must actually provide uniform projected bounds for the G3 contributions to H_ij and K_ai on the declared box.

A useful numerical value for that projected bound is not proved or evaluated here. The curvature is finite: strict radicand/denominator margins on the compact optical box times lambda in [0,1] give a real-analytic neighborhood and continuous bounded derivatives. Finiteness must not be confused with having a sufficiently small certified upper bound. The fixed-shift obstruction also warns against pretending the exact finite-angle tail has a bounded finite harmonic dictionary. The finite nineteen-tone statement belongs only to P2.

## 8. Precise implication for the existing bottleneck

The earlier computation reported a center linear bound near 0.0508 but a full-box sufficient lag majorant 1.37074. Those numbers came from different levels of enclosure; their difference does not prove actual nonlinear growth of that size.

The present theorem replaces the decision by

    alpha_lin + ||E2||inf < 1.                         (20)

If a verified alpha_lin<=0.050811724 is retained, the exact remaining sufficient obligation is

    ||E2||inf < 0.949188276,

using the direct projected curvature definition (11). The displayed threshold is only the arithmetic consequence of that hypothesized verified center bound; it is not a new evaluated certificate. The original center data or its endpoint enclosure must justify the alpha bound.

This removes specific avoidable losses: baseline/DC magnitude, independent treatment of 200 quadratic samples, pure geometry Hessians, and a high-dimensional maximization of the linear term. It does not remove the genuine need to control nonlinear optical curvature. The exact cubic-and-higher projected tail is the residual hard obligation.

Accordingly:

- There is a concrete reusable compression and a finite sufficient theorem, with no new nonlinear solve or high-dimensional subdivision.
- The quadratic contribution and the affine-geometry structure can be bounded explicitly from the existing optical formulas.
- The current evidence does not show whether the remaining projected tail satisfies (20).
- No new passing lag, bivariate, chord, noise, native-accuracy, or global-branch certificate is claimed.

The useful next decision is whether to implement a correlation-preserving projected-jet/Laurent-tail compiler as a general proof tool. Repeating scalar record-ball bounds or shrinking the target would not establish this structural step.

## 9. An explicit finite route for the cubic term and the higher tail

The following formulas specify the next analytic bound. They do not run another numerical refinement.

For each of the five geometry coefficient columns j=0,...,4, form the matrix-valued function

    Phi_j(lambda,x)=L0 H_(Pi_j(lambda,x)) R,

where Pi_j is the exact 2-vector coefficient column from (2), with all wedge sines multiplied by lambda. Let

    K_(p,j)(x)=(1/p!) partial_lambda^p Phi_j(0,x).

The projected cubic-and-higher contribution in that column is exactly

    G_(3,j)=K_(3,j)
             +(1/6) integral_0^1 (1-lambda)^3
                        partial_lambda^4 Phi_j(lambda,x) d lambda. (21)

Projection is inside the integral, and differentiation in optical parameters commutes with both projection and integration. For any optical multi-index alpha with |alpha|<=2,

    D_x^alpha G_(3,j)=D_x^alpha K_(3,j)
       +(1/6) integral_0^1 (1-lambda)^3
              D_x^alpha partial_lambda^4 Phi_j(lambda,x) d lambda.

Hence any entrywise uniform projected jet bound M_(j,alpha,4) gives

    |D_x^alpha G_(3,j)|
       <= |D_x^alpha K_(3,j)| + M_(j,alpha,4)/24.        (22)

The factor 1/24 is exact. Use alpha of order two for the centered-geometry A curvature in (11), and alpha of order one for its B_a geometry-cross terms. Combine the five columns with the known center geometry u0 before taking absolute values when bounding A.

### The cubic term is another explicit finite low-rank calculation

The homogeneous cubic record has 44 signed nonconstant multi-indices: six with |m|1=1 and 38 with |m|1=3. Thus K3 has 22 conjugate pairs of the same outer-product form as (7). Its coefficients are homogeneous cubic polynomials in the three signed wedge sines, times exp(i m dot phi), with rational five-shape coefficients affine in geometry.

The exact plane slope must now include its cubic term:

    u_i(lambda)=lambda e_i v_i+(lambda^3 e_i^3/2)v_i+O(lambda^5).

Ignoring this correction would omit physical fundamental-frequency contributions at cubic order. The lambda-degree-three Laurent compiler includes it automatically. Its unnormalized forward coefficients likewise have no inverse-chart poles U, B_perp or f1; they are obtained through the same positive-h zero-wedge recursion. The six cubic fundamental modes are additionally killed in the filtered right-hand side (17) whenever their lag nodes are roots of the frozen center p_c; this is an exact conditional cancellation of the value, not an assumption that a trial frequency equals its center. Speed and mixed derivatives must retain the nonzero p_c' and p_c'' terms from differentiating p_c(exp(i ell omega)); pointwise annihilation does not annihilate the curvature.

All cubic coefficient and rotor derivatives therefore admit the same small rational/polynomial operations as Section 4. Only the order-four-and-higher remainder in (22) requires an exact finite-angle tail bound.

### What physical constants make the tail bound finite and computable

A uniform real-box derivative majorant can be constructed from:

- native beam cosine lower bounds, for the tangent chart;
- max |e_i|=E<1 and delta_s=1-E^2, for the scaled exit-plane slopes;
- shape/index/beam upper bounds and h_i lower bounds;
- positive minima of every glass radicand and exit Snell radicand on the same box times lambda in [0,1];
- positive lower bounds for every P_i and Z_i denominator;
- direction upper bounds, e.g. |X_i|<=sqrt(Q_max) in the scaled ray chart;
- the propagated 2x5 position-coefficient block bounds;
- the fixed sample times and the exact frozen arrays L0,R,cbar, with native degree/radian conversions.

The already audited physical/analytic guards address the real-domain positivity requirements. Their numerical values alone do not constitute the high-order derivative bound; the following finite recurrence still has to be applied.

For example, the position-block bound can be propagated without any new geometry cases. With a_i=X_in/H_i, w_i=X_out/Z_i, and plane slope u_i,

    A_i=I+(a_i-w_i)u_i^T/(1-u_i^T a_i),

so bounds for |a_i|,|w_i|,|u_i| and the positive denominator give a bound for A_i. The block recurrence in Section 2 then bounds every Pi_j. The fixed source distance six and prism thickness three are retained.

### A finite projected-jet compiler rather than an unknown Hessian oracle

Use normalized multivariate Taylor coefficients in an auxiliary lambda increment and common optical parameter increments. Retain lambda order four and optical order two. For a scalar circuit node a, write a_nu for its normalized coefficient; all samples share the same parameter monomials.

Addition and multiplication use ordinary coefficient convolution. For an inverse b=1/a, after its constant term is guarded away from zero,

    b_0=1/a_0,
    b_nu=-(1/a_0) sum_(0<beta<=nu) a_beta b_(nu-beta).

For a positive square root b=sqrt(a),

    b_0=sqrt(a_0),
    b_nu=[a_nu-sum_(0<beta<nu) b_beta b_(nu-beta)]/(2b_0).

Here beta<=nu is componentwise; the square-root sum includes all beta with 0<=beta<=nu except beta=0 and beta=nu. These identities work for multi-indices. Interval versions are finite and use the stated positive lower bounds. Trigonometric input jets are explicit, with time factors k/20 and native angle conversions. Matrix-affine geometry is propagated as a block.

At each required coefficient, form the signed projected sums L0 H_(coefficient) R before taking its magnitude. The coefficients must remain joint functions of the common base point (lambda,x), represented by shared arithmetic circuits or a validated common Taylor model with its remainder. Merely giving all samples the same monomial labels is not enough: if their coefficient/base-point ranges have already been independently intervalized, projection afterward does not restore the lost dependence. Plain interval recurrences remain a sound fallback, but are not by themselves a correlation-preserving implementation. A lambda-four, optical-alpha normalized coefficient is converted to the derivative in (22) by the factor 4! alpha!. Thus the M arrays have a finite algebraic construction from guarded primitive operations; they are not a new nonlinear optimization problem.

As a deliberately weaker fallback, a valid per-sample mixed-derivative bound B_(j,alpha,4) implies

    ||D_x^alpha partial_lambda^4 Phi_j||inf
       <= 7 ||L0||inf ||R||inf B_(j,alpha,4).           (23)

Using (23) only for the higher tail preserves all explicit quadratic and cubic cancellations. It is sound but can still be too large. A projected common-parameter bound is preferable. For the augmented or filtered operator, replace the fixed right-operator factor by its corresponding known finite row norm.

### Exact remaining quantitative obligation

Compute bounds for the explicit G2 and K3 terms from (7)-(9) and their cubic analogue. Obtain M_(j,alpha,4) from the guarded projected circuit, or a sufficiently small certified fallback such as (23). Insert (22) into H_ij,K_ai of (11), then test (20).

This is a meaningful finite formula and an explicit list of required physical constants. It still does not show that the resulting number is below the available approximately 0.9492 margin. If the tail bound is loose, the conclusion remains unproved; if it is genuinely large, this particular sufficient certificate can fail. No numerical result about either possibility is asserted here.

## Audit status

The compression, geometry-affine remainder, sign enumeration, and filtered-update identities passed independent mathematical review in correlation_preserving_lag_independent_audit.md. The additional cubic/tail route passed mathematical review, including its finite-derivative qualification and the requirement that projected jet coefficients remain joint functions of the common base point. Its useful numerical bound remains unevaluated. No numerical work is part of this memo.
