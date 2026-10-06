# Exact optical-only feasibility from geometry circuits

This is an algebraic elimination of the four geometric unknowns, not a new data protocol. It supplements elimination.md. No circuit enumeration or recovery experiment is performed here.

## 1. Remove both source coordinates without dividing by small coefficients

For one axis and fixed optical coordinates q, write the already proved ordered response as

\[
F_k=A_k p+B_k+C_k g+E_k d,\qquad A_k>0.
\]

Let Y_k=y_k-B_k. The observation strips imply

\[
\frac{Y_k-C_kg-E_kd-\eta}{A_k}\le p\le
\frac{Y_k-C_kg-E_kd+\eta}{A_k},\qquad -5\le p\le5.
\]

The source interval intersection is nonempty precisely when all lower endpoints do not exceed all upper endpoints. Positivity of A gives the equivalent division-free halfplanes

\[
-C_kg-E_kd\le5A_k+\eta-Y_k,
\quad C_kg+E_kd\le5A_k+\eta+Y_k,
\tag{1}
\]
\[
-(C_kA_l-C_lA_k)g-(E_kA_l-E_lA_k)d
\le\eta(A_k+A_l)-Y_kA_l+Y_lA_k.
\tag{2}
\]

Use every ordered sample pair k,l on each axis, and add the four bounds for g in[2,15], d in[50,200]. Equations(1) compare observation intervals with the native source interval; (2) compares observation lower and upper endpoints. The native lower<=upper condition is tautological. Thus no source constraint is omitted.

All dependence on q is in the finite halfplane coefficients. Multiplication by positive A_k or A_k A_l preserves the inequalities. The representation avoids source normalization when A approaches zero. Coefficients are rational functions of lifted optical variables with strictly positive denominators; further positive denominator clearing, or coefficient-defining auxiliary equations, gives polynomial-in-lifted-optics constraints.

## 2. Exhaustive infeasibility certificates in two dimensions

Let the resulting halfplanes be n_j^T x<=h_j, x=(g,d). Their intersection is empty if and only if at least one of the following bad circuits occurs.

1. A zero normal n_j=0 has h_j<0.
2. Two nonzero normals are opposing collinear, and there exist positive weights lambda1,lambda2 with lambda1 n1+lambda2 n2=0 and lambda1 h1+lambda2 h2<0.
3. Three normals have rank2. Put

\[
\lambda=(\det(n_2,n_3),\det(n_3,n_1),\det(n_1,n_2)).
\tag{3}
\]

The nonzero entries of lambda must have a common sign, so one of lambda or -lambda is nonnegative. For that nonnegative orientation require sum lambda_j h_j<0. Zero coefficients reduce the support to a smaller circuit, already covered above.

**Sufficiency.** Each circuit supplies lambda>=0, sum lambda_j n_j=0 and sum lambda_j h_j<0. Multiplying feasible inequalities by lambda and summing would give0<0, a contradiction.

**Necessity.** For a finite infeasible family of closed halfplanes inR2, the finite-dimensional convex intersection theorem supplies an infeasible subfamily of at most3. Equivalently, a minimal nonnegative dependence with negative right side has support at most3. A support1 certificate has zero normal. A support2 certificate uses opposing collinear normals. For a minimal support3 certificate the normals must have rank2; otherwise a support1 or2 certificate exists. Its nullspace is one-dimensional and the determinant vector(3) spans it. This yields precisely the three cases, including every rank stratum.

The geometry rectangle is part of the family. Therefore native bounds participate in the certificates, and no additional boundedness assumption is needed.

## 3. Exact consequence and implementation boundary

For any strictly physical optical q in its native domain, all unknown geometry exists if and only if none of these bad circuits occurs. The native compatible set is therefore represented by:

* the exact optical lift and native optical constraints;
* absence of every bad one-, two-, or three-row geometry circuit;
* source reconstruction by the interval intersections after choosing any point of the remaining geometry polygon.

This is an exact finite-noise predicate in optical14 alone. It does not imply a finite list of optical solutions, global uniqueness, or an efficient symbolic expansion. There can be tens of thousands of halfplanes, making explicit triple enumeration prohibitive.

A principled implementation is lazy: solve the small conditional/outer LP, obtain a sparse active dual/circuit proposal, and verify its coefficient inequalities on an optical region. A verified strict circuit rejects the region; an inconclusive or sign-changing circuit requires refinement or another certificate. Floating infeasibility alone is insufficient. Native angle readout, strict optical boundaries and complete global cover bookkeeping remain as in algebraic_record.md and global_initialization.md.
