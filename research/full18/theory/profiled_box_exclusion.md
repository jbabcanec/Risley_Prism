# Certified optical-box exclusion with exact geometry profiling

## Purpose

This addendum proves a reusable sound exclusion certificate and states exact finite-cover stopping conditions. It is a bounded theory development, not an implementation or an assertion that the present 200-sample inverse has already been localized. The missing practical theorem remains a quantitative all-branches candidate cover/exterior gap, as explained in remaining_constructive_gap.md.

Let x denote the fourteen optical chart variables, q=(p_x,p_y,g,d) the four geometry variables, and y the fixed stacked record. Use the exact sampled vector map

F(x,q)=M(x)q+c(x).

The optical chart prior X and geometry prior Q are the original closed boxes. The noise bound epsilon remains symbolic. Physical admissibility always retains the original strict transmitted, axial, normal, and traversal conditions.

## 1. Guarded optical boxes and derivative certificates

Let B be a convex optical box with algebraic center x_0 and infinity-norm radius r. Assume a certificate verifies all six radicands and all required optical normal/axial quantities are strictly positive on B. This concerns refraction, not traversal order. Then the six-root trace defines M,c analytically throughout B. Its affine position formula also defines F(x,q) for all q in Q, even where traversal order fails; evaluating that extension is a valid relaxation for lower bounds, never a declaration that those q are physical.

A certified derivative enclosure is constructive. Differentiate the rational rotor formulas and the six-root recurrence by the following rules:

D(fg)=f Dg+g Df,
D(f/g)=(g Df-f Dg)/g^2,
D sqrt(R)=DR/(2sqrt(R)).

Use only positive-denominator products from the physical branch. Every radical denominator is bounded below by its certified positive root margin; every position denominator is bounded below by the certified P_j,Z_j margins. Propagate exact algebraic or outward-rounded rational intervals through these finitely many formulas, keeping the position expressions affine in q. This gives intervals enclosing every component of D_xF(x,q) for x in B,q in Q. Derivative coefficients affine in q can be bounded on a tighter geometry polytope if available.

For a fixed lambda in R^(2K), define any certified bound

L_(B,lambda) >= sup_(x in B,q in Q_B)
                      ||D_xF(x,q)^T lambda||_1,       (1)

where Q_B is a nonempty compact outer geometry polytope valid for B and contained in the original geometry box Q. If Q_B is empty, use its infeasibility certificate instead. If a larger outer polytope is chosen, derivative bounds must be recomputed over that larger set rather than reused from Q. Such a bound is obtained by first enclosing each component of D_xF^T lambda, then summing the maximum absolute interval endpoints. Computing the linear combination before taking absolute values can improve the bound. The argument never substitutes a floating-point derivative without an enclosure.

These operations terminate for a particular box once the required strict lower margins have actually been certified. A broad box whose interval tests are inconclusive is subdivided or retained as unresolved. Positivity at its center alone is insufficient.

## 2. A valid common outer geometry polytope

The original geometry box Q is always an admissible choice for Q_B. A stronger choice incorporates traversal information without assuming its coefficients fixed across B.

Write each exact traversal condition as

g_s(x,q)=a_s(x)^Tq+b_s(x)>0.

This affine form follows from the paired position recurrence and its positive optical denominators. Compute a certified bound

E_s(B)>=sup_(x in B,q in Q)|g_s(x,q)-g_s(x_0,q)|.

Then every physical system with x in B has

g_s(x_0,q)>=-E_s(B).

Thus the compact polytope

Q_B=Q intersect {q:g_s(x_0,q)>=-E_s(B) for every s}    (2)

contains every geometry compatible with any physical optical point in B. Weak inequalities in (2) are intentional outer relaxations. Their feasibility does not prove strict physical feasibility. In particular, a geometry point surviving only on a zero-traversal boundary is never reported as a physical witness.

All coefficients and enclosure bounds can be algebraically encoded. A Farkas certificate for emptiness of Q_B rejects B independently of the observation record or epsilon.

## 3. LP-dual lower certificate

Write Q_B={q:Aq<=b}. Let lambda and mu satisfy

||lambda||_1<=1, mu>=0, A^Tmu=-M(x_0)^Tlambda.         (3)

Set

alpha=lambda^T(c(x_0)-y)-b^Tmu.                       (4)

Then every physical system with x in B obeys

||F(x,q)-y||_infinity >= alpha-L_(B,lambda)r.          (5)

Proof. The observation norm is at least lambda^T(F(x,q)-y). At x_0, conditions (3) imply

lambda^T M(x_0)q=-mu^TAq>=-mu^Tb,

so the pairing is at least alpha. Along the segment from x_0 to x, convexity of B and (1) bound the change of that pairing by L_(B,lambda)||x-x_0||_infinity. Every physical q lies in Q_B. Combining the inequalities proves (5).

Therefore B is excluded for all symbolic noise levels

epsilon<alpha-L_(B,lambda)r.                         (6)

At equality, (5) alone does not exclude compatibility. That boundary must remain unless a stronger certificate resolves it.

For the unaugmented box Q=q_c+product_i[-R_i,R_i], one may avoid explicit mu by using

alpha=lambda^T(F(x_0,q_c)-y)
      -sum_i R_i |(M(x_0)^Tlambda)_i|.                (7)

Optimizing the dual in (3)-(4) gives exactly

min_(q in Q_B)||M(x_0)q+c(x_0)-y||_infinity.

The primal is a four-geometry-variable LP with an additional residual variable; the dual contains observation/constraint multipliers. Strong LP duality holds for nonempty compact Q_B. Optimality is not required for a valid certificate: any exactly verified dual-feasible pair suffices. Algebraic arithmetic or conservative rational certificates can be used.

A numerically proposed dual pair must not be treated as satisfying the equality in (3) merely because its residual is small. There is an explicit residual correction. For verified ||lambda||_1<=1 and mu>=0, put

e_dual=M(x_0)^T lambda+A^T mu.

Then (5) remains valid after replacing alpha by any certified lower bound on

lambda^T(c(x_0)-y)-b^T mu+min_(q in Q) e_dual^T q.

For Q=q_c+product_i[-R_i,R_i], the last minimum is e_dual^T q_c-sum_i R_i |e_dual,i|. Using Q rather than Q_B is conservative because Q_B is contained in Q. All coefficient and residual evaluations still require exact arithmetic or outward bounds. This allows a floating or rational proposal to become a valid lower certificate without asserting an unverified equality.

A separate Farkas infeasibility witness is mu>=0, A^Tmu=0, b^Tmu<0. Such a witness proves Q_B empty and hence proves no physical system exists in B.

## 4. Treatment of strict optical boundaries

An optical box intersecting a critical branch surface is not silently removed or guarded away. Conditional interval propagation can enclose positive square roots by clipping a radicand enclosure to its nonnegative part. If its upper bound is<=0, the strict-positive physical branch is impossible and the box may be rejected. The same implication applies to an upper bound<=0 for a required P_j or Z_j, provided all preceding expressions enclose every potentially physical branch.

If a box has potentially positive radicands but no certified positive lower margin, derivative rule(1) may be unbounded or unusable. The derivative-based exclusion lemma does not apply there. Valid options are:

- a branch-preserving algebraic sign certificate excluding the box;
- a different rigorously bounded residual argument that remains valid at the approached boundary;
- subdivision, with no unproved termination promise;
- retention of the exact predicate on that box in the complete algebraic backend.

The same caution applies to strict traversal faces: their weak LP relaxation is an outer set, not an exact replacement. No uniform derivative bound or finite subdivision termination follows merely from the bounded native prior.

## 5. Exact finite-cover completion theorem

Suppose a finite collection of optical boxes covers the entire prior X, including its boundary. Overlaps are allowed. For a specified epsilon, classify every box as follows:

- Excluded: a verified optical-branch contradiction, geometry Farkas certificate, or residual inequality (6) proves no compatible physical point lies there.
- Validated retained: a local certificate supplies a sound enclosure of every compatible system with optical coordinates in that box, and any reported feasible witnesses are checked against the complete physical/noise predicate.
- Unresolved: retain the exact restricted predicate Phi(x,q,epsilon) with x in that box, or continue the complete algebraic backend there.

Every compatible system then lies in at least one retained or unresolved box. This follows directly from coverage and sound exclusion; it makes no use of the 17+1 initializer being complete.

Local validation plus the finite cover proves all-branch localization only when every nonexcluded box is covered by the validated retained regions, or has been resolved exactly by the backend. Local validation must be an enclosure of all compatible points in its box. A converged Newton iterate, one isolated exact root, or one successful fit is not such an enclosure. For positive noise, a regular compatible set generally contains a continuum; it must not be mislabeled a unique physical system.

For native point-estimation accuracy delta_j, a sufficient stopping certificate consists of:

1. the complete finite cover and sound classification;
2. at least one verified physical compatible witness, ensuring nonemptiness;
3. verified native coordinate ranges for the union of every surviving enclosure, each of diameter<=2delta_j.

The midpoint of those ranges then guarantees the requested errors for every compatible system. The ranges may be conservative, so failure of this sufficient test is not itself an impossibility certificate. Two verified compatible systems separated by>2delta_j in any coordinate do prove impossibility. If neither outcome is certified and unresolved boxes remain, the procedure has not finished the global inference and must say so.

For an exact complete inverse representation, the surviving exact predicates and their covering boxes already retain every branch; obtaining tractable explicit parameter cells requires finishing the local/backend representation work rather than merely naming those predicates.

For a symbolic epsilon interval, each exclusion certificate (6) carries its exact valid range. Partition the epsilon axis at certificate thresholds and local-validation endpoints as needed. At each boundary retain the original strict/weak inequality direction. An exclusion established below a threshold is not reused at or above it.

## 6. Conditional finite termination and why it is not yet global practicality

A useful quantitative sufficient condition is available on a compact optical exterior region G with uniform positive refraction margins. Assume:

- Bounded derivatives give a verified common Lipschitz constant L for F(x,q), in the selected optical infinity-norm scaling and observation sup norm, uniformly over q in Q and a validated neighborhood of G.
- A common radius r_guard>0 ensures the optical guard and derivative certificates apply to boxes of that radius centered in G.
- The geometry-box-relaxed residual has a proved gap

min_(q in Q)||F(x,q)-y||_infinity >= epsilon+gamma

for every x in G, with gamma>0.

At a center x_0 in G, obtain a dual certificate within gamma/4 of the LP optimum. For radius r<=min(r_guard,gamma/(4L)), (5) excludes the corresponding box with positive slack. Here gamma/(4L) is treated as infinity when L=0. Thus finitely many such boxes exclude G. A covering-number bound on a region of diameter D is of order

(1+D/min(r_guard,gamma/(4L)))^14,

up to dimension-dependent constants. Centers can be chosen in G, and covering radii adjusted by fixed factors when using a grid. Here D is measured in the selected optical infinity norm. This is a conditional bound on the number of covering boxes, not on the total cost of constructing or checking their certificates. In particular, it does not bound the work needed to certify the guards, derivative bounds, exterior gap, or near-optimal LP duals. It is not a data-independent runtime claim.

The profile gap is stronger than mere physical uniqueness because Q is a relaxation. Stronger common outer polytopes (2) may improve it, but their convergence and resulting gap must be proved rather than assumed. Near excluded physical boundaries, r_guard may tend to zero and L may diverge; an unguarded exterior need not satisfy these hypotheses. Exact gauges or remote near-aliases can make gamma zero.

Accordingly, the next missing theorem is not another local-rank lemma: it is a data-dependent complete candidate cover plus positive, quantitatively useful exterior gaps, with every boundary and degenerate stratum accounted for. The 17+1 construction can propose retained regions and the 14+4 reduction supplies the certificates above. Neither currently guarantees that a small number of such regions and certificates covers the full original prior. This addendum states exactly what a successful finite certificate would prove and exactly when the CAD backend remains necessary.
