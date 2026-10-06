# Certified optical-box exclusion with exact geometry profiling

## Purpose

This addendum proves a reusable sound exclusion certificate and states exact finite-cover stopping conditions. It is a bounded theory development, not an implementation or an assertion that the present 200-sample inverse has already been localized. The missing practical theorem remains a quantitative all-branches candidate cover/exterior gap, as explained in [remaining_constructive_gap.md](remaining_constructive_gap.md).

Let \(x\) denote the fourteen optical chart variables, \(q=(p_x,p_y,g,d)\) the four geometry variables, and \(y\) the fixed stacked record. Use the exact sampled vector map

\[
F(x,q)=M(x)q+c(x).
\]

The optical chart prior \(X\) and geometry prior \(Q\) are the original closed boxes. The noise bound \(\epsilon\) remains symbolic. Physical admissibility always retains the original strict transmitted, axial, normal, and traversal conditions.

**Scoped notation.** Here \(\epsilon\) is the coordinatewise observation-error bound, and \(\gamma\) in Section 6 is an exterior residual gap. That \(\gamma\) is distinct from the regular-branch Jacobian injectivity constant used in [parametric_noise_inverse.md](parametric_noise_inverse.md). The fourteen-dimensional optical chart and the four-dimensional geometry chart are those of [global_inverse_completeness.md](global_inverse_completeness.md).

## 1. Guarded optical boxes and derivative certificates

Let \(B\) be a convex optical box with algebraic center \(x_0\) and infinity-norm radius \(r\). Assume a certificate verifies all six radicands and all required optical normal/axial quantities are strictly positive on \(B\). This concerns refraction, not traversal order. Then the six-root trace defines \(M,c\) analytically throughout \(B\). Its affine position formula also defines \(F(x,q)\) for all \(q\in Q\), even where traversal order fails; evaluating that extension is a valid relaxation for lower bounds, never a declaration that those \(q\) are physical.

A certified derivative enclosure is constructive. Differentiate the rational rotor formulas and the six-root recurrence by the following rules:

\[
D(fg)=f\,Dg+g\,Df,
\qquad
D(f/g)=\frac{g\,Df-f\,Dg}{g^2},
\qquad
D\sqrt{R}=\frac{DR}{2\sqrt{R}}.
\]

Use only positive-denominator products from the physical branch. Every radical denominator is bounded below by its certified positive root margin; every position denominator is bounded below by the certified \(P_j,Z_j\) margins. Propagate exact algebraic or outward-rounded rational intervals through these finitely many formulas, keeping the position expressions affine in \(q\). This gives intervals enclosing every component of \(D_xF(x,q)\) for \(x\in B,q\in Q\). Derivative coefficients affine in \(q\) can be bounded on a tighter geometry polytope if available.

For a fixed \(\lambda\in\mathbb R^{2K}\), define any certified bound

\[
L_{B,\lambda}\ \ge\
\sup_{x\in B,\ q\in Q_B}
\left\|D_xF(x,q)^{\mathsf T}\lambda\right\|_1.
\tag{1}
\]

Here \(Q_B\) is a nonempty compact outer geometry polytope valid for \(B\) and contained in the original geometry box \(Q\). If \(Q_B\) is empty, use its infeasibility certificate instead. If a larger outer polytope is chosen, derivative bounds must be recomputed over that larger set rather than reused from \(Q\). Such a bound is obtained by first enclosing each component of \(D_xF^{\mathsf T}\lambda\), then summing the maximum absolute interval endpoints. Computing the linear combination before taking absolute values can improve the bound. The argument never substitutes a floating-point derivative without an enclosure.

These operations terminate for a particular box once the required strict lower margins have actually been certified. A broad box whose interval tests are inconclusive is subdivided or retained as unresolved. Positivity at its center alone is insufficient.

## 2. A valid common outer geometry polytope

The original geometry box \(Q\) is always an admissible choice for \(Q_B\). A stronger choice incorporates traversal information without assuming its coefficients fixed across \(B\).

Write each exact traversal condition as

\[
g_s(x,q)=a_s(x)^{\mathsf T}q+b_s(x)>0.
\]

This affine form follows from the paired position recurrence and its positive optical denominators. Compute a certified bound

\[
E_s(B)\ \ge\
\sup_{x\in B,\ q\in Q}
\left|g_s(x,q)-g_s(x_0,q)\right|.
\]

Then every physical system with \(x\in B\) has

\[
g_s(x_0,q)\ge -E_s(B).
\]

Thus the compact polytope

\[
Q_B
=Q\cap\{q:g_s(x_0,q)\ge -E_s(B)\text{ for every }s\}
\tag{2}
\]

contains every geometry compatible with any physical optical point in \(B\). Weak inequalities in (2) are intentional outer relaxations. Their feasibility does not prove strict physical feasibility. In particular, a geometry point surviving only on a zero-traversal boundary is never reported as a physical witness.

All coefficients and enclosure bounds can be algebraically encoded. A Farkas certificate for emptiness of \(Q_B\) rejects \(B\) independently of the observation record or \(\epsilon\).

## 3. LP-dual lower certificate

Write \(Q_B=\{q:Aq\le b\}\). Let \(\lambda\) and \(\mu\) satisfy

\[
\|\lambda\|_1\le 1,\qquad
\mu\ge0,\qquad
A^{\mathsf T}\mu=-M(x_0)^{\mathsf T}\lambda.
\tag{3}
\]

Set

\[
\alpha
=\lambda^{\mathsf T}\bigl(c(x_0)-y\bigr)-b^{\mathsf T}\mu.
\tag{4}
\]

Then every physical system with \(x\in B\) obeys

\[
\|F(x,q)-y\|_\infty
\ \ge\ \alpha-L_{B,\lambda}r.
\tag{5}
\]

**Proof.** The observation norm is at least \(\lambda^{\mathsf T}(F(x,q)-y)\). At \(x_0\), conditions (3) imply

\[
\lambda^{\mathsf T}M(x_0)q
=-\mu^{\mathsf T}Aq
\ge-\mu^{\mathsf T}b,
\]

so the pairing is at least \(\alpha\). Along the segment from \(x_0\) to \(x\), convexity of \(B\) and (1) bound the change of that pairing by \(L_{B,\lambda}\|x-x_0\|_\infty\). Every physical \(q\) lies in \(Q_B\). Combining the inequalities proves (5).

Therefore \(B\) is excluded for all symbolic noise levels

\[
\epsilon<\alpha-L_{B,\lambda}r.
\tag{6}
\]

At equality, (5) alone does not exclude compatibility. That boundary must remain unless a stronger certificate resolves it.

For the unaugmented box \(Q=q_c+\prod_i[-R_i,R_i]\), one may avoid explicit \(\mu\) by using

\[
\alpha
=\lambda^{\mathsf T}\bigl(F(x_0,q_c)-y\bigr)
-\sum_i R_i\left|\bigl(M(x_0)^{\mathsf T}\lambda\bigr)_i\right|.
\tag{7}
\]

Optimizing the dual in (3)–(4) gives exactly

\[
\min_{q\in Q_B}\|M(x_0)q+c(x_0)-y\|_\infty.
\]

The primal is a four-geometry-variable LP with an additional residual variable; the dual contains observation/constraint multipliers. Strong LP duality holds for nonempty compact \(Q_B\). Optimality is not required for a valid certificate: any exactly verified dual-feasible pair suffices. Algebraic arithmetic or conservative rational certificates can be used.

A numerically proposed dual pair must not be treated as satisfying the equality in (3) merely because its residual is small. There is an explicit residual correction. For verified \(\|\lambda\|_1\le1\) and \(\mu\ge0\), put

\[
e_{\mathrm{dual}}
=M(x_0)^{\mathsf T}\lambda+A^{\mathsf T}\mu.
\]

Then (5) remains valid after replacing \(\alpha\) by any certified lower bound on

\[
\lambda^{\mathsf T}\bigl(c(x_0)-y\bigr)-b^{\mathsf T}\mu
+\min_{q\in Q}e_{\mathrm{dual}}^{\mathsf T}q.
\]

For \(Q=q_c+\prod_i[-R_i,R_i]\), the last minimum is

\[
e_{\mathrm{dual}}^{\mathsf T}q_c
-\sum_iR_i|e_{\mathrm{dual},i}|.
\]

Using \(Q\) rather than \(Q_B\) is conservative because \(Q_B\subseteq Q\). All coefficient and residual evaluations still require exact arithmetic or outward bounds. This allows a floating or rational proposal to become a valid lower certificate without asserting an unverified equality.

A separate Farkas infeasibility witness is

\[
\mu\ge0,\qquad
A^{\mathsf T}\mu=0,\qquad
b^{\mathsf T}\mu<0.
\]

Such a witness proves \(Q_B\) empty and hence proves no physical system exists in \(B\).

## 4. Treatment of strict optical boundaries

An optical box intersecting a critical branch surface is not silently removed or guarded away. Conditional interval propagation can enclose positive square roots by clipping a radicand enclosure to its nonnegative part. If its upper bound is \(\le0\), the strict-positive physical branch is impossible and the box may be rejected. The same implication applies to an upper bound \(\le0\) for a required \(P_j\) or \(Z_j\), provided all preceding expressions enclose every potentially physical branch.

If a box has potentially positive radicands but no certified positive lower margin, derivative rule (1) may be unbounded or unusable. The derivative-based exclusion lemma does not apply there. Valid options are:

- a branch-preserving algebraic sign certificate excluding the box;
- a different rigorously bounded residual argument that remains valid at the approached boundary;
- subdivision, with no unproved termination promise;
- retention of the exact predicate on that box in the complete algebraic backend.

The same caution applies to strict traversal faces: their weak LP relaxation is an outer set, not an exact replacement. No uniform derivative bound or finite subdivision termination follows merely from the bounded native prior.

## 5. Exact finite-cover completion theorem

Suppose a finite collection of optical boxes covers the entire prior \(X\), including its boundary. Overlaps are allowed. For a specified \(\epsilon\), classify every box as follows:

- **Excluded:** a verified optical-branch contradiction, geometry Farkas certificate, or residual inequality (6) proves no compatible physical point lies there.
- **Validated retained:** a local certificate supplies a sound enclosure of every compatible system with optical coordinates in that box, and any reported feasible witnesses are checked against the complete physical/noise predicate.
- **Unresolved:** retain the exact restricted predicate \(\Phi(x,q,\epsilon)\) with \(x\) in that box, or continue the complete algebraic backend there.

Every compatible system then lies in at least one retained or unresolved box. This follows directly from coverage and sound exclusion; it makes no use of the \(17+1\) initializer being complete.

Local validation plus the finite cover proves all-branch localization only when every nonexcluded box is covered by the validated retained regions, or has been resolved exactly by the backend. Local validation must be an enclosure of all compatible points in its box. A converged Newton iterate, one isolated exact root, or one successful fit is not such an enclosure. For positive noise, a regular compatible set generally contains a continuum; it must not be mislabeled a unique physical system.

For native point-estimation accuracy \(\delta_j\), a sufficient stopping certificate consists of:

1. the complete finite cover and sound classification;
2. at least one verified physical compatible witness, ensuring nonemptiness;
3. verified native coordinate ranges for the union of every surviving enclosure, each of diameter \(\le2\delta_j\).

The midpoint of those ranges then guarantees the requested errors for every compatible system. The ranges may be conservative, so failure of this sufficient test is not itself an impossibility certificate. Two verified compatible systems separated by \(>2\delta_j\) in any coordinate do prove impossibility. If neither outcome is certified and unresolved boxes remain, the procedure has not finished the global inference and must say so.

For an exact complete inverse representation, the surviving exact predicates and their covering boxes already retain every branch; obtaining tractable explicit parameter cells requires finishing the local/backend representation work rather than merely naming those predicates.

For a symbolic \(\epsilon\) interval, each exclusion certificate (6) carries its exact valid range. Partition the \(\epsilon\) axis at certificate thresholds and local-validation endpoints as needed. At each boundary retain the original strict/weak inequality direction. An exclusion established below a threshold is not reused at or above it.

## 6. Conditional finite termination and why it is not yet global practicality

A useful quantitative sufficient condition is available on a compact optical exterior region \(G\) with uniform positive refraction margins. Assume:

- Bounded derivatives give a verified common Lipschitz constant \(L\) for \(F(x,q)\), in the selected optical infinity-norm scaling and observation sup norm, uniformly over \(q\in Q\) and a validated neighborhood of \(G\).
- A common radius \(r_{\mathrm{guard}}>0\) ensures the optical guard and derivative certificates apply to boxes of that radius centered in \(G\).
- The geometry-box-relaxed residual has a proved gap

  \[
  \min_{q\in Q}\|F(x,q)-y\|_\infty\ge\epsilon+\gamma
  \]

  for every \(x\in G\), with \(\gamma>0\).

At a center \(x_0\in G\), obtain a dual certificate within \(\gamma/4\) of the LP optimum. For radius

\[
r\le\min\!\left(r_{\mathrm{guard}},\frac{\gamma}{4L}\right),
\]

(5) excludes the corresponding box with positive slack. Here \(\gamma/(4L)\) is treated as infinity when \(L=0\). Thus finitely many such boxes exclude \(G\). A covering-number bound on a region of diameter \(D\) is of order

\[
\left(
1+\frac{D}{\min(r_{\mathrm{guard}},\gamma/(4L))}
\right)^{14},
\]

up to dimension-dependent constants. Centers can be chosen in \(G\), and covering radii adjusted by fixed factors when using a grid. Here \(D\) is measured in the selected optical infinity norm. This is a conditional bound on the number of covering boxes, not on the total cost of constructing or checking their certificates. In particular, it does not bound the work needed to certify the guards, derivative bounds, exterior gap, or near-optimal LP duals. It is not a data-independent runtime claim.

The profile gap is stronger than mere physical uniqueness because \(Q\) is a relaxation. Stronger common outer polytopes (2) may improve it, but their convergence and resulting gap must be proved rather than assumed. Near excluded physical boundaries, \(r_{\mathrm{guard}}\) may tend to zero and \(L\) may diverge; an unguarded exterior need not satisfy these hypotheses. Exact gauges or remote near-aliases can make \(\gamma\) zero.

Accordingly, the next missing theorem is not another local-rank lemma: it is a data-dependent complete candidate cover plus positive, quantitatively useful exterior gaps, with every boundary and degenerate stratum accounted for. The \(17+1\) construction can propose retained regions and the \(14+4\) reduction supplies the certificates above. Neither currently guarantees that a small number of such regions and certificates covers the full original prior. This addendum states exactly what a successful finite certificate would prove and exactly when the CAD backend remains necessary.

Constructive-center qualification for Section 6: selecting algebraic centers in G requires an effective semialgebraic description of G over the encoded algebraic constants. Without that additional input hypothesis, the displayed covering count is a geometric existence bound, not an effective center-construction claim.
