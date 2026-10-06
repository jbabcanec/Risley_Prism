# Conditional subresultant route to rational last-index recovery

## Current status

This was the conditional planning note for the degree-one subresultant route. Its hypothesis has now been proved and independently audited. The completed theorem and reconstruction algorithm are in exact_rational_last_index_recovery.md; the witness certificate is in sagittal_two_chart_coprimality.md and its completed independent audit is in sagittal_subresultant_certificate_audit.md. The conditional derivation below is retained as background.

Use two fixed five-sample charts I=(0,1,2,3,4) and J=(0,1,2,3,5). Let P_I(G), P_J(G) be their formally ungrouped 32-sign norm polynomials, using the explicit augmented rows of the established sagittal reduction. They have degree at most 64. A repeated incoming squared norm may make an ungrouped polynomial vanish identically; this is harmless for the necessary-root relation below and is retained as an exceptional case.

The benefit of fixed ungrouped polynomials is that every coefficient is a fixed real-analytic semialgebraic function of the retained thirteen optical coordinates and measured screen positions, including through equal-radicand loci. Grouped field norms remain the preferable nonzero test for the earlier single-chart fallback.

## 1. A directly checkable rational reconstruction chart

Construct the first polynomial subresultant, using the fixed degree-64 Sylvester construction,

S_1(G)=a G+b.

The subresultant has a universal polynomial Bezout representation in P_I and P_J. Every physical solution satisfies both norm equations at its true G=n_3^2. Therefore it also satisfies

a G+b=0.

Consequently, whenever the computed coefficient a is nonzero,

**G=−b/a.**

This statement is conditional on a!=0 at the actual retained optical trial and observed data, and is exact. It does not require first claiming a generic theorem. It also immediately excludes every other common norm root at that trial. The resulting number still requires its original index bounds, positive-root Snell branch, affine geometry solve, and all 200 paired observations to be checked.

The expression is rational in the norm coefficients. Those coefficients use the observed positions and upstream ray quantities determined by the remaining thirteen optical coordinates. Thus this would eliminate the last index using arithmetic and a fixed subresultant after tracing only the first two prisms. It would not recover all thirteen retained optical coordinates or produce a data-only rational full18 inverse.

## 2. The witness hypothesis needed for a generic theorem

At the single established rational-rotor physical witness, G*=9/4 and each P has the exact factor G−G*. Compute the deflated polynomials Q_I=P_I/(G−G*) and Q_J=P_J/(G−G*).

A sufficient certificate is:

1. Each P has a nonzero degree-64 leading coefficient, so the specialization preserves the fixed degrees.
2. The ordinary resultant Res(Q_I,Q_J) is nonzero.

The second item includes both quotient simplicity and absence of all extra common complex roots, including roots introduced by unrelated negative-radical norm branches. It therefore addresses the actual norm-conjugacy issue rather than testing only physical positive roots.

Together these conditions show gcd(P_I,P_J)=G−G* at the witness. The first subresultant is then a nonzero degree-one polynomial, hence a!=0 at that point. A floating-point polynomial gcd, small numerical residual, or unvalidated singular-value estimate is insufficient. Exact outward coefficient/resultant intervals or an equivalent exact algebraic certificate are required.

Degree preservation matters. A degree-dropping specialization can remove a generic common factor through a root escaping to infinity. A gcd-one conclusion at such a specialization alone would not prove generic rational recovery.

## 3. Generic implication if the certificate succeeds

Let theta vary in the entire connected physical interior O from physical_domain_contraction.md, and evaluate the subresultant leading coefficient at the exact generated record y=F(theta). The resulting a(theta) is analytic and semialgebraic on O. If the witness certificate proves a!=0 somewhere, its zero set has dimension at most 17.

Let W be the compact weak closure and Fbar its observation extension. Enlarge the bad set by the prior/physical boundary and its closure:

B=(W\\O) union closure_W{theta in O:a(theta)=0},
E=Fbar(B).

Then E is compact semialgebraic of dimension at most 17. Every exact record outside E has every physical preimage on the rational-index chart. Thus the same fixed subresultant formula would recover n_3^2 at every compatible thirteen-coordinate optical point for generic records across the whole prior.

This conclusion remains conditional on the pending certificate. It also would not certify that an arbitrary supplied record avoids E. For arbitrary data, a=0 retains the previous grouped-norm and exact-graph fallbacks; it is not an exclusion criterion.
