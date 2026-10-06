# The remaining gap to a tractable certified inverse

## Assessment

The single consequential missing theorem is a quantitative, all-branches optical candidate-cover/exterior-exclusion theorem for the exact finite-angle record. The current theory is already globally complete and terminating in principle. What it does not prove is that a reasonably sized collection of data-derived candidate neighborhoods contains every compatible system, with the remaining fourteen-dimensional optical region excluded by inexpensive certificates.

The exact14-optical-plus4-geometry split removes four nuisance dimensions but does not control the topology or conditioning of the remaining optical fiber. The17+1 construction supplies a local, leading-order candidate mechanism; it is not a global finite-record cover. Certifying its individual candidates cannot rule out remote branches, harmonic/frequency degeneracies, or admissible sequences approaching excluded physical boundaries. Uniform unique recovery over the full prior is false because of the exact gauges, so the missing theorem must allow certified ambiguity and positive-dimensional outputs rather than demand one recovered point everywhere.

A sufficient additional theorem would provide, for each record and symbolic epsilon, a finite certified optical cover C and an explicit positive residual gap on its complement, or a verified ambiguity witness when such localization is impossible. Its constants must depend on stated branch/excitation/conditioning margins, and every exceptional stratum and boundary-approaching region must either remain represented or be excluded by a valid bound. Neither of the available structural reductions currently proves this.

## A bounded next derivation: exact LP-dual optical-box exclusion

The following lemma is already provable from the existing affine geometry structure and supplies a concrete theory-first next step. It does not assume any hidden parameter known.

Write the stacked exact map as F(x,q)=M(x)q+c(x), with x in R^14 and geometry q in its complete prior box Q=q_c+[-R_1,R_1]x...x[-R_4,R_4]. Let B be a convex optical box centered at x_0, of infinity-norm radius r. Assume its refractive square-root, axial-direction and normal-direction margins are uniformly positive, so the exact optical coefficients are differentiable throughout B. The affine position formula extends to every q in Q even if an extended point fails traversal order; this is only a relaxation for lower bounds, not permission to call that point physical.

Choose any vector lambda in R^(2K) with ||lambda||_1<=1. Define

A_lambda=lambda^T(F(x_0,q_c)-y)
         -sum_(i=1)^4 R_i |(M(x_0)^T lambda)_i|,

L_(B,lambda)>=sup_(x in B,q in Q)
                    ||D_x F(x,q)^T lambda||_1.

Then every physical system with x in B and q in Q satisfies

||F(x,q)-y||_infinity >= A_lambda-L_(B,lambda)r.       (1)

Proof: the observation norm dominates its pairing with lambda. The minimum of the centered affine pairing over the geometry box is exactly A_lambda. The mean-value theorem bounds the change of that pairing across B by L_(B,lambda)r. Physical traversal restrictions only shrink Q, so relaxing them cannot invalidate the lower bound.

Consequently B is excluded for every symbolic noise level

epsilon<A_lambda-L_(B,lambda)r.                      (2)

The best A_lambda at x_0 is precisely the dual of the four-variable geometry residual LP

min_(q in Q)||M(x_0)q+c(x_0)-y||_infinity.

One does not need the optimal dual to obtain a sound certificate. Algebraic x_0, exact root isolation, and exact LP arithmetic give exact certificate values; rigorous interval bounds give conservative alternatives. Derivatives and their bounds are generated from the explicit six-root trace. Root lower margins are essential because derivatives may diverge toward a critical branch boundary.

Thus the bounded mathematical development is: derive explicit certified bounds L_(B,lambda) from the six-root recurrences, combine them with exact LP-dual witnesses, and state the resulting epsilon-dependent exclusion certificates. Add interval Schur/Newton certificates only within retained regular candidate boxes. All boxes lacking either certificate remain in the complete backend; no branch is discarded heuristically.

## What would make this provably useful

On a compact guarded optical region of diameter D, suppose an explicit uniform Lipschitz bound L holds and the geometry-relaxed profile residual has gap at least gamma>0 above the target epsilon outside the retained candidate cover. Boxes of radius below a fixed multiple of gamma/L then admit residual exclusion, yielding a covering count of order

(1+LD/gamma)^14

in the chosen fixed coordinate scaling. This is a conditional quantitative improvement over unconstrained algebraic decomposition, not an assertion that the required ratio is moderate for the present problem. LP relaxation can also leave false candidates that stricter geometry constraints would remove, so the stated profile gap is a sufficient condition and must actually be proved.

The current hypotheses do not furnish global gamma, finite useful derivative bounds near every excluded optical boundary, or a finite-angle remainder theorem that makes the17+1 initializer into a complete cover. Establishing such a cover/gap is the real missing mathematical step. More local-rank determinants or another formal initializer do not close it. A practical performance claim must wait for that theorem or for an actual record-specific finite exclusion certificate.
