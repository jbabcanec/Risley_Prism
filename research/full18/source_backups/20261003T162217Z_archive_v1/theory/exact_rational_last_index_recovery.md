# Exact rational recovery of the last squared index

## Status

The preceding sagittal reduction gives a degree-at-most-64 candidate polynomial for G=n_3^2. A stronger two-chart construction now supplies a degree-one subresultant and therefore an exact rational formula for G on a generically valid chart. The necessary single-witness coprimality and full-degree check has been certified by outward-rounded integer interval arithmetic. An independent 8192-bit rerun and a separately implemented original-row, division-free pseudo-remainder calculation verify the certificate; see sagittal_subresultant_certificate_audit.md.

This eliminates one additional optical unknown beyond the existing four affine geometry variables. It is an exact finite-wedge reduction from fourteen searched optical coordinates to thirteen, using the first six original paired screen samples. It does not recover the remaining thirteen optical coordinates, prove full18 global uniqueness, or imply useful noise conditioning.

## 1. Explicit conditional reconstruction theorem

Retain the thirteen optical coordinates x consisting of n_1,n_2, the three wedges, three phases, three speeds, and two beam coordinates. From a trial x, trace the first two direction stages and their affine position maps at timestamps k=0,...,5:

p_k=T_k b+c_k g+e_k,
q_k = external transverse direction entering prism 3.

These quantities use no hidden source position or distances. Position-dependent traversal constraints are carried as affine geometry inequalities. Let y_k be the measured screen positions, u_k the trial third-prism slope, alpha_k=|q_k|^2, and h_k a formal square root with

h_k^2=G-alpha_k.

For each k define v_k=q_k+h_k u_k and delta_k=[q_k,u_k], where [a,b] is the planar determinant. The exact sagittal identity is

[y_k-p_k,v_k]=(3+d)delta_k.

Its augmented affine-geometry row is

R_k(G)=[-[T_k e_x,v_k], -[T_k e_y,v_k], -[c_k,v_k], -delta_k,
         [y_k-e_k,v_k]-3 delta_k].

Use the fixed charts I=(0,1,2,3,4) and J=(0,1,2,3,5). For either chart C, form

Delta_C=det(R_k)_(k in C),
P_C(G)=product over all 32 formal sign choices of Delta_C.

Reduce h_k^2=G-alpha_k. These are polynomials of degree at most 64. Keep the formal sign choices separate in this construction even when some alpha values happen to coincide: the resulting coefficients remain fixed analytic functions across such loci. Duplicate radicals can make a polynomial identically zero, but cannot invalidate the necessary-root relation.

Compute the fixed-degree first polynomial subresultant

S_1(G)=a(x,y)G+b(x,y)

of P_I and P_J, using the degree-64 Sylvester construction. Its coefficients are polynomial expressions in the two norm polynomials' coefficients. The universal subresultant Bezout identity gives polynomial U,V such that

S_1=U P_I+V P_J.

Every exact physical candidate has P_I(G)=P_J(G)=0. Therefore:

- If a!=0, its squared last index is uniquely forced to be **G=−b/a**.
- If a=0 and b!=0, the retained optical trial has no compatible exact physical solution.
- If a=b=0, this chart is unresolved; use a grouped single-chart norm, another chart, or the original exact graph. Do not reject it solely on this condition.

For a!=0, require G in [169/100,81/25] and set n_3=sqrt(G)&gt;0. Then recover geometry from a nonsingular four-row sagittal pivot and retain all original bounds and strict traversal constraints. Such a pivot must exist at any physical solution on a!=0, as proved below. Validate all 200 paired observations with the actual positive-root Snell model. The formula and pivot give unique conditional candidates, not a standalone feasibility certificate. Deficient-rank geometry fibers remain in the a=b=0 fallback, rather than being silently discarded.

### All five eliminated variables are uniquely reconstructed on this chart

At a true common root G*, differentiate the universal Bezout identity with respect to the polynomial variable G, keeping its coefficients fixed. Since P_I=P_J=0 there,

a=U(G*)P_I'(G*)+V(G*)P_J'(G*).

If a!=0, at least one norm polynomial has a simple root at G*. All signed radicals are analytic near G*, because G*-alpha_k&gt;0. Its physical determinant factor must therefore also have a simple root. The earlier column-operation identity gives

Delta_C'(G*)=det[A_C | partial_G F_C] !=0.

Thus A_C has rank four. One of the two fixed charts contains a nonsingular four-row geometry pivot. Once G is fixed, these rows uniquely fix b_x,b_y,g,d. Consequently every physical solution on a!=0 has all five omitted coordinates uniquely determined by the retained thirteen optics and the measured data. This conditional uniqueness does not imply uniqueness of the retained optics themselves.

## 2. What is rational, and what is still unknown

The formula G=−b/a is rational in coefficients computed from the measured positions and the upstream optical trace at x. Upstream direction roots and rotor evaluations are already determined by the trial thirteen coordinates; the formula does not pretend that they were measured or supplied from truth.

This is not a rational inverse in screen data alone, and n_3 itself is obtained by one positive square root. The remaining thirteen optical coordinates still need a constructive recovery or search method. The global full18 fiber can still contain different retained-optics explanations.

Only six original paired samples are used to construct the index formula. Their short time span is not claimed to provide favorable conditioning. The other 194 samples remain essential validation and rejection information for a practical inverse.

## 3. The certified nonzero witness

Use the single fully admissible witness from exact_sagittal_index_elimination.md:

n_j=3/2, tan(a_j)=1/10, phi_j=0,
incident slopes=(1/3,1/10), source=(1,2), g=10, d=100,
per-sample rotor multipliers=(15+8i)/17, (4+3i)/5, (3+4i)/5.

The earlier independent interval checks certify all original bounds and all 200 physical/traversal guards. The six alpha_k values used by the new construction are distinct.

The dedicated certificate in sagittal_two_chart_coprimality.md establishes, for the exact witness coefficients:

1. Both P_I and P_J have degree exactly 64.
2. Their exact common physical root is G*=9/4.
3. After deflating that root, the degree-63 quotients are coprime.

The proof works in X=G−9/4. An exact affine-geometry column operation replaces the determinant's final column by r_k(h_k−H_k*). This operation leaves the original augmented determinant unchanged and proves the zero constant coefficient algebraically. It is used only in the proof witness, not as a reconstruction step using unknown geometry.

The norm computation uses radical-mask polynomial arithmetic and successive sign-conjugate multiplication. Rigorous positive leading-coefficient intervals certify full degree64, avoiding a degree-dropping specialization. The exact deflated quotients have an outward-rounded dyadic polynomial Euclidean chain with remainder degrees

62,61,...,1,0.

All 63 leading pivots exclude zero. After explicitly recorded power-of-two rescalings, the final constant lies between

6519538297218639/10^16 and 6519538297218640/10^16.

It is therefore nonzero. This proves gcd(P_I,P_J)=G−9/4 at the witness, including absence of extra common complex roots introduced by any negative-radical norm branches. It is stronger than merely checking that the positive physical root is locally unique.

The proof check passes at 1024 and 4096 bits using outward integer arithmetic. No rounded coefficient is treated as exact, no small remainder is discarded, and no numerical parameter campaign is performed. Full artifacts and exact interval endpoints are saved beside the proof program.

For two degree-64 polynomials with gcd of degree exactly one, the first subresultant has degree exactly one. Hence a!=0 at this physical witness.

## 4. Generic validity on the entire physical prior

The domain theorem in physical_domain_contraction.md proves that the full physical interior O is connected. Evaluate the fixed ungrouped norm coefficients at the exact generated observation y=F(theta), and let a(theta) be the first-subresultant leading coefficient.

This is a real-analytic semialgebraic function on O. The witness proves it is not identically zero. Therefore

Z={theta in O:a(theta)=0}

has dimension at most 17. No irreducibility shortcut or uncontrolled specialization is required: the universal Bezout identity gives

a(theta)n_3(theta)^2+b(theta)=0

throughout O, and analyticity propagates the nonzero-chart genericity over the connected domain.

Let W be the compact weak closure and Fbar its continuous semialgebraic observation extension. Define

B=(W\\O) union closure_W Z,
E=Fbar(B).

The existing boundary theorem gives dim B&lt;=17 and therefore dim E&lt;=17. The model observation image has dimension 18. For every exact record outside E, every physical preimage is interior and has a!=0. Thus the same fixed rational-index construction applies at every compatible retained-thirteen-coordinate optical point for generic exact records over the whole original prior.

This is not an effective certification that an arbitrary supplied record lies outside E. Arbitrary-data completeness retains the a=b=0 fallback. The theorem does not prove that different retained optical points cannot explain the same data.

## 5. Positive noise and numerical caution

For positive observation noise, feasible indices are generally intervals. Plugging noisy positions into the exact rational formula does not create an exact inverse or justify discarding other indices.

The independently audited sagittal determinant tube remains a sound necessary screening method. It admits an exact one-dimensional interval description using scalar polynomial root isolation of degree at most 64 for each fixed error allowance. Full noisy feasibility still uses the original affine geometry inequalities and every physical branch constraint.

An interval bound on the rational formula can also be valid if its denominator is certified away from zero over the entire relevant observation-error set, with all shared data dependencies preserved or safely over-enclosed. No useful such conditioning bound is supplied here. The subresultant denominator can be small near exceptional or nearly common-conjugate branches, even when the physical forward problem itself is regular.

## 6. Deliverable boundary

Established reduction: on a verified, generically full-prior chart, recover n_3^2 by a fixed rational subresultant expression, then recover or describe all four geometry variables by exact affine methods. This leaves thirteen optical unknowns.

Still open: a useful end-to-end recovery of those thirteen coordinates from the original record, practical full-prior exclusion of alternate explanations, useful finite-noise error constants, and global uniqueness. The new identity and rational chart provide a concrete exact elimination step for that work, rather than a completed full18 solver.

## 7. Optional correction composition

The concise composition formulas and guards are in sagittal_corrector_composition.md, independently audited in sagittal_reduced_corrector_composition_audit.md. They define Phi_y=pi13 T_y composed with E_y and give both state and shared-record derivatives. No convergence radius or noise guarantee is inherited automatically; an invariant domain, new derivative bounds, and full-record validation remain required. No additional n_2 elimination witness was attempted.
