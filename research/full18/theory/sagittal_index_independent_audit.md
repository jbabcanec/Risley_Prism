# Independent audit: last-prism sagittal index elimination

## Verdict

The exact sagittal identity, conditional finite index elimination, grouped-radical norm construction, degree bound 64, and derivative-witness argument are correct. The supplied rational-rotor witness passes an independently implemented rigorous interval check. This establishes a nonempty admissible, nondegenerate exact-data chart. Combined with the audited connected-domain and compact-closure results, it establishes the whole-prior generic exact-data reduction described in Section 6. It does not establish a finite-index reduction on every optical stratum, nor does it extend to a finite set of indices under positive observation noise.

The audit inspected `vector_inverse.md`, `reduced_algebraic_inverse_addendum.md`, and `proof_checks/sagittal_index_witness.py`. A separate reproducible implementation is in `proof_checks/sagittal_index_independent_check.py`.

## 1. Exact last-prism identity

Write the incoming external transverse direction as q, the internal axial momentum as H=sqrt(G−|q|²), the exit-plane tilt as u, the entrance position as p, and the screen position as y. Put L=3+d and P=H−u·q. Tangential Snell refraction gives B=q+(H−Z)u for outgoing direction (B,Z).

On the physical branch, the two-segment intersection formula is

- a=(3+u·p)/P,
- y−p=a q+(L−aH)B/Z.

Consequently y−p=sq+Ku, with

- s=a+(L−aH)/Z,
- K=(L−aH)(H/Z−1).

Direct cancellation gives Hs−K=L. Therefore

    det(y−p,q+Hu)=L det(q,u).

This proof does not divide by det(q,u). Parallel q and u are included, although the resulting row can be uninformative. The identity is necessary for the full physical screen equations; it is not generally sufficient.

## 2. Affine determinant and degree

With the other thirteen optical coordinates fixed, the first two prisms determine q independently of G. Their last-entrance position has the exact affine form p=T b+c g+e. Let v=q+Hu and δ=det(q,u). In the geometry order (b_x,b_y,g,d), an augmented residual row is

    [−det(T e₁,v), −det(T e₂,v), −det(c,v), −δ,
     det(y−e,v)−3δ].

For five selected samples, denote its augmented determinant by Δ. Each row is affine in its own radical H. The d column is radical-independent. Thus Δ is multilinear before identifying repeated radicals and has total radical degree at most four, not five.

Let a_i=|q_i|². If the five a_i are distinct, each G−a_i has an odd valuation at its own distinct linear factor. Their squareclasses are therefore independent over R(G), and adjoining their square roots gives a degree-32 multiquadratic field. The product over all 32 independent radical signs is its field norm. It is a polynomial in G of degree at most 4·32/2=64.

For repeated a_i, identify the equal radicals first. With r distinct a-values, flip the signs of the r distinct radicals and use the resulting genuine norm. Its degree is at most 4·2^(r−1), hence at most 64. The weighted-degree proof still applies when identifying repeated radicals produces powers of G.

A genuine field norm is identically zero exactly when its input is zero. An ungrouped sign product can fail this property: if a₁=a₂, the positive physical function H₁+H₂ is nonzero, but an artificial opposite-sign factor vanishes identically.

A nonzero norm therefore supplies at most 64 distinct candidate G values for that fixed thirteen-coordinate optical point and data set. Recover n₃ from its unique positive root sqrt(G), retain the native index interval, and check every exact physical constraint. Other sign branches can produce spurious norm roots. Geometry rank-deficient fibers and charts with identically zero norm must remain in the fallback representation.

## 3. Derivative witness

Define the five residuals

    F_k(z,G)=det(y_k−T_k b−c_k g−e_k,q_k+H_k u_k)
             −(3+d)det(q_k,u_k).

Here the measured y_k is held fixed when differentiating G. Its generation by the true physical model does not authorize differentiating it in this inverse residual.

At a true solution (z*,G*), add the first four augmented columns multiplied by z* to the final column. The new final column is F(z*,G), which vanishes at G*. The determinant derivative is consequently

    Δ′(G*)=det[F_bx,F_by,F_g,F_d,F_G],
    F_G=det(y−p,u)/(2H).

Thus a nonzero five-by-five derivative determinant establishes rank four of the geometry coefficient matrix and a simple zero of the physical Δ. In particular, Δ is not the zero function, so its grouped norm is a nonzero polynomial. This conclusion does not require distinct a_i, although the witness also verifies their distinctness.

## 4. Witness audit and independent computation

The witness uses:

- n₁=n₂=n₃=3/2;
- tan(a_j)=1/10 and phases zero;
- incident slopes t=(1/3,1/10);
- b=(1,2), g=10, d=100;
- samples t_k=k/20, k=0,…,199;
- exact unit-complex rotor steps (15+8i)/17, (4+3i)/5, (3+4i)/5.

All parameters lie in the stated native prior. In particular, each rotor angle θ lies between zero and π/3 because its cosine exceeds 1/2. The corresponding frequency is 10θ/π<10/3<3.5 Hz. The wedge is arctan(1/10)<18°, and the two beam angles arctan(1/3), arctan(1/10) are below 25°.

### Supplied interval implementation

Its intervals have integer endpoints divided by 2^240. Rational initialization, multiplication and division round outward using exact integer arithmetic. Division checks that its denominator interval avoids zero. The square-root endpoints are obtained from integer square roots with the upper endpoint rounded up. Addition and negation are exact on the grid. Exact rational rotor recurrences avoid angular or phase drift. Floating-point conversions occur only when printing summaries.

The transport matrix is algebraically correct:

    A=I+(a−v)uᵀ/(1−u·a),  a=q/H, v=B/Z.

The update A(p+3a)+ell v agrees with the two-segment intersection formula. The three upstream sensitivity vectors correctly propagate the two source-offset columns and the common-gap column, adding v to the gap column at each of the first two prisms.

### Independent implementation

The independent checker uses Fraction-valued endpoints rounded outward on a 2^280 grid. It computes refraction through the cancellation-free quadratic root

    λ=(n²−1)/(P+sqrt(P²−(1+|u|²)(n²−1))),
    B=q+λu,  Z=H−λ.

It propagates positions by explicit entrance-to-exit and exit-to-flat segments, without the supplied transport matrix. It obtains the upstream b_x, b_y and g derivatives from exact one-unit affine position differences, independently of the supplied sensitivity recurrence. Its determinant is enclosed by a direct signed-permutation expansion. It also checks the outgoing unit-direction and normal identities by interval enclosure.

Both implementations certify a strictly negative first-five-sample derivative determinant. The independent exact enclosure is contained in

    −7061916467402617 / 10^24 < determinant
      < −7061916467402614 / 10^24,

or approximately −7.061916467402616×10^−9. The implementation additionally prints its exact integer endpoints over 2^280 for reproducibility.

The five incoming squared transverse momenta have pairwise-disjoint rigorous enclosures centered approximately at

    0.18844110495324803,
    0.18849425205865877,
    0.16578787559258687,
    0.12944048310412171,
    0.09239879993818094.

For all 200 samples and all three prisms, the independent lower bound for every tested physical/traversal guard exceeds 0.824. The minimum lower bound is approximately 0.8241258530775574. These tests include positive H, P, transmitted normal component, outgoing axial component, internal traversal numerator, and remaining propagation distance. Initial forward axial propagation follows directly from its positive normalization. This is a certificate at the 200 stipulated samples, not between-sample traversal certification.

The arithmetic checks are reproducible rigorous interval calculations, not a parameter sweep and not a formal proof-assistant verification. Interval containment of a residual is only a consistency check; the exact identities are established algebraically above.

## 5. Positive-noise limitation

At any exact solution in the interior of the physical prior, continuity ensures that every positive coordinatewise noise budget permits an open interval of nearby G values, even with geometry and the other thirteen optical coordinates fixed. There can therefore be no general finite-index candidate theorem for positive noise.

For coordinatewise screen error ε, the exact sagittal identity implies the necessary inequality

    |F_k|≤ε||q_k+H_k u_k||₁.

For Euclidean screen error ε, replace the one-norm with the two-norm. For five selected rows, let C_i be the cofactors of the augmented determinant's final column. The same column operation used above yields the necessary determinant estimate

    |Δ|≤ε Σ_i |C_i| ||q_i+H_i u_i||₁.

These are sound noisy screening conditions, not equivalent replacements for the full noisy affine geometry constraints. Exact positive-noise elimination must preserve continuous index fibers, all observation inequalities, and the original strict physical branch/traversal tests.


## 6. Whole-prior generic corollary

The audit additionally inspected `physical_domain_contraction.md`. Using its connected physical interior O, the compact continuous semialgebraic extension Fbar on W, and dim(W\O)≤17, the proposed stronger generic statement is valid.

Let J(theta) be the fixed first-five-sample determinant from Section 3, evaluated at the measured data y=F(theta). The inverse residual is first differentiated with its data held fixed; only then is y replaced by F(theta). This distinction avoids the identically-zero derivative that would result from differentiating a residual along its own generated-data curve.

The positive-branch formulas make J real analytic and semialgebraic throughout O. The witness gives J≠0 at one point. Connectedness and the real-analytic identity theorem imply that its zero set

    Z={theta in O:J(theta)=0}

has empty interior. Semialgebraicity then gives dim Z≤17. Set

    B=(W\O) union closure_W(Z),
    E=Fbar(B).

The set B is compact semialgebraic of dimension at most 17; its continuous semialgebraic image E is also compact and has dimension at most 17. The existing full18 rank witness ensures that the model observation image has dimension 18, so E is genuinely lower-dimensional in that image.

For every exact observation y in Fbar(W) outside E, every weak physical preimage is interior and has J≠0. In particular, at every compatible retained-thirteen-coordinate optical point, the fixed first-five determinant has a true physical root with nonzero derivative. Its grouped norm is therefore nonzero, and all compatible n₃² values belong to that polynomial's at-most-64-root candidate set. This applies even if several full-system preimages share the same retained optical coordinates.

The corollary is generic over the entire original physical prior, not only over a component around the witness. It does not certify that a particular supplied observation avoids E, does not prove uniqueness, and does not bound the total number of retained-thirteen-coordinate solutions. An implementation intended to cover arbitrary data must retain the identically-zero-norm fallback unless a valid observation-level exclusion certificate is available. Repeated-radicand grouping remains necessary even under this generic corollary.

## 7. Constructive low-degree univariate noisy screening

There is a valid constructive refinement of Section 5. Fix the retained thirteen optical coordinates, the measured data, and a noise budget ε≥0. Assume the upstream physical trace is valid, so |q_k|²<1 and every H_k=sqrt(G−|q_k|²) is real and positive throughout the original closed index interval. The result below describes exactly the necessary determinant tube, not full noisy-model feasibility.

The cofactor C_i is the four-by-four determinant obtained by deleting row i and the augmented final column. It uses the other four samples and still contains the root-independent d column. It therefore has total radical degree at most three in at most four distinct radicals. Its genuine grouped norm has degree in G at most

    3·2^(4−1)=24.

Each scalar component v_(k,l)=q_(k,l)+u_(k,l)H_k has field norm

    q_(k,l)²−u_(k,l)²(G−|q_k|²),

of degree at most one in G. This is a field norm, not the ordinary square of v_(k,l).

First identify and omit identically-zero cofactors and components. For the remaining five cofactors and ten vector components, isolate the roots of their genuine norms in the index interval. There are at most

    5·24+10=130

distinct such roots. Some norm roots may belong only to other radical-sign branches; retaining them merely adds unnecessary cuts. On every resulting open interval, the signs of all nonzero physical cofactors and components are fixed.

On one such interval, choose their actual positive-root signs and replace

    S(G)=Σ_i |C_i(G)| (|v_(i,1)(G)|+|v_(i,2)(G)|)

by the corresponding signed polynomial expression in the H_k. Its total radical degree is at most four. Hence the two boundary functions

    B_minus=Δ−εS,  B_plus=Δ+εS

also have total radical degree at most four in at most five distinct radicals. Their genuine grouped norms have degree in G at most 64.

Handle an identically-zero boundary function explicitly: its associated weak inequality holds as equality on the entire cell, and its zero polynomial must not be passed to a finite-root isolation routine. For every nonzero boundary function, isolate the roots of its norm and refine the cell. A physical-branch sign test on each remaining open subinterval determines exactly whether

    B_minus≤0 and B_plus≥0,

equivalently |Δ|≤εS, holds there. Evaluate the original absolute-value inequality separately at every cut point and prior endpoint. This retains isolated feasible points and avoids losing a feasible equality endpoint. Since the tube functions are continuous, the tube itself is a finite union of closed intervals and isolated points in the closed index prior; later intersection with strict physical constraints can produce open endpoints.

Thus the necessary noisy tube has an exact univariate interval description using only scalar root-isolation polynomials of degree at most 64. The construction presupposes exact coefficient arithmetic and identity/sign tests, as does the exact-data norm construction. For symbolic ε it is a parameterized family of univariate problems; the fixed-budget statement does not by itself provide a complete stratification in ε.

This refinement does not restore a finite candidate list of indices, does not make the tube sufficient, and does not eliminate the need to enforce the full affine noisy geometry system and every original physical branch/traversal guard.
