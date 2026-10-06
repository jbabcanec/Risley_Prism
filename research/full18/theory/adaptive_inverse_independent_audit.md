# Independent audit of the certified adaptive inverse

Date: 2026-10-03

Audited synthesis: `certified_adaptive_inverse_algorithm.md`

SHA-256 at audit: `e36caa375549ac4e5fee79ef139b0c002f5dd72c9c299cfdd2fe0139be93a5c8`

## Verdict

**Passed mathematical audit of the new deductions and their exact-solver interfaces**, conditional on the explicitly cited audited optical-model, closure, modulus and mixed-strict circuit results. The note proves conditional quantitative/certificate-size improvements and an exact representation reduction. It does not prove a practical runtime, a useful record-specific gap, or a successful full-record reconstruction.

The final read checked Sections 2, 4, 5 and 6 in detail against `global_boundary_continuation.md`, `profiled_box_exclusion.md`, `reduced_algebraic_inverse_addendum.md` and `parametric_noise_inverse.md`. Relevant critical-margin and filtered-remainder imports were cross-checked. No older audited file was modified; no simulation, sweep, numerical solve or proof-assistant verification was performed.

## 1. Joint violation LP and strict endpoints

The fixed compact geometry rectangle makes the min-max violation value attained. Uniform row moduli pass through both maximum and minimum, proving the stated pairwise modulus without a path inside the optical domain. The simplex-weight box-support formula is a valid dual lower bound and its optimum is the primal value. This remains valid at critical optical anchors.

Optimality is needed for the adaptive count, not for a sound exclusion leaf. The note correctly distinguishes exact optimal values from arbitrary feasible or accuracy-certified dual proposals.

The row-specific anchor outer system is necessary for every strictly physical compatible system. Traversal rows remain strict under the modulus relaxation. Its mixed-strict circuit rejection has support at most five total rows, including geometry bounds. A zero weighted margin rejects strict systems only when some positive multiplier belongs to a strict traversal row; it does not reject weak systems. The separate traversal exponents 1, 1/2 and 1/4 follow from the audited recurrence.

## 2. Exact cell/fiber output and sixteen-variable envelopes

Exact circuit elimination produces an optical/epsilon predicate in fifteen outer variables. A geometry-coordinate query adds two weak affine rows but no sampled trace. An unchanged optical-coordinate query adds an equality or permits substitution. An original-index query uses its positive reconstruction relation, of degree four. Each coordinate projection therefore uses at most sixteen outer variables.

The complete inverse representation is exact: every retained optical point has its explicit parameter-dependent mixed-strict convex geometry fiber, preserving all correlations and continuous gauges. This is neither one constant polytope per optical cell nor a claim to have constructed a full eighteen-dimensional CAD.

At most five sampled traces enter each circuit support. Their combined radical compiler has a larger fixed degree than the single-sample degree 896; after rotor substitution the degree is still O(K). The stated O(K^5) support count and schematic fixed-outer-dimension CAD bound are valid with the declared compiler, coefficient-height and algebraic-encoding costs. Local elimination of uniquely isolated algebraic observations can be performed before assembling a rational global predicate; otherwise the full coefficient field must be charged.

Projected plane cells provide infima and suprema without attainment. Actual exceptional epsilon fibers, strict flags and jumps must be retained, as the note specifies. The subsequent three-variable graph/diameter tests are exact. Coordinate diameters strictly exceeding the target imply actual strict-compatible witness pairs even when their extrema are unattained. Index extrema are correctly queried through reconstruction instead of being inferred from h or c ranges.

## 3. Adaptive stopping and count

For a nonempty node that is neither retained nor excluded, exact optimality gives v(x)<=2 Psi(h) throughout its optical portion. Failed retained containment supplies a point outside U in the same diameter-h box, giving the distance condition defining A_h. Thus each internal node is counted by the occupied dyadic-grid count, and the full-branching tree identity yields equation (12).

The buffered U_0 and joint violation gap give finite termination at the displayed scale capped by one. The proof uses the original optical chart and does not inadvertently reuse its modulus in the reduced index chart. The global exponent is 1/8; a Lipschitz replacement requires the separately stated convex-neighborhood or pairwise certificate. The note correctly charges domain emptiness, anchor extraction, retained containment and arithmetic separately, and distinguishes the joint violation gap from an observation-only residual gap.

## 4. Entropy/error-bound corollary

The added corollary is valid under its global certificates. Put kappa=alpha/beta, t=(2C/a)^(1/beta) h^kappa and r=max(h,t). The error bound implies A_h lies in the t-neighborhood of K. A radius-r cover of K, enlarged by t, meets at most a dimension-dependent multiple of (r/h)^14 grid boxes per ball. Therefore

    N_h(A_h) <= constant * M * r^(14-s) * h^(-14).

This gives p=s+(14-s) max(0,1-alpha/beta). Geometric summation gives the asserted h_*^(-p) bound for p>0 and the asserted 1+log(1/h_*) bound for p=0, with coarse levels and all stated parameter dependence charged. Minimum ball-cover counts are correctly distinguished from occupied-grid counts. Finite K permits M equal to its cardinality for the former.

Hence the candidate-count-times-depth conclusion for alpha=beta=1 and s=0 is justified. Neither finite cardinality, pointwise rank nor semialgebraic dimension alone supplies the required global error bound or moderate covering constants.

## Audit limits

Weak endpoint systems are not physical witnesses. Symbolic epsilon transitions may destroy adaptive margins. An unresolved finite frontier remains an enclosure until the exact cell/fiber fallback is completed. The synthesis retains these distinctions. The audit establishes no literature-novelty claim and no record-specific numerical threshold or performance result.
