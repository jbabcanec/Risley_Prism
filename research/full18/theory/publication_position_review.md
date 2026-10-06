# Critical publication-position review

Date: 2026-10-03. Scope: the original eighteen-unknown, three-prism finite-thickness vector-Snell model, original prior, and 200 timestamped paired observations. This is a bounded primary-literature positioning assessment, not a proof of priority or an independent re-proof of every audited theorem. No experiments or reconstruction campaigns were launched for this review.

## Overall assessment

The new full-vector results support a substantially stronger mathematical paper than the earlier canonical-model draft. The strongest position is **boundary-complete inversion and sharp limitations of passive three-prism calibration**. They do not yet support “practical recovery of all 18 unknowns,” “globally unique calibration,” or a computational breakthrough.

## Candidate central theorem

For the stated model, original prior and original record:

1. The weak transmitted/traversal set equals the closure of the strictly physical domain, with an explicit hierarchical inward-wedge construction.
2. The exact sampled forward map extends continuously and semialgebraically to that compact closure.
3. All compatible strict systems, including singular and continuous fibers, admit an exact optical-cell/affine-geometry-fiber representation; symbolic native-coordinate uncertainty endpoints are decidable with their inclusion flags.
4. A verified adaptive cover remains sound across critical refraction, and an exact fallback supplies termination under finite exact input encodings.
5. Two physically realized obstructions explain why completeness does not imply uniform recovery: an exact nonlinear fifth-order collision family and a sharp 1/8 critical filtered remainder.

The model-specific closure construction and obstruction realizations should carry the intellectual weight. The representation and decision procedure provide the constructive consequence. Conditional continuation is a useful corollary, not the headline novelty.

## Specific and nontrivial contributions

### Physical closure equals the weak set

Merely replacing strict inequalities by weak ones is generally invalid. Here, the three inward derivatives and hierarchical tau^16, tau^4, tau scaling handle simultaneous critical and traversal constraints across every timestamp. That constructive density result closes a real logical gap in global inversion. The later compactness, boundary-image and minimizer-limit deductions are principally consequences of it.

### A realizable sharp critical cascade

Nested square roots giving exponent 1/8 is elementary; establishing a triple-critical configuration inside these native bounds, preserving traversal, realizing nearby strict systems on the original clock, and showing every specified finite-order Prony annihilator retains that exponent is the substantial contribution. Risley total-internal-reflection/FOV constraints were already studied using vector Snell tracing, so “critical refraction matters” is not novel.

Primary comparator: Yuan Zhou et al., “Limits on field of view for Risley prisms,” Applied Optics 57, 9114–9122 (2018), https://doi.org/10.1364/AO.57.009114.

### Exact nonlinear collision lifting

The fifth-order cancellation and two-point lower-bound argument are established phenomena. Batenkov–Goldman–Yomdin analyze bounded-error near-colliding source reconstruction and the 2p−1 dependence. The additional result here is preserving that obstruction under exact nonlinear vector optics, with admissible positive wedges, off-axis excitation and full original 18-column rank, without an additive optical-tail floor. Describe it as a rigorous optical realization/extension, not a new superresolution exponent. Positive amplitude cutoffs and family constants remain unevaluated; the construction is not a practical noise threshold for a supplied record.

Primary comparator: Dmitry Batenkov, Gil Goldman and Yosef Yomdin, “Super-resolution of near-colliding point sources,” https://arxiv.org/abs/1904.09186; published version https://doi.org/10.1093/imaiai/iaaa005.

### Supporting structural tools

The source-transport determinant identity and strict-branch invertibility; five-root index chart; branch-preserving degree reduction; and data-only final critical-margin inequality with explicit polynomial certificates are concrete structural tools. The determinant lemma and backward affine propagation are standard; the optical identities, their whole-domain validity and their combination with strict traversal are the specific contribution. The 896 instantaneous degree bound is a representation improvement, not evidence of manageable computation.

## Established machinery to credit explicitly

- **Exact decidability after semialgebraic encoding:** real quantifier elimination already supplies this. The distinctive issue is obtaining a correct, bounded-dimension physical encoding without losing branches or strict endpoints. “Polynomial in sample count for fixed dimension” needs its enormous constants, degrees and coefficient costs prominently qualified. Author-hosted source: Saugata Basu, “Algorithms in Real Algebraic Geometry: A Survey,” https://www.math.purdue.edu/~sbasu/raag_survey2011_final.pdf; author preprint https://arxiv.org/abs/1409.1534. This surveys the established quantifier-elimination and complexity results; it is not evidence of a new general elimination theorem here.
- **Mixed-strict feasibility:** this belongs to Motzkin/Farkas alternatives. Minimal positive circuits, small supports and determinant tests are standard polyhedral machinery. The optical compiler and threshold bookkeeping are useful applications, not a new theorem of alternatives. Complete formalizations of the general alternatives already exist: Ralph Bottesch, Max W. Haslbeck and René Thiemann, “Farkas' Lemma and Motzkin's Transposition Theorem,” Archive of Formal Proofs (2019), https://isa-afp.org/entries/Farkas.html.
- **All-feasible-parameter bounded-error inversion, boxes and coordinate extrema:** established set-membership estimation. Jaulin–Walter explicitly discuss feasible-set enclosure, convergence and coordinatewise global optimization. The new contribution must be critical-boundary-safe optical structure and exact endpoint handling. Primary paper: Luc Jaulin and Eric Walter, “Set Inversion via Interval Analysis for Nonlinear Bounded-error Estimation,” Automatica 29, 1053–1064 (1993), author-hosted https://webperso.ensta.fr/Jaulin/paper_automatica93.pdf, DOI https://doi.org/10.1016/0005-1098(93)90106-4.
- **Continuation with certified regularity:** proper local diffeomorphisms yielding coverings is classical. Certified numerical homotopy tracking also predates this work. The distinction is the optical compactification, explicit discriminant and requirement to lift every sheet over the entire noise region. Neither finding one branch nor proving path-tracking correctness supplies complete anchor enumeration. Primary comparator: Carlos Beltrán and Anton Leykin, “Robust certified numerical homotopy tracking,” https://arxiv.org/abs/1105.5992. The covering reference used in the audited derivation is Chung-Wu Ho, “A note on proper maps,” Proceedings of the AMS 51, 237–241 (1975), https://doi.org/10.1090/S0002-9939-1975-0370471-3; its DOI was not independently accessible in this search.
- **Adaptive count:** the Hölder-modulus subdivision argument is a standard covering argument. Its specific LP value function avoids an additional changing-polytope regularity assumption; that is useful, but the gamma^(-112) bound is a conservative guarantee, not an efficiency result. Conditional near-unresolved-set covering estimates are not small runtime guarantees without certified constants and charged exact-operation costs.

## Closest optical comparisons

The closest identification comparator remains the Livox paper: exact vector refraction, passive angular observations and EKF parameter/rotation estimation, with a restricted shared-index/shared-wedge model. It does not make the finite-record all-18 certificate unnecessary, but prevents presenting passive optical self-calibration as new. Its claimed wedge/index correlation was not accompanied by a structural invariance derivation in the inspected observability section; the current results should not be advertised as refuting that different model.

Primary source: Ryan G. Brazeal, Benjamin Wilkinson and Hartwig H. Hochmair, “A Rigorous Observation Model for the Risley Prism-Based Livox Mid-40 Lidar Sensor,” Sensors 21, 4722 (2021), https://doi.org/10.3390/s21144722.

Controlled calibration already jointly fits wedge/index/installation corrections, while three-prism nonparaxial scan-pattern formulas already incorporate phases, speed ratios and separations. The present paper concerns the complete inverse set and its boundaries, rather than another forward model or steering inverse.

Primary sources:

- Jinying Li et al., “Improvement of pointing accuracy for Risley prisms by parameter identification,” Applied Optics 56, 7358–7366 (2017), https://doi.org/10.1364/AO.56.007358. Full text was not accessible; the publisher's abstract, parameter table and publisher preview/captions were inspected in the earlier comparison.
- Ce Qin et al., “Closed form analytical solution and scan pattern shaping theory of three-element Risley prism for LiDAR systems,” Optics Communications 570, 130915 (2024), https://doi.org/10.1016/j.optcom.2024.130915.

## Strongest likely reviewer objections

1. **“Is this mostly Tarski plus ray tracing?”** Answer through the constructive closure proof, exact nonlinear obstruction and sharp filter limitation; move generic machinery to supporting sections.
2. **“Where is one completed full-prior inverse?”** It remains unresolved. Verified executable components and an abstract terminating fallback do not establish a completed instance or useful runtime. Implementation correctness, complete parameter-space coverage and successful numerical recovery are separate claims.
3. **“Are the positive hypotheses ever quantitatively useful?”** No certified moderate anchor count, discriminant clearance, conditioning margin or practical noise threshold has yet been established. Data-only final-stage guards do not automatically certify upstream stages.
4. **“Is the pathological optical family experimentally observable?”** The critical witness has extremely large screen coordinates. Finite detector area, prism aperture, transmitted intensity and other unmodeled hardware limits may exclude it in a real instrument. Its validity is for the stated geometrical-ray model; practical importance needs separate justification.
5. **“Does full-vector mean full instrument physics?”** No. Prescribed surfaces, shared gap, ideal rotations and sampled-time validity remain assumptions. Between-sample traversal, speed drift and manufacturing misalignment are not silently covered.
6. **“Is it formally verified?”** Nineteen checked Lean support lemmas are not a Lean formalization of the physical compiler, compactification, complete inverse theorem or whole certificate pipeline. State independent mathematical audit, symbolic/executable checks and proof-assistant coverage separately. Do not describe the entire theorem as formally verified.

## Remaining novelty uncertainty and recommended claim

This bounded search found strong adjacent precedents but no directly matching theorem. It does not establish priority. The principal remaining comparison is against existing work on singular refractive ray maps, critical caustic/branch geometry and exact elimination for multilayer optical calibration, rather than only papers containing “Risley.” The model-specific results are promising novelty candidates, not certified first-in-literature claims.

Recommended claim: **a rigorous, model-specific theory identifying exactly what a complete inverse must retain, with constructive algebraic reductions and sharp counterexamples to uniform stability.** Practical all-18 recovery remains an open application of that theory.

No further experiment, broad reconstruction sweep, external contact or publication is proposed as an action authorized by this memo.
