# Risley Prism Inversion: A Complete Constructive Inverse

J. Babcanec (Benedict College) · B. Campbell (Robert Morris University)

**An exact inverse in principle for all eighteen parameters of a three-prism system, including every compatible solution and ambiguity. The remaining computational obstacle is the cost of global algebraic elimination.**

Current research: [full report](research/full18/REPORT.md) · [research guide](research/full18/README.md) · [complete inverse theorem](research/full18/theory/global_inverse_completeness.md)

Repository overview updated October 6, 2026. The imported report is the October 3 research draft, revision 13.

A Risley system steers a laser through rotating wedge prisms. This project asks the inverse question: from time-stamped screen positions, what physical systems could have produced them? The unknowns include rotation, glass, geometry and the incoming beam. Physical prism order is retained.

The current theorem answers that question for the **exact coupled three-dimensional vector-Snell model**, the full declared parameter box, and a finite sampled record. It constructs the entire compatible set, or proves that set empty. From that set it decides whether a requested worst-case parameter accuracy is possible for the supplied observations and error bounds.

This is a complete mathematical construction with a finite exact algorithm. A practical full-prior implementation has not been demonstrated. The repository also contains a partial certificate engine and an earlier numerical solver for a different, independent-axis optical model; their results have separate scopes.

## What is being recovered?

For three prisms the native parameter vector is

```text
[N1,N2,N3, ax1,ax2,ax3, ay1,ay2,ay3, ng1,ng2,ng3,
 d_W,gap, bm_ax,bm_ay,bm_px,bm_py]
```

| Unknowns | Meaning | Native prior |
|---|---|---|
| `N1..N3` | Signed rotation speeds | −3.5 to 3.5 Hz |
| `ax1..ax3` | Signed wedge angles | −18° to 18° |
| `ay1..ay3` | Initial rotation phases | −18° to 18° |
| `ng1..ng3` | Glass refractive indices | 1.3 to 1.8 |
| `d_W`, `gap` | Workpiece distance and common inter-prism gap | 50 to 200; 2 to 15 |
| `bm_ax`, `bm_ay` | Incoming beam angles | −25° to 25° |
| `bm_px`, `bm_py` | Source offsets | −5 to 5 |

Distances use native model units. Source distance 6 and prism thickness 3 are fixed model inputs. The reference record has 200 paired screen positions at the exact times `t_k = k/20`, ending at 9.95 seconds. The theorem checks physical transmission and traversal at the measured times.

The finite exact algorithm accepts rational observations and error bounds, or explicitly encoded algebraic inputs under the theorem's encoding rules. A rounded measurement can be represented by a rational interval that includes its measurement and rounding error.

## How the inverse works

1. **Encode all eighteen unknowns algebraically.** Tangent charts replace angles and speeds by bounded real coordinates with algebraic prior endpoints. Exact rational rotor formulas give each prism's orientation at every sample; no truncated Fourier model is needed.
2. **Express one optical sample exactly.** Polynomial equations and sign conditions encode Snell refraction, intersections, and the transmitted physical branch. Positive-root and traversal conditions remove spurious solutions introduced by squaring.
3. **Eliminate the internal ray variables once.** Compile the fixed single-sample optical graph into polynomial sign tests. The newer index chart reduces the compiler from six square roots to five.
4. **Combine every observation in the same eighteen variables.** Substitute the sampled rotor formulas and impose all measurement bands. Four geometry variables are affine once the optical variables are fixed, allowing additional exact elimination while retaining rank-deficient cases.
5. **Resolve every remaining algebraic branch.** A complete real-algebraic decomposition returns all compatible cells, including isolated solutions, disconnected alternatives and continuous families. It also detects inconsistency.
6. **Read off recoverability and error.** Project the compatible set onto each native coordinate. Its ranges determine possible accuracy; separated compatible systems certify when the requested accuracy is impossible.

The [complete theorem](research/full18/theory/global_inverse_completeness.md) proves equivalence and termination. The [reduced compiler and affine-fiber addendum](research/full18/theory/reduced_algebraic_inverse_addendum.md) gives the five-root construction, invertible source transport, and exact geometry reductions.

A small elimination example illustrates the principle. Suppose observations imply

```text
a + b = 5
 a² + b² = 13
```

Substituting `b = 5 − a` gives `a² − 5a + 6 = 0`, so the full inverse is `{(2,3), (3,2)}`. Returning one root would discard a compatible explanation. The optical construction applies the same requirement to much larger equations, physical branch conditions and bounded-error data.

## Why it is computationally expensive

Compiling the fixed optical graph before combining samples keeps the global unknown count at eighteen. For fixed prism count, clock and native tolerances, the exact construction has polynomial bit complexity in the number of consecutive samples and encoded input size. Its dimension-dependent exponent and constants can be prohibitive.

The improved compiler has an instantaneous degree bound of 896. After rotor substitution, its conservative degree bound at 200 samples is **1,076,096**. These are representation bounds, not measured running times or necessary work. They explain why a complete finite construction does not yet give a usable global solver.

Current work reduces that cost through exact geometry elimination, smaller optical charts, certified local correction and bounded exclusion. Generic finite fibers and a successful local calculation do not by themselves resolve every branch of the full inverse.

## What a 0.001 accuracy guarantee means

Let `Cε(y)` contain every physical system compatible with record `y` and its hard error allowance `ε`. If this set is nonempty, an **unrestricted point estimate** can guarantee error at most 0.001 in every native coordinate exactly when every coordinate range has width at most 0.002. The coordinatewise midpoint achieves that bound, even when range endpoints are not attained.

That midpoint need not itself be a compatible physical system. Requiring a **compatible estimate** is a different, additional decision: find a point in `Cε(y)` that is within 0.001 of every other compatible point. The exact construction can decide that condition too; the midpoint criterion must not be substituted for it.

There is no unconditional 0.001 guarantee over the whole prior. For example, a zero wedge leaves its rotation phase invisible. Exact ambiguity is part of the inverse's output. Speed collisions are retained by the construction and do not alone establish non-identifiability.

All guarantees here use deterministic bounded errors. Statistical sensitivity formulas are not worst-case certificates.

## Theory, implementation and evidence

| Component | Established scope | Remaining implementation or scope limit |
|---|---|---|
| [Full-vector research](research/full18/REPORT.md) | Complete finite exact inverse in principle, full compatible sets, native accuracy decisions and symbolic error thresholds | No demonstrated practical full-prior inversion or useful actual-record precision certificate |
| [Partial certificate engine](research/full18/certificate_engine/README.md) | Rational interval evaluation, shared-geometry affine LP, sparse exclusions and bounded subdivision | Saved full-prior check retains four unresolved leaves; it is not the complete global solver |
| Full-vector Lean support, documented in the report | Nineteen checked supporting algebraic, inequality and final-prism margin declarations | The complete optical compiler, global inverse and certificate pipeline are not formalized |
| [`risley_lattice/`](risley_lattice/) | Earlier frequency-lattice initialization, numerical refinement and statistical diagnostics for the independent-axis model | Numerical recovery and statistical diagnostics do not establish full-vector global recovery |

The report is a research draft for expert review. Its mathematical audits, saved computations and Lean support have distinct scopes. No physical instrument experiment is claimed. The [report's related-work discussion](research/full18/REPORT.md#11-related-work-proposed-contribution-and-validation-gap) identifies established machinery and limits the novelty claims.

## Earlier numerical package

The earlier model in [`reverse_problem_v2/core.py`](reverse_problem_v2/core.py) traces the axes independently. The frequency-lattice package uses that model; it must not be described as the exact coupled vector-Snell solver.

The diary records a September 27, 2026 development replay that recovered **26 of 30 cases** to its machine-precision criterion in **628 seconds total**. That is historical evidence from the recorded development version and environment. It is not a benchmark of the older numerical code retained in this publication. See [`DIARY.md`](DIARY.md) for the evidence and historical experiments.

The retained earlier numerical package has a standard development-battery entry point. Run from the repository root with NumPy and SciPy installed:

```bash
python experiments/solve18_battery.py
```

The script pins BLAS threads for reproducibility. The older `risley_lattice/certify.py` statistical calculations are retained as historical diagnostics; they must not be used as deterministic guarantees.

This command does not run the complete full-vector inverse. The [research guide](research/full18/README.md) describes the newer sources and their reproduction boundaries.

## Repository map

| Path | Contents |
|---|---|
| [`research/full18/`](research/full18/) | Current full-vector report, exact inverse theory, supporting sources and partial engine |
| [`risley_lattice/`](risley_lattice/) | Earlier independent-axis inversion and statistical diagnostics |
| [`experiments/`](experiments/) | Earlier reproducible experiments and saved evidence |
| [`reverse_problem_v2/core.py`](reverse_problem_v2/core.py) | Earlier independent-axis reference forward model |
| [`paper/main.tex`](paper/main.tex) | Older manuscript draft; not the current full-vector report |
| [`paper/REWRITE_PLAN.md`](paper/REWRITE_PLAN.md) | Older manuscript rewrite plan and remaining editorial work |
| [`DIARY.md`](DIARY.md) | Chronological research log, including superseded approaches and their limitations |
| [`forward_problem/`](forward_problem/) | Legacy gallery model with a different refractive-index scheme |

The older manuscript and its saved figures/PDF do not represent the current full-vector report. Use the report and linked theorem notes as the starting point for the current mathematical result.
