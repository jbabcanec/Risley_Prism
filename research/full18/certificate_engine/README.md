# Bounded full-vector certificate engine

This is an executable, standard-library Python certificate evaluator for the exact coupled-vector three-prism model. The second implementation revision adds a shared-geometry affine LP, sparse exact dual exclusions, and bounded optical subdivision with complete frontier bookkeeping. It remains **not a complete inverse, a parameter fit or an all-eighteen accuracy certificate**. The published research report remains unchanged at version9.

## Run

Requires Python 3.10 or newer; no third-party package is used. This workspace was checked with the already installed Ubuntu WSL Python 3.12.3. No software was installed.

From this folder on Linux/WSL:

    python3 engine.py certify spotcheck/small_box_input.json certificate.json
    python3 engine.py verify spotcheck/small_box_input.json spotcheck/small_box_certificate.json
    python3 verify_spotcheck.py
    python3 joint_lp.py analyze adaptive_checks/expanded_geometry_joint_input.json joint.json --pivot-limit 32
    python3 joint_lp.py verify adaptive_checks/far_distance_joint_input.json adaptive_checks/far_distance_joint_certificate.json --sparse-only
    python3 adaptive.py run spotcheck/full_prior_outer_box_input.json cover.json --max-nodes 3 --max-depth 2 --lp-pivot-limit 16 --max-total-lp-pivots 32
    python3 adaptive.py verify adaptive_checks/full_prior_input.json adaptive_checks/full_prior_certificate.json
    python3 verify_adaptive.py

On this connected Windows computer, use the existing WSL interpreter:

    wsl.exe -d Ubuntu -- python3 /mnt/c/Users/josep/Documents/Codex/2026-10-02/task/full18_research/certificate_engine/engine.py verify /mnt/c/Users/josep/Documents/Codex/2026-10-02/task/full18_research/certificate_engine/spotcheck/small_box_input.json /mnt/c/Users/josep/Documents/Codex/2026-10-02/task/full18_research/certificate_engine/spotcheck/small_box_certificate.json

`spotcheck.py` can regenerate the original labeled synthetic fixture and its deterministic validation cases; it was not run for this revision. `adaptive_spotcheck.py` reads that exact saved record and generates only the bounded mechanism certificates described below. The supplied certificates allow verification without generating observations. Python may create ordinary local bytecode caches; these are not part of the deliverable or proof evidence.

## Exact input contract

The input has exactly eighteen chart-coordinate intervals under `box`:

| Coordinates | Meaning |
|---|---|
| `r1,r2,r3` | Signed wedge slopes, \(r_j=\tan a_j\), with native wedges in ±18 degrees. |
| `p1,p2,p3` | Initial-phase charts, \(p_j=\tan(\phi_j/2)\), with phases in ±18 degrees. |
| `v1,v2,v3` | Speed charts, \(v_j=\tan(\pi N_j/20)\), with signed speeds in ±3.5 Hz. |
| `n1,n2,n3` | Separate native indices in [1.3,1.8]. |
| `tx,ty` | Incident slopes, \(\tan\beta_x,\tan\beta_y\), with angles in ±25 degrees. |
| `bx,by,g,d` | Shared source offsets in [-5,5], repeated gap in [2,15], screen distance in [50,200]. Units are the model's native length units. |

Each interval is a two-element array of exact rational strings or integers. Strings such as `"1/3"` or `"0.05"` denote exact rationals. JSON floating-point numbers and booleans are rejected as endpoints, observations or allowances. The engine uses the original index chart, not the reduced h/c chart.

The model identifier must be `coupled_vector_snell_three_prism`. A provenance description is required. Known canonical/independent-axis labels are rejected, but a label does not independently authenticate a record's origin. No saved canonical record was used.

The sampling object must be exactly `{"count":200,"step":"1/20","start":"0"}`. Optional timestamps must each equal exact \(k/20\), \(k=0,\ldots,199\). The engine never uses a binary64 arange clock. Optional observations contain exactly 200 rational [x,y] pairs and a nonnegative rational `epsilon`, shared by all 400 scalar observations. Without observations, the request is a forward/physical-domain box check.

`bits` selects a dyadic grid, default 80 and allowed range 32–256. This is an arithmetic grid setting, not a promised number of correct digits in a broad-box result.

## What is implemented

The incoming unit transverse direction is \(t/\sqrt{1+|t|^2}\), and the first entrance position is \(b+6t\). Rotor phases on the exact clock use rational complex rotations from the half-angle charts and integer powers. No floating-point trigonometric function is used.

Every prism propagates the coupled two-component direction through the audited H/P/radicand/R/Z formulas and the selected nonnegative exit root. Position and traversal quantities are carried as **five interval coefficients in the shared order [bx,by,g,d,constant]**. Optical coefficients remain interval enclosures. The recurrence is affine in the same four geometry variables at every stage/sample; it never chooses independently fitted geometries per row.

The unchanged `engine.py` evaluates each affine form over the entire geometry rectangle. `joint_lp.py` additionally checks their simultaneous necessary conditions using the same four geometry variables. Interval dependence can make both tests conservative.

The native angular prior endpoints are enclosed rigorously using Machin's formula for pi, alternating rational arctangent sums and Taylor remainders for sin/cos. A box must be proved wholly inside the original prior to receive a whole-box retention certificate. A box merely intersecting an uncertain irrational prior boundary remains unresolved unless another valid exclusion applies.

The data-only final-prism test uses all observations to certify
\[
R_3(k)/Z_3(k)\ge
\max\{0,(50-\overline M_k)/53\},
\]
where \(\overline M_k\) is a rigorously rounded upper bound for
\(\tan18^\circ\sqrt{(|y_{kx}|+\epsilon)^2+(|y_{ky}|+\epsilon)^2}\).
It is conditional on original-prior compatibility and **does not establish feasibility**. It does not substitute a fitted optical point for the prior.

## Outcome and certificate semantics

- **excluded:** the certificate names a proven contradiction covering every potentially physical point in the supplied box: a disjoint original prior, a nonpositive upper bound on a required strict physical guard, or an output enclosure disjoint from an observation band.
- **retained:** every point of the supplied box is inside the original prior and strictly physical at all 200 samples. If observations are present, every output enclosure is inside its band, so the entire box is compatible. This is much stronger than interval overlap, but is local to the supplied box and proves no global recovery.
- **unresolved:** the complete input box remains in consideration. The engine cannot yet establish a uniform sign, safe denominator or whole-box compatibility. It neither discards the box nor declares it feasible.

If a radicand enclosure straddles zero, the evaluator may bound its nonnegative-root portion while marking physical status unresolved. Subsequent outputs then enclose the potentially physical subset; no real forward value is asserted for negative-radicand inputs. A divisor containing zero stops evaluation as unresolved. Early exits retain their sample/stage and exact rational witness.

Certificates contain input/source hashes, prior enclosures, exact sample times, guard intervals, all processed traversal affine coefficients, screen affine coefficients and output intervals. Deterministic `verify` recomputes and compares the whole certificate, rejecting tampered evidence. This replay uses the same evaluator; independent source and alternate-formula checks are recorded separately. Source changes require fresh certificates.

## Rounding guarantees

All primitive arithmetic uses exact Python integers and Fractions followed by directed floor/ceiling onto multiples of \(2^{-\text{bits}}\). Multiplication and division check all endpoint combinations; division through zero raises. Squaring handles zero-crossing intervals explicitly. Square-root endpoints use integer square-root comparisons and exact rational inequalities. Each rounded endpoint is outward; composed enclosures can be much wider than one grid unit.

The interval kernel contains no floating-point arithmetic. Floating wall-time measurements and any displayed decimal summaries are informational only; all certificate comparisons use exact rationals. Rigorous enclosures do not guarantee useful widths on large optical boxes.

## Shared-geometry LP and sparse exclusions

Each observation yields two weak violation inequalities, and each available internal/external traversal yields a strict inequality. A complete trace supplies 800 observation rows and 1,200 traversal rows. Partial traces may already contain sufficient rows for a valid exclusion; missing rows are reported explicitly and never inferred.

For an interval affine coefficient vector, the LP takes its midpoint and lowers the constant by the coefficient-radius bound over the **entire** shared geometry box. If coefficient radii are `rho_j`, the compensation is `rho_constant + sum(rho_j * max(abs(q_j endpoints)))`. Thus every physically compatible point satisfies the relaxed rows. A feasible weak relaxation cannot establish physical feasibility.

The LP minimizes the maximum lowered violation in five variables: four geometry coordinates plus the epigraph variable. A deterministic exact-rational active-basis method checks primal inequalities and uses finite pivot and repeated-basis stops. It may stop before optimality. No floating solver, numerical stationarity tolerance or termination theorem is assumed.

A sparse witness has at most five positively weighted sampled affine rows, with weights summing exactly to one. Any nonzero weighted normal is paid for by its exact minimum over the geometry box. These support faces are additional implicit geometry inequalities; arbitrary nonoptimal proposals are not claimed to be minimal five-total-row Helly circuits. A positive corrected lower bound excludes even the weak system. A zero bound excludes only when a strictly constrained traversal row has positive weight. Negative bounds and weak zero bounds do not exclude.

`joint_lp.py verify` replays the bounded optimizer. `verify --sparse-only` rebuilds the forward and row evidence and checks an exclusion without running the optimizer. Source/input hashes establish identity; exact arithmetic and proof replay establish the mathematical claim. The internal `Row`, `solve_rows`, and `dual_evidence` helpers assume already validated exact `Fraction` data; external requests should use the validating JSON/CLI or `analyze` interface.

## Cover and resource semantics

`adaptive.py` maintains a breadth-first closed-box cover. It bisects only the fourteen optical coordinates at exact rational midpoints, choosing the largest fraction of that coordinate's root width. Every child retains the full original four-dimensional shared geometry box. Sibling overlap at the midpoint preserves coverage.

Leaves are `excluded`, `retained`, or `unresolved`; `split` is an internal-node status. Retention still requires the strong whole-box forward proof above. A nonexcluding LP leads to subdivision or an unresolved leaf. Reaching a budget records every queued box as an explicit unresolved frontier, including the root when the forward budget is zero. No global recovery claim is emitted.

Default limits are seven forward evaluations, depth three, 24 pivots per LP and 96 pivots in total. Each limit accepts integers from zero to 10,000. These are work counters, not wall-time or memory guarantees: exact rational bit growth, forward evaluations, row scans and coefficient extraction also cost resources. At most twice the forward-evaluation budget plus one tree nodes are stored. Verification checks exact root coverage, unchanged geometry, proof status, limits and hashes, and replays each bounded LP. There is no exact fallback at a resource stop.

## The single validation record

The supplied [synthetic record](spotcheck/synthetic_fullvector_record.json) is generated from one explicitly stated rational full-vector chart point with nonzero signed wedges, phases, speeds, two incident slopes and source offsets. It contains exactly 200 pairs on \(k/20\). Its rational observations are midpoints of proved point-forward enclosures, with their outward error recorded.

A box of halfwidth \(10^{-8}\) in **all eighteen chart coordinates** is supplied around that point. Its validation allowance is derived from the entire forward enclosure, not inferred as a real sensor noise bound. The box is deliberately supplied for this implementation check; it was not recovered from data.

The same single record is used for four fixed checks:

| Supplied box | Checked outcome |
|---|---|
| Small all18 box | Retained: all 200 physical traces and 400 bands certified. |
| Same local optics with distant screen geometry | Excluded by the x observation band at sample 0. |
| Full-prior outer box | Unresolved at an axial divisor enclosure; retained for future refinement. |
| Specified nontransmitting optical box | Excluded by a strictly negative exit-radicand upper bound at sample 0. |

[results.json](spotcheck/results.json) records exact bounds and witnesses. The point-forward maximum coordinate width is \(3243/604462909807314587353088\), about \(5.37\times10^{-21}\). The synthetic whole-box allowance is \(33070397787796324767/2417851639229258349412352\), about \(1.37\times10^{-5}\) in screen units. These values describe one proof check, not a benchmark or recovery threshold.

Independent validation uses a separate normalized three-dimensional vector-Snell and ray/plane implementation at higher precision, with exact rational point rotors. It checks all 400 reference outputs against the engine point enclosures and the record's stated bounds. The [independent review](independent_review.md) and [machine-readable results](independent_review.json) report a pass: all 400 alternate-formula outputs are contained, 1,600 shared-geometry affine ranges and 200 final-prism margin bounds check, and 515 exact kernel checks pass. Certificate replay and tamper rejection also pass. No hardware or actual-record success is claimed.

## Remaining implementation gap

The new [bounded checks](adaptive_checks/results.json) reuse the saved record byte for byte. The small supplied box is retained and the existing distant-screen box is excluded. Three full-prior evaluations at depth at most two produce seven tree nodes, three splits and four explicit unresolved leaves. Zero-forward and zero-pivot budgets remain unresolved. With the small optical box and full prior geometry, all 2,000 affine rows are constructed but the 32-pivot limit is reached without an exclusion. The existing distant-screen case supplies a verified one-row sparse exclusion from its available partial trace. These are mechanism checks, not recovery trials.

The [new independent audit](adaptive_independent_review.md), [exact results](adaptive_independent_review.json), and [checker](verify_adaptive.py) cover row lowering at geometry vertices, sparse support correction, LP primal evidence, full tree coverage and tamper rejection. The earlier forward-engine review remains a separate unchanged validation record.

Full-prior useful localization remains unresolved. The engine does not certify continuation/discriminants, compute sixteen-variable exact envelopes, or invoke an exhaustive algebraic fallback. Finite subdivision and the LP provide sound local exclusions and a complete recorded unresolved cover, but do not establish practical inverse completion, uniqueness or native-coordinate recovery accuracy. A qualifying full-vector record with units and an error contract is still required for real-data validation. No new record, parameter fit, sweep, original-project execution or installation was used for this revision. Historical research files included through linked documentation are context only; they are not additional full-vector validation records.

The audited theoretical basis is [full-vector optics](../theory/physical_vector_inverse.md), [boundary/guard formulas](../theory/global_boundary_compactification.md), [data-dependent margin](../theory/data_dependent_critical_margin.md) and [adaptive interface](../theory/certified_adaptive_inverse_algorithm.md). The existing nineteen Lean results support some identities and the margin implication; they do not formally verify this Python engine.
