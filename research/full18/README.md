# Exact full18 inverse: current research

Published in this repository on **October 6, 2026**. The research report is
**revision 13, dated October 3, 2026**.

**The complete mathematical inverse is constructed. Making its global
algebraic computation practical remains unfinished.** This package is the
current entry point for the exact coupled three-dimensional vector-Snell
problem with all eighteen native hardware parameters unknown.

## Start here

- [Full report](REPORT.md): construction, physical assumptions, error bounds,
  computational complexity, later reductions, and implementation status.
- [Complete inverse theorem](theory/global_inverse_completeness.md): the
  finite exact algorithm, proof of completeness, and input conventions.
- [Reduced algebraic inverse](theory/reduced_algebraic_inverse_addendum.md):
  five-root compiler and exact elimination of affine geometry variables.
- [Adaptive inverse construction](theory/certified_adaptive_inverse_algorithm.md):
  validated exclusions, remaining regions, and the exact completion obligation.
- [Partial certificate engine](certificate_engine/README.md): executable
  coupled-vector enclosures, shared-geometry constraints, and saved certificates.
- [Original-record provenance](RECORD_PROVENANCE.md): the distinction between
  historical independent-axis observations and full-vector synthetic evidence.

## How the mathematical algorithm works

1. Take timed laser positions and a hard bound on their measurement error.
2. Rewrite sampled rotor motion, vector refraction, and actual surface
   intersections as exact polynomial equations and inequalities. Preserve
   transmission signs and forward traversal.
3. Require the same eighteen parameters to satisfy every sample. Eliminate
   sample-specific internal rays before combining the observations.
4. Use complete real-algebraic elimination and cell decomposition to retain
   every compatible system, including isolated answers, disconnected branches,
   and continuous families.
5. Project that set onto each native parameter to obtain its possible range.
   For a nonempty compatible set and an unrestricted estimate, error at most
   0.001 in every coordinate is
   possible exactly when every range has width at most 0.002. Requiring the
   estimate itself to be a compatible physical system is a separate test.

The costly step is computing the complete compatible set and its projections.
For fixed optics and sampling clock, the theoretical dependence on sample
count is polynomial, but the constants and exponents can be enormous. The
reduced compiler's conservative degree bound at 200 samples is 1,076,096.
This is a polynomial degree bound, not a measured running time or storage bound.

## What has been completed

| Layer | Established result | Current limit |
| --- | --- | --- |
| Exact inverse | A terminating construction for every compatible physical system under the stated finite-input and sampled-time model | No practical complete decomposition of the full-prior 200-sample problem |
| Native uncertainty | Exact parameter envelopes and recovery-to-tolerance decisions in principle | Useful record-specific envelopes have not been computed globally |
| Algebraic reductions | Five-root forward compiler; affine geometry elimination; later guarded optical reductions | Exceptional branches and denominator conditions must remain accounted for |
| Certificate engine | Exact rational interval evaluation, shared-geometry LP exclusions, and bounded subdivision with replayable evidence | Saved full-prior subdivision still has four unresolved leaves |
| Lean support | Nineteen supporting declarations with saved verification evidence | The full inverse and Python certificate pipeline are not Lean-formalized |

The theorem includes ambiguity. It does not assert that every possible scan
uniquely determines its hardware. Measurement error, zero wedges, and other
degenerate configurations can leave genuinely different explanations.

The full-vector optical model here couples the two transverse directions.
The older `risley_lattice/` package at the repository root uses the canonical
independent-axis model. Its numerical results are recorded separately in
section 9 of the report and in [EVIDENCE_REPORT.md](EVIDENCE_REPORT.md).

## Replay saved implementation evidence

The certificate engine uses Python 3.10 or newer and only the standard
library. From the repository root:

```sh
cd research/full18/certificate_engine
python engine.py verify spotcheck/small_box_input.json spotcheck/small_box_certificate.json
python adaptive.py verify adaptive_checks/full_prior_input.json adaptive_checks/full_prior_certificate.json
```

These commands replay saved certificates. The small box was supplied to the
engine; it was not recovered from observations. Verification of the full-prior
certificate preserves its unresolved regions and does not establish global
recovery. See the engine README for further commands and outcome semantics.

Historical proof checks and numerical experiments have their own dependencies
and execution contracts. Some preserve original machine paths or source hashes;
they are evidence snapshots, not a newly unified portable solver. The imported
engine and proof sources are unchanged so their recorded hashes remain usable.
Build any Lean package outside the Dropbox checkout as directed by the root
repository guidance. Compiled Lean artifacts are omitted.

## Source preservation

This directory brings together work that previously lived in a separate local
research workspace. [SOURCE_SNAPSHOT.json](SOURCE_SNAPSHOT.json) records the
size and SHA-256 of each imported file and names omitted machine artifacts and
Library transfer metadata. Imported scientific files are copied byte-for-byte;
the original workspace is preserved. `.gitattributes` disables line-ending
conversion here to retain certificate and source-hash identity.

Earlier report revisions, source backups, saved synthetic observations,
supporting proofs, and result files remain available as provenance. Dates and
claims in those historical files belong to their original revision. This
README and report revision 13 describe the current status.
