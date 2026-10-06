# Symbolic and exact-rational witness reproduction checks

This folder contains ten supplied Python 3 proof-check sources. The original seven symbolic files require SymPy. The second-order files load their matching first-order sibling, so keep all files together. The three later exact-rational sources use the Python standard library only (`fractions` and `math`).

Their source owner supplied them as previously executed targeted checks and independent audit counterparts; the parent workspace also reported successful Python syntax compilation of the original seven. Local packaging saved the source bytes and verified the supplied SHA256 hashes. None of the ten witness scripts was executed or imported, and no mathematical check was repeated during packaging. The three new exact-rational files also passed local `ast.parse` syntax checks under Python 3.12.3 in the existing Ubuntu WSL environment, with exit code 0 and unchanged hashes; [syntax_verification.json](syntax_verification.json) records those checks. Parsing source text does not reproduce the supplied mathematical execution or audit.

The checks address the exact algebraic witness identities stated in the companion theory, rather than statistical performance or parameter searches:

| Files | Scope |
| --- | --- |
| `vector_firstorder_rank.py`, `audit_vector_firstorder_rank.py` | The physical-vector oblique first-order shape determinant. |
| `vector_secondorder_gauge.py`, `audit_vector_secondorder_gauge.py` | The physical second-harmonic derivative along the first-order ambiguity; the primary script also converts the tangent to native coordinates. |
| `vector_last_prism_rank.py`, `audit_vector_last_prism_rank.py` | The physical-vector formal last-prism four-dimensional determinant and coefficient identities. |
| `audit_risley_single_witness.py` | The separately scoped independent-axis canonical model's cubic hardware determinant. It is not a physical-vector theorem. |
| `audit_reduced_inverse_exact.py` | Exact-rational checks of the five-root degree recurrence, interlaced divided-difference weights and moments, original-clock phase and speed bounds, and the stated small-amplitude physical guards. |
| `full_prior_tail_witness.py` | Exact rational-interval witness for the large whole-prior absolute Taylor tails, including slope and sine-coordinate Taylor jets through order four. |
| `critical_filtered_tail_witness.py` | Exact rational-interval witness for the triple-critical limiting configuration, positive cascade coefficients and traversal margins supporting the filtered-tail obstruction. |

These scripts are not a practical globally complete solver, a finite-noise recovery certificate, or external peer review. Exact identities at the stated witnesses do not establish global uniqueness, useful conditioning, or full-prior recoverability.

The files reproduce the supplied bytes as UTF-8 without a byte-order mark, with LF line endings and one final LF. Transport-escaped less-than signs were decoded before verification where needed. [manifest.json](manifest.json) records every supplied and verified SHA256 value; all ten match. The sources are intentionally unchanged, including their original notation. In the original symbolic witness scripts, `epsilon` refers to wedge scaling, not the measurement-error parameter used in the updated report.

Companion documentation: [physical oblique inverse](../physical_oblique_inverse.md), [physical inverse certificates](../physical_inverse_certificates.md), [canonical local-rank proof](../full18_local_rank.md), [reduced algebraic inverse](../reduced_algebraic_inverse_addendum.md), [exact collision barrier](../finite_record_collision_barrier.md), [full-prior spectral cover](../full_prior_spectral_cover.md), and [critical filtered remainder](../critical_filtered_remainder.md). The scripts address only their stated witness calculations; the broader theorems also require the arguments and assumptions in those documents. The rational witnesses do not evaluate a useful all-branches cover, the collision amplitude cutoff, or an actual-record noise threshold.
