# Independent audit of the joint LP and adaptive cover

**Result: passed on 2026-10-03.** The executable [checker](verify_adaptive.py) completed with exit 0. The [machine-readable review](adaptive_independent_review.json) records source hashes, exact rational results and rejected mutations. A second independent read-only source review found no soundness defect in the validated request-to-certificate path.

This is a finite code and certificate audit, not formal verification, an all-prior inverse result, or a practical runtime claim. The research report and earlier forward-engine evidence were not changed.

## Scope and input

The extension is [joint_lp.py](joint_lp.py) and [adaptive.py](adaptive.py). It uses unchanged [engine.py](engine.py) and [interval.py](interval.py); their hashes match the [earlier independent review](independent_review.md).

Exactly one existing [synthetic full-vector record](spotcheck/synthetic_fullvector_record.json) was reused, with SHA-256:

    1c83fb2ce416e7e52971e0feb0c7e2f70d91e5d92088d516a3b3ce8aad99de9c

It contains 200 paired observations at exact k/20 times. No new observation generation, hardware or canonical per-axis record, fitting, sweep, original-project execution or installation was used. Test boxes are supplied; they are not recovered from the observations. All eighteen coordinates remain represented, and every child keeps the same four shared geometry intervals.

## Arithmetic proof

For each interval affine form, the LP takes coefficient midpoints and lowers the constant by the constant radius plus the sum of each coefficient radius times the largest absolute geometry endpoint. The lowered row is no greater than the true violation throughout the optical enclosure and shared geometry rectangle.

The independent checker re-identifies observation and traversal forms from forward evidence. At all sixteen geometry vertices it minimizes the signed coefficient intervals exactly and verifies the lower row. The difference between the exact coefficient lower support and the proposed affine row is concave in geometry, so vertex checks establish the inequality on the complete rectangle. The audit checked **64,160 row/vertex inequalities across saved certificates**, including the complete 2,000-row construction twice with different resource settings.

For positive weights summing exactly to one, the checker directly minimizes the weighted normal over those sixteen vertices. This independently verifies the geometry support correction without assuming stationarity. A positive corrected lower bound excludes the weak affine system. A zero bound excludes strictly physical systems only with positive weight on a strict traversal row. Weak zero and negative bounds do not exclude.

At most five sampled affine rows are explicitly weighted; geometry-box support is additional. Arbitrary nonoptimal proposals are not claimed to be minimal five-total-row Helly circuits. Partial forward rows may prove a contradiction, but missing rows never imply complete feasibility. A feasible relaxed LP does not prove physical feasibility.

Forward arithmetic remains outward 80-bit dyadic arithmetic for these inputs, backed by exact integers and fractions. Subsequent LP, coefficient-error and support calculations use exact Fraction arithmetic. No floating stationarity tolerance or residual is discarded. Wall-time fields in the generation summary are informational. Low-level Row and LP helpers assume trusted exact-rational data; external requests should use the validating JSON/CLI interface.

## Cover and resource evidence

The checker derives the root from the input and every child from its parent's exact rational midpoint. It does not trust stored child endpoints. Only fourteen optical coordinates may split; all four geometry intervals remain unchanged. Closed children overlap at the midpoint and their union is the complete parent.

Every terminal node is excluded by replayed evidence, retained by a whole-box forward proof, or explicitly unresolved with its entire box preserved. Duplicate, orphan and missing nodes are rejected. The ordinary adaptive verifier replays the bounded LP run. The independent checker additionally verifies primal inequalities and dual support directly; its mathematical bound check does not need the optimizer. The sparse-only verifier replays an exclusion without running the optimizer.

[Saved results](adaptive_checks/results.json):

| Check | Verified outcome |
|---|---|
| Existing small all18 box | One evaluation; retained by a whole-box physical/observation proof. |
| Existing distant-screen box | One evaluation; excluded. |
| Full-prior outer box | Three evaluations, depth limit two, seven nodes: three splits and four unresolved frontier leaves; no LP pivots. |
| Zero forward budget | Entire root remains unresolved; no evaluation. |
| Zero LP pivot budget | Small optical box with full original geometry remains unresolved at depth zero; 2,000 rows retained, zero pivots. |
| Expanded-geometry joint LP | All 800 observation and 1,200 traversal rows present; 32-pivot limit reached, unresolved. A positive current primal objective is not treated as an exclusion. |
| Distant-screen joint LP | Ten available rows suffice for a one-row sparse exclusion, zero pivots; missing rows explicitly reported. |

The distant-screen joint certificate has exact lower bound:

    318993470498614242301291614495377/120892581961462917470617600000000

Its normal is nonzero and the geometry support penalty is included. Sparse-only replay and rejection of a forged bound passed.

Budgets count evaluations, depth and pivots, not wall-time, memory, row scans or rational bit growth. Every queued branch survives a resource stop. The full-prior result is a coverage-preserving **unresolved cover**, not a solved inverse.

## Exact unit checks and tamper tests

Small deterministic rational LP identities, separate from the optical record, passed:

- Two individually possible rows have joint lower bound exactly 1/4.
- Strict zero-bound exclusion is accepted; the all-weak variant is not excluded.
- A normal residual incurs support penalty -1, converting an uncorrected constant 1/2 to the valid lower bound -1/2.
- Zero pivots leave the optimizer unresolved.

All six tree mutations were rejected: deleted leaf, altered split midpoint, altered leaf endpoint, changed observation input, unsupported retention, and unsupported recovery claim. Three generic dual mutations (weights, strict flag and support correction) and one altered stored sparse lower bound were also rejected.

A source-review suggestion to bind each node's allowance to the LP certificate's recorded pivot limit was implemented before the final generation and audit. Exact used-pivot and total-budget checks also pass.

## Limits and reproduction

The extension implements shared-geometry necessary conditions, replayable sparse exclusions and a sound bounded cover. It does not implement the exhaustive algebraic fallback, establish useful full-prior localization, certify continuation, compute exact native-coordinate envelopes, or demonstrate real-record recovery. Interval dependence and uncertain optical guards leave the tested full prior unresolved. Practical recovery still needs a qualifying full-vector record and a checked cover resolving every remaining branch.

Run from the certificate-engine directory:

    python3 verify_adaptive.py

[Review JSON](adaptive_independent_review.json) records the successful source and checker hashes. Certificate hashes intentionally reject changed proof code.
