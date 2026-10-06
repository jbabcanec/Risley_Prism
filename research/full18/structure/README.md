# Original passive full18 structure diagnostics

This experiment preserves the original 200 positions over 10 seconds, the
canonical independent-axis optics, all native bounds, and the supplied physical
prism order. It never sorts the hardware or treats an unknown parameter as a
known calibration input. The twelve shared fixtures comprise eight random cases
and four stress cases. Evaluating their true points here measures local
conditioning; it is **not a blind-recovery experiment**.

Run from the workspace with Python 3.11 and bytecode disabled:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -B full18_research/structure/run_diagnostics.py
python -B full18_research/structure/validate_results.py
python -B full18_research/structure/audit_candidates.py
```

The runner pins BLAS to one thread, imports the Dropbox source read-only, and
writes `results.json` and `results.csv` beside the script. The JSON records
package versions and SHA256 hashes of the case manifest and imported model
sources. `diagnostics.py` also independently checks the new `solver/poc3.py`
paired-face implementation against the existing interface-by-interface map.

## What the numerical results say

The existing `risley_lattice/separable.py` already has the exact factorization

    F(q,c) = b(q) + A(q)c,
    c = (d_W, gap, bm_px, bm_py), q = the other 14 native coordinates.

Eliminating c is optimization over four unknowns, not assuming them known.
At an interior least-squares solution, the first-order reduced derivative is
the projection of the q derivative orthogonal to A, plus a residual-dependent
term. The existing code includes that term. Bounded fits are piecewise smooth
on a fixed active set. The independent derivative tests intentionally use
off-fit candidates with substantial residual, so they exercise this term.

Across the twelve supplied fixtures, the box-scaled affine matrix has condition
number 20.3 to 67.9. The full18 Jacobian ranges from 30,835 to 191,895,630;
the reduced14 derivative still ranges from 18,866 to 191,864,956. Thus a more
accurate four-variable linear solve does not resolve the main material
coupling. Schur-complement weak directions usually combine glass indices and
gap/distance. In `weak_first`, the lifted normalized weak direction has
coefficient 0.9999907 in ng1; the corresponding first wedge is very weak.

The independent source-first construction removes the two disjoint source
columns, solves only a two-column distance/gap problem, and restores source
positions. It agrees with the joint unconstrained affine fit within 9.2e-14
in predicted positions on these cases. This is a lossless algebraic crosscheck,
not a new identifiability result. Clipping its solution would not correctly
solve the bounded affine problem; production fits retain joint bounded LS.

## Hard-error sensitivity, explicitly local

For J evaluated at the stated fixture, let K be its left pseudoinverse, formed
with box scaling for numerical stability. A linearized least-squares estimate
responds as delta_theta = K e. For arbitrary coordinatewise |e_j| <= eta,
the exact bound for this linear map is

    |delta_theta_i| <= eta * sum_j |K_ij|.

No independent-noise or Gaussian assumption is used. This is nevertheless
**only a first-order local sensitivity calculation**: it is neither a nonlinear
error certificate nor a guarantee that the blind solver reaches this basin.
Likewise, the reported twice-eta quantities describe the local linearized
compatible-pair map, not a proven diameter of the nonlinear feasible set.

| Fixture | Worst native coordinate | Amplification per unit eta | At eta=1e-6 | At eta=1e-8 |
|---|---|---:|---:|---:|
| random_00 | d_W | 619.0 | 0.000619 | 0.00000619 |
| random_01 | d_W | 47.26 | 0.0000473 | 0.000000473 |
| random_02 | d_W | 215.2 | 0.000215 | 0.00000215 |
| random_03 | d_W | 157.9 | 0.000158 | 0.00000158 |
| random_04 | d_W | 236.7 | 0.000237 | 0.00000237 |
| random_05 | d_W | 1073.7 | 0.001074 | 0.0000107 |
| random_06 | d_W | 5205.6 | 0.005206 | 0.0000521 |
| random_07 | d_W | 61.82 | 0.0000618 | 0.000000618 |
| moderate_unsorted | d_W | 11968.4 | 0.011968 | 0.0001197 |
| wide_unsorted | d_W | 389.9 | 0.000390 | 0.00000390 |
| exact_collision | d_W | 8523.8 | 0.008524 | 0.0000852 |
| weak_first | ng1 | 37273.8 | 0.037274 | 0.0003727 |

The machine-readable table includes every native coordinate at eta levels
1e-4, 1e-6, 1e-8 and 1e-10. No one precision level is imposed as a task premise.

## Independent checks

* On all twelve fixtures, the old smooth map and new paired-face map agree
  to floating-point roundoff; their box-scaled full18 Jacobians have relative
  Frobenius discrepancies between 5.8e-16 and 1.4e-15.
* The largest canonical-core versus smooth-map position difference is 4.14e-10.
  This finite-precision implementation discrepancy is recorded, not hidden
  inside an assumption of perfectly exact observations.
* On random_00, random_04 and weak_first, the smooth complex-step derivative
  agrees with independent interval automatic-differentiation midpoints within
  6e-16 in relative Frobenius norm; every tested derivative lies inside the
  returned point interval enclosure.
* Five-point finite differences at three step sizes check both smooth and
  canonical models. Smooth relative Frobenius errors reach 4.1e-11 or lower.
  Canonical small-step differences are less reliable for the weak first prism:
  the maximum column error reaches 1.09e-4 at normalized step 1e-6. This supports
  using the smooth algebra for derivatives and the canonical map for independent
  final residual checking.
* Off-fit reduced-Jacobian checks retain the same active sets and have relative
  Frobenius errors below 1.7e-10, including the residual-dependent correction.

These checks do not establish global uniqueness, full-box recovery, or a
finite-noise native-coordinate accuracy guarantee. Their practical implication
is to retain exact variable projection and accurate derivatives while directing
search/preconditioning toward coupled material directions. A small objective
gradient alone is an unsafe parameter-accuracy stopping criterion; the native
correction J^+r is a more informative local diagnostic.

## Returned-candidate checks

`audit_candidates.py` reads observations and saved returned candidates only;
it never reads a true parameter vector or changes the frozen solver. Its
`candidate_audit.json` independently confirms the exact-collision result's
canonical maximum residual 1.189e-11 and RMS 1.760e-12. Interval point evaluation
gives positive physical lower margins: TIR 0.9590, forward direction 0.9934,
and intersection denominator 0.9962. The remaining proposed native Newton
correction is at most 1.46e-9 (gap). The recovered point's local sensitivity
matches the earlier fixture diagnostic: workpiece distance amplification is
about 8524 per unit coordinatewise measurement allowance.

Three frozen failures are not merely precision-polishing misses. Original
random_03 and held-out random_00/random_02 have canonical maximum residuals
18.97, 0.2599 and 10.63. Their unconstrained native Newton corrections reach
193, 32.4 and 713 units and point outside active bounds. In particular,
held-out random_00 has projected KKT-gradient maximum only 5.52e-8 while its
residual remains large: the saved candidate is close to a wrong constrained
local minimum. These observation-only findings support broader spectral and
physical-order candidate exploration rather than stricter optimizer tolerances.
