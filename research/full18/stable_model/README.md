# Stable evaluation of the original full18 optical model

`forward.py` is a standalone NumPy evaluation of the original **independent-axis**
mathematical optics. It preserves all eighteen parameter meanings, their native
box, physical prism order, the source distance 6, reference thickness 3, common
gap, and passive single-plane observations. It is not the different full-vector
model studied in note AC. Original source and historical observations remain
untouched.

The module exposes `vec2pat_stable(theta)` and
`fast_forward_stable(params)`, the latter accepting the existing
`core.PrismParameters` object without importing or modifying the original
module. Both default to the exact original floating timestamp construction:

```python
np.arange(0, time_limit, time_limit/n_points)[:n_points]
```

The optional `times=` argument uses the exact supplied binary64 timestamps.
Replacing them by rational k/20 is not part of this implementation.

## Numerical defect and replacement

The legacy program recovers a signed ray angle from

    sign(sf0) * acos(sf2 / sqrt(sf0**2 + sf2**2))

then converts degrees back to radians and takes its tangent. Near the optical
axis, the cosine differs from one quadratically in the small transverse
direction. It can round to exactly one while `sf0` is nonzero, erasing that ray
direction. The `abs(sf0)<1e-12` override does not protect against this: the loss
can occur for much larger `sf0`.

The new map retains ray components. For incoming unit components `(ix, iz)`,
effective face sine/cosine `(s, c)`, and refractive-index ratio r, it computes

    q = -c*ix - s*iz
    R = sqrt(1 - (r*q)**2)
    ox = -r*c*q - s*R
    oz = -r*s*q + c*R

For the next face `z = Z + m*p`, the homogeneous propagation is

    lambda = (Z + m*p - z) / (oz - m*ox)
    p_next = p + lambda*ox
    z_next = Z + m*p_next

No ray angle, artificial line endpoint, inverse face slope, or determinant
intersection is needed. Effective face sine, cosine and slope are computed
directly from the rotating normal, retaining the canonical per-axis projection.

`atan2(sf0,sf2)` is a useful minimal correction to the dominant defect. The audit
includes an atan2-only ablation that retains all other legacy arithmetic. The
standalone implementation also removes effective-face angle cancellation,
flat-face approximation, and line-endpoint intersection cancellation.

## Branch policy

The module checks strict transmission, positive output axial direction, and
nonzero grazing denominators. It raises `PhysicalBranchError` with the axis,
interface and sample on failure; it does not clip TIR into a synthetic ray.
This is the strict domain used by `risley_lattice/fmodel.py`, rather than an
attempt to reproduce the legacy clamps outside that domain. It adds no
positive-travel-distance assumption. Returned floating margins are diagnostics,
not interval certificates.

## Reproduction and independent reference

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -B full18_research/stable_model/check_forward.py
```

The audit records twelve shared cases, 100 independent draws from the entire
native box without sorting or speed floors, eight saved nonzero-wedge endpoints,
and seven adversarial cases. It records physically rejected draws instead of
silently clipping or replacing them. Checks include:

* Bitwise reproduction of the legacy core by the ablation harness before its
  angle formula is changed.
* Equality of the two standalone APIs and the default versus explicit original
  time-grid paths.
* Comparison with the existing smooth affine implementation and point interval
  forward enclosures.
* An independent 70-digit directed-interval scalar reference (`oracle.py`),
  using exact binary-rational input parameters **and each original timestamp**.
  Every saved/adversarial record is checked at all 200 times; random/shared
  records are checked at their largest legacy discrepancy and two fixed times.
* Saved input vectors, source hashes, branch margins, trace details and error
  upper bounds in `results.json`.

One analytic native-box control uses flat wedges, n=1.5, gap 7, distance 137,
and beam x angle `(180/pi)*2**-30` degrees. At the first entry, the legacy
computed cosine becomes one despite transverse direction about 6.21e-10.
The missing downstream displacement is approximately
`157 * 2**-30 = 1.4621764421463e-7`. The saved nonzero-wedge cases and additional
all-active near-cancellation cases ensure the investigation also covers systems
with nonzero wedges.

## Completed audit, 2026-10-02

The authorized audit completed in 9.72 seconds. All checks passed on the **121
strictly valid cases out of 127**. Six of the 100 fresh native-box draws were
explicitly refused for negative TIR radicands at the final exit face; they were
neither clipped nor replaced. Their IDs are `native_random/004`, `005`, `030`,
`043`, `053`, and `054`, with recorded radicands between -0.2570 and -0.01750.
The full input vectors and refusal locations remain in `results.json`.

Across the high-precision comparisons performed by this audit:

| Implementation | Largest position-error upper bound |
|---|---:|
| Legacy core | 1.5582588395e-6 |
| Legacy arithmetic with only atan2 substituted | 1.1009440e-12 |
| Standalone stable component propagation | 2.5235546e-12 |

The atan2 ablation isolates the dominant defect and also performs well on these
cases. This table does not claim the standalone implementation has a smaller
error at every point; its purpose is to remove the avoidable angle and
intersection round trips while preserving the strict mathematical equations.

The largest legacy error occurred in the **all-nonzero-wedge** case
`adversarial/nonzero_cancel_5`, at the initial x-axis sample. Its independently
bounded legacy error was 1.5582588395e-6, while the standalone error over that
entire 200-sample case was at most 1.3781368e-13. The saved nonzero-wedge
`saved/3/base` failure reproduced at original timestamp index 18 on the y axis:
legacy error at most 1.6757497588e-7 versus standalone error at most
1.2013017e-13 over its full record. Its interval physical margins all exceed
0.993, so proximity to a TIR or grazing boundary does not explain the defect.
The flat-wedge analytic control reproduced the predicted 1.4621764421e-7 error.

Every standalone output at all 200 timestamps of each valid case lay inside
the independent repository point-interval enclosure. The legacy clone matched
the original program bit for bit, both standalone APIs agreed, and the exact
original `arange` timestamp array was preserved. For reference, replacing that
array by the binary64 calculation `np.arange(200)/20` would change some stored
timestamps by as much as 1.7763568394e-15; the high-precision checks did not make
that replacement.

The high-precision maxima above are over all 200 timestamps for saved and
adversarial cases, and the selected worst-legacy-error plus fixed timestamps
for shared/random cases, as specified above. They are **finite test evidence,
not a uniform error bound over the full parameter box or all timestamps of
untested systems**. Strict-branch conditioning still matters near critical and
grazing configurations.

## What changes in error interpretation

A legacy-core-generated position record is not automatically an exact sample of
the strict mathematical map. Its arithmetic error can exceed an assumed small
measurement allowance. A very small legacy-to-legacy fitting residual therefore
does not establish agreement with strict mathematical optics to the same scale.

Stable evaluation removes an avoidable implementation error for future model
evaluations. It does not improve the information content of existing measured or
rounded observations. Keep historical records and model labels intact, report
legacy and strict-model residuals separately, and include arithmetic discrepancy
when a guarantee relates old synthetic data to strict optics. The largest error
observed in this finite audit is not a uniform full-box error bound; near critical
or grazing branches, conditioning still requires explicit treatment.
