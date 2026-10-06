# Same-witness native chord evaluation: corrected-record domain obstruction

## Decisive result

The prescribed bounded evaluation **does not establish** the chord certificate on the full original native ±0.001 box. The automatic chord inverse at the center and the optical input guards were certified separately. The present corrected-record enclosure cannot yet certify the frozen lag-11 recurrence inverse throughout that box. Thus it cannot soundly proceed to rotor isolation, the all-200 coefficient/confluent inverses, the bivariate A2 branch, or a full-box chord Jacobian/Hessian bound.

This is a failed sufficient interval-domain test, **not** a proved singularity of the lag inverse, nonexistence of the bivariate branch, failure of the actual chord map, or instability of the optical inverse.

## Fixed scope

The witness, nominal record, and target were unchanged:

- κ=1/10, all wedge angles arcsin(1/10)·180/π degrees.
- N=(1,7,49)/20; phases zero; t=(1/3,0); source=(1,2).
- h=(3/2,7/5,5/3), n²=(17/8,233/125,13/5), g=3, d=100.
- Exact nominal y=F(θ⋆); no invented measurement-noise allowance.
- Native order: N[3], wedge degrees[3], phase degrees[3], indices[3], beam-slope angles degrees[2], source[2], g, d.
- S=0.001 I, r=1. No target shrinkage, parameter change, damping choice, candidate search, or new nonlinear solve.
- Frozen numerical proposals were reused from finite_wedge_corrector_interval.json. The exact frozen lag inverse D=(P Jc)^−1 P was enclosed, rather than treating a binary proposal as exact.

## Enclosure calculations

The direct native first-jet calculation evaluates the same exact and degree-one/two optical circuits over the whole native box, including sine/arcsine and native-angle transformations. Corrected records are enclosed by

s=F1(θ⋆)−[R1(θ)−R1(θ⋆)],
w=F2(θ⋆)−[R2(θ)−R2(θ⋆)].

Every corrected derivative is taken in the scaled native directions. The mean-value theorem bounds each record increment by the sum of the absolute derivative intervals. In particular, no independent y−F+Fi value subtraction is used.

Direct centered bounds across all 400 real coordinates:

- max |s−s0| ≤ 0.27054370362943536.
- max |w−w0| ≤ 0.30208042278099145.
- The corresponding center derivative row norms are only 0.07028723693169645 and 0.03339534327266989, respectively.

A second extension uses the exact integral identities from quantitative_algebraic_correction.md. A degree-three λ polynomial whose coefficients are native first jets is evaluated on each of eight fixed closed subintervals of [0,1]. Each cell gives uniform, outward intervals for Fλλ/2 and Fλλλ/6 and their native derivatives. Exact nonnegative integral weights are used; this is interval integration, not point-sample quadrature. The integrated and direct derivative intervals are intersected.

Improved centered record bounds:

- max |s−s0| ≤ 0.22559333524007844.
- max |w−w0| ≤ 0.12178782767038183.

These are valid product envelopes of the actual coupled corrected inputs, but their width still discards shared-parameter correlations.

## Frozen lag-11 domain test

The exact six-column DC-constrained center inverse has

- center inverse proposal defect ≤ 1.8657926809353833·10^−16;
- ||D||∞ ≤ 3.664687301667778.

Let A(s)=D Js. At the center A(s0)=I. For each scaled native coordinate q, the interval derivative of A is formed by applying D, the lag shifts, and the Helmert matrix to the corrected-record derivative array **before** taking absolute row sums. This retains signed linear projection cancellations within each derivative enclosure. Summing those derivative bounds gives a nonnegative 6×6 majorant M with |A(s)−I|≤M.

The direct first-jet extension gives ||M||∞≤1.8904345463098835. Its center-linearized analogue is only 0.050811723605083875; neither number should be mistaken for a full-box true derivative estimate.

The λ-integral extension improves the bound to

||M||∞ ≤ 1.3707389514134782.

The deterministic positive-weight proposal w=(I−M)^−1 1 has all components negative. In addition, a separate outward Collatz–Wielandt calculation certifies for the exact dyadic majorant stored in the JSON

1.0359049589940914 ≤ ρ(M) ≤ 1.0359049589940934.

Therefore no positive diagonal weighting can make **this particular majorant** contractive. This statement concerns the enclosure matrix M alone. It does not concern the spectrum or invertibility of an actual physical lag matrix.

The recurrence forcing radius remains broad: the largest component is 2.742286443039695, compared with a center-linearized maximum of 0.009920534818145808. This identifies significant lost dependence across the common native parameters and the 200 timestamps. A better value-correlated or Taylor-model enclosure might resolve the domain; that further compiler work was not performed in this bounded evaluation.

## Separate requested conclusions

1. **Physical input envelope:** independently passes the fixed native box and λ∈[0,1] checks; see native_box_guards_chord_center.md. That does not supply the off-image A2 branch domain.
2. **Automatic chord inverse enclosure:** independently passes at the unchanged center, with ||P||∞≤40099.85616680308 and inverse residual ≤1.9387182380847987·10^−9. See the same memo and endpoint JSON.
3. **Bivariate A2 branch / raw reconstruction envelope:** unresolved. No new beam-root box was asserted, and no failure was inferred from a raw A2 output exceeding original hardware priors or ±0.001. A valid root box would still have to contain every matching native beam input to preserve the left-inverse identity.
4. **Full-box chord derivative/Hessian:** not certified. The center derivative is canceled by exact P by definition, but that gives no finite box bound.
5. **Native ±0.001 self-map and contraction:** not certified. The lag-domain obstruction prevents a sound evaluation of either a direct chord-Jacobian bound or the quantitative Hessian radii inequalities.
6. **Noise ε:** stays symbolic. No positive ε allowance follows from these results because even the noiseless full-box domain has not been certified.
7. **All-400 fit:** y=F(θ⋆) is compatible by construction. This is not recovery from supplied unknown observations, a certificate for an off-model fixed point, or validation of another candidate. No recovered candidate was produced by this evaluation.
8. **Delivered native error / global branches:** no 0.001 delivered-estimate or all-prior guarantee. Alias, prism, root, and other global branch exclusion remain unresolved.

## Reproduction

New files only were created:

- proof_checks/finite_wedge_native_chord_envelope.py, .json, .log, and _arrays.json.
- proof_checks/finite_wedge_native_integral_envelope.py, .json, .log, and _arrays.json.
- proof_checks/finite_wedge_native_integral_spectral.py, .json, and .log.
- proof_checks/finite_wedge_native_lag_majorant_audit.py and .json.

Run the direct-envelope script first, then the integral-envelope script, then its spectral analysis and the majorant audit. They use installed NumPy/mpmath and the existing optical proof circuit; no software was installed. The integral optical stage took about 256.5 seconds, including endpoint serialization.


## Archival limitation and stopped replay

The completed eight-cell evaluation preserved its aggregate outward first-jet arrays,
the resulting six-by-six operator-radius matrix, all reported bounds, code, and logs.
It did not initially serialize the coefficient arrays of each λ cell separately.
An identical archival replay was started to add them, then stopped immediately when
the parent closed computational checks. Only the completed replay cells in
proof_checks/finite_wedge_native_integral_cells/ are available. The replay log is
proof_checks/finite_wedge_native_integral_cells_replay.log. Missing cell artifacts
are not claimed to exist, and no further numerical work is active. The original
completed aggregate eight-cell result is unchanged and remains reproducible from
its preserved code.

The mathematical bottleneck is loss of shared phase and parameter dependence
before the six-by-six lag inverse: the 200 samples depend on the same three
frequencies and eighteen physical parameters, but many of those correlations are
replaced by independent interval ranges. A possible structural improvement would
project the exact finite-harmonic quadratic term F2−F1 through the lag operator
before bounding it, then bound the higher-order remainder separately. That is not
implemented here and is not a proposed search over cases, wedges, or boxes.
