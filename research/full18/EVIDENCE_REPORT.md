# Passive full18 Risley recovery: construction and evidence

Research checkpoint, 2026-10-02. Original research: `C:\Users\josep\Dropbox\Babcanec Works\Mathematics\Wedge`. New scripts, observations, results and notes are confined to this workspace folder. No original research file was edited.

## What is now established

There is concrete progress in finding all eighteen unknown coordinates from the original passive 200-position record. The frozen combined solver recovered 7/10 fresh noiseless systems. Observation-driven repairs recovered the remaining three, giving **10/10 after adaptation**. This is empirical evidence for several constructive routes, not a 10/10 result for a previously frozen unified algorithm or a global theorem.

There is also an exact reduction of bounded-error geometry recovery to a **two-dimensional polygon** conditional on the fourteen optical coordinates. It eliminates the two source positions analytically, retains all native bounds, handles hard observation strips, and needs no assumption that a least-squares answer is feasible. Exact sampled rotor recurrences and branch-preserving algebraic Snell equations provide a separate route to complete set reconstruction. The mathematical notes linked below distinguish exact equivalence from practical computational complexity.

Uniform `.001` accuracy over the entire native box is nevertheless impossible for some records at the tested positive input allowances. Independent interval calculations verify distinct full18 systems compatible with the same record. This does not prevent recovering a compatible system, listing alternatives, or proving tighter accuracy on other records. A gain-coordinate construction now finds a compatible solution for a record on which ordinary completion stalled, and then exposes its remaining uncertainty.

## Exact problem and conventions

The native vector is `(N1,N2,N3, ax1,ax2,ax3, ay1,ay2,ay3, ng1,ng2,ng3, d_W,gap,bm_ax,bm_ay,bm_px,bm_py)`.

| Coordinates | Meaning and units | Native bounds |
|---|---|---|
| `N1..3` | Signed rotations, Hz | `[-3.5,3.5]` |
| `ax1..3` | Signed wedge angles, degrees | `[-18,18]` |
| `ay1..3` | Rotor phases at the given time origin, degrees | `[-18,18]` |
| `ng1..3` | Glass indices | `[1.3,1.8]` |
| `d_W`, `gap` | Last-exit reference height to screen; common inter-prism gap | `[50,200]`, `[2,15]` |
| `bm_ax,bm_ay` | Source angles, degrees | `[-25,25]` |
| `bm_px,bm_py` | Source-plane coordinates | `[-5,5]` |

Lengths use the source model's native units, not an assumed millimetre calibration. Source distance 6 and thickness 3 are fixed. Physical prism order matters; scoring never permutes prisms. The canonical model refracts the two transverse axes independently, rather than using full-vector 3D Snell optics.

The intended observations are 200 pairs at `t=k/20`, `k=0..199`: a configured 10-second interval whose last sample is 9.95 seconds. The executable uses stored binary64 `arange` timestamps. Numerical audits preserve those exact stored timestamps. Exact-clock algebraic theorems explicitly use rational `k/20`.

Noise means a hard per-coordinate bound `|y-F(theta)| <= eta`. The target is `.001` in every native parameter coordinate. `eta=1e-8` was an assistant-selected prior research benchmark, not a precision specified by the user. This investigation sweeps allowances instead of silently making the input arbitrarily precise. Internal arithmetic precision does not increase observation accuracy.

The exact Snell and intersection recurrences are implemented and explained in [stable_model/README.md](stable_model/README.md) and [stable_model/forward.py](stable_model/forward.py). The mathematical reference in the original project is `risley_lattice/fmodel.py`, especially `_trace_axis`; `reverse_problem_v2/core.py` is the legacy floating evaluator.

## Prior research, with its scope preserved

The latest substantive source is `DIARY.md`, particularly lines 10032 onward, and `paper/P2_THEORY_2026_09_28/BRIEF.md`.

* `Y_global_quartic_shape.md` constructs twelve-coordinate P3 shape inversion from thirty coupled coefficients over its declared R5 family. Those coefficients are not direct measurements in the passive 200-record problem. The unresolved bridge is quantitative finite-record observability with unknown rotors.
* `AA_two_screen_angular_inverse.md` and `AB_two_screen_geometry.md` give a controlled paired-plane protocol: 52 two-dimensional readings, calibrated parked states and synchronization, a restricted R5 family, and explicit finite-noise compatible-pair bounds. This changes the supplied information; it is not a result for the original passive record.
* `AC_vector_layer_stripping.md` gives an exact inverse for a different, full-vector optical model and controlled states. Its uniform finite-noise bound is not proved.
* The September `P3NativeRecovery.lean` result closes a narrow calibrated-box claim. It does not establish broad native-box passive recovery. The diary separately records local injectivity for 15,759 admissible archived specifications and 504 saved recoveries. Neither count proves global exclusion or completion of the remaining archived cases.

## Benchmark design and frozen-policy results

[contract.py](contract.py) draws independent uniform coordinates across the original rectangular prior. It neither sorts physical speeds nor clips slow speeds. Only failure of strict sampled physical transmission rejects a draw, and rejected draws are saved. Truth vectors are separated from the observation-only solver inputs. The original `model.battery_cases` had sorted speeds and a small-speed floor; those restrictions were not carried into the new generator.

The initial eight random cases exposed failure of the original standard `solve18`: **0/8** met `.001`, while a separate moderate control did. The new all-order bounded variable-projection solver plus FFT fallback recovered those eight and six additional random holdouts after development/adaptation. Those repaired cases are development/regression evidence, not a pristine combined-policy validation population. See [baseline/REPORT.txt](baseline/REPORT.txt) and the `verification_*.json` files.

A fresh ten-case batch was then drawn with seed `20261005`, after the combined pencil/FFT policy was set. All ten cases remain in every denominator. The same fixed bounded waveform is added at each nonzero level:

`e[k,a] = eta * sin((k+1)*(sqrt(2) + a*sqrt(3)))`, `a=0,1`.

| Added-noise allowance | All18 native errors <= .001 | Fits the strict-model hard band |
|---|---:|---:|
| 0 | 7/10 | Numerical noiseless experiment, not exact eta=0 feasibility |
| 1e-6 | 7/10 | 7/10 |
| 1e-4 | 6/10 | 7/10 |
| 1e-3 | 2/10 | 7/10 |
| 1e-2 | 0/10 | 8/10 |

The complete fifty-run table is [ACCURACY_SUMMARY.md](ACCURACY_SUMMARY.md), with every coordinate error in [FULL18_ACCURACY.csv](FULL18_ACCURACY.csv) and vectors/checks in [ACCURACY_SUMMARY.json](ACCURACY_SUMMARY.json). For example, fresh `random_01` at `eta=1e-4` fits every observation strip but misses `d_W` by `.0012362186`. Compatibility and parameter accuracy are different tests. A single deterministic noise waveform is not worst-case noise coverage or a statistical performance estimate.

Generation used legacy binary64 `vec2pat`. Initial residual scoring also used that evaluator; native parameter errors compare saved vectors directly and do not depend on a forward evaluator. Strict-model point enclosures were separately checked. Stable-evaluator cross-checks preserve all original records and are reported separately; no old record is silently regenerated. No added noise means legacy-generated positions without an added perturbation, not infinitely precise real observations.

### Repairs of the three fresh noiseless failures

| Case | Frozen max native error | Observation-only repair | Repaired max native error |
|---|---:|---|---:|
| `random_02` | 19.9782 | Remove unsupported `.1 Hz` proposal floor | 7.54596e-12 |
| `random_03` | 15.9310 | Preserve strong measured lines as physical generator proposals | 2.91323e-13 |
| `random_07` | 33.9906 | Fit stronger components, propose missing speed from residual spectrum, release all18 | 1.28608e-12 |

All repairs used the observed record, original bounds and physical order alternatives. They did not receive the hidden truth. The final case has a positive TIR margin about `.01983`, so it was not excluded by imposing a stronger transmission margin. All failed preliminary attempts remain in [verification_low_frequency.json](verification_low_frequency.json). Multiple attempts for one case must not be counted as different validation cases.

The same repair routes were then applied to the three actual noisy records at `eta=1e-4`, starting from noisy-data candidates rather than their recovered noiseless hardware. A final all18 minimax step made the per-coordinate objective match the hard bands. The adapted cohort now has **9/10 all18 accuracy passes and 10/10 compatible fits**. The three repaired errors are `.0003784823` (`ax3`), `.0000181214` (`bm_px`), and `.0000631222` (`d_W`), respectively. The remaining accuracy miss is the already compatible fresh `random_01`; its distance error is `.0012362186`. See [ADAPTED_RESULTS.json](ADAPTED_RESULTS.json), [score_adapted.py](score_adapted.py), and [low_frequency/refine_noise_bands.py](low_frequency/refine_noise_bands.py). These adaptations do not change the frozen 6/10 count at that noise level.

Repeated-speed cancellation has a separate constructive initializer, which models two signed contributions to one complex line. It recovered the development collision and two fresh repeated-speed holdouts. The latter had full18 maximum errors `7.11e-13` and `3.60e-11`; see [collision/REPORT.md](collision/REPORT.md) and [verification_collision_holdout.json](verification_collision_holdout.json). The development collision's truth had been visible in the initial research context; the new holdout solver inputs contained no truths.

## Mathematical construction now available

### Exact geometry reduction

At a prism entry let the external unit horizontal ray component be `X`. For effective face sine/cosine `(s,c)` and index `n`, define

`H=sqrt(n^2-X^2)`, `Q=c*X+s*H`, `R=sqrt(1-Q^2)`,
`X'=c*Q-s*R`, `Z'=s*Q+c*R`, `P=c*H-s*X`.

The physical roots are positive; `Z'>0`. Within the native bounds, `P >= cos(18deg)*sqrt(1.3^2-1)-sin(18deg) > .48`. Refraction through the paired flat/tilted faces and propagation over the following reference gap `ell` give exactly

`p_next=L*p+K+ell*t`, where
`L=R*H/(Z'*P)>0`, `K=3*X*R/(Z'*P)`, `t=X'/Z'`.

For three prisms the successive `ell` values are `g,g,d`. At each axis and sample,

`F=A*p_source+B+C*g+D*d`,
`A=L3*L2*L1`,
`B=6*A*tan(beta)+L3*L2*K1+L3*K2+K3`,
`C=L3*L2*t1+L3*t2`, `D=t3`.

All four coefficients depend only on the fourteen optical coordinates. This proves the geometry affine structure directly. Normalizing by positive `A`, put `z=(y-B)/A`, `U=C/A`, `V=D/A`, `e=eta/A`. The admissible source position is the intersection of `[-5,5]` and every interval `[z-gU-dV-e, z-gU-dV+e]`. This intersection is nonempty precisely when each lower endpoint is no larger than each upper endpoint. Both axes therefore produce a polygon of feasible `(g,d)`, intersected with their native rectangle. Source intervals reconstruct every remaining compatible geometry.

This is an exact conditional reduction, not recovery of the fourteen optical coordinates. Near a transmission boundary, `A` can approach zero and normalization becomes poorly conditioned. The unnormalized four-variable interval-LP formulation is the certificate backend in that situation. Approximate LP status alone is not a proof: directed evaluation of a dual support inequality supplies a replayable exclusion certificate.

The standalone [elimination benchmark](theory/elimination_benchmark.json) compared the polygon with the full four-variable LP at four observation-derived optical candidates. Feasibility agreed in all four; the three feasible geometry extrema agreed within `4.44e-16`. The naive polygon took `.60` to `1.09` seconds on feasible cases, versus `.013` to `.035` seconds for the four-variable LP. Therefore the mathematical dimension reduction does not justify claiming a speed improvement from this implementation. These are floating numerical checks of the exact reduction; they are not rigorous infeasibility certificates.

In the noiseless case, subtract sample 0 separately on each axis after normalization. The resulting 398 equations are `Delta z = g*Delta U + d*Delta V`. A rank-two pair determines `(g,d)` and the remaining augmented 3-by-3 minors must vanish. Rank-one and rank-zero charts require their own lower-order consistency conditions; a rank-two derivation must not discard them.

The derivation, sparse dual witnesses and box exclusion inequalities are in [theory/elimination.md](theory/elimination.md). The complete outer search still has to exclude every incompatible optical box; the polygon alone does not establish global recovery.

### Gain coordinates and constructive uncertainty

[gain_continuation/full_domain_gain.py](gain_continuation/full_domain_gain.py) uses `u_i=(n_i-1)*tan(ax_i)`, retaining the exact coupled constraints `|u_i| <= (n_i-1)*tan(18deg)`. It covers the original full wedge/glass domain, unlike an artificially small rectangular gain prior. All18 remain unknown; four affine coordinates are solved within their bounds.

Starting only from a stalled observation-derived estimate, this chart reduced residual about `.0008647` to a stable-model residual about `1e-13` in 56 evaluations. The interval bound is `3.34e-11`; the legacy residual is `3.60e-10`, both below the record allowance `1e-8`. Domain roundtrips, boundary coverage and a complex-step derivative check passed, with maximum relative derivative difference `3.77e-12`. The same route did not fix wrong-frequency basins in fresh `random_03` or `random_07`. See [gain_continuation/REPORT.txt](gain_continuation/REPORT.txt).

### Complementary methods, including negative results

| Method | Measured outcome | What it establishes |
|---|---|---|
| Exact bounded affine projection and all six physical orders | Core constructive improvement | All18 remain free; order is not a scoring symmetry |
| Low-mode harmonic weighting | 2/4 development recoveries versus direct 4/4 | Suppressing nonlinear information can hurt |
| Harmonic emphasis | 3/4; weak prism still fails | No robust replacement for direct completion shown |
| Block-phase continuation | 3/4 development all18 accuracy; no repair of two separate failures | Releasing rotor phase coherence can help some basins |
| Explicit constrained ray-state optimization on 40 sample nodes, then full200 polish | 0/4 | This tested formulation did not solve the problem |
| Full-domain gain chart | Resolves one severe hardware valley; fails two wrong-frequency basins | Useful coordinate change, not global convergence |

See [harmonic/COMPARISON_REPORT.txt](harmonic/COMPARISON_REPORT.txt) and [shooting/REPORT.txt](shooting/REPORT.txt). These failures remain part of the evidence. The new reflection-tied six-rotor Chebyshev frontend under mathematical study is a different proposal from the already tested harmonic weighting.

## Verified finite-noise alternatives

[profiles/verify_profile_pairs.py](profiles/verify_profile_pairs.py) independently evaluates every one of the 400 coordinates per endpoint with 70-digit directed intervals. It imports neither the original optical program nor parameter truths. Inputs are exact stored binary64 parameters, timestamps and observations; allowances are interpreted as their stated decimal values. Positive branch margins and native bounds are checked.

| Actual stored record | Allowance | Endpoint error upper bounds | Distance separation / minimax lower bound |
|---|---:|---|---|
| Original development `random_00` with bounded noise | 1e-4 | 9.6309208611e-5; 9.7089081406e-5 | .0022 / .0011 |
| Observation-only gain-continuation record | 1e-8 | 9.7222858559e-9; 9.7221680456e-9 | .0022 / .0011 |

The first record is **not** fresh combined-holdout `random_00`. Full vectors, records and checks are in [profiles/independent_pair_verification.json](profiles/independent_pair_verification.json), [profiles/random00_eta1e4.json](profiles/random00_eta1e4.json), and [gain_continuation/compatible_profiles.json](gain_continuation/compatible_profiles.json).

For either pair, every single returned distance is at least half their separation away from one of the two compatible distances. This proves a record-specific obstruction to a uniform `.001` point guarantee. It does not prove that the two endpoints exhaust the compatible set, or that no useful answer can be returned.

[ambiguity/witnesses.json](ambiguity/witnesses.json) also preserves constructed shared-record witnesses checked independently at exact rational timestamps, plus legacy executable comparisons. The `1e-8` example uses nonzero wedges and separated ordinary speeds, rather than zero-speed or coincident-rotor ambiguity. A larger-wedge example at `1e-4` has unavoidable errors at least `.025` in distance, `.013978` in gap and `.001617` degrees in one wedge. See [ambiguity/SUMMARY.txt](ambiguity/SUMMARY.txt). These are constructed records, not a success-rate population.

## Numerical model correction and verification

The legacy evaluator recovers a small angle through `sign(sf0)*acos(sf2/norm)`. The cosine can round to one while the transverse component is nonzero, erasing the outgoing slope. The standalone stable evaluator carries ray components and intersects surfaces directly. It preserves the original mathematical independent-axis model, original physical order and floating timestamps; it refuses invalid physical branches rather than clipping them.

The audit contains 127 inputs: 121 strict physical cases and six explicitly recorded TIR refusals. The largest checked legacy error was `1.55826e-6`, on a case with all wedges nonzero. The largest checked stable error was `2.52355e-12`. All valid stable records lie within repository interval enclosures. Saved/adversarial records were checked at all200 times with an independent high-precision oracle; random cases used selected informative times. These maxima are finite audit results, not uniform bounds near critical/grazing branches. The atan2-only ablation also removed the main cancellation defect and is reported candidly.

Original observations and scores remain intact. Independent witness verification is essential when an allowance is smaller than legacy forward-evaluation error. In particular, the independently checked pairs above survive the stable mathematical evaluation; their conclusions do not rely on matching two executions of the same flawed legacy routine.

The subsequent [saved-evidence cross-check](stability_audit/REPORT.txt) covered 155 comparisons, 102 distinct parameter/time inputs and 51 full-record interval-oracle evaluations. It checked all thirty benchmark truths, all fifty frozen candidate/actual-record pairs, saved repairs and nine witness pairs. The maximum legacy/stable difference in this scored collection was `1.63693e-9`, and the maximum checked stable/oracle error upper bound was `4.57223e-11`. Every positive-noise classification and native accuracy count was unchanged; every noisy truth remained inside its declared allowance; all nine witness pairs passed the independent oracle. The only eta=0 classification change is expected: a legacy-generated record equals the legacy truth output bit-for-bit but not the exact mathematical output. Noiseless counts use numerical fit and direct parameter errors, not literal zero-error compatibility. A historical failed `random_07_profile` was refused as physically invalid; its successful replacement passes. Full records and hashes are in [stability_audit/results.json](stability_audit/results.json).

## Reproducibility and limits

Use Python with NumPy, SciPy and mpmath; the current environment uses Python 3.11, NumPy 1.26.4 and SciPy 1.10.1. Set `PYTHONDONTWRITEBYTECODE=1` or use `python -B`. Original imports are read-only. `contract.py` records five original source hashes, all of which were independently checked unchanged; this is not a hash audit of every repository dependency.

```powershell
python -B full18_research/recover_scan.py --observations full18_research/combined_holdout_observations.json --id random_00 --out candidate.json
python -B full18_research/validate_combined.py --etas 0 1e-6 1e-4 1e-3 1e-2 --workers 2
python -B full18_research/summarize_accuracy.py
python -B full18_research/stable_model/check_forward.py
python -B full18_research/profiles/verify_profile_pairs.py
```

`recover_scan.py` is the frozen pencil/FFT candidate finder. It reports compatibility and local sensitivity separately and explicitly sets global accuracy and complete-set enumeration to false. The low-frequency, cancellation and gain repairs are separate reproducible research routes; they have not yet been combined into a newly frozen and independently validated production solver. Wall-time caps and concurrent workload can change how many proposed starts are explored; saved outputs preserve the measured experiments.

The outstanding mathematical goal is a practical global cover or exclusion method for the full compatible set, followed by a per-record native accuracy certificate or an explicit uncertainty report. Small residual, numerical local rank, one local interval certificate, and success on finite synthetic cases do not supply that global conclusion. No user-selected actual measurement precision or real measured scan was supplied in this investigation.
