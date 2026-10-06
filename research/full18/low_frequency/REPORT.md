# Full18 frequency-proposal repairs, 2026-10-02

Three previously failed passive full18 cases now have strictly physical
observation-selected solutions with independently checked maximum native
coordinate errors 7.55e-12 (02), 2.91e-13 (03), and 1.29e-12 (07). This worker never read
`combined_holdout.json` or its true parameters. The parent performs independent
coordinate-error scoring after outputs are frozen. This is adaptive repair of
the existing cases; the original frozen solver's 7/10 result remains unchanged.

All attempts retain 200 observations at 20 Hz, the original native 18-coordinate
bounds, physical prism order, and all 18 unknowns. Four affine geometry variables
are solved by bounded variable projection. No glass, speed, phase, distance or
beam parameter is supplied as known. The final optical completion frees every
speed even when its initial value came from a spectral proposal. No controlled
measurement, extra timestamp, true parameter or noise-free recovered seed is
used to solve a noisy record.

## Concrete findings and repairs

1. **An obsolete speed cutoff drops an observed physical-speed candidate.**
   `risley_lattice/lattice.py` declares `SPEED_MIN=.10`, explicitly motivated by
   the old battery's speed clamp. `spectral.clean_lines` also discards candidate
   lines below .05 Hz. For random_02 the original line list already contains a
   strong .08155 Hz line (amplitude about 10.36), yet the physical-generator
   selection excludes it. Private process-only lowering of the proposal cutoffs
   to 1e-5, and reducing the lattice merge constant to .001, yields a full18 fit
   with maximum canonical residual 1.34e-12 in 6.45 seconds.

2. **Arithmetic lattice bases need not be the useful physical-speed triples.**
   For random_03 the three strongest FFT lines are approximately
   `[-.256272, 1.482562, -1.539459]`. The existing arithmetic selection substitutes
   a different triple, and simply lowering the cutoff introduces spurious
   near-DC frequencies. `profile_bases.py` always retains the strongest three
   physical-frequency candidates, adds separately DC-fitted slow-frequency
   profiles, screens all six physical orders, then performs full18 completion.
   The retained strong-line proposal converges to numerical precision.

3. **A weak missing frequency becomes clear in the optical residual.**
   random_07 remains unresolved after both preceding policies. Starting from
   the smallest-residual fitted candidate, `residual_release.py` identifies its
   weakest fitted prism deflection, proposes signed frequencies from the actual
   residual spectrum and observed sideband differences, tests all physical
   orders and phases -12/0/+12 degrees, and frees all18 in each fit. Its strongest
   residual-frequency proposal is -2.792358 Hz. The first screen reduces MSE
   from 4.55 to 4.45e-4; full18 completion reaches maximum canonical residual
   1.08e-12. The -2.7947 Hz line was present in the original FFT diagnostics but
   was weaker than several distortion lines. No true speed was supplied.

These are constructive numerical proposal policies, not exact spectral
identifiability theorems. Their successful candidates do not exclude remote
compatible hardware or establish a uniform finite-noise guarantee.

The parent independently scored these frozen outputs in
`../verification_low_frequency.json`: the largest errors are d_W for 02,
bm_py for 03, and d_W for 07. The second profile route also recovers 02 to
1.61e-11 maximum native error. Thus all three are actual full18 numerical
recoveries, rather than conclusions drawn solely from trajectory residuals.

## Preserved experiments

| Case / policy | Seconds | Max canonical residual | Outcome |
|---|---:|---:|---|
| 02, lowered cutoff | 6.45 | 1.34e-12 | Numerical fit |
| 03, lowered cutoff | 67.14 | 20.24 | Failed |
| 07, lowered cutoff | 100.30 | 10.36 | Failed |
| 02, DC/profile candidates | about 30 | near 1e-12 | Numerical fit |
| 03, DC/profile candidates | about 33 | near 1e-12 | Numerical fit |
| 07, DC/profile candidates | 44.52 | 11.42 | Failed |
| 07, residual-frequency release | 38.06 | 1.08e-12 | Numerical fit |

The first two policies were declared and applied identically to all three cases.
The third was a separately declared finite follow-up on the remaining failure.
All outputs, including failed branches, remain in this directory. Any combined
adaptive recovery count must stay separate from the original frozen 7/10 test.

Strict physical margins for final random_07 are TIR .01983, forward .34608,
intersection .86558. All output files contain the chosen full18 vector, actual
canonical residual and physical check. A small residual alone is not scored as
parameter recovery: independent parent coordinate checks are the deciding test.

## Noisy follow-up and execution status

The generic interface reads the actual noisy observations saved by the original
combined solver. Completed random_03 at injected eta=1e-4 uses the same profile
policy without a clean recovered seed. Its full canonical RMS is 6.10e-5,
maximum residual 1.0441e-4, and runtime 42.71 s; parent coordinate-error evaluation
is separate. Because maximum residual exceeds eta, this is not a hard-band
feasibility certificate. No uniform claim follows from one noise waveform.

The noisy random_07 cutoff precursor completed in 44.21 s and remains unresolved
(RMS 2.162, max residual 11.65). The requested residual-frequency continuation
was prepared with a 75 s cap, using only this noisy precursor and the original
noisy combined candidate. It did not execute in this worker: automatic approval
review rejected the command. It also rejected the noisy random_02 profile run.
The stated reason was conflict with the earliest read-only/no-code-execution
request and failure to recognize later parent-delegated authorization. No retry
or execution workaround was attempted. These two runs remain blocked in this worker;
their absence must not be counted as success or failure of the numerical method.
The parent reports that review in the user-facing thread accepted the actual
later user authorization and is executing those prepared commands there.

## Files and reproduction

- `recover_low.py`: first declared private low-cut policy, unchanged original
  repository; original and modified spectral diagnostics are saved per case.
- `profile_bases.py`: finite DC/profile and strongest-line proposal policy.
- `residual_release.py`: finite observation-residual frequency release.
- `run_generic.py`: `--observations`, `--id`, `--out`, repeatable `--candidate`,
  `--mode lowcut|profile|residual`, and `--seconds` interface. It makes private
  source copies changing only file plumbing, identity and the run budget;
  manifests record both original-policy and private-driver hashes.
- `random_*.json`, `random_*_profile.json`, `random_07_release.json`: all noiseless
  results. `noise_1e4/`: completed noisy results and prepared run inputs.
- `source_hashes.json`: policy source hashes. Mandatory proposal metadata uses
  JSON null for an intentionally unscored spectral rank; this serialization-only
  change does not alter candidate generation or numerical fitting.

Generic example, from this directory with the installed Python 3.11 interpreter:

```powershell
& 'C:\Users\josep\AppData\Local\Programs\Python\Python311\python.exe' .\run_generic.py --mode profile --id random_03 --seconds 120 --observations ..\combined_holdout_observations.json --out .\replay_03.json
```

For residual mode pass one or more observation-selected saved parameter files
with `--candidate`; the program selects their smallest full-data canonical MSE.
Inputs may be an observations-only `cases` list or a combined result containing
`actual_observations`. It copies only observations and candidate estimates into
the private run directory. Every script pins BLAS to one thread and disables
bytecode writes. Original research code and the frozen poc3 solver were not
edited.
