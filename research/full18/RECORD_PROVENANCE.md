# Available-record provenance

Read-only source audit, 2026-10-03. No project code, simulator, solver, or proof script was run. No observation was generated, extracted into a new dataset, perturbed, or modified. The original Dropbox research remains unchanged.

## One identified original-protocol record

The clearest existing record for the original **three-prism, 200-pair, 20 Hz protocol** is:

- File: `C:\Users\josep\Documents\Codex\2026-10-02\task\full18_research\observations.json`.
- Record selector: `cases[0]`, with `id == "random_00"`; its ID is at file line 266 and its `observed` array immediately follows.
- This is one record within a twelve-record container. No claim is made that it is the user's unique intended dataset or that it was collected from hardware.
- Top-level schema: `timestamps`, `names`, `lower`, `upper`, `cases`. Each case has `id` and `observed`. The selected `observed` array contains exactly 200 rows of exactly two numeric values, ordered `[x,y]`.
- `timestamps` contains exactly 200 values. Every stored value equals the binary64 computation `k*0.05` for `k=0,...,199`. Intended ideal clock: `t_k=k/20` seconds, from 0 through 9.95 seconds; the final stored decimal is `9.950000000000001`. No physical clock certification is implied by this format check.
- First pair: `[70.89623746590591,40.09534334142048]`.
- Last pair: `[3.2970413153658784,69.94469232656722]`.
- Screen coordinates use the source model's native length units; no millimetre calibration was found.
- File SHA256: `fc61cf3e7181c78c206370e0fe022246ed538bbec857923a1354ae1cc5417d74`.

This is a **canonical independent-axis synthetic record**, not a verified full-vector physical record or hardware measurement. It was saved during the earlier workspace investigation on 2026-10-02. “Original-protocol” describes its sample count and clock; it does not mean an original hardware acquisition.

## Verified provenance chain

The companion [cases.json](cases.json), line 265, explicitly declares:

> canonical reduced per-axis; source distance 6, thickness 3; common gap; P=3

Its `screen_protocol` at line 271 is “one passive plane, 200 timed positions; no interventions.” It includes the seed 20261002, original native parameter bounds, generation/evaluation truth, and source-file hashes. Its first case's `observed` values match the selected observation-only record exactly after JSON parsing. No truth values are required to identify the record.

[contract.py](contract.py), line 38, obtains the observations with `y = vec2pat(v)`. Lines 57 and 66–70 save those values and the timestamps. The generation path does not add a measurement perturbation to this particular base record. This does not establish zero discrepancy against the full-vector model.

In the original Dropbox tree, whose root is `C:\Users\josep\Dropbox\Babcanec Works\Mathematics\Wedge`:

1. `risley_lattice\model.py:23–25` defines 200 points, a 10-second configured interval, and a 0.05-second step.
2. `risley_lattice\model.py:49–58` maps the native eighteen-vector to `reverse_problem_v2.core.fast_forward`.
3. `reverse_problem_v2\core.py:182` constructs the timestamp array.
4. `reverse_problem_v2\core.py:200–203` evaluates `_trace_axis` separately for x and y and stacks the results. This implementation is independent-axis refraction, despite its descriptive use of “physical wedge prisms.” It is not the coupled three-dimensional vector-Snell map of the current theoretical report.
5. `reverse_problem_v2\core.py:208–244` implements that scalar-axis trace.

The current source hashes were read and match the stored generation hashes:

| Source | SHA256 |
|---|---|
| `risley_lattice\model.py` | `4422748bd8b979c75503191f1fe44ec336b457a24bd48ecb50bcc6af78b34d8e` |
| `reverse_problem_v2\core.py` | `7e25bc358413cabc991b970624fc9125f63b8266b0b922e104d69930dfa2c4a6` |

The companion `cases.json` SHA256 is `250e9d42aba6a81fcb26c0683767a751b552e1e6b888606c12b660ed5fd22cec`. These checks identify the saved source/data chain; they are not a new execution-based reproduction.

## Other candidates and why they are different

- `C:\Users\josep\Dropbox\Babcanec Works\Mathematics\Wedge\experiments\results\poc2_2026_09_29\scans\hardware_box.csv` has header `t,x,y` and exactly 200 data rows at the nominal 0.05-second step. It is **synthetic two-prism canonical data**. `experiments\POC2_HARDWARE_BOUNDS.md:104` explicitly calls it synthetic. `experiments\poc2_recovery.py:1` identifies the two-prism experiment; lines 197–208 generate `canonical(truth, args.samples)`, add a deterministic perturbation, and export the CSV. Its filename is not evidence of a hardware measurement. It is not the original three-prism all18 problem.
- `...\experiments\results\full18_target_2026_09_16\162450.inputs.npz` contains `x0.npy` with shape `(18,)` and `observations.npy` with shape `(1600,2)`, both little-endian float64, as directly read from the NPZ/NPY headers without loading project code. Its companion `162450.manifest.json:30–37` specifies 1600 points over 80 seconds and “Regenerated synthetic observation with current case_at/vec2pat; original cloud observation array/revision unavailable.” It is not a saved original200 record. The same target directory also has other 1600-point archived-case inputs.
- The inspected legacy gallery example `...\reverse_problem_v2\output\examples\20250928_130633_simulation\workpiece_projections.csv` has 60 data rows over 0–5.9 seconds at 0.1-second intervals, as confirmed by its companion analysis file. It is not the original protocol.
- `...\experiments\results\p3_vector_2026_10_01\vector_layer_stripping.json` explicitly concerns full three-dimensional vector Snell, but records symbolic identity results, with `"optical_trajectories_or_fits": false`. It contains no measured/simulated 200-pair record.

## Consequence for the next certificate

No saved 200-pair **coupled full-vector or hardware** observation record was verified in the inspected research/workspace files. This is a provenance limitation, not a proof that such a file is absent everywhere on the computer. The inspected scope includes the current workspace observation files, original experimental result files, relevant simulator/generator sources, and the original gallery/archived input candidates above.

The selected canonical record is available and fully identified. It may be used for a canonical-model study, or as arbitrary fixed data whose consistency with the full-vector model must itself be tested. It cannot be assumed to have a full-vector generating truth merely because it has the right eighteen-coordinate labels and 200-sample clock. A targeted certificate asserting recovery of a full-vector generating system needs a supplied or otherwise identified full-vector/hardware record and its error/encoding contract. No noise level has been chosen and no such consistency or recovery computation has been performed in this audit.

