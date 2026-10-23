# Archived experiment scripts

Historical one-off experiments, superseded solvers, and debugging scripts.
Kept as the factual record behind DIARY.md; **not maintained**. They were
written to run from the repo root with `paper/` on sys.path — after the move
here their relative sys.path inserts point one level too deep, so rerunning
one needs a one-line path fix (or run from `paper/archive/` with
`sys.path` adjusted to `../..` and `../../reverse_problem_v2`).

## Superseded by the `risley_lattice` package + `experiments/` (2026-07)

| archived | successor |
|---|---|
| spectral_speeds.py       | risley_lattice/{lattice,spectral}.py |
| solve9_spectral.py       | risley_lattice/{angles,solve}.py, experiments/solve9_battery.py |
| solve18_spectral.py      | risley_lattice/{angles,solve}.py, experiments/solve18_battery.py |
| certify.py               | risley_lattice/certify.py, experiments/certification.py |
| test_assumptions.py      | experiments/assumptions.py |
| test_noise_spectral.py   | experiments/noise.py |
| diag_lattice.py          | experiments/assumptions.py (A4) |
| test_matrix_pencil.py    | v1 record (per-line arithmetic, 8/30) |
| test_lattice_varpro.py   | v2 record (greedy VarPro, 10/30) |
| debug_varpro*.py         | debugging session records |

## April 2026 solver-search era (ruled-out approaches; see DIARY)

test_basin_probe, test_trueseed_angles, test_alphax-era grids, CMA-ES,
homotopy/continuation, coordinate descent, flips, multi-triple variants,
random restarts, ML v2/v3 training, manifold analysis, and the staged
recovery batteries. Each file's negative or partial result is recorded in
DIARY.md; the still-active baselines (ml_staged_solver.py,
solve_preconditioned.py, solve9_grid.py, test_alphax_grid.py) remain in
`paper/`.
