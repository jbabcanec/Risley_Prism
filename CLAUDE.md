# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research codebase for the **Risley prism inverse problem**: recovering the full physical specification of a multi-prism beam-steering system (rotation speeds, wedge angles, glass indices, geometry, beam source) from an observed laser scan pattern. The work backs a paper targeting *Inverse Problems* (IOP); the LaTeX source and figures live in `paper/`.

`DIARY.md` is the authoritative research log — it records what has been tried, what works, what failed, and why. **Read it before making claims about the solver's capabilities or starting new solver experiments.** It is far more current than any README. The user's memory index (`memory/MEMORY.md`) tracks paper status and open problems.

## Architecture: know which layer you're in

1. **`risley_lattice/` — THE method (current, formalized 2026-07).** The
   frequency-lattice inversion package: `model.py` (18-vector box, `vec2pat`
   bridge, `battery_cases` — the single source of truth for the standard
   seed-2026 battery), `lattice.py` (matrix pencil, lattice VarPro),
   `spectral.py` (`extract_speeds`, P-agnostic), `angles.py` (phase/amplitude
   → angles), `solve.py` (`solve9`, `solve18` + verified ladder),
   `certify.py` (success bounds + failure certificates). Torch-free. Import
   as `from risley_lattice import ...` with repo root on sys.path.

2. **`experiments/` — every reproducible battery.** speeds_battery,
   solve9_battery, solve18_battery, certification, assumptions (A1–A8),
   noise, prism_count. Run from repo root; each pins BLAS threads in its
   header (multithreaded eig/svd is not bitwise reproducible and boundary
   cases flicker without pinning).

3. **`reverse_problem_v2/core.py` — the canonical forward model.** A prism =
   two interfaces (flat entry face φ=0, tilted exit face), both rotating in
   lockstep at speed `N_i`; refractive index alternates air→glass→air.
   `PrismParameters` + `SystemGeometry` → `fast_forward(params, n_points,
   time_limit)` → `(n_points, 2)` workpiece positions. Everything imports
   `core` from here. Note: `ay_i` enters ONLY as a rotation phase offset and
   `ax_i` only as tilt magnitude — the basis of the spectral angle readout.

4. **`paper/` — the paper + historical baselines.** `main.tex` (being
   rewritten per `paper/REWRITE_PLAN.md`), figures, and the four baseline
   scripts the paper compares against (`ml_staged_solver.py`,
   `solve_preconditioned.py`, `solve9_grid.py`, `test_alphax_grid.py`).
   `paper/archive/` holds all superseded one-off experiments (see its
   README for the old→new map).

5. **`forward_problem/` — LEGACY.** Non-alternating refractive index scheme
   that does **not** match the paper's physics. Gallery generator only.
   `_old/reverse_problem/` is older still — ignore it.

## The 18-D parameter vector

The paper solvers operate on a flat 18-element vector, defined in `paper/ml_staged_solver.py` (`NAMES`, `LO`, `HI`, `RG`):

```
[N1,N2,N3,  ax1,ax2,ax3,  ay1,ay2,ay3,  ng1,ng2,ng3,  d_W, gap,  bm_ax, bm_ay, bm_px, bm_py]
 speeds      wedge angle x  wedge angle y  glass indices  geom    beam angles   beam positions
```

Key helpers (in `ml_staged_solver.py`, re-exported by `solve_preconditioned.py`):
- `vec2pat(v)` — bridge from the 18-vector to `fast_forward` (the forward map F: θ → pattern).
- `canon(v)` — sorts prisms by descending `|speed|`.
- `DiffFwd` — PyTorch differentiable port of the forward model (validated to ~1.8e-6 vs the numpy `core`), used for Adam refinement and autograd Jacobians.

## The solver pipeline

`paper/solve_preconditioned.py` is the current best 18-D solver. Pipeline:
1. **FFT → speeds.** Extract top-8 peaks → all C(8,3) frequency triples (multi-triple search — fixes "FFT picks a harmonic instead of a fundamental", the #1 historical failure mode).
2. **Screen.** For each triple × 8 sign combos: ML init (`AngleNet` then `RemainNet`) and a physics-based paraxial init, each given a quick Adam screen.
3. **Polish.** Top candidates → 3000 Adam steps (float64) → `scipy.optimize.least_squares` with `method='trf'` and a **3-point numerical Jacobian on the exact numpy forward model**. This reaches machine precision (~1e-11 parameter error).
4. **Basin-hop** along weak Jacobian singular directions if not converged.

Hard-won lessons encoded here (full reasoning in `DIARY.md`):
- The optimizer is **not** the bottleneck — initialization is. TRF converges to machine precision *whenever the init is in the correct basin*. The basin is ~1–5° wide in wedge-angle space, so ML init lands inside it only ~30% of the time on random cases. This success rate is the central open problem; many init strategies (CMA-ES, homotopy, random restarts, coordinate descent) have been tried and ruled out.
- scipy TRF on the **exact numpy model** beats Adam, hand-rolled Gauss-Newton, and DiffFwd-based optimization (the PyTorch model has small inaccuracies at extreme parameters).
- **Prism ordering matters in the non-paraxial model**: `vec2pat(v) ≠ vec2pat(canon(v))`. Permuting prisms is *not* a symmetry. When generating a target and solving, both must use consistent ordering.
- Identifiability is **speed-based, not glass-based** (glass has ~10% effect). The condition `|N_i| ≠ |N_j|` must hold. The 18-D Jacobian is full rank but ill-conditioned (κ ≈ 5×10⁵; σ₁/σ₁₈ spans 5 orders of magnitude — speeds dominate, the d_W↔gap geometry tradeoff is weakest).

## Commands

```bash
# --- Current method (run from repo root) ---
python experiments/solve18_battery.py       # FULL 18-D, nothing known: 26/30 PERFECT ~1e-11
python experiments/solve9_battery.py        # 9-D protocol: 24/30 at 1e-12, <1 s/case
python experiments/speeds_battery.py        # signed-speed extraction vs FFT baseline
python experiments/certification.py         # per-parameter bounds + failure certificates
python experiments/assumptions.py           # A1-A8 assumption verification suite
python experiments/noise.py                 # pipeline + certificates under noise
python experiments/prism_count.py           # P=2 / P=4 generality

# --- Historical baselines the paper compares against ---
python paper/solve_preconditioned.py        # April multi-triple + ML + TRF (448-candidate screen)
python paper/solve9_grid.py                 # June alpha_x-grid solver (16/30 reference)
python paper/ml_staged_solver.py            # staged ML; weights paper/*.pt

# --- Reverse solver v2 (NN + differential evolution; the README's --wc flag is stale, use --prisms) ---
cd reverse_problem_v2 && python pipeline.py --prisms 2 3 --samples 100000 --epochs 200

# --- Legacy forward model / gallery (must run from inside forward_problem/) ---
cd forward_problem && python model.py              # reads parameters from forward_problem/inputs.py
cd forward_problem && python generate_examples.py  # writes galleries to output/examples/

# --- Paper ---
cd paper && pdflatex main.tex && pdflatex main.tex
```

## Conventions & gotchas

- **Running scripts.** `paper/` scripts insert `..` and `../reverse_problem_v2` onto `sys.path` at import time, so run them from the repo root with `python paper/<script>.py`. The legacy `forward_problem/` code uses bare imports (`from inputs import *`, `from utils.funs import *`) and **must** be run with `forward_problem/` as the working directory.
- **`forward_problem/inputs.py` is a mutable global config module.** Functions call `import inputs` mid-body to re-read fresh values; `generate_examples.py` mutates it between runs. There is no CLI for the legacy model — edit `inputs.py`.
- **Forward-model parity.** `core.fast_forward` is validated against a serial reference (`forward_problem/validation/`) and `DiffFwd` is validated against `core`. Preserve these when touching the physics.
- The only conventional unit test is `forward_problem/utils/test_funs.py`; there is no project-wide test runner or linter config.
