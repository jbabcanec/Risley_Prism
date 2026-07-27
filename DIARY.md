# Research Diary — Risley Prism Inverse Problem

J. Babcanec (Benedict College) & B. Campbell (Robert Morris University)

---

## Paper Status (2026-04-03)

**Title:** Exact Parameter Recovery for Multi-Prism Risley Systems from Beam Scan Patterns via Global Optimisation

**Target journal:** Inverse Problems (IOP)

**Status:** Submission-ready. 9 pages, 6 figures, 7 tables, 25 references. All analytical claims verified across 5+ full audits. GitHub: https://github.com/jbabcanec/Risley_Prism

**What the paper proves:**
- First solver for the multi-prism parameter-recovery inverse (P=2-4)
- Floating-point precision recovery on 23 test cases including adversarial configs
- Risley manifold theory: C∞ quasi-periodic structure, Stone-Weierstrass density, Sobolev convergence rates
- Speed-based identifiability condition (necessary; sufficient conjectured)
- Noise robustness (30 trials/SNR, reliable to 40 dB)
- Model mismatch tolerance (<3% thickness error)
- OOD diagnostic via residual MSE
- Comparison with FFT inverse (~15 orders of magnitude improvement)

---

## Open Problems

### 1. Global Uniqueness (potential follow-up paper)

The Risley parameter-recovery inverse is unique in the paraxial limit (Fourier coefficient uniqueness — classical theorem). By the inverse function theorem, uniqueness extends locally to the non-paraxial regime. **Global uniqueness is conjectured but unproved.**

**Path to proof:** The Hadamard Global Inverse Function Theorem states: if F: R^n → R^n is C¹, det(DF) ≠ 0 everywhere, and F is proper (||F(x)|| → ∞ as ||x|| → ∞), then F is a global diffeomorphism. Need to verify:
1. Jacobian is nonsingular on all of Θ (we know it's nonsingular at α=0)
2. The map is proper (large angles → large deflections — plausible)

Nobody has done this for Risley systems. Tools exist (Hadamard). Likely provable. Would be a clean self-contained math paper.

### 2. Beam Source Parameter Recovery

Currently the solver assumes known beam entry conditions: position (r_x, r_y) and angles (θ_x⁰, θ_y⁰). These are part of SystemGeometry and NOT recovered.

Pragmatically, you don't want to assume anything about the source. Recovering these adds 4 free parameters (P=3 goes from 9-D to 13-D). The 14-D case (with glass + distances) already works, so this should be feasible. **Not tested yet.**

### 3. Experimental Validation

No physical hardware tested. All results are simulation-on-simulation. Need at least one benchtop demo (2-prism system) to validate the forward model against reality. Acknowledged in paper as future work.

### 4. 100-Trial Noise Study

Running (or completed) — upgrades the noise table from 30 to 100 trials per SNR level. Tightens confidence intervals but doesn't change the story.

---

## Key Corrections Made During Development

- **Identifiability:** Originally claimed glass-type-based. Numerically disproved (glass has ~10% effect). Corrected to speed-based.
- **Manifold nesting:** Originally claimed M_P ⊂ M_{P+1}. Numerically disproved (adding a prism changes the total optical path). Corrected: nesting doesn't hold for physical model; density proved via abstract Q_P spaces.
- **NN contribution:** Originally claimed 1.4x speedup. Real ablation: no speedup. Value is classification (98.8%), not regression.
- **Machine precision language:** Replaced with "floating-point precision" + caveat about model fidelity.

---

## File Inventory

### Analysis Data
- `paper/analysis_results.json` — v1: noise, ablation proxy, Yang comparison, extended battery, P=4, Hessian validation
- `paper/analysis_results_v2.json` — v2: model mismatch, P=4 extended (4 cases), real NN ablation
- `paper/analysis_results_v2_noise.json` — 30-trial noise study cache
- `paper/noise_100trial.json` — 100-trial noise study (when complete)

### Analysis Scripts
- `paper/run_analysis.py` — v1 analysis
- `paper/run_analysis_v2.py` — v2 analysis (model mismatch, P=4, real NN)
- `paper/generate_figures.py` — all figures (including cut ones still in figures/)

### Archive
- `paper/ARCHIVE_NOTES.md` — all supplementary results not in the paper (Hessian validation details, per-prism NN MAE, full mismatch table, glass identifiability disproof, first OOD experiment)

---

## Audit Notes (2026-04-03, audit #6)

### Decisions on 20-issue review:

**#1 Corollary 1 logic:** AGREE. The d>0 claim for P≠P₀ doesn't follow from Theorem 1. It's empirical. Renamed to "Observation 1."

**#2 Error propagation diagonal Hessian:** AGREE. Added explicit diagonal-dominance statement. The sensitivity hierarchy (10⁵ : 1 : 10⁻²) makes off-diagonal coupling negligible, but should say so.

**#3 Theorem 1 proof ambiguity:** AGREE. Clarified: the union ∪Q_P forms the algebra, not individual Q_P.

**#4 Theorem 2 rate suboptimal:** ACKNOWLEDGED. Our bound P^{-(2s-1)} is valid but possibly loose. The best-P-term rate from DeVore theory could give P^{-2s}. Stated as an upper bound, not claimed as tight.

**#5 Sign ambiguity:** AGREE. cos(2πNt+φ) = cos(-2πNt-φ) creates 2^P discrete degeneracies. Noted in uniqueness paragraph.

**#6 Manifold dimension:** Already says "generically." Added brief paraxial justification.

**#7 Uncited refs:** REMOVED IoffeSzegedy2015 and KingmaBa2015.

**#8-9 Date discrepancies:** These are filename vs publication year issues in the resources/ folder — the bibitems match the actual publications. Not changed (filenames are not in the paper).

**#10 Journal format:** Left as revtex4-2 for now. The content matters; class file is changed at submission time.

**#11 Yan 2026:** Verified real — we found it via web search (JOSA A, vol 43, issue 4, 2026).

**#12 Forward model code divergence:** Added legacy README to forward_problem/.

**#13-20 Minor:** Fixed: uncited refs removed, beam params limitation added, floating-point floor noted, P=4 permutation verified, cross-coupling quantified, normal vector notation unified.

### Audit #7 (2026-04-04):

**#1 MSE formula ÷T vs ÷2T:** FIXED. Code uses np.mean over (T,2) array = ÷2T. Changed Eq. 10 denominator to 2T.

**#2 Eq. 7 first→second derivatives:** FIXED. At a minimum, first derivatives are zero. Changed to ∂²MSE/∂N², etc.

**#3 23-case archive:** ACKNOWLEDGED. The cases span multiple analysis files. Not consolidated into a single file yet — a housekeeping task, not a paper error. All cases are reproducible from the scripts.

**#4 Q_P overloaded:** FIXED. Introduced T_P for P-term trigonometric polynomials. Theorem 2 now uses T_P (finite-dim paraxial space), distinct from Q_P (infinite-dim quasi-periodic space with all harmonics).

**#5 |N_i|=|N_j| gap:** FIXED. Identifiability condition changed from N_i≠N_j to |N_i|≠|N_j|. Explained paraxial sign symmetry and that nonlinear Snell breaks it.

**#6 Table I scaling factors:** FIXED. Corrected to ~10⁴ for N, ~60 for α_x, ~60 for α_y.

**#7 Corollary 2 at s=1/2:** FIXED. Changed to s→1/2⁺ (limiting case).

**#8 Snell's law per-axis note:** FIXED. Added note before Eq. 2 that it's the per-axis reduced form.

**#9 Normalization notation:** FIXED. Now writes n̂ = n/||n|| where n = [tanφ, 0, -1]ᵀ.

**#10 Cross-coupling quantified:** FIXED. Stated ~7% at 15°, noted it would appear as model mismatch against real hardware.

**#11 Unused packages:** REMOVED algorithm, algpseudocode.

**#12 Yan 2026:** Previously verified via web search. Real publication.

**#13 Percentages:** FIXED. Added ~ to 3% and 17%.

**#14 TIR clamping:** Not in paper — acceptable modeling choice, minor.

**#15 Computational cost ranges:** Table gives representative values, not exhaustive — acceptable for a summary table.

**#16 "matches" → "is the inverse of":** FIXED. Error hierarchy is inverse of sensitivity, as expected.

---

## Discussion Notes (2026-04-04)

### Q_P vs M_P — why both exist in the paper

**M_P** = patterns from real P-prism hardware (specific geometry, bounded angles).
**Q_P** = abstract P-frequency quasi-periodic functions (no geometry, unbounded).

M_P manifolds do NOT nest (adding a prism changes the hardware). Q_P spaces DO nest (adding a frequency to a sum of sinusoids doesn't change existing terms). Stone-Weierstrass density is proved for Q_P, not M_P. The connection holds in the paraxial limit where M_P ≈ Q_P (up to amplitude scaling).

**Decision:** The density theorem and Fourier convergence support the OOD diagnostic and wedge-count determination story. They do NOT support the main recovery result (which just needs: correct P → search M_P → find the unique zero). The paper should frame the theory as supporting the secondary questions, not as the foundation for recovery.

### Paraxial limit — what "small angles" means

Paraxial = sin(α) ≈ α. Works well up to ~15° (1% error). The paper tests up to 15°. The theorem is proved for α → 0. The gap: no proof covers 15° rigorously. Empirically it works. The convergence rate P^{-1.64} is observed, not proved for M_P at large angles.

### Missing: α constraint as function of geometry

The achievable pattern amplitude is bounded by:
  pattern_amplitude ≈ (n_g - 1) · α_max · d_W

With α_max = 18° and d_W = 100: max amplitude ≈ 15.7 units. The paper never states this. A target pattern larger than this range cannot be fit regardless of P. This constraint should be noted somewhere — either in the solver description or in limitations.

Also: the Fourier convergence table (P=1-6) works because the target was generated from a 3-prism system and is already within the achievable amplitude range. The convergence rate would look different for a target outside this range.

### 100-trial noise study: failed (solver too weak)

Ran with 1 restart, 100 maxiter, popsize 15. Got ~30% success at ALL SNR levels — the solver wasn't finding the basin, not a noise issue. The 30-trial study in the paper used 2 restarts, 120 maxiter, popsize 18 and got 100% at 60 dB. The 30-trial data is correct; the 100-trial run is garbage. If we want 100 trials, need to match the 30-trial solver settings (~10 hour run).

---

## CRITICAL REALIZATION (2026-04-05): The solver is brute force and that's embarrassing

### The problem

The current solver is differential evolution (scipy) — a population-based random search. It evaluates the forward model ~100k-2M times to find the answer. For 19-D recovery, it takes HOURS and often fails. This is not a contribution. Anyone can call scipy.optimize.

### The insight we missed

In the paraxial limit, the scan pattern is a sum of sinusoids:

  p_x(t) = Σ_i A_i cos(2π N_i t + φ_i)

The FFT gives ALL prism parameters directly:
- **Frequency peaks → N_i** (rotation speeds) — we already extract this
- **Amplitude at each peak → α_x,i** via A_i = d_eff · (n_g - 1) · α_x,i — WE DON'T USE THIS
- **Phase at each peak → α_y,i** — WE DON'T USE THIS

We've been extracting the frequencies and THROWING AWAY the amplitudes and phases, then spending hours of brute-force DE to rediscover what the FFT already contained.

### The intelligent approach

**Stage 1: Spectral decomposition (milliseconds)**
1. FFT or Prony/ESPRIT for super-resolution frequency estimation → N_i
2. Complex Fourier coefficients at each N_i: amplitude → α_x,i, phase → α_y,i
3. Centroid → beam angles
4. Pattern scale → d_W (if unknown)

This gives an analytical paraxial solution for all parameters. No optimization.

**Stage 2: Non-paraxial refinement (seconds)**
NM polish from the paraxial solution. It's already in the right basin — just correcting for Snell's-law nonlinearity. Converges in seconds.

**Total: seconds, not hours. For ALL parameters including beam and geometry.**

### What this changes for the paper

The paper's solver section would change from "we run DE for hours" to "we extract the analytical paraxial solution from the Fourier spectrum and refine with local optimization." This is an actual algorithmic contribution — not an application of existing tools.

The manifold theory still applies (it explains WHY the spectral decomposition works). The noise/mismatch analysis still applies. The OOD diagnostic still applies. But the solver becomes intelligent.

### For geometry parameters (d_W, gap, beam params)

These affect the amplitude mapping A_i → α_x,i. If d_W is unknown:
- A_i = d_eff(d_W, gap) · (n_g - 1) · α_x,i
- The ratio A_i/A_j eliminates d_eff if all prisms have the same n_g
- If glass indices differ, the ratios constrain d_W, gap, and n_g simultaneously
- A small NM search over [d_W, gap, n_g] with α extracted from FFT is ~5-D, not 19-D

### TODO
- [x] Implement spectral parameter extraction (amplitude + phase from FFT)
- [x] Test on the 9-D case (should give near-exact paraxial solution)
- [ ] Test paraxial solution + NM polish on 19-D
- [ ] Compare speed and accuracy vs brute-force DE
- [ ] If it works, rewrite the solver section of the paper

---

## ML Inverse Solver Experiments (2026-04-05)

### Goal
Replace the brute-force DE solver with pure machine learning prediction. No scipy.optimize, no grid search. A trained network that takes a scan pattern and outputs all 18 system parameters.

### What was tried

**V1: End-to-end regression, dual-branch (CNN + spectral MLP)**
- 200k training samples, 325k model params, 80 epochs (~30 min on CPU)
- Input: raw pattern (2×200) + FFT features (404)
- Output: all 18 params via sigmoid → [0,1] → denormalize
- Best val loss: 0.041025
- Result: speeds ~5% error, beam angles ~3%, but glass/geometry/α_y ~20%+
- Inference: 3 ms/case (600,000× faster than DE)

**V2: Bigger model + explicit peak features**
- Same 200k data, 1.16M params, added 19-dim peak features (freq, amp, phase per peak)
- Best val loss: 0.041858 — *worse* than V1 due to severe overfitting (train/val = 0.029/0.044)
- Larger model memorized training data, didn't generalize

**V3: Fixed peak features + V1-sized model**
- Sorted peaks by frequency (matching canonical |speed| ordering)
- Sin/cos phase encoding (eliminated ±π wrapping discontinuity)
- 485k params, 25-dim peak features
- Best val loss: 0.041060 — essentially identical to V1
- Same accuracy profile: speeds OK, glass/geometry/α_y poor

### Why end-to-end regression fails for 18-D

All three variants plateau at val_loss ≈ 0.041 regardless of features, model size, or architecture. The bottleneck is not engineering — it is the problem structure:

1. **Weak observability**: Glass indices have ~10% effect on patterns. The signal is buried under speed/angle variation. The network averages over degenerate parameter combos.
2. **Phase ambiguity**: α_y enters as cos(γ + α_y). Multiple α_y values produce similar patterns under permutation/sign changes. The network cannot resolve these.
3. **Data sparsity**: 200k samples in 18-D gives ~2.5 samples per dimension edge. The inverse mapping is highly nonlinear (iterated Snell's law), so interpolation between sparse samples is inaccurate.
4. **Coupled loss**: MSE on all 18 normalized params treats speed errors (easy) equally with glass errors (hard). The optimizer spends capacity on the easy params and neglects the hard ones.

### Key finding: the DiffForward model works

Ported the full numpy forward model to PyTorch (batched, differentiable). Verified: max |PyTorch − NumPy| = 1.82×10⁻⁶. This enables gradient-based refinement through the physics — but only if the initial prediction is in the correct basin. The end-to-end network's predictions are too far off for gradient polish to converge to the true solution.

### The fix: staged ML prediction

The 18-D problem decomposes naturally by observability:

| Stage | What | From what | Difficulty | Why |
|-------|------|-----------|------------|-----|
| 1 | 3 speeds | FFT peak frequencies | Trivial | Peaks are exact; only sign is ambiguous |
| 2 | 6 angles (α_x, α_y) | Peak amplitudes/phases + speeds | Easy | Paraxial formula gives ~90% of the answer |
| 3 | 9 remaining (glass, geo, beam) | Full pattern + speeds + angles | Hard | Subtle effects, but only 9-D search |

Each stage is a smaller, better-conditioned ML problem. Stage 1 uses the FFT analytically (not optimization — deterministic signal processing). Stages 2–3 are neural networks conditioned on previous outputs.

Expected improvement: speeds → <0.5% error, angles → <2%, glass/geometry → 5–15%. With gradient polish from such close initial estimates, everything should converge to machine precision.

### TODO
- [x] Implement staged ML solver (`ml_staged_solver.py`)
- [x] Benchmark against end-to-end V1 on same test cases
- [ ] If staged solver works well, integrate with paper narrative

---

## Staged ML Solver V1 Results (2026-04-05)

### Architecture
- **Stage 1**: FFT peak extraction → speed magnitudes (analytical, <1 ms). Sign resolution via trying 8 combos with angle network + full forward model.
- **Stage 2**: AngleNet MLP (28→256→256→128→6), conditioned on true speeds + 25 peak features. 3 min training.
- **Stage 3**: RemainNet (CNN + conditioning MLP → 9 outputs), conditioned on true speeds + true angles + peaks. 25 min training.

### Results (200k training, paper test case)
- **Speeds: 0.0% error** — FFT extraction is exact when frequencies land on FFT bins
- **Pattern MSE: 3.67** vs end-to-end's 211 → **57× improvement**
- α_x: 0.2–6.5% (mixed), α_y: 7.5–30% (still poor)
- Glass/geometry/beam: 6–35% (still poor for weakly observable params)

### Results (200k, 100 random cases)
- Speed median: 0.06–0.52 Hz (good), but sign resolution fails on ~10% of cases → mean 20–29%
- α_y: 20–24% mean (no better than end-to-end)
- Glass: 27–37% (worse than end-to-end — conditioning on approximate speeds/angles introduces cascading error)

### Diagnosis
Two bottlenecks remain:

1. **Data**: 200k samples is sparse for 9-D Stage 3. The forward model generates data at 950 samples/s — we can trivially produce 1M+. More data reduces overfitting and improves coverage of the parameter space.

2. **Loss function**: MSE on normalized parameters treats all parameters equally. But glass indices have ~10% effect on patterns while speeds have ~1000% effect. The network has no incentive to learn subtle parameters.

Fix: **physics-informed loss**. During Stage 3 training, assemble predicted remaining params with known speeds/angles, run through DiffForward, compare predicted pattern with input pattern. This automatically weights each parameter by its physical observability. Parameters that strongly affect the pattern get more gradient signal.

### Plan: V2 with 1M data + physics loss
- Generate 1M training samples (~17 min)
- Stage 2: train with 1M data (angle network, ~10 min)
- Stage 3: train with param_loss + physics_loss via DiffForward (~2 hours)
  - Physics loss computed every 4th batch (keeps cost manageable on CPU)
  - DiffForward runs in float32 during training for speed
- Expected: glass/geometry accuracy should improve significantly (physics loss tells the network "this glass index doesn't reproduce the pattern")

---

## Solution Manifold Analysis (2026-04-06)

### The experimental finding

After extensive testing of the 18-D inverse solver (ML init + gradient descent through DiffForward, multi-start across glass/geometry, Adam + LBFGS, per-parameter LR, reparameterization), the results are consistent:

- **Speeds (N_i):** recovered to < 10⁻⁴ in all cases
- **Phase angles (α_y,i):** recovered to < 0.15° in all cases
- **Beam angles (θ_x, θ_y):** recovered to < 0.1° in all cases
- **Wedge angles (α_x,i):** recovered to ~1-3% — coupled to glass/geometry
- **Glass indices, d_W, gap, beam positions:** 10-50% error DESPITE pattern MSE < 2×10⁻³

The solver finds a set of parameters that reproduces the target pattern almost exactly, but the parameter values disagree with the true ones. Different initial conditions converge to DIFFERENT parameter vectors with similarly low pattern MSE.

### The mathematical question

Let F: Θ ⊂ R^18 → R^(200×2) be the forward map (parameters → scan pattern).

For a given target pattern p*, define the solution set (fiber):
  S(p*) = { θ ∈ Θ : ||F(θ) - p*||² < ε }

**Question 1:** What is dim(S)? Is S a discrete set (dim=0), a curve (dim=1), a surface (dim=2), ...?

**Question 2:** Which parameters are constant across S (identifiable) and which vary (degenerate)?

**Question 3:** Does S have a nice structure (smooth manifold, connected)?

### The approach: Jacobian null-space analysis

At any solution θ*, the tangent space to S is the null space of the Jacobian:
  DF(θ*) ∈ R^(400 × 18)

rank(DF) = number of identifiable directions
null_dim = 18 - rank(DF) = dimension of the degeneracy manifold

Compute DF numerically using torch.autograd.functional.jacobian through DiffForward. Then SVD:
  DF = U Σ V^T

The singular values σ_i reveal:
- Large σ_i: well-conditioned (identifiable) parameters
- Small σ_i: ill-conditioned (nearly degenerate) parameters
- σ_i ≈ 0: truly degenerate directions

The right singular vectors (columns of V) corresponding to small σ_i span the degeneracy manifold.

### What this gives for the paper

1. **Quantitative identifiability map**: which parameter combinations are observable from the pattern, with numerical sensitivity
2. **Degeneracy characterization**: the solution set is a d-dimensional manifold, parameterized by d degenerate directions (likely: glass/geometry trade-offs)
3. **Manifold tracing**: starting from the true solution, follow the null space direction to generate a family of solutions that all match the pattern
4. **Solver validation**: the ML+gradient solver correctly finds A point on the solution manifold, and the residual MSE is bounded by the manifold's curvature

This transforms the "glass can't be recovered" negative result into a positive structural theorem about the inverse problem.

### KEY RESULT (2026-04-06): Full rank Jacobian — no true degeneracy

**The 18-D Jacobian has FULL RANK 18.** The solution IS locally unique. There is no null space.

The problem is the **condition number κ = 4.9 × 10⁵**. The sensitivity hierarchy spans 5 orders of magnitude:

| Singular value | Direction | Sensitivity |
|----------------|-----------|-------------|
| σ₁ = 6229 | N₂ (speed) | Huge — 0.001 Hz speed change creates visible pattern change |
| σ₄ = 351 | n_g₂ (glass) | Moderate — glass affects Snell deflection |
| σ₇ = 36 | beam_ax | Moderate |
| σ₁₂ = 0.80 | α_y₁ (phase) | Weak — phase enters as cos(γ+α_y) |
| σ₁₆ = 0.12 | α_x₁ + gap | Very weak |
| σ₁₈ = 0.013 | d_W − gap tradeoff | 50,000× weaker than speeds |

**Manifold trace confirms:** stepping along the weakest direction (d_W ↔ gap), pattern MSE grows from 0 to only 1.9×10⁻⁴ at ±5 units. The gradient IS there — it's just 50,000× smaller than the speed gradient.

**Subproblem conditioning:**
- 9-D (speeds+angles): κ = 7.8×10³ — well-conditioned
- 13-D (+beam): κ = 1.5×10⁴ — well-conditioned
- 18-D (everything): κ = 4.9×10⁵ — ill-conditioned but FULL RANK

**Implication:** The 18-D problem is solvable to machine precision with sufficient optimization precision. The earlier gradient solver failures were due to Adam's inability to resolve σ₁₈ = 0.013 against σ₁ = 6229 — not fundamental degeneracy. A preconditioned optimizer that accounts for the condition number should converge.

### BREAKTHROUGH: Full 18-D Recovery to Machine Precision (2026-04-06)

**Result: ALL 18 parameters recovered to ~10⁻¹¹ absolute error.**

Paper test case: max abs error = 1.75×10⁻¹¹ (speeds, angles, glass, geometry, beam — ALL perfect).
Random battery: 5/5 PERFECT (max errors: 6.15e-11, 1.85e-9, 6.26e-13, 2.16e-12, 3.34e-11).
Pattern MSE: 10⁻²³ to 10⁻²⁶. Total time: ~110s per case on CPU.

**Critical bug found: prism ordering matters in non-paraxial model.**
- `vec2pat(tv) ≠ vec2pat(canon(tv))` — MSE = 2.42 between them
- The canonicalization reorders prisms by |speed|, but in the full vector Snell's law model, the physical ordering (which prism the beam hits first) affects the refraction cascade. Permuting prisms is NOT a symmetry of the non-paraxial model.
- Previous solver stuck at MSE ≈ 10⁻³ because it was searching in canonical ordering while the target was generated from a different ordering. The solver found the best fit in the wrong permutation — a genuine local minimum, not a numerical failure.
- Fix: target and solver must use the same ordering. For the inverse problem, the prism ordering IS a physical observable (determined by the hardware).

**Pipeline (solve_preconditioned.py):**
1. FFT → speed magnitudes (exact to Fourier resolution)
2. 8 sign combos × ML init × 150 Adam steps → pick basin (~20s)
3. 5000 coarse Adam steps (float64) → MSE ~10⁻³ (~85s)
4. `scipy.optimize.least_squares` with `method='trf'`, numerical 3-point Jacobian on the exact NumPy forward model → MSE 10⁻²³ (~2s, ~60 iterations)

**Key insight:** scipy's trust-region reflective method with numerical Jacobians on the EXACT forward model is far superior to:
- Adam (first-order, can't resolve σ_min/σ_max = 50,000× sensitivity ratio)
- Hand-rolled Gauss-Newton/LM (line search was too conservative, stalled at MSE 3×10⁻³)
- DiffFwd-based optimization (the PyTorch model has subtle inaccuracies at extreme parameter values)
- SVD-preconditioned Adam (correct theory but still first-order, converged slowly)

**Why it works:** scipy's TRF implementation uses a trust-region scheme that automatically adapts the step size and damping. The numerical Jacobian (3-point central differences) costs 37 vec2pat evaluations per iteration but gives accurate derivatives of the TRUE forward model. No modeling approximation errors. ~60 TRF iterations × 37 evaluations = ~2200 vec2pat calls, each taking ~1ms = 2s total.

**For the paper:** This result proves the 18-D Risley inverse problem is solvable to machine precision with the pipeline: spectral analysis → ML initialization → trust-region optimization. The ML component provides the basin (correct sign combo + approximate parameters); scipy provides the precision.

### Paper Rewrite & 50-Case Battery Failure (2026-04-07)

**Paper rewrite completed:** Title, abstract, contributions, Sec 4 (solver), Sec 5 (results), Sec 6 (discussion), conclusion all rewritten. New mathematical content: Jacobian SVD identifiability proof, error propagation, Gauss-Newton connection. New figures: SVD spectrum, pipeline convergence, OOD (redone with new solver). Compiles at 10 pages, revtex4-2.

**New theoretical results:**
1. Local identifiability PROVEN: rank(J) = 18 at all test points → inverse function theorem → locally unique solution. This is a theorem, not empirical.
2. Sensitivity hierarchy is structural: σ₁ ≈ 6700 (speeds), σ₁₈ ≈ 0.013 (geometry). 5 orders of magnitude from physics, not numerics.
3. Prism ordering symmetry breaks beyond paraxial: vec2pat(v) ≠ vec2pat(perm(v)), MSE ≈ 2.4 between orderings. Paraxial limit has P! degeneracy; non-paraxial model does not.
4. κ(H) ≈ κ(J)² ≈ 10¹⁰ explains exactly why Adam fails — treats all directions equally when they differ by 10¹⁰ in curvature.
5. Global uniqueness remains OPEN — local is proved, global needs Hadamard's theorem.

**50-case random battery EXPOSED robustness failure:**
- Seed=42 (original 5-case test): 5/5 PERFECT
- Seed=2026 (50-case battery): only 3/11 PERFECT after ~11 cases buffered
- Failure mode: ML init lands in wrong basin, sign selection picks wrong combo, Adam converges to spurious local minimum. Trust-region then polishes the wrong solution.
- The pipeline works when ML init is good enough (paper test case, easy random cases). Fails when ML init misses the basin.

**Root cause:** The ML models (AngleNet, RemainNet) were trained on 1M samples but the prediction quality varies across the parameter space. For some parameter combinations, the ML prediction is far enough from the truth that the 150-step sign selection can't distinguish the correct sign combo, and 5000 Adam steps aren't enough to escape the wrong basin.

**Planned fixes:**
1. Top-K sign combos: instead of picking the single best sign combo, keep top-3 and run full Adam + TRF on each, pick lowest final MSE. 3× cost but much better basin coverage.
2. Perturbed ML inits: for each sign combo, generate multiple perturbations (jitter angles, glass, geometry) and pick the one with best post-Adam MSE. Explores the local landscape around the ML prediction.
3. Both combined: top-3 signs × M perturbations each = 3M candidates. Run quick Adam on each, pick best, then full TRF.

### Root Cause Identified: FFT Harmonic Confusion (2026-04-07)

**The optimizer was never the problem. The FFT was.**

Systematic analysis of 30 random cases showed:
- 17/30 FFT correctly extracted speed magnitudes → solver always worked
- 8/30 FFT picked harmonics/cross-terms instead of fundamentals → solver always failed
- 5/30 speeds too close (< 0.15 Hz separation) → fundamentally degenerate

**Why the FFT fails:** The forward model generates frequencies at k₁N₁ + k₂N₂ + k₃N₃. The naive peak-picker (top-3 by power) grabs the 3 tallest peaks, which can be harmonics (2N₁) or cross-terms (N₁+N₂) instead of fundamentals. Example: Case 4 true speeds [1.23, 0.54, 0.21], FFT returned [2.5, 1.2, 0.5] — the 2.5 is ≈ 2×1.23 (harmonic), and 0.21 Hz (only 2.1 cycles in T=10s) was missed.

**Attempted fixes that DIDN'T work:**
- Top-3 signs + perturbations: same wrong FFT → same wrong basin
- Random inits with known (wrong) speeds: DE with 45k evals still converged to wrong basin
- Hierarchical 9-D subproblem: all candidates found same wrong minimum
- Basin-hopping along Jacobian weak directions: no connected better basin
- Harmonic filtering: fixed some cases but broke others (can't distinguish harmonic from coincidental 2:1 ratio)

**Fix that WORKS: multi-triple search.**
- Extract top-8 FFT peaks (instead of 3)
- Generate all C(8,3) = 56 frequency triples
- For each triple × 8 signs = 448 candidates: ML init + 150 quick Adam steps → score
- Top-5 scores → 3000 Adam + scipy TRF
- The correct triple is guaranteed to be among the 56 (if the true fundamentals appear in the top-8 peaks)

**Result: Case 4 SOLVED.** MSE = 4.83e-25, max_err = 2.51e-11. The winning candidate used the correct triple [1.2, 0.5, 0.2] while the other 4 finalists were stuck at MSE ≈ 0.84 with the wrong triple.

**Cost:** 448 candidates × 150 steps = ~35 min per case (CPU). Expensive but correct. Can be reduced by:
- Fewer triples: C(6,3) = 20 instead of C(8,3) = 56
- Shorter screening: 50 steps instead of 150
- Parallel processing across sign combos

### Progress Summary (2026-04-07)

**What we've built (4-day arc):**

Day 1 (Apr 4-5): Pure ML approach. End-to-end neural net → val_loss 0.041, nowhere near 10⁻³. Staged ML (FFT + AngleNet + RemainNet) better but angles/glass still off. Added differentiable forward model (DiffFwd) for gradient refinement through PyTorch.

Day 2 (Apr 6): Jacobian SVD breakthrough — full rank 18, κ = 4.9×10⁵, proving local identifiability. Discovered prism ordering matters in non-paraxial model (vec2pat(v) ≠ vec2pat(perm(v))). With consistent ordering + scipy TRF: paper test case → all 18 params to 10⁻¹¹. Original 5-case battery: 5/5 PERFECT.

Day 3 (Apr 6-7): Paper rewrite (title, abstract, Sec 4-6 completely new). New figures (SVD spectrum, pipeline convergence, OOD). 50-case battery exposed 68% failure rate → systematic diagnosis.

Day 4 (Apr 7): Root cause: FFT harmonic confusion, not optimizer failure. Tried and ruled out: perturbations, random multi-start, hierarchical 9-D subproblem, basin-hopping, DE. All failed because they used wrong speeds from bad FFT. Solution: multi-triple search — enumerate C(8,3)=56 frequency triples from top-8 FFT peaks, score all 448 (triples × signs) candidates. Case 4 (previously uncrackable): PERFECT at 10⁻²⁵. Cases 6, 7, 9 running.

**Current pipeline:**
1. FFT → top-8 peaks → C(8,3)=56 frequency triples
2. 56 triples × 8 signs × ML init × 150 Adam → 448 candidates scored
3. Top-5 → 3000 Adam (float64) + scipy TRF → machine precision
4. Basin-hopping (if needed) along Jacobian weak directions

**What works:** Any case where the true fundamentals appear in the top-8 FFT peaks (vast majority). Machine-precision recovery guaranteed.

**What doesn't work:** Cases with speeds < ~0.15 Hz (too few cycles) or speed separation < FFT resolution (0.1 Hz). These are physics limits, not algorithmic failures.

**Remaining work:**
- Confirm Cases 6, 7, 9 with multi-triple solver (running now)
- Speed optimization: reduce 35 min → target ~5 min (fewer triples, shorter screening, parallelism)
- Full 50-case battery with robust solver
- Update paper tables/numbers with final results
- Generate battery statistics figure

### Spectral Inversion & Initialization Experiments (2026-04-08)

**Goal:** Improve the ~30% success rate (9/30 at 9-D) by finding better initializations for TRF. The bottleneck is NOT the optimizer — TRF converges to machine precision whenever the init is within the basin. The bottleneck is that ML init misses the basin ~70% of the time.

**Key finding from subproblem tests:** Success rate is ~30% at ALL dimensions tested (9-D, 12-D, 14-D, 18-D with test_dimensions.py). This proves the bottleneck is in the first 9 parameters (speeds + angles), not the geometry/glass/beam parameters.

#### Approach 1: Spectral Phase Extraction (FFT bins)

**Idea:** Extract αᵧ directly from the FFT phase at each speed's frequency bin. In the paraxial limit, the pattern at frequency |Nᵢ| has phase = αᵧ,ᵢ.

**Result: COMPLETE FAILURE.** The FFT grid has resolution 0.1 Hz (T_OBS=10, T_PTS=200). True speeds are off-grid (e.g., |N|=2.247 maps to bin 2.2 Hz). The frequency mismatch creates a phase error of ~π×ΔF×T ≈ 85° — completely corrupting the phase information.

Even with parabolic peak interpolation to refine the frequency estimate, the phase errors remained catastrophic: **median αᵧ error = 62.6°, only 7/90 within 5°.**

Amplitude → |αₓ| worked slightly better: **median 1.46° error, 38/90 within 1°.**

#### Approach 2: Harmonic Least-Squares Decomposition

**Idea:** Instead of reading single FFT bins (corrupted by discretization), fit the pattern as a linear combination of sinusoids at the exact estimated frequencies:
```
p_x(t) = Σᵢ [aᵢ cos(2πfᵢt) + bᵢ sin(2πfᵢt)] + offset
```
This is a standard linear LS problem — no FFT discretization, handles cross-contamination by fitting all prisms jointly.

**Results (with TRUE frequencies + TRUE geometry):**
- **αᵧ: BIMODAL distribution.** Exactly 50% have error <5° (excellent), exactly 50% have error ~180° (off by half turn). NOT a failure — it's a **systematic (αₓ, αᵧ) ↔ (-αₓ, αᵧ+180°) degeneracy** in the paraxial limit.
- **|αₓ|: median 4.55° error** — still poor. Non-paraxial multi-prism interactions amplify the apparent amplitude by ~1.4-1.7×.

**Key insight:** The 180° ambiguity is a PHYSICAL degeneracy, not a numerical artifact. A prism with (αₓ, αᵧ) produces exactly the same pattern as (-αₓ, αᵧ+180°) in the paraxial limit. The degeneracy is broken only by higher-order (non-paraxial) effects, which are small.

**9-D TRF recovery with harmonic init:** 1/30 PERFECT (far worse than ML's 9/30). The αₓ estimates are too inaccurate.

#### Approach 3: 180° Flip Augmentation (ML αₓ + Spectral αᵧ)

**Idea:** Combine the strengths of ML (good αₓ) and spectral decomposition (good αᵧ up to 180°). For each ML init:
1. Extract αᵧ from harmonic LS at each speed's frequency
2. Generate 8 "flipped" variants: for each prism subset, try (αₓ→-αₓ, αᵧ→αᵧ+180°)
3. Screen with single forward eval, TRF from best
4. Keep the better of {ML-only TRF, best-flip TRF}

**Results:** ML only = 9/30, ML+flips = **10/30** (+1 case). The flip saved case 22 but otherwise had no effect. The ML doesn't suffer from the spectral 180° degeneracy in the same way — it's wrong for other reasons.

#### Approach 4: Target-Deformation Homotopy Continuation

**Idea:** Instead of trying to start in the correct basin, smoothly deform from a trivial problem to the real one:
```
target_λ = (1-λ) × F(θ₀) + λ × target_real
```
At λ=0, θ₀ is the exact solution (trivially). Track the solution as λ → 1. Each step is a small perturbation → TRF converges.

**Results (n_steps=20):** ML only = 9/30, Continuation = **10/30** (+1), Union = **11/30** (+2). Saved cases 12 and 21. More promising than flips — saved different cases.

The continuation method is the only approach that can escape a wrong basin by continuously deforming the objective. Higher step counts (50, 100) may improve further but are very slow (~150s/case at 20 steps).

#### Approach 5: CMA-ES (Global Optimizer)

**Idea:** CMA-ES adapts a full covariance matrix and is considered the gold standard for non-convex optimization in moderate dimensions. Seeded at ML init with σ₀=3.0 (angles) and 5000 function evaluations.

**Result: WORSE than ML+TRF.** CMA-ES MISSED cases that plain TRF from ML init solved. The basin is so narrow (~10⁻⁴ of the parameter space) that the CMA-ES population scatters across multiple wrong basins and follows the majority to a wrong minimum. Population-based methods are fundamentally ill-suited for problems with extremely narrow basins.

#### Approach 6: Random Restarts (Ceiling Baseline)

Tested 50 random restarts (near-correct speeds, random angles) on failure cases. **Zero improvement.** The basin is ~1° wide in angle space, but the parameter range is 36° per angle. The probability of a random point landing in the basin is ~(1/36)⁶ ≈ 5×10⁻¹⁰.

#### Approach 7: Model-Based Homotopy (Paraxial → Full)

**Idea:** Deform the forward model from paraxial (where we have an exact analytical solution) to full non-paraxial:
```
R_λ(θ) = (1-λ) × P(θ) + λ × F(θ) - target = 0
```
At λ=0: solve paraxial model exactly via harmonic LS (no init error).
At λ=1: solve full model (the real problem).

**Status:** Implemented but not yet tested (session ended).

**Why this is the most promising remaining approach:** It starts from an EXACT solution (zero init error) and tracks the solution through a smooth model deformation. The paraxial and non-paraxial models agree for small angles and gradually diverge. The continuation should follow the correct branch as long as no bifurcation occurs.

### Summary of What We Know (2026-04-08)

**The fundamental bottleneck is basin width:**
- The 9-D basin of attraction is ~1° wide in each angle dimension
- The total angle parameter space is 36° per dimension
- The basin occupies ~10⁻⁹ of the parameter volume
- ML init gets within ~5° on average → inside basin only ~30% of the time
- No initialization method tested (spectral, harmonic LS, ML, combined) is consistently accurate enough

**What works for the 30% that succeed:**
- FFT → correct speed magnitudes
- ML → angles within ~1-2° (lucky)
- TRF → machine precision in ~60 iterations

**What fails for the 70%:**
- FFT picks harmonics/cross-terms (some cases)
- ML predicts angles >5° off (most cases)
- Once in wrong basin, no local method can escape
- Population-based global methods (CMA-ES) can't find the narrow basin either

**Most promising approaches for next session:**
1. Model homotopy (paraxial → full): starts from exact solution, no init error
2. Continuation with more steps: showed 2 extra cases at 20 steps, more steps may help
3. Combined union of all complementary methods: ML + flip + continuation ≈ 12-14/30
4. Better ML training: larger/better networks, data augmentation, ensemble

**What the paper should say:** The 18-D solver achieves machine precision when initialized within the basin of attraction. The ML initialization succeeds in ~30% of random cases. The remaining 70% represent a fundamental challenge of narrow basins in high-dimensional parameter spaces — not a solver limitation but an initialization problem. The homotopy continuation approach shows promise for expanding the success rate.

### Failure Diagnosis & Multi-Triple Fix (2026-04-08 continued)

**Diagnostic of 30-case 9-D battery (test_diagnose_failures.py):**

Root cause breakdown of 21 failures:
1. **Wrong FFT peaks (spd_match=False): 12 cases (57% of failures)** — FFT picks harmonics/cross-terms instead of fundamental frequencies. This is the dominant failure mode.
2. **Very close speeds (sep < 0.1 Hz): 3 cases** — physics limit, FFT can't resolve. Need longer T_obs.
3. **Correct speeds but bad ML init: 6 cases** — ML angle prediction off by >10°, outside basin.

Solved cases (9/30):
- ALL had spd_match=True (correct FFT peaks)
- Median ML αₓ error = 2.8° (vs 10.3° for failures)
- Median speed separation = 0.515 Hz (vs 0.330 Hz for failures)

**Multi-triple search validation (test_multitriple_fast.py, 30 Adam steps screening):**

Through first 8 cases:
- Single-triple: 3/8 PERFECT
- Multi-triple: 4/8 PERFECT
- Union: **5/8 PERFECT (62.5%)**
- Cases 4, 8: MULTI SAVED (both had spd_match=False — wrong FFT peaks fixed by multi-triple)
- Case 2: MULTI HURT (screening picked wrong candidate — 30-step Adam insufficient)
- Cases 6, 7: both failed despite multi-triple (close speeds + bad ML)

**Key insight: the multi-triple search addresses the #1 failure mode** (wrong FFT speed extraction) but doesn't help with close speeds or bad ML init. The union of single + multi approaches is significantly better than either alone.

**Estimated full-battery performance with multi-triple:**
- 9-13 solved by single-triple (same as before)
- 4-6 additional solved by multi-triple (wrong FFT cases)
- **Total: ~13-15/30 (43-50%)**, up from 30% with single-triple only

**Remaining barriers:**
- Close speeds: need T_obs > 10s (more observation cycles)
- Bad ML init: need better ML training or alternative init strategies
- Both are well-characterized failure modes with clear remedies

### Dead Ends & The Real Bottleneck (2026-04-08/09)

**Tested and ruled out (none improved beyond +1-2 cases):**
- 180° flip augmentation (+1/30)
- Homotopy continuation, target-deformation (+2/30)
- Model homotopy, paraxial→full (failed — models too different)
- CMA-ES (worse — population scatters across narrow basin)
- Coordinate descent, 2-D per prism (+0 — coupled parameters)
- Multi-scale / Gaussian smoothing (+0 — removes signal)
- 6-D differential evolution (case 1 only — basin too narrow in 6-D)
- 3-D DE with fixed harmonic αᵧ (0/3 — coupling kills it)
- Random restarts, 50 starts (0 — basin is 10⁻⁹ of volume)

**Root cause analysis (definitive):**
The 9-D basin of attraction is ~5° wide in αₓ directions (confirmed by diagnostic: all 9 solved cases had ML αₓ error < 7.4°, all 21 failures had error > 4.6° or wrong speeds). αᵧ error doesn't matter much (solved cases had up to 33° αᵧ error) — the basin is wide in the phase direction, narrow in the amplitude direction.

Failure taxonomy (21 failures out of 30):
1. Wrong FFT peaks (spd_match=False): **12 cases** — dominant mode, fixable by multi-triple
2. Close speeds (sep < 0.1 Hz): **3 cases** — physics limit
3. Correct speeds, ML αₓ error > 5°: **6 cases** — ML quality limit

**The ML was barely trained:**
- AngleNet: 108K params, 28→256→256→128→6
- Training: 1M samples, 80 epochs, ~20 minutes on CPU
- Loss: parameter MSE (L2 on normalized angles)
- Val loss was likely still decreasing — we left massive gains on the table

**Projected success rates vs ML improvement factor k:**

| k | Training budget | 9-D | 12-D | 18-D |
|---|---|---|---|---|
| 1 (current) | 20 min | 30% | 25% | 15% |
| 1.5 | ~1 hr | 50% | 40% | 25% |
| 2 | ~3-5 hr | 63% | 50% | 35% |
| 3 | ~10-20 hr | 70% | 60% | 50% |
| 5 | ~2-3 days | 77% | 70% | 65% |
| 10 | ~1-2 weeks | 87% | 80% | 80% |

k is defined as improvement factor on ML αₓ prediction error. Estimates from extrapolating the actual error distribution of our 30 test cases.

**18-D hard wall:** σ₁₈ = 0.013 (d_W↔gap tradeoff) creates a basin 50,000× narrower than the speed direction. Even at k=10, this remains challenging. Needs either SVD-preconditioned loss or dedicated geometry-prediction head.

### ML Retraining Strategy (2026-04-09)

**Quick test (Model v2, ~20 min):** Retrain AngleNet with physics-first loss — make ||F(θ_pred) - target||² the PRIMARY loss instead of parameter MSE. The network learns to be accurate where the forward model is most sensitive. Same architecture, same data, different loss.

**Grand hybrid (Model v3, multi-day):** Bigger architecture + more data + physics loss + ensemble. Target k=5-10.

---

## The bottleneck was never the angle basin — it's speed extraction (2026-06-18)

**This overturns the "narrow α_x basin is the wall" conclusion above.** That framing
conflated two independent sub-problems. Separating them changes everything.

### Proof: angles are trivial GIVEN the speeds

With glass/geometry/beam fixed to truth and the TRUE speeds supplied, on the
seed-2026 30-case battery (`test_trueseed_angles.py`):

- TRF from a **zero-angle init** (α=0): **17/30**
- A **joint 3-D grid over the three α_x** (α_y=0), screened by DiffFwd, one joint
  TRF per top node: **29/30**
- Grid ∪ zero-init: **30/30**

So once the speeds are right, the 6-angle recovery is solved. The α_y basin is
wide (a single midpoint init suffices); the only narrow directions are the three
α_x, and a deterministic 3-D grid covers them. This is the missing middle between
6-D DE (needle in 6-D → ~1/30) and 2-D-per-prism coordinate descent (coupling → +0):
grid the three α_x **jointly**, let one joint TRF finish.

Direct basin probe (`test_basin_probe.py`) confirmed every actual failure had
`FFTmatch=False` or sub-resolution/low-cycle speeds — never an angle-basin miss.

### So the real wall is the FFT speed extraction

Failure taxonomy on the 30-case battery is entirely speed-side:
1. **Harmonic confusion** — top-3 FFT peaks are harmonics/cross-terms, not
   fundamentals. Fixed by multi-triple search (enumerate triples from top-K peaks).
2. **Close speeds** (|N_i|−|N_j| < ~0.06 Hz, e.g. cases 3, 11, 27) — two tones
   merge into one FFT bin. Resolution limit at T=10s. *With* true speeds, TRF
   resolves them fine (case 3 sep=0.027 solves), so it's an extraction limit, not
   a degeneracy.
3. **Low-cycle / clustered fundamentals** (min|N|·T < ~2, or 3 speeds within ~0.3 Hz;
   cases 7, 10, 14, 18, 19) — the slow/clustered prism's fundamental is weak or
   absent from the spectrum.

### New solver: `paper/solve9_grid.py`

Pipeline: top-K FFT peaks → rank candidate speed-triples by a cheap per-triple
coarse-grid forward screen (avoids the global-screen dilution that buries the
correct triple) → for the best triples, joint α_x grid → **batched Adam through
DiffFwd** (refines all grid nodes in parallel — the key speed fix; per-node scipy
TRF was ~13 min/failed-case) → scipy-TRF polish the best few to ~1e-12.

**9-D battery result (FFT speeds, glass/geo/beam fixed):** GRID ≈ **16/30** vs the
ML-init baseline ≈ **9/30** — roughly 2×, ~8 clean "GRID saved" cases. Every GRID
failure is a speed-side physics limit (close/clustered/low-cycle), plus a few
solvable-but-missed cases (13, 14, 26) where triple-ranking/budget dropped the
right triple — fixable, not fundamental.

### What this means for the paper / next steps

- The recovery story should be re-framed: **spectral speed identification is the
  hard part; angle recovery given speeds is deterministic** (grid + TRF, 30/30).
- Next lever is super-resolution frequency estimation (ESPRIT / matrix pencil) +
  with-replacement seeding for close speeds, NOT more angle-init cleverness.
- New files: `solve9_grid.py` (solver), `test_trueseed_angles.py`,
  `test_basin_probe.py`, `test_alphax_grid.py` (experiments).

---

## Lattice VarPro: the brute force is gone (2026-07-17)

**Goal set today: solve the inverse problem "absolutely and analytically (or ML),
without brute force" — eliminate the C(8,3)=56 triple enumeration, the 8 sign
combos, and the 729-node alpha_x grid. Achieved for 24/30 battery cases.**

### The three structural observations

1. **The complex analytic signal kills the sign search.** Work with
   z(t) = x(t) + i·y(t). Each prism's fundamental sits at SIGNED frequency
   N_i; positive and negative speeds are different spectral lines. (The
   conjugate leak at -N_i, from x/y gain asymmetry, is weaker than the main
   line — a physical sign test.)
2. **In `core.py`, ay_i is exactly a rotation phase offset** (`gamma + sphiy`)
   and ax_i only sets tilt magnitude (`tan(sphix)`). Hence
   arg(c_i) at the fundamental = ay_i (+180° iff ax_i < 0, resolved by the
   ±18° box), and |c_i| encodes tan(ax_i)·(lever-arm gain). Angles are READ
   OFF the spectrum — no grid.
3. **The pattern lives on the lattice {k·N : k ∈ Z³}**, so speed extraction is
   generator recovery, not peak picking. Harmonic confusion becomes structure:
   2N₁ is the point (2,0,0), N₁+N₂ is (1,1,0). A harmonic cannot masquerade
   as a fundamental because the full line set is inconsistent with it.

### The method (paper/spectral_speeds.py)

1. **De-glitch**: TIR-clip samples (sq≤0 in the trace; diag_lattice.py showed
   cases 10, 30 have jumps of 300–700 units) detected by pattern-jump
   threshold, masked; all fits run on masked samples.
2. **CLEAN line growth (B=1)**: repeatedly take the strongest matrix-pencil
   line of the residual, refit ALL line frequencies jointly (VarPro: amps
   linear, freqs by damped GN). At order 1 a line explains only itself, so no
   compromise basis can absorb foreign lines. Novelty preference (skip lines
   representable as small combos of existing ones) keeps harmonics from
   exhausting the 8 slots before a weak fundamental appears.
3. **Basis selection by lattice coverage**: score candidate generator triples
   by amplitude-weighted small-integer coverage of all lines (|k|₁≤3, aliases
   f±1/dt included), with a consensus reweight that zeroes lines no top-8
   basis can explain (CLEAN artifacts). Coverage is the PRIMARY gate; the
   B=3 lattice-fit residual only referees candidates within 0.05 coverage.
4. **Full lattice VarPro fit** at |k|₁≤3 (B=4 refinement), with two
   protections: RIDGE amplitudes (unregularized LS explodes canceling pairs
   on near-coincident lattice lines and the giant |c| poisons ranking) and
   MERGING of lattice lines closer than 0.012 Hz (min-|k|₁ representative —
   the data cannot distinguish them; ridge otherwise SPLITS big components
   across near-collinear columns since ||c||² halves).
5. **Canonicalization**: physical fundamentals = rank-extending largest-|c|
   lines of the fitted model (first order dominates). One extraction on the
   CHOSEN fit is beneficial (it can rescue a mis-selected third generator via
   a (1,0,1)-type row); extraction after REFINEMENT fits is poison (grabs
   split twins) — refined generators ARE the speeds, never re-extract.
6. **Polish**: guarded GN refits (movement < 0.07), a final SHARP fit (no
   ridge/merge) for the last decimal, self-consistent glitch remasking, and
   the ±e_i amplitude sign test.

### Results (seed-2026 30-case battery)

- **Signed speeds**: 24/30 exact (<0.02 Hz) at rank 1, ~25/30 within top-3
  bases; median error 3×10⁻⁴ Hz (FFT top-3 magnitude match: 13/30, and it
  never sees signs). ~1 s/case, ZERO forward-model evaluations.
- **End-to-end 9-D (`paper/solve9_spectral.py`)**: speeds + phases→ay +
  amplitude→|ax| (per-prism cubic amp = a·tanα + b·tan³α calibrated with 2
  forward evals per prism) + sign-of-ax from the phase branch + one scipy-TRF
  → **24/30 PERFECT (err ~1e-12)** vs solve9_grid 16/30 and ML-init 9/30.
  Solved cases: 'primary' rung, <1 s (grid was ~60 s). Deterministic
  verification ladder (phase-branch flips, zero-angle init, flip-weakest-
  speed, alternate bases), each rung verified by pattern MSE — no search.
- Notable solves: case 3 (close pair, sep 0.027 Hz — FFT-impossible),
  case 30 (TIR-glitched, masked), case 2 (accidental relation
  N₂ ≈ N₁+2N₃ to 1.2e-3), case 28 (1.5-cycle slow prism), case 16
  (aliased order-3 lines — the lattice model folds exactly).

### The six remaining failures (all spectral-stage, none angle-side)

1. Cases 11 (sep 0.007 Hz) and 27 (sep 0.059, leak-cluster): close-pair
   resolution at T=10 s. ΔfT ≤ 0.6 — at the information-theoretic edge.
2. Cases 4, 19: the slowest prism is spectrally tiny (fundamental amp ~0.3%
   of signal) AND low-cycle; its line is found (novelty-CLEAN sees +0.207 for
   case 4) but sign/bias are unreliable at that SNR-equivalent.
3. Case 18: EXACT accidental relation N₃ ≈ N₁+N₂ (within 5e-4) — the true
   lattice is numerically rank-deficient at T=10 s.
4. Case 10: TIR case under pinned BLAS; masking recovers speeds in some runs
   (5e-3) but the fit residual floor (0.1) leaves angle inits polluted.

All six would yield to longer observation (T=20–40 s); worth ONE experiment.

### Hard-won implementation lessons (each cost a battery round)

- Residual-guided greedy basis growth at B=3 is BROKEN by design: a finer or
  compromise lattice always fits at least as well (case 22: g=1.096 absorbs
  the -3.289 line as k=-3 and -1.846 as (1,-2)). Grow at B=1, select by
  arithmetic coverage, only then fit B=3.
- lstsq on near-coincident columns → canceling amplitude explosions; ridge →
  amplitude SPLITTING (||c||² halves); the cure is merging + ridge + a
  sharp-only final step.
- An underdetermined lattice fit (4 gens at B=4 = 321 columns > 200 samples)
  interpolates exactly and reports residual 1e-15 — guard the design size.
- Never re-extract fundamentals from a refit; keep the guarded generators.
- Multithreaded BLAS makes eig/svd non-bitwise-reproducible; the greedy
  amplifies it into different line lists run-to-run. Pin threads
  (OMP/MKL/OPENBLAS_NUM_THREADS=1) for reproducibility.

### Next steps

1. **18-D presets**: the spectral fingerprint (signed N, DC, complex amps of
   all |k|₁≤2 lines + conjugate-leak ratios) as input to a small NN →
   glass/geometry/beam init, then 18-D TRF. The fingerprint is ~50 numbers in
   the RIGHT coordinates (frequency structure factored out) — this is where
   "or machine learning" enters, replacing the raw-pattern nets.
2. Close pairs: longer-T experiment; and a dedicated two-line splitter seeded
   at the merged line ± CRB-scale offsets.
3. Weak-prism sign (cases 4/19): both signs are cheap verified candidates —
   wire 'flipweak' earlier into the ladder with full angle re-init.
4. Paper: the story is now "quasi-periodic lattice inversion: pencil + integer
   programming on the frequency lattice + VarPro + TRF", replacing every
   enumeration in Sec. IV. New files: spectral_speeds.py, solve9_spectral.py,
   test_matrix_pencil.py, test_lattice_varpro.py, diag_lattice.py,
   debug_varpro*.py.

---

## Full 18-D + certificates + every assumption tested (2026-07-18)

Directive: "get the whole way done", and for what we miss, "an error bound
where we say within such tolerance, based on a function of whatever, we
cannot do it — super tight." Both delivered.

### FULL 18-D recovery, NOTHING assumed known: 26/30 PERFECT

`paper/solve18_spectral.py`. Same spectral front end; then:
- ay from fundamental phases; |ax| from amplitudes via the cubic gain
  calibrated at NOMINAL glass/geometry (2 forward evals/prism);
- beam angles analytically from the pattern DC (rotating deflections average
  out, the DC is the static ray): bm_a = atan(DC / L_nom);
- glass = 1.55, d_W = 125, gap = 8.5 mid-box start;
- one masked 18-D TRF + the verified ladder (recalibration with the fitted
  geometry, phase-branch flips, zero-angle, flip-weakest, alternate bases).

**Result: 26/30 with all 18 parameters at ~1e-11 (pattern MSE 1e-23..1e-26),
20 s/case average, 1-4 s for clean cases.** The full-18 problem beats the
frozen-geometry 9-D protocol (24/30): TRF's freedom in glass/geo/beam plus
the richer ladder rescues cases 19 (alt basis), 27 (alt basis), 30 (zero
rung). Failures: 4, 10, 11, 18 — exactly the certified information-limited
set. For reference: April's best was 5/5 on an easy battery and ~30% on this
one, at ~35 min/case with 448-candidate screening.

### Certificates (`paper/certify.py`) — the "super tight" bounds

- SUCCESS: per-parameter bound = 3*sqrt(diag s^2 (J^T J)^-1) + |(J^T J)^-1
  J^T r| (covariance + optimality-gap/Newton-step term for early-stopped
  TRF), J the exact-model numerical Jacobian at the solution. Battery:
  **26/26 coverage, median tightness 5x** (bound / actual error), i.e.
  bounds like 2e-10 against errors 5e-11 — parameter-wise, certified.
  Subtlety: rank(J)=18 forbids covariance truncation — the weakest
  (d_W↔gap) eigenvalue is ~2e-15 of the largest and carries the dominant
  bound; a pinv rcond=1e-14 silently dropped it and broke coverage on one
  case until fixed (rcond=1e-18).
- FAILURE: Fisher information of the fitted lattice model (merged design, no
  ridge) yields sigma(N_i), sigma(amp_i); certificates fire quantitatively:
    close-pair    |ΔN| < 3σ           → "need T ≥ T(3σ/Δ)^(2/3)"
    weak-prism    amp < 5σ_amp        → "any |ax| < X° is invisible at this T"
    relation      |k·N − N_j| < 3σ    → "subspace degenerate at this T"
    glitch-floor  TIR mask + residual → "all σ inflated ~Nx"
  Every failed case fires the correct mode(s); e.g. case 11 (sep 0.007 Hz):
  "need T ≥ 35-55 s" — and the A8 experiment measured it solving at T = 40 s.

### Assumption test suite (`paper/test_assumptions.py`) — all green

  A1 flip symmetry (ax,ay)->(-ax,ay+180) EXACT: max|dF| 2.2e-11
  A2 conjugate leak < main: median ratio 0.095, 1/90 violations
  A3 phase = ay + 180[ax<0]: median 0.85 deg (p90 8.5)
  A4 lattice support: B=4 residual median 5.5e-4; 3/30 inadequate (TIR)
  A5 fundamental dominance: 3/24 violations, tolerated by rank-extension
  A6 amplitude->|ax|: median 0.26 deg (true geom) / 1.49 deg (nominal)
  A7 MEASURED 18-D TRF basin: 28/30 at actual spectral init-error scale,
     24/30 at 4x that — the "narrow basin" folklore is retired
  A8 information scaling: EVERY remaining failure solves with longer
     observation — case 11 at T=40 s, cases 4/18/27 at T=20 s

So the failure taxonomy is fully constructive: nothing is mysteriously hard;
every miss is certified as information-limited at T=10 s with a prescribed
observation time that cures it (verified empirically).

### Next session: the paper rewrite

`paper/REWRITE_PLAN.md` holds the complete blueprint: section plan, the
theorem/proposition list (each mapped to its test), the salvage map for the
old text, and the headline numbers. Pre-submission experiment TODOs recorded
there: noise battery (`test_noise_spectral.py`, written, not yet run), P=2
and P=4 batteries, model-mismatch redo on the new pipeline.

---

## Formalization, canonical batteries, and the paper rewrite (2026-07-18b)

### Restructure (commit b6fb011)

The method is now a torch-free package `risley_lattice/` (model, lattice,
spectral [P-agnostic n_gen], angles, solve, certify) with every battery a
thin script in `experiments/`; 53 superseded one-offs moved to
`paper/archive/` (old→new map in its README); CLAUDE.md updated. The
parameter box and the seed-2026 battery generator live in ONE place
(risley_lattice/model.py).

### Canonical numbers (the package runs; these are what the paper cites)

- speeds:      25/30 exact signed (top-3: 26/30), median 4.5e-4 Hz; FFT 13/30
- 9-D:         24/30 PERFECT (~1e-12); failures 4,10,11,18,19,27
- 18-D:        25/30 PERFECT (~1e-11), median ~2 s/case;
               failures 4,10,11,18,19 (case 27 rescued by alt tuple; case 19
               is the boundary-flicker case — succeeded in the pre-refactor
               run via alt1, fails under the package's FP path; certified
               close-pair/weak-prism with T_req 40–80 s)
- certificates: 25/25 coverage, median tightness 6x
- noise (10 clean cases): inf 10/10 cov 10/10 (bounds 2e-10);
  60 dB 9/10 cov 8/9 (1.7 / 0.16); 50 dB 10/10 cov 10/10 (6.0 / 1.3);
  40 dB 9/10 cov 9/9 (16 / 2.7); 30 dB 8/10 cov 7/8 (61 / 13).
  Coverage 34/36 total = ~2 parameter-check misses of ~650 at 3σ vs 1.9
  expected: the bounds are statistically CALIBRATED. The inflation is
  concentrated in d_W↔gap exactly as the singular spectrum predicts.
- prism count: P=2 18/20 (median 2e-6); P=4 6/20 at T=10 s but 15/20 at
  T=40 s — the information-scaling law, not the algorithm, sets the
  prism-count frontier.

### Paper rewritten from scratch (paper/main.tex, 9 pp, compiles clean)

"Frequency-Lattice Inversion of Multi-Prism Risley Systems: Exact Parameter
Recovery with Information-Theoretic Certificates." Theorem/proposition
structure with every claim mapped to a named test (A1–A8 audit table in the
paper): flip symmetry (Prop 1, exact — and the old paper's global-smoothness
claim is CORRECTED: Prop 2 characterizes the TIR set, 2/30 ensemble),
lattice support (Thm 1), signed fundamentals + conjugate-leak inequality,
phase/amplitude readout laws, the finer-lattice lemma (why residual-guided
selection provably fails — three failed solver generations documented as
negative results with content), spectral identifiability conditions (i)–(iv)
with T_req prescriptions, certificate section (no covariance truncation —
rank-18 argument), canonical results tables, calibrated-noise section,
prism-count generality. New figure figures/certificates.pdf (bound-vs-error
scatter + measured basin; generated by experiments/paper_figures.py from the
canonical runs). NN content deleted. Precision matters: A8 claims are worded
as what was measured (speed-extraction cure at T_req, not full recovery —
full-recovery-at-T generalization of solve18 is a listed TODO).

### Remaining before submission

- Model-mismatch battery on the new pipeline; hardware validation.
- Case 19 boundary flicker: certified honestly, but a CLEAN-stage
  robustness pass could reclaim it (junk-heavy line lists on lowcyc cases).
- Referee-proofing pass on the manuscript (overfull boxes, figure sizing).

---

## N=617 ensemble complete + autopsy + paper (2026-07-19)

**Adaptive battery finished: 617 configurations** (seed 7777; solve @10 s,
certificate-prescribed ladder 20/40/80 s):
- 489 (79.3%) at T=10; +86/+13/+2 at 20/40/80 → **590/617 = 95.6%
  recovered**, median err 1.2e-11, median 1.8 s/case.
- 27 unsolved, autopsy (scratchpad/autopsy.py, reclassified with the
  corrected merge-floor T_req + truth checks):
  * **16 certified-infeasible at ≤80 s, VERIFIED**: relations with
    0.2–5 mHz gaps (corrected T_req 104–2400 s — the new formula's first
    outing at scale, incl. 800 s and 1200 s prescriptions on fresh certs);
    weak-prism thresholds CONFIRMED against ground truth (e.g. cert
    "any |ax|<6.9° invisible", true 3.62°).
  * **11 (1.8%) algorithmic residual, SELF-REPORTED**: cert says
    T_req ≤ 80 yet solve failed (7 ladder misses incl. certs computed
    from a wrong T=10 fit — case 161's threshold provably false vs
    truth; 4 no-speeds: 81, 130, 223, 317). Key insight: this mismatch
    is detectable AT RUN TIME without truth — cert-feasible + failed
    solve = the algorithm's own confession. NO SILENT FAILURES.
- Caveats recorded in the paper: certs from a mis-fitted T=10 model can
  misstate individual thresholds (still self-reporting); multi-deficit
  cure-time composition is not derived.
- Paper: new Results subsection (ensemble table + self-reporting frame),
  abstract-adjacent cert paragraph updated; Li et al. 2017 (GA
  calibration of 6 params around known nominal) cited & delineated after
  a fresh literature check — novelty phrasing hardened. 11 pp clean.

**Open forensics (next):** line-level autopsy of the 4 no-speeds cases;
optionally harden CLEAN/selection to shrink the 1.8% residual; deep
multi-database literature sweep before submission.

### Residual autopsy (2026-07-19b, experiments/autopsy_residual.py, 63fb2f2)

The 11 "algorithmic residual" cases decompose into three mechanisms:
1. **Hidden weak prisms (102, 130, 161, 313, 317)** — true wedges
   0.04°–1.2° (fund amps 0.03–2.2), genuinely at/below detectability.
   Mislabeled because (a) rank<3 aborts certification before the Fisher
   verdict, (b) certificate prism indices refer to the fitted (wrong)
   basis, not canonical truth. These belong in certified-infeasible.
2. **Glitch-budget overflow (81, 223; partly 268, 529)** — TIR so dense
   the 15% mask budget aborts masking; unmasked impulses poison CLEAN
   (near-Nyquist junk lines) and pin acceptance MSE above 1e-12 even
   when extraction succeeds (case 81: extracts to 3.3e-5 at T=80, still
   fails acceptance). Fixes: softer budget, iterated remask, robust
   acceptance residual.
3. **Margin selection (408, 434)** — truth in the line list at T=80 but
   selection picks wrong near the merge floor; 434 is the single case in
   617 with no visible pathology at all.

True algorithmic residual ≈ 6/617 = 1.0%. Paper table intentionally NOT
yet updated: it reports what the shipped certifier says; the split moves
to ~3.4% infeasible / ~1.0% residual only after implementing (and
re-running) the two certification-plumbing fixes: 2-generator Fisher
verdict on rank<3, and index-aligned certificate reporting.

---

## PERFECT BARRING MATHEMATICS (2026-07-19c, commits ce1b64c…2d3d9e3)

The 434 hunt ("it only takes one — a counterexample to the error
analysis") produced **condition (v): front-end capacity** — dense-comb
spectra (tooth spacing = minimal small-k lattice value) overload any
fixed-order pencil into returning cluster centroids; self-diagnosed by
res_clean ≫ lattice floor; cured IN PLACE by capacity-free FFT-peak
seeding + joint GN (no extra observation needed). Plus: leak-inequality
sign test now UNCONDITIONAL (a rejected polish had skipped it; 434
violated it 14× unseen), lattice_fit hardened vs non-finite, rank<3
extraction stashes partial generators, and the certifier renders
**rank-deficient detectability verdicts** (case 130: cert "any remaining
prism |ax|<0.77° invisible", truth 0.39° ✓; case 317: 0.11° vs 0.05° ✓).

Regressions all positive: speeds 26/30 (was 25), canonical 18-D 26/30
(case 19 recovered), residual sweep 6/11 rescued — every rescue via the
overload retry (223: unsolvable→solved at T=10 in 1 s).

**Definitive clean run (N=600, final pipeline, results/): 493 (82.2%)
at T=10 s; 582 (97.0%) adaptive; median err 1.1e-11, median 2.0 s;
18 unrecovered, ALL certified in one pass (certify_unsolved.py) in
truth-verified classes: 10 relations (T_req 76–1856 s), sub-threshold
wedges incl. the two rank-deficient verdicts, 1 TIR floor, 2 close-pair
marginals within the safety constant. ZERO unexplained, ZERO silent.**

Paper updated (11 pp clean): condition (v) + estimator/signal
distinction in the Definition; overload retry + unconditional sign test
in Sec IV; falsification-loop narrative (merge floor episode + 434
episode) in Discussion; ensemble table = the clean-run numbers with the
self-report-driven-fix story. Known caveats stated: basis-relative cert
indices; multi-deficit cure composition not derived.

Remaining for submission: human proofread of the PDF; model-mismatch
battery; deep literature sweep; GitHub README; hardware (stated
limitation).

---

## T-generalization, hero figure, N=500 adaptive battery (2026-07-18c)

- solve18/solve9 now accept ANY recording length (n_pts/time_limit threaded;
  fixed latent bug: line-merge tolerance was hardwired to 10 s resolution —
  now MERGE_C/T_span). **Full 18-D recovery verified at the certified
  prescriptions**: case 11 (0.007 Hz pair) 3.3e-10 @ T=40 s; case 18 (exact
  relation) 3.9e-12 @ 40 s; case 4 1.4e-11 @ 40 s; case 19 8.3e-12 @ 80 s.
  29/30 fully recovered given adequate observation; case 10 = TIR (non-T
  pathology). Commit e35d2ca.
- Paper figures per user directive: hero.pdf (observed dense pattern → its
  lattice line spectrum → reproduction from recovered 18 params, 4.4 s,
  2e-11) as Fig 1; svd_fresh.pdf regenerated from the package; OOD figure
  DROPPED, wedge-count subsection compressed. Compiles 9 pp clean.
- **N=500 adaptive battery** (experiments/adaptive_battery.py, seed 7777,
  ladder T=10→20→40→80): PAUSED overnight at **186/500 banked**
  (resume-safe JSONL; relaunch 5 workers: `--start {0,100,200,300,400}
  --count 100`). Interim at n=148: 79% solved @10 s; +cures mostly at 20 s;
  adaptive ≈95%; 9 unsolved, ALL certified. Two findings:
  1. **T_req under-prescribes for near-exact lattice relations** (gaps
     1.4–3.2 mHz): the formula extrapolates 3σ scaling but omits the MERGE
     FLOOR (lines closer than 0.12/T are deliberately indistinguishable →
     gap g needs T ≳ 2·0.12/g on top of σ-scaling). Fix: T_req =
     max(σ-term, merge-floor term) in certify. Cases 16/116/408/427.
  2. **2/148 'no-speeds' cases (130, 223) with no visible pathology** —
     possibly CLEAN/selection algorithmic misses, NOT information-theoretic.
     Autopsy required; if algorithmic, the paper owns an ~1–2% algorithmic
     residual class explicitly.

### DONE same evening (2026-07-18d): the analytical error-bound section

Paper Sec. "Error Analysis and Certificates" now DERIVES everything
(commit follows): Prop (error decomposition — the Newton-gap term is a
Taylor identity, not a heuristic; proof included), Cor (certificate
coverage at 3σ with stated assumptions; noiseless case = deterministic
floor bound), remarks (no-truncation with the rank-18 argument; PATH
INDEPENDENCE — acceptance ⇒ certificate regardless of initializer; 
calibration testability), Prop (lattice Fisher = exactly the matrix
certify inverts; closed-form σ(N̂_i)² = 3σ_w²/(2π² f_s T³ S_i) under
separation, proof sketch via cisoid orthogonality + centered second
moment), Cor (T^{-3/2} law), Prop (pair degradation Θ((gT)^{-2}) via the
Gram eigenvalue + the MERGE FLOOR as a deliberate design constraint ⇒
**T_req = max(T(3σ/g)^{2/3}, c·C_m/g)**, c≈2 — validated: prescribes
171 s for the 1.4 mHz relation case that failed at 80 s), Prop
(detectability α_min ∝ T^{-1/2}), and an honest "derived vs measured"
paragraph (basin A7 and the constant c are measured; everything else is
computed per instance). certify.py updated with t_required() (merge
floor included). Paper now 10 pp, compiles clean.

### Original plan (kept for reference): ANALYTICAL error bounds

"We need to actually bound our error with analysis." The bounds are
currently computed (Fisher/covariance + optimality gap) and empirically
calibrated; the paper needs the DERIVATIONS as theorem-grade analysis:
1. Completion-stage bound: Gauss–Markov/CRB derivation for nonlinear LS at
   a converged/early-stopped iterate — state assumptions (local linearity,
   noise model, rank-18), derive Eq. (bounds) incl. the Newton-gap term,
   and the conditions under which 3σ coverage holds.
2. Spectral-stage: derive σ(N̂) for the lattice model (multi-line CRB),
   prove the T^{-3/2} law at fixed f_s, and derive the corrected
   T_req = max(3σ-scaling, c·MERGE_C/gap) with the merge-floor term.
3. Detectability threshold: derive the 5σ amplitude test → minimum
   detectable wedge angle formula (through the cubic gain).
4. Propagate spectral→completion: show spectral init error within the
   MEASURED basin (A7) ⇒ certified endpoint — closing the pipeline-level
   guarantee.
Then: finish the N=500 battery, add the merge-floor term to certify.py,
autopsy cases 130/223, re-aggregate, update paper Secs V–VI.

## 2026-07-20 — Pre-registration restructure: math-only paper + campaign skeleton (commit fcbe597)

User directive: "No 30 no nothing basically I just want all the hard
mathematics in. I just want big placeholders where we run millions of
simulations etc and analyze the results... the 4N+6 dimension... needs
to be SUPER HYPER RIGOROUS." Plus mid-turn: placeholder empty graphs
that say what data we will need; keep methodology exposition.

What changed in paper/main.tex (14 pp, compiles clean, 0 undefined):

1. **Every small-N number is gone.** Abstract, contributions, results,
   discussion, conclusion: no 30-case, no 26/30, no 600-ensemble, no
   noise table. Embedded A1–A8 measured values inside proofs became
   \PH{} placeholders (yellow \colorbox macro). Battery anecdotes that
   are development *history* (merge-floor episode, case-434 story,
   pinv-truncation coverage break) kept but reworded as
   "development-scale" without headline stats. Hero + SVD figures kept
   (single-instance illustrations, relabeled "representative random
   configuration"). certificates.pdf figure dropped (was 26-case data).

2. **NEW Sec VI: Scaling Theory for Arbitrary Prism Count** (dim 4P+6;
   fixed intro's wrong 3P+9). All proved:
   - Lemma (lattice population): |K(P,B)| = sum 2^j C(P,j)C(B,j) =
     Theta(P^B). Cost polynomial; readout linear in P.
   - Prop (exact crowding law): min pair gap of P uniform magnitudes:
     survival (1-(P-1)s/L)_+^P, E = L/(P^2-1) (simplex proof).
     Corollary: T_pair = c*C_m*P(P-1)/(L*delta) = Theta(P^2/delta).
   - Prop (relation-gap anti-concentration): Pr[g_rel<eps] <=
     2eps/L * |K(P,K)| = Theta(eps P^K/L), honest converse via first
     moments only. Corollary: T_rel = Theta(C_m P^K/(L delta)).
   - Prop (pigeonhole): |k.N| <= PQW/((Q+1)^P - 1) at |k|_inf<=Q —
     exponentially small; why bounded-order window is load-bearing.
   - Prop (capacity threshold): overload generic once T >~
     (C_m/f_s)Theta(P^B). Prop (detectability P-invariant).
   - Theorem (feasibility frontier): T* = O(P^max(2,K)/delta),
     Omega(P^2). Conjecture: empirical gamma=2 for P<=6.

3. **Results → Sec VII: Large-Scale Computational Campaign.**
   Pre-registered framing: sweep.py/aggregate.py frozen, plots designed
   before data. Design subsection (order-free case stream, margins at
   truth, adaptive protocol, Wilson/bootstrap stats, pinned BLAS).
   E1 atlas (1e6, zero-unexplained target), E2 calibration (1e7 checks,
   1e-4 binomial resolution), E3 exponent fits (-3/2,-2,-1/2 as point
   predictions), E4 prescription bisection, E5 P×T frontier (margin
   collapse + gamma + capacity + amplitude non-decay), E6 noise,
   E7 baselines (full ML methodology retained + oracle-speed control,
   1e4 common subset), E8 audit table all-\PH. Each E has a framed
   \PHBLOCK "PENDING CAMPAIGN DATA" empty-figure spec: axes, binning,
   overlays, data files, generator script.

Gotcha logged: python-heredoc splice via bash ate \ in tabular rows
(single backslash survived) — misplaced-alignment cascade + killed
pdflatex left corrupt aux locked by zombie process. Fix: restore \,
kill process, rm aux, clean double compile.

Next: run the campaign (sbatch commands in slurm_sweep.sh), then
replace every \PH/\PHBLOCK with data via aggregate.py; E7 needs a
baseline-scoring sweep mode (not yet in sweep.py — TODO); A1–A6/A8
audit extraction from atlas records needs an aggregator pass (TODO).
