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

## 2026-08-12 — DETERMINISTIC error control: no statistics, validated numerics

User directive (verbatim intent): "we need to deterministically and
analytically control the error bounds. you can make physical assumptions
but you may never use statistical heuristics." This retires the
3σ/Fisher certificates as the *primary* error-control story: Gaussian
tails, s² surrogates, 5σ detectability — all statistical. Replacement:
worst-case validated-numerics certificates, computer-assisted-proof
grade.

### The framework (implemented, working)

**Assumptions — all deterministic, all stated:**
- A-data: per-sample data error |y_i − F_i(θ*)| ≤ η, a hard bound
  (sensor quantization / generator arithmetic floor). NO distribution.
- A-fp: IEEE-754 binary64 round-to-nearest semantics.
- A-libm: |libm sin/cos − exact| ≤ 1e-15 abs for |x| ≤ 1e4 (≥4× worst
  known ulp error of mainstream libms; swap crlibm for a formally
  airtight chain).
- A-blas: standard elementwise matmul error bound 2nU|A||B|.
- Physical branch margins certified PER INSTANCE over the whole box:
  no TIR (sq ≥ margin), forward propagation (sf2 > 0), non-grazing
  intersections (|1 − m·t| > 0). These are the "physical assumptions"
  — and they are *verified*, not assumed.

**Mathematics** (risley_lattice/certify_det.py docstring has the full
statements):
1. Anisotropic box X = θ̂ ± r∘RG; certified interval enclosure [J](X)
   via rigorous midpoint-radius interval arithmetic + first-order
   interval AD (risley_lattice/ivx.py — outward rounding via nextafter,
   containment contract per op).
2. Row-wise mean-value theorem ⇒ F(a)−F(b) = J̃(a−b), J̃ ∈ [J](X).
   With ANY preconditioner C (pinv of scaled midpoint J; C needs no
   rigor), E = C[J_s](X) − I, β = ‖|E|‖∞ < 1 gives:
   (i) F injective on X — certified exclusion box: no second
       explanation of the data inside X;
   (ii) if θ* ∈ X: componentwise |δ_s| ≤ num + |E||δ_s|,
        num = |Cr| + η·rowsum|C| ⇒ verified fixed-point bounds b
        (downward iteration from the uniform bound, every iterate
        valid). Coverage is 1 BY CONSTRUCTION — a miss falsifies an
        assumption; there is no "unlucky draw";
   (iii) dichotomy: either |θ̂−θ*| ≤ b or θ* outside X (quantified
        ambiguity — which the failure side now exhibits
        CONSTRUCTIVELY: certify_ambiguity certifies ‖F(θ_alt)−y‖∞ ≤ η
        by interval evaluation; the alternative may be found by any
        optimizer, only the certified inequality matters — the
        deterministic replacement of the Fisher close-pair/weak-prism
        certificates).
3. Box shape by ε-inflation (r ∝ d: tiny along t-amplified speeds,
   large along weak d_W/gap), exclusion box grown outward in a
   verified chain. Search heuristic, rigor entirely in the final
   evaluated (r, d, β) triple. Path-independent as before.

**Key enabler** — the forward model per axis reduces ALGEBRAICALLY to
sin/cos/tan of affine parameter functions composed with +,−,×,÷,√:
core's arccos/degrees round-trips cancel (tan θ_new = sf0/sf2 exactly;
surface slope m = u/√(w²+1); sinφ = u/‖·‖). fmodel.py implements the
exact mathematical model; core's fp guards (1e-30 regularizer,
|sf0|<1e-12 branch, mod-360) are approximation error of the GENERATOR,
absorbed in η. Parity: 28/30 clean battery cases |Fm−core| ≤ 6.8e-10;
cases 10/30 raise certified TIR MarginError (they're the known glitch
cases — the interval model *detects* them analytically). Masked
samples patched with dummies; margins certified over retained samples.

### First numbers (case 1 smoke test; battery running)

θ̂ from TRF polish (mse 1.2e-23): certified data floor 2.0e-10;
η = 1e-8 a-priori ⇒ closed certificate, β = 0.49, 18/18 coverage,
speeds b ≈ 9e-11, angles ≈ 3e-7 deg, d_W ≈ 3.6e-6. Deterministic vs
old 3σ ≈ 1e4× looser — the honest price of worst-case + η ≫ actual
floor. At η = 2×floor the bounds tighten ~25×. Certificate cost
~0.3 s/case (forward_iv 25 ms, T=200).

**Current limitation (v1):** β has a constant part ≈ 0.4 from the
point-run J enclosure width (libm+fp slop amplified by ‖C‖ ~ 1/σ_min),
so the exclusion box only exceeds the enclosure by ~3× and is tiny
along speeds (~1e-11 scaled — far short of the 2e-15 statistical σ).
Tightening path: sharper rnd() constants, second-order (slope) forms,
per-component chain growth. The STATISTICAL certificates remain in the
paper as the tightness benchmark; the deterministic ones are now the
guarantee.

Files: risley_lattice/{ivx,fmodel,certify_det}.py,
experiments/certify_det_battery.py (30-case battery: solve18 → floor →
certify_det at a-priori η AND certified per-case floor → coverage must
be N/N → old-vs-new tightness; constructive d_W ambiguity demo).
Paper Sec. "Error Analysis and Certificates" must be rewritten to the
deterministic form (bounded-noise Corollary replaces Gaussian; Fisher
section becomes the tightness/benchmark discussion; new Prop for the
mean-value/Krawczyk certificate; constructive ambiguity replaces
3σ close-pair) — NEXT.

### Same day, addendum: the eta* threshold (important)

First battery attempt at a blanket a-priori eta = 1e-8 FAILED to close on
most cases ("eps-inflation did not close, beta ~0.5"). Diagnosis
(scratch/cert_diag): the constant fp/libm part of E is NEGLIGIBLE
(beta = 0.000 at r = 1e-13 — the interval arithmetic is essentially
free); beta is entirely box-driven and dominated by the t-amplified
SPEED columns (uniform r = 1e-9 already gives beta = 2.0, all from N_i).
The failure is REAL mathematics, not slop: at data-error eta, the
worst-case uncertainty box has speed radii ~ eta*l1C_N, and once that
exceeds ~a few 1e-9 scaled, the Jacobian's variation across the box
(second derivatives ~ (2 pi T)^2-amplified) destroys the contraction.

Consequence: each instance has a **maximal certifiable data-error level
eta*** — the linearization-validity noise threshold. Below eta*, closed
deterministic certificates exist; above it, a first-order validated
argument cannot close (would need subdivision/higher-order methods).
This is a new, honest, fully deterministic quantity with hardware
meaning: a sensor with worst-case error <= eta* inherits the
certificate. certify_eta_max() bisects for it (geometric bisection,
every probe independently verified — search heuristic, rigor in the
final triple).

Fixes that mattered: (1) seed the eps-inflation with the DECOUPLED
enclosure num = |Cr| + eta*rowsum|C| (correct anisotropy: tiny speed
radii, wide d_W/gap) — a uniform seed poisons the speed components
through the E-coupling and never closes; (2) monotone per-component
growth with dampened back-off; (3) primary certification at
eta = 2x the certified per-case data floor (A-data VERIFIED for the
synthetic generator by data_floor(), interval-certified at truth).

First 5 cases: floors 7e-11..5e-10, all certify at 2x floor with 18/18
coverage, beta 0.3-0.8, max-b 9e-8..1.4e-6 (absolute), eta* between
2.9e-9 and 5.0e-8. ~8 s/case including the eta* bisection. Full battery
running -> experiments/results/_certify_det.log + certify_det.jsonl.

### Final battery numbers (2026-08-12, experiments/results/_certify_det.log)

26/30 solved (unsolved 4/10/11/18 = the known information-limited set).
**25/26 deterministically certified, coverage 25/25 — ZERO misses**, as
the construction demands. eta = largest of {2, 1.5, 1.2, 1.05}x the
certified per-case floor that closes (floors 5e-11..7e-9). beta
0.20-0.95; bounds max-b 1.4e-7..3.4e-6 absolute; det/3sigma median
~7.5e3x (the honest price of worst-case + hard noise bound).
eta* (certifiable-noise threshold): min 5.0e-11, median 1.3e-8, max
9.8e-8 — hardware with sensor error below ~1e-8 typically inherits the
certificate at T = 10 s.

Honest residual: case 22 — eta* < 1.05x its own arithmetic floor, so
NO first-order deterministic certificate exists for it at the
generator's fp noise level (beta stalls ~0.72). Cure paths: tighter
enclosures (sharper rnd constants / slope forms), second-order or
subdivided boxes, or a cleaner generator (extended-precision y).

Constructive ambiguity demo (case 1): d_W pinned 1e-6 scaled off
(12.5x beyond the certified exclusion radius 8e-8), rest refit ->
certified sup-deviation 1.30e-6: at any data-error level above that,
the two systems are PROVABLY indistinguishable — the deterministic
replacement of the Fisher close-pair story, and a nice tightness
witness for the exclusion box (ambiguity appears ~1 order beyond it).

Paper: Sec. certify rewritten deterministic-first same day (assumptions
A-data/A-phys/A-arith, Prop validated-certificate with proof, eta*
paragraph, zero-tolerance remark, sec:constructive, E2 falsification
protocol, E6 bounded noise, abstract + contributions). Compiles clean:
0 errors / 0 undefined / 0 overfull. certify.py docstring marks the
statistical machinery RETIRED as guarantee (benchmark + scales only);
CLAUDE.md updated. Campaign TODO: sweep.py needs a certify_det mode
(E2/E6 now specify deterministic zero-miss protocols).

## 2026-08-12b — The ANALYTIC hardware map (user: "hardware-driven:
## this hardware can't work, everything else analytically SOLVED")

Directive escalation: the deterministic certificates must become
closed-form, hardware-driven verdicts. Implemented in
risley_lattice/hardware.py + experiments/hardware_map.py; paper gains
Sec. "Hardware feasibility: the analytic error map" (Prop minimax
floors + proof, Prop affine bound sheet, Cor hardware dichotomy;
19 pp, compiles 0/0/0).

**Three closed forms:**
1. **Minimax floors (two-point method).** For any direction v with
   certified segment gain g: if s·g ≤ 2ε̄, the midpoint data is
   admissible for both endpoints ⇒ EVERY method errs ≥ s|v_j|/2.
   The killer direction is phase-compensated speed (ay is exactly the
   rotation phase ⇒ v = e_N − 180T·e_ay cancels the T-amplification):
   err_N ≥ ε̄/(πTW), pair-infeasibility gap g_min = 2ε̄(Wi+Wj)/(πT WiWj),
   hardware time prescription T_hw = same/g. Worst-case resolvability
   is T ∝ 1/g — HARSHER than the statistical T^{-3/2} (adversarial
   noise). Also floors for ay (ε̄/W), ax, prism-invisibility, and the
   weak d_W↔gap direction (what this hardware can NEVER learn).
2. **Affine bound sheet.** d(ε̄) ≤ A + ε̄B (fixed point is linear +
   monotone): two certified coefficient vectors per instance bound the
   error for EVERY ε̄ ≤ η*_confirmed. Validated: sheet/direct-certificate
   ratio 1.00–1.01 (the formula IS the certificate).
3. **Closed-form η*.** β is affine along the ν-ray: two probes + ONE
   confirming evaluation (no search). Consistently 2.7–5x conservative
   vs the bisected η* — both rigorous. (Bug fixed en route: use the
   TIGHT closing box β, not the grown exclusion box's near-limit β.)

**Battery results (_hardware_map.log, hardware_map.jsonl):** 25/26
formula sheets (case 22 = the sub-floor instance, honestly refused),
**coverage 25/25 ZERO misses**. Speed floors ~1e-12 Hz at ε̄=1e-9.
Confusable witnesses certified on every instance (dev ~ 2ε̄ ✓).
Hardware classification of the battery: ε̄=1e-11 → 25 SOLVED;
1e-9 → 21 SOLVED + 4 INCREASE_T; 1e-7, 1e-5 → all INCREASE_T. Zero
IMPOSSIBLE at tol=1e-3: the minimax floors show information survives
even at ε̄=1e-5 — the 1e-7+ verdicts are certificate-reach (η*), not
information death. That honest gap (floors << η*-frontier) is the open
band a second-order/subdivision certificate would close.

Margin-safety fix: _two_point_floor now treats a physical-margin-
limited span as a valid conservative floor (case 19's TIR-adjacent
segment crashed the first run).

Status: the error control is deterministic end-to-end AND analytic in
the hardware spec: b_i(ε̄) = (A_i + B_i ε̄)RG_i for ε̄ ≤ η* (closed
form), minimax floors below, T_hw prescriptions between. Campaign
TODO unchanged (sweep.py needs certify_det + hardware-map modes).

## 2026-08-12c — Concision restructure: physics / model / error (+ placeholders kept)

User: "make this paper a little more concise... focus on the physics,
the model and the error" — then mid-edit: "We need placeholders still,
we're trying to get 10 million simulations on a computer somewhere."
Reconciled shape (backup of the pre-cut version:
paper/archive/main_pre_concise_2026-08-12.tex):

- **CUT: Sec. Scaling Theory for Arbitrary Prism Count** (~280 lines)
  → one sentence in contributions + conclusion: deferred to a
  COMPANION REPORT (with the P×T frontier experiment E5, exponent
  fits E3, and baselines E7).
- **COMPRESSED: Campaign E1–E8 sprawl (~330 lines, 10 PHBLOCK figure
  specs) → one tight Sec. "The Validation Campaign"** (~75 lines):
  frozen-protocol paragraph + four experiment paragraphs (E1 atlas,
  E2 zero-tolerance certificate falsification incl. eta*/exclusion/
  hardware-map/affine-sheet/confusable-pair records, E4
  prescriptions, E6 bounded noise) + the A1–A8 audit table — ALL
  values still \PH placeholders per the pre-registration directive.
- Abstract rewritten (~40% shorter): physics/model/error + one
  campaign sentence; no dev-scale numbers (user reaffirmed
  placeholder framing mid-edit — an interim version briefly had
  battery numbers; reverted).
- Contributions: 5 bullets → 3 (structure theory, enumeration-free
  inversion, deterministic certificates+hardware map) + a short coda
  (P-generality + A1–A8).
- Fixed all dangling refs (sec:scaling → hardware/companion wording,
  sec:baselines → companion, conclusion rewritten). The embedded
  \PH{A*}/\PH{E1*} boxes in kept proofs/tests UNTOUCHED (still
  pending campaign data). 39 \PH boxes remain — arXiv still blocked
  on the campaign by design.
- Result: 14 pp (from 19), compiles 0 errors / 0 undefined /
  0 overfull.
- Gotcha (AGAIN, same as 2026-07-20 log): bash-heredoc python splice
  ate the \ tabular row terminators in the A-audit table → 32
  compile errors + a stale main.out from a killed pdflatex caused a
  bogus "Runaway argument \@@BOOKMARK" — rm main.out + restore \.
  RULE: never splice tabular LaTeX through a bash heredoc; use Write/
  Edit tools directly.

Companion-report scope now explicitly: P-scaling theory (proofs
preserved in the archive backup), E3/E5/E7, plus their PHBLOCK
figure specs.

## 2026-08-14 — Lean formalization begins: certificate logic MACHINE-CHECKED

User (after the sabotage test settled the heuristics scare): "we need
to use the lean to nail this down impossibly tight." Toolchain: elan
4.2.3 + Lean 4.33.0 + mathlib (pinned via lake-manifest), project at
C:\Users\josep\lean\risley (OUTSIDE Dropbox — the multi-GB .lake cache
must not sync); sources mirrored to repo formal/ (README + pins).

**Nine theorems, zero sorry, build green (8710 jobs):**
- Certificate.lean (Tier 1 — the certificate logic): enclosure_bound
  (x = c − Ax, ‖A‖<1 ⟹ ‖x‖ ≤ ‖c‖/(1−‖A‖)), perturbed_injective
  (exclusion box), mulVec_mono + fixedpoint_invariant (THE invariant
  that makes every downward iterate a valid bound — the formal answer
  to "search heuristics can't fake a certificate"),
  midpoint_admissible, two_point_floor, minimax_floor (the two-point
  minimax argument end-to-end).
- PhaseCompensation.lean (Tier 2 — model algebra, previously "parity
  tested"): phase_compensation (the compensated speed direction shifts
  the rotation angle by exactly 2πδ(t−t_c) — underpins every
  closed-form speed floor), cos/sin_compensated_dev (Lipschitz
  deviation bounds), refraction_tangent
  (tan(sign(a)·arccos(b/√(a²+b²))) = a/b — fmodel's collapse of
  core's arccos round trip is now a theorem, not a parity check).

Iteration notes: first build failed on ONE goal (field_simp leftover
a²+b²−b²=a² → ring); omit-before-docstring placement; style linter
wants copyright headers. Everything else compiled first try —
mathlib's MVT/norm/matrix API covered all needs (le_div_iff₀,
lipschitzWith_cos, tan_arccos, sqrt_div, module tactic).

NEXT (Tier 3, the big one): verified interval evaluation of the
forward model in Lean — discharges A-fp/A-libm entirely, making each
battery certificate literally a Lean theorem. Route decision pending:
girving/interval (check trig coverage at |x|≤220 rad) vs bespoke
dyadic-rational intervals. Also: concrete-instance bridge (C¹-on-box
hypothesis from certified margins → apply the abstract theorems to
the 18-D model inside Lean).

### Same day: expository companion (paper/exposition.tex, 10 pp)

User asked for a followable, Tao-clarity narrative PDF: forward
problem (undergrad arc, with %% TODO hooks for personal color) ->
inverse problem + literature -> lattice insight -> algorithm -> the
ML detour (honest: AngleNet/RemainNet located the hard sub-problem,
mathematics removed it) -> hardware feasibility (floors, eta*,
affine sheet, verdicts) -> deterministic error (3 moves, sabotage
test) -> Lean (9-theorem table) -> open problems. Reuses hero.pdf +
svd_fresh.pdf. Compiles 0 errors / 0 overfull. Uses dev-scale
numbers freely (expository doc, NOT the pre-registered submission —
main.tex placeholders untouched).

### Same day: Beamer deck (paper/slides.tex, 18 frames, 16:9)

Presentation companion to exposition.tex, same arc, one idea per
frame: machine -> two problems -> forward/undergrad (with the
phi = 2piNt + ay identity flagged early) -> hero data -> hardness
(svd figure + basin fact) -> literature -> lattice insight ->
finer-lattice trap -> 5-step algorithm + battery -> ML detour +
punchline -> hardware floors -> verdicts table -> error 3
assumptions/3 moves -> can-the-certificate-lie (zero-miss + sabotage
test) -> Lean table -> open problems -> closing arc. Boadilla/
seahorse, no nav symbols. Compiles 0/0. Copies of both PDFs at
C:\Users\josep\Desktop (LOCAL desktop — user's visible desktop is
OneDrive-redirected; user explicitly wants no OneDrive).

### Same day: deck REDESIGNED around find-vs-verify (user: "THIS is
### the stuff we need... redo that beamer ENTIRELY. simple steps")

slides.tex fully rewritten, 26 frames, spacious, four parts:
I The problem. II How recovery actually works: FIND != VERIFY as the
organizing idea (overdetermined 400-vs-18 asymmetry), the 3 finding
steps (read the spectrum like a barcode / guess the boring rest /
polish), "why reading works: a theorem not a trick", what
deterministic does NOT mean (the search) vs DOES mean (the
guarantee), the audit, path independence, the sabotage test.
III Can hardware do it: THREE VERDICTS table (cannot=proved floors /
did=proved certificate / will=EMPIRICAL GAP) + the closure plan
(Lemma A deterministic spectral bound; Lemma B Newton-Kantorovich
certified convergence ball reusing ivx — basin stops being folklore).
IV Provenance: three buckets (ours outright / ours on classical
machinery incl. Lean 9 / borrowed+cited) + the two honesty notes
(short proofs, right statements; NOT proved: global uniqueness,
arrival). Closing: "Find however you like. Verify so it cannot lie.
And say out loud which is which." Compiles 0/0. Desktop copy updated.

NOTE the user's reaction sequence that produced this: heuristics
scare -> sabotage test -> find/verify explanation -> provenance
question -> "THIS is the stuff we need." The find-vs-verify framing
and honest provenance are now the canonical way to present this
work. Arrival Lemmas A+B are now user-visible commitments.

### Same day: deck rebuilt ORGANICALLY (user: no buckets, capture the
### why-complex intuition, theorems from need, humanizer, no dashes)

slides.tex rewritten again, 23 frames, single chronological arc with
NO part dividers: machine -> forward/undergrad (phi formula planted:
"file that away") -> the inverse question -> round one (FFT peaks +
enumerations, each patch named) -> round two (ML, 30%, "initialization
was the disease") -> "Look at the pattern again" (epicycles, Ptolemy,
"there is a number system whose whole purpose is circular motion") ->
"Why we went complex" (arm = a e^{2pi i N t}; Fourier because it IS
one; spin direction = sign of frequency; the sign enumeration "in the
complex signal never existed") -> lattice from Snell mixing -> then
THEOREMS AS NEEDS: lattice theorem (selection must account for every
line), readout laws (kill the angle grids), finer-lattice lemma (the
subharmonic that fooled us) -> finder assembled -> need: error ->
find-then-verify -> audit -> guarantee scope -> sabotage -> engineer's
question -> floors -> verdict -> not-yet-theorems (Lemmas A/B, global
uniqueness) -> Lean (born from our own scare) -> one integrated
provenance frame -> closing couplet ("the machine is made of
rotations, so we handed it to the number system made of rotations").
Humanizer skill applied: ZERO em/en dashes in source (verified by
grep), no bucket headers, no bold-colon lists, plain sentences.
Compiles 0/0. THE WHY-COMPLEX INTUITION SLIDE IS NOW CANONICAL for
presenting this work. Exposition.tex still has the old bucketed
structure + dashes; same treatment pending if user wants it.

### Same day: deck stripped flat + real history + new figures

Three user passes on slides.tex: (1) plain title "Blind Parameter
Recovery for Risley Prism Systems", no subtitle, no invented "years"
timeline, Ptolemy/spirograph/closing-couplet cut; (2) ALL flourish
out ("follow the light" etc.), frame titles now literal (The forward
problem / Theorem 1, needed for selection / The algorithm / Stress
tests / Hardware limits); (3) forward slide now credits BEN CAMPBELL
(the user's professor) + Joseph with the VECTOR Snell solution
(formula on slide) and states plainly the forward problem was already
solved in the literature, "ours was an independent rediscovery" —
THIS IS THE REAL HISTORY, keep it. New figures (paper/
slide_figures.py, reproducible): epicycles.pdf (arm-chain diagram,
real case-1 speeds, Okabe-Ito colors, direct labels) and
lattice_spectrum.pdf (signed-frequency spectrum of case 1, generators
marked, three integer-combination labels, staggered to avoid
collisions). hero.pdf now appears exactly ONCE (inverse-problem
slide). 23 frames, 0 errors, 0 overfull, zero dashes.
STYLE RULE from this session (applies to all future user-facing
docs): simple, basic, no flourish, no book-style title:subtitle, no
cute frame titles, no metaphors, no invented history; verify counts
shown in formulas match claims; look at every rendered page before
delivering.

### 2026-08-15 — Algorithm section expanded to a worked walkthrough

User: "this algorithm is the project essentially" — wanted each step
explained, the proof that licenses it, and a worked example. Deck now
29 frames; the single algorithm slide became seven:

  overview (5 steps + case-1 truth) / Step 1 mask (licensed by the TIR
  proposition) / Step 2 lines (licensed by the lattice theorem) /
  Step 3 generators (licensed by the finer-lattice lemma) / Step 4
  readout (licensed by phase + amplitude propositions) / Step 5 Newton
  (licensed by identifiability, full rank 18) / the instance end to end.

All numbers are REAL, traced by the new paper/walkthrough_trace.py
(reproducible, case 1 = battery_cases()[0]):
- masking: case 1 has 0 masked; CASE 10 has 12 masked with |z| up to
  896 vs 110 typical (used as the TIR example).
- lines: 8 found; table shows N3/N2/N1 at rel amp 1.000/0.567/0.314
  plus N3-N2 and N2+N3 at 0.027/0.020 (weak lines carry ~0.01 Hz
  error, refined jointly later).
- SUBHARMONIC TRAP REPRODUCED NUMERICALLY at B=4: true trio residual
  1.09e-2 vs N1-halved 6.03e-3 (finer lattice fits BETTER, exactly as
  the lemma says), while coverage prefers the true trio 42.25 vs
  41.90. Note: at B=3 the finer lattice fits worse purely because the
  order bound truncates before it can cover — the demo needs B>=4.
- READOUT FLIP demonstrated on real data: prism 1 phase -165.10 with
  ax=-4.66 (flip), prism 2 +168.14 with ax=-5.22 (flip), prism 3
  +5.12 with ax=+10.46 (no flip); phase -/+ 180 recovers ay to a
  fraction of a degree. This is the cleanest illustration of the
  phase proposition we have; keep it.
- Newton: MSE 1.25e-23, median param err 1.6e-12, worst 6.9e-11 on
  d_W (the weak direction), route "primary".

Also fixed the framing the user objected to: step 5 is described as
Newton (Jacobian + one linear solve per step, quadratic convergence),
explicitly "not a search", after the user reacted to an earlier
"guess and check" phrasing. Time complexity discussed in chat (pencil
O(M^3) is the only cubic; Newton dominates wall clock via ~2n forward
evals per iteration; nothing enumerates).

## 2026-09-06: SECOND-ORDER certificate. Reach up ~7900x; hardware at 1e-7 SOLVED 26/26

User: "take another look ... try something fundamentally different so that
we break through this." The wall: the deterministic certificate closed only
for data error eta <= eta* ~ 1e-8 (median 1.3e-8, min 5e-11, case 22 never),
so every realistic sensor level (1e-7 and coarser, pattern amplitude ~30-100)
was INCREASE_T. The hardware map was empty exactly where hardware lives.

### Diagnosis (scratch diag_beta.py, diag_norm.py; cases 1, 20, 22)

Three stacked losses in the first-order certificate, each measured:

(L1) NORM. beta = plain max row sum of Eabs, but the box radii span
     1e-11 (speeds) .. 1e-4 (glass). The speed columns dominate every row
     sum while carrying weight r_N/r_j ~ 1e-5 in the actual fixed point.
     The right contraction measure for an anisotropic box is the
     box-weighted beta_r = max_j sum_k E_jk r_k / r_j (any positive
     diagonal weighting is admissible in the Neumann/Banach argument; the
     Lean enclosure_bound is norm-generic). Alone: eta* x44-68.
(L2) DOUBLE ABSOLUTE VALUE. Eabs = |C| |J_rad| discards the cancellation
     encoded in C J = I. True |C dJ| at box vertices vs Eabs: 550x (case
     1), 3900x (case 20), 1e4x (case 22). This is the big one.
(L3) INTERVAL-AD DEPENDENCY. J_rad vs true |dJ|: median 8-9x, max 400x.

Case 20 at eta = 1e-7: beta_plain 3970 -> beta_r(Eabs) 22.7 (L1 fixed) ->
beta_r(|C| true|dJ|) 1.89 (L3 removed) -> beta_r(true C dJ) 0.007 (all
removed). The genuine Kantorovich limit of the box method (true weighted
contraction = 1) is eta ~ 1e-5 (case 20) .. 7e-4 (case 1). Everything
between 5e-11 and that was bookkeeping, not mathematics.

Structural fact, confirmed numerically: F is EXACTLY affine in d_W, gap,
bm_px, bm_py (second differences 1e-14 vs 0.1-80 in angles and speeds).
Ray directions depend only on angles and indices; positions are affine in
the distances. So the Hessian blocks among the four geometry parameters
vanish identically.

### The fix: second-order Taylor certificate

Files: risley_lattice/ivx2.py (IvDual2: value + gradient + packed symmetric
Hessian intervals, same primitives and rigor model as ivx, identities in
the docstring), fmodel.forward_iv2 (Hessian enclosure over a box; the
tracer takes a class hook), risley_lattice/certify_det2.py (the
certificate), experiments/certify_det2_battery.py.

Validation of the AD: bitwise parity of values and gradients with
forward_iv; Hessian vs finite differences 1e-7 relative; containment of
exact Hessians at 40 random interior points of a box; enclosure pessimism
median 14x (the L3 of the Hessian).

Mathematics (docstring of certify_det2 + paper Prop. validated):
  MVT on each gradient component along [0, x] in scaled coordinates:
    d_k f_i(x) = d_k f_i(0) + sum_l x_l H0_ikl + rho_ik(x),
    |rho_ik| <= sum_l |x_l| Hrad_ikl,   H0 = fp midpoint Hessian (any
    reference matrix works), Hrad = box enclosure radius (same midpoint,
    since midpoint-radius arithmetic has radius-independent midpoints).
  ENCLOSURE: |delta*| <= Phi(|delta*|),
    Phi(d) = nu + |E0| d + 1/2 (|G|[d] + |C| Hrad[d]) d,   G = C H0 EXACT
    (A-blas rounding only; this is where the cancellation is kept).
    Closure test Phi(r) <= r; iterate downward, every iterate valid
    (Phi monotone). The 1/2 is int_0^1 s ds.
  INJECTIVITY: K(r) = |E0| + |G|[r] + |C| Hrad[r]; beta_r(K) < 1 forces
    Delta = 0 (weighted-norm kernel argument).
  Search (heuristic, rigor in the final triple): eps-inflation seeded by
    nu, grown by the image Phi(r), shrunk when the nonlinear excess
    (Phi(r) - nu)/r >= 1 (beyond the fold); eta* by geometric bisection,
    each probe seeded by the previous closing box rescaled by eta ratio.
Cost: ~0.8 s per box evaluation (numpy per-op overhead; packing the
Hessian did not change wall time), ~3 min per case for the eta* bisection.

### Battery (results/_certify_det2.log, certify_det2.jsonl; 5306 s)

26/30 solved (same set), 26/26 CERTIFIED (case 22 now closes),
coverage 26/26 at 2x floor AND 26/26 at eta*_2 (zero misses).
eta*_2: min 1.3e-6 (case 20), median 1.2e-4, max 4.1e-4.
Gain eta*_2/eta*_1: min 3843x, median 7866x, max 25397x.
Bounds at 2x floor unchanged vs first order (noise-term dominated there).
Hardware (SOLVED = sensor error <= eta*_2):
  1e-11: 26/26 (first order 25)   1e-9: 26/26 (21)   1e-7: 26/26 (0)
  1e-5: 24/26 (0)   1e-4: 15/26   1e-3: 0/26.
The two 1e-5 holdouts are cases 20 and 22 (eta*_2 1.3e-6, 2.9e-6).

### What limits the reach now

The O(r^2) remainder |C| Hrad[d] d still pays the double absolute value
and the ~14x interval-AD pessimism of Hrad. Achieved eta*_2 sits 3-10x
below the true Kantorovich limit measured by vertex sampling. Next lever
if needed: third-order remainder (exact C T0 products, interval fourth
derivatives) or centered/affine forms for Hrad. The inf-norm worst-case
data model (rowsum|C| in nu) is inherent to A-data and not a loss.

### Lean: 12 theorems, zero sorry (build green, 8710 jobs)

Added to Certificate.lean: monotone_fixedpoint_invariant and
monotone_iterate_bound (the nonlinear monotone map iteration of
certify_det2; every iterate valid), weighted_contraction_zero (the
beta_r < 1 exclusion argument in the box-adapted norm; proof by the
maximal ratio |Delta_k|/r_k). formal/README.md table updated.

### Paper (main.tex, compiles 0 errors)

Prop. validated rewritten in the second-order form (eq:Kmat, eq:bounds
with Phi, eq:gradmvt, weighted beta_r, monotone iteration from r), new
"Why second order" paragraph (the cancellation argument + the affine
geometry structure), eta* paragraph ("certifiable-noise threshold",
second-order argument, seeded bisection), Prop. sheet now with
K_half = E0 + 1/2(|G|[r] + |C|Hrad[r]) on the closing box (linear map
dominates Phi on [0, r], invariant for every eps <= eta*), Cor. wording.
Placeholders untouched (campaign design unchanged); A-arith now covers
second-order interval differentiation.

### Also this session

experiments/sabotage_det2.py: regression test of the dichotomy with
corrupted answers (outcomes: refused / covered / truth outside box;
required violations 0). experiments/hardware_map.py moved to the
second-order sheet (formula_sheet2, hardware_verdict(order=2, sheet=...),
grid to 1e-3; first-order results preserved as hardware_map.jsonl).
Results of both appended below when finished.

### Same session, later: A9 + interim sabotage count

experiments/assumptions.py gained A9 (second-order interval AD): midpoint
parity with forward_iv True, Hessian vs finite differences max rel
6.1e-8, containment violations 0/40 interior points. PASS (A1-A8
unchanged: all PASS, A7 basin 29/28/26/24 of 30 at x0.5/1/2/4).
sabotage_det2 (cases 1, 20, 22; 135 corrupted answers): interim 45
trials, outcomes refused / covered only, 0 violations. The second-order
hardware map (hardware_map.py) runs at ~30 min per case because failing
bisection probes spin the full 40-iteration search; the SOLVED counts
are already fixed by eta*_2 (certificate battery); left running, log in
results/_hardware_map2.log. Search speed-up (bisection on the box scale
along the current shape instead of grow/shrink alternation) is the
obvious next engineering step; it changes no rigor.

### Sabotage test FINAL (results/_sabotage_det2.log, 2176 s)

135 corrupted answers on cases 1, 20, 22 (N1, ax1, ay1, ng1, d_W, gap,
bm_px and random directions; scales 1e-7..1e-3 of range): refused 42,
closed with truth inside the box and covered 93, truth outside box 0,
VIOLATIONS 0 (required 0). The dichotomy holds on every trial.
Exposition remark updated with these numbers.

### Hardware map FINAL, second order (results/_hardware_map2.log, hardware_map2.jsonl, 8130 s)

26 formula sheets (K_half sheets on the closing box), coverage 26/26
(zero misses), beta_half 0.40-0.44. Classification (tol 1e-3 of range):
  1e-11, 1e-9, 1e-7: 26 solved / 0 impossible / 0 increase-T
  1e-5: 24 / 0 / 2      1e-4: 15 / 0 / 11      1e-3: 0 / 3 IMPOSSIBLE / 23
The three IMPOSSIBLE verdicts at 1e-3 are certified by the minimax
floors (two-point witnesses), the first non-empty IMPOSSIBLE column on
the battery. Speed floors ~1e-12 Hz at eps 1e-9, witness deviations
~1e-9 as before. Slides table and exposition updated accordingly.

## 2026-09-07: Separable completion, two new recoveries, explicit clock ambiguity

Implemented an optional completion in risley_lattice/separable.py using the
affine geometry structure recorded on 2026-09-06. Eliminate d_W, gap, bm_px,
bm_py by bounded linear least squares at every nonlinear step (18 -> 14
nonlinear coordinates). Propagate affine ray-position coefficients; use the
smooth direction-ratio algebra from fmodel instead of the arccos/tan round
trip for initialization. A batched complex-step derivative supplies the FULL
variable-projection Jacobian, including its residual-dependent term, computed
with QR. Scale nonlinear coordinates by their box ranges. Final polishing
and acceptance still use the canonical numpy vec2pat. No certificate or
canonical forward-model code changed.

solve18(pattern) remains the original default. completion="separable" selects
the raw new engine; completion="hybrid" tries that engine, then runs the
original solver if its canonical MSE does not pass, retaining the smaller
residual. The raw reduced stage is capped at 250 evaluations and final polish
at 100 per call (smaller caller caps apply); numerical exceptions fall back
to the original completion budget. The comparison measures this combined
implementation, not an ablation of individual changes.

Full paired canonical seed-2026 battery, 200 samples / 10 seconds, same
parameter criterion as solve18_battery (max absolute error < 1e-3), pinned
BLAS, NumPy 2.3.5 / SciPy 1.18.1:
- Original: 26/30. Raw separable: 27/30.
- New recoveries: case 10, max error 1.435e-11, MSE 2.132e-24, zero rung;
  case 11, max error 2.858e-9, MSE 1.069e-22, recal+flip1.
- Raw regression: case 30 (original zero rung succeeds; raw separable fails).
- Raw remaining failures: 4, 18, 30. Original failures: 4, 10, 11, 18.
- On the 25 cases both engines solve: median paired speed ratio 1.865x;
  median original time 1.530 s, median raw separable time 0.788 s.
- Raw recovered-case median max-parameter error 1.945e-11; worst 3.673e-7
  (case 22, still certified and covered). Precision was not uniformly improved.
- Total solver time: original 788.95 s, raw 91.25 s. This aggregate includes
  shorter failure budgets; use shared-success timings for the main speed claim.
- 26/27 raw recoveries pass second-order deterministic certification at
  max(2*data_floor, 1e-12), and all 26 bounds cover the actual errors.
  Case 10 is the exception: physical TIR margin -0.08908 on retained data.
  Recovering core's TIR-clipped synthetic pattern is NOT a physical no-TIR
  certificate. The benchmark records refusal rather than claiming a guarantee.

Hybrid integration runs on cases 30, 10, 11, 1 all recover:
- case 30 via original fallback, max error 1.478e-12, certified and covered;
- cases 11 and 1 via separable, certified and covered;
- case 10 via separable, the same documented physical certificate refusal.
The paired component runs plus these integration checks support the 28/30
union (all original recoveries plus 10 and 11), with 27 certified cases.
A separate full 30-case hybrid sweep was NOT run. The unresolved union is
4 and 18. From measured component times, the hybrid median shared-success
speed ratio is 1.833x and total 476.12 s; these are DERIVED, not fresh hybrid
timings. Do not label them as a separate measured hybrid battery.

CORRECTION TO EARLIER FAILURE CLAIMS: the old blanket description of the
T=10 failures as information-theoretic limitations is not justified for
these noiseless data. Case 11 now recovers at the SAME observation length
with a deterministic certificate. Case 10's clipped numerical inverse also
recovers. These experiments do not establish identifiability at arbitrary
sensor noise, and continued solver failure on cases 4/18 is not an
impossibility proof.

Validation:
- affine/smooth versus canonical forward parity on all 30 cases: worst
  max-coordinate discrepancy / max(1, max|pattern|) = 1.443e-11;
- reduced Jacobian versus independent centered directional differences:
  worst relative error 8.535e-10 for interior cases, 2.183e-9 with active
  geometry bounds (two distinct active sets);
- forward parity at 320 samples / 16 seconds and 173 / 8.65 also passes;
- no new noise or hardware campaign performed.

Clock-free API: risley_lattice/clockfree.py, solve18_clockfree(pattern).
Requires ORDERED UNIFORM samples; returns cycles/sample, signed speed ratios,
and the other fifteen parameters in the original conventions. Phases are
relative to the first sample and still obey the existing parameter box.
The original spectral priors remain in virtual sample-index units (nominal
fundamental cutoff 0.005 cycles/sample, box maximum 0.175).
This is an explicit time-unit reparameterization of the existing problem,
NOT recovery of ordering from an image.

Constructive non-identifiability witness: phase depends on N_i*dt. Halving
all speeds and doubling dt leaves the canonical data bitwise identical
(tested on case 1, 200 samples / 10 vs 20 seconds). The one clock-free
recovery maps to both physical vectors at the appropriate dt with error
1.259e-10. Absolute Hz needs one real clock calibration; the API requires
a finite positive dt for conversion. Raw unordered photograph inversion
remains unimplemented and unresolved here.

Reproduction:
  python experiments/separable_battery.py --certify
  python experiments/separable_battery.py --hybrid --cases 30,10,11,1 --certify --output experiments/results/hybrid_checks.jsonl
  python experiments/separable_battery.py --checks-only
Results: experiments/results/separable_battery.jsonl (60 component records),
hybrid_checks.jsonl (4 integration records), separable_summary.json (measured
and derived quantities distinguished). Usage/math: experiments/SEPARABLE_RECOVERY.md.
SciPy installed locally into ignored .codex-deps for the Windows runtime.

## 2026-09-07: Static etch work separated; lattice bounds resume

The user requested the etch research remain a separate branch. It is preserved
as research/etch-recovery, commit 66200d1bd3181aa0ce656c600518b228d733c9b7,
with a standalone checkout at ../Wedge-etch-recovery. That branch retains the
complete etch research log below the original marker, code, experiments and
results, plus its shared current lattice/interval dependencies. The etch-only
files have been removed from this main checkout after verifying the committed
copy. Existing pending lattice, formal and paper changes were preserved.

Main focus resumes deterministic frequency-lattice recovery bounds, including
per-parameter precision, certifiable noise reach, exclusion regions and
constructive hardware lower bounds. Statistical heuristics remain excluded
from guarantees. The last full baseline has 26/30 solved, 26/26 certified and
covered, eta*_2 median about 1.2e-4; the separable/hybrid work above is distinct.

### 2026-09-07: L1 inverse proposals tighten all 18 deterministic bounds

Added risley_lattice/minimax.py. For the scaled Jacobian J, each row of C
minimizes ||c||_1 subject to c J = e_j^T (QR-conditioned HiGHS proposal).
The certificate needs any approximate C, not an exact LP solution: all
finite residuals enter interval E0 = |C J - I| and the second-order
Hessian products. LP tolerances supply no guarantee. Optional
preconditioner="minimax" in certify_det2/certify_eta_max2 and hardware
sheets; the pseudoinverse and default lattice recovery remain unchanged.

CORRECTION TO EARLIER TIGHTNESS LANGUAGE: rowsum|C| is inherent for fixed C
under componentwise bounded data noise. Choosing C as the L2 pseudoinverse
is avoidable and was an additional source of looseness.

Paired certificates reuse the stored standard estimates (26/30) from
separable_battery.jsonl, with noiseless synthetic observations and specified
hard uncertainty bounds. This is not a new noisy-recovery campaign.
At eta=1e-7: both versions close 26/26; truth is in each closing box and
covered in all 26 new certificates. Across 468 coordinates, old/new bound
ratio min 1.3506, median 1.72349, max 2.6902, regressions zero.
At historical numerical data floors: median gain 1.68818, no regressions.
At historical eta*_2: fresh minimax closes 26/26, fresh unseeded pinv 17/26;
shared-case median gain 2.19602. These boundary search failures do not
invalidate the historical warm-seeded certificates or prove impossibility.
No new maximal-noise search was run, so the old eta* values are unchanged.

Mask correction: the first new campaign omitted deglitch_mask; case 30 was
refused on a physical margin. It was rerun with the standard 196 retained
samples and merged into minimax_bounds.json. The other 25 evaluated cases
retain all 200 samples, so their results were unaffected. Four original
unsolved cases were skipped, not recategorized as certificate failures.

LP duals also propose coordinate-specific alternatives. Cases 1, 20, 22
at eta=1e-7: all 18 pairs per case have certified model separation <=2eta;
BOTH endpoints fit the actual observed pattern within eta; all 54 pairs
are inside the corresponding certificate boxes. Half their parameter
separation is a constructive uncertainty floor at those observations.
Upper/floor ratios: case 1 median 1.00826, worst 1.00873; case 20 median
1.03367, worst 1.11534; case 22 median 1.01731, worst 1.03416. Thus all 18
case-1 bounds are within 0.9% of the demonstrated unavoidable ambiguity.
Eight bisection steps find witnesses; no claim of universally optimal LP
solutions, witness spans, or tightness on the other 23 cases.

### Hardware accuracy correction and fixed-ceiling sheets

Found hardware_verdict's SOLVED branch checked certificate availability
but omitted the requested parameter tolerance. It now requires EVERY
outward-rounded scaled bound <= tol. Missing accuracy is unresolved
(legacy INCREASE_T label); no guaranteed observation time is fabricated.

CORRECTION to the hardware-map FINAL counts above, tol=1e-3 of each range:
saved pseudoinverse sheets give 16/0/10 at eta=1e-5, and 2/0/24 at 1e-4
(SOLVED/IMPOSSIBLE/unresolved), replacing 24/0/2 and 15/0/11. The old
24 and 15 count certificate availability, not requested accuracy. The
26/0/0 low-noise rows and 0/3/23 at 1e-3 are unchanged by this audit.
hardware_accuracy_audit.json derives the correction from saved coefficients;
raw historical sheets remain untouched. Exposition and slides corrected,
rebuilt without warnings, relevant pages rendered and visually checked.

Added formula_sheet2_at: independently certify a supplied noise ceiling and
derive the affine accuracy sheet, avoiding a fresh ceiling search. Fresh
minimax sheets confirm all 26 historical eta*_2 values. At tol=1e-3 they
meet accuracy on 26 cases at eta=1e-7, 17 at 1e-5, 2 at 1e-4. Case 3 is
newly certified at the 1e-5 accuracy target. Uniform sheets may be looser
than certificates evaluated specifically at a smaller noise level.

Reproduction and limits: experiments/BOUND_TIGHTENING.md.
New experiments/minimax_bounds.py writes full paired certificates,
actual-data witness pairs, and fixed-ceiling sheets; --summarize derives
minimax_summary.json. experiments/minimax_validation.py passes analytic
LP, accuracy verdict, and invalid noise-input checks. All new full physical
certificates/witnesses are interval checked under the existing arithmetic
assumptions. No changes to canonical physics, solver default, or Lean source.
Higher confirmed noise ceilings, global exclusion, and full-battery
coordinate lower-bound comparisons remain open; local coverage is not a
global identifiability theorem.

## 2026-09-15: Lisa's large campaign found in Gmail; preliminary report audit

Correcting the earlier email-status assessment: the Benedict Outlook thread
did not contain the latest delivery. Lisa's September 10 Gmail message in
the subject "Claude" links risley_results.tar.gz, summary.txt, analysis.txt,
and merge/finalize/aggregate/analyze scripts. Her September 8 note says the
cloud app was stopped at 669,497 runs with x18; the later analysis reports
669,574 retained cases. The 77-row difference is not yet reconciled.

The unchanged downloaded reports are saved under
experiments/results/lisa_2026_09_10/. They REPORT, not independently recounted:
669,574 noiseless cases; cumulative recovery 83.47%, 94.25%, 96.51%, 97.25%
by T=10,20,40,80; 651,172 recovered cases with x18 and 18,402 unresolved.
These last two counts sum correctly. This is substantial numerical-recovery
evidence, not a 669k deterministic physical-certification campaign, and the
exact producing solver/runner revision remains unverified.

Read the four analysis/preparation scripts via their Drive previews; none
was executed. The 221 MB raw archive download was blocked by Codex's browser;
the user was asked to save it locally. Raw-data audit remains pending at
this checkpoint. No new solver experiment or cloud run was launched.

Material interpretation issues: A1 labels T_req <= T_solved as sufficient,
which does not test recovery at T_req, and reverses the interpretation of
T_solved/T_req < 1. Its 308 unresolved cases with T_req <= 80 are statistical
prescription failures, not violations of certify_det2. Four noiseless
duration points do not prove a universal T^-3/2 recovery law or a permanent
2.75% impossibility floor. A6's printed interpretation about basin-margin
ratios near one contradicts reported ratios 0.07, 0.13, 0.20. Failure labels
remain diagnostics, not proofs of their stated causes.

Provenance issues to audit: merge key omits seed/source identity; tag priority
selects the first row; finalization drops rows missing min_cycles; parsers
skip malformed JSON. Whether these affected the reported rates is unknown.
Preserve null-x18 failures and all excluded-row counts. Do not run the
supplied merge scripts over current results: they move existing files.

Detailed source links, evidence limits, and checklist:
experiments/results/lisa_2026_09_10/AUDIT.md. Updated
paper/TIME_SERIES_RESEARCH_PLAN.md: audit/reuse this campaign first, diagnose
a stratified subset, then integrate the current hybrid/deterministic method
and perform a modest frozen evaluation. The earlier 30-case integration
gap remains real; repeating the large noiseless campaign is not the priority.

### Same session: raw archive available; independent recount completed

The archive subsequently appeared at C:/Users/josep/Downloads/risley_results.tar.gz.
Read it directly without extracting or running any member. Added the independent
read-only auditor experiments/audit_lisa_archive.py. Archive SHA256:
beb9e1e22cc0c57aacb52e4ed77566658094440e34a95793a506b46076ea8914.

Raw inventory: 24,024 JSONL files, 713,572 rows, one seed (424242), adaptive
and snr=null throughout. Deduplication under the supplied tag priority removes
43,983 rows, leaving 669,589 unique identities; 15 lack required diagnostics,
leaving exactly 669,574 retained cases. No malformed/missing-identity rows or
cross-seed key collisions were encountered. There are 204 duplicate comparisons
whose T_solved values differ (comparison count, not necessarily unique cases).
Exact source/environment provenance and this variability still need review.

Independently regenerated truth with current model.case_at and checked every
finite saved vector: 842,126 retained rungs, 7,945 null-x18 rungs, 834,181 finite
vectors; max discrepancy between stored and recomputed parameter error = 0.
No vector-shape, ladder, T_solved, or successful-final-rung inconsistency found.
Recomputed cumulative recoveries:
T<=10: 558,919 / 669,574 = 83.4738206681%
T<=20: 631,053 / 669,574 = 94.2469390986%
T<=40: 646,198 / 669,574 = 96.5088250141%
T<=80: 651,172 / 669,574 = 97.2516854000%.
All 18,402 unresolved cases and the delivered failure-class counts match.
Unresolved with numeric T_req<=80 = 308, >80 = 9,594, no numeric T_req = 8,500.

The selected records contain no singular-value spectra, actual sample masks
(only counts), or second-order deterministic boxes/bounds. No forward solve or
physical certificate was rerun. This validates the saved numerical-recovery
statistics, not physical validity, global identifiability, noisy-data accuracy,
or the newer hybrid/minimax pipeline. Case IDs are noncontiguous through 995899;
unfinished-task selection needs assessment before generalizing population rates.

Compact audit: experiments/results/lisa_2026_09_10/raw_audit.json.
All unresolved cases with full trails were saved outside Dropbox at
C:/Users/josep/.codex/visualizations/2026/09/15/01a0a318-de6d-7990-bd5a-b7352ee3b92f/lisa-raw-audit/unresolved_cases.jsonl.
Updated the evidence audit and paper plan to reflect completed recount and the
remaining provenance/integration work. Original reports and archive preserved.

### Same session: failure-driven algorithm improvement and deterministic limits

User authorized working through the downloaded failures to establish hardware
restrictions or improve the algorithm. Added a complete unresolved-tail inventory:
18,402 cases, 1,139 null final vectors, only 371 with all signed speeds within
1e-5, and 136 WRONG vectors passing the old MSE<1e-12 rule. That residual rule
is not an accuracy certificate. Fixed selections use SHA256(seed,case), two per
exclusive label combination (31 cases); no selection by new recovery outcome.

Saved-estimate separable refinement at T=10 recovered only one of those 31
(476502), whose full physical branch refuses. An FFT CLEAN ablation exposed weak
lines missed by the pencil. Added optional frontend='fft': twelve line slots,
near-unregularized B=1 subtraction, unchanged regularized higher-order lattice
selection. The existing solve18 defaults remain standard/pencil.

Added risley_lattice/windowed.py::solve18_windowed. Phase error 2*pi*dN*t makes
long-record completion sensitive to initialization. Frequencies use all T=80
samples; physical completion first uses the genuine first T=10 prefix, followed
by full-record evaluation/polish. Both candidate selection and stopping are
data-only. This is an optional bounded candidate strategy, not global convergence
or an accuracy guarantee. Full-record masked MSE is returned explicitly.

On 12 preselected archived failures, identical windowed budgets recover 2 with
pencil, 4 with FFT; the data-only retry policy has union 5. Fresh production-helper
integration reruns confirm all five: 37551 (error3.98e-11),199242(2.90e-10),
274881(3.71e-9),515381(7.91e-10),476502(3.21e-12). Only FIRST THREE pass interval
physical checks on ALL 1600 observations. Cases 515381 and 476502 fail unmasked
TIR-margin checks. All five pass second-order conditional local bounds on the
RETAINED FIRST-200-SAMPLE subset at eta=1e-8, with truth in their local boxes and
inside every bound. This does not certify omitted samples or the whole 80s
physical branch. We did not pass nonstandard data to the standard-grid certifier.

The 'rank-deficient' archived case199242 recovers: the label was not physical
nonidentifiability. Two direct full-record fitting pilots were stopped due cost:
the first had no completed row; the capped retry finished25034 and29640 in
about63/66s with neither recovered, then was stopped during37551.
Raw-basis/joint-amplitude trials on four
cases added no recovery; extra polish reduced case25034 to .00363 (still fails
.001), and55086 remained unresolved. All outcomes preserved; no invented success.
Paired development trials are not held-out rates; timing processes overlapped.

Added ambiguity.py::certify_pair for explicit-grid, interval-checked two-system
witnesses. ALL1600samples, no mask, on12 hash-selected old MSE false accepts:
12 endpoint pairs physically interval-checked;11 force some native-coordinate
worst-case error >1e-3 when eta=1e-5 model position units. BOTH endpoints fit
the actual observation at that eta. No statistical heuristic in this argument.
The remaining pair is compatible but its half-separation is too small to prove
failure of that tolerance. A negative result is not classified impossible.

Concrete hardware ambiguity: case257705(first wedge .012954deg), both systems
fit the actual80s observation within4.577e-7, but glass-index uncertainty floor
is .032146. Case470640 has glass-index floor .001403 at allowance<1.437e-7.
These are conditional nonzero-noise restrictions, not noiseless impossibility,
an optimal sensor threshold, a universal wedge cutoff, or a measured hardware
specification. All11 tolerance failures involve glass index. Independent glass
calibration/narrower priors/additional informative data are appropriate next
tests; none is yet proved sufficient. Exactly zero wedge makes its speed and
phase vanish from the model, an actual structural restriction.

Validation: analytic flat-plate two-point example, strict tolerance comparison,
actual/common-observation distinction, input/grid guards; five production-helper
integration checks with subset certificates; unchanged default hybrid successes
on seed2026 cases1,11,30. All passed, as did git diff --check. No cloud runs,
messages, commits, pushes, physical-model edits, or Lean changes.

Full mathematical argument, qualification of masks/noise/units, reproduction,
unsuccessful attempts and paper priorities: experiments/LISA_FAILURE_FOLLOWUP.md.
Saved records and source/runtime hashes: experiments/results/lisa_2026_09_10/.

### Full 18,402-case classification and a corrected identifiability distinction

User requested the classification for the whole tail or stronger theory. Completed
all18,402 cases, no omissions, duplicates or runtime errors, in527.83s using4
local processes. All1600samples/T80, no masking. Dataset/source hashes frozen.
Proved physical-branch violations:2,643. Certified admissible:15,759. EVERY
admissible case has a verified two-endpoint witness and an explicit noise
threshold at which its half-separation exceeds the historical native .001 target.
These are sufficient ambiguity thresholds, NOT optimal noise boundaries.

At eta=1e-5 model position units and native tolerance .001:2,643 physical
exclusions +6,701 certified accuracy impossibilities +9,058 unresolved=18,402.
Original campaign is noiseless: this is NOT a proof of original noiseless
failure. Counts of admissible cases with native ambiguity at eta1e-9,1e-7,1e-5,
1e-3 are0,200,6701,15365. Same stored pairs against .001*RG tolerances give
0,26,1498,2090; directions were not separately optimized for that scaled target.
Witness noise thresholds min1.774e-9,median1.461e-5,max.04435. No universal
hardware wedge cutoff or sensor specification is inferred. Witness proposals
often targetd_W(9486cases); the earlier12glass-heavy examples were a special
false-accept selection, not a description of the entire tail.

New fmodel.forward_point uses the identical interval trace with zero AD seeds,
omitting the unused Jacobian. Against forward_iv: exact value/radius parity on
56 physically passing cases and4 matching refusals across the30-case battery
atT10/T80. Physical exclusion now requires a NEGATIVE UPPER bound on a signed
condition; a failed lower bound alone is inconclusive. _check attaches the
upper bound/sample index/proved_violation metadata. Corrected misleading
MarginError prose. Existing physics/value arithmetic unchanged.

Weak-direction proposals solve a linearized cancellation problem with fixed
coordinate displacement using an SVD-based inverse Gram matrix. The prefix
Jacobian and its numerical rank are proposals only. Finite endpoints must pass
box, physical and actual-observation interval checks on all80s; the triangle
inequality supplies the error floor. No Fisher/statistical guarantee is used.
Truth proposes simulation witnesses, never initializes blind recovery.

Stronger theory:3,048 failure cases have exactly equal speed magnitudes, driven
by the generator's minimum-speed clamp creating atoms at+/-0.15. All6predeclared
collision cases29640,55086,128325,313904,73884,357166 nevertheless have CERTIFIED
LOCAL INJECTIVITY of the full18D map in nonzero boxes around truth atT10.
Contraction factors .0575--.3684. This is an oracle-centered local theorem audit,
not global uniqueness or blind recovery. Distinct speed magnitudes are NOT
necessary for local full-model identifiability; they are an assumption of the
earlier independent-generator spectral method. Do not classify equal speeds as
a universal hardware impossibility based on that initializer assumption.

New optional collision.py::solve18_collision proposes repeated signed observed
generators and bounded amplitude splits, then fits nonlinear physics. Blind
recovery on the6cases succeeds on29640(error7.137e-11) and128325(2.570e-12);
four remain unresolved. Fresh reruns confirm both; unmasked80s physical checks
pass; retained-prefix T10/eta1e-8 local certificates cover truth in their boxes.
Together with earlier windowed work:7targeted numerical rescues,5full-sample
physical. This is NOT a full18k solver rerun. Frozen classification knows5prior
rescues; derivedfindings overlays2new ones. Recovery and finite-noise ambiguity
can coexist:274881 and29640 are examples. Default solver unchanged.

Validation: all input identities/hashes accounted, zero errors;48additional
full18-seed AD replays(24witness cases,24physical violations) agree with faster
point path. Analytic ambiguity and strict-tolerance tests pass, as do new
collision recovery/physical/certificate/input checks. All results checkpointed;
no cloud, external messages, push, commit or Lean edits.

Full report:experiments/LISA_FULL_CLASSIFICATION.md. Per-case proofs, manifest,
audit and derivedfindings:experiments/results/lisa_2026_09_10/classification_all/.
OutputSHA256:a64f0f61281d4e0fe95d2d40c63d895a2dfd83c28eebd30b7e14f71790c4bcfe.
Next: collision/near-relation initializer evaluation on a frozen cohort;
application-specific hardware/noise targets; remaining9,058 unresolved at the
reference settings. Physically audit successful campaign cases before claiming
recovery rates for the physically admissible subpopulation.

## September 16 — user requires continued work on the unresolved tail

Work in progress, recorded to preserve the research decisions. The criterion
remains original noiseless native max error <.001. No noise/tolerance adjustment
is used to turn open numerical cases into impossibility claims.

New relations.py preserves a two-generator signed frequency hypothesis during
initial fitting and enumerates both physical orders for opposite-sign equal
speeds. Earlier collision.py omitted one order and released speeds prematurely.
Same six exact-collision cases improve from2/6 to5/6: new rescues313904(6.31e-13),
55086(4.31e-9),73884(7.04e-8);357166 remains open after the first continuation
and rest-grid attempts. No true parameters enter these initializers.

New frequency_candidates.py screens physical frequency triples with speeds
fixed, then releases them. Frozen stratified30-case development cohort yields
18rescues; clusters.py closes9of the remaining12 through repeated-generator
fitting followed by splitting on a prefix: combined27/30. The final3have weak
fundamentals present in the extracted line list but discarded by amplitude/
screen ranking; a broader candidate test is running. These are development
results, not held-out/population recovery rates.

The339admissible archived near-fits(MSE<1e-8) are undergoing paired saved-estimate
polish: current120-budget full-record route versus1000-budget prefix route with
canonical full verification. Three prefix pilots recover367816,693471,13901.
Controlled700-iteration ablation for367816 shows ORIGINALgradient tolerance
also recovers in509evaluations; do not attribute the improvement to disabling
gradient stopping. Hand-written Gauss-Newton line search failed on all3pilots.

Whole-admissible-population local-identifiability audit running on15,759truth
centers. Validated||C J-I||<1 proves rank, then explicit interval boxes prove
local injectivity. No inference of global uniqueness or blind recovery. One
very weak-wedge case157014 needs a weighted norm for an explicit anisotropic
box: beta_weighted=.35664, unweighted>101on that samebox. Exactly zero wedge
negative control is correctly refused. Smooth complex-step Jacobian agrees
with interval midpoint to4.4e-16relative in the control test.

Current methods, evidence paths, limitations: experiments/CLOSURE_WORK.md.
Default solver and frozen September15classification remain unchanged.

### September 16 completed theorem audit, cohort closure and full replay launch

Completed all15,759 admissible cases in1035.65s with2workers. Every case has an
explicit certified locally injective box:15,757ordinary max-norm boxes plus
weighted boxes for157014(beta=.3566373) and116873(.0701030). The latter even
failed the unweighted point test(1.166879): certificate failure was not rank
loss. Full accounting and18additional replays(16hash-selected +2weighted) pass.
No global uniqueness or blind-recovery conclusion follows from these local boxes.

Near-fit cohort339: original120-budgetfull fit124recoveries; extended1000-budget
prefix291. Glass grid adds36/48; exact u=(n-1)tan(ax) reparameterization removes
curved glass/wedge optimization valleys in further weak cases. Chain-rule test
relativeerror3.92e-8. Residual-frequency proposals repair missing weak speeds.
Three final same-speed cases13936,511874,99565 are solved by small amplitude
shares(.01/.99,.1/.9) rather than the earlier moderate splits. ALL339now recover.
The gain transformation preserves the exact forward model but uses a restricted
safe rectangle in gain/index coordinates; no global domain-completeness claim.

The30stratifiedcases close30/30after preserving all12observedlines and20screened
candidates solves260303,145328,649657. All6exactcollisiondevelopmentcases close;
357166 needs independent glass starts and achieves4.36e-8. Integrated frozen
ladder passes7/7developmentchecks and16/16new hash-selected cases excluded from
the339+30+6developmentcohorts. No tuning on the fresh16 during that run.

Independent summary recomputes every stored error and picks candidates by FULL
unmasked canonicalMSE without truth. Exactly391unique cases,391recovered,
391pass all1600physical sample checks. This is375adaptive developmentcases+
16freshpilotcases, not a population recovery percentage. Full errorcertificate
checks cover357166,649657,674223;723421 is numerically recovered but certificate
refuses the tested4.87e-9synthetic error allowance. Refusal preserved explicitly.

User's requirement to keep resolving the population: launched a6worker replay
of the remaining15,368admissiblecases viafull_recovery_campaign.py. A complete
source snapshot is imported by workers; original input and prior391evidence
are hashed. Checkpoints and atomic progress summary under
experiments/results/lisa_2026_09_10/full_replay_v1/. Initial rows recovered and
canonical residual replays pass. Full run NOT complete; never count queued cases
as solved. Any new numerical failures require further diagnosis, not relabeling.

Reports:experiments/CLOSURE_WORK.md; closure_summary.json and .cases.jsonl.
Local theorem outputSHA256:c78dbc0ed0654b2ded5a22988f3c230522819383d55acc55f8ad740d0229a793.
AGENTS.md/CLAUDE.md corrected the old universal distinct-speed identifiability
claim. Defaultsolver, physicsvaluearithmetic, Lean and external services unchanged.

### September 16 curved finite-noise witness for case 723421

While the frozen population replay continues, followed up its separate refused
error certificate. First wedge3.40225e-6deg; exact gain-coordinate nonlinear
endpoint fits fix ng1 at truth+/-.0011. Full1600sample interval endpoint checks
bound distance from ACTUAL observation by2.3147779513e-9 and2.5363433773e-9.
Thus eta2.54e-9 admits both explanations, with ng1 errorfloor.0011000000000000994
strictly above.001. This proves a noise-dependent accuracy obstruction at the
previous refused4.874e-9 allowance; no claim that any local/looser enclosure
must fail, that this threshold is optimal, or that noiseless recovery is
impossible. Extra priors may exclude a witness. Original numerical recovery
is unchanged. Truth-assisted witness generation is separate from blind solving.

Full interval-AD replay exactly reproduces the two endpoint bounds; lowereta
1e-10control correctly declines the impossibility assertion. Saved nonlinear
endpoints/sourcehashes:curved_ambiguity_723421.json and.audit.json. This improves
the prior sufficient actual-observation threshold4.35e-7 by following the curved
valley instead of taking only a linear weak-direction displacement. Reports
and canvas updated; no frozen classification or running snapshot was changed.

### September 16 first population replay failure repaired

Case271161 is the first new frozen-policy failure, error4.568,mse2.5557e-6.
Finalresidual repair found the correct near-equal speeds but no long local fit
ran afterwards. New observation-only near-relation refinement temporarily ties
the nearest signed-speed pair (within1/T), runs3000-budgetprefixfit, then releases
it. Result2.5441e-10 parametererror; helper replay and independently frozen repair
pass reproduce the result with fullphysicalchecks. Unrestricted extendedfit plus
glassgrid also recovers4.8954e-10, so no claim relation tying is necessary here.

Runningfull_replay_relation_followup.py --follow separately consumes baseline
failures fromfull_replay_v1. Sourcesnapshots/hashes/percaserows are preserved in
relation_followup_v1; summary reports baseline failures and successful repairs
separately. Originalfrozenpolicy still records271161asfailed; adaptiveevidence
mustnotbeusedtoinflate thatpolicy's own recoveryrate. Consumer stops atproducer
completion oronehourwithoutinputand supports --resume --follow. Atthischeckpoint,
85newcasescompleted:84baseline recoveries+1followup repair, combined391+85=476.
Fullpopulationrun stillincomplete; future failures requirefurtherdiagnosis.

### September 16 — user stopped the local reruns

The user explicitly instructed not to rerun these cases. Stopped the population
replay parent, its six workers, and the follow-up consumer; verified all eight
processes exited. Completed checkpoints and frozen source snapshots are retained.
Do not restart these jobs or launch further recovery reruns without fresh explicit
user authorization. Continue from the archived evidence and mathematical analysis.
Earlier statements that the full replay is running are superseded by this stop.

### September 16 — evidence-led mathematical synthesis, no new numerical runs

User reaffirmed: use existing numerical results to guide mathematical rigor;
no reruns. This pass used saved files, algebraic derivations and manuscript
editing only. No solver, forward evaluation, certificate battery, cloud job,
or stopped process was run or restarted. Three separate read-only reasoning
reviews checked the local-certificate logic, weak-wedge mechanism and manuscript
claims. The new theorem section received two mathematical reviews and corrections.

New paper/failure_geometry.tex (included in main.tex) proves:
1. In tolerance-weighted max norm with unrestricted estimator outputs, the
   optimal actual-data error is half the compatible-set diameter. Uniform
   deterministic minimax error is omega_tau(2eta)/2. Physical-output restrictions
   retain the lower bound but require a separate equality argument.
2. A weighted interval inverse-defect bound below one gives an explicit LOCAL
   Lipschitz inverse bound; no global upper bound or truth-in-box inference.
3. A flat FIRST prism has exact index/source-position compensation:
   r_n(q)=q/sqrt(n^2+(n^2-1)q^2), b(n)=b(n0)+h[r_n0(q)-r_n(q)]. Matching the exit
   state preserves the entire downstream trace. Zero wedge also hides its speed
   and phase. Later flat prisms can retain time-dependent index information.
4. On a uniformly regular admissible fixed-gain family, double integration of
   a bounded mixed derivative gives ||Psi(u,n')-Psi(u,n)||inf<=K|u||n'-n|.
   K is symbolic, not a newly measured/certified archive constant; no O(u^2),
   optimal sensor cutoff or universal wedge threshold is claimed. Saved723421
   endpoints independently supply the concrete finite-noise obstruction.
5. Temporal collisions aggregate torus coefficients, with additional sampled
   aliases; they can defeat independent spectral readout while physical local
   injectivity remains. Exact gain reparameterization changes no physical
   information when the whole domain and physical loss are preserved.

Corrected main.tex's full-gap ambiguity floor, unjustified eta/g segment floor,
necessary distinct-speed assertion, unconditional rank-P and phase/amplitude
readout claims, physical refusal versus exclusion, certificate-search ceiling,
closure versus tolerance distinction, unsupported prism-count residual claim,
universal recovery/cure-time prose and pending campaign-as-evidence wording.
Added explicit hypotheses to affine sheets. Numerical code, frozen evidence
and Lean sources were not changed. The old rewrite blueprint was archived and
replaced by an existing-evidence/analytic plan; the earlier experimental plan is
marked superseded. Main PDF has not been regenerated and remains an older draft.

Companion evidence map: paper/FAILURE_THEORY.md. Remaining work: quantitative
mixed-derivative constants, local-posterior containment/outside-box alternatives,
later-prism nuisance directions, collision initializer conditions and arithmetic/
source provenance. The saved certificates are conditional implementation audits;
preconditioner hashes and shared-primitive replays are not standalone independently
machine-checked numerical proof objects. Some weighted boxes are extremely small.
Text-only checks found no unmatched LaTeX environments or missing/duplicate labels;
git diff --check passes. Existing result links resolve. Canvas updated with the
mathematical thread; all computational reruns remain stopped.

### September 16 — global inverse design with affine elimination

User requires either a defensible explanation of the whole failure tail or an
algorithm capable of handling it. Preserved the explicit no-rerun instruction.
This pass derived an algorithm and proofs from the model; no forward-model,
solver, interval certificate, experiment, or campaign was executed.

New paper/GLOBAL_INVERSE_DESIGN.md gives four direct propositions:
1. F(q,l)=b(q)+A(q)l reduces nonlinear subdivision from eighteen to fourteen
   coordinates while retaining the complete four-dimensional affine fiber.
   Interval coefficient bounds provide a four-variable outer LP for each q box.
   Sixteen affine sign regions optionally preserve the R|z| dependence.
2. Approximate LP multipliers yield rigorously checkable exclusion and affine
   coordinate bounds, including a norm term for their dual equality defect.
3. A signed data-space direction gives a verified residual LOWER bound for an
   entire region. Numerical fits can propose directions but their local residuals
   cannot exclude regions. No new numerical lower-bound values were computed.
4. Compact regular domains, convergent enclosures, fair subdivision, increasing
   precision and complete LP feasibility processing give outer convergence and
   finite inconsistency/strict-accuracy decisions. Ambiguity needs separate
   compatible endpoint or existence proofs. Boundary decisions are not covered.

Two independent reasoning reviews checked the affine/dual algebra and global
convergence logic. Corrected the LP completeness assumption, separated regular
domain restrictions from affine elimination, and disclosed the noiseless core
versus mathematical F consistency issue. This is reviewed symbolic mathematics,
not a machine-checked theorem or newly executed numerical certificate.

The design preserves ordering and collision branches. A subset of samples can
exclude a region or bound the full-data feasible set from outside; actual-y
ambiguity witnesses must fit the full record. A coordinate-midpoint guarantee
does not itself establish physical admissibility or nonempty feasible data.
Current binary64 ivx is not the unbounded-precision backend required by the
abstract completion theorem. Restricting to positive physical margins changes
the domain and cannot silently omit boundary regions of the original prior.

Global interval inversion and variable projection are established methods;
primary references are linked in the design. No novelty is claimed for those
general methods. The proposed model-specific integration and tight directional
bounds still need an implementation and performance evidence before any fast
all-case solver claim. Fourteen-dimensional subdivision may remain prohibitive.
The 2,643 truth-side physical violations do not by themselves exclude all valid
explanations of those observations. The 15,759 local truth boxes do not give
blind initialization or global accuracy. No original failure was reclassified
or counted as a new recovery in this pass.

Updated the paper blueprint to prioritize global containment, linked the design
from FAILURE_THEORY.md, and updated the existing research canvas. Main manuscript,
PDF, numerical code, saved result records, and Lean sources were not changed in
this pass. All numerical reruns remain stopped.

### September 16 — full eighteen-parameter scope reaffirmed

User rejects weakening the problem to fourteen dimensions. Clarified the global
design, paper blueprint and canvas: all eighteen parameters remain unknown,
inferred from the same data and included in every accuracy check. The proposed
fourteen-plus-four split uses exact affine structure internally; it keeps the
entire four-coordinate compatible set and reconstructs the full eighteen-vector.
It supplies no ground-truth geometry, offsets or other calibrated coordinates.
The full-eighteen requirement does not remove the already stated regular-domain
or conditional-termination limitations. No solver or evidence files changed and
no numerical runs were performed.

### September 16 — literature search and exact algebraic formulation

User explicitly requested stronger mathematical theory for the full inverse.
Searched primary literature in real algebraic geometry, polynomial optimization,
certified homotopy, structural identifiability and spectral recovery. Three
independent reasoning subtasks and two reviews checked the resulting derivation.
No numerical evaluations, installations, certificate runs or solver reruns.

New paper/ALGEBRAIC_INVERSE_THEORY.md derives an exact polynomial graph of the
sampled reduced per-axis model. Eighteen physical coordinates remain unknown:
h=tan(pi*N*dt), p=tan(phase/2), a=tan(wedge), tan(beam), glass and geometry/source
coordinates. On dt=1/20 the transforms are invertible on the full declared
coordinate priors. Cubic rotor recurrences retain all collisions and zero wedges;
positive-root Snell equations and guarded signed ray intersections retain the
current physical branch. Additional ray states are determined auxiliaries.
Positive path length was NOT silently added; fmodel does not impose that check.

Native speed/angle tolerances have polynomial tangent-difference comparisons.
With exact rational/algebraic observations and allowances, rational tolerances
and the declared grid/branch, effective real quantifier elimination therefore
decides compatibility and quantified accuracy, including exact boundary cases.
This is stronger than the earlier interval-refinement completion theorem. It
permits ambiguous or inconsistent outcomes, not universally unique recovery.
The <= tolerance pair criterion and the archive's strict < criterion are
explicitly separated. Estimates in the strict formula are restricted to the
transformed prior box by a harmless native-coordinate clipping argument.

The ideal grid is not automatically bit-identical to np.arange timestamps.
Finite stored rational timestamps also admit a lift by a common time unit and
repeated-squaring rotation graphs, but their algebraic constants may have huge
degree. A justified timing/model/arithmetic allowance is an alternative; none
was calculated or silently added here. Exact native angles/speeds are obtained
through inverse maps of algebraic transformed outputs and need not be algebraic.

Practical candidates: sparse moment/SOS certificates and structured polynomial
homotopy, preserving full-data and physical-branch certification. Current
derivation proves neither affordable algebraic elimination, low-order SOS
exactness, bounded auxiliary variables near grazing, nor homotopy completeness.
Generic identifiability and finite-atom spectral theorems do not cover every
exceptional finite-data case. No archived failure was resolved by computation.

Reviewed corrections include axis-specific face auxiliaries, first-entry-plane
initial coordinates, pair versus estimate threshold factors and strict boundary
handling. Sources, derivation and limitations are recorded in the new note.
Updated paper blueprint/global-design cross-links and existing research canvas.
No main manuscript/PDF, numerical code, frozen evidence or Lean sources changed.

### September 16 — bounded deterministic implementation after fresh authorization

User explicitly asked to push the deterministic theory analytically or
numerically. This authorizes small predeclared checks with hard limits; the old
population replay and repair processes remain stopped. No population recovery
or new inverse run was launched. All eighteen coordinates remain unknown.

Implemented risley_lattice/algebraic_graph.py: a truth-independent sparse
polynomial graph of the reduced sampled model, exact Fraction coefficients,
symbolic algebraic prior endpoints, branch guards and rational observation
constraints. Both the cubic and optional quadratic rotor graphs are available.
The quadratic graph has 44+132*m variables (18 physical), maximum degree two;
at m=40: 5,324 variables, 6,386 equalities and 1,687 branch inequalities.
Quadratic does not mean convex or first-relaxation exact.

experiments/algebraic_graph_check.py checked five fixed systems at 40 samples:
ordinary, zero first wedge, same signed collision, opposite signed collision,
and saved 723421 endpoint. Maximum scaled equation residual 3.22e-16 and quadratic
output difference from interval-model midpoint 8.53e-14. Positive-root negative
control, exact rational observations and zero-wedge invariance passed. These
are numerical witness/parity diagnostics, not certified roots or recoveries.
Final run 2.097 s with a 180 s watchdog. Prior passes during implementation were
also bounded; total graph checks under 8 s. No forward physics was changed.

Exactness controls distinguish k/20 from stored np.arange timestamps (maximum
difference 1/5629499534213120 seconds on this 40-sample check), and exact decimal
glass bounds 13/10..9/5 from binary64 model LO/HI (each endpoint higher by
1/22517998136852480). Neither timing discrepancy nor canonical-model rounding
has a newly proved universal allowance. Angular endpoint root isolation remains
unimplemented. These distinctions are explicit in paper notes and saved checks.

Implemented risley_lattice/affine_exclusion.py: interval coefficient enclosures
over a nonlinear region retain the entire four-coordinate affine prior. The LP
only proposes a direction/multipliers. Final Farkas bounds and residual lower
bounds use exact Fraction arithmetic, including dual-equality defect and input
rounding. Coefficient proofs still inherit existing ivx A-fp/A-libm assumptions
and exact affinity. Rotor arguments are guarded; tangent guards are inherited
from ivx. Failed physical enclosures are refusals, never exclusions.

Four tiny shifted strong-speed boxes excluded at eta=1e-8; all four known-feasible
controls retained, plus a zero-wedge hidden-speed alternative. Wide-noise,
negative-multiplier and elementary-function-domain controls passed. 96 samples
per fixture, T=4.8; latest initial run 1.84 s. Independent stdlib Fraction replay
verified all four signs, bounds and stored rounding inequalities. Nonlinear
radii were about 1e-10 times prior widths, so this does not establish broad-box
pruning. Inputs are truth-assisted synthetic midpoint fixtures, not the archived
observations. No archived failure is reclassified.

paper/ALGEBRAIC_SPARSITY.md gives an explicit running-intersection tree and
auxiliary bounds on a declared regular domain. The implemented quadratic
graph's maximum bag 64 gives first-order moment block 65, independent of sample
count. The implemented cubic order-two block is 1431. Three-sample symbolic
support/connectedness checks passed. 19,200 optical blocks at 1600 samples use
329 MB for one padded packed matrix each; this is raw representation arithmetic,
not measured RAM or runtime. Low-order tightness is open. Closed equivalent
root guards are distinguished from genuinely restrictive positive optical
margins needed for the compact sparse-SOS convergence argument.

formal/Risley/AlgebraicLift.lean adds 9 local lemmas, checked successfully by
lake env lean Risley/AlgebraicLift.lean in C:/Users/josep/lean/risley, Lean 4.33.0,
exit 0, no diagnostics, no sorry. Full Mathlib imports were slow and stopped;
minimal imports completed. No cache download or .lake build in Dropbox.
Repository/external Lean sources and entrypoint match. This proves local
rotation/Snell/intersection/tangent identities, not full graph equivalence.
An independent sparse-rational checker also passed 6 identities and 4 negative
controls. Saved source hashes were independently checked against current files.

Implementation results and remaining obligations: paper/ALGEBRAIC_PROGRESS.md.
Raw proof/diagnostic records: experiments/results/algebraic_2026_09_16/.
The existing research canvas and paper blueprint now reflect fresh bounded
authorization and current results. Main manuscript/PDF and original archive
evidence were not changed in this pass. A practical complete full-18 inverse,
global uniqueness, and an affordable all-case decision guarantee remain open.

The fixed width follow-up then checked three shifted fixtures at five radius
fractions of each nonlinear prior width: 1e-8, 1e-6, 1e-4, 1e-3 and 1e-2.
All fifteen checks completed: eleven exact-verified exclusions and four interval
physical-margin refusals. Largest excluded radius fractions were 1e-4 for
29640 and 1e-3 for 367816 and 723421. All four affine coordinates retained their
full priors. The truth-centered feasible control at 1e-4 was retained.
Refusals are inconclusive, not physical invalidity proofs. Charged numerical
time was 3.84 seconds under a 120-second cap, including a conservative two-second
charge for a Dropbox checkpoint interruption; completed checks were preserved.
This establishes wider regional pruning on synthetic fixtures, not global
coverage or archived recovery. Results: affine_exclusion_widths.json in the
same evidence directory.

### September 16 — full-prior optical theorem and capped global relaxations

Fresh "go go go" authorization was used for bounded deterministic tests with all
18 physical parameters unknown. No archive recovery campaign was restarted and
no archived failure was newly resolved. Full report and links:
paper/DETERMINISTIC_GLOBAL_PROGRESS.md.

Main mathematical result: paper/GLOBAL_OPTICAL_MARGINS.md proves that one
transmitted, forward per-axis prism sends incoming vertical unit component z to
t >= z^2/(2*n). The original stored prior is enclosed by n <=181/100 and z0>=8/9;
delta_(i+1)=(50/181)*delta_i^2 yields positive air-exit bounds approximately
0.2182661483, 0.01316025179 and 0.00004784315669. Glass slope <=5/4 and exit-face
slope <=1/3 imply slope-form propagation denominator >=7/12. The unnormalized
graph denominator bound is 14/39 and the unit-normal bound is 224/663; these are
different conventions. Exact recurrences give finite Cartesian positions and
signed path lengths. The conservative final position bound is about 84 million
model distance units, not a hardware tolerance or useful recovery guarantee.

This CORRECTS the earlier claim that Cartesian auxiliaries could become
unbounded through grazing on this particular three-prism prior. No additional
hardware restriction is needed for these forward/intersection bounds. Exact
feasible homogeneous intermediate scales cannot vanish on the rational prior.
The theorem applies wherever the selected transmitted, forward branch is
admissible; it does not prove a positive exit Snell-root/TIR margin, continuous
admissibility, inverse conditioning, global uniqueness or recovery. Relevant
algebraic/projective/global-design notes and the paper blueprint were corrected.

Added formal/Risley/ProjectiveLift.lean (three lemmas) and OpticalMargins.lean
(three lemmas). External checks in C:/Users/josep/lean/risley completed in
10.109 s and 14.469 s, exit 0, no diagnostics or sorry. Repository/external
sources and entrypoint are synchronized. These supplement the previous nine
algebraic lemmas; complete physical-prior mapping and graph equivalence remain
paper arguments. No Lean build occurred inside Dropbox.

New risley_lattice modules: projective_graph.py (bounded quadratic homogeneous
graph), optical_bounds.py (exact derived auxiliary bounds), polynomial_lp.py
(rational McCormick relaxation), poly_dual.py (exact residual-corrected LP
bounds), moment_cuts.py (exact affine products and nonnegative square cuts),
highs_proposals.py (persistent numerical LP proposals) and sparse_moment.py
(numerical SDP proposals). The LP verifier passed 49 controls; moment cuts
passed 21. Projective graph parity checked three four-sample fixtures, maximum
scaled equation residual 3.06e-16 and output difference 1.42e-14. These parity
checks are numerical diagnostics. Exact optical bounds tightened 232 auxiliary
coordinates on a four-sample check without observed containment violations.

All global tests used one fixed synthetic fixture, exact ideal dt=1/20 and
position allowance eta=1e-6. Truth generated observations only; no local truth
box, known coordinate, initialization or calibration was supplied to a global
optimizer. Angular outer intervals enlarge the native prior, and the relaxed
graph includes closed chi>=0. All 18 parameters were free throughout.

Two-sample SDP: 17.58 s total, 15 s solver cap; four observations only, an
implementation diagnostic. Ten-sample SDP: 35.27 s total, 25 s solver cap;
20 observations, 1730 graph variables, largest moment matrix 19 by 19. Both
statuses optimal_inaccurate, with no useful speed narrowing. The ten-sample
physical moment matrix had numerical rank 18 and its mean had maximum scaled
graph residual about 0.659. No physical solution or exact SDP bound was obtained.
Tested bags were constraint supports plus the physical-parameter bag, not the
previously derived size-65 Cartesian junction tree. No sparse-hierarchy
convergence claim applies to this prototype bag selection.

Ten-sample LP plus 144 exact square cuts: 89.0 s, 10931 lifted variables,
40 objectives, nine exact certificates and 31 time-limited proposals without
certificates. Follow-up with proved optical bounds and persistent HiGHS basis:
82.06 s under a predeclared computation budget, 32/36 coordinate objectives
attempted, 31 numerical time limits; all 32 finite proposals yielded exact
residual-corrected bounds. Four source-position objectives were unattempted.
Neither LP study narrowed any encoded physical prior interval. Independent
stdlib Fraction replay verified all 41 certificates and their LP prefixes,
source hashes and residual corrections in 4.53 s. This establishes no useful
narrowing within the caps, not intrinsic failure of every first-order method.
No speed advantage from basis reuse is established. Local .codex-deps gained
CVXPY/SCS/HiGHS dependencies; existing NumPy and SciPy versions were preserved.

Finally experiments/projective_exact_witness.py certified one exact transformed
rational point compatible with both saved ten-sample studies. 80-bit outward
dyadic Fraction intervals and integer isqrt prove 429 strict guards, 120 finite
nongrazing intersections and all 20 output errors <=
17804599111/1208925819614629174706176 <1.473e-14 <eta. It completed in 1.140 s
under a 60 s hard cap. The independent read-only audit passed. Exact inner boxes
prove original-prior membership. The native coordinates are the inverse-atan
image of the recorded rational point, not necessarily the displayed generation
theta. Existence follows from composing well-defined exact operations and
graph identities, not from equation intervals containing zero. This enclosure
uses no floating-point/libm accuracy assumption; Python integer/Fraction/isqrt
correctness and the paper's model-equivalence arguments remain dependencies.
It is a known feasible forward witness, not blind recovery or uniqueness.

All records are in experiments/results/algebraic_2026_09_16/. Existing result
records were preserved; Dropbox output-write repairs are explicit metadata,
not hidden repeated solves. The research canvas includes the theorem, exact
witness and bounded negative results. Main manuscript/PDF and archived campaign
evidence were not changed in this pass. Next priority: derive stronger physical
consistency identities or eliminate intermediate variables, and demand a
nontrivial exact full-prior bound before scaling any global solver.

### September 16 — exact elimination and a data-dependent collective bound

User authorized the next mathematical step ("lets do it"). All 18 physical
unknowns were retained. No archived solver population or optical battery was
restarted. Main result: paper/ELIMINATION_PROGRESS.md, with derivations in
PRISM_ELIMINATION.md, RAY_TRANSFER_ELIMINATION.md and SCAN_EXCURSION_BOUND.md.

NEW DATA-DEPENDENT EXCLUSION: for the same saved fixed_regular ten-sample data
at eta=1/1000000, every compatible system satisfies
max_i |tan(ax_i)| >1933/250000=.007732. A rigorous native-angle lower bound is
7611035824702023/17187500000000000 degrees, approximately .4428239025281177.
Thus at least one wedge magnitude exceeds .442 degrees. All other fifteen
parameters retain their full original priors. This is a collective exclusion
of the all-small-wedges cube, not a per-prism bound, individual recovery,
global uniqueness theorem or classification of archived failures.

Proof: compare each candidate with its same-parameter all-flat system, whose
scan is constant. On wedge tangent radius r<=1/60, coarse Snell induction
establishes |X_air|<3/5, Z_air>4/5 and positive optical margins independently
of a failed interval computation. Tangential Snell gives cumulative horizontal
deflection <=i*(101/100)*r; exact scalar slope Lipschitz bounds give a rational
position-error recurrence D4(r). The coordinate excursion is <=2*D4(r).
The saved x range minus 2 eta is about 20.92915472536, whereas twice the envelope
at r=.007732 is about 20.92757473647. Exact strict gap:
1019625855099547255962039366912580429 /
645337358555532202920362849075200000000 >0.

risley_lattice/scan_excursion.py and experiments/scan_excursion_check.py use
stdlib Fraction only; the fixed denominator 1e6 scalar radius search took
.016s, with zero optical forward evaluations and zero inverse optimizer calls.
The next grid radius fails this sufficient inequality, which is inconclusive
rather than a feasibility claim. Independent proof/code/data audit passed.
The saved array is identical to the preceding projective/optical global tests,
so their exact compatible forward witness applies. Deterministic eta and the
reduced per-axis model are explicit hypotheses. Evidence: scan_excursion_bound
and scan_excursion_audit JSON records in the existing algebraic results folder.

NEW PRIOR-WIDE IDENTITIES: let x be incoming horizontal air component and (u,t)
outgoing, B=sqrt(n^2-x^2), Q=sqrt(1+secondary^2), V=primary. The complete
direction equations are outgoing unit norm, Q(u-x)=V(B-t), and transmitted
guard E=Qt-Vu>=0, with B,Q,t positive. The closed zero-root boundary is an
outer enlargement; a rational forward wrong-root counterexample proves the
guard cannot be discarded. With C=B-t and D=QB-Vx, derive
D^2-E^2=(1+a_i^2)*(n_i^2-1), shared across both axes and all samples;
Q(D-E)=(1+a_i^2)*C; and uniform 24/85<=C<=151/100. No nonzero wedge or distinct
speed assumption enters. Six free-polynomial identities, six rational controls
and two negative branch controls passed in .907s under a 10s hard cap.

EXACT POSITION ELIMINATION: Pnext=A*P+3*Bg+G*T with A=B*E/(t*D),
Bg=E*x/(t*D), T=u/t. Composing three stages gives
Y=K*(source+6*beam_tangent)+3*H+gap*J+d_W*T3. All four affine parameters
remain unknown; this eliminates auxiliary states rather than calibrating or
discarding physical coordinates. New transfer proof gives
0<=A_i<=min(2,1+1/(3*delta_in)); caps are 11/8, 2, 2. Consequently 0<=K<=11/2,
a uniform source-position amplification bound at fixed remaining parameters.
This does not establish inverse conditioning or a positive exit TIR margin.

Implemented risley_lattice/reduced_transfer_graph.py: 903 variables at 10 samples
versus 1730 for the previous projective graph, 18 unknown physical coordinates,
no internal positions/heights or separate X_g,Z_g direction pairs, maximum
degree 2. The scaled glass component B remains as a refraction auxiliary. Includes
shared prism contrast/gain and adjacent-rotor dot/cross identities. Numerical
parity on six 10-sample fixtures (including zero wedges and signed collisions)
had maximum output difference 4.2633e-14. Final v3 check took .172s under a 30s cap.
Earlier .203s and .157s diagnostics are retained: v2 fixed exact-rational
observation conversion; v3 added the stronger gain caps and identities. These
are numerical consistency checks, not root or recovery certificates.

One predeclared joint-objective LP study used saved observations only and
minimized sum_i tan(ax_i)^2, WITHOUT supplying the analytic excursion cut.
The reduced graph plus objective had 904 original and 6910 lifted variables.
Three proposals, each capped at 4 solver seconds, with two rounds adding 96 exact
square cuts completed in 5.437s; all three numerical statuses were optimal.
All 18 parameters stayed free. Exact residual-corrected bounds gave no useful
positive objective lower bound; the retained bound was the prior zero.
This does not prove an exact mathematical relaxation optimum of zero or an
all-method impossibility. Source hashes, exact LP coefficients, dual proposals
and cut witnesses are saved in reduced_transfer_bounds.json/.lp.json.gz.
The successful analytic sample-coupling inequality was stronger than this
tested relaxation for this objective; a recovery-speed advantage is unproved.

Added formal/Risley/PrismElimination.lean with three local lemmas, compiled
once externally in 10.094s under a 45s cap, exit 0/no diagnostics/no sorry. Exact
normal-contrast, gain-difference and positive-gain implications are checked;
physical mapping, uniform transfer cap and excursion theorem remain paper
proofs. Repository/external sources, imports and README are synchronized.

Updated current research canvas and progress-note links. Main manuscript/PDF,
canonical forward model, old proof records and original archive are unchanged
by this pass. Next priority is regional exclusion using differences between
samples with shared nuisance parameters, combined with the reduced graph.
Require new certified exclusions before expanding solver budgets. A practical
complete full-18 inverse remains open; no archived case was reclassified here.

The new excursion_cuts.py helper compiles the audited exclusion into six
alternative strict signed-wedge branches (a UNION, never their conjunction),
plus the weaker necessary quadratic cut sum_i a_i^2 >=
3736489/62500000000=.000059783824. Graph type/prior checks protect the stated
scope; a failed excursion criterion returns no cut. Six exact signed-branch
controls, coefficient checks and a constant-data no-exclusion control passed
with zero forward evaluations and zero solver calls. The controls ran below
the clock's reported resolution; no speed claim is made from that value.
Source hashes and exact constraints: excursion_cuts_check.json.

Final independent audits passed: scan_excursion_audit.json reconstructs the
exact saved-data recurrence and source provenance in .375s;
reduced_transfer_audit.json replays all three LP certificates with their exact
row prefixes and all 96 appended square cuts in .438s. Neither invokes an
optical model or numerical solver. Read-only review of excursion_cuts.py also
passed: the quadratic non-strict bound is a valid weakening, and the six strict
halfspaces are correctly alternatives. All current report source hashes,
Python syntax, local links and external/repository Lean copies were checked.

## 2026-09-16: User correction — full18 recovery, physical weak-prism dictionary

The user explicitly rejected collective wedge bounds and graph reductions as
a substitute for recovering all eighteen physical parameters or resolving the
archived failures. Refocused this pass on actual case recovery. No population
replay was restarted. All numerical work used one development case, a frozen
three-case pilot, and a separate bounded repair of its one failure.

READ-ONLY COUNT CORRECTION: actual saved JSONL rows give500 distinct earlier
recoveries:391 prior cohort cases,108 successes among109 stopped-replay rows,
and the separately repaired replay failure271161. Stale summary checkpoints
showed499. Thus before this pass15259 of15759 physically admissible original
failures had no saved recovery;2643 additional original failures have proved
physical exclusions. Original failure total18402. Old diagnostic labels are
heuristics, not causal theorems. Counts and hashes:
experiments/results/full18_target_2026_09_16/archive_target_audit.json.

NEW METHOD: risley_lattice/physical_weak_dictionary.py. At a saved background,
zero the weakest estimated wedge and differentiate the entire downstream
optical chain with respect to its two Cartesian rotating-normal components.
The first-order signal D_N*c includes physical sideband modulation. Project
off14 nuisance tangent columns; solve2 phasor coefficients in the native signed
phase sectors; search at most1121 signed frequencies and refine at most5 minima.
The saved glass/background are initialization proposals, not known quantities.
All18 coordinates are released in polish_extended:14 nonlinear includingall3
glass,4 unknown affine coordinates, then canonical all18 refinement. No true
parameters are passed to the worker. Numerical proposal code is not a certified
frequency selector. A proposed small-MSE integration gate was withdrawn: it
would reject two of the successful new cases. No change was made to default
solve18 or the stopped campaign.

DEVELOPMENT CASE320811: first proposal plus58 prefix nfev recovered all18;
max native error7.370845156451367e-8, canonical MSE2.3994710343113236e-25.
Physical interval checks retainall1600 samples/80seconds. Worker computation
1.046s, supervised worker1.906s; neither includes input preparation. These are
one-case timings, not a benchmark. Original failed weak speed3.4494 was repaired
to2.0536368, with an extremely small physical wedge. Independent exact saved-
error/hash audit passed. Frozen runner experiments/full18_targeted_recovery.py.

FROZEN FOLLOW-UP: SHA256 selection physical-weak-pilot-v1:<seed>:<case> over5294
eligible records selected162450,316177,428374. Eligible means preexisting
certified_admissible status, finite savedx18, min|savedax|<.01degrees, no earlier
recovery/probe record. Preexisting physical status uses simulation truth;
true parameter values and saved errors were not selection criteria. Algorithm
and manifest frozen before model calls. Each worker capped90s externally,
75s between stages,5 completions of at most600 prefix nfev each. Runtime module
hash81abff74a0a66d7df929642e42c5e3115a5b630cfbcb154dd24df5c621ce67bc.

Frozen result2/3:316177 maxerr5.094591415399918e-12,MSE4.4727790125916244e-26;
428374 maxerr1.5351453441780905e-10,MSE3.7152099939531236e-25. Both first
dictionary candidates.162450 failed after5 candidates: maxerr17.1241740566,
MSE1.9788181565e-8; result preserved. Supervised worker times9.969,1.938,1.688s
for162450,316177,428374 respectively; preparation saved separately. No timeout.
Independent read-only auditor reconstructed all eligible/excludedIDs and the
selection order, checked bytes/hashes, exact18 errors and full physical-check
metadata. Audit passes for the valid recorded failure as well as the successes.

ADAPTIVE ORDER REPAIR162450: the dictionary had already found missingN=.15;
its initial profile improved99.977%. Truth-side diagnosis showed the weak prism
in the wrong physical slot. Transfers do not commute, so a reordered block is
a distinct model proposal. New separate runner froze the observation-selected
failed estimate and triedall5 nonidentity permutations in lexical order,
permutingN/ax/ay/ng consistently and releasingall18 on completion. All5 outcomes
saved; full canonical MSE and physical checks alone selected(1,2,0). Maxerr
3.968779083152185e-9,MSE5.459981509901538e-24;13.047s including supervisor/input
setup under60s hard worker cap. Original frozen failure unchanged. Independent
audit confirms exact observation-byte reuse, parent estimate,all5orders,
source/input/truth hashes, exact18 errors and full1600 physical metadata.
Diagnosis used truth; candidate enumeration and selection did not. This is an
adaptive repair, NOT a3/3 frozen pilot. No optical calls in either auditor.

PROVENANCE: all4 observations regenerated with current case_at/vec2pat from
archived IDs; original cloud observation arrays and exact executable revision
are unavailable. Workers receiveonly x0/observations NPZ; truth stored separately.
Reported recoveries concern these regenerated current-model case records, not a
bitwise replay of the original cloud experiments. Positive interval physical
checks were performed by workers; independent saved-evidence audits do not
recompute optical interval arithmetic or prove parameter accuracy with noise.

CURRENT ACCOUNTING:504 distinct saved numerical recoveries of admissible
original failures,15255 admissible cases without saved recovery,2643 physical
exclusions. Most remaining cases are unprocessed after the campaign stop, not
demonstrated failures of the new method. Local injectivity ofall15759 is a
separate result; finite-noise ambiguity does not resolve noiseless failures.

MATHEMATICAL RESULT: paper/PHYSICAL_DICTIONARY_CERTIFICATION.md gives a
conditional deterministic exclusion theorem: if r=A_N*c+e with||e||<=epsilon,
any frequency cell with validated minimum profile distance>epsilon is excluded.
Wrong projected signal subspaces separated by>2epsilon cannot win. Continuous
cells use d_lower(N0)-rho*L*h, with explicit optical-template derivativeL.
The note proves convex tangent lower bounds for the2D coefficient fit and an
error budget including measurement, background/glass, Taylor remainder and
arithmetic errors. Current fitted scores are upper bounds; background coverage
and rigorous numerical hypotheses have NOT been established on the4 recoveries.
No universal recovery/uniqueness theorem is claimed. First-order glass/gain
confounding requires full nonlinear completion.

paper/FULL18_RECOVERY_TARGET.md preserves the complete-inverse research route:
exact affine elimination, minimal polynomial subsystem, full start-fiber/path
accounting, exact real/physical certification, and separate singular/noisy-set
handling. Algebraic degree and a practical runtime remain unknown. Updated
paper/FULL18_RECOVERY_PROGRESS.md, physical dictionary derivation, paper plan
and existing status canvas around actual18-parameter recovery. Main manuscript,
PDF, canonical physics and previously frozen numerical sources unchanged.

Final read-only verification passed:7 Python sources parse,47 source-hash
bindings remain unchanged,15 result-file hash links match,all3 independent
audit reports pass, and23 relative links in the4 new research notes resolve.
No additional model calls were used for these checks. The existing canvas was
updated using its existing components; a rendered UI/typecheck was unavailable.

## 2026-09-16: Stop case hunting; universal full-18 theory is the requested target

The user explicitly rejected the four-case repair strategy as the research
direction and requested a theory covering every degeneracy and the complete
eighteen-parameter problem. Individual-case recovery runs are now stopped as
well as the population replay. Do not interpret the earlier bounded-test
allowance as authorization for another recovery cohort.

This pass performs mathematical reasoning and primary-source verification
only. The target is one compiled inverse relation over observation space:
classify inconsistent data, return an accurate compatible physical estimate
where one exists, and certify failure of the accuracy requirement otherwise.
All eighteen physical unknowns remain represented, including unobservable
coordinates on genuinely ambiguous fibers. Singularities and speed collisions
must remain in the domain; generic-root assumptions are insufficient.

The distinction is unavoidable: exact degeneracy can remove information, so
no method can force unique recovery there. This is not an explanation of the
locally injective archived tail. Finite decidability of the complete inverse
does not establish affordable runtime or resolve any unprocessed archive row.

NEW UNIVERSAL THEORY: paper/UNIVERSAL_FULL18_THEORY.md now states and proves
the complete decision relation with free observations and allowance. Exact
real quantifier elimination partitions the data into inconsistent, accurately
recoverable, and nonempty-but-unresolvable classes under the stated output
convention. CAD selection supplies a compatible physical estimate on success
or a symbolic adversary against proposed estimates on failure. It retains
singular, positive-dimensional and zero-wedge strata, strict physical guards,
native-unit tolerances and all eighteen physical coordinates. Strict/open-set
and physically constrained estimates require the full quantified criterion;
coordinate midpoints and two-point ambiguity witnesses are not complete in
those settings. The universal partition/selectors have not been computed.

STRONGER STRUCTURAL COROLLARY: for fixed three-prism optics, fixed exact uniform
sample step, fixed original prior and fixed rational native tolerances, eliminate
the fixed-size instantaneous optical trace once into a Boolean sign predicate
Psi. Its degree and polynomial count are fixed constants, currently unknown.
Substitute U_k+iW_k=a(1+ip)^2(1+ih)^(2k)/[(1+p^2)(1+h^2)^k]. Denominators are
positive and degrees grow linearly in k. An n-sample compatible-set formula
therefore uses only 18 physical variables, O(n) polynomials of degree O(n),
and polynomial encoding size. No sample-specific ray variables remain.
For rational observations/eta, exact feasibility uses 18 variables and the
accurate-compatible-center decision uses 36 (estimate plus possible truth).
Fixed-dimensional CAD yields a deterministic polynomial bit-complexity bound
in n+B, with potentially astronomical fixed exponent and preprocessing cost.
This avoids treating the 903-variable lift as an unavoidable decision dimension.
The one-sample predicate is NOT computed and no practical speed is established.

Two independent mathematical reviews checked this reduction. Limits: the
bound concerns rational data, or one fixed coefficient field; fixed optics,
grid and tolerances; samplewise physical conditions; componentwise bounded
noise. It does not establish polynomial-size compilation over all free observed
coordinates, or polynomial bit complexity for arbitrary binary timestamp grids
whose common-step exponents may be enormous. Exact uniform-grid versus stored
timestamp and floating-output discrepancies remain explicit input obligations.

The same note proves an exact flat-prism indistinguishability example and a
continuity obstruction to uniform positive-noise recovery even for arbitrarily
small nonzero wedges. Neither argument classifies the locally injective archive.
A separate compact-semialgebraic injective-map corollary gives a global Holder
inverse bound without requiring a nonsingular Jacobian (Lojasiewicz theory).
Useful constants and injectivity are not supplied automatically. Primary
sources: Basu's real-algebraic-geometry survey and Basu--Mohammad-Nezhad 2024.

Next target is explicit symbolic construction and size assessment of the ONE
instantaneous compatibility predicate, preserving all branches. This is a
universal model construction, not another case-specific recovery protocol.
No numerical experiments, archive case evaluations or optimizer calls in this
pass. Existing recovery counts remain unchanged.

## 2026-09-16: Explicit polynomial full-18 compatibility construction

The authorized next step was symbolic model construction, with both population
replays and individual archived-case experiments stopped. This pass constructs
the actual polynomial sign predicate promised by the universal theory. It does
not solve an observation record or change any archived recovery count.

OPTICAL ELIMINATION: paper/EXPLICIT_OPTICAL_RELATION.md derives a homogeneous
three-prism recurrence with nine positive radicals per axis. Internal ray
positions are eliminated by exact affine transfer composition. The transfer
requires the factor r=1+V^2+W^2 in r*E*(B*P+3*X); independent polynomial checks
reject the mutant that omits it. Strict radicand/forward guards and nonzero
intersection denominators preserve the selected mathematical fmodel branch.
The comparison (numerator-threshold*denominator)*denominator preserves its sign
for either nonzero denominator sign. Zero wedges, zero speeds and coincident
speeds require no division or special generic-case assumption.

risley_lattice/exact_optical_predicate.py implements exact radical-sign
elimination with factored rational polynomial circuits. For A+B*sqrt(R), R>0,
the signs of A, B and A^2-B^2*R determine the result, including cancellation.
No conjugate-norm division is used; dependent and perfect-square radicals are
permitted. Every declared root-domain guard remains explicit. Independent
controls passed 1,176 exact sign comparisons and 6,245 guarded comparisons,
including invalid radicands, and twelve free-polynomial optical identities.
These checks and source review are not a proof-assistant proof of the compiler.

PRESERVED LIMIT: the first direct two-axis squared-band lowering reached its
500,000 arithmetic-node cap at x_observation_band after 18.25 seconds. Its
incomplete construction.json and radical source are preserved in
experiments/results/universal_symbolic_2026_09_16/. The successful construction
shares one axis/threshold template, rather than hiding that capped attempt.

COMPLETED TEMPLATE: exact_optical_axis.py and compile_optical_axis.py produced
optical_axis.polynomial.json.gz in 13.328 seconds: 119,105 reachable arithmetic
nodes, 16,850 sign nodes, 11,197 polynomial atoms, and structural degree upper
bound 17,408. Compressed size 1,186,530 bytes; SHA256
834447c6d223aeff706dddb74db42c2a49be3b2a6f77436fff3c87e6bd54597c.
Independent serialized-DAG inspection verified exact rational coefficients,
root-free arithmetic, dependency order, counts, degree bound, source hashes,
nineteen threshold-independent physical guards, and the exposed comparator.
The template's comparator entry accepts all signs intentionally; consumers
MUST constrain its sign for the desired lower/upper observation threshold.

EXACT PRIOR: exact_prior.py represents all eighteen native-transformed prior
coordinates, with rational or real-algebraic endpoints. Exact Sturm isolation
specifies each tangent constant, including root choice. The default glass
bounds preserve the original literal binary64 endpoints; exact-decimal glass
bounds are a separately named domain. The current speed chart supports exactly
Delta=1/20. Closed coordinate endpoints and open isolating intervals have
different meanings and remain explicitly distinguished.

FULL18 BINDING: exact_rotor_binding.py implements
U_i+iW_i=a_i*(1+i*p_i)^2*(1+i*h_i)^(2*k)/D_i,
D_i=(1+p_i^2)*(1+h_i^2)^k>0. Each template polynomial is transformed using its
numerator and three denominator exponents; positive denominator clearing keeps
all sign/equality tests exact. Both axes, all four observation-band inequalities,
eta>=0 and the exact original coordinate prior are explicitly conjoined. All
eighteen physical unknowns remain shared, including glass, geometry and beam.
No radical or internal ray variable remains in the polynomial arithmetic.

compile_full18_predicate.py materialized the symbolic k=1 sample in 12.125
seconds under an external 55-second watchdog: 426,042 arithmetic nodes, 64,156
sign nodes, 42,541 polynomial atoms, and 43 sign-conjunction entries plus the
eighteen-coordinate prior. Its degree upper bound is 60,416; compressed size
4,191,046 bytes. The saved full18_sample_1.polynomial.json.gz SHA256 is
d477e6009c11e30d536c418708fb006356d06d3a472e40141f901c3ff3697210.
full18_binding.json binds both source hashes and the input template/prior hashes.
The API accepts multiple distinct nonnegative indices; only the single symbolic
k=1 record was materialized here, with no observation data. One sample is not
claimed to identify eighteen parameters. Degree grows with the maximum sample
index; the O(n) degree theorem requires consecutive or suitably bounded indices.

INTERPRETATION: this is a concrete quantifier-free compatibility representation
for the strict mathematical model, including degenerate inputs, with an exact
full18 substitution implementation. It is not a dense polynomial expansion, a
CAD decomposition, a universal observation-space partition, an inverse selector,
a practical runtime guarantee, or a proof that all archived failures are now
resolved. The degree figures are conservative circuit bounds, not exact degrees
or solution counts. Floating acquisition/model discrepancies and between-sample
physics remain the input obligations described in UNIVERSAL_FULL18_THEORY.md.
The next mathematical implementation is the complete inverse decision/selection
step, with structural simplification and realistic cost assessment. No optical
forward evaluations, inverse optimizer calls or archive-record reads occurred
in this pass; exact algebra controls and symbolic construction were the checks.

Independent full18 artifact audit passed in 2.025 seconds and is reproducible
with experiments/exact_rotor_binding_audit.py. It verifies all source/input/
output hashes, exact prior embedding, all eighteen live physical symbols, the
four signed observation constraints, coefficient exactness and DAG structure,
and independently expands the three saved small denominator expressions.
It does not replay every large substitution or evaluate an optical parameter
point. Final read-only checks parsed eleven Python sources, verified twenty-four
source/input hash bindings and twenty-six relative document links.

The additional source-bound exact_rotor_binding_check.json passed independent
free-polynomial rotor identities at k=0,1,2,7, seventy-two synthetic exact
rational value/sign comparisons and seventeen invalid-input rejections. This
checks denominator scaling, powers, constant terms, cancellations and the
eighteen-coordinate prior interface without loading the optical template.

## 2026-09-16: Exact simplification with all18 retained; degree bottleneck remains

The user requested pushing the exact simplification. This pass used symbolic
construction, exact algebra controls, source/artifact audits and local Lean
proofs only. Population replay and individual archived-case experiments remain
stopped. No optical forward calls, observations, archive reads or inverse runs.

NEW OPTICAL LEMMAS: paper/EXACT_SIMPLIFICATION_LEMMAS.md proves that the original
prior makes Q/B root domains and tilted intersections automatically regular.
The selected exit root itself implies strictly positive outgoing vertical
direction; a separate forward guard is redundant. With |beam|<=1/2,
|V|<=1/3 and 13/10<=n<=181/100, the FIRST exit radicand divided by Q^2 is at
least 413/24000. Thus only the SECOND and THIRD exit-radicand guards are
nontrivial per axis. Those two guards remain strict. The output denominator
is positive, so sign((T-qL)*L) is replaced by sign(T-qL). These equivalences
require the exact original prior and the actual rotor map; they do not apply
to arbitrary unrelated instantaneous inputs. Source/recurrence comparison
confirms the unchanged numerator, denominator and all nine root radicands.

formal/Risley/ExactSimplification.lean machine-checks three local lemmas:
automatic outgoing forward direction, positive-denominator inequality
equivalence, and positive rotor denominators. The external build completed with
exit zero and no diagnostics, sorry or admit; both source copies are identical.
Evidence: results/simplification_2026_09_16/lean_exact_simplification_check.json
under experiments/. This is not formal verification of the complete compiler,
prior-to-hypothesis map or inverse method. Six additional free-polynomial
identities and exact rational constants support the optical proof.

GENERIC SIGN REDUCTION: new exact_sign_simplify.py preserves zero and negative
factors while decomposing root-free product signs, inferring positive sums,
and extracting common factors before radical sign elimination. Its exact
controls passed 1,098 sign comparisons and 5,635 guarded comparisons. No
statistical estimate is used. The frozen base compiler is unchanged.

MEASURED AXIS CONSTRUCTION: prior-only specialization gives 10,829 polynomial
atoms; adding exact structural sign reductions gives 9,901 versus the original
11,197. The structural result has 116,309 arithmetic nodes, 15,106 sign nodes
and degree upper bound 17,408 (unchanged), constructed in 13.578 seconds.
The named template keeps nineteen physical guard slots for compatibility;
seventeen are proved constants and only E2/E3 are nontrivial. These results
and all source hashes are saved under experiments/results/simplification_2026_09_16/.

ROTOR-NORM ABLATION: the exact identity U_i^2+W_i^2=a_i^2 allows replacing
verified 1+V_i^2+W_i^2 nodes by 1+a_i^2 before denominator clearing. New
exact_rotor_simplify.py implements this with domain and source-pattern checks;
synthetic identities and negative controls passed. It is correct, but its
materialized k=1 artifact grew to 473,588 arithmetic nodes and 4,448,047 bytes,
with 38,507 atoms and degree upper 60,416. This path is retained as an audited
alternative, not selected as the smaller representation.

SELECTED REPRESENTATION: use the prior-specialized structural template with
the unchanged original rotor binder, then remove only accepted literal-constant
guard entries. prior_structural_plain_full18_1.polynomial.json.gz retains all
eighteen unknowns, the exact original prior, the four strict later-exit tests,
eta>=0 and all four observation bounds. It has 416,712 arithmetic nodes,
58,712 sign nodes, 38,507 polynomial atoms, nine sign-conjunction entries plus
36 coordinate-bound comparisons, and 4,083,364 compressed bytes. Construction
took 13.094 seconds; this is not an inverse runtime. Its SHA256 is
1e2f27f1635c6caa91629c1eaac346938c3adbeb2355cadf341006a763b0ccb9.
selected_result.json records the selection and comparison. Relative to the
frozen full18 k=1 artifact, atoms fall 9.5%, arithmetic nodes 2.2%, compressed
bytes 2.6%, and redundant top-level entries fall from 43 to 9. The maximum
syntactic degree upper bound remains 60,416. No degree or inverse-speed gain
is established. Only one symbolic sample was materialized, with no data.

PRESERVED UNSUCCESSFUL VARIANT: exact_distributive.py expands squares of at most
four terms and flattens scalar-times-sum expressions. Eleven free-polynomial
identities, three cancellation controls and four input checks passed. Its one
optical symbolic construction hit 500,000 arithmetic nodes at the threshold
comparison after 30.172 seconds. No complete output was produced; the report
and logs are retained, without increasing the cap or rerunning it.

WHAT REMAINS: structural analysis attributes almost all degree growth to the
observation threshold expression (axis degree upper 17,408), while physical
guards were at most 512 even before simplification. Generic expansion and the
rotor-norm shortcut did not remove this bottleneck. The next substantial
reduction must exploit the threshold/algebraic structure before repeated norm
elimination, with the complete full18 inverse decision and accuracy question
still required. No case recovery, global uniqueness, practical all-case runtime
or additional archived-failure resolution follows from this pass. See
paper/EXACT_SIMPLIFICATION_PROGRESS.md for the retained representation and
the measured tradeoffs.

Final independent selected-artifact audit passed in 2.490 seconds, saved as
exact_plain_binding_audit.json with a reproducible Python checker. It verifies
source/input/output hashes, original prior and all18 variables, all nine exact
conjunctions, thirty-four accepted constant removals, correct observation/beam
axes, root-free coefficient grammar and complete dependency reachability after
pruning. Both selected and norm-shortcut artifacts pass their respective audits;
only the smaller plain binding is selected. Read-only final validation also
checked forty-two source/input hash bindings and four completed output hashes,
then the final selected-audit source hash and updated document links. No
additional construction or optical evaluation was used for these checks.

## 2026-09-16: Executable exact full18 decision queries; no optical decisions run

The user requested substantive progress toward resolving every failure. This
pass kept the standing prohibition on population replay and individual case
hunting. Work comprised exact symbolic construction, scalar algebra controls,
synthetic real-algebraic solver decisions and read-only source/artifact audits.
No optical forward evaluation, archived record, measured observation, inverse
optimizer or full optical satisfiability decision was used.

LAST-INTERFACE ELIMINATION: exact_threshold_axis.py constructs the observation
threshold without the third exit root E. Writing C=E*P+A*D*u*L and
E^2=r*H-A^2 gives the exact conjugate norm r*N, where
N=A^2*((q*D*L-Q*K)^2+(G*D*L-V*K)^2)-H*P^2. Positivity of r,D,L leaves an
exact ternary sign rule in sign(A*u), sign(P), sign(N), including all zero
cases. The strict third-exit radicand condition remains. This is eight
constructed roots per axis, with no physical unknown removed. Six exact
free-polynomial identities and 75 scalar sign controls (eleven exact zeros)
passed. One capped compile completed in 13.922 seconds, producing 119,223
reachable arithmetic nodes, 14,945 sign nodes, 9,861 polynomial atoms and
degree upper bound 16,896, versus 17,408 for the previous axis template.
Arithmetic nodes and compressed size increase slightly: this is a modest
representation improvement, not an inverse speed result. The 1,177,406-byte
artifact SHA256 is
92601bfadbdfdb3c650ca671fa8dd6f8c05671372dfccf6a7c5fe3a8237c90fd.
structured_threshold_audit.json independently verifies the source/formulas,
coefficients, dependencies, remaining guards, counts and hashes without
recompilation. Proof: paper/STRUCTURED_THRESHOLD_ELIMINATION.md.

SPARSE OPTICAL EQUATIONS: exact_smt_graph.py exports the same strict physical
branch as polynomial real constraints with explicit ray auxiliaries. Its
symbolic k=1 export has 18 physical unknowns, 30 trace auxiliaries and seven
shared constants/data symbols, 457 arithmetic definitions, 116 assertions and
30,303 bytes. Polynomial degrees after definition inlining are at most 14 for
the optical constraints and 20 including the prior constants. This does not
give an eighteen-variable degree-14 representation: the trace auxiliaries
are deliberately retained. They are determined by the positive-root branch.
The exporter supports more exact grid indices; only symbolic k=1 was exported
and checked here. One sample is not claimed to identify eighteen parameters.
Evidence: experiments/results/exact_smt_2026_09_16/graph_check.json.

NATIVE ACCURACY: exact_native_accuracy.py converts all eighteen native-unit
comparisons exactly, without floating trigonometric thresholds. For an angle
threshold p*pi/q it defines tan(p*pi/q) by the imaginary part of (1+i*T)^q
and a strict rational isolating interval. A repeated-squaring complex circuit
uses O(log q) auxiliary variables with quadratic constraints. Machin bounds
for pi, rational Taylor enclosures and finite-root spacing greater than 3/q
select the intended root. Four small independent Sturm counts, known tangent
values, all18 coordinate comparisons and strict/closed boundary controls
passed. The illustrative angle pi/180000000 uses 76 auxiliaries; its algebraic
degree has not thereby become small. Isolation has an explicit 256-term
budget and can fail for extreme exact inputs. The surrounding caller requires
the trusted original prior for both parameter vectors. Caller-supplied rational
tolerances are a convention; 1/1000 is not silently equated with the binary64
literal .001 used by an archived computation.

EXACT DECISION BACKEND: installed official z3-solver 5.1.0.0 into the isolated
ignored .codex-deps/z3_exact directory. The wheel origin and SHA256 are saved in
experiments/results/exact_backend_2026_09_16/z3_install.json. New
exact_real_backend.py restricts inputs to polynomial real arithmetic and uses
exact NLSat or quantified NRA, with exact zero handling and bounded hidden
worker processes. The original 12 synthetic decisions and nine invalid-input
controls passed in 3.328 seconds. A subsequent resource-error classification
fix has its own final-source-bound seven controls and four regressions; the
original evidence remains intact. Resource exhaustion returns UNKNOWN, never
an impossibility conclusion. SAT values are exact rationals or real algebraic
root representations. UNSAT is currently trusted Z3 output: the proof-enabled
probe did not supply an independently checkable proof. This is not a Lean-
verified decision backend, and synthetic timings give no optical runtime.

COMPLETE QUESTIONS NOW IMPLEMENTED: exact_recovery_queries.py builds a separate
compatibility query and an exists-compatible-estimate / forall-compatible-truth
native accuracy query. All18 estimate coordinates remain free existential
unknowns. Every competing physical coordinate and every competing trace
auxiliary is universally bound. Definitions are local nested lets; shared
algebraic caps are anchored by the positive compatible-estimate conjunct.
Independent source reviews confirmed this scope, the exact native comparisons
and strict boundary behavior. The classifier requires feasibility SAT before
reporting that accuracy is impossible for a compatible estimate, so an empty
solution set cannot produce a vacuous accuracy claim. Any resource failure
remains unresolved. Exact transformed models carry explicit inverse-atan
expressions for the native angles and speeds. No rounded-output guarantee,
arbitrary-center minimax claim or independently checked obstruction certificate
is inferred.

The saved symbolic full optical queries were parsed, polynomial-type checked
and scope checked in 0.390 seconds, with ZERO solver.check calls. The feasibility
query has 55 declared constants and 116 assertions. The accuracy query has
276 declared constants, 228 assertions and one universal quantifier binding
48 variables (18 competing physical plus 30 trace auxiliaries). It retains
all18 free estimate coordinates, with no leaked free competing variable. Its
illustrative native tolerances are exact 1e-6; this does not change the archived
success criterion. No observation values were supplied. Full evidence and
source/output hashes: experiments/results/decision_2026_09_16/
recovery_queries_check.json and native_query_proof_review.json.

STATUS: there is now executable code connecting the full physical equations,
native accuracy predicate, exact decision backend and exact model extraction.
No actual optical inverse query has been executed, no additional archived
failure is resolved and no practical all-case runtime is established. The
universal observation-space partition and independently checkable decision
certificates remain uncomputed. The next substantive barrier is tractable,
verifiable optical decision, not another declaration that decidability alone
solves the archive. See paper/EXACT_DECISION_ENGINE.md for the full contract and
the distinction between implementation, checked syntax and observed recovery.

The final tangent-to-backend integration controls passed eight synthetic
decisions in 1.297 seconds (four SAT, four UNSAT, no UNKNOWN). They verify the
generated constant circuits for tan(pi/4)=1 and tan(pi/6)=1/sqrt(3), exact model
extraction, exclusion of wrong defining facts, and strict versus closed
comparison at exact equality. Each worker had a three-second hard cap. The
complete queries, answers and final source hashes are saved in
decision_2026_09_16/native_tangent_backend_check.json under experiments/results/.
No optical graph, observation value or archive was used by these controls.

Final read-only verification passed forty source/input hash bindings, six
completed-output hashes, Python syntax for eighteen sources, and fifty-eight
relative document links. The previously selected full18 artifact is unchanged.
Evidence: decision_2026_09_16/final_provenance_check.json under experiments/results/.
This verification ran no compiler, algebra control or decision solver again.

## 2026-09-16: Research handoff requested

Created HANDOFF.md at the user's request. It records the full18 objective,
standing stop on population/targeted recovery reruns, current archive evidence,
the exact decision implementation's untested optical status, input/provenance
obligations, and the proposed priority of a useful model-specific global bound.
The user challenged the prior pass's lack of recovery progress; the agent
acknowledged that infrastructure and modest algebraic reductions did not deliver
new recoveries or population-wide failure bounds. The handoff preserves this
distinction for the successor. All nineteen local links were checked. No new
research computation, experiment authorization or external task was created.

## 2026-09-16/17: Outside mathematical attack; reverse optics and exact affine fibers

The user requested a powerful outside perspective and a theory/algorithm for
the entire full18 inverse. Read HANDOFF and the latest log, then split three
independent investigations across global geometry, optical structure and
observability. Primary literature was checked for structural identifiability,
algebraic solving, Prony collisions, set-membership inversion, optimal recovery,
Hamiltonian optics, conic relaxation and layer stripping. No stopped population
or individual recovery run was restarted. Work used mathematical derivation,
exact scalar/polynomial controls, synthetic exact decisions and read-only saved
evidence. No new optical scan or inverse optical decision was evaluated.

REVERSE OPTICAL RESULT: OUTSIDE_OPTICAL_STRUCTURE_2026_09_17.md derives a reverse
angular relation with d=sqrt(n^2-h^2), d^2>=69/100 on the original prior. Critical
transmission becomes the strict guard chi>0; chi is not a reverse radical or
denominator. A global secant bound, bounded rank-one Hessian and exact convex
hull of this one root graph are proved. The latter uses a second-order cone
plus a four-corner lower hull. The synthesis proves the uniform vertical
relaxation gap <=(42761/40960)*(width_n^2+width_h^2). This is a bound on a root
relaxation, not the whole optical feasible set or full18 inverse error. A
quadratic bidirectional prism block retains backward direction constraints and
forward position propagation with denominator td>=(4/5)*delta_i. The exact
symplectic identity A*f'=1 links position gain and ray angular derivative.
Outgoing angles remain unobserved latent variables; no global confinement or
recovery follows yet. Nine polynomial and four rational controls passed.

AFFINE GEOMETRY RESULT: OUTSIDE_GLOBAL_GEOMETRY_2026_09_17.md uses strictly
positive source-position coefficients to eliminate both unknown source offsets
exactly, giving a polygon in the shared (d_W,gap) plane plus full reconstruction
intervals for both offsets. It derives the exact deterministic infinity-norm
affine-fiber residual and its planar convex minimax formulation. All18 unknowns
remain represented; fixed-q polygon exclusions must be verified across a whole
nonlinear region to give global exclusions. One polynomial identity, nine
rational bounds and 100 synthetic scalar minimax controls passed and are saved.

FORWARD-BOUND OBSTRUCTION: the same note constructs an explicit compensated
weak-first-prism family approaching downstream critical transmission, inside
the strict original prior. Its gain/glass mixed derivative diverges. A review
caught and fixed the state definition: use the second prism's fixed flat entry,
not the first tilted exit without its changing height. The result rules out
one uniform forward mixed-derivative/Hessian bound over this prior. Continuity
extends the divergence to arbitrarily small nonzero first wedges; merely
removing the exactly flat stratum does not fix it. Compact regular-family
bounds remain valid.

ACCURACY DECISION RESULT: OUTSIDE_CENTER_GEOMETRY_2026_09_17.md replaces the
simultaneous exists-center/forall-compatible-system question by coordinate
extrema/attainment followed by one compatible-center feasibility query. For
strict native tolerance, center interval endpoints are excluded exactly when
the corresponding opposite extrema are attained. There are 18 projections or
36 global extremal problems, NOT 36 guaranteed SAT calls. Those projections
remain the global bottleneck. A noncompact counterexample proves finite witness
cuts need not terminate at a strict-boundary obstruction. 24 synthetic exact
decisions passed. Two earlier input-typing failures and their source versions
are retained; no optical equation or observation was used in these controls.

WEAK-PHASE RESULT: OUTSIDE_OBSERVABILITY_2026_09_17.md proves an explicit
all-time obstruction with all parameters unknown: if all wedges <= 2 degrees,
both beam angles <= 5 degrees, and one wedge <= .008 degrees, an inward phase change
of .003 degrees alters either ideal output by less than 20449/2067187500 < 1e-5 at
every time. At ideal y=F(theta), eta=1e-5, any estimator has worst native phase
error at least .0015 degrees. Uniform physical margins are proved throughout the
subfamily, with exact response constants 1352, 1208, 1043. The .005 degree version
has more discrepancy slack. Independent optical review and exact Fraction
controls passed. This is an infinite-family finite-noise bound, not a noiseless
failure explanation. A read-only tally of all 18402 saved classifications found
6 admissible specifications within the 2 degree/5 degree regular subfamily and
ZERO satisfying its weak-wedge condition. No archive classification was added.

Synthesis and recommended mathematical architecture:
paper/OUTSIDE_SYNTHESIS_2026_09_17.md. Evidence and reproducible controls:
experiments/results/outside_2026_09_17/. The original physical model, frozen
solver/decision sources and manuscript were not modified. Recovery counts
remain 504 saved and 15255 admissible without saved recovery. A practical all-case
inverse, global confinement of the nonlinear coordinates and a new noiseless
archive resolution are still unproved. HANDOFF now links these developments.

NONUNIFORM EXTENSION, SAME PASS: OUTSIDE_PHASE_ENVELOPES_2026_09_17.md and
experiments/outside_phase_envelopes.py extend the weak-phase proof to independent
wedge envelopes and a beam envelope. The direction bound uses the monotone
corner of the input/normal/index box, with index monotonicity invoked only
after maximizing the normal to its nonnegative corner. Exact dyadic root
brackets, rational sine bounds and finite-difference recurrences prove physical
margins and all-time response constants. No selected optical trajectory is
simulated. Unclosed envelopes return inconclusive, not physical exclusion.
The final 100 synthetic scalar controls bind source/proof hashes; the earlier
pre-proof report and its source snapshot remain preserved.

Only after freezing that source, a read-only full sweep of the 18,402 saved
classifications checked all 15,759 admissible specifications. It obtained
11,605 all-time envelope certificates and 4,154 inconclusive envelopes. The
phase change .003 degrees supplies a worst-case native phase floor .0015
degrees at ideal y=F(theta), eta=1e-5, for 444 specifications. Of those, 443
overlap the previous saved finite-noise obstruction flags. The source formula
was not changed or tuned after reading these results. The sweep took 17.015
seconds for scalar inequalities and input processing; this is NOT an inverse
solver timing. No scan generation, truth regeneration or optical inverse
decision occurred. Report: outside_2026_09_17/phase_envelope_archive_audit.json
under experiments/results/, with exact cap/constant/bound records and hashes.

These are population-linked analytic results on ideal mathematical outputs,
not 444 new recoveries or original-observation classifications. No new bound
between those ideal outputs and original or regenerated floating observations
was established. The actual-observation count 6,701 and saved recovery count
504 remain unchanged; nonzero-wedge noiseless failures are not explained by
these finite-noise obstructions.

Independent read-only phase-envelope review passed 33,081 checks with no
findings: all 444 retained rows/source lines, exact binary64 angle caps, full18
original prior, inward phase alternatives, rational bound arithmetic and
both rational/binary64 .001 tolerance comparisons. It independently recomputed
the Machin pi enclosure used in the bounds; it did not rerun optical scans or
the envelope recurrence. The 11,605 aggregate envelope results and stored M
coefficients are provenance/consistency checked, not independently recomputed.
Evidence: outside_2026_09_17/phase_envelope_readonly_review.json under results.

## 2026-09-17: Executed mathematical attack; contraction theorem, sharp limit, exact ambiguity

After discussing a staged attack, the user authorized execution and parallel
agents. Three mathematical agents and the coordinating agent worked on state
contraction, global exclusion, exact ambiguity and structural alternatives.
The standing stops were preserved: no optical evaluations, inverse decisions,
new archive reads or recovery campaigns. The synthesis is
paper/ATTACK_SYNTHESIS_2026_09_17.md.

STATE CONTRACTION: paper/ATTACK_STATE_CONTRACTION_2026_09_17.md supplies an
explicit interval operator whose computed ray-state and affine-coefficient
widths contract when the fourteen nonlinear physical coordinates shrink. It
needs no physical midpoint or truth initialization, and no independent ray
subdivision. All four affine physical coordinates retain their complete fibers.
Only their own shrinkage would force the full position intervals to shrink.
The uniform finite-horizon rate is O(rho^(1/4)); an exact double-critical
construction proves this exponent is optimal even for measured positions and
strictly physical epsilon versus 2epsilon pairs. Raw reverse-root hulls can
retain a false root at fixed parameters; an explicit counterexample shows why
the shrinking state bounds are necessary. Sixty-eight exact scalar controls
passed; an initial cap-saturation assertion failure is preserved with its source.

STRONGER PRIOR MARGINS: paper/ATTACK_PRIOR_MARGINS_2026_09_17.md improves the
first transmitted-root bound to chi1>1/5 and vertical air bounds to
(1/2,1/12,1/440). First-exit regularity was known earlier with weaker constants.
No later critical-root margin is assumed. Twenty-six exact scalar checks and
independent mathematical review passed. These constants improve enclosures,
not demonstrated inverse performance.

GLOBAL RESIDUAL: paper/ATTACK_GLOBAL_EXCLUSION_2026_09_17.md combines signed
source-eliminated inequalities with a compact four-variable affine LP fallback.
The regional lower-bound error is controlled by C_F(T)*rho^(1/4), including
zero source gains at the closed critical boundary. There are 106 exact
synthetic/scalar checks. Review corrected the finite-covering scope: forbidden
regions must retain their entire affine rectangle, and classifying wholly
nonphysical cells has an extra cost not bounded by the residual gap alone.
Earlier notes/reports and both corrections are preserved with source hashes.
The illustrative T=10, gap=1 sufficient uniform partition is 2^2058, about
10^619.5 cells. This conservative upper-bound calculation fails to provide a
practical guarantee; it is not actual runtime or a lower bound on adaptive
work. Positive-gap conditional termination remains valid with its domain and
strict-boundary obligations. Practical global confinement is still missing.

EXACT NOISELESS AMBIGUITY: paper/ATTACK_EXACT_AMBIGUITY_2026_09_17.md derives a
stationary-first-prism/source gauge. Three first-prism shape/index coordinates
vary while exact beam angle/position compensation preserves the ray entering
prism two. An explicit fullprior family has all wedges between1 and2degrees,
downstream speeds1 and3Hz, and an exactly identical nonconstant all-time scan.
Endpoints with first wedge1 versus2degrees force a worst-case error at least
0.5degrees already at eta=0. A stationary prefix of k prisms has at least3k
local source-compensated directions under the stated interior/strict conditions.
This is a stationary-prefix stratum, not ambiguity among the locally injective
archived truth centers. Seventeen polynomial and23 rational controls passed.

STRUCTURAL ALTERNATIVES: paper/ATTACK_RAY_COORDINATES_2026_09_17.md derives
regular face/transfer formulas from incoming/outgoing rays using sigma=xu+Bt,
with n^2-sigma>=39/100. Six reference ray angles replace six physical angles
off zero-wedge strata; all18 parameters must still be reconstructed. One
regular reference cannot remove critical projections at other samples. Thirteen
symbolic/scalar checks and two independent reviews passed. A useful global
collection of such charts remains unproved.
paper/ATTACK_BRANCH_VISIBILITY_2026_09_17.md proves that with fixed upstream
state and an independent final rotor, the last position generates exactly the
last transmitted root's quadratic extension. A simple complex zero proves
nonsquareness even after adjoining both quadratures and both projected normal
cosines. Seventeen polynomial and10 rational checks passed; an initial scalar
test-construction TypeError and its source were preserved. Independent
function-field review passed. This does not recover the unknown upstream field,
resolve collisions, or imply uniqueness from finite samples.

NEXT OBLIGATION: observation-dependent signed separators across substantial
parameter regions, or a finite-data structural separation theorem, must supply
the global confinement that the conservative modulus cannot make practical.
The new proofs are not full Lean formalizations. No recovery or original-data
classification counts changed:504 saved recoveries,15255 admissible without a
saved recovery. No practical all-case full18 inverse is claimed.

## 2026-09-17: Continued attack; observations exclude critical optical layers

The user requested continued attack and deferred T10 until T1-T9 are complete.
Three mathematical agents and the coordinating agent worked in parallel.
No optical evaluations, inverse decisions, new observation fixtures or archive
classifications were produced. Synthesis: paper/ATTACK_CONTINUATION_2026_09_17.md.

OBSERVATION-FORCED MARGINS: ATTACK_OPTICAL_SEPARATORS_2026_09_17.md proves on
the original full18 closed transmitted relation that |Y_axis|<=50 implies
chi2>3/3250 and chi3>1/25 at that sampled ray. All eighteen coordinates remain
unknown; signed internal propagation is retained. Near-critical source gain
vanishes while positive gap/workpiece terms force a large displacement. This
is a global exclusion before finding an inverse. The measured condition
|y|+eta<=50 is sufficient; apply it to every sampled coordinate for uniform
sampled margins. Unsampled times and arbitrary recentering are not covered.
No observation array was checked here. Forty-eight exact controls passed and
three independent mathematical reviews found no issues.

CONTRACTION AND CLOSURE: ATTACK_OBSERVATION_CONTRACTION_2026_09_17.md uses
these data-implied margins to give O(rho) computed coefficient widths instead
of the unrestricted fourth-root rate, preserving all four affine fibers.
Its sampled domain is compact and strictly physical. If a proposed output's
strict native .001 box contains all compatible systems, every nonempty closed
coordinate-violation region has an attained positive residual gap. Critical
closure ghosts cannot cause the earlier zero-gap pathology on this class.
No useful gap size or tolerance-equality decision follows. The conservative
T=10 constant is approximately 9.44439348241222e14; illustrative gap 1 gives
level 51 and 2^714 cells instead of old level 147 and 2^2058. This sufficient
partition is still impractical, and additional physical-domain classification
cost remains unbounded. These are not measured times or lower bounds.
Five hundred sixty-four exact controls passed. An independent 47-check review
binds final proofs/reports and verifies constants, counts and compactness.

FINITE-LAG EXCLUSION: ATTACK_OBSERVATION_SEPARATORS_2026_09_17.md proves the
increment bound 2*sum(V_i*|tan ax_i|*|sin(pi*N_i*tau)|). With all nuisance
priors and sum|tan ax_i|<=.08, returns within .001 turns force increments below
.322212. Larger noise-adjusted data increments exclude the entire speed/wedge
region through a three-variable LP. No observation was used to choose the
constants. Two hundred twenty-five exact controls passed. Review corrected
one strict sign at zero speed; the initial proof/report and corrected binding
are preserved. This is a necessary constraint, not complete localization.

FINITE-GRID OBSTRUCTION: ATTACK_FINITE_DATA_2026_09_17.md fixes speeds 10/3 at
exact t_k=k/20. Six rotor configurations compress the record to 12 outputs,
while an explicit strictly physical 13-dimensional nonspeed cube has radius .01.
Borsuk-Ulam gives systems with equal entire grid scans and native separation
.02, hence arbitrary-estimator worst error at least .01 at some noiseless data.
All wedges lie between .99 and 1.01 degrees, all speeds are nonzero, and the
cube's scans are nonconstant. The pair is not located or evaluated. Seven-state
variants and generic local fibers are also derived. Literal binary64 timestamps
have a separate finite-noise transfer: midpoint allowance<4.506e-12 or clean
endpoint allowance<9.012e-12. Floating optical outputs still require their own
discrepancy certificate. Nearby distinct speeds receive a finite-noise corollary
only. One hundred thirteen scalar controls and independent proof reviews passed.

OUTSIDE THEOREM: ATTACK_GENERIC_SAMPLING_2026_09_17.md applies Sontag's analytic
identification theorem to 37 generic distinct times on a connected full18 family
with a common strict analytic interval. It distinguishes whole traces modulo
all-time equivalence; no given grid, parameter uniqueness or noise tolerance
is certified by genericity. Primary-source reviews passed. Sampling redesign
remains optional; the original observation contract is unchanged.

STATUS: T5 has substantive large-region exclusions, but complete confinement
is missing. T6 resolves one boundary issue on a declared observation class and
adds ambiguity, but complete exceptional decisions remain missing. T7 has cheap
tests and a better exponent, but no useful all-case computation. Next attack:
regular non-returning compatible regions surviving these constraints. T10 is
deferred. Counts remain 504 saved recoveries and 15,255 admissible specifications
without a saved recovery. No complete inverse or new population result is claimed.

## 2026-09-17: Regular-region attack; guarded quartic, complete affine elimination, cubic noise limit

The user requested more mathematical attack toward closure. Three independent
agents and the coordinating agent worked on optical monotonicity, coupled
global inequalities, exact decision boundaries and a new noise obstruction.
No optical evaluation, inverse decision, archive input read or recovery campaign
occurred. T10 remains deferred until T1-T9 are complete. Synthesis:
paper/REGULAR_ATTACK_SYNTHESIS_2026_09_17.md.

TERMINAL INVERSE: REGULAR_OPTICAL_BOUNDS_2026_09_17.md proves a positive terminal
slope derivative dY/dm>=144G/2125 when |Y|<=G, for fixed incoming momentum,
position, index and workpiece distance. The exact guarded polynomial has degree
at most four and at most one physical root in that band. Compatible scalar
slope width is at most85/288 times output-interval width. This conditional
optical inverse retains the unknown upstream arguments and all shared rotor
constraints; it does not turn the full18 inverse into a12-dimensional problem.
A separate observed |Y|<=50 bound forces final vertical direction>sqrt(21)/11,
about.4166, instead of prior-only1/440. Forty-five exact controls and three
independent mathematical reviews passed without substantive findings.

EXACT AFFINE FIBER: REGULAR_COUPLED_EXCLUSION_2026_09_17.md appends four identity
prior rows to the exact scan matrix. Every five-row signed cofactor inequality
is necessary, and together they are sufficient for existence of all four
affine coordinates within their original priors/noise bands. The augmented
matrix has rank4 even when the optical rows lose rank. Smaller circuits extend
to these cofactors. This applies Rockafellar's classical elementary-vector
principle; generic affine elimination is not presented as newly invented.
Native Cramer/noise inequalities and a sharp synthetic moving-cofactor example
are proved. Its global residual is1/6, and allowance1/8 excludes its entire
nonlinear interval while every fixed output hyperplane fails. This does not
demonstrate equivalent global optical confinement. All14 nonlinear coordinates
remain free.113 exact controls and independent reviews passed. One determinant
orientation wording clarification preserves the initial proof and report.

DIRECT PHYSICAL EXCLUSION: REGULAR_SIGNED_CONE_2026_09_17.md proves on each
same-sign beam/normal region that V=sigma*(x0+sum((n_i-1)*s_i)) satisfies
sigma*Y>=-5+50V/sqrt(1-V^2). This gives a quadratic necessary condition directly
in beam, glass, wedge, phase and speed coordinates, preserving all geometry
and source-position priors. Unknown-sign complementary regions remain.
An explicit ordinary full18 region with wedge tangents[.12,.125], small beam
components, and all remaining original priors is uniformly strictly physical
at every time, yet has Yx(0)>.025. A retained yx(0)+eta<=.025 excludes the whole
region without fitting. No observation was read.32 exact controls, root and
independent mathematical reviews passed.

CUBIC NOISE OBSTRUCTION: REGULAR_CUBIC_NOISE_2026_09_17.md fixes a centered
vertical source as a witness family, with first wedge<=.9degrees and other
wedges<=2degrees. Every member has a fullprior alternative changing first
index by.003 and first wedge to hold(n-1)sin(alpha) fixed. The exact directional
derivative is -s^3*(n+1)/(c+chi)*(1+n*s^2/(c*chi)). Its linear response cancels;
the surviving cubic term and exact downstream secant inequalities give the
alltime position difference bound47948497487161/5762400000000000000<8.322e-6.
At ideal y=F(theta), eta=1e-5, both explanations are compatible and every
estimator has worst native error>=.0015. All speeds may be distinct/nonzero;
all starting wedges may be>=.5degrees, and the altered wedge remains>.49.
Noiseless equality is not asserted. Arbitrarily long records cannot defeat
this deterministic noise obstruction. Floating outputs still need their own
certified discrepancy allowance.39 exact controls and three independent
reviews passed; no archive population coverage was measured.

DECISION AUDIT: REGULAR_DECISION_THEORY_2026_09_17.md credits existing exact
equality-inclusive algebraic decidability and coordinate-center geometry.
Compact moderate compatible sets have at most36 attained extrema witnessing
all arbitrary accurate centers, but finding them is a global problem. A new
identity-map counterexample proves arbitrary center/counterexample iteration
can continue forever at strict tolerance equality. The valid schedule4^-k
replaces an explicitly rejected2^-k scratch schedule. A credited weak-phase
corollary, with an alltime output bound below5, rules out uniform positive
parameter separation on the regular moderate distinct-speed class. No new
finite200-sample completeness or useful Pfaffian bound is claimed.115 exact
controls passed; the initial syntax failure/source and metadata-check failures
remain documented, with no failed optical experiment hidden or rerun.

EVIDENCE AND STATUS:344 primary exact symbolic/scalar/synthetic controls passed;
independent proof and source-binding reviews are separate. New audit artifacts
live under experiments/results/regular_attack_2026_09_17/. Before updating the
living handoff, diary, roadmap and existing target display, their exact prior
versions were snapshotted so preceding audit hashes remain reproducible.
No full Lean formalization or new optical benchmark was performed. T5 gained
exact complete affine inequalities and an ordinary optical-region exclusion,
but useful complete nonlinear confinement is still missing. T6 has additional
obstructions and a sharper termination requirement; T7 remains without useful
all-case execution. T4/T8/T9 remain unfinished and T10 deferred. Historical
counts remain504 saved recoveries and15,255 admissible specifications without
a saved recovery. No complete inverse or new archive resolution is claimed.

## 2026-09-17: T5-only sprint; fixed completion gate not passed

The user required one existing target per sprint with fixed acceptance criteria,
then authorized continued work on T5. The sprint did not close T5, and no other
remaining major target was crossed off. The verdict is recorded directly in
paper/T5_SPRINT_VERDICT_2026_09_17.md. T10 remains deferred. No optical model
evaluation, inverse decision, observation-array read, archive input or recovery
campaign occurred. The original full18 prior and strict native .001 target
remain unchanged.

AFFINE EXTREMUM COVER: T5_AFFINE_EXTREMUM_COVER_2026_09_17.md proves that every
one of the 36 attained native coordinate extrema has a witness at a vertex of
the four-affine fiber. Signed four-active-band Cramer charts cover those
vertices and the entire nonlinear feasible projection. A division-free compact
completion replaces selected bands by exact endpoint faces and applies the
complete cofactor criterion; rank-zero chart points retain genuine feasible
fibers. This requires a compact compatible set, supplied by the moderate
observation class, and is not an attained-extremum theorem on arbitrary open
physical domains. Exact synthetic examples show active determinants can tend
to zero despite identity augmentation, continuous affine selectors can fail,
and abnormal stationarity can retain whole singular feasible sets. The atlas
does not compute the nonlinear extrema. Eighty-two exact controls passed;
independent mathematical review found no substantive issue.

TERMINAL ROTOR: T5_TERMINAL_ROTOR_REDUCTION_2026_09_17.md reconstructs the last
rotor conditionally from its exact normal sequence. It retains zero wedge's
entire speed/phase fiber. At nonzero wedge, two exact grid normals determine
the rotor triplet subject to all later recurrence and prior tests. With noise,
at fixed speed the initial normal lies in two planar convex intersections of
rotated rectangles and signed disk sectors. Observation-band clipping is
essential; a root at the measured center need not exist. Fifteen outer unknowns
remain. The transformed Jacobian is a positive row rescaling of the original,
so conditional twist does not prove upstream identifiability. An irreducible
coefficient-specialized quartic also rules out a universal rational quadratic
factorization shortcut. Eighty exact controls and independent review passed.

BUFFERED EXCHANGE: T5_BUFFERED_EXCHANGE_2026_09_17.md proves that inner radius
.0009 and strict outer radius .001 require at most 325 counterexample additions,
retaining at most 36 witnesses. Complete global center and witness queries are
prerequisites. Inner-radius failure is not a .001 obstruction. A two-sided
scheme eventually decides strict accuracy away from exact minimax equality on
compact compatible sets, conditional on terminating global queries. Equality
and useful inner cost remain unsolved. Even exact empirical minimax centers
can loop forever at equality. The 180 exact synthetic controls and independent
reviews passed; no optical query was executed.

FINITE-HORIZON ROUTE: T5_FINITE_HORIZON_2026_09_17.md tests polynomial/Nash
observability as an outside route. A fixed-reference ideal can plateau and
then grow; an exact two-state counterexample prevents using that plateau as
a stopping certificate. Paired-state pullback invariance is the sound repair
in a preserved Noetherian function ring. No 200-sample optical horizon follows.
A fixed compact synthetic rotating system with a principal-root output has
arbitrarily delayed information despite strictly positive roots at every
integer sample and uniform prefix margins. It is not an optical realization;
the common invariant analytic domain is precisely the missing hypothesis.
Sixty-eight exact controls and independent reviews passed.

EVIDENCE AND VERDICT: 410 primary exact controls passed. These are symbolic,
scalar, finite-field and synthetic checks, not an optical benchmark or full
Lean formalization. The prior living status files were snapshotted before
updating the handoff, roadmap and existing target display. The sprint audit
binds final sources and the preserved preceding audit artifacts. The unresolved
core is a certified global nonlinear feasibility/witness and boundary mechanism
with useful cost. Another conditional inverse or excluded region cannot count
as T5 completion. Counts remain 504 saved recoveries and 15,255 admissible
specifications without a saved recovery.

## 2026-09-17: Full-power T5 attack; finite witness construction replaces the oracle

The user asked for a sustained full-power attack until significant breakthroughs.
Three parallel mathematical agents and the coordinating agent developed and
independently reviewed a constructive global route. Fixed target T5 remains
open: no optical global decision or useful execution bound was produced. No
remaining major target was crossed off, and T10 remains deferred. Read
paper/T5_FULL_POWER_VERDICT_2026_09_17.md. No optical evaluation, inverse decision,
observation array, archive input or recovery campaign ran. The native strict
.001 tolerance, full18 unknowns, full original priors and physical order remain.

FINITE ACTIVE-BLOCK CONSTRUCTION: T5_GLOBAL_FEASIBILITY_ATTACK_2026_09_17.md
replaces the previously unspecified multivariate feasibility oracle in theory.
Primary positive infinitesimal floors preserve every strict condition;
independent smaller constraint-level perturbations give regular active strata;
still smaller independent objective perturbations give finite saturated complex
critical systems. At an optimal four-affine vertex, at most18 constraints are
active and at most14 are additional to the affine basis. Only the selected
scalar optical blocks enter candidate generation; all retained observations,
prior rows and branch guards remain exact filters. Nine homogeneous optical
roots per scalar block give at most361 variables including multipliers and
saturation. The complex finiteness proof covers all saturated sheets, not only
isolated real physical points.

Groebner/quotient-algebra identities, trace-radical reduction, a separating
linear form, rational univariate representation, exact root/sign determination,
ordered standard parts and positive floor specialization specify terminating
operations. Original conditions are rechecked after the smaller perturbations
are removed. Full-set endpoints require a floor-limit; strict output witnesses
instead require positive floor specialization. This is a specialization of
classical critical-point methods, not a new decidability claim. The optical
engine is not implemented. A synthetic singular nonrational replay exercises
the finite root construction and a limit where raw rational substitution fails.

EXACT CENTER/EQUALITY: T5_MINIMAX_ATTACK_2026_09_17.md turns the original native
accuracy condition into exact shifted chart faces plus one common slack.
The slack is a fifth affine variable, leaving14 nonlinear coordinates. Its
positive feasibility is equivalent to a compatible accurate output; endpoint
attainment decides strict versus weak faces. At most19 constraints are active,
but an affine basis must contain one slack-dependent nonoptical row. Therefore
at most18 optical blocks and363 variables suffice. The same finite construction
handles this query. A compact disconnected regular-root example proves that
positive perturbed slack can have original limiting value zero; an open-set
example proves that accurate centers for every floor-restricted set need not
work for the full set. Exact limits and original attainment queries are
essential. The final positive slack must itself receive the strict floor.

GROUPED DEGREE BOUND: Preserving nonlinear, affine, root and multiplier groups
changes rotor-degree growth to D^14, where D=28(6K+7), rather than raising D to
the number of all auxiliary variables. With l affine coordinates and R roots,
the coefficient bound is C_l(r,R)*binomial(14+R,14)*D^14*28^R. Saturation adds
no multiplicative degree factor to this grouped count. Root and observability
independently derived the same coefficient formula and integer totals. At
K199,m400 the exhaustive upper candidate sums are about5.97e374 and9.04e377,
still unusable as practical guarantees. They are not runtimes, necessary work,
or lower bounds. Coefficient growth and algebraic limits have additional cost.

PHYSICAL DOMAIN TOPOLOGY: T5_DOMAIN_TOPOLOGY_2026_09_17.md proves that sequential
last-to-first wedge flattening retracts the strict physical prior onto its
all-flat convex subfamily. Both axes and every retained time stay transmitted;
signed propagation and all original bounds are respected. Thus the physical
domain is contractible. Compatible observation fibers need not be. Under a
matched full-rank witness, one fixed18-scalar subrecord is generically finite
to one, excluding lower-dimensional critical/prior-boundary images. This does
not prove uniqueness, classify a particular record, or transfer literal-time
rank certificates to an ideal rational grid.

NEW EXACT MOVING AMBIGUITY: T5_OPTICAL_CLOSURE_ATTACK_2026_09_17.md combines
quadrature and scalar optical reflection. Speeds(1,1,-3) on exact20Hz timestamps
give20 distinct joint rotor configurations but only10 independent scalar slots.
An11D native cube has every wedge in[.99,1.01]degrees, all-time exit roots>.99,
output magnitudes<14 and a provably nonconstant scan. Borsuk-Ulam gives an
identical-grid pair separated by.02 and native minimax floor.01. One speed
collision remains; no three-distinct-speed theorem is claimed. Strict optical
regularity is distinguished from inverse rank: these families have compatible
continua and do not classify the archived locally injective truth centers.
Literal-time transfer is only a separate finite-noise inequality, with no
floating-output discrepancy certificate inferred.

VERDICT: The theoretical oracle has been replaced by a specified finite
algebraic construction, including the final strict accuracy decision. Useful
optical confinement and execution are still absent. The next obligation is
complete support/region exclusion and degree reduction that makes this route
usable, without reintroducing guessed active sets or unexamined solutions.
All preceding frozen artifacts and prior living status versions were preserved.
Initial control failures (a structural expression comparison, an insufficient
pi lower bound, and an unavailable SymPy import) remain recorded with their
sources; no optical experiment failed or was rerun. Counts remain504 saved
recoveries and15,255 admissible specifications without a saved recovery.

FINAL EVIDENCE: The four final reports pass472 exact controls: global156,
minimax223, optical symmetry/inequality72 and topology21. Each proof received
independent review; the construction and grouped bound received multiple
independent reviews. These are exact scalar/synthetic checks and mathematical
reviews, not optical inverse validation or new Lean formalization. The final
provenance/status audit is experiments/results/t5_full_power_2026_09_17/audit.json.
The target canvas has source-level content checks only; no host-rendering or
TypeScript check is asserted. T5 remains open under its unchanged criterion.


## 2026-09-17: T5 computational attack; certified pruning and smaller ray graph

Following "lets do it", attacked the computational bottleneck with three
parallel agents and coordinating mathematical/code review. The fixed T5
acceptance criterion did not change. T5 remains open: no optical observation
received complete confinement or a global inverse decision. T10 stays deferred.
Read paper/T5_PRUNING_VERDICT_2026_09_17.md. No optical evaluation, observation
array, archive input or recovery campaign ran. Full18 unknowns, original priors,
physical order and strict native .001 deterministic guarantees remain intact.

REGIONAL COMPRESSION: T5_CONSTRAINT_COMPRESSION_2026_09_17.md and its checker
supply exact moving Farkas identities, graph-ideal witnesses and rational
Bernstein/square sign certificates. Same-axis positive source gains reduce
pairwise dominance to four geometry-corner signs. The synthetic family retains
all four affine coordinates and shared geometry; 200 scalar bands reduce to
four with392 explicit half-band deletion certificates. Compression precedes
fresh independent levels, and deleted original rows are restored after level
limits. Physical guards remain unless separately discharged. A degree-two
positive-source counterexample proves that arbitrarily many globally necessary
rows are possible in this structural class; no fixed small global coreset is
inferred for optics. Corrected final report passes464 checks.

SUPPORT FAMILIES: T5_SUPPORT_PRUNING_2026_09_17.md reduces the positive-source
four-dimensional affine objective cone to a two-dimensional geometry polygon
plus cone; center slack gives three geometry dimensions. A supplied direction
with nonpositive dot product against every allowed normal and positive objective
dot product rejects every descendant support. The finite verifier binds every
row, proves exact signs and rejects corrupted certificates. Rank/source/pair
filters precede it. One synthetic family passes rank-four and favored-source
requirements but a single cone certificate rejects1,689,061,678,433,595,784
structural supports. Report passes322 checks. The full root with both affine
prior signs cannot be rejected this way: its normal cone is the full space.
Certificates must hold on an outer region containing perturbed candidates,
not merely an original tight face; original-system compression has different
perturbation semantics.

OPTICAL GRAPH: T5_OPTICAL_DEGREE_REDUCTION_2026_09_17.md uses outgoing unit
rays and glass components, with shared first-glass and beam roots per axis.
The exact position transfer is [S(BP+3x)+G*u*N]/(t*N), with
S=B*t+x*u-1 and N=n^2-B*t-x*u>0. Position equations have no explicit rotor
variables; rotor dependence remains only in squared direction alignment.
Exact intrinsic branch selection is never independently relaxed. Complex
Jacobian saturation plus a generic-divisor incidence proof avoids V=0 only
during perturbed generation; all original zero cases are restored in limits.
Maximal auxiliary count is148 instead of162; full kernels are333/335 variables.
Observation numerator total degree11 and auxiliary degree10 are verified by
symbolic expansion. Native beam sine faces retain radians conversion, original
prior clipping and endpoint attainment. Report passes172 checks including23
polynomial identities; three independent proof reviews passed.

UNIFORMLY INACTIVE GUARDS: Existing full-prior margins give positive real
standard parts for B,C,z0,t and S1, including infinitesimally relaxed exact
prior endpoints. These guards cannot be active at the primary infinitesimal
floor. Only S2,S3 need remain in the physical active-support list, although
all guards stay enforced or proved automatic. Base potentially active rows
fall from17m+36 to4m+36, without observed-amplitude assumptions.

DEGREE GROUPS: T5_SPEED_DEGREE_2026_09_17.md separates the three speed-chart
variables from the other eleven nonlinear coordinates, reducing the K exponent
in geometric candidate counts from14 to3. Its81 exact controls pass. Combining
that partition with the new optical graph gives degree form
4K(H1+H2+H3)+8Y+10Z. At K199,m400 the exhaustive upper candidate counts become
about1.60e235 and8.04e237, versus5.97e374 and9.04e377 previously. They remain
astronomical upper counts, not runtimes, necessary work or measured speedups.
No useful all-case complexity follows.

COMPOSED EXECUTION: experiments/t5_pruning_composition_controls.py uses both
public checker APIs on one exact synthetic family. It replays392 deletion
certificates and a whole-family cone certificate on original q-region[1/4,3/4]
with outer certificate region[0,1]. The specified branch's structural count
falls from7,971,970,140,824,560,431,624 to72,411,024 after compression; one cone
certificate rejects the remaining branch. Direct signs also pass for all200
original upper rows. Eleven checks pass and independent review found no issue.
This is certificate composition, not an integrated optical inverse.

REVIEW FINDING AND EVIDENCE: The first compression run passed459 synthetic
checks, then coordinating review found that its public API did not validate
the declared nonlinear box. A forged larger box could be accepted using only
unit-box signs. Initial source/proof/report are preserved; the corrected API
rejects nonunit domains, mismatched/unbound symbols and reversed geometry
bounds, with negative tests. No failed evidence was overwritten. Five final
reports pass1050 exact controls (81+464+322+172+11); the392 certificate replays
are not separately added to that total. Prior living status files were
snapshotted before updates. Final audit: experiments/results/t5_pruning_2026_09_17/audit.json.
The target display has source-content checks only, no claimed host render or
TypeScript check. No new Lean formalization is asserted.

VERDICT: Exact reusable pruning tools and a smaller complete theoretical graph
now exist. A useful complete optical region/support cover is still missing.
The next scientific obligation is that cover and a useful bound on what remains;
guessed supports or a failed certificate cannot count as exclusion. No major
target crossed off. Counts remain504 saved recoveries and15,255 admissible
specifications without a saved recovery.

AUDIT BOOKKEEPING: The first pruning audit launch stopped before writing its
report because the support report stores passed=322 rather than a Boolean.
Initial audit source and failure metadata are preserved in its results directory.
The corrected audit normalizes that schema and still checks every result/count.
This was an audit-interface failure, not an algebra or optical experiment.


## 2026-09-17: T5 monotone-range global search; executed, obstruction measured

Following "lets see if you can crack it", selected T5 with its criterion
unchanged and attacked it by executing a rigorous global search rather than
by another candidate bound. T5 remains open. T10 remains deferred. Read
paper/T5_MONOTONE_SEARCH_VERDICT_2026_09_17.md. No archive input, replay,
recovery campaign or optical inverse on stored observations ran; forward
evaluations were of the seed-2026 battery case 0 and random synthetic boxes.

MONOTONE LEMMA: the reduced per-axis chain theta_out = arcsin(n sin(a+phi)) - phi,
a = arcsin(sin theta_in / n), has d/dtheta_in = cos b cos theta_in/(cos c cos a) > 0,
d/dphi = n cos b/cos c - 1 > 0 iff n > 1, d/dn = sin phi/(cos a cos c) with the
sign of phi. The exit direction is therefore increasing in the beam angle and
every tilt sine and monotone in each n_i with the sign of that tilt; guards
share the structure. Exact direction ranges over any native box come from two
corner evaluations per axis; wrapped rotor phases give the full tilt interval
instead of a refusal. A clipped TIR corner is NOT a bound (on the TIR surface
theta_out = 90 deg - phi is larger for smaller tilts; 26 of 78 first tests
violated); the admissible bound theta_out < pi/2 - phi_min fixed it. Positions
are affine in (beam_p, gap, d_W) with interval coefficients from exact
directions. risley_lattice/monotone.py. Validation: parity 6.8e-10 with core;
36,000 box/point containment tests, zero violations.

EXECUTED SEARCH: branch-and-bound on the full prior, per-sample exact-range
pre-test plus affine LP, split by width x sensitivity on unwrapped samples,
tol 1e-2, eta 1e-5, case 0. Four configurations (full prior, TIR-free
sub-prior, speeds fixed, speeds+phases fixed), 5,000-box budget each (30,000
in scratch): every one excludes exactly half its boxes and reaches no
survivor. Measured cluster radius of the per-sample test (smallest offset in
box widths excluded, scale invariant at widths 0.03/0.01/0.003): speeds 1-2,
wedges 4-8, phases 16 to >32, indices 16 to >32, beam angles 1-2. A cluster of
radius r in D coordinates is (2r)^D boxes per level; the search cannot
terminate at any budget.

JOINT TEST: the mean-value LP on fmodel.forward_iv's Jacobian enclosure is
REFUSED for every box with free speeds down to width 1e-2 of the prior
(phase wrap breaks the interval chain's norm/Snell margins), so it exists
only when speeds are known to about 1e-3 Hz. With speeds fixed it refuses
widths >= 0.0625, and at 0.03/0.01 has slack 3.5/0.30 pattern units (scale
about 95), excluding random offsets of 2-4 box widths but not 1;
single-coordinate radius at 0.01: wedges 1, phases 2-4, indices 8-16.

ROTOR SWEEP: a box-wise "every system TIRs at some sample" test was built
and never fires: TIR in the corner needs several rotors aligned and 200
samples of a generic orbit rarely hit it; 400 corner-biased boxes, zero
exclusions, zero false exclusions on battery truths. Wide-speed corner boxes
are over 99 percent clipped and cost 2^(3L) speed boxes per shape box.

VERDICT: for box methods the obstruction is the nonlinearity scale (about 3
percent of the prior in shape coordinates, about 3e-4 in speeds) against
fifteen strongly coupled coordinates, not the algebraic degree: joint
relaxations are informative only below that scale, non-joint tests have the
cluster radii above, so the tree from the prior down to that scale is
essentially full, of order (1/0.03)^15. Restricting to a TIR-free sub-prior
does not help. The Prony/lattice route needs under a hundred retained nodes
while admissible tails need lattice order four or more at eta 1e-5. Missing
piece, stated exactly: a joint multi-sample test informative at box widths
of one quarter to one half of the prior with wrapped phases. Candidates
(none executed): exact-range Jacobian along the monotone chain; a lifted
per-stage secant relaxation with rotor-recurrence coupling; a proved
harmonic-richness bound for admissible alternatives. Counts remain 504
saved recoveries and 15,255 admissible specifications without one.

EVIDENCE: experiments/t5_monotone_search.py (validate/search/radius/mvlp),
results and audit under experiments/results/t5_monotone_2026_09_17/,
scratch_logs for the 30,000-box runs and the abandoned clip-driven split
rule (which bisected speeds to 1/128 of range with all else at full width).

## 2026-09-18: T5 continued; shape-free ordering test for the rotor coordinates

Following "we need to keep digging", attacked the missing piece named in the
09-17 verdict (a joint coarse-scale test with wrapped phases). Work in
progress, recorded so nothing is lost. T5 remains open. No archive input,
replay or recovery campaign ran; forward evaluations were of battery cases
and random synthetic boxes only.

POSITION MONOTONE IN THE TILTS (empirical, adversarial): the full workpiece
position, not only the exit direction, is increasing in every tilt sine at
19,984 uniform admissible prior points and at 292,984 points with half of
them pushed to the lateral-heavy corners (min derivative 15.3, never
negative; the ratio to the direction term d_W dt_6/ds is 0.93 to 1.45).
Analytically dx/dm_3 = D_3 t_6' - p_5 (t_6 - t_5)/(1 - m_3 t_5), D_3 the
remaining propagation distance and p_5 the lateral exit position; the
negative part needs |p_5| of order d_W/(3 m_3), which the sampled admissible
prior does not reach. Proof obligation open.

ORDERING TEST (shape-free): with wedge signs fixed, for samples k > j the
tilt differences are A_i(cos g_k - cos g_j) = -2 A_i sin(mean) sin(half
step) (x) and 2 A_i cos(mean) sin(half step) (y), with mean = ay_i + (k+j)
w_i/2 and half step = (k-j) w_i/2 affine in (ay_i, w_i): exact ranges, so
the sign is certain even when the absolute phase is wide. If all three tilt
differences are certainly >= 0, the data must satisfy x_k >= x_j - 2 eta.
Sound: zero violations on all 30 battery truths at four widths (40 to
10,000 active constraints each). Power on random rotor boxes with every
shape coordinate at full prior (case 0): excluded 32% at speed width 0.5 Hz,
63% at 0.25, 90% at 0.1 Hz with phases 18 deg wide, 99.5% at 0.02 Hz / 2
deg; about 10 ms per box (scratch order.py, order_power.py).

BLIND SPOT (measured): the rotor-only branch-and-bound (6 rotor dims, wedge
signs of case 0 = (-,-,+)) never terminates: 100,000 boxes, 31,766
survivors, all in the region N_1 = N_2 = N_3 in [3.28, 3.5] Hz with equal
phases near 18 deg. There each prism alone has 14,894 certain pairs but the
three signs never agree, because two rotors in sync with opposite wedge
signs always predict opposite tilt-difference signs. Such in-sync
mixed-sign families (two rotors in sync, third free, is a 4-D family) carry
zero sign constraints. Closing them needs magnitudes: weighted dominance or
an amplitude LP, both requiring gain bounds G_i = dx/ds_i over the box.

GAIN BOUNDS: an exact-range Jacobian along the monotone chain was built
(scratch jac.py: stage derivatives on exact angle ranges, chain rule as
intervals; zero containment violations at widths 0.03 to 0.5 after fixing a
tan-over-pi/2 bug; midpoint equals central differences to 2e-7). Its
tightness: relative width 0.17 at 1% of the prior, 0.45 at 3%, 1.1 at 6%,
1.9 at 12%, 8 at 25%, 59 at 50%. For the tilt gains over shape boxes at
rotor width 0.02 Hz / 2 deg: no finite bound at full width (TIR), 4 to 10%
of samples finite at half width, lower bounds negative and ratios 50 to
10^12 at quarter width, ratios 4 to 26 at 12.5% width (four cases). The
product of seven exact factor ranges is inherently loose while the true
gain variation over such boxes is about a factor two or three. The existing
certified Jacobian (fmodel.forward_iv) is worse: it refuses every box whose
speed width exceeds about 2e-3 Hz and, where it works, has relative width
0.3 to 2.

STATUS: the rotor coordinates now have a rigorous shape-free test that
resolves the dominant prism at coarse scale, but its blind families make a
rotor-only search non-terminating, and every magnitude-based closure needs
tight gain bounds at coarse shape widths, which no available enclosure
provides. Next: test whether the tilt gain is itself coordinatewise
monotone (then two corners would give exact gain ranges).

CONCLUSION OF THE 09-18 SPRINT (paper/T5_ROTOR_ORDERING_VERDICT_2026_09_18.md):
the gain survey (40,000 prior points) shows the tilt gain is increasing in
d_W, in its own index and in the gap, and not monotone in any angle
(log-derivative about 1.2/rad in beam angle, 2 per unit own index). A
sampling proxy shows magnitudes separate the in-sync blind families once
their rotor boxes are at or below 0.25 Hz and 9 deg (14/780 pairs outside a
30 percent inflated sampled range at 0.25 Hz; 266/1770 at 0.02 Hz; 0 for the
truth), so a rigorous quantitative pair test would close them: exact
endpoints in the affine coordinates, corners in the own index, beam angle
and own index subdivided with mean-value slack from chainjac (tight on
narrow sub-boxes). Not built; estimated about 100 s per shape cell in
Python, days for the 10^4 blind-family boxes. All-positive-wedge cases (no
in-sync blind family) with the sign test alone: case 5: signs (1, 1, 1): processed 100001 excluded 49990 survivors 0 pending 22  truth-in-survivor False  903s; case 7: signs (1, 1, 1): processed 100002 excluded 49991 survivors 0 pending 21  truth-in-survivor False  840s.
Instrumented on case 7: deep undecided boxes far from the truth carry
about 350 certain constraints and violate none, because the surviving
constraints are strongly correlated (same k-j, consecutive k) and carry
only of order ten independent bits; the test is decisive only near 0.02 Hz
and 2 deg, so the rotor tree is of order 10^7 boxes even without blind
families. T5 open. New package modules: risley_lattice/ordering.py, chainjac.py;
experiment: experiments/t5_rotor_ordering.py; results and audit under
experiments/results/t5_rotor_2026_09_18/.

## 2026-09-18: Identifiability theory stated; two structural lemmas proved

The user rejected search techniques ("any fool could do these") and asked
where the unified theory stands. Written as
paper/IDENTIFIABILITY_THEORY_2026_09_18.md: a two-layer theorem. Layer 1
(ideal record): outside an explicit exceptional set (a zero wedge, a
rational relation among signed speeds, a visibility failure) the continuous
infinite record determines all eighteen parameters; proof = Bohr uniqueness
+ Lemma A + Lemma B + Proposition 2.4 + Lemma C. Layer 2 (finite record,
allowance): a stability theorem whose constants come from the analyticity
strip (TIR margin) and the discrepancy of the sampled orbit; stated, not
proved; predicts exactly the observed failure classes.

LEMMA A (polarization, proved): along the flow the lattice line with
integer vector m and signs sigma has x-to-y phase lag (sum sigma_i m_i)
pi/2, so lines with even coefficient sum are exactly linear in the complex
scan, lines with sum 1 mod 4 are elliptical with the handedness of
sign(a_m b_m), sum 3 mod 4 reversed. Verified on a 400 s record of case 3:
generators 84398:1639, 47181:961, 5094:110 (+f:-f); 2w1 and w1+w2 and
w1-w2 equal to the last digit; 2w1-w2 positive; w1+w2+w3 and 3w1 negative.
LEMMA B (first harmonic, proved modulo Assumption T for positions,
unconditional for directions): a_{e_i} b_{e_i} > 0 for every A_i != 0.
LEMMA C (shape from flat-configuration Taylor coefficients): normal
incidence closed forms h_i = A_i(n_i-1)L_i, h_ii = -2A_i^2(n_i-1)p, h_ij =
-A_iA_j(n_i-1)(n_j-1)p/n_j, h_iii = 6A_i^3[D_i c_3(n_i) + plate terms],
c_3 = (n-1)(n^2-n+1)/2, verified against the model to 1e-9 (orders 1, 2)
and 1e-5 (order 3); dilation ambiguity broken because r(n) = (n^2-n+1)/(2
(n-1)^2) is strictly decreasing. General incidence: finite elimination,
not done. Evidence: experiments/identifiability_lemmas.py, results under
experiments/results/identifiability_2026_09_18/. No search, no archive
input, no recovery. T5 open; the finite-record layer is the remaining
mathematics.

## 2026-09-18 (later): Layer 1 shape step proved at general incidence; Layer 2 reduced

LEMMA C GENERAL INCIDENCE (proved by analytic continuation): the last
prism's single-tilt response X_3(s) = d_W t_6 + Q(1 - m t_6)/(1 - m tau) is
real-analytic exactly on (-sin(beta+alpha), sin(beta-alpha)) in the tilt
sine (TIR branch points of square-root type, verified exponents 0.500),
with the lateral pole at |s| = cos alpha beyond; sin beta = 1/n, sin alpha
= sin theta_0/n. The three radii give sin beta = S/P, tan alpha = C/(P cos
beta), A = cos alpha/P with S, C the half sum and half difference of the
TIR radii and P the pole radius; verified exact (1e-15) on 20,000 draws.
Then (d_W, Q) from c_0, c_1 (negative determinant). Prisms 2, 1 likewise
(a flat plate never reflects totally on entry), then g from the lever
arms, p from Q_1. Layer 1 is complete modulo Assumption T and the
write-up of Proposition 2.4.

ANGULAR IDENTITIES: D'(0) = F, D''(0) = F(F+2) tan theta_0, D'''(0) =
F(F+2)(F+1)(1+3 tan^2 theta_0), tan alpha = tan theta_0/(F+1), so (n,
theta_0) are explicit in (F, D''); the last prism's c_0..c_3 are closed
forms in (A, F, T, d_W, Q), verified against finite differences.

LAYER 2 MEASURED: rotor step has small explicit constants at K=200
(generator leakage ratios 0.001-0.02; 0.34 for the 3-degree wedge of case
3, the weak-wedge class; torus coefficients decay exp(-1.1..-1.6) per
shell). The function-space route for the shape (orbit -> torus function
-> parameters) has a floor: the fixed-point leakage bound on sup|G - G'|
is 7e-2, 2.3e-2, 1.4e-2, 1.3e-2, 9.7e-3 of scale at K = 200..3200 (case 0;
similar for 3 and 7) at allowance 1e-5 scale, because ~700 lines above
1e-7 cost ~700 x allowance in a sup bound and the shape-carrying small
lines are not individually resolvable; a Taylor route on the tilt box
meets the same wall (degree > 10 needed for 1e-5, more coefficients than
samples). Hence the finite-record shape statement is exactly the
injectivity modulus of the 12-parameter sampled analytic family, not a
function-space question. Decomposition F = E o C + R (E linear evaluation
of low-order Taylor coefficients at the 200 known tilt points, C the
explicit algebraic parameter-to-coefficient map, R the Cauchy-bounded
remainder) makes an explicit modulus possible once C has an explicit global
inversion: that inversion is the open algebra. Evidence:
experiments/identifiability_layer2.py, results under
experiments/results/identifiability_2026_09_18/.


## 2026-09-18 (later still): decomposition F = E C + R measured; stable Lemma C numerically injective

Fourth-order closed forms of the last prism's response verified (1e-6):
D''''(0) = F(F+2)T[(F+1)^2(9+15T^2) - (1+3T^2)], Theta_4 = D'''' + 4
Theta_2, with the tangent and lateral expansions. The map (A, F, T, d_W,
Q) -> (c_0..c_4) returned the truth from 3,000 random starts (60 trials
x 50) with zero spurious solutions: the stable form of Lemma C holds
numerically with five coefficients; its proof is an elimination in five
variables. Decomposition of the sampled shape map (rotor known) into
E C_d + R_d on case 0: sigma_min(E) = 0.77, 0.30, 0.11, 0.007, 0.002 for
degrees 4..8 (x axis); the weakest coefficient direction (gap against
n_2, n_3, p_y, d_W) carries 0.715 pattern units per full prior range at
every degree, equal to sigma_min of the full sampled shape Jacobian
(0.7149), while the remainder's sensitivity along it is 2.4e-2, 7.8e-3,
2.2e-3, 6.3e-4, 2.0e-4: ratios 30 to 3600. The local contest is won in
every direction; a global modulus needs the global modulus of C_d (stable
Lemma C) and direction-dependent global Lipschitz bounds on R_d (a crude
2 sup|R| bound fails against the weak signal). Both are finite explicit
tasks, not done. Script: experiments/identifiability_decomp.py.

CORRECTION (same day): one axis's five coefficients do NOT determine the
last prism robustly: the five-coefficient map has inverse condition
1e-5 to 1e-9 over the prior (near normal incidence even coefficients carry
only the lateral term, odd ones only the angular term), and the saved
random-start run found near-solutions with the index off by 0.04 and
d_W by 3 percent at all five coefficients matched to 1e-10. Both axes
together resolve it: the ten-coefficient map (A, n, theta_x, theta_y,
d_W, Q_x, Q_y) returned the truth from 2,400 exact fits with zero
spurious solutions. The stable Lemma C is a two-axis statement; its
proof is an elimination in seven variables.

## 2026-09-18 (evening): Assumption T certified per instance; exact elimination per instance

ASSUMPTION T -> CERTIFICATE. The theorem needs tilt monotonicity only at
the true system over its own torus. experiments/certify_tilt_monotone.py
covers the phase torus by 48^3 cells, encloses d(position)/d(tilt sine)
on each cell with the exact-range Jacobian (valid enclosure), refines
failing cells to depth 3. All 28 admissible battery truths certified,
lower bounds 17 to 88 pattern units per unit tilt sine, 1-4 s each; the
two truths whose torus reaches TIR (9, 29) are refused. Layer 1 is now
unconditional for every certified instance.

STABLE LEMMA C, EXACT PER INSTANCE. experiments/lemma_c_elimination.py:
rationalized data, (d, Q) eliminated linearly, resultants in A (degrees
75, 93 in (F, T), common factor of degree 7 removed), resultant in T of
degree 284 in F, Sturm root count, back-substitution: for the case-0
last prism exactly one solution in the box (the truth) in 48 s. Per
instance and exact; a symbolic all-data proof is out of reach at these
degrees; the full 12-parameter map not attempted.

REMAINDER BOUND (piece 3): reduced to a chain of certified enclosures of
J_F - E J_C along the weak direction over the box left after pinning the
strong directions; not built.

CORRECTION (evening): the exact elimination on the near-degenerate example
finds TWO exact real solutions of the one-axis five-coefficient system in
the box (A, n, d_W, Q) = (0.2403, 1.7758, 166.77, -10.31) and (0.2343,
1.8178, 161.18, -10.14), same beam angle 1.73 degrees, residuals 1e-15
(experiments/results/identifiability_2026_09_18/lemma_c_elimination_degenerate.json).
So one axis's five coefficients are two-to-one there; the two-axis
ten-coefficient map is the right object (numerically injective, 2,400
fits, zero spurious). Theory note corrected.


## 2026-09-20: Lemma D removes the visibility condition; chain position proved; Theorem 2 (qualitative); two-axis exact certificate; one correction

Target: T5 (criterion unchanged). Verdict: STILL OPEN. No archive input,
no search over parameters, no counts changed.

LEMMA D (generator dominance), proved. For a system whose positions are
nondecreasing in every tilt sine (class T) and A_i != 0, every lattice
line that involves rotor i is strictly smaller than generator i:
|zeta_m| < |zeta_{e_i}| for m_i != 0, m != e_i. One integration by parts in
gamma_i turns a_m into an average of (dH/ds_i) A_i sin(gamma_i)
sin(m_i gamma_i)/m_i times cosines, and |sin(m g)| <= m |sin g| with
dH/ds_i >= 0 gives |a_m| <= 2^{q-1} |a_{e_i}|, strict unless m = e_i.
Explicit margin kappa |A_i| (g_x + g_y), kappa = (1 - 2/pi)/2, with g the
certified minimum tilt gains. Consequence (Proposition 2.4, new form):
the generators are read from the spectrum by the rule largest line, then
largest line outside the module generated so far, twice. No visibility
condition. The old condition (three largest lines are the generators) is
false on battery cases 12, 16, 18, 27; the new rule returns the generators
on all 28 admissible truths; dominance ratio at most 0.26; margins
respected (experiments/identifiability_lemmas.py dominance ->
dominance.json). Exceptional set of Theorem 1 is now: zero wedge, or
rational relation. Weak wedge is a finite-record difficulty only.

CHAIN POSITION, proved. The first version of Theorem 1 matched rotors of
the two systems by index without proof. Exact single-rotor response at
position i: X_i = Q_i (1 - m t)/(1 - m tau_i) + D_i t + 3 sum_{j>i} P_j(t),
P_j(t) = t/sqrt(n_j^2 + (n_j^2 - 1) t^2) (agrees with the model to 1e-13
on 2,400 responses). The pole, the entire term and the plate terms with
distinct branch points are independent, so the number of plate terms, which
is the number of prisms downstream, is intrinsic to each frequency label.

THEOREM 2 (finite records, qualitative), proved. For theta* in a compact
class T_c with nonzero wedges and (N*, f_s) rationally independent: every
delta has K_0, epsilon_0 such that agreement of the first K_0 samples
within epsilon_0 forces |theta' - theta*| < delta for all theta' in T_c;
with a rank-18 sample Jacobian, Lipschitz in the allowance. Proof: sampled
Bohr uniqueness, a rank argument that disposes of competitors with zero
wedges or rational relations, Proposition 2.4 modulo 2 pi, Kronecker,
Lemma C with chain position, then compactness. Constants NOT explicit and
not uniform (near-rational speeds). So it does not meet the T5 criterion.

TWO-AXIS EXACT CERTIFICATE (experiments/lemma_c_two_axis.py ->
lemma_c_two_axis.json). Exactly rational instances through the rational
parametrization of (F+1)^2 = n^2 + (n^2-1) T^2. Per axis: linear
elimination of (d_W, Q); dropped factors are F, F+1, T^2+1,
F T^2 + F + 2 T^2 + 1 only (this is the degree-7 common factor of 09-18,
now identified and nonvanishing in the box); eliminants of degree 284 in F
and 290 in T; root isolation to 1e-120; exact rational interval
back-substitution. Both instances: one solution per axis inside the
prior, one joint solution, the truth (304 s and 502 s).

CORRECTION of the 09-18 evening entry. The second solution of the
near-degenerate one-axis example has index 1.825 (exact: A = 0.23408, n =
1.82469, d_W = 161.03, Q = -10.137), outside the index prior [1.3, 1.8].
The 09-18 box bounded F and not n, so "two solutions in the box" was true
of that box and not of the prior. The pair straddles a fold near n = 1.80
for those parameters (truth 1.73 pairs with 1.917, 1.75 with 1.875); a
float scan of 300 random truths found two second solutions, both outside
the prior. Whether one can fall inside is not known. Every axis also has
the mirror solution (A, T, d_W) -> (-A, -T, -d_W). Theory note corrected.

SCOPE OF THE DECOMPOSITION F = E C + R, stated plainly. The tilt expansion
converges like rho^d with rho = A / sin(beta - alpha); rho = 0.896 already
at the prior corner of the first prism, so the remainder is of the order of
the scale for admissible shapes near TIR. The decomposition is a local
tool. It cannot exclude distant shapes. Explicit constants at K = 200 have
no known route other than exhaustive computation.

ASSUMPTION T, LAST PRISM, CLOSED FORM (same day). dX/dm_3 = (cos phi
sin(c-b)/cos theta) [l cos phi/(sin b cos c cos theta) - p_e/cos b], l =
d_W - m p_e; verified against finite differences (4e-7, 185,000 draws).
So T for the last prism <=> l cos phi > p_e tan b cos c cos theta, and
max |tan b| cos c = n - sqrt(n^2 - 1) <= 0.4693 on the index prior, hence T
holds for the last prism whenever |p_e| < 1.22 d_W (beam within 61 mm of
the axis on the last exit face). Prisms 1 and 2: no closed form; the
per-instance certificate stands.

THEOREM T, ALL PRISMS, EXACT (same day, later). The ray leaving prism i is a
line (intercept y_0 = p_i (1 - m_i t), slope t); downstream direction maps
do not depend on position (plane faces) and downstream position maps are
y -> lambda_j (y + 3 tau_j) plus free propagation. Chain rule gives the
effective lever recursion E_4 = 0, E_j = tau_j' (3 + m_j p_j) + (D_j - m_j
p_j + E_{j+1}) T_j'/lambda_j, the effective remaining distance l_i = D_i -
m_i p_i + E_{i+1}, and

  dX/dm_i = Lambda_{i+1} (cos phi_i sin(c_i-b_i)/cos theta_i)
            [l_i cos phi_i/(sin b_i cos c_i cos theta_i) - p_i/cos b_i].

So position increasing in tilt i <=> l_i cos phi_i > p_i tan b_i cos c_i
cos theta_i. Verified against finite differences: 2e-7, 60,000 random
admissible states, all three prisms, sign of derivative = sign of margin
on all. max |tan b| cos c = n - sqrt(n^2-1) <= 0.4693. Regime corollary:
positive glass and air paths, |p_j| <= P, |theta_in| <= Theta at prisms 2,
3, and (d_W - 0.325 P) cos^6 Theta > 0.4934 P imply T (T_j'/lambda_j >=
cos^3 theta_in). Battery: margin >= 48.4, sufficient condition >= 47.3 on
all 28 (24^3 grid); some model rays have glass path down to -0.41 (no
aperture in the model), criterion holds regardless.
experiments/tilt_monotone_exact.py -> tilt_exact_verify.json,
tilt_exact_battery.json. The class T of Theorems 1 and 2 is now an
explicit scalar inequality. Theory note has a final statement (section 6).
T5 verdict unchanged: STILL OPEN.

CONTINUOUS RECORDS (same day, after the user asked why the theory was
sample based). It was sample based only because the pipeline and the T5
criterion define the observation as 200 samples. For the continuous
time-stamped trace: z(t) is real-analytic on the line (no TIR on the
compact torus), so agreement on ANY interval is agreement everywhere, and
Theorem 1 gives uniqueness for every window length (Theorem 1'). The
compactness argument then gives stability for every window length
(Theorem 2'), with no sampling-rate condition and no K_0. Constants still
come from compactness. Theory note section 3.0a. The random-state and
battery numbers quoted elsewhere are checks of proved formulas, never part
of a guarantee.


## 2026-09-21: constructive inversion by layer stripping; Theorem 3 (inert prisms)

Target: T5 (criterion unchanged), verdict STILL OPEN; one sentence added to
T6's "Have". No archive input, no search, no counts changed.

LAYER STRIPPING (theory note 2.6). Theorem 1 is constructive, and only one
step is unstable. (1) rotor by the rule of Proposition 2.4, which is
exactly the rank-extending largest-line rule in
risley_lattice/lattice.py::fundamentals; Lemma D is its proof of
correctness (the "three largest lines" rule that Lemma D replaced was in
my first write-up, not in the solver). (2) torus function. (3) chain
positions by plate count. (4) the last prism reads the incoming ray: at
any upstream state the function of the last tilt is d_W t + Q lat with the
beam angle replaced by the incoming direction, so the radii and c_0, c_1
give (A_3, n_3, d_W) and the full ray entering prism 3 at every upstream
state; Q is a nonconstant analytic function of the state, so the pole used
for the third radius is present on an open dense set, which removes a
codimension-one gap of the flat-configuration proof. (5), (6) prisms 2
and 1 are EXPLICIT: at zero tilt a prism is a plate, so the entering
direction is the value of the exit direction at u = 0, and with c_1 = A F,
c_2 = A^2 F (F+2) T/2: F = 2 c_1^2 T/(2 c_2 - c_1^2 T), A = c_1/F, n^2 =
((F+1)^2 + T^2)/(1+T^2) (exact to 1e-40 at 40 digits, 300 prisms); gap and
lateral terms by 2x2 linear solves. So the whole difficulty of the shape
step is the last prism.

THEOREM 3 (note 2.7). Class Z: one inert prism (zero wedge anywhere, or a
stationary first prism), two rotating prisms with independent speeds.
Within Z the record determines the position of the inert prism (plate
counts (1,0), (2,0), (2,1)) and everything except: speed and phase of a
zero wedge; and for an inert first prism the block (N_1, A_1, ay_1, n_1,
beam) only through the constant ray entering prism 2 (plate gauge and the
09-17 stationary family are one fiber). The index of a flat plate at
position 2 or 3 IS determined (cubic part of the plate shift), settling
the FAILURE_THEORY remark. Competitors outside Z with rank-two records
(stationary tilted prism 2 or 3; three rotors with one relation) are not
excluded. Check (mode strata, one system per stratum, 74 s dense record):
exact null directions 3 / 2 / 2 for zero wedge at 1 / 2 / 3, with n_2, n_3
weak at 6e-10 and 3e-10; stationary first prism 3; stationary 2, 3 no
exact nulls; N_2 = N_1, N_3 = N_2, N_2 = -N_1, N_2 = 2 N_1 all regular.
A first draft of the theorem wrongly read the thresholded nullity 3 at
positions 2, 3 as an exact null index direction; the singular values show
it is weak, not null, as the proof says.

FLOATS. Stored speeds are rational, so no stored system satisfies rational
independence literally; Theorem 2' plus the triangle inequality covers
every system in a neighbourhood of a non-exceptional one. Written into
note section 4.

RIGOR PASS ON LEMMA C (same day). The general-incidence proof asserted
three facts that had only been checked numerically. Now proved, under
beta + |alpha| < pi/2 (true at the flat upstream configuration of every
prior system; fails only above 56 degrees incidence at n < sqrt 2), alpha
!= 0, Q != 0 (open dense set of upstream states): (a) real-analytic on
(s_-, s_+); (b) s_+- are genuine square-root branch points because dq/ds
> 0 and the coefficient of t_6 is the remaining distance d_W - m p_e > 0;
(c) continuing past the nearer branch point along the upper side of the
real axis, arcsin q = pi/2 + i arccosh q keeps tan(c - phi) finite, no
second TIR solution exists, and the first singularity is the simple pole
at sign(alpha) cos alpha with residue Q (1 - m t_6)/(-tau (1 - s^2)^{-3/2}),
nonzero because cos c = -i sinh w. Checks at 30 digits: exponent 0.500 to
2e-4, no intervening singularity, residue formula to 1e-9
(identifiability_lemmas.py continuation). What is left to close the work:
the user's decision on the T5 criterion, and the user's go-ahead for
moving the theory into the paper and for committing.

USER DIRECTION (09-21, evening): keep the theory in markdown; no re-scoping
of T5; keep hunting until it is done in theory or perfectly bounded.

TORUS COVERAGE MEASURED (paper/T5_HUNT_PLAN_2026_09_21.md;
identifiability_layer2.py coverage). Covering radius of the rotor orbit on
the one-axis folded torus: 0.70 to 0.82 rad at 10 s, 0.29 to 0.42 at 100 s,
0.15 to 0.28 at 300 s, 0.05 to 0.14 at 1000 s. Implied pointwise error of
the torus function by local interpolation up to order 8 (estimate, true
system's derivative norms, interpolation constant ignored): 6e-3 to 8e-1
of the scale at 10 s, 6e-6 to 4e-3 at 100 s, 3e-8 to 1.5e-4 at 300 s. The
1e-2 function-space floor at 10 s is this sparsity, seen geometrically: a
property of the record. Consequence: the layer-stripping decomposition
(12 coupled shape coordinates -> one 4-unknown problem + two monotone
2-unknown problems) is available once the record is minutes long, and not
at 10 s, where frozen upstream states are never visited. Hunt plan: rotor
step with explicit window constants; pointwise reconstruction lemma with
covering radius and class derivative bounds; last prism with explicit
modulus (4 nonlinear unknowns); Lipschitz constants of the explicit
upstream formulas; assembly. For the 10 s record no decomposition exists.

## 2026-09-22: finite algebraic shape inverse; adverse conditioning control

User requested a fresh research assessment, then an intensive attempt to
finish the problem. Target: T5, criterion unchanged. Verdict: STILL OPEN.
No archived-case hunting, recovery campaign, paper/main.tex edits or commits.
The new computations are exact symbolic/rational algebra controls and one
high-precision synthetic conditioning control, not additional recoveries.

LAST-PRISM ALGEBRA. paper/LAST_PRISM_IMPLICIT_INVERSE_2026_09_22.md proves
an exact finite inverse for the full last-prism response, instead of its
five-term Taylor truncation. With m=Au/sqrt(1-A^2u^2), tau=sigma/kappa,
R=d-m(d tau+Q), S=(1-m tau)X-Q, the squared Snell identity is
(S+mR)^2-kappa^2(tau+m)^2(R^2+S^2)=0. Its tilt-root norm is a polynomial
H(v,X), v=u^2, of bidegree (4,4): 25 possible coefficients. In the stated
regular regime, a field/norm proof establishes that H is primitive and
irreducible. Any 33 distinct squared-tilt values determine it exactly:
the resultant with another polynomial in the same rectangle has degree
at most 32. The leading X^4 coefficient eliminates d and Q and factors
as (1-rv)^2(1-2pv+qv^2). A simple root of the reciprocal quartic's cubic
derivative gives r; one strictly increasing scalar function, with
derivative at least one, gives a=A^2. Remaining shape coordinates,
signed incidence, distance and offset have explicit coefficient formulas.
Normal incidence does not require division by sigma. The exact
competitor argument uses the universal polynomial identity, so the
regular truth is compared with physical competitors outside its regular
subclass as well. Zero-Q and a finite nearby-state construction are
treated separately in the note. These are statements about exact frozen
responses; they are not a solver for the native time samples.

CHAIN ORDER AND UPSTREAM FORMULAS. paper/CHAIN_POSITION_ALGEBRAIC_2026_09_22.md
proves that each distinct flat downstream index adds an independent square
root over the angular circle field. Minimal degree in position is 4 for
the last rotor, 8 for the middle, and 16 for the first (8 if its two
downstream indices coincide). A finite resultant argument gives a
193-value exact last-position test; after removal, 81 values order the
remaining two. Zero beam angle, Q=0 and repeated glass indices are covered.
Upstream angular inversion uses three distinct outgoing directions in a
linear system, or two plus a quadratic at normal incidence; derivatives
at zero are unnecessary. This supplies an algebraic completion of the
normal-flat-state gap in the earlier chain-position proof.

FINITE-SCAN TRANSFER. paper/TARGETED_FINITE_SCAN_READOUT_2026_09_22.md
proves a direct linear-functional certificate with joint bounded-noise
errors, explicit Chebyshev tails, a stronger Parseval ellipsoid/kernel
bound, uncertain-rotor terms, and inverse-residual certificates for an
implicit coefficient chart. A full-prior local complex domain is proved:
|u_i|<=.03 is holomorphic with |h|<203; the cube |u_i|<=.02 therefore has
a Bernstein-ellipse bound R=2,M=203. Downstream-first flattening connects
each admissible sample state to the flat state while preserving the
physical guards. With common rationally independent rotors, exact equality
on a time interval then extends to the flat germ by algebraic resultants
and Zariski density, without assuming whole-torus admissibility. These
facts do not manufacture an analytic germ from 200 isolated samples.
The note supplies both a finite algebraic norming condition and explicit
counterexamples to the corresponding inference in a relaxed analytic
class. Arbitrary speed relations and full physical-family norming remain
open obligations.

EXACT CONTROLS. experiments/last_prism_implicit_control.py checks the
Snell/norm identities, coefficient factors, monotone scalar identity,
quintic, subresultant, simple critical root, signed-incidence/distance
formulas and angular readout. Two exact rational instances (nonzero and
zero incidence) have irreducible bidegree-(4,4) curves and 33 rational
graph points with exact rank 24. Modular elimination over prime
1000000007 proves the lower rank bound; the nonzero exact relation gives
the upper bound. Results: experiments/results/last_prism_implicit_2026_09_22/control.json.
These controls verify algebra and examples; the universal proofs are in
the notes and received an independent mathematical audit.

CRITICAL NEGATIVE CONTROL. The new finite theorem must NOT be sold as a
useful noisy linear solver. experiments/last_prism_conditioning_control.py
uses one fixed regular curve A=1/5, sigma=3/13, kappa=20/13, d=100,Q=2,
with substantial physical margins. Even after output centering/scaling
and unit-column tensor-Chebyshev normalization, the smallest nonzero
singular value of the free 25-coefficient matrix is 7.32e-34 for 33
positive tilts and 7.59e-27 for 66 signed values. The latter linearized
normalized data-to-coefficient infinity gain is 8.63e25. Calculations at
80 and 110 digits agree in all 24 reported significant digits. The
five-physical-parameter RMS Jacobian is much better: its range-scaled,
output-normalized condition is about 8.27e4 for the signed data, versus
3.21e26 for the coefficient matrix. These are sensitivity diagnostics,
not interval guarantees, statistical error bounds, or physical
impossibility theorems. Huge linearized perturbations cannot be interpreted
as valid finite-error estimates. The result shows that fitting 25 free
coefficients throws away essential low-dimensional physical structure.
Results: experiments/results/last_prism_implicit_2026_09_22/conditioning.json.

CORRECTIONS TO EARLIER STATUS. The claimed intrinsic function-space floor
and "no decomposition exists at ten seconds" were unsupported. The
coefficientwise upper bound loses cancellation; sampled coverage estimates
use random test points, truth-only derivatives and omitted interpolation
constants. Neither is a lower bound on all methods. The rotor calculation
uses a truth-centered 32^3 FFT and fitted decay, without certified tails
or all-competitor exclusion, so it does not certify ten-second rotor
confinement. The measured C in identifiability_decomp.py is a discrete
polynomial least-squares projection on a Chebyshev grid, not the center
Taylor map used in the separate elimination. The theory note, hunt plan,
handoff and T5 status now record these distinctions.

NEXT ACTUAL BOTTLENECK. Preserve the physical coefficient constraints
while extracting parameter information; do not repeat unrestricted
25-coefficient fitting at higher precision or merely collect more nearby
slice points. A useful global physical secant/norming bound must connect
the actual finite record to the structural inverse, with globally
justified rotor bounds and deterministic arithmetic. The algebraic shape
inverse is now finite and explicit, but neither its practical noisy use
nor the original full18 finite-scan gate is closed.

LATE COMPLETION OF THE ZERO-OFFSET BRANCH. The last-prism note now also
inverts Q=0 directly: remove content (1-rv)^2 to obtain an irreducible
bidegree-(2,4) curve. Seventeen values determine it when Q=0 is known;
25 exclude competitors with unknown Q. Thus 33 values form a uniform
exact protocol covering Q=0 and Q!=0. Distance and wedge are selected
roots of two explicit quadratics in the flat-prior regime, with formulas
regular at normal incidence. The signed-incidence formula is explicit.
The new identities passed independent exact symbolic controls in the
same control script. The optional 26-nearby-state construction remains
as another exact way to obtain a nonzero-Q slice, not a required detour.

## 2026-09-22: T5 structured inverse; physical slice-noise obstruction

User asked to attack the remaining obstacle. Selected target: T5, with
the original full18, 200 moving samples, deterministic native <0.001
criterion unchanged. Verdict: STILL OPEN. No archive campaign, optimizer
run, main.tex edit, or commit. Three independent theory/audit tasks and
one bounded symbolic/interval control supported this pass.

DIRECT PHYSICAL INVERSE. paper/STRUCTURED_LAST_PRISM_INVERSE_2026_09_22.md
avoids the 25 free implicit coefficients entirely at known normal
incidence. Eight values at four signed frozen tilt magnitudes give the
odd part O=d t, eliminating Q. Set x=O^2,z=x/u^2,Y=d^2. The physical
identity is z=C(x+Y)-D sqrt(x+Y), C=A^2(n^2+1),D=2A^2 n d.
The ratio of adjacent second divided differences equals
Phi(Y)=(r2+r4)(r3+r4)/[(r1+r2)(r1+r3)], r_i=sqrt(x_i+Y), whose
log derivative is (1/2)(1/r2+1/r3)(1/r4-1/r1)<0. One globally monotone
scalar solve gives d; explicit positive-root formulas give n,A,Q.
The theorem includes Q=0. It excludes every competitor with known
normal incidence, not arbitrary unknown-incidence competitors.

SMALL EXTENSION. When Q=0 is imposed on all competitors, one exactly
odd pair forces incidence sigma=0 through the unsquared paired Snell
sum. Four positive values plus one paired negative therefore suffice
in this submodel. With bounded error, the pair gives
|sigma| <= (|E_hat|+epsilon)/(d_min sqrt(1-A_max^2 u^2)). Unknown Q
can cancel the even part exactly for one pair, so that incidence
inference cannot be silently extended to the full unknown-offset class.

FINITE-ERROR CONTROL. Explicit monotone corner bounds and outward
arithmetic give a global enclosure procedure within the known-normal
class. experiments/structured_last_prism_control.py verifies the core
identities symbolically and runs one fixed rational instance
A=1/5,n=3/2,d=100,Q=2, magnitudes 1/4,1/2,3/4,1, distance prior [50,200].
Under ivx's A-fp contract the d radius is about 3.43e-4 at zero added
allowance, 1.68e-3 at 1e-12, and .134 at 1e-10; 1e-8 and 1e-6 are
inconclusive. Roundoff and interval dependency are included. These are
conservative algorithm widths, not physical lower bounds or a model
noise threshold. Independent code review checked the all-candidate
retention of the monotone exclusions. Results:
experiments/results/structured_last_prism_2026_09_22/control.json.

GENUINE PHYSICAL NOISE LIMIT. paper/STRUCTURED_SLICE_NOISE_AUDIT_2026_09_22.md
constructs a normal Q=0 family with b=A^2(n^2-n+1) and g=d(n-1)A fixed.
Its linear and cubic response terms agree exactly; differences begin
at fifth order. The pair n0=1.5,A0=.1,d0=100 and n1=1.5022 with
A1^2=437500/43860121,d1^2=438601210000/44135847 has exact entire-slice
discrepancy <11/3136000 (about 3.508e-6). A rational derivative bound
and uniform optical margins prove this without local linearization.
At midpoint-data allowance eta=11/6272000, every estimator has glass
index error at least .0011 for one of these two explanations. Thus
even continuous frozen-slice information cannot guarantee native
<.001 for this pair at that allowance. This is NOT a full18 moving-
record obstruction. The exact rational constants and identities are
reproduced by the same control script.

WHAT STILL HOLDS OUT. The original data do not supply frozen slices.
Unknown incidence with unknown offset, all-competitor rotor/shape
confinement, and useful noise control for the actual moving record are
not solved by the new normal-stratum theorem. The strongest exact
full18 reduction remains the existing source/geometry elimination in
paper/OUTSIDE_GLOBAL_GEOMETRY_2026_09_17.md: R(q) is the minimum native
record residual over four affine coordinates for fourteen nonlinear
coordinates q. The missing ingredient is a useful global bound on R(q)
over wrong nonlinear regions, with rigorous ambiguity decisions where
appropriate. Full-prior positive secant bounds are impossible because
exact sampled-record ambiguities already exist. No finish-line target
was crossed off, and the archived recovery ledger did not change.

## 2026-09-22: user defers global uniqueness; constructive recovery first

After the finish-line explanation, the user asked to deal with uniqueness
later. Global uniqueness/all-competitor exclusion is therefore deferred
as an immediate research priority. It must no longer be imposed as a
prerequisite to progress on finding one explanation.

Immediate milestone: constructive blind recovery of one physically
admissible full18 candidate from the actual 200 moving-scan samples,
followed by deterministic checks of physical validity and observation
compatibility. Existing local certification may provide neighborhood-
conditional conclusions. Such a candidate does not alone establish
native <.001 error relative to the unknown generating specification;
another distant compatible explanation may remain.

This changes the order of work, not the evidence already established.
T5's eventual all-competitor criterion stays open. The restrictions on
archive campaigns, manuscript edits and commits remain in force; no
such work was started by this scheduling update. The next constructive
step should address reliable initialization/inversion from the native
record without demanding a global uniqueness proof first.

## 2026-09-22: constructive full18 finite-record theorem on an open class

User requested a serious attempt at constructive recovery after deferring
global uniqueness. New result: a data-initialized full18 convergence
theorem on a nonempty open small-wedge, near-normal, nonaliased class.
This is a mathematical advance, not full-prior closure or a demonstrated
optical solver. No archive recovery/forward reruns, main.tex edits or
commits. The controls were symbolic, rational, or explicitly polynomial
fixtures. The current full18 solver was not changed.

EXPLICIT QUADRATIC PHYSICAL READOUT. At normal incidence with nonzero
source p and wedges A_i, let c_i=n_i-1. The center Taylor polynomial has
l_i=A_i c_i L_i, q_ii=-p A_i^2 c_i, q_ij=-p A_i A_j c_i c_j/n_j (i<j),
where L3=d, L2=d+g+3/n3, L1=d+2g+3/n2+3/n3. Mixed quadratic ratios give
rho=q12*q23/(q13*q22)=(n2-1)/n2 and n2=1/(1-rho);
n1=1+q12^2*n2^2/[q11*q22*(n2-1)]; R=q22*q33*(n2-1)/q23^2 gives
n3=2/(1+sqrt(1-4/R)), the unique branch below2. Then signed A, d, gap
and source are explicit. These formulas invert the physical relationships
without requiring cubic or quintic coefficients on this normal stratum.
See paper/QUADRATIC_PHYSICAL_INITIALIZER_2026_09_22.md.

UNKNOWN BEAM DIRECTIONS. A total-degree-three Snell/intersection expansion
gives the exact derivatives of the quadratic jet with respect to beam
incidence. At a rational witness, the 20 two-axis quadratic coefficients
have a rank12 Jacobian for ALL twelve nonrotor quantities, including
both unknown beam directions. The selected12x12 minor is nonzero mod
1000000007 (residue933379057). An independent tangent-based derivation
at a second rational witness gives exact determinant
-1666307939/257298363000000000; the two beam-column derivations agree
exactly. The inverse function theorem supplies a full12 physical jet
inverse near the normal witness. Newton can evaluate it from the normal
closed-form preseed for sufficiently small unknown incidence. This is
local and quantitative radii are not yet computed.

FINITE RECORD, NOT FROZEN STATES. paper/CONSTRUCTIVE_MOVING_SCAN_2026_09_22.md
keeps the original200 samples and distinguishes discrete projection
from Taylor coefficients: P*y=C3(theta)+P*R3(theta). For known rotors,
the physical correction C(theta)+B[y-F(theta)] removes the candidate's
complete finite-record leakage. Under A=epsilon*alpha, normalized
degree0/1/2 coefficients have an epsilon-independent physical inverse;
the degree3 remainder and its parameter derivatives are O(epsilon^4).
Consequently the direct seed error and correction derivative are
O(epsilon^2) in scaled shape coordinates. A fixed invariant ball and
contraction proof establish convergence for sufficiently small wedges.
The full12 jet inverse extends this to unknown near-normal beam angles.

ALL SIX ROTOR COORDINATES ALSO UNKNOWN. Section7 removes the known-rotor
assumption. First differences of the complex scan have three dominant
exponentials near normal incidence. A three-term Prony recurrence gives
data-only frequency seeds within O(|incidence|+epsilon). An unregularized
cubic lattice fit has63 exponential columns. Its three frequency
derivatives add confluent columns t*exp(i*N_i*t); distinct sampled lattice
nodes give rank66 using at most66 of the200 samples. The normalized local
frequency Hessian is therefore invertible, with a uniform Newton basin.
The data-only preseed enters that basin for a sufficiently small class.
An analytic implicit-function argument gives frequency bias O(epsilon^3)
in C1, rather than merely a small fit residual. Cubic fundamental phase
parity gives the same phase accuracy, including unknown beam direction.
These rotor errors perturb the native trace only by O(epsilon^4), so
the normalized physical coefficient and shape errors remain O(epsilon^2).

The resulting approximate inverse H satisfies
H(F(Theta))=Theta+delta(Theta), delta and Ddelta=O(epsilon^2), where
Theta uses scaled wedge sines alpha=A/epsilon. Thus the data-derived
seed H(y) and exact correction
Theta_next=Theta+H(y)-H(F(Theta))
converge geometrically to the generating full18 specification on this
open class. Keep the SAME order/sign/root chart throughout a run;
dovetail the six physical-order runs and local solves so a wrong chart
cannot block the convergent branch. No truth initialization, inaccessible
frozen response, or global-uniqueness proof is used. The rank18 native
Jacobian also follows from D(H o F)=I+Ddelta, rather than being assumed.

AN EXPLICIT WELL-CONDITIONED SPECTRAL WITNESS. At N=(2.4,.7,.2)Hz and
t_k=k/20, cubic labels map to63 distinct DFT bins in[-72,72]. Therefore
V*V=200I exactly. Tensor-Chebyshev degree3 columns are also orthogonal
after normalization. The class is nonempty without requiring irrational
speeds or invoking infinite rational independence. This does not bound
the nonlinear physical inverse or its practical convergence radius.

CODE AND CONTROLS. risley_lattice/quadratic_init.py implements the normal
coefficient inverse, degree3 native projection, and at most12 full18
proposals from both axes and all6orders. Beam zeros are proposal values,
not known hardware; all18 must be released in physical completion.
It refuses invalid charts rather than clipping them. The full spectral/
off-normal Newton/outer-correction pipeline of the theorem is NOT yet
implemented. experiments/quadratic_initializer_control.py passes exact
rational readout identities, signed-source/zero-chart controls, a200-node
polynomial fixture (max matched-vector discrepancy1.87e-9), and an exact
63+3 confluent spectral minor (905086314 mod1000000009). This fixture
is explicitly NOT optical data or a new physical recovery. Independent
permutation/phase-wrap/other-axis checks pass. The separate
experiments/quadratic_jet_rank_control.py verifies both exact local series
and the full rational shape Jacobian. Results:
experiments/results/quadratic_physical_2026_09_22/.

OTHER DERIVATION. paper/WEAK_GLASS_PROFILE_2026_09_22.md gives exact
zero-wedge baseline and first-order sideband dependence on the unknown
weak glass. This can support a joint glass/frequency profile, but no
saved failure has been attributed to that mechanism and no performance
claim follows. It was not integrated into the solver this turn.

LIMITS AND NEXT WORK. The new theorem covers an open small-wedge,
near-normal, nonzero-source/wedge, finite-nonalias class. It gives no
computed useful wedge threshold, basin size, noise allowance or runtime,
and does not cover the full prior or exceptional strata. Its exact
Prony/Vandermonde argument uses the mathematical uniform grid k/20;
actual timestamp and forward roundoff must enter deterministic discrepancy
allowances rather than being silently ignored. Practical implementation
and physical/native-record validation remain open. Global uniqueness
remains deferred. No archive recovery count changed. Three independent
mathematical/code reviews checked the constructive chain and its scope.

User's aside about the large run was answered from the independent
669,574-case recount:83.47% recovered by10s,97.25% by80s,18,402 unresolved.
Its later physical/local-identifiability audit helped distinguish missing
rotor information, weak physical compensation and invalid specifications
from intrinsic noiseless nonidentifiability. Those numerical results
guided the mathematical attack; they do not prove the new theorem.

## 2026-09-22: source-independent cubic construction and executable full18 prototype

User asked to keep attacking hard. The immediate constructive milestone is
still OPEN; eventual T5 all-competitor localization remains OPEN and global
uniqueness stays deferred. This pass produced a stronger scoped theorem,
explicit formulas, an implementation and polynomial controls. No optical
recovery/forward campaign, archive input, main.tex edit or commit occurred.
The existing optical solvers and archived recovery ledger were not changed.

NEW SOURCE-INDEPENDENT CUBIC READOUT. At normal incidence, the odd linear and
cubic position coefficients are independent of source offsets. Five exact
cubic identities reduce recovery of the8 hardware quantities(A3,n3,d,g) to
one scalar quintic for n3 followed by explicit positive-root and rational
formulas. The other glasses, signed wedge sines, distance and gap follow.
No division by source position is needed. At the rational witness the quintic
has exactly one root in[1.3,1.8], with exact nonzero derivative
-3807156888/912342025 at8/5. Generic symbolic identities and exact rational
roundtrip controls pass; numerical discrepancy1.43e-13 or less on the primary
fixture. Numerical quintic roots supply proposals, not certified global root
coverage. See paper/CUBIC_CONSTRUCTIVE_INVERSE_2026_09_22.md and
risley_lattice/normal_cubic_init.py.

CENTERED SOURCE IS INCLUDED IN THE NEW CONSTRUCTIVE CLASS. The13 odd cubic/
linear coefficients have rank8 for(A3,n3,d,g) at the previous hardware
witness with BOTH source offsets and beam incidences zero; modular minor
310152495 mod1000000007. Each axis's constant/q11 coefficients distinguish
its source and beam sine, determinant2303407/1920000. Parity separates these
blocks, giving rank12 for the full40-coordinate cubic jet. The scalar
simple-root inverse supplies a data-derived normal preseed. A local all40-row
least-squares normal-equation inverse extends it to unknown near-normal beams
and centered/nearby sources. Off-image coefficient tuples are fitted locally,
not falsely asserted to satisfy all40 equations exactly.

NATIVE RECORD EXTENSION. At N=(3.1,1.2,.1), the129 lattice labels |m|1<=4
give distinct DFT bins modulo200. Their three frequency derivative columns
give rank132, still within the original200 sample rows. Degree4 projection
has an O(epsilon^5) physical tail under A=epsilon*alpha. The same local
frequency critical-point argument gives rotor/phase biases O(epsilon^4) in
C1, their trace effect O(epsilon^5), and normalized cubic-jet/shape errors
O(epsilon^2) in C1. Consequently the full18 approximate-inverse defect and
its derivative remain O(epsilon^2), and fixed-chart additive correction
converges for sufficiently small wedges on a nonempty open class INCLUDING
centered source. The new theorem removes the nonzero-source hypothesis;
it does not remove nonzero wedges, sampled nonaliasing, near-normal beam,
local chart/basin, or sufficiently-small-wedge hypotheses. Native finite
arithmetic and useful quantitative radii remain unproved. Global uniqueness
is not used or claimed.

PRACTICAL AUDIT CHANGED THE INVERSE CHOICE. On an exact quadratic coefficient
fixture with beam sines(.0002,-.0001), the selected12-row inverse reaches a
different exact root with distance error4.193; unused8 coefficients reveal
the disagreement. This does not refute the sufficiently local theorem, but
demonstrates a narrow practical chart. All20 coefficient inversion resolves
the same fixture. The new source-independent cubic preseed plus all40 fit
recovers the centered-source unknown-beam coefficient fixture in3 evaluations,
with maxshape discrepancy7.53e-13. These are coefficient, not optical controls.

EXECUTABLE PIECES. risley_lattice/physical_jet.py now evaluates degree2/3 jets
and exact chain-rule shape Jacobians using truncated Taylor algebra plus AD,
and supplies bounded scaled Gauss-Newton/backtracking coefficient inverses.
Their Jacobians agree with independent exact rational controls to7.11e-15.
risley_lattice/constructive_rotors.py implements data-only rank3 Hankel/Prony,
unregularized63/129-column VarPro, the full projected residual derivative,
signed phases and frozen rotor charts. A plain all-window order3 recurrence
was too fragile on a polynomial perturbation; the rank3 Hankel preseed uses
the same leading-exponential limit more stably. This change and all rank,
boundary and sign refusals are recorded in its controls.

risley_lattice/constructive_inverse.py composes full18 data-only proposals
over physical orders/axes/quintic branches and round-robin defect correction.
The physical coefficient weights use a positive scale derived from the normal
preseed wedges, then frozen; no known epsilon or wedge calibration is needed.
Correction recomputes H(y) with exactly the same fixed chart/options used for
H(F(theta)); an arbitrary earlier candidate is not substituted for it. It
stops numerically on the whole native-record residual. A separate opt-in
point checker uses fmodel intervals for physical margins and compatibility
under the caller's hard total discrepancy allowance. It is not a generating-
hardware accuracy or global uniqueness certificate, and no optical point
validation was executed in this pass.

FULL18 POLYNOMIAL CONTROL. experiments/constructive_inverse_control.py creates
an explicitly NONOPTICAL degree4 polynomial in the rotor states, using
physical cubic coefficients plus arbitrary quartic nuisance. Source offsets
are zero and beam sines(.0002,-.0001); all18 are unknown to the inverse.
All physical orders are tried, and candidate selection/correction uses
forward polynomial residuals rather than truth. Recovered native discrepancy
is2.1445e-11 and full-record residual7.905e-13. Adding an intentionally omitted
synthetic degree5 term gives seed error0.0012969235, reduced after TWO updates
to9.9332e-10. Residuals are1.12431e-5,5.00756e-7,3.57418e-10. This verifies the
finite-record algorithm and correction mechanism, NOT optical recovery.

EXACT POINT SENSITIVITY. New rational primal/dual controls optimize linear
inverses under independent bounded coefficient errors. For the nonzero-source
quadratic witness, all20 rows reduce distance amplification from1.3812e8 to
7.2437e6, about19x. For the centered-source cubic witness, the distance factor
is48362.78 and glass factors116.96,159.60,486.99. These are different source
settings/readouts, not a population improvement factor. Exact all-epsilon
cubic bounds include47316/epsilon^3 < gamma_d <=1047/epsilon+47317/epsilon^3.
Thus the cubic chart has much better point constants at this witness and a
worse asymptotic noise exponent. The coefficient-box lower bounds are NOT
native-record impossibility bounds: sample-induced errors are correlated.
See paper/QUADRATIC_JET_SENSITIVITY_2026_09_22.md.

VERIFICATION AND LIMITS. Independent reviews audited the cubic identities,
rank/parity, C1 extension, physical/spectral code and composition. Two material
implementation issues were fixed: rank-deficient zero-residual inputs now
refuse inverse success, and correction uses a freshly evaluated consistent
H(y) target. Negative controls retain the selected12 wrong-root example.
All symbolic/rational, coefficient and polynomial controls pass. Results are
under experiments/results/quadratic_physical_2026_09_22/ and
experiments/results/constructive_rotors_2026_09_22/.

The work still needs explicit physical remainder/derivative bounds on a useful
finite neighborhood, a wedge/noise/arithmetic threshold, and permitted optical
native-record validation. The prototype is not integrated into solve18. No
historical recovery count or finish-line gate was promoted to complete.

## 2026-09-22: first finite constructive tolerances; conditional physical correction

USER REQUEST. Begin capturing practical tolerances. Global uniqueness remains
deferred. This pass used exact algebra, coefficient controls and validated
interval derivatives only: no optical observation arrays were generated,
read, fitted or validated; no archive recovery campaign, main.tex edit or
commit occurred. New aggregate proof: paper/CONSTRUCTIVE_TOLERANCES_2026_09_22.md.
Machine ledger: experiments/results/constructive_tolerances_2026_09_22/ledger.json.

ROTOR CONDITIONING. At exact N=(31/10,6/5,1/10) Hz and t_k=k/20, the129
quartic lattice labels occupy distinct DFT bins. Exact rational bounds give
||V(N)-V(N*)||_2 <=522||deltaN||_infinity for normalized columns with centered
times tau_k=(k-99.5)/20 (column rephasing preserves singular values). In the
radius1/2000 Hz box, singular values lie in[.739,1.261], condition<1.707.
The three projected unit-leading-amplitude frequency Gram directions exceed
.664I at the witness and .407I on a1e-5 Hz neighborhood. These are sampling/
leading-Gram bounds, NOT blind rotor basin or full objective certificates.
Projection noise/tail/rotor/phase transfer is explicit. New implementation:
constructive_rotor_tolerances.py; exact control and proof note accompany it.

FINITE PRESEED. The normal-incidence source-independent eight-hardware
quintic readout is uniformly valid for |delta linear|<=16eta and selected
|delta cubic|<=8eta with exact eta1e-10. Every coefficient tuple has exactly
one last-index root in[1.599,1.601]. Endpoint signs, derivative negativity,
exact center identities, implicit derivatives and interval mean-value
propagation give all eight native errors<.001; max bound~.000927389deg in
ax2. This is a coefficient-input certificate, not optical noise or unknown-
beam robustness. Larger allowances remain recorded as weaker/inconclusive
bounds, not impossibility. See NORMAL_CUBIC_TOLERANCES_2026_09_22.md and control.

NONLINEAR CUBIC BOX. certify_physical_jet.py implements rigorous interval
cubic coefficients and their twelve-shape Jacobian. For T_z=beta+C(z-G(beta)),
a fixed binary12x40 preconditioner is validated over a whole box. Fine box
distance radius is .000499 (deliberately below .0005); weighted q<=.000884258.
Every native radius is strictly below exact1/2000, so pairwise native widths
are strictly below1/1000. Larger d-radius.05/.1 boxes have q<.089/.178. All
forty coefficient equations need not hold for an off-image projected fixed
point; compatible in-box roots are unique and G is locally injective. The
coefficient-midpoint noise certificate includes center-rounding drift.

PHYSICAL TAYLOR TAIL. constructive_tail_bounds.py bounds the exact guarded
ray map's fifth wedge-scale coefficient and all twelve shape derivatives
uniformly over tau in[0,epsilon] and rotor states[-1,1]^3. Taylor remainder
and derivative are <=epsilon^5 times their interval envelopes. Rationalizing
Snell as a_out=a+(n^2-1)s/(T+R), using T^2-R^2=n^2-1, preserves cancellation
lost by direct interval subtraction. Positive TIR, branch and propagation
margins are checked throughout. Fourteen independent exact coefficient/
derivative controls pass, plus scale/branch checks; subnormal scale misuse
is refused. At a separate small reference box, remainder~1.8934e-8 peraxis
at epsilon1/16. Composition ALWAYS recomputes bounds on its actual box.

CONDITIONAL PHYSICAL MAP. constructive_tolerance_bridge.py proves contraction
for T_y(beta)=beta+C D_epsilon^-1 P[y-F_epsilon(beta)], where P projects the
same200 exact-grid observations onto degree4 rotor polynomials and extracts
the forty cubic/lower coefficients. With the six rotors FIXED EXACTLY,
Q=D^-1 PF=G+E and |I-CDQ|<=|I-CDG|+|C||DE|. This yields local injectivity
of the full physical record with respect to twelve shape coordinates.
The fine box at epsilon1/16 gives q=.324938079 upper-rounded and nominal
eta>1.723e-12; epsilon1/8 is inconclusive(q>1.32), not impossible. Smaller
wedges improve contraction but worsen position-noise sensitivity.

UNIFORM FAMILY ACCURACY. On the broader d-radius.1 box at epsilon1/16,
q<=.502586958 and eta_nominal>2.56054e-10. For true beta* in center±r/2,
the target-centered beta*±r/2 box is inside the certified outer box and
contains the COMMON starting center. eta<=eta_nominal/2 makes it invariant.
Banach gives weighted error <=eta max_i(g_i/r_i)/(1-q). Native derivatives
are bounded on the whole outer box, so this converts to native errors.

At exact hard position error1/10^12 (enclosed UPWARD before interval use),
all12 native errors are strictly below1/1000. Maximum is distance
.000393016793236; gap<.000226129; wedges<9.52e-7deg; indices<3.958e-6;
source positions<1.582e-5; beam angles<7.293e-6deg. Largest nominal wedge is
about.45deg. True distance varies by at least±.05; this is actual contraction
of an initially wider range, not assuming the target accuracy as a prior.
All12 shape variables vary; the6 rotor variables are fixed in this theorem.

PRECISE LIMITS. The physical nominal allowance is centered at the EXACT
UNEVALUATED F(center), not a saved cubic midpoint. This is an exact projected
iteration theorem; it does not certify the numerical prototype, finite-stop
error or full-record noisy compatibility. A timestamp-to-position transfer
is available; its clock-only limits are conservative sufficient bounds,
not necessary hardware requirements or the full accuracy example's budget.
Actual model, projection, root and iteration arithmetic need separate error
budgets. Existing floating forward parity figures cannot be assumed adequate
at these extremely small tolerances.

VERIFICATION. Exact/rational and interval component controls pass. Independent
audits checked Snell rationalization, Taylor/AD remainder logic, polynomial
projection caps and normalization, the local physical composition, and the
inner-half-box theorem. The ledger validates source-report scopes and hashes
20 evidence/code files without rerunning their controls. No optical records.

REMAINING. Quantitative data-only rotor initialization/refinement, rotor
perturbations through the full inverse, preseed entry including unknown beam
and optical tails, finite-precision implementation/stopping bounds, and
actual observation compatibility. The first finite constants are a foothold
with a severe precision bottleneck, NOT practical blind full18 tolerance
closure. Immediate verified optical reconstruction and eventual T5 remain open.

## 2026-09-22: complete finite tolerance chain on a restricted full18 family

USER REQUEST. Keep going until all remaining tolerances are obtained. The
finite links now COMPOSE on a declared local family: data-derived rotor
entry, physical preseed entry, coupled18 correction, numerical arithmetic,
finite stopping, native output and full-record residual. This is NOT a
practical operating-range or original-full-prior result. No optical record
was generated/read/fitted/validated; no archive campaign, main.tex edit or
commit occurred. Main note: paper/CONSTRUCTIVE_FULL18_TOLERANCES_2026_09_22.md.

EXACT SCOPE. epsilon2^-24, A=epsilon*alpha near(.1,.125,-1/12), largest
nominal wedge~4.3e-7deg. True hardware/source coordinates range over1/32 of
the broad d-radius.1 box; beam sines±1e-7. Distance radius.003125, gap
radius~.001798. Speeds are in(3.1,1.2,.1)±5e-6Hz with fixed physical order;
phases±17deg. All18 coordinates vary, but the speed prior alone is already
narrower than native.001. Total input discrepancy from the exact ideal-grid
model is<=1e-40 peraxis. This includes measurement, clock and model errors,
not statistical sigma. Existing floating optical outputs are not presumed
to satisfy it. These are sufficient bounds, not necessary hardware limits.

ROTOR ENTRY. constructive_rotor_entry.py and control prove seven-sample
Prony on native indices0,24,...144. Exact node separation>1.54094 and
Vandermonde Gram>1.756 give error_L2<=4*delta0. Fixed declared windows resolve
aliases without giving the estimator true frequencies. The complete129-line
quartic VarPro has a finite GN basin: frequency Jacobian singular value>1,
residual Hessian norm<=4e6, radius1e-7Hz and e_next<=delta+.2e. Thirty-five
updates at the chosen case yield frequencyerror<7.54e-31Hz and midpoint
phaseerror<1.03e-27rad, with arithmetic allowances explicitly discharged.
The nuisance coefficient norm includes corrections at fundamental labels;
leading amplitudes>=4 and actual quartic fundamentals>=3.999 are distinct.

PHYSICAL PRESEED. constructive_preseed_entry_control.py proves the data-
derived quintic/formula readout enters the broad correction box for extra
effective recorderror<=epsilon^3*2.5e-11. Unknown beam variation is bounded
by exact normal source-independence plus interval beam derivatives. A
correlated no-exit continuation proof avoids destructive independent
interval subtraction: only algebraically justified value ranges are
intersected, never unproved derivative ranges. Root isolation uses fixed
[1.599,1.601], not the unknown true index. Observed constants initialize
both sources; zero beam is a proposal and both coordinates are then free.
Seed radius/outerradius<.867021 including execution, and initial distance
from unknown truth<.898271 in the weighted shape norm. The actual coupled
rotor/measurement/projection arithmetic input is below the proved gate.

ALL18 CORRECTION. The key extension subtracts the estimated exact fourth-
order Taylor remainder R4=F-F4 before EACH rotor refit. This removes the
systematic bias left by a permanently frozen quartic rotor estimate.
Simultaneously update shape by the preconditioned projected full-model
residual at current rotors. Both operations use the previous iterate.

Let a=||shapeerror||_r, b=max|midphaseerror|+40max|speederror| and
w=epsilon^2*1e-12. Uniform remainder derivatives give
|deltaR4_complex|<=S*a+U*b. Together with physical rotor-mismatch derivatives
and the finite refit theorem this gives a positive2x2 error recurrence for
(a,b/w). Its row sums are<.177684 and<.002610. A target-centered joint
radius.95 is invariant and stays in the shape outerbox. Cached frequency
entry uses the L2 conversion sqrt3<2; all refit record-energy/perturbation
hypotheses are checked uniformly. This is contraction of error to the true
member, not a new pairwise/global uniqueness claim.

EXECUTION. validated_arithmetic.py and rotor_validated_arithmetic.py supply
exact-rational linear solves and dyadic integer intervals. sqrt uses isqrt;
pi/trig/atan use bounded rational series. Root bisection64steps+192bit
formulas has hardware arithmeticerror<6.026e-21. Rotor384bit update error
<6.012e-76Hz and phase arithmetic<1.630e-95rad includes storage as rational
turns, exact gauge changes and native degrees. Physicalupdate and exact
remainder subtraction use256bits in the composed certificate. Iterates
are explicit dyadic endpoints, satisfying the next-step input contract.
Final asin output uses validated monotone bisection with error<=1e-20deg.
No libm or floating linear-solve guarantees are assumed for these kernels.
Interval physical derivative certificates still state their ivx A-fp contract.

FINITE RESULT. Forty outersteps, each with80 cached frequency updates,
give all18 native errors<2.574e-9 and full-record residual<2.074e-10 peraxis,
including native output conversion. Output compatibility is explicitly
declared as1e-9; this is distinct from the input error hypothesis1e-40.
No claim of residual<=1e-40 on arbitrary noisy data is made. The full-record
bound uses actual uniform physical derivatives, not projected residual alone.
Clock-only and input-representation allowances are recorded separately.

EVIDENCE. experiments/constructive_end_to_end_tolerances_control.py recomputes
and links all component bounds; results in
experiments/results/constructive_tolerances_2026_09_22/end_to_end.json.
The aggregate ledger records both this full18 family and the earlier
larger-wedge fixed-rotor case. Original14 exact tail controls plus9 new
rotor-derivative identities pass. New data-only polynomial, rational LS,
Prony, full VarPro derivative, readout and phase-storage controls pass.
Independent audits checked component proofs and the final composition,
including explicit perturbed-root/Newton-start margins and phase conversion.

LIMITS AND NEXT SCIENTIFIC WORK. These severe constants are not laboratory
tolerances. The original full prior is not covered and T5 stays open.
No actual optical record or runtime benchmark was used. Validated kernels
and proof controls exist; the old binary64 constructive_inverse.py driver
does not execute this complete certified variant, and solve18 is unchanged.
The earlier0.45deg theorem remains conditional on fixed exact rotors.
Practical work must enlarge the family/noise allowance, retain correlations
in the bounds and establish useful execution on observations. Global
uniqueness remains deferred. No historical recovery count was changed.

## 2026-09-22: larger-wedge initialized release and a physical noise limit

USER CORRECTION. "We need to free the bird": the tiny-wedge/noise result
is not a useful practical finish. Work targeted restriction removal while
preserving the immediate goal of one compatible full18 explanation and
deferring global uniqueness. No optical records, archive replay or targeted
recovery cohorts were evaluated. No main.tex edit or commit was made in
this pass. Main note: paper/CONSTRUCTIVE_RELEASE_2026_09_22.md.

CORRELATIONS AND ROTORS. constructive_noise_release.py multiplies the
coefficient inverse through D^-1 and the polynomial conversion before
taking absolute values. Exact rational covariance bounds avoid the former
independent-noise inflation. A covariance-weighted binary preconditioner
is independently recertified on the broad physical coefficient box.
constructive_rotor_release.py uses35 real quartic nuisance coefficients
per axis, exact all-phase Gram bounds and finite residual derivatives to
give a six-coordinate GN basin radius.001 in q=(phase,20*frequency).
The cached radius.0001 has noise gain<1.222495 in q. Observed signed DFT
fundamentals and declared speed centers give data-derived phase entry.
Neither true speeds nor true phases initialize the estimator.

OBLIQUE READOUT. constructive_coupling_release*.py cancels first-order
phase mismatch and the leading frequency tangent. Exact Laurent identities
and a finite interval frequency/phase box certify the cancellation and
remaining terms. Keeping covariance before absolute values gives fixed-
rotor shape contraction<.484543 with the newC. OriginalC comparisons and
failed higher-wedge bounds are retained. All480 algebraic reproduction
controls pass. Complete129 Fourier fitting is used only for the shape
readout; the structured35 real-column profile supplies the rotor update.

INITIALIZED FINITE COMPOSITION. constructive_joint_release14_control.py
keeps twelve separate normalized shape errors and maximum frequency/phase
errors. At epsilon1/16 (largest nominal wedge~.45deg), truth in half the
broad shape box and true N=(3.1,1.2,.1)±5e-6Hz, native phases±17deg,
declared order/signs, the common shape center is a permitted initialization.
Observed DFT phases enter the .001 rotor ball. Sixteen GN updates, then
one tail-subtracted sixteen-step refit, give qerror<2.627e-9 and enter the
joint invariant box. Every subsequent rotor refit subtracts the current
exact R4; shape correction uses the projected full-model residual.

The positive14x14 recurrence has max row sum.485977067. At hard per-axis
input error7.62958549e-12,120 outersteps (sixteen GNinner updates/refit)
give all18 native errors<.001; largest bound.0009, gap<.000444673.
The derived native accuracy noise cap is8.47731721e-12. The chosen eta is
nine tenths of the smallest accuracy/invariance/entry cap. The independent
FULL record residual bound is4.45412023e-5, exceeding1e-5 and vastly exceeding
the input eta. It must not be reported as eta-compatible reconstruction.

Exact inverse execution is a premise of this NEW composition. Upward
dyadic rounding in the certificate recurrence does not discharge execution
arithmetic or native output rounding. The older epsilon2^-24 chain retains
its own complete arithmetic proof; it cannot be transferred automatically.
No optical solver runtime/recovery was benchmarked. New result:
experiments/results/constructive_release_2026_09_22/joint14.json.
Independent audits checked/reran its normalization, per-coordinate matrix,
startup, cache, gauge containment, native indexing and whole-record bound;
all passed, with source hashes matching.

HIGHER ORDER AND ITS LIMIT. constructive_preseed_release.py supplies
interval Taylor/remainder derivatives through configurable degree16 and
an exact degree8 reference preseed. Degree4 agrees with all earlier value,
shape and rotor derivative controls. Exact degree8 native DFT projection
retains aliases. That separate affine preseed has explicit effective-error
and arithmetic premises; it is not needed by joint14's common-center prior.
An unrestricted degree6 nuisance fit is NOT uniformly invertible: an exact
Chebyshev identity at native phases(15,-15,7.5)deg makes its Ydesign rank
deficient on all200 samples. This is a nuisance-space degeneracy, not a
physical inverse nonidentifiability proof. Preserve physical structure in
any higher-order extension.

REAL FINITE-NOISE ACCURACY LIMIT. The new ambiguity control constructs
two exact physical systems with d=100±.0011, common n,g, normal source,
and alpha_i(d)=alpha_i(100)L_i(100)/L_i(d). Their leading coefficients
coincide exactly. Degree8 exact polynomial differences plus interval
remainder derivatives bound the complete scan difference uniformly over
every common rotor state. Both endpoints and their segment are enclosed
outward; all branch margins are>.937 on the saved wedge ladder. Half-
separation bounds are1.97068931e-9 at epsilon1/16 and8.86528112e-6 at
epsilon1 (largest nominal wedge~7.18deg). Thus hard noise1e-5 can hide
which distance generated one midpoint record. Every estimator has distance
error at least.0011 in one of the two possibilities. This is NOT a
noiseless uniqueness claim and does NOT obstruct one compatible explanation.
No midpoint observation array was generated. Independent audit passed.
Evidence: constructive_release_2026_09_22/ambiguity.json and the linked
CONSTRUCTIVE_PRESEED_RELEASE_AMBIGUITY_2026_09_22.md proof.

STATUS. The microscopic-wedge assumption is removed for the initialized
mathematical chain, but the practical bird is not free. Speeds and hardware
are still narrowly localized beforehand; many prior coordinates already
vary less than.001. True distance may vary±.05 and gap±.02877, so those
coordinates do require substantive correction. Useful full-prior entry,
attainable-noise whole-record compatibility, new execution arithmetic and
actual optical execution remain open. The1e-5 detector-error benchmark is
provisional research context, not a supplied physical specification. T5
and the immediate verified reconstruction milestone remain unfinished.

## 2026-09-22: turbo pass closes broad blind speed + attainable-noise compatibility

USER REQUEST. "Kick it into turbo drive ... don't stop." Continued the
restriction-removal work toward one admissible full18 explanation, with
global uniqueness deferred. The stopped optical campaigns were not resumed;
all controls remain symbolic, interval, rational or explicitly nonoptical
multitone/affine fixtures. No optical record was evaluated, no manuscript
claim was changed, and no commit was made.

MAIN COMPOSED RESULT. paper/CONSTRUCTIVE_BLIND_COMPATIBILITY_2026_09_22.md
and experiments/constructive_blind_compatibility_control.py now connect
broad blind rotor entry to one physical scan-compatible output at hard
per-axis input discrepancy1e-5. The full-record residual bound is
2.042007487e-5, with declared output tolerance2.1e-5. Input and output
budgets remain distinct; native accuracy to generating hardware is not
claimed. This removes near-known speed centers and tiny input noise
together, not merely in separate incompatible component lemmas.

SPEED SCOPE. .3<=|N_i|<=3.5Hz, signed pair gaps>=.5Hz, circular separation
modulo1Hz>=.02Hz. Unknown signed speeds range across this class; no N*
center is supplied. Physical order is unknown and handled by six branches.
Native phases±17deg and declared physical wedge signs(+,+,-) remain.
Hardware is STILL the prior halfbox near0.45deg, with d100±.05,g6±.02877
and narrow glass/wedge/source/beam ranges. The original hardware prior and
excluded/colliding speed classes are not covered. The older witness with
speed.1Hz is outside this new broad class; do not silently merge scopes.

BLIND FRONTEND. constructive_blind_rotor.py double-centers90x91 Hankel
matrices and uses shifts1,20. This removes arbitrary source DC without
differencing away slow tones. Exact Vandermonde/subspace/resolvent bounds
give unordered signed node disks and data-only alias matching. Cubic
positive fundamentals are absorbed into their free complex amplitudes;
remaining harmonics, physical R3 and measurement error form an arbitrary
normalized pointwise perturbation<.000984293. This bound is uniform over
all rotors on the declared shape halfbox.

Generic three-tone+DC profile conditioning uses actual finite Dirichlet
derivatives, not the earlier DFT witness. Its projected derivative Gram
is>129, noisy Jacobian singular value>=52 and residual Hessian<=32000.
The pencil enters radius.0005Hz. Sixteen GN steps enter.00005Hz; sixteen
more give frequency L2 error1.9844598272e-5Hz and midpoint phasor-phase L2
error.0005637524791rad. A covariance-sensitive coefficient calculation
keeps the phase bound small. Independent audit checked arbitrary DC,
complex phases, nonzero-residual derivatives, invariant balls, cubic
parity/absorption and harmonic aliases; no blocker was found.

DIRECT COMPATIBILITY. constructive_residual_release.py propagates the
exact physical15-coordinate(shape,rotor-state) Hessian uniformly over the
outer box and independent rotor states. The affine remainder includes
second-order trigonometric phase/frequency terms. No inverse-Jacobian norm
or rank assumption enters the minimax compatibility lemma. A true feasible
step gives affine residual<=eta+R+a; an LP within gap g returns physical
residual<=eta+2(R+a)+g+o. Hardware accuracy is not needed for this argument.
For the new broad rotor bounds, R<=5.190037437e-6; a=g=o=1e-8 gives the
stated2.042008e-5 output bound.

Six tone-to-prism branches subtract the declared negative-wedge half-turn,
unwrap, and filter using an enlarged native phase chart. The correct one
survives. Frequency steps are intersected with the native±3.5Hz prior,
including true endpoints. Choose the smallest certified affine residual
among box-feasible candidates; the correct branch supplies the upper bound.
All surviving outputs obey original native bounds, including phase magnitude
<17.135686deg. Exact branch/gauge/endpoint controls pass. No optical LP ran.

ARITHMETIC. affine_compatibility_lp.py implements a HiGHS proposal and exact
rational primal/dual check: normalize lambda to L1<=1, then lower-bound
the minimax objective by lambda*b-support_box(A^Tlambda). No optimizer
status or rank is trusted. A400x18 nonoptical rational control has verified
gap4.154e-16; singular/asymmetric/fixed-coordinate/tiny-unit/bad-proposal
controls also pass. This is a per-input gap certificate, not a bound on
runtime or guaranteed numerical proposal success on every possible input.

affine_validated_arithmetic.py adds dyadic interval AD and a uniform width
proof over all reference shapes/rotors. At192bits, full value+Jacobian
affine error<7.57e-55. Phases are rational midpoint TURNS, with outward
conversion of radian step bounds.96 validated inverse-sine bisections
give output-representation forward error<3.768e-27; other coordinates and
phase gauges stay exact rational. Output membership is checked after
rounding. Independent audits passed. Any later binary64 conversion needs
its own error check. Frontend SVD/eigen/GN execution still assumes the exact
map and its phase evaluation has a separate1e-8 allowance; the complete
optical implementation/runtime has not been validated.

HARDWARE BREAKTHROUGH AT THE COEFFICIENT LEVEL. global_normal_cubic.py
uses D=c102*l2-c012*l1 and H=c201*l2^2+c021*l1^2-c111*l1*l2. Exact algebra
gives c003*H/D^2=3(n3^3+1)/(n3(3n3+1)^2), strictly decreasing on the FULL
glass prior[1.3,1.8], with derivative magnitude>=4325/221184. D never
vanishes for nonzero wedges and positive original geometry. One globally
monotone scalar cubic and explicit formulas recover all8 normal hardware
coordinates without a near-correct prior center. The old selected(U,V)
chart actually folds at an admissible n3=7/4,d50,g~11.826; the new mixed
information avoids that coordinate failure. Five exact coefficient controls
and bounded-coefficient propagation pass. At centered source/normal beam,
the full12 cubic coefficient Jacobian has rank12 throughout the nonzero-
wedge hardware domain; each even block determinant is
A3^2*d*(n3-1)*(3n3+1)/2. This is not yet finite-record recovery.

GLOBAL OBSERVABLE HARDWARE COORDINATES. observable_hardware_route.py uses
two flat constants and three linears to eliminate all wedge sines and both
sources, with NO nonzero-wedge/source/beam assumption. Seven variables
remain(n1,n2,n3,d,g,t_x,t_y). A uniform leading-gain lower160/13 holds over
the original physical prior, so the coordinate map has no vanishing gain.
Exact arbitrary-beam last-only cubic formulas absorb upstream hardware
into the observed constant. For fixed n3,d,C, q33/l3^2 is strictly increasing
in beam tangent with derivative>3/(20d)>=3/4000 on the whole physical prior.
An exact216-coefficient Bernstein proof and finite dyadic root isolation
certify this. When the last wedge is nonzero, two beam inversions leave
only(n1,n2,n3,d,g) free; shared-wedge and other coefficient constraints are
retained. The seven-variable chart remains for zero last wedge. These
global coefficient reductions still need a justified noisy finite-record
Taylor readout and a useful feasible remaining-parameter construction.

EVIDENCE. Main composition: constructive_turbo_2026_09_22/blind_compatibility.json.
Other new reports: same folder global_normal_cubic.json, affine_lp.json,
affine_arithmetic.json; constructive_release_2026_09_22/blind_rotor.json and
residual.json; observable_hardware_route_2026_09_22/control.json. Proof notes
link the exact scopes and source hashes. Main scientific remainder is now
broad hardware entry and the finite-record coefficient bridge, with frontend
validated execution and optical evidence still outstanding. T5 remains open.

## 2026-09-22: direct native features remove the local hardware box from five-variable recovery

CONTINUED TURBO ATTACK. The following is a new direct-data result, stronger
than an exact-flat-coefficient chart. It still does not complete the
full18 reconstruction milestone. All new checks were interval boxes,
symbolic/rational algebra or nonoptical designs. No optical observation,
case hunt, stopped campaign, or manuscript change was performed.

NATIVE FEATURE BREAKTHROUGH. Fix seven free physical coordinates
(n1,n2,n3,d,g,t_x,t_y) and a candidate rotor tuple on the stated separated
ABSOLUTE-speed class. Two fitted scan constants plus three X fitted leading
coefficients define five DIRECT native features. For the physical flat
coordinate z=(Cx,Cy,lx1,lx2,lx3), their exact full-model feature map is
Phi(z)=z+P(F-P1)(z). The nonlinear term is computed from the same physical
hardware; no free high-degree coefficients or unobserved Taylor quantities
are substituted for measured features.

Uniform interval AD now proves ||D(Phi-I)||2<=.948900193 on ALL original
glass, distance50..200, gap2..15, source positions[-5,5], and BOTH independent
beam angles[-25,25] degrees, with signed wedge sines|A_i|<=1/80 (about.716deg).
Zero wedges are included. There is no near-true hardware center. A smaller
|A|<=1/160 family (about.358deg) has q<=.652224271. The full glass range
includes the exact binary64 upper endpoint stored in model.HI.

SAMPLING. On the original200 times j/20, candidate absolute speeds lie in
[.2995,3.5005]Hz with absolute-speed gaps>=.499Hz, and phases are arbitrary.
This includes component errors<=.0005 about the true class |N| in[.3,3.5]
with absolute gaps>=.5. Signed gaps alone do not suffice. Exact Dirichlet
caps and a rational7x7 positive-LDL comparison prove normalized real
constant+three cosine/sine design Gram>=.72. This gives collective X feature
norm<=sqrt(2/.72) and Y-constant norm<=1/sqrt(.72). Rotor discovery or its
error on this broader hardware family is not supplied by this theorem.

GLOBAL PHYSICAL DOMAIN. At fixed free7, native source/wedge bounds become
the convex polytope |Ca-offset_a|<=5, |lx_i|<=Amax*Gxi(Cx), with positive
affine gains Gxi. Projection of b-P(F-P1)(z) onto this domain is a q<1
contraction. Every feasible start converges to the unique projected
candidate on that fixed free7 fiber. For noiseless compatible features it
recovers the corresponding five physical coordinates. For arbitrary/noisy
features the fixed point can sit at a prior boundary without matching them
exactly; a fixed point is not proof that the entire scan is compatible.

EVIDENCE IS A COMPLETE COVER, NOT A COHORT. direct_feature_contraction.py
uses36 geometry leaves atA1/160 and181 atA1/80. Each leaf covers8 beam cells
and all64 independent X/Y pairs, keeping the cross-Cx-to-Y wedge coupling.
Sources and wedges remain whole intervals; all rotor values[-1,1] are
enclosed. Exact volumes, containment and disjoint interiors verify full
coverage. Independent root audit recomputed2304 and11584 pair bounds from
the saved derivative enclosures and checked the full cover. Ten exact
source/gain identities and independent DualSeries controls pass.

FINITE PHYSICAL PROJECTION. projected_feature_chart.py reduces projection
to at most4 scalar quadratic pieces, then independently checks all32
polytope vertex variational inequalities with exact rationals. Algebraic
gain coefficients are outward-enclosed; a rational inner polytope keeps
every approximate iterate physically feasible. An explicit directed
Hausdorff bound certifies its projection error. A generic oblique flat-
algebra control has projection error<3.85e-15. This is not an optical
iteration. Exact-model feature evaluation still needs its per-update
arithmetic gate; a polynomial physical tail also counts as update error.

At512 updates with total feasible update error<=1e-12, the native-feature
composition bounds distance to the projected fixed point by6.13e-11
(A1/80) and2.88e-12(A1/160). These are conditional finite-iteration bounds,
not executed optical runs. At per-axis input error1e-5, bounds relative to
a compatible true FEATURE tuple are about.000399462 and.0000586942;
neither number is a native-hardware error or a whole-record residual.
Using T8 with its5.817e-8 physical tail instead must explicitly charge
about1.19e-7 feature error per update, not silently use the1e-12 gate.

WHY FREE CUBIC READOUT WAS A TRAP. finite_polynomial_bridge.py gives an
exact resonance obstruction inside the broad speed class: N=(.4,1.2,2.7),
zero phases, has u2-4u1^3+3u1=0 on X and v2-3v1+4v1^3=0 on Y for every
time. Free linears/cubics are not separately observable. Even without
resonance, eta*T5 and eta*T7 prove free-polynomial cubic accuracy floors
20eta and56eta. Nonoptical degree6 sampling/control gains confirm that
a small optical tail alone cannot resolve this. These are free-polynomial
obstructions, not physical hardware nonuniqueness proofs. Keeping all
higher terms on the physical coefficient manifold led to the native
feature correction above.

OTHER COMPLETED SUPPORT. global_hardware_tail.py proves full-hardware,
full-beam forward polynomial remainder<5.817e-8 atA1/80 and degree8;
degree6 gives5.416e-6. AtA1/40 (~1.433deg), degree10 gives5.163e-6.
These are uniform forward approximation bounds, not solver guarantees.
At normal beam with fixed leading coefficients, degree6/8 tails over the
full glass/geometry/source ranges are1.962e-6/1.509e-8.

observable_distance_components.py completes the exact-coefficient distance
branch decomposition without assuming connectedness: at most25 components
for an exact quadratic ratio, each with at most one cubic-ratio distance.
This reduces the exact-coefficient problem to4 continuous variables plus
finite branches. An uncertain quadratic ratio gives at most47 complete
components with exact n,C; cubic-band interval fiber checks give necessary
enclosures/exclusions only. The controls retain a singleton atd200 and a
valid ratio band whose midpoint branch is empty. Independent audits pass.
These exact-coefficient reductions do NOT turn the native problem into4
free variables; the present direct-native profile has7 free hardware
coordinates, and noise slack/rotors still require explicit treatment.

FRONTEND EXECUTION GAP CLOSED FOR ACCEPTED RECORDS IN THE EARLIER FAMILY.
frontend_arithmetic.py replaces trusted SVD/eigen/atan execution by exact
rational subspace/pencil checks, dyadic unit-node eigenvector residuals,
32 validated frequency updates and an interval phase-sector check. An
explicit coarse+fine error<.5Hz gate justifies principal unwrapping. The
earlier broad-speed/local-hardware compatibility result now charges
1e-12 per update and gives residual2.04200751301657e-5 atinput1e-5, still
below2.1e-5. Independent audit and nonoptical execution controls pass.
Universal proposal acceptance/runtime and an actual optical run remain
unproved. This local-hardware theorem must not be silently combined with
the newer full-hardware native-feature chart.

CURRENT SCIENTIFIC GAP. Construct/select the remaining7 hardware coordinates
and rotors from the scan while preserving physical/noise correlations,
then certify the complete observed-record residual. Fitting five selected
features alone does not preserve the original minimax noise budget. Global
uniqueness remains deferred. No claim of completed broad full18 analytical
recovery, practical full-prior coverage, or T5 closure is made.

NEW REFERENCES. DIRECT_FEATURE_CONTRACTION_2026_09_22.md,
PROJECTED_FEATURE_CHART_2026_09_22.md, GLOBAL_HARDWARE_TAIL_2026_09_22.md,
FINITE_POLYNOMIAL_BRIDGE_2026_09_22.md,
OBSERVABLE_DISTANCE_COMPONENTS_2026_09_22.md, FRONTEND_ARITHMETIC_2026_09_22.md.
Final full-hardware covers are direct_feature_a160_final.json and
direct_feature_a80_final.json in constructive_turbo_2026_09_22; independent
aggregation is direct_feature_cover_audit.json. The older exact-map
joint.json benchmark retains a stale dependency hash and is not current
execution evidence. Current frontend/broad-compatibility reports are fresh.

## 2026-09-22: source elimination reduces the broad native inner solve to three wedges

The next pass sharpens the same broad-hardware direct-data construction.
The exact full-model fitted DC is affine in its own source coordinate:
P0 F(A,p)=U(A)+S(A)p. Uniform bounds give S>0. The clipped quotient
clip((b_DC-U)/S,[-5,5]) therefore eliminates each source explicitly. Only
v_i=Gxi(0) A_i remains in a three-dimensional rectangle contraction.
The Y source is read after the wedges and never feeds the X iteration.

The contraction bound from the saved complete covers is .568597236443 at
wedge sines <=1/160 and .803148671770 at <=1/80. This is a different
projected map from the five-feature polytope construction; its noisy
candidate and noise calculation must not be interchanged with that map.
The unsuccessful partial five-feature q<.8 cover remains incomplete;
it is not evidence for the new bound, which follows from source elimination.

With the same seven free hardware coordinates and rotors as a compatible
true member, per-axis native error <=1e-5, 128 feasible wedge updates each
accurate to 1e-12, and final feasible source readouts accurate to 1e-12:
at Amax=1/80 the scaled-wedge error is <8.466814e-5, native wedge-angle
vector error <.000394185 degrees, px error <5.735e-5, and py error
<6.961e-5. The three wedge-sine errors are each <7e-6; both source errors
are <7e-5. At Amax=1/160 the corresponding angle bound is .000179855
degrees, px <2.594e-5 and py <2.894e-5. These are conditional inner
coordinate errors, not recovery of free7 or the unknown rotors.

The source transfer retains the Y derivative numerator divided by X
zero-source gains, because v uses X scaling. All 2,304 and 11,584
independent beam-pair chains were aggregated from the complete saved
covers. No new physical boxes were needed for this step. Root reran the
finite composition control; the source clipping, cross-axis chain and
shared-candidate noise bounds passed independent read-through audits.

The exact full-record factor-two counterexample explains why the noisy
feature profile needs a correction. A true zero-wedge system has residual
eta to symbolic data (X,-Y), while a physical last-wedge-only system has
the same five selected features as the data and residual 2*eta. Native
sample quarter-period symmetry proves this for a continuous amplitude
family including eta=1e-5. It does not obstruct finding a compatible
physical explanation. A boundary-safe normal-cone/ellipse argument also
supplies general conditional residual bounds, with operator constants
still required. No optical observation array was evaluated.

One audit found and corrected a separate exact-algebra endpoint bug in
observable_distance_components.py. The isolating interval (1,2) for
sqrt(2), with defining polynomial (x-1)(x^2-2), caused a false zero sign
for x-1 by counting an unrelated endpoint root. Removing endpoint factors
before interval gcd/root counting fixes the issue. Six exact regressions
cover positive and negative sqrt(2) and either/both endpoints; singleton
semantics and coarse-isolation point/band controls pass. Both refreshed
distance reports bind current module hash 67f6bc68df8488ece21b1b14e4d9f43c0fcb62661d87ecbe918ac4c07a640392.

References: NATIVE_SOURCE_FIRST_CHART_2026_09_22.md,
SOURCE_FIRST_READOUT_2026_09_22.md, FEATURE_PROFILE_RESIDUAL_2026_09_22.md.
Reports: source_first_contraction.json, native_source_first_chart.json,
source_first_readout.json and feature_profile_residual.json in
constructive_turbo_2026_09_22. No optical inverse iteration, replay,
case hunt, archive recovery, or paper/main.tex change was performed.

## 2026-09-22: a full-record candidate-or-exclusion theorem for each broad hardware fiber

The data-derived inner localization now supports a rigorous full-record
correction, removing the factor-two selected-feature loss without a
feature-noise grid. The seven free hardware coordinates and candidate
rotors are fixed for each invocation, but need not be assumed correct.
The original full glass, distance, gap, source and independent beam ranges
are covered, with the existing wedge-sine bound 1/80 and absolute-speed
separation. Final output also checks the original native rotor priors.

FIRST, LOCALIZE EVERY COMPATIBLE INNER POINT. For any eta=1e-5-compatible
source/wedge member at those fixed outer coordinates, the same observed
source-first map yields the same feasible center. The uniform contraction
and source transfer therefore place every such member within 7e-6 in
each wedge sine and 7e-5 in each source coordinate of that center. This
implication is valid for every proposed outer tuple, even if no compatible
member actually exists. It does not require a known true center.

SECOND, ONE FULL-RECORD AFFINE MINIMAX CORRECTION. Intersect the center's
displacement rectangle with the original physical A/source box. It is
convex, contains the center, and contains every eta-compatible member if
one exists. Whole-prior directional Taylor arithmetic, with all rotor
states enclosed simultaneously, bounds the first-order remainder on every
such segment by R=2.241659172045633e-7. The second formal coefficient is
already one half of the directional Hessian; the Taylor integral has no
additional factorial. The minimum physical margin is >.7697. Zero-step
and source-only remainders are exactly zero; independent nonoptical
polynomial, reciprocal and radical coefficient controls pass.

Use all 400 native record coordinates in the affine LP, with exact
rational interval-midpoint coefficients. Let L and U be its certified
dual lower bound and primal upper bound; alpha bounds its arithmetic
error. If L>eta+R+alpha, NO eta-compatible physical inner point exists at
the fixed outer tuple. If the gap U-L<=delta=1e-8 is accepted and that
exclusion does not hold, the physical candidate has full-record residual
at most eta+2R+2alpha+delta+output_rounding. This is a genuine fixed-fiber
candidate-or-exclusion dichotomy, not a claim that noisy features match
exactly. Unaccepted numerical proposals remain unresolved unless their
primal or dual alone establishes a verdict.

The uniform LP F/J arithmetic error at 192 bits is <5.685e-54. Final
native-output rounding contributes <1.894e-26; inverse-sine interval
clipping handles native endpoints without an inner-half-box assumption.
The resulting convenient full-record bound is <1.046e-5; the saved bound
is approximately 1.045833183440913e-5. It is deliberately larger than the
input noise cap, by about 4.6 percent. The exact sine-coordinate candidate
is in the fixed fiber; rounding native beam angles can move the final
output infinitesimally off that fiber, covered by the output allowance.
The whole-fiber exclusion always concerns the original exact outer tuple.

The two affine source variables can also be eliminated exactly. At a
fixed wedge step each source belongs to the intersection of its native
displacement interval and all row residual intervals. Pair inequalities
and the two interval endpoints produce an explicit convex maximum of
affine functions of three wedge steps. Sources are reconstructed by
interval intersection. This yields a four-variable LP including the
residual cap, but has 80,401 planes for the native record, so it is not
claimed faster than the original 800-inequality five-variable fit.
Exact sparse dual-plane controls, source endpoints/collapsed intervals,
a coupled nonoptical LP and 400-row gate controls pass. Root and the
practical-bound agent independently audited the composition and source
elimination.

EXECUTION AND OUTER GAP. The feasible source-first updates accurate to
1e-12 and final clipped-source readouts accurate to 1e-12 are still
explicit execution gates. This new arithmetic result closes the LP F/J
and native output allowances only. No optical record, physical inner
iteration or optical LP was run. Selecting the remaining seven hardware
coordinates and rotors from the record remains unsolved. The new oracle
tests one such choice; it does not prove useful global outer complexity.
The archive ledger and full18/T5 completion status remain unchanged.

The bounded outer analysis now gives the exact interior native-profile
derivative F_h-F_x(P F_x)^(-1)P F_h. It retains direct/compensating hardware
correlations. A rational physical cubic-polynomial control at native speeds
(3.1,1.3,.3) has a rank-seven minor using X cosine harmonics .5,.6,.7,.9,
1.5,2.3 Hz and Y cosine .6 Hz. These are actual finite-sample functionals
of the polynomial, including cubic contributions to fitted fundamentals;
no optical values are generated. Root reran the exact control. This is
not a full-model or uniform rank theorem. A leading-only geometry/wedge
alternation has exact identity Jacobian in (d,g), so affine updates alone
cannot supply the missing contraction. Full-model transfer, boundary and
inactive-prism branches, and data-derived outer entry remain open.

New references: PROFILE_AFFINE_COMPATIBILITY_2026_09_22.md,
PROFILE_AFFINE_ARITHMETIC_2026_09_22.md, NATIVE_OUTER_PROFILE_2026_09_22.md.
Reports in constructive_turbo_2026_09_22: profile_affine_remainder.json,
profile_affine_arithmetic.json, profile_affine_compatibility.json and
native_outer_profile.json. Final direct-feature covers reevaluated all
physical derivatives with current code; older seed reports supplied only
geometric partitions and remain historical provenance, not current
derivative certificates. Source hashes and partition-input file hashes
are checked according to those roles.

## 2026-09-22: attacking the 0.716-degree restriction with exact slope windows

The user explicitly rejected the 0.716-degree wedge range as impractical
and asked to free that restriction substantially. The working target is
at least five degrees while retaining the original broad glass, geometry,
independent beam/source priors and generic separated-speed/arbitrary-phase
sampling class. No special rotor schedule is substituted. Outer hardware
and rotor selection remains open; global uniqueness stays deferred.

A new exact-response route replaces the small-wedge Taylor-defect test.
Transform each actual observed coordinate by atan(y/d-B0/d), with B0 the
known candidate hardware's zero-source flat upstream offset. The exact
angular/centered-position recurrence cancels the final tangent amplification
before interval evaluation. Source is now a nonlinear monotone scalar
inverse in z=p/d, not affine division. With relative wedge-slope radii ri
and source slope window c0±s, joint fitted-source/linear covariance gives
q² <= 2*(211/400)*sum(ri²)/[.72-(s/c0)²]. Twelve general symbolic identities
and independent kernel/tolerance audits pass. All calculations are exact
algebra, nonoptical controls or whole parameter/effective-state intervals;
no optical record or physical inverse iteration is evaluated.

Initial five-degree state-box controls passed at difficult fixed hardware,
but are not full-prior coverage. A first full-cover attempt using the raw
exact response remained much too loose. Glass-normalized atan derivatives
improved coverage: cancel each column's own ni²-1 algebraically, retaining
that actual known factor in the physical wedge scaling. After 2,000
hardware nodes the preserved partial cover had 995 accepted leaves and
covered 64.26697 percent of the native hardware/positive-beam volume. The
last 1,000 nodes added only about 1.5 percent; that subdivision strategy
was stopped. The report remains INCOMPLETE. Partial accepted-cell noise
bounds also became looser than the earlier small-wedge bounds; no old
precision or full-record error constants are transferred automatically.

Further exact reductions now being combined: monotone incoming-angle and
wedge endpoints bound q and outgoing angle throughout each prism box;
intersections retain T²=n²-q² and monotonicity of Delta and 1/(T+R).
Normalized positions and derivative numerators are affine in
(1,1/d,g/d,p/d). Their native geometry/source polytope has eight vertices.
Dividing a derivative numerator by the positive known lever
Li=1+(3-i)*g/d preserves vertex extrema. Thus ci(h)=(ni²-1)*Li*midHi(hcell)
removes known geometry dependence before bounding. Independent exact
convex-combination/linear-fractional controls and full-geometry flat-gain
controls pass. This reduces a continuum by proof, not by sampled cases.
The combined complete cover is still being checked at this log entry.

Separate branch analysis proves that the whole |Xi|<=7/80 cube (containing
five degrees), all native glass and beam ranges, remains transmitted and
forward: minimum exit TIR margin exceeds .470793. The full unrestricted
ten-degree cube contains an aligned TIR configuration; physically
admissible ten-degree subfamilies are not ruled out. This is a physical
branch distinction, not an inverse impossibility claim.

New references: BROAD_WEDGE_RELEASE_2026_09_22.md (scope and construction),
BROAD_WEDGE_INVERSE_2026_09_22.md, EXACT_MONOTONE_WEDGE_2026_09_22.md,
EXACT_MONOTONE_WEDGE_KERNEL_AUDIT_2026_09_22.md and
EXACT_MONOTONE_WEDGE_BRANCH_2026_09_22.md. New core files include
direct_slope_contraction.py, broad_wedge_inverse.py,
broad_wedge_tolerances.py, broad_wedge_geometry.py,
broad_wedge_geometry_monotone.py and monotone_prism_bounds.py. The old
0.716-degree full-record candidate-or-exclusion theorem remains valid
under its original assumptions. Five-degree full-record compatibility
and actual inverse-update arithmetic remain separate obligations.

## 2026-09-22: COMPLETE five-degree full-hardware inner cover

The wedge-range target is now proved: |Ai|<=7/80 contains at least ±5
native degrees for every prism, approximately seven times the previous
.716-degree range. Full native glass1.3..1.8, distance50..200, gap2..15,
sources±5, independent beam angles±25deg, generic separated speeds and
arbitrary phases are retained. The seven free hardware coordinates and
rotors remain fixed candidate inputs; this is the five-coordinate inner
inverse, not completion of full18 reconstruction.

The successful proof combines monotone prism endpoint enclosures with
geometry-affine propagation and known glass/lever normalization. A new
weighted partition and earlier state refinement improved the first250
nodes to68.9453 percent coverage, exceeding the old2000-node attempt.
Eight-process state evaluation is exactly equal to serial interval union
and accelerated continuation. The final report
wedge_release_bounds_affine_cover4.json has637 hardware/beam leaves,
269,360 accepted effective-state subboxes, exact covered fraction1 and
ZERO pending boxes. The binary tree has1273 nodes. The worst certified
contraction is .9948779512944627; the minimum physical margin is
.4707933568097432 (displayed decimal approximations to exact bounds).
The report SHA256 is
5e190a04e2623105af5517b61d5e0550415cb04ef40a644b99f13f91ebca8264.

Root's independent final audit is PASS: exact whole partition and volume,
all saved rational q² values, physical ci(h) glass/lever factors, positive
source margins, Gram=.72, individual rotor RMS²<=211/400, joint data-gain
bounds, and preserved seed leaves/splits/source hashes. Harmonic reviewed
all downstream consumed constants and independent-axis aggregation. The
final audited finite report is broad_wedge_cover_audit.json. Source and
kernel files were frozen throughout the complete cover.

CONDITIONAL NOISE/EXECUTION LIMITS. For native per-axis RMS discrepancy
1e-5,8192 feasible wedge updates each within1e-12, and final normalized
source readout error1e-12, uniform bounds are wedge-angle vector error
<.011790deg, px error<.015707 and py error<.029137 native length units.
The finite iteration part in scaled wedges is<1.953e-10; noise dominates.
The Y source bound deliberately uses conservative cross-leaf maxima while
retaining the actual shared glass/geometry factors. These tolerances are
materially looser than the old small-wedge result. No original full-record
residual guarantee is inferred from feature recovery. Actual feasible
update arithmetic and a five-degree full-record compatibility correction
remain separate gates; the old.716deg LP constants are not reused.

Further refinement, not needed for the completed cover: exact own-flat
angular normalization K_i=sqrt(ni²+(ni²-1)tan(beam)²)-1 removes the entire
known zero-wedge angular gain before interval evaluation. Product,
additive-defect and quotient identities have independent exact controls
and audits. It certifies the entire lower half of the positive beam
range over full hardware in one64-state cell, q<.958, but does not close
the high-beam region unsplit. It is kept as a separate kernel and is NOT
mixed into the final cover. Independent X/Y K factors differ; exact bounds
include the simple shared-index ratio cap6/5. This may permit a smaller
future cover or sharper conditional bounds.

Scope remains explicit: no optical observations, scan arrays, inverse
iterations or optical fit trials were evaluated; no recovery campaign or
paper/main.tex change was made. The outcome removes the.716deg restriction
for the full-prior inner contraction. Outer selection and the wider-angle
full-record noise theorem are still open. HANDOFF.md and FINISH_LINE.md
now reflect this result. Main proof note: BROAD_WEDGE_RELEASE_2026_09_22.md;
additional exact geometry and flat-scale notes/controls are saved beside
it. All current reports are in constructive_turbo_2026_09_22.

## 2026-09-23: user requires at least fifteen degrees; remaining scope clarified

The user now explicitly requires at least15-degree wedges and asks whether
rotation speed, distance and initial angle are unrestricted enough. This
supersedes five degrees as the practical target; the completed five-degree
inner theorem remains a valid intermediate result. No fifteen-degree
inverse guarantee is claimed.

Checked current model.py/core.py, the authoritative latest diary entries,
HANDOFF and the native sampling proof. At five degrees, full original
screen-distance50..200, inter-prism gap2..15, independent incoming beam
angles±25deg, glass1.3..1.8 and source positions±5 are retained. The
source-to-first-prism distance6 and prism thickness3 remain fixed model
constants. Nominal speed magnitudes are.3..3.5Hz (18..210rpm), either
rotation direction, with pairwise magnitude separation at least.5Hz
(30rpm); the proof has the already documented tiny candidate buffer.
This is for200 ideal samples at20Hz over the10-second observation window.
Equal/near-equal absolute speeds are outside this particular guarantee,
not proved physically unidentifiable. Rotor-phase sampling bounds allow
all phases, but the original native output prior remains±18deg. Initial
beam angle and starting rotor phase must not be conflated.

Range coverage is conditional on fixed candidate rotor settings and the
seven other hardware coordinates. Recovering those coordinates from the
scan remains open. The five-degree full-record noise guarantee and
actual feasible inverse-update arithmetic remain separate obligations.

For15degrees, the old blanket independent box cannot be declared
transmitted: the already proved10-degree aligned TIR example is inside
it. This does not preclude reconstruction of physically valid15-degree
systems. Independent bounded direction-envelope reasoning found whole
transmitted15-degree families with full glass range and beam±4deg, or
with glass1.3..1.35 and beam±25deg. These are existence examples and
physical-constraint diagnostics, NOT silently substituted target priors
or fifteen-degree inversion results. The wider target requires a
transmitted-margin domain, which need not be rectangular; the current
rectangle projection/contraction does not automatically extend.

Only repository reads, exact angular envelopes and research-scope updates
were executed here. No optical records, position arrays,
physical inverse trials or stopped campaigns were run. HANDOFF now makes
fifteen degrees the explicit minimum target.

## 2026-09-23: material realism without hard-coded glass identities

The user clarified that the intended optics use standard materials and
air/vacuum-like surroundings. Refractive indices around 4, 5 or 6 should
be treated as a warning for this application, not accepted as an escape
route to a fit. This does not authorize hard-coding a glass species or a
convenient numerical index. Preserve continuous unknown prism indices.

Readback confirms the native model.py prior is 1.3..1.8 independently for
each prism; core.py currently uses a surrounding index of 1.0 at every
air segment. The surrounding value is an explicit existing assumption,
not an inferred medium parameter. No numerical bounds, forward physics,
solver or frozen proof reports were changed. No optical evaluations ran.
The earlier aligned TIR obstruction uses index 1.8, so the 15-degree
recovery obligation remains relevant inside the intended material range.
The preference is recorded in HANDOFF.md for future work.

## 2026-09-23: quantitative fifteen-degree domain and exact prism inverse

The user requested continued work with realistic continuous glass indices.
Three independent mathematical tasks and root's exact whole-box contractor
work addressed the fifteen-degree physical domain and inverse primitives.
No optical record, sample-state sequence, physical inverse iteration,
case hunt, population replay, or main.tex edit was performed.

NEW DOMAIN RESULT. For a>−1,a<1 and native n, write q=a*c+B*s,
T=B*c−a*s,R=sqrt(1−q²),b=q*c−R*s,V=q*s+R*c. With Delta=T−R>0 and
w=(R+V)/(1+c)>0, b=(w*a+Delta*q)/(w+Delta). Hence a strict common air-sine
bound |a|<Q and face bound |q|<=Q preserve the same outgoing bound strictly.
For Q=sqrt(1−delta), all outgoing vertical squares exceed delta. The
incoming convex weight is at least sqrt(delta)/n_max. This turns the
previous qualitative reverse-order flattening topology into a quantitative
margin-preserving theorem.

At fixed candidate glass, beams and rotors, the full common-margin wedge
domain is exactly triangular: A1 in I1, A2 in I2(A1), A3 in I3(A1,A2),
where A_i=sin(ax_i). Each interval intersects the constraints of both axes
and every retained time, and contains zero in its interior. At delta=.01
the native ranges give uniform conditional halfwidths 9/50,9/1000,9/20000,
respectively. These are guaranteed inner widths, NOT smaller replacement
wedge caps; the actual intervals reach the native/physical boundaries.
A sign-preserving piecewise-linear triangular map is a continuous cube
chart. Prefix-order clipping is a feasible retraction. Its nonexpansiveness
and the observation inverse's contraction remain unproved.

Root's new prism_margin_domain.py encloses the exact local face boundaries
s±=(±B*Q−a*sqrt(n²−Q²))/n², intersected with the face cap. It works for
h<=1/3, which contains the full native eighteen-degree angle range, with
the incoming common-margin hypothesis explicit. Four monotone corners
give outward possible and inward universally safe intervals over a whole
continuous incoming/index box. Formal arcsine folds outside the native
face branch are handled by a proved clipping argument. A separate exact
rational pullback handles negative and zero rotor factors. The possible
outer interval is not a universal transmission guarantee. Independent
code/mathematical audit passed. Control: prism_margin_domain_control.py.

NEW INDIVIDUAL-PRISM INVERSE. From supplied forward incoming/outgoing air
sines a,u and n, let B=sqrt(n²−a²),V=sqrt(1−u²),d=u−a,e=B−V,
L=sqrt(d²+e²),H=B*V+a*u−1. The positive face branch exists exactly when
e>0,H>0, and has s=d/L,c=e/L. Both incoming/outgoing ANGLE partials of
the recovered face angle are bounded by 1/(n−1)<=10/3. At fifteen degrees
the continuous-index partial is below1.357 radians per index unit. These
avoid the exit-normal reciprocal singularity, but intermediate directions
are not observed at the screen. Twelve symbolic identities and a complete
direction/index box control passed. Source multiplier M=B*R/(V*T) tends
to zero near critical exit refraction, so regular angular inversion does
not imply a uniformly well-conditioned full-hardware inverse.

SOURCE TOLERANCE CONSEQUENCE. At fifteen degrees, the entire initial beam
box |a0|<=17/40 (containing both native±25-degree ranges), all native glass,
and effective face cap13/50 (containing fifteen degrees) have first exit
normal square >=1/4. Root's whole-box endpoint control verifies that the
full face interval survives. Let D_i be angular direction transfer. Then
D_i<=sqrt(n²−1+R_i²)/(n*R_i). The literal native upper index gives D1<1.77
and D2,D3<8.35 if every later exit-normal square is at least .01. The source
gain telescopes as K=(V0/V3)/(D1*D2*D3), with V0>.9,V3<=1. Therefore
K>12000/1645451>1/138. With all other parameters fixed correctly, a source
readout has error <=(1645451/12000)*eta<138*eta under deterministic per-axis
RMS or sup record allowance eta; compatible source ambiguity is at most
twice this. This is an exact-coefficient, source-only result, not joint
wedge/hardware recovery, full-record compatibility, or an executed solver.
At eta=1e-5 the conditional source error is below .00138 model length units.

REMAINING GAP. Independent integration audit rejects automatic transfer of
the five-degree contraction: the chart's prefix dependence adds derivative
couplings, changing active constraints produces corners, and its clipping
has no proved nonexpansiveness. An actual data-driven operator must enforce
all observed positions, shared wedge/rotor identities and physical margins,
with a global convergence/residual proof. Geometry/source four-coordinate
affine profiling was already available at arbitrary transmitted angles;
the audit deliberately did not relabel that existing result as new.

New proof notes: QUANTITATIVE_ADMISSIBILITY_CHART_2026_09_23.md and
EXACT_PRISM_INVERSE_2026_09_23.md. Reports and reproducible symbolic/whole-box
controls are in constructive_turbo_2026_09_23. Frozen five-degree kernels
and certificates remain unchanged. The fifteen-degree scan inverse and
full18 blind reconstruction remain open.

Final root verification for this pass: all three new controls PASS; the
domain endpoint source hashes, chart source hashes and linked source-bound
report hash agree. The frozen five-degree cover retains SHA256
5e190a04e2623105af5517b61d5e0550415cb04ef40a644b99f13f91ebca8264.
Independent endpoint/branch and telescoping-source audits found no defect;
the integration audit explicitly keeps the data inverse/convergence gap open.

## 2026-09-27: parallel full18 attack, primary research, and exact shared-cell tools

The user requested more parallel agents and outside research toward the full
problem. Root and three subagents used all four available slots, with separate
mathematical tasks followed by independent audits. The fifteen-degree target,
continuous ordinary glass and all eighteen unknowns remain. The immediate
constructive target is still one physical full-record compatible explanation;
global uniqueness remains deferred. No optical record, physical-point forward
evaluation, inverse trial, population replay or main.tex edit was performed.
No recovery/archive count changed. The full18 goal remains OPEN.

WIDE-ANGLE FEASIBILITY GAP CLOSED, NOT THE INVERSE GAP. At fixed candidate
glass, beams and rotors, the triangular common-margin wedge retraction is
globally nonexpansive in an explicit weighted maximum norm. At margin .01,
one valid scale vector is (1,650,8658000). A convex local (incoming sine,
effective face sine) domain justifies two-point derivative integration even
when the complete prefix domain is nonconvex. The small-rotor inactive-boundary
argument removes a spurious division by the rotor factor. The same construction
extends to every positive common-margin class through the native18-degree
wedge cap. Its scales deteriorate as the margin vanishes, so there is no
uniform norm over the union of all strict records. An explicit continuous
native subbox disproves Euclidean nonexpansiveness, with expansion>4/3.
This does not transfer the five-degree data contraction to15 or18 degrees.
Proof/control: WIDE_ANGLE_OPERATOR_ATTACK_2026_09_27.md and
wide_angle_operator_control_2026_09_27.py. Independent mathematical audit:
WIDE_ANGLE_INDEPENDENT_AUDIT_2026_09_27.md.

NONFIT STATIONARITY IS A REAL FULL18 OBSTRUCTION TO GENERIC OPTIMIZER THEOREMS.
An exact symbolic family has all three target wedges positive through15deg,
target speeds(.6,1.8,3)Hz, zero phase/beam/source, full independent native glass
and full native geometry. A whole-state envelope gives exit-normal squares>.14.
The target has100-sample periodicity and50-sample sign reversal. A flat candidate
at separated speeds(.5,1.5,2.5) has zero gradient of the full200-sample summed
least-squares objective but positive residual. Its wedge columns cancel against
the target between the two100-sample blocks. This is an allowed wrong candidate,
not a restriction of the solver's target family to flat wedges.

The candidate is generically a strict saddle. Choosing the better native phase
among +/-9deg exposes a wedge/speed Hessian minor with negative determinant.
With the last target wedge15deg, the negative eigenvalue is below
-18225/17408 in native degree/Hz coordinates for J=(1/2)*sum residual^2.
Objective averaging scales this bound accordingly. At this stage a finite
feasible escape step needed a neighborhood/remainder bound; the follow-through
below closes that family-specific gap. No global strict-saddle or
global residual-decrease theorem was established. Proof and primary convex/KL
literature audit: WIDE_ANGLE_RESEARCH_2026_09_27.md. Independent derivation
audits stationarity, mixed-Hessian factors and the all-active extension; controls
do not themselves automate every full18 derivation. Both controls PASS.

FINITE TERMINAL-ROTOR ELIMINATION. Given the guarded terminal normal rectangles
from the earlier fixed-fifteen-upstream reduction, planar Helly gives complete
infeasibility witnesses using a disk and at most two halfplanes, or at most
three halfplanes. Explicit determinant, multiplier-sign, three-row Farkas and
disk-tangency events give a finite exact continuous-speed projection, including
isolated speeds, zero wedges and collisions. At200 samples their degree is at
most796 in the rotor half-tangent OVER THE SUPPLIED COEFFICIENT FIELD; this is
not the degree after eliminating all optical algebraic coefficients down to Q.
The loose event-root bound is34,729,322,790, not a practical runtime promise.
The executable event control handles rational unit-normal coefficients with
explicit resource caps; general native algebraic coefficients remain an
implementation obligation. Finite_record_event_control passes618 exact checks.

Root's exact rational disk checker uses O(m^2) arithmetic operations, permits
all degeneracies, and independently verifies feasible points or at most3-row
exclusions. All802 rows are retained in a200-sample synthetic normal control.
Independent enumeration audited59,049 coefficient systems and rejected corrupt
witnesses. New whole-cell certificates lift short fixed-speed witnesses to
continuous speed intervals AND every upstream parameter whose normals are
inside supplied outer rectangles. Factoring common sample-index rotation makes
the derivative cost depend only on index differences. A Cauchy bound gives
exact sufficient exclusion and a necessary linear cut on shared row RHS values.
The new control excludes complete wrong-speed/sign cells, preserves tangency,
and contracts a synthetic shared affine coefficient from[1,3] to a lower bound
>1.98 where independent-row bounds give only1. No optical normal bounds are
claimed from those synthetic inputs. See ROTOR_DISK_CERTIFICATES_2026_09_27.md,
FINITE_RECORD_ATTACK_2026_09_27.md and ROTOR_CELL_EXCLUSION_2026_09_27.md.

SHARED-VARIABLE CONTRACTORS IMPLEMENTED. The new polynomial_contractors.py
uses one coordinate wherever the same hardware variable occurs, a dependency
queue, exact rational quadratic term isolation, outward square preimage hulls,
and safe division only when zero is excluded. Strict/nonzero guards retain
their semantics; uncertain arithmetic or resource exhaustion stays unresolved.
Constructive disjunction applies the full network to each closed shared-coordinate
slice and retains the hull of all nonexcluded branches. Examples prove shared
coordinate contraction/exclusion that disappear if hardware is incorrectly
duplicated by sample. Controls:2,916 scalar quadratic calls retain118,098 exact
compatible points;540 network/CID calls test composition. Root's manual audit
finds the pruning and disjunction rules sound. The actual reduced graph was
adapted symbolically with all18 coordinates, without physical evaluation.
Float inputs mean their exact stored binary rationals, not intended decimals.
The reduced graph explicitly uses those native binary64 glass endpoints; the
generic algebraic_graph builder's exact13/10..9/5 prior must not be confused
with this reduced builder. The reduced graph still enlarges angular endpoints
to rational outer caps and closes strict guards; its nonempty relaxation is
not a physical solution.

An independent integration control composes the continuous-speed rotor cut
with exact affine RHS substitution and the shared polynomial contractor.
It contracts the same synthetic shared parameter, retains all18 coordinates,
and excludes a full incompatible coefficient/speed cell. Exact substitution
before interval propagation matters: separate auxiliary RHS intervals can
lose the correlation. Proof: SHARED_POLYNOMIAL_CONTRACTORS_2026_09_27.md;
integration evidence: universal_attack_2026_09_27/rotor_contractor_integration.json.

SPARSE GLOBAL PROOF PREREQUISITE. A new exact graph cover for the reduced
200-sample polynomial model retains the18 physical coordinates in one core
bag and covers every complete polynomial support. Its12,014 maximal bags
have maximum size39, for17,243 graph variables and27,424 constraints. An
independent union-find/subtree audit verifies running intersection, maximality,
support coverage, all saved hashes and4,096 small-graph covers. The largest
order-two moment block would have size820, with368,544,117 dense entries summed
over all blocks. This is a structural count, not an allocated/solved SDP.
Redundant local ball descriptors make an Archimedean hierarchy possible; the
hierarchy, verified SOS dual identities and finite-order exactness are not
implemented or proved by the cover. The graph is a closed outer model with
known endpoint/strict-guard relaxations. See SPARSE_GLOBAL_RESEARCH_2026_09_27.md.

PRIMARY RESEARCH HAS CONCRETE TRANSFER LIMITS. Deterministic matrix-pencil,
Vandermonde conditioning and decimated Prony papers support conditional rotor
gates only after exact optical remainder, amplitude and collision hypotheses
are proved. CID/ACID contractor programming and certified corner cuts support
the new shared-cell route. CS-TSSOS gives a convergent sparse hierarchy under
specific graph and compactness hypotheses, not exact first-order SDP recovery.
An exact four-corner argument shows coordinate-additive ISA enclosures of xy
retain worst pointwise width at least wx*wy/2 regardless of grid resolution;
this is a representation limitation, not an optical noise floor. The reverse
prism relation also has a strict saddle Hessian jointly in outgoing sine and
squared index, so a naive joint convexity claim fails. Verified primary sources
and local acquisition evidence are recorded in GLOBAL_INVERSION_RESEARCH,
FINITE_RECORD_RESEARCH, SPARSE_GLOBAL_RESEARCH and WIDE_ANGLE_RESEARCH notes
dated2026_09_27.

STATUS. Several missing proof/implementation components are now checked. The
conditional five-degree inner inverse is still the actual established broad
wedge inverse result. The15/18-degree results above concern feasibility and
explicit obstruction/escape structure. Sharp observation-derived contraction
or exclusion across the JOINT unknown hardware/rotor domain, an executed
compatible optical output, and full18 universal reconstruction remain open.
New controls/results live in constructive_turbo_2026_09_27 and
universal_attack_2026_09_27. Earlier frozen kernels, reports and counts remain.

FOLLOW-THROUGH: JOINT GRAPH/ROTOR INTERFACE. The new reduced_rotor_ports.py
removes the need to fix upstream hardware merely to apply the whole-cell
rotor test. It extracts each physical rotor's U/W coordinates directly from
a whole box of the reduced graph, after checking the actual polynomial
identities. These are TAN-wedge rotor coordinates, with radius squared bounded
by the shared a_i^2 variable; they must not be mixed with normalized sine
components. The graph h_i is exactly the certificate's rotor half-tangent on
the stated ideal grid. Any of the three physical indices may be selected,
without permutation or removal of other coordinates. At200 samples there
are803 rows per sign, including an explicit sign row needed for zero-width
phase sectors. Zero wedges and every necessary sign branch are retained.

Selected supports also give linear feedback constraints on signed graph
variables +/-U,+/-W. A guarded attachment function rebuilds the proof and
constraint, checks source-cell containment and checks the wedge sign whenever
sector rows were used. Rectangle-only cuts are sign-independent. Such cuts
must be discarded if their certified cell is broadened. The adapter validates
the reduced builder's actual native binary glass endpoints; it does not import
the generic graph builder's different decimal convention. Its own31 controls,
27 exact rational rotor assignments and the200-sample symbolic port check pass;
the independent audit passes125 checks. See REDUCED_ROTOR_PORTS_2026_09_27.md
and REDUCED_ROTOR_PORTS_INDEPENDENT_AUDIT_2026_09_27.md. This is now an interface
in both directions; contraction sharpness on unknown optical hardware remains
unmeasured, and no actual optical graph contractor was run.

FOLLOW-THROUGH: COMPOSED NORM TEST. The finite max-row comparison of a data
update followed by triangular clipping now has exact rational alternatives:
positive contraction weights, or one row-policy matrix and a nonnegative
vector proving this absolute envelope cannot contract. A positive-perturbation
Perron argument proves completeness, including reducible policies. Independent
audit passes. In a synthetic comparison, direct composition improves a failed
111/110 universal-weight bound to65/66. But the specified absolute conversion
of the saved five-degree column estimates fails every one of637 leaves, with
optimal factors1.14190445..1.72165815. This is a limitation of that conversion,
not the actual data map; all saved Euclidean certificates remain valid.
See COMPOSED_CLIPPING_CERTIFICATE and COMPOSED_CLIPPING_INDEPENDENT_AUDIT notes
dated2026_09_27. No fifteen-degree update matrix has been proved.

FOLLOW-THROUGH: FINITE OBSERVED-DATA DESCENT AT FIFTEEN DEGREES. The
FINITE_FEASIBLE_SADDLE_ESCAPE_2026_09_27.md theorem turns the previous strict
saddle into a finite feasible update, with deterministic per-axis RMS (or
sup) noise up to1e-5. Its target family retains all three positive wedges,
last wedge15deg, speeds(.6,1.8,3)Hz, phases/beam/source zero, and all native
continuous glass/geometry. Its flat candidate has arbitrary native glass
and geometry; their equality to the target is not assumed. This is a stated
family, not all possible beams, rotors, observations or candidate states.

The last candidate phase can be changed to either +/-9deg at no objective
cost. An observed-data mixed-Hessian test b^2>a+1 is guaranteed to pass for
at least one phase, including the declared noise. At the flat candidate
the last-wedge/last-speed Hessian is[[a,b],[b,0]], even for noisy data. A
30-bit dyadic approximation to an explicit rational-form direction retains
curvature<-7/16. All candidates within a radius1/2 in these degree/Hz
coordinates are physically transmitted and remain within the native prior.

Exact scalar calculus and all-state bounds give |Yobs|<401 and full-objective
fourth directional derivative<10^14 over that neighborhood. The average of
the two objectives at plus/minus a step cancels both the gradient and cubic
terms. At rational step1e-7, at least one sign decreases the ACTUAL summed
observed-data objective by at least17*step^2/96. Computing objective-difference
intervals of widthD/4, whereD=step^2/12=1/(1.2*10^15), suffices to select and
certify an output with decrease at leastD. This explicitly retains finite
comparison error; subtracting two large objective values is avoided.

The guaranteed decrease is tiny and is not a practical convergence-rate
claim. The control executes exact derivative/denominator/Taylor/noise algebra
only; the optical step and its interval comparison have not been executed.
Independent candidate-derivative and full-proof audits support the constants.
Repeated descent leaves the original flat-candidate hypotheses, so no global
convergence or compatible reconstruction follows. This closes the finite
escape theorem for the exhibited wrong stationary family, while the joint
unknown-hardware inversion goal remains open.

JOINT SHARED-CELL FOLLOW-THROUGH (2026-09-27). Root and three agents used
all four slots to connect the previously separate necessary-condition tools.
The executed work remains symbolic/exact algebra and nonoptical contractor
controls. No optical record, physical state, forward call or recovery trial
was generated or evaluated; stopped campaigns remain stopped. No main.tex
or frozen five-degree proof file was changed. The requested full18 inverse
with at least15-degree wedges remains open, and the conditional five-degree
inner theorem remains the established broad inverse result.

The new verified_lp_contractor_2026_09_27.py adapter substitutes singleton
coordinates exactly through all shared polynomial constraints, constructs
the existing quadratic outer LP on remaining variables, and maps certified
bounds back into the original full coordinate map. Every proposed dual is
rechecked against the actual rebuilt LP, objective and box. Self-replay of
a foreign witness is insufficient. Poor duals retain exact box-residual
correction; malformed/foreign proposals and numerical failures retain the
unresolved box. All-singleton consistency is not called physical existence.
Its46-control fixture includes a correlation that interval propagation
cannot remove: c*u=3*h,u+h=1,c=3 withh,u in[0,1] becomes exactlyh=u=1/2.
Singletons are a declared synthetic-control input, not a hidden restriction
of the requested physical priors. Source-extra and branch-local constraints
remain scoped to the relevant call. See VERIFIED_LP_CONTRACTOR_2026_09_27.md.

The root's rotor_bernstein_2026_09_27.py retains cancellation in the squared
norm of a nonnegative sum of rotating normals. Combining equal indices
first, the norm is P(h)/(1+h^2)^D, with degree at most2D and D the relative
index span. Equal-degree Bernstein coefficients give a quotient range bound
when denominator coefficients are positive; other leaves use the old
validated bound. Exact closed subdivision, degree/bit caps and whole-cell
fallback preserve coverage. The new checker reconstructs bounds from actual
rows, weights, radius and speed interval, rather than trusting reported
certificate values. An independent auditor caught a malformed-witness
AttributeError path, which was fixed to reject that witness.

For adjacent-index coefficients(1,-2,1), the exact norm is16h^4/(1+h^2)^2.
At h in[-.01,.01], the new squared bound16/100020001 replaces2/625, a
20,004.0002 ratio. This factor is specific to that synthetic cancellation,
not optical performance. The2,418 root controls include1,100 exact-point
identity/enclosure comparisons across100 random rational cells, with74
strict improvements, plus endpoint, sign, resource and corrupt-cut checks.
The whole-interval guarantee rests on the polynomial/convex-combination
proof, not those finite sample comparisons. A separate two-normal support
at indices0 and199 gave a degree398 calculation taking about11.9seconds
with cap400. That timing exposes cost; it is not a200-row/full-graph run.

The three-way integration is now substantive rather than a disconnected
API test. In an explicit21-variable synthetic network with18 shared
coordinates, b0=p+t-2,b1=p-2,b2=p-t-2. The new rotor inequality is supplied
in b0,b1,b2 without manually eliminatingt. Interval propagation even with
the cut leavesp in[1,3], and uncut LP also leaves it unchanged. Old rotor
cut plus verified LP givesp>=about1.985857864376269; the new bound plus
the same verified LP givesp>=about1.999900009999. It excludes the entire
p<=1.999 continuous-speed cell by a verified negative upper bound on the
zero objective; the old bound cannot. All14 integration checks pass,
including local-cut lifetime and zero-budget preservation. This still
does not demonstrate unknown optical-hardware contraction.

The joint_cell_2026_09_27.py driver now runs the supplied shared graph,
guarded rotor ports, closed wedge-sign branches, selected-support feedback,
optional verified LP and optional shared-coordinate CID under explicit
budgets. All supplied constraints remain obligations and all18 physical
coordinate slots remain present. Each child receives only cuts proved on
that child; cuts are discarded before a survivor hull. Unprocessed or
resource-limited branches remain. Graph changes cause stale contractions
to be discarded. A separate rotor-only ReducedTransferGraph fixture
(without optical transfer/output equations or observation values) narrows
a synthetic shared coefficient more than35 times beyond the equally
budgeted sign-split LP. The default older Lipschitz cuts do not improve
that fixture, which is recorded rather than hidden.

Root and the independent auditor found a scheduling weakness: capped
passes restarted at the same first constraint and could stop on an
unchanged box before later constraints were visited. The driver now rotates
original-constraint priority, keeps local cuts first, and does not mistake
a step-limited unchanged pass for convergence while visit budget remains.
Late-constraint regression and independent branch/source/budget checks
accompany the change. This fixes that starvation mode; it is not a proof
of global solver convergence. Round limits apply per CID child; other
specified proof-call budgets are global.

The new proof notes are ROTOR_BERNSTEIN_BOUND, VERIFIED_LP_CONTRACTOR,
JOINT_CELL_PIPELINE, JOINT_CELL_INDEPENDENT_AUDIT and JOINT_CELL_LP_BUDGET_REVIEW,
all dated2026_09_27. Research connection: Garloff/Schabert/Smith,
Bounds on the Range of Multivariate Rational Functions,2012,
doi10.1002/pamm.201210313. Its Crossref publisher-deposited abstract and
metadata were checked, not its full text; the required Bernstein proof
is provided directly in our note. No novelty claim is made for that
standard enclosure method. Earlier primary-source research remains in
GLOBAL_INVERSION_RESEARCH and FINITE_RECORD_RESEARCH.

The old parallel_attack_manifest.json is preserved as an immutable
snapshot. joint_shared_cell_manifest_v2.json records the new sources,
controls and audits separately and checks earlier proof code/reports plus
the frozen five-degree cover. The integrity check also found that
FINITE_FEASIBLE_SADDLE_ESCAPE_2026_09_27.md had changed after the old snapshot
(note timestamp01:08:55, snapshot01:04:04). Its earlier/current hashes are
recorded explicitly; the old note binding is not represented as current.
The unchanged independent scalar audit was rerun into a NEW report with
all10 checks passing and the current note hash. The historical report and
manifest were preserved. No exact textual delta or author is inferred from
the hash mismatch. The living research logs intentionally advance.
The next unresolved obligation is effective
joint contraction and constructive recovery for the actual optical
equations over the requested unknown hardware domain; synthetic coefficient
contraction, finite branch coverage and local proofs alone do not close it.

## September27,2026 — connect the new cuts to the actual optical equations

Root and the three specialist agents continued with whole-domain symbolic
proofs, source-checked graph compilation, and abstract algebraic controls.
No optical observations or physical states were instantiated, no forward
call or optical-graph contractor visit occurred, and no stopped campaign
was restarted. The paper main.tex and all prior frozen proofs/reports were
left unchanged. The independent audits below check new sources, not recovery
performance on observations.

The new prior-wide margin argument retains the exact correlation
B*t+u*x>=1 instead of losing it in componentwise absolute bounds. With
n<1801/1000, the critical forward support yields3*t+x_in^2>=1. Correlated
first-prism beam/face bounds give t1>51/100, then t2>9/100 and
t3>27/10000 throughout the rational angular outer graph's transmitted
branch, including later critical exit roots. The new final slope cap is
10000/27 (about370.37 versus20901.63), and the output cap is32216945/432
(about74576.26 versus84361307.12). These improve enclosures by factors
56.4 and1131.2; they are not inverse-error or success-rate improvements.
The separately proved native18-degree wedge/25-degree beam constants give
final slope46875/256 and output5920027/160. They are explicitly not applied
to the larger rational angular outer graph.

optical_margin_sharpening_2026_09_27.py checks every actual original source
equation, guard, prior and bound, then supplies tighter auxiliary bounds
and four quadratic forward relations per axis/sample/prism. No physical
parameter interval is fixed or narrowed by those whole-prior bounds. Its
73 controls and86 independent checks pass; the independent audit uses a
different rational Taylor proof of the native angular sine bounds. Later
exit-root derivatives may still be singular; positive air-forward values
are not the same as positive tilted-exit-root margins.

Root's optical_cone_envelopes_2026_09_27.py supplies exact rational Lorentz
support cuts for the actual glass/air/face/beam root equations. It also
uses concavity of sqrt(n^2-x^2) to construct lower glass-root planes from
downward-rounded corner values, including all singleton rectangle cases.
Their local validity is reconstructed from actual source identities and
their complete n/x rectangle; reported planes are not trusted. The1892
controls include1755 scalar-plane comparisons, but the whole-domain proof
rests on concavity, not samples. A32-check independent review passes. A
deliberately nonphysical abstract root-block LP pseudopoint is separated
by a new plane; no attained optical-state claim is attached to it. This
implements the earlier reverse-root hull idea, not a novelty claim for
Lorentz support or convex-hull mathematics.

The shared material theorem bounds each paired-axis deflection energy:
(n-1)^2*a^2/(1+a^2)<=e_k<=(n^2-1)*a^2/(1+a^2). It holds with the same
unknown glass and wedge across samples. The first source-bound kernel
turns whole-cell energy intervals into necessary material/wedge cuts;
35 controls and185 independent checks pass. The follow-through retains
each selected e_k as an actual quadratic auxiliary and adds both envelopes
plus(n+1)e_l-(n-1)e_k>=0. This preserves variable correlations that interval
endpoints can lose; energies at different samples are not asserted equal.
The quadratic energy graph has59 controls and103 independent checks. The
source checks include paired axes, roots, rotation and shared material
identities, retain zero wedges and critical closure, and reject changed
source equations or a broader attachment cell.

The output/geometry bridge begins from the exact same-axis source-cancelled
identity R=d_W*D+gap*G, using products of output coefficients. It then forms
cross-axis determinants and division-free relations N_d=d_W*Delta and
N_g=gap*Delta, plus native geometry cones. Delta=0 is retained; no division
or rank assumption is introduced. Beam dependence remains inside the
coefficients even though the explicit source/beam offset cancels. The
bridge has41 controls and29 independent checks. An exact Farkas argument
proves that its lift strictly strengthens a specified free output-block
McCormick relaxation. That fixture is not an observed or attained optical
trace.

Root's optical_relaxation_2026_09_27.py composes these four layers in the
required order: margin theorem, cone envelopes, shared energies, output
geometry. Full-source reconstruction verifies the final graph, every
original constraint remains, and local rectangles remain permanent graph
bounds. A child can inherit the extension only inside the complete source
cell; a broader hull must rebuild it. Failed construction caps return
unresolved with no replacement graph. All18 native priors, signs, zero
wedges, speed collisions and zero determinants survive. Its40 controls
include full200 symbolic compilation:17243 original variables plus27 new
auxiliaries,22661 equalities and9888 inequalities. All4800 margin cuts are
included, with32 of3602 cone ports selected plus sparse energy/geometry
couplings. This is actual-source symbolic integration, not a full optical
LP or contractor benchmark. Evaluation entry points were patched to fail
during the control. The independent38-check composition audit reconstructs
all four layers, verifies local-cell scope and all source obligations, and
rejects source mutation, forged self-consistent graph hashes and namespace
collisions. No correctness defect was found.

A separate independent all-prism gain proof strengthens0<=A<11/8 without
an incoming-angle or wedge upper bound, using n>=13/10 and the retained
branches. The exact sign is A-1=-m*C*(u+m*t)/[t*(B-m*x)], m=V/Q.
After sign reflection, the only amplifying case is bounded by
U*(1-U)/(t*B). Three exact Bernstein intervals prove
U^2*(1-U)/(1+U)<=3/32, giving(A-1)^2<=25/184<9/64. Thus the product
gain is below1331/512. The intermediate sign ambiguity was resolved by
direct source-polynomial expansion before the note/control was finalized.
This additional gain theorem is not substituted into the now-frozen margin
module or compiled controls.

The progress recommendation derives an explicit Holder1/4 output modulus
with constant1.2e11 in the transformed eighteen physical coordinates on
the200-sample exact-grid outer prior. First-stage transmission has a
uniform positive exit root; each later critical square root can cost one
half in the exponent. Positive tD value bounds permit quotient enclosures.
The note separately defines a clipped-radical/clamped-denominator continuous
extension agreeing on physical transmitted points, so an infeasible cell
center need not be fed to an undefined physical forward branch. A center
residual exceeding eta+C*r^(1/4), with verified evaluation allowance, would
exclude its entire physical box. This suggests an enclosure fallback that
subdivides18 physical coordinates rather than17243 graph variables. No
extension evaluator was implemented or run. The constant is deliberately
impractical alone; data-dependent separation, physical validation and
useful global runtime are not supplied. The gain/modulus algebra control
passes26 exact checks and the independent modulus review passes29. The
independent output coefficient is30499200118207/270, below1.2e11; the
extension must propagate raw rotated unit directions and preserve raw E,
while clamping only the stated transfer quantities/denominators. The center
and source box must remain in the declared rational outer prior. Primary
contractor/CID and sparse-SOS research in
the earlier research notes supplies sound composition context, not an
optical recovery theorem.

The broad certified inverse remains conditional at about5degrees with
outer hardware supplied. Blind full18 recovery and a general7/15-degree
inverse are open. The current work improves the actual optical bounds and
cross-coordinate constraints needed for that task; it does not close it.
The new optical_symbolic_manifest_v3.json preserves both earlier manifests,
their unchanged proof code/reports, the separately documented historical
note revision and the frozen five-degree cover. Living logs intentionally
advance. The next substantive obligation remains certified progress from
observed outputs to compatible unknown physical hardware.

## 2026-09-27 — Output-required margins and an exact affine-fiber backend

Root and three agents continued the mathematical/source-only work using
all four available slots. No optical record or physical state was evaluated,
no forward call or actual optical contractor was run, and no stopped
campaign was restarted. The broad inverse angle claim stays conditional at
about5degrees per prism with the outer hardware and rotors supplied.
General7-degree blind recovery and the later15-degree target remain open.

An essential prior-art correction was made before freezing this phase.
ATTACK_OPTICAL_SEPARATORS_2026_09_17 already proves |Y|<=50 implies
chi2>3/3250 and chi3>1/25, with an amplitude-dependent terminal threshold150.
ATTACK_GLOBAL_EXCLUSION and ATTACK_OBSERVATION_CONTRACTION already retain
the full4 affine fiber, zero-gain cases,14D subdivision and data-dependent
linear regularity versus the global quarter-power rate. These qualitative
results are not new. The new notes explicitly credit them. Also the prior
turn's56.4/1131.2 bound-improvement factors compare against stale builder
bounds, not against the best September17 analytic t1>.5,t2>1/12,t3>1/440.

The first new output/root module proved |Y|+C45*E2>=45, with
C45=4063517/4032, and (150+4M)*E3+3|Y|E3+|Y|>=150, M=192535/512.
It passed50 controls and an independent88-check audit. The first inequality
is valid but would narrow the older bounded-scan class if used alone.
The separate refinement therefore preserves that module and its cuts while
adding |Y|+C*E2>=A, A=165063/3125=52.82016,
C=1024942546639/995400000. It retains the positive middle propagation term
when the reflected final-face slope is negative; then A3>=1. The other
face-sign case gives a still stronger bound. Exact endpoint checks cover
E2<=1/19, and C/19>A makes the remaining branch automatic.

Thus a justified absolute model-output cap R<A implies
chi2>=(200/211)*(A-R)/C; the retained terminal cut similarly bounds chi3.
At R50 the new chi2 floor is561437452800/216262877340829, about.002596,
strictly above3/3250. The new terminal floor also exceeds the older explicit
6144000/144367141. These are screen-position units, not wedge degrees, and
the original physical origin matters. Record noise must be added to the
cap, converting an RMS allowance to coordinatewise allowances if needed.
No actual record was supplied to these inequalities.

The refined source adapter reconstructs all original optical obligations,
preserves all18 priors and both old cuts, and adds the stronger inequality
using the same absolute-output auxiliaries. Full200 compilation adds400
variables,400 equalities and1600 inequalities, with no physical evaluation.
The root control passes43 checks. The global agent's independent scalar/sign
audit passes17; the separate source audit passes108. Complete source-cell binding
prevents attaching local auxiliary bounds to a broader hull. Zero absolute
outputs are retained by the logical graph; no inherited floating witness
evaluator is claimed for that boundary.

The exact affine-fiber implementation is a separate substantive software
advance. The native distance, gap and two source coordinates remain free
in their full box. Closed affine-sign branches turn interval-row minimax
bounds into five-variable rational LPs. A feasible initial slack basis,
Bland pivot rule, exact primal/dual replay and explicit resource limits
avoid a full-rank or nonzero-gain requirement. Independent dual witnesses
bind the actual rebuilt LP; caller-owned row keys bind observations to the
intended output rows. The physical coefficient extractor binds all400 rows
to the original symbolic graph. Its140-check control,48 independent exact
LP vertex comparisons,93-check external audit and96 reconstructed branch
LP witnesses pass. These controls use abstract coefficient problems and
symbolic optical sources, not instantiated optical records.

The updated profile notes connect this backend to explicit clipped-center
extensions and new quantitative constants. Common-z comparisons justify
the value-function modulus without uniqueness or continuous minimizers.
The uniform14D constant is112960000033 times radius^(1/4); the optional
bounded-output branch has a linear modulus below the refined threshold.
Controls22/51 and an independent20-check review pass. This specializes the
existing theory; it does not first establish14D profiling or linear
regularity. A validated extension-center coefficient evaluator remains
unimplemented. A physical-graph enclosure is not automatically an enclosure
of that extension at an unphysical center. LP allowances must not be counted
twice, and an affine/extension upper fit is not physical membership.

The new sharpness family is also credited accurately. September17 already
proved global forward quarter-power sharpness and strict-pair failure of
larger exponents, with first wedge tangent.3 (about16.7degrees). The new
exact radical family puts every wedge strictly below15degrees and proves
noncancellation for every native geometry/source choice. Its46 checks pass.
This is an improved obstruction range, not a recovery theorem.

No full18 compatible optical reconstruction, new inverse angle guarantee,
useful global runtime, or T5 completion follows. The frozen five-degree
cover and all earlier snapshots/proof code/reports remain preserved;
living status documents advance with this explicit scope.

The focused external review read primary full texts by Ben-Tal et al.
(adjustable robust optimization), Hladik (AE interval systems), van Leeuwen
and Aravkin (nonsmooth variable projection), and Shary and Moradi (interval
least squares). The exact theorem hypotheses matter: fixed recourse,
constraint-wise independence, uniform strong convexity or full rank are
unavailable for the general optical fiber where the cited theorem requires
them. The resulting recommendation is a small finite policy of affine
witnesses over coefficient boxes, improving an upper certificate without
requiring one hardware witness for the whole coefficient family. It does
not strengthen the exact optimistic lower bound on the same independent
coefficient box. The missing optical exclusions still require information
about correlated coefficients from the same upstream hardware. Primary
retrievals and hashes are retained in research_affine_fiber; the detailed
applicability analysis is AFFINE_FIBER_EXTERNAL_RESEARCH_2026_09_27.md.

The follow-up COEFFICIENT_PARTITION_PROFILE_ADDENDUM proves the distinction
quantitatively. For coefficient-box diameter D in the native bounded-hardware
row metric, the exact robust-minus-optimistic profile gap is at most D;
robust-minus-worst-realized-profile is at most D/2 by a midpoint minimizer.
A finite complete coefficient cover with a full4 hardware witness per leaf
therefore approximates the worst profile without rank or continuous-optimizer
assumptions. The abstract existing fixture has exact lower5/9, worst profile1
and common-witness upper5/3; sixteen half-width coefficient boxes attain
upper1. Minima over the coefficient cover leave the exact lower unchanged.
Resource-limited inner LPs require their own certified gap; D alone says
nothing about an unfinished optimization. A policy is not the old API's
single witness and does not identify a physical optical system. No wrapper
is implemented, and increasing arithmetic precision already closes the
unpartitioned common-witness gap at one fixed center. This makes the policy
an optional value-bound tool; validated center enclosures and correlated
optical lower bounds remain the more direct next obligations.

The final independent policy review caught and corrected one scope sentence:
the worst-profile lower bound max_leaf L applies to the whole coefficient
box, not automatically to a smaller realizable optical image. A scalar
beta[0,1], singleton hardware, two half-interval leaves and realizable image
{0} give worst-profile lower1/2 for the box but actual worst profile0 on
that image. The universal pointwise enclosure min_leaf L<=V<=max_leaf U
does transfer. The addendum now states this explicitly; no implemented
solver or earlier proof module had used the incorrect transfer.

The final independent policy control passes70 checks: sixteen children are
verified by exact algebra without new child LP solves, and one deliberately
zero-pivot abstract LP demonstrates that D=0 can still yield reported[0,5]
around exact profile1. All twelve phase reports are consolidated with their
source hashes in data_physics_bridge_manifest_v4.json. The three earlier
manifests, their immutable files/reports, known documented note revision and
frozen five-degree cover are preserved. The full18 goal remains active and
unachieved; this phase supplies stronger checked components and an explicit
research priority, not a general7/15-degree recovery guarantee.

## 2026-09-27 — Source-bound extension coefficient oracle and exact LP binding

Root and the same three specialists continued using all four slots. The
missing clipped-extension program/enclosure interface is now implemented,
with independent arithmetic, optical-source and composition reviews. This
supersedes the preceding entry's unimplemented-evaluator status. Executed
work remains generic arithmetic, abstract coefficient LP controls and
symbolic optical compilation: no optical record, numerical optical state,
extension value, forward call or optical contractor visit occurred. The
callable numerical optical enclosure and adapter APIs were not executed.
No stopped campaign was restarted and paper/main.tex was not edited.

CENTER_EXPRESSION_ARITHMETIC implements an immutable expression DAG and
arbitrary-precision outward dyadic operations for rational arithmetic,
square roots, min/max and clips. Its replay binds actual inputs, expression
nodes, output mappings, precision, resource caps and trace. Unresolved
domain/resource results discard partial output certificates. The author
control passes 103 checks, including 25 independently interpreted rational
probes; independent audit passes 102. A metadata canonicalization bug was
fixed before freezing: nonstring mapping keys are rejected rather than
colliding with string keys after conversion.

Zero-root convergence requires structurally nonnegative interval operands,
not merely an exactly nonnegative algebraic value. The expression
sqrt(sqrt(2)-sqrt(2)) remains unresolved because dependency loss gives a
negative lower radicand at every precision. Explicit max(0,radicand) guards
in the optical program preserve its exact function and make critical-root
evaluation available. Generic double-root controls show the quarter-power
rounding rate. These are nonoptical examples, not critical optical trials.

CENTER_EXTENSION_COMPILER reconstructs the actual reduced source, native
prior descriptors, rational sample grid, input cell and row mapping before
compiling the existing universal or data-required clipped extension. The
full 200-sample program has 94,750 nodes, 400 coefficient rows, 8,000 named
outputs and 1,200 diagnostic stages. All fourteen nonlinear coordinates
and four native affine coordinates remain. Physical prism order, collisions
and zero wedges are retained. Source/specification control passes 62,
root compiler control 34 and independent compiler audit 104.

The compiler includes proved redundant clips and positive denominator
floors, propagates raw rotated directions and forms E from raw t. It
records source affine restrictions separately while profiling over the
complete native distance/gap/source box. This is a valid outer relaxation
for exclusion; its upper witness need not satisfy extra source constraints.
Fixed native prior checks prevent future model-range changes from silently
inheriting the present constants. The emitted diagnostic chi is clipped:
it cannot certify the original physical root margin. Native/source and
unclipped direction-agreement obligations remain separate.

CENTER_PROFILE_ACCURACY_BRIDGE connects generic arithmetic outputs to the
existing exact interval LP. The stopping quantity is the actual replayed
upper-minus-lower gap, including unfinished optimization, not coefficient
diameter alone. For finite point accuracy, first refine and freeze a
sufficiently narrow coefficient problem, then increase exact LP resources
until that fixed problem finishes. This termination argument does not
implement an outer refinement loop or bound useful runtime.

Whole-q-cell coefficient enclosures feed the interval LP directly with
spatial_error=0, avoiding a second charge of the giant global modulus.
Increasing precision alone does not shrink genuine cell variation. An
alternative rowwise transport expands intercept intervals by separately
proved row errors. Abstract controls demonstrate stricter bounds than a
single scalar allowance; the independent unequal-row/noise example gives
[9/8,15/8] versus scalar [1,2], with one shared full native hardware witness.
No optical row errors or cell widths were evaluated.

CENTER_AFFINE_BINDING supplies the final actual-source/observation interface.
It replays the enclosure before constructing the LP from its coefficients,
native affine box and caller-owned rows, targets and deterministic allowances.
In data mode the exact premise is max_i(|y_i|+eta_i+threshold)<=R<Astar.
The combined gate uses that SAME threshold for strict LP exclusion, with
no duplicate coefficient or spatial error. Storage/noise provenance and
RMS-to-coordinate conversion remain caller obligations. The bridge and
binding controls pass 50 and 16; independent joint review passes 78. The
complete optical positive path was source-reviewed only. An extension
upper fit still proves no physical existence, original branch validity,
native angular membership or restricted-source membership.

This implements previously missing checked components; it does not first
prove affine profiling, fourteen-coordinate reduction, continuous clipped
extensions or coefficient-diameter estimates. The September 17 and earlier
September 27 results are explicitly credited. All eight new reports are
bound in center_oracle_manifest_v5.json, preserving v1-v4, their earlier
immutable files/reports, the documented historical note revision and the
unchanged five-degree cover. Only the living status logs advance.

The follow-up literature synthesis favors correlated cofactor separators.
September 17 REGULAR_COUPLED_EXCLUSION already proves identity-augmented
elementary-circuit completeness, using Rockafellar's elementary-vector
theorem. The concrete missing implementation is support extraction and
source-bound compilation of selected five-row cofactor residuals as whole
expressions, before interval evaluation loses shared-coefficient dependence.
Native-prior rows and zero-rank cases must remain, with no determinant
division. Selected supports need a fair fallback; completeness at a point
does not imply useful broad-cell bounds. Convex interval Taylor cuts from
Araya, Trombettoni and Neveu are a possible later strengthening on proved
smooth domains. Critical roots and clipping seams prevent importing their
smoothness assumptions globally. Contractor composition and constructive
disjunction motivate the fair fourteen-coordinate driver. This synthesis
uses the already retained primary-source research; it claims no new optical
effectiveness result. More coefficient-box upper policies would not tighten
the already exact optimistic lower bound on the same coefficient box.

The broad certified inner inverse remains |sin(ax_i)|<=7/80, containing
at least ±5 degrees per prism. It reconstructs three wedges and two source
positions with seven other hardware settings and all six rotor settings
supplied. It is not a five-degree initialization tolerance or blind full18
result. General seven-degree recovery and the requested transmitted
fifteen-degree inverse remain open, as do T5 and useful global T7. Restricted
small-angle full18 constructive results remain intact. The goal remains
active and unachieved; no recovered optical system or wider-angle inverse
guarantee is claimed by this implementation phase.

## 2026-09-27 — Correlated cofactor compiler, support extraction and cell composition

The previous turn made progress by implementing the missing source-bound
coefficient interface. Root and all three specialists now implemented the
research-recommended correlated cofactor route, instead of adding another
independent-coefficient upper policy. The complete identity-augmented
circuit theorem, rank-deficient treatment and original abstract barrier
example were already proved in REGULAR_COUPLED_EXCLUSION_2026_09_17, with
Rockafellar's elementary-vector theorem explicitly credited. New work is
executable source-bound compilation, exact support extraction and their
bounded composition. No new global circuit theorem is claimed.

COFACTOR_EXPRESSION collects exact sparse polynomial terms in shared DAG
atoms before interval enclosure. The whole C=sum(v_i*b_i) is collected,
not merely its separately bounded products. Cofactors use the correct
ordered five-row determinants; both orientations C-E and -C-E are retained.
Optical radii receive allowance+threshold, while the four appended native
prior radii remain fixed. No determinant, source gain or cofactor norm is
divided out. Zero-rank selections return the vacuous zero test.

Square roots, reciprocals and min/max nodes are opaque exact atoms. Source
expansion caps can retain an original subtree as an atom, never truncate
terms. Final collection or work exhaustion yields unresolved. The complete
original DAG remains as a prefix with all outputs retained, so cancellation
cannot erase a domain failure. Regressions retain unresolved sqrt(-1) and
1/0 nodes even if their formal contribution cancels. Formal data mode adds
uninstantiated target/allowance inputs, avoiding numerical optical records
even during source compilation. The author control passes64 and independent
compiler audit38; additional generic/source review is recorded below.

The earlier five-row q-in[-1,1] family now runs through the validated
compiler: C=1 exactly and the positive surplus lower is1/4 at allowances1/8.
The independent-coefficient affine LP has exact lower0 and an explicit
zero-residual relaxed coefficient witness, which uses intercept/slope values
that cannot come from one common q. The cofactor removes that false
explanation by preserving their algebraic relation. This is an executable
comparison on an existing abstract family, not a new optical example.

The independent compiler audit also supplies a distinct shared-root control:
u=sqrt(1+x^2), x in[-1,1], four identity bands centered at mu_i+u, and a fifth
row with coefficients(u,u,u,u) and center u*sum(mu)+4u^2+1. It uses the full
native affine box. The whole cofactor has C=1, while independent coefficient
intervals admit a zero-residual explanation and their LP lower is0. Root
dependence is preserved without an unsafe radical identity. This is again
generic algebra, not asserted optical realizability.

COFACTOR_SUPPORT extracts a small support without enumerating binomial row
subsets. A supplied exact separator is augmented with -A^T*lambda on the
four native identity rows. Exact sign-compatible nullspace steps preserve
kernel membership, total absolute mass and positive separation while
removing support entries. At most five active rows remain. Rank-increasing
padding supplies exactly five rows, and the final cofactors are recomputed
and checked from actual bands. One exact fixed-coefficient minimax LP can
propose the initial weights; primal and dual replay, not solver status,
authorize their use. Author/independent controls pass62/61, with32 additional
independently checked normalized-kernel directions. An independent exact
full-box minimax fixture attains13/220; zero rows, singleton coordinates,
unequal noise, resource caps and forged optimizer metadata are covered.

Padding choice matters regionally even when it is irrelevant at one point.
The optional optical-first padding tries unused optical rows that increase
point rank, with native identity rows as the guaranteed fallback. At q=0 in
the earlier family the circuit has only four active rows. Native distance
padding is sound but weak on[-1,1]; adding the remaining optical row recovers
the correlated C=1 identity and1/4 surplus. This is a useful proposal
heuristic, not a universal dominance claim.

COFACTOR_CELL_PIPELINE composes one bounded generic attempt. It encloses
the source DAG at a rational proposal point, uses rational midpoint
coefficients only to suggest support, then recompiles the ORIGINAL shared
DAG and encloses its cofactor over the ENTIRE original input box. Only the
strict whole-cell inequality can exclude. Author/independent controls
pass27/27. A sharp independent regression uses exactly equal expressions
sqrt(2) and sqrt(18)/3: at32 bits their surrogate coefficients give a false
positive minimax lower1/4294967296 for an exactly feasible singleton fiber.
The actual whole-cell cofactor correctly retains it. A pointwise separator
whose region contains a fit and an injected fabricated support proposal
also cannot authorize exclusion. No point-surrogate gap is promoted to a
regional guarantee.

OPTICAL_COFACTOR_COMPILER binds this machinery to the actual clipped optical
source, preserving all14 nonlinear coordinates and the full4 native affine
box. Full200 symbolic construction retains all400 formal target/allowance
pairs and expands94750 source nodes to95963 for the selected distant-row
support. The resulting DAG has814 inputs:14 physical plus800 uninstantiated
data symbols. Sparse row order and prior-row offsets, native source
restrictions, physical prism order, data mode and threshold all remain bound.
Source fingerprints are checked again after compilation and enclosure to
reject an in-flight source change.

Actual symbolic programs demonstrate a nonvacuous optical cancellation:
for five optical rows, cofactor annihilation removes the direct6*b_axis*K
terms from C. With source-prior rows present, their surviving terms remain;
the compiler does not apply the optical-only simplification indiscriminately.
Transfer coefficients still depend on the beam, so beam coordinates are
not eliminated. The author source control passes41 and independent
generic/source audit48. The numerical optical APIs were reviewed at source
level only. No actual optical record, state, expression value, forward call,
contractor visit or recovery trial was instantiated or executed.

The data-dependent gate still needs max_i(|y_i|+eta_i+threshold)<=R<Astar
with the SAME threshold used in the cofactor bands. Whole-cell scalar
enclosure needs no extra coefficient or global spatial error charge. A
surviving selected test supplies no feasible point, physical branch,
native-angle membership or restricted-source membership. Clipped roots
remain unusable as certificates of original physical root margins.

The eight author/independent reports are consolidated with current source
hashes in correlated_cofactor_manifest_v6.json. All five earlier snapshots,
their immutable files/reports, the documented historical note revision and
the frozen five-degree cover remain unchanged; only living logs advance.
No paper/main.tex edit or stopped-campaign restart occurred.

The remaining gap is substantive: selected supports and one bounded attempt
do not provide fair14D coverage, useful optical interval widths, a practical
all-case progress bound or final original-physics acceptance. The broad
conditional five-degree result remains unchanged, with outer hardware and
rotors supplied. General seven-degree and transmitted fifteen-degree full18
recovery, T5 and useful global T7 remain open. The goal stays active and
unachieved; the new implementation now retains correlations that its earlier
independent-coefficient backend necessarily discarded.

## 2026-09-27 — Resumable profiled coverage and original-physics acceptance

The v6 follow-up now supplies an exact generic controller, its actual-source
adapter and a separate original-physics acceptance compiler. September17
already proved conditional fair14D positive-gap coverage. Earlier exact
specialized covers already verified closed partitions and resumptions.
The new result is executable scheduling/source binding, not a new splitting
method, dimensional reduction or unconditional completion theorem.

PROFILED_COVER retains a complete closed binary partition. Only a freshly
replayed strict whole-cell LP lower bound or cofactor surplus can remove a
leaf. Every unknown, failed or unfinished leaf remains in the checkpoint.
Both children are constructed before a split replaces their parent, shared
faces remain covered, and zero-width coordinates are retained. FIFO service
and normalized longest-side splitting provide the stated fairness. Singleton
cells can increase arithmetic precision without a spatial split.

Arithmetic resource refusals retry the SAME cell with increasing caps;
validated domain refusals refine it, while unexpected exceptions retain it.
Children inherit learned precision rounds. Once coefficient arithmetic
succeeds, every LP retry uses the IDENTICAL rational problem. All required
row/sign/pivot/rational-size caps grow, including from an initial zero cap.
The nonexcluding refinement gate is actual_verified_gap <= D(B)+epsilon,
where D is the proved coefficient diameter and epsilon decreases with depth
and precision round. Demanding only gap<=epsilon can stall forever on a
wide coefficient family. Whole-cell intervals already include variation;
no second global spatial error is charged. Optional bounded cofactor work
always falls through to the mandatory coefficient/LP route if it fails.

An exact JSON transport preserves rational values and nested compiled DAGs;
decoding alone proves nothing. Replay reconstructs the entire FIFO event
history and partition against caller-owned source, data, bounds and settings.
Independent review found and resolved an initial codec omission for the DAG
inside a cofactor certificate. LP witnesses prove mathematical bounds, not
historical resource counters: the frozen backend does not bind diagnostic
pivot counts to proposal caps. max_steps counts new service quanta, excluding
prior replay, integer cost, serialization and memory. No practical resource
guarantee is inferred from it.

Author/independent generic controls pass75/48. The independent fourteen-input
beta=q0^2-q0+1 fixture requires a split and completes in six events, retaining
all four native affine coordinates. Separate controls retain exact threshold
equality and undefined reciprocal boundaries, recover a tiny singleton
denominator with increasing precision, and complete an exact profile2 LP
after zero-cap retries. Optional cofactor failure cannot bypass the fallback.
Changed context, missing leaves, forged physical claims and altered frozen
rows are rejected. These are abstract arithmetic controls only.

PROFILED_COVER_SOURCE compiles the actual full200 source with94750 DAG nodes,
400 coefficient rows,14 nonlinear inputs and the full4 native affine box.
The29 source checks retain actual source restrictions, physical order, row
selection, exact grid and source fingerprint. Narrower affine source bounds
remain separate; the full native box is a valid exclusion relaxation, not
an existence certificate. Numerical wrapper source review independently
checks start/advance/replay bindings. The same threshold appears in the
target/noise amplitude premise and the strict profile gate. No numerical
optical wrapper is called, even a start that would bind data without yet
evaluating the optical DAG.

PHYSICAL_ACCEPTANCE rebuilds canonical original operations rather than
trusting a caller's diagnostic operation list. For each prism it uses the
unclipped radicand R2-L^2 and original positive exit direction; all canonical
equations follow from checked operation identities. Every source bound,
original/additional guard and added equality remains mandatory. The existing
exact_prior algebraic/Sturm checks establish native membership for exact
rational transformed candidate coordinates. Added auxiliary variables need
their own exact rational witness, and added equalities require enclosure
exactly[0,0], never merely an interval containing zero.

Review corrected an important scope distinction: closed critical-root
acceptance cannot alone claim a strict physical solution. The earlier closed
outer graph was not proved equal to the strict physical closure. Accordingly
closed_source_witness_proved is separate; physical_existence_proved and full
physical-record flags additionally require strictly positive lower endpoints
for EVERY original exit radicand, even in default closed mode. A generic
clipping trap gives a positive diagnostic root3/5 although the original
radicand is-3; it cannot pass the original-physics gate.

Acceptance author/independent controls pass73/94. The independent audit
rechecks raw prism and rotor polynomial identities, exact native caps,
closed/strict boundary semantics, every original guard and source bound,
extra constraints, sparse rows and self-hashed forged declarations. Numerical
acceptance flag construction was inspected at source level only. No remaining
correctness issue was found in this scoped review.

Full200 original-physics compilation yields149599 nodes and818 formal inputs:
18 candidate parameters plus800 uninstantiated target/allowance symbols.
No optical candidate, observation vector, state, expression value, forward
call, contractor visit, cover step or physical acceptance calculation was
executed. Numerical optical APIs remain source-reviewed only. Exact native
inverse-tangent expressions, if accepted later, would not certify rounded
exported angle values. Irrational extra witnesses and added identities whose
intervals do not collapse can remain unresolved. This is a sufficient gate,
not a complete algebraic existence algorithm.

Conditional positive-gap completion concerns the CHOSEN continuous extension
profile over the entire starting region. Physical infeasibility alone need
not give that gap: unphysical extension fits may survive. Original domain
classification, useful optical interval widths, global confinement/uniqueness
and a complete certified recovery procedure remain obligations. T7/T8 gain
implemented components but their completion status is unchanged. The v7
manifest preserves all six earlier snapshots, their immutable source/reports,
the documented historical note revision and the unchanged five-degree cover.

The broad inner inverse still covers |sin(ax_i)|<=7/80, containing at least
±5 degrees for each wedge, with seven other hardware coordinates and all
six rotor settings supplied. It is not a five-degree initialization tolerance.
General seven-degree and transmitted fifteen-degree inversion, broad full18,
T5 and useful global T7 remain open. The active goal is unachieved. A separate
follow-up is examining the exact seven-degree branch-safety/inverse distinction
and primary research that could improve the remaining inverse bounds.

## 2026-09-27 — Seven-degree full-beam branch safety and inverse research

A separate exact scalar induction now extends the explicit full native
beam/glass transmission statement from the earlier five-degree cube to
all three native wedge angles in[-7,7]degrees. Both beam axes retain±25deg,
glass retains its original binary endpoints and all rotor speeds/phases
remain arbitrary. The result holds at every real time. No geometry or
source coordinate is fixed, since this direction-transfer argument does
not depend on them. It does not assert positive signed propagation-leg
lengths, aperture clearance or a floating-implementation discrepancy bound.

The coordinate map is essential: incoming a is a unit-direction sine, not
a slope; the projected face sine is s=sin(alpha)*cos(psi) or
sin(alpha)*sin(psi). Native priors give |a0|<7/16, |s|<1/8 and
13/10<n<29/16. With B=sqrt(n^2-a^2), c=sqrt(1-s^2), q=a*c+B*s,
T=B*c-a*s and R=sqrt(1-q^2), the outgoing components are
b=q*c-R*s and v=q*s+R*c. Exact identities include b-a=s(T-R) and
T^2-R^2=n^2-1. On |a|<=3/4, B>1, c>7/8 and T>25/32.

The sequential bounds are |q1|<85/128<2/3, R1>11/15 and
|b1|<1099/1920<7/12; then |q2|<311/384<13/16, R2>9/16 and
|b2|<71/96<3/4. Finally |q3|<125/128. All three original normalized
exit radicands therefore exceed759/16384, every exit root exceeds1/5,
and every outgoing vertical component exceeds1/20. This proves the next
stage is forward before using its horizontal component in the induction.
Signed reflection/absolute bounds cover all shared rotor states and both
axes, not only aligned positive wedges.

The same proof covers a useful rational transformed outer box: wedge
tangents±1/8 and beam tangents±15/32, with the existing rotor bounds and
native glass. Exact trigonometric Taylor inequalities prove tan7deg<1/8
and tan25deg<15/32. The resulting face sine squared is<=1/65<1/64 and
incoming direction sine squared is<=225/1249<49/256. This box includes
the desired native angular ranges; it does not make every outer point a
native-prior member.

The method is not new: the September22 branch note already uses scalar
envelopes at five degrees and gives an aligned ten-degree TIR obstruction;
the September23 fifteen-degree note handles smaller-beam/glass transmitted
families. The new specialization supplies explicit uniform full-beam,
full-glass seven-degree margins, with independent exact scalar/source
review. No sampled optical state, target record, forward call, optical
expression evaluation, contractor visit or inverse iteration was executed.

The inverse bottleneck remains separate. The frozen five-degree proof's
worst contraction is approximately.994878, leaving little slack to transfer
its constants to seven degrees. Positive transmission margins establish
smoothness, not a global injectivity or stability constant. Primary-source
research is directed at quantitative strong monotonicity/global univalence
and correlated Taylor enclosures. The concrete missing proof is a positive
lower bound for an appropriate preconditioned symmetric Jacobian over the
whole five-coordinate inner domain, or another complete inverse criterion,
with the same supplied hardware/rotor and deterministic data contract.
The full18 prior still includes zero-wedge ambiguities; no uniform positive
full18 Jacobian criterion can hold on that entire prior.

The research pass read Ryu and Boyd's 2016 monotone-operator primer and
Nemirovski's 2004 prox-method paper from their author-hosted primary texts,
and revisited Araya/Trombettoni/Neveu's convex interval Taylor contractor.
The new note states the exact metric/preconditioned symmetric-Jacobian and
Lipschitz inequalities required. A verified strong-monotonicity constant m
and Lipschitz constant L would make the projected step with alpha=m/L^2
contract with norm factor sqrt(1-m^2/L^2). Existence of a compatible feature
fit and full-record acceptance stay separate. An exact abstract skew example
demonstrates why this can pass after a unit-step contraction bound fails;
the optical matrix inequality is not yet proved. The publisher's Gale-Nikaido
metadata was checked, but its subscription text was not obtained or used as
an uninspected theorem. Author/independent branch controls pass30/35.

SEVEN_DEGREE_TAYLOR_RESEARCH supplies explicit original single-prism jets
on the regular (incoming sine,effective angle in radians,index) block.
The sums of absolute Hessian entries obey Mu<209 and Mv<1350, giving
first-order remainder bounds (209/2)*r^2 and675*r^2 in maximum coordinate
radius r. Free derivative identities and rational majorants pass30 checks;
root independently reviewed the bound table, chain formulas and raw transfer
denominators. These are single-prism bounds, not a completed14D coefficient
or inverse estimate. The note gives a rank-free Taylor/cofactor exclusion
test that retains determinant zeros and an explicit preconditioned Jacobian
target. Source-bound derivative propagation and useful optical sharpness
remain unimplemented. Fetched primary PDFs/text and all new reports are
bound in the separate research snapshot.

The v8 research snapshot preserves v1-v7 and the unchanged five-degree
inverse cover. The result advances a branch-safety prerequisite and gives
a smoother domain for subsequent estimates. General seven-degree inversion,
transmitted fifteen-degree inversion, useful global full18 recovery and
the active goal remain open. No new inverse angle range, initialization
tolerance or recovered archive case is claimed.

## 2026-09-27 — Seven-degree source elimination and a correlated Taylor gate

The v9 follow-through advances two distinct parts of the same inverse gap.
No optical point, record, expression value, derivative box, contractor,
forward call or inverse iteration was evaluated. Controls use exact generic
algebra or symbolic actual-source compilation. Earlier v1-v8 sources/reports
and the existing five-degree inverse cover remain unchanged.

SEVEN_DEGREE_SOURCE_COVARIANCE proves a uniformly monotone raw-mean source
readout on the seven-degree branch domain. This is the conditional fiber
with seven hardware coordinates and all six rotor coordinates supplied;
three signed wedges and two source positions are unknown. The separated
absolute-frequency assumptions remain necessary for the stated wedge Gram
floor. At fixed wedges, source affinity F=B+pG and the transformed response
W=atan(beta+zG), z=p/d, give S=W_z=G/(1+(beta+zG)^2)>0. Prior-wide branch
and gain inequalities yield |beta+zG|<41 and

    S > (16/145)^3/(1+41^2) = 2048/2563893625 > 2^-21.

Thus each clipped mean-source inverse is unique and Lipschitz at fixed
wedges. This is a conservative analytical floor, not a measured condition
number or a guarantee for the coupled wedge problem. A target outside the
endpoint mean range is clipped rather than fitted. At the same fixed wedge
and hardware fiber, deterministic per-axis coordinate sup or RMS error eta
changes the recovered physical source by at most min(10,2^21*eta), before
adding arithmetic error. Two independently eta-compatible records have
factor2 before applying the width cap. This is not a full18 or wedge-noise bound.

The four raw X moments are a fixed invertible recombination of the existing
four fitted X features. The raw Y mean replaces the fifth fitted Y DC
feature, using the already available full Y record. Positive sample slopes
alone would not prove a positive fitted-DC derivative because its weights
can have either sign. The new readout avoids that issue. Source elimination
gives the exact wedge derivative <S> Cov_S(u_i,kappa_j*u_j), where
kappa_j=F_sj/(d*c_j*F_p). Its atan denominator cancels and kappa is affine
in source position. Source saturation is included by a generalized clipping
parameter lambda in[0,1]. The baseline weighted Gram block stays positive;
the coupled response-error block still needs a sufficiently strong bound.

The precise remaining target is G_lambda+sym(E_lambda)>=m*I with m>0,
together with a finite Lipschitz upper bound, throughout the actual inner
domain. Those inequalities would give a contracted projected iteration.
An exact generic example shows that shared source weighting can preserve
positivity after independent response windows fail. A separate generic
positive-slope countermodel changes covariance determinant sign and rules
out a universal fixed preconditioner proof from those generic assumptions
alone. Neither example is asserted to be a canonical optical system or the
native200 rotor design. The actual prism covariance inequality is unproved.
Author and independent controls pass38/40; the independent six-row rational
fixture checks Schur identities, clipping and weighted Gram inequalities.

SMOOTH_DAG_TAYLOR implements exact outward dyadic value/gradient and
directional second-derivative propagation through the existing arithmetic
DAG. Every original node still receives value/domain validation, including
unused or algebraically canceled roots and reciprocals. Smooth derivatives
are required only on ancestors of selected outputs. Every varying input
contributes to the centered Taylor remainder; formal data can be omitted
as coordinates only when their actual input intervals are singletons.
Natural bounds, centered quadratic bounds and first-order opposite-corner
cuts are intersected. The second-order coefficient already contains its
one-half factor and displacement radii. Active nonsmooth/zero-root seams,
undefined domains and resource failures return unresolved. Author controls
pass60 plus74 independent analytic probes; independent audit passes45.

COFACTOR_TAYLOR composes this layer with the existing exact cofactor
collector. It encloses C and five signed cofactors rather than differentiating
their absolute-value allowance expression. The necessary-condition gate is
C.lower-sum(e_i*maxabs(v_i))>0, or its negative counterpart. Observation
radii include the caller's deterministic allowance and profile threshold;
native affine-prior radii remain unchanged. Equality and rank-zero supports
retain their cells without determinant division or added spatial allowance.

The new abstract family f(x)=sqrt(1+x)+sqrt(1-x)-31/16 on[-1/4,1/4]
has f>1/32. With four identity affine rows and a fifth zero-coefficient row
of center f and radius1/64, the new gate proves surplus>1/64. Natural
cofactor intervals cross zero, and the rectangular coefficient LP has an
exact zero-residual relaxed witness. This is a proved improvement for that
nonlinear family, not a general dominance theorem over LP relaxations or
an observed optical exclusion. Root controls pass32; the independent audit
uses a different convex-radical family and boundary-equality control.

SEVEN_DEGREE_COEFFICIENTS reconstructs the original unclipped canonical
coefficients on the caller's nonlinear box intersected with the proved
seven-degree rational box. Actual output polynomials are split exactly in
the four affine variables, without substituting geometry/source values.
Every original square-root and reciprocal argument has an analytical guard
binding, but those floors do not override numerical interval validation.
Full200 symbolic construction has88560 nodes,400 rows,14 nonlinear inputs
and1200 raw axiswise exits. The full native four-variable affine box is
retained; additional source restrictions remain separate obligations.

SEVEN_DEGREE_COFACTOR_TAYLOR binds those coefficients to actual selected
rows, formal targets/allowances, native-prior bands and profile threshold.
Full200 symbolic composition has139155 nodes and814 inputs. A four-term
source expansion cap keeps exact original subtrees as opaque atoms rather
than truncating them; larger expansion hit the existing degree cap even
on a small symbolic source. This can weaken algebraic cancellation and
does not establish optical effectiveness. The source compiler passes43
checks, root adapter26 and combined independent audit42. Review found and
fixed Python Boolean/integer metadata aliases in exact replay; canonical
digest comparisons now reject them. This was a binding defect, not a
demonstrated false optical exclusion.

Numerical optical enclosure/replay APIs are implemented and source-reviewed
only. The new gate is not integrated into the frozen v7 cover driver, and
the part of the original prior outside its restricted domain stays unresolved.
Useful optical Taylor widths, the actual coupled covariance bound, global
confinement and a complete recovery procedure remain missing. The v9
seven_degree_taylor_covariance_manifest records all eight reports and
preserves the earlier evidence and primary research bindings. The broad
inner inverse remains |sin(ax_i)|<=7/80, containing at least±5degrees with
outer hardware/rotors supplied. It is not an initialization-error radius.
General seven-degree and transmitted fifteen-degree inversion, broad full18,
T5, useful global T7 and the active goal remain open.

## 2026-09-27 — Uniform seven-degree position response and source stability

The v10 work changes an actual optical structural claim, not only a generic
certificate implementation. On the established seven-degree branch domain,
both axes retain arbitrary rotors at all real times and every native glass,
beam, distance, gap and source value. No spectral separation is required for
the scalar response statements. No optical point, record, numerical optical
expression, derivative box, contractor, forward call or inverse iteration
was evaluated; the proofs use free identities and exact scalar envelopes.

SEVEN_DEGREE_GAIN_REFINEMENT keeps the correlated factors in
M=1/[(c-a*s/B)*(c+q*s/R)] before taking their bounds. The sequential prism
gain floors are3072/3577,55296/69445,57344/100589. Their product is
1391569403904/3569540986655>3/8, improving the v9 conservative source-gain
floor by a factor between290 and291. The old upper G<(11/8)^3 remains.
With wedges and all other hardware fixed, the direct normalized source
estimate has deterministic physical error<=min(10,(8/3)*eta) for per-axis
sup or RMS observation error eta. Comparing two native sources compatible
with the SAME observed record at that fixed fiber doubles eta before the
width cap. Unrelated records do not have this comparison bound.

The root also derived M^2*v^3<=1 from a sum of nonnegative terms. This is
useful in backward response recurrences only after their numerator is
proved positive. A further Delta=T-R<4/3 bound gives final outgoing
|b3|<29/32 and v3>2/5. Exact signed-position envelopes, retaining distance
normalization, reduce the atan argument bound from41 to21/5. Therefore
the transformed raw-mean source derivative satisfies W_z>75/3728>1/50,
where z=p/d; it is not a physical-p derivative. The corresponding physical
source-noise factor can use50; the direct normalized estimator's8/3 remains
stronger. Original source/angular reciprocity also gives
512/1331<d a_out/d a_in<8/3 in unit-sine coordinates at fixed optics.
Neither bound is a full18 singular-value estimate. Author/independent
gain controls pass42/62 with the same-record noise units explicitly checked.

SEVEN_DEGREE_RESPONSE_RATIOS proves uniform positivity of the full position
derivative in each effective face sine, including the adverse signed source
and all upstream/downstream geometry contributions. Exact contact identities
bound |C|<3/5 and |Dcontact|<1/5; positive cubic Bernstein coefficients
certify the two scalar rational envelopes. Combining these with
M^2*v^3<=1 proves the three effective lever floors L1>4,L2>6,L3>17,
and normalized floors L1/d>2/25,L2/d>3/25,L3/d>17/50. The resulting
actual response ratios are

    F_s1/(dF_p)>96/4375,
    F_s2/(dF_p)>1152/48125,
    F_s3/(dF_p)>26112/529375,

all exceeding1/50. Author/independent controls pass75/55. Root also checked the
contact, backward recurrence, source cancellation and affine corner chains.
An alternative free identity also gives |Dcontact|<37/256<1/5; the author
report retains its independently reviewable original contact proof.

The statement concerns independent effective face sines. A physical wedge
angle column includes its signed rotor component and angle-unit conversion;
it need not be positive at every time. Pointwise positive responses still
do not prove the joint covariance positive. Zero wedges, speed collisions,
mixed-sign changes and previously known full18 ambiguities remain present.
No positive signed propagation-leg or aperture-clearance assertion is made.

Combining the two proofs yields F_sj>3d/400. Along a componentwise ordered
effective-sine segment, the output increment exceeds(3d/400)*sum(delta_s)
when at least one increment is positive. With two observed rows, the sum
of their coordinatewise error allowances must be added to the observed
increment. A per-axis RMS allowance eta instead permits pair error
sqrt(2N)*eta, equal to20*eta at N=200; it is not a two-eta pair allowance.
This supplies a quantitative seven-degree premise for the old
September18 rotor-ordering mechanism; it is not a new ordering-completeness
theorem. Only a positive sum with mixed-sign increments is insufficient.
Unknown wedges/rotors remain in the inequalities, so no hidden known-speed
assumption or blind full18 confinement follows.

SOURCE_NORMALIZED_RESPONSE gives a different regional inner map:
r=(Y-B)/(dG), z_hat=clip(mean(r)), Psi=-mean((u-mean(u))*r), with the
hardware and rotor design fixed. Every derivative of z_hat cancels exactly,
including at clipping transitions. Its derivative is a fixed centered Gram
plus a response-ratio error and a residual-times-grad(logG) error. If the
entire convex wedge region has Gram>=gI, component second moments<=h,
ratio errors a_j, residual<=epsilon and RMS log-gain derivative<=ell,
then b=sqrt(h*sum(a_j^2))+epsilon*ell<sqrt(g) proves strong monotonicity
m=g-sqrt(g)*b and upper norm L=3h+sqrt(3h)*b. There is no Smax/Smin factor.
Those optical whole-region bounds remain unproved; verifying them only at
compatible points or on a disconnected set would be insufficient.

The note gives deterministic distance-to-compatible-point and compatible-set
diameter bounds under that criterion. The projected fixed point need not
be physically feasible or a feature root. Its full18 section retains all
unknown geometry/source intervals and credits the September17 exact source
projection. Unknown rotor weights cannot be frozen when differentiating
the full18 problem. Generic author/independent controls pass45/27, including
a distinct three-coordinate noncentered design whose source estimate clips
inside the parameter box. No actual optical record enters either example.

SOURCE_COVARIANCE_CERTIFICATE implements a generic exact shared-moment test:
clear the positive mean source derivative, collect both clipping-endpoint
symmetric matrices and their scalar leading minors, subtract the requested
margin before a fixed congruence, and use validated Taylor enclosures.
A separate whole-matrix norm-surplus test is required for a contraction
factor. Author/independent controls pass57/45. The independent audit also
records an abstract positive matrix family whose broad-box certificate is
unresolved: this kernel is not a complete positivity test or proof of useful
optical interval widths.

SEVEN_DEGREE_COVARIANCE_SOURCE symbolically constructs the actual u,S,r
expressions needed by that kernel. It differentiates canonical affine
coefficients, uses the exact tangent-to-wedge-sine chain factor, reconstructs
unit rotors without dividing by wedge size, and retains all18 physical
inputs and all4 native affine intervals. Full200 construction has189834
nodes; no observation inputs are instantiated. Author/independent source
controls pass38/50. The actual covariance matrix has not been enclosed or
certified, and the new kernel/source are not integrated into the frozen v7
cover. Added source restrictions and native-angle membership remain separate.

The v10 snapshot preserves v1-v9, all earlier reports/source bindings, the
documented historical note revision and the unchanged five-degree cover.
Current gains and individual position responses are materially sharper,
but joint seven-degree covariance/normalized-map positivity, general
seven-degree inversion, transmitted fifteen-degree inversion, useful broad
full18 confinement, T5, T7 and the active goal remain open. The established
conditional inner inverse remains |sin(ax_i)|<=7/80, containing at least
plus/minus five degrees with outer hardware and rotor settings supplied.

## 2026-09-27 — Joint seven-degree inner uniqueness from finite rotor signs

The v11 work establishes a new actual seven-degree conditional inner inverse
theorem. It does not extend the old five-degree contraction to every old
separated rotor design, and it does not solve blind full18. The three signed
wedges and two independent source positions are unknown. The other seven
hardware coordinates and all six rotor coordinates are supplied, with every
endpoint comparison made in the same physical order and supplied fiber.
All native glass, beam, geometry and source ranges remain, uniformly across
those fibers. The whole native wedge cube[-7,7]degrees is covered.

SEVEN_DEGREE_SIGN_DESIGN uses the v10 actual inequalities G=F_p>3/8 and
r_j=F_sj/(dG)>1/50. On the straight convex path in(A=sin(ax),p) between
arbitrary inner endpoints, including a changing source, the exact secant is

    Delta F_k/d=Gbar_k[Delta p/d+sum_j u_kj Delta A_j rbar_kj],

where Gbar is the integral of G and rbar is averaged with the same positive
G weights. Both lower bounds survive averaging. If one axis's sampled rotor
rows cover all eight strict sign vectors with each component magnitude at
least tau, choose the sign vector matching Delta A when Delta p>=0 and its
opposite when Delta p<=0. Every selected contribution then has the same
sign. Therefore

    ||Delta F_axis||_sup >=(3/8)[|Delta p_axis|
                                +(d*tau/50)||Delta A||_1].

This compares arbitrary endpoints, including zero wedge differences. It
does not need small slope windows, positive compressed covariance, small
regional residuals, source clipping or a fixed source along the path.
X coverage alone identifies all three wedges and pX; then any Y row fixes
pY by positive gain. Both-axis coverage gives the direct joint source bound.
The earlier generic compressed-covariance obstruction is not a counterexample
to full-record injectivity. Compression and the full record have different
mathematical properties.

An explicit six-dimensional native rotor neighborhood has now been proved
analytically. On t_k=k/20, nominal speeds(5/8,5/4,5/2)Hz and zero phases
give arguments k*pi/16,2k*pi/16,4k*pi/16. Exact quadrant identities give all
eight X sign vectors at k=1,3,...,15 and all eight Y vectors at k=1,5,...,29.
The half-angle radical identity proves every selected magnitude>=sin(pi/16)
>3/16. Independent speed deviations<=1/250Hz and phases<=1degree perturb
these selected arguments by less than1/16radian, preserving tau=1/8 in both
axes over the entire closed box. These are supplied-design conditions, not
estimated rotors or truth-assisted initialization.

With speed deviations<=1/2000Hz, the same proof applies to all96 odd indices
1..191. Six complete32-index blocks have each sign vector twice in each
axis, yielding12 witnesses per sign group. RMS still averages over all200
rows, hence the lower-bound multiplier is sqrt(12/200)=sqrt(3/50). Exact
clock-grid statements are distinguished from the NumPy time array. Both
design proofs tolerate an explicitly bounded clock error<=1/10000second;
no stored timing array, numerical trigonometric value or optical state is
evaluated. Permuting the design's frequency assignments preserves coverage,
but physical prism order remains fixed and is never declared a symmetry.

For tau=1/8, the joint sup modulus is

    ||Delta p||_infinity+(d/400)||Delta A||_1 <=(8/3)R_sup.

For the repeated design the corresponding max-axis RMS right side is
(8/3)*sqrt(50/3)*R_RMS<=11R_RMS. Two candidates each compatible with the
SAME eta-error record use2eta; a candidate of independently certified residual
rho versus an eta-compatible point uses rho+eta. At d>=50 and eta=1e-5,
the conservative wedge-angle L1 compatible-pair diameters are below.025802
degrees under sup noise and below.106432degrees under per-axis RMS noise
for the respective rotor boxes. These are proved bounds, not measured errors
or claimed recovered systems. Zero discrepancy uses nonstrict inequalities.
Author/independent controls pass50/62 and bind all source hashes.

Each fixed row is also monotone in its axis source and in each signed wedge
sine with sign u_kj. Thus its exact extrema on an inner rectangle occur at
the corresponding sign-selected corners. No corner oracle or inverse
iteration was run. A useful recovery implementation and wider finite-design
coverage remain next steps; the theorem supplies uniqueness and stability of
the inverse on its image, not arbitrary-data feasibility or practical cost.

SEVEN_DEGREE_RESPONSE_ENVELOPES supplies complementary quantitative bounds
for the full18 necessary-condition route. Independent forward sensitivity
propagation verifies F_sj/d<(49,44,56), with source-relative caps(125,112,143)
obtained from the unrounded affine geometry envelopes and exact gain floor.
More significantly,

    F_sj/d=Phi*w_j, Phi=1/(G*v3^3)>0,

with common-weight lower endpoints
(32/175,294912/3129875,2717908992/43470833875) and upper endpoints
(3168/721,3740/707,16940/3609). The proof retains signed contact terms and
uses negative-part bounds before substituting h<=1. The lower chi envelopes
are convex in gap, so their endpoint maxima cover the whole geometry prior.
The common positive factor survives secant averaging and proves a sign for
the formal mixed increment(1/100,-1/4000,0), where independent response
windows cannot. No particular optical pair is instantiated or asserted to
realize that increment. Author/independent controls pass69/46.

FINITE_LAG_RESPONSE_CONES keeps the common factor and direct column caps in
one four-variable response polytope. Its exact rational support requires at
most eight factor candidates: feasible endpoints and the six possible column
crossovers. Interval increments and distance use their correct signed
endpoints. Common synchronized increments Delta s_j=A_j*c+e_j retain c
before interval relaxation; two c endpoints suffice by affine dependence.
Exact returns, zero columns and mixed-sign cancellation remain present.
The author controls pass94, including24 comparisons with independent exact
polytope-vertex enumeration. The replay verifier binds caller-owned bounds
and explicitly does not certify their optical provenance. No observation,
including a zero observation, is instantiated and no optical exclusion runs.

The cone inequalities can keep all18 unknown coordinates within the seven-
degree wedge restriction, including collisions and zero wedges. They improve
some mixed-sign finite-lag necessary tests and credit the September17/18
ordering/projection work. They do not prove broad rotor confinement or
complete synchronized-family exclusion. A two-time difference within one
RMS-eta record has allowance sqrt(2N)*eta=20eta atN200; this differs from
the2eta whole-record comparison in the new conditional inverse theorem.

The five-degree contraction and all v1-v10 evidence remain unchanged.
Actual seven-degree joint injectivity/stability is now proved on the new
sign-design class; general old-class seven-degree inversion, practical
seven-degree inner execution, blind full18, transmitted fifteen-degree
inversion, T5, useful global T7 and the active goal remain open. No optical
point, record, derivative box, contractor, forward call or inverse iteration
was evaluated during this work. The evidence snapshot is
seven_degree_sign_inverse_manifest_v11.json.

The finite-lag kernel's independent audit passes84 checks with a distinct
exact four-dimensional active-set oracle, signed distance cases, narrowed
factor intervals, interior crossover extrema, zero/singleton branches,
synchronized endpoint enumeration and corrupted replay. No defect was found.

SIGN_INJECTIVITY_RESEARCH reads the primary full text of Muller, Feliu,
Regensburger, Conradi, Shiu and Dickenstein, Foundations of Computational
Mathematics16(2016),69-97, DOI10.1007/s10208-014-9239-3, arXiv1311.5493v2.
Its generalized-polynomial family theorem is relevant background, not directly
applicable to the radical optical map. The new direct sign-class argument
checks every nonzero ternary direction: some row must have all nonzero
products of one sign. This gives a zero-safe qualitative-rank criterion and
global injectivity when the integrated Jacobian remains in that sign class.
All eight strict wedge orthants are necessary for the unrestricted strict-row
positive-magnitude class; exact zero rows can supply different certificates,
and actual bounded/correlated optical magnitudes may need fewer orthants.
The necessity claim is carefully restricted to avoid deleting these cases.

The research control passes117 checks, including80 direction signs and eight
exact missing-orthant null constructions. It also shows the frozen generic
compressed-covariance countermodel has globally injective full records with
lower modulus |Delta z|+(7/80)||Delta v||_1. This reinforces the distinction
between a failed compressed feature proof and failure of the full inverse.
The note proposes eight bounded-coefficient secant LPs as a next quantitative
test, using existing exact LP infrastructure. No optical LP, rotor scan or
numerical optical value is evaluated. Primary full text, metadata and source
index hashes are included in the evidence.

SEVEN_DEGREE_TRIANGULAR_READOUT adds a constructive result for the EXACT
nominal supplied rotor design, separate from neighboring-box injectivity.
At speeds(5/8,5/4,5/2)Hz, zero phases and t_k=k/20, Y rotor rows at indices
0,8,4,2 are(0,0,0),(1,0,0),(r,1,0),(h,r,1), where r=sqrt(2)/2 and
h=sin(pi/8). Thus Y0 directly gives pY from the full flat offset
(6+2g+d)*tan(beam_y)+3 sum a0/sqrt(n_i^2-a0^2). This includes the final
d*tan(beam_y) term, unlike the earlier atan Bflat. Successive Y rows then
give A1,A2,A3 through globally monotone scalar inversions on the full
seven-degree interval. X row2 has rotor(q,r,0), q=cos(pi/8), so its affine
source readout obtains pX without feeding the A3 error into that step.
No optical expression is instantiated to obtain these analytic identities.

The scalar slopes exceed m=3d/400. Each inverse is clipped to its endpoint
image, because errors in earlier coordinates can put the next target outside
that image even for exact compatible data. The clipped inverse obeys
|T_f(y)-x|<=|y-f(x)|/m for every native x. Midpoint interval queries of
width nu retain the correct half when the comparison is strict, or return
the midpoint when ambiguous with error<=nu/m. No endpoint queries or exact
equality decisions are required. J queries give error<=max(a_star/2^J,nu/m),
with a_star=sin(7degrees). This is an explicit oracle contract, not a claim
that the current floating model supplies those enclosures.

The response caps and r<3/4,h<2/5 give noiseless sine-error multipliers
1,4901,64701043/3. The X-source coefficient is(1600/3)*(49+33*4901), with
no third-wedge term. With exact source arithmetic, J38 needs at most114
scalar queries and proves each wedge error<0.001degree and each source
error<0.001. For a complete finite-arithmetic contract use J40, query widths
and source-computation errors<=2^-60, returned native sine-point rounding
<=2^-60 and final degree-conversion rounding<=2^-60. The common sine-error
base becomes1/(8*2^40)+(2035/192)*2^-60; all five errors still meet the
same tolerances with at most120 scalar queries. Author controls pass104.

No optical scalar inversion, observation or recovery trial is executed.
The query count excludes bit-complexity or practical timing claims and
requires validated original-model enclosures at the stated precision. Exact
data compatibility and supplied hardware are explicit; measurement-error
recurrences are conservative and may saturate native-width caps. The exact
rotor zeros are essential, so the procedure is not established for the
whole six-dimensional neighboring rotor boxes. It recovers five inner
coordinates with thirteen supplied and does not solve the full18 objective.

The independent triangular audit passes61 checks with no defect. It binds
the final author report and verifies120 distinct generic scalar cases,
four independent noisy/source-rounded triangular systems with six
out-of-image intermediate stages, the source quotient and final J40/120-query
contract. The extra J38 finite-return-error calculation in that audit has
its own hypothesis and is not substituted for the advertised query-width
contract. These are generic controls, never optical observations or states.
The final v11 collection has nine reports and687 checks, plus retained
primary research. All prior snapshots, frozen sources/reports and the
five-degree cover remain preserved; T1-T10 status labels remain unchanged.

## 2026-09-27: full13 outer transport and conditional full18 localization (v12)

Continued the four-worker mathematical attack from frozen v11, with primary
research and independent audits. The restriction on numerical optical work
remains: no optical states, records, observations, centers, derivative boxes,
DAG values, contractors, forward calls or inverse trials are evaluated. All
controls are free symbolic identities, exact scalar envelopes or explicitly
generic nonoptical systems. Frozen v1-v11, paper/main.tex and prior solver
campaigns remain unchanged.

HARDWARE_TRANSPORT gives explicit forward sensitivity bounds on the complete
native seven-degree branch domain, retaining all hardware/source ranges:
|F_nj|<d*(12,12,10), |F_betaRadians|<52d,
|F_betaDegrees|<(286/315)d, |F_d|<145/64, |F_gap|<12265/4032.
The new index identity is
M_n=-M*n[a*s/(B^2*T)+s^2/(B*R^2*v)], with b_n=n*s*v/(B*R).
The proof includes the source-to-prism6tan(beta), the glass plate term and
both shared gap terms. Exact positive affine envelope coefficients place
normalized d/g maxima at d50,g15. Author controls pass51; independent
primitive quotient differentiation and FORWARD sensitivity composition
reproduce the author's backward-adjoint envelopes and pass39. These are
uniform inequality coefficients, not evaluated optical derivatives.

SEVEN_DEGREE_ROTOR_TRANSPORT uses the frozen raw response caps(49,44,56),
retaining |A_j|*min(2,phase_difference_radians). The exact native-grid center
is199/40 seconds, with centered phase difference zeta=Delta ay+1791Delta N
in degrees and time variance13333/1600. The squared RMS speed coefficient
is1079973; the sup coefficient is1791. Signed phase/speed compensation is
formed before taking absolute values. Author controls pass53. The existing
September17 phase-envelope and hardware two-point work is explicitly credited;
the new result extends quantitative transport to the full seven-degree domain.
Hardware and rotor differences telescope through the same inner tuple to give
an explicit bound for all13 outer coordinates. No outer lower sensitivity,
blind identifiability or exact-fit existence follows.

SEVEN_DEGREE_OUTER_PROFILE_V12 uses a sign-complete anchor fiber h0 and any
native inner anchor x0 with certified residual rho. With candidate allowance
eta and whole-cell outer variation E, every compatible inner point satisfies

  ||Delta p||infinity+(d0*tau/50)||Delta A||1 <=(8/3)(rho+eta+E).

Only the anchor needs sign coverage and it need not lie inside the candidate
outer cell. Candidate rotors can be arbitrary native values, including speed
collisions, provided E covers them. The whole source/wedge domain is retained;
no truth or exact-fit anchor is assumed. The exact generic kernel preserves
all32 coupled halfspaces and all18 variables. Its dual support is
max(max_j|a_Aj|/w,|a_px|+|a_py|), w=d0*tau/50. A generic399-row sign block
plus a held residual proves a whole13-outer-cell exclusion that the coordinate
hull misses. A separate400-row family T(x+B h), with every outer column active,
shows why inner injectivity alone cannot imply outer identifiability. Neither
example is a canonical optical system.

A separately named feedback helper requires a signed affine residual minorant
on the ENTIRE original outer-cell times inner-domain product. If its anchor
value including model error is b and k=(8/3) times the weighted dual norm,
then each candidate residual t obeys t>=b-k(rho+E+t). Therefore the full
profile lower bound includes max(0,[b-k(rho+E)]/(1+k)). A lower bound valid
only on the eta-localized set cannot be called a full profile lower bound.
The generic example yields global lower3/1216, compared with the larger
localized-row lower1/128. Author controls pass68 and the independent audit
passes61, including729 exact support comparisons and18 feedback multiplier
cases. Source binding remains explicitly unverified in every returned
certificate. No actual optical residual or exclusion is instantiated.

FULL18_FINITE_RECORD_RESEARCH newly reads Das--Yorke1506.06810v5 and
Laskar math/0305364v3, and rereads primary Aubel--Boelcskei1604.07196,
Moitra1408.1681 and Batenkov--Goldman--Yomdin1904.09186. It credits existing
finite-record and constructive Prony work. The analytic three-torus shell
count is4j^2+2, with cumulative counts7,25,63,129,231 through orders1..5.
An explicit complex-strip Fourier tail formula connects a bounded analytic
waveform to a finite exponential sum only if its strip width/norm are proved.
The full order4 shell exceeds the200-sample matrix-pencil theorem's node
budget; order3 permits no integer stride greater than1 at full support.
Merged aliases can cancel and require amplitude/support branches. The exact
native triple(sqrt10,sqrt11,(20-sqrt10-2sqrt11)/3) is continuously rationally
independent but has(1,2,3).N=20, so two order3 modes alias on k/20. This
illustrates the already known sampling-rate exception, not a new optical
ambiguity theorem. A generic polynomial-interpolation argument shows that
analyticity and positive response derivatives alone cannot identify unknown
frequencies from finite data without physical coefficient restrictions.
Research controls pass53; primary full texts and provenance hashes are saved.

This is progress toward source-bound full18 exclusion. The resulting outer
variation bounds are conservative and can be vacuous on broad cells. The
actual residual-minorant bridge, useful global cover size, full-prior branches,
optical implementation/execution and blind full18 recovery remain open. The
conditional five-degree contraction and v11 seven-degree inverse statements
retain their exact scopes. T1--T10 status labels and the active goal remain
unchanged. The v12 snapshot is seven_degree_outer_transport_manifest_v12.json.

The independent rotor audit passes60 with no substantive defect. Its six
generic discrepancy-Gram fixtures and six exact interval-image profile
fixtures verify RMS composition, the cross-term hardware telescope, strict
boundary retention and why an anchor residual is not a lower profile bound.
The final v12 collection contains seven reports and385 checks. All prior
snapshot/report/source bindings are replayed before exclusive snapshot
creation; no new kernel is integrated into the frozen optical cover driver.

### 2026-09-27: v13 spectral structure and actual-source residual minorants

Continued with three parallel agents and independent cross-audits. The
execution boundary remains unchanged: free symbolic identities, exact scalar
envelopes, generic nonoptical controls and optical symbolic compilation only.
No optical point, state, observation, extension, derivative box, forward or
inverse evaluation was performed. All v1-v12 artifacts remain immutable.

SEVEN_DEGREE_GENERATOR_GAP strengthens the earlier ideal monotonic generator
argument on the full native seven-degree branch domain. With m=3d/400,
subtract m*s_j from BOTH coordinate responses before pairing X+iY. The
subtracted term is exactly m*A_j*exp(i*gamma_j); it changes only the +e_j
generator, including negative wedges/speeds. Nonnegative residual derivatives
then give |zeta_l|<=|zeta_ej|-m*|A_j| for l_j nonzero and l!=e_j.
Using upper caps d*(49,44,56)*|A_j| yields uniform relative gap3/22400.
Arbitrary phase offsets preserve magnitudes. Zero wedges stay invisible,
and no uniform finite-noise visibility floor for tiny wedges is asserted.
The exact finite-catalog kernel requires a complete order-K support catalog,
valid magnitude/frequency intervals, and sampled separation of ALL labels,
including zero-amplitude modes and DC. With frequency error epsilon it
requires delta>2*(K+1)*epsilon and prunes using radius(K+1)*epsilon.
Its output remains conditional; it does not certify an optical catalog.
Author/independent audits pass33/44, with126/50 hidden-label pruning
comparisons and six distinct generic monotone polynomial pairs.

SEVEN_DEGREE_COMPLEX_STRIP provides an explicit holomorphic phase strip
|Im(theta_j)|<=1/128 with |F_axis|<1000 and |X+iY|<2000. All real native
hardware/source ranges and seven-degree wedges remain; hardware is not
complexified. Principal square roots stay on their original branches via
strict positive-real radicand disks, with exact propagated perturbation
and position bounds. This extends the earlier local flat-centered complex
domain and compactness arguments to an explicit whole-torus strip.
Author/independent scalar controls pass82/32. The order3 Fourier-tail
majorant lies between33billion and34billion and is unusably coarse.
This is a bound on the upper-bound formula, NOT an actual-tail lower bound.

OPTICAL_HARMONIC_SELECTION credits the earlier reflection/parity theory.
At normal beam F(s;0,p)=B(s)+p*G(s), with B odd and G even. The paired
source-free signal obeys sum(l_j)=1 mod4. Circular support counts through
degrees2/4/6 are3/22/73; generic odd support counts are6/44/146. The finite
lag filter (z_k-z_(k+L))/2 has explicit mismatch for candidate unknown-rotor
cells near simultaneous odd half-periods. Normal-beam source terms cancel
at exact relation; arbitrary beam/source deviations have explicit budgets.
Exact relations can force collisions; separated-node recovery is not claimed.
The filtered q-mode approximation yields a necessary Hankel singular-value
or Hermitian-minor test without prior frequency estimation. Noise, Hankel
multiplicities and tail allowances are carried explicitly. An informative
rank test needs at least2q+1 samples. The tail norm premise was clarified
to mean the COMPLEX strip throughout, not the real torus. Author/independent
controls pass115/73. The explicit global tail still prevents useful optical
accuracy/exclusion conclusions from these counts alone.

SEVEN_DEGREE_RESIDUAL_MINORANT_V13 closes the symbolic actual-equation
provenance gap for a future residual bound. It translates every original
coefficient node, replaces wedge tangent by A/sqrt(1-A^2), and reconstructs
each selected affine output before making a signed residual combination.
The sine cap31/250 contains seven degrees while mapping inside tangent1/8;
original narrowed chart restrictions remain enforced. Full200 compilation
has93,386 nodes,18 physical inputs and400 FORMAL uninstantiated targets.
No zero optical record or other numerical data were supplied.

The new generic partial-inner Taylor kernel evaluates the base over H times
{x0}, retaining all thirteen outer intervals. Its directional half-second
coefficient is used once, and signed slope-error intervals give an affine
minorant on the ENTIRE original product H times X. Original unused domain
nodes remain guarded. The actual-source wrapper binds row weights/data,
whole domain and exact v12 feedback coordinates. Weighted row noise is
used only in selected-row compatibility; the global profile uses the raw
minorant without a duplicate noise subtraction. Anchor/sign/transport
premises remain explicitly unverified, so feedback stays conditional.
Author controls pass69 with81 generic polynomial probes. Independent
generic controls pass32 with786 exact function comparisons; a separate
source/math audit finds no substantive defect. Numerical optical entrypoints
were patched forbidden during symbolic controls and remain unexecuted.

The v13 snapshot seven_degree_spectral_minorant_manifest_v13.json contains
eight reports and480 checks. These counts inventory evidence, not progress
percentages. Useful optical remainder bounds, actual observed exclusion,
full-prior coverage, practical cost, integration and blind full18 recovery
remain open. T1--T10 completion statuses and the archive ledger are unchanged.
Seven degrees here refers to physical wedge magnitude in the stated
theorems, not a proved seven-degree initialization-error basin.

### 2026-09-27: v14 exact corners, missing generators and a finite-record obstruction

Previous goal turn classified as progress: v13 changed source-bound state
and was independently replayed. The goal remains active and incomplete.
This round again used three parallel agents, with independent cross-audits.
All v1-v13 sources/reports are immutable. Only these three living logs
advance. No optical point, parameter-box, state, observation, extension,
derivative, contractor, forward or inverse evaluation was performed.

SEVEN_DEGREE_AFFINE_CORNERS_V14 uses the older exact four-affine split
before the v13 partial Taylor bound. For common wedge/source slopes,
minimizing f-a_p.p over(d,g,pX,pY) is exactly the minimum of16 corners.
Corner Taylor bounds therefore require only three wedge derivatives while
the eleven remaining outer intervals stay unknown. The common slopes are
essential; separate corner planes cannot be merged by their intercepts.
The whole original product and all18 parameters remain covered.

Independent audit found a conservative but avoidable loss: coefficient
reconstruction expanded(D-G)h into D*h-G*h. The final compiler proves
affinity structurally, then substitutes corner constants directly into
the original ancestor DAG, preserving factorization and reusing every
affine-independent node. All original nodes remain as guarded prefix.
In a distinct quadratic fixture, the original v13 intercept is11/8;
expanded corner reconstruction gives2; corrected direct corners give11/4.
The author's simpler fixture improves4 to6. These are generic nonoptical
comparisons, not demonstrated optical gains. Author/independent controls
pass66/30, with300/256 separate rational comparisons and all16 exact V/G/Q
formulas. Full200 symbolic compilation has121,378 nodes,18 physical inputs,
400 uninstantiated formal targets and16 corners. No optical data were bound.

CORNER_PROFILE_PERSPECTIVE_V14 optimizes the common five slopes from already
certified corner base/gradient/half-second enclosures, without rerunning
derivatives. The exact formulation has55 variables and218 inequalities.
For scale t and scaled slopes v, t+(8/3)D(v)<=1 and the corner Taylor
inequalities give linear objective z+v.x0-(1-t)(rho+E). This is the classical
linear-fractional perspective applied to the existing weighted inverse
modulus. At t0 the objective cannot be positive; actual certificates need
t>0. Numerical LP status is not evidence: only proposed slopes are used,
and intercept/scale are reconstructed as exact rationals. The generic
control gets positive499/65536 from an inconclusive zero-slope baseline.
Author/independent audits pass32/35, including729/2187 exact generic
comparisons, a nonzero anchor, unequal boxes and a singleton wedge.
Optical enclosure provenance and anchor/sign/transport premises stay false.

PARTIAL_GENERATOR_CATALOG_V14 removes the need to detect every nonzero
retained Fourier line. Each detected line still needs an injective label
correspondence and all-label sampled separation. If omitted generators
are bounded by Gamma, the strict selection gate becomes
L>(1-3/22400)*max(Gamma,U_remaining). Unselected physical slots retain a
detected alternative OR d*|A_j|<=400Gamma/3, with speed/phase unrestricted.
All injections into physical prism positions remain; no ordering symmetry
is invented. Missing caps, failed separation and resource limits retain
unrestricted or unpruned supersets. Zero and arbitrarily weak wedges survive.

A residual representation r=Vb+e with verified V*V>=sigma²I gives omitted
coefficient cap(R+E+T)/sigma. It needs q<=200, not the2q samples for blind
matrix-pencil recovery; q129 order4 is dimensionally possible conditionally.
Selective residual functionals can cap a chosen generator even when nuisance
columns alias, provided the target response is nonzero and every leakage
has a residual-coefficient cap. Target aliases can cancel and forbid that
inference. Regional optical frames/functionals, useful physical tails and
detected correspondences are not established. Moitra's retained primary
finite Vandermonde text was reread for its exact scale/separation hypotheses.
Author controls pass66 with24 enumerated partial-support covers. Independent
controls pass50 with64 additional cover/resource cases and30 dyadic-root
checks; complex PSD, selective leakage and all physical-slot injections
were separately reviewed with no defect found.

SEVEN_DEGREE_TAIL_BOUNDS_V14 supplies a new generator-conditioned energy
constraint. With a_j=|sin(ax_j)|, m=3d/400, L_j=dU_j and g_j=|zeta_ej|,
sum_(l!=ej) l_j²|zeta_l|² <=(g_j-m*a_j)(L_j*a_j-g_j).
This follows from the real sector inequality, paired derivative norm and
Parseval, subtracting the generator BEFORE bounding. Summing removes the
three generators once. Balanced integer shell weights give a torus-L2
tail bound. A generic concentration example shows why it cannot be used
as finite-record RMS without additional information.

The consequential result is an original-model finite-record obstruction.
The first two wedges may be zero, the last±7degrees, beams normal and all
unused glasses/gap/source positions native and unknown. The symbolic odd
response is d*tan(arcsin(n*s)-arcsin(s)); its source term is even. All odd
power coefficients are positive. At N1=N2=N3=2Hz and phases0, the exact
functional(1/200)sum(-1)^k X_k kills every total-order<=4 approximant at
those SAME generators and kills the source-even term. Roots-of-unity
orthogonality gives Lambda(cos^5)=1/16 with positive higher contributions.
Since c5(13/10)=177399/800000 and sin7>1/9, both sup and per-axis RMS
approximation errors exceed19711/1679616000>1e-5 on the original200 grid.
No optical entries are instantiated: this is a free finite-group and
coefficient proof. Uniform tiny degree3/4 tails are therefore a false
target, not merely awaiting a sharper upper-bound calculation.

At equal10/7Hz, a14-point alternating rule on the first196 samples similarly
gives degree<=6 sup error>1/190000 and full200 RMS error>1/200000 in a native
high-index/distance subfamily. This does NOT rule out1e-5 degree-six accuracy.
Different free frequency dictionaries, alias-adapted supports, higher order
and exact-model inversion remain possible. Neither approximation lower bound
is an inverse ambiguity or an impossibility theorem for full18 recovery.
Tail author/independent controls pass115/28, including separate formal-series,
Machin sine and root-of-unity checks. The frozen v13 strip proof remains valid.

Snapshot: seven_degree_corner_spectral_manifest_v14.json, eight reports
with422 checks. Useful actual
optical minorants, certified detections/regional frames, practical global
coverage, implementation integration and blind full18 recovery remain open.
T1--T10 labels and the archive recovery ledger are unchanged.

## 2026-09-27 — Fifteen-degree margin inverse and joint row/alias constraints (v15)

FIFTEEN_DEGREE_RESPONSE_V15 supplies a substantive angle extension: on the
explicit common transmission-margin class R_i²>=1/2, all three physical
wedges may range through±15degrees. All native hardware/beam/source ranges
remain variable within that coupled class. The supplied-thirteen inner
inverse still requires identical outer coordinates at its two endpoints
and finite rotor sign coverage. It does not infer those thirteen unknowns
or cover every strictly transmitted15-degree configuration.

Free scalar identities give contact caps |C|<13/16, |D|<23/100. Keeping the
same gap/distance correlations in the exact backward levers proves
L1/d>1/160, L2/d>9/100, L3/d>7/100. Each normalized response F_sj/(dF_p)
exceeds1/625 and source gain exceeds1/5. On a sign-witness row the effective
sine increments are all ordered. Before imposing transmission, T>0 proves
q increases in both incoming and face sine. Ordered endpoints bound every
intermediate q; forward v>0 is proved before using positive b derivatives.
Induction preserves that row's margin. Other rows need not remain feasible
along the comparison path. Thus nonconvexity of the full domain is harmless
for this endpoint argument, without pretending the whole domain is convex.

With both axes covered, ||Delta p||inf+(d*tau/625)||Delta A||1<=5 R_sup.
The known tau1/8 design family applies unchanged. A conservative native
two-eta-compatible wedge L1 diameter is62500*eta degrees; this is an analytic
bound, not an observed error or an initialization tolerance. Author/independent
controls pass156/52 with distinct Bernstein partitions, direct contact
identities, exact affine minima and81 zero-inclusive generic sign patterns.

Separately, output elimination in the last prism gives
F_s3=Delta*(d*J-F*K)/(c*R²*v), J>1393/1600, |K|<2831/4000. A target with
|y|/d<6965/5662 has at most one last-wedge amplitude on its strict transmitted
same-fiber interval when the chosen rotor factor is nonzero. Clipped-output
monotonicity proves this without requiring all outputs along the full interval
to stay in-band. Two candidates with residual eta have amplitude diameter
<=2eta/(|u3|mW), W=|y|+eta in-band and mW=(3/10)(d*j0-W*k0). This result
allows arbitrarily small positive transmission margins but fixes every
other parameter and does not assert root existence.

JOINT_CORNER_WEIGHTS_V15 extends the frozen perspective LP to signed row
weights lambda, ||lambda||1<=1, and common five slopes. There are55+2m
variables and219 inequalities. Scaled positive/negative weights sum to at
most the perspective scale. Only proposed rational weights/slopes are used;
all16 combined enclosures and intercepts are reconstructed exactly. An
independent audit corrected one proof sentence: t0 does not force v0.
The weight constraint forces u0 and interpolation at the source anchor
still proves objective<=-(rho+E), even for nonzero v. Code was correct.
Author/independent controls43/40 include240 signed endpoint extrema and
2,187 independent abstract full-record comparisons.

SEVEN_DEGREE_JOINT_ROWS_V15 binds the proposal to ordered original rows and
formal target symbols. It compiles the weighted expression in original
grouping and with exact polynomial/atom collection, retaining every original
value-domain node. Each variant needs all16 same-slope corner proofs;
the stronger valid intercept wins. Collection/resource failure preserves
a complete grouped proof. The automatic generic h²+p±1 example improves
rowwise/grouped1/2 to collected1. An independent radical/quadratic example
gives15/8 and243 direct plane comparisons. Author/independent39/28 pass.
Full200 symbolic compilation has108,589 nodes,18 physical symbols,400
formal targets and6,400 individual row corners. Numerical optical proof
entrypoints were patched forbidden during those controls. Actual optical
source checking remains unexecuted; generic gains are not optical gains.

ALIAS_ENERGY_QUOTIENT_V15 projects the generator-conditioned coefficient
energy onto exact sampled-character classes. With positive derivative
weights Q_l, nuisance class sums hC satisfy sum|hC|²/HC<=B,
HC=sum_(l in C nuisance)1/Q_l. The explicit minimizing lift proves sharpness
for that relaxed ellipsoid only. A finite200 functional bound uses
sqrt(B*sum HC|muC|²)+||w||E; no full-label separation, Gram inverse or
frequency recovery is assumed. E must independently include the finite-record
omitted tail. The old torus energy does not bound it at sampling sites.

Eight closed sign sectors use the independently verified native phase
prior±18degrees; it is not inferred from wedge size. Same-sign generators
in one class have a conditional sum cap from phase-sector noncancellation.
Mixed signs, weak/zero wedges, uncanceled unknown DC, missing statistics
and failed alias premises remain explicit branches. Additional ungrouped
aliases do not invalidate the dual inequality. The generic200 examples
are unrelated exact characters, not optical records. Author/independent
58/57 pass, with independent56 energy decompositions and8 true sign sectors.
Prior shifted-Parseval/kernel and signed-row/cofactor work is credited.

Snapshot fifteen_degree_joint_alias_manifest_v15.json binds eight reports
with473 checks and preserves every v1-v14 source/report. Only this diary,
HANDOFF.md and paper/FINISH_LINE.md advance. No optical point, record,
state, enclosure, derivative, contractor, forward or inverse evaluation
was performed. No observation, including a zero observation, was instantiated.
Blind full18 reconstruction, useful actual optical exclusions, global cost
and integration remain open. T1--T10 and the recovery ledger are unchanged.

## 2026-09-27 — Output-forced fifteen-degree margins and shared sample constraints (v16)

OUTPUT_FORCED_FIFTEEN_MARGIN_V16 proves that an original, uncentered model
coordinate satisfying |F|<=d/14-17501/25984 forces all three exit-root
squares R_i²>=1/2, for physical wedges within±15degrees and all remaining
native priors. At the native minimum distance50 this band is10757/3712
(about2.898 position units). The proof chooses the first failed margin,
propagates positive direction floors through the remaining prisms, and
keeps the shared gap when bounding signed positions. Independent free
derivation, transfer-matrix and rational-envelope checks find no defect.
Author/independent controls pass83/41; no optical state was evaluated.

For a candidate cell use its distance lower bound in that threshold.
Equivalently, any error-adjusted model-output cap r gives the necessary
disjunction: all three row half-margins hold OR d<14r+17501/1856. Both
branches remain; this is not recovered geometry. Only selected rows need
pass the gate, but their rotor sign coverage must be proved separately.
Those rows then supply the v15 conditional inner inverse premise from
data. The same13 outer coordinates are still required for comparing inner
solutions. Larger-output and arbitrary transmitted15-degree branches remain
unresolved. Coordinate allowances must include the original per-axis RMS
budget and any model/arithmetic error; recentering the scan is invalid here.

TERMINAL_OUTPUT_BAND_V16 sharpens the last-face contact ratio to3/11.
The derivative bound is F_s>(3d-2|F|)/11 over the native18-degree prior,
giving the monotone output band |F|/d<3/2. With terminal15degrees it is
F_s>(3057d-1791|F|)/11000, with band1019/597. A degree-six positivity proof
uses exact Bernstein coefficients; an independent exact Sturm chain verifies
the same global sign without that partition. Author/independent94/79 pass.

Novelty correction: REGULAR_OPTICAL_BOUNDS_2026_09_17 already implies the
unspecialized last-face band27/19. The v15 band6965/5662 is valid but weaker
than that prior consequence, and must not be called a new terminal extension.
The new v16 bands3/2 and1019/597 do improve it. The v15 all-three conditional
15-degree inverse remains a separate new result. Frozen v15 files are kept
unchanged; this entry records the correction explicitly.

The terminal module also supplies a center-free rectangle recurrence-defect
test. It retains empty/zero-wedge/zero-speed possibilities and never requires
an exact inverse at the measured center. All396 full200 symbolic recurrence
consequences preserve the original18 coordinates; they do not add independent
information to the old rotor equations. The terminal dimension reduction,
noisy inverse fibers and rank invariance were already in the September17
T5_TERMINAL_ROTOR_REDUCTION and are credited. Inverse diameters apply per
fixed upstream fiber, not to the union over an uncertain upstream cell.

MONOTONE_SAMPLE_TRANSPORT_V16 gives an exact finite relaxation from the
proved seven-degree response bounds. With x_kj=d*A_j*u_kj, the common scalar
response has gradient in [3/400,(49,44,56)]. Its asymmetric support costs
c(x_i-x_j) bound every directed output difference. A finite interval fit to
some Lipschitz response exists exactly when l_i<=h_j+c(x_i-x_j) for every
pair; a minimum of translated support cones constructs the extension.
This response need not be optical. For outward/incomplete costs, shortest
paths and negative cycles give the exact finite potential constraints.
The older September18 rotor ordering test and v11 lag cones are credited.

Nonnegative flows combine those edges with one shared RMS noise charge:
lambda*y-B<=sqrt(N)*eta*||lambda||2, lambda being the flow divergence.
Intermediate-row errors cancel. A generic200-row mixed-sign example survives
every independently inflated pair test but is excluded by the aggregate
flow; its exact relaxed RMS distance is1. This is a nonoptical construction,
not an observed recovery or optical exclusion. Author/independent52/22 pass,
including729 distinct sparse potential systems. The asymmetric extension
is proved directly; McShane1934 metadata was checked, but inaccessible full
text is not claimed as read or used for an unverified theorem.

SUPPORT_FLOW_SOURCE_V16 binds that necessary test to the authenticated
original clock, sample axes and all18 unknowns. The smaller exact feature
program retains common d*A and the rational rotor recurrence; free unit
identities justify clipping direction components to[-1,1]. Combined costs
keep both original grouping and exact polynomial/atom collection. An upper
bound from either complete expression is valid; source restrictions remain
obligations. The full200 symbolic program has7,239 feature nodes and14,234
cost nodes, with90,983 original optical nodes authenticated. No physical
rotor or optical expression was numerically evaluated. A generic common
quadratic/radical cancellation example and independent source corruption
checks pass; author/independent40/45. The numerical source gate remains
unexecuted and proves no actual optical exclusion.

Snapshot data_margin_transport_manifest_v16.json binds eight reports with456
checks, preserving v1-v15, the frozen cover and driver. These checks support
the stated lemmas and source binding; their count is not distance to the
finish line. Only DIARY.md, HANDOFF.md and paper/FINISH_LINE.md advance among
living prior files. No optical observation, including zero, was instantiated.
Full18 reconstruction, useful whole-cell optical confinement, practical cost
and integration remain open. T1--T10 and the archive ledger are unchanged.

## 2026-09-27 — Terminal constraints over shared cells and15-degree order hulls (v17)

TERMINAL_HARMONIC_CUT_V17 proves a new consequence of the v16 output gate.
On a qualifying15-degree row, flattening the last face preserves the
half-margin along the entire scalar path. The hypothetical flat-reference
output obeys |F0|/d<9861/6104<13/8<1019/597, automatically supplying the
terminal inverse's reference-band premise. The proof uses exact transfer
elimination, |1/M-1|<39/109 and an integrated tangent change<65/56.
No reference output is numerically evaluated or assumed calibrated.

Along that path the terminal effective-sine response lies between
(1173/88000)*d and12*d. For m selected original samples at which BOTH axes
pass the error-adjusted v16 gate, write A3=epsilon*a with a>=0 and either
closed sign epsilon. The exact finite correlation
P=epsilon*sum u_ka*(y_ka-F0_ka) satisfies

    (1173/88000)*d*a*m-b <= P <=12*d*a*m+b,
    b²=N*m*(eta_x²+eta_y²).

N is the original record count. This is a finite functional of the exact
flat-reference residual, not a finite Fourier model. No omitted tail,
frequency separation, nonzero rotor or nonzero wedge is required. Zero
amplitude is retained in both sectors. P depends on unknown upstream
hardware and terminal speed/phase; it is not observed in isolation.

Amplitude in[a_-,a_+] is eliminated exactly in this ONE relaxation by using
a_- in the lower endpoint and a_+ in the upper endpoint. The two residual
gaps remain affine in the same four geometry/source unknowns and admit the
existing16-corner elimination. The generic compiler receives UNPOLARIZED
C=sum u*(y-F0) and applies epsilon itself; independent review prompted that
API clarification. It verifies whole-box grouped/collected bounds and original
domains. Author/independent38/50 pass, including distinct amplitude projection,
noise-boundary and corner-completeness controls. Optical correlation/source
association is still unimplemented; no actual optical cell was excluded.

REFERENCE_FACE_TERMINAL_V17 supplies the more general threshold comparison
sigma(F-y)>=sigma(F0-y)+mu(d)*sigma*(s-s0), with the increment sign certified.
Its whole-cell kernel retains14 nonlinear coordinates and the four common
affine unknowns, polynomial coefficient models with uniform remainders, and
separate per-axis RMS supports. It also handles an admissible reference
outside the output band when the candidate moves further outward. Missing
domain/enclosure premises remain unresolved. This is a compatible-subset
inequality, not an unconditional residual bound for v12 feedback.
Author/independent65/28 pass. A generic common-reference cancellation gives
20/11 against noise1 where separate row minima are nonpositive; the independent
three-row/two-axis fixture gives1/2 and729 exact product checks. Those functions
and observations are explicitly nonoptical; source declarations remain hypotheses.

TERMINAL_TUBE_V17 keeps one joint squared-error budget after scalar inverse
projection. A measurement is projected onto the closure of the restricted
attainable output interval. At an open endpoint the inverse is a unique
Lipschitz limit, not a claimed physical root. Empty and unattained-endpoint
branches remain distinct. For a compatible row,
|F-y|²>=delta²+m²|s-c|². A Fenchel dual combines all selected rows and the
existing whole-speed Bernstein rotor-disk support. Optional exact equality-
graph multipliers retain shared implicit normals without finding a center
root. Original domains and both noise budgets remain in fresh replay.

The decisive generic fixture varies15 shared nuisance coordinates and has
whole-cell lower>2.49999 against original-N200 energy budget2.42, while every
pair admits a common nuisance/rotor witness with energy<=2. Independent boxes
also survive. A separate audit fixture uses both axes, shared radicals and
nonzero projection distances. Author/independent70/47 pass. Actual optical
range/projection coverage and graph association remain unproved. X-Newton
and Jaulin's centered affine graph enclosure are credited from previously
catalogued primary full texts; they offer a proposal mechanism, not optical
sharpness or a global-cost guarantee.

FIFTEEN_DEGREE_ORDER_HULL_V17 extends qualified half-margins to the order
rectangles bracketed by ACTUAL comparable anchors. Direction ordering and
forward vertical positivity are proved before response derivatives are used.
Inherited rows supply directed costs with response floor d/3125; incomparable
anchors and arbitrary convex hulls do not. Factored rotor gaps preserve the
shared products d*|A_j|. Exact nonnegative dual weights give joint product
bounds, and a distance upper bound only with a proved positive weighted
amplitude floor. Author/independent65/49 pass, including729 separate generic
dual comparisons. These are necessary conditions on a common compatible
subset; no actual source association or optical exclusion is claimed.

A useful limitation is also proved: for any magnitude cap r>0, all six
rotor coordinates, three glasses, distance and gap retain their full native
projection when the beams are normal and three NONZERO wedges obey
|A_j|<=min(1/16,r/60000). The frozen seven-degree caps give
|F-p|<=149*d*max|A|<r/2 uniformly in time. Suitable native sources make
|F|<r. Thus arbitrarily many small-magnitude bounds alone cannot confine
those eleven coordinates. This credits earlier weak-wedge work and concerns
magnitude-only information, not compatibility with arbitrary actual values.

Snapshot terminal_shared_cell_manifest_v17.json binds eight reports with412
checks. All v1-v16 sources, reports, cover and driver are preserved; only
DIARY.md, HANDOFF.md and paper/FINISH_LINE.md advance among living files.
No optical or physical-rotor value, observation, derivative, contractor,
forward or inverse computation was performed. These are new necessary
constraints and generic whole-cell certificates. Blind full18 reconstruction,
useful actual optical confinement, integration and runtime remain open.
T1--T10 and the archive recovery ledger are unchanged.

### 2026-09-27: v18 reverse-flattening references and original-source constraints

The work remains free symbolic algebra, exact scalar bounds, unrelated
generic controls, and actual-source symbolic compilation. No optical state,
record, observation (including zero), rotor value, expression enclosure,
derivative box, forward or inverse calculation was executed. Frozen v1-v17
files and the cover driver are unchanged. Only this log, HANDOFF.md and
paper/FINISH_LINE.md advance among existing living files.

REVERSE_FLATTENING_REFERENCE_CONE_V18 proves that a qualified15-degree row
can flatten faces in order3,2,1 while retaining every half-margin. At each
stage the current incoming direction is fixed, q stays between its original
value and a, outgoing positivity is established before differentiation, and
all later flat plates preserve the resulting safe direction. Integrating
the three scalar responses gives the exact telescope

    F-B0=sum_j gamma_j*A_j*u_j,
    d/3125<=gamma_j<=d*(11/2,11/2,6)_j,
    B0=p+(6+2g+d)*t0+3*a0*sum_j(1/sqrt(n_j²-a0²)).

B0 is one common unknown baseline per axis across time. The safe staircase
does not imply any arbitrary mixed-sign candidate path is transmitted.
Gain endpoints are outer relaxations, not freely realizable optical hardware.
The exact upper envelopes preserve downstream flat-plate identities and
shared geometry. Delta<1 follows from a positive difference-of-squares
margin147/1280. The native geometry maxima are5801749/1075200,
46320871/8601600 and5269415/917504, below11/2,11/2,6 respectively.

For a row directly qualified by the v16 original-output gate, this sharper
Delta bound gives |F0|/d<10079/9156<9/8 for the terminal-only reference.
The terminal response lower bound improves to8337d/88000, a factor2779/391
over v17, with upper6d. In the common-baseline telescope only the third lower
coefficient gets this strengthening; rows known only through half-margins or
order-hull inheritance retain1/3125 throughout. Author/independent53/36 pass,
with125 generic polynomial checks and216 separate gain-corner comparisons.

FLAT_TERMINAL_REFERENCE_SOURCE_V18 authenticates the full original reduced
graph and universal extension, then substitutes a3=0 only inside an external
reference copy. Original candidate bounds and coupled restrictions are not
flattened. The zero reference need not lie in a narrowed candidate cell.
All reference outputs exclude h3,p3,a3 ancestors and retain n3. Safe exact
constant folding preserves every original and substituted operand domain.
Full200 compilation retains18 inputs,400 references and161,154 nodes.
Author/independent64/63 pass, including positive/negative nonzero candidate
cells, changed equations/priors/clock/rows, hidden-domain failures and fresh
typed replay. Source association is proved symbolically, not by optical values.

TERMINAL_HARMONIC_SOURCE_V18 closes the v17 generic-correlation interface gap.
It reconstructs the unpolarized correlation sum u*(y-F0) from the authenticated
reference and original unit-rotor clock; y remains formal. The same18 physical
inputs and400 formal data inputs produce165,346 nodes. The new sharp kernel
uses8337/88000 and6 with the frozen generic corner/collection logic.
The future numerical wrapper would check all wedge tangents against2-sqrt3,
enclose |a3|/sqrt(1+a3²) as a sine amplitude in one closed sign sector, and
verify each actual selected row's original-N RMS output-margin gate. A box
crossing signs needs both intersections; zero survives both. The unpolarized
correlation is signed exactly once. All16 shared affine corners remain.
Author/independent64/51 pass, including405 distinct generic product checks.
The actual numerical wrapper was reviewed statically and never called.

SHARED_FLAT_BASELINE_V18 derives the exact finite row-cone RMS profile

    min_B sum_k dist(B,[y_k-u_k,y_k-ell_k])².

The rational breakpoint/active-mean search includes all minimizers, also
for bounded baseline intervals and flat minima. Its residual dual attains
the exact squared distance. Completeness is for the stated finite convex
relaxation only. Fixed signed duals combine whole-cell shared features with
two separately charged original-axis RMS supports. Zero total weight on an
axis cancels that axis's baseline before enclosure. Every selected row must
qualify under the same data and noise contract. Inherited qualification is
not silently integrated into the implemented future source API.

The compact actual-source compiler binds the baseline and d*A*u features to
the original18 coordinates, physical order and clock in7,912 nodes with1,202
outputs. It authenticates the optical carrier but omits its value prefix in
the smaller necessary-condition DAG. That is sound for excluding compatible
physical points; it is not a physical-domain or existence check. Its numerical
source API remains unexecuted. A generic200-row fixture has minimum energy450
against original budget200 while every pair has minimum at most9/2. The
fixture does not instantiate optical values, rotor features or observations.
Author/independent52/57 pass; the independent audit proves324 generic exact
projection witnesses with9,072 complete convex-support vertex comparisons.

COMMON_BASELINE_AFFINE_RESEARCH_V18 retrieves and checks Boyd/Vandenberghe,
Convex Optimization(2004), author full text, Sections3.1.1/3.2.3 and8.1.3.
Its support/separation statements guide a source-specific simplification:
because each feature is d*xi(q) and native d>0, positive homogeneity moves
the SAME distance outside every max gain term. The actual support is exactly
beta(q)+d*c_d(q)+gap*c_g(q)+w_x*p_x+w_y*p_y. Its maximum over the full four-Z
box equals the maximum of16 complete corner expressions, each retaining
common q across all coefficients and rows. Coupled physical restrictions
make the product-box result an outer bound. Fixed duals, source/row binding,
original RMS budgets and every row's qualification are still mandatory.
A negative distance, shifted generic feature, missing corner or arbitrary
choice of d_max breaks the corresponding unsupported simplification.
The exact nonoptical example improves a natural interval upper11 to7.
Author26 checks plus36 rational homogeneity comparisons pass. The retrieved
full text, metadata and source hashes are retained. The specialized source
corner compiler is not implemented; useful whole-q optical bounds remain
open. This credits classical convex analysis without claiming a new generic
duality theorem or a demonstrated optical performance increase.
The independent research audit passes25 checks plus100 distinct exact
raw/factored comparisons,20 barycentric comparisons and9 singleton cases.
Snapshot shared_reference_source_manifest_v18.json binds ten author/audit
reports with491 checks, current source hashes, the primary full text and
seventeen preserved prior snapshots. It retains the single documented
historical note revision and the unchanged frozen cover/driver. These are
control counts, not optical recoveries or a measure of distance to completion.

These are wedge-size results through conditional15degrees, not optimizer
initialization radii or blind full18 robustness claims. Useful actual optical
confinement, physical reconstruction, integration and computational cost
remain open. T1--T10 and the archive recovery ledger are unchanged.

### 2026-09-27: v19 stronger physical gains and shared rotor-product support

The preceding goal turn was progress: v18 froze source-bound constraints,
stronger conditional15-degree references and an exact affine research route.
This turn implements that route, strengthens its physical gains materially,
and derives a further necessary test with only six rotor dependencies.
The full18 objective is unchanged. Work remains free algebra, scalar envelope
arithmetic, unrelated generic controls and actual-source symbolic compilation.
No optical/rotor values, observations, derivative boxes, forward/inverse runs,
source numerical certificates or campaigns were executed. Frozen v1-v18 and
the existing cover/driver are immutable; only the three living logs advance.

REVERSE_FLATTENING_GAIN_TIGHTENING_V19 uses the exact later-flat formula

    F_sj=K_j*(d+(3-j)g-Q_j*C_j)+f_sj*sum_(r>j)J_r.

Delta>=n-1>=3/10 and c,R,v<=1 give K>=3/10. Keeping |q|/R<=1 yields
f_s>7/32, and each later flat plate has J>=3/n>48/29. Crucially, the
complete contact brackets are proved positive before multiplying lower
factors: native minima23943/512,158925/4096,115151/32768. Shared geometry
and signed-source adversity remain in the q_j(g) bounds. Exact minimization
gives floors2190561/7424000,2851311/11878400,345453/16384000. Convenient
closed lower gains are(59/200,6/25,21/1000)*d, against prior d/3125.
Directly output-qualified third gains retain8337d/88000 from v18.

These are staircase derivative/integral bounds with later faces flat. They
do not improve the v15 arbitrary unflattened response or its inverse modulus.
An optional formula retains unknown glass and geometry correlations; no
true hardware is supplied. Author/independent43/27 pass. The independent
proof uses positive affine decompositions over the complete native geometry
rectangle. Its initial structural Sympy equality caused a false test failure;
the exact zero-polynomial comparison fixed that test without changing any
physical bound. Generic gain-box vertex checks confirm support nesting.

AFFINE_BASELINE_SUPPORT_V19 implements the v18 positive-distance reduction
H=beta(q)+d*c_d(q)+gap*c_g(q)+w_x*p_x+w_y*p_y. It structurally verifies that
normalized features xi=A*u, beam tangents and plate terms are independent
of all four affine inputs. Fresh whole-box xi bounds justify any selected
max branch; otherwise max remains. Grouped and collected full expressions
each need all16 corner upper enclosures. An incomplete corner budget cannot
claim a complete variant, but a separate complete variant remains usable.
All original operation domains and same-cell correlations are retained.

The source adapter authenticates the original model and retains the entire
161,154-node reference carrier before adding xi, beam and plate outputs.
Full200 source compilation has168,976 nodes,400 rows and18 inputs. The
future numerical wrapper would check exact15-degree membership, same-row
output qualification, original-N separate-axis RMS budgets and source replay.
It was not called. The unrelated generic support upper improves from natural
11 to verified7; independent exact product comparisons number162. Author61
passes; optical usefulness and runtime remain unmeasured.

SHARED_SCENARIO_DUAL_V19 replaces hand-picked row weights with one rational
proposal shared by a whole finite scenario family. A sparse LP uses support
epigraphs, optional exact axiswise mass constraints and conservative
eta_a*sqrt(N*m_a)*||lambda_a||infinity proposal penalties. Solver status,
objective and support variables are untrusted. Rationalized weights have
their masses restored exactly and are replayed using actual finite supports
and the sharper exact Euclidean per-axis noise supports. None of this proves
continuous scenario coverage or exact optimality of the rationalized proposal.

The32-scenario generic fixture has two nuisance cases times16 common-affine
corners. The automatic shared dual succeeds although either single-case
profile dual fails the other case. Its unchanged weights pass the new full
cell gate on an affine family. A nonlinear family with the same endpoint
scenarios contains an interior fit; the complete gate correctly returns
unresolved. This is an explicit test of the required proposal-to-verification
boundary. Author52 passes; no optical scenario or target was instantiated.

ROTOR_PRODUCT_SUPPORT_V19 takes one dual with exactly zero mass on EACH axis.
The common baseline cancels identically, leaving on a closed wedge sector

    H=sum_j r_j*c_j(h,p),  r_j=d*abs(sin ax_j)>=0,
    c_j=sum_ka max(L_j*lambda_ka*eps_j*u_kaj,
                   U_j*lambda_ka*eps_j*u_kaj).

The common-product domain is exactly the polytope
d_-<=d<=d_+, A_j^-*d<=r_j<=A_j^+*d, with16 endpoint vertices. Thus
amplitudes and distance can be removed from the selected support through
complete corner maximization; glass, beam, gap and source have already
canceled or been conservatively covered by the gain intervals. Only the six
speed/phase coordinates remain as selected expression dependencies. This
projects a necessary constraint, not the entire full18 inverse. Existing
September18 rotor-only tests, v11/v16 quantitative cones and v17 common
products are explicitly credited; there is no first-six-dimensional claim.

Original tangent endpoints are mapped to sine magnitudes by symbolic positive
radicals. Eight closed sign sectors retain zeros in both signs and preserve
collisions. One sector's exclusion does not exclude a box crossing signs.
All18 source inputs, candidate intervals, original clock and full operation
prefix remain; extra physical constraints are still obligations. The future
actual-source numerical API requires direct output qualification and original
axis budgets, and remains unexecuted. Full200 symbolic compilation has167,781
nodes and400 rows; every selected coefficient/corner ancestry is checked to
be contained in the six rotor names. Author45 plus2,187 unrelated exact
product comparisons pass. A generic common-distance upper0 improves the
independent-product upper8; true supremum-2 shows the bound remains conservative.

Independent affine-gate audit68 passes with signed/two-axis fixtures, corner
and max-branch scope checks, fresh source reconstruction and static review
of the unexecuted numerical wrapper. An initial audit test incorrectly
expected a complete grouped interval bound to exclude; it was corrected to
check its valid-upper contract, with no implementation or theorem change.
Independent selector audit45 passes with2,048 exact box-vertex comparisons,
an independently constructed feasible LP vector and a distinct two-axis
hidden-interior-fit example. Independent rotor-product audit53 enumerates
the active-constraint polytope vertices, checks125 signed supports and
reconstructs all three source coefficients plus six positive radical endpoints.
No author correctness defect was found. All report/source bindings are current.

Snapshot shared_rotor_support_manifest_v19.json binds eight reports with394
controls and preserves all eighteen prior snapshots, the documented historical
note revision, and the frozen cover/driver. Source programs, generic proofs,
physical gain scope and typed zero-execution counters remain separately
recorded. Control counts do not measure solved scans or progress percentages.

Useful actual optical confinement, certified physical reconstruction,
integration and global runtime remain open. T1--T10 and the archive recovery
ledger are unchanged; the full18 goal remains active.

### 2026-09-27: v20 coupled row profiles and local fifteen-degree transport

The preceding goal turn was progress: v19 was exclusively frozen and its
hashes/scopes independently replayed. The next action changed because the
row-independent gain cone admits false fits even at stationary rotor states.
This turn addresses that concrete relaxation defect while preserving the
full18 compatible-set/reconstruction objective. Frozen v1-v19 and the frozen
cover/driver are unchanged. Work remains symbolic identities, exact scalar
envelopes, unrelated generic numerical controls and actual-source symbolic
compilation only. No optical or physical-rotor evaluation, observation,
forward/inverse call, numerical source certificate or campaign ran.

ROW_COUPLING_V20 proves an explicit local mixed-row comparison through15degrees.
One original half-margin anchor and effective-sine differences delta satisfying
8|delta1|+4|delta2|+2|delta3|<=3/20 force the whole straight segment to remain
inside quarter transmission margins. A first-loss argument uses the triangular
q sensitivity rows(2,0,0),(4,2,0),(8,4,2), and the strict reserve
3/4-(71/100+3/20)^2=13/1250. This does not assume that arbitrary half-margin
endpoints have a transmitted mixed segment.

Original unflattened absolute derivative caps on that tube are
|F_sj|/d<(2011,890,196). They use the frozen global M<11/8 theorem, explicit
native position envelopes and the full adjoint chain; the v19 later-flat
lower bounds are not substituted for unflattened derivatives. Constants are
large and have no measured optical usefulness. Their relevant property is
that the resulting difference cost tends to zero with delta. Exact identical
effective states give identical same-axis outputs on admitted original
branches without an angle/margin premise. Zero wedges remain, without division.

The source compiler retains all18 inputs, original output indices, clock and
operation prefix. It forms signed differences as A times the base rotor times
(step^lag-1), with a cancellation-preserving recurrence rather than subtracting
independent rotor boxes. The numerical source wrapper is implemented but
unexecuted; its near-row qualification and native15 guards remain required.
Generic flows and coupled profiles combine row weights before charging the
original separate-axis RMS budgets.

COUPLED_ROTOR_PROFILE_V20 retains one common (d,r1,r2,r3), both baseline
offsets, all fitted rows, their gain intervals and optional proved row bands.
The declared fixed-feature relaxation is polyhedral before the two RMS balls.
Its strict rational separation-certificate family is complete, using closed
polyhedral projection plus compact-noise separation and attaining LP support
duals. This handles zero noise, empty or lower-dimensional domains without
asserting SOCP Slater feasibility. Completeness concerns this relaxation only;
the implemented conservative LP is an untrusted proposal, not a complete
numerical solver. Exact primal witnesses certify only relaxation compatibility.

Six unrelated generic rows demonstrate the benefit: every duplicate-pair
constraint alone fits the same original noise budget, but the three together
do not. The exact excess profile is1/6; the dual excludes below that threshold
and preserves equality. A separate fixture requires conflicting products that
independent product intervals allow but one common distance forbids.

SHARED_PRODUCT_CONTRACTION_V20 intersects multiple necessary contrasts on the
same16-vertex product domain. Exact residual support repairs imperfect proposed
LP multipliers instead of trusting solver status/objective or demanding exact
bounded-coordinate cancellation. Generic cuts individually feasible prove a
joint contradiction0<=-1; an independent-product correction would give+2.
Another fixture yields d[3,6],r1=3,r2=2,r3[0,3]. Combined original row weights
also tighten a generic lower distance to9/4 while the individual cuts do not.
These are mathematical consequences of caller-owned cuts, not certified
optical coefficients or actual geometry recovery.

COUPLED_PROFILE_TRANSPORT_V20 transports one exact nonnegative dual across a
whole nonlinear cell. Constant row/baseline blocks stay exact; every unbounded
baseline coefficient must cancel. Varying product coefficients and right sides
remain on one shared DAG, with all16 complete product-corner expressions and
all original operation domains retained. One original RMS support is charged
after combination. Generic controls retain a nonlinear endpoint fit that an
anchor-only certificate excludes, while a smaller whole-cell variation leaves
a genuine strict gap. A conditional finite-cover theorem requires a strict
separation gap at every point of the compact region; it does not establish
that gap, a useful covering count or optical confinement.

Primary Boyd/Vandenberghe full text was checked at4.4.2(p156),5.2.3(pp226-227)
and8.1.2(pp399-400), complementing v18's support-duality read. A focused search
found Jansson2004, Rigorous Lower and Upper Bounds in Linear Programming,
DOI10.1137/S1052623402416839. Publisher-deposited abstract/metadata were saved
with a read-scope record; the publisher full-text request returned403. No
theorem from that unread text was invoked. Bounded residual correction is
proved directly; unbounded baseline cancellation remains explicit.

Independent row audit found a real serialization defect: tuple-keyed cost
maps could not pass the strict certificate digest. The author fixed generic
and future-source storage to deterministic typed tuple entries, and fresh
replay now passes. An initial audit input also needed exact division instead
of a floating zero. Neither issue changed the physical theorem. Other
independent audits found no author defect. The product audit initially expected
an exactly attained rational from a rounded proposal; it was corrected to the
verified outward allowance, with the exact witness checked separately.

Final author/independent controls pass80/51 for physical row coupling,56/50
for the fixed-feature profile,51/40 for joint product contraction, and33/40
for whole-cell dual transport. The profile audit checks a distinct two-axis
excess boundary3/2 and2,304 product/row supports. The transport audit directly
converts the actual profile inequality blocks into DAG coefficients and
replays identical row weights, product support and noise; nonlinear widening
correctly preserves an unresolved endpoint fit. It includes144 independently
expanded corner comparisons. These are meaningful controls, not recoveries.

Snapshot coupled_row_profile_manifest_v20.json binds eight reports with401
checks, all new source/note/research artifacts and the three living logs,
while preserving nineteen earlier snapshots, the documented historical-note
revision and the frozen cover/driver. Source-only compilation, generic
relaxation completeness, local physical margins and unproved global gap are
recorded separately. No numerical optical exclusion or recovery is claimed.

Useful actual optical confinement, strict physical reconstruction, global
runtime and full solver integration remain open. T1--T10 labels and the
historical recovery ledger are unchanged. The full18 goal remains active.

### 2026-09-27: v21 convex quarter margins and output-conditioned row transport

The preceding goal turn was progress: v20 was exclusively frozen and its
source/report bindings independently replayed. This turn targets its large
local derivative constants rather than treating another generic certificate
as actual confinement. Three agents and root worked in parallel on physical
response estimates, finite-path comparison, independent audits and primary
research. Frozen v1-v20 and the frozen cover/driver are unchanged. Execution
remains free identities, exact scalar envelopes, unrelated generic numerical
controls and original-source symbolic compilation only. No optical state,
rotor, observation, forward/inverse call or numerical source certificate ran.

QUARTER_MARGIN_TRANSPORT_V21 reuses the v15 convex direction identity
b=(w*a+Delta*q)/(w+Delta), with both weights positive. The new application
propagates |a|,|b|<sqrt3/2 through all three quarter-margin exits, giving
v>1/2 rather than v20's independent lower bound1/4. Positivity is established
first from the older elementary vertical floor, so the proof is not circular.
Correlated glass ratios give B,T>24/25,1/M<9/5,Jglass<6,|C|<7/8 and
|D|<31/100. The complete native position/adjoint chain then gives original
unflattened |F_sj|/d<(93,50,24), versus v20's(2011,890,196).

The first-loss argument also enlarges the sufficient comparison tube.
One original half-margin anchor and weighted effective-sine difference
(15/4)|delta1|+(25/12)|delta2|+(19/10)|delta3|<=5/32 keep the full segment
inside quarter transmission. The rational reserve is23/9216. This contains
the old sufficient tube and does not assert convexity for arbitrary
half-margin endpoints. All unknown glass, geometry, source and beam ranges
remain. The source wrapper retains the frozen lag-factorized DAG, all18
inputs, clock, rows and constraints; its numerical API is unexecuted.
Author51/independent60 controls pass, including independently checked scalar
contact polynomials, a distinct shared-noise chain and symbolic source replay.

OUTPUT_CONDITIONED_RESPONSE_V21 eliminates contact positions before applying
absolute bounds. The exact backward recurrence yields F_sj=L_j*F+H_j,
|L_j|<(16,21/2,13/2), |H_j|/d<(70,75/2,13). H_j has no source-position
term. Source-normalized and contact identities are explicitly credited to
v15/v16; the new work is the all3 quantitative recurrence after elimination.
An independent tangent differentiation verifies the identities without
assuming the author's pre-expanded derivatives. Exact native envelopes
and common geometry support the coefficients. Author79/independent41 pass.
One draft control used a strict comparison where two rational envelope
values are equal; it was corrected. The physical strict premise and theorem
were unchanged. These are forward upper bounds, not inverse lower moduli.

OUTPUT_CONDITIONED_ROW_TRANSPORT_V21 proves directly that |f'|<=a|f|+b
implies |f1-f0|<=tanh(a/2)(|f0|+|f1|)+(2/a)tanh(a/2)*b, with the a=0
limit handled separately. The intrinsic integral coordinate covers sign
crossings as well as same-sign paths. The rational weaker coefficients
(a/2,1) also suffice. Applied on a separately proved connecting tube, two
endpoint bounds |F|<=rho*d give finite response caps B+rho*A without any
interior output-band assumption. At rho=1 they are(86,48,39/2); at1/14
they are(498/7,153/4,377/28), all below the new unconditional caps.

The generic implementation encloses the exponential with exact rational
series/remainder bounds and retains its valid rational fallback on resource
limits. Necessary endpoint intervals come from the original RMS ball;
absolute-value chords yield constant output-row coefficients. The composed
v20 gate encloses all16 complete product corners over a whole nonlinear
cell and charges the original separate-axis RMS support once after row
combination. Nonnegative forcing and all original operation domains remain
obligations. A hidden interior fit survives even when both outer points
exclude; the independent audit also retains an exact noise boundary and
combines a three-row chain whose individual edges cannot exclude. This
module is not a new numerical physical-source adapter.

The focused optical literature read found Luo et al., Sensors26(7):2013
(2026), DOI10.3390/s26072013. The original openly licensed article XML was
read in the stated model/calibration/experiment sections and independently
checked. It fixes two11.35-degree prisms, index1.515 and camera geometry,
then estimates only two orientation offsets against54 checkerboard corners
located by external stereo geometry. Its phase-offset results are not
wedge-angle errors or a blind18 robustness theorem. Li,Liu,Sun2017,
DOI10.1364/OE.25.007677, was read only at the published abstract/bibliography
level because publisher full text was blocked. It describes three-prism
pointing inversion with damped least squares and a conditioned motion law,
not a verified unknown-hardware finite-record recovery theorem. The useful
measurement/calibration principles and exact read scopes are saved; no
unread theorem or unavailable calibration information is imported.

Bihari1956, DOI10.1007/BF02022967, was retrieved from the original Hungarian
Academy journal archive. Printed81 and83-85 were inspected, including a
visual check of the theorem/proof on83-84; the entire retained article is
not claimed read. The reciprocal integral and inverse-domain condition
support the scalar-comparison route. The signed symmetric endpoint bound
is proved independently: Bihari's nonnegative-half-line hypothesis is not
silently applied to a signed function. All source artifacts and read scopes
are hash-bound. A possible future improvement retains the positive product
term for opposite-sign endpoints, but no such optical certificate is claimed.

On the new admitted tube, the output-affine comparison rate is at most63/80.
At most25 Taylor terms prove an exponential width below2^-80. A fixed
declared rate63/80 also permits rational coefficients(3/8,20/21), since
exp(63/80)<11/5. The forcing coefficient20/21 is not uniform over smaller
chosen rates, whose zero-rate limit is1. These control scalar arithmetic;
they provide no global covering-count or optical runtime theorem.

For the user's angle question, the existing seven-degree claims must be
kept distinct. Full native +/-7-degree wedge transmission is proved over
all rotor states and native beam/glass ranges. Conditional joint recovery
of three wedges and two source coordinates is proved with the other13
coordinates supplied and finite rotor sign coverage; a nominal design has
a constructive oracle theorem. These do not establish blind full18 recovery
or a5-7degree initialization basin. The v21 fifteen-degree results require
their admitted margins and connecting path, and do not change that scope.

Final author/independent controls pass51/60 for quarter transport,79/41 for
output-affine response, and57/46 for endpoint transport,334 checks in total.
Snapshot output_conditioned_transport_manifest_v21.json binds the six
reports, exact source/research artifacts and three living logs while
preserving all twenty prior snapshots and their documented exceptions.
Control counts measure proof/replay checks, not recovered optical systems.

Useful actual optical confinement, strict physical reconstruction, global
runtime, full solver integration and the historical recovery ledger remain
open. T1--T10 completion labels are unchanged. The full18 goal remains active.

### 2026-09-27: v22 global half-margin row comparison and a relaxation obstruction

The preceding v21 turn was verified progress: its exclusive snapshot and
all source/report bindings were independently replayed. This turn used
three agents plus root for separate physical proofs, graph constraints,
limitation analysis and independent audits. Frozen v1-v21 remain unchanged.
Only free identities, exact scalar envelopes, unrelated generic controls
and original-source symbolic compilation ran. No numerical optical state,
rotor, observation, source expression, contractor or inverse was evaluated.

GLOBAL_HALF_MARGIN_PATH_V22 removes the small-step premise for same-system
same-axis endpoint pairs whose three exit-root squares are all at least1/2.
The native faces remain at most15degrees. The conditional admissible face
interval is nonempty, has nonincreasing endpoint functions, and is
100/109-Lipschitz in the incoming direction. The positive branch and
forward direction are established before using the old convex identity.
Successive clipping of raw face-coordinate lines constructs a path staying
inside all three half-margins and fixing both original endpoints.

On a moving margin, the outgoing total derivative is v/B: the two partial
terms cancel before taking absolute bounds. This prevents spurious
amplification. The first incoming direction is monotone, which implies
the second clipped face is monotone even when clipping is active. With
delta_j the absolute endpoint face-sine differences and k=1525/1308,
TV(s1)=delta1,TV(s2)=delta2 and
TV(s3)<=max(delta3,k*(3*delta1/2+delta2)). No monotonicity of the last
coordinate or convexity of the straight segment is assumed. The path is
an analytic comparison, not a claimed physical rotor trajectory. Fixed
smaller face caps and zero wedges are retained without division.

HALF_MARGIN_RESPONSE_V22 separately tightens the original unflattened
response on this domain. Convex direction propagation yields v>70/99,
B,T>109/100,1/M<3/2,b_s<61/48,Jglass<4,|C|<5/6,|D|<2/7.
The complete native position and adjoint envelopes give
|F_sj|/d<(18,11,6). Integrating these along the new path gives the global
max cost18*delta1+11*delta2+6*max(delta3,k*(3*delta1/2+delta2)),
and its linear upper(57/2)*delta1+18*delta2+6*delta3. The exact rounding
reserves are3/436 and1/218. Both endpoints must qualify. A single half
anchor alone still uses the older local tube. None of these forward
upper bounds is an inverse lower modulus or an angle-error basin.

The new row/source module retains original rows, clock, all18 inputs and
the complete operation prefix. Generic flow certificates recompute the
global cost and charge the original RMS support after row combination.
Source-only replay checks a five-sample pair DAG with2979 nodes. The
separate phase-free source has4566 nodes: squared complex lag recurrences
bind return expressions to individual speed coordinates only, and repeated
lags share expressions. Wedge amplitudes have individual wedge ancestry.
Both axes for every selected sample, candidate restrictions and fresh
source fingerprints remain bound. Numerical wrappers are implemented and
statically reviewed but not executed. Independent symbolic source fixtures
use a distinct four-sample graph and retain negative-wedge and additional
equality restrictions. No optical observations were instantiated.

PHASE_FREE_GRAPH_ENERGY_V22 combines the two axes through the exact
identity DeltaCos^2+DeltaSin^2=4*sin(pi*N*tau)^2. Each axis first uses
its valid physical row bound; no equality of x/y optical derivatives is
assumed. Weighted incidence and a verified Laplacian bound Lambda give
one noise term sqrt(N*Lambda*(eta_x^2+eta_y^2)). The physical graph
term retains the common distance/wedge-product polytope across all edges,
with exact16-vertex support. Strict radical comparison handles equality
correctly. Both axes at every endpoint must satisfy the half-margin gate.
Generic graph proofs alone explicitly lack source/qualification provenance;
the separate future source wrapper supplies those obligations.

An equal-lag matching with100 edges and originalN200 has paired noise
allowance2*sqrt(eta_x^2+eta_y^2). Conditionally on paired RMS increment
at least1 and each eta<=1e-5, the native product bounds exclude simultaneous
returns within1/20000 turns of integers: the physical allowance is429/500
and noise below3/100000. At lag5 this is a1e-5Hz radius about return
frequencies. This is an exact thin-band consequence of the theorem,
not measured optical confinement or general speed localization. Earlier
September17 return constraints are credited; the new ingredient combines
global qualified-row comparison, paired geometry and shared graph error.

GAIN_CONE_SHADOW_LIMITATION_V22 proves a structural obstruction for the
covered universal-gain/row-cost relaxation. Its constraints factor through
six rotor coordinates, three signed amplitudes, distance and two exact
all-flat baselines. Fixing those12 quantities leaves6 independent local
directions: vary the three indices, gap and two beam slopes and compensate
the source positions to hold both baselines fixed. At an admitted strict
native-interior point the full local fiber remains admissible. Every
covered relaxed fit persists for arbitrary data, including zero noise,
nonzero wedges and independent speeds. This is not a physical ambiguity
theorem, nor a statement that interval enclosure widths are identical.
Additional original output/residual equations can and must break the fiber.

A particular constant-gain affine shadow has a separate7-dimensional
fiber when distance and amplitudes vary at fixed products. That stronger
dimension is not asserted universally for every relaxed fit. The earlier
calibrated terminal linear/cubic response identities exhibit hardware
information discarded by independent gains; they are explicitly credited
to September18/22. Arbitrary moving scans do not supply the required
fixed-upstream Taylor slice automatically. Retaining an original DAG
prefix authenticates domains but does not impose F=y on relaxed rows.
Consequently the next recovery step must retain shared nonlinear
hardware-dependent response or original residual relations; adding only
uniform response envelopes cannot resolve these six directions.

Final author/independent controls pass94/37 for the global path,64/64 for
half response and source transport,50/45 for graph energy and47/35 for
the limitation,436 controls total. The path audit independently derives
the cancellation by inverse rotation and checks234 unrelated exact scalar
clamp pairs, including caps, time reversals and a third-coordinate
backtracking example. The graph audit uses a distinct weighted cycle and
isolated row with originalN25, exact PSD principal minors, all16 product
vertices and an attained two-axis noise boundary. These are proof and
implementation controls, not recovered systems. No defects were found in
the final independent audits.

Snapshot global_half_response_manifest_v22.json binds all eight reports,
new sources and three living logs while preserving all21 prior snapshots,
their documented historical exception, and the frozen cover/driver. The
v21 primary literature and exact read scopes remain preserved. No new
optical performance or full18 reconstruction is claimed. The user's
seven-degree transmission and conditional-five-unknown inversion results
retain their earlier qualifications; a5-7degree blind initialization
tolerance remains unproved. T1--T10 labels and the archive ledger are
unchanged. The full18 goal remains active.

### 2026-09-27: v23 shared-hardware RMS exclusion and exact finite secants

The preceding v22 snapshot was independently replayed. This turn used three
agents plus root for separate proofs, exact generic controls and independent
audits. Frozen v1--v22 are unchanged. Work stayed within free identities,
exact scalar envelopes, explicitly unrelated generic controls and actual
source symbolic compilation. No numerical optical state, rotor, record,
observation, forward map, derivative, contractor or inverse was evaluated.

AFFINE_RMS_SOURCE_V23 retains F_i(q,z)=beta_i(q)+A_i(q)z with one common
four-coordinate affine box and all fourteen nonlinear coordinates unknown.
Normalize z=m+Rx with x in [-1,1]^4 and form C=AR, b=y-beta-Am. For fixed
nonnegative axis weights summing to one, G=C^TWC, h=C^TWb, c=b^TWb and the
allowed weighted energy is B=N*(lambda_x*eta_x^2+lambda_y*eta_y^2). N is
the original sample count even when selected rows are sparse. No independent
copy of either axis's error budget is allocated to each row.

For positive delta and fixed nonnegative box multipliers, H=G+delta*I is
positive definite even when the original design is singular. Put
v=h+(mu_minus-mu_plus)/2. The original box residual has lower bound
c-sum(mu_plus+mu_minus)-4*delta-v^T*H^-1*v. The complete 4*delta correction
is essential: the regularized objective alone can falsely exclude a perfect
original fit. The determinant of the block matrix with H, v and lower-right
entry c-sum(mu)-4*delta-B equals det(H) times the certified surplus.
Its strict whole-cell positivity excludes the entire native affine fiber.
The determinant is polynomial, has fixed order five, and requires no
rank choice or division; det(H)>=delta^4. All original shared coefficient
expressions and their domain prefix are retained before optional collection.

AFFINE_RMS_PROFILE_V23 provides an exact generic box KKT oracle. It searches
at most 81 faces and accepts only an independently replayable KKT witness.
A minimum with a singular free block can move along a null direction to a
smaller face, so only positive-definite free blocks need solving. A resource
limit without a witness remains unresolved. Nonnegative axis weights form
a complete strict exclusion family for the closed convex upper image of
the two residual energies over the compact box. A finite tested schedule
is not complete. Positive ridge changes the minimum by at most 4*delta;
rational weights, sufficiently small ridge and rational approximations of
optimal box multipliers retain any strict fixed-fiber gap. Continuity then
gives a neighborhood for a fixed strict certificate. It supplies neither
the missing global separation gap nor a useful fourteen-dimensional cover.

The RMS source compiler reconstructs the frozen actual coefficient source,
original clock, prism order, selected rows, full affine box, restrictions
and fourteen nonlinear inputs. Every target and both noise allowances stay
formal. The full 200-sample source has 119,064 nodes, including the unchanged
94,750-node coefficient prefix, 400 formal target inputs and 416 total input
names. Numerical wrappers are implemented and statically audited but never
called. The optional data-radius mode checks |y_i|+sqrt(N)*eta_axis<=R in a
future numerical call before using its saved extension theorem. Universal
extension retains its separate transmitted-point agreement scope. Symbolic
provenance does not prove physical admission or an optical exclusion.

The source author/audit pass 67/53 controls. Distinct generic examples show
strict whole-cell gaps recovered through cross-row polynomial cancellation
although independent coefficient intervals admit false fits. Other examples
have excluded endpoints and exact hidden interior fits; the whole-cell gate
correctly retains them. The independent audit reconstructs six dense Schur
determinants and checks 486 exact cube points with arbitrary nonoptimal box
multipliers. The oracle author/audit pass 50/29 controls; the independent
audit prescribes 54 dense minima with translated unequal boxes and checks a
separate asymmetric case requiring unequal axis weights. These are generic
proof/implementation controls, not optical fits or performance measurements.

FINITE_SNELL_SECANTS_V23 gives exact same-hardware endpoint differences on
the native fifteen-degree region when all three exit-root squares are at
least 1/2 at both endpoints. Symmetric root/product identities yield
delta_b=U*delta_a+V*delta_s with 29/100<U<5/3 and 3/80<V<9/5.
The endpoint bounds use their joint mean corrections; no intermediate path,
wedge division or unchanged upstream state is assumed. Positivity concerns
angular secants, not the full workpiece-position response.

The exact gain quotient and triangular position recurrence retain glass,
distance, repeated gap, beam and source. In the affine-column mean update,
delta_M*delta_(x+h*e0)/4 cannot be dropped. A common four-coordinate vector
serves all rows. Coincident-state differences vanish algebraically. This is
a mathematical recurrence, not a newly implemented optical source compiler.
The original RMS support is charged only after incidence weights combine
on original rows. Independent free identities expose why common hardware
is required rather than silently applying the formula across glass changes.

A separate formal flat-face calculation differentiates response while
compensating source position to preserve the exact baseline. With all other
material and beam coordinates fixed, the own-index derivative exceeds
(156145/184320)*d>4*d/5 at native points. Gap derivatives are positive for
the first two stages. The local compensated source must remain native.
This exhibits hardware dependence discarded by the v22 relaxation, but
does not supply observed jets or simultaneous finite-record identification.
Earlier normal-beam and calibrated terminal responses are explicitly
credited. Secant author/audit pass 65/49 controls, including distinct
generic affine transfers and independent reconstruction of the derivative.
An initially strict q bound was corrected to |q|<=1/sqrt(2) before the final
audit; all advertised rounded constants remain valid and unchanged.

GLASS_SOURCE_PROJECTION_OBSTRUCTION_V23 proves a precise limitation of fixed
linear source removal. On the normal-beam stratum with two flat upstream
prisms and a nonzero terminal wedge, write x=s^2. Its exact source gain is
K(n,x)=sqrt(1-n^2*x)/(sqrt(1-x)*(n*x+sqrt(1-x)*sqrt(1-n^2*x))).
Fixed row weights cancel unknown source uniformly over an open glass
interval if and only if their sum is zero in every equal-x group. A formal
series at n=1 has degree-j polynomial coefficients in z=x/(1-x), with
nonzero leading coefficients. Analytic continuation and the resulting
Vandermonde determinant prove the characterization. The proof coordinate
n=1 is not asserted to be a native observation or optical evaluation.

An analytic native clock N3=1/80, phase zero, t_k=k/20 for k=0,...,199 has
strictly ordered squared cosine and sine face coordinates for any nonzero
terminal wedge through seven degrees. Every group is a singleton on both
axes, so the only uniformly source-cancelling fixed 400-row functional has
zero weights. This is a theorem about the declared fixed projection, not
full-model ambiguity or an obstruction to candidate-dependent weights,
original cofactor relations, the new Gram test or nonlinear methods. The
two-row source leak is at least (3/10)*|delta_source|*|delta_x|; it bounds
the source contribution only, since other output terms can cancel it.
Author/audit pass 51/50 controls. The audit independently obtains a quadratic
formal recurrence and verifies the grouping and analytic-clock scope.

Primary research extends the retained Boyd--Vandenberghe Convex Optimization
text read to Sections 5.1.1--5.1.3 (printed 215--216) and 5.2.3 (226--227),
including refined Slater and dual attainment. The positive-ridge cube has
strictly feasible x=0, so those convex statements apply. The explicit ridge
correction is proved here; no stationary convergence theorem is treated as
global nonlinear recovery. A separate read-only audit checked all three
source-index artifacts and the excerpt against the frozen original text.
The new source_index.json records the exact limited read scope; no whole-book
read is claimed and earlier research artifacts remain unchanged.

Snapshot hardware_dependent_rms_manifest_v23.json binds eight reports and
414 controls, the new artifacts and three living logs, while preserving all
22 prior snapshots, their documented historical exception and the frozen
cover/driver. No new optical recoveries or useful global cost have been
established. T5 confinement, T7 computation, T8 integration and T9 validation
remain open; all T1--T10 labels and the archive ledger are unchanged. The
established seven-degree transmission and conditional five-unknown inverse
retain their qualifications. Blind full18 recovery and a 5--7-degree blind
initialization tolerance remain unproved. The full18 goal remains active.

### 2026-09-27: v24 moving RMS support, admitted curvature and affine rank structure

The preceding v23 turn was verified progress: its exclusive snapshot and
all source/report/history bindings passed an independent read-only replay.
This turn again used three agents plus root, followed by independent
cross-audits. Frozen v1--v23 remain unchanged. Only free identities, exact
scalar envelopes, unrelated generic controls and original-source symbolic
compilation ran. No numerical optical state, rotor, observation, source
expression, derivative, contractor, forward map or inverse was evaluated.

RMS_TANGENT_V24 uses the original four-affine cube and fourteen shared
nonlinear coefficient functions. For G=C^TWC,h=C^TWb,c=b^TWb and fixed
normalized anchor a, the support-plane lower is
c-a^TGa-||2(Ga-h)||_1. Its surplus subtracts only the original weighted
RMS budget N*(lambda_x*eta_x^2+lambda_y*eta_y^2). No ridge, inverse,
determinant or design-rank assumption is needed. At a cube KKT minimizer
the bound is exact. Rational anchors approximate strict fixed-fiber gaps;
one fixed anchor or a finite anchor list need not work on a whole q-cell.

The implementation collects the intercept and each of four gradients
before applying absolute-value support, retaining all shared source atoms
and the complete original operation prefix. Global collection limits keep
the complete grouped fallback. The pointwise mathematics is standard
convex duality. V23 already retains source-dependent effective duals; the
new component does not claim their first introduction or dominate that
exact determinant family.

An unrelated polynomial normal family has residual norm squared
(1+q^2)^2*(1+x^2), q in [-1,1]. Opposite nonzero residuals occur at its
two ends, placing zero in the union's convex hull. Thus no common fixed
linear row functional can strictly separate the whole family. The moving
collected tangent nevertheless has exact surplus at least 7/8 with its
stated generic RMS allowance. A different rank-three polynomial family
in the independent audit gives surplus at least 53/64 and its own exact
opposite residuals. Modified families with hidden interior fits survive.
Author/audit 49/24 pass; the audit independently constructs 27 dense KKT
minima and checks 81 additional cube points. These are algebra controls,
not optical records or recoveries.

RMS_CURVATURE_SOURCE_V24 admits a stronger correction only after verifying
original source curvature over the same whole nonlinear cell. Choose a
fixed invertible rational S and m>0; require all 32 expressions
(S^TGS)_jj-sum_(k!=j) sigma_k*(S^TGS)_jk-m, sigma_k=+/-1, to have
nonnegative lower enclosures. This proves S^TGS>=mI without assumed
off-diagonal signs. Put e=h-Ga+(mu_minus-mu_plus)/2 with fixed nonnegative
box multipliers, and box0=mu_plus.(a-1)+mu_minus.(-a-1). The lower bound is
f_q(a)+box0-||S^T e||^2/m. The strict surplus
m*(f_q(a)+box0-B)-||S^T e||^2 has coefficient degree at most four and no
regularization charge. All guards must hold; a positive surplus alone is
insufficient. The basis change belongs to the proof and does not rotate
the physical cube.

Shared collection of the 32 guards and surplus is transactional. Exact
original coefficients, row keys, box normalization, caller proposals and
both RMS allowances are freshly reconstructed. Source authentication
does not prove the proposed floor. Rank-zero or singleton-column cases
cannot pass a positive full-dimensional floor and remain unresolved for
this gate. Arbitrary nonoptimal anchors and nonnegative multipliers still
produce valid lower bounds on admitted curvature cells.

Generic examples prove an actual whole-cell improvement over tangent at
the same anchor and show dense preconditioning can make valid guards pass
when identity-coordinate guards fail. Independent controls distinguish
the correct S^T e from S e, verify 96 signed guards against a separate
matrix reconstruction and check 243 cube points. Author/audit 48/47 pass.
The full200 source has 129,791 nodes versus 119,064 for v23, so lower
polynomial degree is not advertised as a smaller carrier or faster optical
runtime. All 400 targets and two noise allowances remain formal.

RMS_CURVATURE_TRANSPORT_V24 explains the quantitative distinction. At an
unregularized KKT anchor, proved objective variation L*rho and defect
||S^Te||<=K*rho give profile lower p(q0)-L*rho-K^2*rho^2/m, provided the
same cell has the verified floor. Only the profiling correction is
quadratic; the objective can vary linearly. Optimizing the fallback ridge
loss 4*delta+K^2*rho^2/delta instead gives 4*K*rho. A fixed preconditioner
cannot manufacture a full-prior floor across native rank-degenerate strata.

A free-coordinate block provides an alternative for some singular full
Gram matrices. With active cube coordinates fixed at their signed endpoints,
the stationary free-face point may be outside the cube. A verified free
Gram floor and whole-cell correctly signed active gradients still make
its value a lower bound for every cube point by convexity. Separate free
displacement guards are required to prove cube admission and attainment.
The exact rational/Bernstein generic checker implements both sets of
guards. It covers a rank-two full design, an off-cube virtual lower point,
empty-free corners and a hidden interior active-sign switch which would
invalidate an endpoint-only argument. The hybrid active-support penalty
is proved but is not implemented as an optical source interface.

The independent audit found a real domain issue in the initial generic
face utility: original C,b entries containing 1/q could cancel from all
guards and conceal q=0. The final checker verifies each original rational
coefficient denominator before Gram/residual construction, including
zero-weight rows. Positive and negative nonzero denominator domains are
accepted; unresolved poles remain unresolved. A caller-simplified input
cannot supply lost operation history, so this remains a rational-expression
control utility rather than the optical DAG-domain interface. Final author/
audit 53/50 pass, with 405/192 distinct exact cube probes. No source theorem
or frozen optical implementation changed in this repair.

HARDWARE_WEIGHTED_AFFINE_RANK_V24 uses the actual candidate-dependent source
gains in rows (D,J,K,0) or (D,J,0,K). On axis a define
S_a=sum w*K^2, m_a=sum w*K*(D,J), R_a=sum w*(D,J)*(D,J)^T.
If S_a>0, H_a=R_a-m_a*m_a^T/S_a; if S_a=0, its cross moment vanishes
and H_a=R_a without division. The full rank is the number of positive
source masses plus rank(H_x+H_y). No zero-mass branch is discarded.

For positive S_a the exact edge identity is
H_a=(1/S_a)*sum_(i<j in a) w_i*w_j*v_ij*v_ij^T, with
v_ij=K_i*(D_j,J_j)-K_j*(D_i,J_i). The two-dimensional determinant is a
sum of weighted squared two-edge minors. Two nonparallel edges, possibly
from different axes, yield an explicit conditional lower floor from their
separation and trace bounds. The existing half-margin gain bound gives a
source-mass floor, but does not prove the required geometry separation.
Individual gains are never divided out in the implemented identities.

The source carrier appends polynomial moments, Q=S_y*(S_x R_x-m_xm_x^T)
+S_x*(S_y R_y-m_ym_y^T), and det Q. When both masses are positive,
Q=S_x*S_y*H and det G=det Q/(S_x*S_y). Thus strict whole-cell lower
bounds for the masses and det Q would prove full affine rank. The source
compiler itself supplies no such bounds. The compact moment construction
has degree-twelve determinant; it is not claimed uniformly tighter than
the other source gates. Selected two-edge minors have lower degree and
are a concrete future physical target.

Native bounded-source minimization clips each source's scalar quadratic
optimum. Two sources produce at most nine closed quadratic pieces in the
same distance/gap rectangle. Zero source mass means that source is
unidentified. Source clipping can add geometry curvature on an admitted
piece, but does not restore rank of the original design. At clipping and
singleton boundaries, the reported matrix describes a selected closed
polynomial piece rather than a unique ambient Hessian. Independent exact
partial-profile comparisons verify this convention.

The original physics also gives a precise native degeneracy. If only
prism j has a nonzero wedge, flat prisms have M=1 and preserve direction,
so J=(3-j)*T3+(j-1)*t0*K on each axis. Across both axes the affine null
direction in (d,g,p_x,p_y) is
(-(3-j),1,-(j-1)*t0x,-(j-1)*t0y). Small signed displacements at interior
native affine points preserve every original output with nonlinear
hardware fixed. This includes nonnormal beam and moving rotors on the
stated one-active-prism stratum. It is an exact original-model ambiguity,
not a relaxation shadow and not a classification of all-active systems.
Earlier zero-wedge impossibility and affine elimination are credited.

Rank author/audit 163/71 pass. The audit uses distinct scaled seven-row
systems, 75 exact Gram principal-minor comparisons, twenty independently
minimized source profiles and source-only reconstruction. No positive
rank or curvature of an instantiated optical system was computed.

Full200 source compilation preserves the 94,750-node coefficient prefix:
the tangent program has 111,185 nodes, curvature 129,791 and rank moments
103,199. Tangent and curvature keep 400 formal targets and 416 inputs;
rank moments keep fourteen inputs and introduce no observations. The
tangent's bounded full200 collection attempt falls back to its complete
grouped source and records that limit. All numeric optical wrappers remain
unexecuted; the rank module deliberately supplies none.

Primary research extends the retained Boyd--Vandenberghe text read to
Section 9.1.2, printed 459--460, equations 9.7--9.10, and Appendix A.5.5,
printed 650--651. Root independently read the retained original text at
PDF pages 473--474 and 664--665. The Schur excerpt/index is separately
hash-bound; the strong-convexity read scope is bound by the author report.
The text supports the standard quadratic lower-model and block elimination
facts. The explicit source/face contracts are derived here. No new download,
whole-book read or global nonlinear recovery theorem is claimed.

Snapshot shared_curvature_manifest_v24.json binds eight reports and 505
controls, new artifacts and three living logs, preserving all 23 prior
snapshots, their documented historical exception and the frozen cover/driver.
Useful original-data optical confinement, complete reconstruction, global
cost, integration and validation remain open. T1--T10 labels, the archive
ledger and the seven-degree guarantees are unchanged. Full18 blind recovery
and a 5--7-degree blind initialization tolerance remain unproved. The full
goal remains active.

### 2026-09-27: controlled geometry rank, robust returns and phase-free clock coverage (v25)

This continuation addresses the physical nonparallel-edge premise left
open by the v24 four-affine Schur reduction. It preserves every frozen
v1-v24 artifact and permits only free identities, scalar prior-envelope
arithmetic, unrelated generic controls and actual-source symbolic
compilation. No optical state, record, observation (including zero data),
rotor value, numerical DAG, derivative, forward or inverse was evaluated.
Three research agents and root supplied four author/audit pairs.

NORMALIZED_GEOMETRY_EDGE_V25 uses the original M=B R/(v T), so the
normalized slope is W=(b/v)/M=b T/(B R). Its free angular derivative
retains the cancellation giving

    W_s=(T-R)*(T/B)*(1+b q R/(c T))/R^3.

For strictly transmitted forward endpoints with wedge magnitudes <=15
degrees, T/B>127/200 and the final bracket>43/48 imply W_s>1/6.
A monotone q and outgoing-angle argument proves the intervening scalar
path is transmitted and forward. This does not presume convexity of the
full three-face transmission set or cover critical R=0.

On the full native seven-degree domain, the frozen branch cube gives
W_s>645/2912>1/5. A terminal row pair with its first two effective face
sines unchanged has zero normalized gap change and normalized distance
change >(64/605)|Delta s3|. A middle pair with its first face unchanged
has normalized gap change >(8/55)|Delta s2|, regardless of its terminal
face. The pairs may use different axes. The frozen gain K>3/8 gives
an original gain-weighted minor >(81/266200)|Delta s3 Delta s2|.
Nonzero increments, both source axes and positive weights imply full
four-affine rank. Repeated endpoints retain gain multiplicity. Uniform
regional increment bounds supply an explicit conservative Gram floor.
These are same-candidate coefficient comparisons, not a comparison of
different unknown nonlinear hardware. Author/audit 80/35 pass, including
180/270 independent generic principal-minor comparisons.

ROBUST_GEOMETRY_RETURNS_V25 removes the exact matching requirement using
the entire independent seven-degree effective-sine cube. Canceled
reciprocal-gain derivatives and the frozen contact bounds yield

    |partial_s U| < (306,296,307),
    |partial_s V| < (14,10,0),  U=D/K, V=J/K.

The zero is exact. All glass and beam values remain native and unknown.
Virtual hybrid endpoints retain the same hardware and stay in the proved
cube. For target changes t,u, terminal upstream errors e1,e2, and middle
first-face error eM, set

    A=(64/605)t-306 e1-296 e2,
    B=(8/55)u-14 eM, E=14 e1+10 e2, H=145/12.

Positive A,B and A B-E H give a normalized determinant lower bound;
the original weighted minor is greater than (3/8)^4 times that bound.
Both component guards are essential: a positive product of two negative
putative floors cannot prove rank. Equality, zero targets and failed
allowances remain unresolved. Optional complete middle-change bounds can
sharpen H. Author/audit 47/49 pass; the audit independently expands the
chain and checks 288 signed/interior unrelated edge matrices. The sharper
contact bound 37/256 is tied to its frozen independent proof.

PHASE_FREE_RETURN_COVERAGE_V25 connects these physical bounds to the
original 200-sample clock t_k=k/20. Additive frequency orders characterize
exact upstream returns, while paired-axis increment energy removes the
unknown phase. Exact returns are not available for arbitrary speeds, and
pigeonhole upstream recurrence alone gives no separated target rotor.
Zero amplitudes, collisions and arbitrary close returns remain explicit.

For each integer Q from 6 through 99 and independent signs, the analytic
family N=(+/-20/Q,+/-10/Q,+/-20/(3Q)) supplies the middle lag Q and terminal
lag 2Q. Some independently chosen axis has the required target change.
All native phases, indices, beam, source and geometry remain allowed.
The first wedge may vanish; the second and third target amplitudes must
be nonzero. Exact-return classification is an iff for this construction,
not for optical rank itself.

There is also an explicit thin open neighborhood: |sin ax2|,|sin ax3|>=1/10
within the seven-degree cube, speed radii 1/(200000 Q) Hz, and a supplied
per-sample clock-error bound of 1 ns. Exact scalar propagation proves a
normalized minor >1/12500 and an original gain-weighted minor >1/650000
for at least one of four axis combinations. No clock error was measured;
the supplied allowance is a theorem premise.

The fixed six rows (0,Q,2Q) on both axes, each axis weight 1/2, give a
phase-uniform analytic Gram floor by summing all four squared minors.
With Omega=1/650000, Kbar=1331/512 and V2=116939225/8128512,

    gamma=min(Omega^2/(108 Kbar^4 V2),27/128)/256>0.

This bound can be extremely small and is not a useful conditioning claim.
Other rows only add PSD contributions. Affine normalization multiplies
the floor by at least 25 on the full native affine box; zero-width local
coordinates require their own free-coordinate treatment. The original
per-axis RMS budget still uses N=200. Full14 nonlinear identification,
broad speed coverage and useful observed-data confinement are open.
Clock author/audit 56/51 pass, with independent subgroup and rational
interval comparisons and separate reconstruction of the clock-error
and Gram constants. No numerical trigonometric evaluation occurred.

SELECTED_GEOMETRY_EDGES_V25 compiles the selected degree-four minor
directly from the original shared coefficient source. Cross-axis edges
give det G>=lambda_x^2 lambda_y^2 Omega^2 by exact mass cancellation;
same-axis edges retain the opposite/selected source mass ratio. Six
whole-region guards bound both masses below/above, the oriented minor
below, and trace above. A determinant lower delta and trace upper T give
G>=(27 delta/T^3)I. Failed guards retain the candidate. Column scales,
clock, rows, source restrictions and every original domain operation
remain bound. Bounded polynomial collection falls back transactionally.

The full200 symbolic carrier preserves the 94,750-node original prefix
and has 100,388 total nodes, fourteen nonlinear inputs and no observations.
Its fixed oriented pair does NOT implement the phase-uniform four-minor
sum. No numerical optical wrapper is supplied. Generic source author/
audit 55/60 pass, including distinct dense/scaled systems, shared radical
families, hidden interior rank losses and unused domain obligations.
Source binding never proves that a proposed optical guard holds.

Primary research adds a retained-original read of Boyd--Vandenberghe,
Convex Optimization (2004), Appendix A.5.4, printed 648--649 (PDF 662--663).
Root independently read both original text pages and checked the exact
excerpt/index. The SVD/Gram/operator-norm statements support the standard
matrix interpretation; the prism-specific perturbation theorem is derived
here. A draft subsection reference was corrected before freezing. No
new download, whole-book read or external global inverse theorem is claimed.

Eight final reports total 433 top-level proof/implementation checks. The
controlled_geometry_manifest_v25 snapshot binds the release and these
three living logs, preserving all 24 prior snapshots and their historical
exception. No frozen cover, main paper, archive category or T1--T10 label
changes. Full18 recovery, useful global cost, integration, optical validation
and a 5--7-degree blind initialization tolerance remain unproved. The goal
stays active. A possible next step is a rigorously sourced real-analytic
zero-set argument for generic affine rank; it is not proved by this release.

### 2026-09-27: user-directed change to direct full-system reconstruction

After the user challenged the lack of complete recovery, they explicitly
stopped further verification/certificate-integration work and requested an
analytic result or routine addressing the full unknown system. The pending
v26 notes and reports remain as written; no v26 snapshot was frozen. Proposed
v27 RMS-cover work was stopped before implementation. Do not resume that
audit/manifest cycle as the default next action.

New files `risley_lattice/ray_state_inverse.py`,
`risley_lattice/ray_state_initialization.py`, and `paper/RAY_STATE_INVERSE.md`
instead formulate and implement a direct inverse-state candidate routine.
Six outgoing air-ray directions per retained timestamp, twelve shared globals,
and adjacent rotor equations retain all eighteen original unknowns. Inverse
Snell gives each signed face normal explicitly from consecutive ray directions
and glass index; no forward exit-radical evaluation is required in this lift.
The exact correspondence includes transmitted-branch guards, original prism
order, actual timestamp increments, native phase origin and zero-wedge fibers.
The rotor-constraint Jacobian has full row rank in the lifted states even at
zero wedges and speed collisions. The local linear system has fixed-size
time blocks and a twelve-coordinate global border. This does not prove global
convergence or remove physical nonidentifiability.

The implementation combines scan-only initialization, sparse TRF, augmented
Lagrangian rotation equations, inequality penalties, and joint bounded affine
fits between rounds. It returns numerical lifted candidates or unresolved;
it does not certify reconstructed-vector forward agreement or uniqueness.
Native wedge priors remain eighteen degrees, not silently restricted to seven.
No optical numerical evaluation, observation instantiation, recovery run,
test campaign, audit report or manifest was performed for these new files.

The same analytic note gives a new all-active six-degree finite-noise
obstruction: wedges (+6,+6,-6), strictly native glasses and arbitrarily slow
nonzero speeds independent of the sampling rate can make the entire workpiece
distance range [50,200] indistinguishable at any prescribed positive allowance
over the fixed record. It is not a positive-speed noiseless ambiguity claim.
The historical recovery ledger is unchanged. Full18 global recovery, useful
cost and demonstrated convergence of the new routine remain open.

Follow-up implementation after the user requested research and a working
routine: primary multiple-shooting, variable-projection and sparse-spectrum
research led to analytic local/sparse derivatives, scaled data residuals,
stationarity-aware augmented-Lagrangian updates, competing spectral bases,
and real work limits. `ray_state_derivatives.py` shares the exact inverse-state
algebra. `ray_state_constrained.py` adds a separate constrained trust-region path. The bounded
`experiments/ray_state_development.py` harness predeclares three synthetic cases
and supplies no true parameters to either inverse solver. It has not been run:
clarification of the earlier numerical-optics prohibition is pending. No new
recovery, angle robustness, global convergence or success rate is established.

## 2026-09-27 (evening): ray-state inverse executed for the first time

The bounded harness `experiments/ray_state_development.py` was run on its
three predeclared cases (method `al`, 2 starts, 60-90 s cap). No truth value
was used for initialization; the standard 200-sample, 10 s record was used.

- moderate (wedges 5, -4, 3 deg): status `unresolved` (finite work budget),
  42 s, max native parameter error 6.5e-4, all 18 coordinates inside the
  1e-3 rule. Lifted axis RMS 2e-7, rotor closure 5e-10, strict branch.
- seven (wedges 7, -6, 5 deg): 35 s, max native error 1.6e-3, carried by
  d_W (137.0016 vs 137); every other coordinate below 7e-4, speeds to 1e-9.
- collision (N1 = N2 = 1.2 exactly): failed. The scan-only initializer
  cannot separate two equal-speed prisms; the iterate pinned to prior
  corners (glass 1.8, d_W 200, wedges -18) with axis RMS 0.6.

The existing back end composes with it: `solve.trf` on the exact model,
started from the two blind outputs, reaches 4.6e-9 (moderate) and 4.7e-10
(seven) in 0.2 s and at most 400 evaluations.

What this is: the first executed evidence that the lifted ray-state
formulation lands in the correct basin blind at 5-7 degree wedges, and that
the verified polish finishes it. What it is not: a success rate (three fixed
development cases), a 15-degree result, a collision result, or a certificate.
The `unresolved` label is the routine's own acceptance threshold, not a
recovery failure. Nothing from the archived campaign was replayed.

Regression, same evening: `python experiments/solve18_battery.py` gives
26/30 PERFECT in 628 s, failures at cases 4, 10, 11, 18, the documented
baseline. The uncommitted 09-15 edits to `solve.py`/`spectral.py` and
today's untracked files did not change the core method's result.

## 2026-09-28: sprint on synthetic recovery stopped by the user; theorem-only attack launched

Morning: a sprint measured blind recovery of the ray-state inverse on a
frozen set of 30 admissible cases with wedges 10 to 18 degrees
(`experiments/ray_state_sprint_15deg.py`, results in
`experiments/results/ray_state_sprint_2026_09_28/`). Findings, recorded for
the log only: the scan-only initializer's misses were wrong physical order
in eleven of twelve cases (speed magnitudes correct, prisms permuted, a
separate least-squares basin with scan RMS about 0.3); enumerating all
(basis, order) proposals with a three-iteration screen separates the true
order by six orders of magnitude in residual; a near speed collision (0.024
Hz apart) has no proposal with the right line set. The certificate must be
gated on the candidate's own residual; before that gate one certificate
closed around a non-fitting point. The user stopped the sprint: recovering
synthetic cases, in any number, is not the deliverable and will not be run
again. The stopped screen-mode run and a collision probe were killed.

Afternoon: pure theorem-and-proof attack, starting at P = 2. Brief and
targets in `paper/P2_THEORY_2026_09_28/BRIEF.md`; nine parallel agents own
targets A to I (rotor constants; large-sieve coefficient recovery;
analyticity strip and tail; plate-state reduction; one-prism explicit
modulus with unknown incidence, the crux; upstream Lipschitz constants;
order separation; algebraic and holonomic structure of the Fourier
coefficients; exceptional set and quantitative nonresonance). The assembly
theorem they feed is stated as a skeleton in `J_assembly.md`. No solver
runs, no case batteries, no statistics are permitted in this line of work.

## 2026-09-29: explicit tails and a monotone entering-ray inverse for P=2

The theorem-only continuation is in `paper/P2_THEORY_2026_09_28/`.
`C_analyticity_tail.md` proves a uniform complex tube from whole-torus
transmission and forward margins, with an explicit positive strip width,
supremum bound, Fourier decay, tails and first weighted coefficient sum.
The constants are conservative; no useful fixed200 tail bound is claimed.

`E2_entering_ray_readout.md` proves a new interface after last-prism
hardware is known. The zero-tilt value and quadratic position coefficient
give a scalar equation for the unknown entering tangent. Its derivative
is at least 111/33800 for |T|<=1/2, |X0|/d<=9/10 and native glass indices.
Every native two-prism system meets these premises when the first
normalized tilt has modulus at most 1/100. The position then follows
explicitly. All parameter/data perturbations have stated error bounds.

`F_upstream_inverse.md` uses that small interval: first and third angular
coefficients give a strictly monotone inverse for the wedge and index,
including normal incidence, and position coefficients give gap and source.
Explicit finite interpolation remainders replace derivatives. This is
complementary to the existing overnight `F_upstream_lipschitz.md` finite
symmetric-pair construction, which is preserved.

`E_unknown_incidence_local.md` proves exact seven-parameter rank at the
normal, zero-offset center and a local Holder modulus on an explicitly
tiny cube. This is not global E. `K_joint_rotor_coefficients.md` proves an
explicit confluent-matrix lower bound and a conditional contraction for
joint rotor/coefficient refinement, with measurement error plus tail and
no separate window-leakage term in its local error formula. It requires
K>=2L and entry into its explicit ball; entry is not proved for the
intended record. `A2_masked_rotor_rule.md` adds an alternative rotor proof
while preserving the completed existing A note.

`I_nonresonance_and_fibers.md` gives a finite exact separation test, an
explicit sampling-independent family, a deterministic excluded-area bound,
and exact zero-wedge fibers. It does not classify all-active colliding
rotors. `H_holonomic_structure.md` constructs a differential operator of
order at most four for one axis of one prism and its bilateral Fourier
recurrence, without asserting a finite stable hardware inverse.

The assembly now explicitly retains the full-prior target and separates
it from the restricted competitor class used by A/C/D. B's missing rank
premise, rotor-error numerator, P=1 alias interval and masked-row bound
are corrected using the existing audit. Existing overnight A/D/F/G and
the corrected parameter count were present on rereading and are retained.

The new algebra and rational bounds were checked with
`experiments/p2_theory_2026_09_28/EF_symbolic_2026_09_29.py`. No optical
sample, synthetic record, recovery run, statistical argument or manifest
was produced. Global last-prism stability, explicit separation against
freely changed wrong-order hardware, out-of-regime competitor exclusion,
and a useful positive noise threshold remain open. Full-prior confinement
and the native 0.001 target are not established.

## 2026-09-29: global free-shape order separation; global centered-axis inverse

`paper/P2_THEORY_2026_09_28/G3_global_order_curvature.md` resolves the
free-shape order-separation obligation at the labelled-torus interface.
It covers arbitrary native beam incidence and offset and all freely
changed wrong-order native hardware. A corrected expression in the
linear, quadratic and mixed cubic coefficients has the sign of order
and magnitude at least 2*A_min^8. Its response separation constant is
the explicit expression (G3.7), with no compactness minimum or nearby
competitor assumption. The coefficient comparison already works on a
small flat-state tilt square, where every native model is analytic.

The proof is computer-assisted: the rational curvature function in
G3(3.3)-(3.4) was enclosed over a box containing the entire native
parameter domain. The completed interval Bernstein covering returned
the lower bound 0.000015231554579386929, above the rational floor
1/100000. `experiments/p2_theory_2026_09_28/G_general_orientation_interval.py
--bernstein --max-boxes 50000` reproduces the proof. It retains the
common reciprocal workpiece distance in gap/d and source-position/d;
earlier enlarged-domain enclosures were inconclusive. Its arithmetic
contract is correctly rounded binary64 rational operations with outward
nextafter, with no libm or BLAS premise. This is one validated inequality
on a declared box, not evaluation of optical records or fitted cases.

`G2_free_order_orientation.md` gives a separate analytic sign formula
at normal incidence and a stronger bound on its centered truth stratum.
`E3_global_centered_inverse.md` gives global native-competitor confinement
when one true axis is centered and normally incident. Its hardware
A,n,d spans the declared native intervals, the other true axis may have
arbitrary native incidence and offset, and the competitors have no
known-incidence or nearby-shape restriction. Constant/quadratic response
coefficients force incidence and offset; a strictly monotone invariant
of the first, third and fifth odd coefficients recovers glass and the
remaining hardware. All coefficient and parameter loss factors are
explicit. This is distinct from the earlier tiny local E cube.

Exact identities, coordinate covariance and rational analytic bounds
were verified by `EG_global_symbolic.py`. J and the original G status
now point to these results. No solver, synthetic or archived optical
record, recovery count, statistical guarantee, or manifest was used.

The overall theorem is still incomplete. General E with neither true
axis centered, full-prior rotor/reconstruction coverage from the moving
finite record, a useful native 0.001 allowance, and complete exceptional
fiber classification remain open. G3's positive universal constant is
conservative; it must not be reported as an instrumental-noise result.

## 2026-09-29: explicit longer-record P=2 confinement, full native competitors

The later assembly in `paper/P2_THEORY_2026_09_28/J_assembly.md` now
proves a longer-record P=2 theorem with native error at most 1/2000 in
all fourteen coordinates, leaving 1/2000 for reporting-roundoff within
the requested 1/1000 target. The sufficient record length K0 and hard
allowance eta0>0 are specified by finite integer recurrences, arithmetic,
radicals and ceilings. `N_native_accuracy.md` gives the full formulas
and a proof that the truth regime contains a nonempty open domain.
Every compatible native specification physical at the measured times
is covered, whether or not it belongs to that truth regime.

`E4_global_algebraic_modulus.md` closes general E at arbitrary incidence
and offset on both axes, including Q=0. It uses the exact physical
33-value inverse and a complete projection-height bound on its physical
graph, giving an explicit global Holder modulus. It does not fit the
unrestricted implicit-polynomial coefficients or rely on a local entry
condition. Its exponent and constants are extremely conservative.

`L_full_prior_extension.md` proves native P=2 tilt monotonicity throughout
the transmitted domain. Saturating the last face at its TIR boundary
gives every native competitor a continuous monotone torus extension
matching all physical samples. Uniform Holder, generator-dominance and
Fejer bounds follow. `M_full_prior_finite_record.md` uses these to derive
candidate wedge floors and spectral separation from the truth's finite
nonresonance, rather than assuming them. Matching in both directions
and generator dominance force the signed rotor permutation. Three
distinct Fourier degrees close the rotor/coefficient/torus error budget
without a tail-amplification loop. G3 fixes physical order; E4, E2 and F
then give the native-coordinate bounds in N. No contraction entry
hypothesis remains in this route.

The result is computer-assisted solely through the previously completed
G3 curvature inequality. New checks E4_polynomial_bounds.py,
L_native_monotonicity.py, MN_budget_symbolic.py and O_graph_symbolic.py
verify exact formal identities, polynomial degree/height bounds and
rational continuum estimates. EG_global_symbolic.py was also checked.
No optical solver, synthetic or archived record, statistical argument,
recovery count or manifest was used.

`O_exceptional_fibers.md` supplies finite complete algebraic invariants
of a physical analytic scan germ. Its polynomial differential model
has 69 variables per system and degree at most 196; a specified ideal-
stabilization bound supplies a uniform finite jet order. This yields an
effective exact fiber classification on q*N1-p*N2=0, including colliding
speeds and an unknown common speed. A closed-form catalog of every
individual all-active resonant hardware fiber has not been computed.
The result concerns a connected physical interval, and is not a claim
to recover those derivatives stably from 200 noisy samples.

O also gives stationary-first-prism fibers and an explicit analytic
short-record obstruction. For every finite K and eta>0, two fully
active native systems with distinct nonzero rationally independent
speeds can have records differing by less than eta while their
workpiece distances differ by one. Their differing source positions
are given exactly by the affine transfer. The same construction at
N's separated speed witness disproves uniform K=200, eta=1e-5 recovery
on N's declared truth regime. This is a proved parameter family, not
an evaluated optical example.

Thus the explicitly permitted longer-record theorem is closed. The
fixed-200 T5 criterion is not achieved and must not be marked complete.
A useful short-record theorem on a stronger speed regime and a P=3
extension require additional work. The saved bounded decision backend
has not run a physical-fiber query here; N's representative-selection
statement is a complete theoretical construction, not a runtime claim.

The final selector in N3 can be constrained to the proved truth regime
without weakening the full-prior competitor quantifier: the unknown
truth guarantees that this selection set is nonempty. The nonresonance
conditions become polynomial clock inequalities
Re(z1^v1*z2^v2)<=cos(pi/L). Reapplying the theorem to this compatible
representative gives the entire compatibility fiber radius 1/2000;
an additional coordinate rounding radius 1/2000 yields 1/1000.
I4 additionally gives the exact excluded speed-region area by finite
polygon inclusion-exclusion, a piecewise quadratic rational formula,
complementing I3's simpler upper bound.

## 2026-09-29: full signed two-prism wedge range

`paper/P2_THEORY_2026_09_28/P_full_wedge_range.md` removes the chosen
5.74-degree lower cutoff. N and J now allow any declared integer
a>=10 with |sin(ax_i)|>=1/a. Their union covers every nonzero wedge
in [-18,18], including both endpoints and all sign choices. E4, M,
G3 and N's inverse modulus already tracked this scale symbolically;
P gives the substitution, finite constants and precise quantifiers.
The same construction gives any rational native target 0<t<=1.
Full native physical competitors remain covered. The finite speed
test and physical-sample conditions remain hypotheses.

An algebraic norm argument proves
dist(v1*sqrt(2)+v2*sqrt(3),20Z)>1/(18Q)^3 for
0<|v|_1<=2Q. Thus the same fixed speeds sqrt(2),sqrt(3) work at
every wedge scale using Q=4q2^2 and delta=1/[2(18Q)^3]. A direct
analytic inequality proves whole-torus physicality for every native
wedge pair when 1.3<n_i<1.4 and |sin(beta_a)|<.001. Hence the
angle coverage is nonvacuous at every nonzero angle pair, without
sending the speeds to zero.

Exactly zero is an explicit obstruction, not a recovered fourteen-
coordinate point: that prism's speed and phase are invisible, and
I's plate formulas give additional hardware freedoms. P also proves
that no common positive noise allowance works over arbitrarily small
nonzero wedges, even with infinite records and fixed algebraically
separated speeds. Its two phases differ by two degrees while the
all-time output discrepancy is <=2000000*|A1|. Thus the dependence
of the sufficient noise bound on the wedge scale is necessary.

`P_wedge_scale_symbolic.py` completed using exact symbolic identities
and rational continuum inequalities only. No optical record, solver,
sampled recovery test or statistical argument was used. The recovery
theorem still inherits the single computer-assisted G3 curvature
inequality. No new computer-assisted enclosure was needed. The zero
strata satisfy the brief's explicit-obstruction alternative; unique
fourteen-coordinate recovery on the entire closed interval is false.

## 2026-09-29: empirical two-prism POC requested and executed

After the theorem and wedge-range extension, the user explicitly asked
to build a POC and see how it performs. This authorizes the following
empirical work beyond the earlier theorem-only phase. The theorem notes
are unchanged by these numerical results.

Added `risley_lattice/poc2.py`: a blind 14-coordinate candidate solver.
It uses the existing two-generator spectral front end, a joint Fourier
phase/sign readout, both physical orders, and three fixed nominal hardware
starts. Exact Snell transfer exposes affine dependence on d, gap, px, py;
bounded variable projection eliminates them while a complex-step reduced
Jacobian refines the other ten coordinates. A sequential linear minimax
refinement targets the per-axis hard error bands. Final residuals use the
independent canonical model. True parameters are never passed to solve14.
The existing three-prism solver and forward-model files were not edited.

`experiments/poc2_recovery.py` supplies fixed fixtures, bounded deterministic
noise, native-coordinate scoring and a CSV entry point. No physical-order
permutation is allowed when scoring errors. The native +/-18-degree prior
is retained, including small wedges. This finite-start numerical solver
does not execute the theorem's global algebraic selection or certify
global parameter recovery. Its spectral proposal cutoff remains 0.10 Hz;
arbitrarily slow native speeds are not covered by this implementation.

Executed results in `experiments/results/poc2_2026_09_29/`:

- `noiseless_200.json`: all 17 active cases met 0.001 in every coordinate;
  largest error 2.2226670259861692e-7, on a 0.01-degree wedge case. There
  were nine declared active development fixtures and eight fixed-seed
  additional native fixtures. Exact +/-18-degree endpoints, signed wedges,
  reversed physical order and 0.024 Hz speed separation were included.
  Median active-case runtime 0.59 s. The complete run, including two
  exact-zero plate controls, took 48.6 s. Both plate controls fit the
  records but did not recover the original fourteen coordinates, as
  expected from their exact ambiguities; they are not counted as active
  recoveries. These are finite-case measurements, not a success probability.
- `minimax_noisy_200.json`: at deterministic perturbation amplitude 1e-5,
  five of seven active cases met 0.001. The 0.1- and 0.01-degree cases
  missed; largest error 0.005968429085822402 in n1 at 0.01 degree.
- `minimax_noisy_800.json`: all seven of the same noisy cases met 0.001;
  largest error 0.0005307434850339021 in gap at 0.01 degree.

The initial least-squares-only noisy run is retained as `noisy_200.json`.
It slightly exceeded the hard sample allowance despite small RMS; the
minimax refinement resolves that fit objective mismatch. It does not
turn a compatible point into a global accuracy guarantee.

`risley_lattice/poc2_verify.py` independently evaluates the exact P=2
transfer with outward interval arithmetic at a proposed point. It checks
the physical branches and its residual at exact times k/20, under ivx's
documented A-fp and A-libm assumptions. It certifies compatibility only,
not parameter accuracy or uniqueness. All 19 noiseless candidates passed
at the explicitly stated numerical allowance 1e-8; exact zero residual
is not certified from floating-point-generated observations. All seven
final noisy candidates at each length passed at the requested 1e-5.
`poc2_verify_results.py` reproduces these checks without rerunning an
inverse solver. The reference generator's interval data floor is recorded
separately wherever its truth enclosure does not close at 1e-5.

For the 0.01-degree noisy 200-sample record, two native physical points
are interval-verified compatible at allowance 1.0000013582468576e-5,
including that reference arithmetic floor. Their native separation has
lower bound 0.005968429085822401, exceeding 0.002. Thus no single center
can be within 0.001 of both at this explicitly recorded allowance.
This is a constructive two-point obstruction, not a conclusion drawn
from failure of an optimizer or a singular-value heuristic.

Validation completed: canonical parity on all declared fixtures at 200
and 317 samples (largest scaled difference 1.640902655050181e-11), reduced
Jacobian directional checks with interior and bounded affine variables
(relative differences below 3.4e-9), interval consistency and refusals,
zero-rotor and gap/distance fibers, and generated-CSV blind round trip.
Usage, limitations and reproduction commands are in
`experiments/POC2_RECOVERY.md`. These numerical results do not alter the
theorem's conservative constants or prove its global confinement at 200
or 800 samples.

## 2026-09-29: uniform deterministic hardware/noise bounds

The user requested explicit hardware and noise conditions under which
every case is recovered deterministically within the target epsilon.
Added `paper/P2_THEORY_2026_09_28/Q_uniform_hardware_guarantee.md` and an
executable sufficient condition in `risley_lattice/poc2_bounds.py`.
This is a distinct guarantee on an independently declared convex hardware
tolerance box X, not full-native-prior exclusion around a fitted answer.

Q1 gives the exact universal two-point criterion for an epsilon-radius
center: forward distance <=2eta must imply parameter distance <=2epsilon.
Q2 records why exact continuous-parameter recovery at positive noise is
impossible in general. Q3 proves uniform projected contraction from the
nominal hardware, including finite iteration and arithmetic error budgets.
It covers every truth in X and every componentwise bounded noise vector.
The output encloses every compatible member of X but need not itself be
a noise-compatible representative. No condition is claimed outside X.

`poc2_verify.py` now shares the same optical graph between point values
and first/second-order interval derivatives on boxes. Existing point
compatibility behavior is preserved. `poc2_bounds.py` bounds I-CJ over
all of X using the cancellation-preserving reference product C H0 and
interval Hessian variation. A numerically proposed preconditioner and
positive weights are accepted only after the outward contraction
inequality closes. Positive sums/products include underflow guards.
The projected inverse tracks deterministic componentwise errors, uses
inward clipping, and falls back to arbitrary-precision intervals if the
binary64 update exceeds its proved arithmetic budget. Certificates are
bound to the actual preconditioner and sampling count.

`experiments/poc2_hardware_bounds.py` certifies custom declared boxes,
accepts external scan CSVs, and reproduces two example regimes at exact
times k/20, K=200, target 0.001 in each native coordinate:

- `uniform_wide.json`: beta=0.16462893399961281,
  sufficient eta_max=1.3327758740862206e-7, iteration bound 6.
- `uniform_edge.json`: beta=0.4161300529546671,
  sufficient eta_max=3.237470752062376e-8, iteration bound 13.
  Its signed wedge intervals touch +18 and -18 degrees.

Both are narrow calibration/manufacturing tolerance boxes, with all
fourteen coordinates variable. Speed and glass-index prior radii are
already below 0.001. The ranges are fully listed in
`experiments/POC2_HARDWARE_BOUNDS.md`; neither is a guarantee throughout
the original full fourteen ranges or throughout all signed wedge angles.
Their physical branch margins are positive uniformly over the boxes.
These sufficient noise thresholds are not optimality claims.

The proof's arithmetic assumptions are ivx A-fp, A-libm and A-blas;
arbitrary-precision fallback additionally assumes mpmath's outward
interval contract. No statistical or Fisher bound enters the guarantee.
The full-native-competitor long-record P1/N2 theorem remains the broad
theoretical result. Zero, arbitrarily small wedges, and arbitrarily slow
speeds retain their proved universal-noise obstructions; full-prior
confinement at practical K=200/800 is not established by this work.

Validation: both uniform interval certificates closed; each passed six
implementation controls (three unknown points and two deterministic
noise patterns), an independent Jacobian comparison, precision-fallback
containment, and rejection of excess noise/mismatched preconditioners.
The earlier POC checks passed again, including 38 canonical forward
comparisons and interval consistency. Python compilation passed.
An external-CSV round trip at eta=1e-7 returned a maximum certified
parameter radius 0.0004153133838153449 after four steps; its declared
hardware and synthetic input are saved separately from the solver.
The finite control cases validate implementation behavior; the interval
inequality over each whole box supplies the universal quantifier.

## 2026-09-29: exact-arithmetic Lean recovery formalization (active verification)

The user requested the rigorous two-prism algorithm in Lean before moving
to three prisms. The reusable proof chain now passes pinned Lean 4.33.0:
`UniformRecovery`, `DerivativeBridge`, `SecondOrderBridge`,
`RationalEnclosure`, `TrigTrace`, `OpticalExpr`, `OpticalEnclosure`,
`PrismChain`, `RationalBudget`, `IntegerBudget`, `OpticalRecovery`,
`CertifiedRecovery`, `RationalStep`, and `CheckerControls`.
Evidence: `experiments/results/lean_2026_09_29/recovery_core.json` and log.

`CertifiedRecovery.Valid.exact_recovery` connects proof-carrying optical
value/gradient/Hessian enclosures and finite rational checks to the actual
projected inverse, started at the declared nominal hardware. It covers
every compatible truth in the entire declared box and every hard-bounded
noise vector. The optical premises are enclosure proof trees, not assumed
derivative inequalities. Exact rational reciprocal/root rules and Taylor
plus angle-doubling trigonometric certificates remove A-fp/A-libm from
this new proof path. `RationalStep.Run.Valid.recovery` proves soundness of
a rational implementation whose actual point evaluations carry those
certificates. These theorems print only propext, Classical.choice and
Quot.sound. No custom axiom, sorry, native_decide or probability is used.

The strict mathematical optical model is specified in `PrismChain` for
4P+6 coordinates, retaining physical prism order and the exact k/20 clock.
This is not a formal proof that the separate Python forward implementation
equals that specification. The complete original full-native-prior
P1/N2 theorem is also not thereby formalized.

Concrete instance assembly is still in progress in this entry. The Wide
box uses seven selected times from the 200-sample record, eta=1e-8 and
12 iterations. The +/-18-degree endpoint box uses nine times, eta=1e-9
and 14 iterations. The seven-time endpoint proposal failed the sufficient
inequality and is not a certificate or impossibility result. These are
the SAME independently declared tolerance boxes as the previous entry,
with more conservative formal-protocol noise allowances. Their full
optical and arithmetic modules must pass before the final closed recovery
theorems may be claimed. Generators label outputs GENERATED_NOT_YET_LEAN_CHECKED;
the final checker alone writes a *_closed.json record after all premise
modules and the assembled recovery theorem have passed and its axioms
have been audited. Older P2WideBudget/P2EdgeBudget exports remain explicitly
conditional on their optical step estimate and must not be conflated
with these new closed-instance attempts.

The exact-rational inverse producer has also executed a non-central
unknown-hardware control with bounded noise aligned with the weak
workpiece-distance row. True hardware is not supplied to the solver.
Those run outputs remain labelled NOT_YET_LEAN_CHECKED; the control
validates execution but does not replace universal certificate checking.

Subsequent checkpoint: `P2SevenRecovery.lean` is now FULLY CLOSED and
kernel checked. `P2Seven.recovery_from_200` has only hardware-membership
and noise-bound premises; there is no assumed optical derivative bound,
floating-point contract or numerical recurrence premise. It proves
epsilon=1/1000 after twelve projected steps from the nominal hardware,
uniformly over the Wide box and all coordinate noise <=1e-8. Every
concrete enclosure and matrix inequality is included in the proof chain.
Both its fourteen-observation theorem and its 200-record corollary print
only the three standard axioms above. Evidence: `P2Seven_closed.json`
and `P2SevenRecovery.log`. One optical module needed a higher compiler
memory limit; its successful retry is included in that final closure.
Endpoint and computed-run closure remain pending at this checkpoint.

Computed-run checkpoint: P2SevenComputed.lean now passes Lean. Its
computed_recovery theorem covers every hardware in the Wide box compatible
with this fixed rational observation vector. The actual_accuracy theorem
has no hypotheses: the control truth's box membership and its data/noise
compatibility are themselves checked, as are all twelve rational updates
and their optical evaluations. Axiom audit: only propext, Classical.choice,
Quot.sound. P2Seven_computed_closed.json binds the run, instance and proof
sources by hash. This is stronger than an unchecked numerical trial, but
one computed run does not prove universal termination of the Python producer.

Three-prism preparation: the exact certificate producer and assembler now
infer n=4P+6 from the hardware box, retaining physical order. The refactor
reproduced all P2 enclosures, matrices and finite budgets exactly. A distinct
18-dimensional hardware box is declared in hardware_p3.json; twelve prior
coordinate radii exceed the target, while speed and glass-index radii are
already narrower. Nine selected sample times and eta=1e-10 yield a proposed
three-step bound with final integer radius 540162567/10^12. The numerical
preconditioner is frozen as rational input; only checked rational inequalities
are accepted. P3NineNumbers and the first matrix row now pass Lean; remaining
optical and matrix checking is active, so no closed P3 claim is made here.
The first matrix encoding exceeded reduction capacity at maxRecDepth 100000;
the same entire row passes at 1000000. No mathematical inequality was changed.

The P3 rational execution control has actual max native error 2.06547e-7 with
nonzero noise and noncentral truth, which was not given to the solver.
Four canonical-forward controls had max discrepancy 1.6680e-11. These are
implementation controls, not substitutes for the uniform certificate or a
formal equivalence theorem for the separate Python implementation. Protocol,
scope and reproduction commands are recorded in experiments/POC3_FORMAL.md.

P2 ENDPOINT CLOSURE: P2EdgeNineRecovery.lean has now fully passed Lean,
including all optical value/gradient/Hessian modules, all fourteen matrix
rows, the 200-record recovery theorem and exact +/-18-degree boundary
lemmas. Only truth membership in the declared endpoint box and coordinate
noise <=1e-9 are assumed. After fourteen projected steps the error is
<=1/1000 in each native coordinate. There is no assumed derivative bound,
FP/libm contract, custom axiom or proof hole. Axioms are propext,
Classical.choice and Quot.sound. Evidence: P2EdgeNine_closed.json and
P2EdgeNineRecovery.log. Both promised P2 hardware-box instances are closed;
the Wide rational execution is also closed. These results do not formalize
the paper's full-native-prior P1/N2 theorem or imply all signed angles are
covered by the two narrow hardware boxes. Work continues on the distinct
three-prism instance with those scope limits retained.

The first complete P3 optical module, P3NineS000A0Box, has passed all
12 split proof modules and its facade. Its value, all eighteen gradient
entries, and all 324 ordered Hessian entries are certified over the whole
declared box against PrismChain.axisExpr 3. The scalable producer now
shares constructor-level derivative equality proofs, rather than repeatedly
expanding the full expression tree. This changes proof representation only.
P3 recovery remains pending until every other optical and matrix module and
the final assembled theorem pass.

Stronger P2 noise budgets are now ALSO fully kernel checked, reusing the
unchanged optical proofs. P2SevenOperational permits eta=5e-8 with fourteen
steps and final table maximum 0.000928842308. P2EdgeNineOperational permits
the same eta=5e-8 with twenty steps and final maximum 0.000977214159.
Their model/hardware hypotheses and target remain unchanged. The matching
*_closed.json records bind new runnable *_instance.json inputs. The only
axioms remain the three standard foundations. The new exporter can derive
noise/iteration corollaries without repeating optical enclosure proofs.

For P3, the analogous proposed Operational corollary permits eta=1e-7
with four steps and final table maximum 0.000641150929; it remains pending
on the base P3 optical proof. Its separate noncentral noisy control run
has maximum actual error 0.00020668158112016534. This stronger budget is
analytic, not fitted to that control. The lower-noise three-step base
instance and the stronger four-step instance are distinct runnable inputs.

P3 SCOPE CORRECTION: I found that the initial P3Nine box placed ay3 at
19 degrees, outside the paper's native phase interval [-18,18]. Its pending
checks were stopped. All P3Nine and P3NineOperational numbers above refer
ONLY to that out-of-prior control and must not be cited as native-prior
recovery. No completed P2 statement is affected. The corrected independent
box is hardware_p3_native.json, with ay3 centered at 17 degrees and all
other coordinates unchanged. New proof namespace P3Native prevents mixing
old and corrected evidence. The producer now rejects out-of-prior boxes
by default, and the assembled Lean theorem additionally proves hardware-box
containment in every native coordinate range.

The corrected P3Native exact budget closes at eta=1e-7, four steps, with
integer final radius 0.000976530449 <0.001. Its noncentral noisy rational
control has actual maximum error 0.00027683637270512493. These corrected
proofs are being checked from their new sources; no native P3 closure is
claimed at this checkpoint. All three-prism numerical claims now use the
corrected box unless explicitly marked P3Nine historical control.

P3 NATIVE CLOSURE: `P3NativeRecovery.lean` has now fully passed Lean 4.33.0.
All eighteen matrix rows and all 36 optical box/point modules passed, as
did the final assembled recovery and native-prior containment theorems.
For every truth in the independently declared native hardware box and
every componentwise observation error <=1e-7, four projected steps from
the nominal center give error <=1/1000 in every native coordinate. The
integer final error table has maximum 0.000976530449. The protocol uses
both axes at k=0,17,39,63,88,112,139,167,199 from the k/20, K=200 record,
with source distance 6, prism thickness 3, and physical prism order.
Evidence: P3Native_closed.json and P3NativeRecovery.log. The instance SHA256
is aab9e42aa31c717f18cd3cd84feaae42b88201741e71e3570b661b27be7cf695.

`P3NativeComputed.lean` is also closed. It checks all four rational updates,
their optical evaluations, and the noncentral control truth's membership
and noise compatibility. The boundary-noise control reserves exactly the
reference enclosure radius before applying the allowed noise amplitude;
its numerical maximum actual native error is 0.0005536732144553943.
`computed_recovery` covers every compatible truth for these fixed data,
and `actual_accuracy` has no hypotheses. Evidence:
P3Native_computed_closed.json, binding rational_p3_native_boundary_run.json
with SHA256 9ffa60837a50bea60a9742a54344c65431a2e71556924414ccbad5e457467e57.
The generation-time labels in input JSON are intentionally unchanged;
the separate closure records bind those inputs and the proof sources.
All final P3 theorem audits list only propext, Classical.choice and
Quot.sound. There are no custom axioms, proof holes, native_decide calls,
statistical premises, or assumed optical derivative inequalities.

This closes the requested concrete P=3 extension of Q's calibrated-box
algorithm. All eighteen coordinates vary, but speed and glass-index
priors are already narrower than the target. It does not prove blind
recovery over the full prior, formalize J1's global selector, prove the
separate Python forward program equivalent to the strict formal model,
or prove universal termination of the Python certificate producer. The
earlier P3Nine phase-19 control remains outside the native prior and is
not a native recovery result. The two P2 Operational noise-budget audits
and the P2Seven computed-run audit were rechecked successfully.

Compiler bookkeeping now uses per-module OS locks to prevent duplicate
batches writing the same artifacts. A concurrent-writer control passed.
Explicit module-list batches no longer require an unrelated default P2
instance; that CLI behavior and conflicting-option rejection were checked.
One P3 run enclosure exceeded a 4 GB compiler cap and passed at 6 GB;
no mathematical premise was weakened. All builds stayed outside Dropbox.

## 2026-09-29: three-prism solver runtime measurement

In response to the user's runtime/conditions question, replayed the existing
Lean-closed P3Native boundary-noise input three times through the exact
rational `solve` function. Measured times were 2.8584483, 2.8565122 and
2.9494253 seconds, median 2.8584483 seconds, on this Windows machine using
Python 3.11.4. All returned fields matched the saved closed-run input
exactly; no proof sources or closed input files changed. This measures
four-step solving only, excluding acquisition, imports/loading, and new
Lean execution-certificate generation/checking. It is not a worst-case
runtime theorem. Evidence and input hashes: `P3Native_runtime.json` in
`experiments/results/lean_2026_09_29/`. The hardware tolerances are encoded
in the declared instance and formal hypotheses; this work has not measured
or configured a physical device. Physical box membership and the hard
observation-error bound must be independently established for a real scan.

## 2026-09-29: user correction of the broad inverse objective

The user explicitly rejected treating the narrow nominal hardware boxes as
the solution. The intended task is blind recovery of all native parameters
over broad ranges, or a broad family specified by explicit physical conditions,
with a deterministic native-coordinate error guarantee. Approximate true
speeds, glass indices, angles, geometry and beam parameters are not supplied
to the solver. A tiny neighborhood around a selected configuration does not
meet this objective, even when every inequality in that neighborhood is proved.
Earlier conversational wording claiming the three-prism problem was solved
under conditions overstated the relevance of the completed local theorem.

Working ambient ranges remain the original ones: each speed in [-3.5,3.5]
Hz; each wedge and phase in [-18,18] degrees; each glass index in [1.3,1.8];
workpiece distance in [50,200]; common gap in [2,15]; beam angles in [-25,25]
degrees; beam positions in [-5,5]. All physical orders remain candidates.
Any sufficient wedge-strength, frequency-separation or transmission conditions
must be stated and proved, with explicit dependence on noise, scan length and
target epsilon=1/1000. Such conditions must not encode an approximate answer.
The recovery theorem must address distant compatible candidates in the broad
admissible domain, not only establish injectivity near a proposed solution.
The original J1 P2 result has full-native-competitor coverage but extremely
conservative record/noise bounds; it has not been extended to P3 by the recent
Lean calibration-box work.

The user points to the large saved campaign as guidance. Located its raw audit,
classification and local-identifiability records under
`experiments/results/lisa_2026_09_10/`, and the subsequent target audit under
`experiments/results/full18_target_2026_09_16/`. Use that evidence to investigate
structure and exceptions without mistaking empirical coverage or local rank
for a global deterministic theorem. No archived recovery count changed in
this scope-correction pass, and no archive recovery campaign was restarted.
The broad three-prism objective remains OPEN. The proved local instances and
their hashes are preserved, and their documentation now states this distinction
prominently.

## 2026-09-29: broad three-prism finite-record theorem on a declared class

The user requested the actual broad theoretical extension from P=2, with
all eighteen parameters unknown. New note:
`paper/P2_THEORY_2026_09_28/R_broad_three_prism_theorem.md`.
This is a mathematical theorem with exact supporting arithmetic checks,
NOT a new Lean theorem and not a solver-runtime claim.

R1 quantifies over a broad native hardware class C_b, defined by uniform
positive optical branch margins and positive position derivatives in each
physical sine tilt over the whole independent tilt box. These are explicit
forward-physics conditions, not local inverse-Jacobian conditions or supplied
approximate answers. Competitors must belong to the same C_b, but need not
satisfy the truth's wedge floor or finite frequency-separation test. This is
weaker competitor coverage than P2 J1: arbitrary native competitors physical
only at sampled times are not covered by the P3 theorem.

An analytic interval argument proves C_2 contains the full product family:
indices [1.3,1.4] independently, beam sine angles [-.02,.02] independently
(approximately +/-1.146 degrees), positions [-1,1] independently, full
native distance [50,200] and gap [2,15], all signed wedges and phases in
[-18,18]. A scale |sin(ax_i)|>=1/a, a>=10, excludes exact-zero wedges;
the union over scales covers every nonzero signed angle triple, including
all endpoint combinations. Speeds remain unknown in [-3.5,3.5], subject to
an explicit finite nonresonance test, not selected nominal centers.

The proof obtains exact global shape/order injectivity from the existing
algebraic-degree ordering and exact last-prism inverse, followed by the
upstream first/third angular-coefficient inverse (including normal incidence).
A quantified polynomial comparison of two whole normalized responses has
160 variables. E4's complete projection bound, applied for 159 steps, gives
a specified positive global separation for requested coordinate accuracy,
including wrong physical orders. No fitted polynomial coefficients or
unspecified minimum/singular value enters the bound.

The finite-record rotor argument extends M to dimension three using explicit
adjugate bounds. Its truth test is dist(v.N,20Z)>=delta for 0<|v|_1<=Q,
Q=64q2^3; r=delta/(256q2^3). Three Fejer cutoffs keep the budget acyclic.
R16 defines sufficient K0 and eta0 for every declared scale, guard margin
and target epsilon. A finite exact algebraic selector uses only observations
and the declared class. Pairwise error <=epsilon/2 plus rounding gives a
deterministic epsilon guarantee on all eighteen native coordinates.

Nonemptiness is explicit: (sqrt(2),sqrt(3),sqrt(5)) and an open neighborhood
meet the speed test for delta=1/[2(16Q)^7]. A deterministic strip-volume bound
shows that this finite test rejects less than 1e-7 of the native speed-cube
volume. This is a size bound for the declared set, not a probabilistic
accuracy claim. The associated record length and precision remain extremely
conservative; neither K=200 nor practical runtime follows.

Exact supporting checks passed in
`experiments/p2_theory_2026_09_28/R_broad_three_prism_checks.py`:
formal optical identities, rational endpoint enclosures for the broad family,
uniform lever margins, the common small-cube guards, graph polynomial
degree/norm/count bounds for all six label matchings, integer rotor bounds,
and budget inequalities. Report:
`experiments/results/p3_broad_2026_09_29/exact_checks.json`.
Actual graph degree <=4, integer coefficient norm <=1423, conservative
all-permutation sign count <=1613, below the proof budgets 6, 2^50 times
the declared integer factors, and 4096. The enormous projection integers
are specified by recurrence, not expanded by this check script. One initial
supporting check used an unnecessarily strong .9 root lower bound after
relaxing its input to .5; replaced it by the valid .85 lower bound, still
strictly above the .8 graph guard. No final theorem margin was weakened.

Documentation distinguishes this broad analytic theorem from the previous
narrow P3Native Lean certificate. The latter's hashes and proof sources are
unchanged. Full native sample-physical competitor coverage, a practical
200-sample guarantee and a Lean formalization of R1 remain open. No archived
recovery count, T5 fixed-record status, main manuscript or git history changed.

## 2026-09-29: primary-source research and sharper broad P3 bounds

The user requested a deeper mathematical attack drawing on known methods,
including algebraic geometry. Read the current theory and earlier research
notes before choosing new work. Retrieved primary PDFs by Trefethen and
Weideman, Jeronimo/Perrucci/Tsigaridas, Sontag, Nie/Schweighofer and
O'Leary/Rust; inspected the relevant theorems/proofs and rendered the JPT
theorem page to verify its exponents. URLs, scope and retrieval hashes are
in `paper/P2_THEORY_2026_09_28/S_research_sharper_bounds.md`. Earlier Prony,
interval and affine-fiber research is credited rather than presented as new.
The saved raw and local-identifiability audit metadata were read for context;
no archive replay, forward experiment or new success rate was produced.

S proves that R's finite-record argument can use hard Fourier cutoffs with
exponential tails on the same C_b truth/competitor class. Fejer damping
must actually be removed: analyticity does not make its low-mode bias
exponentially small. Generator dominance and the existing conservative
matching/large-sieve budgets survive that change. C supplies the general
C_b strip; a sharper rational complex-majorant calculation proves strip
half-width 1/20 radians and |Z|<800 throughout the existing R5 product
family. This family still includes every signed wedge through 18 degrees,
unknown indices in [1.3,1.4] and full native distance/gap intervals. The
explicit tail is 169444800*(20/21)^(q+1). It bounds the entire continuum;
no sampled nominal hardware is used. q=765 suffices for response-tail
1e-8, but its full cube contains 3,588,604,291 lattice points. That number
is not a sample count or a native-accuracy guarantee.

The JPT positive-minimum theorem replaces E4's 210-step projection
recurrence by eta=(16*H*6^211)^(-211*12^211)/2, where
H=2^26*(a+1)^2*(m+1). The proof splits coordinate-separation branches and
uses compact positive-branch graphs; singular constraints are allowed by
the cited theorem. This remains extremely conservative: at a=10,m=134000,
log10(1/eta) is between 10^232 and 2*10^232.

For P3, S constructs a fixed rational determining design for all twelve
normalized shape coordinates and physical order. It combines the existing
193-node last-label test, 33-node last-prism inverse, and 81-node recursive
order test over all ordered coordinate pairs. Exact angular readout and a
degree-four zero count finish geometry recovery. The construction has
15361 distinct states, denominator 960000, and is permutation invariant.
An added all-box entering-ray bound |sigma_in|<.43, |Q|<27<d verifies that
the last-prism uniqueness argument applies at every needed upstream state.
These are normalized-response evaluations supplied by the torus bridge,
not independently observed or commanded timestamps.

The finite design permits a compact physical-graph minimum with
N=2027677, H=2^50*960000^2*(a+1)^2*(m+1), even degree 6 and fewer than
10^7 constraints. JPT gives Delta=(16*H*6^N)^(-N*12^N). Using Delta/4
in R16, together with exponential tails, gives alternative fully specified
broad recovery constants without R's 159-stage height recurrence. No huge
elimination/minimization was run. The analytic theorem and published
positive-minimum bound supply this conclusion, not finite check counts.

A separate dimension proof shows that good 25-state designs contain an
open dense set of rational tuples for each declared a,m. Exact symbolic
feasibility makes a deterministic search terminating. No such 25-state
tuple was constructed or certified here; the completed bound uses the
explicit 15361-state design. The generic theorem alone is not a usable
record or a noise certificate.

The note specifies the next concrete algebraic certificate target:
epsilon-c=sum(squares times branch inequalities)+equality-ideal terms.
An exact identity with c>0 excludes the entire corresponding bad-pair
cell. Putinar's existence theorem or an unverified numerical SDP does not
provide a useful value of c. The missing practical step is a sharp global
optical separation certificate preserving correlations and covering all
competitors, not another nominal local inverse.

Exact supporting arithmetic and symbolic bookkeeping passed in
`experiments/p2_theory_2026_09_28/S_sharper_bounds_checks.py`; report:
`experiments/results/p3_broad_2026_09_29/sharper_bounds_checks.json`.
R's prerequisite exact checks also passed. Python syntax and scoped
whitespace checks passed. These are mathematical results with supporting
arithmetic, not new Lean proofs. K=200, useful uniform noise tolerance and
coverage beyond R's declared competitor class remain open. No solver,
physics, formal proof source, main manuscript or git history was changed.

## 2026-09-30 — Constructive broad P3 shape inverse, replacing the generic height bound

Continued the user's theoretical push for broad unknown hardware. New
notes T_scale_invariant_inverse.md, U_three_prism_order.md and
V_constructive_three_prism_shape.md are in paper/P2_THEORY_2026_09_28.
They are computer-assisted mathematical proofs, not new Lean theorems.
No nominal hardware, sampled recovery count, archive fit or statistical
bound supplies the guarantees.

T derives the exact last-prism coefficients c_j=d*A^j*f_j(F,T,r), where
F=sqrt(n^2+(n^2-1)T^2)-1 and r=X(0)/d=T+Q/d. Three coefficient invariants
cancel both unknown scales A and d. A fixed preconditioner gives a global
contraction <.9 on F in [.3,.401], T in [-.021,.021], r in [-.061,.061].
This includes the entire R5 flat-last-prism input family. It simultaneously
recovers unknown glass, incidence and offset from any starting point in
the rectangle. Algebra restores wedge and distance. The final complete
cover has 543 leaves, 1085 visited nodes, and maximum certified norm less
than .898. Every operation is an enclosing integer/dyadic interval
operation at scale 2^70; the saved cover is independently replayed for
inequalities and complete prefix-tree coverage. Python/SymPy and the
interval implementation remain the trusted computation, not Lean.

T gives an explicit native error bound 23000*E/(.9*a0)^5 from errors E
in c0,c1,c2,c3,c5, with E<=a0^3/1000 and a truth wedge-sine floor a0.
It forces the competitor's nonzero wedge and matching sign. A complex
Snell contraction proves analyticity on |u|<=2 and |X|<150, allowing
Chebyshev coefficient extraction: 49 normalized last-slice values with
pairwise discrepancy <=5e-21 suffice for native error <.000772 at a0=.1.
These are normalized response values, not raw moving-scan samples.

U extends G3's corrected curvature invariant to every pair in a THREE
prism chain. It explicitly retains the third plate before, between or
after the active pair. The rational expression R satisfies R>1/50 on
the entire R5 outer box; all three cases certify on their unsplit root
boxes. Returning to measured normalized coordinates gives
|S|>65000*a0^8 with sign equal to physical order, regardless of signed
wedges, arbitrary allowed incidence and source offset. Direct derivative
bounds give magnitude <50 for all seven coefficients used by the
invariant. Therefore coefficient discrepancy at most
65000*a0^8/(48*50^5) excludes every changed wrong-order R5 competitor,
including zero-wedge competitors. This avoids algebraic rank tolerances
or an implicit-polynomial fit.

V composes T/U with F's explicit angular inverse and the corrected P3
source formula. Last-prism inversion is uniform while one upstream
normalized tilt ranges over [-.005,.005]; its input remains inside T's
rectangle. The recovered entering angular functions determine each
upstream wedge/glass, including normal incidence. The entering position
coefficients recover the common gap and both source positions. The source
formula includes both gaps and both upstream plate tangents.

The concrete finite design uses at most 35049 normalized states before
deduplication: three 65x65 pair squares for order, plus six 33x113 ordered
pair rectangles for peeling. At truth wedge-sine floor .1 and pairwise
normalized-response discrepancy <=1e-45, all twelve native shape
coordinates differ by <.000573 and physical order agrees. The dominant
certified error is gap (<.000572347); upstream wedge error is <1.374e-7
degrees and glass error <1.675e-9. Both truth and competitors range over
R5: n=[1.3,1.4], beam sines +/-.02, positions +/-1, g=[2,15], d=[50,200],
signed wedges through 18 degrees. Only the truth needs the wedge floor.
All positive floors and all positive target accuracies are covered by a
terminating exact interpolation-degree/noise-budget construction. The
concrete .1 floor is about 5.739 degrees, not the limit of the theorem.

This closes a constructive quantitative shape/order step on R5, replacing
the generic algebraic-height estimate there by ordinary explicit bounds.
It does not improve the constants for the whole C_b competitor class.
The 1e-45 allowance is still extreme, is sufficient rather than necessary,
and is NOT a sensor-noise guarantee. Speeds/phases, the original moving
record's noise amplification and useful sample count remain a separate
interface. No practical K=200 guarantee or full fast 18-D implementation
is claimed. Dense certified feasibility selection gives a deterministic
estimator in principle; runtime for that selection has not been bounded.

Reproducers in experiments/p2_theory_2026_09_28:
T_scale_invariant_inverse.py, U_three_prism_order.py,
V_constructive_shape_budget.py. Reports are in
experiments/results/p3_broad_2026_09_29/ as scale_invariant_inverse.json,
three_prism_order.json and constructive_shape_budget.json. J, R and S
link the results and preserve the older theorem scopes. No optical model,
production solver, formal proof source, main manuscript or git history
was changed by this research pass.

## 2026-09-30 - Two-window full-record bounds and a fast short-record obstruction

Continued the user's instruction to push the remaining scan interface.
Read the authoritative log, R/S/T/U/V, the prior A convolution-window
proof, B's large-sieve proof, and existing normal-cubic/source-first
work. No solver, optical trajectory or archived recovery experiment ran.
New W_two_window_record.md and X_short_record_obstruction.md are exact
mathematical/computer-assisted results, not Lean formalizations.

W reuses A's normalized p-fold box-convolution window at TWO centres.
R's global integer-lattice matching survives with exponentially small
off-line leakage. A coarse main-lobe bound first excludes phase wraps;
comparing the two centred phases then proves final speed error
<=20*alpha/D and initial phase error <=2*pi*alpha, linear rather than
square-root in the line-error ratio alpha. The centred supports span
K=2D+1 samples. Positive weights preserve deterministic hard noise bounds;
no independent-noise or probabilistic argument is used.

S's arbitrary complex sine-tilt continuation gives a uniform per-axis
tilt derivative <25000, wedge-coefficient bound C_A=50000, and intrinsic
complex phase derivative <15500 per rotor throughout R5. These bounds
replace R's larger recurrence constant and its sum over separate Fourier
derivatives. Direct optical phase transport plus B yields a torus error
<=2*tau+2*sqrt(L)*(3*eta+1e6*alpha+2*tau). The complete exact budget
composes this with V's shape input epsilon=1e-45 and proves all eighteen
native coordinates confined within .000573 (rounding within .0004 still
meets .001). Both truth and competitors are R5; only the truth initially
needs wedge floor .1 and the finite speed guard. Candidate zero wedges
and unconstrained competing speeds are handled by the matching proof.

The computed constants are q=3607, p=274, Q=3003436130752,
delta=1/(10000*Q^3), K approximately 3.56733709e58, eta approximately
8.33766459e-69 per axis. The exact K and rational quantities are saved.
The speed guard excludes less than .0033 of native speed-cube volume,
a deterministic volume fact, not a recovery probability. At the SAME
line-match radius and line-error target, the rectangular-window
sufficient K is over 4e79 times larger. This is a comparison of sufficient
proof budgets, not a runtime benchmark or optimal lower bound. W's new
K is still unusable (about 5.65e49 years at 20 Hz). No claim that a
practical short-record problem has been solved follows from this bound.

X proves that the old class also has a REAL obstruction at K=200,
accuracy .001 and hard per-axis noise 1e-6, even with FAST rotors. Take
n_i=4/3, g=5, zero source angles/positions/phases, A1=A2=.1, and define
A3 exactly by the face-from-two-rays identity to cancel the outgoing
direction at the common unit tilt. Exact intervals give
A3 approximately -.198720705061032, strictly between -.25 and -.1.
All wedges are nonzero and meet the declared .1 floor and 18-degree cap.

For coherent independent tilts (A1*u,A2*u,A3*u), an exact dyadic interval
cover proves |t_exit(u)|<1/4000 for ALL u in [-1,1]. The cover has 5628
leaves; every leaf and complete prefix-tree coverage are replayed. A
separate analytic derivative calculation gives sum_i |partial t_exit /
partial gamma_i|<2 throughout R5 (rational bound about 1.60054244).

Choose speeds in the cube centred at (1,1,1) Hz of radius b=1/(100Q).
A finite slab-volume calculation excludes at most 13/50 of this small
cube under W's ENTIRE finite speed test; hence a nonempty open subset
passes strictly. Rationally independent speeds can be constructed in
that subset by a small algebraic perturbation. Thus this example does
not use zero speeds, exact speed collisions or a violated W guard.
All rotors complete almost ten turns, but their RELATIVE phase motion
is extremely small. Their actual exit tangent differs from the coherent
one by at most 2*pi*b*T times the derivative sum.

Keep those same speeds/hardware and take workpiece distances 100 and
100.003. The exact difference of their output coordinates is .003 times
the exiting tangent. Their midpoint observations have per-axis errors
<3.7500000051e-7, so BOTH fit at eta=1e-6. Every estimator has distance
error >=.0015 for one explanation. This proves impossibility of a
uniform .001 guarantee at this allowance over the current class. It
even holds for the whole continuous observation interval, so denser
sampling over the same duration does not fix the ambiguity.

X also records a slow-speed version in the cube about zero: two distances
100 and 101 are compatible at allowance 1/Q, about 3.33e-13, despite
nonzero independent speeds satisfying the same finite guard. The fast
example shows why merely imposing an individual speed floor is not enough.
Useful short-record theory must lower the noise, increase observation of
relative motion, or exclude near-cancelled/near-coherent configurations
through stronger declared conditions. Such exclusions are not yet a
proved sufficient short-record regime. W remains valid with its stated
much longer record and much smaller noise; X does not contradict it.

Reproducers: experiments/p2_theory_2026_09_28/W_two_window_record.py and
X_short_record_obstruction.py. Reports:
experiments/results/p3_broad_2026_09_30/two_window_record.json and
short_record_obstruction.json. Exact checks passed. No production solver,
physical model, formal proof source, main manuscript or git history was
changed. J, R and V link the new scope and obstruction.

## 2026-09-30: coupled fourth-order shape inverse across R5

The user rejected the infeasible raw-record constants and asked for a
better theory. New note: paper/P2_THEORY_2026_09_28/Y_global_quartic_shape.md.
This is pure symbolic and validated continuum work, with no synthetic or
archived optical solve, success rate, or statistical argument.

The new theorem determines all twelve P3 shape parameters and physical
order from 30 coupled scalar coefficients: the twenty X coefficients
through total degree three, six mixed quartics of types (2,2,0) and (2,1,1),
and the Y constant plus three pure quadratics. The unknown hardware varies
over the entire existing R5 family: n in [1.3,1.4], source sines +/-.02,
positions +/-1, g in [2,15], d in [50,200], every nonzero signed wedge
through +/-18 degrees. This is not a full-native-glass extension or a
statement that those coefficients are observations in a 200-point scan.

For X(u1,u2,v)=z(u)+a(u)v+b(u)v^2+c(u)v^3+..., use the two upstream
variables to form the second derivative of a with respect to b at fixed z.
The observable formula is w^T(Ha-k_z Hz-k_b Hb)w, with
w=Jac(z,b)^(-1)(0,1) and (k_z,k_b)=grad(a)Jac(z,b)^(-1).
Multiplication by c cancels the last wedge and distance. The resulting
K=f3[(D^2 f1)(D f2)-(D f1)(D^2 f2)]/(D f2)^3 depends only on F,T,r,
where D differentiates in T at FIXED GLASS and r. At normal incidence and
zero offset it equals twice the previously proved global normal cubic
invariant. The Hessian corrections remove upstream parametrization
curvature at unknown incidence and source position.

(K,V,U) has a global 9/10 contraction inverse over T's entire rectangle
[.3,.401] x [-.021,.021] x [-.061,.061]. The exact integer interval proof
uses 393 leaves with complete cover replay; its certified bound is below
.8996212341496354. It also proves .7<=D f2<=1.1. No isolated fifth-order
coefficient is required. A free symbolic chain-rule check independently
verifies both Hessian correction terms.

Once the last hardware is recovered, the gradients of z,b give the two
entering tangent derivatives T_i. The ratios R_i=z_i/T_i cancel upstream
wedges. Together with 2*C_300/(T_1^3*R1)-1, they form a three-variable map
in F1,F2,g. An exact cancellation of the leading geometry term proves a
uniform contraction bound below .6372354294326613, hence a declared 2/3,
on the whole seven-dimensional box including independent unknown last
context F3,T,q,d. One interval evaluation suffices, without subdivision.
Earlier uncombined expressions were too loose; no failed cover was
promoted to a proof. The final code checks the cancellation symbolically.

The two upstream directions cannot collapse in the normalized response:
4<R1-R2<18. Thus det Jac(z,b)=d*A3^2*(D f2)*T_1*T_2*(R1-R2) is nonzero,
and its magnitude exceeds (63/5)*a0^4 when all wedge sines exceed a0.
U's existing cubic invariant fixes physical order. Explicit restoration
then gives every remaining X-axis shape coordinate; a monotone quadratic
last-prism equation gives the other beam angle and source position.

The note includes invariant-space deterministic error bounds and context
propagation. In the upstream preconditioned weighted norm, context
derivative bounds for F3,T,q,d are .001,1.988,.012,.001. It does not assert
a practical jet-noise threshold, a raw-record runtime, or native .001
accuracy from 200 samples. Both global contractions start from arbitrary
points of their boxes; compatible-input residuals are still required.

This removes the fifth-coefficient/interpolation cascade from the algebraic
shape inversion. The unresolved step is direct, quantitative finite-record
observability of the coupled optical information together with unknown
rotors. Coefficient counts and local Jacobian rank are insufficient; the
free-polynomial obstructions and X's near-coherent finite-window ambiguity
remain valid. No new Lean formalization or hardware implementation is claimed.

Reproducers: experiments/p2_theory_2026_09_28/Y_mixed_curvature_inverse.py
and Y_upstream_geometry_inverse.py. Reports in
experiments/results/p3_broad_2026_09_30/mixed_curvature_inverse.json and
upstream_geometry_inverse.json. Both final scripts pass their symbolic
identities and complete interval checks. The production optical model,
solver, main manuscript, formal sources, and git history are unchanged.

## 2026-09-30: input precision is a constraint, not a proof variable

The user clarified that the input points must not require extreme numbers
of digits. BRIEF.md now separates the accuracy of given information from
internal arithmetic precision and makes fixed bounded input error part
of the research target. Increasing working precision cannot improve the
information content of rounded observations.

An initial assistant-selected working benchmark is hard per-axis input
error eta=1e-8 in native position units, with the existing native parameter
accuracy epsilon=.001. The user did not specify eight decimal places;
this is a concrete research benchmark, not an attributed user requirement
or achieved guarantee. The existing 200-point protocol remains a working
assumption pending any change to what information is supplied. All
measurement and rounding errors must be included in eta.

No new theorem or experiment was claimed in this clarification. In
particular, Y's thirty coupled coefficients are not direct observations
and its invariant-space bounds are not a measurement-noise guarantee.
The new requirement cannot be met by shrinking eta after choosing a proof,
assuming exact derivative data, or reporting only local identifiability.
The precision-constrained global full18 result remains open.

## 2026-10-01: constructive inverses with controlled paired-plane measurements

The user requested a broader constructive attack and rejected another
nearby-speed ambiguity as the main result. The results below explore
additional measurements; the user has not approved replacing the original
passive 200-position protocol. That original finite-precision problem
remains open. No optical recovery trials or production solver changes
were made.

AA_two_screen_angular_inverse.md proves a direct angular inverse for the
canonical per-axis model. Two parallel screen positions at known axial
separation supply an outgoing direction. Exact parked rotor orientations
isolate one effective axis at a time. An all-plate baseline and four
ordinary phase states per rotor recover each glass and signed wedge from
a polynomial of degree at most two and unsquared branch checks. No
optical derivatives, Fourier coefficients or initial estimate are inputs.

AB_two_screen_geometry.md completes that inverse. An affine cancellation
of the baseline position gives an isolated normalized response whose
strict order identifies physical prism order. Distance, common gap and
source position then follow by direct formulas. Fourteen static states
recover the twelve shape coordinates. Six phase-synchronized isolated
runs, retaining two times each, supply the six rotor coordinates. The
full protocol uses 26 optical states, each read at two planes: 52
two-dimensional positions. All paired-plane and axis-isolation repeats
must reproduce the target rotor phase at the same time origin.

Theorem AB6 proves a compatible-pair diameter below .001 in every native
coordinate for the R5 family with |sin(ax_i)|>=.1, screen separation 50
and hard per-screen coordinate allowance eta=1e-8. The family retains
n_i in [1.3,1.4], source direction sines bounded by .02, source positions
bounded by 1, d in [50,200] and g in [2,15]. Parked orientations, screen
separation and replay synchronization are exact calibration inputs.
Explicit diameter bounds are: glass <.000026401, wedge <.000565 degrees,
d <=.000186344, g <.000437146, source position <.000026803, source angle
<=4.64e-8 degrees, initial phase <.000641 degrees and speed <.000089 Hz.
These are global compatible-set bounds under the expanded protocol, not
local fit errors or a guarantee for the original passive observations.

AC_vector_layer_stripping.md gives a separate exact inverse for full
three-dimensional vector Snell optics. This changes the optical model:
the canonical repository model traces the two axes independently. For a
single rotating prism, b-a=(B-v)m and constant |m| imply a quadric in the
five observed features (b_x,b_y,v,v^2,1). Five distinct orientations force
rank four when the incoming transverse direction is nonzero. The recovered
quadric reduces glass, wedge and unknown incidence to one scalar cubic;
the note proves its unique physically admissible root for n>=1.3 and
0<|ax|<=18 degrees. Exact inverse refraction then removes the calibrated
last prism, permitting the same construction for the middle and first.

The same angular rank guarantees a usable pair for the geometry formula
U_j=Q+G V_j. Distance, gap and source position therefore need no extra
states. Fifteen controlled states, or thirty two-dimensional screen
positions, recover all twelve shape coordinates; four rank-checked states
per prism can reduce this to twelve states. Known physical actuator order,
strict transmission and nonzero incoming transverse direction at EACH
sweep are hypotheses. Timed recovered face vectors also determine speeds
and release-time phases under the stated clock and sign convention. This
does not reconstruct an earlier untracked phase origin. The exact rank
theorem alone provides no uniform finite-noise conditioning bound; that
bound is not proved for AC.

Reproducers: experiments/p2_theory_2026_09_28/AA_two_screen_angular_inverse.py,
AB_two_screen_geometry.py and AC_vector_layer_stripping.py. Their exact
symbolic identities and rational inequalities pass. Reports are under
experiments/results/p3_broad_2026_10_01/ and p3_vector_2026_10_01/.
Proofs include the monotonicity, physical-root and compatible-pair
arguments; symbolic checks alone are not asserted to formalize them.
The production forward model, main manuscript and Lean sources are unchanged.

## 2026-10-06: publish the current full-vector inverse at the repository forefront

The user corrected a status summary that treated the practical computation gap
as absence of a completed inverse. Located the newer research in the separate
October 2 full18_research workspace: REPORT.md revision 13, dated October 3.
That work was missing from the Wedge repository's previous entry points.

Imported the scientific source and saved evidence into research/full18/.
Its README is the current guide. The complete finite real-algebraic construction
retains every physical system compatible with the finite record, including
singular and positive-dimensional ambiguities, and provides exact native
recovery-to-tolerance decisions in principle. It applies to coupled full-vector
Snell optics; the earlier independent-axis numerical solvers remain separate.
The original 200 ideal-clock positions enter this construction directly.

The central unresolved issue is useful computational cost and implementation.
The complete compatible-set decomposition and its coordinate projections can
have enormous algebraic complexity. No completed practical full-prior recovery
is newly claimed. The published bounded engine is partial and its saved
full-prior subdivision retains four unresolved leaves. The nineteen Lean
support declarations do not formalize the complete inverse or Python pipeline.
The later correction work, its unstable raw map and inconclusive prescribed box
are preserved with the report's stated limits.

For a nonempty compatible set and an unrestricted estimate, native maximum
error <=.001 is possible exactly
when all eighteen compatible-set diameters are <=.002. The coordinate midpoint
need not itself be compatible; demanding a compatible estimate requires the
additional quantified center test. Completeness includes correct ambiguity and
inconsistency outputs, rather than universal uniqueness on the entire prior.

The GitHub front page and agent handoff now foreground this completed inverse
and its computational bottleneck. Imported scientific files retain their exact
bytes and have SHA-256 entries in research/full18/SOURCE_SNAPSHOT.json. Build
outputs, bytecode and Library transfer metadata are omitted. The original
external workspace is unchanged. This is a publication and navigation update;
it does not add mathematical claims, recovery trials or new Lean verification.
