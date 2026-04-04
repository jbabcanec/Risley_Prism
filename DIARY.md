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
