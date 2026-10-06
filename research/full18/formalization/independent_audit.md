# Independent audit of the bounded Lean support

The frozen [source](RisleySupport.lean) compiled successfully with Lean 4.33.0. The [compiler log](compiler-output.txt) records exit code 0 and no warnings or errors. This audit independently read the source, script, log and dependency manifest, matched the source/object hashes to the successful log, counted all theorem declarations and axiom listings, and checked the statements against the optical claims they support. It did not edit or rerun the proof source.

There are **eleven** checked theorem declarations, not ten. Each has an individual `#print axioms` result. No `sorry`, `admit`, custom `axiom` declaration, `sorryAx` dependency, or other unproved custom axiom occurs in the claimed results. The failed draft log is superseded provenance and is not evidence of verification.

## Compiler and evidence

- Lean version: 4.33.0, x86_64-w64-windows-gnu, Release; compiler commit `d8b18978322de05a8f3dba51ef03cf5461676c17`.
- Executable: `C:\Users\josep\.elan\toolchains\leanprover--lean4---v4.33.0\bin\lean.exe`.
- Existing mathlib manifest: tag `v4.33.0`, revision `db584cd6d46c92f209a44c0f1c829460d327499d`.
- Imports are standard mathlib modules for real square roots, matrix determinants and proof tactics. No original Risley-project theorem is imported.
- [check.ps1](check.ps1) uses the existing cached package build paths and records unchanged source SHA256 across compilation plus object SHA256. No installation, dependency update, or numerical campaign is involved.
- The actual successful command is the direct Lean invocation shown in the log, with absolute source/output paths. Standard output and error are combined in that log.

| Artifact | Independently verified SHA256 |
| --- | --- |
| `RisleySupport.lean` | `402D3F1555DABE0639F54D8AF43A770F30ED0A134AC6308846F4444E74D573E8` |
| `RisleySupport.olean` | `9BD201BD7D13E0C241A26EF1906E58C275717C31F755D89B992732AEF880E68B` |
| `compiler-output.txt` | `9502F23A92627676E8463AA4602683E4234A24555167C12A03E4C33242FE9778` |
| `check.ps1` | `ED69F8CB840F0DE3D6177C9944B59397C9EB682F1BFF473B877E04B3BC0BC2EA` |

The axiom dependencies below are Lean/mathlib's standard foundational axioms: propositional extensionality, classical choice, and quotient soundness. They are not additional optical assumptions. Axiom-free verification is not claimed.

## Per-theorem assumptions and dependencies

Every scalar parameter has type real. Write \(d_a=1-u_1a_1-u_2a_2\), \(d_v=1-u_1v_1-u_2v_2\). The source defines the two-by-two matrix
\[
\operatorname{transport}(a,v,u)=I+(a-v)u^\top/d_a.
\]
The inverse interpretation of this expression is proved under its stated denominator hypotheses; its definition alone is not an optical-model axiom.

| Theorem | Complete scalar hypotheses; conclusion scope | Printed axioms |
| --- | --- | --- |
| `entrance_det` | No hypotheses; determinant of \(I-au^\top\) is \(d_a\). | `propext, Classical.choice, Quot.sound` |
| `transport_factorization` | \(d_a\ne0\); the transport multiplied by \(I-au^\top\) equals \(I-vu^\top\). | `propext, Classical.choice, Quot.sound` |
| `transport_det` | \(d_a\ne0\); determinant of transport equals \(d_v/d_a\). | `propext, Classical.choice, Quot.sound` |
| `transport_inverse` | \(d_a\ne0\), \(d_v\ne0\); transport times reversed transport equals the identity. The reverse product follows by swapping \(a,v\), rather than being a separately counted declaration. | `propext, Classical.choice, Quot.sound` |
| `transport_det_pos` | \(d_a>0\), \(d_v>0\); transport determinant is positive. | `propext, Classical.choice, Quot.sound` |
| `combined_spatial_recurrence` | \(H-\mathrm{dotX}\ne0\), \(Z\ne0\), \(T\ne0\); one scalar component of the two-segment expression equals the displayed combined rational numerator. Other scalar variables are unrestricted. | `propext, Classical.choice, Quot.sound` |
| `critical_inward_identity` | \(1+b\ne0\), \((H-a)^2=(1+b)\nu\); the critical derivative expression equals \(-2(H-a)(Hb+a)/(1+b)\). | `propext, Classical.choice, Quot.sound` |
| `critical_inward_negative` | The preceding two hypotheses, \(H-a>0\), \((Hb+a)/(1+b)>0\); that derivative expression is strictly negative. | `propext, Classical.choice, Quot.sound` |
| `external_flight_inward_identity` | \(H-a\ne0\), \(\ell=(Hb+3a)/(H-a)\); \(-H(Hb+3a)/(H-a)^2=-H\ell/(H-a)\). | `propext, Classical.choice, Quot.sound` |
| `external_flight_inward_negative` | \(H-a>0\), \(H>0\), \(\ell>0\), the same active-flight equality; the preceding expression is strictly negative. | `propext, Classical.choice, Quot.sound` |
| `sqrt_difference_bound` | \(s\ge0\), \(t\ge0\); \(\lvert\sqrt{s}-\sqrt{t}\rvert\le\sqrt{\lvert s-t\rvert}\). | `propext, Classical.choice, Quot.sound` |

The table's hypotheses are exactly those of the source; their complete Lean signatures are preserved below. No namespace-level variables or hidden instance assumptions add optical restrictions. `set_option autoImplicit false` disallows undeclared implicit variables.

## What these checks do and do not establish

The transport identities apply to arbitrary real transverse slopes under explicit denominator guards. Applying them to the optical model still requires identifying \(a=X_{\rm in}/H\), \(v=X_{\rm out}/Z\), and proving the branch guards. The combined spatial identity treats `alpha`, `dotX` and the vector components as arbitrary real scalars; identifying them with optical dot products is a separate substitution, so the algebraic theorem has fewer assumptions than its optical specialization.

The critical and external-flight lemmas verify formulas and signs under their listed equalities and inequalities. They contain no `HasDerivAt` claim: the differentiation that produces those expressions is not formalized. In particular, \(b=|u|^2\) in the critical lemma and \(b=u\cdot p\) in the external-flight lemma are different optical substitutions; the abstract scalar \(b\) is not a globally identified model coordinate.

The square-root modulus is a genuine inequality theorem, but the complete three-prism Hölder recurrence, hierarchical strictification, weak-set closure equality, projected covering/noise-cube theorem, five-root compiler degree bound, geometry-circuit elimination, collision construction and full inverse remain outside this Lean formalization. This audit establishes the stated supporting checks, not full-model proof-assistant verification, useful actual-record conditioning, practical inverse runtime, or publication readiness.

## Exact Lean theorem signatures

These signatures are extracted without alteration from the successfully compiled source whose hash is recorded above. The code block omits proof bodies only.


```lean
theorem entrance_det (a₁ a₂ u₁ u₂ : ℝ) :
    Matrix.det !![1 - a₁*u₁, -a₁*u₂; -a₂*u₁, 1-a₂*u₂] =
      1 - (u₁*a₁ + u₂*a₂)

theorem transport_factorization (a₁ a₂ v₁ v₂ u₁ u₂ : ℝ)
    (ha : 1-(u₁*a₁+u₂*a₂) ≠ 0) :
    transport a₁ a₂ v₁ v₂ u₁ u₂ *
      !![1-a₁*u₁, -a₁*u₂; -a₂*u₁, 1-a₂*u₂] =
      !![1-v₁*u₁, -v₁*u₂; -v₂*u₁, 1-v₂*u₂]

theorem transport_det (a₁ a₂ v₁ v₂ u₁ u₂ : ℝ)
    (ha : 1-(u₁*a₁+u₂*a₂) ≠ 0) :
    Matrix.det (transport a₁ a₂ v₁ v₂ u₁ u₂) =
      (1-(u₁*v₁+u₂*v₂))/(1-(u₁*a₁+u₂*a₂))

theorem transport_inverse (a₁ a₂ v₁ v₂ u₁ u₂ : ℝ)
    (ha : 1-(u₁*a₁+u₂*a₂) ≠ 0)
    (hv : 1-(u₁*v₁+u₂*v₂) ≠ 0) :
    transport a₁ a₂ v₁ v₂ u₁ u₂ * transport v₁ v₂ a₁ a₂ u₁ u₂ = 1

theorem transport_det_pos (a₁ a₂ v₁ v₂ u₁ u₂ : ℝ)
    (ha : 0 < 1-(u₁*a₁+u₂*a₂))
    (hv : 0 < 1-(u₁*v₁+u₂*v₂)) :
    0 < Matrix.det (transport a₁ a₂ v₁ v₂ u₁ u₂)

theorem combined_spatial_recurrence
    (H Z dotX alpha Xi Yi Ai T ell : ℝ)
    (hP : H-dotX ≠ 0) (hZ : Z ≠ 0) (hT : T ≠ 0) :
    Ai/T + Xi*(3+alpha/T)/(H-dotX) +
      (Yi/Z)*(ell - (alpha/T + dotX*(3+alpha/T)/(H-dotX))) =
    (Z*(H-dotX)*Ai + (Z*Xi-H*Yi)*alpha +
      3*(Z*Xi-dotX*Yi)*T + ell*(H-dotX)*Yi*T) /
      (Z*(H-dotX)*T)

theorem critical_inward_identity (H a b nu : ℝ)
    (hb : 1+b ≠ 0)
    (hc : (H-a)^2 = (1+b)*nu) :
    -2*(a*(H-a)+b*nu) = -2*(H-a)*((H*b+a)/(1+b))

theorem critical_inward_negative (H a b nu : ℝ)
    (hb : 1+b ≠ 0) (hc : (H-a)^2 = (1+b)*nu)
    (hP : 0 < H-a) (hZ : 0 < (H*b+a)/(1+b)) :
    -2*(a*(H-a)+b*nu) < 0

theorem external_flight_inward_identity (H a b ell : ℝ)
    (hP : H-a ≠ 0) (hactive : ell = (H*b+3*a)/(H-a)) :
    -H*(H*b+3*a)/(H-a)^2 = -H*ell/(H-a)

theorem external_flight_inward_negative (H a b ell : ℝ)
    (hP : 0 < H-a) (hH : 0 < H) (hell : 0 < ell)
    (hactive : ell = (H*b+3*a)/(H-a)) :
    -H*(H*b+3*a)/(H-a)^2 < 0

theorem sqrt_difference_bound (s t : ℝ) (hs : 0 ≤ s) (ht : 0 ≤ t) :
    |Real.sqrt s - Real.sqrt t| ≤ Real.sqrt |s-t|
```
