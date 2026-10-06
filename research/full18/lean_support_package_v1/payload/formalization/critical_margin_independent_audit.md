# Independent audit of the final-prism Lean margin certificate

The separate [CriticalMargin.lean](CriticalMargin.lean) package passed terminal verification with **eight** theorem declarations, eight individual axiom inventories, compiler exit code **0**, and no compiler errors or warnings. This audit independently read the final source, [compiler output](critical-margin-compiler-output.txt), [check script](check-critical-margin.ps1), [verification metadata](critical-margin-verification.json), and mathematical [critical-margin memo](../theory/data_dependent_critical_margin.md). Source and object hashes match the successful log. No proof source was edited or rerun by this audit.

All eight declarations depend exactly on `propext`, `Classical.choice` and `Quot.sound`. These are standard Lean/mathlib foundational axioms. There is no `sorry`, `admit`, custom axiom declaration, or `sorryAx` dependency in the claimed results. This is not a claim of axiom-free mathematics.

## Exact result and noncircular proof chain

For one final-prism sample, put \(u=(u_x,u_y)\), \(p=(p_x,p_y)\), and let \((X,Y,Z)\) be the outgoing direction. The main theorem assumes precisely:

- \(Z>0\), \(R\ge0\), and \(R=Z-(u_xX+u_yY)\).
- \(B=d-(u_xp_x+u_yp_y)\), \(B\ge0\), and \(3+u_xp_x+u_yp_y\ge0\).
- \(F_x=p_x+(X/Z)B\), \(F_y=p_y+(Y/Z)B\).
- \(d\ge50\), \(u_*\ge0\), \(u_x^2+u_y^2\le u_*^2\).
- \(\lvert F_x-y_x\rvert\le\epsilon\) and \(\lvert F_y-y_y\rvert\le\epsilon\), with the same real \(\epsilon\). These inequalities already imply \(\epsilon\ge0\).

From these hypotheses it proves
\[
\frac RZ\ge\max\!\left(0,\frac{50-u_*
 \sqrt{(\lvert y_x\rvert+\epsilon)^2+
       (\lvert y_y\rvert+\epsilon)^2}}{53}\right).
\]

The desired bound is not assumed. The proof first derives
\[
 Z(d-u\cdot F)=RB,\qquad 0\le B\le3+d.
\]
It then derives \(u\cdot F\le M\) from the two-dimensional Lagrange identity, the slope bound, and the measured-coordinate bounds. Finally it uses
\[
 \frac{d-M}{3+d}\ge\frac{50-M}{53},
 \qquad (d-50)(M+3)\ge0,
\]
and \(R/Z\ge0\). The abstract scalar helper assumes a combined identity, but the final linked theorem discharges that hypothesis from the individual ray equations. No desired critical margin or positive lower bound on \(R\) appears among the premises; \(R=0\) is allowed when the observational bound permits it.

The lack of an upper bound on \(d\) is deliberate: the statement holds for all \(d\ge50\), hence in particular for the original interval \([50,200]\). Likewise the proof is a general real-coordinate implication, not restricted to one fitted system.

## Per-theorem hypotheses and axiom inventory

All variables have type real. The exact Lean signatures are included below.

| Theorem | Hypotheses and conclusion | Individual printed axioms |
| --- | --- | --- |
| `final_plane_identity` | \(Z\ne0\), both outgoing intersection equations, and the definitions of \(B,R\); derives \(Z(d-u\cdot F)=RB\). | `propext, Classical.choice, Quot.sound` |
| `weak_flight_bounds` | Definition of \(B\), nonnegative internal height \(3+u\cdot p\), and \(B\ge0\); derives \(0\le B\le3+d\). | `propext, Classical.choice, Quot.sound` |
| `lagrange_identity` | No hypotheses; proves \((u\cdot F)^2+(u_xF_y-u_yF_x)^2=(u_x^2+u_y^2)(F_x^2+F_y^2)\). | `propext, Classical.choice, Quot.sound` |
| `screen_box_dot_bound` | \(u_*\ge0\), squared slope bound, \(\lvert F_x\rvert\le a_x\), \(\lvert F_y\rvert\le a_y\); derives \(u\cdot F\le u_*\sqrt{a_x^2+a_y^2}\). Nonnegative \(a_x,a_y\) are derived from these bounds. | `propext, Classical.choice, Quot.sound` |
| `scalar_ratio_margin` | \(\kappa\ge0\), \(M\ge0\), \(d\ge50\), \(B\le3+d\), \(d-\mathrm{dot}=\kappa B\), and \(\mathrm{dot}\le M\); derives \(\max(0,(50-M)/53)\le\kappa\). This scalar helper does not require a lower bound on \(B\). | `propext, Classical.choice, Quot.sound` |
| `final_prism_ratio_margin` | All physical/slope premises above, plus arbitrary screen-coordinate absolute bounds \(a_x,a_y\); derives the ratio bound with \(M=u_*\sqrt{a_x^2+a_y^2}\). | `propext, Classical.choice, Quot.sound` |
| `observation_screen_bound` | One residual bound \(\lvert F-y\rvert\le\epsilon\); derives \(\lvert F\rvert\le\lvert y\rvert+\epsilon\). | `propext, Classical.choice, Quot.sound` |
| `final_prism_observation_margin` | Exactly the complete list of physical/slope/observation premises above; derives the displayed record-dependent ratio bound. | `propext, Classical.choice, Quot.sound` |

## Provenance and preserved work

Compiler: Lean 4.33.0, Windows x86_64, commit `d8b18978322de05a8f3dba51ef03cf5461676c17`. Existing mathlib manifest: tag `v4.33.0`, revision `db584cd6d46c92f209a44c0f1c829460d327499d`. The direct command is `lean.exe -o CriticalMargin.olean CriticalMargin.lean` with absolute paths and the existing cached package search path; it imports no original-project or local theorem module.

| New artifact | Independently verified SHA256 |
| --- | --- |
| `CriticalMargin.lean` | `D711B8B89CC89C9AFC86C6426C0136A03C2295D598774F2B58C5133EBC49FDB7` |
| `CriticalMargin.olean` | `EBC3251B6A6118B94A75BA5C3D5183552BD447DCFE7A773DE5E96E92B5648B35` |
| `critical-margin-compiler-output.txt` | `1A57194D1F3C66EA1FD89F2460134B23BEB41428D51D12CAAF36D630D20FD9F9` |
| `check-critical-margin.ps1` | `283912886F459922A367EC6BF4EA74A4433FCC0DAE79B3D167348A4B57FEF401` |

The earlier eleven-lemma package is preserved: independent hash checks confirmed that `RisleySupport.lean`, `RisleySupport.olean`, `compiler-output.txt`, `check.ps1`, and [independent_audit.md](independent_audit.md) still match their previously recorded values. The current package adds eight declarations; it does not replace or reclassify the earlier eleven.

## Limits of the formalization

This proves the final-prism implication under explicit, physically sound hypotheses. It does not formalize the full three-prism Snell trace or prove that every member of the native prior supplies these hypotheses. In particular, specializing \(u_*=\tan(\pi/10)\), deriving its slope bound from the wedge prior, and establishing the original branch/axial guards remain model-level instantiations outside this Lean file. The proof needs the normal definition and nonnegativity of \(R\); it does not independently derive those facts from a selected Snell square root.

The theorem can be applied separately to every sample satisfying its premises. No particular 200-sample record, global compatible set, or positive resulting numerical margin has been certified. The exact noise-cutoff formulas, sharpened angular-sector support, upstream exposure inequalities, global boundary compactification, complete inverse and useful practical conditioning remain outside this bounded formalization.

No numerical campaign, original-project execution, software installation or dependency update was performed for this audit.

## Exact Lean theorem signatures

The following signatures are extracted without alteration from the frozen successfully compiled source above; only the proof bodies are omitted.


```lean
theorem final_plane_identity
    (ux uy px py X Y Z B d R Fx Fy : ℝ)
    (hZ : Z ≠ 0)
    (hFx : Fx = px + (X/Z)*B)
    (hFy : Fy = py + (Y/Z)*B)
    (hB : B = d - (ux*px+uy*py))
    (hR : R = Z - (ux*X+uy*Y)) :
    Z*(d-(ux*Fx+uy*Fy)) = R*B

theorem weak_flight_bounds (ux uy px py B d : ℝ)
    (hB : B = d-(ux*px+uy*py))
    (hInternal : 0 ≤ 3+(ux*px+uy*py))
    (hExternal : 0 ≤ B) :
    0 ≤ B ∧ B ≤ 3+d

theorem lagrange_identity (ux uy Fx Fy : ℝ) :
    (ux*Fx+uy*Fy)^2 + (ux*Fy-uy*Fx)^2 =
      (ux^2+uy^2)*(Fx^2+Fy^2)

theorem screen_box_dot_bound (ux uy Fx Fy ax ay uStar : ℝ)
    (huStar : 0 ≤ uStar)
    (hu : ux^2+uy^2 ≤ uStar^2)
    (hFx : |Fx| ≤ ax) (hFy : |Fy| ≤ ay) :
    ux*Fx+uy*Fy ≤ uStar*Real.sqrt (ax^2+ay^2)

theorem scalar_ratio_margin (kappa B d dot M : ℝ)
    (hkappa : 0 ≤ kappa) (hM : 0 ≤ M) (hd : 50 ≤ d)
    (hBupper : B ≤ 3+d)
    (hIdentity : d-dot = kappa*B)
    (hDot : dot ≤ M) :
    max 0 ((50-M)/53) ≤ kappa

theorem final_prism_ratio_margin
    (ux uy px py X Y Z B d R Fx Fy ax ay uStar : ℝ)
    (hZ : 0 < Z) (hRnonneg : 0 ≤ R)
    (hFx : Fx = px + (X/Z)*B)
    (hFy : Fy = py + (Y/Z)*B)
    (hB : B = d-(ux*px+uy*py))
    (hR : R = Z-(ux*X+uy*Y))
    (hInternal : 0 ≤ 3+(ux*px+uy*py))
    (hExternal : 0 ≤ B)
    (hd : 50 ≤ d)
    (huStar : 0 ≤ uStar)
    (hu : ux^2+uy^2 ≤ uStar^2)
    (hScreenX : |Fx| ≤ ax) (hScreenY : |Fy| ≤ ay) :
    max 0 ((50-uStar*Real.sqrt (ax^2+ay^2))/53) ≤ R/Z

theorem observation_screen_bound (F y epsilon : ℝ)
    (hError : |F-y| ≤ epsilon) : |F| ≤ |y|+epsilon

theorem final_prism_observation_margin
    (ux uy px py X Y Z B d R Fx Fy yx yy epsilon uStar : ℝ)
    (hZ : 0 < Z) (hRnonneg : 0 ≤ R)
    (hFx : Fx = px + (X/Z)*B)
    (hFy : Fy = py + (Y/Z)*B)
    (hB : B = d-(ux*px+uy*py))
    (hR : R = Z-(ux*X+uy*Y))
    (hInternal : 0 ≤ 3+(ux*px+uy*py))
    (hExternal : 0 ≤ B)
    (hd : 50 ≤ d)
    (huStar : 0 ≤ uStar)
    (hu : ux^2+uy^2 ≤ uStar^2)
    (hErrorX : |Fx-yx| ≤ epsilon) (hErrorY : |Fy-yy| ≤ epsilon) :
    max 0 ((50-uStar*Real.sqrt ((|yx|+epsilon)^2+(|yy|+epsilon)^2))/53) ≤ R/Z
```
