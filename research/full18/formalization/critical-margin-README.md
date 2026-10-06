# Lean proof of the final-prism observation margin

This separate package checks the main final-prism ratio inequality from `theory/data_dependent_critical_margin.md`. It preserves the previous eleven-lemma package without edits. It imports cached mathlib modules only, with no original-project or other local proof imports.

Final verification passed: eight theorem declarations, compiler exit code 0, no errors or warnings. Every individual axiom inventory is exactly `[propext, Classical.choice, Quot.sound]`; there are no admitted goals or custom axioms. The superseded `critical-margin-failed-draft.txt` is development provenance only and is not verification evidence.

Frozen source SHA256: `d711b8b89cc89c9afc86c6426c0136a03c2295d598774f2b58c5133ebc49fdb7`. Successful object SHA256: `ebc3251b6a6118b94a75ba5c3d5183552bd447dcfe7a773de5e96e92b5648b35`. Final compiler-log SHA256: `1a57194d1f3c66ea1fd89f2460134b23beb41428d51d12caaf36d630d20fd9f9`. The original package's source, object, compiler log and check script still match its previously recorded hashes.

The main theorem `RisleyCriticalMargin.final_prism_observation_margin` proves

```text
R/Z >= max(0, (50-uStar*sqrt((|yx|+epsilon)^2+(|yy|+epsilon)^2))/53).
```

Every physical and observational assumption is explicit:

- All quantities are real. `p_exit=(px,py)`, outgoing transverse direction is `(X,Y)`, its axial component is `Z`, and plane slope is `(ux,uy)`.
- `Z>0`, `R>=0`, and `R=Z-(ux*X+uy*Y)`.
- `B=d-(ux*px+uy*py)`; weak traversal is `B>=0` and `3+(ux*px+uy*py)>=0`.
- The actual screen coordinates obey `Fx=px+(X/Z)*B` and `Fy=py+(Y/Z)*B`.
- `d>=50`, `uStar>=0`, and `ux^2+uy^2<=uStar^2`.
- The measured record obeys `|Fx-yx|<=epsilon` and `|Fy-yy|<=epsilon`, using the same symbolic epsilon. These hypotheses imply `epsilon>=0`; no additional sign premise is needed.

The proof derives the combined identity `Z*(d-u·F)=R*B` from the two coordinate intersection equations; it does not assume that combined identity in the final theorem. It derives `B<=3+d` from traversal and derives the dot-product bound from the 2D Lagrange identity and the measured-coordinate bounds. The scalar step uses `(d-50)*(M+3)>=0` to prove the required monotonicity. No upper bound on `d` is needed, so the theorem applies in particular to the original interval `[50,200]`.

## Theorem inventory and hypotheses

The complete exact Lean signatures are in `CriticalMargin.lean`. There are eight declarations:

| Theorem | Exact scope |
| --- | --- |
| `final_plane_identity` | Derives `Z*(d-(ux*Fx+uy*Fy))=R*B` from `Z!=0`, both intersection equations, and the displayed definitions of `B,R`. |
| `weak_flight_bounds` | Derives `0<=B` and `B<=3+d` from the definition of `B`, internal weak traversal, and external weak traversal. |
| `lagrange_identity` | Proves `(u·F)^2+(ux*Fy-uy*Fx)^2=(ux^2+uy^2)*(Fx^2+Fy^2)` for arbitrary real components. |
| `screen_box_dot_bound` | Derives `u·F<=uStar*sqrt(ax^2+ay^2)` from `uStar>=0`, the squared slope-disk bound, and `\|Fx\|<=ax`, `\|Fy\|<=ay`. The latter premises imply nonnegative `ax,ay`. |
| `scalar_ratio_margin` | From `kappa>=0`, `M>=0`, `d>=50`, `B<=3+d`, `d-dot=kappa*B`, and `dot<=M`, derives `max(0,(50-M)/53)<=kappa`. The lower bound on `B` is not required for this scalar implication. |
| `final_prism_ratio_margin` | Links all physical equations and guards to the ratio conclusion using arbitrary screen-coordinate box bounds `ax,ay`. |
| `observation_screen_bound` | From `\|F-y\|<=epsilon` derives `\|F\|<=\|y\|+epsilon`. |
| `final_prism_observation_margin` | Links the original coordinatewise observation inequalities directly to the displayed main bound, with every physical hypothesis listed above. |

Each declaration has its own `#print axioms` command. `critical-margin-compiler-output.txt` records their individual inventories and the terminal compiler status. `critical-margin-verification.json` records hashes, sizes and exact toolchain information.

## Formal scope and limits

This is a proof of the conditional final-prism certificate from explicit real-coordinate premises. It does not formalize the complete three-prism forward trace, prove all prior ray guards, or instantiate `uStar=tan(pi/10)` and its trigonometric positivity/prior slope bound. Those facts remain inputs supplied by the mathematical model. It proves a statement for every sample satisfying the stated premises; a concrete 200-sample trace or record has not been verified. Upstream exposure certificates, global inversion, threshold sharpness, useful conditioning, and physical validation remain outside this package.

No numerical campaign, original project code, installation, or download is involved. The existing toolchain/cache is used read-only:

- Lean 4.33.0, Windows x86_64, commit `d8b18978322de05a8f3dba51ef03cf5461676c17`.
- mathlib manifest tag `v4.33.0`, commit `db584cd6d46c92f209a44c0f1c829460d327499d`.
- Direct compiler path and all imported package locations are the same as the frozen earlier package.

Reproduce from the workspace using:

```powershell
& '.\full18_research\formalization\check-critical-margin.ps1'
```

The script invokes `lean.exe -o CriticalMargin.olean CriticalMargin.lean` with absolute paths and a `LEAN_PATH` built from the pre-existing cached package build directories. It records combined stdout/stderr, exit code, source SHA256 before/after compilation, and successful object SHA256. It makes no elan or Lake resolution/update call.
