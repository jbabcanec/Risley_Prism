# Bounded Lean support for the Risley addenda

This isolated folder contains supporting algebraic and inequality proofs for the reduced-algebra and global-boundary addenda. It does not formalize the complete optical model, boundary strictification, global inverse, continuation theorem, noise threshold algorithm, or collision construction. Inward-derivative lemmas verify the stated algebraic derivative expressions and their signs; differentiation itself is not part of those lemmas.

The source imports standard cached mathlib modules only. It imports no theorem or generated code from the existing user Risley project. No original research file is modified and no reconstruction, sampling, or numerical campaign is run.

Final verification passed: all eleven theorem declarations compiled with exit code 0, without errors or warnings. Every individual axiom listing is exactly `[propext, Classical.choice, Quot.sound]`, the standard Lean logical foundations used by mathlib. There are no admitted goals, `sorryAx` dependencies, or custom axioms. The earlier `compiler-output-failed-draft.txt` is retained as superseded development provenance only; its failed goals are not checked results.

The final source SHA256 is `402d3f1555dabe0639f54d8af43a770f30ed0a134ac6308846f4444e74d573e8`; the successful `.olean` SHA256 is `9bd201bd7d13e0c241a26ef1906e58c275717c31f755d89b992732aef880e68b`; the final compiler log SHA256 is `9502f23a92627676e8463aa4602683e4234a24555167c12a03e4c33242fe9778`. The machine-readable `verification.json` also records file sizes, the check-script hash, and the verification timestamp. The script checked that the source hash stayed unchanged throughout compilation.

## Toolchain and reproduction

- Installed compiler: Lean 4.33.0, x86_64-w64-windows-gnu, commit `d8b18978322de05a8f3dba51ef03cf5461676c17` (Release).
- Existing mathlib dependency: tag `v4.33.0`, manifest commit `db584cd6d46c92f209a44c0f1c829460d327499d`.
- Compiler executable: `C:\Users\josep\.elan\toolchains\leanprover--lean4---v4.33.0\bin\lean.exe`.
- Read-only package cache: `C:\Users\josep\lean\risley\.lake\packages`.
- `check.ps1` sets `LEAN_PATH` from existing package build directories, invokes that executable directly, and writes all output here. It does not invoke elan resolution, Lake updates, installation, or downloads.

Run from the research workspace in PowerShell:

```powershell
& '.\full18_research\formalization\check.ps1'
```

The effective compiler command is `lean.exe -o RisleySupport.olean RisleySupport.lean`; the script supplies absolute paths. The recorded compiler output is `compiler-output.txt`. Successful exit and the final axiom inventory are the evidence for the checked results. `#print axioms` is included for every theorem so custom assumptions cannot be hidden behind imported local theorem declarations.

## Exact checked statements

All scalar variables below range over the real numbers. The Lean source gives the complete statements and proofs. Let `a=(a1,a2)`, `v=(v1,v2)`, `u=(u1,u2)`, `d_a=1-u·a`, and `d_v=1-u·v`. The matrix `transport a v u` is `I+(a-v)uᵀ/d_a`, with this formula expanded componentwise in the source.

| Lean theorem | Statement and assumptions |
| --- | --- |
| `entrance_det` | `det(I-a uᵀ)=1-u·a`, without assumptions. |
| `transport_factorization` | `transport a v u * (I-a uᵀ)=I-v uᵀ`, assuming `d_a ≠ 0`. |
| `transport_det` | `det(transport a v u)=d_v/d_a`, assuming `d_a ≠ 0`. |
| `transport_inverse` | `transport a v u * transport v a u=I`, assuming `d_a ≠ 0` and `d_v ≠ 0`. Swapping `a,v` gives the reverse product too. |
| `transport_det_pos` | The transport determinant is positive, assuming `d_a>0` and `d_v>0`. |
| `combined_spatial_recurrence` | One exact scalar component of the two-segment position formula equals the combined rational numerator, assuming `H-dotX ≠ 0`, `Z ≠ 0`, and `T ≠ 0`. `alpha`, `dotX`, `Xi`, `Yi`, `Ai` are arbitrary real scalars representing `u·A`, `u·X`, and the corresponding vector components. The proof is therefore stronger than its optical specialization. |
| `critical_inward_identity` | `-2(a(H-a)+b nu)=-2(H-a)(H b+a)/(1+b)`, assuming `1+b ≠ 0` and `(H-a)^2=(1+b)nu`. |
| `critical_inward_negative` | The left side of the preceding identity is negative, additionally assuming `H-a>0` and `(H b+a)/(1+b)>0`. In the addendum, `a=u·X`, `b=|u|²`, and these are the critical branch guards. |
| `external_flight_inward_identity` | `-H(H b+3a)/(H-a)^2=-H ell/(H-a)`, assuming `H-a ≠ 0` and `ell=(H b+3a)/(H-a)`. |
| `external_flight_inward_negative` | The preceding quantity is negative, assuming `H-a>0`, `H>0`, `ell>0`, and the same active-flight equality. |
| `sqrt_difference_bound` | `|sqrt(s)-sqrt(t)| ≤ sqrt(|s-t|)`, assuming `s≥0` and `t≥0`. This is the square-root modulus used by the boundary hierarchy and global error recurrence. |

The optical interpretation still requires the separately proved model identities and physical guards. Lean does not infer those hypotheses for the full parameter prior here. These supporting proofs make no assertion about practical inverse runtime, an actual record's conditioning, or publication readiness.
