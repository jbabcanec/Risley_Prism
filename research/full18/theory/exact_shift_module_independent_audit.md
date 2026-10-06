# Independent audit: exact shift-module obstruction

Audited document: `exact_shift_module_obstruction.md`.

Audit date: 2026-10-03.

## Verdict

**Pass within the document's stated scope.** The physical construction, function-field obstruction, generic 200-sample Hankel consequence, finite-record persistence for nearby fully active systems, and Section 5's 25-slot nonlinear relation are mathematically sound. The native-phase conversion in equation (2) was corrected during review and has been checked as `w_0 = exp(i pi phi/180)` for phase measured in degrees.

No simulations, reconstruction sweeps, or numerical determinant evaluations were used. This is an independent algebraic and analytic verification, not a conditioning or runtime certificate. Earlier memos were not modified by this audit.

## Checked arguments

- The two flat slabs preserve the external direction and give the stated entrance coordinate of the final prism. Formula (1) includes the exact tilted exit-plane intersection and outgoing propagation.
- All native bounds and physical branch inequalities hold. In particular, the displayed radicand bound implies `E > 3/4`, while `C > -1/30`, giving `C+E > 43/60 > 0`. The final flight and internal traversal also have the stated strict margins.
- The nondegenerate Möbius dependence and equation (3) establish `C(w,f) = C(w,E)`. Positivity of `B_0` and `q+Hu` excludes physical denominator cancellation.
- The four positive simple roots of the radicand, the infinite order of the sampled rotor, and disjoint rotated root sets give independent square classes by valuations. This proves degree `2^K` and the claimed linear independence over `C(w)`.
- The newest shift occurs only at the bottom-right corner of the Hankel matrix. Its independent quadratic extension makes the determinant induction valid. The physical analytic branch exists on an annulus; a nonzero determinant therefore has finitely many zeros on the compact allowed phase arc.
- The rank-100 matrix uses samples 0 through 198 and excludes recurrence order below 100 across the original 200-sample record. Dense-orbit continuation proves the separate infinite-sequence statement for the constructed subfamily. Continuity proves the stated finite-record result near fully active interior systems; it is not used to extend the infinite-sequence claim to that whole neighborhood.

## Section 5 degree and support check

Both `L` and `M` are linear in `X`, so `M^2-RL^2` has `X`-degree at most two. With the document's `b_0,b_1`, the coefficients of `u^6` in `M^2` and `RL^2` both equal

`b_1^2 q^2`.

Their coefficients of `u^5` both equal

`-2 b_0 b_1 q^2 - 2 b_1^2 Hq + 2 b_1 q^3 X`.

Thus the `u`-degree is at most four. The `X^2` coefficient is exactly `P^2(C^2-R)`. Using `H^2+q^2=9/4` gives

`C^2-R = D[q^2+H^2 a^2-1+2Hqu]`,

which has `u`-degree one. The `X^2` coefficient therefore has degree at most three. Substitution of `u=a(w+w^(-1))/2` gives the respective Laurent support bounds `[-4,4]`, `[-4,4]`, and `[-3,3]`, totaling at most 25 coefficient slots. The resulting polynomial is nonzero: its `X^2` coefficient is not the zero polynomial.

Consequently the true-speed 200-by-25 feature matrix has a nonzero right kernel. Clearing negative rotor powers yields polynomial necessary rank-defect conditions in the trial rotor node. This check does not establish nonvacuity, finitely many trial nodes, sufficiency of the condition, physical branch recovery, or unique reconstruction. The document properly withholds those conclusions.

## Limits retained

The exponential degree statement is functional over `C(w)`, not a number-field degree claim after numerical specialization. Lift dimensions refer to the stated vector spaces or multiplicatively closed algebras over that base field. Exact nonsingularity supplies no useful singular-value bound. The result obstructs a fixed finite linear shift representation; it does not obstruct nonlinear implicit relations, compact arithmetic circuits, or all structured exact inverse methods. It does not solve the full18 inverse or establish a computational lower bound for that problem.
