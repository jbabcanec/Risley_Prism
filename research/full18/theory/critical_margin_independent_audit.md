# Independent audit: data-dependent critical margins

Audit date: 2026-10-03. Target: `data_dependent_critical_margin.md`.

Verdict: the final-stage exclusion and conditional upstream certificate are mathematically valid under the stated original weak physical domain and downstream guards. No full-vector record-specific certificate has been evaluated. The locally available canonical per-axis record is not a substitute for a full-vector record.

## Final-stage margin and signs

The exact plane relation is ell-u·p_next=(R/Z)B, with0<=B<=3+ell and R/Z>=0 on the weak domain. Therefore (R/Z)(3+d)>=d-u·F, without dividing by B; the proof remains valid at zero external traversal. The record bound u·F<=M and monotonicity of(d-M)/(3+d) in d give the stated ratio lower bound max(0,(50-M)/53).

For0<=mu<50/53, C_mu=50-53mu is positive and1-mu is nonnegative. These signs make every product in identity(8) nonnegative under R<=mu Z. They also justify the positive-side squaring in(9). Omitting C_mu>0 would make the proposed squared criterion invalid.

The exact epsilon cutoff is the positive root of2epsilon²+2(a+b)epsilon+a²+b²-S². The condition a²+b²<S² guarantees a positive root. The strict cutoff and its excluded endpoint are correctly distinguished. These statements concern success of this sufficient certificate, not the largest possible physical noise allowance.

The lower bound on H/P and hence on the final source-transport determinant is valid using|X_in|<=1 and H>=sqrt(1.3²-1). A uniform ratio bound, together with the preexisting positive axial bound, supplies a positive normal-root bound. It supplies no upstream margin by itself.

The optional phase-aware support bound is correct. At samples0 and1, the allowed signed rotor sector reduces the maximization to max_{0<=theta<=alpha}(A cos(theta)+B sin(theta)), giving precisely the stated piecewise formula. At later samples, the signed sectors cover all directions. The piecewise support expression is separate from the simpler disk-based epsilon formula and product certificate.

## Upstream transport and finite corners

Backward transport through already-guarded downstream prisms gives an affine function of the two distances and two screen-error components. The current prism's inverse is not used, so its own R=0 is permitted in this test.

For fixed optics, the ratio expression in(13) is a linear-fractional function with a strictly positive affine denominator. Its value at any point in the distance rectangle is a positive denominator-weighted average of the four vertex values. This proves the four-vertex minimum. Equivalently, clearing the positive denominator gives an affine inequality in the four variables(g,d,e_x,e_y), so all sixteen rectangle corners suffice. Universal validity over the chosen optical superset remains a separate substantive obligation.

The reverse step(15) is correct. With R as an independent unit-ray graph variable and the relation R=Z-u·X_out retained, its denominator HR has degree2 and its matrix/vector numerators degree at most3. One backward step from y+epsilon sigma has numerator/denominator degrees at most(4,2), and two steps at most(7,4). Multiplication by the current normal gives the stated cleared degrees at most5 and8. These bounds do not describe dense native-chart elimination.

The suggested polynomial product/sum-of-squares identity is a sound way to verify a supplied sign certificate. No bounded-degree existence or efficient discovery of that certificate has been proved; the draft correctly keeps this limitation.

## Boundary retention

The separately established hierarchical strictification theorem converts an algebraically encoded weak R=0 point into strict physical systems at every tolerance strictly above its weak residual. Continuity also makes R arbitrarily small, proving absence of any positive uniform normal margin at that larger tolerance. The residual-equality endpoint still needs a separate feasibility test.

A single direct strict physical witness has a narrower implication: if it satisfies a specified near-critical threshold, it retains that target set. It does not by itself prove absence of every positive margin. This wording clarification was sent to the author for the final retention bullet.

## Reproducible checks

`proof_checks/audit_critical_margin_identities.py` checks identities(8),(9),(15) and the epsilon root(7) by exact symbolic expansion. All assertions passed. No broad numerical test or parameter campaign was performed.
