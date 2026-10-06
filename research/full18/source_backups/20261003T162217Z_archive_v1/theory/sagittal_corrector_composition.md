# Optional sagittal reconstruction inside the two-beam corrector

## Scope

This short note composes two established constructions; it does not add a new numerical witness or assert a larger certified convergence region. The last squared index and four geometry variables can be reconstructed exactly on the sagittal subresultant chart. The off-model two-beam algebraic corrector updates all18 coordinates. Composing them gives an explicit thirteen-coordinate iteration with computable derivatives. New denominator, branch, and derivative bounds are still required. Scalar radii majorants belong to the companion corrector work, not to this note.

## 1. Fixed reconstruction chart and guards

Write x for the thirteen retained optical coordinates and w=(b_x,b_y,g,d). Choose one fixed four-row sagittal geometry pivot P contained in the six samples used by the two norm charts. Let the first subresultant be a(x,y)G+b(x,y), and write those four sagittal rows as

M(x,G)w+c(x,G;y)=0.

Define

G_y(x)=−b(x,y)/a(x,y),
w_y(x)=−M(x,G_y(x))^−1 c(x,G_y(x);y),
n_3(x,y)=sqrt(G_y(x))&gt;0.

Let E_y(x) insert n_3 and w into their native full18 positions while leaving x unchanged, and let pi project full18 parameters onto x. Thus pi E_y(x)=x.

For a differentiable correction branch, keep the pivot choice fixed and verify throughout its domain:

- a is bounded away from zero;
- M is nonsingular with a verified inverse bound;
- G stays in the original positive squared-index range;
- E_y(x) lies inside the original parameter and strict physical domain;
- both corrected records lie in the selected off-model algebraic corrector branch, including frequency, root, label, and denominator guards.

At an exact physical solution with a!=0, at least one of the two fixed charts has a rank-four geometry pivot, by the simple-root/Bezout corollary in exact_rational_last_index_recovery.md. Local continuity then supplies a fixed valid pivot around that point. This does not supply useful uniform lower bounds without further certification.

For arbitrary off-image or noisy data, E_y is an algebraic continuation. It enforces the first-subresultant equation and the four selected sagittal rows, but need not satisfy either full norm equation, the remaining sagittal rows, or the physical record. An exact solution in its chart is represented by E_y; the reverse implication requires full validation.

## 2. Explicit reconstruction derivatives

All differentials below are taken on this fixed chart. Since aG+b=0,

DG=−(G Da+Db)/a.

For x variation at fixed y,

D_x w=−M^−1[(M_x+M_G D_xG)w+c_x+c_G D_xG],
D_x n_3=D_xG/(2sqrt(G)).

The same formula applies to y variation with partial y derivatives. M has no direct dependence on y, so

D_y w=−M^−1[M_G(D_yG)w+c_y+c_G D_yG],
D_y n_3=D_yG/(2sqrt(G)).

Here products such as M_G(D_xG) are the ordinary chain-rule contractions. Partial derivatives of the norm coefficients and subresultants can be obtained by differentiating their fixed arithmetic circuits; no parameter optimization or numerical differentiation is required. D_y E has support only in the first six paired screen samples used to form the index formula and pivot.

These formulas expose the new potentially large factors 1/a and M^−1. Exact elimination does not make their sensitivity disappear.

## 3. Reduced corrector and Jacobian

Let the established two-beam corrector be

T_y(theta)=A17+1(y−R1(theta),y−R2(theta)),

with its selected off-model branch. Define the thirteen-coordinate iteration

Phi_y(x)=pi T_y(E_y(x)).

Every exact physical solution theta*=E_y(x*) in the chart remains a fixed point, because T_y(theta*)=theta*. A fixed point of Phi_y for arbitrary data need not solve the full record; the reduced iteration has discarded correction output coordinates and retains redundant residuals only through subsequent validation.

Its exact derivative is

D_x Phi_y=pi [D_theta T_y](E_y(x)) D_x E_y(x)
           =−pi [A_s D_theta R1+A_w D_theta R2] D_x E_y.

Thus the appropriate native-scaled contraction matrix is the thirteen-by-thirteen product above. A bound for the original eighteen-coordinate corrector alone does not automatically bound this product below one; the reconstruction derivative can magnify it. Conversely, exploiting its actual matrix structure may yield a sharper bound than separate norm estimates. A contraction theorem additionally requires an invariant retained-coordinate set Phi_y(X) subset X; physical validity of E_y(X) alone does not imply this. No improvement is claimed until the derivative product and this self-mapping condition are certified.

This removes five algebraic parameter coordinates from the iteration. It does not remove the physical weak quadratic information direction: that direction can survive in the retained optics and in the reconstruction's data sensitivity.

## 4. Shared-noise continuation derivative

The same observed record enters both corrector slots and E_y. At fixed x,

D_y Phi_y=pi [(A_s+A_w)+(D_theta T_y) D_y E_y].

The second term must not be omitted. A noisy compatible theta* generally equals E_(y_true)(x*), not E_y(x*) for the perturbed reported record. Consequently a compatible-set error bound for the composed iteration must control this full record derivative, plus the output reconstruction derivatives D_x E and D_y E. Reusing the original corrector's noise constant alone would be unsound.

The existing scalar radii/majorant framework can use the displayed state and data matrices once all domains are enclosed. The present note supplies their exact formulas and guards only. It makes no finite-noise accuracy, global basin, or all-prior convergence claim.

## 5. Why the same step does not immediately eliminate n_2

The last-prism identity succeeds because its outgoing endpoint is the observed screen and the upstream transverse direction is already known conditional on the retained optics. At prism2, the next-flat endpoint is unobserved. Pulling it backward through prism3 introduces q_2(n_2), the third transmitted-root branch, and the recovered n_3 expression. The resulting coefficients no longer have the same simple shared-radical form sqrt(G_2−known alpha_k).

A generic norm/resultant elimination is still algebraically possible, but no analogous low-degree physical invariant has been established. Repeating the witness merely to demonstrate another formal elimination would not resolve this obstruction. The justified next use is certifying the displayed reduced-corrector matrix on an existing candidate branch, with the new subresultant and pivot guards retained.
