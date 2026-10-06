# Independent audit: operational chord rule and native 0.001 target

## Verdict

Sections 5–7 of `principled_corrector_acceleration.md` give a valid **conditional, operational certificate**. The exact chord inverse has zero nominal center derivative. A fixed finite inverse approximation must retain its verified center defect. The inverse enclosure, transformed radius/noise inequalities, literal native tolerance box, and delivered-estimate error decomposition are sound.

This is a symbolic audit only. No chord matrix, relaxed map, new wedge, new physical case, convergence basin, or noise budget was numerically tested. The preceding undamped-instability certificate remains a separate result.

Two precision clarifications were requested and verified in the revised source: the finite approximation's center defect is included in the consecutive-iterate error denominator, and the thirteen-coordinate output-lift derivatives explicitly include S13. The formulas are detailed below. The final Sections 5–7 pass this audit.

## 1. Automatically defined exact inverse versus its enclosure

Let Q0=D_x Psi(0,0) for the nominal record, A0=I-Q0, and W be one fixed numerical proposal. If the verified residual satisfies

    E=I-W A0,    ||E||inf <= e < 1,

then W A0 is invertible. Since the matrices are square, both W and A0 are invertible, and

    P=A0^-1=(I-E)^-1 W.

The proposed inverse is therefore a computational aid; the exact mathematical P does not depend on which successful W encloses it. The Neumann formulas

    P_m=sum_(j=0)^m E^j W,
    ||P-P_m||inf <= e^(m+1)||W||inf/(1-e),
    ||P||inf <= ||W||inf/(1-e)

are correct, including when Q0 and E are evaluated through outward interval enclosures. The interval must enclose the exact nominal Q0; substituting an unverified midpoint is insufficient.

Freeze P across both state x and record perturbation h. Then

    Psi_P(x,h)=x+P[Psi(x,h)-x],
    D_x Psi_P(0,0)=I-P(I-Q0)=0.

This exact cancellation holds even if the candidate is not a fixed point and Psi(0,0) is nonzero. The nonzero center update is still charged by Y_P. Since P is nonsingular, the preconditioned and original maps have exactly the same fixed points on their shared domain.

A finite matrix P_tilde instead gives

    Z_tilde >= ||I-P_tilde(I-Q0)||inf.

Its center derivative is not declared zero. Its own center bounds are Y_tilde=upper(||P_tilde Psi(0,0)||) and L_tilde=upper(||P_tilde D_h Psi(0,0)||), with its own certified matrix norm in the Hessian bounds. A passing contraction condition implies Z_tilde<1 and hence also verifies nonsingularity of P_tilde. Alternatively, an interval implementation may enclose the intended exact-P update using the inverse tail as numerical evaluation error; it must actually propagate that error rather than silently identify P_tilde with P.

## 2. Transformed curvature and finite-radius test

For fixed P, every second derivative of the affine identity terms vanishes. Thus bounds

    Hxx=p Mxx,    Hxh=p Mxh,    Hhh=p Mhh

with p>=||P|| are valid. Computing actual matrix-times-Hessian expressions before taking norms can sharpen them, without changing the theorem.

Taylor expansion at (0,0) gives

    ||Psi_P(x,h)|| <= Y_P+L_P epsilon+(Hhh/2)epsilon^2
                     +Hxh r epsilon+(Hxx/2)r^2

and

    ||D_x Psi_P(x,h)|| <= Hxx r+Hxh epsilon.

The displayed self-map and strict-contraction inequalities therefore follow. They require all derivative bounds and every physical/root/reconstruction guard on the entire state/data envelope. A center inverse certificate alone does not establish that envelope or make the constants small.

For P_tilde, add Z_tilde r to the self-map majorant and Z_tilde to the contraction majorant. All other constants must be the transformed constants for that actual P_tilde. This is essential, not an optional safety factor.

## 3. Literal native-unit box and independent noise

Setting S=diag(tau_i), tau_i=0.001, and r=1 means the centered domain

    |theta_i-theta0_i| <= 0.001

for each coordinate in its original units. These are half-widths: Hz for speeds, degrees for all native angles, dimensionless refractive indices, and native model lengths for the geometry coordinates. Bounds certified in another chart or scale must be converted or regenerated before using this choice.

For exact P the self-map condition becomes

    Y_P+(L_P+Hxh)epsilon+(Hhh/2)epsilon^2+Hxx/2 <= 1,

and contraction requires Hxx+Hxh epsilon<1. With

    C=1-Y_P-Hxx/2,    B=L_P+Hxh,

the stable quadratic-root expression

    epsilon <= 2C/[B+sqrt(B^2+2Hhh C)]

is correct when C>0 and Hhh>0. It must also obey the independently declared epsilon_bar, the strict contraction condition, and every domain guard. The constants cannot change silently when epsilon is selected. The earlier theorem supplies the linear and zero-coefficient cases.

For P_tilde replace C by 1-Y_tilde-Z_tilde-Hxx/2 and contraction by Z_tilde+Hxx+Hxh epsilon<1. No relation between the physical wedge scale and measurement epsilon is needed or implied.

## 4. Error of the actual delivered iterate

For exact P let q0=Hxx<1 on the nominal unit box. Comparison of the perturbed and nominal fixed points gives

    r_noise <= [(L_P+Hxh)epsilon+(Hhh/2)epsilon^2]/(1-q0).

This uses the same record perturbation in both algebraic-corrector inputs. The true system need only be compatible with the original observation strips; the nominal fixed point need not itself fit the entire record.

For an exact nominal iterate, the residual bound

    ||x_m-x_hat(0)|| <= ||Psi_P(x_m,0)-x_m||/(1-q0)

is valid while x_m is in the certified box. This is a consecutive-iterate bound when the next exact update is available. Using a previous consecutive difference divided by 1-q0 is also conservative. Numerical evaluation errors and inverse-tail errors must be added explicitly.

For a fixed P_tilde, **q0=Z_tilde+Hxx**, and both the numerical-iteration and noise denominators must be 1-Z_tilde-Hxx. It would be incorrect to retain 1-Hxx in the numerical-iteration example for this case.

Combining verified r_num with r_noise yields

    |theta_m,i-theta_star,i| <= tau_i(r_num+r_noise).

Therefore r_num+r_noise<=1 certifies the stated local delivered-estimate accuracy. Merely putting two points in the same +/-0.001 box would only give a possible 0.002 separation; the separate delivered-error test correctly avoids that mistake.

## 5. Thirteen-coordinate output lift and all eighteen errors

The lift must include its direct data dependence. To state the scaling unambiguously, write retained native coordinates as xi=xi0+S13 x, and define

    H(x,h)=E_(y+h)(xi0+S13 x).

For every full native output coordinate i, certify on the necessary envelope

    a_(i,a) >= |[D_xi E_(y+h) S13]_(i,a)|,
    b_i >= sum_j |[D_y E_(y+h)]_(i,j)|.

If r_num,a and r_noise,a bound the reduced **scaled** coordinate errors, and delta_eval,i bounds final lift evaluation error, then

    |theta_m,i-theta_star,i|
      <= sum_a a_(i,a)(r_num,a+r_noise,a)
         + b_i epsilon + delta_eval,i.

Require this to be <=0.001 separately for all eighteen outputs. If derivatives are instead tabulated against retained native coordinates xi, convert the reduced coordinate error bounds to native units first. Do not mix the two conventions.

These bounds must cover the subresultant denominator, geometry-pivot inverse, positive square root, and their data dependence. The reduced corrector's own state/data derivatives must also include the lift composition terms already required by the earlier memo. A thirteen-coordinate contraction alone does not certify five additional native coordinate errors.

## 6. What the theorem still requires

The checklist in Section 7 correctly retains the actual record/error model, data-derived candidate, retained aliases and root charts, complete domain guards, exact-center derivative enclosure, transformed center/curvature bounds, finite iteration error, all observation residuals, and coverage or explicit unresolved status for other original-prior branches.

A successful inverse residual certifies a linear preconditioner. A successful radius/noise test certifies one local iteration. Only the additional delivered-error and branch-coverage conditions can turn it into the requested recovery guarantee. None of those numerical successes is asserted by this symbolic rule alone.
