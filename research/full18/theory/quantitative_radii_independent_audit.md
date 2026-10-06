# Independent audit: finite radii/noise certificate and long-lag extraction

## Scope and current verdict

This audit checks the finite a posteriori theorem, its arithmetic derivative certificate, and the proposed DC-constrained long-lag extraction. The baseline is `two_variable_algebraic_correction.md` and `algebraic_defect_correction.md`; the quantitative draft `quantitative_algebraic_correction.md` has also received an equation-by-equation pass.

The scalar radii polynomial, the mixed-Hessian majorants, the frozen-left-inverse recurrence, and the long-lag alias construction are mathematically sound subject to the explicit domain and branch conditions below. None of these facts certifies an actual nonzero finite-wedge radius without completing all of those conditions and passing the scalar inequalities for specific data, scales, radii, and wedge sizes.

Final revised-draft verdict: pass at the stated conditional scope. The draft has incorporated the DC factor and strict Rouche criterion, native wedge-scaling qualification, scalar endpoint cases, symmetric mixed-Hessian interpretation, and output-lift noise propagation. A final purely notational correction keeps the Rouche disk label j fixed and uses n for its polynomial-power summation index, as written in this audit.

## 1. Finite radii/noise theorem

Use scaled coordinates `x=S^{-1}(theta-theta0)` and define

    G(x,h)=S^{-1}[A(y+h-R1(theta0+Sx),
                       y+h-R2(theta0+Sx))-theta0].

All norms in this section are real infinity norms, with induced bilinear norms for Hessians. Assume `G` is C2 on a neighborhood of the closed product `||x||<=r`, `||h||<=epsilon_max`; the same A branch and all physical and algebraic guards must hold throughout this product. Let verified upper bounds satisfy

    ||G(0,0)|| <= Y,
    ||G_x(0,0)|| <= z0,
    ||G_h(0,0)|| <= L0,
    ||G_xx|| <= Mtt, ||G_xh|| <= Mty, ||G_hh|| <= Myy.

Taylor expansion first in h at x=0, then in x at fixed h, gives

    ||G(x,h)|| <= Y + L0 epsilon + Myy epsilon^2/2
                  + (z0+Mty epsilon)r + Mtt r^2/2.

Also

    ||G_x(x,h)|| <= z0+Mtt r+Mty epsilon = q_epsilon.

Thus the proposed radii inequality `P<=0` and `q_epsilon<1` certify self-mapping and uniform contraction for every record `y+h` in the chosen measurement ball. Strict `P<0` is sufficient but unnecessary. There is exactly one fixed point in the scaled parameter box for each such record, and iteration from any point of that box converges to it. The two inequalities are sufficient; neither can generally be omitted merely because the other passes.

For exact compatible physical solutions, additionally require the matching left-inverse identity

    A(F1(theta),F2(theta))=theta

on the selected physical chart in the parameter box. A smooth root branch unrelated to the physical root does not suffice. Under the matching condition, any physical theta in the box with `F(theta)=y+h` equals the corresponding fixed point. The fixed point need not itself satisfy the full 400-coordinate record. This is an at-most-one compatible-solution statement unless compatibility is independently validated.

### Explicit noise allowance

Fix all majorants on the already verified `(r,epsilon_max)` domain. Set

    C=r-Y-z0 r-Mtt r^2/2,  B=L0+Mty r.

For `C>0` and `Myy>0`, the positive root of the radii polynomial is

    epsilon_P = 2C/[B+sqrt(B^2+2 Myy C)].

This is algebraically correct and avoids cancellation. Under `P<=0` the noise root itself is allowed, provided contraction remains strict. For `Myy=0, B>0`, use `C/B`. For `Myy=B=0`, the radii polynomial imposes no additional noise restriction. If `C=0`, only epsilon zero is allowed unless `B=Myy=0`; if `C<0`, no nonnegative epsilon passes. In every case retain the independently certified domain cap and the contraction condition, including `epsilon<(1-z0-Mtt r)/Mty` when `Mty>0`. If the majorants are recomputed as functions of epsilon, the displayed root is not a closed-form allowance until their uniform validity on the resulting interval is checked.

For the draft's noiseless radius formula, the correct interval is `r_minus<=r<(1-Z)/Mxx`, restricted to positive radii within the verified domain. The upper endpoint gives contraction bound one and is excluded. When `Mxx=0`, retain `Z<1` together with `Y<=(1-Z)r`.

### Tube around the nominal fixed point

Put `q0=z0+Mtt r`. If `theta_hat_h` and `theta_hat_0` denote the fixed points for `y+h` and y, respectively, a slightly sharper valid tube is

    ||S^{-1}(theta_hat_h-theta_hat_0)||
      <= [(L0+Mty r)||h||+(Myy/2)||h||^2]/(1-q0).

To prove it, compare `G(x_h,0)` and `G(x_0,0)` using the nominal contraction, and integrate the record perturbation at fixed `x_h`. Replacing `q0` with the larger uniform `q_epsilon` is safe. A simpler derivative-based sensitivity bound using `(L0+Mty r+Myy epsilon)/(1-q_epsilon)` is also safe but more conservative.

## 2. Hessian chain rule and finite derivative compiler

Let the A first- and second-partial bounds include the native output scale `S^{-1}`. Let

    Bi >= ||D Ri S||,  Ci >= ||D^2 Ri[S,S]||

on the physical parameter box. With notation `As,Aw,Ass,Asw,Aww` for the corresponding A derivative bounds, the following majorants are correct:

    Mtt <= Ass B1^2 + 2 Asw B1 B2 + Aww B2^2
           + As C1 + Aw C2,
    Mty <= (Ass+Asw)B1 + (Asw+Aww)B2,
    Myy <= Ass+2 Asw+Aww.

The minus signs from `-Ri` disappear only after taking upper bounds. Center derivatives should retain the actual sums whenever possible, particularly `A_s+A_w` for shared measurement noise. A bound on two independently perturbed records is safe but can lose useful cancellation.

Finite operation-level interval or ball arithmetic is a genuine implementation of these bounds. It must include:

- exact finite-angle F and the actual Taylor-ring F1/F2, not substituted asymptotic powers;
- all unit conversions, phase/frequency argument derivatives, native wedge sine and inverse sine conversions;
- certified inverse bounds for every matrix solve and root Jacobian;
- scalar denominator, square-root, sign, and argument-chart margins;
- a uniform real simple-root enclosure for the two beam equations;
- verified center enclosures, making Y, z0, and L0 upper bounds rather than floating-point estimates.

The corrected inputs lie in the explicit product enclosure

    ||s-s0|| <= B1 r+epsilon,  ||w-w0|| <= B2 r+epsilon,
    s0=y-R1(theta0), w0=y-R2(theta0).

Checking A on that product is sufficient, though a coupled enclosure may be much sharper. The compiler can require high-dimensional arithmetic enclosures; it does not hide a high-dimensional nonlinear root solve or optimization. It is allowed to return failure or unresolved when dependency loss makes a bound too broad.

## 3. DC-constrained long-lag recurrence

For `1<=ell<=28`, form the stacked rows from all indices `0<=k<=199-7ell`, constrain the degree-seven recurrence by `p(1)=0`, and write

    c=cbar+R a, cbar=-(1/7)1, R^T R=I, R^T 1=0,
    J_s=H_s R, u_s=v_s+H_s cbar.

Both J and u are invariant under independent constant offsets of the two real output channels. This exactly eliminates the baseline. Require `rank J_s=6`; the lag range alone does not imply this. In particular lag 28 has only four sample positions per real output, and stacked rank still needs verification. Non-DC lag nodes must be distinct and separated from 1 for the selected simple six-root chart.

### Frozen left inverse

Let D be frozen on the entire record/noise branch with `D J_c=I`, and define

    a(s)=-(D J_s)^{-1}D u_s,  a_c=-D u_c.

This is a valid alternative off-model map. On exact first-order records the true recurrence is recovered whenever `D J_s` is invertible. For a record ball of radius d, define fixed linear-operator bounds

    mu >= ||D (D_s J)||,
    gamma >= ||D D_s(u+J a_c)||.

Because J and u are affine in s and `D(u_c+J_c a_c)=0`, the proposed estimates are valid when `mu d<1`:

    Delta_a <= gamma d/(1-mu d),
    ||D_s a|| <= (gamma+mu Delta_a)/(1-mu d),
    ||D_s^2 a|| <= 2mu ||D_s a||/(1-mu d).

The operator norms can be bounded by finite coefficient row sums; no optimization is necessary. An approximate floating left inverse cannot silently be treated as exact. A proposal P may be normalized exactly by `D=(P J_c)^{-1}P`, with certified interval evaluation, or its defect must explicitly enter the Neumann denominator.

The draft's crude choices `mu<=7||D||inf||R||inf` and `gamma<=||D||inf(1+||c_c||1)` are safe upper-bound recipes. The first follows from the seven entries of each perturbed Hankel row; the second uses the scalar filter `v+H c_c`.

### Root disk certificate

For the degree-seven monic polynomial, the Rouche lower product must include all seven center roots, including the known DC root 1. On a disk of radius delta around an exact center root `z_j`, the sufficient test is strictly

    delta_c sum_(n=0)^6 (|z_j|+delta)^n
      < delta product_(l!=j)(|z_j-z_l|-delta).

Every product factor must be positive. If numerical disk centers approximate the exact center roots, subtract the certified center-root errors as well. The six retained disks are separated from one another, the DC root, and the real axis. Together with `p_s(1)=0`, this accounts for every root; derivative-product bounds must likewise include DC. Strictness of Rouche is separate from the allowed non-strict self-map inequality.

### Aliases and full-record fitting

For a chosen representative of each conjugate lag-root pair, retain every lift

    nu=20[Arg(w)+2pi m]/(2pi ell)

inside the signed original speed interval. Use `|nu|` in the real cosine/sine design and recover physical signed speed from the fitted determinant orientation. This preserves positive original frequencies whose lagged node lies in the negative-argument half-plane. Taking only positive original lifts of a positive-argument lag root is incomplete. Deduplicate conjugate and permutation copies while retaining physical prism assignments.

On exact regular first-order records, full-200-sample fitting resolves different unsigned lift sets: the union of two seven-tone supports has at most thirteen distinct sampled nodes, so a nonzero difference cannot vanish for 200 consecutive samples. Jointly nonzero mode coefficients are essential. On arbitrary off-model/noisy records, a lowest residual does not prove that a discarded lift is impossible. Preserve all not soundly excluded lift branches and apply the finite certificate branchwise.

## 4. Spectral certificate script

The inspected script is `proof_checks/certify_long_lag_spectral_margins.py`. It uses the existing rational witness, exact interval trigonometric enclosures at rational phase angles, double-precision matrix proposals, and interval defect verification. No sweep or wedge-radius experiment is involved.

For a proposal P and exact rectangular matrix A, with `E=I-PA`, the exact corrected left inverse is

    D=(I-E)^{-1}P.

The infinity-norm bound `||D||inf<=||P||inf/(1-||E||inf)` is valid. Likewise the Frobenius bounds on P and E safely imply a spectral inverse bound and hence a lower bound for the smallest singular value of A. The DC-constrained lag-11 J0 construction in the script matches the first-order witness formulas and its Helmert basis is appropriate.

For the complex demixer, the corrected target row obeys

    ||ell||1 <= ||P_target||1
                 + ||E_target||1 ||P||inf/(1-||E||inf).

This is valid, but the resulting row is the exact-left-inverse row `e_target^T(PV)^{-1}P`, not generally the Moore-Penrose row `(V*V)^{-1}V*`. To use its numerical constant for A, explicitly adopt the former locally with P frozen and variable V, or state the constant only as existence of an alternate row. That alternate row preserves the confluent annihilation identities and their differentiated cancellation, since it remains an exact left inverse throughout its verified domain.

The initially inspected script used ordinary floating arithmetic in its final rational combinations of interval upper endpoints. Those combinations have now been moved inside interval arithmetic. Every published bound should use these corrected outward enclosures and sufficient threshold slack.

Finally, `sum |ell_k|` is the operator norm from a complex-modulus infinity input norm. Real coordinatewise measurement noise satisfies `|hx+i hy|<=sqrt(2)epsilon`. Include that conversion, or compute the exact real projected row norm `sum(|Re a_k|+|Im a_k|)` when the output is a real quadrature. The quantitative draft explicitly adopts the frozen-proposal exact demixer, so the corrected-row certificate has the right map scope.

## 5. Integral remainder and thirteen-coordinate composition

The draft's equation (14) is the exact integral Taylor remainder for the auxiliary sine-wedge scaling lambda. Its factors integrate to `1/2` for the second lambda derivative and `1/6` for the third. Differentiating under the integral is legitimate on a compact physical analytic branch with all forward-model roots and denominators uniformly guarded for `0<=lambda<=1`.

An order-three lambda jet carrying second parameter jets computes the required mixed derivatives without constructing a full five-way eighteen-variable tensor. The resulting operation bounds are valid for any specified native scale S. However, preservation of the asymptotic kappa powers after parameter differentiation additionally requires scaled wedge components of S to be `O(kappa)`. Unscaled native wedge derivatives can remove powers of kappa. This qualification must accompany any order statement; it does not invalidate direct finite bounds for arbitrary S.

For the optional composition `Phi(x,y)=pi T(E(x,y),y)`, the first derivative formulas and the xx/xy Hessian formulas in (24)-(25) are correct. The yy Hessian, written explicitly on arbitrary directions h and k, is

    Phi_yy[h,k] = pi[T_yy[h,k]
                   + T_thetay[E_y h,k] + T_thetay[E_y k,h]
                   + T_thetatheta[E_y h,E_y k]
                   + T_theta E_yy[h,k]].

The draft's factor two is diagonal-quadratic/symmetrized shorthand; its factor-two norm bound is valid. It must not be read as twice a nonsymmetric bilinear term on two different directions.

The scalar theorem for Phi directly bounds thirteen retained coordinates. Returning a full18 noise tube additionally requires propagation through the output lift:

    ||E(x_hat_h,y+h)-E(x_hat_0,y)||
      <= E_x_bound ||x_hat_h-x_hat_0|| + E_y_bound ||h||,

with all norms and derivative bounds carrying the declared input and native output scales and domains. The lift's record derivative is needed both inside Phi and in this final output reconstruction. The subresultant and geometry-pivot guards remain additional proof obligations.

## 6. What has not been certified

Normalized spectral constants at one witness are meaningful ingredients, but do not establish the nonlinear two-beam-root inverse bounds, physical margins, all remainder Hessians, or the radii/noise inequalities for a particular nonzero wedge size. A numerical finite-angle basin is established only after those remaining finite checks pass. No conclusion about the full 18-degree prior, global uniqueness, completeness of a candidate enumeration, or compatibility of an unvalidated fixed point follows from this audit.
