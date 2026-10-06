# Independent audit: the fixed native box and automatic center inverse

## Verdict

**PASS for the physical/native-domain guards and the exact automatic center inverse.** These are independently checked partial results for the one unchanged witness at kappa=1/10, on the predeclared all18 native box theta0+[-1/1000,1/1000]^18. They do **not** prove a corrected-record branch domain, a finite-box contraction, a positive observation-noise allowance, or successful recovery from unknown data.

The source result audited is `proof_checks/native_box_guards_chord_center.json`, with SHA-256

    29374ab7437fc5f4f5bf2d58cbf095730440d771988238dcf13aa4d93d96ed98

The audit independently reproduces

    ||I-W(I-Q_native)||_inf <= 1.9387182380847987e-9,
    Neumann tail per entry <= 2.9220450229824003e-22,
    ||(I-Q_native)^-1||_inf <= 40099.85616680308.

No original files were modified. No iterates, smaller target boxes, alternative wedges, damping values, new optical cases, or installations were used.

## What was checked

The new script `proof_checks/audit_native_box_guards_chord_center.py` uses outward mpmath intervals at 100 decimal digits and exact Python rational arithmetic. Its result is `proof_checks/audit_native_box_guards_chord_center.json`.

1. All 23,637 saved interval records were decoded from their exact internal dyadic tuples. Their stated binary64 lower and upper endpoints outwardly contain the exact dyadics. Every tuple's mantissa bit count was checked.
2. The source record contains exactly 200 sample times and three prisms, once each, for both the original box and its common lambda analytic extension. All 21,600 saved guard inequalities (two domains times 600 pairs times 18 guards) have strictly positive exact lower endpoints. Every reported minimum is a valid downward bound on the corresponding saved guard family.
3. The exact center, original native box, original priors, and native-to-eta box conversion were independently instantiated at higher precision and are contained in the saved enclosures.
4. The prior point derivative's recorded SHA-256 matches its actual file. The native matrix was independently rebuilt from that derivative and analytic coordinate Jacobians. The saved native matrix contains this reconstruction.
5. The frozen numerical inverse proposal's JSON entries agree with its hexadecimal binary64 entries. The left-residual enclosure, residual norm, proposal norm, Neumann tail, whole-inverse norm, and sharper entrywise norm were checked. Scalar norm/tail comparisons use exact rational arithmetic.
6. A separate unit-ray/unit-normal optical implementation passes every same-domain sample/prism pair. It does not use the source Q-scaled optical recursion.

The previous independent point-derivative audit establishes the imported Q_eta's model/branch meaning. This new audit checks its file identity and conversion into the present native certificate; it does not unnecessarily rerun the prior full point-derivative compiler.

## Native coordinates, priors, and nonlinear conversion

The native order is

    (N1,N2,N3,a1_deg,a2_deg,a3_deg,phi1_deg,phi2_deg,phi3_deg,
     n1,n2,n3,beta_x_deg,beta_y_deg,b_x,b_y,g,d).

The priors agree with `oblique_inverse.md`: speeds in [-3.5,3.5] Hz; wedges and phases in [-18,18] degrees; indices in [1.3,1.8]; beam angles in [-25,25] degrees; source coordinates in [-5,5]; g in [2,15]; and d in [50,200]. All 18 box coordinates are strictly interior. Decimal prior endpoints are exact rationals, not silently replaced by binary approximations.

The beam convention is t_a=tan(pi beta_a/180), component by component. The native refractive indices are independently variable; the auxiliary h coordinates are consequently

    h_j=sqrt(n_j^2+(n_j^2-1)|t|^2).

This is the correct nonlinear conversion. Treating h as an independent native index or omitting its dependence on the beam would produce the wrong box. The source optical calculation instead uses n and t directly, so it does not falsely shrink the domain by discarding this dependence. The saved eta box is only an outward coordinate image.

Angles, phases, and wedge sines retain their exact degree/radian and sine/arcsine conversions. The source point has all wedge sines 1/10 and native wedges (180/pi)asin(1/10); no equality between a degree tolerance and a sine tolerance is assumed.

## Physical normalization and independent optical verification

Write Q_opt=1+|t|^2, reserving Q_native for the iteration derivative. The source transverse X and axial Z are external unit-ray components multiplied by sqrt(Q_opt). At a glass exit with plane slope u and D=1+|u|^2, source P=H-u dot X and Delta=P^2-D(n^2-1)Q_opt give

    incoming glass unit-normal projection = P/(n sqrt(Q_opt D)),
    physical unit-normal exit Snell radicand = Delta/(Q_opt D),
    outgoing unit-normal projection = sqrt(Delta)/sqrt(Q_opt D),
    outgoing unit axial component = Z/sqrt(Q_opt).

These distinctions are correct in the source script and memo. In particular Delta/Q_opt is not the unit-normal Snell radicand; it still uses an unnormalized normal. The source additionally exports the properly normalized quantity.

The independent optical circuit starts with the actual unit air ray, uses the unit surface normal

    v=(-e cos(gamma),-e sin(gamma),sqrt(1-e^2)),

and glass optical momentum m=(q_x,q_y,sqrt(n^2-|q_xy|^2)). Writing C=m dot v, it computes R=1-n^2+C^2 and

    q_out=m-[(n^2-1)/(C+sqrt(R))] v.

Travel is checked independently with internal slopes, positive exit-intersection denominator, actual internal axial distance, and remaining external axial distance. All 600 pairs pass on the original box and all 600 pass on the analytic common-lambda extension e_j(lambda)=lambda sin(a_j), lambda in [0,1]. The extension is used only as an analytic remainder domain, not as a substitute witness.

For the original box, independently reproduced downward bounds include

    unit-normal Snell radicand > 0.6794434351351366,
    outgoing unit axial component > 0.8762029410871408,
    outgoing unit-normal projection > 0.8242835890245156,
    actual internal axial travel > 2.2367633610132622,
    actual external axial travel > 2.378850585302347.

On the lambda extension, the independent lower bounds remain positive; in particular the outgoing unit axial component exceeds 0.8727626792727026 and external axial travel exceeds 2.364791262747313. Different valid interval circuits need not produce identical lower bounds. These checks establish sampled-time physical admissibility, not a new continuous-time claim.

## Native similarity and exact inverse orientation

Let C=D_eta theta and J=D_theta eta at the exact fixed point. The source's mixed index/beam blocks are correct:

    dn_j/dh_j = h_j/(n_j Q_opt),
    dn_j/dt_a = t_a(1-h_j^2)/(n_j Q_opt^2),
    dh_j/dn_j = n_j Q_opt/h_j,
    dh_j/dbeta_a_deg = (n_j^2-1)t_a(1+t_a^2)pi/(180h_j).

Together with the phase, wedge, and beam conversions, these are analytic inverse Jacobians. At 100 decimal digits the independent enclosure gives ||CJ-I||_inf<7.284e-101. This small interval identity defect is a numerical check on exact formulas, not an approximation substituted for the inverse chart.

Because S=(1/1000)I18 is scalar, the derivative in the declared scaled native coordinates is exactly

    Q_scaled=S^-1 C Q_eta J S=C Q_eta J=Q_native.

No eta-space norm or adapted norm is being mistaken for the original native requirement.

The residual orientation is also correct. With A=I-Q_native and E=I-WA, one has WA=I-E and hence

    A^-1=(I-E)^-1 W.

Thus the degree-two approximation is (I+E+E^2)W, with tail at most e^3||W||/(1-e) in the matrix infinity norm, and therefore in every individual entry. There is no illicit reversal of matrix multiplication. The independently rebuilt degree-two polynomial plus the full tail is contained in the saved inverse enclosure.

The exact rational audit confirms both reported norm bounds:

    ||W||/(1-e) <= 40099.8562445391,
    maximum absolute row sum of saved P enclosure <= 40099.85616680308.

Taking their minimum is valid. The exact matrix P has zero center chord defect by definition. Using the finite proposal W as an implemented update instead requires charging its nonzero verified defect.

## What still blocks a complete fixed-box certificate

The rules in `quantitative_algebraic_correction.md` and `principled_corrector_acceleration.md` require the same native box's entire corrected-record branch domain and transformed curvature bounds before a contraction test is meaningful. Physical admissibility and invertibility of I-Q_native do not imply either condition.

At minimum, an eventual positive result still needs the selected spectral/root/demixer/bivariate branch to exist on the full coupled corrected-record envelope, native-scaled chord curvature bounds on that domain, a passing fixed-radius self-map and strict contraction inequality, and an independent measurement-error domain if a positive noise allowance is claimed. Recovery from unknown data additionally requires the initializer/branch coverage, exact residual, delivered-iterate, and global qualifications stated in the theoretical rule.

An interval upper bound failing a sufficient domain or curvature test is an unresolved certificate, not proof that the exact modified map diverges or is undefined. Conversely the already proved instability of the undamped center derivative does not invalidate this successful automatic center inverse calculation.

## Reproduction

From `/workspace/shared/risley_theory` in the existing environment:

    OPENBLAS_NUM_THREADS=1 python proof_checks/audit_native_box_guards_chord_center.py

This reads the original records and writes only the new audit JSON. The audit is reproducible numerical interval mathematics, not proof-assistant formalization.
