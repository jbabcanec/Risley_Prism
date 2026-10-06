# Independent audit of the one finite-wedge corrector derivative

## Verdict and scope

**Pass: the predeclared undamped finite-angle iteration is locally unstable at this one physical point.** This is a certified derivative obstruction, not merely failure of an upper-bound contraction test. An independently reconstructed interval certificate encloses two real derivative eigenvalues outside the unit disk:

- [-23.42673294838928, -23.426732948194797]
- [-5.6564709968713505, -5.656470996664911]

A separate, simpler exact-rational trace check gives spectral radius greater than 1.54917466578605 without using any proposed eigenvectors. A change of equivalent norm or a smooth invertible native-coordinate chart cannot turn this derivative into a contraction.

This does **not** invalidate the two-variable algebraic inverse structure, its local Taylor-pair left-inverse identity, or the exact fixed-point identity. It does not establish a finite convergence radius, an independent observation-noise allowance, a data-derived initializer, global completeness, or performance elsewhere. No new physical case, wedge choice, damping value, or modified iteration was tested.

## 1. Unchanged witness and physical trace

The sole witness is

    kappa = e1 = e2 = e3 = 1/10,
    native wedges = (180/pi) asin(1/10) degrees,
    N = (1,7,49)/20, phases = 0,
    t = (1/3,0), b = (1,2),
    h = (3/2,7/5,5/3), n^2 = (17/8,233/125,13/5),
    g = 3, d = 100,
    sample times = k/20, k=0,...,199.

Every native coordinate is inside its original prior. All 200 timestamps and all three prisms pass the strict sampled physical and traversal guards. The proof code uses Q-scaled momenta with Q=10/9. Therefore its outgoing axial direction is the unit axial direction multiplied by sqrt(Q), and its Snell radicand is the unit-direction radicand multiplied by Q. This scaling is consistently propagated between prisms.

The independent optical implementation instead starts with the unit incident transverse direction t/sqrt(Q). It propagates positions using

    p_next = p + 3v + ell w
             + (v-w) [u dot (p+3v)]/(1-u dot v),

where v and w are internal and outgoing slopes. This differs from the source implementation's explicit exit-intersection/flight update. The same-point exact records agree to 2.04e-13 in double arithmetic over every sample and both axes. Positivity of the source guard `internal_travel` is positivity of 3+u dot p; together with H,P>0 it certifies positive internal travel. Its external numerator ell-u dot p_exit is the actual remaining positive axial flight.

These are sampled-time guards, as required by the specified model; no all-continuous-time optical claim is made.

## 2. Taylor records and independent parameter derivatives

The differentiated coordinates are exactly

    eta = (N1,N2,N3, e1/kappa,e2/kappa,e3/kappa,
           phi1,phi2,phi3 [radians], h1,h2,h3,
           tx,ty,bx,by,g,d).

The source compiler expands the physical plane slope after e is replaced by lambda e. Its first two Taylor coefficients agree with replacing the slope by lambda e times its rotor direction, because the next slope term is cubic. F1 includes the full baseline plus degree one. F2 includes the full baseline, degree one, and every degree-two term, including physical DC and mixed harmonics.

The independent implementation uses unscaled first and second lambda tangents rather than polynomial coefficients. It then obtains eta derivatives by complex-step differentiation of the analytic arithmetic circuit. These are numerical cross-checks at the same fixed point, not rigorous domain enclosures or additional physical examples.

Across all 200 paired samples, the largest discrepancies are:

- F1, independent tangents versus polynomial compiler: 2.07e-13
- F2, independent tangents versus polynomial compiler: 2.01e-13
- DF, independent complex step versus source first jets: 1.29e-11
- DF1 and DF2: each below 1.29e-11
- The full feature Jacobian G: below 1.78e-15

The explicit first-order coefficient formula and the Taylor compiler are also checked on the same point. The independent extracted third self-second harmonic differs from I U^2 by less than 3.74e-15 in double arithmetic. The interval production run encloses this defect by approximately 4.27e-23 in each real component. This is consistent with the exact physical self-harmonic identity, including its coefficient normalization.

## 3. Frequency lift and off-model preprocessing derivative

The frozen long-lag construction is the DC-constrained chart in `quantitative_algebraic_correction.md`, not an unconstrained seven-coefficient recurrence. Its Helmert matrix satisfies R^T 1=0 and R^T R=I. Lag 11 uses 123 starts per output axis, hence 246 rows, with maximum index 199. The source derivative

    dc = -R D [sum_(j=0)^7 c_j ds_(k+11j)]

is the derivative of the exact frozen rational chart at its matching Taylor record. Since dc lies in the columns of R, the exact DC root remains fixed under off-model perturbations.

For a selected simple lag root z,

    dz = -sum_(j=0)^6 dc_j z^j / p'(z),
    dN = [20/(2 pi 11)] Im(dz/z).

This is the correct derivative of the radially normalized root argument. The original-frequency lift integer is fixed on the branch. In particular the third selected lag root has principal phase numerator 139 modulo 400, while its unwrapped numerator is 539: the third frequency uses lift integer one. The derivative remains the displayed expression. This calculation certifies one selected branch; it does not reject other speed lifts or physical prism assignments.

Let L7 be the exact frozen left inverse of the real seven-tone design. Its coefficient differential at the matching record is

    dcoeff = L7 [ds - (partial_N F1) dN].

No derivative of L7 is missing: differentiating L7(N)V7(N)=I on a matching model record gives precisely this formula.

The weak extractor contains all 25 quadratic tone nodes and the six scaled confluent fundamental columns. The 25 integer node numerators are distinct modulo 400. At the matching F2 record, differentiation of its exact extraction identity gives

    dC_plus2 = ell [dw - (partial_N F2) dN].

Thus the source's weak frequency correction is also correct. The confluent columns preserve the intended first-order frequency cancellation; they cannot simply be discarded. With U the positive complex fundamental coefficient,

    d Re(I) = Re[dC_plus2/U^2 - 2 I dU/U].

The factor 1/2 in U=(complex cosine - i complex sine)/2 and the positive third self-second target are consistent in both implementations.

## 4. Why the 18-by-18 linear inverse is legitimate

The feature map consists of three speeds, fourteen real first-order coefficients, and one real normalized quadratic invariant. Its Jacobian G is square. The inverse-defect certificate establishes that G is nonsingular at this point.

Locally the stated bivariate algebraic construction is the inverse of this feature map on its regular branch. Therefore computing its differential by G^{-1} is legitimate. It is an eighteen-by-eighteen **linear differential calculation**, not an extra eighteen-variable nonlinear solve or a replacement for the bivariate reconstruction.

The independent exact rational bivariate audit was rerun. It again verifies the nonzero elimination denominator

    f1 = 399563722409659/317911540674000,

and the nonzero two-by-two beam Jacobian of the eliminant and real weak invariant. The native ellipses, positive hardware reconstruction, and narrow wedge/phase branch are regular at the unchanged point. Their amplitude factors cancel from the shape invariants, so this exact bivariate check applies at e_i=1/10.

Writing K for the preprocessing differential applied to (DF1-DF,DF2-DF), the corrector derivative is

    DT = G^{-1} K.

Applying the same preprocessing to (DF1,DF2) gives G. The independent numerical check yields

    ||DA(DF1,DF2)-I||_infinity < 5.60e-9.

The interval production calculation bounds the preprocessing identity defect by 6.083e-24. The independent DT differs from the source floating proposal by less than 9.96e-9 in infinity norm. Both reproduce the same two large negative eigenvalues and trace.

## 5. Rigorous linear-algebra certificate

Every floating inverse in the production certificate is only a fixed binary proposal. For E=I-PA with ||E||<1, the exact left inverse (PA)^{-1}P is enclosed by

    (I+E+E^2)P + tail,
    ||tail|| <= ||E||^3 ||P||/(1-||E||).

The real/complex interval type discrimination was repaired before the successful run. Real lag and coefficient left inverses now remain real. The complex demixer receives a complex tail rectangle that contains the modulus-bounded tail. The reconstruction inverse is corrected by the same Neumann argument. Its reported inverse-defect bound is below 4.104e-11, and its final common entry-tail bound is below 8.811e-27. The full exported DT intervals include propagated arithmetic uncertainty as well as this tail; the tail alone must not be mistaken for the full entry enclosure.

The independent certificate reads those outward binary endpoint matrices and sums their diagonal endpoints using exact rational arithmetic. It obtains

    trace(DT) in [-27.885143984149146, -27.885143984149067].

Since an 18-by-18 matrix satisfies |trace(DT)| <= 18 rho(DT), this proves

    rho(DT) > 1.54917466578605 > 1.

This conclusion does not depend on an eigenvalue proposal or a chosen norm.

For a sharper independent check, the audit rebuilds the real adapted similarity in 70-digit interval arithmetic. If R is the proposed inverse of the proposed real basis S and E=I-RS, then

    S^{-1} DT S = (I-E)^{-1} (R DT S).

The audit encloses the inverse correction and tests disk separation with exact rational arithmetic on the binary centers and outward radii. Each of the first two Gershgorin disks is isolated and contains exactly one eigenvalue. Because the actual matrix is real and the disks have real centers, those singleton eigenvalues are real. Their intervals are those in the verdict. The independent certificate also gives Re(lambda)<0.781406 for every eigenvalue, without selecting or evaluating any relaxation parameter.

The source initially printed a few Gershgorin arithmetic endpoints using ordinary floating addition. Its final rerun now uses outward interval endpoint calculations and separation tests. The independent audit additionally uses exact-rational separation predicates. The final source run and the independent endpoint-certificate rerun both passed. Explicit interval assertions now also check every sample's F1 value and derivative against the direct formula (error bound below 1e-40), the self harmonic (below 1e-20), and the feature differential identity (below 1e-20).

## 6. Consequences and exclusions

At the exact compatible record y=F(theta_star), this point is a fixed point of the undamped map, and its derivative has expanding real modes. Consequently the undamped map is not locally asymptotically stable there and cannot satisfy a neighborhood contraction certificate in any equivalent norm. Smooth conversion from eta to native wedges, phases, indices, and beam angles conjugates the derivative at the fixed point and preserves its spectrum.

The failure is in this finite-angle iteration's dynamics. It is not a failure of the two-variable algebraic feature inverse or a proof of non-identifiability. No finite-radius, noisy-data, broad-prior, global branch-count, or practical runtime conclusion follows. A known witness was used for this declared proofcheck; it is not a demonstration of a data-derived blind initialization.

## 7. General acceleration note

The separate feasibility note's scalar complex-eigenvalue criterion and its weighted Lyapunov-norm inequality are correct. The frozen chord preconditioner has the claimed center derivative and second-derivative bounds; it must stay fixed on the parameter/data envelope, and its conditioning can enlarge the noise and curvature constants.

A coordinate notation correction was reported and verified in the revised note: when Q denotes the native-scaled derivative, the compatible-point identity is

    I-Q = S^{-1}(A_s+A_w) DF S,

unless all derivatives on the right have already been expressed in scaled coordinates. The corresponding unscaled derivative identity is I-D_theta T=(A_s+A_w)DF. This audit does not select, tune, or test any acceleration at the witness.

## Reproducible audit artifacts

- `proof_checks/audit_finite_wedge_derivative_independent.py`: independent unit-direction optical and differential cross-check
- `proof_checks/finite_wedge_derivative_independent_audit.log` and `.json`: its numerical checks
- `proof_checks/audit_finite_wedge_interval_certificate.py`: independent exact-rational trace and interval-similarity certificate
- `proof_checks/finite_wedge_interval_certificate_independent_audit.log` and `.json`: rigorous certificate results
- `proof_checks/finite_wedge_bivariate_branch_audit.log`: rerun of the existing exact rational bivariate branch audit

All audit files are new. The source proof script and original theory notes were not edited by this auditor.
