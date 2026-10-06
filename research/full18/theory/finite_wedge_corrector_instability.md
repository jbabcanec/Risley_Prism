# One finite-angle proof check: the undamped 17+1 corrector is unstable

## Result

At the **single, preselected original oblique witness** with all three signed wedge sines equal to 1/10, the exact finite-angle fixed point of the specified undamped A17+1 correction has an unstable derivative. Outward interval arithmetic certifies two isolated real eigenvalues in

- (-23.426732948389, -23.426732948195),
- (-5.656470996867, -5.656470996669).

Consequently its spectral radius exceeds one, and **no equivalent norm can make this derivative a contraction**. This is a failure of this undamped correction map at this point, rather than merely a failed infinity-norm estimate. It does not prove that the physical inverse itself fails or that every reconstruction method is unstable.

An independent, simpler spectral obstruction is also certified:

    -27.885143984150 < trace(DT) < -27.885143984148.

Since |trace(DT)| <= 18 rho(DT), this alone forces rho(DT)>1.54917. The isolated-eigenvalue certificate is much sharper.

No wedge, speed, source, hardware, or phase was changed to obtain this outcome. No parameter sweep, other proof point, damping selection, modified-map test, or software installation was performed.

## The one witness and physical guards

The native parameters are

- all three wedges: (180/pi) asin(1/10) = 5.739170477266787 degrees;
- all three phases: 0;
- speeds: (1,7,49)/20 = (0.05,0.35,2.45) Hz;
- beam angles: ((180/pi) atan(1/3),0) = (18.43494882292201,0) degrees;
- indices: (sqrt(17/8),sqrt(233/125),sqrt(13/5));
- source b=(1,2), common gap g=3, screen distance d=100.

Equivalently t=(1/3,0), h=(3/2,7/5,5/3). These parameters are strictly within the stated original priors.

All 200 original timestamps k/20 and all three prism traversals pass outward-interval checks. Directions in the compiler are scaled by sqrt(Q), Q=10/9. Rounded-down lower bounds are

- glass radicand >1.92251;
- scaled exit Snell radicand >0.770275;
- scaled incoming-normal numerator P >1.34780;
- scaled outgoing axial direction Z >0.927914;
- scaled outgoing normal numerator Z-u dot X >0.877653;
- internal traversal numerator 3+u dot p >2.29779;
- actual external axial propagation length >2.38446.

For unit directions, Z is divided by sqrt(Q), and the exit Snell radicand is divided by Q. Thus safe unit-direction-chart lower bounds are Z_unit>0.88029, radicand_unit>0.69324, and the outgoing normal numerator >0.83261. For projection onto the unit surface normal, divide the latter by sqrt(D), D=100/99; the corresponding lower bound is >0.82843. Internal traversal numerator positivity together with H,P>0 certifies positive internal axial travel.

The first-order ellipse and bivariate reconstruction chart guards also pass: rho>0, Bv!=0, det M_i>0, beta_i>0, h3>1, nonzero U, distinct lag roots and nonzero root derivatives. The elimination coefficient is the previously audited exact rational

    f1=399563722409659/317911540674000 >0.

The verified 18-by-18 feature Jacobian inverse, the audited bivariate equivalence, and these nonzero chart guards identify the selected regular A17+1 branch locally.

## Derivative construction

The coordinates used for differentiation are

    eta=(N1,N2,N3, e1/kappa,e2/kappa,e3/kappa,
         phi1,phi2,phi3, h1,h2,h3, tx,ty,bx,by,g,d),
    kappa=1/10,

with phases in radians. All 18 coordinates are differentiated. The physical indices inside the forward evaluator are

    n_i^2=(h_i^2+|t|^2)/(1+|t|^2).

The exact Snell circuit and its lambda-degree-two truncation are evaluated with first parameter jets. The auxiliary lambda multiplies all wedge sines. This computes F,DF,F1,DF1,F2,DF2 entirely from the model; no observed temporal derivative or nonlinear full18 reconstruction solve is used.

For each of the 18 input directions, put delta s=(DF1-DF) delta eta and delta w=(DF2-DF) delta eta. The off-model input derivatives are assembled in this order:

1. Lag-11, 246-by-6, DC-constrained Helmert recurrence with exact frozen corrected left inverse D=(P J)^(-1)P.
2. Degree-seven polynomial-root differentiation, radial phase derivative, and the matching speed-lift branch for this proof point. The actual inverse algorithm must retain all lifts; no known-frequency claim is made for reconstruction from unknown data.
3. A frozen corrected left inverse of the original 200-by-7 real fundamental design, including the subtraction (D_N F1) delta N.
4. The 200-by-31 quadratic/confluent design, with confluent columns (k/199)z_i^(+/-k), and the exact third self-second-harmonic extraction row. Its frequency derivative is incorporated as delta C2=ell[delta w-(D_N F2)delta N].
5. The complete normalized invariant derivative Re(delta C2/U^2-2 I delta U/U).
6. A linear inverse of the physical feature Jacobian G for (N3, all 14 baseline/harmonic coefficients, Re I). The physical invariant jet differentiates B=b+t[6+2g+d+3 sum 1/h_i], including all its beam, source, and hardware dependence.

The resulting matrix is exactly DT=A_s(DF1-DF)+A_w(DF2-DF) at y=F(theta). The feature-Jacobian inversion is ordinary linear algebra for this proof; it does not replace the algorithm's two-variable nonlinear reconstruction step with a hidden full18 nonlinear solve.

Same-point consistency checks include agreement between the Taylor F1 jets and the explicit M_i first-order formula; the correct self-harmonic identity ell F2=I U^2; and the full differential left-inverse identity. In interval arithmetic, the extracted feature differential differs from its explicit physical Jacobian by an enclosure with infinity norm below 6.083e-24. The corresponding exact mathematical identity is zero.

Floating binary64 matrices only propose left inverses. Interval arithmetic encloses their defects, sums two Neumann correction terms, and bounds every omitted tail. The final certificate exports the binary proposals and all entries of the enclosed derivative. A fixed real eigenbasis proposed numerically is also corrected for its inverse defect before applying Gershgorin's theorem. The two large negative disks are isolated, so each contains exactly one eigenvalue; because the matrix is real, each is a real eigenvalue.

## Native-coordinate conversion and spectral diagnosis

The nonsingular eta-to-native tangent transformation is explicit:

    da_i(degrees)= (180/pi) kappa/sqrt(1-e_i^2) d(e_i/kappa),
    dphi_i(degrees)= (180/pi) dphi_i,
    dbeta_a(degrees)= (180/pi)/(1+t_a^2) dt_a,
    dn_i= h_i/(n_i Q) dh_i
          + sum_a t_a(1-h_i^2)/(n_i Q^2) dt_a,
    Q=1+|t|^2.

Speeds, source coordinates, g, and d are unchanged. Eigenvalues are invariant under this change of coordinates. Its mixed-native infinity norm is below 57.295780 at the witness; this is a unit-conversion bound, not a condition number or an error tolerance.

The interval Gershgorin cover also certifies Re(lambda)<0.781406 for every eigenvalue. The floating rightmost eigenvalue is approximately 0.645940730. The remaining floating eigenvalue proposals are recorded in the certificate JSON; the spectral-radius proposal is approximately 23.426732948292.

The two unstable eigenvectors mix multiple parameters. Expressed in the original mixed native units and normalized by |delta d|=1, the largest components of the first are approximately (d,a1_degrees,bx,g)=(-1,0.439,0.143,0.112), and of the second (d,bx,a1_degrees,a2_degrees,g)=(1,-0.191,0.102,0.073,-0.071). These describe two coupled tangent directions, not independent coordinate sensitivity bounds.

As a general mathematical observation only, if every eigenvalue lambda of DT has Re(lambda)<1, then sufficiently small positive scalar relaxation can place the eigenvalues of (1-alpha)I+alpha DT inside the unit disk: each requires alpha<2(1-Re lambda)/|1-lambda|^2. No alpha was selected, optimized, or tested here; no modified algorithm or finite-radius claim follows from this observation alone.

## Scope, non-results, and reproducibility

- Derivative contraction at this point: **disproved for the specified undamped map**.
- Positive numerical finite convergence radius: **not established**.
- Positive measurement-error tolerance epsilon: **not established**.
- Exact all-400 fit: the constructed synthetic y=F(theta) is compatible by definition, and the left-inverse identity makes theta its fixed point. No arbitrary supplied record was solved or certified.
- Global branches or uniqueness: **not established**.

Run from the proof_checks directory, using the existing Python environment:

    OPENBLAS_NUM_THREADS=1 python finite_wedge_point_guards.py
    OPENBLAS_NUM_THREADS=1 python finite_wedge_corrector_derivative.py --interval

Files created for this bounded check:

- proof_checks/finite_wedge_point_guards.py
- proof_checks/finite_wedge_corrector_derivative.py
- proof_checks/finite_wedge_corrector_proposal.log and .json
- proof_checks/finite_wedge_corrector_interval.log and .json

The interval JSON contains the frozen left-inverse proposals, the enclosed derivative endpoints, its trace enclosure, the fixed adapted basis and its inverse proposal, all floating eigenvalue proposals, and the verified Gershgorin disks. Numerical implementations and interval certificates are not proof-assistant formalization.
