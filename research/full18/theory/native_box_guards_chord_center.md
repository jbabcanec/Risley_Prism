# One native-box guard check and automatic center inverse

## Exact scope and outcome

This calculation uses the unchanged theoretical witness, kappa=1/10, and one fixed box of radius 1/1000 in every original native coordinate. The center has speeds (1,7,49)/20, equal wedge angles (180/pi)asin(1/10), zero phases, indices (sqrt(17/8),sqrt(233/125),sqrt(13/5)), beam angles ((180/pi)atan(1/3),0), source b=(1,2), gap g=3, and distance d=100. The coordinate order below is stated explicitly because the older point derivative uses a different chart:

    theta=(N1,N2,N3,a1_deg,a2_deg,a3_deg,phi1_deg,phi2_deg,phi3_deg,
           n1,n2,n3,beta_x_deg,beta_y_deg,b_x,b_y,g,d).

All original priors and all 200 samples' three-prism physical and ordered-traversal guards pass on this entire native box. The same physical guards also pass on the optional common wedge-sine analytic extension e_j(lambda)=lambda sin(a_j), 0<=lambda<=1. That extension is only a domain check for Taylor integral remainders; it does not replace the finite-wedge witness or change the target box.

The exact center matrix P=(I-Q_native)^-1 is enclosed rigorously, with

    ||I-W(I-Q_native)||_infinity <= 1.9387182380847987e-9,
    ||P||_infinity <= 40099.85616680308.

These are two partial calculations. They establish neither the off-image algebraic inverse's entire box domain nor a self-map/contraction/noise certificate. No corrector iterates, relaxation values, smaller boxes, or other optical cases were tested.

## Reproducible calculation and exact endpoints

Run `python proof_checks/native_box_guards_chord_center.py` with the already installed Python, NumPy and mpmath. No installation is needed. The script writes `proof_checks/native_box_guards_chord_center.json` and is based on the Snell/traversal circuit in `finite_wedge_point_guards.py` and the point-derivative coordinate definitions in `finite_wedge_corrector_derivative.py`. It consumes, without changing, `finite_wedge_corrector_interval.json`, whose SHA-256 is recorded in the new JSON.

All proof arithmetic uses mpmath outward real intervals at 70 decimal digits. Each saved interval includes outward binary64 lower and upper bounds and the exact mpmath dyadic endpoint tuples (sign, integer mantissa, base-two exponent, mantissa bit count). Each endpoint therefore represents (-1)^sign times mantissa times 2^exponent. The frozen inverse proposal is saved both as round-trippable JSON binary64 numbers and as hexadecimal binary64 strings. Floating-point inversion is used only to propose W; every claim about it is checked by interval arithmetic.

The exact box is theta0+[-1/1000,1/1000]^18, enclosed with interval pi, square roots and atan2. Prior decimal bounds such as 1.3 and 1.8 are introduced as exact decimal rationals rather than binary floating approximations. The native-to-eta box image is calculated from the exact expressions

    t_a=tan(pi beta_a/180), rho=t_x^2+t_y^2, Q=1+rho,
    h_j=sqrt(n_j^2+(n_j^2-1)rho), e_j=sin(pi a_j/180),
    eta=(N,e/kappa,pi phi_deg/180,h,t,b,g,d).

The physical trace uses the shared original native n,t expressions directly, rather than treating the saved eta interval image as independent input coordinates. This retains important n/t correlation. Squared intervals use interval powers, which retain nonnegativity across zero. D=1/(1-e_j(lambda)^2) is used exactly instead of widening cos^2+sin^2 independently. The rationalized Snell coefficient avoids the P-E cancellation.

## Physical circuit and normalization

Directions are scaled optical momentum coordinates. With Q fixed by the incident slope, the external direction is (X,Z)/sqrt(Q), initially (t,1)/sqrt(Q). For one prism, write u=tan(a)(cos gamma,sin gamma), D=1+|u|^2, with the appropriate lambda-dependent wedge slope for the analytic extension. The circuit is

    H=sqrt(n^2 Q-|X|^2), P=H-u dot X,
    Delta=P^2-D(n^2-1)Q, E=sqrt(Delta),
    c=(n^2-1)Q/(P+E), Z_out=H-c, X_out=X+u c,
    A=3+u dot p, p_exit=p+X A/P,
    external_flight=ell-u dot p_exit,
    p_next=p_exit+X_out external_flight/Z_out.

Here ell is g after the first two prisms and d after the last. The first entrance position is b+6t. For the first H only, n^2+(n^2-1)rho is evaluated to retain its exact cancellation. The two traversal conditions are A>0 and external_flight>0; the actual internal axial travel is H A/P>0. Entrance and exit axial and normal-branch guards are checked explicitly.

Normalization matters:

- H^2 is a Q-scaled glass radicand; H^2/Q=n^2-|q|^2.
- P is a scaled incoming optical-momentum normal numerator. The incoming glass unit-direction projection onto the unit surface normal is P/(n sqrt(QD)).
- Delta/Q is the outgoing unit-ray, unnormalized-normal discriminant. The actual unit-normal Snell radicand is Delta/(QD).
- E is the outgoing scaled unnormalized-normal numerator, equal algebraically to Z_out-u dot X_out. The outgoing external unit-ray numerator is E/sqrt(Q), while the projection onto the unit surface normal is E/sqrt(QD).
- Z_out/sqrt(Q) is the outgoing unit-ray axial component.

Selected strict lower bounds on the entire native box are:

| Guard | Lower bound |
|---|---:|
| Q-scaled glass radicand | 1.919250845596771 |
| Q-scaled incoming normal numerator P | 1.3465941200520413 |
| Q-scaled exit discriminant Delta | 0.7625120157938498 |
| Unit-normal Snell radicand Delta/(QD) | 0.6793879168733478 |
| Outgoing unit axial component | 0.8761713909150738 |
| Outgoing unit-normal projection | 0.8242499116611101 |
| Internal traversal numerator A | 2.2869958926786316 |
| Actual internal axial travel | 2.2325286719350173 |
| Actual external axial travel | 2.378850464719938 |

All quoted lower bounds are outward rounded downward. On the full lambda interval, the internal traversal numerator is greater than 2.244184583410213, internal axial travel greater than 2.1692132297886046, and external flight greater than 2.3650830279296464. The optical lower bounds above also hold on that extension. The JSON records all 18 guards at all 600 sample/prism pairs for both domains, including the locations of each attained interval minimum.

## Exact native differential and inverse rule

Let C=D_eta theta at the exact center and J=D_theta eta. Apart from identity entries, the nonzero changes are

    C[a_j,e_j/kappa]=(180/pi) kappa/sqrt(1-kappa^2),
    C[phi_j_deg,phi_j_rad]=180/pi,
    C[n_j,h_j]=h_j/(n_j Q),
    C[n_j,t_a]=t_a(1-h_j^2)/(n_j Q^2),
    C[beta_a_deg,t_a]=(180/pi)/(1+t_a^2),

and

    J[e_j/kappa,a_j]= (pi/180) sqrt(1-kappa^2)/kappa,
    J[phi_j_rad,phi_j_deg]=pi/180,
    J[h_j,n_j]=n_j Q/h_j,
    J[h_j,beta_a_deg]=(n_j^2-1)t_a(1+t_a^2)pi/(180 h_j),
    J[t_a,beta_a_deg]=(1+t_a^2)pi/180.

These formulas are analytic inverse Jacobians, not floating inverse proposals. Their interval identity check gives ||CJ-I||_infinity<=9.197174064433381e-71. Since the native scaled coordinates use S=(1/1000)I, scalar S cancels exactly:

    Q_scaled=S^-1 C Q_eta J S=C Q_eta J=Q_native.

The existing point certificate's outward Q_eta endpoints are imported as exact binary endpoints. W is the frozen binary64 inverse proposal for the midpoint of A0=I-Q_native. The proof checks

    E=I-W A0, e>=||E||_infinity<1.

Thus the exact A0 and W are nonsingular and the automatically defined matrix is

    P=A0^-1=(I-E)^-1 W.

The saved enclosure uses P2=(I+E+E^2)W with a common rigorous entry tail

    e^3||W||_infinity/(1-e)<=2.9220450229824003e-22.

The elementary Neumann norm bound is ||W||/(1-e)<=40099.8562445391. Taking absolute row sums of the sharper saved entrywise enclosure instead gives the reported bound 40099.85616680308. The exact P has zero center derivative defect for the ideal chord map by definition. A computational implementation that uses W itself must keep the verified nonzero defect e. Neither conclusion removes the need for box Hessians, corrected-record branch guards, an independent measurement-noise domain, full-record residuals, and global branch coverage.

## Arithmetic replay and alternative physical formulation

`proof_checks/native_box_guards_chord_replay.py` reads exact dyadic endpoints at 90 digits, reconstructs the binary W from its hexadecimal strings, and recomputes E and the Neumann polynomial by plain nested-list matrix multiplication. It reproduces the defect upper bound and verifies that the recomputed P2 plus the recorded remainder lies within every saved P entry. Its new JSON/log record the source certificate hash.

An additional unit-ray calculation uses the direct unit-normal Snell coefficient, without Q scaling or rationalization: K=(H-u dot q)/sqrt(D), delta=1-n^2+K^2, c=(K-sqrt(delta))/sqrt(D). It checks the same native box and all 600 sample/prism combinations. All guards pass, including unit-normal Snell radicand >0.6789507113545055, outgoing unit axial >0.8732265781330142, internal traversal numerator >2.285826073454098, and external axial flight >2.378371907127911. The difference in lower bounds reflects ordinary interval dependency differences. This is a same-author arithmetic cross-check, not a claim of independent review.
