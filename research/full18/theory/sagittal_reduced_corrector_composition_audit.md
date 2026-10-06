# Composition audit: thirteen-coordinate corrector

The proposed composition and derivatives are correct. This note adds the two necessary iteration qualifications: a retained-coordinate invariant domain is required, and a projected fixed point is not automatically a full-model solution.

Fix the observed data y. Let x be the retained thirteen optical coordinates, let the first subresultant be a(x,y)T+b(x,y), and choose a fixed four-row sagittal geometry pivot

    A_P(x,G) w+c_P(x,G,y)=0.

On its nonzero chart define

    G(x)=−b/a,
    w(x)=−A_P(x,G(x))^−1 c_P(x,G(x),y),
    n₃(x)=sqrt(G(x)),
    E_y(x)=(x,n₃(x),w(x)), reordered into the native eighteen-coordinate order.

The square root is the positive physical branch. With y held fixed, the derivatives are

    DG=−(G Da+Db)/a,
    Dw=−A_P^−1[(D_x A_P+A_(P,G) DG)w
               +D_x c_P+c_(P,G) DG],
    Dn₃=DG/(2 sqrt(G)).

In the Dw expression, the bracket is a four-vector-valued one-form: apply it to a retained-coordinate increment to obtain the usual implicit-differentiation formula. The derivative DE_y consists of the identity block for retained coordinates and these eliminated-coordinate derivatives, followed by the native-order permutation. If coefficient construction uses a different algebraic coordinate chart, include its ordinary chart-to-native Jacobian factors.

For a differentiable original eighteen-coordinate corrector T_y, let π₁₃ select the retained coordinates and set

    Φ_y=π₁₃ T_y∘E_y.

Then

    DΦ_y(x)=π₁₃ DT_y(E_y(x)) DE_y(x).

## Domain and iteration guards

The proposed retained-coordinate domain must ensure:

- a≠0 and det A_P≠0, with controlled lower bounds when proving uniform derivative estimates;
- the reconstructed positive index and all reconstructed geometry variables satisfy their original native bounds;
- the upstream radical branches and the complete reconstructed system obey the required strict physical and traversal guards at every stipulated sample;
- E_y(x) belongs to the chosen branch/domain of T_y;
- for an iteration theorem, Φ_y maps the retained-coordinate domain into itself.

On the generic interior chart, retain the original interior-prior requirements as well. Boundary cases need their own treatment. A contraction theorem requires a suitable complete invariant domain and a newly proved contraction bound in the chosen scaled norm. An old bound for DT_y alone does not suffice: DE_y contains inverse subresultant and geometry-pivot factors and can amplify errors.

## Fixed points and noise

If an exact physical solution theta has a≠0 and the selected geometry pivot is valid, then E_y(π₁₃ theta)=theta by the certified conditional uniqueness. Assuming T_y fixes exact solutions, π₁₃ theta is therefore a fixed point of Φ_y.

The converse does not follow from projection alone. A projected fixed point may hide a change that T_y makes only in the five eliminated coordinates. It must be checked against the original full-model residual and guards, or covered by an additional equivalence theorem.

Away from exact compatible data, the rational continuation G=−b/a only solves the linear subresultant equation. It need not solve either norm equation. Solving the selected four affine rows need not satisfy the other sagittal equations or the full observed record. In particular, this construction does not parameterize the complete positive-noise feasible set and cannot silently replace its retained residual/error variables.

There is no automatic elimination of weak directions, no inherited contraction guarantee, and no noise-stability theorem from the dimension reduction alone. Those claims require fresh derivative, invariant-domain, residual, and, where relevant, data-sensitivity bounds.

## Shared-record data derivative

The shared-noise formulas in `sagittal_corrector_composition.md` are correct. At fixed retained coordinates x,

    D_y G=−(G D_y a+D_y b)/a,
    D_y w=−M^−1[M_G(D_y G)w+c_y+c_G D_y G],
    D_y n₃=D_y G/(2 sqrt(G)).

Here c_y is the partial derivative at fixed G. The sagittal geometry coefficient matrix M has no direct dependence on y, so there is no additional M_y term. All occurrences of D_y G in the matrix expression are ordinary chain-rule contractions. Because the fixed norm charts and pivot use only the first six paired samples, D_y E_y has support only in those observed coordinates.

For T_y(theta)=A17+1(y−R1(theta),y−R2(theta)), with R1 and R2 independent of y, its direct data derivative at fixed theta is A_s+A_w. Therefore the complete derivative of the composed map at fixed x is

    D_y Φ_y=π₁₃[(A_s+A_w)+(D_theta T_y)D_y E_y].

Both terms are evaluated at theta=E_y(x) and the associated two corrected records. The two slot derivatives are added because the same perturbation of y enters both slots; they are not independent noise variables. The second term accounts for the simultaneous movement of the reconstruction itself.

These identities supply terms that a future noise bound must control, together with reconstruction state/data derivatives and the domain guards. They do not constitute such a bound, and do not justify carrying over the original corrector's noise constant unchanged.
