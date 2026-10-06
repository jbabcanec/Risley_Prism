# Completed independent audit: two-chart norm coprimality certificate

## Conclusion

The targeted certificate passes. At the exact rational-rotor witness, the fixed norm polynomials for samples (0,1,2,3,4) and (0,1,2,3,5) both have degree 64. After removing the known exact common factor G−9/4, their degree-63 quotient polynomials are coprime. Consequently the original specialized norm polynomials have gcd exactly G−9/4, up to a nonzero scalar.

This was checked by a full arithmetic/construction audit, an 8192-bit rerun, and a second 2048-bit calculation with independently generated original augmented rows and division-free polynomial pseudo-remainders. The hypotheses of `sagittal_subresultant_conditional_audit.md` are therefore now certified for this witness. Its generic subresultant consequence applies with the scope and exceptional-set qualifications stated there.

The computation is a rigorous outward-interval proof check at one specified witness. It is not a numerical gcd tolerance test, a reconstruction campaign, or a proof-assistant formalization.

## 1. Files inspected and reproduced

Original implementation:

- `proof_checks/sagittal_two_chart_coprimality.py`
- Its 4096-bit and 1024-bit execution logs and original certificate JSON

Higher-precision independent rerun:

- `proof_checks/sagittal_two_chart_coprimality_audit_rerun.py`
- `proof_checks/sagittal_two_chart_coprimality_independent_8192.log`
- `proof_checks/sagittal_two_chart_coprimality_independent_8192_certificate.json`

Second arithmetic path:

- `proof_checks/audit_sagittal_subresultant_pseudodivision.py`
- `proof_checks/sagittal_subresultant_independent_pseudodivision.log`
- `proof_checks/sagittal_subresultant_independent_pseudodivision_certificate.json`

The second path shares the audited integer-enclosure primitives and formal norm multiplication with the original implementation. Its position propagation, geometry sensitivities, augmented-row construction, and polynomial elimination path are independently implemented. This distinction matters: it is not described as a wholly independent implementation of every primitive.

The witness is the same exact physical system certified for all 200 samples in `sagittal_index_independent_audit.md`. The new six-sample calculations retain its exact rational rotor powers, rational native slopes and indices, and positive-root algebraic screen data.

## 2. Outward interval arithmetic

An interval endpoint integer m denotes m/2^BITS. Rational initialization uses exact floor and ceiling. Addition and negation are exact on this grid. Multiplication takes the extrema of the four exact endpoint products and rounds outward. Division checks that the denominator interval strictly avoids zero, evaluates all four endpoint quotients with correct handling of negative denominators, and rounds outward. Square roots use integer square roots with upward adjustment of the upper endpoint when needed.

The `ldexp` operation multiplies by an exact positive power of two; when its exponent is negative, the endpoints are again rounded outward. Normalization chooses this power using integer bit lengths, not floating-point estimates. Floating-point conversion is used only for printed summaries and elapsed time; no proof assertion depends on it.

Interval dependencies are allowed to be forgotten, making enclosures wider but not unsound. The few exact algebraic cancellations used to narrow an interval are justified separately below.

## 3. Exact norm construction

The code uses X=G−9/4, so the formal radicals obey

    h_i²=X+beta_i,  beta_i=9/4−|q_i|².

A bit mask records a square-free product of the h_i. Polynomial multiplication uses symmetric difference of masks for surviving radicals and replaces each common bit by X+beta_i. This implements the stated quotient-ring multiplication exactly before interval enclosure.

To eliminate one radical, the current expression is written E+h_i O. The code replaces it with

    E²−(X+beta_i)O².

Eliminating all five radicals therefore gives the product over all 32 formal sign choices. The final output has only mask zero and at most degree 64 in X. The root-independent d column gives the original determinant total radical degree at most four, establishing this degree bound independently of the implementation's list lengths.

Normalization between eliminations changes the polynomial only by a known nonzero positive scalar. If the current scale exponent is e, a norm step doubles it, and subsequent normalization adds its chosen exponent. The final scales relative to the unnormalized fixed norms were independently checked to be 2^539 and 2^508 for the two charts. Such scales do not alter degrees, roots, or gcds.

Both enclosed degree-64 coefficients are strictly positive. For the normalized polynomials they are approximately 3.1162515572237×10^−43 and 5.7933562382665×10^−42. These are scaled leading coefficients, not leading coefficients of the unnormalized generic norms.

## 4. Witness column operation and exact deflation

The original implementation adds the first four augmented columns multiplied by the true geometry to the final column. This leaves the determinant unchanged for every formal radical value. Writing

    r_i=det(y_i−p_i,u_i),

the exact generated-data sagittal identity then makes the final column r_i(h_i−H_i,true). This is a legitimate exact identity at this one proof witness. It avoids interval cancellation in constructing the constant coefficient.

The generic norm and subresultant must still be defined from the original inverse-data rows, whose coefficients depend only on retained optical coordinates, observations, and known upstream quantities. The witness-only coefficient representation using H_i,true is not a license to introduce the unknown index into generic coefficient formulas.

The independent 2048-bit path verifies precisely this distinction. It computes positions by explicit two-segment propagation; obtains the three upstream geometry columns from exact affine one-unit differences; reconstructs the upstream constant e; and uses the original last-column coefficients

    det(y−e,q)−3 det(q,u),  det(y−e,u).

It does not use the true-geometry column operation or the replacement by −H_true r. It nevertheless yields the same norm degrees and matching normalized coefficient values.

At X=0, the physical positive-radical determinant factor vanishes exactly by the already proved sagittal identity. Both exact norm polynomials therefore have constant coefficient exactly zero. Dropping that coefficient produces the exact quotient by X. The assertion that its interval contains zero is only a consistency check; the mathematical identity supplies the equality. No small nonzero constant is numerically truncated.

## 5. Euclidean cancellation and degree certification

In the original Euclidean calculation, a long-division step uses q=r_top/b_lead. The next highest coefficient is then exactly

    r_top−(r_top/b_lead)b_lead=0.

Discarding that coefficient is sound even though independent interval operations would fail to reproduce exact cancellation. It preserves the enclosure of the actual polynomial remainder. This is not trimming a coefficient merely because its interval contains zero.

All other coefficients are updated by outward interval arithmetic. Every divisor leading coefficient excludes zero. Every successive remainder's retained leading coefficient also excludes zero; the code never drops an unresolved leading term. The certified remainder degrees are exactly

    62, 61, …, 1, 0.

Positive power-of-two rescaling preserves the gcd at each step. A nonzero terminal constant therefore proves that the two deflated degree-63 polynomials are coprime.

The independent path replaces quotient divisions with pseudo-division steps

    r_new=b_lead r−r_top X^k b.

Its top term is exactly zero by commutativity, and is discarded for that algebraic reason. Normalization after each step is by a nonzero positive scalar. This independently preserves the gcd and reaches the same degree sequence without coefficient division during polynomial elimination.

## 6. Exact certificate results

The 8192-bit rerun certifies all 63 strict remainder pivots. Its terminal scaled Euclidean constant lies strictly between

    0.65195 and 0.65196,

with interval width less than 2^−7410.

The 2048-bit original-row, division-free pseudo-division calculation also certifies all 63 strict remainder pivots. Its differently scaled terminal constant lies strictly between

    0.97136 and 0.97137,

with interval width less than 2^−1132.

The decimal rational bounds above were checked against the exact integer endpoints in the saved certificate files. All degree labels and all strict pivot signs were independently re-read and checked. The differing terminal values reflect different harmless polynomial scalings; neither value is being asserted to be the numerical value of the quotient resultant. Nonzero terminal constants prove that resultant is nonzero.

## 7. What the certificate now establishes

The two exact specialized norms have gcd X, equivalently G−9/4, and both preserve their generic degree bound 64. The fixed degree-(64,64) first subresultant therefore has a nonzero linear coefficient at the witness.

Together with the universal Bezout identity and the connected physical-domain theorem, this supplies the promised generic formula

    G=−b/a,

where S₁(T)=aT+b is formed in an unshifted indeterminate T representing G. The shifted coordinate X=G−9/4 is used only by this witness certificate; it must not be confused with the generic formula's coordinate convention.

This establishes rational reconstruction of the squared last index conditional on the retained thirteen optical coordinates and data, in the norm-coefficient field. It does not establish a rational inverse from observations alone, a unique full optical system, uniform numerical stability, positive-noise rational reconstruction, or a practical global reconstruction algorithm. Zero-subresultant and other exceptional strata still require the complete fallback representation for arbitrary observations.

## 8. Unique geometry on every physical nonzero-subresultant chart

There is a further exact consequence, valid at every actual physical solution with a≠0, not only at the witness. Fix the retained thirteen optical coordinates and the observed data throughout this argument, and let G* be the physical common root.

Differentiate the universal Bezout identity

    aT+b=U(T)P(T)+V(T)Q(T)

at T=G*. Since P(G*)=Q(G*)=0,

    a=U(G*)P′(G*)+V(G*)Q′(G*)≠0.

Thus at least one norm polynomial has a simple root at G*. All five radicands are strictly positive near G*, so each of its 32 signed determinant factors is an analytic function of T there. Its physical positive-radical determinant factor vanishes at G*. A simple norm root therefore forces that physical determinant to have a simple root as well: Δ_C′(G*)≠0. It also forces the other factors in that particular norm to be nonzero there.

At the actual geometry z*, the exact column operation from the earlier derivative audit yields

    Δ_C′(G*)=det[A_C(G*), F_(C,G)(z*,G*)].

Hence this chart's five-by-five residual Jacobian in (b_x,b_y,g,d,G) is nonsingular. In particular, its five-by-four geometry matrix A_C has rank four. One of its four-row minors is a valid geometry pivot. The combined six-row geometry matrix necessarily has rank four as well.

Equivalently, the proposed rank-deficiency argument is correct: if both chart geometry matrices had rank at most three, their augmented matrices would have the same rank at a solution, each determinant would have at least a double zero, and both norms would have double roots. A nonzero linear Bezout combination is incompatible with that. The simple-root argument above supplies the stronger nonsingular five-variable Jacobian conclusion directly.

After computing G=−b/a, choose a nonzero four-row geometry minor in the qualifying chart and solve the four affine equations. The geometry solution is unique. Therefore all five eliminated scalar variables (G,b_x,b_y,g,d), and consequently the positive index n₃, are uniquely reconstructible conditional on the retained thirteen optical coordinates and exact observations on this chart.

This is conditional uniqueness only. Different retained optical coordinates can still produce distinct complete systems with the same data. Geometry recovery is rational in its affine coefficients, which include the positive radicals H_i=sqrt(G−|q_i|²); it is not asserted to be rational in the bare norm coefficients alone. Reconstructed candidates still require full-model, prior, and physical-branch validation. Geometry gauges can survive in the zero-subresultant fallback strata, not at a compatible point with a≠0.
