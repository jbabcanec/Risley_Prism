# Independent model and certificate review contract

This note records a read-only source review before implementation testing. It is not a pass of the eventual executable.

## Correct model and provenance

The engine must implement the coupled two-transverse-coordinate model in [physical_vector_inverse.md](../theory/physical_vector_inverse.md), sections 1–2, and [global_boundary_compactification.md](../theory/global_boundary_compactification.md), section 1. The scaled alternative is in [explicit_six_root_compiler.md](../theory/explicit_six_root_compiler.md), sections 1–2.

The original Dropbox code inspected read-only on 2026-10-03 is axis-separated: risley_lattice/fmodel.py defines _trace_axis, and physical_jet.py propagates inside a loop over the two axes. The workspace's stable_model/forward.py explicitly labels itself independent-axis. These are not suitable full-vector parity oracles. No original code was executed or edited for this review. The already identified 200-pair canonical record is not a qualifying full-vector input.

## Eighteen-coordinate contract

The fourteen optical coordinates are three signed wedge slopes r=tan(a), three phase charts p=tan(phi/2), three speed charts v=tan(pi N/20), three native refractive indices, and two incident slopes t=(tan(beta_x),tan(beta_y)). The four affine geometry coordinates are q=(b_x,b_y,g,d).

With rot(w)=((1-w^2)/(1+w^2),2w/(1+w^2)), sample k uses the complex product u_j(k)=r_j rot(p_j)rot(v_j)^k. This is exactly the specified clock k/20, for k=0,...,199, without a floating clock or numerical trigonometric conversion.

The source-to-first-entrance state is X=t/sqrt(1+|t|^2), z=1/sqrt(1+|t|^2), position b+6t. At each prism:
- H=sqrt(n^2-|X|^2), P=H-u dot X, D=1+|u|^2.
- Delta=P^2-D(n^2-1), R=sqrt(Delta), c=(P-R)/D.
- X_out=X+c u, Z=H-c.
- A=3+u dot position; p_exit=position+X A/P.
- B=ell-u dot p_exit; p_next=p_exit+(X_out/Z)B, with ell=g,g,d.

For a strict physical trace, H,P,Z,R,A,B are positive at every prism/sample. Delta>0 is equivalent to R>0 on the selected root branch. Weak critical boundaries are genuine limits but not strict physical solutions.

## Affine geometry invariant

Represent each position and traversal scalar by five interval coefficients in the order (b_x,b_y,g,d,constant). Optical coefficients may be outward enclosures; geometry must remain shared through the complete recurrence. Every ray direction is independent of geometry. Only addition and optical-scalar multiplication of affine forms occur, so the invariant is exact symbolically.

Evaluating each resulting affine form over the entire geometry rectangle is a sound outer enclosure. It need not decide joint feasibility across rows; independently intersecting row ranges does not create a shared feasible geometry. No LP, root completeness, parameter uniqueness, or 0.001 all-coordinate estimate follows from affine enclosure alone.

## Outcomes and evidence obligations

An excluded box needs an explicit contradiction with outward bounds: for example an upper bound <=0 on a required strict physical margin, a negative radicand upper bound, or a screen interval disjoint from the full observation interval. A denominator overlapping zero, a radicand straddling zero, or a failed interval sign test is unresolved unless another valid exclusion already exists.

Retained must state its precise meaning. A proved all-physical forward enclosure covers every parameter in the admitted box but is not an inverse certificate. Mere observation overlap is unresolved. A whole-box compatibility assertion requires every output enclosure to lie within the observation allowance and every strict physical lower bound to be positive.

Prior handling must use exact algebraic tests or outward bounds. An input box extending outside an irrational prior endpoint may serve as an outer cover; it must not be called wholly inside the original prior.

The final-prism data certificate uses the original global wedge bound and distance lower bound, all 200 paired observations, and epsilon>=0. With M=u_*sqrt((|y_x|+epsilon)^2+(|y_y|+epsilon)^2), a rigorously rounded upper M gives R_3/Z_3>=max(0,(50-M)/53). Target-margin exclusion requires the strict squared inequality and 0<=mu<50/53. It does not require a fitted system and does not establish record consistency.

## Planned independent executable review

Read the frozen interval and engine sources, check integer/rational rounding, replay saved certificate requests/results, reject tampered evidence, and validate one clearly labeled deterministic synthetic full-vector spot-check if provided. No sweep, fit, original-project execution or hardware-success claim is authorized by this review.

