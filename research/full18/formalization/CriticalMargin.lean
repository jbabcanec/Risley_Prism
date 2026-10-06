import Mathlib.Analysis.Real.Sqrt
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

/-!
Final-prism data-dependent critical-normal margin, in explicit real coordinates.
Every geometry, weak-branch, and observation assumption is in the statements.
No original-project module or other local formalization is imported.
-/

set_option autoImplicit false

namespace RisleyCriticalMargin

/-- The final-plane identity follows from the two outgoing intersection equations,
the external axial flight B, and the outgoing normal component R. -/
theorem final_plane_identity
    (ux uy px py X Y Z B d R Fx Fy : ℝ)
    (hZ : Z ≠ 0)
    (hFx : Fx = px + (X/Z)*B)
    (hFy : Fy = py + (Y/Z)*B)
    (hB : B = d - (ux*px+uy*py))
    (hR : R = Z - (ux*X+uy*Y)) :
    Z*(d-(ux*Fx+uy*Fy)) = R*B := by
  rw [hFx, hFy, hR, hB]
  field_simp
  ring

/-- Weak internal and external traversal give the required flight interval. -/
theorem weak_flight_bounds (ux uy px py B d : ℝ)
    (hB : B = d-(ux*px+uy*py))
    (hInternal : 0 ≤ 3+(ux*px+uy*py))
    (hExternal : 0 ≤ B) :
    0 ≤ B ∧ B ≤ 3+d := by
  constructor
  · exact hExternal
  · linarith

/-- The real two-dimensional Lagrange identity used for the screen-radius bound. -/
theorem lagrange_identity (ux uy Fx Fy : ℝ) :
    (ux*Fx+uy*Fy)^2 + (ux*Fy-uy*Fx)^2 =
      (ux^2+uy^2)*(Fx^2+Fy^2) := by
  ring

/-- Componentwise screen bounds and the slope disk imply the exact dot-product bound. -/
theorem screen_box_dot_bound (ux uy Fx Fy ax ay uStar : ℝ)
    (huStar : 0 ≤ uStar)
    (hu : ux^2+uy^2 ≤ uStar^2)
    (hFx : |Fx| ≤ ax) (hFy : |Fy| ≤ ay) :
    ux*Fx+uy*Fy ≤ uStar*Real.sqrt (ax^2+ay^2) := by
  have hax : 0 ≤ ax := le_trans (abs_nonneg Fx) hFx
  have hay : 0 ≤ ay := le_trans (abs_nonneg Fy) hFy
  have hx2 : Fx^2 ≤ ax^2 := by
    have habs2 := (sq_le_sq₀ (abs_nonneg Fx) hax).2 hFx
    simpa only [sq_abs] using habs2
  have hy2 : Fy^2 ≤ ay^2 := by
    have habs2 := (sq_le_sq₀ (abs_nonneg Fy) hay).2 hFy
    simpa only [sq_abs] using habs2
  have hcs : (ux*Fx+uy*Fy)^2 ≤ (ux^2+uy^2)*(Fx^2+Fy^2) := by
    have hid := lagrange_identity ux uy Fx Fy
    nlinarith [sq_nonneg (ux*Fy-uy*Fx)]
  have hnorm := mul_le_mul_of_nonneg_right hu
    (add_nonneg (sq_nonneg Fx) (sq_nonneg Fy))
  have hbox := mul_le_mul_of_nonneg_left (add_le_add hx2 hy2) (sq_nonneg uStar)
  have hdot2 : (ux*Fx+uy*Fy)^2 ≤ uStar^2*(ax^2+ay^2) :=
    le_trans hcs (le_trans hnorm hbox)
  have hsqrt := Real.sq_sqrt (add_nonneg (sq_nonneg ax) (sq_nonneg ay))
  have hM : 0 ≤ uStar*Real.sqrt (ax^2+ay^2) :=
    mul_nonneg huStar (Real.sqrt_nonneg _)
  have hdotM : (ux*Fx+uy*Fy)^2 ≤ (uStar*Real.sqrt (ax^2+ay^2))^2 := by
    simpa only [mul_pow, hsqrt] using hdot2
  have habs : |ux*Fx+uy*Fy| ≤ uStar*Real.sqrt (ax^2+ay^2) := by
    apply (sq_le_sq₀ (abs_nonneg _) hM).1
    simpa only [sq_abs] using hdotM
  exact le_trans (le_abs_self _) habs

/-- Scalar margin step: lower distance 50 and thickness 3 produce the divisor 53. -/
theorem scalar_ratio_margin (kappa B d dot M : ℝ)
    (hkappa : 0 ≤ kappa) (hM : 0 ≤ M) (hd : 50 ≤ d)
    (hBupper : B ≤ 3+d)
    (hIdentity : d-dot = kappa*B)
    (hDot : dot ≤ M) :
    max 0 ((50-M)/53) ≤ kappa := by
  have hden : 0 < 3+d := by linarith
  have hprod := mul_le_mul_of_nonneg_left hBupper hkappa
  have hbound : (d-M)/(3+d) ≤ kappa := by
    apply (div_le_iff₀ hden).2
    nlinarith
  have hmono : (50-M)/53 ≤ (d-M)/(3+d) := by
    apply (div_le_div_iff₀ (by norm_num : (0:ℝ)<53) hden).2
    nlinarith [mul_nonneg (sub_nonneg.mpr hd) (show 0 ≤ 3+M by linarith)]
  exact max_le hkappa (le_trans hmono hbound)

/-- Complete linked final-prism ratio certificate from the displayed physical
intersection equations, weak traversal, and screen-coordinate bounds.
uStar can be specialized to tan(pi/10), ax,ay to |y_x|+epsilon,|y_y|+epsilon.
No conclusion about upstream prisms or all-sample feasibility is asserted here. -/
theorem final_prism_ratio_margin
    (ux uy px py X Y Z B d R Fx Fy ax ay uStar : ℝ)
    (hZ : 0 < Z) (hRnonneg : 0 ≤ R)
    (hFx : Fx = px + (X/Z)*B)
    (hFy : Fy = py + (Y/Z)*B)
    (hB : B = d-(ux*px+uy*py))
    (hR : R = Z-(ux*X+uy*Y))
    (hInternal : 0 ≤ 3+(ux*px+uy*py))
    (hExternal : 0 ≤ B)
    (hd : 50 ≤ d)
    (huStar : 0 ≤ uStar)
    (hu : ux^2+uy^2 ≤ uStar^2)
    (hScreenX : |Fx| ≤ ax) (hScreenY : |Fy| ≤ ay) :
    max 0 ((50-uStar*Real.sqrt (ax^2+ay^2))/53) ≤ R/Z := by
  have hid := final_plane_identity ux uy px py X Y Z B d R Fx Fy
    (ne_of_gt hZ) hFx hFy hB hR
  have hflight := weak_flight_bounds ux uy px py B d hB hInternal hExternal
  have hdot := screen_box_dot_bound ux uy Fx Fy ax ay uStar huStar hu hScreenX hScreenY
  have hquot : d-(ux*Fx+uy*Fy) = (R/Z)*B := by
    apply (mul_left_cancel₀ (ne_of_gt hZ))
    calc
      Z*(d-(ux*Fx+uy*Fy)) = R*B := hid
      _ = Z*((R/Z)*B) := by field_simp
  exact scalar_ratio_margin (R/Z) B d (ux*Fx+uy*Fy)
    (uStar*Real.sqrt (ax^2+ay^2))
    (div_nonneg hRnonneg (le_of_lt hZ))
    (mul_nonneg huStar (Real.sqrt_nonneg _)) hd hflight.2 hquot hdot

/-- A measured-coordinate residual bound implies the screen-coordinate box.
Nonnegativity of epsilon is already implied by the hypothesis. -/
theorem observation_screen_bound (F y epsilon : ℝ)
    (hError : |F-y| ≤ epsilon) : |F| ≤ |y|+epsilon := by
  calc
    |F| = |(F-y)+y| := by congr 1; ring
    _ ≤ |F-y|+|y| := abs_add_le _ _
    _ ≤ |y|+epsilon := by linarith

/-- The final-prism certificate stated directly in terms of measured y and a
single shared componentwise error allowance epsilon. The two error hypotheses
themselves imply epsilon>=0. The physical trace hypotheses are not suppressed. -/
theorem final_prism_observation_margin
    (ux uy px py X Y Z B d R Fx Fy yx yy epsilon uStar : ℝ)
    (hZ : 0 < Z) (hRnonneg : 0 ≤ R)
    (hFx : Fx = px + (X/Z)*B)
    (hFy : Fy = py + (Y/Z)*B)
    (hB : B = d-(ux*px+uy*py))
    (hR : R = Z-(ux*X+uy*Y))
    (hInternal : 0 ≤ 3+(ux*px+uy*py))
    (hExternal : 0 ≤ B)
    (hd : 50 ≤ d)
    (huStar : 0 ≤ uStar)
    (hu : ux^2+uy^2 ≤ uStar^2)
    (hErrorX : |Fx-yx| ≤ epsilon) (hErrorY : |Fy-yy| ≤ epsilon) :
    max 0 ((50-uStar*Real.sqrt ((|yx|+epsilon)^2+(|yy|+epsilon)^2))/53) ≤ R/Z := by
  exact final_prism_ratio_margin ux uy px py X Y Z B d R Fx Fy
    (|yx|+epsilon) (|yy|+epsilon) uStar
    hZ hRnonneg hFx hFy hB hR hInternal hExternal hd huStar hu
    (observation_screen_bound Fx yx epsilon hErrorX)
    (observation_screen_bound Fy yy epsilon hErrorY)

#print axioms final_plane_identity
#print axioms weak_flight_bounds
#print axioms lagrange_identity
#print axioms screen_box_dot_bound
#print axioms scalar_ratio_margin
#print axioms final_prism_ratio_margin
#print axioms observation_screen_bound
#print axioms final_prism_observation_margin

end RisleyCriticalMargin
