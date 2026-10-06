import Mathlib.Analysis.Real.Sqrt
import Mathlib.LinearAlgebra.Matrix.Determinant.Basic
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.FinCases
import Mathlib.Tactic.LinearCombination
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

/-!
Bounded supporting formalization for the reduced-algebra and boundary addenda.
This file does not formalize the full optical model or its inverse theorem.
The coordinate statements below expose every algebraic/positivity hypothesis.
-/

set_option autoImplicit false

namespace RisleySupport

/-- The determinant of I - a u^T in two transverse dimensions. -/
theorem entrance_det (a₁ a₂ u₁ u₂ : ℝ) :
    Matrix.det !![1 - a₁*u₁, -a₁*u₂; -a₂*u₁, 1-a₂*u₂] =
      1 - (u₁*a₁ + u₂*a₂) := by
  rw [Matrix.det_fin_two]
  simp
  ring

/-- Coordinate representation of (I-vu^T)(I-au^T)⁻¹. -/
noncomputable def transport (a₁ a₂ v₁ v₂ u₁ u₂ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![1 + (a₁-v₁)*u₁/(1-(u₁*a₁+u₂*a₂)),
       (a₁-v₁)*u₂/(1-(u₁*a₁+u₂*a₂));
       (a₂-v₂)*u₁/(1-(u₁*a₁+u₂*a₂)),
     1 + (a₂-v₂)*u₂/(1-(u₁*a₁+u₂*a₂))]

/-- The transport really factors the incoming/outgoing intersection matrices. -/
theorem transport_factorization (a₁ a₂ v₁ v₂ u₁ u₂ : ℝ)
    (ha : 1-(u₁*a₁+u₂*a₂) ≠ 0) :
    transport a₁ a₂ v₁ v₂ u₁ u₂ *
      !![1-a₁*u₁, -a₁*u₂; -a₂*u₁, 1-a₂*u₂] =
      !![1-v₁*u₁, -v₁*u₂; -v₂*u₁, 1-v₂*u₂] := by
  ext i j
  fin_cases i <;> fin_cases j
  all_goals simp only [Matrix.mul_apply, Fin.sum_univ_two]
  all_goals dsimp [transport]
  all_goals generalize hd : (1-(u₁*a₁+u₂*a₂)) = da at *
  all_goals field_simp [ha]
  all_goals rw [← hd]
  all_goals ring

/-- The source-transport determinant ratio; the entrance denominator is nonzero. -/
theorem transport_det (a₁ a₂ v₁ v₂ u₁ u₂ : ℝ)
    (ha : 1-(u₁*a₁+u₂*a₂) ≠ 0) :
    Matrix.det (transport a₁ a₂ v₁ v₂ u₁ u₂) =
      (1-(u₁*v₁+u₂*v₂))/(1-(u₁*a₁+u₂*a₂)) := by
  rw [Matrix.det_fin_two]
  dsimp [transport]
  generalize hd : (1-(u₁*a₁+u₂*a₂)) = da at *
  field_simp [ha]
  rw [← hd]
  ring

/-- The proposed inverse is the reversed transport, with both physical normal guards. -/
theorem transport_inverse (a₁ a₂ v₁ v₂ u₁ u₂ : ℝ)
    (ha : 1-(u₁*a₁+u₂*a₂) ≠ 0)
    (hv : 1-(u₁*v₁+u₂*v₂) ≠ 0) :
    transport a₁ a₂ v₁ v₂ u₁ u₂ * transport v₁ v₂ a₁ a₂ u₁ u₂ = 1 := by
  ext i j
  fin_cases i <;> fin_cases j
  all_goals simp only [Matrix.mul_apply, Fin.sum_univ_two, Matrix.one_apply, if_true]
  all_goals dsimp [transport]
  all_goals generalize hd : (1-(u₁*a₁+u₂*a₂)) = da at *
  all_goals generalize he : (1-(u₁*v₁+u₂*v₂)) = dv at *
  all_goals field_simp [ha, hv]
  all_goals rw [← hd, ← he]
  all_goals ring

/-- Positive incoming and outgoing normal factors give a positive determinant. -/
theorem transport_det_pos (a₁ a₂ v₁ v₂ u₁ u₂ : ℝ)
    (ha : 0 < 1-(u₁*a₁+u₂*a₂))
    (hv : 0 < 1-(u₁*v₁+u₂*v₂)) :
    0 < Matrix.det (transport a₁ a₂ v₁ v₂ u₁ u₂) := by
  rw [transport_det a₁ a₂ v₁ v₂ u₁ u₂ (ne_of_gt ha)]
  exact div_pos hv ha

/-- One component of the combined position numerator, using P=H-u·X.
Here alpha=u·A and dotX=u·X; Xi and Yi are corresponding vector components. -/
theorem combined_spatial_recurrence
    (H Z dotX alpha Xi Yi Ai T ell : ℝ)
    (hP : H-dotX ≠ 0) (hZ : Z ≠ 0) (hT : T ≠ 0) :
    Ai/T + Xi*(3+alpha/T)/(H-dotX) +
      (Yi/Z)*(ell - (alpha/T + dotX*(3+alpha/T)/(H-dotX))) =
    (Z*(H-dotX)*Ai + (Z*Xi-H*Yi)*alpha +
      3*(Z*Xi-dotX*Yi)*T + ell*(H-dotX)*Yi*T) /
      (Z*(H-dotX)*T) := by
  field_simp
  ring

/-- Algebraic critical inward derivative identity. The hypothesis is Delta=0.
The quantity (H*b+a)/(1+b) is the critical outgoing axial direction Z. -/
theorem critical_inward_identity (H a b nu : ℝ)
    (hb : 1+b ≠ 0)
    (hc : (H-a)^2 = (1+b)*nu) :
    -2*(a*(H-a)+b*nu) = -2*(H-a)*((H*b+a)/(1+b)) := by
  field_simp
  linear_combination b*hc

/-- The critical derivative is strictly negative when the physical guards hold. -/
theorem critical_inward_negative (H a b nu : ℝ)
    (hb : 1+b ≠ 0) (hc : (H-a)^2 = (1+b)*nu)
    (hP : 0 < H-a) (hZ : 0 < (H*b+a)/(1+b)) :
    -2*(a*(H-a)+b*nu) < 0 := by
  rw [critical_inward_identity H a b nu hb hc]
  nlinarith [mul_pos hP hZ]

/-- At an active external flight constraint, its derivative simplifies to -H*ell/P. -/
theorem external_flight_inward_identity (H a b ell : ℝ)
    (hP : H-a ≠ 0) (hactive : ell = (H*b+3*a)/(H-a)) :
    -H*(H*b+3*a)/(H-a)^2 = -H*ell/(H-a) := by
  rw [hactive]
  field_simp

/-- The preceding derivative is strictly negative for positive internal H, flight, and P. -/
theorem external_flight_inward_negative (H a b ell : ℝ)
    (hP : 0 < H-a) (hH : 0 < H) (hell : 0 < ell)
    (hactive : ell = (H*b+3*a)/(H-a)) :
    -H*(H*b+3*a)/(H-a)^2 < 0 := by
  rw [external_flight_inward_identity H a b ell (ne_of_gt hP) hactive]
  exact div_neg_of_neg_of_pos (mul_neg_of_neg_of_pos (neg_neg_of_pos hH) hell) hP

/-- The square-root modulus used by the hierarchical strictification and global modulus. -/
theorem sqrt_difference_bound (s t : ℝ) (hs : 0 ≤ s) (ht : 0 ≤ t) :
    |Real.sqrt s - Real.sqrt t| ≤ Real.sqrt |s-t| := by
  have hs0 := Real.sqrt_nonneg s
  have ht0 := Real.sqrt_nonneg t
  have hd0 := Real.sqrt_nonneg |s-t|
  have hss := Real.sq_sqrt hs
  have htt := Real.sq_sqrt ht
  have hdd := Real.sq_sqrt (abs_nonneg (s-t))
  rcases le_total t s with hts | hst
  · have hr : Real.sqrt t ≤ Real.sqrt s := Real.sqrt_le_sqrt hts
    rw [abs_of_nonneg (sub_nonneg.mpr hr)]
    rw [abs_of_nonneg (sub_nonneg.mpr hts)] at hd0 hdd ⊢
    nlinarith [mul_nonneg ht0 (sub_nonneg.mpr hr)]
  · have hr : Real.sqrt s ≤ Real.sqrt t := Real.sqrt_le_sqrt hst
    rw [abs_of_nonpos (sub_nonpos.mpr hr)]
    rw [abs_of_nonpos (sub_nonpos.mpr hst)] at hd0 hdd ⊢
    nlinarith [mul_nonneg hs0 (sub_nonneg.mpr hr)]

#print axioms entrance_det
#print axioms transport_factorization
#print axioms transport_det
#print axioms transport_inverse
#print axioms transport_det_pos
#print axioms combined_spatial_recurrence
#print axioms critical_inward_identity
#print axioms critical_inward_negative
#print axioms external_flight_inward_identity
#print axioms external_flight_inward_negative
#print axioms sqrt_difference_bound

end RisleySupport
