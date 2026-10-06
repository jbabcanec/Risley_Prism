"""Local structure diagnostics for the ORIGINAL passive full18 model.

These are numerical conditioning diagnostics, not uncertainty certificates.
Every parameter remains unknown; affine elimination is exact minimization,
not fixing those parameters. Original repository files are only imported.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_name] = "1"
sys.dont_write_bytecode = True

import numpy as np
from scipy.optimize import lsq_linear

REPO = Path(os.environ.get("WEDGE_RESEARCH_ROOT",
    r"C:\Users\josep\Dropbox\Babcanec Works\Mathematics\Wedge"))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from risley_lattice.model import LO, HI, RG, NAMES, vec2pat
from risley_lattice.separable import LINEAR, NONLINEAR, affine_geometry, ProjectedResidual


def smooth_forward(theta):
    b, a = affine_geometry(np.asarray(theta), 200, 10.)
    return b + a @ np.asarray(theta)[LINEAR]


def full_jacobian(theta):
    """Batched complex-step of the existing smooth affine forward map."""
    theta = np.asarray(theta, float)
    h = 1e-25
    batch = np.broadcast_to(theta, (14, 18)).astype(complex).copy()
    batch[np.arange(14), NONLINEAR] += 1j*h
    b, a = affine_geometry(batch, 200, 10.)
    j = np.empty((400, 18))
    j[:, NONLINEAR] = (b.imag/h + (a.imag/h) @ theta[LINEAR]).T
    j[:, LINEAR] = affine_geometry(theta, 200, 10.)[1]
    return j


def source_first_geometry(b, a, y):
    """Independent, unconstrained source-first four-affine least squares.

    Source columns have disjoint support. Remove each with its own exact
    one-column orthogonal projection; solve the remaining two columns by SVD.
    This equals joint four-column least squares. Bounded candidate solving
    still uses ProjectedResidual; clipping this solution is NOT equivalent.
    """
    rhs = np.asarray(y).ravel() - b
    s, e = a[:, 2:], a[:, :2]
    norms = np.sum(s*s, axis=0)
    if np.any(norms <= 0):
        raise np.linalg.LinAlgError("Zero source column")
    pe = e - s @ ((s.T @ e)/norms[:, None])
    pr = rhs - s @ ((s.T @ rhs)/norms)
    dg = np.linalg.lstsq(pe*RG[LINEAR[:2]], pr, rcond=None)[0]*RG[LINEAR[:2]]
    src = (s.T @ (rhs-e@dg))/norms
    return np.r_[dg, src], pe


def spectrum(matrix):
    s = np.linalg.svd(matrix, compute_uv=False)
    cutoff = np.finfo(float).eps*max(matrix.shape)*s[0]
    return dict(singular_values=s.tolist(), rank=int(np.sum(s > cutoff)),
                sigma_min=float(s[-1]), sigma_max=float(s[0]),
                condition=float(s[0]/s[-1]) if s[-1] else None,
                numerical_rank_cutoff=float(cutoff))


def native_noise_map(j):
    """Native-coordinate left inverse using box scaling only for numerics.

    For the linearized unconstrained LS estimator delta=K e and |e|<=eta,
    max |delta_i|=eta*sum_j |K_ij|. This is a first-order local diagnostic,
    not a nonlinear/global hard-noise guarantee. Rank loss is reported.
    """
    js = j*RG
    u, s, vt = np.linalg.svd(js, full_matrices=False)
    threshold = np.finfo(float).eps*max(js.shape)*s[0]
    rank = int(np.sum(s > threshold))
    invs = np.divide(1., s, out=np.zeros_like(s), where=s > threshold)
    k = RG[:, None]*((vt.T*invs)@u.T)
    return k, rank


def top_components(vector, names=NAMES, count=7):
    order = np.argsort(-np.abs(vector))[:count]
    return [{"name": names[i], "coefficient": float(vector[i])} for i in order]


def analyze(theta, target, error_levels=(1e-4, 1e-6, 1e-8, 1e-10)):
    """Inspect a declared parameter point, without using it to initialize a solve."""
    theta = np.asarray(theta, float)
    y = np.asarray(target).ravel()
    b, a = affine_geometry(theta, 200, 10.)
    f = b+a@theta[LINEAR]
    j = full_jacobian(theta)
    aa = a*RG[LINEAR]
    d = j[:, NONLINEAR]*RG[NONLINEAR]
    ua, sa, _ = np.linalg.svd(aa, full_matrices=False)
    arank = int(np.sum(sa > np.finfo(float).eps*max(aa.shape)*sa[0]))
    qa = ua[:, :arank]
    reduced = d-qa@(qa.T@d)
    _, _, vt = np.linalg.svd(reduced, full_matrices=False)
    weakq = vt[-1]
    weakc = -np.linalg.lstsq(aa, d@weakq, rcond=None)[0]
    lifted = np.empty(18)
    lifted[NONLINEAR], lifted[LINEAR] = weakq, weakc
    lifted /= np.linalg.norm(lifted)
    k, rank = native_noise_map(j)
    amp = np.sum(np.abs(k), axis=1)
    worst = int(np.argmax(amp))
    conditional = np.linalg.pinv(aa)*RG[LINEAR, None]
    c_amp = np.sum(np.abs(conditional), axis=1)
    direct = np.linalg.lstsq(aa, y-b-a@LO[LINEAR], rcond=None)[0]
    direct = LO[LINEAR]+RG[LINEAR]*direct
    seq, pe = source_first_geometry(b, a, y)
    projected = ProjectedResidual(theta, y, np.ones(400, bool), 200, 10.)
    projected.fun(theta[NONLINEAR])
    jp = projected.jac(theta[NONLINEAR])*RG[NONLINEAR]
    # Entrywise correlation of normalized native columns reveals coupling;
    # this is algebraic correlation of sensitivities, not statistical covariance.
    cn = np.linalg.norm(j*RG, axis=0)
    normalized = (j*RG)/np.maximum(cn, np.finfo(float).tiny)
    corr = normalized.T@normalized
    pairs = [(abs(corr[i, l]), i, l) for i in range(18) for l in range(i)]
    pairs.sort(reverse=True)
    noise = []
    for eta in error_levels:
        noise.append(dict(eta=float(eta), full_rank_linearized_map=rank==18,
                          single_fit_linearized_error_by_parameter=(eta*amp).tolist(),
                          compatible_pair_linearized_diameter_by_parameter=(2*eta*amp).tolist(),
                          worst_native_error=float(eta*amp[worst]),
                          worst_parameter=NAMES[worst]))
    return dict(
        theta=theta.tolist(), all_18_parameters_unknown=True,
        diagnostics_evaluated_at_declared_case_point=True,
        is_recovery_experiment=False, is_finite_noise_certificate=False,
        canonical_smooth_max_abs=float(np.max(np.abs(vec2pat(theta).ravel()-f))),
        full18_native=spectrum(j), full18_box_scaled=spectrum(j*RG),
        affine4_box_scaled=spectrum(aa), reduced14_box_scaled=spectrum(reduced),
        source_eliminated_geometry2_box_scaled=spectrum(pe*RG[LINEAR[:2]]),
        full_varpro_jacobian_box_scaled=spectrum(jp),
        varpro_residual_max_abs=float(np.max(np.abs(projected.residual))),
        affine_free_mask=projected.free.tolist(),
        schur_lift_unit_box_coordinates=lifted.tolist(),
        schur_lift_largest_components=top_components(lifted),
        schur_lift_native_direction=(RG*lifted).tolist(),
        schur_lift_response_norm=float(np.linalg.norm((j*RG)@lifted)),
        schur_projection_orthogonality=float(np.max(np.abs(qa.T@reduced))),
        sensitivity_rank=rank,
        native_linearized_error_amplification_l1=amp.tolist(),
        worst_native_sensitivity_parameter=NAMES[worst],
        local_linearized_eta_for_all_native_errors_001=float(.001/amp[worst]) if rank==18 else None,
        conditional_affine_only_amplification_l1=c_amp.tolist(),
        conditional_affine_note="Diagnostic with nonlinear coordinates held fixed; not the full18 uncertainty.",
        largest_sensitivity_column_correlations=[
            dict(first=NAMES[i], second=NAMES[l], dot_product=float(corr[i,l]))
            for _, i, l in pairs[:8]],
        source_first_vs_joint4_max_native_difference=float(np.max(np.abs(seq-direct))),
        source_first_vs_joint4_max_prediction_difference=float(np.max(np.abs(a@(seq-direct)))),
        bounded_affine_fit=projected.v[LINEAR].tolist(),
        error_budget_table=noise)


def five_point_jac(fun, theta, normalized_step):
    theta=np.asarray(theta, float)
    cols=[]
    for i in range(len(theta)):
        h=normalized_step*RG[i]
        e=np.zeros(18);e[i]=h
        cols.append((fun(theta-2*e)-8*fun(theta-e)+8*fun(theta+e)-fun(theta+2*e))/(12*h))
    return np.stack(cols, axis=1)


def validate_derivatives(theta, target, seed=2601002):
    """Three independent forward derivative calculations and profile checks."""
    from risley_lattice.fmodel import forward_iv
    theta=np.asarray(theta, float)
    j=full_jacobian(theta)
    fm, fr, ji, jr, margins=forward_iv(theta, np.zeros(18))
    result=dict(
        interval_ad_midpoint_relative_frobenius=float(np.linalg.norm((j-ji)*RG)/np.linalg.norm(ji*RG)),
        interval_ad_max_scaled_absolute_difference=float(np.max(np.abs((j-ji)*RG))),
        interval_ad_entrywise_enclosure_excess=float(np.max(np.abs(j-ji)-jr)),
        interval_point_vs_smooth_max_abs=float(np.max(np.abs(fm-smooth_forward(theta)))),
        interval_point_max_radius=float(np.max(fr)), margins=margins,
        finite_difference=[])
    for step in (1e-4, 1e-5, 1e-6):
        for label, fun in (("smooth", smooth_forward), ("canonical", lambda v: vec2pat(v).ravel())):
            fd=five_point_jac(fun, theta, step)
            result["finite_difference"].append(dict(model=label, box_normalized_step=step,
                relative_frobenius=float(np.linalg.norm((fd-j)*RG)/np.linalg.norm(j*RG)),
                maximum_column_relative_error=float(np.max(np.linalg.norm((fd-j)*RG, axis=0)/
                    np.maximum(np.linalg.norm(j*RG, axis=0), 1e-30)))))
    # Purposefully move the nonlinear candidate so residual-dependent terms
    # are exercised. These are derivative tests, not truth-start solver scores.
    rng=np.random.default_rng(seed)
    candidate=np.clip(theta+RG*2e-4*rng.uniform(-1,1,18), LO+1e-9*RG, HI-1e-9*RG)
    profile=ProjectedResidual(candidate, np.asarray(target).ravel(), np.ones(400,bool),200,10.)
    q=candidate[NONLINEAR]
    analytic=profile.jac(q)*RG[NONLINEAR]
    free=profile.free.copy()
    numerical=[]
    masks=[]
    h=2e-7
    for i in range(14):
        e=np.zeros(14);e[i]=h*RG[NONLINEAR[i]]
        vals=[]
        for t in (-2,-1,1,2):
            vals.append(profile.fun(q+t*e).copy())
            masks.append(profile.free.copy())
        numerical.append((vals[0]-8*vals[1]+8*vals[2]-vals[3])/(12*h))
    numerical=np.stack(numerical,axis=1)
    profile.fun(q)
    result["profile_jacobian_test"]=dict(
        residual_max_abs=float(np.max(np.abs(profile.residual))),
        same_active_set_all_perturbations=bool(all(np.array_equal(free,x) for x in masks)),
        free_affine_mask=free.tolist(),
        relative_frobenius=float(np.linalg.norm(numerical-analytic)/np.linalg.norm(analytic)),
        max_column_relative_error=float(np.max(np.linalg.norm(numerical-analytic,axis=0)/
            np.maximum(np.linalg.norm(analytic,axis=0),1e-30))))
    return result


def validate_poc3(theta):
    """Independent parity audit of the new paired-face three-prism transfer."""
    solver_path=Path(__file__).parents[1]/"solver"
    sys.path.insert(0,str(solver_path))
    import poc3
    theta=np.asarray(theta,float)
    b,a=affine_geometry(theta,200,10.)
    bb,aa,guards,penalty=poc3.affine(theta,200,.05,safe_trials=False)
    h=1e-25
    vv=np.broadcast_to(theta,(18,18)).astype(complex).copy()
    vv[np.arange(18),np.arange(18)]+=1j*h
    bc,ac,_,_=poc3.affine(vv,200,.05,safe_trials=False)
    jnew=(bc+np.einsum('kmi,ki->km',ac,vv[:,LINEAR])).imag.T/h
    jold=full_jacobian(theta)
    return dict(
        old_vs_new_base_max_abs=float(np.max(abs(bb-b))),
        old_vs_new_design_max_abs=float(np.max(abs(aa-a))),
        old_vs_new_forward_max_abs=float(np.max(abs(bb+aa@theta[LINEAR]-smooth_forward(theta)))),
        canonical_vs_new_forward_max_abs=float(np.max(abs(poc3.forward(theta)-vec2pat(theta)))),
        scaled_jacobian_relative_frobenius=float(np.linalg.norm((jnew-jold)*RG)/np.linalg.norm(jold*RG)),
        scaled_jacobian_max_abs=float(np.max(abs((jnew-jold)*RG))),
        minimum_trial_guard=float(np.min(guards)),maximum_penalty=float(np.max(abs(penalty))))
