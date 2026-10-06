"""Small exact arithmetic controls; no optical model or optimizer is imported.

This verifies algebraic consequences and uses already saved strict optical
witness certificates as inputs. It is not a new optical interval verifier and
does not close the original full-prior cover.
"""
from fractions import Fraction as Q
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def dot(a, b):
    return sum((x*y for x, y in zip(a, b)), Q(0))


def support(lo, hi, v):
    return sum((max(a*w, b*w) for a, b, w in zip(lo, hi, v)), Q(0))


def directional_upper(jac, residual, eta, lo, hi, lam, v, remainder_lo):
    """Equation (7), evaluated exactly for rational submitted data."""
    defect = [v[j]-sum((lam[i]*jac[i][j] for i in range(len(lam))), Q(0))
              for j in range(len(v))]
    return (-dot(lam, residual)+dot([abs(x) for x in lam], eta)
            +support(lo, hi, defect)-remainder_lo)


def fraction_record(q):
    return {"exact": str(q), "decimal_approximation": float(q)}


def run():
    reports = {}
    # Exact affine inverse: observation errors (+eta,-eta) attain the bound.
    jac = [[Q(1), Q(1)], [Q(1), Q(101, 100)]]
    eta = [Q(1, 10000)]*2
    lo, hi = [Q(-1)]*2, [Q(1)]*2
    residual = [Q(0)]*2
    v = [Q(1), Q(0)]
    exact_lam = [Q(101), Q(-100)]
    tight = directional_upper(jac, residual, eta, lo, hi, exact_lam, v, Q(0))
    extremum = [Q(201, 10000), Q(-1, 50)]
    assert [dot(row, extremum) for row in jac] == [eta[0], -eta[1]]
    assert tight == extremum[0] == Q(201, 10000)
    # An inexact dual is safe only with its support-function defect correction.
    approx_lam = [Q(100), Q(-99)]
    loose = directional_upper(jac, residual, eta, lo, hi, approx_lam, v, Q(0))
    assert loose == Q(299, 10000) and loose >= tight
    uncorrected = dot([abs(x) for x in approx_lam], eta)
    assert uncorrected < tight  # Dropping that correction really is unsound.
    reports["affine_ill_conditioned_control"] = {
        "sharp_coordinate_upper": fraction_record(tight),
        "inexact_dual_safe_upper": fraction_record(loose),
        "inexact_dual_uncorrected_unsound_upper": fraction_record(uncorrected),
        "extremum_attains_sharp_bound": True,
    }

    # Nonlinear scalar combination:
    # f1=x+x^2+y^2; f2=y+2xy; f1-f2=(x-y)+(x-y)^2.
    # Thus lambda=(1,-1) has remainder >=0 exactly, including its cross term.
    # Verify the polynomial identity coefficient by coefficient with rationals.
    f1 = {(1, 0): Q(1), (2, 0): Q(1), (0, 2): Q(1)}
    f2 = {(0, 1): Q(1), (1, 1): Q(2)}
    difference = {p: f1.get(p, Q(0))-f2.get(p, Q(0)) for p in set(f1)|set(f2)}
    assert difference == {(1, 0): Q(1), (0, 1): Q(-1),
                          (2, 0): Q(1), (1, 1): Q(-2), (0, 2): Q(1)}
    nonlinear_bound = directional_upper(
        [[Q(1), Q(0)], [Q(0), Q(1)]], [Q(0)]*2, [Q(1, 100)]*2,
        [Q(-1, 10)]*2, [Q(1, 10)]*2, [Q(1), Q(-1)],
        [Q(1), Q(-1)], Q(0))
    assert nonlinear_bound == Q(1, 50)
    compatible_grid_count = 0
    for ix in range(-50, 51):
        for iy in range(-50, 51):
            x, y = Q(ix, 500), Q(iy, 500)
            values = [x+x*x+y*y, y+2*x*y]
            if max(map(abs, values)) <= Q(1, 100):
                compatible_grid_count += 1
                assert x-y <= nonlinear_bound
    reports["signed_quadratic_control"] = {
        "exact_identity_checked": "f1-f2=(x-y)+(x-y)^2",
        "signed_remainder_lower": "0 by an exact square",
        "direction_upper": fraction_record(nonlinear_bound),
        "coarser_symmetric_absolute_coefficient_upper": fraction_record(Q(3, 50)),
        "compatible_rational_grid_points_checked": compatible_grid_count,
        "scope": "Polynomial identity proves nonnegative remainder; grid is a regression control, not its proof.",
    }

    # Narrow local boxes do not imply a narrow global uncertainty cover.
    boxes = [(Q(-1), Q(-999, 1000)), (Q(999, 1000), Q(1))]
    assert all(b-a == Q(1, 1000) for a, b in boxes)
    global_lower = min(a for a, b in boxes)
    global_upper = max(b for a, b in boxes)
    assert (global_upper-global_lower)/2 == 1
    candidate = Q(1999, 2000)
    candidate_error = max(abs(a-candidate) for box in boxes for a in box)
    assert candidate_error == Q(3999, 2000)
    reports["disconnected_cover_control"] = {
        "each_leaf_width": "1/1000",
        "global_minimax_radius": "1",
        "candidate_near_right_leaf_global_error": fraction_record(candidate_error),
    }

    # Exact deductions from independently saved interval optical certificates.
    # We do not reevaluate F here. Each recorded upper is treated as the
    # already-certified upper bound it purports to be, with explicit provenance.
    specs = [("benchmark_verified_100.json", Q(1, 100000000)),
             ("ordinary_noise_verified.json", Q(1, 10000))]
    lo18 = [-3.5]*3+[-18]*6+[1.3]*3+[50, 2, -25, -25, -5, -5]
    hi18 = [3.5]*3+[18]*6+[1.8]*3+[200, 15, 25, 25, 5, 5]
    witnesses = []
    for filename, allowance in specs:
        path = ROOT/"ambiguity"/filename
        data = json.loads(path.read_text(encoding="utf-8"))
        assert len(data["records"]) == 1
        rec = data["records"][0]
        assert rec["status"] == "DIRECTED_ARBITRARY_PRECISION_INTERVAL_ENDPOINT_WITNESS"
        assert rec["inside_native_box"] and rec["prism_order_unchanged"]
        assert len(rec["common_observations"]) == 400
        assert all(isinstance(value, (float, int)) for value in rec["common_observations"])
        for guards in [rec["guards_base"], rec["guards_alternative"]]:
            assert all(Q.from_float(float(x)) > 0 for x in guards.values())
        base = list(map(lambda x: Q.from_float(float(x)), rec["base"]))
        alternative = list(map(lambda x: Q.from_float(float(x)), rec["alternative"]))
        assert len(base) == len(alternative) == 18
        for point in [base, alternative]:
            assert all(Q.from_float(float(a)) <= x <= Q.from_float(float(b))
                       for a, x, b in zip(lo18, point, hi18))
        upper = max(Q.from_float(float(rec[key])) for key in
                    ["base_observation_error_upper", "alternative_observation_error_upper"])
        assert upper <= allowance
        half_separations = [abs(a-b)/2 for a, b in zip(base, alternative)]
        assert half_separations[12] > Q(1, 1000)
        witnesses.append({
            "source": str(path.relative_to(ROOT)),
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "noise_allowance": fraction_record(allowance),
            "saved_common_record_upper": fraction_record(upper),
            "distance_minimax_lower": fraction_record(half_separations[12]),
            "all18_coordinate_minimax_lowers": [fraction_record(x) for x in half_separations],
            "refutes_uniform_0_001_native_recovery_for_this_record": True,
            "scope": "Exact endpoint-distance consequence of saved strict optical verification; no new optical execution.",
        })
    reports["saved_strict_witness_consequences"] = witnesses
    reports["status"] = "ALL_EXACT_ALGEBRA_CONTROLS_PASSED"
    reports["global_optical_cover_certified"] = False
    reports["optical_model_executed"] = False
    reports["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output = Path(__file__).with_suffix(".json")
    output.write_text(json.dumps(reports, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"status": reports["status"], "output": str(output),
                      "grid_compatible_count": compatible_grid_count,
                      "global_optical_cover_certified": False}, indent=2))


if __name__ == "__main__":
    run()
