"""One deterministic, labeled synthetic full-vector proof spot-check.

No random sampling, fitting, inverse search, hardware record, or canonical record.
All cases use the same 200-pair rational record. The small box is supplied, not recovered.
"""
from copy import deepcopy
from fractions import Fraction as F
from pathlib import Path
import json
import time
from engine import (MODEL, NAMES, certify, dump_json, prior_limits,
                    precision, digest, verify)

HERE = Path(__file__).resolve().parent
OUT = HERE / "spotcheck"
POINT = {
    "r1": "1/100", "r2": "-1/80", "r3": "1/120",
    "p1": "1/50", "p2": "-1/40", "p3": "1/60",
    "v1": "1/200", "v2": "1/125", "v3": "-1/175",
    "n1": "7/5", "n2": "3/2", "n3": "8/5",
    "tx": "1/20", "ty": "-1/30",
    "bx": "1/2", "by": "-1/4", "g": "5", "d": "100",
}


def template(box):
    return {"model": MODEL, "provenance": "synthetic_fullvector_proof_spotcheck",
            "description": "One constructed algebraic point; not hardware or inverse recovery",
            "sampling": {"count": 200, "step": "1/20", "start": "0"},
            "timestamps": [str(F(k, 20)) for k in range(200)], "bits": 80, "box": box}


def run():
    started = time.perf_counter()
    OUT.mkdir(exist_ok=True)
    point_request = template({n: [POINT[n], POINT[n]] for n in NAMES})
    point_forward = certify(point_request)
    assert point_forward["status"] == "retained"
    assert len(point_forward["samples"]) == 200
    observations = [[str((F(a)+F(b))/2) for a, b in s["output"]]
                    for s in point_forward["samples"]]
    point_error = max((F(b)-F(a))/2 for s in point_forward["samples"]
                      for a, b in s["output"])

    # Every one of the 18 coordinates has nonzero uncertainty; this box is not fitted.
    radius = F(1, 10**8)
    small_box = {n: [str(F(POINT[n])-radius), str(F(POINT[n])+radius)] for n in NAMES}
    small_forward_request = template(small_box)
    small_forward = certify(small_forward_request)
    assert small_forward["status"] == "retained"
    # Derive the validation allowance from the rigorous box envelope, not a noise sweep.
    allowance = max(max(abs(F(a)-F(observations[k][c])),
                        abs(F(b)-F(observations[k][c])))
                    for k, s in enumerate(small_forward["samples"])
                    for c, (a, b) in enumerate(s["output"]))
    allowance += F(1, 1 << 80)
    record = {"model": MODEL, "provenance": "synthetic_fullvector_proof_spotcheck",
              "point_chart": POINT, "sampling": point_request["sampling"],
              "timestamps": point_request["timestamps"], "observations": observations,
              "point_rounding_enclosure_error": str(point_error),
              "box_validation_allowance": str(allowance),
              "warning": "Synthetic proof fixture. The parameter box was supplied, not reconstructed."}
    dump_json(OUT/"synthetic_fullvector_record.json", record)
    dump_json(OUT/"point_forward_input.json", point_request)
    dump_json(OUT/"point_forward_certificate.json", point_forward)

    def with_data(box):
        req = template(box)
        req.update(observations=observations, epsilon=str(allowance))
        return req

    cases = {"small_box": with_data(small_box)}
    far = deepcopy(small_box)
    far["d"] = ["150", str(F(150)+radius)]
    cases["far_distance_box"] = with_data(far)
    with precision(80):
        bounds, _ = prior_limits()
        broad = {n: [str(bounds[n][0].lo), str(bounds[n][1].hi)] for n in NAMES}
    cases["full_prior_outer_box"] = with_data(broad)

    critical = deepcopy(small_box)
    # Deterministic within-prior optical domain rejection, same observed record.
    for name, value in {"r1": "8/25", "p1": "3/20", "n1": "9/5",
                        "tx": "23/50", "ty": "23/50"}.items():
        critical[name] = [value, value]
    cases["nontransmitted_box"] = with_data(critical)

    expected = {"small_box": "retained", "far_distance_box": "excluded",
                "full_prior_outer_box": "unresolved", "nontransmitted_box": "excluded"}
    results = {}
    small_certificate = None
    for name, request in cases.items():
        certificate = certify(request)
        dump_json(OUT/(name+"_input.json"), request)
        dump_json(OUT/(name+"_certificate.json"), certificate)
        assert certificate["status"] == expected[name], (name, certificate["status"], certificate["witness"])
        results[name] = {"status": certificate["status"], "witness": certificate["witness"],
                         "samples_processed": len(certificate["samples"]),
                         "request_sha256": digest(request),
                         "minimum_final_prism_ratio_lower":
                             certificate["final_prism_margin"]["minimum_ratio_lower"]}
        if name == "small_box":
            small_certificate = certificate
    assert verify(cases["small_box"], small_certificate)
    altered = deepcopy(small_certificate)
    altered["samples"][0]["output"][0][0] = "123456789"
    assert not verify(cases["small_box"], altered)
    invalid_float = deepcopy(cases["small_box"])
    invalid_float["box"]["tx"][0] = 0.05
    try:
        certify(invalid_float)
    except ValueError:
        float_rejected = True
    else:
        raise AssertionError("Binary float input was accepted")
    invalid_clock = deepcopy(cases["small_box"])
    invalid_clock["timestamps"][-1] = "9.950000000000001"
    try:
        certify(invalid_clock)
    except ValueError:
        clock_rejected = True
    else:
        raise AssertionError("Nonexact sample clock was accepted")
    canonical = deepcopy(cases["small_box"])
    canonical["provenance"] = "canonical_independent_axis"
    try:
        certify(canonical)
    except ValueError:
        canonical_rejected = True
    else:
        raise AssertionError("Canonical provenance was accepted")

    min_guards = {}
    for s in small_certificate["samples"]:
        for stage in s["stages"]:
            for name, interval in stage["guards"].items():
                value = F(interval[0])
                min_guards[name] = min(min_guards.get(name, value), value)
    summary = {"status": "passed", "dataset_count": 1,
               "dataset_kind": "synthetic_fullvector_proof_spotcheck",
               "samples": 200, "scalar_observations": 400, "exact_clock": "k/20",
               "all_18_box_widths_positive": all(F(b)>F(a) for a,b in small_box.values()),
               "supplied_chart_halfwidth": str(radius), "bits": 80,
               "point_max_output_width": str(2*point_error),
               "derived_box_validation_allowance": str(allowance),
               "minimum_strict_guard_lower_bounds": {k: str(v) for k,v in min_guards.items()},
               "cases": results, "certificate_replay": "passed", "tamper_rejected": True,
               "float_rejected": float_rejected, "nonexact_clock_rejected": clock_rejected,
               "canonical_provenance_rejected": canonical_rejected,
               "wall_seconds_display_only": time.perf_counter()-started,
               "scope": "Conditional box validation only; no global inverse, fitted estimate, hardware result, or benchmark"}
    dump_json(OUT/"results.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run()
