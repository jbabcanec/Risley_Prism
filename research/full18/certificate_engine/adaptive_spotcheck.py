"""Bounded mechanism checks, reusing the sole saved full-vector record verbatim.

This does not generate observations, fit parameters, or run a search campaign.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
import adaptive
import engine
import joint_lp

HERE = Path(__file__).resolve().parent
OUT = HERE / "adaptive_checks"
FIXTURE_HASH = "1c83fb2ce416e7e52971e0feb0c7e2f70d91e5d92088d516a3b3ce8aad99de9c"
FROZEN = {
    "engine.py": "7cbce954039aa9cb93e00cedb6ae2b4d4a760de8eb46edc8f41a76c5637eba6b",
    "interval.py": "3ddc0eaa3430028211d41962fcb2ce6d22c5c70dc24f89c3fae83245b8c90cbe",
}


def main():
    fixture_path = HERE / "spotcheck" / "synthetic_fullvector_record.json"
    assert hashlib.sha256(fixture_path.read_bytes()).hexdigest() == FIXTURE_HASH
    fixture = engine.load_json(fixture_path)
    for name, expected in FROZEN.items():
        assert hashlib.sha256((HERE/name).read_bytes()).hexdigest() == expected
    OUT.mkdir(exist_ok=True)
    sources = adaptive.source_hashes()
    local = engine.load_json(HERE/"spotcheck"/"small_box_input.json")
    broad = engine.load_json(HERE/"spotcheck"/"full_prior_outer_box_input.json")
    far = engine.load_json(HERE/"spotcheck"/"far_distance_box_input.json")
    expanded = deepcopy(local)
    for name in engine.GEOMETRY:
        expanded["box"][name] = deepcopy(broad["box"][name])
    for request in (local, broad, far, expanded):
        assert request["observations"] == fixture["observations"]
        assert request["sampling"] == fixture["sampling"]
        assert request["timestamps"] == fixture["timestamps"]
    results = {"dataset_count": 1, "fixture_sha256": FIXTURE_HASH,
               "new_observations_generated": False, "original_project_execution": False,
               "source_sha256": sources, "adaptive": [], "joint": []}
    cases = [
        ("small_box", local, dict(max_nodes=1,max_depth=0,lp_pivot_limit=16,max_total_lp_pivots=16)),
        ("far_distance", far, dict(max_nodes=1,max_depth=0,lp_pivot_limit=16,max_total_lp_pivots=16)),
        ("full_prior", broad, dict(max_nodes=3,max_depth=2,lp_pivot_limit=16,max_total_lp_pivots=32)),
        ("zero_budget", broad, dict(max_nodes=0,max_depth=2,lp_pivot_limit=16,max_total_lp_pivots=32)),
        ("zero_pivots", expanded, dict(max_nodes=1,max_depth=0,lp_pivot_limit=0,max_total_lp_pivots=0)),
    ]
    for name, request, budget in cases:
        started = time.monotonic()
        certificate = adaptive.run(request, **budget)
        assert adaptive.verify(request, certificate)
        engine.dump_json(OUT/(name+"_input.json"), request)
        engine.dump_json(OUT/(name+"_certificate.json"), certificate)
        result = {"name": name, "limits": budget, **certificate["summary"],
                  "replay_passed": True, "informational_seconds": time.monotonic()-started}
        results["adaptive"].append(result)
        print(json.dumps(result), flush=True)
    for name, request in (("expanded_geometry_joint", expanded), ("far_distance_joint", far)):
        started = time.monotonic()
        forward = engine.certify(request)
        certificate = joint_lp.analyze(request, forward, pivot_limit=32)
        assert joint_lp.verify(request, certificate, forward=forward, replay_forward=False)
        sparse_passed = None
        if certificate["status"] == "excluded":
            assert joint_lp.verify_sparse(request, certificate, forward=forward, replay_forward=False)
            damaged = deepcopy(certificate)
            damaged["certificate"]["lower_bound"] = "0"
            assert not joint_lp.verify_sparse(request, damaged, forward=forward, replay_forward=False)
            sparse_passed = True
        engine.dump_json(OUT/(name+"_input.json"), request)
        engine.dump_json(OUT/(name+"_certificate.json"), certificate)
        result = {"name": name, "status": certificate["status"],
                  "coverage": certificate["coverage"], "diagnostics": certificate["diagnostics"],
                  "objective": certificate["lp"].get("objective"),
                  "sparse_support_size": (certificate["certificate"] or {}).get("support_size"),
                  "replay_passed": True, "sparse_replay_and_tamper_passed": sparse_passed,
                  "informational_seconds": time.monotonic()-started}
        results["joint"].append(result)
        print(json.dumps(result), flush=True)
    assert sources == adaptive.source_hashes()
    assert hashlib.sha256(fixture_path.read_bytes()).hexdigest() == FIXTURE_HASH
    engine.dump_json(OUT/"results.json", results)


if __name__ == "__main__":
    main()
