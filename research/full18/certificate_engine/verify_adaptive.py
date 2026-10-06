"""Independent arithmetic and cover audit for the bounded adaptive engine.

Reuses the one saved synthetic full-vector record; never generates observations.
Forward replay uses the already independently reviewed evaluator. Row-bound,
dual-support and tree-partition checks below do not use their implementation's
corresponding construction or verification functions.
"""
from __future__ import annotations
from copy import deepcopy
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import hashlib
import json

import engine
import joint_lp

HERE = Path(__file__).resolve().parent
CHECKS = HERE / "adaptive_checks"


def corners(bounds):
    return list(product(*[(F(lo), F(hi)) for lo, hi in bounds]))


def source_rows(request, forward):
    """Independently identify physical signed rows in their prescribed order."""
    for sample in forward["samples"]:
        k = sample["sample"]
        for stage in sample["stages"]:
            for kind in ("internal", "external"):
                name = kind + "_affine"
                if name in stage:
                    yield (f"k{k}:p{stage['prism']}:{kind}",
                           stage[name], -1, F(0), 0, True)
        if request.get("observations") is not None and "output_affine" in sample:
            for axis, values in enumerate(sample["output_affine"]):
                y = F(request["observations"][k][axis])
                for sign in (1, -1):
                    yield (f"k{k}:axis{axis}:sign{sign}",
                           values, sign, -sign*y, 1, False)


def audit_rows(request, forward, rows):
    bounds = [tuple(map(F, request["box"][n])) for n in engine.GEOMETRY]
    vertices = corners(bounds)
    raw = list(source_rows(request, forward))
    assert len(raw) == len(rows)
    count = 0
    for row, (name, intervals, sign, shift, delta, strict) in zip(rows, raw):
        assert row.label == name and row.delta == delta and row.strict is strict
        assert len(row.a) == 4 and row.coefficient_error >= 0
        signed = []
        for pair in intervals:
            lo, hi = map(F, pair)
            assert lo <= hi
            signed.append((lo, hi) if sign == 1 else (-hi, -lo))
        signed[4] = (signed[4][0]+shift, signed[4][1]+shift)
        mids = [(lo+hi)/2 for lo, hi in signed]
        radii = [(hi-lo)/2 for lo, hi in signed]
        expected_error = radii[4] + sum(
            (radius*max(abs(lo), abs(hi))
             for radius, (lo, hi) in zip(radii[:4], bounds)), F(0))
        assert tuple(mids[:4]) == row.a
        assert row.b == mids[4]-expected_error
        assert row.coefficient_error == expected_error
        # Exact minimization over coefficient intervals at all sixteen Q vertices.
        # The true interval lower support minus this affine lower row is concave
        # in q, so nonnegativity at every vertex proves it on the entire box.
        for q in vertices:
            true_lower = signed[4][0] + sum(
                (min(lo*v, hi*v) for (lo, hi), v in zip(signed[:4], q)), F(0))
            proposed = row.b + sum((a*v for a, v in zip(row.a, q)), F(0))
            assert proposed <= true_lower
            count += 1
    return count


def audit_dual(rows, bounds, epsilon, evidence):
    """Use direct vertex minimization, independently of the support formula."""
    if evidence is None:
        return {"checked": False}
    weights = evidence["weights"]
    assert 1 <= len(weights) <= 5
    ids = [item["row"] for item in weights]
    assert len(ids) == len(set(ids))
    assert all(type(i) is int and 0 <= i < len(rows) for i in ids)
    w = [F(item["weight"]) for item in weights]
    assert all(v > 0 for v in w) and sum(w, F(0)) == 1
    for item, index in zip(weights, ids):
        assert item["id"] == rows[index].label
    selected = [rows[i] for i in ids]
    assert evidence["support_rows"] == [row.json() for row in selected]
    normal = [sum((weight*row.a[j] for weight, row in zip(w, selected)), F(0))
              for j in range(4)]
    support = min(sum((a*qj for a, qj in zip(normal, q)), F(0))
                  for q in corners(bounds))
    a0 = support + sum((weight*row.b for weight, row in zip(w, selected)), F(0))
    slope = sum((weight*row.delta for weight, row in zip(w, selected)), F(0))
    lower = a0 - F(epsilon)*slope
    strict = any(row.strict for row in selected)
    assert 0 <= slope <= 1
    assert list(map(F, evidence["weighted_normal"])) == normal
    assert F(evidence["geometry_support_correction"]) == support
    assert F(evidence["a0"]) == a0
    assert F(evidence["epsilon_coefficient"]) == slope
    assert F(evidence["epsilon"]) == F(epsilon)
    assert F(evidence["lower_bound"]) == lower
    assert evidence["support_size"] == len(w)
    assert evidence["strict_row_has_positive_weight"] is strict
    assert evidence["excludes_weak_affine"] is (lower > 0)
    assert evidence["excludes_strict_physical"] is (lower > 0 or (lower == 0 and strict))
    validity = evidence["symbolic_validity"]
    assert F(validity["slope"]) == slope
    assert validity["threshold"] == (str(a0/slope) if slope else None)
    assert validity["endpoint_excluded_for_strict_systems"] is bool(slope and strict)
    assert validity["all_nonnegative_epsilon_excluded"] is bool(
        not slope and (a0 > 0 or (a0 == 0 and strict)))
    return {"checked": True, "support_size": len(w),
            "normal_residual_nonzero": any(normal),
            "lower_bound": str(lower),
            "strict_equality_exclusion": bool(lower == 0 and strict)}


def audit_lp(rows, bounds, epsilon, solved):
    result = audit_dual(rows, bounds, epsilon, solved.get("dual"))
    if "point" in solved:
        point = list(map(F, solved["point"]))
        assert len(point) == 5
        assert all(lo <= v <= hi for v, (lo, hi) in zip(point[:4], bounds))
        violations = [sum((a*v for a, v in zip(row.a, point[:4])), F(0))
                      + row.b-F(epsilon)*row.delta for row in rows]
        assert all(v <= point[4] for v in violations)
        if "objective" in solved:
            assert F(solved["objective"]) == point[4]
        if solved["optimal"]:
            assert solved["dual"] is not None
            assert F(solved["dual"]["lower_bound"]) == point[4]
            assert solved["relaxation_feasible"] is (point[4] <= 0)
    else:
        assert not solved["optimal"]
    return result


def canonical_box(box):
    assert set(box) == set(engine.NAMES)
    return {name: tuple(map(F, box[name])) for name in engine.NAMES}


def audit_cover(request, tree):
    """Derive exact covered boxes recursively; never trust stored partitions."""
    entries = tree["nodes"]
    assert isinstance(entries, list) and entries
    ids = [node["id"] for node in entries]
    assert len(ids) == len(set(ids))
    nodes = dict(zip(ids, entries))
    seen = set()
    counts = {"excluded": 0, "retained": 0, "unresolved": 0, "split": 0}
    row_checks = 0
    dual_checks = 0
    forward_checks = 0
    geometry = {name: canonical_box(request["box"])[name] for name in engine.GEOMETRY}

    def visit(identity, expected, depth):
        nonlocal row_checks, dual_checks, forward_checks
        assert identity in nodes and identity not in seen
        seen.add(identity)
        node = nodes[identity]
        assert node["depth"] == depth
        actual = canonical_box(node["box"])
        assert actual == expected
        assert all(actual[name] == value for name, value in geometry.items())
        outcome = node["outcome"]
        assert outcome in counts
        counts[outcome] += 1
        local = deepcopy(request)
        local["box"] = node["box"]
        forward = node.get("forward")
        if forward is not None:
            assert engine.verify(local, forward)
            forward_checks += 1
        joint = node.get("joint")
        if joint is not None:
            assert forward is not None
            rows, bounds, coverage = joint_lp.extract_rows(local, forward)
            row_checks += audit_rows(local, forward, rows)
            check = audit_lp(rows, bounds, F(local.get("epsilon", 0)), joint["lp"])
            dual_checks += int(check["checked"])
            assert joint["coverage"] == coverage
            assert joint["rows_sha256"] == engine.digest([row.json() for row in rows])
            assert joint["row_count"] == len(rows)
            if joint["status"] == "excluded":
                assert joint["certificate"] == joint["lp"]["dual"]
                assert joint["certificate"]["excludes_strict_physical"]
            else:
                assert joint["status"] == "unresolved" and joint["certificate"] is None
        if outcome == "excluded":
            assert ((forward is not None and forward["status"] == "excluded")
                    or (joint is not None and joint["status"] == "excluded"))
        elif outcome == "retained":
            assert forward is not None and forward["status"] == "retained"
        elif outcome == "split":
            split = node["split"]
            coordinate = split["coordinate"]
            assert coordinate in engine.OPTICAL
            lo, hi = actual[coordinate]
            mid = F(split["midpoint"])
            assert lo < mid < hi and mid == (lo+hi)/2
            children = split["children"]
            assert isinstance(children, list) and len(children) == 2
            assert children[0] != children[1]
            left, right = deepcopy(actual), deepcopy(actual)
            left[coordinate], right[coordinate] = (lo, mid), (mid, hi)
            visit(children[0], left, depth+1)
            visit(children[1], right, depth+1)
        else:
            # An unresolved leaf preserves the complete expected box.
            assert "split" not in node
        if outcome != "split":
            assert "split" not in node

    visit("r", canonical_box(request["box"]), 0)
    assert seen == set(nodes), "Unreachable or missing nodes invalidate coverage evidence"
    assert counts["excluded"]+counts["retained"]+counts["unresolved"] == counts["split"]+1
    return {"nodes": len(nodes), "counts": counts, "forward_replays": forward_checks,
            "exact_row_vertex_inequalities": row_checks, "dual_vertex_recomputations": dual_checks}



def unit_checks():
    """Small exact LP identities, not optical datasets or simulated records."""
    zero = (F(0),)*3
    bounds = tuple((F(0), F(1)) for _ in range(4))
    conflict = [
        joint_lp.Row("q<=1/4", (F(1),)+zero, F(-1,4), 0, False),
        joint_lp.Row("q>=3/4", (F(-1),)+zero, F(3,4), 0, False)]
    # Neither row excludes Q by itself. Together they require 3/4 <= q <= 1/4.
    one = [joint_lp.dual_evidence(conflict, bounds, F(0), {i:F(1)})
           for i in range(2)]
    assert all(not cert["excludes_strict_physical"] for cert in one)
    shared = joint_lp.dual_evidence(conflict, bounds, F(0), {0:F(1,2),1:F(1,2)})
    assert F(shared["lower_bound"]) == F(1,4)
    audit_dual(conflict, bounds, F(0), shared)
    solved = joint_lp.solve_rows(conflict, bounds, pivot_limit=16)
    audit_lp(conflict, bounds, F(0), solved)
    assert solved["dual"]["excludes_strict_physical"]

    strict_rows = [
        joint_lp.Row("q<0", (F(1),)+zero, F(0), 0, True),
        joint_lp.Row("q>=0", (F(-1),)+zero, F(0), 0, False)]
    centered = tuple((F(-1),F(1)) for _ in range(4))
    strict = joint_lp.dual_evidence(strict_rows, centered, F(0), {0:F(1,2),1:F(1,2)})
    assert strict["excludes_strict_physical"] and not strict["excludes_weak_affine"]
    audit_dual(strict_rows, centered, F(0), strict)
    weak_rows = [joint_lp.Row(r.label,r.a,r.b,r.delta,False) for r in strict_rows]
    weak = joint_lp.dual_evidence(weak_rows, centered, F(0), {0:F(1,2),1:F(1,2)})
    assert not weak["excludes_strict_physical"]
    audit_dual(weak_rows, centered, F(0), weak)

    # A positive constant cannot justify separation if residual normal can offset it.
    residual_rows = [joint_lp.Row("nonstationary", (F(1),)+zero, F(1,2), 0, False)]
    residual = joint_lp.dual_evidence(residual_rows, centered, F(0), {0:F(1)})
    assert F(residual["geometry_support_correction"]) == -1
    assert F(residual["lower_bound"]) == F(-1,2)
    assert not residual["excludes_strict_physical"]
    audit_dual(residual_rows, centered, F(0), residual)

    damaged = deepcopy(residual)
    damaged["geometry_support_correction"] = "0"
    damaged["lower_bound"] = "1/2"
    assert not joint_lp.verify_dual(residual_rows, centered, damaged)
    damaged = deepcopy(shared)
    damaged["weights"][0]["weight"] = "1/3"
    assert not joint_lp.verify_dual(conflict, bounds, damaged)
    damaged = deepcopy(strict)
    damaged["strict_row_has_positive_weight"] = False
    assert not joint_lp.verify_dual(strict_rows, centered, damaged)
    limited = joint_lp.solve_rows(conflict, bounds, pivot_limit=0)
    assert limited["pivots"] == 0 and not limited["optimal"]
    assert limited["relaxation_feasible"] is None
    audit_lp(conflict, bounds, F(0), limited)
    return {"shared_conflict_lower_bound":"1/4",
            "individually_nonexcluding_rows":2,
            "strict_equality_excludes_only_strict_systems":True,
            "weak_equality_not_excluded":True,
            "nonzero_residual_support_penalty":"-1",
            "residual_corrected_lower_bound":"-1/2",
            "zero_pivot_budget_retains_unresolved_optimizer":True,
            "dual_tamper_rejections":3}


def expect_rejection(function, *args):
    assert function(*args) is False, "Altered certificate was accepted"


def main():
    import adaptive
    frozen = adaptive.source_hashes()
    previous = engine.load_json(HERE/"independent_review.json")["source_sha256"]
    assert all(frozen[name] == previous[name] for name in ("engine.py","interval.py"))
    fixture_path = HERE/"spotcheck"/"synthetic_fullvector_record.json"
    fixture = engine.load_json(fixture_path)
    fixture_hash = hashlib.sha256(fixture_path.read_bytes()).hexdigest()
    assert fixture["provenance"] == "synthetic_fullvector_proof_spotcheck"
    observations = fixture["observations"]
    assert len(observations) == 200
    requests, objects = {}, []
    for folder in (HERE/"spotcheck", CHECKS):
        if folder.exists():
            for path in sorted(folder.glob("*.json")):
                if not (path.name.endswith("_input.json") or path.name.endswith("_certificate.json")):
                    continue
                value = engine.load_json(path)
                objects.append((path,value))
                if isinstance(value,dict) and "box" in value and "model" in value:
                    requests[engine.digest(value)] = value
    tree_results = []
    tamper_results = {}
    for path, tree in objects:
        if not isinstance(tree,dict) or tree.get("engine") != adaptive.VERSION:
            continue
        request = requests[tree["input_sha256"]]
        assert request["observations"] == observations
        assert request["sampling"] == {"count":200,"step":"1/20","start":"0"}
        assert adaptive.verify(request,tree)
        details = audit_cover(request,tree)
        assert details["nodes"] == tree["summary"]["nodes"]
        assert details["counts"] == tree["summary"]["outcomes"]
        tree_results.append({"file":str(path.relative_to(HERE)),**details})
        if details["counts"]["split"] and not tamper_results:
            leaves = [n for n in tree["nodes"] if n["outcome"] != "split"]
            split = next(n for n in tree["nodes"] if n["outcome"] == "split")
            bad = deepcopy(tree)
            bad["nodes"] = [n for n in bad["nodes"] if n["id"] != leaves[-1]["id"]]
            expect_rejection(adaptive.verify,request,bad)
            tamper_results["deleted_leaf"] = True
            bad = deepcopy(tree)
            altered = next(n for n in bad["nodes"] if n["id"] == split["id"])
            altered["split"]["midpoint"] = str(F(altered["split"]["midpoint"])+F(1,10**8))
            expect_rejection(adaptive.verify,request,bad)
            tamper_results["altered_split_midpoint"] = True
            bad = deepcopy(tree)
            altered = next(n for n in bad["nodes"] if n["id"] == leaves[-1]["id"])
            changed = engine.OPTICAL[-1]
            altered["box"][changed][0] = str(F(altered["box"][changed][0])+F(1,10**8))
            expect_rejection(adaptive.verify,request,bad)
            tamper_results["altered_leaf_endpoint"] = True
            bad = deepcopy(tree)
            altered = next(n for n in bad["nodes"] if n["id"] == leaves[-1]["id"])
            altered["outcome"],altered["reason"] = "retained","whole_box_certificate"
            expect_rejection(adaptive.verify,request,bad)
            tamper_results["unsupported_retention"] = True
            changed_input = deepcopy(request)
            changed_input["observations"][0][0] = str(F(changed_input["observations"][0][0])+1)
            expect_rejection(adaptive.verify,changed_input,tree)
            tamper_results["changed_observation_input"] = True
            bad = deepcopy(tree)
            bad["summary"]["global_recovery_claim"] = True
            expect_rejection(adaptive.verify,request,bad)
            tamper_results["unsupported_recovery_claim"] = True
    assert tree_results, "No saved adaptive covers were found"
    assert tamper_results, "No subdivided saved cover was available for coverage tamper tests"

    joint_results = []
    sparse_replays = 0
    sparse_tamper_rejections = 0
    for path, value in objects:
        if not isinstance(value,dict) or value.get("engine") != joint_lp.VERSION:
            continue
        request = requests[value["input_sha256"]]
        assert request["observations"] == observations
        forward = engine.certify(request)
        assert joint_lp.verify(request,value,forward=forward,replay_forward=False)
        rows,bounds,coverage = joint_lp.extract_rows(request,forward)
        checks = audit_rows(request,forward,rows)
        dual = audit_lp(rows,bounds,F(request.get("epsilon",0)),value["lp"])
        if value["status"] == "excluded":
            assert joint_lp.verify_sparse(request,value,forward=forward,replay_forward=False)
            sparse_replays += 1
            bad = deepcopy(value)
            bad["certificate"]["lower_bound"] = str(F(bad["certificate"]["lower_bound"])+1)
            expect_rejection(joint_lp.verify_sparse,request,bad,forward)
            sparse_tamper_rejections += 1
        joint_results.append({"file":str(path.relative_to(HERE)),
                              "coverage":coverage,"row_vertex_inequalities":checks,
                              "dual":dual,"lp_termination":value["lp"]["termination"],
                              "lp_pivots":value["lp"]["pivots"],
                              "lp_objective":value["lp"].get("objective"),
                              "status":value["status"]})
    assert joint_results, "No standalone shared-geometry LP certificate was found"
    assert any(item["coverage"]["complete_affine_rows"] for item in joint_results)
    units = unit_checks()
    assert frozen == adaptive.source_hashes(), "Reviewed proof sources changed"
    assert fixture_hash == hashlib.sha256(fixture_path.read_bytes()).hexdigest()
    result = {
        "status":"passed",
        "scope":"Independent finite code and certificate audit, not formal verification or global inverse completion",
        "dataset_count":1,"dataset_provenance":fixture["provenance"],
        "fixture_sha256":fixture_hash,"source_sha256":frozen,
        "independent_checker_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cover_checks":tree_results,
        "standalone_joint_checks":joint_results,
        "optimizer_independent_sparse_replays":sparse_replays,
        "sparse_bound_tamper_rejections":sparse_tamper_rejections,
        "coverage_tamper_rejections":tamper_results,
        "rational_lp_units":units,
        "original_engine_and_interval_sources_unchanged":True,
        "new_observation_generation":False,
        "original_project_execution":False,
        "remaining":"Unresolved boxes preserved; no exact fallback, full-prior solved cover, or real-record recovery"
    }
    engine.dump_json(HERE/"adaptive_independent_review.json",result)
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__ == "__main__":
    main()
