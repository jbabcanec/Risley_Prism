"""Independent bounded review of the single saved synthetic fixture.

Does not generate observations, fit parameters, run a sweep, or import original
Dropbox project code. It replays evidence and checks the direct-vector reference.
"""
from copy import deepcopy
from fractions import Fraction as F
from pathlib import Path
import hashlib
import json
from interval import I, precision
from independent_reference import trace_point, independent_kernel_checks
from engine import load_json, verify, digest, source_hashes

HERE=Path(__file__).resolve().parent
DATA=HERE/"spotcheck"


def read(name):
    return load_json(DATA/name)


def exact_affine_range(coefficients, box, names=("bx","by","g","d")):
    """Independent interval support evaluation with unrounded rational arithmetic."""
    lower,upper=map(F,coefficients[4])
    for coefficients_i,name in zip(coefficients[:4],names):
        a,b=map(F,coefficients_i)
        c,d=map(F,box[name])
        corners=[a*c,a*d,b*c,b*d]
        lower+=min(corners)
        upper+=max(corners)
    return lower,upper


def check_contains(stored, lower, upper):
    a,b=map(F,stored)
    assert a<=lower<=upper<=b,(str(a-lower),str(b-upper))


def main():
    current_hashes=source_hashes()
    kernel=independent_kernel_checks()
    fixture=read("synthetic_fullvector_record.json")
    assert fixture["provenance"]=="synthetic_fullvector_proof_spotcheck"
    assert len(fixture["observations"])==200
    assert list(map(F,fixture["timestamps"]))==[F(k,20) for k in range(200)]
    point_input=read("point_forward_input.json")
    point=read("point_forward_certificate.json")
    assert point["source_sha256"]==current_hashes
    assert point["input_sha256"]==digest(point_input)
    assert point["status"]=="retained" and len(point["samples"])==200
    assert all(pair==[fixture["point_chart"][name]]*2 for name,pair in point_input["box"].items())
    reference=trace_point(fixture["point_chart"],bits=224)
    error=F(fixture["point_rounding_enclosure_error"])
    comparisons=0
    max_reference_width=F(0)
    for k,pair in enumerate(reference["screen"]):
        for axis,interval in enumerate(pair):
            check_contains(point["samples"][k]["output"][axis],interval.lo,interval.hi)
            y=F(fixture["observations"][k][axis])
            assert y-error<=interval.lo<=interval.hi<=y+error
            max_reference_width=max(max_reference_width,interval.width)
            comparisons+=1

    expected={"small_box":"retained","far_distance_box":"excluded",
              "full_prior_outer_box":"unresolved","nontransmitted_box":"excluded"}
    for name,status in expected.items():
        request=read(name+"_input.json")
        certificate=read(name+"_certificate.json")
        assert request["observations"]==fixture["observations"]
        assert request["epsilon"]==fixture["box_validation_allowance"]
        assert certificate["status"]==status
        assert certificate["source_sha256"]==current_hashes
        assert certificate["input_sha256"]==digest(request)
    small_input=read("small_box_input.json")
    small=read("small_box_certificate.json")
    epsilon=F(small_input["epsilon"])
    assert all(F(a)<F(b) for a,b in small_input["box"].values())
    affine_checked=0
    for k,sample in enumerate(small["samples"]):
        for stage in sample["stages"]:
            assert all(F(pair[0])>0 for pair in stage["guards"].values())
            for affine_key,guard_key in (("internal_affine","internal_traversal"),
                                         ("external_affine","external_traversal")):
                lower,upper=exact_affine_range(stage[affine_key],small_input["box"])
                check_contains(stage["guards"][guard_key],lower,upper)
                affine_checked+=1
        for axis,coefficient in enumerate(sample["output_affine"]):
            lower,upper=exact_affine_range(coefficient,small_input["box"])
            check_contains(sample["output"][axis],lower,upper)
            y=F(fixture["observations"][k][axis])
            a,b=map(F,sample["output"][axis])
            assert y-epsilon<=a<=b<=y+epsilon
            affine_checked+=1

    # Check data-only margin independently with the exact radical for tan(18deg).
    margin=small["final_prism_margin"]
    with precision(224):
        five=I(5).sqrt()
        ustar=(five-1)/(10+2*five).sqrt()
        for k,pair in enumerate(fixture["observations"]):
            ax,ay=[abs(F(y))+epsilon for y in pair]
            support=ustar*(I(ax).square()+I(ay).square()).sqrt()
            saved=margin["samples"][k]
            assert F(saved["screen_support_upper"])>=support.hi
            justified=max(F(0),(F(50)-support.hi)/53)
            assert F(saved["ratio_R_over_Z_lower"])<=justified
    assert F(margin["minimum_ratio_lower"])>0

    # This first-window exclusion replays quickly; a tampered witness must fail.
    far_input=read("far_distance_box_input.json")
    far=read("far_distance_box_certificate.json")
    assert verify(far_input,far)
    tampered=deepcopy(far)
    tampered["witness"]["sample"]=199
    assert not verify(far_input,tampered)
    changed_input=deepcopy(far_input)
    changed_input["provenance"]+="_tampered"
    assert not verify(changed_input,far)
    assert current_hashes==source_hashes(),"Proof source changed during review"

    source_names=("engine.py","interval.py","independent_reference.py","verify_spotcheck.py","spotcheck.py")
    report={"status":"passed","dataset_count":1,"dataset_kind":fixture["provenance"],
            "kernel":kernel,"reference_precision_bits":224,"engine_precision_bits":point["bits"],
            "independent_direct_vector_output_containments":comparisons,
            "exact_shared_geometry_affine_range_checks":affine_checked,
            "all_400_reference_coordinates_within_stated_point_error":True,
            "all_400_small_box_output_intervals_within_allowance":True,
            "all_1200_traversal_and_2400_optical_guard_lower_bounds_positive":True,
            "reference_max_output_width":str(max_reference_width),
            "reference_minimum_strict_lower_bounds":
                {k:str(v) for k,v in reference["minimum_strict_lower_bounds"].items()},
            "final_prism_margin_independent_radical_checks":200,
            "minimum_final_prism_ratio_lower":margin["minimum_ratio_lower"],
            "case_statuses":expected,"replay_passed":True,"tampered_evidence_rejected":True,
            "changed_input_rejected":True,
            "source_sha256":{name:hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                             for name in source_names},
            "fixture_sha256":hashlib.sha256((DATA/"synthetic_fullvector_record.json").read_bytes()).hexdigest(),
            "scope":"Bounded independent implementation review of one supplied synthetic fixture, not global inversion, hardware validation, a formal proof, or a benchmark"}
    (HERE/"independent_review.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2,sort_keys=True))


if __name__=="__main__":
    main()

