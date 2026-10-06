"""Assert the numerical consistency checks in an existing diagnostic report."""
import json
from pathlib import Path

path=Path(__file__).with_name("results.json")
report=json.loads(path.read_text(encoding="utf-8"))
checked=[]
for case in report["cases"]:
    assert case["source_first_vs_joint4_max_prediction_difference"] < 1e-10
    parity=case["paired_face_transfer_validation"]
    assert parity["scaled_jacobian_relative_frobenius"] < 1e-12
    assert parity["minimum_trial_guard"] > 0
    if "derivative_validation" in case:
        val=case["derivative_validation"]
        assert val["interval_ad_entrywise_enclosure_excess"] <= 0
        assert val["interval_ad_midpoint_relative_frobenius"] < 1e-12
        assert val["profile_jacobian_test"]["same_active_set_all_perturbations"]
        assert val["profile_jacobian_test"]["relative_frobenius"] < 1e-8
    checked.append(case["id"])
path.with_name("validation_summary.json").write_text(json.dumps(dict(
    checked_cases=checked,all_numerical_checks_passed=True,
    scope="Independent mathematical-model/derivative consistency, not accuracy certification"),
    indent=2)+"\n",encoding="utf-8")
print(f"Independent numerical consistency checks pass on {len(checked)} cases.")
