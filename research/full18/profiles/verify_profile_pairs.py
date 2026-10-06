"""Independent full-record checks of observation-conditioned profile pairs.

No truth data, optimizer, repository model, or repository interval implementation
is imported. The independent scalar oracle uses directed high-precision
intervals and exact binary64 parameters, timestamps, and stored observations.
"""
import hashlib
import json
import math
import sys
import time
from fractions import Fraction
from pathlib import Path

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"stable_model"))
from oracle import reference,exact,abs_upper,bounds

NAMES=["N1","N2","N3","ax1","ax2","ax3","ay1","ay2","ay3",
       "ng1","ng2","ng3","d_W","gap","bm_ax","bm_ay","bm_px","bm_py"]
LOWER=[-3.5]*3+[-18.]*6+[1.3]*3+[50.,2.,-25.,-25.,-5.,-5.]
UPPER=[3.5]*3+[18.]*6+[1.8]*3+[200.,15.,25.,25.,5.,5.]


def load(path):
    raw=path.read_bytes()
    return json.loads(raw),hashlib.sha256(raw).hexdigest()


def check_endpoint(theta,observations,timestamps,eta,dps):
    if len(theta)!=18 or not all(math.isfinite(x) for x in theta):
        raise ValueError("Expected a finite 18-parameter endpoint")
    inside=all(lo<=x<=hi for x,lo,hi in zip(theta,LOWER,UPPER))
    if not inside:
        return dict(theta=theta,inside_full_native_bounds=False,compatible=False)
    values,margins=reference(theta,timestamps,dps)
    errors=[[abs_upper(predicted-exact(observed)) for predicted,observed in zip(pair,y)]
            for pair,y in zip(values,observations)]
    maximum=max(max(pair) for pair in errors)
    worst=max(((e,k,axis) for k,pair in enumerate(errors) for axis,e in enumerate(pair)))
    # The allowance printed in JSON (1e-4 or 1e-8) is interpreted as that exact
    # decimal value, rather than taking advantage of any upward binary rounding.
    allowance=Fraction(str(eta))
    compatible=Fraction.from_float(maximum)<=allowance
    return dict(theta=theta,inside_full_native_bounds=True,
        samples_checked=len(timestamps),scalar_coordinates_checked=2*len(timestamps),
        compatible=compatible,observation_error_upper=maximum,
        per_axis_error_upper=[max(row[axis] for row in errors) for axis in range(2)],
        every_coordinate_error_upper=errors,
        worst_sample=worst[1],worst_axis=worst[2],strict_physical_margin_lower=margins,
        all_physical_margins_positive=all(m>0 for m in margins.values()))


def check_pair(identifier,endpoints,observations,timestamps,eta,dps=70):
    assert len(timestamps)==200 and len(observations)==200
    assert all(len(pair)==2 and all(math.isfinite(x) for x in pair) for pair in observations)
    checked=[check_endpoint(v,observations,timestamps,eta,dps) for v in endpoints]
    both=all(r["compatible"] for r in checked)
    lower=[]
    upper=[]
    for a,b in zip(*endpoints):
        lo,hi=bounds(abs(exact(a)-exact(b))/2)
        lower.append(max(0.,lo))
        upper.append(max(0.,hi))
    worst=max(range(18),key=lambda j:lower[j])
    obstruction=both and Fraction.from_float(lower[worst])>Fraction("0.001")
    return dict(id=identifier,eta=eta,eta_interpretation="exact decimal JSON allowance",
        endpoint_a=checked[0],endpoint_b=checked[1],
        pair_compatible_with_actual_stored_record=both,
        minimax_coordinate_error_lower_bound=lower,
        minimax_coordinate_error_upper_bound=upper,
        largest_forced_coordinate=NAMES[worst],
        largest_forced_error_lower_bound=lower[worst],
        native_accuracy_001_impossible_for_all_compatible_systems=obstruction,
        mathematical_implication="For each coordinate separately, any one estimate from this same observation record has error at least that coordinate's reported half-separation for at least one of these two admissible systems.",
        observed_record_sha256=hashlib.sha256(json.dumps(observations,separators=(',',':')).encode()).hexdigest(),
        complete_compatible_set_enumerated=False)


def main():
    start=time.perf_counter()
    metadata,clock_hash=load(ROOT/"observations.json")
    timestamps=metadata["timestamps"]
    # Read observation-only timestamps and returned profiles; never open cases
    # files containing generating true hardware.
    noisy_path=ROOT/"profiles/random00_eta1e4.json"
    noisy,noisy_hash=load(noisy_path)
    witness=noisy.get("actual_record_pair_witness",noisy.get("compatible_pair"))
    if witness is None:
        raise ValueError("No actual-record pair witness in noisy profile report")
    endpoints=[noisy["candidate"],witness["endpoint_b"]["theta"]]
    if "endpoint_a" in witness:
        assert endpoints[0]==witness["endpoint_a"]["theta"]
    records=[check_pair("random00_actual_noisy_record_eta1e4",endpoints,
        noisy["actual_observations"],timestamps,noisy["eta"])]
    print(json.dumps(dict(id=records[-1]["id"],compatible=records[-1]["pair_compatible_with_actual_stored_record"],
        endpoint_errors=[records[-1][key]["observation_error_upper"] for key in ("endpoint_a","endpoint_b")],
        dW_minimax_lower=records[-1]["minimax_coordinate_error_lower_bound"][12])),flush=True)
    hashes={str(noisy_path.relative_to(ROOT)):noisy_hash,"observations.json":clock_hash}
    gain_path=ROOT/"gain_continuation/compatible_profiles.json"
    adverse_path=ROOT/"profiles/adverse_observations_only.json"
    if gain_path.exists() and adverse_path.exists():
        gain,gain_hash=load(gain_path)
        adverse,adverse_hash=load(adverse_path)
        observation=next(c["observed"] for c in adverse["cases"] if c["id"]=="eta_1e-8_separated_speeds")
        negative=next(r["theta"] for r in gain["profiles"] if r["coordinate"]=="d_W" and r["shift"]<0)
        positive=next(r["theta"] for r in gain["profiles"] if r["coordinate"]=="d_W" and r["shift"]>0)
        records.append(check_pair("gain_continuation_actual_record_dW_minus_plus",[negative,positive],
            observation,timestamps,gain["eta"]))
        hashes[str(gain_path.relative_to(ROOT))]=gain_hash
        hashes[str(adverse_path.relative_to(ROOT))]=adverse_hash
        print(json.dumps(dict(id=records[-1]["id"],compatible=records[-1]["pair_compatible_with_actual_stored_record"],
            endpoint_errors=[records[-1][key]["observation_error_upper"] for key in ("endpoint_a","endpoint_b")],
            dW_minimax_lower=records[-1]["minimax_coordinate_error_lower_bound"][12])),flush=True)
    output=dict(status="INDEPENDENT_DIRECTED_INTERVAL_ACTUAL_RECORD_ENDPOINT_CHECKS",
        model="Original strict independent-axis full18, physical order preserved, no parameter restrictions added",
        arithmetic="mpmath.iv directed intervals at 70 decimal digits; not a formal verification of Python or mpmath",
        binary64_inputs="Parameters, original stored timestamps, and actual observations all enter through exact integer ratios",
        no_truth_read=True,no_repository_optics_imported=True,names=NAMES,
        native_bounds=dict(lower=LOWER,upper=UPPER),timestamps=timestamps,
        input_sha256=hashes,
        verification_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        oracle_source_sha256=hashlib.sha256((ROOT/"stable_model/oracle.py").read_bytes()).hexdigest(),
        records=records,elapsed_seconds=time.perf_counter()-start)
    destination=ROOT/"profiles/independent_pair_verification.json"
    destination.write_text(json.dumps(output,indent=2)+"\n",encoding="utf-8")
    assert all(r["pair_compatible_with_actual_stored_record"] for r in records)
    assert all(r["native_accuracy_001_impossible_for_all_compatible_systems"] for r in records)
    print(str(destination),flush=True)


if __name__=="__main__":
    main()
