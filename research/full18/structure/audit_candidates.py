"""Observation-only diagnostics of frozen returned candidates; no solver tuning."""
import json
from pathlib import Path
import numpy as np
from diagnostics import (LO, HI, RG, NAMES, full_jacobian, analyze, validate_poc3,
                         vec2pat)
from risley_lattice.fmodel import forward_point

ROOT=Path(__file__).parents[1]


def inspect(path, observations):
    result=json.loads(path.read_text(encoding="utf-8-sig"))
    theta=np.array(result.get("theta",result.get("x18")),float)
    y=np.asarray(observations,float).ravel()
    f=vec2pat(theta).ravel()
    r=f-y
    j=full_jacobian(theta)
    js=j*RG
    step=np.linalg.lstsq(js,-r,rcond=1e-15)[0]
    gradient=js.T@r
    lower=(theta-LO)<1e-10*RG
    upper=(HI-theta)<1e-10*RG
    kkt_gradient=gradient.copy()
    kkt_gradient[lower]=np.minimum(gradient[lower],0)
    kkt_gradient[upper]=np.maximum(gradient[upper],0)
    room=np.full(18,np.inf)
    pos,neg=step>0,step<0
    room[pos]=((HI-theta)/RG)[pos]/step[pos]
    room[neg]=((LO-theta)/RG)[neg]/step[neg]
    cap=max(0.,min(1.,float(np.min(room))))
    s=np.linalg.svd(js,compute_uv=False)
    fm,fr,margins=forward_point(theta)
    row=dict(source=str(path.relative_to(ROOT)),canonical_max_residual=float(np.max(abs(r))),
        all18_within_full_native_bounds=bool(np.all(theta>=LO) and np.all(theta<=HI)),
        canonical_rms=float(np.sqrt(np.mean(r*r))),physical_margins=margins,
        mathematical_point_vs_canonical_max_bound=float(np.max(abs(fm-f)+fr)),
        full18_scaled_condition=float(s[0]/s[-1]),
        proposed_native_correction=(step*RG).tolist(),
        proposed_native_correction_max_abs=float(np.max(abs(step*RG))),
        proposed_scaled_step_max_abs=float(np.max(abs(step))),
        largest_correction_parameter=NAMES[int(np.argmax(abs(step*RG)))],
        residual_fraction_after_unconstrained_linear_step=float(np.linalg.norm(r+js@step)/np.linalg.norm(r)),
        bound_feasible_fraction_of_unconstrained_step=cap,
        active_bounds=[NAMES[i]+(" lower" if lower[i] else " upper") for i in range(18) if lower[i] or upper[i]],
        scaled_gradient_max_abs=float(np.max(abs(gradient))),
        scaled_projected_kkt_gradient_max_abs=float(np.max(abs(kkt_gradient))),
        observation_only_diagnostic=True,changes_to_frozen_solver=False)
    return row


def main():
    original={c["id"]:c["observed"] for c in json.loads((ROOT/"observations.json").read_text())["cases"]}
    holdout={c["id"]:c["observed"] for c in json.loads((ROOT/"holdout_observations.json").read_text())["cases"]}
    rows=[]
    for folder, cases, ids in [
        ("validation/remaining_development",original,["random_03"]),
        ("validation/holdout",holdout,["random_00","random_02"]),
        ("collision",original,["result"])]:
        for key in ids:
            obs=cases["exact_collision" if key=="result" else key]
            row=inspect(ROOT/folder/(key+".json"),obs)
            rows.append(row)
            print(json.dumps(row),flush=True)
    extra=ROOT/"collision/holdout/observations.json"
    if extra.exists():
        for case in json.loads(extra.read_text())["cases"]:
            candidate=extra.parent/"run"/(case["id"]+".json")
            if candidate.exists():
                row=inspect(candidate,case["observed"])
                rows.append(row)
                print(json.dumps(row),flush=True)
    recovered=json.loads((ROOT/"collision/result.json").read_text())["x18"]
    collision=analyze(recovered,original["exact_collision"])
    collision["paired_face_validation"]=validate_poc3(recovered)
    output=dict(note="Observation-only check at returned candidates; no truth read or solver policy altered.",
                candidates=rows,recovered_collision_sensitivity=collision)
    path=Path(__file__).with_name("candidate_audit.json")
    path.write_text(json.dumps(output,indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
