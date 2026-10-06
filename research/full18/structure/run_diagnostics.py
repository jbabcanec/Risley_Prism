"""Reproduce the passive full18 structure report, without modifying Dropbox."""
import os
import sys
for name in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[name]="1"
sys.dont_write_bytecode=True
import argparse
import csv
import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np
import scipy
from diagnostics import analyze, validate_derivatives, validate_poc3, REPO, LO, HI, NAMES


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--cases",type=Path,default=Path(__file__).parents[1]/"cases.json")
    parser.add_argument("--output",type=Path,default=Path(__file__).parent/"results.json")
    args=parser.parse_args()
    data=json.loads(args.cases.read_text(encoding="utf-8-sig"))
    start=time.perf_counter()
    output=dict(scope="Original passive per-axis full18, 200 samples, 10 seconds, full native box; arbitrary physical order.",
        limitations="Local conditioning diagnostics, not blind recovery or finite-noise/global certificates.",
        python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
        source_cases_sha256=hashlib.sha256(args.cases.read_bytes()).hexdigest(),
        bounds=dict(lower=LO.tolist(),upper=HI.tolist()),names=NAMES,
        source_sha256={str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [REPO/"risley_lattice"/name for name in ("model.py","separable.py","gains.py","fmodel.py")]},
        cases=[])
    for i, case in enumerate(data["cases"]):
        row=analyze(case["truth"],case["observed"])
        row["id"],row["kind"]=case["id"],case["kind"]
        row["paired_face_transfer_validation"]=validate_poc3(case["truth"])
        if i in (0,4,len(data["cases"])-1):
            row["derivative_validation"]=validate_derivatives(case["truth"],case["observed"])
        output["cases"].append(row)
        print(json.dumps(dict(id=row["id"],full_condition=row["full18_box_scaled"]["condition"],
            affine_condition=row["affine4_box_scaled"]["condition"],
            reduced_condition=row["reduced14_box_scaled"]["condition"],
            worst_parameter=row["worst_native_sensitivity_parameter"],
            amplification=max(row["native_linearized_error_amplification_l1"]),
            canonical_parity=row["canonical_smooth_max_abs"])),flush=True)
    output["elapsed_seconds"]=time.perf_counter()-start
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2)+"\n",encoding="utf-8")
    with args.output.with_suffix(".csv").open("w",newline="",encoding="utf-8") as stream:
        fields=["case","eta","full18_rank","interpretation"]+NAMES
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for row in output["cases"]:
            for budget in row["error_budget_table"]:
                writer.writerow(dict(case=row["id"],eta=budget["eta"],
                    full18_rank=row["sensitivity_rank"],
                    interpretation="First-order local LS error only; not certified finite-noise error",
                    **dict(zip(NAMES,budget["single_fit_linearized_error_by_parameter"]))))
    print(str(args.output),flush=True)


if __name__=="__main__":
    main()
