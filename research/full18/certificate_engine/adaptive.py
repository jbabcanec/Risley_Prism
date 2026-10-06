"""Coverage-preserving bounded optical subdivision, never an unchecked inverse.

Each leaf retains the complete shared geometry box. Resource stops preserve the
unprocessed frontier explicitly. Verification replays leaf proofs and bounded LP runs.
"""
from __future__ import annotations
from collections import deque
from copy import deepcopy
from fractions import Fraction as F
import argparse
import hashlib
from pathlib import Path
from engine import (OPTICAL, GEOMETRY, NAMES, certify, validate, digest,
                    load_json, dump_json, verify as verify_forward)
import joint_lp

VERSION = "bounded-adaptive-cover-1"


def source_hashes():
    here = Path(__file__).resolve().parent
    return {name: hashlib.sha256((here/name).read_bytes()).hexdigest()
            for name in ("adaptive.py", "joint_lp.py", "engine.py", "interval.py")}


def limits(max_nodes, max_depth, lp_pivot_limit, max_total_lp_pivots):
    values = dict(max_nodes=max_nodes, max_depth=max_depth,
                  lp_pivot_limit=lp_pivot_limit, max_total_lp_pivots=max_total_lp_pivots)
    for name, value in values.items():
        if type(value) is not int or not 0 <= value <= 10000:
            raise ValueError(name+" must be an integer in [0,10000]")
    return values


def exact_box(box):
    return {name: [str(F(a)), str(F(b))] for name, (a,b) in box.items()}


def request_for(request, box):
    out = deepcopy(request)
    out["box"] = deepcopy(box)
    return out


def split_axis(box, root):
    choices = []
    for position, name in enumerate(OPTICAL):
        width = F(box[name][1])-F(box[name][0])
        root_width = F(root[name][1])-F(root[name][0])
        if width > 0 and root_width > 0:
            choices.append((width/root_width, -position, name))
    return max(choices)[2] if choices else None


def summary(nodes, evaluations, pivots):
    counts = {name: sum(n["outcome"] == name for n in nodes)
              for name in ("excluded", "retained", "unresolved", "split")}
    leaves = counts["excluded"]+counts["retained"]+counts["unresolved"]
    completion = ("unresolved_cover" if counts["unresolved"] else
                  "input_box_excluded" if counts["retained"] == 0 else
                  "input_box_partition_certified")
    return {"nodes": len(nodes), "leaves": leaves, "outcomes": counts,
            "forward_evaluations": evaluations, "lp_pivots": pivots,
            "coverage_preserved": True, "completion": completion,
            "global_recovery_claim": False}


def run(request, *, max_nodes=7, max_depth=3, lp_pivot_limit=24, max_total_lp_pivots=96):
    validate(request)
    if request.get("observations") is None:
        raise ValueError("Adaptive joint feasibility requires the original paired observations")
    budget = limits(max_nodes,max_depth,lp_pivot_limit,max_total_lp_pivots)
    root = exact_box(request["box"])
    queue = deque([("r",0,root)])
    nodes = []
    evaluations = pivots = 0
    while queue:
        node_id, depth, box = queue.popleft()
        node = {"id": node_id, "depth": depth, "box": box}
        nodes.append(node)
        if evaluations >= max_nodes:
            node.update(outcome="unresolved", reason="forward_node_budget")
            continue
        local = request_for(request,box)
        forward = certify(local)
        evaluations += 1
        node["forward"] = forward
        if forward["status"] == "excluded":
            node.update(outcome="excluded", reason="forward_certificate")
            continue
        if forward["status"] == "retained":
            node.update(outcome="retained", reason="whole_box_certificate")
            continue

        pivot_allowance = min(lp_pivot_limit, max_total_lp_pivots-pivots)
        joint = joint_lp.analyze(local, forward, pivot_limit=pivot_allowance)
        node["joint"] = joint
        node["lp_pivot_allowance"] = pivot_allowance
        used = joint["diagnostics"]["pivots"]
        if type(used) is not int or not 0 <= used <= pivot_allowance:
            raise ArithmeticError("Joint LP reported an invalid pivot count")
        pivots += used
        if joint["status"] == "excluded":
            node.update(outcome="excluded", reason="joint_sparse_dual")
            continue
        if depth >= max_depth:
            node.update(outcome="unresolved", reason="depth_budget")
            continue
        axis = split_axis(box,root)
        if axis is None:
            node.update(outcome="unresolved", reason="no_optical_width")
            continue
        lo,hi = map(F,box[axis])
        midpoint = (lo+hi)/2
        left,right = deepcopy(box),deepcopy(box)
        left[axis][1] = str(midpoint)
        right[axis][0] = str(midpoint)
        children = [node_id+".0",node_id+".1"]
        node.update(outcome="split", reason="unresolved_subdivision",
                    split={"coordinate":axis,"midpoint":str(midpoint),"children":children})
        queue.append((children[0],depth+1,left))
        queue.append((children[1],depth+1,right))
    return {"engine": VERSION, "input_sha256": digest(request),
            "source_sha256": source_hashes(), "root_box":root,
            "limits":budget, "nodes":nodes,
            "summary":summary(nodes,evaluations,pivots),
            "scope":("Exact closed-box coverage of the supplied root with shared geometry. "
                     "Only excluded leaves are discarded; unresolved leaves remain. "
                     "No complete practical inverse or native-accuracy claim.")}


def verify(request, certificate):
    """Replay every claimed exclusion/retention and prove the tree covers its root.

    This checks the saved cover and replays each bounded LP run. Unresolved leaves
    are always conservative; they cannot be promoted to a recovery claim.
    """
    try:
        validate(request)
        if request.get("observations") is None:
            return False
        if certificate["engine"] != VERSION or certificate["input_sha256"] != digest(request):
            return False
        if certificate["source_sha256"] != source_hashes():
            return False
        budget = limits(**certificate["limits"])
        root = exact_box(request["box"])
        if certificate["root_box"] != root:
            return False
        node_list = certificate["nodes"]
        if not node_list or len(node_list) > 2*budget["max_nodes"]+1:
            return False
        nodes = {node["id"]:node for node in node_list}
        if len(nodes) != len(node_list):
            return False
        seen=set()
        evaluations=pivots=0
        stack=[("r",0,root)]
        while stack:
            node_id,depth,box=stack.pop()
            if node_id in seen or node_id not in nodes:
                return False
            seen.add(node_id)
            node=nodes[node_id]
            if node["depth"]!=depth or node["box"]!=box or depth>budget["max_depth"]:
                return False
            if any(node["box"][g] != root[g] for g in GEOMETRY):
                return False
            local=request_for(request,box)
            forward=node.get("forward")
            joint=node.get("joint")
            if forward is not None:
                if not verify_forward(local,forward):
                    return False
                evaluations+=1
            if joint is not None:
                if forward is None or forward["status"]!="unresolved":
                    return False
                allowance=node.get("lp_pivot_allowance")
                if type(allowance) is not int or not 0 <= allowance <= budget["lp_pivot_limit"]:
                    return False
                if joint["resource_limits"]["pivot_limit"] != allowance:
                    return False
                used=joint["diagnostics"]["pivots"]
                if type(used) is not int or not 0 <= used <= allowance:
                    return False
                if not joint_lp.verify(local,joint,forward=forward,replay_forward=False):
                    return False
                pivots+=used
            outcome=node["outcome"]
            reason=node["reason"]
            if outcome=="excluded":
                valid_forward=forward is not None and forward["status"]=="excluded"
                valid_joint=joint is not None and joint["status"]=="excluded"
                if not ((reason=="forward_certificate" and valid_forward) or
                        (reason=="joint_sparse_dual" and valid_joint)):
                    return False
            elif outcome=="retained":
                if reason!="whole_box_certificate" or forward is None or forward["status"]!="retained":
                    return False
            elif outcome=="unresolved":
                if reason=="forward_node_budget":
                    if forward is not None or joint is not None:
                        return False
                elif reason in ("depth_budget","no_optical_width"):
                    if (forward is None or forward["status"]!="unresolved" or
                            joint is None or joint["status"]!="unresolved"):
                        return False
                    if reason=="depth_budget" and depth!=budget["max_depth"]:
                        return False
                    if reason=="no_optical_width" and split_axis(box,root) is not None:
                        return False
                else:
                    return False
            elif outcome=="split":
                if (reason!="unresolved_subdivision" or depth>=budget["max_depth"] or
                        forward is None or forward["status"]!="unresolved" or
                        joint is None or joint["status"]!="unresolved"):
                    return False
                data=node["split"]
                axis=data["coordinate"]
                if axis not in OPTICAL or axis!=split_axis(box,root):
                    return False
                lo,hi=map(F,box[axis])
                mid=F(data["midpoint"])
                if not lo<mid<hi or mid!=(lo+hi)/2:
                    return False
                children=[node_id+".0",node_id+".1"]
                if data["children"]!=children:
                    return False
                left,right=deepcopy(box),deepcopy(box)
                left[axis][1]=str(mid)
                right[axis][0]=str(mid)
                stack.extend([(children[1],depth+1,right),(children[0],depth+1,left)])
            else:
                return False
            if outcome!="split" and "split" in node:
                return False
        if seen!=set(nodes):
            return False
        if evaluations>budget["max_nodes"] or pivots>budget["max_total_lp_pivots"]:
            return False
        if any(n["reason"]=="forward_node_budget" for n in node_list) and evaluations!=budget["max_nodes"]:
            return False
        return certificate["summary"]==summary(node_list,evaluations,pivots)
    except (KeyError,ValueError,TypeError,ArithmeticError,AssertionError):
        return False


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest="command",required=True)
    build=commands.add_parser("run")
    build.add_argument("input")
    build.add_argument("output")
    build.add_argument("--max-nodes",type=int,default=7)
    build.add_argument("--max-depth",type=int,default=3)
    build.add_argument("--lp-pivot-limit",type=int,default=24)
    build.add_argument("--max-total-lp-pivots",type=int,default=96)
    check=commands.add_parser("verify")
    check.add_argument("input")
    check.add_argument("certificate")
    args=parser.parse_args()
    request=load_json(args.input)
    if args.command=="run":
        result=run(request,max_nodes=args.max_nodes,max_depth=args.max_depth,
                   lp_pivot_limit=args.lp_pivot_limit,max_total_lp_pivots=args.max_total_lp_pivots)
        dump_json(args.output,result)
        print(result["summary"])
    else:
        valid=verify(request,load_json(args.certificate))
        print({"verified":valid})
        if not valid:
            raise SystemExit(1)


if __name__=="__main__":
    main()
