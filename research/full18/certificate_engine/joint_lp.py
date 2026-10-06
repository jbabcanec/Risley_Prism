"""Exact shared-geometry LP relaxation and sparse exclusion certificates.

Forward interval coefficients are provided by the unchanged full-vector engine.
Every midpoint row is lowered by an exact coefficient-radius support bound.
Thus a feasible LP is only a necessary test, never a physical-feasibility claim.
No numerical LP dependency or floating-point arithmetic is used.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as F
import argparse
import hashlib
from pathlib import Path
import json

import engine

VERSION = "exact-joint-affine-certificates-1"
GEOMETRY = tuple(engine.GEOMETRY)


@dataclass(frozen=True)
class Row:
    """a.q + b - epsilon*delta is a lower bound for a true violation."""
    label: str
    a: tuple
    b: F
    delta: int
    strict: bool
    coefficient_error: F = F(0)

    def json(self):
        return {"id": self.label, "a": list(map(str, self.a)), "b": str(self.b),
                "delta": self.delta, "strict": self.strict,
                "coefficient_error": str(self.coefficient_error)}


def _dot(a, b):
    return sum((x*y for x, y in zip(a, b)), F(0))


def _bounds(request):
    return tuple((engine.rational(request["box"][n][0]),
                  engine.rational(request["box"][n][1])) for n in GEOMETRY)


def _lower_row(label, intervals, sign, shift, delta, strict, bounds):
    if len(intervals) != 5:
        raise ValueError("An affine form must have four shared coefficients and one constant")
    mid, rad = [], []
    for pair in intervals:
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError("Invalid interval coefficient")
        lo, hi = map(engine.rational, pair)
        if lo > hi:
            raise ValueError("Reversed coefficient enclosure")
        mid.append(sign*(lo+hi)/2)
        rad.append((hi-lo)/2)
    error = rad[4]+sum((rad[j]*max(abs(lo), abs(hi))
                       for j, (lo, hi) in enumerate(bounds)), F(0))
    return Row(label, tuple(mid[:4]), mid[4]+shift-error,
               delta, strict, error)


def extract_rows(request, forward):
    """Use every available row, including complete prefixes of partial traces.

    Forward enclosures must be those obtained for this request. analyze validates
    their identity; verification independently regenerates them.
    """
    bounds = _bounds(request)
    rows = []
    observations = request.get("observations")
    obs_count = 0
    traversal_count = 0
    samples = forward.get("samples", [])
    for sample in samples:
        k = sample["sample"]
        if type(k) is not int or not 0 <= k < engine.COUNT:
            raise ValueError("Invalid sample index")
        for stage in sample.get("stages", []):
            j = stage["prism"]
            if type(j) is not int or not 1 <= j <= 3:
                raise ValueError("Invalid prism index")
            for kind in ("internal", "external"):
                name = kind+"_affine"
                if name in stage:
                    rows.append(_lower_row(
                        f"k{k}:p{j}:{kind}", stage[name], -1, F(0), 0, True, bounds))
                    traversal_count += 1
        if observations is not None and "output_affine" in sample:
            forms = sample["output_affine"]
            if len(forms) != 2:
                raise ValueError("Two output affine forms are required")
            for axis in range(2):
                y = engine.rational(observations[k][axis])
                for sign in (1, -1):
                    rows.append(_lower_row(
                        f"k{k}:axis{axis}:sign{sign}", forms[axis], sign,
                        -sign*y, 1, False, bounds))
                    obs_count += 1
    if len({row.label for row in rows}) != len(rows):
        raise ValueError("Duplicate affine row identifiers")
    expected_obs = 800 if observations is not None else 0
    coverage = {
        "observation_rows": obs_count,
        "expected_observation_rows": expected_obs,
        "traversal_rows": traversal_count,
        "expected_traversal_rows": 1200,
        "geometry_box_rows": 8,
        "complete_affine_rows": obs_count == expected_obs and traversal_count == 1200,
        "scope": ("Complete here refers only to available affine rows; outgoing critical "
                  "normal and all other optical physical tests remain separate."),
    }
    return rows, bounds, coverage


def dual_evidence(rows, bounds, epsilon, weights):
    """Replay any nonnegative simplex weights with exact stationarity correction.

    weights maps row indices to exact nonnegative rationals.  A=sum lambda*a
    need not vanish: min_{q in Q} A.q is explicitly included.  Hence imperfect
    stationarity is safe and no tolerance can silently weaken this certificate.
    There are at most five explicit weighted affine rows. A nonoptimal proposal
    can additionally use geometry-box support faces; it is not asserted to be
    an inclusion-minimal circuit with at most five total inequalities.
    """
    if not weights or any(type(i) is not int or i < 0 or i >= len(rows)
                          for i in weights):
        raise ValueError("Invalid dual support")
    weights = {i: engine.rational(w) for i, w in weights.items()}
    if any(w <= 0 for w in weights.values()) or sum(weights.values(), F(0)) != 1:
        raise ValueError("Weights must be positive on their support and sum exactly to one")
    if len(weights) > 5:
        raise ValueError("This certificate format is restricted to at most five supported rows")
    A = tuple(sum((w*rows[i].a[j] for i, w in weights.items()), F(0))
              for j in range(4))
    center = tuple((lo+hi)/2 for lo, hi in bounds)
    radii = tuple((hi-lo)/2 for lo, hi in bounds)
    support_correction = _dot(A, center)-_dot(tuple(abs(v) for v in A), radii)
    b0 = sum((w*rows[i].b for i, w in weights.items()), F(0))
    slope = sum((w*rows[i].delta for i, w in weights.items()), F(0))
    a0 = b0+support_correction
    lower = a0-slope*epsilon
    strict = any(rows[i].strict for i in weights)
    excludes = lower > 0 or (lower == 0 and strict)
    relation = "positive" if lower > 0 else "zero_with_strict_row" if lower == 0 and strict else "nonexcluding"
    return {
        "weights": [{"row": i, "id": rows[i].label, "weight": str(weights[i])}
                    for i in sorted(weights)],
        "support_rows": [rows[i].json() for i in sorted(weights)],
        "support_size": len(weights),
        "support_count_scope": "Explicit affine rows; geometry-box support is additional",
        "weighted_normal": list(map(str, A)),
        "stationarity_residual_treatment": "Exact geometry-box support; no zero-residual assumption",
        "geometry_support_correction": str(support_correction),
        "a0": str(a0), "epsilon_coefficient": str(slope),
        "epsilon": str(epsilon), "lower_bound": str(lower),
        "strict_row_has_positive_weight": strict,
        "excludes_strict_physical": excludes,
        "excludes_weak_affine": lower > 0,
        "relation": relation,
        "symbolic_validity": {
            "slope": str(slope),
            "threshold": str(a0/slope) if slope else None,
            "endpoint_excluded_for_strict_systems": bool(slope and strict),
            "all_nonnegative_epsilon_excluded": bool(
                not slope and (a0 > 0 or (a0 == 0 and strict))),
            "condition": "a0 - slope*epsilon > 0, or equality with positive strict-row weight",
        },
    }


# Public descriptive alias used by the adaptive driver and independent audit.
build_rows = extract_rows


def verify_dual(rows, geometry_box, certificate):
    """Verify a generic sparse bound, including a nonexcluding bound."""
    try:
        listed = certificate["weights"]
        indices = [item["row"] for item in listed]
        if len(set(indices)) != len(indices):
            return False
        weights = {item["row"]: engine.rational(item["weight"]) for item in listed}
        bounds = tuple(tuple(engine.rational(v) for v in pair) for pair in geometry_box)
        if len(bounds) != 4 or any(lo > hi for lo, hi in bounds):
            return False
        return dual_evidence(rows, bounds, engine.rational(certificate["epsilon"]), weights) == certificate
    except (ValueError, KeyError, TypeError, ArithmeticError, IndexError):
        return False


def _solve(matrix, rhs):
    """Small exact Gaussian solve; singular bases are rejected."""
    n = len(rhs)
    augmented = [[F(v) for v in row]+[F(rhs[i])] for i, row in enumerate(matrix)]
    for j in range(n):
        pivot = next((i for i in range(j, n) if augmented[i][j]), None)
        if pivot is None:
            raise ArithmeticError("Singular LP basis")
        augmented[j], augmented[pivot] = augmented[pivot], augmented[j]
        scale = augmented[j][j]
        augmented[j] = [v/scale for v in augmented[j]]
        for i in range(n):
            if i != j and augmented[i][j]:
                scale = augmented[i][j]
                augmented[i] = [a-scale*b for a, b in zip(augmented[i], augmented[j])]
    return tuple(row[-1] for row in augmented)


def solve_rows(rows, bounds, epsilon=F(0), pivot_limit=64):
    """Bounded exact vertex walk for min_q max_i lower_violation_i(q).

    Resource limits and detected repeated bases return an explicitly unresolved
    optimizer. All proposed sparse dual bounds are valid independently of the
    walk, including nonoptimal proposals. No termination theorem is relied on.
    """
    if type(pivot_limit) is not int or not 0 <= pivot_limit <= 10000:
        raise ValueError("pivot_limit must be an integer in [0,10000]")
    epsilon = engine.rational(epsilon)
    if epsilon < 0 or len(bounds) != 4 or any(lo > hi for lo, hi in bounds):
        raise ValueError("Invalid LP epsilon or geometry bounds")
    if not rows:
        return {"termination": "no_available_affine_rows", "pivots": 0,
                "optimal": False, "dual": None, "relaxation_feasible": None}
    for row in rows:
        if len(row.a) != 4 or row.delta not in (0, 1) or type(row.strict) is not bool:
            raise ValueError("Invalid affine row")
    # A_i.(q,t) <= rhs_i. Geometry bounds are additional true constraints.
    coeff = [tuple(row.a)+(F(-1),) for row in rows]
    rhs = [-(row.b-epsilon*row.delta) for row in rows]
    m = len(rows)
    lower_ids = []
    for j, (lo, hi) in enumerate(bounds):
        e = [F(0)]*5
        e[j] = F(-1)
        lower_ids.append(len(coeff))
        coeff.append(tuple(e))
        rhs.append(-lo)
        e = [F(0)]*5
        e[j] = F(1)
        coeff.append(tuple(e))
        rhs.append(hi)
    q = tuple(lo for lo, _ in bounds)
    violations = [_dot(row.a, q)+row.b-epsilon*row.delta for row in rows]
    t = max(violations)
    anchor_row = next(i for i, v in enumerate(violations) if v == t)
    point = q+(t,)
    basis = lower_ids+[anchor_row]
    objective = (F(0), F(0), F(0), F(0), F(1))
    best = None
    visited = set()
    pivots = 0

    def consider(weights):
        nonlocal best
        if not weights:
            return None
        total = sum(weights.values(), F(0))
        if total <= 0:
            return None
        proposal = dual_evidence(rows, bounds, epsilon,
                                 {i: w/total for i, w in weights.items() if w > 0})
        if (best is None or F(proposal["lower_bound"]) > F(best["lower_bound"])
            or (proposal["lower_bound"] == best["lower_bound"] and
                proposal["strict_row_has_positive_weight"])):
            best = proposal
        return proposal

    # A one-row box-support proof is cheap and useful before any LP pivots.
    for i in range(m):
        candidate = consider({i: F(1)})
        if candidate["excludes_strict_physical"]:
            return {"termination": "sparse_certificate", "pivots": 0,
                    "optimal": False, "dual": candidate, "relaxation_feasible": None,
                    "point": list(map(str, point)), "basis": basis}

    termination = "pivot_limit"
    optimal = False
    while True:
        key = tuple(sorted(basis))
        if key in visited:
            termination = "repeated_basis"
            break
        visited.add(key)
        B = [coeff[i] for i in basis]
        transpose = list(map(tuple, zip(*B)))
        multipliers = _solve(transpose, tuple(-v for v in objective))
        proposal = consider({i: lam for i, lam in zip(basis, multipliers)
                             if i < m and lam > 0})
        if all(lam >= 0 for lam in multipliers):
            optimal = True
            termination = "optimal"
            # Exact KKT and box support must agree with the primal objective.
            if proposal is None or F(proposal["lower_bound"]) != point[4]:
                raise ArithmeticError("Exact primal-dual equality failed")
            best = proposal
            break
        if proposal is not None and proposal["excludes_strict_physical"]:
            termination = "sparse_certificate"
            best = proposal
            break
        if pivots >= pivot_limit:
            break
        # Deterministic smallest constraint index among negative multipliers.
        leave = min((slot for slot, lam in enumerate(multipliers) if lam < 0),
                    key=lambda slot: basis[slot])
        desired = [F(0)]*5
        desired[leave] = F(-1)
        direction = _solve(B, desired)
        if direction[4] >= 0:
            raise ArithmeticError("LP direction failed exact descent check")
        basis_set = set(basis)
        blockers = []
        for i, row in enumerate(coeff):
            if i in basis_set:
                continue
            rate = _dot(row, direction)
            if rate > 0:
                slack = rhs[i]-_dot(row, point)
                if slack < 0:
                    raise ArithmeticError("LP primal feasibility was lost")
                blockers.append((slack/rate, i))
        if not blockers:
            raise ArithmeticError("Compact geometry and a nonempty row set cannot yield an unbounded LP")
        step, enter = min(blockers)
        point = tuple(v+step*d for v, d in zip(point, direction))
        basis[leave] = enter
        pivots += 1
    # Check every primal inequality in exact arithmetic before reporting a point.
    if any(_dot(row, point) > b for row, b in zip(coeff, rhs)):
        raise ArithmeticError("LP final point is not feasible")
    return {"termination": termination, "pivots": pivots, "optimal": optimal,
            "dual": best, "point": list(map(str, point)), "basis": basis,
            "objective": str(point[4]),
            "relaxation_feasible": point[4] <= 0 if optimal else None,
            "meaning": "Feasibility concerns only the lowered weak affine relaxation"}


def _forward_identity(request, forward):
    if forward.get("input_sha256") != engine.digest(request):
        raise ValueError("Forward certificate belongs to a different request")
    if forward.get("source_sha256") != engine.source_hashes():
        raise ValueError("Forward certificate comes from different source bytes")
    if forward.get("geometry_order") != list(GEOMETRY)+["constant"]:
        raise ValueError("Unexpected geometry coefficient order")


def analyze(request, forward=None, pivot_limit=64):
    """Construct a certificate; supplied forward evidence is checked on replay."""
    engine.validate(request)
    if forward is None:
        forward = engine.certify(request)
    _forward_identity(request, forward)
    rows, bounds, coverage = extract_rows(request, forward)
    epsilon = engine.rational(request.get("epsilon", 0))
    solved = solve_rows(rows, bounds, epsilon, pivot_limit)
    dual = solved["dual"]
    excluded = dual is not None and dual["excludes_strict_physical"]
    here = Path(__file__)
    result = {
        "engine": VERSION, "input_sha256": engine.digest(request),
        "source_sha256": {**engine.source_hashes(),
                          "joint_lp.py": hashlib.sha256(here.read_bytes()).hexdigest()},
        "forward_sha256": engine.digest(forward),
        "status": "excluded" if excluded else "unresolved",
        "meaning": ("Sparse exact dual excludes every strict physical compatible point in the box"
                    if excluded else
                    "Retain the complete box; a nonexcluding outer LP does not prove physical feasibility"),
        "rounding": ("Exact Fraction arithmetic after forward outward interval enclosures; coefficient "
                     "radii lowered by exact geometry support; exact nonzero dual residual correction"),
        "coverage": coverage, "rows_sha256": engine.digest([row.json() for row in rows]),
        "row_count": len(rows),
        "geometry_order": list(GEOMETRY),
        "geometry_box": [[str(lo), str(hi)] for lo, hi in bounds],
        "epsilon": str(epsilon),
        "resource_limits": {"pivot_limit": pivot_limit},
        "diagnostics": {"pivots": solved["pivots"], "termination": solved["termination"]},
        "lp": solved,
        "certificate": dual if excluded else None,
        "support_row_indices": [item["row"] for item in dual["weights"]] if excluded else [],
    }
    return result


def verify(request, certificate, forward=None, replay_forward=True):
    """Replay complete evidence deterministically, including the bounded LP.

    The forward certificate is recomputed unless the caller supplies a forward
    value already independently verified and explicitly sets replay_forward=False.
    Never pass untrusted stored forward bytes in that optimization. Replaying all arithmetic is intentionally
    stronger than accepting a self-reported feasible LP or dual status.
    """
    try:
        if forward is None:
            forward = engine.certify(request)
        elif replay_forward and not engine.verify(request, forward):
            return False
        limit = certificate["resource_limits"]["pivot_limit"]
        return analyze(request, forward, pivot_limit=limit) == certificate
    except (ValueError, KeyError, TypeError, ArithmeticError, IndexError):
        return False


def verify_sparse(request, certificate, forward=None, replay_forward=True):
    """Check an exclusion without rerunning the optimizer.

    All row derivations, coefficient radii, support corrections and strict flags
    are regenerated.  Metadata, source hashes and input hashes are checked.
    """
    try:
        engine.validate(request)
        if forward is None:
            forward = engine.certify(request)
        elif replay_forward and not engine.verify(request, forward):
            return False
        _forward_identity(request, forward)
        if certificate.get("engine") != VERSION or certificate.get("status") != "excluded":
            return False
        expected_sources = {**engine.source_hashes(),
                            "joint_lp.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        if (certificate["input_sha256"] != engine.digest(request)
            or certificate["source_sha256"] != expected_sources
            or certificate["forward_sha256"] != engine.digest(forward)):
            return False
        rows, bounds, coverage = extract_rows(request, forward)
        if (certificate["rows_sha256"] != engine.digest([row.json() for row in rows])
            or certificate["row_count"] != len(rows) or certificate["coverage"] != coverage
            or certificate["geometry_box"] != [[str(lo), str(hi)] for lo, hi in bounds]
            or certificate["geometry_order"] != list(GEOMETRY)
            or certificate["epsilon"] != str(engine.rational(request.get("epsilon", 0)))):
            return False
        proposed = certificate["certificate"]
        listed = proposed["weights"]
        if not isinstance(listed, list):
            return False
        indices = [item["row"] for item in listed]
        if len(set(indices)) != len(indices):
            return False
        weights = {item["row"]: engine.rational(item["weight"]) for item in listed}
        regenerated = dual_evidence(rows, bounds,
                                    engine.rational(request.get("epsilon", 0)), weights)
        return regenerated == proposed and regenerated["excludes_strict_physical"]
    except (ValueError, KeyError, TypeError, ArithmeticError, IndexError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("analyze")
    build.add_argument("input")
    build.add_argument("output")
    build.add_argument("--pivot-limit", type=int, default=64)
    check = sub.add_parser("verify")
    check.add_argument("input")
    check.add_argument("certificate")
    check.add_argument("--sparse-only", action="store_true")
    args = parser.parse_args()
    request = engine.load_json(args.input)
    if args.command == "analyze":
        result = analyze(request, pivot_limit=args.pivot_limit)
        engine.dump_json(args.output, result)
        print(json.dumps({"status": result["status"], "rows": result["row_count"],
                          "termination": result["lp"]["termination"],
                          "pivots": result["lp"]["pivots"]}))
    else:
        certificate = engine.load_json(args.certificate)
        valid = (verify_sparse if args.sparse_only else verify)(request, certificate)
        print(json.dumps({"verified": valid, "sparse_only": args.sparse_only}))
        if not valid:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
