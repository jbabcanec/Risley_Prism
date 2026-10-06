"""Bounded rational-interval full-vector certificate engine.

All eighteen chart coordinates are inputs, never fitted or fixed internally.
This is a box certificate evaluator, not a complete inverse or a search engine.
"""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
from interval import I, precision, tan_pi_fraction, IntervalDomainError

OPTICAL = ["r1", "r2", "r3", "p1", "p2", "p3",
           "v1", "v2", "v3", "n1", "n2", "n3", "tx", "ty"]
GEOMETRY = ["bx", "by", "g", "d"]
NAMES = OPTICAL + GEOMETRY
MODEL = "coupled_vector_snell_three_prism"
VERSION = "bounded-box-certificates-1"
COUNT = 200


def rational(value):
    if isinstance(value, bool) or isinstance(value, float):
        raise ValueError("Use exact rational strings or integers, never JSON floats/bools")
    if not isinstance(value, (str, int, F)):
        raise ValueError("Expected an exact rational value")
    return F(value)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      parse_float=lambda x: (_ for _ in ()).throw(
                          ValueError("JSON decimal numbers prohibited; quote exact decimals")))


def dump_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def source_hashes():
    here = Path(__file__).resolve().parent
    return {n: hashlib.sha256((here/n).read_bytes()).hexdigest()
            for n in ("engine.py", "interval.py")}


def validate(request):
    if request.get("model") != MODEL:
        raise ValueError("The coupled-vector model identifier is required")
    provenance = str(request.get("provenance", "")).lower()
    if not provenance or "canonical" in provenance or "independent_axis" in provenance:
        raise ValueError("State data provenance; canonical independent-axis records are excluded")
    if request.get("sampling") != {"count": 200, "step": "1/20", "start": "0"}:
        raise ValueError("Only the original 200 times k/20 are supported")
    bits = request.get("bits", 80)
    if type(bits) is not int or not 32 <= bits <= 256:
        raise ValueError("bits must be an integer in [32,256]")
    box = request.get("box", {})
    if set(box) != set(NAMES):
        raise ValueError("Exactly all eighteen original chart coordinates are required")
    for name in NAMES:
        pair = box[name]
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError("Each chart coordinate needs a [lower,upper] pair")
        if rational(pair[0]) > rational(pair[1]):
            raise ValueError("Reversed box interval")
    if "timestamps" in request:
        ts = request["timestamps"]
        if len(ts) != COUNT or any(rational(t) != F(k, 20) for k, t in enumerate(ts)):
            raise ValueError("Timestamps must equal exact k/20")
    obs = request.get("observations")
    if obs is not None:
        if len(obs) != COUNT or any(not isinstance(p, list) or len(p) != 2 for p in obs):
            raise ValueError("Observations must be 200 exact [x,y] pairs")
        for pair in obs:
            for y in pair:
                rational(y)
        if rational(request.get("epsilon")) < 0:
            raise ValueError("epsilon must be a nonnegative exact rational")
    return bits


def prior_limits():
    # Exact algebraic endpoints are enclosed via certified rational pi/Taylor arithmetic.
    wedge = tan_pi_fraction(1, 10)
    phase = tan_pi_fraction(1, 20)
    speed = tan_pi_fraction(7, 40)
    beam = tan_pi_fraction(5, 36)
    result = {}
    for names, limit in ((OPTICAL[0:3], wedge), (OPTICAL[3:6], phase),
                         (OPTICAL[6:9], speed), (["tx", "ty"], beam)):
        for name in names:
            result[name] = (-limit, limit)
    for name in ["n1", "n2", "n3"]:
        result[name] = (I("13/10"), I("9/5"))
    for name in ["bx", "by"]:
        result[name] = (I(-5), I(5))
    result["g"] = (I(2), I(15))
    result["d"] = (I(50), I(200))
    return result, wedge


def prior_check(box, limits):
    evidence = {}
    whole = True
    excluded = None
    for name in NAMES:
        value = box[name]
        lower, upper = limits[name]
        inside = value.lo >= lower.hi and value.hi <= upper.lo
        outside = value.hi < lower.lo or value.lo > upper.hi
        evidence[name] = {"input_enclosure": value.json(), "native_chart_lower": lower.json(),
                          "native_chart_upper": upper.json(),
                          "relation": "outside" if outside else "inside" if inside else "unresolved"}
        whole = whole and inside
        if outside and excluded is None:
            excluded = {"kind": "prior_disjoint", "coordinate": name,
                        "details": evidence[name]}
    return whole, excluded, evidence


def clip_unit(v):
    # Each exact complex rotor factor is unit modulus; these intersections are safe.
    lo, hi = max(v.lo, F(-1)), min(v.hi, F(1))
    if lo > hi:
        raise ArithmeticError("Unit rotor enclosure became empty")
    return I(lo, hi)


def cmul(a, b):
    return (clip_unit(a[0]*b[0]-a[1]*b[1]),
            clip_unit(a[0]*b[1]+a[1]*b[0]))


def rotor_chart(t):
    den = 1+t.square()
    return (clip_unit((1-t.square())/den), clip_unit(2*t/den))


def rotors(p, v):
    start = rotor_chart(p)
    power = [rotor_chart(v)]
    for _ in range(7):
        power.append(cmul(power[-1], power[-1]))
    out = []
    for k in range(COUNT):
        z = start
        for j in range(8):
            if k & (1 << j):
                z = cmul(z, power[j])
        out.append(z)
    return out


def azero():
    return [I(0) for _ in range(5)]


def aconst(value):
    out = azero()
    out[4] = I(value)
    return out


def abasis(index):
    out = azero()
    out[index] = I(1)
    return out


def aadd(a, b):
    return [x+y for x, y in zip(a, b)]


def ascale(s, a):
    return [s*x for x in a]


def adiv(a, s):
    return [x/s for x in a]


def adot(u, p):
    return aadd(ascale(u[0], p[0]), ascale(u[1], p[1]))


def aevaluate(a, geometry):
    return a[4]+sum((a[j]*geometry[j] for j in range(4)), I(0))


def aj(a):
    return [v.json() for v in a]


def observation_margin(observations, epsilon, wedge):
    rows = []
    for k, pair in enumerate(observations):
        ax, ay = abs(rational(pair[0]))+epsilon, abs(rational(pair[1]))+epsilon
        radius = (I(ax).square()+I(ay).square()).sqrt()
        M = wedge*radius
        lower = max(F(0), (F(50)-M.hi)/53)
        rows.append({"sample": k, "time": str(F(k, 20)), "screen_support_upper": str(M.hi),
                     "ratio_R_over_Z_lower": str(lower)})
    minimum = min(F(r["ratio_R_over_Z_lower"]) for r in rows)
    return {"status": "positive_margin_certified" if minimum > 0 else "no_positive_margin_certified",
            "minimum_ratio_lower": str(minimum), "samples": rows,
            "scope": "Conditional on original-prior weak/strict full-vector compatibility; not a feasibility test"}


def certify(request):
    bits = validate(request)
    with precision(bits):
        box = {name: I(rational(request["box"][name][0]),
                       rational(request["box"][name][1])) for name in NAMES}
        limits, wedge = prior_limits()
        prior_whole, excluded, prior_evidence = prior_check(box, limits)
        result = {"engine": VERSION, "model": MODEL, "input_sha256": digest(request),
                  "source_sha256": source_hashes(), "bits": bits, "status": "unresolved",
                  "meaning": "The complete input box is retained; this engine is not a complete inverse",
                  "sampling": request["sampling"], "provenance": request["provenance"],
                  "prior": prior_evidence, "samples": [], "witness": None,
                  "geometry_order": GEOMETRY+["constant"],
                  "rounding": "Exact fractions with outward dyadic arithmetic and certified integer square roots"}
        observations = request.get("observations")
        epsilon = rational(request["epsilon"]) if observations is not None else None
        if observations is not None:
            result["final_prism_margin"] = observation_margin(observations, epsilon, wedge)
        if excluded:
            result.update(status="excluded", witness=excluded,
                          meaning="No parameter in the input box satisfies the original prior")
            return result

        all_physical = prior_whole
        all_observations = True
        geometry = [box[n] for n in GEOMETRY]
        rotations = [rotors(box["p"+str(j)], box["v"+str(j)]) for j in (1, 2, 3)]
        beam_den = (1+box["tx"].square()+box["ty"].square()).sqrt()
        initial_X = [box["tx"]/beam_den, box["ty"]/beam_den]

        for k in range(COUNT):
            sample = {"sample": k, "time": str(F(k, 20)), "stages": []}
            result["samples"].append(sample)
            X = initial_X[:]
            pos = [aadd(abasis(0), aconst(6*box["tx"])),
                   aadd(abasis(1), aconst(6*box["ty"]))]
            for j in range(3):
                stage = {"prism": j+1}
                sample["stages"].append(stage)
                u = [box["r"+str(j+1)]*v for v in rotations[j][k]]
                n = box["n"+str(j+1)]
                stage["u"] = [v.json() for v in u]
                guards = {}
                stage["guards"] = guards

                def guard(name, interval):
                    nonlocal all_physical
                    guards[name] = interval.json()
                    if interval.hi <= 0:
                        result.update(status="excluded",
                                      meaning="A required strict physical guard is nonpositive throughout the box",
                                      witness={"kind": "strict_physical_guard", "sample": k,
                                               "prism": j+1, "guard": name, "interval": interval.json()})
                        return False
                    if interval.lo <= 0:
                        all_physical = False
                    return True

                H2 = n.square()-X[0].square()-X[1].square()
                if not guard("glass_radicand", H2):
                    return result
                if H2.lo <= 0:
                    result["witness"] = {"kind": "unresolved_glass_root", "sample": k, "prism": j+1}
                    return result
                H = H2.sqrt()
                P = H-u[0]*X[0]-u[1]*X[1]
                if not guard("incoming_normal_P", P):
                    return result
                if P.lo <= 0:
                    result["witness"] = {"kind": "unresolved_divisor_P", "sample": k, "prism": j+1}
                    return result
                D = 1+u[0].square()+u[1].square()
                delta = P.square()-D*(n.square()-1)
                if not guard("exit_radicand_delta", delta):
                    return result
                # Overlap with negative radicands is a domain uncertainty, never an exclusion.
                # Bound the nonnegative-root branch over the potentially physical subset.
                R = I(max(F(0), delta.lo), delta.hi).sqrt()
                c = (P-R)/D
                Xout = [X[a]+c*u[a] for a in range(2)]
                Z = H-c
                if not guard("outgoing_axial_Z", Z):
                    return result
                if Z.lo <= 0:
                    result["witness"] = {"kind": "unresolved_divisor_Z", "sample": k, "prism": j+1}
                    return result
                internal = aadd(aconst(3), adot(u, pos))
                internal_range = aevaluate(internal, geometry)
                if not guard("internal_traversal", internal_range):
                    return result
                pexit = [aadd(pos[a], ascale(X[a]/P, internal)) for a in range(2)]
                ell = abasis(2 if j < 2 else 3)
                flight = aadd(ell, ascale(I(-1), adot(u, pexit)))
                flight_range = aevaluate(flight, geometry)
                if not guard("external_traversal", flight_range):
                    return result
                pos = [aadd(pexit[a], ascale(Xout[a]/Z, flight)) for a in range(2)]
                stage.update(internal_affine=aj(internal), external_affine=aj(flight),
                             incoming_X=[v.json() for v in X], H=H.json(), R=R.json(),
                             outgoing_X=[v.json() for v in Xout], Z=Z.json())
                X = Xout
            outputs = [aevaluate(pos[a], geometry) for a in range(2)]
            sample["output_affine"] = [aj(a) for a in pos]
            sample["output"] = [v.json() for v in outputs]
            if observations is not None:
                residual = []
                for a, out in enumerate(outputs):
                    y = rational(observations[k][a])
                    low, high = y-epsilon, y+epsilon
                    residual.append((out-I(y)).json())
                    if out.hi < low or out.lo > high:
                        sample["residual"] = residual
                        result.update(status="excluded",
                                      meaning="An output enclosure is disjoint from its observation band",
                                      witness={"kind": "observation_disjoint", "sample": k,
                                               "coordinate": a, "output": out.json(),
                                               "band": [str(low), str(high)]})
                        return result
                    if out.lo < low or out.hi > high:
                        all_observations = False
                sample["residual"] = residual
        if all_physical and all_observations:
            result.update(status="retained",
                          meaning=("Every parameter in this entire eighteen-dimensional input box is within "
                                   "the prior and strictly physical at all 200 samples" +
                                   (" and satisfies all 400 observation bands" if observations is not None else "") +
                                   "; this is a conditional box certificate, not global recovery"))
        else:
            result["witness"] = {"kind": "unresolved_overlap",
                                 "whole_box_strict_physical": all_physical,
                                 "whole_box_in_observation_bands": all_observations}
        return result


def verify(request, certificate):
    # Deterministic certificate replay. Independent mathematical/code review is separate.
    return certify(request) == certificate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    make = sub.add_parser("certify")
    make.add_argument("input")
    make.add_argument("output")
    check = sub.add_parser("verify")
    check.add_argument("input")
    check.add_argument("certificate")
    args = p.parse_args()
    request = load_json(args.input)
    if args.command == "certify":
        result = certify(request)
        dump_json(args.output, result)
        print(json.dumps({"status": result["status"], "certificate": args.output,
                          "samples_processed": len(result["samples"])}))
    else:
        valid = verify(request, load_json(args.certificate))
        print(json.dumps({"verified": valid}))
        if not valid:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
