#!/usr/bin/env python3
"""The four atlas runs share seed0 = 424242, so case indices overlap: 713,572
rows hold 669,589 distinct configurations. Cases run in more than one atlas
are an exact reproducibility test: same input, independent run."""
import os, re, json, collections
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', '..', 'results')
pc = re.compile(rb'"case": (\d+)'); pt = re.compile(rb'"T_solved": (null|[\d.]+)')
pe = re.compile(rb'"err": ([^,]+),')
runs = collections.defaultdict(dict)
for a in ['atlas2', 'atlas3', 'atlas4', 'atlas5']:
    for f in os.listdir(os.path.join(RES, a)):
        for line in open(os.path.join(RES, a, f), 'rb'):
            c = pc.search(line); t = pt.search(line)
            if not c or not t:
                continue
            errs = pe.findall(line)
            ts = None if t.group(1) == b'null' else float(t.group(1))
            runs[int(c.group(1))][a] = (ts, float(errs[-1]) if errs else None)
multi = {k: v for k, v in runs.items() if len(v) > 1}
same_T = sum(len({x[0] for x in v.values()}) == 1 for v in multi.values())
pairs = collections.Counter(tuple(sorted(v)) for v in multi.values())
out = dict(rows=sum(len(v) for v in runs.values()), unique=len(runs),
           multi=len(multi), identical_outcome=same_T,
           atlas_combos={'+'.join(k): n for k, n in pairs.items()})
diff = [(k, v) for k, v in multi.items() if len({x[0] for x in v.values()}) > 1]
out['examples_differing'] = [[k, {a: x[0] for a, x in v.items()}] for k, v in diff[:10]]
json.dump(out, open(os.path.join(HERE, 'out', 'atlas_overlap.json'), 'w'), indent=1)
print(json.dumps(out, indent=1))
