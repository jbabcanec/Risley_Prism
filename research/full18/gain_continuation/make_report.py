"""Assemble the continuation evidence without reading any hidden truths."""
import json,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
def read(name):return json.loads((HERE/name).read_text())
a=read('result_native.json');b=read('result_replay.json');p=read('compatible_profiles.json')
v=read('validation.json');j=read('jacobian_sweep.json');f3=read('fresh_random_03.json');f7=read('fresh_random_07.json')
assert a['theta']==b['theta']
assert a['check']['interval_compatible'] and a['check']['canonical_compatible']
assert p['accepted_profiles']==6
lines=[
'Full-domain gain continuation: observation-only constructive recovery',
'',
'The successful adverse-record run starts exclusively from the saved candidate in profiles/blind_adverse_1e8.json and the supplied observations in profiles/adverse_observations_only.json. No truth vector, ambiguity endpoint, known glass index or known distance is read.',
'',
'Domain: u_i=(n_i-1)tan(ax_i), n_i in [1.3,1.8]. With t=tan(18 degrees), the full native wedge/index domain is exactly the gain rectangle |u_i|<=.8t intersected with the six linear inequalities +/-u_i<=(n_i-1)t. Inversion ax_i=atan(u_i/(n_i-1)) gives signed native wedges. This retains the original domain; the old weak-prism rectangle |u|<=.3t would remove part of it.',
'',
'All eighteen coordinates remain unknown. Four affine variables (d_W,gap,bm_px,bm_py) are eliminated by bounded least squares using their original bounds. Fourteen nonlinear coordinates are optimized. Explicit coupled-domain and optical penalties guide trial points; final candidates must independently satisfy every original bound and physical guard.',
'',
f"Successful first run: {a['evaluations']} evaluations, {a['seconds']:.6f} seconds after imports; former maximum residual 8.647446230964917e-4.",
f"Returned stable-model max residual: {a['check']['strict_numerical_max_residual']:.17g}.",
f"Original interval-model all-sample error upper bound: {a['check']['interval_error_upper_under_ivx']:.17g} under its documented deterministic arithmetic assumptions.",
f"Canonical executable all-sample max residual: {a['check']['canonical_max_residual']:.17g}.",
f"Native bounds pass; physical margins: {a['check']['margins']}.",
'',
'Compatibility is not a point-accuracy guarantee. Six observation-derived alternatives pass interval compatibility at eta=1e-8: d_W shifts +/-0.0011, gap shifts +/-0.0005, and ng3 shifts +/-0.0000025, with all other coordinates profiled. All six also pass canonical executable residual checks.',
f"Found compatible d_W span: {p['observed_compatible_coordinate_spans'][12]:.17g}; gap span: {p['observed_compatible_coordinate_spans'][13]:.17g}.",
'These are lower witnesses of uncertainty in this actual observation record, not a complete compatible-set enclosure. A single returned representative is not unique generating-hardware recovery.',
'',
f"Domain tests: 200 random full-native roundtrips max error {v['max_native_roundtrip_error']:.3g}; signed +/-18 degree/high-index endpoints explicitly covered.",
f"Derivative audit: best finite-difference step-sweep maximum relative error {j['best_fd_max']:.3g}; independent complex-step normal-equation projection maximum relative error {j['complex_max_relative_error']:.3g}. The initial 2.1e-5 weak-column finite-difference mismatch was cancellation.",
'Frozen generic CLI replay reproduced every returned parameter bit-for-bit.',
'',
'Fresh failure checks, with the identical frozen route and a 120-second cap per case:',
f"random_03: {f3['evaluations']} evaluations, {f3['seconds']:.6f} seconds, max residual {f3['check']['canonical_max_residual']:.17g}; unresolved.",
f"random_07: {f7['evaluations']} evaluations, {f7['seconds']:.6f} seconds, max residual {f7['check']['canonical_max_residual']:.17g}; unresolved.",
'Both terminate locally before the cap; chart changes do not supply global basin selection. These were observation-only runs and no hidden-truth accuracy criterion was applied.',
'',
'Returned parameter vector, in the recorded NAMES order:',
json.dumps(dict(zip(a['names'],a['theta'])),indent=2),
'',
'Input hashes:',json.dumps(b['input_sha256'],indent=2),'Code hashes:',json.dumps(b['code_sha256'],indent=2),
'',
'Files: full_domain_gain.py (generic CLI), result_native.json (first run), result_replay.json (frozen replay), compatible_profiles.json (actual-record alternatives), validation.json, jacobian_sweep.json, fresh_random_03.json, fresh_random_07.json.',
'Reproduce successful run: python -B full_domain_gain.py --seconds 120 --out new_result.json',
'Generic inputs: --candidate PATH --observations PATH --id CASE_ID --threshold 1e-8 --out NEW_OUTPUT.json',
'The threshold is numerical compatibility testing, not an assumed all-coordinate accuracy guarantee.',
'No Dropbox source or existing poc3.py was edited. Only new isolated workspace files were written.'
]
(HERE/'REPORT.txt').write_text('\n'.join(lines)+'\n')
print(json.dumps({'replay_bitwise_identical':True,'profiles':p['accepted_profiles'],'report':'REPORT.txt'}))
