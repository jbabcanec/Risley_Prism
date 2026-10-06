"""Export compact reusable witnesses, complete tables, and validation evidence."""
import os,sys,json,math,hashlib
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
from pathlib import Path
from fractions import Fraction
import numpy as np
from profile_pairs import NAMES
ROOT=Path(r'C:\Users\josep\Dropbox\Babcanec Works\Mathematics\Wedge')
sys.path.insert(0,str(ROOT))
from risley_lattice.model import vec2pat

def exact_difference_upper(xs,ys):
    value=max(abs(Fraction.from_float(float(x))-Fraction.from_float(float(y))) for x,y in zip(xs,ys))
    return {'exact_fraction':str(value),'outward_binary64':math.nextafter(float(value),math.inf)}

low60=json.loads(Path('benchmark_verified_60.json').read_text())['records'][1]
low100=json.loads(Path('benchmark_verified_100.json').read_text())['records'][0]
ordinary=json.loads(Path('ordinary_noise_verified.json').read_text())['records'][0]
chosen=[('eta_1e-8_separated_speeds',1e-8,low100),('eta_1e-4_large_wedges',1e-4,ordinary)]
items=[]
for label,eta,r in chosen:
    a=vec2pat(np.array(r['base'])).reshape(-1).tolist()
    b=vec2pat(np.array(r['alternative'])).reshape(-1).tolist()
    y=r['common_observations']
    ca=exact_difference_upper(a,y);cb=exact_difference_upper(b,y)
    max_pair=exact_difference_upper(a,b)
    item={'id':label,'noise_allowance':eta,'parameter_names':NAMES,'strict_interval_witness':r,
          'canonical_float_outputs':{'base':a,'alternative':b},
          'exact_stored_canonical_errors_to_same_record':{'base':ca,'alternative':cb},
          'exact_stored_canonical_pair_difference':max_pair,
          'strict_record_admissible_for_both':r['common_observation_error_upper']<=eta,
          'canonical_float_record_admissible_for_both':max(ca['outward_binary64'],cb['outward_binary64'])<=eta}
    assert item['strict_record_admissible_for_both']
    assert item['canonical_float_record_admissible_for_both']
    assert r['minimax_coordinate_lower_bound'][12]>.001
    items.append(item)
comparison={'same_endpoints':low60['base']==low100['base'] and low60['alternative']==low100['alternative'],
            'same_shared_record':low60['common_observations']==low100['common_observations'],
            'same_outward_eta':low60['common_observation_error_upper']==low100['common_observation_error_upper']}
assert all(comparison.values())
summary={'protocol':'original ordered full18 native box, 200 passive x/y positions, mathematical times k/20',
         'conclusion':'No estimator uniformly achieves native .001 at eta1e-8 over the original full native box: explicit shared-record witness.',
         'proof':'For a common admissible y, triangle inequality gives max(|estimate_j-a_j|,|estimate_j-b_j|)>=|a_j-b_j|/2. The adversarial truth may depend on j.',
         'scope':'These are constructed shared records, not the frozen random development records; truths are used to construct impossibility witnesses, not as solver initialization.',
         'rigor':'Independent directed arbitrary-precision interval optical calculations with exact rational times and parameters; not Lean and not a formal proof of mpmath itself. Canonical executable outputs are also compared as exact stored binary64 rationals.',
         'precision_repeat':comparison,'witnesses':items}
Path('witnesses.json').write_text(json.dumps(summary,indent=2))
lines=[summary['conclusion'],'',summary['proof'],'',summary['scope'],'',summary['rigor'],'']
for item in items:
    r=item['strict_interval_witness'];lines.extend([item['id'],f"Noise allowance: {item['noise_allowance']:.17g}",f"Strict required error upper bound: {r['common_observation_error_upper']:.17g}",f"Strict per-axis pair-difference upper bounds: {r['xy_difference_upper']}",f"Canonical exact stored errors: {item['exact_stored_canonical_errors_to_same_record']}",f"Physical margins A: {r['guards_base']}",f"Physical margins B: {r['guards_alternative']}",'coordinate | base | alternative | difference | unavoidable-error lower bound'])
    for j,name in enumerate(NAMES):
        lines.append(f"{name} | {r['base'][j]:.17g} | {r['alternative'][j]:.17g} | {r['parameter_difference'][j]:.17g} | {r['minimax_coordinate_lower_bound'][j]:.17g}")
    lines.append('')
lines.extend(['Precision repeat (60 and 100 decimal digits): '+str(comparison),'',
              'Additional ordinary family: verified_pairs.json covers wedge scales 2,3,4,6,8,12 with d_W difference .0022.',
              'Important preserved diagnostic: repository_parity.json record3 shows up to 1.67575e-7 difference between core floating arccos implementation and strict algebra. That exploratory record is NOT used in witnesses.json.',
              'Reproduce: python -B profile_pairs.py; python -B benchmark_probe.py; python -B verify_pairs.py --input benchmark_candidates.json --out benchmark_verified_60.json --indices 0,1,2,3 --dps 60;',
              'python -B verify_pairs.py --input benchmark_candidates.json --out benchmark_verified_100.json --indices 1 --dps 100; python -B ordinary_noise_probe.py;',
              'python -B verify_pairs.py --input ordinary_noise_candidate.json --out ordinary_noise_verified.json --indices 0 --dps 80; python -B summarize_witnesses.py',
              'Original repository sources were read/imported only, with bytecode disabled; all outputs live in this workspace directory.'])
Path('SUMMARY.txt').write_text('\n'.join(lines)+'\n')
print(json.dumps({'witnesses':[{'id':x['id'],'eta':x['noise_allowance'],'strict_eta_bound':x['strict_interval_witness']['common_observation_error_upper'],'dw_floor':x['strict_interval_witness']['minimax_coordinate_lower_bound'][12],'core_also_passes':x['canonical_float_record_admissible_for_both']} for x in items],'precision_repeat':comparison},indent=2))
