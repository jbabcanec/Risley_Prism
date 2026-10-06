"""Derive the P3 candidate implementation from the read-only P2 baseline.

All replacements are asserted; the generated source is an independent local
file. No source repository file is changed.
"""
from pathlib import Path

ROOT=Path(r'C:\Users\josep\Dropbox\Babcanec Works\Mathematics\Wedge')
text=(ROOT/'risley_lattice/poc2.py').read_text(encoding='utf-8')
def change(old,new):
    global text
    if old not in text: raise RuntimeError('Missing source fragment: '+old)
    text=text.replace(old,new)

change('Experimental blind two-prism recovery','Experimental blind three-prism recovery')
change('both physical orders','all six physical orders')
change('solve14','solve18')
change('from .model import pat_P','from risley_lattice.model import pat_P')
change('from .spectral import extract_speeds','from risley_lattice.spectral import extract_speeds')
change('from .lattice import kset','from risley_lattice.lattice import kset')
start=text.index('NAMES = ');end=text.index('\n\n\ndef canonical',start)
text=text[:start]+'''from risley_lattice.model import NAMES, LO, HI
LINEAR = np.array([12,13,16,17])
NONLINEAR = np.array([i for i in range(18) if i not in LINEAR])
SPAN = HI-LO
'''+text[end:]
change('v[:2], v[2:4], v[4:6], v[6:8], v[8:]','v[:3], v[3:6], v[6:9], v[9:12], v[12:]')
change('v[:, :2, None]','v[:, :3, None]')
change('v[:, 4:6, None]','v[:, 6:9, None]')
change('v[:, 2:4, None]','v[:, 3:6, None]')
change('v[:, 10+axis, None]','v[:, 14+axis, None]')
change('for i in range(2):','for i in range(3):')
change('v[:, 6+i, None]','v[:, 9+i, None]')
change('1 if i == 0 else 0','1 if i < 2 else 0')
change('(len(NONLINEAR), 14)','(len(NONLINEAR), 18)')
change('kset(2, 3)','kset(3, 3)')
change('np.eye(2)','np.eye(3)')
change('permutations(range(2))','permutations(range(3))')
change('v[:2] = speeds[order]','v[:3] = speeds[order]')
change('v[4:6] = phases[order]','v[6:9] = phases[order]')
change('v[6:8] = glass','v[9:12] = glass')
change('v[8:10] = distance, 8.5','v[12:14] = distance, 8.5')
change('np.array([distance+8.5+3/glass, distance])','np.array([distance+17+6/glass, distance+8.5+3/glass, distance])')
change('v[2:4] = signs[order]','v[3:6] = signs[order]')
change('6+distance+8.5+6/glass','6+distance+17+9/glass')
change('v[10:12] = np.clip','v[14:16] = np.clip')
change('v[12:14] = 0','v[16:18] = 0')
change('np.full(14, .05)','np.full(18, .05)')
change('radius[:2] = .02/(count*dt*SPAN[:2])','radius[:3] = .02/(count*dt*SPAN[:3])')
change('(14, 14)','(18, 18)')
change('np.arange(14)','np.arange(18)')
change('np.zeros(15)','np.zeros(19)')
change('lp.x[:14]','lp.x[:18]')
change('n_gen=2','n_gen=3')
change('did not find two rotors','did not find three rotors')
change('candidate.shape == (2,)','candidate.shape == (3,)')
change("    from .poc2_verify import verify_candidate\n    result['compatibility_verification'] = verify_candidate(\n        best, y, eta if eta > 0 else numerical_tolerance)","    result['compatibility_verification'] = {'verified': False, 'reason': 'candidate solver only; run full18 interval verification separately'}")
change('max_seconds: float = 45.','max_seconds: float = 45.')
change('screen_evaluations: int = 18','screen_evaluations: int = 24')
change('finalists: int = 4','finalists: int = 6')
change('ftol=1e-13,\n                                xtol=1e-13, gtol=1e-13','ftol=1e-14,\n                                xtol=1e-14, gtol=None')
# The underlying P2 code already derived the affine recurrence. The P3
# transfer adds the shared gap after BOTH upstream prisms.
header='''# Adapted 2026-10-02 from risley_lattice/poc2.py (2026-09-29).
# Main changes: P=3/full18, shared two-gap propagation, all six orders,
# 14 nonlinear variables with four bounded affine geometry variables.
# This bounded numerical candidate procedure is NOT globally complete.
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
sys.path.insert(0,r'C:\\Users\\josep\\Dropbox\\Babcanec Works\\Mathematics\\Wedge')

'''
(Path(__file__).parent/'poc3.py').write_text(header+text,encoding='utf-8')
print('Wrote local full18 solver')
