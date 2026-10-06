"""Small algebra/Jacobian checks only; no optical solves or truth files."""
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[key]='1'
import json
from math import comb,exp
from pathlib import Path
import numpy as np
from harmonic_dictionary import (dictionary,indices,variable_projection,chebyshev_coefficient,
                                 fixed_rotor_jet_readout,jet_weights)

times=np.arange(200)/20
rotor=np.array([2.71,-1.93,.83,7.,-11.,17.])
rng=np.random.default_rng(902101)
report={'scope':'Algebra and finite-difference checks; no physical recovery or tail certificate.'}
checks=[]
for degree in (3,5):
    y=np.column_stack([dictionary(rotor,times,degree,axis)@rng.normal(size=len(indices(degree)))
                      for axis in (0,1)])+.07*rng.normal(size=(200,2))
    answer=variable_projection(rotor,times,y,degree)
    errors=[]
    for j in range(6):
        step=1e-7 if j<3 else 1e-5
        shift=np.zeros(6);shift[j]=step
        difference=(variable_projection(rotor+shift,times,y,degree)['residual']-
                    variable_projection(rotor-shift,times,y,degree)['residual'])/(2*step)
        errors.append(float(np.linalg.norm(difference-answer['jacobian'][:,j])/
                            max(1,np.linalg.norm(difference))))
    checks.append(dict(degree=degree,columns=len(indices(degree)),jacobian_errors=errors,
                       diagnostics=answer['diagnostics']))
    assert max(errors)<2e-6

# Independent numpy polynomial representation checks all coefficients through n=12.
for n in range(13):
    expected=np.polynomial.chebyshev.cheb2poly(np.eye(13)[n])
    for r in range(13):
        assert chebyshev_coefficient(n,r)==(expected[r] if r<len(expected) else 0)

# The unconstrained surrogate cannot distinguish individual rotor reversal:
# x columns are unchanged, y columns get (-1)^h_j factors.
h=indices(5)
reversal=[]
for j in range(3):
    changed=rotor.copy();changed[j]*=-1;changed[j+3]*=-1
    ex=np.max(abs(dictionary(changed,times,5,0)-dictionary(rotor,times,5,0)))
    ey=np.max(abs(dictionary(changed,times,5,1)-dictionary(rotor,times,5,1)*(-1.)**h[None,:,j]))
    reversal.append([float(ex),float(ey)])
    assert max(ex,ey)<2e-12

report.update(jacobian_checks=checks,exact_chebyshev_coefficients_checked_through=12,
              rotor_reversal_dictionary_errors=reversal,
              columns={str(d):len(indices(d)) for d in (3,4,5,6,7,8,9)},
              passed=True)

# At zero rotor speeds and phases every column on x is constant: exact rank
# deficiency. Choose bounded coefficients attaining the nullspace-row charge.
flat=np.zeros(6)
phi=dictionary(flat,times,3,0)
empty=np.zeros((len(times),2))
bounded=fixed_rotor_jet_readout(flat,times,empty,(1,0,0),3,0,
                               coefficient_bounds=np.ones(len(indices(3))))
coefficients=np.sign(bounded['nullspace_row'])
observations=np.column_stack([phi@coefficients,np.zeros(len(times))])
checked=fixed_rotor_jet_readout(flat,times,observations,(1,0,0),3,0,
                               coefficient_bounds=np.ones(len(indices(3))))
error=abs(checked['value']-jet_weights((1,0,0),3)@coefficients)
assert checked['nullspace_charge']>1
assert abs(error-checked['nullspace_charge'])<1e-12
report['rank_deficient_jet_check']=dict(rank=int(np.linalg.matrix_rank(phi)),
    columns=phi.shape[1],observed_jet_error=float(error),
    coefficient_bound=1.,nullspace_charge=checked['nullspace_charge'])

# Illustrations of the shell majorant only; these strip widths are not
# established for the native optical family and these are not interval proofs.
def normalized_tail(degree,sigma):
    x=exp(-sigma)
    k=degree+1
    return x**k*((4*k*k+2)/(1-x)+8*k*x/(1-x)**2+4*x*(1+x)/(1-x)**3)
tails=[]
for sigma in (.5,1.):
    degree=next(h for h in range(1000) if normalized_tail(h,sigma)<=1e-8)
    tails.append(dict(assumed_strip_width=sigma,target_eta_over_M=1e-8,
        first_degree_meeting_function_tail=degree,columns=comb(degree+3,3),
        normalized_tail=normalized_tail(degree,sigma),
        previous_normalized_tail=normalized_tail(degree-1,sigma),
        degree5_normalized_tail=normalized_tail(5,sigma),
        degree8_normalized_tail=normalized_tail(8,sigma)))
report['illustrative_tail_budget']=tails
output=Path(__file__).with_name('harmonic_dictionary_checks.json')
output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
