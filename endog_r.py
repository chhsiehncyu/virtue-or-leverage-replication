import sympy as sp, pickle
from model import *

HA = agency(h); LA = agency(l)
out = {}
for name, S, q in [('HA', HA, h), ('LA', LA, l)]:
    Om = sp.together(S['Omega'])
    dOm = sp.together(sp.diff(Om, r))
    num, den = sp.fraction(dOm)
    num = sp.factor(num)
    print(f'--- {name}: dOmega/dr numerator factored ---')
    print(num)
    print('degree in r of each factor:')
    for fac, mult in sp.factor_list(num)[1]:
        print('   deg_r =', sp.degree(fac, r), ' deg_lam =', sp.degree(fac, lam), ' factor:', fac if sp.degree(fac, r) <= 1 else '(long)')
    print('den factored:', sp.factor(den))
    out[name] = dict(num=num, den=den)
    # second derivative sign check later numerically
pickle.dump(out, open('focr.pkl', 'wb'))
