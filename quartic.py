"""Extract the quartic first-order condition of the platform-chosen commission (Section 5.1) from focr.pkl; writes quartic.pkl."""
import sympy as sp, pickle
from model import r
foc = pickle.load(open('focr.pkl', 'rb'))
quartic = {}
for name in ['HA', 'LA']:
    facs = sp.factor_list(foc[name]['num'])[1]
    quartic[name] = [f_ for f_, m in facs if sp.degree(f_, r) == 4][0]
    print(name, 'quartic in r of degree', sp.degree(quartic[name], r))
pickle.dump({'quartic': quartic}, open('quartic.pkl', 'wb'))
