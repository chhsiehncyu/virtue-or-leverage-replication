"""Section 8.2: continuous quality of the integrated hotel under the flexible commission (c = 0.75)."""
import numpy as np, sympy as sp, pickle
from model import h, l, c, r, lam, agency, merchant
Q = pickle.load(open('quartic.pkl', 'rb'))['quartic']
F = {nm: {k: sp.lambdify((h, l, c, r, lam), S[k], 'numpy') for k in ['Omega', 'DT', 'DI']} for nm, S in [('HA', agency(h)), ('LA', agency(l))]}
coef = {nm: [sp.lambdify((h, l, c, lam), cf, 'numpy') for cf in sp.Poly(Q[nm], r).all_coeffs()] for nm in Q}
def best(nm, hv, lv, cv, la):
    roots = np.roots([float(cf(hv, lv, cv, la)) for cf in coef[nm]])
    cand = [x.real for x in roots if abs(x.imag) < 1e-9 and 0 < x.real < 1
            and F[nm]['DI'](hv, lv, cv, x.real, la) > 0 and F[nm]['DT'](hv, lv, cv, x.real, la) > 0]
    if not cand: return -np.inf, np.nan
    rs = max(cand, key=lambda x: F[nm]['Omega'](hv, lv, cv, x, la))
    return F[nm]['Omega'](hv, lv, cv, rs, la), rs
cv = 0.75
for la in [0.0, 0.5, 1.0]:
    lo = [(q, *best('LA', 1.3, q, cv, la)) for q in np.linspace(0.30, 0.999, 700)]
    hi = [(q, *best('HA', q, 0.9, cv, la)) for q in np.linspace(1.001, 2 / cv - 1 - 0.001, 700)]
    allv = lo + hi
    q, om, rs = max(allv, key=lambda t: t[1])
    print(f'flexible, lam={la}: q* = {q:.3f}, r* = {rs:.3f}, Omega = {om:.4f}')
