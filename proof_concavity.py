"""Lemma 3, uniqueness: the platform objective is strictly concave in kappa = c/(2(1-r)).
Writes the second derivative in t = kappa - c/2, sigma = 1 - lambda, eta = h - 1 (nu = 1 - l) and checks coefficient signs."""
import sympy as sp
from model import h, l, c, r, lam, agency
kappa = sp.symbols('kappa', positive=True)
t, sigma, eta, nu = sp.symbols('t sigma eta nu', nonnegative=True)
for tier in ['H', 'L']:
    S = agency(h if tier == 'H' else l)
    Om = sp.together(S['Omega'].subs(r, 1 - c / (2 * kappa)))
    d2 = sp.factor(sp.together(sp.diff(Om, kappa, 2)))
    num, den = sp.fraction(d2)
    print(f'--- {tier}: denominator of d2 Omega/d kappa2 =', sp.factor(den))
    sub = {kappa: t + c / 2, lam: 1 - sigma}
    sub.update({h: 1 + eta} if tier == 'H' else {l: 1 - nu})
    N = sp.expand(num.subs(sub))
    const, facs = sp.factor_list(N)
    print(f'{tier}: numerator = {const} x', [str(f)[:30] if sp.Poly(f, t, c).total_degree() < 2 else 'G' for f, m in facs])
    G = [f for f, m in facs if sp.Poly(f, t, c).total_degree() >= 2][0]
    P = sp.Poly(G, t, c)
    other = (eta, sigma) if tier == 'H' else (sigma, nu)
    for (i, j), cf in sorted(P.terms(), reverse=True):
        cp = sp.Poly(sp.expand(cf), *other)
        signs = set(int(sp.sign(x)) for x in cp.coeffs())
        print(f'   t^{i} c^{j}: coefficient signs in {other}: {signs}')
