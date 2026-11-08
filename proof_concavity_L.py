import sympy as sp, numpy as np
from model import h, l, c, r, lam, agency
kappa = sp.symbols('kappa', positive=True); t, sigma, nu = sp.symbols('t sigma nu', nonnegative=True)
S = agency(l)
Om = sp.together(S['Omega'].subs(r, 1 - c / (2 * kappa)))
num, den = sp.fraction(sp.factor(sp.together(sp.diff(Om, kappa, 2))))
N = sp.expand(num.subs({kappa: t + c / 2, lam: 1 - sigma, l: 1 - nu}))
const, facs = sp.factor_list(N)
G = [f for f, m in facs if sp.Poly(f, t, c).total_degree() >= 2][0]
P = sp.Poly(G, t, c)
f = sp.lambdify((sigma, nu), 0, 'numpy')
for (i, j), cf in sorted(P.terms(), reverse=True):
    fc = sp.factor(cf)
    g = sp.lambdify((sigma, nu), cf, 'numpy')
    M, V = np.meshgrid(np.linspace(0, 1, 401), np.linspace(1e-4, 1 - 1e-4, 401))
    vals = g(M, V)
    print(f't^{i} c^{j}: factored = {fc}\n      min on grid = {np.min(vals):.4g}')
