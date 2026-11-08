"""Exact certificate that the platform-chosen commission r*(lambda) strictly decreases in lambda (Lemma 3).

Method. Write the platform objective along the agency pricing equilibrium as a function of the incumbent's
effective unit cost kappa = c/(2(1-r)). Strict concavity in kappa is shown in the proof of Lemma 3. Here we show
d2 Omega / d kappa d lambda < 0 on the whole admissible domain, so d kappa*/d lambda < 0 and r* = 1 - c/(2 kappa*)
strictly decreases.

Substitutions (all variables in the open unit interval except eta > 0):
  high-end: h = 1 + eta, lambda = 1 - sigma, c = 2 s/(1 + h)  [c < 2/(1+h)],
            kappa = c/2 + u (kbar_H - c/2), kbar_H = [c(h+1) + 2(1-lambda)] / [2(2-lambda)]
  economy : l = 1 - nu,  lambda = 1 - sigma, c = 2 s/(1 + l)  [c < 2/(1+l)],
            kappa = c/2 + u (kbar_L - c/2), kbar_L = 1 - c l / [2(2-lambda)]
The numerator of the cross-partial is written in the tensor Bernstein basis on the unit cube and the sign of every
Bernstein coefficient is checked in exact rational arithmetic."""
import sympy as sp, itertools, random
from math import comb
from model import h, l, c, r, lam, agency

kappa = sp.symbols('kappa', positive=True)
u, s, sigma, eta, nu = sp.symbols('u s sigma eta nu', positive=True)

def bernstein(poly_expr, vars_):
    P = sp.Poly(sp.expand(poly_expr), *vars_)
    n = [P.degree(v) for v in vars_]
    a = {k: v for k, v in P.terms()}
    B = {}
    for idx in itertools.product(*[range(d + 1) for d in n]):
        acc = sp.Integer(0)
        for jdx in itertools.product(*[range(i + 1) for i in idx]):
            if jdx in a:
                w = sp.Integer(1)
                for i, j, d in zip(idx, jdx, n):
                    w *= sp.Rational(comb(i, j), comb(d, j))
                acc += w * a[jdx]
        B[idx] = sp.expand(acc)
    return n, B

def write_certificate(fname, header, F, n, B):
    """Write the polynomial and every Bernstein coefficient in exact arithmetic, so the certificate can be checked
    without rerunning the symbolic computation."""
    with open(fname, 'w') as f:
        for line in header:
            f.write('# ' + line + '\n')
        f.write('# degrees ' + ' '.join(str(d) for d in n) + '\n')
        f.write('F = ' + str(sp.expand(F)) + '\n')
        for idx in sorted(B):
            f.write(' '.join(str(i) for i in idx) + '\t' + str(B[idx]) + '\n')
    print('wrote', fname, 'with', len(B), 'coefficients')


def cross_partial(tier):
    S = agency(h if tier == 'H' else l)
    Om = sp.together(S['Omega'].subs(r, 1 - c / (2 * kappa)))
    return sp.together(sp.diff(Om, kappa, lam))

def normalize(const, facs, gens):
    """sympy's sign convention for factors depends on the symbol names; put a negative constant into the main factor."""
    if const < 0:
        assert all(m == 1 for f, m in facs if sp.Poly(f, *gens).total_degree() >= 3)
        const = -const
        facs = [(-f, m) if sp.Poly(f, *gens).total_degree() >= 3 else (f, m) for f, m in facs]
    return const, facs


report = {}
# ---------- high-end ----------
X = cross_partial('H')
num, den = sp.fraction(sp.factor(X))
print('H: denominator of the cross-partial =', sp.factor(den))
kbar = (c * (h + 1) + 2 * (1 - lam)) / (2 * (2 - lam))
sub = {kappa: c / 2 + u * (kbar - c / 2)}
N = sp.together(num.subs(sub))
N = sp.together(N.subs({c: 2 * s / (1 + h)}))
N = sp.together(N.subs({h: 1 + eta, lam: 1 - sigma}))
nN, dN = sp.fraction(sp.factor(N))
const, facs = sp.factor_list(nN)
const, facs = normalize(const, facs, (u, s, sigma, eta))
print('H: substituted numerator = const', const, 'x factors', [f if sp.Poly(f, u, s, sigma, eta).total_degree() < 3 else 'F_H' for f, m in facs], '/ (', dN, ')')
F_H = [f for f, m in facs if sp.Poly(f, u, s, sigma, eta).total_degree() >= 3][0]
n, B = bernstein(F_H, (u, s, sigma))
write_certificate('bernstein_certificate_H.txt', [
    'Lemma 3, decline of the optimal commission, high-end integration.',
    'Numerator of the cross-partial of the objective in kappa and lambda = 16 (eta + 1) F / [(eta + 2)^5 (sigma + 1)],',
    'with h = 1 + eta, lambda = 1 - sigma, c = 2 s / (1 + h), kappa = c/2 + u (kbar_H - c/2).',
    'Each line after F gives i j k and the coefficient of the Bernstein basis polynomial',
    'C(5,i) u^i (1-u)^(5-i) * C(5,j) s^j (1-s)^(5-j) * C(5,k) sigma^k (1-sigma)^(5-k) in the expansion of F.',
    'Every coefficient is zero or a polynomial in eta whose coefficients are all negative, so F < 0 on the open unit cube for eta > 0.'],
    F_H, n, B)
signs = {'zero': 0, 'neg': 0, 'pos_or_mixed': 0}
for k, v in B.items():
    cs = sp.Poly(v, eta).coeffs() if v != 0 else []
    if v == 0: signs['zero'] += 1
    elif all(x < 0 for x in cs): signs['neg'] += 1
    else: signs['pos_or_mixed'] += 1
print('H: Bernstein degrees (u,s,sigma) =', n, ' coefficient classes:', signs)
report['H'] = signs
# ---------- economy ----------
X = cross_partial('L')
num, den = sp.fraction(sp.factor(X))
print('L: denominator of the cross-partial =', sp.factor(den))
kbar = 1 - c * l / (2 * (2 - lam))
sub = {kappa: c / 2 + u * (kbar - c / 2)}
N = sp.together(num.subs(sub))
N = sp.together(N.subs({c: 2 * s / (1 + l)}))
N = sp.together(N.subs({l: 1 - nu, lam: 1 - sigma}))
nN, dN = sp.fraction(sp.factor(N))
const, facs = sp.factor_list(nN)
const, facs = normalize(const, facs, (u, s, sigma, nu))
print('L: substituted numerator = const', const, 'x factors', [f if sp.Poly(f, u, s, sigma, nu).total_degree() < 3 else 'F_L' for f, m in facs], '/ (', sp.factor(dN), ')')
F_L = [f for f, m in facs if sp.Poly(f, u, s, sigma, nu).total_degree() >= 3][0]
n, B = bernstein(F_L, (u, s, sigma, nu))
write_certificate('bernstein_certificate_L.txt', [
    'Lemma 3, decline of the optimal commission, economy integration.',
    'Numerator of the cross-partial of the objective in kappa and lambda = 16 F / [(sigma + 1) (nu - 2)^5],',
    'with l = 1 - nu, lambda = 1 - sigma, c = 2 s / (1 + l), kappa = c/2 + u (kbar_L - c/2).',
    'Each line after F gives i j k m and the coefficient of the Bernstein basis polynomial',
    'C(5,i) u^i (1-u)^(5-i) * C(5,j) s^j (1-s)^(5-j) * C(5,k) sigma^k (1-sigma)^(5-k) * C(8,m) nu^m (1-nu)^(8-m) in the expansion of F.',
    'Every coefficient is a nonnegative rational number and not all are zero, so F > 0 on the open unit hypercube; since (nu - 2)^5 < 0 the numerator is negative.'],
    F_L, n, B)
vals = list(B.values())
print('L: Bernstein degrees (u,s,sigma,nu) =', n, ' positive', sum(1 for v in vals if v > 0), ' zero', sum(1 for v in vals if v == 0), ' negative', sum(1 for v in vals if v < 0))
# ---------- numerical spot check of the sign of the cross-partial ----------
random.seed(1)
for tier in ['H', 'L']:
    f = sp.lambdify((h, l, c, kappa, lam), cross_partial(tier), 'math')
    worst = -1e9
    for _ in range(2000):
        hv = 1 + random.uniform(0.01, 2); lv = random.uniform(0.05, 0.99); la = random.uniform(0.01, 0.99)
        cb = 2 / (1 + hv) if tier == 'H' else 2 / (1 + lv)
        cv = random.uniform(0.01, 0.99) * cb
        kb = (cv * (hv + 1) + 2 * (1 - la)) / (2 * (2 - la)) if tier == 'H' else 1 - cv * lv / (2 * (2 - la))
        kv = cv / 2 + random.uniform(0.01, 0.99) * (kb - cv / 2)
        worst = max(worst, f(hv, lv, cv, kv, la))
    print(tier, ': largest sampled value of the cross-partial (should be negative):', worst)
