import sympy as sp, numpy as np
from scipy.optimize import brentq, minimize_scalar
from model import *

HM = merchant(h); LM = merchant(l); HA = agency(h); LA = agency(l)
B = dict(BASE)

def f(expr, lv, extra=None):
    d = dict(B)
    if extra: d.update(extra)
    return float(expr.subs(d).subs(lam, lv))

print('=== (1) Win-win threshold, Prop 5(ii) ===')
dpi0 = sp.simplify(sp.diff(HA['piR'], lam).subs(lam, 0))
num, den = sp.fraction(sp.together(dpi0))
print('numerator factored:', sp.factor(num))
print('denominator factored:', sp.factor(den))
cbar = 4*(1-r)/(2*h*(1-r)+(1+r)**2)
print('cbar(baseline h=1.3, r=0.2) =', float(cbar.subs(B)))
print('cbar(h=1.3, r=0.35) =', float(cbar.subs({h: 1.3, r: 0.35})))
print('dpiR^HA/dlam at lam=0, baseline =', f(dpi0, 0))

print('\n=== (2) Control paradox band at baseline, high-end ===')
dOm = HA['Omega'] - HM['Omega']      # >0 : platform prefers agency
dCS = HM['CS'] - HA['CS']            # >0 : consumers prefer merchant
grid = np.linspace(0, 1, 2001)
om = np.array([f(dOm, x) for x in grid]); cs = np.array([f(dCS, x) for x in grid])
def roots(arr):
    out = []
    for i in range(len(arr)-1):
        if arr[i]*arr[i+1] < 0:
            out.append(grid[i] + (grid[i+1]-grid[i])*arr[i]/(arr[i]-arr[i+1]))
    return out
print('Omega^HA - Omega^HM sign changes at lam =', roots(om), ' value at 0:', om[0], ' at 1:', om[-1])
print('CS^HM - CS^HA sign changes at lam =', roots(cs), ' value at 0:', cs[0], ' at 1:', cs[-1])
# economy tier
dOmL = LA['Omega'] - LM['Omega']; dCSL = LM['CS'] - LA['CS']
omL = np.array([f(dOmL, x) for x in grid]); csL = np.array([f(dCSL, x) for x in grid])
print('Economy: Omega^LA - Omega^LM sign changes at', roots(omL), ' at 0:', omL[0], ' at 1:', omL[-1])
print('Economy: CS^LM - CS^LA sign changes at', roots(csL))
# profit comparison pi only
dpi = HA['piR'] - HM['piR']
pr = np.array([f(dpi, x) for x in grid]); print('piR^HA - piR^HM sign changes at', roots(pr), ' at 0:', pr[0], ' at 1:', pr[-1])

print('\n=== (3) Tier switch under agency, Prop 3(ii) ===')
DA = HA['Omega'] - LA['Omega']
da = np.array([f(DA, x) for x in grid]); print('Delta^A(0)=', da[0], ' Delta^A(1)=', da[-1], ' crossing:', roots(da))
# region in q_L at baseline others: find lowest l with sign change
for lv_ in [0.80, 0.82, 0.83, 0.84, 0.85, 0.86, 0.9, 0.95]:
    LA_l = {k_: v_.subs(l, lv_) for k_, v_ in LA.items()}
    DAl = HA['Omega'] - LA_l['Omega']
    vals = np.array([float(DAl.subs({h: 1.3, c: 0.7, r: 0.2}).subs(lam, x)) for x in grid])
    print(f'  q_L={lv_}: Delta^A(0)={vals[0]:.4f}, Delta^A(1)={vals[-1]:.4f}, crossings={roots(vals)}')

print('\n=== (4) Welfare peaks under agency at baseline ===')
for name, S in [('HA', HA), ('LA', LA)]:
    tw = lambda x: -f(S['TW'], x)
    res = minimize_scalar(tw, bounds=(0, 1), method='bounded')
    print(name, 'TW peak at lam =', res.x, ' TW =', -res.fun, ' TW(0)=', f(S['TW'], 0), ' TW(1)=', f(S['TW'], 1))

print('\n=== (5) Global win-win check at (h=1.3,c=0.9,r=0.35) and baseline ===')
for ex in [{c: 0.9, r: 0.35}, {}]:
    vals = [f(HA['piR'], x, ex) for x in [0, 0.25, 0.5, 0.75, 1.0]]
    print(ex, ['%.5f' % v for v in vals])

print('\n=== (6) Endogenous commission at baseline (OTA picks r to max Omega) ===')
def rstar(S_builder, lv, cval=0.7, hv=1.3, lvq=0.9):
    S = S_builder
    Om = S['Omega'].subs({h: hv, l: lvq, c: cval}).subs(lam, lv)
    g = sp.lambdify(r, Om, 'numpy')
    res = minimize_scalar(lambda x: -g(x), bounds=(0.001, 0.999), method='bounded')
    return res.x
for name, S in [('HA', HA), ('LA', LA)]:
    for lv in [0, 0.5, 0.75, 0.95, 1.0]:
        rs = rstar(S, lv)
        d = {h: 1.3, l: 0.9, c: 0.7, r: rs, lam: lv}
        print(f'{name} lam={lv}: r*={rs:.4f} piR={float(S["piR"].subs(d)):.4f} piI={float(S["piI"].subs(d)):.4f} CS={float(S["CS"].subs(d)):.4f} TW={float(S["TW"].subs(d)):.4f} Omega={float(S["Omega"].subs(d)):.4f} DI={float(S["DI"].subs(d)):.4f} margI={float(((1-r)*S["pI"]-c/2).subs(d)):.4f}')
    print()
print('Merchant comparison values:')
for name, S in [('HM', HM), ('LM', LM)]:
    for lv in [0, 0.5, 0.75, 0.95, 1.0]:
        d = dict(B); d[lam] = lv
        print(f'{name} lam={lv}: piR={float(S["piR"].subs(d)):.4f} piI={float(S["piI"].subs(d)):.4f} CS={float(S["CS"].subs(d)):.4f} Omega={float(S["Omega"].subs(d)):.4f}')
