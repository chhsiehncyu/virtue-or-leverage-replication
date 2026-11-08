"""Merchant equilibrium under non-uniform densities, solved via polynomial first-order conditions."""
import sympy as sp, numpy as np
from scipy.optimize import fsolve, minimize_scalar

th, pT, pI, w, lam, cval = sp.symbols('theta p_T p_I w lambda c', real=True)

def build(qT, dens):
    f = {'uniform': 1, 'inc': 2 * th, 'dec': 2 * (1 - th)}[dens]
    if qT > 1:
        thTI = (pT - pI) / (qT - 1); thI = pI
        DT = sp.integrate(f, (th, thTI, 1)); DI = sp.integrate(f, (th, thI, thTI))
        CS = sp.integrate((th * qT - pT) * f, (th, thTI, 1)) + sp.integrate((th - pI) * f, (th, thI, thTI))
        region = lambda a, b: (0 < b < (a - b) / (qT - 1) < 1)
    else:
        thIT = (pI - pT) / (1 - qT); thT = pT / qT
        DI = sp.integrate(f, (th, thIT, 1)); DT = sp.integrate(f, (th, thT, thIT))
        CS = sp.integrate((th - pI) * f, (th, thIT, 1)) + sp.integrate((th * qT - pT) * f, (th, thT, thIT))
        region = lambda a, b: (0 < a / qT < (b - a) / (1 - qT) < 1)
    cost = cval * qT**2 / 2
    Om = (pI - w) * DI + (pT - cost) * DT + lam * CS
    foc = [sp.diff(Om, pT), sp.diff(Om, pI)]
    hess = sp.hessian(Om, (pT, pI))
    fn = sp.lambdify((pT, pI, w, lam, cval), foc, 'numpy')
    hs = sp.lambdify((pT, pI, w, lam, cval), hess, 'numpy')
    objs = sp.lambdify((pT, pI, w, lam, cval), [DT, DI, CS, (pI - w) * DI + (pT - cost) * DT, (w - cval / 2) * DI], 'numpy')
    return fn, hs, objs, region

def solve_ota(fn, hs, region, wv, lv, cv, guess):
    sol, info, ier, msg = fsolve(lambda x: fn(x[0], x[1], wv, lv, cv), guess, full_output=True)
    H = np.array(hs(sol[0], sol[1], wv, lv, cv), dtype=float)
    ok = ier == 1 and region(sol[0], sol[1]) and np.all(np.linalg.eigvals(H) < 0)
    return sol, ok

def merchant(qT, dens, lv, cv=0.7):
    fn, hs, objs, region = build(qT, dens)
    guess = np.array([0.95, 0.70]) if qT > 1 else np.array([0.59, 0.68])
    def negpiI(wv):
        sol, ok = solve_ota(fn, hs, region, wv, lv, cv, guess)
        if not ok: return 1.0
        return -objs(sol[0], sol[1], wv, lv, cv)[4]
    # coarse grid then refine
    wg = np.linspace(cv / 2 + 0.005, 0.85, 170)
    vals = np.array([negpiI(x) for x in wg]); i = np.argmin(vals)
    res = minimize_scalar(negpiI, bounds=(wg[max(i - 1, 0)], wg[min(i + 1, len(wg) - 1)]), method='bounded', options=dict(xatol=1e-8))
    wv = res.x; sol, ok = solve_ota(fn, hs, region, wv, lv, cv, guess)
    DT, DI, CS, piR, piI = objs(sol[0], sol[1], wv, lv, cv)
    return dict(w=wv, pT=sol[0], pI=sol[1], DT=DT, DI=DI, CS=CS, piR=piR, piI=piI, Om=piR + lv * CS, ok=ok)

for dens in ['uniform', 'inc', 'dec']:
    print(f'=== density {dens} ===')
    E = {}
    for qT, name in [(1.3, 'H'), (0.9, 'L')]:
        for lv in [0, 0.5, 1.0]:
            e = merchant(qT, dens, lv); E[(name, lv)] = e
            print(f'{name} lam={lv}: ok={e["ok"]} w={e["w"]:.4f} pT={e["pT"]:.4f} pI={e["pI"]:.4f} DT={e["DT"]:.4f} DI={e["DI"]:.4f} DT/DI={e["DT"]/e["DI"]:.4f} piR={e["piR"]:.4f} piI={e["piI"]:.4f} CS={e["CS"]:.4f} Omega={e["Om"]:.4f}')
    for lv in [0, 0.5, 1.0]:
        print(f'   Omega^H - Omega^L at lam={lv}: {E[("H", lv)]["Om"] - E[("L", lv)]["Om"]:.5f};  CS/piR at lam=0: H {E[("H",0)]["CS"]/E[("H",0)]["piR"]:.3f}, L {E[("L",0)]["CS"]/E[("L",0)]["piR"]:.3f}')
