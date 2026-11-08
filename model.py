"""
Dual-purpose OTA model. Symbolic re-derivation of the four
scenarios (H/L x M/A) and helper functions. All in original parameters h, l, c, r, lam.
"""
import sympy as sp

h, l, c, r, lam, mu = sp.symbols('h l c r lambda mu', positive=True)
pT, pI, w = sp.symbols('p_T p_I w', real=True)
th = sp.symbols('theta', real=True)


def demand_cs(qT, pT, pI):
    """Return (D_T, D_I, CS) for integrated quality qT vs incumbent quality 1."""
    if qT == h:  # high-end
        thTI = (pT - pI) / (h - 1)
        DT = 1 - thTI
        DI = thTI - pI
        CS = sp.integrate(th * h - pT, (th, thTI, 1)) + sp.integrate(th - pI, (th, pI, thTI))
    else:  # economy, qT = l
        thIT = (pI - pT) / (1 - l)
        thT = pT / l
        DI = 1 - thIT
        DT = thIT - thT
        CS = sp.integrate(th - pI, (th, thIT, 1)) + sp.integrate(th * l - pT, (th, thT, thIT))
    return sp.simplify(DT), sp.simplify(DI), sp.simplify(CS)


def merchant(qT):
    """Merchant equilibrium: incumbent sets w, OTA sets pT, pI. Returns dict of closed forms."""
    DT, DI, CS = demand_cs(qT, pT, pI)
    cost_T = c * qT**2 / 2
    Omega = (pI - w) * DI + (pT - cost_T) * DT + lam * CS
    sol = sp.solve([sp.diff(Omega, pT), sp.diff(Omega, pI)], [pT, pI], dict=True)[0]
    piI = (w - c / 2) * DI.subs(sol)
    wstar = sp.solve(sp.diff(piI, w), w)[0]
    pTs = sp.simplify(sol[pT].subs(w, wstar))
    pIs = sp.simplify(sol[pI].subs(w, wstar))
    out = eq_objects(qT, pTs, pIs, fmt='M', w_=wstar)
    out['w'] = sp.simplify(wstar)
    return out


def agency(qT, r_=r):
    """Agency equilibrium with commission r_: incumbent sets pI, OTA sets pT simultaneously."""
    DT, DI, CS = demand_cs(qT, pT, pI)
    cost_T = c * qT**2 / 2
    piI = ((1 - r_) * pI - c / 2) * DI
    Omega = r_ * pI * DI + (pT - cost_T) * DT + lam * CS
    sol = sp.solve([sp.diff(Omega, pT), sp.diff(piI, pI)], [pT, pI], dict=True)[0]
    pTs = sp.simplify(sol[pT])
    pIs = sp.simplify(sol[pI])
    return eq_objects(qT, pTs, pIs, fmt='A', r_=r_)


def eq_objects(qT, pTs, pIs, fmt, w_=None, r_=None):
    DT, DI, CS = demand_cs(qT, pTs, pIs)
    cost_T = c * qT**2 / 2
    if fmt == 'M':
        piR = (pIs - w_) * DI + (pTs - cost_T) * DT
        piI = (w_ - c / 2) * DI
    else:
        piR = r_ * pIs * DI + (pTs - cost_T) * DT
        piI = ((1 - r_) * pIs - c / 2) * DI
    d = dict(pT=pTs, pI=pIs, DT=sp.simplify(DT), DI=sp.simplify(DI), CS=sp.simplify(CS),
             piR=sp.simplify(piR), piI=sp.simplify(piI))
    d['TW'] = sp.simplify(d['piR'] + d['piI'] + d['CS'])
    d['Omega'] = sp.simplify(d['piR'] + lam * d['CS'])
    return d


BASE = {h: sp.Rational(13, 10), l: sp.Rational(9, 10), c: sp.Rational(7, 10), r: sp.Rational(1, 5)}

if __name__ == '__main__':
    HM = merchant(h); LM = merchant(l); HA = agency(h); LA = agency(l)
    print('w^HM =', HM['w'], '   w^LM =', sp.factor(LM['w']))
    print('pT^HM =', sp.factor(HM['pT'])); print('pI^HM =', sp.factor(HM['pI']))
    print('pT^LM =', sp.factor(LM['pT'])); print('pI^LM =', sp.factor(LM['pI']))
    print('pT^HA =', sp.factor(HA['pT'])); print('pI^HA =', sp.factor(HA['pI']))
    print('pT^LA =', sp.factor(LA['pT'])); print('pI^LA =', sp.factor(LA['pI']))
    print()
    print('Table 3 check (baseline):')
    for name, S in [('HM', HM), ('HA', HA), ('LM', LM), ('LA', LA)]:
        for lv in [0, sp.Rational(1, 2), 1]:
            vals = {k: float(S[k].subs(BASE).subs(lam, lv)) for k in ['pT', 'pI', 'DT', 'DI', 'piR', 'piI', 'CS', 'TW']}
            print(name, float(lv), ' '.join(f'{k}={v:.4f}' for k, v in vals.items()))
