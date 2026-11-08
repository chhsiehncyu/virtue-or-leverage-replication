"""Verify the numbers of the extensions (hotel-side dual purpose; continuous quality)."""
import sympy as sp, numpy as np
from scipy.optimize import minimize_scalar, brentq
from model import *
mu_ = sp.symbols('mu', nonnegative=True)
B = {h: 1.3, l: 0.9, c: 0.7, r: 0.2}

# ---- 8.1 hotel-side dual purpose, merchant, high-end ----
DT, DI, CS = demand_cs(h, pT, pI)
Om = (pI - w) * DI + (pT - c * h**2 / 2) * DT          # platform profit maximizer (lam = 0)
sol = sp.solve([sp.diff(Om, pT), sp.diff(Om, pI)], [pT, pI], dict=True)[0]
OmI = ((w - c / 2) * DI + mu_ * CS).subs(sol)
wmu = sp.solve(sp.diff(OmI, w), w)[0]
print('w(mu) merchant H:', sp.factor(wmu), ' claimed:', sp.factor(c * (2 * (h + 1) - h * mu_) / (2 * (4 - mu_))))
CSmu = sp.simplify(CS.subs(sol).subs(w, wmu))
f = sp.lambdify(mu_, CSmu.subs(B))
print('CS at mu=0,1:', f(0), f(1), ' dCS/dmu at 0:', float(sp.diff(CSmu, mu_).subs(B).subs(mu_, 0)))
piImu = sp.simplify(((w - c / 2) * DI).subs(sol).subs(w, wmu))
print('incumbent profit at mu=0,1:', float(piImu.subs(B).subs(mu_, 0)), float(piImu.subs(B).subs(mu_, 1)))
# compare with OTA-side lambda: dCS/dlam at 0 under merchant = ?
HM = merchant(h); print('merchant: dCS/dlam at 0 =', float(sp.diff(HM['CS'], lam).subs(B).subs(lam, 0)), ' CS(1)=', float(HM['CS'].subs(B).subs(lam, 1)))

# ---- 8.1 agency (rigid r), hotel-side mu, high-end ----
piI_mu = ((1 - r) * pI - c / 2) * DI + mu_ * CS
OmR = r * pI * DI + (pT - c * h**2 / 2) * DT            # lam = 0
solA = sp.solve([sp.diff(OmR, pT), sp.diff(piI_mu, pI)], [pT, pI], dict=True)[0]
CSA = sp.simplify(CS.subs(solA)); piIA = sp.simplify((((1 - r) * pI - c / 2) * DI).subs(solA))
print('agency: dCS/dmu at 0 =', float(sp.diff(CSA, mu_).subs(B).subs(mu_, 0)), ' CS(mu=1)=', float(CSA.subs(B).subs(mu_, 1)))
HA = agency(h); print('agency: dCS/dlam at 0 =', float(sp.diff(HA['CS'], lam).subs(B).subs(lam, 0)), ' CS(lam=1)=', float(HA['CS'].subs(B).subs(lam, 1)))
g = sp.lambdify(mu_, piIA.subs(B)); mub = brentq(g, 0.3, 1.0); print('incumbent profit zero at mu =', mub)

# ---- 8.2 continuous quality, c=0.75, r=0.2 ----
HMf = sp.lambdify((h, c, r, lam), merchant(h)['Omega'], 'numpy'); LMf = sp.lambdify((l, c, r, lam), merchant(l)['Omega'], 'numpy')
HAf = sp.lambdify((h, c, r, lam), agency(h)['Omega'], 'numpy'); LAf = sp.lambdify((l, c, r, lam), agency(l)['Omega'], 'numpy')
qs_hi = np.linspace(1.001, 1.9, 900); qs_lo = np.linspace(0.3, 0.999, 700)
for fmt, (Hf, Lf) in [('M', (HMf, LMf)), ('A', (HAf, LAf))]:
    for lv in [0, 0.5, 1.0]:
        vh = Hf(qs_hi, 0.75, 0.2, lv); vl = Lf(qs_lo, 0.75, 0.2, lv)
        # interiority: ignore nan; check demands positive would need more; report argmax
        allq = np.concatenate([qs_lo, qs_hi]); allv = np.concatenate([vl, vh])
        allv = np.where(np.isfinite(allv), allv, -np.inf)
        print(f'continuous quality, format {fmt}, lam={lv}: q* = {allq[np.argmax(allv)]:.3f}  (Omega={allv.max():.4f})')
