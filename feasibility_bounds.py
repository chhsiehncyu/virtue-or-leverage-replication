"""Lemma 2: demand of the integrated hotel, the bound rhat_H, and the ordering of rhat_H and rbar_H."""
import sympy as sp
from model import h, l, c, r, lam, agency
HA, LA = agency(h), agency(l)
LamH = 4*h - 1 - r - lam*(2*h - 1); LamL = 4 - l*(1 + r) - lam*(2 - l)
PhiH = 2*c*h**2 + c*h - 4*h + 2*r; PhiL = c*(l - 1) + 2*(r - 1)
print('D_T^HA + Phi_H/(2 Lambda_H) =', sp.simplify(HA['DT'] + PhiH/(2*LamH)))
print('D_T^LA + Phi_L/(2 Lambda_L) =', sp.simplify(LA['DT'] + PhiL/(2*LamL)))
rhat = h*(4 - c*(2*h + 1))/2
rbar = (c*(h - 1) + 2 - lam*(2 - c))/(c*(h + 1) + 2 - 2*lam)
print('D_T^HA at rhat_H =', sp.simplify(HA['DT'].subs(r, rhat)), '; D_I^HA at rbar_H =', sp.simplify(HA['DI'].subs(r, rbar)))
print('rhat_H - rbar_H =', sp.factor(sp.simplify(rhat - rbar)))
print('rbar_H at lambda = 1:', sp.simplify(rbar.subs(lam, 1)))
rbarL = ((4 - 2*lam) - c*(2 + l - lam))/((4 - 2*lam) - c*l)
print('rbar_L at lambda = 1:', sp.factor(rbarL.subs(lam, 1)))
pt = {h: sp.Rational(13, 10), l: sp.Rational(9, 10), c: 1, r: sp.Rational(3, 10), lam: 0}
print('example h=1.3, c=1, r=0.3 (c > 2/(1+h)): D_T^HA =', float(HA['DT'].subs(pt)), ' rhat_H =', float(rhat.subs(pt)))
