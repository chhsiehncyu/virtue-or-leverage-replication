"""Rigid-regime tables: paradox boundaries, tier threshold, win-win threshold, welfare peak; sensitivity in r and c.
Also r*(lambda) sensitivity for the flexible regime."""
import sympy as sp, numpy as np, pandas as pd, pickle, json
from scipy.optimize import brentq, minimize_scalar
from model import *

HM = merchant(h); LM = merchant(l); HA = agency(h); LA = agency(l)
args = (h, l, c, r, lam)
F = {nm: {k: sp.lambdify(args, v, 'numpy') for k, v in S.items() if k != 'w'} for nm, S in [('HM', HM), ('LM', LM), ('HA', HA), ('LA', LA)]}
Q = pickle.load(open('quartic.pkl', 'rb'))['quartic']
coef = {nm: [sp.lambdify((h, l, c, lam), cf, 'numpy') for cf in sp.Poly(Q[nm], r).all_coeffs()] for nm in Q}
def rstar(nm, hv, lv, cv, la):
    roots = np.roots([cf(hv, lv, cv, la) for cf in coef[nm]])
    cand = [x.real for x in roots if abs(x.imag) < 1e-9 and 0 < x.real < 1]
    return max(cand, key=lambda x: F[nm]['Omega'](hv, lv, cv, x, la)) if cand else None

def crossings(fun, lo=0.0, hi=1.0, n=2001):
    g = np.linspace(lo, hi, n); v = np.array([fun(x) for x in g]); out = []
    for i in range(n - 1):
        if v[i] == 0: out.append(g[i])
        elif v[i] * v[i + 1] < 0: out.append(brentq(fun, g[i], g[i + 1]))
    return out, v[0], v[-1]

def admissible(hv, lv, cv, rv):
    # Section 6: both hotels sell at every intensity under both tiers.
    # High-end: r < rhat_H = h[4 - c(2h+1)]/2 and r < h/(h+1); economy: r < [2 - c(1+l)]/(2 - cl).
    if cv >= min(4 / (2 * hv + 1), 2 / (1 + lv)): return False
    if rv >= hv * (4 - cv * (2 * hv + 1)) / 2: return False
    if rv >= hv / (hv + 1): return False
    if rv >= (2 - cv * (1 + lv)) / (2 - cv * lv): return False
    return True

def paradox(hv, lv, cv, rv, tier):
    A = tier + 'A'; M = tier + 'M'
    om = lambda x: F[A]['Omega'](hv, lv, cv, rv, x) - F[M]['Omega'](hv, lv, cv, rv, x)
    cs = lambda x: F[M]['CS'](hv, lv, cv, rv, x) - F[A]['CS'](hv, lv, cv, rv, x)
    om_c, om0, om1 = crossings(om); cs_c, cs0, cs1 = crossings(cs)
    # agency-preferred set
    pts = [0.0] + om_c + [1.0]; agency_int = []
    for a, b in zip(pts[:-1], pts[1:]):
        if om((a + b) / 2) > 0: agency_int.append((a, b))
    cs_thr = cs_c[0] if cs_c else (None if cs0 < 0 else 0.0)
    # overlap
    overlap = []
    if cs_thr is not None:
        for a, b in agency_int:
            lo_, hi_ = max(a, cs_thr), b
            if hi_ > lo_: overlap.append((lo_, hi_))
    return agency_int, cs_thr, overlap

rows = []
base = dict(h=1.3, l=0.9, c=0.7, r=0.2)
variants = [('baseline', base)]
for rv in [0.1, 0.15, 0.25, 0.3]: variants.append((f'r={rv}', dict(base, r=rv)))
for cv in [0.5, 0.6, 0.8]: variants.append((f'c={cv}', dict(base, c=cv)))
for hv in [1.2, 1.5]: variants.append((f'h={hv}', dict(base, h=hv)))
for lv in [0.8, 0.95]: variants.append((f'l={lv}', dict(base, l=lv)))
print('=== Control-paradox boundaries (rigid commission) ===')
for name, p in variants:
    for tier in ['H', 'L']:
        if not admissible(p['h'], p['l'], p['c'], p['r']):
            print(name, tier, 'not admissible'); continue
        ag, thr, ov = paradox(p['h'], p['l'], p['c'], p['r'], tier)
        rows.append(dict(variant=name, tier=tier, agency_intervals=[(round(a, 4), round(b, 4)) for a, b in ag], cs_threshold=None if thr is None else round(thr, 4), overlap=[(round(a, 4), round(b, 4)) for a, b in ov]))
        print(f'{name:9s} {tier}: platform prefers agency on {rows[-1]["agency_intervals"]}; consumers prefer merchant above {rows[-1]["cs_threshold"]}; paradox band {rows[-1]["overlap"]}')
pd.DataFrame(rows).to_csv('table_paradox.csv', index=False)

print('\n=== Tier threshold lambda_hat under rigid agency; win-win threshold cbar; welfare peak ===')
rows2 = []
for name, p in variants:
    if not admissible(p['h'], p['l'], p['c'], p['r']):
        continue
    gap = lambda x: F['HA']['Omega'](p['h'], p['l'], p['c'], p['r'], x) - F['LA']['Omega'](p['h'], p['l'], p['c'], p['r'], x)
    cr, g0, g1 = crossings(gap)
    gapM = F['HM']['Omega'](p['h'], p['l'], p['c'], p['r'], 0) - F['LM']['Omega'](p['h'], p['l'], p['c'], p['r'], 0)
    cbar = 4 * (1 - p['r']) / (2 * p['h'] * (1 - p['r']) + (1 + p['r'])**2)
    peaks = {}
    for tier in ['H', 'L']:
        res = minimize_scalar(lambda x: -F[tier + 'A']['TW'](p['h'], p['l'], p['c'], p['r'], x), bounds=(0, 1), method='bounded')
        tw0 = F[tier + 'A']['TW'](p['h'], p['l'], p['c'], p['r'], 0); tw1 = F[tier + 'A']['TW'](p['h'], p['l'], p['c'], p['r'], 1)
        peaks[tier] = round(res.x, 4) if (-res.fun > max(tw0, tw1) + 1e-10) else ('1' if tw1 >= tw0 else '0')
    rows2.append(dict(variant=name, DeltaA0=round(g0, 4), DeltaA1=round(g1, 4), lambda_hat=[round(x, 4) for x in cr], merchant_prefers=('H' if gapM > 0 else 'L'), cbar=round(cbar, 4), peak_H=peaks['H'], peak_L=peaks['L']))
    print(rows2[-1])
pd.DataFrame(rows2).to_csv('table_rigid_thresholds.csv', index=False)

print('\n=== Flexible regime: r*(lambda) sensitivity ===')
rows3 = []
for name, p in variants:
    if name.startswith('r='): continue
    rec = dict(variant=name)
    for tier in ['H', 'L']:
        for la in [0, 0.5, 1.0]:
            rs = rstar(tier + 'A', p['h'], p['l'], p['c'], la)
            rec[f'{tier}_r*_{la}'] = None if rs is None else round(rs, 4)
    rows3.append(rec); print(rec)
pd.DataFrame(rows3).to_csv('table_rstar_sensitivity.csv', index=False)

