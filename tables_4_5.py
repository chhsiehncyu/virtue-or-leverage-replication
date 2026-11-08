"""Tables 4 and 5 of the paper, printed with the same number of decimals as in the text.
Table 4: merchant and rigid agency at the baseline (h = 1.3, l = 0.9, c = 0.7, r = 0.2).
Table 5: flexible agency at the baseline; the commission is the admissible root of the quartic in quartic.pkl.
The equilibrium objects are evaluated with 30-digit precision and rounded half up, so exact ties such as the
wholesale rate 0.4025 appear as 0.403."""
import pickle, sympy as sp
from decimal import Decimal, ROUND_HALF_UP
from model import h, l, c, r, lam, merchant, agency, BASE

S = {'HM': merchant(h), 'LM': merchant(l), 'HA': agency(h), 'LA': agency(l)}
keys = ['pT', 'pI', 'DT', 'DI', 'piR', 'piI', 'CS', 'TW']
dec = {'pT': 3, 'pI': 3, 'DT': 3, 'DI': 3, 'piR': 4, 'piI': 4, 'CS': 4, 'TW': 4}
Q = pickle.load(open('quartic.pkl', 'rb'))['quartic']


def fmt(v, d):
    """Round half up on a 30-digit decimal string, so exact ties such as 0.5915 round to 0.592."""
    q = Decimal(str(sp.N(v, 30))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)
    return str(abs(q)) if q == 0 else str(q)


def rstar(nm, la):
    sub = dict(BASE); sub.pop(r); sub[lam] = la
    poly = sp.Poly(Q[nm].subs(sub), r)
    cand = []
    for x in poly.nroots(n=30):
        if abs(sp.im(x)) < sp.Float('1e-20') and 0 < sp.re(x) < 1:
            xv = sp.re(x)
            if S[nm]['DI'].subs(sub).subs(r, xv) > 0 and S[nm]['DT'].subs(sub).subs(r, xv) > 0:
                cand.append(xv)
    return max(cand, key=lambda xv: S[nm]['Omega'].subs(sub).subs(r, xv))


print('Table 4. Case | lambda | p_T | p_I | D_T | D_I | pi_R | pi_I | CS | TW')
for nm in ['HM', 'HA', 'LM', 'LA']:
    for la in [0, sp.Rational(1, 2), 1]:
        sub = dict(BASE); sub[lam] = la
        vals = [S[nm][k].subs(sub) for k in keys]
        print(' | '.join([nm, str(float(la)).rstrip('0').rstrip('.')] + [fmt(v, dec[k]) for k, v in zip(keys, vals)]))

print('\nTable 5. Case | lambda | r* | p_T | p_I | D_T | D_I | pi_R | pi_I | CS | TW')
for nm in ['HA', 'LA']:
    for la in [0, sp.Rational(1, 4), sp.Rational(1, 2), sp.Rational(3, 4), 1]:
        rs = rstar(nm, la)
        sub = dict(BASE); sub.pop(r); sub[lam] = la; sub[r] = rs
        vals = [sp.N(S[nm][k].subs(sub), 30) for k in keys]
        print(' | '.join([nm, str(float(la)).rstrip('0').rstrip('.'), fmt(rs, 3)] + [fmt(v, dec[k]) for k, v in zip(keys, vals)]))
