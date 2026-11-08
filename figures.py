"""Figures 1 to 7. Line plots, one line style per regime, legends inside the axes away from the data."""
import sympy as sp, numpy as np, pickle, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy.optimize import brentq
from model import *

OUT = 'figures'; os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, AQUA, VIOLET, GRAY = '#2a78d6', '#eb6834', '#1baf7a', '#6250d6', '#6b6b6b'
plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 9, 'axes.labelsize': 9.5, 'legend.fontsize': 8,
                     'axes.spines.top': False, 'axes.spines.right': False, 'lines.linewidth': 1.6, 'mathtext.fontset': 'dejavuserif'})
HM = merchant(h); LM = merchant(l); HA = agency(h); LA = agency(l)
args = (h, l, c, r, lam)
F = {nm: {k: sp.lambdify(args, v, 'numpy') for k, v in S.items() if k != 'w'} for nm, S in [('HM', HM), ('LM', LM), ('HA', HA), ('LA', LA)]}
W = {'H': sp.lambdify((h, l, c), HM['w'], 'numpy'), 'L': sp.lambdify((h, l, c), LM['w'], 'numpy')}
Q = pickle.load(open('quartic.pkl', 'rb'))['quartic']
coef = {nm: [sp.lambdify((h, l, c, lam), cf, 'numpy') for cf in sp.Poly(Q[nm], r).all_coeffs()] for nm in Q}
def rstar(nm, hv, lv, cv, la):
    roots = np.roots([cf(hv, lv, cv, la) for cf in coef[nm]])
    cand = [x.real for x in roots if abs(x.imag) < 1e-9 and 0 < x.real < 1]
    return max(cand, key=lambda x: F[nm]['Omega'](hv, lv, cv, x, la)) if cand else np.nan
B = dict(h=1.3, l=0.9, c=0.7, r=0.2)
lg = np.linspace(0, 1, 201)
def ev(nm, k, rv=None, **kw):
    p = dict(B); p.update(kw); rv = p['r'] if rv is None else rv
    return np.array([F[nm][k](p['h'], p['l'], p['c'], rv, x) for x in lg])
def ev_flex(tier, k, **kw):
    p = dict(B); p.update(kw)
    rs = np.array([rstar(tier + 'A', p['h'], p['l'], p['c'], x) for x in lg])
    return np.array([F[tier + 'A'][k](p['h'], p['l'], p['c'], rr, x) for rr, x in zip(rs, lg)]), rs

def shared_legend(fig, ax, ncol, bottom=0.26, order=None):
    hnd, lab = ax.get_legend_handles_labels()
    if order is not None:
        hnd = [hnd[i] for i in order]; lab = [lab[i] for i in order]
    fig.legend(hnd, lab, loc='lower center', ncol=ncol, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(bottom=bottom)

def save(fig, name):
    fig.savefig(f'{OUT}/{name}.png', dpi=1200, bbox_inches='tight'); fig.savefig(f'{OUT}/{name}.pdf', bbox_inches='tight'); plt.close(fig)

# ---------- Figure 1: channel structure ----------
def box(ax, x, y, w_, h_, text, ec, fc):
    ax.add_patch(FancyBboxPatch((x, y), w_, h_, boxstyle='round,pad=0.02,rounding_size=0.03', ec=ec, fc=fc, lw=1.4))
    ax.text(x + w_ / 2, y + h_ / 2, text, ha='center', va='center', fontsize=8.5)
def arrow(ax, p, q):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=12, lw=1.2, color='#444444'))
fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4))
titles = ['(a) Merchant contract\nplatform holds price authority', '(b) Agency contract\nincumbent retains price authority']
for ax, ttl, fmt in zip(axes, titles, ['M', 'A']):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off'); ax.set_title(ttl, fontsize=9, pad=4)
    box(ax, 0.04, 0.72, 0.38, 0.2, 'Incumbent hotel $I$\n(quality $q_I = 1$)', ORANGE, '#fdeee6')
    box(ax, 0.58, 0.72, 0.38, 0.2, 'Integrated hotel $T$\n($q_T \\in \\{h, l\\}$)', AQUA, '#e5f6ef')
    box(ax, 0.28, 0.36, 0.44, 0.22, 'Platform $R$\n$\\Omega_R = \\pi_R + \\lambda\\, CS$', BLUE, '#e7f0fb')
    box(ax, 0.28, 0.04, 0.44, 0.16, 'Consumers\n$\\theta \\sim U[0,1]$', GRAY, '#f3f3f3')
    arrow(ax, (0.30, 0.72), (0.42, 0.58)); arrow(ax, (0.70, 0.72), (0.58, 0.58)); arrow(ax, (0.50, 0.36), (0.50, 0.20))
    ax.text(0.86, 0.63, 'ownership\n(internal)', ha='center', va='center', fontsize=7.5, color='#1a7a55')
    if fmt == 'M':
        ax.text(0.13, 0.63, 'wholesale $w$\n(hotel sets)', ha='center', va='center', fontsize=7.5, color='#8a4b14')
        ax.text(0.54, 0.27, 'platform sets $p_I$ and $p_T$', ha='left', va='center', fontsize=7.3, color='#1d4f8f')
    else:
        ax.text(0.12, 0.63, 'commission $r\\,p_I$\n(hotel sets $p_I$)', ha='center', va='center', fontsize=7.5, color='#8a4b14')
        ax.text(0.54, 0.345, 'platform sets $p_T$', ha='left', va='center', fontsize=7.3, color='#1d4f8f')
        ax.text(0.54, 0.29, 'posts $r$ (flexible regime)', ha='left', va='center', fontsize=6.8, color='#1d4f8f')
        ax.text(0.54, 0.235, 'or takes $r$ as given (rigid)', ha='left', va='center', fontsize=6.8, color='#1d4f8f')
fig.tight_layout(); save(fig, 'fig1')

# ---------- Figure 2: prices, merchant and rigid agency ----------
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharex=True)
for ax, tier, ttl in zip(axes, ['H', 'L'], ['(a) High-end integration', '(b) Economy integration']):
    M, A = tier + 'M', tier + 'A'
    q = B['h'] if tier == 'H' else B['l']
    ax.plot(lg, ev(M, 'pT'), color=BLUE, ls='-', label='$p_T$, merchant')
    ax.plot(lg, ev(M, 'pI'), color=ORANGE, ls='-', label='$p_I$, merchant')
    ax.plot(lg, ev(A, 'pT'), color=BLUE, ls='--', label='$p_T$, rigid agency')
    ax.plot(lg, ev(A, 'pI'), color=ORANGE, ls='--', label='$p_I$, rigid agency')
    ax.axhline(W[tier](B['h'], B['l'], B['c']), color=GRAY, ls='-.', lw=1.1, label='wholesale rate $w$')
    ax.axhline(B['c'] * q**2 / 2, color=GRAY, ls=':', lw=1.1, label='unit cost of $T$')
    ax.set_title(ttl, fontsize=9); ax.set_xlabel('Dual-purpose intensity $\\lambda$'); ax.set_xlim(0, 1)
    ax.set_ylabel('Price')
axes[0].set_ylim(0.3, 1.0); axes[1].set_ylim(0.2, 0.75)
fig.tight_layout(); shared_legend(fig, axes[0], ncol=3, bottom=0.32)
save(fig, 'fig2')

# ---------- Figure 3: three regimes at the baseline ----------
flex = {k: ev_flex('H', k)[0] for k in ['piI', 'CS', 'TW', 'piR']}
rs_H = ev_flex('H', 'piI')[1]; rs_L = ev_flex('L', 'piI')[1]
fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))
ax = axes[0, 0]
ax.plot(lg, rs_H, color=VIOLET, ls='-', label='flexible, high-end')
ax.plot(lg, rs_L, color=VIOLET, ls='--', label='flexible, economy')
ax.axhline(0.2, color=GRAY, ls=':', lw=1.1, label='rigid, $r = 0.2$')
ax.set_title('(a) Commission rate', fontsize=9); ax.set_ylabel('$r$'); ax.set_ylim(0, 0.55); ax.legend(loc='upper right', frameon=False)
panels = [('piI', '(b) Incumbent profit', '$\\pi_I$'), ('CS', '(c) Consumer surplus', '$CS$'), ('TW', '(d) Total welfare', '$TW$')]
for ax, (k, ttl, yl) in zip([axes[0, 1], axes[1, 0], axes[1, 1]], panels):
    ax.plot(lg, ev('HM', k), color=BLUE, ls='-', label='merchant')
    ax.plot(lg, flex[k], color=VIOLET, ls='-.', label='agency, flexible commission')
    ax.plot(lg, ev('HA', k), color=ORANGE, ls='--', label='agency, rigid commission')
    ax.set_title(ttl, fontsize=9); ax.set_ylabel(yl)
for ax in axes[1]: ax.set_xlabel('Dual-purpose intensity $\\lambda$')
for ax in axes.flat: ax.set_xlim(0, 1)
axes[0, 1].set_ylim(0, 0.02); axes[1, 1].set_ylim(0.15, 0.22)
fig.tight_layout(); shared_legend(fig, axes[0, 1], ncol=3, bottom=0.14)
save(fig, 'fig3')

# ---------- Figure 4: positioning under rigid agency ----------
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
ax = axes[0]
dM = ev('HM', 'Omega') - ev('LM', 'Omega'); dA = ev('HA', 'Omega') - ev('LA', 'Omega')
ax.plot(lg, dM, color=BLUE, ls='-', label='merchant, $\\Delta^M$')
ax.plot(lg, dA, color=ORANGE, ls='--', label='rigid agency, $\\Delta^A$')
ax.axhline(0, color=GRAY, lw=0.8)
lh = brentq(lambda x: F['HA']['Omega'](1.3, 0.9, 0.7, 0.2, x) - F['LA']['Omega'](1.3, 0.9, 0.7, 0.2, x), 0.05, 0.5)
ax.axvline(lh, color=GRAY, ls=':', lw=1.0); ax.text(lh + 0.02, -0.028, '$\\hat{\\lambda}_A$', fontsize=9)
ax.set_title('(a) Positioning gap at the baseline', fontsize=9); ax.set_xlabel('Dual-purpose intensity $\\lambda$'); ax.set_ylabel('$\\Omega_R^{H} - \\Omega_R^{L}$')
ax.set_xlim(0, 1); ax.set_ylim(-0.03, 0.012); ax.legend(loc='upper right', frameon=False)
ax = axes[1]
qgrid = np.linspace(0.80, 0.99, 96)
for rv, ls, lab in [(0.1, '-', '$r = 0.1$'), (0.2, '--', '$r = 0.2$'), (0.3, '-.', '$r = 0.3$')]:
    th = []
    for qv in qgrid:
        g = lambda x: F['HA']['Omega'](1.3, qv, 0.7, rv, x) - F['LA']['Omega'](1.3, qv, 0.7, rv, x)
        g0, g1 = g(0), g(1)
        th.append(brentq(g, 0, 1) if g0 > 0 > g1 else np.nan)
    ax.plot(qgrid, th, color=ORANGE, ls=ls, label=lab)
ax.set_title('(b) Switch threshold against economy-tier quality', fontsize=9); ax.set_xlabel('Economy-tier quality $l$'); ax.set_ylabel('$\\hat{\\lambda}_A$')
ax.set_xlim(0.8, 1.0); ax.set_ylim(0, 0.5); ax.legend(loc='upper left', frameon=False)
fig.tight_layout(); save(fig, 'fig4')

# ---------- Figure 5: format comparison under rigid commission ----------
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
ax = axes[0]
ax.plot(lg, ev('HM', 'CS') - ev('HA', 'CS'), color=BLUE, ls='-', label='high-end integration')
ax.plot(lg, ev('LM', 'CS') - ev('LA', 'CS'), color=AQUA, ls='--', label='economy integration')
ax.axhline(0, color=GRAY, lw=0.8)
ax.set_title('(a) Consumer-surplus gap, merchant minus agency', fontsize=9); ax.set_xlabel('Dual-purpose intensity $\\lambda$'); ax.set_ylabel('$CS^M - CS^A$')
ax.set_xlim(0, 1); ax.legend(loc='upper left', frameon=False)
ax = axes[1]
rgrid = np.linspace(0.05, 0.35, 61); lo, hi, thr = [], [], []
for rv in rgrid:
    om = lambda x: F['HA']['Omega'](1.3, 0.9, 0.7, rv, x) - F['HM']['Omega'](1.3, 0.9, 0.7, rv, x)
    cs = lambda x: F['HM']['CS'](1.3, 0.9, 0.7, rv, x) - F['HA']['CS'](1.3, 0.9, 0.7, rv, x)
    g = np.linspace(0, 1, 801); v = np.array([om(x) for x in g]); idx = np.where(np.diff(np.sign(v)) != 0)[0]
    roots = [brentq(om, g[i], g[i + 1]) for i in idx]
    lo.append(roots[0] if roots else np.nan); hi.append(roots[1] if len(roots) > 1 else (1.0 if roots else np.nan))
    v2 = np.array([cs(x) for x in g]); j = np.where(np.diff(np.sign(v2)) != 0)[0]
    thr.append(brentq(cs, g[j[0]], g[j[0] + 1]) if len(j) else np.nan)
lo, hi, thr = map(np.array, (lo, hi, thr))
ax.plot(rgrid, lo, color=ORANGE, ls='-', label='platform switches to agency')
ax.plot(rgrid, hi, color=ORANGE, ls='--', label='platform returns to merchant')
ax.plot(rgrid, thr, color=BLUE, ls='-.', label='consumers prefer merchant above')
ax.fill_between(rgrid, np.maximum(lo, thr), hi, where=hi > np.maximum(lo, thr), color=ORANGE, alpha=0.15, lw=0, label='paradox band')
ax.set_title('(b) Boundaries against the commission, high-end', fontsize=9); ax.set_xlabel('Commission rate $r$'); ax.set_ylabel('Dual-purpose intensity $\\lambda$')
ax.set_xlim(0.05, 0.35); ax.set_ylim(0.3, 1.02)
fig.tight_layout(); shared_legend(fig, axes[1], ncol=2, bottom=0.34)
save(fig, 'fig5')

# ---------- Figure 6: profits and welfare, merchant vs rigid agency ----------
fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))
series = [('HM', BLUE, '-', 'high-end, merchant'), ('HA', BLUE, '--', 'high-end, rigid agency'), ('LM', AQUA, '-', 'economy, merchant'), ('LA', AQUA, '--', 'economy, rigid agency')]
for ax, (k, ttl, yl, loc) in zip(axes.flat, [('piR', '(a) Platform profit', '$\\pi_R$', 'lower left'), ('piI', '(b) Incumbent profit', '$\\pi_I$', 'upper left'), ('CS', '(c) Consumer surplus', '$CS$', 'upper left'), ('TW', '(d) Total welfare', '$TW$', 'lower right')]):
    for nm, col, ls, lab in series:
        ax.plot(lg, ev(nm, k), color=col, ls=ls, label=lab)
    if k == 'TW':
        for nm, col in [('HA', BLUE), ('LA', AQUA)]:
            y = ev(nm, k); i = np.argmax(y); ax.plot(lg[i], y[i], 'o', color=col, ms=4)
    ax.set_title(ttl, fontsize=9); ax.set_ylabel(yl); ax.set_xlim(0, 1)
for ax in axes[1]: ax.set_xlabel('Dual-purpose intensity $\\lambda$')
axes[1, 1].set_ylim(0.15, 0.22)
fig.tight_layout(); shared_legend(fig, axes[0, 0], ncol=4, bottom=0.13)
save(fig, 'fig6')

# ---------- Figure 7: win-win example under rigid agency ----------
fig, ax = plt.subplots(figsize=(3.6, 2.8))
pr = ev('HA', 'piR', c=0.9, r=0.35); cs_ = ev('HA', 'CS', c=0.9, r=0.35)
ax.plot(lg, 100 * (pr / pr[0] - 1), color=ORANGE, ls='-', label='platform profit')
ax.plot(lg, 100 * (cs_ / cs_[0] - 1), color=AQUA, ls='--', label='consumer surplus')
ax.set_xlabel('Dual-purpose intensity $\\lambda$'); ax.set_ylabel('Change from $\\lambda = 0$ (percent)'); ax.set_xlim(0, 1)
ax.set_title('Win-win under a rigid commission ($c = 0.9$, $r = 0.35$)', fontsize=9); ax.legend(loc='upper left', frameon=False)
fig.tight_layout(); save(fig, 'fig7')
print('figures written')
