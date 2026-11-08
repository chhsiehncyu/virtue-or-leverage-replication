"""Baseline numbers reported in Sections 5 to 7 and 10: commissions, shares, profit peak, welfare thresholds, tier-switch ranges, commission cap."""

# ======== audit_nums.py ========
import numpy as np, sympy as sp, pickle
from scipy.optimize import brentq, minimize_scalar
from model import *
HA = agency(h); LA = agency(l)
args=(h,l,c,r,lam)
F={nm:{k:sp.lambdify(args,v,'numpy') for k,v in S.items()} for nm,S in [('HA',HA),('LA',LA)]}
Q=pickle.load(open('quartic.pkl','rb'))['quartic']
coef={nm:[sp.lambdify((h,l,c,lam),cf,'numpy') for cf in sp.Poly(Q[nm],r).all_coeffs()] for nm in Q}
def rstar(nm,hv,lv,cv,la):
    roots=np.roots([cf(hv,lv,cv,la) for cf in coef[nm]])
    cand=[x.real for x in roots if abs(x.imag)<1e-9 and 0<x.real<1]
    return max(cand,key=lambda x:F[nm]['Omega'](hv,lv,cv,x,la)) if cand else None
hv,lv,cv=1.3,0.9,0.7
print('== shares under flexible commission at baseline ==')
for nm in ['HA','LA']:
    for la in [0,0.5,1]:
        rs=rstar(nm,hv,lv,cv,la); DT=F[nm]['DT'](hv,lv,cv,rs,la); DI=F[nm]['DI'](hv,lv,cv,rs,la)
        print(nm,la,f'r*={rs:.4f} DT={DT:.4f} DI={DI:.4f} share_T={DT/(DT+DI):.4f} coverage={DT+DI:.4f}')
print('== lambda at which r* equals a cap ==')
for nm in ['HA','LA']:
    for cap in [0.15,0.2,0.25]:
        f=lambda la: rstar(nm,hv,lv,cv,la)-cap
        try: print(nm,'cap',cap,'binds for lambda below', round(brentq(f,0,1),4))
        except Exception as e: print(nm,cap,'no root',e)
print('== flexible profit peak at baseline ==')
for nm in ['HA','LA']:
    piR=lambda la: F[nm]['piR'](hv,lv,cv,rstar(nm,hv,lv,cv,la),la)
    res=minimize_scalar(lambda x:-piR(x),bounds=(0,0.2),method='bounded')
    print(nm,'peak at lambda',round(res.x,4),'gain %',round(100*(piR(res.x)/piR(0)-1),4), 'piR(0)=',round(piR(0),6))


# ======== audit_cap.py ========
import numpy as np, sympy as sp, pickle
from model import *
HA=agency(h); LA=agency(l)
args=(h,l,c,r,lam)
F={nm:{k:sp.lambdify(args,v,'numpy') for k,v in S.items()} for nm,S in [('HA',HA),('LA',LA)]}
Q=pickle.load(open('quartic.pkl','rb'))['quartic']
coef={nm:[sp.lambdify((h,l,c,lam),cf,'numpy') for cf in sp.Poly(Q[nm],r).all_coeffs()] for nm in Q}
def rstar(nm,hv,lv,cv,la):
    roots=np.roots([cf(hv,lv,cv,la) for cf in coef[nm]])
    cand=[x.real for x in roots if abs(x.imag)<1e-9 and 0<x.real<1]
    return max(cand,key=lambda x:F[nm]['Omega'](hv,lv,cv,x,la))
hv,lv,cv,cap=1.3,0.9,0.7,0.2
for nm in ['HA','LA']:
    lg=np.linspace(0,1,101)
    rr=np.array([min(rstar(nm,hv,lv,cv,la),cap) for la in lg])
    # check capped optimum by brute force on r grid
    bf=[]
    for la in [0,0.5,0.9,1.0]:
        grid=np.linspace(0.0005,cap,4000); vals=F[nm]['Omega'](hv,lv,cv,grid,la)
        bf.append((la,round(grid[np.argmax(vals)],4),round(min(rstar(nm,hv,lv,cv,la),cap),4)))
    piI=np.array([F[nm]['piI'](hv,lv,cv,x,la) for x,la in zip(rr,lg)])
    i=np.argmin(piI)
    print(nm,'capped optimum brute-force vs min(r*,cap):',bf)
    print(nm,'incumbent profit under 20%% cap: lam=0 %.5f, min %.5f at lam=%.2f, lam=1 %.5f'%(piI[0],piI[i],lg[i],piI[-1]))


# ======== analyze_v2_thr.py ========
import numpy as np, sympy as sp, pickle
from scipy.optimize import brentq
from model import *
S={'HM':merchant(h),'LM':merchant(l),'HA':agency(h),'LA':agency(l)}
args=(h,l,c,r,lam)
F={nm:{k:sp.lambdify(args,S[nm][k],'numpy') for k in ['TW','Omega','CS','piR','piI','DT','DI']} for nm in S}
Q=pickle.load(open('quartic.pkl','rb'))['quartic']
coef={nm:[sp.lambdify((h,l,c,lam),cf,'numpy') for cf in sp.Poly(Q[nm],r).all_coeffs()] for nm in Q}
def rstar(nm,hv,lv,cv,la):
    roots=np.roots([float(cf(hv,lv,cv,la)) for cf in coef[nm]])
    cand=[x.real for x in roots if abs(x.imag)<1e-9 and 0<x.real<1 and F[nm]['DI'](hv,lv,cv,x.real,la)>0 and F[nm]['DT'](hv,lv,cv,x.real,la)>0]
    return max(cand,key=lambda x:F[nm]['Omega'](hv,lv,cv,x,la))
def gapTW(t,hv,lv,cv,la):
    nmA=t+'A'; nmM=t+'M'
    return F[nmA]['TW'](hv,lv,cv,rstar(nmA,hv,lv,cv,la),la)-F[nmM]['TW'](hv,lv,cv,0,la)
hv,lv=1.3,0.9
cH=brentq(lambda x: gapTW('H',hv,lv,x,0),0.2,0.6); cL=brentq(lambda x: gapTW('L',hv,lv,x,0),0.5,0.86)
print('lambda=0 TW tie: H tier at h=1.3: c=',round(cH,4),'   L tier at l=0.9: c=',round(cL,4))
for t in ['H','L']:
    x=brentq(lambda la: gapTW(t,hv,lv,0.7,la),0,1); print('baseline TW crossing',t,round(x,4))
# baseline CS loss from format choice, H tier
for la in [0,1]:
    a=F['HA']['CS'](hv,lv,0.7,rstar('HA',hv,lv,0.7,la),la); m=F['HM']['CS'](hv,lv,0.7,0,la)
    print('H baseline lambda',la,' CS loss share of merchant CS',round(1-a/m,4))


# ======== check_switch_1d.py ========
import numpy as np, sympy as sp
from scipy.optimize import brentq
from model import *
HA=agency(h); LA=agency(l)
args=(h,l,c,r,lam)
OH=sp.lambdify(args,HA['Omega'],'numpy'); OL=sp.lambdify(args,LA['Omega'],'numpy')
def sw(hv,lv,cv,rv):
    ok = rv<min(hv*(4-cv*(2*hv+1))/2, hv/(hv+1)) and rv<(2-cv*(1+lv))/(2-cv*lv)
    g0=OH(hv,lv,cv,rv,0)-OL(hv,lv,cv,rv,0); g1=OH(hv,lv,cv,rv,1)-OL(hv,lv,cv,rv,1)
    return ok and g0>0 and g1<0
ls=np.round(np.arange(0.5,0.996,0.0025),4); print('vary l (h=1.3,c=0.7,r=0.2): switch for l in',ls[[sw(1.3,x,0.7,0.2) for x in ls]].min(),'to',ls[[sw(1.3,x,0.7,0.2) for x in ls]].max())
cs=np.round(np.arange(0.1,1.05,0.0025),4); m=[sw(1.3,0.9,x,0.2) for x in cs]; print('vary c (h=1.3,l=0.9,r=0.2): switch for c in',cs[m].min(),'to',cs[m].max())
rs=np.round(np.arange(0.01,0.5,0.0025),4); m=[sw(1.3,0.9,0.7,x) for x in rs]; print('vary r (h=1.3,l=0.9,c=0.7): switch for r in',rs[m].min(),'to',rs[m].max())
# threshold at r=0.1 and 0.3 baseline
for rv in [0.1,0.2,0.3]:
    f=lambda la: OH(1.3,0.9,0.7,rv,la)-OL(1.3,0.9,0.7,rv,la); print('lambda_hat at r=',rv,round(brentq(f,0,1),4))


# ======== Figure 4(b): economy quality above which the switch exists ========
import numpy as np, sympy as sp
from model import *
OH=sp.lambdify((h,l,c,r,lam),agency(h)['Omega'],'numpy'); OL=sp.lambdify((h,l,c,r,lam),agency(l)['Omega'],'numpy')
q=np.linspace(0.80,0.999,800)
for rv in [0.1,0.2,0.3]:
    m=[(OH(1.3,x,0.7,rv,0)-OL(1.3,x,0.7,rv,0)>0) and (OH(1.3,x,0.7,rv,1)-OL(1.3,x,0.7,rv,1)<0) for x in q]
    print('r =',rv,' switch exists for q_L above', round(q[int(np.argmax(m))],3))
