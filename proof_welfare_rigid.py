"""Rigid commission: shape of total welfare in lambda (closed-form classification) and verification."""
import sympy as sp, numpy as np
from model import *
res={}
for nm,S in [('HA',agency(h)),('LA',agency(l))]:
    dTW=sp.factor(sp.together(sp.diff(S['TW'],lam)))
    num,den=sp.fraction(dTW)
    print(nm,'dTW/dlam factored numerator factors:',sp.factor_list(num))
    print(nm,'denominator:',sp.factor(den))
    res[nm]=dTW
# closed-form conditions
XH=c*(4*h**2*(1-r)+h*(1-r)*(1-2*r)-(2-r)*(1+r))-2*(4*h-3)*(1-r)      # >0  <=> falls at lambda=0 (HA)
YH=h*(1-r)*(1-2*r)-r*(2-r)                                            # <0  <=> rises at lambda=1 (HA)
BL=l**2*(1-r**2)-l*(1-r)*(2+r)+2*(2-r)
XL=2*(1-r)*(l*(1+r)-2*r)-c*BL                                         # >0  <=> falls at lambda=0 (LA)
YL=2*(1-r)*(1-2*r+l*r)-c*(1+l*(1-r)**2+l**2*r*(1-r))                  # <0  <=> rises at lambda=1 (LA)
f={k:sp.lambdify((h,l,c,r,lam),v,'numpy') for k,v in res.items()}
g={k:sp.lambdify((h,l,c,r),v,'numpy') for k,v in dict(XH=XH,YH=YH,XL=XL,YL=YL).items()}
DT={nm:sp.lambdify((h,l,c,r,lam),S['DT'],'numpy') for nm,S in [('HA',agency(h)),('LA',agency(l))]}
DI={nm:sp.lambdify((h,l,c,r,lam),S['DI'],'numpy') for nm,S in [('HA',agency(h)),('LA',agency(l))]}
rng=np.random.default_rng(0); bad={'H0':0,'H1':0,'L0':0,'L1':0,'Ushape':0}; n={'H':0,'L':0}
for _ in range(200000):
    hv=rng.uniform(1.01,3); lv=rng.uniform(0.05,0.99); cv=rng.uniform(0.01,1)*min(4/(2*hv+1),2/(1+lv)); rv=rng.uniform(0.001,0.95)
    for nm,t in [('HA','H'),('LA','L')]:
        if not (DT[nm](hv,lv,cv,rv,0)>0 and DT[nm](hv,lv,cv,rv,1)>0 and DI[nm](hv,lv,cv,rv,0)>0 and DI[nm](hv,lv,cv,rv,1)>0): continue
        n[t]+=1
        s0=f[nm](hv,lv,cv,rv,0.0); s1=f[nm](hv,lv,cv,rv,1.0)
        if t=='H':
            if (s0<0)!=(g['XH'](hv,lv,cv,rv)>0): bad['H0']+=1
            if (s1>0)!=(g['YH'](hv,lv,cv,rv)<0): bad['H1']+=1
        else:
            if (s0<0)!=(g['XL'](hv,lv,cv,rv)>0): bad['L0']+=1
            if (s1>0)!=(g['YL'](hv,lv,cv,rv)<0): bad['L1']+=1
        if s0<0 and s1>0: bad['Ushape']+=1
print('random feasible draws',n,' mismatches',bad)
for nm,(hv,lv,rv) in [('H',(1.3,0.9,0.2)),('L',(1.3,0.9,0.2))]:
    pass
print('baseline: H falls at 0 iff c >',round(float(sp.solve(XH.subs({h:1.3,r:0.2}),c)[0]),4))
print('H rises at parity iff r > root of',[round(float(x),4) for x in sp.solve(YH.subs({h:1.3}),r)])
print('L falls at 0 iff c <',round(float(sp.solve(XL.subs({l:0.9,r:0.2}),c)[0]),4),';  L rises at parity iff c >',round(float(sp.solve(YL.subs({l:0.9,r:0.2}),c)[0]),4))
for rv in [0.1,0.15,0.25,0.3]:
    print(' r=',rv,' H falls-at-0 threshold at h=1.3:',round(float(sp.solve(XH.subs({h:1.3,r:rv}),c)[0]),4),'  L thresholds at l=0.9 (falls below, rises to parity above):',round(float(sp.solve(XL.subs({l:0.9,r:rv}),c)[0]),4),round(float(sp.solve(YL.subs({l:0.9,r:rv}),c)[0]),4))
