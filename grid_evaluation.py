"""v2 recomputation of every numerical observation.
Flexible regime: h in [1.1,2], l in [0.5,0.95], c in [0.1, 2/(1+h)), lambda in [0,1].
Rigid regime: same h,l; c in [0.1, min(4/(2h+1), 2/(1+l))); r in {0.10,...,0.30};
  high-end valid iff r < min(rhat_H, h/(h+1)); economy valid iff r < (2-c(1+l))/(2-cl)."""
import numpy as np, sympy as sp, pickle, time, pandas as pd
from model import *
t0=time.time()
S={'HM':merchant(h),'LM':merchant(l),'HA':agency(h),'LA':agency(l)}
args=(h,l,c,r,lam)
KEYS=['pT','pI','DT','DI','CS','piR','piI','TW','Omega']
F={nm:{k:sp.lambdify(args,S[nm][k],'numpy') for k in KEYS} for nm in S}
Q=pickle.load(open('quartic.pkl','rb'))['quartic']
coef={nm:[sp.lambdify((h,l,c,lam),cf,'numpy') for cf in sp.Poly(Q[nm],r).all_coeffs()] for nm in Q}
LG=np.linspace(0,1,51)
H=np.round(np.arange(1.10,2.001,0.05),3); L=np.round(np.arange(0.50,0.951,0.05),3)
def rbar(nm,hv,lv,cv,la):
    if nm=='HA': return (cv*(hv-1)+2-la*(2-cv))/(cv*(hv+1)+2-2*la)
    return ((4-2*la)-cv*(2+lv-la))/((4-2*la)-cv*lv)
def rhat(hv,cv): return hv*(4-cv*(2*hv+1))/2
def rstar_path(nm,hv,lv,cv):
    out=np.full(len(LG),np.nan)
    cfs=[np.broadcast_to(np.asarray(cf(hv,lv,cv,LG),dtype=float),LG.shape) for cf in coef[nm]]
    for i,la in enumerate(LG):
        roots=np.roots([cf[i] for cf in cfs])
        ub=rbar(nm,hv,lv,cv,la)
        if nm=='HA': ub=min(ub,rhat(hv,cv))
        cand=[x.real for x in roots if abs(x.imag)<1e-9 and 0<x.real<ub]
        if cand: out[i]=max(cand,key=lambda x:F[nm]['Omega'](hv,lv,cv,x,la))
    return out
def evalpath(nm,hv,lv,cv,rv):
    return {k:np.asarray(F[nm][k](hv,lv,cv,rv,LG),dtype=float)*np.ones(len(LG)) for k in KEYS}
# ---------------- flexible regime ----------------
flex=[]
for hv in H:
    cb=2/(1+hv)
    cs=list(np.round(np.arange(0.1,cb-1e-9,0.025),4))
    if cb-0.005>cs[-1]: cs.append(round(cb-0.005,4))
    for lv in L:
        for cv in cs:
            rec=dict(h=hv,l=lv,c=cv)
            for nm in ['HA','LA']:
                rs=rstar_path(nm,hv,lv,cv); rec[nm+'_r']=rs
                for k,v in evalpath(nm,hv,lv,cv,rs).items(): rec[nm+'_'+k]=v
            for nm in ['HM','LM']:
                for k,v in evalpath(nm,hv,lv,cv,0.0).items(): rec[nm+'_'+k]=v
            flex.append(rec)
flex=pd.DataFrame(flex); flex.to_pickle('v2_flex.pkl')
print('flexible points',len(flex),' time',round(time.time()-t0,1))
# ---------------- rigid regime ----------------
rig=[]
for hv in H:
    for lv in L:
        cb=min(4/(2*hv+1),2/(1+lv))
        cs=list(np.round(np.arange(0.1,cb-1e-9,0.025),4))
        if cb-0.005>cs[-1]: cs.append(round(cb-0.005,4))
        for cv in cs:
            for rv in [0.10,0.15,0.20,0.25,0.30]:
                rec=dict(h=hv,l=lv,c=cv,r=rv)
                rec['H_ok']= rv<min(rhat(hv,cv),hv/(hv+1))
                rec['L_ok']= rv<(2-cv*(1+lv))/(2-cv*lv)
                for nm in ['HA','LA']:
                    for k,v in evalpath(nm,hv,lv,cv,rv).items(): rec[nm+'_'+k]=v
                for nm in ['HM','LM']:
                    for k,v in evalpath(nm,hv,lv,cv,0.0).items(): rec[nm+'_'+k]=v
                rig.append(rec)
rig=pd.DataFrame(rig); rig.to_pickle('v2_rigid.pkl')
print('rigid rows',len(rig),' H_ok',rig.H_ok.sum(),' L_ok',rig.L_ok.sum(),' time',round(time.time()-t0,1))
