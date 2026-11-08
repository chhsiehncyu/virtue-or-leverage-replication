"""Numerical observations of Sections 5 and 6 over the grid of grid_evaluation.py, and the support table (Table R1)."""

# ======== flex ========
import numpy as np, pandas as pd
LG=np.linspace(0,1,51)
f=pd.read_pickle('v2_flex.pkl')
print('flexible points:',len(f))
nan=f.apply(lambda R: np.isnan(R.HA_r).any() or np.isnan(R.LA_r).any(),axis=1); print('points without interior r*:',nan.sum())
inc=lambda x: np.all(np.diff(x)>0); dec=lambda x: np.all(np.diff(x)<0)
def share(fun): return np.mean([fun(R) for _,R in f.iterrows()])
print('r* strictly decreasing: HA',share(lambda R:dec(R.HA_r)),' LA',share(lambda R:dec(R.LA_r)))
print('D_T,D_I>0 at r*: ',share(lambda R: (R.HA_DT>0).all() and (R.HA_DI>0).all() and (R.LA_DT>0).all() and (R.LA_DI>0).all()))
for nm in ['HA','LA']:
    print(nm,'piI increasing',share(lambda R:inc(R[nm+'_piI'])),' margin increasing',share(lambda R:inc((1-R[nm+'_r'])*R[nm+'_pI']-R.c/2)),' CS inc',share(lambda R:inc(R[nm+'_CS'])),' TW inc',share(lambda R:inc(R[nm+'_TW'])),' prices dec',share(lambda R:dec(R[nm+'_pT']) and dec(R[nm+'_pI'])))
    pk=np.array([LG[np.argmax(R[nm+'_piR'])] for _,R in f.iterrows()]); gain=np.array([R[nm+'_piR'].max()/R[nm+'_piR'][0]-1 for _,R in f.iterrows()])
    print('   profit peak lambda on the 0.02 grid (profit_gain_flex.py locates it on a 0.0005 grid): max',pk.max(),' mean',pk.mean().round(3),'  max gain %',round(100*gain.max(),4),' profit decreasing after peak:',share(lambda R: dec(R[nm+'_piR'][np.argmax(R[nm+'_piR']):]) if np.argmax(R[nm+'_piR'])<50 else True))
    r0=np.array([R[nm+'_r'][0] for _,R in f.iterrows()])
    print('   r*(0) range',r0.min().round(3),r0.max().round(3),'  share r*(0)<0.4',np.mean(r0<0.4).round(4))
    lo=f[r0<0.4]
    if len(lo): print('   r*(0)<0.4 occurs at c/cbound >=',(lo.c/(2/(1+lo.h))).min().round(3),' c>=',lo.c.min(),' h range',lo.h.min(),lo.h.max(), ' l range',lo.l.min(),lo.l.max())
    r1=np.array([R[nm+'_r'][-1] for _,R in f.iterrows()]); print('   r*(1) range',r1.min().round(3),r1.max().round(3))
# shares of served market
R0=f[(f.h==1.3)&(f.l==0.9)&(np.isclose(f.c,0.7))]
print('baseline rows:',len(R0))


# ======== flex2 ========
import numpy as np, pandas as pd
LG=np.linspace(0,1,51)
f=pd.read_pickle('v2_flex.pkl')
rows=[]
for _,R in f.iterrows():
    tM=np.where(R.HM_Omega>R.LM_Omega,'H','L'); tA=np.where(R.HA_Omega>R.LA_Omega,'H','L')
    gM=(R.HM_Omega-R.LM_Omega)/np.maximum(R.HM_Omega,R.LM_Omega)   # relative merchant tier gap
    OmM=np.maximum(R.HM_Omega,R.LM_Omega); OmA=np.maximum(R.HA_Omega,R.LA_Omega)
    piM=np.where(tM=='H',R.HM_piR,R.LM_piR); piA=np.where(tA=='H',R.HA_piR,R.LA_piR)
    CSM=np.where(tM=='H',R.HM_CS,R.LM_CS); CSA=np.where(tA=='H',R.HA_CS,R.LA_CS)
    IM=np.where(tM=='H',R.HM_piI,R.LM_piI); IA=np.where(tA=='H',R.HA_piI,R.LA_piI)
    rows.append(dict(h=R.h,l=R.l,c=R.c,
        agree_all=np.all(tM==tA), agree_frac=np.mean(tM==tA),
        tierA_changes=len(set(tA))>1, tierM_changes=len(set(tM))>1,
        min_abs_gapM=np.min(np.abs(gM)),
        plat_agency_obj=np.all(OmA>OmM), plat_agency_profit=np.all(piA>piM),
        cons_merchant=np.all(CSM>CSA), inc_merchant=np.all(IM>IA), inc_merchant_frac=np.mean(IM>IA),
        csgap_widens=np.all(np.diff(CSM-CSA)>0),
        csgap_H_widens=np.all(np.diff(R.HM_CS-R.HA_CS)>0), csgap_L_widens=np.all(np.diff(R.LM_CS-R.LA_CS)>0)))
d=pd.DataFrame(rows)
print('tier agreement at every lambda:',d.agree_all.mean().round(4),' mean fraction:',d.agree_frac.mean().round(4))
print('flexible-agency tier changes with lambda at',d.tierA_changes.sum(),'points; merchant tier never changes:',(~d.tierM_changes).all())
dis=d[~d.agree_all]
print('disagreement points: min |merchant tier gap| relative:',dis.min_abs_gapM.describe().round(4).to_dict())
print('agreement points: min |merchant tier gap| quantiles', d[d.agree_all].min_abs_gapM.quantile([0,0.01,0.05]).round(4).to_dict())
print('platform prefers agency (objective) at every lambda:',d.plat_agency_obj.mean(),' (profit):',d.plat_agency_profit.mean())
print('consumers prefer merchant at every lambda:',d.cons_merchant.mean())
print('incumbent prefers merchant at every lambda:',d.inc_merchant.mean().round(4))
ni=d[~d.inc_merchant]; print('incumbent exceptions: h',ni.h.min(),ni.h.max(),' l',ni.l.min(),ni.l.max(),' c',ni.c.min(),ni.c.max(),' frac of lambdas',ni.inc_merchant_frac.describe().round(3).to_dict())
print(ni[['h','l','c','inc_merchant_frac']].head(30).to_string())
print('CS gap (preferred tiers) widens:',d.csgap_widens.mean(),'  H-tier gap widens',d.csgap_H_widens.mean(),'  L-tier gap widens',d.csgap_L_widens.mean())
d.to_pickle('v2_flex_cmp.pkl')


# ======== flex3 ========
import numpy as np, pandas as pd
LG=np.linspace(0,1,51)
f=pd.read_pickle('v2_flex.pkl'); d=pd.read_pickle('v2_flex_cmp.pkl')
# (a) tier switching under flexible agency: direction and merchant gap
sw=[]
for i,R in f.iterrows():
    tA=np.where(R.HA_Omega>R.LA_Omega,'H','L')
    if len(set(tA))>1:
        gM=np.abs(R.HM_Omega[0]-R.LM_Omega[0])/max(R.HM_Omega[0],R.LM_Omega[0])
        gA0=np.abs(R.HA_Omega-R.LA_Omega)/np.maximum(R.HA_Omega,R.LA_Omega)
        sw.append(dict(h=R.h,l=R.l,c=R.c,dir=tA[0]+'->'+tA[-1],gapM=gM,mingapA=gA0.min(),maxgapA=gA0.max()))
sw=pd.DataFrame(sw)
print('switch directions:',sw.dir.value_counts().to_dict())
print('merchant rel gap at switching points: max',sw.gapM.max().round(4),'  agency rel gap max over lambda at switching points:',sw.maxgapA.max().round(4))
# (b) incumbent exceptions vs tier disagreement
ex=d[~d.inc_merchant]
print('incumbent exceptions:',len(ex),' of which tier disagreement:',(~ex.agree_all).sum())
# per-tier incumbent comparison
pt=[]
for i,R in f.iterrows():
    pt.append(dict(h=R.h,l=R.l,c=R.c,H_inc_M=np.all(R.HM_piI>R.HA_piI),L_inc_M=np.all(R.LM_piI>R.LA_piI)))
pt=pd.DataFrame(pt)
print('within tier, incumbent prefers merchant at every lambda: H',pt.H_inc_M.mean().round(4),' L',pt.L_inc_M.mean().round(4))
eH=pt[~pt.H_inc_M]; eL=pt[~pt.L_inc_M]
print('H-tier exceptions c range',eH.c.min() if len(eH) else None, eH.c.max() if len(eH) else None)
print('L-tier exceptions c range',eL.c.min() if len(eL) else None, eL.c.max() if len(eL) else None, ' h',eL.h.min() if len(eL) else None, eL.h.max() if len(eL) else None)
# relation of L exceptions to c relative to 2/(1+l)? and to l
if len(eL):
    eL=eL.assign(crel=eL.c/(2/(1+eL.l)), crelh=eL.c/(2/(1+eL.h)))
    print('  L exceptions c/(2/(1+l)) range',eL.crel.min().round(3),eL.crel.max().round(3),'  c/(2/(1+h)) range',eL.crelh.min().round(3),eL.crelh.max().round(3))
# (c) L-tier CS gap monotonicity exceptions
cg=[]
for i,R in f.iterrows():
    g=R.LM_CS-R.LA_CS
    cg.append(dict(h=R.h,l=R.l,c=R.c,mono=np.all(np.diff(g)>0),g0=g[0],g1=g[-1]))
cg=pd.DataFrame(cg); e=cg[~cg.mono]
print('L CS-gap non-monotone points:',len(e),' c range',e.c.min(),e.c.max(),' l range',e.l.min(),e.l.max(),' h range',e.h.min(),e.h.max())
print(e.head(12).to_string())


# ======== flex4 ========
import numpy as np, pandas as pd
LG=np.linspace(0,1,51)
f=pd.read_pickle('v2_flex.pkl'); d=pd.read_pickle('v2_flex_cmp.pkl')
sw=d.tierA_changes.values; nm_=~d.csgap_widens.values
print('non-monotone preferred-tier CS gap points:',nm_.sum(),'  of which flexible tier switches:',(nm_&sw).sum(),'  switch points with monotone gap:',(sw&~nm_).sum())
def crossings(g):
    s=np.sign(g); s=s[s!=0]; return int(np.sum(s[1:]!=s[:-1]))
rows=[]
for i,R in f.iterrows():
    for t in ['H','L']:
        g=R[t+'A_TW']-R[t+'M_TW']   # agency minus merchant
        k=crossings(g)
        lam_x=np.nan
        if k==1:
            j=np.where(np.sign(g[1:])!=np.sign(g[:-1]))[0][0]
            lam_x=LG[j]+(LG[j+1]-LG[j])*g[j]/(g[j]-g[j+1])
        rows.append(dict(h=R.h,l=R.l,c=R.c,tier=t,A_better0=g[0]>0,A_better1=g[-1]>0,ncross=k,lam_x=lam_x))
w=pd.DataFrame(rows)
for t in ['H','L']:
    x=w[w.tier==t]
    print(t,': agency better at lambda=0 share',x.A_better0.mean().round(4),' agency better at lambda=1 share',x.A_better1.mean().round(4),' crossings dist',x.ncross.value_counts().to_dict())
    nb=x[~x.A_better0]
    print('   lambda=0 merchant better: c range',nb.c.min(),nb.c.max(),' c/bound range',(nb.c/(2/(1+nb.h))).min().round(3) if len(nb) else None,(nb.c/(2/(1+nb.h))).max().round(3) if len(nb) else None)
    ab=x[x.A_better0]
    print('   lambda=0 agency better: c range',ab.c.min(),ab.c.max())
    cx=x[x.ncross==1]
    print('   crossing location: min',cx.lam_x.min().round(3),' max',cx.lam_x.max().round(3),' share below 0.5',(cx.lam_x<0.5).mean().round(4))
    above=cx[cx.lam_x>=0.5]
    if len(above): print('   crossings >=0.5 at c range',above.c.min(),above.c.max(),' h range',above.h.min(),above.h.max(),' l range',above.l.min(),above.l.max())
w.to_pickle('v2_flex_tw.pkl')
# threshold c for lambda=0 TW ranking, per h for H tier and per l for L tier
for t in ['H','L']:
    x=w[w.tier==t]
    key='h' if t=='H' else 'l'
    tab=[]
    for v,g in x.groupby(key):
        gg=g.groupby('c').A_better0.all()
        tab.append((v, gg[gg].index.min() if gg.any() else None, gg[gg].index.max() if gg.any() else None, (~x[x[key]==v].A_better0).sum()))
    print(t,'(value, min c with agency better, max c with agency better, n merchant better):',tab[:20])


# ======== rigid ========
import numpy as np, pandas as pd
LG=np.linspace(0,1,51)
g=pd.read_pickle('v2_rigid.pkl')
def crossings(x):
    s=np.sign(x); s=s[s!=0]; return int(np.sum(s[1:]!=s[:-1]))
out=[]
for i,R in g.iterrows():
    for t in ['H','L']:
        if not R[t+'_ok']: continue
        csg=R[t+'M_CS']-R[t+'A_CS']; omg=R[t+'A_Omega']-R[t+'M_Omega']; tw=R[t+'A_TW']
        k=int(np.argmax(tw))
        out.append(dict(h=R.h,l=R.l,c=R.c,r=R.r,tier=t,
            cs_A_better0=csg[0]<0, cs_M_better1=csg[-1]>0, cs_cross=crossings(csg),
            plat_M0=omg[0]<0, plat_A_any=np.any(omg>0), plat_A_cross=crossings(omg),
            tw_peak=LG[k], tw_interior=0<k<50, tw_drop=(tw[k]-tw[-1])/tw[k],
            pi_dec=np.all(np.diff(R[t+'A_piI'])<0), DI_pos=np.all(R[t+'A_DI']>0) and np.all(R[t+'A_DT']>0)))
o=pd.DataFrame(out); o.to_pickle('v2_rigid_cmp.pkl')
for t in ['H','L']:
    x=o[o.tier==t]
    print('==',t,' valid rows',len(x),' demands positive at all lambda:',x.DI_pos.mean(),' incumbent profit falls:',x.pi_dec.mean())
    print('  CS: agency better at 0:',x.cs_A_better0.mean().round(4),' merchant better at 1:',x.cs_M_better1.mean().round(4),' crossings:',x.cs_cross.value_counts().to_dict())
    print('  platform prefers merchant at 0:',x.plat_M0.mean().round(4),' agency preferred somewhere:',x.plat_A_any.mean().round(4),' crossings:',x.plat_A_cross.value_counts().to_dict())
    nm=x[~x.plat_M0]; 
    if len(nm): print('  agency preferred at lambda=0 at: r',sorted(nm.r.unique()),' c range',nm.c.min(),nm.c.max(),' h',nm.h.min(),nm.h.max(),' l',nm.l.min(),nm.l.max())
    print('  welfare interior peak share',x.tw_interior.mean().round(4),' max post-peak drop %',round(100*x.tw_drop.max(),3))
    for rv in [0.1,0.15,0.2,0.25,0.3]:
        y=x[np.isclose(x.r,rv)]
        print(f'   r={rv}: interior peak share {y.tw_interior.mean():.3f}; c range with interior peak {y[y.tw_interior].c.min() if y.tw_interior.any() else None}-{y[y.tw_interior].c.max() if y.tw_interior.any() else None}')


# ======== rigid2 ========
import numpy as np, pandas as pd
o=pd.read_pickle('v2_rigid_cmp.pkl')
def intervals(cs, flags):
    cs=np.array(cs); flags=np.array(flags); segs=[]; start=None
    for cv,fl in zip(cs,flags):
        if fl and start is None: start=cv
        if not fl and start is not None: segs.append((start,prev)); start=None
        prev=cv
    if start is not None: segs.append((start,prev))
    return segs
for t,key in [('H','h'),('L','l')]:
    x=o[o.tier==t]
    print('=====',t,'tier: c-intervals with an interior welfare peak, by r and',key)
    for rv in [0.1,0.15,0.2,0.25,0.3]:
        y=x[np.isclose(x.r,rv)]
        line=[]
        for v,z in y.groupby(key):
            z=z.groupby('c').tw_interior.all().sort_index()   # for L: same over h
            segs=intervals(z.index.values,z.values)
            line.append(f'{v}:{[(round(a,3),round(b,3)) for a,b in segs]}')
        print(f' r={rv}:',' | '.join(line[::2]))
# CS exceptions under H
x=o[(o.tier=='H')&(~o.cs_A_better0)]; print('H rows where merchant gives more CS already at lambda=0:'); print(x[['h','l','c','r']].drop_duplicates(['h','c','r']).to_string())
x=o[(o.tier=='H')&(~o.pi_dec)]; print('H rows where incumbent profit not strictly falling:'); print(x[['h','l','c','r']].drop_duplicates(['h','c','r']).to_string())


# ======== rigid3 ========
import numpy as np, pandas as pd
LG=np.linspace(0,1,51)
g=pd.read_pickle('v2_rigid.pkl'); o=pd.read_pickle('v2_rigid_cmp.pkl')
def crossings(x):
    s=np.sign(x); s=s[s!=0]; return int(np.sum(s[1:]!=s[:-1]))
# welfare peaks at 0?
print('welfare non-interior peaks located at:',o[~o.tw_interior].tw_peak.value_counts().to_dict())
# platform never prefers agency
for t in ['H','L']:
    x=o[(o.tier==t)&(~o.plat_A_any)]
    print(t,'never prefers agency: r',sorted(x.r.unique()),' c range',x.c.min(),x.c.max(),' h',x.h.min(),x.h.max(),' l',x.l.min(),x.l.max())
    y=o[(o.tier==t)&(~o.plat_M0)]
    for rv in sorted(y.r.unique()):
        z=y[np.isclose(y.r,rv)]
        print(f'   agency preferred at lambda=0, r={rv}: min c {z.c.min()}  (h {z.h.min()}-{z.h.max()}, l {z.l.min()}-{z.l.max()})')
# tier switch (Prop 4) over rows where both tiers valid
b=g[g.H_ok & g.L_ok].copy()
res=[]
for i,R in b.iterrows():
    gap=R.HA_Omega-R.LA_Omega
    cov=(R.LA_DT+R.LA_DI)-(R.HA_DT+R.HA_DI)
    res.append(dict(h=R.h,l=R.l,c=R.c,r=R.r,sw=(gap[0]>0) and (gap[-1]<0),ncross=crossings(gap),covL=np.all(cov>0)))
s=pd.DataFrame(res); sw=s[s.sw]
print('rows with both tiers valid:',len(s),'  switch rows:',len(sw),'  crossings among switch rows:',sw.ncross.value_counts().to_dict(),'  coverage larger under L at every lambda:',sw.covL.mean())
print('switch rows: h',sw.h.min(),sw.h.max(),' l',sw.l.min(),sw.l.max(),' c',sw.c.min(),sw.c.max(),' r',sorted(sw.r.unique()))
b13=sw[np.isclose(sw.h,1.3)]
print('at h=1.3: l',sorted(b13.l.unique()),' c',b13.c.min(),b13.c.max(),' r',sorted(b13.r.unique()))
s.to_pickle('v2_rigid_switch.pkl')


# ======== rigid4 ========
import numpy as np, pandas as pd
o=pd.read_pickle('v2_rigid_cmp.pkl')
o['cls']=np.where(o.tw_peak==0,'0',np.where(o.tw_peak==1,'1','i'))
def runs(cs,cl):
    out=[];cur=None
    for cv,k in zip(cs,cl):
        if cur is None or k!=cur[0]: 
            if cur: out.append(cur)
            cur=[k,cv,cv]
        else: cur[2]=cv
    out.append(cur); return ' '.join(f'{k}[{a:.3f},{b:.3f}]' for k,a,b in out)
for t,key,vals in [('H','h',[1.1,1.3,1.5,2.0]),('L','l',[0.5,0.7,0.9,0.95])]:
    x=o[o.tier==t]
    print('=====',t)
    for rv in [0.1,0.15,0.2,0.25,0.3]:
        for v in vals:
            y=x[np.isclose(x.r,rv)&np.isclose(x[key],v)]
            if t=='L': y=y[np.isclose(y.h,1.3)] if (np.isclose(y.h,1.3)).any() else y.drop_duplicates('c')
            else: y=y[np.isclose(y.l,0.9)]
            y=y.sort_values('c')
            print(f' r={rv} {key}={v}:',runs(y.c.values,y.cls.values))


# ======== support_table ========
import numpy as np, pandas as pd
LG=np.linspace(0,1,51)
f=pd.read_pickle('v2_flex.pkl'); d=pd.read_pickle('v2_flex_cmp.pkl'); w=pd.read_pickle('v2_flex_tw.pkl')
g=pd.read_pickle('v2_rigid.pkl'); o=pd.read_pickle('v2_rigid_cmp.pkl'); s=pd.read_pickle('v2_rigid_switch.pkl')
n=len(f); rows=[]
def pct(x): return f'{100*x:.2f}%'
inc=lambda x: np.all(np.diff(x)>0); dec=lambda x: np.all(np.diff(x)<0)
def share(fun): return np.mean([fun(R) for _,R in f.iterrows()])
rows.append(('F1','Optimal commission strictly decreases in the intensity (proved in Lemma 3)', n, pct(share(lambda R: dec(R.HA_r) and dec(R.LA_r))), 'none'))
rows.append(('F2','Incumbent margin and profit strictly increase in the intensity, both tiers', n, pct(share(lambda R: all(inc(R[t+'A_piI']) and inc((1-R[t+'A_r'])*R[t+'A_pI']-R.c/2) for t in 'HL'))), 'none'))
rows.append(('F3','Total welfare strictly increases in the intensity, both tiers', n, pct(share(lambda R: inc(R.HA_TW) and inc(R.LA_TW))), 'none'))
fg=pd.read_pickle('flex_gain.pkl')  # written by profit_gain_flex.py, intensities in steps of 0.0005
pk=fg.lam_peak.max(); gn=fg.gain.max(); en=fg.lam_end.max()
falls=share(lambda R: all(dec(R[t+'A_piR'][np.argmax(R[t+'A_piR']):]) for t in 'HL' if np.argmax(R[t+'A_piR'])<50))
rows.append(('F4',f'Platform profit peaks at an intensity of at most {pk:.3f} with a gain of at most {100*gn:.3f}%, falls thereafter, and is below its zero-intensity level from an intensity of at most {en:.3f}', n, pct(falls*float(fg.gain.min()>0)*float(fg.lam_end.notna().all())), 'none'))
r0=np.array([min(R.HA_r[0],R.LA_r[0]) for _,R in f.iterrows()])
rows.append(('F5','Optimal commission at zero intensity is at least 0.4', n, pct(np.mean(np.array([R.HA_r[0] for _,R in f.iterrows()])>=0.4)*0+np.mean([ (R.HA_r[0]>=0.4) and (R.LA_r[0]>=0.4) for _,R in f.iterrows()])), 'only where c > 0.8'))
rows.append(('F6','Tier under flexible agency equals merchant tier at every intensity', n, pct(d.agree_all.mean()), 'only where the two merchant tiers are within 10% in objective value'))
rows.append(('F7','Agency tier does not change with the intensity', n, pct(1-d.tierA_changes.mean()), 'same set as F6; switches in both directions'))
rows.append(('F8','Platform objective and profit higher under agency at every intensity', n, pct(d.plat_agency_obj.mean()*d.plat_agency_profit.mean()), 'none'))
rows.append(('F9','Consumer surplus higher under merchant at every intensity (preferred tiers)', n, pct(d.cons_merchant.mean()), 'none'))
rows.append(('F10','Incumbent profit higher under merchant at every intensity (preferred tiers)', n, pct(d.inc_merchant.mean()), 'only where the formats lead to different tiers; within a tier 100%'))
rows.append(('F11','Consumer-surplus gap between formats widens in the intensity (preferred tiers)', n, pct(d.csgap_widens.mean()), 'only where the agency tier changes with the intensity'))
xH=w[w.tier=='H']; xL=w[w.tier=='L']
rows.append(('F12','High-end: total-welfare ranking of formats crosses at most once, below 0.5', len(xH), pct(np.mean((xH.ncross<=1)&((xH.ncross==0)|(xH.lam_x<0.5)))), 'none'))
rows.append(('F13','Economy: total-welfare ranking of formats crosses at most once', len(xL), pct(np.mean(xL.ncross<=1)), 'c <= 0.15 with l >= 0.9'))
oH=o[o.tier=='H']; oL=o[o.tier=='L']; sw=s[s.sw]
rows.append(('R1','Tier switch (high-end to economy): positioning gap crosses zero exactly once where sign conditions hold', len(sw), pct(np.mean(sw.ncross==1)), 'none'))
rows.append(('R2','Tier switch (high-end to economy): coverage larger under economy integration at every intensity', len(sw), pct(sw.covL.mean()), 'none'))
rows.append(('R3','Consumer-surplus gap between formats crosses zero exactly once (high-end)', len(oH), pct(np.mean(oH.cs_cross==1)), 'h = 1.1, r = 0.3, c at its upper bound'))
rows.append(('R4','Consumer-surplus gap between formats crosses zero exactly once (economy)', len(oL), pct(np.mean(oL.cs_cross==1)), 'none'))
rows.append(('R5','Platform prefers merchant at zero intensity (high-end)', len(oH), pct(oH.plat_M0.mean()), 'r >= 0.25 and c > 0.65'))
rows.append(('R6','Platform prefers merchant at zero intensity (economy)', len(oL), pct(oL.plat_M0.mean()), 'r >= 0.25 and c > 0.8'))
rows.append(('R7','Platform prefers agency at some intensity (high-end)', len(oH), pct(oH.plat_A_any.mean()), 'c <= 0.2 with h <= 1.2'))
rows.append(('R8','Platform prefers agency at some intensity (economy)', len(oL), pct(oL.plat_A_any.mean()), 'l = 0.95, c = 0.95, r = 0.1'))
rows.append(('R9','Welfare decline from peak to parity below 2% (both tiers)', len(o), pct(np.mean(o.tw_drop<0.02)), 'none'))
# reverse tier switch under the rigid commission: economy at zero intensity, high-end at parity
gb=g[g.H_ok & g.L_ok]
gap0=np.array([R.HA_Omega[0]-R.LA_Omega[0] for _,R in gb.iterrows()]); gap1=np.array([R.HA_Omega[-1]-R.LA_Omega[-1] for _,R in gb.iterrows()])
rv=gb[(gap0<0)&(gap1>0)]
ncr=np.array([len(np.where(np.diff(np.sign(R.HA_Omega-R.LA_Omega))!=0)[0]) for _,R in rv.iterrows()])
rows.append(('R10','Reverse tier switch (economy to high-end): positioning gap crosses zero exactly once where it is negative at zero intensity and positive at parity', len(rv), pct(np.mean(ncr==1)), 'none'))
rows.append(('R11','Reverse tier switch: consumer surplus at parity higher under high-end integration', len(rv), pct(np.mean([R.HA_CS[-1]>R.LA_CS[-1] for _,R in rv.iterrows()])), 'only at the upper end of the quality-cost range of the switch, c >= 0.65 with r >= 0.15'))
rows.append(('R12','Reverse tier switch: coverage at parity larger under economy integration', len(rv), pct(np.mean([R.LA_DT[-1]+R.LA_DI[-1]>R.HA_DT[-1]+R.HA_DI[-1] for _,R in rv.iterrows()])), 'h <= 1.15 with c <= 0.225'))
t=pd.DataFrame(rows,columns=['id','observation','points','support','exceptions'])
t.to_csv('support_table.csv',index=False)
print(t.to_markdown(index=False))
print('flexible points',n,' rigid rows',len(g),' H valid',int(g.H_ok.sum()),' L valid',int(g.L_ok.sum()),' both',int((g.H_ok&g.L_ok).sum()))
