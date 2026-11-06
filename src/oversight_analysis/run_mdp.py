import time, numpy as np, pandas as pd, mdp_alpha as M
M.BMAX=900; M.GAM=0.99
t0=time.time(); out=[]
def runs(pi,C,kappas,bound="point"):
    res=[]
    for k in kappas:
        m,pm,pol=M.solve(pi,C,k,bound)
        e=M.evaluate(m,pm,pol,C)
        # 정책 요약: 대기량 구간별 α
        segs=[]; start=0
        for b in range(1,M.BMAX+2):
            if b==M.BMAX+1 or pol[b]!=pol[start]:
                if start<=M.BMAX: segs.append((start,b-1,float(m.alpha[pol[start]])))
                start=b
        res.append((k,e,segs))
    return m,res
for pi,C,label in [(0.10,450,"π=.10, 3명×150"),(0.10,600,"π=.10, 4명×150"),(0.10,300,"π=.10, 2명×150"),(0.25,450,"π=.25, 3명×150")]:
    m,res=runs(pi,C,[1,10,100,1000])
    print("\n###",label,"C=",C,"  (고정 α=0.05 기준:",end=" ")
    fx=M.evaluate(m,*[None]*0,pol_or_fixed=3,C=C) if False else None
    pm=[]
    from scipy.stats import binom
    for _,r in m.iterrows():
        p=r.load; lo=int(binom.ppf(1e-6,M.N,p)); hi=int(binom.ppf(1-1e-6,M.N,p)); A=np.arange(lo,hi+1); w=binom.pmf(A,M.N,p); pm.append((A,w/w.sum()))
    e=M.evaluate(m,pm,3,C); print({k:round(v,3) for k,v in e.items()},")")
    for k,e,segs in res:
        print(f" κ={k:<5}", {kk:round(v,3) for kk,v in e.items()}, " 정책(대기량→α):", [(a,b,al) for a,b,al in segs][:6])
print("time",round(time.time()-t0,1))
