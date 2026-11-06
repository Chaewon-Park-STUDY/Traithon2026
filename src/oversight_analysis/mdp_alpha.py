# 소형 MDP: 하루 단위로 α(보정 수준)를 대기량(backlog)에 따라 고르는 정책을 푼다.
# 입력은 레포 results/alpha_sweep_axft.csv의 α별 (자동통과율 a+, a-)와 표본 수(비민감 클릭베이트 1370, 정상 955)뿐.
import numpy as np, pandas as pd, math, json
from scipy.stats import beta, binom
import os
R_ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datascience", "results") + os.sep
sw = pd.read_csv(R_+"alpha_sweep_axft.csv", encoding="utf-8-sig"); sw = sw[sw["mode"]=="maxq"].reset_index(drop=True)
S, N, NCB, NOK = 0.278, 1000, 1370, 955
def cp(k,n,a=0.05):
    lo = 0.0 if k==0 else beta.ppf(a/2,k,n-k+1); hi = 1.0 if k==n else beta.ppf(1-a/2,k+1,n-k); return lo,hi
def menu(pi, bound="point"):
    rows=[]
    for _,r in sw.iterrows():
        ap, an = r.autopass_cb_nonsens, r.autopass_ok_nonsens
        kcb, kok = round(ap*NCB), round(an*NOK)
        if bound=="worst":   # 부하는 크게(자동통과율 하한), 미포착은 크게(미포착률 상한)
            ap_load, an_load = cp(kcb,NCB)[0], cp(kok,NOK)[0]; ap_miss = cp(kcb,NCB)[1]
        else: ap_load, an_load, ap_miss = ap, an, ap
        load = S+(1-S)*(pi*(1-ap_load)+(1-pi)*(1-an_load)); miss=(1-S)*N*pi*ap_miss
        rows.append((r.alpha, load, miss))
    return pd.DataFrame(rows, columns=["alpha","load","miss"])
BMAX=1200; GAM=0.995
def solve(pi, C, kappa, bound="point"):
    m = menu(pi,bound); b = np.arange(BMAX+1); V=np.zeros(BMAX+1); K=len(m)
    pm = []
    for _,r in m.iterrows():
        p=r.load; lo=int(binom.ppf(1e-6,N,p)); hi=int(binom.ppf(1-1e-6,N,p)); A=np.arange(lo,hi+1); pm.append((A, binom.pmf(A,N,p)/binom.pmf(A,N,p).sum()))
    for it in range(4000):
        Q=np.zeros((BMAX+1,K))
        for k,(A,w) in enumerate(pm):
            nb=np.clip(b[:,None]+A[None,:]-C,0,BMAX)
            Q[:,k]=kappa*m.miss[k] + (w[None,:]*(nb+GAM*V[nb])).sum(1)
        Vn=Q.min(1)
        if np.abs(Vn-V).max()<1e-6*max(1,np.abs(V).max()): V=Vn; break
        V=Vn
    pol=Q.argmin(1); return m,pm,pol
def evaluate(m,pm,pol_or_fixed,C):
    pol = pol_or_fixed if hasattr(pol_or_fixed,"__len__") else np.full(BMAX+1,pol_or_fixed)
    P=np.zeros((BMAX+1,BMAX+1)); b=np.arange(BMAX+1)
    for s in b:
        A,w=pm[pol[s]]; nb=np.clip(s+A-C,0,BMAX); np.add.at(P[s],nb,w)
    d=np.ones(BMAX+1)/(BMAX+1)
    for _ in range(20000):
        d2=d@P
        if np.abs(d2-d).sum()<1e-12: d=d2;break
        d=d2
    miss=(d*np.array([m.miss[pol[s]] for s in b])).sum()
    return dict(avg_backlog=float((d*b).sum()), p_backlog_gt0=float(d[1:].sum()), p_backlog_gt_C=float(d[C+1:].sum()), miss_per_day=float(miss), p_saturated=float(d[-20:].sum()))
if __name__=="__main__":
    for pi in (0.10,):
        print("=== α 메뉴 (π=%.2f) 점추정 / 최악구간"%pi); mp, mw = menu(pi), menu(pi,"worst")
        print(pd.concat([mp, mw[["load","miss"]].add_suffix("_worst")],axis=1).round(3).to_string())
