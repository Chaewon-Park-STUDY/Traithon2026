# Interval-censored time-to-discovery: how wide is the uncertainty? (scenario simulation, NOT an estimate)
import numpy as np
from scipy.optimize import minimize
rng=np.random.default_rng(1)

def sim(N,k,med,delta,W):
    lam=med/np.log(2)**(1/k)
    T=lam*rng.weibull(k,N)
    L=np.floor(T/delta)*delta; R=L+delta
    obs=T<=W
    # items not discovered by W: right-censored at W (L=W,R=inf). Observed ones: interval (L,R] clipped to W
    L=np.where(obs,L,W); R=np.where(obs,np.minimum(R,np.inf),np.inf)
    return L,R,obs

def nll(p,L,R):
    k=np.exp(p[0]); lam=np.exp(p[1])
    S=lambda t: np.exp(-(t/lam)**k)
    SL=S(L); SR=np.where(np.isinf(R),0.0,S(np.where(np.isinf(R),1.0,R)))
    return -np.sum(np.log(np.clip(SL-SR,1e-300,None)))

def fit(L,R):
    best=None
    for k0 in (0.8,1.2):
        r=minimize(nll,[np.log(k0),np.log(40.)],args=(L,R),method='Nelder-Mead')
        if best is None or r.fun<best.fun: best=r
    k=np.exp(best.x[0]); lam=np.exp(best.x[1])
    return lam*np.log(2)**(1/k), lam*np.log(10)**(1/k)   # median, q90

def run(N,k,med,delta,W,reps=300):
    out=[];mid=[]
    for _ in range(reps):
        L,R,obs=sim(N,k,med,delta,W)
        if obs.sum()<5: continue
        out.append(fit(L,R))
        # naive: midpoint of interval among observed only, sample median
        m=(L[obs]+R[obs])/2; mid.append(np.median(m))
    o=np.array(out); 
    return o[:,0], np.array(mid), len(o)

print("scenario: Weibull shape k, true median 30d, follow-up W=180d; estimand = median time-to-discovery")
print("N_items k   Δ(d)  events~  IC-MLE: bias%  90%range(rel.width%) | naive-midpoint(observed only): bias%")
for N in (30,100,300):
  for k in (0.8,1.5):
    for delta in (1,7,30):
        m,mid,n=run(N,k,30,delta,180)
        ev=int(N*(1-np.exp(-np.log(2)*(180/30)**k)))
        lo,hi=np.percentile(m,[5,95])
        print(f"{N:5d}   {k:.1f}  {delta:3d}  {ev:5d}   {100*(np.median(m)/30-1):+6.1f}   {100*(hi-lo)/30:6.1f}            | {100*(np.median(mid)/30-1):+6.1f}")
