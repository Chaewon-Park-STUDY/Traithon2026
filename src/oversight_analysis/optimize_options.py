# 검토 용량 선택지 비교: (1) 정적 전수 탐색(인력 R × 이관 범위 s × α), (2) 동적 정책 비교(MDP 최적 vs greedy vs 고정 α)
# 입력: results/alpha_sweep_axft.csv 집계값. N, π, c(1인 처리량), s, κ는 시나리오 값이다.
import sys, time, numpy as np, pandas as pd
import mdp_alpha as M

PIS = (0.05, 0.10, 0.25)
CS  = (100, 150, 200)          # 1인 일 처리량 시나리오(건/인/일). 150이 예선 A4 가정, 실측은 파일럿에서
RS  = (2, 3, 4, 5, 6)
SCOPES = {"정치·경제 전량(현행 s=0.278)": 0.278,
          "정치만 전량(s≈0.139, 시험 구성비 근사·정책 변경)": 0.139}   # 후자는 a+, a−를 고정한 근사

def static():
    rows = []
    for sname, s in SCOPES.items():
        M.S = s
        for pi in PIS:
            mp, mw = M.menu(pi), M.menu(pi, "worst")
            for c in CS:
                for R in RS:
                    C = c * R
                    ok = mw.load * M.N <= C                    # 부하 상한(CP)으로 용량 만족
                    if ok.any():
                        i = mw[ok].miss.idxmin()               # 만족하는 α 중 미포착 상한이 가장 작은 것
                        rows.append(dict(scope=sname, pi=pi, c=c, R=R, C=C, feasible=True, alpha=mw.alpha[i],
                                         load_point=round(mp.load[i]*M.N), load_worst=round(mw.load[i]*M.N),
                                         miss_point=round(mp.miss[i], 2), miss_worst=round(mw.miss[i], 2)))
                    else:
                        rows.append(dict(scope=sname, pi=pi, c=c, R=R, C=C, feasible=False, alpha=None,
                                         load_point=round(mp.load.min()*M.N), load_worst=round(mw.load.min()*M.N),
                                         miss_point=None, miss_worst=None))
    return pd.DataFrame(rows)

def dynamic(pi, C, kappas=(1, 10, 100, 1000)):
    M.S = 0.278; M.BMAX = 500; M.GAM = 0.99
    out = []
    for kappa in kappas:
        m, pm, pol = M.solve(pi, C, kappa)
        res = {"MDP 최적": M.evaluate(m, pm, pol, C)}
        g = M.GAM; M.GAM = 0.0
        m, pm, polg = M.solve(pi, C, kappa); M.GAM = g
        res["greedy(당일 최선)"] = M.evaluate(m, pm, polg, C)
        k5 = int((m.alpha - 0.05).abs().idxmin())
        res["고정 α=0.05"] = M.evaluate(m, pm, k5, C)
        for name, r in res.items():
            out.append(dict(pi=pi, C=C, kappa=kappa, policy=name, **{k: round(v, 3) for k, v in r.items()}))
    return pd.DataFrame(out)

if __name__ == "__main__":
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    st = static()
    print("=== (1) 정적 전수 탐색: 용량 제약(부하 상한 ≤ C)을 만족하는 α 중 미포착 상한 최소 ===")
    print(st.to_string(index=False))
    print("\n=== 필요 최소 인력 R*(만족하는 가장 작은 R) ===")
    f = st[st.feasible]
    print(f.groupby(["scope", "pi", "c"]).R.min().unstack("c").to_string())
    if "--dynamic" in sys.argv:
        t0 = time.time()
        print("\n=== (2) 동적 정책 비교(안정 영역만: 최소 부하 < C) ===")
        d = pd.concat([dynamic(0.10, 450), dynamic(0.25, 600)])
        print(d.to_string(index=False)); print("time", round(time.time()-t0))
