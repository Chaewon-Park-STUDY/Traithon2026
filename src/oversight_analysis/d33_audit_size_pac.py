# d33 (a) 감사 표본 크기의 ε-δ 비교(CLT, Bernstein, Chebyshev, Hoeffding, exact) (b) 보정 표본 변동: 실현 놓침률의 분포와 PAC형 α 조정. 규칙: PREREG_d33_audit_size_pac.md
# 데이터 없이도 돈다. outputs 폴더를 주면 보정 표본 수(n_calib)를 split_manifest에서 읽는다. 출력은 집계뿐.
# 사용: python d33_audit_size_pac.py [outputs 폴더] [출력폴더=d33_out]
import os, sys, math
import numpy as np, pandas as pd
from scipy.stats import beta, norm, binom
DELTA = 0.05; EPS_GRID = [0.01, 0.02, 0.03, 0.05]; M_GRID = [0.02, 0.0307, 0.05, 0.10]; PI_GRID = [0.05, 0.10, 0.25]; PN = 0.9571   # PN=관측 정상->자동 통과율(d32 출력)
ALPHA = 0.05

def n_clt(m, e, d=DELTA): return math.ceil(norm.ppf(1 - d / 2) ** 2 * m * (1 - m) / e ** 2)
def n_cheb(m, e, d=DELTA): return math.ceil(m * (1 - m) / (d * e ** 2))
def n_hoef(e, d=DELTA): return math.ceil(math.log(2 / d) / (2 * e ** 2))
def n_bern(m, e, d=DELTA): return math.ceil((2 * m * (1 - m) / e ** 2 + 2 / (3 * e)) * math.log(2 / d))   # |X-m|<=1
def n_exact(m, e, d=DELTA, cap=20000):
    """k=round(n*m)일 때 Clopper-Pearson 양측 구간의 반폭(큰 쪽)이 e 이하가 되는 최소 n (대표값 기준이며 모든 k에 대한 보장은 아님)"""
    for n in range(10, cap):
        k = int(round(n * m)); lo = 0.0 if k == 0 else beta.ppf(d / 2, k, n - k + 1); hi = 1.0 if k == n else beta.ppf(1 - d / 2, k + 1, n - k)
        if max(hi - k / n, k / n - lo) <= e: return n
    return float("nan")
def n_exact_worst(e, d=DELTA, cap=20000):
    """모든 k에 대해 CP 반폭이 e 이하가 되는 최소 n (최악 k, 보장형). 이분 탐색이 아니라 단조가 아니므로 뒤에서부터 확인"""
    ok_from = None
    for n in range(10, cap, 1):
        w = 0.0
        for k in range(0, n + 1, max(1, n // 200)):
            lo = 0.0 if k == 0 else beta.ppf(d / 2, k, n - k + 1); hi = 1.0 if k == n else beta.ppf(1 - d / 2, k + 1, n - k); w = max(w, hi - k / n, k / n - lo)
        if w <= e: return n
    return float("nan")
def adj_alpha(n, step=0.0005):
    """P(실현 놓침 <= ALPHA) >= 1-DELTA 가 되도록 하는, ALPHA 이하의 가장 큰 보정용 α' (step 격자). 없으면 nan"""
    x = ALPHA
    while x > step / 2:
        l = math.floor((n + 1) * x)
        if l >= 1 and beta.ppf(1 - DELTA, l, n + 1 - l) <= ALPHA: return round(x, 4)
        x -= step
    return float("nan")
def kappa(pi, pn, m): return pi * m / (pi * m + (1 - pi) * pn)

def main(outputs=None, out="d33_out"):
    os.makedirs(out, exist_ok=True); rows = []
    for e in EPS_GRID:
        for m in M_GRID:
            r = dict(eps=e, m=m, CLT=n_clt(m, e), Bernstein=n_bern(m, e), Chebyshev=n_cheb(m, e), Hoeffding=n_hoef(e), Exact_CP=n_exact(m, e))
            for pi in PI_GRID: r[f"감사건수/κ π={pi}"] = math.ceil(r["Bernstein"] / kappa(pi, PN, m))
            rows.append(r)
    A = pd.DataFrame(rows); A.to_csv(os.path.join(out, "d33_audit_size.csv"), index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(f"[감사에 필요한 클릭베이트 건수 (δ={DELTA}, 반폭 ε)] 방법별. 오른쪽 열은 자동 통과 풀에서 뽑아야 하는 기사 수 = Bernstein 건수 / κ (PN={PN}, κ=π·m/(π·m+(1-π)·PN))."); print(A.to_string(index=False))
    print("해석: CLT는 정규근사라 작은 m에서 과소하고, Hoeffding은 분포 가정 없이 m(1-m)를 쓰지 않아 가장 크다. Bernstein은 분산을 쓰는 분포 무관 보장이라 둘 사이다. 보장이 필요하면 Bernstein 이상, 계획용 대표값은 Exact_CP.")
    # 보장형 exact (최악 k) 몇 개만
    W = pd.DataFrame([dict(eps=e, exact_worstcase_n=n_exact_worst(e)) for e in (0.03, 0.05)]); print("\n[최악 k에서도 CP 반폭 ≤ ε인 n (m을 모를 때)]"); print(W.to_string(index=False)); W.to_csv(os.path.join(out, "d33_exact_worst.csv"), index=False)
    # (b) 보정 표본 변동
    ns = {"전체 Direct(예시)": 391, "auto_check(예시)": 388, "category당(예시)": 78, "category당 Direct(예시)": 55}
    if outputs and os.path.exists(os.path.join(outputs, "split_manifest.csv")):
        man = pd.read_csv(os.path.join(outputs, "split_manifest.csv")); ns = {}
        c = man[(man.subset == "calib") & (man.label == 1) & ~man.category.isin(["정치", "경제"])]; a = man[(man.subset == "calib_auto_check") & (man.label == 1) & ~man.category.isin(["정치", "경제"])]
        ns["calib 클릭베이트(비민감)"] = len(c); ns["auto_check 클릭베이트(비민감)"] = len(a)
        for g, v in c.groupby("category").size().items(): ns[f"calib {g}"] = int(v)
    rows = []
    for nm_, n in ns.items():
        l = math.floor((n + 1) * ALPHA)
        if l < 1: rows.append({"대상": nm_, "n": n, "비고": "n+1 < 1/α 라서 qhat=1.0(전부 검토)"}); continue
        q_hi = float(beta.ppf(1 - DELTA, l, n + 1 - l)); q_lo = float(beta.ppf(DELTA, l, n + 1 - l))
        a_adj = adj_alpha(n)
        rows.append({"대상": nm_, "n": n, "l": l, "기대_놓침": round(l / (n + 1), 4), "실현_놓침_5분위": round(q_lo, 4), "실현_놓침_95분위": round(q_hi, 4), "P(놓침<=α)": round(float(beta.cdf(ALPHA, l, n + 1 - l)), 3), "조정_α(95% 보장)": a_adj})
    P = pd.DataFrame(rows); P.to_csv(os.path.join(out, "d33_calib_pac.csv"), index=False, encoding="utf-8-sig")
    print(f"\n[보정 표본 변동] 보정이 주어졌을 때 실현 놓침률 M ~ Beta(l, n+1-l), l=floor((n+1)α). α={ALPHA}로 보정해도 M이 α 이하일 확률은 보장되지 않는다(평균만 ≤α)."); print(P.to_string(index=False))
    print("해석: 'P(놓침<=α)'이 0.95보다 작으면, 한 번 뽑은 보정 표본에서 실현 놓침이 α를 넘을 수 있다. 조정_α는 그 확률을 0.95로 맞추려고 보정에 쓰는 더 작은 α이며, 그만큼 검토량이 늘어난다(d28 곡선에서 읽는다). maxq는 Direct·Auto 각각을 이 식으로 본다.")
if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else None, sys.argv[2] if len(sys.argv) > 2 else "d33_out")
