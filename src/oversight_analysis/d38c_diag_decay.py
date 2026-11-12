"""d38c 사후 진단 (사전 등록 아님): H1 위반의 원인 확인.
d38c H1은 'ρ=0에서 노출만(A)은 감시 없음(N)과 같다'를 기대했다. 실제로는 0.7778 칸에서 달랐다.
가설: 습관화 규칙이 '맞은 AI 제안'에만 v를 줄이므로, canary(틀린 제안)가 맞은 제안을 대체해 습관화를 늦춘 것이다(코드 오류 아님).
확인: 습관화를 모든 항목에 적용하는 변형에서 같은 비교를 한다. 변형에서 ρ=0 효과가 사라지면 가설이 맞다.
사용: python -X utf8 d38c_diag_decay.py [출력폴더]     (d38c_recovery_assumption.py 와 같은 폴더에서, 약 1분)
"""
import sys, os, csv, math, itertools
import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else "d38c_out"
os.makedirs(OUT, exist_ok=True)
here = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(here, "d38c_recovery_assumption.py"), encoding="utf-8").read().split("if __name__")[0]
OLD = "v = np.where(~wrong, v * (1 - lam), v)"
assert OLD in src
variants = {"원안(맞은 제안에서만 습관화)": src, "변형(모든 항목에서 습관화)": src.replace(OLD, "v = v * (1 - lam)")}
rows = []
for name, code in variants.items():
    sys.argv = ["x", os.path.join(OUT, "_tmp")]
    ns = {}; exec(compile(code, "d38c", "exec"), ns)
    sim = ns["simulate"]
    for rho in (0.0, 1.0):
        bad = tot = 0; effs = []
        for k, (lam, e, v0, r) in enumerate(itertools.product(ns["LAM"], ns["E"], ns["V0"], ns["RATE"])):
            N = sim(lam, e, 0.0, v0, "none", np.random.default_rng([1, k]), 1000, 1800, rho=rho, alerts=False)
            A = sim(lam, e, r, v0, "none", np.random.default_rng([2, k]), 1000, 1800, rho=rho, alerts=False)
            eff = N["wf"] - A["wf"]; se = math.hypot(N["wf_se"], A["wf_se"])
            tot += 1; bad += abs(eff) > 2 * se; effs.append(eff)
        rows.append(dict(습관화=name, rho=rho, 칸수=tot, 효과_2SE초과_비율=round(bad / tot, 3),
                         효과_중앙값=round(float(np.median(effs)), 2), 효과_최대=round(max(effs), 2)))
        print(rows[-1])
try:
    os.rmdir(os.path.join(OUT, "_tmp"))
except OSError:
    pass
with open(os.path.join(OUT, "d38c_diag_decay.csv"), "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("저장:", OUT)
