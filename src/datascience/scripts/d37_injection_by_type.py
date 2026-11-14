"""d37: 인젝션 결과를 공격 유형별로 나눠 최악 유형과 Clopper-Pearson 95% 구간을 보고한다(멘토링 11/13 반영).
입력 outputs/injection_results_axft.csv(집계용, 원문 없음) → outputs/injection_by_type_axft.csv, injection_by_type_axft.json"""
import os, json, pandas as pd
from scipy.stats import beta
B = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); O = os.path.join(B, "outputs")
d = pd.read_csv(os.path.join(O, "injection_results_axft.csv"))
def cp(k, n): return (0.0 if k == 0 else beta.ppf(.025, k, n - k + 1), 1.0 if k == n else beta.ppf(.975, k + 1, n - k))
rows = []
for scope, sub in [("전체", d), ("비민감", d[d.route0 != "sensitive"])]:
    for t, g in list(sub.groupby("inj_type")) + [("합계", sub)]:
        k, n = int(g.changed.sum()), len(g); lo, hi = cp(k, n)
        ch = g[g.changed]
        rows.append(dict(scope=scope, inj_type=t, n=n, changed=k, rate=round(k / n, 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                         to_pass=int((ch.route1_final == "pass").sum()), to_hold=int((ch.route1_final == "hold").sum()),
                         format_violation=int(g.format_violation.sum())))
r = pd.DataFrame(rows); r.to_csv(os.path.join(O, "injection_by_type_axft.csv"), index=False, encoding="utf-8-sig")
pos = d.groupby(["inj_type", "position"]).changed.agg(["sum", "count"]).reset_index()
w = r[(r.scope == "전체") & (r.inj_type != "합계")].sort_values("rate").iloc[-1]
S = dict(worst_type=w.inj_type, worst_rate=w.rate, worst_ci=[w.ci_lo, w.ci_hi], pooled=float(r[(r.scope == "전체") & (r.inj_type == "합계")].rate.iloc[0]),
         by_position=pos.to_dict("records"),
         note="변경 건 중 자동 통과로 바뀐 건(to_pass)이 위험 방향, 판단 보류(hold)는 안전 방향. 시험 5계열은 미세조정 증강(d18)에도 같은 계열(문구 다름)이 있어 새 계열은 미시험.")
json.dump(S, open(os.path.join(O, "injection_by_type_axft.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=int)
print(r.to_string(index=False)); print(json.dumps({k: S[k] for k in ["worst_type", "worst_rate", "worst_ci", "pooled"]}, ensure_ascii=False))
