"""D7 보강: α 민감도, 검토 용량, 카테고리 조건부 보정 비교 (BaseKit 1 A1~A5, BaseKit 6 α 결정 규칙, BaseKit 5 §5-4).
- α 결정 규칙: 검토 부하 r(α; π)·N ≤ C 를 만족하는 가장 작은 α (BaseKit 6 6-0).
- 검토 부하 = 정치·경제 비율 s + (1-s)·[π·(1-자동통과₊) + (1-π)·(1-자동통과₋)]  (확정 후보도 사람 승인이 필요하므로 부하에 포함)
- 가정: N=1,000/일(A1), C=300/일(A4: 150건×2명), π∈{0.05,0.10,0.25}(A2), s=0.278(BaseKit 1 Training 구성비)
사용법: python d9_sensitivity.py <name>
"""
import os, sys, math, json
import numpy as np, pandas as pd
from scipy.stats import beta
from common import OUT, SENSITIVE, clopper_pearson, save_json, calib_min_n

name = sys.argv[1]
N, C, S, PIS, PER = 1000, 300, 0.278, [0.05, 0.10, 0.25], 150
ALPHAS = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30]
man = pd.read_csv(os.path.join(OUT, "split_manifest.csv"))
d = man.merge(pd.read_csv(os.path.join(OUT, f"scores_{name}.csv")), on="news_id")
cal_cb = d[(d.subset == "calib") & (d.label == 1)]
cal_ok = d[(d.subset == "calib") & (d.label == 0)]
cal_auto = d[d.subset == "calib_auto_check"]
test = d[d.subset == "test"]


def qhat(s, a):
    n = len(s); k = math.ceil((n + 1) * (1 - a))
    return float(np.sort(s)[k - 1]) if k <= n else 1.0


def decide(df, q_cb, q_ok):
    inc_cb, inc_ok = (1 - df.p_cb) <= q_cb, df.p_cb <= q_ok
    return np.select([inc_cb & ~inc_ok, ~inc_cb & inc_ok], ["cb", "normal"], "ambiguous")


def metrics(t, sets):
    t = t.assign(set=sets)
    ns = t[~t.category.isin(SENSITIVE)]
    out = {}
    for lab, k in [("direct", (t.label == 1) & (t.process_type == "Direct")), ("auto", (t.label == 1) & (t.process_type == "Auto"))]:
        sub = t[k]; c = int((sub.set != "normal").sum())
        out[f"cov_{lab}"], out[f"cov_{lab}_lo"] = c / len(sub), clopper_pearson(c, len(sub))[0]
    ok = t[t.label == 0]
    out["false_confirm"] = float((ok.set == "cb").mean())
    ap_pos = float((ns[ns.label == 1].set == "normal").mean())   # 비민감 클릭베이트 자동 통과율(=미포착)
    ap_neg = float((ns[ns.label == 0].set == "normal").mean())   # 비민감 정상 자동 통과율
    out["autopass_cb_nonsens"], out["autopass_ok_nonsens"] = ap_pos, ap_neg
    for pi in PIS:
        load = S + (1 - S) * (pi * (1 - ap_pos) + (1 - pi) * (1 - ap_neg))
        out[f"load_pi{pi}"] = load
        out[f"reviewers_pi{pi}"] = math.ceil(load * N / PER)
        out[f"missed_per_day_pi{pi}"] = (1 - S) * N * pi * ap_pos   # 사람을 거치지 않고 통과하는 클릭베이트 건수/일
    return out


rows = []
for a in ALPHAS:
    q_ok = qhat(cal_ok.p_cb.values, a)
    q_dir = qhat(1 - cal_cb.p_cb.values, a)
    q_aut = qhat(1 - cal_auto.p_cb.values, a)
    for mode, q_cb in [("direct", q_dir), ("maxq", max(q_dir, q_aut))]:
        rows.append(dict(alpha=a, mode=mode, q_cb=q_cb, q_ok=q_ok, **metrics(test, decide(test, q_cb, q_ok))))
sw = pd.DataFrame(rows)
sw.to_csv(os.path.join(OUT, f"alpha_sweep_{name}.csv"), index=False, encoding="utf-8-sig")

# 카테고리 조건부(maxq): 카테고리별로 q̂ 계산
a = 0.05
cat_rows, sets = [], pd.Series(index=test.index, dtype=object)
for c in sorted(test.category.unique()):
    cc, ca, co = cal_cb[cal_cb.category == c], cal_auto[cal_auto.category == c], cal_ok[cal_ok.category == c]
    q_cb = max(qhat(1 - cc.p_cb.values, a), qhat(1 - ca.p_cb.values, a)); q_ok = qhat(co.p_cb.values, a)
    tc = test[test.category == c]
    sets.loc[tc.index] = decide(tc, q_cb, q_ok)
    n = min(len(cc), len(ca), len(co)); l = math.floor((n + 1) * a)
    p_guar = float(beta.sf(1 - a - 0.02, n + 1 - l, l)) if l >= 1 else 0.0
    cb = tc[tc.label == 1]; k = int((sets.loc[cb.index] != "normal").sum())
    cat_rows.append(dict(category=c, n_calib_min=n, p_cov_ge_093=round(p_guar, 3), cov_cb=k / len(cb), cov_cb_lo=clopper_pearson(k, len(cb))[0]))
cat = pd.DataFrame(cat_rows)
glob = pd.read_csv(os.path.join(OUT, f"eval_{name}_maxq.csv")).set_index("slice")
cat["cov_cb_global_maxq"] = [float(glob.loc[f"CAT={c}", "coverage_cb"]) for c in cat.category]
cat.to_csv(os.path.join(OUT, f"category_conditional_{name}.csv"), index=False, encoding="utf-8-sig")
mcat = metrics(test, sets.values)

need = calib_min_n(0.05, 0.02, 0.05)
feasible = sw[(sw["load_pi0.1"] * N <= C)]
summary = dict(score_name=name, assumptions=dict(N=N, C=C, s=S, pi=PIS, per_reviewer=PER),
               sensitive_only_load=S, sensitive_only_reviewers=math.ceil(S * N / PER),
               smallest_alpha_meeting_capacity_pi01=(None if feasible.empty else feasible.alpha.min()),
               category_conditional_maxq_a005=dict(cov_direct=mcat["cov_direct"], cov_auto=mcat["cov_auto"], load_pi0_1=mcat["load_pi0.1"]),
               category_calib_needed_per_cat_per_class=need, category_calib_have_min=int(cat.n_calib_min.min()))
save_json(summary, f"sensitivity_summary_{name}.json")
pd.set_option("display.width", 220)
cols = ["alpha", "mode", "cov_direct", "cov_direct_lo", "cov_auto", "cov_auto_lo", "false_confirm", "load_pi0.1", "reviewers_pi0.1", "missed_per_day_pi0.1", "reviewers_pi0.25"]
print(sw[cols].round(3).to_string(index=False)); print(cat.round(3).to_string(index=False)); print(json.dumps(summary, ensure_ascii=False, indent=1))
