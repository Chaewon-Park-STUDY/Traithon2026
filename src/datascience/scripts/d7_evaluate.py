"""D7: 테스트 셋 평가 (BaseKit 5 MDL-01, BaseKit 7 §7-5). 모든 비율은 Clopper–Pearson 95% 구간과 함께 보고.
사용법: python d7_evaluate.py <name>
처리 규칙: {cb}만 → 확정 후보(검토 큐) / {normal}만 → 자동 통과 / 둘 다·공집합 → 애매 이관 / 정치·경제 → 항상 이관.
"""
import os, sys, json
import numpy as np, pandas as pd
from common import OUT, POLICY, SENSITIVE, clopper_pearson, ece, save_json

tag = sys.argv[1]                    # 예: baseline 또는 baseline_maxq
name = tag.replace("_maxq", "")       # 점수 파일 이름
cal = json.load(open(os.path.join(OUT, f"calibration_{tag}.json"), encoding="utf-8"))
man = pd.read_csv(os.path.join(OUT, "split_manifest.csv"))
sc = pd.read_csv(os.path.join(OUT, f"scores_{name}.csv"))
d = man[man.subset.isin(["test", "calib_auto_check"])].merge(sc, on="news_id")
inc_cb = (1 - d.p_cb) <= cal["q_clickbait"]
inc_ok = d.p_cb <= cal["q_normal"]
d["set"] = np.select([inc_cb & ~inc_ok, ~inc_cb & inc_ok], ["cb", "normal"], "ambiguous")
d["sensitive"] = d.category.isin(SENSITIVE)
d["route"] = np.where(d.sensitive, "human_sensitive", np.select([d.set == "cb", d.set == "normal"], ["confirm_queue", "auto_pass"], "human_ambiguous"))
d["covered"] = d.set != "normal"           # 모델 수준: 클릭베이트가 자동 통과되지 않음
d["false_confirm"] = d.set == "cb"         # 정상 기사가 확정 후보로 감
d["referred"] = d.set == "ambiguous"
d.to_csv(os.path.join(OUT, f"test_decisions_{tag}.csv"), index=False, encoding="utf-8-sig")

rows = []
def add(slice_id, sub):
    cb, ok = sub[sub.label == 1], sub[sub.label == 0]
    r = dict(slice=slice_id, n_cb=len(cb), n_ok=len(ok))
    for key, part, col in [("coverage_cb", cb, "covered"), ("false_confirm_ok", ok, "false_confirm"),
                           ("refer_cb", cb, "referred"), ("refer_ok", ok, "referred"),
                           ("recall_at_0.5", cb, None), ("fpr_at_0.5", ok, None)]:
        k = int((part.p_cb >= 0.5).sum()) if col is None else int(part[col].sum())
        n = len(part)
        lo, hi = clopper_pearson(k, n)
        r[key], r[key + "_lo"], r[key + "_hi"] = (k / n if n else np.nan), lo, hi
    rows.append(r)

t = d[d.subset == "test"]
add("ALL", t)
add("DIRECT_cb+all_ok", pd.concat([t[(t.label == 1) & (t.process_type == "Direct")], t[t.label == 0]]))
add("AUTO_cb+all_ok", pd.concat([t[(t.label == 1) & (t.process_type == "Auto")], t[t.label == 0]]))
add("CALIB_AUTO_CHECK(train)", d[d.subset == "calib_auto_check"])
for c in sorted(t.category.unique()):
    add(f"CAT={c}", t[t.category == c])
for p in sorted(t.part.unique()):
    add(f"PART={p}", t[t.part == p])
for lv in sorted(t[(t.process_type == "Direct")].level.dropna().astype(str).unique()):
    add(f"DIRECT_LEVEL={lv}", pd.concat([t[(t.label == 1) & (t.process_type == "Direct") & (t.level.astype(str) == lv)], t[t.label == 0]]))
add("POLITICS", t[t.category == "정치"]); add("NON_POLITICS", t[t.category != "정치"])
add("ECON", t[t.category == "경제"]); add("NON_ECON", t[t.category != "경제"])
ev = pd.DataFrame(rows)
ev.to_csv(os.path.join(OUT, f"eval_{tag}.csv"), index=False, encoding="utf-8-sig")

def g(s, k): return float(ev.set_index("slice").loc[s, k])
a, e = POLICY["alpha"], POLICY["eps"]
checks = {
    "MDL01_(1)_coverage_lo>=1-a-eps_DIRECT": g("DIRECT_cb+all_ok", "coverage_cb_lo") >= 1 - a - e,
    "MDL01_(1)_coverage_lo>=1-a-eps_AUTO": g("AUTO_cb+all_ok", "coverage_cb_lo") >= 1 - a - e,
    "refer_rate_overall<=0.30": float(t.referred.mean()) <= 0.30,
    "politics_gap_fpr<=0.05": abs(g("POLITICS", "false_confirm_ok") - g("NON_POLITICS", "false_confirm_ok")) <= 0.05,
    "econ_gap_fpr<=0.05": abs(g("ECON", "false_confirm_ok") - g("NON_ECON", "false_confirm_ok")) <= 0.05,
    "politics_gap_cov<=0.05": abs(g("POLITICS", "coverage_cb") - g("NON_POLITICS", "coverage_cb")) <= 0.05,
    "econ_gap_cov<=0.05": abs(g("ECON", "coverage_cb") - g("NON_ECON", "coverage_cb")) <= 0.05,
}
cats = ev[ev.slice.str.startswith("CAT=")]
summary = dict(score_name=tag, n_test=len(t), ece_test=ece(t.p_cb.values, t.label.values),
               refer_rate_test=float(t.referred.mean()), route_share=t.route.value_counts(normalize=True).round(4).to_dict(),
               category_cov_range=[float(cats.coverage_cb.min()), float(cats.coverage_cb.max())],
               category_fc_range=[float(cats.false_confirm_ok.min()), float(cats.false_confirm_ok.max())],
               checks={k: bool(v) for k, v in checks.items()})
save_json(summary, f"eval_summary_{tag}.json")
pd.set_option("display.width", 200)
print(ev[["slice", "n_cb", "n_ok", "coverage_cb", "coverage_cb_lo", "false_confirm_ok", "refer_cb", "refer_ok", "recall_at_0.5", "fpr_at_0.5"]].round(3).to_string(index=False))
print(json.dumps(summary, ensure_ascii=False, indent=1))
