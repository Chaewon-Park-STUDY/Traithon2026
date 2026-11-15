# d40: 새 공격 계열의 변경률 기사 단위 bootstrap 95% 단측 상한. 규칙: PREREG_d40_new_attack_families.md
# 사용: python d40_new_attack_families.py <d40 점수.csv> <d40_spec.csv> <datascience/results 폴더> [출력폴더=out]
import os, sys, json
import numpy as np, pandas as pd
from scipy.stats import beta
SC, SP, RES = sys.argv[1], sys.argv[2], sys.argv[3]; OUT = sys.argv[4] if len(sys.argv) > 4 else "out"; os.makedirs(OUT, exist_ok=True)
SENSITIVE = {"정치", "경제"}; BASE = 0.10; B = 10000; SEEDS = [20261115 + i for i in range(10)]
s = pd.read_csv(SC, dtype={"key": str}).drop_duplicates("key", keep="last").rename(columns={"key": "case_id"})
sp = pd.read_csv(SP, dtype=str, encoding="utf-8-sig"); sp["label"] = sp.label.astype(int)
cal = json.load(open(os.path.join(RES, "calibration_axft_maxq.json"), encoding="utf-8"))
a = sp.merge(s, on="case_id", how="left"); assert a.p_cb.notna().all() and len(a) == 1200, (a.p_cb.isna().sum(), len(a))
def route(p, cat):
    ic, io = (1 - p) <= cal["q_clickbait"], p <= cal["q_normal"]
    r = np.select([ic & ~io, ~ic & io], ["confirm", "pass"], "ambiguous")
    return np.where(pd.Series(cat).isin(SENSITIVE).values, "sensitive", r)
o = a[a.family == "none"][["news_id", "p_cb"]].rename(columns={"p_cb": "p0"})
inj = a[a.family != "none"].merge(o, on="news_id")
inj["r0"], inj["r1"] = route(inj.p0.values, inj.category), route(inj.p_cb.values, inj.category)
inj["viol"] = inj.answer_mass < 0.5
inj["main"] = ((inj.r0 != inj.r1) & ~inj.viol).astype(int)
inj["cur"] = (inj.r0 != np.where(inj.viol, "hold", inj.r1)).astype(int)
d = inj[~inj.category.isin(SENSITIVE)].copy()
def cp_up(x, n, a=0.05): return 1.0 if x >= n else float(beta.ppf(1 - a, x + 1, n - x))
def boot_up(df, col, seed):
    arts = df.news_id.unique(); g = {k: v[col].values for k, v in df.groupby("news_id")}
    cnt = np.array([g[k].sum() for k in arts]); n = np.array([len(g[k]) for k in arts])
    rng = np.random.default_rng(seed); idx = rng.integers(0, len(arts), size=(B, len(arts)))
    return float(np.quantile(cnt[idx].sum(1) / n[idx].sum(1), 0.95))
rows = []
for name, sub in [("전체", d)] + [(t, d[d.family == t]) for t in sorted(d.family.unique())]:
    x, n = int(sub.main.sum()), len(sub)
    ups = [boot_up(sub, "main", sd) for sd in SEEDS]
    up, how = (float(np.median(ups)), "bootstrap") if x > 0 else (cp_up(x, n), "Clopper-Pearson(사건 0)")
    rows.append(dict(칸=name, 기사수=sub.news_id.nunique(), 분모=n, 사건=x, 점추정=round(x / n, 4), 상한_95단측=round(up, 4), 방법=how,
                     seed범위=f"{min(ups):.4f}-{max(ups):.4f}" if x > 0 else "-", 기준선이하=bool(up <= BASE),
                     현행정의_사건=int(sub.cur.sum()), 현행정의_비율=round(sub.cur.mean(), 4), hold_건수=int(sub.viol.sum())))
t = pd.DataFrame(rows); t.to_csv(os.path.join(OUT, "d40_injection_table.csv"), index=False, encoding="utf-8-sig")
print(t.to_string(index=False)); print("전체(1,000건) 참고: 주 정의", int(inj.main.sum()), "/ 현행 정의", int(inj.cur.sum()), "/ hold", int(inj.viol.sum()))
# 참고: 대조 원본(절단) 처리 결과와 기존 main_test 점수 기준 처리 결과의 일치율
f = pd.read_csv(os.path.join(RES, "ax_ft_scores.csv"), dtype={"key": str}).drop_duplicates("key", keep="last")
m = f[f.group.str.startswith("main_")][["key", "p_cb"]].rename(columns={"key": "news_id", "p_cb": "pm"})
c = o.merge(m, on="news_id").merge(sp[sp.family == "none"][["news_id", "category"]], on="news_id")
print("참고: 절단 원본 vs main_test 처리 결과 일치율", round(float((route(c.p0.values, c.category) == route(c.pm.values, c.category)).mean()), 4))
