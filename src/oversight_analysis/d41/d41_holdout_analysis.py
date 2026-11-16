# d41: 고정 임계값을 holdout에 적용해 헤드라인 지표와 d34 재현을 계산. 규칙: PREREG_d41_holdout.md
# 사용: python d41_holdout_analysis.py <d41_scores.csv> <holdout_meta.csv> <datascience/results> <oversight_analysis/d34_out> [out=out]
import os, sys, json
import numpy as np, pandas as pd
from scipy.stats import beta, binom
from sklearn.metrics import roc_auc_score
SC, META, RES, D34 = sys.argv[1:5]; OUT = sys.argv[5] if len(sys.argv) > 5 else "out"; os.makedirs(OUT, exist_ok=True)
SENS = {"정치", "경제"}; B = 10000; SEED = 20261116; ALPHA = 0.05
s = pd.read_csv(SC, dtype={"key": str}).drop_duplicates("key", keep="last")
m = pd.read_csv(META, dtype={"key": str}).merge(s[["key", "p_cb", "answer_mass"]], on="key", how="left")
assert m.p_cb.notna().all(), f"점수 누락 {m.p_cb.isna().sum()}"
def cp(k, n, a=0.05):
    if n == 0: return (np.nan, np.nan)
    return (0.0 if k == 0 else float(beta.ppf(a / 2, k, n - k + 1)), 1.0 if k == n else float(beta.ppf(1 - a / 2, k + 1, n - k)))
def ece(p, y, bins=10):
    idx = np.clip(np.digitize(p, np.linspace(0, 1, bins + 1)[1:-1]), 0, bins - 1); e = 0.0
    for b in range(bins):
        k = idx == b
        if k.any(): e += k.mean() * abs(p[k].mean() - y[k].mean())
    return e
cal = json.load(open(os.path.join(RES, "calibration_axft_maxq.json"), encoding="utf-8"))
th = pd.read_csv(os.path.join(D34, "d34_thresholds_axft.csv"), encoding="utf-8-sig"); th = th[th.alpha == ALPHA]
def thr(sc):
    if sc == "S0": return m.assign(qcb=cal["q_clickbait"], qok=cal["q_normal"])
    t = th[th.scenario.str.startswith(sc)].set_index("group")
    g = m.category.map(t.q_cb); h = m.category.map(t.q_ok); return m.assign(qcb=g, qok=h)
def route(x):
    ic, io = (1 - x.p_cb) <= x.qcb, x.p_cb <= x.qok
    return ic, io, np.select([ic & ~io, ~ic & io], ["cb", "normal"], "ambiguous")
y = (m.label == 1).astype(int).values
# 1~4 헤드라인 (S0)
ic, io, r0 = route(thr("S0")); m["ic"], m["io"], m["r0"] = ic, io, r0
rows = []
for name, sub in [("Direct 커버리지(전체 category)", m[m.grp == "direct_cb"]), ("Auto 커버리지(전체 category)", m[m.grp == "auto_cb"]),
                  ("Direct 커버리지(비민감)", m[(m.grp == "direct_cb") & ~m.category.isin(SENS)]), ("Auto 커버리지(비민감)", m[(m.grp == "auto_cb") & ~m.category.isin(SENS)])]:
    k, n = int(sub.ic.sum()), len(sub); lo, hi = cp(k, n)
    rows.append(dict(지표=name, 값=round(k / n, 4), 하한=round(lo, 4), 상한=round(hi, 4), n=n, 기준="하한 ≥ 0.93", 충족=bool(lo >= 0.93)))
nm = m[m.grp == "normal"]
for name, k in [("정상 커버리지(정상 후보 포함)", int(nm.io.sum())), ("정상 오확정(cb 확정)", int((nm.r0 == "cb").sum()))]:
    lo, hi = cp(k, len(nm)); rows.append(dict(지표=name, 값=round(k / len(nm), 4), 하한=round(lo, 4), 상한=round(hi, 4), n=len(nm), 기준="보고만", 충족=""))
rng = np.random.default_rng(SEED); idx = rng.integers(0, len(m), size=(B, len(m))); au, ec = [], []
p = m.p_cb.values
for i in range(B):
    j = idx[i]; au.append(roc_auc_score(y[j], p[j])); ec.append(ece(p[j], y[j]))
a0, e0 = roc_auc_score(y, p), ece(p, y); alo, ehi = np.quantile(au, .025), np.quantile(ec, .975)
rows.append(dict(지표="AUC(클릭베이트 대 정상)", 값=round(a0, 4), 하한=round(alo, 4), 상한=round(np.quantile(au, .975), 4), n=len(m), 기준="하한 ≥ 0.85", 충족=bool(alo >= 0.85)))
rows.append(dict(지표="ECE(정상:클릭베이트=1:2 표본)", 값=round(e0, 4), 하한=round(np.quantile(ec, .025), 4), 상한=round(ehi, 4), n=len(m), 기준="상한 ≤ 0.10", 충족=bool(ehi <= 0.10)))
h = pd.DataFrame(rows); h.to_csv(os.path.join(OUT, "d41_headline.csv"), index=False, encoding="utf-8-sig"); print(h.to_string(index=False))
# 5 d34 재현 (비민감 category)
ns = m[~m.category.isin(SENS)].copy(); out = []; per = {}
for sc in ["S0", "S1", "S2"]:
    x = thr(sc); x = x[~x.category.isin(SENS)]; ic2, io2, r = route(x); per[sc] = pd.Series(r, index=x.index)
    t = x.assign(r=r)
    for c, g in [("전체", t)] + list(t.groupby("category")):
        cb = g[g.label == 1]; k = int((cb.r == "normal").sum()); n = len(cb); lo, hi = cp(k, n)
        nmg = g[g.label == 0]
        out.append(dict(시나리오=sc, category=c, 클릭베이트_n=n, 놓침_k=k, 놓침=round(k / n, 4), 하한=round(lo, 4), 상한=round(hi, 4), exceeds_alpha=bool(lo > ALPHA),
                        오확정fc=round(float((nmg.r == "cb").mean()), 4), amb_normal=round(float((nmg.r == "ambiguous").mean()), 4)))
c = pd.DataFrame(out); c.to_csv(os.path.join(OUT, "d41_category_s0s1s2.csv"), index=False, encoding="utf-8-sig"); print(c.to_string(index=False))
# 생활&문화 클릭베이트에서 S0 대 S1·S2 exact McNemar (단측: S0만 놓치고 S1/S2는 안 놓친 쪽이 많은가)
g = ns[(ns.category == "생활&문화") & (ns.label == 1)].index; mc = []
for sc in ["S1", "S2"]:
    a = (per["S0"].loc[g] == "normal").values; b = (per[sc].loc[g] == "normal").values
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum()); pv = float(binom.sf(n10 - 1, n10 + n01, 0.5)) if n10 + n01 > 0 else 1.0
    mc.append(dict(비교=f"S0 대 {sc} (생활&문화 클릭베이트)", S0만놓침=n10, 시나리오만놓침=n01, 단측_p=round(pv, 4), 기준="p < 0.05", 충족=bool(pv < 0.05)))
mc = pd.DataFrame(mc); mc.to_csv(os.path.join(OUT, "d41_mcnemar.csv"), index=False, encoding="utf-8-sig"); print(mc.to_string(index=False))
