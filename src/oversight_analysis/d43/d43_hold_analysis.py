"""d43: answer_mass 기반 hold 임계값(tau) 운영 비교. 사후 분석(결과를 본 뒤 설정), 집계만 출력.
사용: python d43_hold_analysis.py <ax_ft_scores.csv> <split_manifest.csv> <d41_scores.csv> <holdout_meta.csv> [out=out]
입력 행 단위 파일은 로컬 전용이며 저장소에 올리지 않는다. 출력은 집계표만.
규칙: 정치/경제는 항상 사람 -> 비민감 입력만 집계. confirm/pass/ambiguous는 calibration_axft_maxq.json의 q를 사용.
오판 정의: 자동 판정(사람에게 안 간 것) 중 confirm이면 clickbait(1), pass이면 normal(0)으로 읽었을 때 라벨과 다른 비율.
"""
import sys, os, numpy as np, pandas as pd
from scipy.stats import beta
SC, MF, HS, HM = sys.argv[1:5]; OUT = sys.argv[5] if len(sys.argv) > 5 else "out"
QCB, QN, TAU_DEFAULT = 0.17167949676513672, 0.5799638032913208, 0.95
sc = pd.read_csv(SC, dtype={"key": str})
mf = pd.read_csv(MF, dtype={"news_id": str}, encoding="utf-8-sig").rename(columns={"news_id": "key"})
test = sc[sc.group == "main_test"].merge(mf[["key", "label", "category"]], on="key")
ho = pd.read_csv(HS, dtype={"key": str}).merge(pd.read_csv(HM, dtype={"key": str})[["key", "label", "category"]], on="key")
def cp(k, n, a=0.05):
    if n == 0: return (np.nan, np.nan)
    return (0.0 if k == 0 else beta.ppf(a/2, k, n-k+1), 1.0 if k == n else beta.ppf(1-a/2, k+1, n-k))
rows = []
for name, x in [("main_test", test), ("d41_holdout", ho)]:
    x = x[~x.category.isin(["정치", "경제"])].copy()
    cb = (1 - x.p_cb <= QCB); nm = (x.p_cb <= QN)
    r = np.where(cb & ~nm, "confirm", np.where(nm & ~cb, "pass", "amb"))
    for tau in [0, 0.5, 0.9, 0.95, 0.97]:
        hold = (x.answer_mass < tau).values; amb = (r == "amb") & ~hold
        human = hold | amb; auto = ~human
        wrong = auto & ((r == "confirm").astype(int) != x.label.values)
        lo, hi = cp(int(wrong.sum()), int(auto.sum()))
        rows.append(dict(set=name, tau=tau, default=(tau == TAU_DEFAULT), n_nonsensitive=len(x),
            hold_rate=hold.mean(), ambiguous_only_rate=amb.mean(), human_rate=human.mean(),
            n_auto=int(auto.sum()), n_auto_wrong=int(wrong.sum()), auto_wrong_rate=wrong.sum()/auto.sum(),
            auto_wrong_ci_lo=lo, auto_wrong_ci_hi=hi))
os.makedirs(OUT, exist_ok=True)
t = pd.DataFrame(rows); t.to_csv(os.path.join(OUT, "d43_hold_table.csv"), index=False, float_format="%.4f")
print(t.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
