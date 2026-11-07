# 지름길(shortcut) 점검: test에서 정상은 전부 Auto, Direct는 클릭베이트에만 있다. 모델이 클릭베이트가 아니라
# 문체(사람이 쓴 제목 vs 모델이 만든 제목)를 쓰는지 본다. 출력은 집계뿐이다. 규칙: PREREG_d27_shortcut_check.md
# 입력: outputs/test_decisions_<tag>.csv (label, process_type, category, set, p_cb, subset)
# 사용: python d27_shortcut_check.py <outputs 폴더> [tag=axft_maxq] [출력폴더=d27_out]
import os, sys
import numpy as np, pandas as pd
from scipy.stats import beta, fisher_exact, rankdata

SENSITIVE = {"정치", "경제"}
B_BOOT = 2000   # bootstrap 반복 수(수치 안정성용)
SEED = 20261107

def cp(k, n, a=0.05):
    lo = 0.0 if k == 0 else beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a / 2, k + 1, n - k)
    return float(lo), float(hi)

def wilson(k, n, z=1.959964):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h

def newcombe(k1, n1, k2, n2):
    p1, p2 = k1 / n1, k2 / n2; l1, u1 = wilson(k1, n1); l2, u2 = wilson(k2, n2)
    d = p1 - p2
    return d, d - np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)

def auc(pos, neg):
    r = rankdata(np.concatenate([pos, neg])); n1, n0 = len(pos), len(neg)
    return (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def main(outputs, tag, out):
    d = pd.read_csv(os.path.join(outputs, f"test_decisions_{tag}.csv"))
    if "subset" in d.columns:
        print("subset 구성:", d.subset.value_counts().to_dict()); d = d[d.subset == "test"].copy()
    d = d[~d.category.isin(SENSITIVE)].copy()
    d["cls"] = np.where(d.label == 1, "클릭베이트", "정상")
    os.makedirs(out, exist_ok=True)
    print(f"tag={tag} test 비민감 {len(d)}건. 구성:\n{pd.crosstab(d.cls, d.process_type).to_string()}")

    # 1) 클릭베이트의 경로별 비율을 Auto와 Direct로 나눠 비교 (Direct - Auto)
    cb = d[d.cls == "클릭베이트"]; rows = []
    for sv, lab in (("normal", "미포착(자동 통과)"), ("cb", "확정 후보"), ("ambiguous", "판정 유보")):
        g = {t: cb[cb.process_type == t] for t in ("Auto", "Direct")}
        k = {t: int((g[t].set == sv).sum()) for t in g}; n = {t: len(g[t]) for t in g}
        diff, lo, hi = newcombe(k["Direct"], n["Direct"], k["Auto"], n["Auto"])
        _, p = fisher_exact([[k["Direct"], n["Direct"] - k["Direct"]], [k["Auto"], n["Auto"] - k["Auto"]]])
        for t in ("Auto", "Direct"):
            a, b = cp(k[t], n[t]); rows.append(dict(metric=lab, group=t, n=n[t], k=k[t], rate=round(k[t] / n[t], 4), lo=round(a, 4), hi=round(b, 4)))
        rows.append(dict(metric=lab, group="차이(Direct-Auto)", n=n["Direct"] + n["Auto"], k=None, rate=round(diff, 4), lo=round(lo, 4), hi=round(hi, 4), fisher_p=round(p, 4)))
    R = pd.DataFrame(rows); R["k"] = R.k.astype("Int64"); R.to_csv(os.path.join(out, f"d27_rates_{tag}.csv"), index=False, encoding="utf-8-sig")
    print("\n[클릭베이트 경로별 비율: Auto 대 Direct, 차이는 Newcombe 95% 구간]"); print(R.to_string(index=False))

    # 2) p_cb 순위의 AUC: 전체, Auto 클릭베이트만, Direct 클릭베이트만 (정상은 모두 Auto)
    nm = d[d.cls == "정상"]; sc = {"전체": cb, "Auto": cb[cb.process_type == "Auto"], "Direct": cb[cb.process_type == "Direct"]}
    base = {k: auc(v.p_cb.to_numpy(), nm.p_cb.to_numpy()) for k, v in sc.items()}
    rng = np.random.default_rng(SEED); bs = {k: [] for k in sc}; bd = []
    for _ in range(B_BOOT):
        bn = nm.p_cb.to_numpy()[rng.integers(0, len(nm), len(nm))]; vals = {}
        for k, v in sc.items():
            x = v.p_cb.to_numpy(); vals[k] = auc(x[rng.integers(0, len(x), len(x))], bn); bs[k].append(vals[k])
        bd.append(vals["Direct"] - vals["Auto"])
    A = []
    for k in sc:
        lo, hi = np.percentile(bs[k], [2.5, 97.5]); A.append(dict(positive=f"클릭베이트({k})", negative="정상(Auto)", n_pos=len(sc[k]), n_neg=len(nm), auc=round(base[k], 4), lo=round(lo, 4), hi=round(hi, 4)))
    lo, hi = np.percentile(bd, [2.5, 97.5]); A.append(dict(positive="차이(Direct-Auto)", negative="정상(Auto)", n_pos=None, n_neg=len(nm), auc=round(base["Direct"] - base["Auto"], 4), lo=round(lo, 4), hi=round(hi, 4)))
    A = pd.DataFrame(A); A["n_pos"] = A.n_pos.astype("Int64"); A.to_csv(os.path.join(out, f"d27_auc_{tag}.csv"), index=False, encoding="utf-8-sig")
    print("\n[p_cb 순위 AUC, 정상(Auto) 대비, bootstrap 95% 구간]"); print(A.to_string(index=False))
    print("\n주의: 정상이 전부 Auto라서 Direct 클릭베이트는 문체만으로도 구분될 수 있다. Direct가 Auto보다 뚜렷이 쉽게 잡히면 문체 지름길의 징후이고, 아니면 이 데이터로는 지름길의 증거가 없다는 뜻이다(없다는 증명은 아님).")

if __name__ == "__main__":
    if len(sys.argv) < 2: sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "axft_maxq", sys.argv[3] if len(sys.argv) > 3 else "d27_out")
