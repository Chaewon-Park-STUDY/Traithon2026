# 제목 표현 슬라이스(F1 지시어, F3 의문형)에서 모델의 판정이 달라지는지 본다. 출력은 집계뿐이다.
# 입력: 소이님 파이프라인 산출물 outputs/test_decisions_<tag>.csv (행 단위 판정)와 outputs/clean_main.parquet (제목)
# 사용: python d24_slice_errors.py <outputs 폴더> [tag=axft_maxq] [출력폴더=d24_out]
# 규칙은 d23_slice_count.py의 사전 등록 규칙을 그대로 쓴다. 주 비교 플래그를 F1, F3로 둔 것은 d23 개수(모델 오류와 무관)를 보고 정한 것이므로 탐색적이다.
import os, sys, csv
import numpy as np, pandas as pd, pyarrow.parquet as pq
from scipy.stats import beta, fisher_exact
from d23_slice_count import FLAGS, SENSITIVE

def cp(k, n, a=0.05):
    if n == 0: return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a / 2, k + 1, n - k)
    return lo, hi

def main(outputs, tag, out):
    dec = pd.read_csv(os.path.join(outputs, f"test_decisions_{tag}.csv"))
    if "subset" in dec.columns:   # 보정 점검용 행(calib_auto_check)이 섞여 있으므로 main test만 쓴다
        print("subset 구성:", dec.subset.value_counts().to_dict()); dec = dec[dec.subset == "test"].copy()
    ids = set(dec.news_id.astype(str))
    t = pq.read_table(os.path.join(outputs, "clean_main.parquet"), columns=["news_id", "title"]).to_pandas()
    t["news_id"] = t.news_id.astype(str); t = t[t.news_id.isin(ids)]
    dec["news_id"] = dec.news_id.astype(str)
    d = dec.merge(t, on="news_id", how="left")
    d = d[~d.category.isin(SENSITIVE)].copy()                       # 비민감만(민감은 전량 사람 이관)
    d["title"] = d.title.fillna("").astype(str)
    for name in ("F1_지시어", "F3_의문형"):
        d[name] = d.title.map(FLAGS[name])
    d["cls"] = np.where(d.label == 1, "클릭베이트", "정상")
    d["miss"] = d.set == "normal"                                   # 클릭베이트가 자동 통과(미포착)
    d["referred_amb"] = d.set == "ambiguous"
    d["false_confirm"] = d.set == "cb"
    rows = []
    for (cls, ptype), g in d.groupby(["cls", "process_type"]):
        for name in ("F1_지시어", "F3_의문형"):
            for lab, metric in (("미포착(클릭베이트→자동통과)", "miss"), ("애매 이관", "referred_amb"), ("확정 후보(오탐 후보)", "false_confirm")):
                if cls == "정상" and metric == "miss": continue
                if cls == "클릭베이트" and metric == "false_confirm": continue
                a = g[g[name]]; b = g[~g[name]]
                ka, na, kb, nb = int(a[metric].sum()), len(a), int(b[metric].sum()), len(b)
                p = fisher_exact([[ka, na - ka], [kb, nb - kb]])[1] if na and nb else float("nan")
                la, ha = cp(ka, na); lb, hb = cp(kb, nb)
                rows.append(dict(cls=cls, process_type=ptype, flag=name, metric=lab, n_flag=na, k_flag=ka,
                                 rate_flag=round(ka / na, 4) if na else None, lo_flag=round(la, 4), hi_flag=round(ha, 4),
                                 n_other=nb, k_other=kb, rate_other=round(kb / nb, 4) if nb else None,
                                 lo_other=round(lb, 4), hi_other=round(hb, 4), fisher_p=round(p, 4)))
    r = pd.DataFrame(rows); os.makedirs(out, exist_ok=True)
    r.to_csv(os.path.join(out, f"d24_slice_errors_{tag}.csv"), index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(f"tag={tag}  비민감 행 {len(d)}건 (미포착 등 사건 수가 적으면 구간이 넓다)")
    print(r.to_string(index=False))
    print(f"\n검정 {len(r)}개 -> 다중비교 Bonferroni 기준 p < {0.05/len(r):.4f}. 이 분석은 탐색적이며 사건 수 k가 작은 행은 해석하지 않는다.")

if __name__ == "__main__":
    if len(sys.argv) < 2: sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "axft_maxq", sys.argv[3] if len(sys.argv) > 3 else "d24_out")
