"""D8: Webis 보조 평가 (BaseKit 4 D8, BaseKit 7 §7-1). 보정·최종 테스트에는 쓰지 않는다.
입력: outputs/scores_webis_<name>.csv (id, p_cb), outputs/calibration_<name>.json (한국어 보정 q̂ 그대로 적용)
산출: 모델 확률-truthMean 순위상관, 판단자 불일치(표준편차)와 애매 구간의 관계, 영어 편향 점검 지표
사용법: python d8_webis_eval.py <name>
"""
import os, sys, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr, mannwhitneyu
from common import OUT, clopper_pearson, ece, save_json

name = sys.argv[1]
w = pd.read_parquet(os.path.join(OUT, "clean_webis.parquet")).merge(pd.read_csv(os.path.join(OUT, f"scores_webis_{name}.csv"), dtype={"id": str}), on="id")
res = dict(score_name=name, n=len(w))
rho, p = spearmanr(w.p_cb, w.truth_mean)
res["spearman_p_vs_truthMean"] = dict(rho=float(rho), p=float(p))
res["ece_vs_truthClass"] = ece(w.p_cb.values, w.label.values)
calp = os.path.join(OUT, f"calibration_{name}.json")
if os.path.exists(calp):
    cal = json.load(open(calp, encoding="utf-8"))
    inc_cb, inc_ok = (1 - w.p_cb) <= cal["q_clickbait"], w.p_cb <= cal["q_normal"]
    w["ambiguous"] = ~(inc_cb ^ inc_ok)
    w["auto_pass"] = ~inc_cb & inc_ok
    cb = w[w.label == 1]
    k, n = int((~cb.auto_pass).sum()), len(cb)
    res["korean_q_applied"] = dict(coverage_cb=k / n, ci=clopper_pearson(k, n), refer_rate=float(w.ambiguous.mean()))
    a, b = w[w.ambiguous].truth_sd, w[~w.ambiguous].truth_sd
    res["judge_sd_ambiguous_vs_not"] = dict(mean_amb=float(a.mean()), mean_not=float(b.mean()),
                                            mannwhitney_p=float(mannwhitneyu(a, b, alternative="greater").pvalue) if len(a) and len(b) else None)
# 판단자 불일치 상위 25%에서 모델 확률이 0.5 근처에 몰리는지
hi = w.truth_sd >= w.truth_sd.quantile(0.75)
res["abs_p_minus_half"] = dict(high_disagree=float((w[hi].p_cb - 0.5).abs().mean()), others=float((w[~hi].p_cb - 0.5).abs().mean()))
w.to_csv(os.path.join(OUT, f"webis_eval_rows_{name}.csv"), index=False, encoding="utf-8-sig")
save_json(res, f"webis_eval_summary_{name}.json")
print(json.dumps(res, ensure_ascii=False, indent=1))
