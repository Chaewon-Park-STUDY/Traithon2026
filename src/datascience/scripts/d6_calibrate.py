"""D6: 클래스 조건부 split conformal 보정 (BaseKit 5 §5-4).
입력: outputs/scores_<name>.csv (news_id, p_cb = 클릭베이트 확률)
s(x,y) = 1 - p(y|x), q̂_y = s_y 의 ⌈(n_y+1)(1-α)⌉/n_y 분위수. 클릭베이트 q̂ 는 Direct 표본으로 산출.
사용법: python d6_calibrate.py <name> [alpha]
"""
import os, sys, math, json
import numpy as np, pandas as pd
from scipy.stats import beta
from common import OUT, POLICY, save_json

name = sys.argv[1]
alpha = float(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else POLICY["alpha"]
mode = "max" if "--maxq" in sys.argv else "direct"  # max: q̂_cb = max(q̂_Direct, q̂_Auto), 운영 시 유형을 모르므로 더 엄격한 쪽
man = pd.read_csv(os.path.join(OUT, "split_manifest.csv"))
sc = pd.read_csv(os.path.join(OUT, f"scores_{name}.csv"))
cal = man[man.subset == "calib"].merge(sc, on="news_id")


def qhat(s):
    n = len(s)
    k = math.ceil((n + 1) * (1 - alpha))
    return float(np.sort(s)[min(k, n) - 1]) if k <= n else 1.0, n


q_cb, n_cb = qhat(1 - cal[cal.label == 1].p_cb.values)
q_auto = None
if mode == "max":
    auto = man[man.subset == "calib_auto_check"].merge(sc, on="news_id")
    q_auto, n_auto = qhat(1 - auto.p_cb.values)
    q_cb = max(q_cb, q_auto)
q_ok, n_ok = qhat(cal[cal.label == 0].p_cb.values)  # s = 1 - p(정상) = p_cb
tag = name + ("_maxq" if mode == "max" else "")
out = dict(score_name=name, mode=mode, q_clickbait_auto=q_auto, alpha=alpha, q_clickbait=q_cb, q_normal=q_ok, n_calib_clickbait_direct=n_cb, n_calib_normal=n_ok,
           calib_coverage_dist={c: dict(n=n, l=math.floor((n + 1) * alpha),
                                        p_cov_ge_1_a_eps=float(beta.sf(1 - alpha - POLICY["eps"], n + 1 - math.floor((n + 1) * alpha), math.floor((n + 1) * alpha))))
                                for c, n in [("clickbait", n_cb), ("normal", n_ok)]},
           rule="C(x) = {cb if 1-p_cb <= q_clickbait} ∪ {normal if p_cb <= q_normal}; 둘 다/공집합 = 애매(인간 이관), 정치·경제 = 항상 이관")
save_json(out, f"calibration_{tag}.json")
print(json.dumps(out, ensure_ascii=False, indent=1))
