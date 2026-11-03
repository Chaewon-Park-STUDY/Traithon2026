"""robustness 근거: 반복 분할 안정성(보정·테스트 재추출 R회). 이미 점수화된 보정+테스트 표본을 유형별로 합쳐
매회 보정 540건(클래스·유형별)을 새로 뽑고 나머지로 평가한다. 수정안(q̂_cb = max(Direct, Auto)) 기준.
사용법: python d13_stability.py <name> [R=200]
"""
import os, sys, math
import numpy as np, pandas as pd
from common import OUT, POLICY, clopper_pearson, save_json

name = sys.argv[1]; R = int(sys.argv[2]) if len(sys.argv) > 2 else 200
a, eps = POLICY["alpha"], POLICY["eps"]; NC = 540
man = pd.read_csv(os.path.join(OUT, "split_manifest.csv"))
d = man[man.subset.isin(["calib", "calib_auto_check", "test"])].merge(pd.read_csv(os.path.join(OUT, f"scores_{name}.csv")), on="news_id")
pools = {"dir": d[(d.label == 1) & (d.process_type == "Direct")], "aut": d[(d.label == 1) & (d.process_type == "Auto")], "ok": d[d.label == 0]}
def qhat(s):
    n = len(s); k = math.ceil((n + 1) * (1 - a)); return float(np.sort(s)[k - 1]) if k <= n else 1.0
rng = np.random.default_rng(7); rows = []
for r in range(R):
    cal, tst = {}, {}
    for k, p in pools.items():
        idx = rng.permutation(len(p)); cal[k], tst[k] = p.iloc[idx[:NC]], p.iloc[idx[NC:]]
    q_cb = max(qhat(1 - cal["dir"].p_cb.values), qhat(1 - cal["aut"].p_cb.values)); q_ok = qhat(cal["ok"].p_cb.values)
    def sets(df):
        ic, io = (1 - df.p_cb) <= q_cb, df.p_cb <= q_ok
        return np.select([ic & ~io, ~ic & io], ["cb", "normal"], "amb")
    res = dict(rep=r)
    for k in ("dir", "aut"):
        s = sets(tst[k]); c = int((s != "normal").sum()); n = len(s)
        res[f"cov_{k}"], res[f"cov_{k}_lo"] = c / n, clopper_pearson(c, n)[0]
    so = sets(tst["ok"]); allt = pd.concat([tst["dir"], tst["aut"], tst["ok"]])
    res["false_confirm"] = float((so == "cb").mean()); res["refer"] = float((sets(allt) == "amb").mean())
    rows.append(res)
s = pd.DataFrame(rows); s.to_csv(os.path.join(OUT, f"stability_{name}.csv"), index=False)
summ = {"score_name": name, "reps": R, "n_test": {k: len(p) - NC for k, p in pools.items()}}
for c in ["cov_dir", "cov_aut", "false_confirm", "refer"]:
    summ[c] = dict(mean=float(s[c].mean()), p05=float(s[c].quantile(.05)), min=float(s[c].min()), max=float(s[c].max()))
summ["share_cov_ge_target_dir"] = float((s.cov_dir >= 1 - a - eps).mean())
summ["share_cov_ge_target_aut"] = float((s.cov_aut >= 1 - a - eps).mean())
summ["share_cov_lo_ge_target_both"] = float(((s.cov_dir_lo >= 1 - a - eps) & (s.cov_aut_lo >= 1 - a - eps)).mean())
save_json(summ, f"stability_summary_{name}.json")
import json; print(json.dumps(summ, ensure_ascii=False, indent=1))
