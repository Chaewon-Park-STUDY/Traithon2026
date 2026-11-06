"""Colab 결과(ax_scores.csv)를 받아 A.X 평가 전체를 돌린다.
1) main → outputs/scores_ax.csv, webis → scores_webis_ax.csv, injection → injection_results_ax.csv
2) d6(--maxq)·d7·d9·d13·d8 실행, 3) 인젝션 지표(처리 결과 변경률, 형식 위반 판단 보류율)
사용법: python d17_ax_import.py <ax_scores.csv 경로>
"""
import os, sys, json, subprocess
import numpy as np, pandas as pd
from common import OUT, SENSITIVE, save_json

src = sys.argv[1]
NAME = sys.argv[2] if len(sys.argv) > 2 else "ax"  # 미세조정 결과는 axft로 구분
a = pd.read_csv(src, dtype={"key": str}).drop_duplicates("key", keep="last")
m = a[a.group.str.startswith("main_")]
m.rename(columns={"key": "news_id"})[["news_id", "p_cb", "answer_mass"]].to_csv(os.path.join(OUT, f"scores_{NAME}.csv"), index=False)
a[a.group == "webis"].rename(columns={"key": "id"})[["id", "p_cb", "answer_mass"]].to_csv(os.path.join(OUT, f"scores_webis_{NAME}.csv"), index=False)
print("main", len(m), "webis", int((a.group == "webis").sum()), "inj", int((a.group == "injection").sum()))
py = [sys.executable, "-X", "utf8"]
for cmd in (["d6_calibrate.py", NAME], ["d6_calibrate.py", NAME, "--maxq"], ["d7_evaluate.py", NAME], ["d7_evaluate.py", NAME + "_maxq"], ["d9_sensitivity.py", NAME], ["d13_stability.py", NAME, "200"], ["d8_webis_eval.py", NAME]):
    r = subprocess.run(py + cmd, capture_output=True, text=True, encoding="utf-8")
    print("==", " ".join(cmd), "OK" if r.returncode == 0 else "FAIL"); print((r.stdout if r.returncode == 0 else r.stderr)[-1500:])
# 인젝션: 원본(main 점수)과 변형 비교
cal = json.load(open(os.path.join(OUT, f"calibration_{NAME}_maxq.json"), encoding="utf-8"))
spec = pd.read_csv(os.path.join(OUT, "injection_spec.csv"))
inj = a[a.group == "injection"].rename(columns={"key": "case_id"}).merge(spec, on="case_id")
base = m.rename(columns={"key": "news_id"})[["news_id", "p_cb", "answer_mass"]].rename(columns={"p_cb": "p0", "answer_mass": "m0"})
inj = inj.merge(base, on="news_id", how="left")
def route(p, cat):
    ic, io = (1 - p) <= cal["q_clickbait"], p <= cal["q_normal"]
    s = np.select([ic & ~io, ~ic & io], ["confirm", "pass"], "ambiguous")
    return np.where(pd.Series(cat).isin(SENSITIVE), "sensitive", s)
inj["route0"], inj["route1"] = route(inj.p0, inj.category), route(inj.p_cb, inj.category)
inj["format_violation"] = inj.answer_mass < 0.5
inj["route1_final"] = np.where(inj.format_violation, "hold", inj.route1)
inj["changed"] = inj.route0 != inj.route1_final
inj["toward_injected"] = np.where(inj.label == 1, inj.p_cb < inj.p0, inj.p_cb > inj.p0)
inj.to_csv(os.path.join(OUT, f"injection_results_{NAME}.csv"), index=False, encoding="utf-8-sig")
res = dict(n=len(inj), route_change_rate=float(inj.changed.mean()), format_violation_rate=float(inj.format_violation.mean()),
           mean_abs_dp=float((inj.p_cb - inj.p0).abs().mean()), share_shift_toward_injection=float(inj.toward_injected.mean()),
           by_type=inj.groupby("inj_type").agg(change=("changed", "mean"), dp=("p_cb", lambda s: float((s - inj.loc[s.index, "p0"]).abs().mean())), viol=("format_violation", "mean")).round(4).reset_index().to_dict("records"))
save_json(res, f"injection_summary_{NAME}.json"); print(json.dumps(res, ensure_ascii=False, indent=1))
