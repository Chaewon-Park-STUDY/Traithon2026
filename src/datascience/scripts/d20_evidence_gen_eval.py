"""5-B4 근거 인용 생성 평가 집계. 입력 gpu/results/ev_gen.csv + outputs/evidence_input.parquet + evidence_rows.csv(TF-IDF 기준선).
출력 outputs/evidence_gen_summary.csv (원문 없음, 집계만)
"""
import os, sys
import numpy as np, pandas as pd
from common import OUT, clopper_pearson
g = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(OUT), "gpu", "results", "ev_gen.csv"))
d = pd.read_parquet(os.path.join(OUT, "evidence_input.parquet"))[["news_id", "part", "ptype", "sents", "gold"]]
base = pd.read_csv(os.path.join(OUT, "evidence_rows.csv"))
d["n"] = d.sents.map(len); d["gold"] = d.gold.map(lambda x: {int(i) + 1 for i in x})   # 1부터 센 번호
m = g.merge(d, on="news_id")
def picks(s): return [int(x) for x in str(s).split()] if isinstance(s, str) and s.strip() else []
m["p"] = m.picks.map(picks)
m["valid"] = [bool(p) and all(1 <= x <= n for x in p) for p, n in zip(m.p, m.n)]
m["top1"] = [int(bool(p) and p[0] in gd) for p, gd in zip(m.p, m.gold)]
m["top3"] = [int(bool(set(p[:3]) & gd)) for p, gd in zip(m.p, m.gold)]
m["echo"] = m.picks.astype(str).str.strip() == "4 1 7"   # 프롬프트 예시를 그대로 베낀 답
rows = []
for (mode, part, pt), s in m.groupby(["mode", "part", "ptype"]):
    k1, k3, n = int(s.top1.sum()), int(s.top3.sum()), len(s)
    b = base[(base.part == part) & (base.ptype == pt)]
    rows.append(dict(mode=mode, part=part, ptype=pt, n=n, top1=k1 / n, top1_ci=tuple(round(x, 3) for x in clopper_pearson(k1, n)), top3=k3 / n,
                     valid=s.valid.mean(), echo_example=s.echo.mean(), tfidf_top1=b.top1.mean(), tfidf_top3=b.top3.mean(), rand_top1=b.rand1.mean()))
r = pd.DataFrame(rows); r.to_csv(os.path.join(OUT, "evidence_gen_summary.csv"), index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250); print(r.round(3).to_string(index=False))
