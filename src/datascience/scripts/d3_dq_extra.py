"""DQ 보강(BaseKit 4 §4-6): 필드 결측·길이 분포, 근거 정답 라벨 표시 건수(§4-3 '표시 건수는 본선 측정')."""
import os, json
import numpy as np, pandas as pd, pyarrow.parquet as pq
from common import OUT, save_json

d = pq.read_table(os.path.join(OUT, "clean_main.parquet"), columns=["news_id", "split", "part", "process_type", "label", "category", "title", "evidence_sent", "title_len", "body_len", "hold", "flag_pii"]).to_pandas()
d["has_ev"] = d.evidence_sent.fillna("") != ""
d["n_ev"] = d.evidence_sent.fillna("").map(lambda s: len([x for x in s.split(",") if x]))
g = d.groupby(["part", "label", "process_type"])
ev = g.agg(n=("news_id", "size"), evidence_marked=("has_ev", "sum"), evidence_rate=("has_ev", "mean"), mean_ev_sents=("n_ev", "mean"),
           title_len_med=("title_len", "median"), body_len_med=("body_len", "median"), body_len_p95=("body_len", lambda x: x.quantile(.95))).reset_index()
ev.to_csv(os.path.join(OUT, "dq_evidence_lengths.csv"), index=False, encoding="utf-8-sig")
comp = {"title_blank": int((d.title.str.strip() == "").sum()), "body_short_lt100": int((d.body_len < 100).sum()),
        "body_len_median": float(d.body_len.median()), "body_gt_3000_share": float((d.body_len > 3000).mean()),
        "title_eq_orig_check": "Part1 정상은 새 제목=원제목이어야 함(아래 별도)"}
save_json(comp, "dq_completeness.json")
pd.set_option("display.width", 200); print(ev.round(3).to_string(index=False)); print(comp)
