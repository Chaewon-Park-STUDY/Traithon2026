"""DQ-02 라벨 재검수 표본(BaseKit 4 §4-6, 감사 표본 공식 n = z²p(1-p)/h²).
p=0.10(AI-Hub 의미 정확성 90% 기준상 최대 오류율), h=0.03 → 385, 여유 두어 400건(클래스별 200, 유형·Part·카테고리 층화).
산출: outputs/label_audit.csv (검수자가 채울 칸 포함, 원문 발췌 포함 → 저장소 업로드 금지), outputs/label_audit_ids.csv (ID만)
검수 후: python d12_label_audit_sample.py --score  → 오류율과 Clopper–Pearson 구간
"""
import os, sys, math
import pandas as pd, pyarrow.parquet as pq
from common import OUT, SEED, clopper_pearson, save_json

P = os.path.join(OUT, "label_audit.csv")
if "--score" in sys.argv:
    a = pd.read_csv(P)
    done = a[a.reviewer_label.isin([0, 1])]
    k = int((done.reviewer_label != done.label).sum()); n = len(done)
    res = dict(n_reviewed=n, n_disagree=k, error_rate=k / n if n else None, ci95=clopper_pearson(k, n),
               by_type=done.assign(err=done.reviewer_label != done.label).groupby(["label", "process_type"]).err.agg(["mean", "size"]).reset_index().to_dict("records"))
    save_json(res, "label_audit_result.json"); print(res); sys.exit()
z, p, h = 1.96, 0.10, 0.03
n_req = math.ceil(z * z * p * (1 - p) / h / h)
man = pd.read_csv(os.path.join(OUT, "split_manifest.csv"))
meta = pq.read_table(os.path.join(OUT, "clean_main.parquet"), columns=["news_id", "split", "label", "process_type", "part", "category", "hold", "flag_pii"]).to_pandas()
pool = meta[~meta.hold & ~meta.flag_pii & meta.label.notna() & ~meta.news_id.isin(man[man.subset == "test"].news_id)]
s = pd.concat([g.sample(min(len(g), round(200 * len(g) / (pool.label == y).sum())) or 1, random_state=SEED)
               for y in (0, 1) for _, g in pool[pool.label == y].groupby(["process_type", "part", "category"])])
txt = pq.read_table(os.path.join(OUT, "clean_main.parquet"), columns=["news_id", "title", "title_orig", "body", "evidence_sent"], filters=[("news_id", "in", list(s.news_id))]).to_pandas()
s = s.merge(txt, on="news_id")
s["body_excerpt"] = s.body.str[:1200]
s = s.drop(columns=["body", "hold", "flag_pii"]).sample(frac=1, random_state=SEED)
s["reviewer_label"] = ""; s["reviewer"] = ""; s["note"] = ""
s.to_csv(P, index=False, encoding="utf-8-sig")
s[["news_id", "label", "process_type", "part", "category"]].to_csv(os.path.join(OUT, "label_audit_ids.csv"), index=False, encoding="utf-8-sig")
print("required", n_req, "drawn", len(s)); print(s.groupby(["label", "process_type", "part"]).size().to_string())
