"""Webis Clickbait 17 정제 (BaseKit 4 §4-3: postText=제목, targetParagraphs=본문, truthClass·truthMean 보존)."""
import os, json
import pandas as pd
from common import RAW, OUT, file_sha

W = os.path.join(RAW, "webis")
inst = {j["id"]: j for j in map(json.loads, open(os.path.join(W, "instances.jsonl"), encoding="utf-8"))}
rows = []
for t in map(json.loads, open(os.path.join(W, "truth.jsonl"), encoding="utf-8")):
    i = inst[t["id"]]
    rows.append(dict(id=t["id"], title=" ".join(i["postText"]).strip(), body=" ".join(i["targetParagraphs"]).strip(),
                     target_title=i.get("targetTitle", ""), label=int(t["truthClass"] == "clickbait"),
                     truth_mean=t["truthMean"], truth_sd=float(pd.Series(t["truthJudgments"]).std(ddof=0)),
                     n_judges=len(t["truthJudgments"])))
df = pd.DataFrame(rows)
df.to_parquet(os.path.join(OUT, "clean_webis.parquet"), index=False)
print(len(df), df.label.mean().round(3), df.truth_mean.describe().round(3).to_dict(), "sd_mean", round(df.truth_sd.mean(), 3))
print({f: file_sha(os.path.join(W, f)) for f in ["instances.jsonl", "truth.jsonl"]})
