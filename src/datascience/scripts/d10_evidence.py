"""근거 위치 찾기 기준선(BaseKit 5 §5-5, BaseKit 7 근거 평가). 테스트 셋 클릭베이트에서
- Part2: 삽입된 이탈 문장(subjectConsistencyYn=N)을 '나머지 본문과 가장 덜 닮은 문장'으로 찾을 수 있는가
- Part1: 근거 문장(referSentenceyn=Y)을 '새 제목과 가장 닮은 문장'으로 찾을 수 있는가
지표: top-1·top-3 적중률(정답 문장이 상위 k 안에 있는 비율), 무작위 기준(k/문장 수) 대비.
"""
import os, json, zipfile
import numpy as np, pandas as pd, pyarrow.parquet as pq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from common import OUT, RAW, clopper_pearson, save_json

man = pd.read_csv(os.path.join(OUT, "split_manifest.csv"))
ids = set(man[(man.subset == "test") & (man.label == 1)].news_id)
meta = pq.read_table(os.path.join(OUT, "clean_main.parquet"), columns=["news_id", "part", "process_type", "category", "src_path"],
                     filters=[("news_id", "in", list(ids))]).to_pandas()
zips = {}
rows = []
for r in meta.itertuples():
    zname, inner = r.src_path.split("/", 1)
    if zname not in zips:  # setdefault는 매번 ZipFile을 새로 열어 느려짐
        zips[zname] = zipfile.ZipFile(os.path.join(RAW, "aihub", zname))
    z = zips[zname]
    j = json.loads(z.read(inner).decode("utf-8-sig"))
    lab, src = j["labeledDataInfo"], j["sourceDataInfo"]
    if r.part == "Part2":
        ss = lab.get("processSentenceInfo") or []
        sents = [s["sentenceContent"] for s in ss]; gold = {i for i, s in enumerate(ss) if str(s.get("subjectConsistencyYn", "Y")).upper() == "N"}
        query = None
    else:
        ss = src.get("sentenceInfo") or []
        sents = [s["sentenceContent"] for s in ss]
        refs = {int(x["sentenceNo"]) for x in lab.get("referSentenceInfo", []) if str(x.get("referSentenceyn", "N")).upper() == "Y"}
        gold = {i for i, s in enumerate(ss) if int(s["sentenceNo"]) in refs}; query = lab.get("newTitle") or src.get("newsTitle")
    if sents and gold:
        rows.append(dict(news_id=r.news_id, part=r.part, ptype=r.process_type, category=r.category, sents=sents, gold=gold, query=query))
print("usable", len(rows), "of", len(meta))
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), min_df=2, sublinear_tf=True).fit([s for x in rows for s in x["sents"]])
out = []
for x in rows:
    S = normalize(vec.transform(x["sents"])); n = S.shape[0]
    if x["part"] == "Part2":
        tot = np.asarray(S.sum(0)).ravel(); st = S @ tot; rn = np.sqrt(np.maximum(tot @ tot - 2 * st + 1, 1e-9))
        score = -(st - 1) / rn           # 덜 닮을수록 높은 점수
    else:
        score = (S @ normalize(vec.transform([x["query"]])).T).toarray().ravel()
    order = np.argsort(-score)
    out.append(dict(news_id=x["news_id"], part=x["part"], ptype=x["ptype"], category=x["category"], n_sent=n, n_gold=len(x["gold"]),
                    top1=int(order[0] in x["gold"]), top3=int(bool(set(order[:3]) & x["gold"])),
                    rand1=len(x["gold"]) / n, rand3=1 - np.prod([(n - len(x["gold"]) - i) / (n - i) for i in range(min(3, n))]) if n > len(x["gold"]) else 1.0))
e = pd.DataFrame(out)
e.to_csv(os.path.join(OUT, "evidence_rows.csv"), index=False, encoding="utf-8-sig")
summ = []
for (p, t), g in e.groupby(["part", "ptype"]):
    k1, k3, n = int(g.top1.sum()), int(g.top3.sum()), len(g)
    summ.append(dict(part=p, ptype=t, n=n, top1=k1 / n, top1_ci=clopper_pearson(k1, n), top3=k3 / n, rand_top1=float(g.rand1.mean()), rand_top3=float(g.rand3.mean()),
                     median_sents=float(g.n_sent.median()), mean_gold=float(g.n_gold.mean())))
s = pd.DataFrame(summ); s.to_csv(os.path.join(OUT, "evidence_summary.csv"), index=False, encoding="utf-8-sig")
pd.set_option("display.width", 200); print(s.round(3).to_string(index=False))
