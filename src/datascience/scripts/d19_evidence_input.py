"""근거 인용 생성 평가(5-B4) 입력 만들기. d10과 같은 테스트 클릭베이트·정답 문장으로 GPU용 parquet을 만든다.
출력: outputs/evidence_input.parquet (news_id, part, ptype, category, title, sents, gold) — 원문 포함이므로 저장소에 올리지 않는다.
"""
import os, json, zipfile
import pandas as pd, pyarrow.parquet as pq
from common import OUT, RAW
man = pd.read_csv(os.path.join(OUT, "split_manifest.csv"))
ids = set(man[(man.subset == "test") & (man.label == 1)].news_id)
meta = pq.read_table(os.path.join(OUT, "clean_main.parquet"), columns=["news_id", "part", "process_type", "category", "src_path"],
                     filters=[("news_id", "in", list(ids))]).to_pandas()
zips, rows = {}, []
for r in meta.itertuples():
    zname, inner = r.src_path.split("/", 1)
    if zname not in zips: zips[zname] = zipfile.ZipFile(os.path.join(RAW, "aihub", zname))
    j = json.loads(zips[zname].read(inner).decode("utf-8-sig")); lab, src = j["labeledDataInfo"], j["sourceDataInfo"]
    if r.part == "Part2":
        ss = lab.get("processSentenceInfo") or []
        gold = [i for i, s in enumerate(ss) if str(s.get("subjectConsistencyYn", "Y")).upper() == "N"]
        title = lab.get("newTitle") or src.get("newsTitle")
    else:
        ss = src.get("sentenceInfo") or []
        refs = {int(x["sentenceNo"]) for x in lab.get("referSentenceInfo", []) if str(x.get("referSentenceyn", "N")).upper() == "Y"}
        gold = [i for i, s in enumerate(ss) if int(s["sentenceNo"]) in refs]; title = lab.get("newTitle") or src.get("newsTitle")
    sents = [s["sentenceContent"] for s in ss]
    if sents and gold:
        rows.append(dict(news_id=r.news_id, part=r.part, ptype=r.process_type, category=r.category, title=title, sents=sents, gold=gold))
d = pd.DataFrame(rows); d.to_parquet(os.path.join(OUT, "evidence_input.parquet"), index=False)
print(len(d), d.groupby(["part", "ptype"]).size().to_dict(), "max sents", d.sents.map(len).max())
