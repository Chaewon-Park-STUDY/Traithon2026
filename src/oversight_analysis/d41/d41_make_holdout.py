"""d41 holdout 입력 생성 (원문 포함 -> 저장소 업로드 금지, 로컬/GPU 서버 전용).
조건: Training 분할, hold/flag_pii 아님, 라벨 있음, 그리고 아래를 모두 제외
  - split_manifest(보정/테스트/dev) 기사
  - d18 학습에 쓰인 기사(pick 7,000x3)와 인젝션 증강 원본(extra 700x3)  <- d18과 동일한 sample(random_state=SEED)로 재현
사용: cd src/datascience/scripts ; python -X utf8 d41_make_holdout.py --n 1000
  --n: 그룹(direct_cb / auto_cb / normal)당 건수. 크기는 d41 PREREG에서 점수를 보기 전에 정한다.
산출: outputs/ax_input_holdout.parquet (key,group,lang,title,body) -> GPU 서버 /root/Data/ 로 복사
      outputs/holdout_meta.csv (key,label,process_type,category) -> 로컬 전용, 저장소 금지
"""
import os, argparse
import numpy as np, pandas as pd, pyarrow.parquet as pq
from common import OUT, SEED

ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, required=True); a = ap.parse_args()
man = pd.read_csv(os.path.join(OUT, "split_manifest.csv")); held = set(man.news_id)
meta = pq.read_table(os.path.join(OUT, "clean_main.parquet"),
                     columns=["news_id", "split", "process_type", "category", "label", "hold", "flag_pii"]).to_pandas()
meta = meta[(meta.split == "Training") & ~meta.hold & ~meta.flag_pii & meta.label.notna() & ~meta.news_id.isin(held)]
groups = {"direct_cb": meta[(meta.label == 1) & (meta.process_type == "Direct")],
          "auto_cb": meta[(meta.label == 1) & (meta.process_type == "Auto")],
          "normal": meta[meta.label == 0]}
used = set()
for g, d in groups.items():                       # d18과 동일 호출 -> 동일 표본(학습 7,000 + 증강 700)
    used |= set(d.sample(7000 + 700, random_state=SEED).news_id)
rng = np.random.default_rng(SEED + 37)
pick = []
for g, d in groups.items():
    pool = d[~d.news_id.isin(used)]
    assert len(pool) >= a.n, (g, len(pool))
    pick.append(pool.sample(a.n, random_state=int(rng.integers(1 << 31))).assign(grp=g))
pick = pd.concat(pick, ignore_index=True)
tab = pq.read_table(os.path.join(OUT, "clean_main.parquet"), columns=["news_id", "title", "body"],
                    filters=[("news_id", "in", list(pick.news_id))]).to_pandas().set_index("news_id")
pick["title"] = tab.loc[pick.news_id, "title"].values; pick["body"] = tab.loc[pick.news_id, "body"].str[:1500].values
pick["key"] = pick.news_id.astype(str)
assert not (set(pick.news_id) & (used | held)) and pick.news_id.is_unique
pick.assign(group="holdout", lang="ko")[["key", "group", "lang", "title", "body"]].to_parquet(os.path.join(OUT, "ax_input_holdout.parquet"), index=False)
pick[["key", "label", "process_type", "category", "grp"]].to_csv(os.path.join(OUT, "holdout_meta.csv"), index=False)
print("holdout", len(pick), pick.groupby("grp").size().to_dict(), "pool left", {g: int((~d.news_id.isin(used)).sum()) for g, d in groups.items()})
