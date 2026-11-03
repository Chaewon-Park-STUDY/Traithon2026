"""카테고리 조건부 보정(BaseKit 5 §5-4)을 보장 조건에 맞는 표본 크기로 다시 계산한다.
- 카테고리 7종 × {Direct 클릭베이트, Auto 클릭베이트, 정상} 각 400건(≥360)을 Training에서 새로 뽑아 '카테고리 보정 셋'으로 쓴다.
- 기준선(TF-IDF+LR)을 저메모리로 다시 학습(학습 표본 20,000, 기존 분할·카테고리 보정 셋과 겹치지 않음)해 테스트·전역 보정·카테고리 보정 셋을 점수화.
- 같은 모델로 전역 보정(수정안)과 카테고리 조건부 보정(수정안)을 테스트 셋에서 비교.
"""
import os, math, gc, json
import numpy as np, pandas as pd, scipy.sparse as sp, pyarrow.parquet as pq
from scipy.stats import beta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from common import OUT, SEED, POLICY, clopper_pearson, save_json

a, eps, NPER = POLICY["alpha"], POLICY["eps"], 400
P = os.path.join(OUT, "clean_main.parquet")
meta = pq.read_table(P, columns=["news_id", "split", "hold", "flag_pii", "label", "process_type", "category"]).to_pandas()
man = pd.read_csv(os.path.join(OUT, "split_manifest.csv"))
pool = meta[(meta.split == "Training") & ~meta.hold & ~meta.flag_pii & meta.label.notna() & ~meta.news_id.isin(man.news_id)].copy()
pool["cell"] = np.where(pool.label == 0, "ok", np.where(pool.process_type == "Direct", "dir", "aut"))
catcal = pool.loc[pool.groupby(["category", "cell"], group_keys=False).apply(lambda g: g.sample(min(NPER, len(g)), random_state=SEED)).index]
rest = pool.drop(catcal.index)
train_ids = set(rest.sample(20000, random_state=SEED + 1).news_id)
need = train_ids | set(catcal.news_id) | set(man[man.subset.isin(["test", "calib", "calib_auto_check"])].news_id)
del meta, pool, rest; gc.collect()
SC = os.path.join(OUT, "scores_catcal_model.csv")
if os.path.exists(SC):
    sc = pd.read_csv(SC)
else:
  tab = pq.read_table(P, columns=["news_id", "title", "body", "label"], filters=[("news_id", "in", list(need))]).to_pandas()
  txt = lambda x: x.title + " [SEP] " + x.body.str[:1500]
  tr = tab[tab.news_id.isin(train_ids)]
  cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3, max_features=150000, sublinear_tf=True, dtype=np.float32)
  clf = LogisticRegression(max_iter=2000, C=4.0).fit(cv.fit_transform(txt(tr)), tr.label.astype(int))
  ev = tab[~tab.news_id.isin(train_ids)]
  sc = pd.DataFrame({"news_id": ev.news_id.values, "p_cb": clf.predict_proba(cv.transform(txt(ev)))[:, 1]})
  sc.to_csv(os.path.join(OUT, "scores_catcal_model.csv"), index=False)
  print("scored", len(sc), flush=True)

def qhat(s):
    n = len(s); k = math.ceil((n + 1) * (1 - a)); return float(np.sort(s)[k - 1]) if k <= n else 1.0
def guar(n):
    l = math.floor((n + 1) * a); return float(beta.sf(1 - a - eps, n + 1 - l, l)) if l >= 1 else 0.0
cc = catcal[["news_id", "category", "cell"]].merge(sc, on="news_id")
g = man.merge(sc, on="news_id")
gcal = g[g.subset.isin(["calib", "calib_auto_check"])]
q_glob_cb = max(qhat(1 - gcal[(gcal.label == 1) & (gcal.process_type == "Direct")].p_cb.values), qhat(1 - gcal[(gcal.label == 1) & (gcal.process_type == "Auto")].p_cb.values))
q_glob_ok = qhat(gcal[gcal.label == 0].p_cb.values)
test = g[g.subset == "test"].copy()
def setof(df, qc, qo):
    ic, io = (1 - df.p_cb) <= qc, df.p_cb <= qo
    return np.select([ic & ~io, ~ic & io], ["cb", "normal"], "amb")
test["set_glob"] = setof(test, q_glob_cb, q_glob_ok)
test["set_cat"] = ""
qrows = []
for c, sub in cc.groupby("category"):
    qc = max(qhat(1 - sub[sub.cell == "dir"].p_cb.values), qhat(1 - sub[sub.cell == "aut"].p_cb.values)); qo = qhat(sub[sub.cell == "ok"].p_cb.values)
    m = test.category == c; test.loc[m, "set_cat"] = setof(test[m], qc, qo)
    n = int(sub.groupby("cell").size().min()); qrows.append(dict(category=c, n_min=n, p_guarantee=round(guar(n), 3), q_cb=qc, q_ok=qo))
rows = []
for c, sub in test.groupby("category"):
    r = dict(category=c)
    for mode in ("glob", "cat"):
        cb = sub[sub.label == 1]; ok = sub[sub.label == 0]
        k = int((cb[f"set_{mode}"] != "normal").sum()); r[f"cov_{mode}"] = k / len(cb); r[f"cov_{mode}_lo"] = clopper_pearson(k, len(cb))[0]
        r[f"fc_{mode}"] = float((ok[f"set_{mode}"] == "cb").mean()); r[f"refer_{mode}"] = float((sub[f"set_{mode}"] == "amb").mean())
    rows.append(r)
res = pd.DataFrame(rows).merge(pd.DataFrame(qrows), on="category")
res.to_csv(os.path.join(OUT, "category_calib_v2.csv"), index=False, encoding="utf-8-sig")
summ = {m: dict(cov_range=[float(res[f"cov_{m}"].min()), float(res[f"cov_{m}"].max())], cov_gap=float(res[f"cov_{m}"].max() - res[f"cov_{m}"].min()),
                refer_overall=float((test[f"set_{m}"] == "amb").mean()),
                cov_direct=float((test[(test.label == 1) & (test.process_type == "Direct")][f"set_{m}"] != "normal").mean()),
                cov_auto=float((test[(test.label == 1) & (test.process_type == "Auto")][f"set_{m}"] != "normal").mean())) for m in ("glob", "cat")}
summ["n_per_cell"] = int(res.n_min.min()); summ["p_guarantee_min"] = float(res.p_guarantee.min())
save_json(summ, "category_calib_v2_summary.json")
pd.set_option("display.width", 220); print(res.round(3).to_string(index=False)); print(json.dumps(summ, ensure_ascii=False, indent=1))
