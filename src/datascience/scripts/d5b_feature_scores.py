"""개선 기준선(저메모리판): 문자 n-gram TF-IDF + 제목-본문 정합성 특성.
특성: cos(제목,본문), cos(제목,첫 3문장), 문장-나머지본문 유사도 최소·하위2평균, 문장-제목 유사도 최소·평균, log(문장 수).
- 필요한 기사만 pyarrow 필터로 읽고, 문장-나머지 유사도는 S·tot 로 희소 계산(밀집 벡터 없음).
사용법: python d5b_feature_scores.py [학습표본수=15000]  → outputs/scores_feat.csv
"""
import os, re, sys, gc
import numpy as np, pandas as pd, scipy.sparse as sp
import pyarrow.parquet as pq, pyarrow.compute as pc
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, normalize
from common import OUT, SEED

n_train = int(sys.argv[1]) if len(sys.argv) > 1 else 15000
P = os.path.join(OUT, "clean_main.parquet")
meta = pq.read_table(P, columns=["news_id", "split", "hold", "flag_pii", "label"]).to_pandas()
man = pd.read_csv(os.path.join(OUT, "split_manifest.csv"))
held = set(man[man.subset != "dev"].news_id)
pool = meta[(meta.split == "Training") & ~meta.hold & ~meta.flag_pii & meta.label.notna() & ~meta.news_id.isin(held)]
tr_ids = set(pool.sample(n_train, random_state=SEED).news_id)
want = tr_ids | set(man.news_id)
del meta, pool; gc.collect()
tab = pq.read_table(P, columns=["news_id", "title", "body", "label"], filters=[("news_id", "in", list(want))]).to_pandas()
tab["body"] = tab.body.str[:3000]
tr, ev = tab[tab.news_id.isin(tr_ids)], tab[tab.news_id.isin(set(man.news_id))]
print("loaded", len(tr), len(ev), flush=True)
SPLIT = re.compile(r"(?<=[.!?])\s+")
wv = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), min_df=5, max_features=50000, sublinear_tf=True, dtype=np.float32).fit(tr.title + " " + tr.body)


def feats(d):
    T = normalize(wv.transform(d.title)); B = normalize(wv.transform(d.body))
    out = np.zeros((len(d), 7), np.float32)
    out[:, 0] = np.asarray(T.multiply(B).sum(1)).ravel()
    for i, (t, b) in enumerate(zip(d.title.values, d.body.values)):
        ss = [s for s in SPLIT.split(b) if len(s) > 5][:40] or [b[:200]]
        S = normalize(wv.transform(ss)); tv = normalize(wv.transform([t]))
        ts = (S @ tv.T).toarray().ravel()
        if S.shape[0] > 1:
            tot = sp.csr_matrix(S.sum(0))
            st = (S @ tot.T).toarray().ravel()            # s_j · tot
            tt = float((tot @ tot.T).toarray()[0, 0])     # ||tot||²
            rn = np.sqrt(np.maximum(tt - 2 * st + 1, 1e-9))  # ||tot - s_j||
            rs = (st - 1) / rn                            # cos(s_j, 나머지)
        else:
            rs = np.array([1.0])
        out[i, 1:] = [ts[:3].mean(), rs.min(), np.sort(rs)[:2].mean(), ts.min(), ts.mean(), np.log1p(len(ss))]
    return out


txt = lambda x: x.title + " [SEP] " + x.body.str[:1500]
cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3, max_features=150000, sublinear_tf=True, dtype=np.float32)
Xc = cv.fit_transform(txt(tr)); Ft = feats(tr); sc = StandardScaler().fit(Ft)
print("features done", flush=True)
clf = LogisticRegression(max_iter=3000, C=4.0).fit(sp.hstack([Xc, sp.csr_matrix(sc.transform(Ft) * 3)]).tocsr(), tr.label.astype(int))
del Xc; gc.collect()
Xe = sp.hstack([cv.transform(txt(ev)), sp.csr_matrix(sc.transform(feats(ev)) * 3)]).tocsr()
pd.DataFrame({"news_id": ev.news_id.values, "p_cb": clf.predict_proba(Xe)[:, 1]}).to_csv(os.path.join(OUT, "scores_feat.csv"), index=False)
names = ["cos_title_body", "cos_title_first3", "min_sent_rest", "low2_sent_rest", "min_sent_title", "mean_sent_title", "log_nsent"]
print("train", len(tr), "scored", len(ev), dict(zip(names, np.round(clf.coef_[0][-7:], 2))))
