"""개선 기준선: 문자 n-gram TF-IDF + 제목-본문 정합성 특성(Part1 불일치, Part2 이탈 문장 탐지용).
특성: cos(제목, 본문), cos(제목, 첫 3문장), 문장별 '나머지 본문 중심'과의 최소 유사도, 문장별 제목 유사도 최소/평균, 문장 수.
사용법: python d5b_feature_scores.py [학습표본수=40000]  → outputs/scores_feat.csv
"""
import os, re, sys
import numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from common import OUT, SEED

n_train = int(sys.argv[1]) if len(sys.argv) > 1 else 40000
df = pd.read_parquet(os.path.join(OUT, "clean_main.parquet"))
man = pd.read_csv(os.path.join(OUT, "split_manifest.csv"))
held = set(man[man.subset != "dev"].news_id)
df = df[~df.hold & ~df.flag_pii & df.label.notna()]
tr = df[(df.split == "Training") & ~df.news_id.isin(held)].sample(n_train, random_state=SEED)
ev = df[df.news_id.isin(set(man.news_id))]
SPLIT = re.compile(r"(?<=[.!?])\s+|(?<=다\.)\s*")
wv = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), min_df=5, max_features=200000, sublinear_tf=True).fit(tr.title + " " + tr.body.str[:1500])


def feats(d):
    T = normalize(wv.transform(d.title)); B = normalize(wv.transform(d.body.str[:3000]))
    out = np.zeros((len(d), 7))
    out[:, 0] = np.asarray(T.multiply(B).sum(1)).ravel()
    for i, (t, b) in enumerate(zip(d.title.values, d.body.values)):
        ss = [s for s in SPLIT.split(b[:4000]) if len(s) > 5][:40] or [b[:200]]
        S = normalize(wv.transform(ss)); tv = normalize(wv.transform([t]))
        ts = (S @ tv.T).toarray().ravel()
        tot = np.asarray(S.sum(0)).ravel()
        rest = [(tot - S[j].toarray().ravel()) for j in range(S.shape[0])]
        rs = np.array([S[j].toarray().ravel() @ (r / (np.linalg.norm(r) + 1e-9)) for j, r in enumerate(rest)]) if len(ss) > 1 else np.array([1.0])
        out[i, 1:] = [ts[:3].mean(), rs.min(), np.sort(rs)[:2].mean(), ts.min(), ts.mean(), np.log1p(len(ss))]
    return out


txt = lambda x: x.title + " [SEP] " + x.body.str[:1500]
cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3, max_features=300000, sublinear_tf=True)
Xc = cv.fit_transform(txt(tr))
sc = StandardScaler().fit(feats(tr.head(1)) * 0 + 0) if False else None
Ft = feats(tr); sc = StandardScaler().fit(Ft)
X = sp.hstack([Xc, sp.csr_matrix(sc.transform(Ft) * 3)]).tocsr()
clf = LogisticRegression(max_iter=3000, C=4.0).fit(X, tr.label.astype(int))
Xe = sp.hstack([cv.transform(txt(ev)), sp.csr_matrix(sc.transform(feats(ev)) * 3)]).tocsr()
pd.DataFrame({"news_id": ev.news_id, "p_cb": clf.predict_proba(Xe)[:, 1]}).to_csv(os.path.join(OUT, "scores_feat.csv"), index=False)
print("train", len(tr), "scored", len(ev), "feature coefs", np.round(clf.coef_[0][-7:], 2))
