"""비교 기준선 점수(문자 n-gram TF-IDF + 로지스틱 회귀). A.X 4.0 Light 대체가 아니라
(1) 파이프라인 점검, (2) 'A.X가 단순 기준선보다 나은가'의 비교용이다. 학습은 dev+Training 잔여 표본, 보정·테스트 표본은 학습에서 제외.
사용법: python d5_baseline_scores.py [학습표본수=40000]
"""
import os, sys
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from common import OUT, SEED

n_train = int(sys.argv[1]) if len(sys.argv) > 1 else 40000
df = pd.read_parquet(os.path.join(OUT, "clean_main.parquet"))
man = pd.read_csv(os.path.join(OUT, "split_manifest.csv"))
held = set(man[man.subset != "dev"].news_id)
df = df[~df.flag_pii]
tr = df[(df.split == "Training") & ~df.news_id.isin(held) & ~df.hold & df.label.notna()]
tr = tr.sample(min(n_train, len(tr)), random_state=SEED)
txt = lambda x: (x.title + " [SEP] " + x.body.str[:1500])
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3, max_features=300000, sublinear_tf=True)
X = vec.fit_transform(txt(tr))
clf = LogisticRegression(max_iter=2000, C=4.0).fit(X, tr.label.astype(int))
ev = df[df.news_id.isin(set(man.news_id))]
pd.DataFrame({"news_id": ev.news_id, "p_cb": clf.predict_proba(vec.transform(txt(ev)))[:, 1]}).to_csv(
    os.path.join(OUT, "scores_baseline.csv"), index=False)
print("train", len(tr), "scored", len(ev))
