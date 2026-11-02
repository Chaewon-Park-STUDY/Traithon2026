"""D3·D4: 슬라이스 태깅과 통계 기반 분할 (BaseKit 4 §4-5, §4-7).
- Training → dev / calib, Validation → test. 기사 ID 단위, 클래스×유형×Part×카테고리 층화, 시드 기록.
- 크기: calib 클래스별 ≥ calib_min_n(α,ε,δ) 이고 클릭베이트 calib 은 Direct 만으로 그 크기를 채운다(보수적 q̂).
  test 는 Direct 클릭베이트 ≥ n_direct, 카테고리별 클릭베이트 ≥ n_cat, 정상도 카테고리별 ≥ n_cat.
사용법: python d4_split.py [--scale 1.0]   (scale 로 최소 크기에 여유 배수를 준다)
"""
import os, sys, json
import numpy as np, pandas as pd
from common import OUT, SEED, POLICY, calib_min_n, test_min_n, save_json

scale = float(sys.argv[sys.argv.index("--scale") + 1]) if "--scale" in sys.argv else 1.5
P = POLICY
df = pd.read_parquet(os.path.join(OUT, "clean_main.parquet"))
df = df[~df.hold & ~df.flag_pii & df.label.notna()].copy()  # DQ-04 입력 이상, DQ-05 개인정보 의심 제외
df["label"] = df.label.astype(int)

# BaseKit 4 §4-5 표 값과 Clopper–Pearson 계산값 중 큰 쪽을 최소 크기로 쓴다
BK_TABLE_TEST = {(0.95, 0.02): 589, (0.95, 0.05): 127}
n_cal = calib_min_n(P["alpha"], P["eps"], P["delta"])
n_dir = max(test_min_n(P["recall_assumed"], P["h_direct"]), BK_TABLE_TEST.get((P["recall_assumed"], P["h_direct"]), 0))
n_cat = max(test_min_n(P["recall_assumed"], P["h_category"]), BK_TABLE_TEST.get((P["recall_assumed"], P["h_category"]), 0))
target = dict(calib_per_class=int(np.ceil(n_cal * scale)), test_direct_cb=int(np.ceil(n_dir * scale)),
              test_per_cat_per_class=int(np.ceil(n_cat * scale)))
rng = np.random.default_rng(SEED)


def strat_sample(pool, n, keys):
    """keys 셀 비율대로 n 건 층화 무작위 추출(셀별 최소 1건)."""
    if n >= len(pool):
        return pool.index
    g = pool.groupby(keys)
    w = g.size() / len(pool)
    alloc = np.maximum(1, np.floor(w * n)).astype(int)
    idx = []
    for k, sub in g:
        m = min(len(sub), alloc[k])
        idx += list(rng.choice(sub.index, m, replace=False))
    rest = pool.index.difference(idx)
    if len(idx) < n:
        idx += list(rng.choice(rest, n - len(idx), replace=False))
    return pd.Index(idx[:n])


tr, va = df[df.split == "Training"], df[df.split == "Validation"]
keys = ["process_type", "part", "category"]
# 보정 셋: 클릭베이트는 Direct 만(보수적), 정상은 전체에서
cal_cb = strat_sample(tr[(tr.label == 1) & (tr.process_type == "Direct")], target["calib_per_class"], ["part", "category"])
cal_ok = strat_sample(tr[tr.label == 0], target["calib_per_class"], keys)
# Auto 클릭베이트 커버리지 별도 실측용(보정에는 안 씀)
cal_auto = strat_sample(tr[(tr.label == 1) & (tr.process_type == "Auto")], target["calib_per_class"], ["part", "category"])
used = cal_cb.union(cal_ok).union(cal_auto)
# dev 셋: 프롬프트·퓨샷 선정용, 클래스별 500건
dev_pool = tr.drop(used)
dev = strat_sample(dev_pool[dev_pool.label == 1], 500, keys).union(strat_sample(dev_pool[dev_pool.label == 0], 500, keys))
# 테스트 셋(Validation): 카테고리×클래스별 최소 + Direct 클릭베이트 최소 + Auto 클릭베이트 동일 크기
test = []
for (c, y), sub in va.groupby(["category", "label"]):
    test += list(strat_sample(sub, target["test_per_cat_per_class"], ["process_type", "part"]))
test = pd.Index(test)
for ptype in ["Direct", "Auto"]:
    have = va.loc[test].query("label == 1 and process_type == @ptype")
    need = target["test_direct_cb"] - len(have)
    if need > 0:
        pool = va[(va.label == 1) & (va.process_type == ptype)].drop(test, errors="ignore")
        test = test.union(strat_sample(pool, need, ["part", "category"]))

parts = [("dev", dev), ("calib", cal_cb.union(cal_ok)), ("calib_auto_check", cal_auto), ("test", test)]
man = pd.concat([df.loc[ix, ["news_id", "split", "label", "process_type", "part", "category", "level"]].assign(subset=n) for n, ix in parts])
assert man.news_id.duplicated().sum() == 0, "분할 간 기사 ID 중복"
man.to_csv(os.path.join(OUT, "split_manifest.csv"), index=False, encoding="utf-8-sig")

# 슬라이스 레지스트리(§4-7)
sl = df[["news_id", "label", "process_type", "part", "category", "level"]].copy()
sl["SLICE_AUTO"] = (sl.label == 1) & (sl.process_type == "Auto")
sl["SLICE_DIRECT"] = (sl.label == 1) & (sl.process_type == "Direct")
sl["SLICE_POLITICS"] = sl.category == "정치"
sl["SLICE_ECON"] = sl.category == "경제"
sl[sl.news_id.isin(man.news_id)].to_csv(os.path.join(OUT, "slice_registry.csv"), index=False, encoding="utf-8-sig")

summary = dict(seed=SEED, policy=P, scale=scale, min_sizes=dict(calib=n_cal, test_direct=n_dir, test_cat=n_cat), target=target,
               counts=man.groupby(["subset", "label", "process_type"]).size().rename("n").reset_index().to_dict("records"))
save_json(summary, "split_summary.json")
print(json.dumps(summary["min_sizes"]), json.dumps(target)); print(man.groupby(["subset", "label"]).size())
