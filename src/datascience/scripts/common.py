"""레드투그린 데이터과학 공통 설정과 통계 함수 (BaseKit 4 §4-3·4-5, BaseKit 5 §5-4, BaseKit 7 §7-5)."""
from __future__ import annotations
import math, json, hashlib, os
import numpy as np
from scipy.stats import beta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data_raw")
OUT = os.environ.get("RTG_OUT", os.path.join(ROOT, "outputs"))
os.makedirs(OUT, exist_ok=True)

# 양성(클릭베이트) 정의는 여기 한 곳에서만 관리한다. AI-Hub clickbaitClass 0 = 클릭베이트.
AIHUB_CLICKBAIT_CODE = 0
def is_clickbait_aihub(code) -> int:
    return int(int(code) == AIHUB_CLICKBAIT_CODE)

SEED = 20261102
POLICY = dict(alpha=0.05, eps=0.02, delta=0.05, h_direct=0.02, h_category=0.05, recall_assumed=0.95)
SENSITIVE = {"정치", "경제"}  # 예측집합과 무관하게 인간 이관(HO-02)


def calib_min_n(alpha: float, eps: float, delta: float, n_max: int = 20000) -> int:
    """보정 후 커버리지 ~ Beta(n+1-l, l), l=floor((n+1)a). P(cov >= 1-a-eps) >= 1-delta 를
    만족하고 그 이상 모든 n에서도 성립하는 최소 n(BaseKit 4 §4-5 표 기준)."""
    ok = np.zeros(n_max + 1, bool)
    for n in range(1, n_max + 1):
        l = math.floor((n + 1) * alpha)
        if l < 1:
            continue
        ok[n] = beta.sf(1 - alpha - eps, n + 1 - l, l) >= 1 - delta
    bad = np.where(~ok[1:])[0]
    return int(bad[-1] + 2) if len(bad) else 1


def clopper_pearson(k: int, n: int, conf: float = 0.95):
    if n == 0:
        return (float("nan"), float("nan"))
    a = 1 - conf
    lo = 0.0 if k == 0 else beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a / 2, k + 1, n - k)
    return float(lo), float(hi)


def test_min_n(p: float, h: float, conf: float = 0.95) -> int:
    """재현율 p를 가정했을 때 Clopper–Pearson 구간 반폭이 h 이하가 되는 최소 n."""
    n = 10
    while True:
        k = round(p * n)
        lo, hi = clopper_pearson(k, n, conf)
        if (hi - lo) / 2 <= h:
            return n
        n += 1 if n < 200 else max(1, n // 200)


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """양성 확률 p, 정답 y(1=클릭베이트)의 Expected Calibration Error (동일 폭 구간)."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def file_sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()[:16]


def save_json(obj, name):
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
