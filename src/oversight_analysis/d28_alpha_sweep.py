# α를 바꿀 때 사람 검토량과 사람이 못 보는 놓침이 어떻게 바뀌는가. 출력은 집계뿐이다. 규칙: PREREG_d28_alpha_sweep.md
# d6(--maxq)와 d7의 계산을 같은 규칙으로 재현한다. 기존 calibration_*.json, test_decisions_*.csv를 덮어쓰지 않는다(읽기만 함).
# 입력: outputs/split_manifest.csv, outputs/scores_<name>.csv (news_id, p_cb), 선택: outputs/test_decisions_<tag>.csv (재현 확인용)
# 사용: python d28_alpha_sweep.py <outputs 폴더> [name=axft] [출력폴더=d28_out]
import os, sys, math
import numpy as np, pandas as pd
from scipy.stats import beta

SENSITIVE = {"정치", "경제"}
ALPHA_GRID = [0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30]   # 표시용 격자. 0.05는 현행(common.POLICY)
PI_GRID = [0.05, 0.10, 0.25]                                            # 표시용 격자

def cp(k, n, a=0.05):
    if n == 0: return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a / 2, k + 1, n - k)
    return float(lo), float(hi)

def qhat(s, alpha):
    n = len(s); k = math.ceil((n + 1) * (1 - alpha))
    return (float(np.sort(s)[min(k, n) - 1]) if k <= n else 1.0), n     # d6_calibrate.qhat과 동일

def route(p_cb, q_cb, q_ok):
    inc_cb = (1 - p_cb) <= q_cb; inc_ok = p_cb <= q_ok
    return np.select([inc_cb & ~inc_ok, ~inc_cb & inc_ok], ["cb", "normal"], "ambiguous")

def thresholds(cal, auto, alpha):
    q_dir, n_cb = qhat(1 - cal[cal.label == 1].p_cb.values, alpha)
    q_auto, n_auto = qhat(1 - auto.p_cb.values, alpha)
    q_ok, n_ok = qhat(cal[cal.label == 0].p_cb.values, alpha)
    return max(q_dir, q_auto), q_ok, n_cb, n_auto, n_ok

def main(outputs, name, out):
    man = pd.read_csv(os.path.join(outputs, "split_manifest.csv"))
    sc = pd.read_csv(os.path.join(outputs, f"scores_{name}.csv"))
    d = man.merge(sc, on="news_id")
    cal, auto = d[d.subset == "calib"], d[d.subset == "calib_auto_check"]
    test = d[(d.subset == "test") & (~d.category.isin(SENSITIVE))]
    nn, nc = test[test.label == 0], test[test.label == 1]
    os.makedirs(out, exist_ok=True)
    print(f"name={name}. calib {len(cal)}건, calib_auto_check {len(auto)}건, test 비민감 {len(test)}건 (정상 {len(nn)}, 클릭베이트 {len(nc)}). 격자(α, π)는 표시용이다.")

    # 재현 확인: α=0.05(현행)에서 기존 test_decisions의 set과 같은지
    tf = os.path.join(outputs, f"test_decisions_{name}_maxq.csv")
    q_cb0, q_ok0, *_ = thresholds(cal, auto, 0.05)
    if os.path.exists(tf):
        old = pd.read_csv(tf); old = old[old.subset == "test"][["news_id", "set"]]
        mine = d[d.subset == "test"][["news_id", "p_cb"]].copy(); mine["set_mine"] = route(mine.p_cb.values, q_cb0, q_ok0)
        m = old.merge(mine, on="news_id"); bad = int((m.set != m.set_mine).sum())
        print(f"[재현 확인] α=0.05에서 기존 test_decisions와 비교: {len(m)}건 중 불일치 {bad}건" + ("  <- 불일치가 있으면 이 결과를 쓰지 말 것" if bad else "  (일치)"))
    else:
        print("[재현 확인] 기존 test_decisions 파일이 없어 건너뜀")

    rows = []
    for a in ALPHA_GRID:
        q_cb, q_ok, n_cb, n_auto, n_ok = thresholds(cal, auto, a)
        sn, sc_ = route(nn.p_cb.values, q_cb, q_ok), route(nc.p_cb.values, q_cb, q_ok)
        fc, pn, an = (sn == "cb").mean(), (sn == "normal").mean(), (sn == "ambiguous").mean()
        tp, ms, ac = (sc_ == "cb").mean(), (sc_ == "normal").mean(), (sc_ == "ambiguous").mean()
        k_ms = int((sc_ == "normal").sum()); lo, hi = cp(k_ms, len(nc))
        base = dict(alpha=a, q_clickbait=round(q_cb, 4), q_normal=round(q_ok, 4), n_calib_cb=n_cb, n_calib_auto=n_auto, n_calib_normal=n_ok,
                    fc=round(fc, 4), pn=round(pn, 4), amb_normal=round(an, 4), tp=round(tp, 4), amb_cb=round(ac, 4),
                    miss=round(ms, 4), miss_lo=round(lo, 4), miss_hi=round(hi, 4), miss_le_alpha=bool(hi <= a))
        for pi in PI_GRID:
            review = 1000 * ((1 - pi) * (fc + an) + pi * (tp + ac))      # 사람이 보는 건수(확정 후보 + 판정 유보), 비민감 1,000건당
            auto_n = 1000 * ((1 - pi) * pn + pi * ms)
            rows.append(dict(**base, pi=pi, review_per1000=round(review, 1), autopass_per1000=round(auto_n, 1),
                             unreachable_per1000=round(1000 * pi * ms, 2), unreach_lo=round(1000 * pi * lo, 2), unreach_hi=round(1000 * pi * hi, 2),
                             autopass_clickbait_share=round(pi * ms / (pi * ms + (1 - pi) * pn), 5) if (pi * ms + (1 - pi) * pn) > 0 else None))
    R = pd.DataFrame(rows); R.to_csv(os.path.join(out, f"d28_alpha_sweep_{name}.csv"), index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    cols = ["alpha", "q_clickbait", "q_normal", "fc", "amb_normal", "tp", "amb_cb", "miss", "miss_lo", "miss_hi", "miss_le_alpha"]
    print("\n[α별 비율, 비민감 test, 클래스 조건부 (miss의 95% 구간은 Clopper-Pearson)]"); print(R[R.pi == PI_GRID[0]][cols].to_string(index=False))
    for pi in PI_GRID:
        print(f"\n[비민감 기사 1,000건당, π={pi}]")
        print(R[R.pi == pi][["alpha", "review_per1000", "autopass_per1000", "unreachable_per1000", "unreach_lo", "unreach_hi"]].to_string(index=False))
    print("\n주의: 검토 건수는 확정 후보와 판정 유보의 합이며 민감(정치·경제) 기사는 포함하지 않았다(α와 무관하게 전량 사람이 봄). 검토자가 정확하다는 가정(h=0). n_calib가 작은 α에서는 qhat이 1.0이 되어 전부 검토 대상이 될 수 있다.")

if __name__ == "__main__":
    if len(sys.argv) < 2: sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "axft", sys.argv[3] if len(sys.argv) > 3 else "d28_out")
