# 검토의 이득 상한(headroom)과 감독 범위 밖 오류. 출력은 집계뿐이다. 규칙: PREREG_d25_headroom.md
# AI 판정 = 시스템이 실제로 내린 conformal 라우팅(임계값을 새로 정하지 않음).
#   set=="cb"      확정 후보(사람이 검토) -> 정상이면 AI 오류(false confirm), 사람이 고칠 수 있음
#   set=="normal"  자동 통과(사람이 안 봄) -> 클릭베이트이면 AI 오류(미포착), 사람이 고칠 수 없음
#   set=="ambiguous" 판정 유보(사람이 판단), 민감(정치·경제)은 항상 사람 -> 오류로 세지 않음
# 입력: outputs/test_decisions_<tag>.csv (컬럼: news_id, category, label, process_type, set)
# 사용: python d25_headroom.py <outputs 폴더> [tag=axft_maxq] [출력폴더=d25_out] [s=0.278]
import os, sys
import numpy as np, pandas as pd
from scipy.stats import beta

SENSITIVE = {"정치", "경제"}

def cp(k, n, a=0.05):
    if n == 0: return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a / 2, k + 1, n - k)
    return float(lo), float(hi)

def rate(k, n):
    lo, hi = cp(k, n)
    return dict(n=n, k=k, rate=k / n if n else float("nan"), lo=lo, hi=hi)

def main(outputs, tag, out, s):
    d = pd.read_csv(os.path.join(outputs, f"test_decisions_{tag}.csv"))
    d["sens"] = d.category.isin(SENSITIVE)
    d["cls"] = np.where(d.label == 1, "클릭베이트", "정상")
    os.makedirs(out, exist_ok=True)

    # 1) 경로별 비율 (클래스 조건부): 비민감, 민감(반사실: 규칙이 없었다면 AI가 내렸을 판정)
    rows = []
    for sens, gname in ((False, "비민감"), (True, "민감(반사실)")):
        for cls in ("정상", "클릭베이트"):
            for ptype in ["전체"] + sorted(d.process_type.unique()):
                g = d[(d.sens == sens) & (d.cls == cls)]
                if ptype != "전체": g = g[g.process_type == ptype]
                if len(g) == 0: continue
                for sv, lab in (("cb", "확정 후보"), ("normal", "자동 통과"), ("ambiguous", "판정 유보")):
                    rows.append(dict(group=gname, cls=cls, process_type=ptype, route=lab, **rate(int((g.set == sv).sum()), len(g))))
    R = pd.DataFrame(rows).round(4)
    R.to_csv(os.path.join(out, f"d25_route_rates_{tag}.csv"), index=False, encoding="utf-8-sig")

    # 2) 서비스 환산: 비민감, 기사 1,000건당, π 곡선. π와 s는 가정이다(s는 BaseKit 1 Training 구성비).
    ns = d[~d.sens]
    nn, nc = ns[ns.cls == "정상"], ns[ns.cls == "클릭베이트"]
    fc = rate(int((nn.set == "cb").sum()), len(nn))          # 정상 -> 확정 후보 (고칠 수 있는 오류)
    tp = rate(int((nc.set == "cb").sum()), len(nc))          # 클릭베이트 -> 확정 후보 (맞은 확정)
    ms = rate(int((nc.set == "normal").sum()), len(nc))      # 클릭베이트 -> 자동 통과 (고칠 수 없는 오류)
    grid = [round(x, 2) for x in np.arange(0.05, 0.51, 0.05)]  # 표시용 격자. 값 선택이 아님
    ds = [0.0, 0.25, 0.5, 0.75, 1.0]                            # 표시용 격자
    H = []
    for pi in grid:
        def q(f, t, m):  # f,t,m: fc,tp,miss 비율 값
            queue = (1 - s) * 1000 * ((1 - pi) * f + pi * t)
            fix = (1 - s) * 1000 * (1 - pi) * f
            unr = (1 - s) * 1000 * pi * m
            e = (1 - pi) * f / ((1 - pi) * f + pi * t) if ((1 - pi) * f + pi * t) > 0 else float("nan")
            share = pi * m / ((1 - pi) * f + pi * m) if ((1 - pi) * f + pi * m) > 0 else float("nan")
            return queue, fix, unr, e, share
        pt = q(fc["rate"], tp["rate"], ms["rate"])
        # 범위: 구간 끝점의 보수적 조합(신뢰구간이 아님). 감독 범위 밖 비율은 miss 하한·fc 상한이 최소, 반대가 최대
        lo_share = q(fc["hi"], tp["rate"], ms["lo"])[4]; hi_share = q(fc["lo"], tp["rate"], ms["hi"])[4]
        lo_fix = q(fc["lo"], tp["rate"], ms["rate"])[1]; hi_fix = q(fc["hi"], tp["rate"], ms["rate"])[1]
        lo_unr = q(fc["rate"], tp["rate"], ms["lo"])[2]; hi_unr = q(fc["rate"], tp["rate"], ms["hi"])[2]
        for dd in ds:
            H.append(dict(pi=pi, d=dd, review_queue_per1000=round(pt[0], 2),
                          ai_error_rate_in_queue_e=round(pt[3], 4),
                          fixable_errors_per1000=round(pt[1], 2),
                          benefit_upper_bound_per1000=round((1 - dd) * pt[1], 2),   # h=0(완벽한 독립 검토자)일 때의 상한
                          benefit_ub_lo=round((1 - dd) * lo_fix, 2), benefit_ub_hi=round((1 - dd) * hi_fix, 2),
                          unreachable_errors_per1000=round(pt[2], 2), unreach_lo=round(lo_unr, 2), unreach_hi=round(hi_unr, 2),
                          share_outside_oversight=round(pt[4], 4), share_lo=round(lo_share, 4), share_hi=round(hi_share, 4)))
    Hd = pd.DataFrame(H)
    Hd.to_csv(os.path.join(out, f"d25_headroom_{tag}.csv"), index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    print(f"tag={tag} test {len(d)}건 (비민감 {len(ns)}, 민감 {int(d.sens.sum())}). s={s}는 가정(BaseKit 1 Training 구성비), π 격자는 표시용.")
    print("\n[경로별 비율, 클래스 조건부, 95% Clopper-Pearson]"); print(R.to_string(index=False))
    print("\n[비민감 핵심 비율] 정상->확정 후보(고칠 수 있는 오류) fc=%.4f [%.4f, %.4f] | 클릭베이트->자동 통과(고칠 수 없는 오류) miss=%.4f [%.4f, %.4f]"
          % (fc["rate"], fc["lo"], fc["hi"], ms["rate"], ms["lo"], ms["hi"]))
    print("\n[기사 1,000건당, d=0(AI를 전혀 따르지 않는 독립 검토)일 때의 상한과 감독 범위 밖 비율]")
    print(Hd[Hd.d == 0.0][["pi", "review_queue_per1000", "ai_error_rate_in_queue_e", "benefit_upper_bound_per1000", "unreachable_errors_per1000", "share_outside_oversight", "share_lo", "share_hi"]].to_string(index=False))
    print("\n주의: 범위(lo/hi)는 비율 구간의 끝점을 보수적으로 조합한 값이며 신뢰구간이 아니다. h는 추정하지 않았고 상한은 h=0 가정이다. 이득이 생기려면 h < e 여야 한다.")

if __name__ == "__main__":
    if len(sys.argv) < 2: sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "axft_maxq",
         sys.argv[3] if len(sys.argv) > 3 else "d25_out", float(sys.argv[4]) if len(sys.argv) > 4 else 0.278)
