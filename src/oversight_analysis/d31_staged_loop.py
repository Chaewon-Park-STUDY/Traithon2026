# d31 단계형 검토(L0~L4)와 피드백 루프 비용. 출력은 집계뿐. 규칙: PREREG_d31_staged_loop.md
# d28_alpha_sweep.py(임계값·판정 규칙)를 재사용한다. 기존 결과 파일은 읽기만 한다.
# 사용: python d31_staged_loop.py <outputs 폴더> [name=axft] [출력폴더=d31_out] [d28결과csv=d28_out/d28_alpha_sweep_<name>.csv]
import os, sys
import numpy as np, pandas as pd
from scipy.stats import binom
from d28_alpha_sweep import thresholds, route, ALPHA_GRID, PI_GRID, SENSITIVE

SEED, B_BOOT, DELTA = 20261109, 2000, 0.05           # DELTA는 common.POLICY와 같은 값이어야 한다(팀 확인 필요)
X_GRID = [0.0, 0.01, 0.05, 0.10, 0.20, 0.50]
MULT = [1.0, 1.5, 2.0, 3.0]; N_AUDIT = [203, 500, 1000, 3000]; FA_GRID = [0.02, 0.05, 0.10]
RHO_GRID = [1, 3, 10, 30, 100, 300, 1000]

def one(pn, pc, q_cb, q_ok):
    """test의 정상 pn, 클릭베이트 pc 점수로 (π,x)별 1,000건당 검토 건수와 못 보는 놓침을 계산"""
    sn, sc = route(pn, q_cb, q_ok), route(pc, q_cb, q_ok); n_n, n_c = len(pn), len(pc)
    fc, an = (sn == "cb").mean(), (sn == "ambiguous").mean(); tp, ac, ms = (sc == "cb").mean(), (sc == "ambiguous").mean(), (sc == "normal").mean()
    p_all = np.concatenate([pn[sn == "normal"], pc[sc == "normal"]])
    is_c = np.concatenate([np.zeros(int((sn == "normal").sum()), bool), np.ones(int((sc == "normal").sum()), bool)])
    is_c = is_c[np.argsort(-p_all, kind="stable")]
    rev, un = np.zeros((len(PI_GRID), len(X_GRID))), np.zeros((len(PI_GRID), len(X_GRID)))
    for i, pi in enumerate(PI_GRID):
        w = np.where(is_c, pi / n_c, (1 - pi) / n_n); cw = np.cumsum(w); cc = np.cumsum(np.where(is_c, w, 0.0)); W = cw[-1] if len(cw) else 0.0
        base = 1000 * ((1 - pi) * (fc + an) + pi * (tp + ac))
        for j, x in enumerate(X_GRID):
            k = int(np.searchsorted(cw, x * W, side="right")) if (x > 0 and W > 0) else 0
            rev[i, j] = base + (1000 * cw[k - 1] if k > 0 else 0.0); un[i, j] = 1000 * pi * ms - (1000 * cc[k - 1] if k > 0 else 0.0)
    return rev, np.maximum(un, 0.0), (sn == "normal").mean()   # 부동소수 오차로 생기는 -0.00 제거

def kappa(pi, pn, m): return pi * m / (pi * m + (1 - pi) * pn)

def main(outputs, name="axft", out="d31_out", d28csv=None, B=B_BOOT):
    man = pd.read_csv(os.path.join(outputs, "split_manifest.csv")); sc = pd.read_csv(os.path.join(outputs, f"scores_{name}.csv")); d = man.merge(sc, on="news_id")
    cal, auto = d[d.subset == "calib"], d[d.subset == "calib_auto_check"]
    test = d[(d.subset == "test") & (~d.category.isin(SENSITIVE))]; pn, pc = test[test.label == 0].p_cb.values, test[test.label == 1].p_cb.values
    os.makedirs(out, exist_ok=True); nA, nP, nX = len(ALPHA_GRID), len(PI_GRID), len(X_GRID)
    print(f"name={name}. test 비민감 {len(test)}건 (정상 {len(pn)}, 클릭베이트 {len(pc)}). 격자는 표시용이다. bootstrap B={B}.")
    th = [thresholds(cal, auto, a) for a in ALPHA_GRID]
    R, U = np.zeros((B + 1, nA, nP, nX)), np.zeros((B + 1, nA, nP, nX)); pn_auto = []
    for ai, t in enumerate(th):
        R[0, ai], U[0, ai], p_ = one(pn, pc, t[0], t[1]); pn_auto.append(p_)
    rng = np.random.default_rng(SEED)
    for b in range(1, B + 1):
        bn, bc = pn[rng.integers(0, len(pn), len(pn))], pc[rng.integers(0, len(pc), len(pc))]
        for ai, t in enumerate(th): R[b, ai], U[b, ai], _ = one(bn, bc, t[0], t[1])
    # 재현 확인
    f28 = d28csv or os.path.join("d28_out", f"d28_alpha_sweep_{name}.csv")
    if os.path.exists(f28):
        o = pd.read_csv(f28); bad = 0
        for ai, a in enumerate(ALPHA_GRID):
            for i, pi in enumerate(PI_GRID):
                r = o[(np.isclose(o.alpha, a)) & (np.isclose(o.pi, pi))].iloc[0]
                bad += int(abs(r.review_per1000 - R[0, ai, i, 0]) > 0.06 or abs(r.unreachable_per1000 - U[0, ai, i, 0]) > 0.006)
        print(f"[재현 확인] x=0 값을 d28 결과와 비교: 불일치 {bad}건 / {nA * nP}건" + ("  <- 불일치가 있으면 이 결과를 쓰지 말 것" if bad else "  (일치)"))
    else: print("[재현 확인] d28 결과 CSV가 없어 건너뜀")
    # 부분 1: L2
    rows = []
    for ai, a in enumerate(ALPHA_GRID):
        for i, pi in enumerate(PI_GRID):
            for j, x in enumerate(X_GRID):
                rl, rh = np.percentile(R[1:, ai, i, j], [2.5, 97.5]); ul, uh = np.percentile(U[1:, ai, i, j], [2.5, 97.5])
                rows.append(dict(alpha=a, pi=pi, x=x, review_per1000=round(R[0, ai, i, j], 1), review_lo=round(rl, 1), review_hi=round(rh, 1),
                                 unreachable_per1000=round(U[0, ai, i, j], 2), unreach_lo=round(ul, 2), unreach_hi=round(uh, 2)))
    L2 = pd.DataFrame(rows); L2.to_csv(os.path.join(out, f"d31_L2_{name}.csv"), index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    a0 = ALPHA_GRID.index(0.05)
    for i, pi in enumerate(PI_GRID):
        print(f"\n[부분 1] α=0.05(현행)에서 자동 통과를 p_cb 순으로 x만큼 추가 검토, π={pi}, 기사 1,000건당")
        print(L2[(np.isclose(L2.alpha, 0.05)) & (np.isclose(L2.pi, pi))][["x", "review_per1000", "review_lo", "review_hi", "unreachable_per1000", "unreach_lo", "unreach_hi"]].to_string(index=False))
    # 부분 2: 경보 지연
    rows = []
    for ai, a in enumerate(ALPHA_GRID):
        for pi in PI_GRID:
            for mu in MULT:
                k0, k1 = kappa(pi, pn_auto[ai], a), kappa(pi, pn_auto[ai], min(mu * a, 1.0))
                for Ba in N_AUDIT:
                    c = int(binom.ppf(1 - DELTA, Ba, k0)); q = float(binom.sf(c, Ba, k1)); E = (1 / q) if q > 0 else float("inf")
                    for fa in FA_GRID:
                        rows.append(dict(alpha=a, pi=pi, mult=mu, audit_batch=Ba, f_a=fa, alarm_prob=round(q, 4), expected_batches=round(E, 2),
                                         audited_until_alarm=round(Ba * E, 0), autopass_until_alarm=round(Ba * E / fa, 0), leaked_clickbait=round(k1 * Ba * E * (1 - fa) / fa, 1)))
    DL = pd.DataFrame(rows); DL.to_csv(os.path.join(out, f"d31_delay_{name}.csv"), index=False, encoding="utf-8-sig")
    for pi in PI_GRID:
        s = DL[(np.isclose(DL.alpha, 0.05)) & (np.isclose(DL.pi, pi)) & (np.isclose(DL.f_a, 0.05))]
        print(f"\n[부분 2] α=0.05, π={pi}, 감사율 f_a=0.05: 경보 확률(묶음당) / 새어 나간 클릭베이트(경보까지 기대)")
        print(s.pivot(index="mult", columns="audit_batch", values="alarm_prob").to_string()); print(s.pivot(index="mult", columns="audit_batch", values="leaked_clickbait").to_string())
    # 부분 3: 비용
    rows = []; ab = ALPHA_GRID.index(0.05)
    for i, pi in enumerate(PI_GRID):
        for rho in RHO_GRID:
            C = R[:, :, i, :] + rho * U[:, :, i, :]; flat = C.reshape(B + 1, -1); best = int(flat[0].argmin()); bi, bj = divmod(best, nX)
            share = float((flat[1:].argmin(axis=1) == best).mean()); base = C[0, ab, 0]
            rows.append(dict(pi=pi, rho=rho, best_alpha=ALPHA_GRID[bi], best_x=X_GRID[bj], best_cost=round(flat[0, best], 1), baseline_cost=round(base, 1),
                             saving=round(base - flat[0, best], 1), boot_share_best=round(share, 3)))
    CS = pd.DataFrame(rows); CS.to_csv(os.path.join(out, f"d31_cost_{name}.csv"), index=False, encoding="utf-8-sig")
    print("\n[부분 3] 비용=검토 건수+ρ×못 보는 놓침. 격자 안 가장 싼 설정과 현행(α=0.05, x=0), bootstrap에서 그 설정이 최저가인 비율"); print(CS.to_string(index=False))
    print("\n주의: 설계 비교이며 최적 설정을 고르는 분석이 아니다. ρ, 배수, 감사 묶음 크기는 가정(표시용 격자)이다. 검토자 정확(h=0). 구간은 test만 bootstrap(보정 고정).")

if __name__ == "__main__":
    if len(sys.argv) < 2: sys.exit(__doc__)
    a = sys.argv; main(a[1], a[2] if len(a) > 2 else "axft", a[3] if len(a) > 3 else "d31_out", a[4] if len(a) > 4 else None)
