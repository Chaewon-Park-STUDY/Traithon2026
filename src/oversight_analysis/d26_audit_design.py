# OPS-01 감사 표본 설계 근거. 출력은 집계뿐이다. 규칙: PREREG_d26_audit_design.md
# 부분 1: 자동 통과에서 무작위 n건을 사람이 라벨링할 때 경보가 울릴 확률(alarm_prob). 진짜 miss > α 이면 검정력, 아니면 오경보 확률.
# 부분 2: 자동 통과를 p_cb 순으로 줄 세웠을 때 놓친 클릭베이트가 얼마나 앞쪽에 몰리는지(lift, AUC), bootstrap 구간.
# 입력: outputs/test_decisions_<tag>.csv (news_id, category, label, process_type, set [+ p_cb 있으면 사용])
#       없으면 outputs/scores_<이름>.csv (key, p_cb ...)를 news_id와 key로 합친다(scores= 로 경로 지정 가능).
# 사용: python d26_audit_design.py <outputs 폴더> [tag=axft_maxq] [출력폴더=d26_out] [scores=<csv 경로>]
import os, sys, glob
import numpy as np, pandas as pd
from scipy.stats import beta, binom, rankdata

SENSITIVE = {"정치", "경제"}
ALPHA = 0.05      # common.POLICY["alpha"]
DELTA = 0.05      # common.POLICY["delta"] (감사 검정의 오경보 허용 확률로 재사용: 팀 결정 사항)
PLAN_N = 203      # 기획본 감사 표본 크기(참조선)
N_MAX = 100000
PI_GRID = [round(x, 2) for x in np.arange(0.05, 0.51, 0.05)]   # 표시용 격자(d25와 동일)
MULT = [1.0, 1.5, 2.0, 3.0]                                    # 표시용: 진짜 miss = 배수 × α
N_SHOW = [100, 203, 300, 500, 750, 1000, 1500, 2000, 3000, 5000, 7500, 10000, 20000, 50000]
X_GRID = [0.01, 0.02] + [round(x, 2) for x in np.arange(0.05, 1.0001, 0.05)]
B_BOOT = 2000     # bootstrap 반복 수(수치 안정성용)
SEED = 20261107

def cp(k, n, a=0.05):
    if n == 0: return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a / 2, k + 1, n - k)
    return float(lo), float(hi)

def kappa(m, pi, pn):
    """자동 통과 안의 클릭베이트 비율 = π·m / (π·m + (1−π)·pn)."""
    return pi * m / (pi * m + (1 - pi) * pn)

def crit(n, k0, delta=DELTA):
    """H0: 비율 ≤ k0 를 단측 정확 이항검정(수준 ≤ δ). X > k_crit 이면 경보."""
    n = np.asarray(n)
    kc = binom.isf(delta, n, k0).astype(int)
    kc = np.where(binom.sf(kc, n, k0) > delta, kc + 1, kc)      # 수준 ≤ δ 보정
    return kc

def power(n, k0, k1):
    kc = crit(n, k0)
    return binom.sf(kc, n, k1), kc

def min_n(k0, k1, target=0.8):
    ns = np.arange(1, N_MAX + 1)
    p, _ = power(ns, k0, k1)
    bad = np.where(p < target)[0]
    if len(bad) == 0: return 1
    last = bad[-1]
    return int(ns[last] + 1) if last + 1 < N_MAX else None     # None: N_MAX 안에서 도달 못 함

def load(outputs, tag, scores):
    d = pd.read_csv(os.path.join(outputs, f"test_decisions_{tag}.csv"))
    print("test_decisions 컬럼:", list(d.columns))
    col = next((c for c in ("p_cb", "score", "p") if c in d.columns), None)
    if col is None:
        cands = [scores] if scores else sorted(glob.glob(os.path.join(outputs, "scores_*.csv")))
        for f in cands:
            s = pd.read_csv(f)
            print("점수 파일 후보:", os.path.basename(f), list(s.columns))
            if "key" in s.columns and "p_cb" in s.columns:
                m = d.assign(_k=d.news_id.astype(str)).merge(
                    s[["key", "p_cb"]].assign(_k=s.key.astype(str)).drop(columns="key"), on="_k", how="left")
                rate = m.p_cb.notna().mean()
                print(f"  news_id↔key 일치율 {rate:.3f}")
                if rate > 0.99:
                    d, col = m.drop(columns="_k"), "p_cb"; break
    if col is None:
        print("점수 열을 찾지 못해 부분 2는 건너뜁니다. scores= 로 점수 파일을 지정하거나 위 컬럼 목록을 알려 주세요.")
    elif col != "p_cb":
        d = d.rename(columns={col: "p_cb"})
    return d, (col is not None)

def part1(d, out, tag):
    ns = d[~d.sens]
    nn, nc = ns[ns.cls == "정상"], ns[ns.cls == "클릭베이트"]
    pn_k, pn_n = int((nn.set == "normal").sum()), len(nn)
    ms_k, ms_n = int((nc.set == "normal").sum()), len(nc)
    pn = pn_k / pn_n; ms = ms_k / ms_n; ms_lo, ms_hi = cp(ms_k, ms_n)
    print(f"\n[입력 비율, 비민감] 정상->자동 통과 pn={pn:.4f} ({pn_k}/{pn_n}) | 클릭베이트->자동 통과 miss={ms:.4f} ({ms_k}/{ms_n}) [{ms_lo:.4f}, {ms_hi:.4f}]")
    alts = [(f"{m:g}α", m * ALPHA) for m in MULT] + [("관측 miss", ms), ("관측 miss 하한", ms_lo), ("관측 miss 상한", ms_hi)]
    rows, mins = [], []
    for pi in PI_GRID:
        for defn in ("B", "A"):
            # B: 보장 대상인 miss ≤ α 를 검정 (κ0 = κ(α)), A: 기획본 정의, 자동 통과 안의 클릭베이트 비율 ≤ α
            k0 = kappa(ALPHA, pi, pn) if defn == "B" else ALPHA
            for lab, m in alts:
                k1 = kappa(m, pi, pn)
                for n in N_SHOW:
                    p, kc = power(n, k0, k1)
                    rows.append(dict(definition=defn, pi=pi, alt=lab, miss_true=round(m, 4), kappa_null=round(k0, 5), kappa_true=round(k1, 5),
                                     n=n, alarm_if_k_gt=int(kc), expected_cb_in_sample=round(n * k1, 2), alarm_prob=round(float(p), 4)))
                mins.append(dict(definition=defn, pi=pi, alt=lab, miss_true=round(m, 4), kappa_null=round(k0, 5), kappa_true=round(k1, 5),
                                 n_at_power_0_8_reference=(min_n(k0, k1) if k1 > k0 else None)))
    P = pd.DataFrame(rows); M = pd.DataFrame(mins)
    P.to_csv(os.path.join(out, f"d26_power_{tag}.csv"), index=False, encoding="utf-8-sig")
    M.to_csv(os.path.join(out, f"d26_min_n_{tag}.csv"), index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 230); pd.set_option("display.max_columns", 30)
    for pi in (0.05, 0.10, 0.25):
        print(f"\n[정의 B: miss ≤ α 검정] π={pi}  귀무 κ0={kappa(ALPHA, pi, pn):.5f} (표본 {PLAN_N}건이면 클릭베이트 기대 {PLAN_N * kappa(ALPHA, pi, pn):.2f}건)")
        t = P[(P.definition == "B") & (P.pi == pi) & (P.n.isin([203, 500, 1000, 3000, 10000, 50000]))]
        print(t.pivot(index="alt", columns="n", values="alarm_prob").reindex([a for a, _ in alts]).to_string())
        print(M[(M.definition == "B") & (M.pi == pi)][["alt", "miss_true", "kappa_true", "n_at_power_0_8_reference"]].to_string(index=False))
    print(f"\n[정의 A: 자동 통과 안의 클릭베이트 비율 ≤ α, 기획본] n={PLAN_N}에서 실제 κ(관측 miss)별 기대 클릭베이트 건수:")
    for pi in (0.05, 0.10, 0.25):
        print(f"  π={pi}: κ={kappa(ms, pi, pn):.5f} → 기대 {PLAN_N * kappa(ms, pi, pn):.2f}건 (경보 기준 {int(crit(PLAN_N, ALPHA))}건 초과)")
    return pn, ms

def part2(d, out, tag):
    rng = np.random.default_rng(SEED)
    ns = d[~d.sens]
    N = {c: ns[ns.cls == c] for c in ("정상", "클릭베이트")}
    Ntot = {c: len(v) for c, v in N.items()}
    def sc(df): return df[df.set == "normal"].p_cb.to_numpy()
    def curve(sn, sc_, nn_tot, nc_tot, pis):
        s = np.concatenate([sn, sc_]); isc = np.concatenate([np.zeros(len(sn), bool), np.ones(len(sc_), bool)])
        o = np.argsort(-s, kind="stable"); isc = isc[o]
        res = {}
        for pi in pis:
            w = np.where(isc, pi / nc_tot, (1 - pi) / nn_tot)
            cw = np.cumsum(w); x = cw / cw[-1]
            ycb = np.cumsum(np.where(isc, w, 0)); y = ycb / ycb[-1] if ycb[-1] > 0 else np.full_like(x, np.nan)
            res[pi] = np.interp(X_GRID, np.r_[0, x], np.r_[0, y])
        return res
    def auc(sn, sc_):
        r = rankdata(np.concatenate([sc_, sn])); n1, n0 = len(sc_), len(sn)
        return (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    pis = (0.05, 0.10, 0.25)
    base = curve(sc(N["정상"]), sc(N["클릭베이트"]), Ntot["정상"], Ntot["클릭베이트"], pis)
    a0 = auc(sc(N["정상"]), sc(N["클릭베이트"]))
    boots = {pi: [] for pi in pis}; aucs = []
    for _ in range(B_BOOT):
        bn = N["정상"].iloc[rng.integers(0, Ntot["정상"], Ntot["정상"])]
        bc = N["클릭베이트"].iloc[rng.integers(0, Ntot["클릭베이트"], Ntot["클릭베이트"])]
        sn_, sc__ = sc(bn), sc(bc)
        if len(sc__) == 0: continue
        r = curve(sn_, sc__, Ntot["정상"], Ntot["클릭베이트"], pis)
        for pi in pis: boots[pi].append(r[pi])
        aucs.append(auc(sn_, sc__))
    rows = []
    for pi in pis:
        arr = np.array(boots[pi]); lo, hi = np.nanpercentile(arr, [2.5, 97.5], axis=0)
        for i, x in enumerate(X_GRID):
            rows.append(dict(pi=pi, x_share_of_autopass=x, y_share_of_missed_cb=round(base[pi][i], 4), y_lo=round(lo[i], 4), y_hi=round(hi[i], 4),
                             lift=round(base[pi][i] / x, 3), lift_lo=round(lo[i] / x, 3), lift_hi=round(hi[i] / x, 3)))
    L = pd.DataFrame(rows); L.to_csv(os.path.join(out, f"d26_lift_{tag}.csv"), index=False, encoding="utf-8-sig")
    alo, ahi = np.percentile(aucs, [2.5, 97.5])
    n_cb, n_n = len(sc(N["클릭베이트"])), len(sc(N["정상"]))
    A = pd.DataFrame([dict(auto_pass_clickbait=n_cb, auto_pass_normal=n_n, auc=round(a0, 4), auc_lo=round(alo, 4), auc_hi=round(ahi, 4), boot_B=len(aucs))])
    A.to_csv(os.path.join(out, f"d26_auc_{tag}.csv"), index=False, encoding="utf-8-sig")
    print(f"\n[부분 2] 비민감 자동 통과 중 놓친 클릭베이트 {n_cb}건, 정상 {n_n}건. 줄 세우기 AUC={a0:.3f} [{alo:.3f}, {ahi:.3f}] (0.5면 점수 순서가 놓침을 못 찾는다)")
    for pi in pis:
        print(f"\n  π={pi}: 자동 통과의 상위 x를 감사했을 때 잡히는 놓침 비율 y (랜덤이면 y=x)")
        print(L[(L.pi == pi) & (L.x_share_of_autopass.isin([0.01, 0.05, 0.1, 0.2, 0.5, 1.0]))][["x_share_of_autopass", "y_share_of_missed_cb", "y_lo", "y_hi", "lift", "lift_lo", "lift_hi"]].to_string(index=False))

def main(outputs, tag, out, scores):
    d, has_score = load(outputs, tag, scores)
    d["sens"] = d.category.isin(SENSITIVE)
    d["cls"] = np.where(d.label == 1, "클릭베이트", "정상")
    os.makedirs(out, exist_ok=True)
    print(f"tag={tag} test {len(d)}건. α={ALPHA}, δ={DELTA} (common.POLICY). 격자(π, 배수, n, x)는 표시용이며 선택한 값이 아니다.")
    part1(d, out, tag)
    if has_score: part2(d, out, tag)
    print("\n주의: 검토자(라벨러)가 정확하다는 가정(h=0)에서의 값이다. 80% 검정력 n은 참조용이며 결정 기준이 아니다. 신뢰구간은 bootstrap 백분위 구간이다.")

if __name__ == "__main__":
    if len(sys.argv) < 2: sys.exit(__doc__)
    a = sys.argv[1:]
    sc_arg = next((x.split("=", 1)[1] for x in a if x.startswith("scores=")), None)
    pos = [x for x in a if not x.startswith("scores=")]
    main(pos[0], pos[1] if len(pos) > 1 else "axft_maxq", pos[2] if len(pos) > 2 else "d26_out", sc_arg)
