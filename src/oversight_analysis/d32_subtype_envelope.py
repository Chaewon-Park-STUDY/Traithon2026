# d32 (수정판) 클릭베이트 유형 구성 변화에 따른 놓침의 범위. 출력은 집계뿐. 규칙: PREREG_d32_subtype_envelope.md
# 입력: outputs/test_decisions_<tag>.csv (기존 판정을 읽기만 함), outputs/clean_main.parquet (news_id, pattern 컬럼만 읽음)
# 사용: python d32_subtype_envelope.py <outputs 폴더> [tag=axft_maxq] [출력폴더=d32_out]
import os, sys
import numpy as np, pandas as pd, pyarrow.parquet as pq
from scipy.stats import beta, binom, norm, fisher_exact
from scipy.special import betaln
from scipy.optimize import minimize
SENSITIVE = {"정치", "경제"}; ALPHA, DELTA = 0.05, 0.05
PARTITIONS = {"P1 pattern×part": ["pattern", "part"], "P2 category": ["category"], "P3 level": ["level"]}
R_GRID = [1.0, 1.5, 2.0, 3.0]; PI_GRID = [0.05, 0.10, 0.25]; N_UNL = [1000, 10000, 100000, 1000000]; N_AUD = [203, 500, 1000, 3000]

def cp(k, n, a=0.05):
    if n == 0: return (float("nan"), float("nan"))
    return (0.0 if k == 0 else float(beta.ppf(a / 2, k, n - k + 1)), 1.0 if k == n else float(beta.ppf(1 - a / 2, k + 1, n - k)))
def kappa(pi, pn, m): return pi * m / (pi * m + (1 - pi) * pn)
def umvue_var(k, n): return k * (n - k) / (n * n * (n - 1)) if n > 1 else float("nan")   # Var(k/n)의 불편추정량
def eb_fit(k, n):
    def nll(p):
        a, b = np.exp(p); return -float(np.sum(betaln(k + a, n - k + b) - betaln(a, b)))
    r = minimize(nll, [0.0, 3.0], method="Nelder-Mead"); a, b = np.exp(r.x); return float(a), float(b)
def homog_perm(n, k, rng, B=10000):
    """전역 균질성 검정 H0: 모든 층의 놓침 비율이 같다. Pearson 통계량의 permutation p값"""
    n = n.astype(int); k = k.astype(int); ids = np.repeat(np.arange(len(n)), n)
    y = np.concatenate([np.r_[np.ones(kk), np.zeros(nn - kk)] for nn, kk in zip(n, k)]); pb = y.mean()
    if pb in (0.0, 1.0): return 0.0, 1.0
    st = lambda yy: float(np.sum((np.bincount(ids, weights=yy, minlength=len(n)) - n * pb) ** 2 / (n * pb * (1 - pb))))
    obs = st(y); c = sum(st(rng.permutation(y)) >= obs for _ in range(B)); return obs, (c + 1) / (B + 1)
def sup_mix(vals, w0, r):
    """w_s <= r*w0_s, sum w = 1 아래 sum w_s*vals_s의 최댓값(큰 값부터 질량 배분)"""
    w = np.zeros(len(vals)); rem = 1.0
    for i in np.argsort(-vals):
        t = min(r * w0[i], rem); w[i] = t; rem -= t
        if rem <= 1e-12: break
    return float(w @ vals), w

N_REP = 500; SEED_V = 20261111
def coverage_check(cb, a_b, out, tag):
    """pseudo-population 점검: 전체 클릭베이트 test를 모집단으로 보고, 절반 표본 A로 만든 상한이 모집단 값을 덮는지 N_REP회 반복해 센다.
    A는 모집단의 절반이라(비복원) 실제 coverage는 명목보다 높게(보수적으로) 나오는 것이 정상이다. 명목(95%) 아래면 상한이 틀린 것이다."""
    rng = np.random.default_rng(SEED_V); rows = []; m_all = float(cb.miss.mean()); n_all = len(cb)
    for p, keys in PARTITIONS.items():
        pop = cb.groupby(keys).miss.agg(["size", "sum"]).reset_index(); pop["m"] = pop["sum"] / pop["size"]
        lab = cb[keys].astype(str).agg("×".join, axis=1).values; idx_of = {l: i for i, l in enumerate(pop[keys].astype(str).agg("×".join, axis=1))}
        li = np.array([idx_of[l] for l in lab]); y = cb.miss.values; mp = pop.m.values; S_ = len(pop)
        hit = {r: 0 for r in R_GRID}; hit_all = 0; hit_h = 0; tot = 0
        for _ in range(N_REP):
            A = rng.choice(n_all, n_all // 2, replace=False); nA = np.bincount(li[A], minlength=S_).astype(float); kA = np.bincount(li[A], weights=y[A], minlength=S_)
            ok = nA > 0
            if not ok.all(): continue
            tot += 1; w0 = nA / nA.sum(); ub = np.array([cp(int(a), int(b), a_b)[1] for a, b in zip(kA, nA)])
            hit_all += int(np.all(ub >= mp))
            for r in R_GRID:
                U = cp(int(kA.sum()), int(nA.sum()))[1] if r == 1.0 else sup_mix(ub, w0, r)[0]
                truth = m_all if r == 1.0 else sup_mix(mp, w0, r)[0]; hit[r] += int(U >= truth)
            wu = np.full(S_, 1 / S_); V = float(np.sum(wu ** 2 / nA)); h = float(np.sqrt(np.log(2 / DELTA) * V / 2))
            hit_h += int(abs(float(wu @ (kA / nA)) - float(wu @ mp)) <= h)
        if tot == 0: continue
        for r in R_GRID: rows.append(dict(partition=p, check=f"상한 U_env(r={r}) >= 모집단 값", coverage=round(hit[r] / tot, 4), reps=tot))
        rows.append(dict(partition=p, check="모든 층 Bonferroni 상한이 동시에 모집단 층 값을 덮음", coverage=round(hit_all / tot, 4), reps=tot))
        rows.append(dict(partition=p, check="중립(균등) UMVUE ± Hoeffding 반폭이 모집단 값을 덮음", coverage=round(hit_h / tot, 4), reps=tot))
    V_ = pd.DataFrame(rows); V_.to_csv(os.path.join(out, f"d32_coverage_{tag}.csv"), index=False, encoding="utf-8-sig")
    print("\n[실데이터 pseudo-population coverage] 명목 0.95 이상이어야 한다. 절반 표본이라 보수적으로 높게 나오는 것이 정상. 그 아래면 해당 상한은 쓰지 않는다."); print(V_.to_string(index=False))
def observed_shift(outputs, cb, tag, out):
    """보정 표본(calib, calib_auto_check)과 test 사이에 실제로 관측된 클릭베이트 유형 구성비 변화: r_obs = max_s w_test_s / w_calib_s"""
    f = os.path.join(outputs, "split_manifest.csv")
    if not os.path.exists(f): print("\n[관측된 r] split_manifest.csv 없음 -> 건너뜀"); return
    m = pd.read_csv(f)
    need = {"news_id", "subset", "label", "category"}
    if not need <= set(m.columns): print(f"\n[관측된 r] split_manifest 컬럼 부족({need - set(m.columns)}) -> 건너뜀"); return
    pat = pq.read_table(os.path.join(outputs, "clean_main.parquet"), columns=["news_id", "pattern"]).to_pandas(); pat["news_id"] = pat.news_id.astype(str); m["news_id"] = m.news_id.astype(str)
    m = m.merge(pat, on="news_id", how="left"); m = m[(m.label == 1) & ~m.category.isin(SENSITIVE)].copy(); m["pattern"] = m.pattern.astype(str)
    print("\n[관측된 r] split_manifest subset 값:", dict(m.subset.value_counts())); rows = []
    cal = m[m.subset.astype(str).str.startswith("calib")]; tst = cb.copy(); tst["pattern"] = tst.pattern.astype(str)
    if len(cal) == 0: print("calib 행 없음 -> 건너뜀"); return
    for p, keys in (("P1 pattern×part", ["pattern"] + (["part"] if "part" in m.columns else [])), ("P2 category", ["category"])):
        wt = tst.groupby(keys).size() / len(tst); wc = cal.groupby(keys).size() / len(cal); jj = pd.concat([wt.rename("w_test"), wc.rename("w_calib")], axis=1).fillna(0.0)
        ratio = (jj.w_test / jj.w_calib.replace(0, np.nan)); rows.append(dict(partition=p, n_calib=len(cal), n_test=len(tst), r_obs_max=round(float(ratio.max()), 3) if ratio.notna().any() else float("nan"),
            r_obs_min=round(float(ratio.min()), 3) if ratio.notna().any() else float("nan"), test에만_있는_층=int(((jj.w_calib == 0) & (jj.w_test > 0)).sum()), total_variation=round(float(0.5 * (jj.w_test - jj.w_calib).abs().sum()), 4)))
    R_ = pd.DataFrame(rows); R_.to_csv(os.path.join(out, f"d32_observed_shift_{tag}.csv"), index=False, encoding="utf-8-sig"); print(R_.to_string(index=False))
    print("r 격자(1, 1.5, 2, 3)를 이 실제 관측값과 비교해 읽는다. calib_auto_check는 Auto만 포함하므로 calib 구성에 Direct가 적게 들어 있을 수 있다(보정 구성 자체가 이미 편향).")

def main(outputs, tag="axft_maxq", out="d32_out"):
    dec = pd.read_csv(os.path.join(outputs, f"test_decisions_{tag}.csv"))
    if "subset" in dec.columns: dec = dec[dec.subset == "test"].copy()
    dec["news_id"] = dec.news_id.astype(str)
    pat = pq.read_table(os.path.join(outputs, "clean_main.parquet"), columns=["news_id", "pattern"]).to_pandas(); pat["news_id"] = pat.news_id.astype(str)
    d = dec.merge(pat, on="news_id", how="left"); d = d[~d.category.isin(SENSITIVE)].copy()
    for c in ("pattern", "level", "part", "process_type", "category"): d[c] = d[c].astype(str)
    d["miss"] = (d.set == "normal").astype(int); d["amb"] = (d.set == "ambiguous").astype(int); d["conf"] = (d.set == "cb").astype(int)
    cb, nm = d[d.label == 1], d[d.label == 0]; os.makedirs(out, exist_ok=True)
    groups = {p: cb.groupby(keys) for p, keys in PARTITIONS.items()}; S_tot = sum(g.ngroups for g in groups.values()); a_b = 0.05 / S_tot
    m0, k0 = float(cb.miss.mean()), int(cb.miss.sum()); lo0, hi0 = cp(k0, len(cb)); pn = float((nm.set == "normal").mean())
    print(f"tag={tag}. test 비민감 {len(d)}건 (정상 {len(nm)}, 클릭베이트 {len(cb)}). 분할 3개의 층 수 합 S_total={S_tot} -> Bonferroni 수준 {a_b:.5f}")
    print(f"[전체] 클릭베이트 놓침 {m0:.4f} ({k0}/{len(cb)}) [{lo0:.4f}, {hi0:.4f}], α={ALPHA}. 정상 -> 자동 통과 pn={pn:.4f}")
    strata, env = [], []; scen = {}; z = norm.ppf(1 - DELTA / 2); rng = np.random.default_rng(20261110)
    for p, g in groups.items():
        T = g.agg(n=("miss", "size"), k=("miss", "sum"), ka=("amb", "sum"), kc=("conf", "sum")).reset_index()
        T["slice"] = T[PARTITIONS[p]].astype(str).agg("×".join, axis=1); n, k = T.n.values.astype(float), T.k.values.astype(float)
        w0 = n / n.sum(); a, b = eb_fit(k, n); T["umvue"] = k / n; T["se_unb"] = np.sqrt([umvue_var(x, y) for x, y in zip(k, n)])
        T["cp_lo"], T["cp_hi"] = zip(*[cp(int(x), int(y)) for x, y in zip(k, n)]); T["bonf_hi"] = [cp(int(x), int(y), a_b)[1] for x, y in zip(k, n)]
        T["eb_mean"] = (a + k) / (a + b + n); rest_k, rest_n = k.sum() - k, n.sum() - n
        T["fisher_p"] = [fisher_exact([[x, y - x], [rk, rn - rk]])[1] for x, y, rk, rn in zip(k, n, rest_k, rest_n)]
        T.insert(0, "partition", p); strata.append(T)
        chi, ph = homog_perm(n, k, rng); ab = a + b
        print(f"\n===== {p}: 층 {len(T)}개. 전역 균질성 검정(모든 층의 놓침이 같다) Pearson={chi:.2f}, permutation p={ph:.4f} =====")
        print("EB 수축: " + (f"a+b={ab:.1f}" if ab < 1e4 else "a+b>1e4 -> 층 간 차이가 이항 변동 이상으로 보이지 않아 사실상 완전 풀링"))
        print(T.sort_values("umvue", ascending=False)[["slice", "n", "k", "umvue", "se_unb", "cp_lo", "cp_hi", "bonf_hi", "eb_mean", "fisher_p"]].round(4).to_string(index=False))
        # 중립 시나리오: 층 균등 가중치(사전 고정)
        wu = np.full(len(T), 1 / len(T)); est = float(wu @ (k / n)); var = float(np.nansum(wu ** 2 * np.array([umvue_var(x, y) for x, y in zip(k, n)]))); V = float(np.sum(wu ** 2 / n))
        hoef = float(np.sqrt(np.log(2 / DELTA) * V / 2)); scen[p] = dict(equal_umvue=est, equal_se=float(np.sqrt(var)), hoeffding=hoef, n_eff=1 / V, upper=est + hoef)
        print(f"[중립: 층 균등 가중치] UMVUE {est:.4f}, 불편 SE {np.sqrt(var):.4f}, Hoeffding 반폭 {hoef:.4f}, n_eff {1 / V:.0f}, 상한(UMVUE+반폭, 고정 w라 층 Bonferroni 불필요) {est + hoef:.4f} = {(est + hoef) / ALPHA:.2f}α")
        pool_hi = cp(int(k.sum()), int(n.sum()))[1]
        for r in R_GRID:
            plug, _ = sup_mix(k / n, w0, r); ebm, _ = sup_mix(T.eb_mean.values, w0, r); ub, _ = sup_mix(T.bonf_hi.values, w0, r)
            if r == 1.0: ub = pool_hi      # r=1이면 w=w0 하나뿐이므로 풀링 Clopper-Pearson 상한이 유효하고 더 좁다
            env.append(dict(partition=p, r=r, plugin_umvue_ref=round(plug, 4), eb=round(ebm, 4), upper_valid=round(ub, 4), upper_over_alpha=round(ub / ALPHA, 2)))
        env.append(dict(partition=p, r="꼭짓점(참고)", plugin_umvue_ref=round(float((k / n).max()), 4), eb=round(float(T.eb_mean.max()), 4), upper_valid=round(float(T.bonf_hi.max()), 4), upper_over_alpha=round(float(T.bonf_hi.max()) / ALPHA, 2)))
    S = pd.concat(strata, ignore_index=True); S.to_csv(os.path.join(out, f"d32_strata_{tag}.csv"), index=False, encoding="utf-8-sig")
    E = pd.DataFrame(env); E.to_csv(os.path.join(out, f"d32_envelope_{tag}.csv"), index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print("\n[범위: 가중치 제한 r 아래 클릭베이트 놓침] plugin_umvue_ref=선택 편향이 있는 참고값, eb=수축 추정, upper_valid=Bonferroni 동시 신뢰 상한(주 결과)"); print(E.to_string(index=False))
    print("\n읽는 법: upper_valid는 r>1에서 층별 Bonferroni 상한의 가중합이라 작은 층이 많을수록 느슨하다(r=1은 풀링 구간). 느슨함 자체가 '지금 test 크기로는 유형 구성 변화에 대한 보장이 약하다'는 결과다.")
    coverage_check(cb, a_b, out, tag); observed_shift(outputs, cb, tag, out)
    sig = S[S.fisher_p < a_b]; print(f"\nBonferroni 기준(p<{a_b:.5f})을 넘는 층: {len(sig)}개" + (": " + ", ".join(sig.partition + ":" + sig.slice) if len(sig) else ""))
    print("\n[정상 slice (설명용)]"); rows = []
    for fam in ("category", "part"):
        for key, g in nm.groupby(fam):
            for nme, col in (("확정 후보(오탐 후보)", "conf"), ("애매 이관", "amb"), ("자동 통과", "miss")):
                kk, nn = int(g[col].sum()), len(g); lo, hi = cp(kk, nn); rows.append(dict(family=fam, slice=key, metric=nme, n=nn, k=kk, rate=round(kk / nn, 4), lo=round(lo, 4), hi=round(hi, 4)))
    N = pd.DataFrame(rows); N.to_csv(os.path.join(out, f"d32_normal_{tag}.csv"), index=False, encoding="utf-8-sig"); print(N.to_string(index=False))
    # 부분 3: 탐지 (주 분할 P1에서 시나리오를 사전 고정 규칙으로 정함)
    p1 = "P1 pattern×part"; eg = E[(E.partition == p1)]; sc = [("중립(층 균등) UMVUE", scen[p1]["equal_umvue"]), ("범위 상한 r=2", float(eg[eg.r == 2.0].upper_valid.iloc[0])), ("범위 상한 r=3", float(eg[eg.r == 3.0].upper_valid.iloc[0]))]
    rows = []; zz = norm.ppf(1 - DELTA)
    for name, m1 in sc:
        for pi in PI_GRID:
            a0, a1 = (1 - pi) * pn + pi * m0, (1 - pi) * pn + pi * m1
            for N_ in N_UNL:
                pw = float(norm.cdf((a1 - a0 - zz * np.sqrt(a0 * (1 - a0) / N_)) / np.sqrt(a1 * (1 - a1) / N_))) if a1 > a0 else float("nan")
                rows.append(dict(scenario=name, m1=round(m1, 4), pi=pi, method="라벨 없음: 자동 통과 비율(트래픽 N)", size=N_, power=round(pw, 4)))
            k0_, k1_ = kappa(pi, pn, m0), kappa(pi, pn, m1)
            for B in N_AUD:
                c = int(binom.ppf(1 - DELTA, B, k0_)); rows.append(dict(scenario=name, m1=round(m1, 4), pi=pi, method="라벨 감사: 자동 통과 표본 B건", size=B, power=round(float(binom.sf(c, B, k1_)), 4)))
    D = pd.DataFrame(rows); D.to_csv(os.path.join(out, f"d32_detect_{tag}.csv"), index=False, encoding="utf-8-sig")
    for name, _ in sc:
        for pi in PI_GRID:
            s = D[(D.scenario == name) & np.isclose(D.pi, pi)]; print(f"\n[탐지 검정력] 시나리오={name}(m1={s.m1.iloc[0]}), π={pi}, 기준 m0={m0:.4f}"); print(s.pivot(index="method", columns="size", values="power").to_string(na_rep="-"))
    print("\n주의: 구축 속성은 운영에서 관찰되지 않으므로 유형 구성이 바뀌는 시나리오로만 읽는다. 탐색적이며 Bonferroni는 겹치는 분할 전체에 적용해 보수적이다. 보정이 주어진 조건부 분석이며 보정 표본 변동은 포함하지 않았다. π, pn, m0를 안다고 가정한 탐지 계산. 검토자 정확(h=0).")
if __name__ == "__main__":
    if len(sys.argv) < 2: sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "axft_maxq", sys.argv[3] if len(sys.argv) > 3 else "d32_out")
