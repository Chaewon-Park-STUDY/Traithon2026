# d34 category 조건부(Mondrian) 보정: 유형별 놓침을 α 근처로 맞추는 대가(검토량)는 얼마인가. 출력은 집계뿐. 규칙: PREREG_d34_mondrian_category.md
# 입력: outputs/split_manifest.csv, outputs/scores_<name>.csv (news_id, p_cb). 기존 파일을 덮어쓰지 않는다(읽기만 함).
# 사용: python d34_mondrian_category.py <outputs 폴더> [name=axft] [출력폴더=d34_out]   (d28_alpha_sweep.py와 같은 폴더에서 실행)
import os, sys
import numpy as np, pandas as pd
from d28_alpha_sweep import qhat, route, cp, SENSITIVE
ALPHAS = [0.05, 0.10]; PI_GRID = [0.05, 0.10, 0.25]; B_BOOT = 500; SEED = 20261112
# 시나리오: (이름, 키 열, 정상 임계값도 키별로 쓰는가)
SCEN = [("S0 전역(현행)", None, False), ("S1 category별(클릭베이트·정상)", "category", True), ("S2 category별(클릭베이트만)", "category", False), ("R1 part별(참고: 운영에서 관찰 불가)", "part", True)]

def thr_rows(cal, auto, alpha, key, split_normal, test):
    """test 각 행에 대한 (q_cb, q_ok). 키가 없으면 전역값."""
    def one(c, a, nm):
        q_dir, _ = qhat(1 - c[c.label == 1].p_cb.values, alpha); q_auto, _ = qhat(1 - a.p_cb.values, alpha); q_ok, _ = qhat(nm[nm.label == 0].p_cb.values, alpha)
        return max(q_dir, q_auto), q_ok
    g_cb, g_ok = one(cal, auto, cal); qcb = np.full(len(test), g_cb); qok = np.full(len(test), g_ok); info = {}
    if key is None: return qcb, qok, info
    for g in test[key].unique():
        c, a = cal[cal[key] == g], auto[auto[key] == g]; cb_g, ok_g = one(c, a, c)
        m = (test[key] == g).values; qcb[m] = cb_g
        if split_normal: qok[m] = ok_g
        info[g] = dict(n_cal_cb=int((c.label == 1).sum()), n_auto=len(a), n_cal_normal=int((c.label == 0).sum()), q_cb=round(cb_g, 4), q_ok=round(qok[m][0], 4))
    return qcb, qok, info

def evaluate(test, qcb, qok):
    s = route(test.p_cb.values, qcb, qok); t = test.assign(s=s); nm, cb = t[t.label == 0], t[t.label == 1]
    r = dict(fc=(nm.s == "cb").mean(), pn=(nm.s == "normal").mean(), amb_n=(nm.s == "ambiguous").mean(), tp=(cb.s == "cb").mean(), amb_c=(cb.s == "ambiguous").mean(), miss=(cb.s == "normal").mean(), k=int((cb.s == "normal").sum()), n=len(cb))
    return r, t

def main(outputs, name="axft", out="d34_out"):
    man = pd.read_csv(os.path.join(outputs, "split_manifest.csv")); sc = pd.read_csv(os.path.join(outputs, f"scores_{name}.csv")); d = man.merge(sc, on="news_id"); os.makedirs(out, exist_ok=True)
    if "part" not in d.columns: print("part 컬럼이 없어 R1(part별)은 건너뜀"); scen = [s for s in SCEN if s[1] != "part"]
    else: scen = SCEN
    cal0, auto0 = d[(d.subset == "calib") & ~d.category.isin(SENSITIVE)], d[(d.subset == "calib_auto_check") & ~d.category.isin(SENSITIVE)]
    glob_cal, glob_auto = d[d.subset == "calib"], d[d.subset == "calib_auto_check"]     # 현행(S0)은 정치·경제 포함 전체 보정 표본을 그대로 쓴다(재현)
    test = d[(d.subset == "test") & ~d.category.isin(SENSITIVE)].copy(); rng = np.random.default_rng(SEED)
    print(f"name={name}. 보정(비민감) 클릭베이트 {int((cal0.label==1).sum())}건·정상 {int((cal0.label==0).sum())}건, auto_check {len(auto0)}건, test 비민감 {len(test)}건. S0는 현행과 같게 정치·경제를 포함한 전체 보정 표본을 쓰고 S1·S2·R1은 비민감 보정 표본만 쓴다.")
    rows, crow, irow = [], [], []
    for a in ALPHAS:
        for nm_, key, split_n in scen:
            cal_, auto_ = (glob_cal, glob_auto) if key is None else (cal0, auto0)
            qcb, qok, info = thr_rows(cal_, auto_, a, key, split_n, test); r, t = evaluate(test, qcb, qok); lo, hi = cp(r["k"], r["n"], 0.05)
            for pi in PI_GRID:
                rv = 1000 * ((1 - pi) * (r["fc"] + r["amb_n"]) + pi * (r["tp"] + r["amb_c"]))
                rows.append(dict(alpha=a, scenario=nm_, pi=pi, miss=round(r["miss"], 4), miss_lo=round(lo, 4), miss_hi=round(hi, 4), fc=round(r["fc"], 4), amb_normal=round(r["amb_n"], 4), tp=round(r["tp"], 4), amb_cb=round(r["amb_c"], 4), review_per1000=round(rv, 1), unreachable_per1000=round(1000 * pi * r["miss"], 2)))
            for g, gg in t[t.label == 1].groupby("category"):
                k, n = int((gg.s == "normal").sum()), len(gg); l2, h2 = cp(k, n, 0.05); crow.append(dict(alpha=a, scenario=nm_, category=g, n_test_cb=n, k_miss=k, miss=round(k / n, 4), lo=round(l2, 4), hi=round(h2, 4), exceeds_alpha=bool(l2 > a)))
            for g, v in info.items(): irow.append(dict(alpha=a, scenario=nm_, group=g, **v))
    R, C, I = pd.DataFrame(rows), pd.DataFrame(crow), pd.DataFrame(irow)
    R.to_csv(os.path.join(out, f"d34_overall_{name}.csv"), index=False, encoding="utf-8-sig"); C.to_csv(os.path.join(out, f"d34_by_category_{name}.csv"), index=False, encoding="utf-8-sig"); I.to_csv(os.path.join(out, f"d34_thresholds_{name}.csv"), index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    for a in ALPHAS:
        print(f"\n===== α={a}: 전체(비민감 test) ====="); print(R[(R.alpha == a) & (R.pi == PI_GRID[0])][["scenario", "miss", "miss_lo", "miss_hi", "fc", "amb_normal", "tp", "amb_cb"]].to_string(index=False))
        for pi in PI_GRID: print(f"[π={pi}] 비민감 1,000건당 검토 건수"); print(R[(R.alpha == a) & (R.pi == pi)][["scenario", "review_per1000", "unreachable_per1000"]].to_string(index=False))
        print(f"[category별 클릭베이트 놓침, α={a}] exceeds_alpha=95% 하한이 α 초과"); print(C[C.alpha == a].pivot(index="category", columns="scenario", values="miss").to_string())
        print(C[(C.alpha == a) & C.exceeds_alpha][["scenario", "category", "n_test_cb", "k_miss", "miss", "lo", "hi"]].to_string(index=False) if C[(C.alpha == a) & C.exceeds_alpha].shape[0] else "하한이 α를 넘는 category 없음")
    print("\n[category별 임계값과 보정 표본 수 (α=0.05, S1)]"); print(I[(I.alpha == 0.05) & (I.scenario.str.startswith("S1"))].to_string(index=False))
    # 보정 표본 변동: 보정 표본을 (라벨×키) 칸 안에서 복원추출해 임계값을 다시 만들고 test에 적용
    a = 0.05; brow = []
    for nm_, key, split_n in scen:
        res = {g: [] for g in test.category.unique()}; allm = []; cal_, auto_ = (glob_cal, glob_auto) if key is None else (cal0, auto0)
        cells_c = [grp.index for _, grp in cal_.groupby(["label"] + ([key] if key else []))]; cells_a = [grp.index for _, grp in auto_.groupby(key)] if key else [auto_.index]
        for _ in range(B_BOOT):
            ci = np.concatenate([rng.choice(ix, len(ix)) for ix in cells_c]); ai = np.concatenate([rng.choice(ix, len(ix)) for ix in cells_a])
            qcb, qok, _ = thr_rows(cal_.loc[ci].reset_index(drop=True), auto_.loc[ai].reset_index(drop=True), a, key, split_n, test); r, t = evaluate(test, qcb, qok); allm.append(r["miss"])
            for g, gg in t[t.label == 1].groupby("category"): res[g].append((gg.s == "normal").mean())
        for g, v in list(res.items()) + [("전체", allm)]:
            v = np.array(v); brow.append(dict(scenario=nm_, category=g, miss_mean=round(float(v.mean()), 4), p05=round(float(np.quantile(v, .05)), 4), p95=round(float(np.quantile(v, .95)), 4), frac_le_alpha=round(float((v <= a).mean()), 3)))
    Bd = pd.DataFrame(brow); Bd.to_csv(os.path.join(out, f"d34_boot_{name}.csv"), index=False, encoding="utf-8-sig")
    print(f"\n[보정 표본 변동: 칸 안 복원추출 {B_BOOT}회, α=0.05] frac_le_alpha = 놓침이 α 이하였던 비율"); print(Bd.pivot(index="category", columns="scenario", values="frac_le_alpha").to_string())
    print(Bd[Bd.category == "전체"].to_string(index=False))
    print("\n주의: 검토자 정확(h=0). 검토량은 민감 기사 제외. category는 운영에서 관찰되므로 S1·S2는 배치 가능하지만, R1(part)은 구축 속성이라 운영 불가의 참고용이다. 보정 표본이 category당 작아 임계값이 최댓값 근처로 올라가므로 검토량 증가는 이 표본 크기에서의 값이다. test 한 번의 결과이며 탐색적이다.")
if __name__ == "__main__":
    if len(sys.argv) < 2: sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "axft", sys.argv[3] if len(sys.argv) > 3 else "d34_out")
