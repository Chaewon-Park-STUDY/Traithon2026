"""d38b: 같은 누적 거짓 경보 수준(12일 5% 이하)에서 감시 규칙 비교: 창 규칙(win), canary CUSUM(cc), 공짜 신호 추가 CUSUM(ccf).
사용: python -X utf8 d38b_matched_fa_detectors.py [출력폴더]    (약 3~8분)
numpy, scipy 사용. 난수 시뮬레이션. 환경변수 D38B_REPS, D38B_CAL, D38B_T 는 스모크 테스트용(결과 보고에 쓰지 않음).
"""
import sys, os, csv, math, itertools
import numpy as np
from scipy.stats import norm

SEED = 20261113
FA_TARGET = 0.05
T = int(os.environ.get("D38B_T", 1800))
R = int(os.environ.get("D38B_REPS", 1000))
RC = int(os.environ.get("D38B_CAL", 2000))
LAM = [0, 0.0005, 0.001, 0.003]
E = [0.05, 0.15, 0.30]
V0 = [0.5, 0.9]
RATE = [0.01, 0.02, 0.05, 0.10]
W = 20
DELTA = [0.3, 0.6]
SIG = {"weak": (0.3, 0.6), "strong": (0.1, 0.8)}
OUT = sys.argv[1] if len(sys.argv) > 1 else "d38b_out"
os.makedirs(OUT, exist_ok=True)


def llr_params(v0, delta, sig=None):
    v1 = v0 * (1 - delta)
    p0, p1 = 1 - v0, 1 - v1
    d = dict(follow=math.log(p1 / p0), notfollow=math.log((1 - p1) / (1 - p0)))
    if sig:
        ac, an = sig
        q0, q1 = v0 * ac + (1 - v0) * an, v1 * ac + (1 - v1) * an
        d["fast"] = math.log(q1 / q0); d["slow"] = math.log((1 - q1) / (1 - q0))
    return d


def cusum_first_alert(seq, follow, notfollow, h):
    """스칼라 CUSUM(시험용). seq: 1=따름, 0=안 따름. 첫 경보 인덱스(1부터), 없으면 None."""
    s = 0.0
    for i, x in enumerate(seq, start=1):
        s = max(0.0, s + (follow if x else notfollow))
        if s > h: return i
    return None


def simulate(lam, e, r, v0, det, rng, R, T, thr=None, delta=None, sig=None, alerts=True):
    use_c = r > 0
    v = np.full(R, float(v0))
    stat = np.zeros(R); M = np.zeros(R)
    buf = np.zeros((R, W), np.int8); ncan = np.zeros(R, int)
    first = np.full(R, -1); nal = np.zeros(R, int)
    wf = np.zeros(R); real_n = np.zeros(R)
    if det in ("cc", "ccf"):
        lp = llr_params(v0, delta, SIG[sig] if det == "ccf" else None)
    for t in range(T):
        can = (rng.random(R) < r) if use_c else np.zeros(R, bool)
        wrong = can | (rng.random(R) < e)
        chk = rng.random(R) < v
        followed = wrong & ~chk
        caught = wrong & chk
        real_n += ~can; wf += (~can) & followed
        v = np.where(~wrong, v * (1 - lam), v)
        v = np.where(caught, v0, v)
        cond = np.zeros(R, bool)
        if det == "win" and use_c:
            idx = np.where(can)[0]
            if idx.size:
                buf[idx, ncan[idx] % W] = followed[idx]
                ncan[idx] += 1
                full = ncan[idx] >= W
                s = np.where(full, buf[idx].sum(1), 0)
                M[idx] = np.maximum(M[idx], s)
                if alerts: cond[idx[full & (s >= thr)]] = True
        elif det in ("cc", "ccf"):
            llr = np.zeros(R)
            llr += np.where(can, np.where(followed, lp["follow"], lp["notfollow"]), 0.0)
            if det == "ccf":
                ac, an = SIG[sig]
                fast = rng.random(R) < np.where(chk, ac, an)
                llr += np.where(fast, lp["fast"], lp["slow"])
            stat = np.maximum(0.0, stat + llr)
            M = np.maximum(M, stat)
            if alerts: cond = stat > thr
        if alerts and cond.any():
            al = np.where(cond)[0]
            nal[al] += 1
            nf = al[first[al] == -1]; first[nf] = t + 1
            v[al] = v0; stat[al] = 0.0; buf[al] = 0; ncan[al] = 0
    det_ = first >= 0
    ratio = wf / np.maximum(real_n, 1)
    return dict(M=M, detect_prob=float(det_.mean()),
                median_first_alert=float(np.median(first[det_])) if det_.any() else float("nan"),
                alerts_per_1000=float(nal.mean() / T * 1000),
                wf=float(ratio.mean() * 1000), wf_se=float(ratio.std(ddof=1) / math.sqrt(R) * 1000))


def calibrate(det, r, v0, delta, sig, seed):
    rng = np.random.default_rng(seed)
    s = simulate(0.0, 0.15, r, v0, det, rng, RC, T, delta=delta, sig=sig, alerts=False)
    Mx = s["M"]
    if det == "win":
        for k in range(1, W + 1):
            if (Mx >= k).mean() <= FA_TARGET: return k
        return W + 1
    return float(np.quantile(Mx, 1 - FA_TARGET, method="higher"))


def configs():
    for r in RATE: yield ("win", r, None, None)
    for r, d in itertools.product(RATE, DELTA): yield ("cc", r, d, None)
    for r, d, sg in itertools.product([0.0] + RATE, DELTA, SIG): yield ("ccf", r, d, sg)


def tests(calib):
    rows = []
    def add(tid, desc, exp, act, basis):
        rows.append(dict(id=tid, 설명=desc, 기대=exp, 실제=act, 판정="PASS" if str(exp) == str(act) else "FAIL", 근거=basis))
    lp = llr_params(0.9, 0.3)
    f, nf = lp["follow"], lp["notfollow"]
    h = 5.0
    n_exp = int(h // f) + 1
    add("T1-항상따름", f"v0=0.9, δ=0.3, h=5: 첫 경보 인덱스", n_exp, cusum_first_alert([1] * 50, f, nf, h), "S_n = n×log(p1/p0) > h 의 최소 n")
    add("T2-따르지않음", "따르지 않으면 S=0 유지, 경보 없음", None, cusum_first_alert([0] * 500, f, nf, h), "max(0, S+음수)=0")
    seq = [1] * 3 + [0] * 3 + [1] * 50
    s, exp = 0.0, None
    for i, x in enumerate(seq, 1):
        s = max(0.0, s + (f if x else nf))
        if s > h: exp = i; break
    add("T3-섞임", "따름 3, 안 따름 3, 따름 연속: 손계산 CUSUM과 일치", exp, cusum_first_alert(seq, f, nf, h), "정의")
    # 벡터화 구현과 스칼라 구현의 일치: λ=0, cc, r=0.05, v0=0.9, δ=0.3, 보정된 h 로 12일 동안 경보 확률
    key = ("cc", 0.05, 0.9, 0.3, None)
    h = calib[key]
    rng = np.random.default_rng([SEED, 90])
    sv = simulate(0.0, 0.15, 0.05, 0.9, "cc", rng, 4000, T, thr=h, delta=0.3)
    rg = np.random.default_rng([SEED, 91]); n = 2000; hit = 0
    for _ in range(n):
        s = 0.0; ok = False
        can = rg.random(T) < 0.05
        fol = rg.random(T) < 0.1   # 따름 확률 p0 = 1 − v0
        for t in range(T):
            if can[t]:
                s = max(0.0, s + (f if fol[t] else nf))
                if s > h: ok = True; break
        hit += ok
    ps = hit / n
    se = math.sqrt(max(ps * (1 - ps), 1e-9) / n + max(sv["detect_prob"] * (1 - sv["detect_prob"]), 1e-9) / 4000)
    add("T4-구현일치", f"벡터화 시뮬레이션 {sv['detect_prob']:.4f} vs 스칼라 {ps:.4f} (3SE={3*se:.4f})", "일치(3SE 이내)",
        "일치(3SE 이내)" if abs(sv["detect_prob"] - ps) <= 3 * se else "불일치", "같은 분포의 독립 구현")
    with open(os.path.join(OUT, "d38b_tests.csv"), "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    bad = [x for x in rows if x["판정"] != "PASS"]
    print(f"기능 시험: {len(rows)}개 중 불일치 {len(bad)}")
    for x in bad: print("  FAIL", x["id"], x["기대"], x["실제"])


def main():
    print(f"d38b 시작 T={T} R={R} RC={RC}")
    calib, crows = {}, []
    ci = 0
    for (det, r, d, sg), v0 in itertools.product(configs(), V0):
        ci += 1
        thr = calibrate(det, r, v0, d, sg, [SEED, 10, ci])
        calib[(det, r, v0, d, sg)] = thr
        crows.append(dict(det=det, r=r, v0=v0, delta=d if d is not None else "", sig=sg or "", thr=thr))
    with open(os.path.join(OUT, "d38b_calib.csv"), "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(crows[0])); w.writeheader(); w.writerows(crows)
    tests(calib)
    base, cells, ei = {}, [], 0
    for lam, e, v0 in itertools.product(LAM, E, V0):
        ei += 1
        s = simulate(lam, e, 0.0, v0, "none", np.random.default_rng([SEED, 20, ei]), R, T, alerts=False)
        base[(lam, e, v0)] = (s["wf"], s["wf_se"])
    for (det, r, d, sg), lam, e, v0 in itertools.product(configs(), LAM, E, V0):
        ei += 1
        thr = calib[(det, r, v0, d, sg)]
        s = simulate(lam, e, r, v0, det, np.random.default_rng([SEED, 30, ei]), R, T, thr=thr, delta=d, sig=sg)
        b, bse = base[(lam, e, v0)]
        red = (1 - s["wf"] / b) if b > 0 else float("nan")
        cells.append(dict(det=det, r=r, delta=d if d is not None else "", sig=sg or "", v0=v0, lam=lam, e=e, thr=thr,
                          detect_prob=s["detect_prob"], median_first_alert=s["median_first_alert"],
                          alerts_per_1000=s["alerts_per_1000"], wf_per_1000=s["wf"], base_wf_per_1000=b,
                          wf_se=s["wf_se"], base_se=bse, reduction=red))
    # ε–δ: 절대 감소량(1,000건당)의 단측 하한. 정규근사(CLT). δ=0.05, Bonferroni 는 δ/m (m = λ>0 칸 수)
    m = sum(1 for c in cells if c["lam"] > 0)
    z_u, z_b = norm.ppf(1 - 0.05), norm.ppf(1 - 0.05 / m)
    for c in cells:
        diff = c["base_wf_per_1000"] - c["wf_per_1000"]
        se = math.sqrt(c["base_se"] ** 2 + c["wf_se"] ** 2)
        c["abs_reduction_per_1000"] = diff
        c["lcb_unadj"] = diff - z_u * se
        c["lcb_bonf"] = diff - z_b * se
    print(f"ε–δ 하한: λ>0 칸 수 m={m}, z(δ=0.05)={z_u:.3f}, z(Bonferroni)={z_b:.3f}")
    with open(os.path.join(OUT, "d38b_sweep.csv"), "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cells[0])); w.writeheader(); w.writerows(cells)
    hyp(cells)


def hyp(cells):
    rows = []
    # H1
    z = [c for c in cells if c["lam"] == 0 and not (c["det"] == "win" and c["thr"] > W)]
    bad = [c for c in z if c["detect_prob"] > FA_TARGET + 2 * math.sqrt(FA_TARGET * (1 - FA_TARGET) / R)]
    mean_fa = float(np.mean([c["detect_prob"] for c in z]))
    rows.append(dict(가설=f"H1(a) λ=0 누적 거짓 경보 평균 {mean_fa:.4f} <= 0.05", 통과=int(mean_fa <= FA_TARGET), 위반=int(mean_fa > FA_TARGET)))
    rows.append(dict(가설=f"H1(b) 0.05+2SE 를 넘는 칸 비율 {len(bad)/len(z):.4f} <= 0.05", 통과=int(len(bad) / len(z) <= 0.05), 위반=int(len(bad) / len(z) > 0.05)))
    for c in bad[:10]: print("  H1 위반", c["det"], c["r"], c["delta"], c["sig"], c["v0"], c["e"], round(c["detect_prob"], 3))
    idx = {(c["det"], c["r"], c["delta"], c["sig"], c["v0"], c["lam"], c["e"]): c for c in cells}
    pos = [c for c in cells if c["lam"] > 0]
    # H2 (서술)
    for d in DELTA:
        w = t = l = 0
        for r, v0, lam, e in itertools.product(RATE, V0, LAM[1:], E):
            a = idx[("cc", r, d, "", v0, lam, e)]["detect_prob"]; b = idx[("win", r, "", "", v0, lam, e)]["detect_prob"]
            w += a > b; t += a == b; l += a < b
        rows.append(dict(가설=f"H2(서술) cc(δ={d}) 대 win 경보 확률: 높음 {w} / 같음 {t} / 낮음 {l}", 통과="", 위반=""))
    # H3 (서술)
    for d, sg in itertools.product(DELTA, SIG):
        w = t = l = 0
        for r, v0, lam, e in itertools.product(RATE, V0, LAM[1:], E):
            a = idx[("ccf", r, d, sg, v0, lam, e)]["detect_prob"]; b = idx[("cc", r, d, "", v0, lam, e)]["detect_prob"]
            w += a > b; t += a == b; l += a < b
        rows.append(dict(가설=f"H3(서술) ccf(δ={d}, 신호 {sg}) 대 cc 경보 확률: 높음 {w} / 같음 {t} / 낮음 {l}", 통과="", 위반=""))
    # H4 (서술): ε–δ 하한
    for det in ("win", "cc", "ccf"):
        sub = [c for c in pos if c["det"] == det]
        n_b = sum(1 for c in sub if c["lcb_bonf"] > 0); n_u = sum(1 for c in sub if c["lcb_unadj"] > 0)
        mx = max(c["lcb_bonf"] for c in sub)
        rows.append(dict(가설=f"H4(서술) {det}: 하한>0 칸 수 Bonferroni {n_b} / 비보정 {n_u} / 전체 {len(sub)}, 지원되는 최대 ε(Bonferroni) {max(mx,0):.3f}건/1,000건", 통과="", 위반=""))
    sub = [c for c in pos if c["det"] == "ccf" and c["r"] == 0]
    rows.append(dict(가설=f"H4(서술) ccf r=0(canary 없이 공짜 신호만): 하한>0 칸 수 Bonferroni {sum(1 for c in sub if c['lcb_bonf'] > 0)} / 전체 {len(sub)}", 통과="", 위반=""))
    top = sorted(pos, key=lambda c: -c["lcb_bonf"])[:30]
    with open(os.path.join(OUT, "d38b_top.csv"), "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(top[0])); w.writeheader(); w.writerows(top)
    with open(os.path.join(OUT, "d38b_hypotheses.csv"), "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["가설", "통과", "위반"]); w.writeheader(); w.writerows(rows)
    for r in rows: print(" ", r)
    print("\n저장:", OUT)


if __name__ == "__main__":
    main()
