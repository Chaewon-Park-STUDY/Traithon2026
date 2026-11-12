"""d38c: 오류를 잡았을 때의 경계심 회복 정도(ρ)를 바꿔 canary 노출 효과와 경보 조치 효과를 분해한다.
사용: python -X utf8 d38c_recovery_assumption.py [출력폴더]    (약 3~6분)
numpy, scipy 사용. 난수 시뮬레이션. 환경변수 D38C_REPS, D38C_CAL, D38C_T 는 스모크 테스트용(결과 보고에 쓰지 않음).
"""
import sys, os, csv, math, itertools
import numpy as np
from scipy.stats import norm

SEED = 20261113
FA_TARGET = 0.05
T = int(os.environ.get("D38C_T", 1800))
R = int(os.environ.get("D38C_REPS", 1000))
RC = int(os.environ.get("D38C_CAL", 2000))
LAM = [0.0005, 0.001, 0.003]
E = [0.05, 0.15, 0.30]
V0 = [0.5, 0.9]
RATE = [0.02, 0.05, 0.10]
RHO = [0.0, 0.1, 0.3, 1.0]
W = 20
DELTA = 0.3
SIGQ = (0.1, 0.8)   # 강한 공짜 신호 (a_c, a_n)
OUT = sys.argv[1] if len(sys.argv) > 1 else "d38c_out"
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
    s = 0.0
    for i, x in enumerate(seq, start=1):
        s = max(0.0, s + (follow if x else notfollow))
        if s > h: return i
    return None


def simulate(lam, e, r, v0, det, rng, R, T, rho=1.0, thr=None, alerts=True):
    use_c = r > 0
    v = np.full(R, float(v0))
    stat = np.zeros(R); M = np.zeros(R)
    buf = np.zeros((R, W), np.int8); ncan = np.zeros(R, int)
    first = np.full(R, -1); nal = np.zeros(R, int)
    wf = np.zeros(R); real_n = np.zeros(R)
    if det == "cc": lp = llr_params(v0, DELTA)
    if det == "ccf": lp = llr_params(v0, DELTA, SIGQ)
    for t in range(T):
        can = (rng.random(R) < r) if use_c else np.zeros(R, bool)
        wrong = can | (rng.random(R) < e)
        chk = rng.random(R) < v
        followed = wrong & ~chk
        caught = wrong & chk
        real_n += ~can; wf += (~can) & followed
        v = np.where(~wrong, v * (1 - lam), v)
        v = np.where(caught, v + rho * (v0 - v), v)
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
            llr = np.where(can, np.where(followed, lp["follow"], lp["notfollow"]), 0.0)
            if det == "ccf":
                ac, an = SIGQ
                fast = rng.random(R) < np.where(chk, ac, an)
                llr = llr + np.where(fast, lp["fast"], lp["slow"])
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
                wf=float(ratio.mean() * 1000), wf_se=float(ratio.std(ddof=1) / math.sqrt(R) * 1000))


def calibrate(det, r, v0, seed):
    rng = np.random.default_rng(seed)
    s = simulate(0.0, 0.15, r, v0, det, rng, RC, T, alerts=False)
    Mx = s["M"]
    if det == "win":
        for k in range(1, W + 1):
            if (Mx >= k).mean() <= FA_TARGET: return k
        return W + 1
    return float(np.quantile(Mx, 1 - FA_TARGET, method="higher"))


def configs():
    for r in RATE: yield ("win", r)
    for r in RATE: yield ("cc", r)
    for r in [0.0] + RATE: yield ("ccf", r)


def tests(calib):
    rows = []
    def add(tid, desc, exp, act, basis):
        rows.append(dict(id=tid, 설명=desc, 기대=exp, 실제=act, 판정="PASS" if str(exp) == str(act) else "FAIL", 근거=basis))
    lp = llr_params(0.9, DELTA); f, nf = lp["follow"], lp["notfollow"]; h = 5.0
    add("T1-항상따름", "v0=0.9, δ=0.3, h=5: 첫 경보 인덱스", int(h // f) + 1, cusum_first_alert([1] * 50, f, nf, h), "S_n = n×log(p1/p0) > h 의 최소 n")
    add("T2-따르지않음", "따르지 않으면 경보 없음", None, cusum_first_alert([0] * 500, f, nf, h), "max(0, S+음수)=0")
    # ρ=0, 경보 없음, λ=0: v가 변하지 않으므로 틀린 제안 따름 확률이 정확히 (1−v0)×e 근처
    rng = np.random.default_rng([SEED, 50])
    s = simulate(0.0, 0.15, 0.05, 0.9, "none", rng, 4000, T, rho=0.0, alerts=False)
    ex = 0.15 * 0.1 * 1000   # 실제 항목의 틀린 제안 확률 e=0.15 × 확인하지 않을 확률 1−v0=0.1
    ok = abs(s["wf"] - ex) <= 3 * s["wf_se"]
    add("T3-정확값", f"λ=0, ρ=0: 실제 항목 1,000건당 틀린 제안 따름 {s['wf']:.2f} vs 기댓값 {ex:.2f} (3SE={3*s['wf_se']:.2f})", "일치(3SE 이내)", "일치(3SE 이내)" if ok else "불일치", "e × (1−v0)")
    with open(os.path.join(OUT, "d38c_tests.csv"), "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    bad = [x for x in rows if x["판정"] != "PASS"]
    print(f"기능 시험: {len(rows)}개 중 불일치 {len(bad)}")
    for x in bad: print("  FAIL", x["id"], x["기대"], x["실제"])


def main():
    print(f"d38c 시작 T={T} R={R} RC={RC}")
    calib, crows, ci = {}, [], 0
    for (det, r), v0 in itertools.product(configs(), V0):
        ci += 1
        thr = calibrate(det, r, v0, [SEED, 60, ci])
        calib[(det, r, v0)] = thr
        crows.append(dict(det=det, r=r, v0=v0, thr=thr))
    with open(os.path.join(OUT, "d38c_calib.csv"), "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(crows[0])); w.writeheader(); w.writerows(crows)
    tests(calib)
    N, A, cells, k = {}, {}, [], 0
    for rho, lam, e, v0 in itertools.product(RHO, LAM, E, V0):
        k += 1
        N[(rho, lam, e, v0)] = simulate(lam, e, 0.0, v0, "none", np.random.default_rng([SEED, 70, k]), R, T, rho=rho, alerts=False)
        for r in RATE:
            k += 1
            A[(rho, lam, e, v0, r)] = simulate(lam, e, r, v0, "none", np.random.default_rng([SEED, 71, k]), R, T, rho=rho, alerts=False)
    for (det, r), rho, lam, e, v0 in itertools.product(configs(), RHO, LAM, E, V0):
        k += 1
        thr = calib[(det, r, v0)]
        s = simulate(lam, e, r, v0, det, np.random.default_rng([SEED, 72, k]), R, T, rho=rho, thr=thr)
        n = N[(rho, lam, e, v0)]
        a = A[(rho, lam, e, v0, r)] if r > 0 else n
        cells.append(dict(rho=rho, det=det, r=r, v0=v0, lam=lam, e=e, thr=thr, detect_prob=s["detect_prob"],
                          wf_none=n["wf"], wf_expo=a["wf"], wf_alert=s["wf"],
                          expo_effect=n["wf"] - a["wf"], expo_se=math.hypot(n["wf_se"], a["wf_se"]),
                          alert_effect=a["wf"] - s["wf"], alert_se=math.hypot(a["wf_se"], s["wf_se"])))
    m = 2 * len(cells)      # 노출 효과와 경보 효과 비교 칸 수
    z_u, z_b = norm.ppf(1 - 0.05), norm.ppf(1 - 0.05 / m)
    for c in cells:
        for nm in ("expo", "alert"):
            c[f"{nm}_lcb_unadj"] = c[f"{nm}_effect"] - z_u * c[f"{nm}_se"]
            c[f"{nm}_lcb_bonf"] = c[f"{nm}_effect"] - z_b * c[f"{nm}_se"]
    print(f"ε–δ 하한: 비교 칸 수 m={m}, z(δ=0.05)={z_u:.3f}, z(Bonferroni)={z_b:.3f}")
    with open(os.path.join(OUT, "d38c_sweep.csv"), "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cells[0])); w.writeheader(); w.writerows(cells)
    hyp(cells)


def hyp(cells):
    rows = []
    # H1: ρ=0 노출 효과가 0
    z = [c for c in cells if c["rho"] == 0.0 and c["r"] > 0]
    bad = [c for c in z if abs(c["expo_effect"]) > 2 * c["expo_se"]]
    rows.append(dict(가설=f"H1 ρ=0 노출 효과 |추정|>2SE 칸 비율 {len(bad)/len(z):.4f} <= 0.05 (칸 {len(z)})", 값=int(len(bad) / len(z) <= 0.05)))
    # H2
    for v0 in V0:
        for det in ("win", "cc", "ccf"):
            c = [x for x in cells if x["rho"] == 1.0 and x["det"] == det and x["r"] == 0.10 and x["lam"] == 0.003 and x["e"] == 0.05 and x["v0"] == v0][0]
            rows.append(dict(가설=f"H2 ρ=1, λ=0.003, e=0.05, r=0.10, v0={v0}, {det}: 노출 효과 {c['expo_effect']:.2f} (하한 Bonf {c['expo_lcb_bonf']:.2f}), 경보 효과 {c['alert_effect']:.2f} (하한 Bonf {c['alert_lcb_bonf']:.2f}), 경보 확률 {c['detect_prob']:.2f}", 값=""))
    # H3
    for rho in RHO:
        for det in ("win", "cc", "ccf"):
            sub = [c for c in cells if c["rho"] == rho and c["det"] == det and c["r"] > 0]
            n_b = sum(1 for c in sub if c["alert_lcb_bonf"] > 0)
            mx = max(c["alert_lcb_bonf"] for c in sub)
            rows.append(dict(가설=f"H3 ρ={rho} {det}: 경보 효과 하한>0 칸(Bonf) {n_b}/{len(sub)}, 지원되는 최대 ε {max(mx,0):.3f}", 값=""))
    # H4
    for rho in RHO:
        sub = [c for c in cells if c["rho"] == rho and c["r"] > 0]
        n_e = sum(1 for c in sub if c["expo_lcb_bonf"] > 0)
        mx = max(c["expo_lcb_bonf"] for c in sub)
        rows.append(dict(가설=f"H4 ρ={rho}: 노출 효과 하한>0 칸(Bonf) {n_e}/{len(sub)}, 지원되는 최대 ε {max(mx,0):.3f}, 노출 효과 추정 중앙값 {float(np.median([c['expo_effect'] for c in sub])):.3f}", 값=""))
    with open(os.path.join(OUT, "d38c_hypotheses.csv"), "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["가설", "값"]); w.writeheader(); w.writerows(rows)
    for r in rows: print(" ", r["가설"])
    print("\n저장:", OUT)


if __name__ == "__main__":
    main()
