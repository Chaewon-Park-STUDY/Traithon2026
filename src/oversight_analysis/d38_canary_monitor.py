"""d38: canary 감시 규칙 시험(1부)과 검토자 시뮬레이션(2부, 가정 기반 민감도 분석).
사용: python -X utf8 d38_canary_monitor.py [출력폴더]      (약 1~3분)
numpy, scipy 사용. 난수 시뮬레이션이며 실제 운영·사람 데이터가 아님.
환경변수 D38_REPS, D38_T 로 시행·기간을 줄일 수 있다(스모크 테스트용, 결과 보고에는 쓰지 않는다).
"""
import sys, os, csv, itertools, math
import numpy as np
from scipy.stats import binom

SEED = 20261113
ALPHA_ALARM = 0.05
T = int(os.environ.get("D38_T", 1800))       # 12일 x 150건/일 (demo CAP_PER)
R = int(os.environ.get("D38_REPS", 1000))
LAM = [0, 0.0005, 0.001, 0.003]
E = [0.05, 0.15, 0.30]
V0 = [0.5, 0.9]
RATE = [0.01, 0.02, 0.05, 0.10]
WIN = [10, 20, 40]
OUT = sys.argv[1] if len(sys.argv) > 1 else "d38_out"
os.makedirs(OUT, exist_ok=True)


def k_alarm(W, p0, alpha=ALPHA_ALARM):
    """P(Bin(W,p0) >= k) <= alpha 를 만족하는 최소 k (없으면 W+1: 경보 불가)."""
    for k in range(1, W + 1):
        if binom.sf(k - 1, W, p0) <= alpha:
            return k
    return W + 1


class Monitor:
    """최근 canary W건 중 따름 횟수 >= k 이면 경보, 경보 후 창을 비운다."""
    def __init__(self, W, k): self.W, self.k, self.buf = W, k, []
    def canary(self, followed):
        self.buf.append(int(followed))
        if len(self.buf) > self.W: self.buf.pop(0)
        if len(self.buf) == self.W and sum(self.buf) >= self.k:
            self.buf = []
            return True
        return False


def alert_indices(pattern, W, k):
    m, out = Monitor(W, k), []
    for i, f in enumerate(pattern, start=1):
        if m.canary(f): out.append(i)
    return out


def simulate(lam, e, r, W, v0, rng, R=R, T=T):
    k = k_alarm(W, 1 - v0) if r > 0 else None
    v = np.full(R, float(v0))
    use_c = r > 0
    buf = np.zeros((R, W), np.int8)
    ncan = np.zeros(R, int)
    first_alert = np.full(R, -1)
    n_alerts = np.zeros(R, int)
    fa_first = np.full(R, -1)             # 첫 번째 가득 찬 창의 경보 여부(0/1), 미도달 -1
    wf = np.zeros(R); real_n = np.zeros(R); can_n = np.zeros(R)
    for t in range(T):
        can = (rng.random(R) < r) if use_c else np.zeros(R, bool)
        wrong = can | (rng.random(R) < e)
        chk = rng.random(R) < v
        followed = wrong & ~chk
        caught = wrong & chk
        real = ~can
        real_n += real; can_n += can
        wf += real & followed
        v = np.where(~wrong, v * (1 - lam), v)
        v = np.where(caught, v0, v)
        if use_c:
            idx = np.where(can)[0]
            if idx.size:
                buf[idx, ncan[idx] % W] = followed[idx]
                ncan[idx] += 1
                full = ncan[idx] >= W
                cond = full & (buf[idx].sum(1) >= k)
                fresh = (ncan[idx] == W) & (fa_first[idx] == -1)
                fa_first[idx[fresh]] = cond[fresh].astype(int)
                al = idx[cond]
                if al.size:
                    n_alerts[al] += 1
                    nf = al[first_alert[al] == -1]
                    first_alert[nf] = t + 1
                    v[al] = v0; buf[al] = 0; ncan[al] = 0
    reached = fa_first >= 0
    det = first_alert >= 0
    return dict(k=k, detect_prob=det.mean(),
                median_first_alert=float(np.median(first_alert[det])) if det.any() else float("nan"),
                alerts_per_1000=float(n_alerts.mean() / T * 1000),
                wrong_followed_per_1000=float((wf / np.maximum(real_n, 1)).mean() * 1000),
                wf_se=float((wf / np.maximum(real_n, 1)).std(ddof=1) / math.sqrt(R) * 1000),
                canary_share=float(can_n.sum() / (can_n.sum() + real_n.sum())),
                n_reached_window=int(reached.sum()),
                fa_first=float(fa_first[reached].mean()) if reached.any() else float("nan"))


def part1():
    rows = []
    def add(tid, desc, expected, actual, basis):
        rows.append(dict(id=tid, 설명=desc, 기대=expected, 실제=actual,
                         판정="PASS" if str(expected) == str(actual) else "FAIL", 근거=basis))
    for W in (20, 10):
        k = k_alarm(W, 0.1)
        pats = {
            "항상 따름": ([1] * (6 * W), W),
            "한 번 건너 따름": ([i % 2 for i in range(6 * W)], W // 2),
            "10건에 1건 따름": ([1 if i % 10 == 0 else 0 for i in range(10 * W)], W // 10),
            "따르지 않음": ([0] * (6 * W), 0),
        }
        for name, (seq, c) in pats.items():
            if name == "항상 따름":
                exp = [W, 2 * W, 3 * W, 4 * W, 5 * W, 6 * W]
            else:
                # 창 안의 따름 횟수가 상수 c 이므로 c >= k 이면 첫 경보는 W번째, 이후 W건마다
                exp = [W * j for j in range(1, len(seq) // W + 1)] if c >= k else []
            add(f"T1-W{W}-{name}", f"W={W}, p0=0.1, k={k}, 패턴={name}", exp, alert_indices(seq, W, k),
                "규칙: 창 안 따름 >= k 이면 경보, 경보 후 창 비움")
    rng = np.random.default_rng([SEED, 1])
    s = simulate(0.0, 0.15, 0.05, 20, 1.0, rng)
    add("T2-v0=1", "v0=1, λ=0(항상 확인): canary 따름 없음, 경보 확률 0", 0.0, float(s["detect_prob"]), "p0=0, k=1")
    s = simulate(0.003, 0.0, 0.0, 20, 0.9, rng)
    add("T3-r=0", "canary 없음, AI 오류 없음: 경보 없음, 틀린 제안 따름 0", (0.0, 0.0), (float(s["detect_prob"]), float(s["wrong_followed_per_1000"])), "정의상")
    # 정확확률 대조: λ=0 첫 번째 가득 찬 창의 경보 확률 = P(Bin(W,p0) >= k)
    for W, v0 in ((20, 0.9), (10, 0.5), (40, 0.9)):
        k = k_alarm(W, 1 - v0)
        ex = float(binom.sf(k - 1, W, 1 - v0))
        rng = np.random.default_rng([SEED, 2, W])
        s = simulate(0.0, 0.15, 0.10, W, v0, rng, R=4000, T=max(T, 1800))
        n = s["n_reached_window"]
        se = math.sqrt(max(ex * (1 - ex), 1e-9) / max(n, 1))
        ok = abs(s["fa_first"] - ex) <= 3 * se + 1e-12
        add(f"T4-exact-W{W}-v{v0}", f"λ=0 첫 창 경보 확률: 시뮬레이션 {s['fa_first']:.4f} vs 정확값 {ex:.4f} (n={n}, 3SE={3*se:.4f})",
            "일치(3SE 이내)", "일치(3SE 이내)" if ok else "불일치", "Bin(W, 1−v0) 꼬리확률")
    with open(os.path.join(OUT, "d38_tests.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    bad = [r for r in rows if r["판정"] != "PASS"]
    print(f"1부 기능 시험: {len(rows)}개 중 불일치 {len(bad)}")
    for r in bad: print("  FAIL", r["id"], r["기대"], r["실제"])
    return rows


def part2():
    cells, base = [], {}
    ci = 0
    for lam, e, v0 in itertools.product(LAM, E, V0):
        ci += 1
        s = simulate(lam, e, 0.0, 10, v0, np.random.default_rng([SEED, 3, ci]))
        base[(lam, e, v0)] = (s["wrong_followed_per_1000"], s["wf_se"])
        cells.append(dict(lam=lam, e=e, v0=v0, r=0.0, W="", k="", **{kk: vv for kk, vv in s.items() if kk != "k"}, 감소율=0.0, 유의감소=""))
        for r, W in itertools.product(RATE, WIN):
            ci += 1
            s = simulate(lam, e, r, W, v0, np.random.default_rng([SEED, 3, ci]))
            b, bse = base[(lam, e, v0)]
            red = (1 - s["wrong_followed_per_1000"] / b) if b > 0 else float("nan")
            sig = (b - s["wrong_followed_per_1000"]) > 2 * math.sqrt(bse ** 2 + s["wf_se"] ** 2)
            cells.append(dict(lam=lam, e=e, v0=v0, r=r, W=W, k=s["k"], **{kk: vv for kk, vv in s.items() if kk != "k"}, 감소율=red, 유의감소=int(sig)))
    with open(os.path.join(OUT, "d38_sweep.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(cells[0])); w.writeheader(); w.writerows(cells)
    return cells


def hypotheses(cells):
    rows = []
    # H1
    h1 = [c for c in cells if c["lam"] == 0 and c["r"] > 0]
    ok = bad = skip = 0
    for c in h1:
        n = c["n_reached_window"]
        if n < 100: skip += 1; continue
        tol = 2 * math.sqrt(ALPHA_ALARM * (1 - ALPHA_ALARM) / n)
        if c["fa_first"] <= ALPHA_ALARM + tol: ok += 1
        else: bad += 1; print("  H1 위반", c["e"], c["v0"], c["r"], c["W"], c["fa_first"])
    rows.append(dict(가설="H1 λ=0 첫 창 거짓 경보 <= 0.05", 통과=ok, 위반=bad, 판정불가=skip))
    # H2
    keyed = {}
    for c in cells:
        if c["lam"] > 0 and c["r"] > 0:
            keyed.setdefault((c["lam"], c["e"], c["v0"], c["W"]), []).append(c)
    ok = bad = 0
    for key, lst in keyed.items():
        lst.sort(key=lambda c: c["r"])
        good = all(lst[i + 1]["detect_prob"] >= lst[i]["detect_prob"] - 0.03 for i in range(len(lst) - 1))
        ok += good; bad += (not good)
        if not good: print("  H2 위반", key, [round(c["detect_prob"], 3) for c in lst])
    rows.append(dict(가설="H2 r가 커질수록 경보 확률이 줄지 않음(허용 0.03)", 통과=ok, 위반=bad, 판정불가=0))
    # H3
    h3 = [c for c in cells if c["lam"] > 0 and c["r"] > 0 and not (isinstance(c["감소율"], float) and math.isnan(c["감소율"]))]
    pos = sum(1 for c in h3 if c["감소율"] > 0)
    nodet = sum(1 for c in h3 if c["detect_prob"] < 0.05)
    sigc = sum(1 for c in h3 if c["유의감소"] == 1)
    rows.append(dict(가설="H3 (서술) λ>0 칸 중 감소율>0 인 칸 수 / 전체", 통과=pos, 위반=len(h3) - pos, 판정불가=0))
    rows.append(dict(가설="H3 (서술) λ>0 칸 중 감소가 2 표준오차 이상인 칸 수 / 전체", 통과=sigc, 위반=len(h3) - sigc, 판정불가=0))
    rows.append(dict(가설="H3 (서술) λ>0 칸 중 경보 확률 < 0.05 인 칸 수(창이 차지 않음 등)", 통과=nodet, 위반="", 판정불가=""))
    with open(os.path.join(OUT, "d38_hypotheses.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["가설", "통과", "위반", "판정불가"]); w.writeheader(); w.writerows(rows)
    for r in rows: print(" ", r)
    return h3


def summary(h3):
    print("\n[요약] λ>0 칸의 감소율 / 경보 확률 (r별 중앙값, 칸 수는 W·e·v0·λ 조합)")
    for r in RATE:
        sub = [c for c in h3 if c["r"] == r]
        if sub:
            print(f"  r={r:<5} 감소율 중앙값 {np.median([c['감소율'] for c in sub]):6.3f}  경보 확률 중앙값 {np.median([c['detect_prob'] for c in sub]):5.3f}  감소율<=0 칸 {sum(1 for c in sub if c['감소율'] <= 0)}/{len(sub)}  유의 감소 칸 {sum(1 for c in sub if c['유의감소'] == 1)}/{len(sub)}")


if __name__ == "__main__":
    print(f"d38 시작 T={T} R={R} (기본값이 아니면 스모크 테스트)")
    part1()
    cells = part2()
    h3 = hypotheses(cells)
    summary(h3)
    print("\n저장:", OUT)
