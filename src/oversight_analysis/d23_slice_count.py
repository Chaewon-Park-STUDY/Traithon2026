# 제목 표현 슬라이스(지시어·말줄임표·의문형·짧은 제목) 개수 세기 + 검토 부하 영향 범위 계산.
# 출력은 집계 CSV뿐이다. 기사 원문, 제목, ID는 저장하지 않는다.
# 사용: python d23_slice_count.py "AI-Hub 데이터 폴더 또는 zip이 든 폴더"  [출력폴더]
# 규칙은 결과를 보기 전에 커밋해 사전 등록한다(결과를 보고 규칙을 바꾸지 않는다).
import os, re, sys, json, zipfile, csv
from collections import Counter
from scipy.stats import beta

# ---- 사전 등록 규칙 (제목 문자열에만 적용) ----
DEMO = ["이것", "이거", "이게", "이건", "그것", "그거", "그게", "그건", "저것", "이곳", "그곳", "이들", "그들",
        "그가", "그는", "그녀", "이 사람", "그 사람", "이런", "이렇게", "이러면", "이래서"]
RE_DEMO = re.compile(r"(?:^|[\s\"'“‘(\[])(?:" + "|".join(map(re.escape, DEMO)) + r")")
FLAGS = {
    "F1_지시어": lambda t: bool(RE_DEMO.search(t)),
    "F2_말줄임표": lambda t: ("…" in t) or ("..." in t) or ("··" in t),
    "F3_의문형": lambda t: "?" in t,
    "F4_짧은제목(어절<=5)": lambda t: len(t.split()) <= 5,
}
SENSITIVE = {"정치", "경제"}

def find(o, key, default=None):
    if isinstance(o, dict):
        if key in o: return o[key]
        for v in o.values():
            r = find(v, key, None)
            if r is not None: return r
    elif isinstance(o, list):
        for v in o:
            r = find(v, key, None)
            if r is not None: return r
    return default

def iter_json(src):
    for dp, _, fs in os.walk(src):
        for fn in sorted(fs):
            p = os.path.join(dp, fn)
            if fn.lower().endswith(".zip"):
                with zipfile.ZipFile(p) as z:
                    for n in z.namelist():
                        if n.lower().endswith(".json") and "원천" not in n:
                            yield json.loads(z.read(n).decode("utf-8-sig"))
            elif fn.lower().endswith(".json") and "원천" not in p:
                with open(p, encoding="utf-8-sig") as f:
                    yield json.load(f)

def cp(k, n, a=0.05):
    lo = 0.0 if k == 0 else beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a / 2, k + 1, n - k)
    return lo, hi

def main(src, out):
    cnt = Counter(); n_json = 0; n_nolabel = 0
    for d in iter_json(src):
        n_json += 1
        code = find(d, "clickbaitClass")
        if code is None: n_nolabel += 1; continue
        label = "클릭베이트" if int(code) == 0 else "정상"           # AI-Hub: clickbaitClass 0 = 클릭베이트
        ptype = str(find(d, "processType", "") or "")
        ptype = "Direct" if ptype.upper().startswith("D") else "Auto"
        cat = str(find(d, "newsCategory", "") or "")
        sens = "민감(정치·경제)" if cat in SENSITIVE else "비민감"
        title = str(find(d, "newTitle") or find(d, "newsTitle") or "").strip()
        hit = {k: f(title) for k, f in FLAGS.items()}
        hit["ANY(F1~F4 중 하나 이상)"] = any(hit.values())
        hit["ANY(F1~F3 중 하나 이상)"] = any(hit[k] for k in list(FLAGS)[:3])
        key = (label, ptype, sens)
        cnt[(key, "N")] += 1
        for k, v in hit.items():
            if v: cnt[(key, k)] += 1
    names = list(FLAGS) + ["ANY(F1~F4 중 하나 이상)", "ANY(F1~F3 중 하나 이상)"]
    keys = sorted({k for (k, _) in cnt})
    rows = []
    for key in keys:
        n = cnt[(key, "N")]
        for nm in names:
            k = cnt[(key, nm)]; lo, hi = cp(k, n)
            rows.append(dict(label=key[0], process_type=key[1], sensitivity=key[2], flag=nm, n=n, k=k,
                             rate=round(k / n, 4), ci95_lo=round(lo, 4), ci95_hi=round(hi, 4)))
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "d23_slice_counts.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"JSON {n_json}건 읽음, 라벨 없음 {n_nolabel}건")
    print(f"{'label':8}{'type':8}{'sens':14}{'flag':26}{'n':>8}{'k':>8}{'rate':>8}   95%CP")
    for r in rows:
        print(f"{r['label']:8}{r['process_type']:8}{r['sensitivity']:14}{r['flag']:26}{r['n']:>8}{r['k']:>8}{r['rate']:>8}   [{r['ci95_lo']}, {r['ci95_hi']}]")
    # ---- 검토 부하 영향 범위 (비민감 기사 기준) ----
    S, N = 0.278, 1000; ap, an = 0.0307, 0.9571          # a+, a-: results/alpha_sweep_axft.csv 의 α=0.05 maxq
    def rate(label, nm):
        n = sum(cnt[((l, t, s), "N")] for (l, t, s) in keys if l == label and s == "비민감")
        k = sum(cnt[((l, t, s), nm)] for (l, t, s) in keys if l == label and s == "비민감")
        return (k / n) if n else float("nan")
    print("\n=== 규칙 플래그를 Trigger에 추가했을 때 일 추가 이관 건수의 범위 (N=1000/일, s=0.278, α=0.05 기준) ===")
    print("하한 0, 독립 가정(플래그와 현재 이관이 무관): (1-s)·N·f·자동통과율, 상한(플래그 전부 현재 자동통과): (1-s)·N·f")
    print("주의: AI-Hub 비율 f는 이 코퍼스(클릭베이트 일부는 모델 생성 제목)의 값이며 서비스 f가 아니다.")
    for nm in ("ANY(F1~F3 중 하나 이상)", "ANY(F1~F4 중 하나 이상)"):
        fc, fo = rate("클릭베이트", nm), rate("정상", nm)
        for pi in (0.05, 0.10, 0.25):
            f_mix = pi * fc + (1 - pi) * fo
            auto = pi * ap + (1 - pi) * an
            print(f"{nm:26} π={pi:.2f}  f={f_mix:.3f}  독립가정 +{(1-S)*N*f_mix*auto:6.1f}건/일   상한 +{(1-S)*N*f_mix:6.1f}건/일")

if __name__ == "__main__":
    if len(sys.argv) < 2: sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "d23_out")
