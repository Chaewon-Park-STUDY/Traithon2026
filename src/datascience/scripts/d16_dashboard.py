"""결과 대시보드(시연 자료 데이터 화면). outputs/*.json·csv 를 읽어 단일 HTML(인라인 SVG, 외부 의존 없음, 폰트만 CDN)을 만든다.
A.X 결과(eval_summary_ax_maxq.json 등)가 있으면 자동으로 세 번째 모델로 들어간다.
사용법: python d16_dashboard.py → outputs/dashboard.html
"""
import os, json, html
import pandas as pd
from common import OUT

J = lambda f: json.load(open(os.path.join(OUT, f), encoding="utf-8")) if os.path.exists(os.path.join(OUT, f)) else None
C = lambda f: pd.read_csv(os.path.join(OUT, f)) if os.path.exists(os.path.join(OUT, f)) else None
MODELS = [("baseline", "TF-IDF 기준선", 1), ("feat", "정합성 기준선", 2), ("ax", "A.X 4.0 Light", 3)]
MODELS = [m for m in MODELS if os.path.exists(os.path.join(OUT, f"eval_{m[0]}_maxq.csv"))]
esc = html.escape

# ---------- 데이터 모으기 ----------
cov = []
for key, label, slot in MODELS:
    e = C(f"eval_{key}_maxq.csv").set_index("slice")
    for t, s in (("Direct", "DIRECT_cb+all_ok"), ("Auto", "AUTO_cb+all_ok")):
        cov.append(dict(model=label, slot=slot, type=t, v=e.loc[s, "coverage_cb"], lo=e.loc[s, "coverage_cb_lo"], hi=e.loc[s, "coverage_cb_hi"], n=int(e.loc[s, "n_cb"])))
summ = {m[1]: J(f"eval_summary_{m[0]}_maxq.json") for m in MODELS}
sweep = {m[1]: C(f"alpha_sweep_{m[0]}.csv") for m in MODELS if C(f"alpha_sweep_{m[0]}.csv") is not None}
cat = C("category_calib_v2.csv")
ev = C("evidence_summary.csv")
stab = {m[1]: J(f"stability_summary_{m[0]}.json") for m in MODELS if J(f"stability_summary_{m[0]}.json")}
dq = J("dq_metrics_summary.csv") if False else C("dq_metrics_summary.csv")
first = MODELS[-1]  # 가장 최신(있으면 A.X) 모델을 KPI 기준으로
fs = summ[first[1]]
fe = C(f"eval_{first[0]}_maxq.csv").set_index("slice")
sw_first = sweep.get(first[1])
rev_a05 = int(sw_first[(sw_first.alpha == 0.05) & (sw_first["mode"] == "maxq")]["reviewers_pi0.1"].iloc[0]) if sw_first is not None else None

# ---------- SVG 도우미 ----------
W = 560
def axis_y(y0, y1, h, ticks, fmt, x0=48, x1=W - 12):
    out = []
    for t in ticks:
        y = h - 28 - (t - y0) / (y1 - y0) * (h - 48)
        out.append(f'<line x1="{x0}" x2="{x1}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/><text x="{x0-6}" y="{y+4:.1f}" class="tick" text-anchor="end">{fmt(t)}</text>')
    return "".join(out)
def ypos(v, y0, y1, h):
    return h - 28 - (v - y0) / (y1 - y0) * (h - 48)

def chart_coverage():
    h, y0, y1 = 260, 0.85, 1.0
    s = [axis_y(y0, y1, h, [0.85, 0.9, 0.95, 1.0], lambda t: f"{t:.2f}")]
    yt = ypos(0.93, y0, y1, h)
    s.append(f'<line x1="48" x2="{W-12}" y1="{yt:.1f}" y2="{yt:.1f}" class="ref"/><text x="{W-14}" y="{yt-5:.1f}" class="reflab" text-anchor="end">목표 하한 0.93</text>')
    groups = ["Direct", "Auto"]; gw = (W - 80) / 2; bw = min(34, gw / (len(MODELS) + 1))
    for gi, g in enumerate(groups):
        gx = 60 + gi * gw
        s.append(f'<text x="{gx + gw/2 - 10:.1f}" y="{h-8}" class="tick" text-anchor="middle">{g} 클릭베이트</text>')
        for mi, r in enumerate([c for c in cov if c["type"] == g]):
            x = gx + gw / 2 - (len(MODELS) * (bw + 2)) / 2 + mi * (bw + 2) - 10
            yv, yb = ypos(r["v"], y0, y1, h), ypos(y0, y0, y1, h)
            tip = f'{r["model"]} · {g}: {r["v"]:.3f} (95% {r["lo"]:.3f}~{r["hi"]:.3f}, n={r["n"]})'
            s.append(f'<g class="hit" data-tip="{esc(tip)}"><line x1="{x+bw/2:.1f}" x2="{x+bw/2:.1f}" y1="{ypos(r["lo"],y0,y1,h):.1f}" y2="{ypos(min(r["hi"],1),y0,y1,h):.1f}" class="whisk"/>'
                     f'<circle cx="{x+bw/2:.1f}" cy="{yv:.1f}" r="6" class="dot s{r["slot"]}"/><text x="{x+bw/2+9:.1f}" y="{yv+4:.1f}" class="tick">{r["v"]:.3f}</text>'
                     f'<rect x="{x-4:.1f}" y="20" width="{bw+8:.1f}" height="{h-48:.1f}" fill="transparent"/></g>')
    return f'<svg viewBox="0 0 {W} {h}" role="img" aria-label="모델별 클릭베이트 커버리지">{"".join(s)}</svg>'

def chart_sweep():
    h = 260; xs = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30]; y0, y1 = 200, 1000
    X = lambda a: 48 + xs.index(a) / (len(xs) - 1) * (W - 70)
    s = [axis_y(y0, y1, h, [200, 400, 600, 800, 1000], lambda t: f"{int(t)}")]
    for a in xs:
        s.append(f'<text x="{X(a):.1f}" y="{h-8}" class="tick" text-anchor="middle">{a:g}</text>')
    yc = ypos(300, y0, y1, h)
    s.append(f'<line x1="48" x2="{W-12}" y1="{yc:.1f}" y2="{yc:.1f}" class="ref"/><text x="{W-14}" y="{yc-5:.1f}" class="reflab" text-anchor="end">검토 용량 300건/일 (검토자 2명)</text>')
    for key, label, slot in MODELS:
        d = sweep.get(label)
        if d is None: continue
        d = d[d["mode"] == "maxq"].set_index("alpha")
        pts = [(X(a), ypos(d.loc[a, "load_pi0.1"] * 1000, y0, y1, h), a, d.loc[a, "load_pi0.1"] * 1000, int(d.loc[a, "reviewers_pi0.1"])) for a in xs if a in d.index]
        s.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y, *_ in pts)}" class="ln s{slot}"/>')
        for x, y, a, v, r in pts:
            s.append(f'<g class="hit" data-tip="{esc(f"{label} · α={a:g}: 하루 {v:.0f}건, 검토자 {r}명")}"><circle cx="{x:.1f}" cy="{y:.1f}" r="4" class="dot s{slot}"/><circle cx="{x:.1f}" cy="{y:.1f}" r="12" fill="transparent"/></g>')
    return f'<svg viewBox="0 0 {W} {h}" role="img" aria-label="α에 따른 하루 검토 부하">{"".join(s)}</svg>'

def chart_category():
    if cat is None: return "<p>카테고리 결과 없음</p>"
    h = 40 + 30 * len(cat); x0v, x1v = 0.88, 1.0
    X = lambda v: 110 + (v - x0v) / (x1v - x0v) * (W - 130)
    s = []
    for t in [0.88, 0.92, 0.96, 1.0]:
        s.append(f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="10" y2="{h-24}" class="grid"/><text x="{X(t):.1f}" y="{h-8}" class="tick" text-anchor="middle">{t:.2f}</text>')
    s.append(f'<line x1="{X(0.93):.1f}" x2="{X(0.93):.1f}" y1="10" y2="{h-24}" class="ref"/>')
    for i, r in enumerate(cat.sort_values("cov_glob").itertuples()):
        y = 24 + i * 30
        s.append(f'<text x="100" y="{y+4}" class="tick" text-anchor="end">{esc(r.category)}</text><line x1="{X(min(r.cov_glob,r.cov_cat)):.1f}" x2="{X(max(r.cov_glob,r.cov_cat)):.1f}" y1="{y}" y2="{y}" class="conn"/>')
        for v, cls, nm in ((r.cov_glob, "s1", "전역 보정"), (r.cov_cat, "s2", "카테고리 조건부")):
            s.append(f'<g class="hit" data-tip="{esc(f"{r.category} · {nm}: {v:.3f}")}"><circle cx="{X(v):.1f}" cy="{y}" r="5" class="dot {cls}"/><circle cx="{X(v):.1f}" cy="{y}" r="12" fill="transparent"/></g>')
    return f'<svg viewBox="0 0 {W} {h}" role="img" aria-label="카테고리별 커버리지">{"".join(s)}</svg>'

def chart_evidence():
    h = 40 + 46 * len(ev); X = lambda v: 150 + v * (W - 170)
    s = []
    for t in [0, 0.25, 0.5, 0.75, 1.0]:
        s.append(f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="10" y2="{h-24}" class="grid"/><text x="{X(t):.1f}" y="{h-8}" class="tick" text-anchor="middle">{t:.2f}</text>')
    for i, r in enumerate(ev.itertuples()):
        y = 18 + i * 46; lab = f"{r.part} {r.ptype}"
        s.append(f'<text x="140" y="{y+14}" class="tick" text-anchor="end">{esc(lab)}</text>')
        for j, (v, cls, nm) in enumerate(((r.top1, "s1", "유사도 기준선 top-1"), (r.rand_top1, "s0", "무작위 top-1"))):
            yy = y + j * 16
            s.append(f'<g class="hit" data-tip="{esc(f"{lab} · {nm}: {v:.3f}")}"><rect x="{X(0):.1f}" y="{yy}" width="{max(X(v)-X(0),2):.1f}" height="12" rx="4" class="{cls}"/></g>')
    return f'<svg viewBox="0 0 {W} {h}" role="img" aria-label="근거 문장 위치 찾기">{"".join(s)}</svg>'

legend = "".join(f'<span class="lg"><i class="sw s{slot}"></i>{esc(label)}</span>' for _, label, slot in MODELS)
def table(df, cols, names):
    th = "".join(f"<th>{esc(n)}</th>" for n in names)
    trs = "".join("<tr>" + "".join(f"<td>{esc(f'{r[c]:.3f}' if isinstance(r[c], float) else str(r[c]))}</td>" for c in cols) + "</tr>" for _, r in df.iterrows())
    return f'<details><summary>표로 보기</summary><table><tr>{th}</tr>{trs}</table></details>'

covdf = pd.DataFrame(cov)
DQN = {"rows": "전체 기사", "unique_ids": "고유 ID", "dup_ids_across_split": "분할 간 중복 ID", "multi_class_ids": "두 클래스에 속한 ID", "label_missing": "라벨 누락",
       "flag_empty": "제목·본문 공백", "flag_short": "본문 100자 미만", "flag_nonko": "한글 비율 30% 미만", "flag_pii": "개인정보 의심(이메일·전화)", "hold": "판단 보류(입력 이상)"}
dq_html = "<table>" + "".join(f"<tr><td>{esc(DQN.get(str(r.iloc[0]), str(r.iloc[0])))}</td><td>{int(r.iloc[1]):,}</td></tr>" for _, r in dq.iterrows()) + "</table>" if dq is not None else ""
stab_rows = "".join(f"<tr><td>{esc(k)}</td><td>{v['cov_dir']['mean']:.3f}</td><td>{v['cov_aut']['mean']:.3f}</td><td>{v['share_cov_ge_target_dir']*100:.0f}% / {v['share_cov_ge_target_aut']*100:.0f}%</td><td>{v['share_cov_lo_ge_target_both']*100:.0f}%</td></tr>" for k, v in stab.items())
kpis = [("테스트 기사", f"{fs['n_test']:,}건", "AI-Hub 샘플, 층화 추출"),
        ("Direct 커버리지", f"{fe.loc['DIRECT_cb+all_ok','coverage_cb']:.3f}", f"95% 하한 {fe.loc['DIRECT_cb+all_ok','coverage_cb_lo']:.3f} · 목표 0.93"),
        ("Auto 커버리지", f"{fe.loc['AUTO_cb+all_ok','coverage_cb']:.3f}", f"95% 하한 {fe.loc['AUTO_cb+all_ok','coverage_cb_lo']:.3f} · 목표 0.93"),
        ("사람 검토 이관율", f"{fs['refer_rate_test']:.2f}", "목표 0.30 이하"),
        ("필요 검토자(α=0.05)", f"{rev_a05}명" if rev_a05 else "-", "가정 2명 · 하루 1,000건")]
kpi_html = "".join(f'<div class="kpi"><div class="kl">{esc(a)}</div><div class="kv">{esc(b)}</div><div class="ks">{esc(c)}</div></div>' for a, b, c in kpis)

page = f"""<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>레드투그린 클릭베이트 탐지 신뢰성 검증 대시보드</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css">
<style>
:root{{--surface:#fcfcfb;--card:#ffffff;--t1:#0b0b0b;--t2:#52514e;--t3:#8a8984;--grid:#e8e7e3;--s0:#c9c8c2;--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--ref:#52514e}}
@media (prefers-color-scheme:dark){{:root{{--surface:#1a1a19;--card:#222220;--t1:#f0efec;--t2:#c3c2b7;--t3:#8f8e86;--grid:#33332f;--s0:#55554f;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--ref:#c3c2b7}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--surface);color:var(--t1);font-family:Pretendard,system-ui,sans-serif}}
header{{padding:28px 32px 8px}}h1{{font-size:22px;margin:0 0 6px}}.sub{{color:var(--t2);font-size:13px}}
.kpis{{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;padding:16px 32px}}.kpi{{background:var(--card);border:1px solid var(--grid);border-radius:10px;padding:14px}}
.kl{{font-size:12px;color:var(--t2)}}.kv{{font-size:26px;font-weight:700;margin:4px 0}}.ks{{font-size:11px;color:var(--t3)}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:16px;padding:8px 32px 32px}}.card{{background:var(--card);border:1px solid var(--grid);border-radius:10px;padding:16px 18px}}
h2{{font-size:15px;margin:0 0 4px}}.note{{font-size:12px;color:var(--t2);margin:0 0 8px;line-height:1.5}}
svg{{width:100%;height:auto;display:block}}.grid{{stroke:var(--grid);stroke-width:1}}.tick{{fill:var(--t2);font-size:11px}}
.ref{{stroke:var(--ref);stroke-width:1;stroke-dasharray:4 3}}.reflab{{fill:var(--t2);font-size:11px}}.whisk{{stroke:var(--t1);stroke-width:1.5}}
.s0{{fill:var(--s0)}}.s1{{fill:var(--s1)}}.s2{{fill:var(--s2)}}.s3{{fill:var(--s3)}}
.ln{{fill:none;stroke-width:2}}.ln.s1{{stroke:var(--s1)}}.ln.s2{{stroke:var(--s2)}}.ln.s3{{stroke:var(--s3)}}.dot{{stroke:var(--card);stroke-width:2}}.conn{{stroke:var(--t3);stroke-width:1.5}}
.lg{{display:inline-flex;align-items:center;gap:6px;margin-right:14px;font-size:12px;color:var(--t2)}}.sw{{width:10px;height:10px;border-radius:3px;display:inline-block}}
.sw.s0{{background:var(--s0)}}.sw.s1{{background:var(--s1)}}.sw.s2{{background:var(--s2)}}.sw.s3{{background:var(--s3)}}
details{{margin-top:8px;font-size:12px}}table{{border-collapse:collapse;width:100%;margin-top:6px}}td,th{{border-bottom:1px solid var(--grid);padding:4px 6px;text-align:left}}th{{color:var(--t2);font-weight:600}}
#tip{{position:fixed;pointer-events:none;background:var(--t1);color:var(--surface);font-size:12px;padding:6px 8px;border-radius:6px;opacity:0;transition:opacity .1s}}
.hit{{cursor:default}}footer{{padding:0 32px 32px;color:var(--t3);font-size:11px;line-height:1.6}}
@media (max-width:900px){{.kpis{{grid-template-columns:repeat(2,1fr)}}.grid2{{grid-template-columns:1fr}}}}
</style></head><body>
<header><h1>클릭베이트 탐지 Agent 신뢰성 검증 대시보드</h1>
<div class="sub">팀 레드투그린 · 제2회 TRAITHON 본선 · 클래스 조건부 conformal 보정(α=0.05, 수정안) · KPI 기준 모델: {esc(first[1])}</div></header>
<section class="kpis">{kpi_html}</section>
<section class="grid2">
<div class="card"><h2>클릭베이트를 사람 검토 없이 통과시키지 않는 비율</h2><p class="note">점은 테스트 셋 커버리지, 세로선은 Clopper–Pearson 95% 구간입니다. 운영 중 생성 유형을 알 수 없어 두 유형 중 더 엄격한 기준으로 보정합니다.</p>
<div>{legend}</div>{chart_coverage()}{table(covdf, ["model","type","v","lo","hi","n"], ["모델","유형","커버리지","하한","상한","n"])}</div>
<div class="card"><h2>놓침 허용률 α에 따른 하루 검토 부하</h2><p class="note">하루 1,000건, 유병률 10%, 정치·경제 전량 사람 검토를 가정했습니다. 정치·경제만으로 278건이라 어떤 α에서도 검토자 2명 용량을 넘습니다.</p>
<div>{legend}</div>{chart_sweep()}</div>
<div class="card"><h2>카테고리별 커버리지: 전역 보정 대 카테고리 조건부 보정</h2><p class="note">카테고리별 보정 표본 400건(보장 확률 0.958)으로 다시 계산하면 개선 효과가 없습니다. 연예가 가장 낮아 별도 위험으로 관리합니다.</p>
<div><span class="lg"><i class="sw s1"></i>전역 보정</span><span class="lg"><i class="sw s2"></i>카테고리 조건부</span></div>{chart_category()}
{table(cat, ["category","cov_glob","cov_cat","refer_glob","refer_cat"], ["카테고리","전역","카테고리 조건부","이관율(전역)","이관율(조건부)"]) if cat is not None else ""}</div>
<div class="card"><h2>근거 문장을 얼마나 정확히 짚는가</h2><p class="note">검토 화면에 보여 줄 근거 문장 후보의 첫 번째가 정답 문장인 비율입니다. A.X가 인용하는 문장은 이 기준선보다 높아야 합니다.</p>
<div><span class="lg"><i class="sw s1"></i>유사도 기준선</span><span class="lg"><i class="sw s0"></i>무작위</span></div>{chart_evidence()}
{table(ev, ["part","ptype","n","top1","top3","rand_top1"], ["Part","유형","n","top-1","top-3","무작위 top-1"])}</div>
<div class="card"><h2>robustness: 보정·테스트 200회 재추출</h2><p class="note">점추정은 이론대로 95% 이상 목표를 넘지만, 테스트 Direct 표본이 344건이라 구간 하한 기준은 약 60%만 통과합니다. 테스트 셋 확대가 필요합니다.</p>
<table><tr><th>모델</th><th>Direct 평균</th><th>Auto 평균</th><th>점추정 0.93 이상</th><th>구간 하한 동시 통과</th></tr>{stab_rows}</table></div>
<div class="card"><h2>데이터 품질</h2><p class="note">AI-Hub 낚시성 기사 탐지 샘플 267,955건, Webis 19,538건. 개인정보 의심 57건과 입력 이상 3건을 제외했습니다.</p>
{dq_html}</div>
</section>
<footer>데이터 출처: 한국지능정보사회진흥원 AI-Hub 「낚시성 기사 탐지 데이터」(본 결과는 AI-Hub 사업 결과물을 활용함), Webis Clickbait Corpus 2017(CC BY 4.0). 기사 원문은 포함하지 않으며 집계값만 표시합니다. 재현 코드: github.com/Chaewon-Park-STUDY/Traithon2026 src/datascience</footer>
<div id="tip"></div>
<script>const tip=document.getElementById('tip');document.querySelectorAll('.hit').forEach(g=>{{g.addEventListener('mousemove',e=>{{tip.textContent=g.dataset.tip;tip.style.left=(e.clientX+12)+'px';tip.style.top=(e.clientY+12)+'px';tip.style.opacity=1}});g.addEventListener('mouseleave',()=>tip.style.opacity=0)}});</script>
</body></html>"""
open(os.path.join(OUT, "dashboard.html"), "w", encoding="utf-8").write(page)
print("dashboard.html", len(page), "models", [m[1] for m in MODELS])
