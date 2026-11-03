"""Traceability Matrix(제출물 3번) 데이터과학 부분: BaseKit ID ↔ 시험·산출물 ↔ 결과 ↔ 상태.
수치는 outputs/*.json 에서 읽어 채운다. 산출: outputs/traceability_datascience.xlsx / .csv
"""
import os, json
import pandas as pd
from common import OUT

J = lambda f: json.load(open(os.path.join(OUT, f), encoding="utf-8")) if os.path.exists(os.path.join(OUT, f)) else {}
eb, ebq, ef = J("eval_summary_baseline.json"), J("eval_summary_baseline_maxq.json"), J("eval_summary_feat_maxq.json")
evb = pd.read_csv(os.path.join(OUT, "eval_baseline.csv")).set_index("slice"); evq = pd.read_csv(os.path.join(OUT, "eval_baseline_maxq.csv")).set_index("slice")
st = J("stability_summary_baseline.json"); dq = J("dq_completeness.json"); sens = J("sensitivity_summary_baseline.json")
ev = pd.read_csv(os.path.join(OUT, "evidence_summary.csv"))
f3 = lambda x: f"{x:.3f}"
rows = [
 ("DATA-01·02", "AI-Hub 낚시성 기사 데이터 정제·분할", "RISK-13, RISK-10", "d2_build_main.py, d4_split.py", "dq_metrics_summary.csv, split_summary.json",
  "샘플 데이터 267,955건, ID 중복 0, 라벨 누락 0, 분할 간 중복 0. 보정 클래스별 540, 테스트 3,232", "완료(샘플 기준). 정식 660,085건 공식 분할로 재실행 필요"),
 ("DATA-03", "Webis 보조 데이터", "RISK-10", "fetch_webis.py, d2_build_webis.py, d8_webis_eval.py", "webis_eval_summary_en_tfidf.json",
  "19,538건(클릭베이트 24.4%), 영어 기준선 확률-사람 평균 Spearman 0.585", "부분 완료. A.X 점수로 재평가 필요"),
 ("DQ-01 완전성", "제목·본문 결측", "RISK-05", "d2_build_main.py, d3_dq_extra.py", "dq_completeness.json",
  f"제목 공백 {dq.get('title_blank')}, 본문 100자 미만 {dq.get('body_short_lt100')}, 본문 중앙값 {dq.get('body_len_median')}자", "완료"),
 ("DQ-02 라벨 품질", "라벨 재검수", "RISK-13", "d12_label_audit_sample.py", "label_audit_ids.csv, LABEL_AUDIT_GUIDE.md",
  "표본 397건 추출(필요 385)", "진행 필요(사람 검수)"),
 ("DQ-04 입력 이상", "한글 비율·길이 이상 판단 보류", "RISK-05", "d2_build_main.py", "dq_metrics_summary.csv", "한글 비율 낮음 3건 제외", "완료"),
 ("DQ-05 개인정보", "이메일·전화번호 정규식 스캔", "RISK-05", "d2_build_main.py", "dq_metrics_summary.csv", "57건 검출, 분석·학습에서 100% 제외", "완료"),
 ("§4-3 근거 정답 라벨", "근거 표시 건수", "MDL-01 근거", "d3_dq_extra.py", "dq_evidence_lengths.csv",
  "Part1 Direct·Part2 클릭베이트 100% 표시, Part1 Auto·정상 0%", "완료"),
 ("§4-5 표본 크기", "보정·테스트 최소 크기", "MDL-01", "common.py", "split_summary.json",
  "보정 360(표와 일치), 테스트 CP 계산 506·88 < 표 589·127 → 표 값 사용", "완료. 표 계산 방식 문구 정정 필요"),
 ("MDL-01 (1) 커버리지", "클래스 조건부 conformal, α=0.05", "IMPACT-01, RISK-01, RISK-13", "d6_calibrate.py, d7_evaluate.py", "eval_baseline*.csv",
  f"TF-IDF 원안 Direct {f3(evb.loc['DIRECT_cb+all_ok','coverage_cb'])}·Auto {f3(evb.loc['AUTO_cb+all_ok','coverage_cb'])}(하한 {f3(evb.loc['AUTO_cb+all_ok','coverage_cb_lo'])} 미달) → 수정안 Auto {f3(evq.loc['AUTO_cb+all_ok','coverage_cb'])}", "기준선 완료, A.X 대기"),
 ("RISK-13 생성 유형", "Direct 보정이 보수적인가", "RISK-13", "d7_evaluate.py, d5b_feature_scores.py", "RESULTS_baseline_2026-11-02.md",
  "어려운 유형이 모델마다 다름(TF-IDF는 Auto, 정합성 특성 모델은 Direct) → q̂=max(Direct, Auto) 제안", "완료. BaseKit 5 §5-4 수정 필요"),
 ("robustness 반복 분할", "보정·테스트 200회 재추출", "RISK-13, RISK-11", "d13_stability.py", "stability_summary_*.json",
  f"TF-IDF Direct 평균 {f3(st['cov_dir']['mean'])}·Auto 평균 {f3(st['cov_aut']['mean'])}, 점추정 0.93 이상 비율 {st['share_cov_ge_target_dir']:.2f}/{st['share_cov_ge_target_aut']:.2f}, CP 하한 기준 동시 통과 {st['share_cov_lo_ge_target_both']:.2f}" if st else "", "완료"),
 ("MDL-01 집단 비교", "정치·경제 대 나머지 차이 ≤ 0.05", "RISK-04, IMPACT-06", "d7_evaluate.py, d9_sensitivity.py, d14_category_calib.py", "eval_*.csv, category_calib_v2.csv",
  "정치·경제 차이는 0.05 이내, 카테고리 간 최대 차이 0.07", "완료(카테고리 조건부 v2 비교 포함)"),
 ("IMPACT-03 / A4 검토 용량", "이관율·검토 부하", "IMPACT-03, HO-01, HO-02", "d9_sensitivity.py", "alpha_sweep_*.csv, sensitivity_summary_*.json",
  f"이관율 {ebq.get('refer_rate_test',0):.2f}(목표 0.30), 정치·경제 이관만 용량 93%, 완벽 모델도 검토자 3~4명 필요", "완료. BaseKit 1·6 인력 가정 수정 필요"),
 ("§5-5 근거 인용", "근거 문장 위치 찾기 기준선", "MDL-01, HO 검토 화면", "d10_evidence.py", "evidence_summary.csv",
  "; ".join(f"{r.part} {r.ptype} top-1 {r.top1:.2f}(무작위 {r.rand_top1:.2f})" for r in ev.itertuples()), "기준선 완료, A.X 인용 평가 대기"),
 ("MDL-03 / RISK-08 인젝션", "본문 지시문 삽입 시험", "RISK-08", "d11_injection_set.py, colab_score_ax.py", "injection_spec.csv",
  "200건 → 1,200건(5유형×3위치)", "세트 완료, A.X 실행 중"),
 ("MDL-01 A.X 4.0 Light", "현재 모델 테스트셋 검증(GPU 지원 조건)", "MDL-01 전체", "colab_score_ax.py", "ax_scores.csv",
  "Colab T4, 4비트 NF4", "실행 중"),
]
df = pd.DataFrame(rows, columns=["BaseKit ID", "시험·점검 항목", "연결 위험·영향", "실행 코드", "근거 파일", "결과", "상태"])
df.to_csv(os.path.join(OUT, "traceability_datascience.csv"), index=False, encoding="utf-8-sig")
with pd.ExcelWriter(os.path.join(OUT, "traceability_datascience.xlsx"), engine="openpyxl") as w:
    df.to_excel(w, index=False, sheet_name="데이터과학")
    ws = w.sheets["데이터과학"]
    for col, wd in zip("ABCDEFG", [22, 26, 22, 30, 32, 70, 34]):
        ws.column_dimensions[col].width = wd
    from openpyxl.styles import Alignment, Font
    for row in ws.iter_rows():
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
    for c in ws[1]:
        c.font = Font(bold=True)
print(len(df), "rows")
