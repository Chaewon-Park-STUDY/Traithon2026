# 레드투그린 본선 데이터과학 파이프라인 (담당: 박소이, 데이터과학자)

BaseKit 4 §4-3~4-7, BaseKit 5 §5-4·MDL-01, BaseKit 7 §7-5 절차를 그대로 코드로 옮겼다. 원본 데이터(`data_raw/`)와 기사 원문이 들어 있는 산출물(`*.parquet`, 행 단위 점수·판정 CSV)은 저장소에 올리지 않는다. AI-Hub 약관상 승인받지 않은 사람에게 데이터를 열람·제공할 수 없으므로 **팀원도 각자 AI-Hub에서 신청**해 `data_raw/aihub/`에 넣고 실행한다.

저장소에는 코드(`scripts/`)와 집계 결과(`results/`)만 있다. `results/split_ids.csv`는 분할 재현용 기사 ID 목록이다. 스크립트를 실행하면 `src/datascience/data_raw/`, `src/datascience/outputs/`가 생기며 둘 다 커밋 대상이 아니다.

## 폴더
- `data_raw/aihub/` : AI-Hub 낚시성 기사(71338) **샘플(경량) 데이터** zip(1.13GB, 라벨링 JSON 267,955건). 2026-11-02 박소이 계정 승인(2027-02-02까지). 정식 데이터는 미다운로드
- `data_raw/webis/` : Webis Clickbait 17 train(170630)의 instances.jsonl·truth.jsonl (19,538건, 이미지 제외 원격 추출, 2026-11-02)
- `scripts/` : 단계별 스크립트, `outputs/` : 산출물

## 실행 순서
| 단계 | 명령 | 산출물 |
|---|---|---|
| D1~D3 정제 | `python -X utf8 scripts/d2_build_main.py` | clean_main.parquet, dq_metrics_summary.csv, dq_category_distribution.csv |
| Webis 정제 | `python -X utf8 scripts/d2_build_webis.py` | clean_webis.parquet (완료) |
| D3·D4 분할 | `python -X utf8 scripts/d4_split.py --scale 1.5` | split_manifest.csv, slice_registry.csv, split_summary.json |
| 기준선 점수 | `python -X utf8 scripts/d5_baseline_scores.py` | scores_baseline.csv |
| A.X 점수(GPU) | Colab(T4 무료, 4비트)에 `colab/colab_score_ax.py`와 입력 parquet를 올려 실행 → 받은 `ax_scores.csv`를 `python scripts/d17_ax_import.py <경로>`로 가져오면 보정·평가·인젝션 지표까지 한 번에 계산 | scores_ax.csv, injection_summary_ax.json |
| 안정성 | `python scripts/d13_stability.py <name> 200` | stability_summary_<name>.json |
| 카테고리 보정 v2 | `python scripts/d14_category_calib.py` | category_calib_v2.csv |
| 추적표·대시보드 | `d15_traceability.py`, `d16_dashboard.py` | traceability_datascience.xlsx, dashboard.html |
| D6 보정 | `python -X utf8 scripts/d6_calibrate.py ax` | calibration_ax.json |
| D7 평가 | `python -X utf8 scripts/d7_evaluate.py ax` | eval_ax.csv, eval_summary_ax.json, test_decisions_ax.csv |
| D8 Webis | `python -X utf8 scripts/d8_webis_eval.py ax` | webis_eval_summary_ax.json |

`ax` 자리에 `baseline`을 넣으면 같은 절차를 기준선 점수로 돌린다.

## 확인된 사항
- 보정 셋 최소 크기 계산이 BaseKit 4 §4-5 표와 일치(α0.05·ε0.02·δ0.05 → 360 등).
- 테스트 셋 표(589, 127)는 Clopper–Pearson 반폭 계산(506, 88)보다 크다. 코드는 둘 중 큰 값을 쓴다. BaseKit 본선본에서 계산 방식 문구를 맞출 것.
- 가짜 AI-Hub 형식 데이터 13,000건으로 전 단계 실행 확인. 보정 후 커버리지가 0.95 근처로 나와 conformal 계산 정상.
- Webis 영어 TF-IDF 기준선(5겹 교차검증): truthMean과 Spearman 0.585, ECE 0.048. 판단자 불일치 상위 25% 글은 확률이 0.5 쪽으로 모임.

## 첫 실데이터 결과
- `outputs/RESULTS_baseline_2026-11-02.md` 참고. 보정 수정안은 `d6_calibrate.py <name> --maxq`, 평가는 `d7_evaluate.py <name>_maxq`.

## 실제 데이터에서 다시 확인할 것
- AI-Hub JSON 필드 위치(코드는 키 이름으로 찾음: newsID, newsCategory, newsTitle, newsContent, partNum, processType, processPattern, processLevel, clickbaitClass, newTitle, referSentenceInfo, processSentenceInfo). 첫 실행 후 dq_metrics_summary.csv의 label_missing·flag 수와 660,085건 일치 여부를 본다.
- 운영팀 GPU는 받지 못해 Colab 무료 T4에서 4비트(NF4)로 실행한다(팀 결정, 2026-11-03). 16비트 원 모델과 확률이 조금 다를 수 있다. 16건 시험에서 답 토큰 확률 합 0.998 이상으로 형식은 정상.
- Colab 세션 파일은 런타임이 끝나면 지워진다. 결과 CSV는 받은 즉시 내려받는다.
