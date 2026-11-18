# d43: hold 임계값 운영 기준 (answer_mass < 0.95 → 사람)

상태: 사후 분석(결과를 본 뒤 운영 기준으로 설정). 집계 수치만 포함.
재현: `python d43_hold_analysis.py <ax_ft_scores.csv> <split_manifest.csv> <d41_scores.csv> <holdout_meta.csv> out` (입력은 로컬 전용)

## 결정
기존 hold 기준 answer_mass < 0.5는 정상 입력에서 한 번도 걸리지 않았다(main_test, holdout, webis, calib 모두 0건. 정상 입력의 answer_mass 최솟값 약 0.90).
운영 기준을 **answer_mass < 0.95 → 사람에게 이관**으로 바꾼다. 이유는 설명이 쉽고(95% 미만 확신이면 사람) 사람 부담이 작기 때문이다.
참고: main_calib 1,080건으로 conformal risk control(자동 판정 오판 ≤ 1%)을 적용하면 τ≈0.968이 나오며, 이는 0.97 선택지를 뒷받침한다.

## 결과 (비민감 입력, 정치·경제 제외)

| 집합 | τ | 사람이 보는 비율 | 자동 판정 중 오판 (95% CI) |
|---|---|---|---|
| main_test (n=2,325) | 0.5 (기존) | 1.1% | 2.74% (2.11–3.49) |
| main_test | **0.95** | 7.0% | **1.48% (1.01–2.08)** |
| main_test | 0.97 | 10.2% | 0.91% (0.55–1.42) |
| d41 holdout (n=4,279) | 0.5 (기존) | 1.4% | 3.53% (2.99–4.13) |
| d41 holdout | **0.95** | 9.2% | **1.21% (0.89–1.61)** |
| d41 holdout | 0.97 | 11.6% | 0.66% (0.43–0.97) |

answer_mass로 오류를 구분하는 AUC: main_test 0.938, holdout 0.950.
기존 ambiguous(1%대)는 낮은 answer_mass와 거의 겹쳐, τ를 올리면 사람 몫이 단순 합산으로 늘지 않고 흡수된다.

## 주의
- CI는 기사 단위 Clopper–Pearson이며 입력이 기사 단위라는 전제다. 인젝션 변형 데이터에는 적용하지 않는다.
- holdout은 clickbait 2/3로 구성된 표본이라 실서비스 비율과 다르다.
- calibration(clickbait 50%)과 test(58.6%)의 비율이 달라 exchangeability가 완전하지 않다.
- 형식 위반 출력의 hold 처리율(MDL-03 (d))은 이 분석에 포함되지 않았다.
