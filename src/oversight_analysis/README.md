# oversight_analysis: 검토 용량·불확실성 분석 (박채원)

`src/datascience/results/`의 **집계 결과만** 입력으로 쓴다. 행 단위 점수나 기사 원문은 쓰지 않으며 레포에도 없다.

| 파일 | 내용 | 입력 |
|---|---|---|
| `load_interval.py` | 검토 부하와 미포착 건수의 Clopper–Pearson 95% 구간, 필요한 1인 일 처리량 | 코드 안 상수(a+ 42/1370, a− 914/955, s=0.278) |
| `mdp_alpha.py`, `run_mdp.py` | 대기량에 따라 α를 고르는 소형 MDP(value iteration) | `results/alpha_sweep_axft.csv`의 maxq 행 |
| `d22_ic_sim.py` | interval-censored 발견 시간의 불확실성 폭 시뮬레이션 | 없음(시나리오 값) |

실행: 각 파일이 있는 폴더에서 `python load_interval.py`, `python run_mdp.py`(약 2분), `python d22_ic_sim.py`.

## 읽을 때 주의
- N=1,000건/일, π(0.05/0.10/0.25), 검토 인력 R, 오류 비용 κ는 **데이터가 아니라 시나리오 값**이다. 데이터에서 나온 값은 a+, a− 와 그 구간뿐이다.
- MDP에서 검토 용량이 최소 부하보다 작으면(예: C=300, π=0.10에서 부하 상한 386건) 대기량이 상한(BMAX)에 붙어 정책 출력이 의미가 없다. 이 경우 결론은 "안정 정책 없음"이다.
- α sweep은 표준 split conformal과 달리 α가 클수록 예측집합이 비는 경우가 늘고, 빈 집합과 두 라벨 모두 포함은 애매로 이관한다. 문서에 명시해야 한다.
- `d22_ic_sim.py`의 Weibull 모수는 추정치가 아니다. 현재 데이터에는 시각 필드가 없어 발견 시간을 추정할 수 없다(AI-Hub 샘플 JSON 확인).
