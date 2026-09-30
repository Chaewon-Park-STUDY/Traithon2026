# TRAITHON 2026 - 팀 프로젝트

제2회 TRAITHON(AI 신뢰성 해커톤) 참가 팀 레포지토리

## 프로젝트 개요
- 주제: 클릭베이트 탐지 AI + 인간감독 기반 신뢰성 산출물
- 필수 모델: A.X 4.0 Light

## 사용 모델
- **필수**: [A.X 4.0 Light](https://huggingface.co/skt/A.X-4.0-Light) (SKT 공개 모델, Hugging Face)
  - 필수 사용이나 이 모델만 써야 하는 것은 아님 — 팀 필요에 따라 다른 모델 추가 병행 가능 (대회 공지 기준)
- **보조 모델**: 필요 시 추가 예정 (팀 논의 후 확정)

## 사용 데이터
메인·보조 데이터를 모두 활용하는 것이 기본입니다.

**① 메인 데이터: AI-Hub 낚시성 기사 탐지 데이터**
- 언어: 한국어
- 다운로드: [AI-Hub 낚시성 기사 탐지](https://aihub.or.kr/aihubdata/data/view.do?pageIndex=1&currMenu=115&topMenu=100&srchOptnCnd=OPTNCND001&searchKeyword=%EB%82%9A%EC%8B%9C&srchDetailCnd=DETAILCND001&srchOrder=ORDER001&srchPagePer=20&aihubDataSe=data&dataSetSn=71338)
- 용도: 한국어 클릭베이트 탐지, 제목-본문 불일치, 과장·정보 은닉 등 위험 유형 분석

**② 보조 데이터: Webis Clickbait Corpus 2017**
- 언어: 영어
- 사이트: [Webis Clickbait Challenge](https://webis.de/events/clickbait-challenge/shared-task.html)
- 다운로드: [Zenodo](https://zenodo.org/records/5530410) — `clickbait17-train-170630.zip` 사용 (Media 폴더 내 이미지는 제외)
- 용도: 클릭베이트 강도, AI-사람 판단 비교, 인간 검토 기준 설정

**데이터 활용 기준**
- 정제·가공·증강, 주석 추가 가능
- 언어·구조가 달라 단순 병합보다 목적별 구분 활용 권장
- 제공 데이터 없이 추가 데이터만으로 수행 불가
- 추가 데이터 필요 시: 필요 이유 / 기존 데이터의 부족한 점 / 어떤 신뢰성 문제를 보완하는지 설명 가능해야 함 + 사전에 멘토·운영팀 확인 권장

## 참고 자료
- BaseKit 샘플, 제1회 TRAITHON 참가팀 사례·결과집 등 원본은 대회 측 Google Drive 링크로 제공됨 (팀 내 공유, 대용량 파일은 레포에 직접 올리지 않음)
- `docs/` 폴더에는 팀이 직접 작성한 BaseKit 문서와 회의록만 정리

## 폴더 구조 (예시)
```
.
├── README.md
├── docs/                # BaseKit(BK01~07) 작성본, 회의록
├── progress/            # 교육 진도 체크 등
│   └── traithon-progress-tracker.md
├── notebooks/           # 데이터 탐색, 실험 노트북
├── src/                 # 모델/전처리 코드
└── data/                # 데이터 (원본은 커밋하지 않음, .gitignore 참고)
```

## 일정
- 1주차 (~9/20): 서비스 정의, 시나리오 정리
- 2주차 (~9/27): 이해관계자·영향 분석, CP2
- 3주차 (~10/4): 위험 등록부·통제, 초기 아키텍처, CP3
- 4주차 (~10/11): 운영·인간감독 계획, 예선 제출(BaseKit 1~7), CP4

## 진도 체크
`progress/traithon-progress-tracker.md` 참고
