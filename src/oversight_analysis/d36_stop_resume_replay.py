"""d36: 용량 초과 대응 규칙(5-B6)의 중단·재개 재생 시연과 테스트 표.
사용: python -X utf8 d36_stop_resume_replay.py [demo_app.py 경로] [출력폴더]
표준 라이브러리만 사용. 모의 재생(simulated replay)이며 실제 운영 데이터가 아님.
"""
import sys, re, csv, os

CAP_PER, LOAD_PER = 150, 0.376      # demo_app.py와 동일 (A4 가정, 실측 부하)
WARN_U, HOLD_U = 0.9, 1.0           # 5-B6: 90% 초과 알림, 100% 초과 보류
WARN_STRICT, HOLD_STRICT = True, True   # '넘으면' = 초과(>). drift check에서 코드와 비교
STOP_DAYS, STOP_WAIT = 3, 24        # 3일 연속 초과, 24시간 이상 대기
STAGE = {0: "정상", 1: "경고", 2: "게시 보류", 3: "자동 처리 중지"}

def util(vol, rev):
    return vol * LOAD_PER / (rev * CAP_PER)

def stage(u, days_over, max_wait):
    """5-B6 단계 판정 (중지는 호출 측에서 latch)."""
    if days_over >= STOP_DAYS or max_wait >= STOP_WAIT: return 3
    if (u > HOLD_U) if HOLD_STRICT else (u >= HOLD_U): return 2
    if (u > WARN_U) if WARN_STRICT else (u >= WARN_U): return 1
    return 0

def can_resume(approved, u, max_wait):
    """팀 제안 재개 조건 R: 승인 기록 + 사용률 <= 90% + 24시간 넘은 대기 없음."""
    return bool(approved) and u <= WARN_U and max_wait < STOP_WAIT

class Ops:
    def __init__(self): self.stopped = False; self.days_over = 0; self.log = []
    def day(self, d, vol, rev, max_wait, approved=False, note=""):
        u = util(vol, rev)
        self.days_over = self.days_over + 1 if u > HOLD_U else 0
        st = stage(u, self.days_over, max_wait)
        actor, action = "시스템", ""
        if self.stopped:
            if can_resume(approved, u, max_wait):
                self.stopped = False; actor = "운영 책임자"
                action = "재개 승인 기록 · 자동 처리 재개 · 단계 재평가"
                st = stage(u, self.days_over, max_wait)
            else:
                st = 3
                action = "중지 유지 (재개 조건 미충족)" if approved else "중지 유지 (승인 없음)"
        else:
            if st == 3:
                self.stopped = True
                action = "신규 기사 자동 처리 중지 · 전량 검토자 확인 전환 · 인력 증원 요청"
            elif st == 2: action = "확인 큐 기사 게시 보류 (자동 통과 확대 금지)"
            elif st == 1: action = "운영 책임자에게 사전 통보"
            else: action = "정상 운영"
        self.log.append(dict(day=d, volume=vol, reviewers=rev, utilization=f"{u:.1%}",
            days_over=self.days_over, max_wait_h=max_wait, stage=STAGE[st],
            actor=actor, approved=int(bool(approved)), action=action, note=note))
        return st

def scenario():
    o = Ops()
    o.day(1, 1000, 3, 4, note="영상 장면: 검토자 3명")
    o.day(2, 750, 2, 6, note="영상 장면: 750건/2명")
    o.day(3, 1000, 2, 10, note="영상 장면: 1,000건/2명, 초과 1일째")
    o.day(4, 1000, 2, 16, note="초과 2일째")
    o.day(5, 1000, 2, 20, note="초과 3일째 → 중지")
    o.day(6, 1000, 2, 20, approved=False, note="승인 없이 대기: 중지 유지 확인")
    o.day(7, 1000, 2, 20, approved=True, note="승인했으나 증원 없음: 재개 조건 미충족 확인")
    o.day(8, 1000, 4, 6, approved=True, note="검토자 4명으로 증원 후 승인 → 재개")
    o.day(9, 1000, 4, 5, note="정상 복귀 확인")
    return o.log

def tests():
    T = []
    def add(tid, desc, inp, exp, act, basis):
        kind = "정상" if tid in ("T-01","T-02","T-03") else ("중단 후 재개" if tid >= "T-09" else "경계·실패")
        T.append(dict(test_id=tid, scenario_type=kind, description=desc, input=inp, expected=exp, actual=act,
                      acceptance="기대와 실제 일치(PASS)", result="PASS" if exp == act else "FAIL", basis=basis,
                      evidence_id="EVD-06-4.9-(번호 부여 필요)"))
    S = lambda *a: STAGE[stage(*a)]
    add("T-01", "정상 구간", "1000건/3명, 연속0일, 대기4h", "정상", S(util(1000,3),0,4), "5-B6 (1)(2)")
    add("T-02", "경고 구간(영상 장면)", "750건/2명", "경고", S(util(750,2),0,6), "5-B6 (2)")
    add("T-03", "게시 보류(영상 장면)", "1000건/2명", "게시 보류", S(util(1000,2),1,10), "5-B6 (3)")
    add("T-04", "3일 연속 초과 중지", "연속초과3일", "자동 처리 중지", S(util(1000,2),3,10), "5-B6 (4)")
    add("T-05", "대기 24h 중지 경계", "대기24h", "자동 처리 중지", S(0.5,0,24), "5-B6 (4)")
    add("T-06", "대기 23h는 중지 아님", "대기23h", "정상", S(0.5,0,23), "5-B6 (4) 경계")
    add("T-07", "사용률 정확히 90%", "u=0.9", "정상", S(0.9,0,0), "5-B6 (2) '넘으면'")
    add("T-08", "사용률 정확히 100%", "u=1.0", "경고", S(1.0,0,0), "5-B6 (3) '넘으면'")
    add("T-09", "중지 중 승인 없음", "stopped, approved=0", "False", str(can_resume(False,0.5,0)), "제안 재개 조건 R(a)")
    add("T-10", "승인했으나 사용률 높음", "approved=1, u=1.25", "False", str(can_resume(True,1.25,0)), "제안 재개 조건 R(b)")
    add("T-11", "승인했으나 24h 대기 잔존", "approved=1, 대기24h", "False", str(can_resume(True,0.5,24)), "제안 재개 조건 R(c)")
    add("T-12", "조건 충족 재개", "approved=1, u<=0.9, 대기<24h", "True", str(can_resume(True,0.63,6)), "제안 재개 조건 R")
    log = scenario()
    add("T-13", "시나리오 5일차 중지", "시나리오", "자동 처리 중지", log[4]["stage"], "시나리오")
    add("T-14", "시나리오 6~7일차 중지 유지", "시나리오", "자동 처리 중지|자동 처리 중지", log[5]["stage"]+"|"+log[6]["stage"], "latch")
    add("T-15", "시나리오 8일차 재개 후 단계", "시나리오", "정상", log[7]["stage"], "재개 후 재평가")
    return T

def drift_check(path):
    """demo_app.py의 상수·임계값이 이 스크립트와 같은지 확인(출력만, 수정 안 함)."""
    if not path or not os.path.exists(path): print("[drift] demo_app.py 경로 없음: 건너뜀"); return
    src = open(path, encoding="utf-8").read()
    m = re.search(r"CAP_PER\s*,\s*LOAD_PER\s*=\s*([\d.]+)\s*,\s*([\d.]+)", src)
    print("[drift] CAP_PER, LOAD_PER =", m.groups() if m else "찾지 못함", "| 이 스크립트:", CAP_PER, LOAD_PER)
    for pat in [r"days_over\s*>=\s*(\d+)", r"max_wait\s*>=\s*(\d+)", r"u\s*>=?\s*([\d.]+)"]:
        print("[drift]", pat, "->", re.findall(pat, src))
    print("[drift] 코드의 u 비교 연산자(>= 또는 >)와 이 스크립트(WARN_STRICT/HOLD_STRICT)를 대조하세요.")

def write(path, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

if __name__ == "__main__":
    demo = sys.argv[1] if len(sys.argv) > 1 else None
    out = sys.argv[2] if len(sys.argv) > 2 else "d36_out"
    os.makedirs(out, exist_ok=True)
    drift_check(demo)
    L, T = scenario(), tests()
    write(os.path.join(out, "d36_events.csv"), L); write(os.path.join(out, "d36_tests.csv"), T)
    print("\n== 이벤트 로그 (모의 재생) ==")
    for r in L: print(f"D{r['day']} {r['volume']}건/{r['reviewers']}명 u={r['utilization']:>6} 초과{r['days_over']}일 대기{r['max_wait_h']}h -> {r['stage']} | {r['actor']}: {r['action']}")
    print("\n== 테스트 ==")
    for t in T: print(t["test_id"], t["result"], t["description"], "| 기대:", t["expected"], "실제:", t["actual"])
    print("\nFAIL 수:", sum(t["result"] == "FAIL" for t in T), "/", len(T))
