#!/usr/bin/env python3
"""진도표에서 한 팀원의 강의 범위를 ✅(또는 ⬜)로 바꾼다.

사용법: python scripts/mark.py 진도표경로 이름 시작번호 끝번호 [check|uncheck]
예시:   python scripts/mark.py 교육진도/traithon-progress-tracker.md 박채원 3 3.5
시작번호와 끝번호도 범위에 포함된다. 파일에 적힌 순서를 기준으로 범위를 정한다.
"""
import sys

DONE, TODO = "✅", "⬜"


def main():
    if len(sys.argv) < 5:
        sys.exit(__doc__)
    path, member, start, end = sys.argv[1:5]
    mode = sys.argv[5] if len(sys.argv) > 5 else "check"
    if mode not in ("check", "uncheck"):
        sys.exit("모드는 check 또는 uncheck 이어야 합니다.")
    mark = DONE if mode == "check" else TODO

    with open(path, encoding="utf-8") as f:
        lines = f.read().split("\n")

    members, col = [], None
    rows = []  # (줄 번호, 강의 번호)
    for i, line in enumerate(lines):
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells and cells[0] == "#":
            members = cells[3:]
            continue
        if len(cells) >= 5 and all(c in (DONE, TODO) for c in cells[3:]):
            rows.append((i, cells[0]))

    if member not in members:
        sys.exit(f"팀원 이름을 찾지 못했습니다: {member} (가능한 이름: {', '.join(members)})")
    col = 3 + members.index(member)  # 줄을 | 로 나눴을 때 앞의 빈 칸을 빼고 센 열 위치
    ids = [r[1] for r in rows]
    for label, value in (("시작", start), ("끝", end)):
        if value not in ids:
            sys.exit(f"{label} 번호를 진도표에서 찾지 못했습니다: {value}")
    a, b = ids.index(start), ids.index(end)
    if a > b:
        sys.exit(f"시작 번호({start})가 끝 번호({end})보다 뒤에 있습니다.")

    changed = 0
    for line_no, _ in rows[a:b + 1]:
        parts = lines[line_no].split("|")
        idx = col + 1  # 맨 앞의 빈 문자열 때문에 1 더한다
        if parts[idx].strip() != mark:
            parts[idx] = f" {mark} "
            lines[line_no] = "|".join(parts)
            changed += 1

    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("\n".join(lines))
    print(f"{member}: {start} ~ {end} ({b - a + 1}개 강의) 중 {changed}개를 {mark}로 바꿨습니다.")


if __name__ == "__main__":
    main()
