#!/usr/bin/env python3
"""교육 진도표(마크다운)를 읽어 팀원별 진행률과 남은 시간을 계산한다.

사용법: python scripts/progress.py [진도표 경로]
출력: 마크다운 요약을 표준 출력으로 내보낸다. 환경변수 SUMMARY_FILE이 있으면 같은 내용을 그 파일에도 덧붙인다.
"""
import os
import re
import sys

DONE, TODO = "✅", "⬜"
DEFAULT_PATH = "교육진도/traithon-progress-tracker.md"


def to_seconds(text):
    """'13분 34초', '1시간 2분', '51초' 형태를 초로 바꾼다."""
    h = re.search(r"(\d+)\s*시간", text)
    m = re.search(r"(\d+)\s*분", text)
    s = re.search(r"(\d+)\s*초", text)
    return (int(h.group(1)) * 3600 if h else 0) + (int(m.group(1)) * 60 if m else 0) + (int(s.group(1)) if s else 0)


def fmt(seconds):
    h, rest = divmod(seconds, 3600)
    m = rest // 60
    return f"{h}시간 {m}분" if h else f"{m}분"


def parse(path):
    chapters, current, members = [], None, []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("## "):
                current = {"title": line[3:].strip(), "rows": []}
                chapters.append(current)
                continue
            if current is None or not line.startswith("|"):
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) < 5 or set(cells[0]) <= set("-: "):
                continue
            if cells[0] == "#":
                members = cells[3:]
                continue
            marks = cells[3:]
            if not marks or not all(m in (DONE, TODO) for m in marks):
                continue
            current["rows"].append({"id": cells[0], "name": cells[1], "sec": to_seconds(cells[2]), "marks": marks})
    return chapters, members


def summarize(chapters, members):
    n = len(members)
    total = [0] * n
    done = [0] * n
    rem_sec = [0] * n
    lines = ["# 교육 진도 현황", ""]
    for ch in chapters:
        if not ch["rows"]:
            continue
        lines += [f"## {ch['title']}", "", "| 팀원 | 완료 | 진행률 | 남은 시간 |", "|---|---|---|---|"]
        for i, name in enumerate(members):
            d = sum(1 for r in ch["rows"] if r["marks"][i] == DONE)
            t = len(ch["rows"])
            rs = sum(r["sec"] for r in ch["rows"] if r["marks"][i] == TODO)
            done[i] += d
            total[i] += t
            rem_sec[i] += rs
            lines.append(f"| {name} | {d}/{t} | {d / t:.0%} | {fmt(rs)} |")
        lines.append("")
    lines += ["## 전체", "", "| 팀원 | 완료 | 진행률 | 남은 시간 |", "|---|---|---|---|"]
    for i, name in enumerate(members):
        pct = done[i] / total[i] if total[i] else 0
        lines.append(f"| {name} | {done[i]}/{total[i]} | {pct:.0%} | {fmt(rem_sec[i])} |")
    finished = [m for i, m in enumerate(members) if total[i] and done[i] == total[i]]
    left = [m for i, m in enumerate(members) if not (total[i] and done[i] == total[i])]
    lines += ["", f"전원 수료까지 남은 팀원: {', '.join(left) if left else '없음 (전원 완료)'}"]
    if finished:
        lines.append(f"수료 완료: {', '.join(finished)}")
    return "\n".join(lines) + "\n"


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("TRACKER_PATH", DEFAULT_PATH)
    chapters, members = parse(path)
    if not members:
        sys.exit(f"진도표에서 팀원 열을 찾지 못했습니다: {path}")
    out = summarize(chapters, members)
    print(out)
    summary = os.environ.get("SUMMARY_FILE")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write(out)


if __name__ == "__main__":
    main()
