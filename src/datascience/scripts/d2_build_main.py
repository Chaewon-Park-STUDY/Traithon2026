"""D1~D3: AI-Hub 낚시성 기사 원본(JSON, zip 또는 폴더)을 정제해 clean_main.parquet 와 품질 지표를 만든다.
사용법: python d2_build_main.py [원본경로=data_raw/aihub]
- 라벨: clickbaitClass 0 → label=1(클릭베이트). common.is_clickbait_aihub 한 곳에서 관리.
- 입력 조립(BaseKit 4 §4-3): Part1 = (newTitle, newsContent), Part2 = (newsTitle, processSentenceInfo 문장 순서대로).
- processType·processPattern 은 분석·슬라이스용으로만 보존하고 모델 입력에는 넣지 않는다.
"""
import os, sys, json, zipfile, re
import pandas as pd
from common import RAW, OUT, is_clickbait_aihub, file_sha

SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(RAW, "aihub")


def find(d, key, default=None):
    """중첩 dict/list 에서 key 를 처음 찾은 값(필드 위치가 Part별로 달라도 대응)."""
    if isinstance(d, dict):
        if key in d:
            return d[key]
        for v in d.values():
            r = find(v, key, None)
            if r is not None:
                return r
    elif isinstance(d, list):
        for v in d:
            r = find(v, key, None)
            if r is not None:
                return r
    return default


def sent_text(s):
    for k in ("sentenceContent", "sentence", "content", "text"):
        if isinstance(s, dict) and k in s:
            return str(s[k])
    return str(s)


def sent_no(s, i):
    for k in ("sentenceNo", "sentenceNum", "no"):
        if isinstance(s, dict) and k in s:
            try:
                return int(s[k])
            except Exception:
                pass
    return i


def iter_json():
    """(상대경로, dict) 생성. zip 안의 JSON, 폴더 안의 JSON 모두 지원."""
    for dp, _, fs in os.walk(SRC):
        for fn in sorted(fs):
            p = os.path.join(dp, fn)
            if fn.lower().endswith(".zip"):
                with zipfile.ZipFile(p) as z:
                    for n in z.namelist():
                        if n.lower().endswith(".json") and "원천" not in n:  # 라벨링 JSON에 원천 내용이 모두 포함됨
                            yield f"{fn}/{n}", json.loads(z.read(n).decode("utf-8-sig"))
            elif fn.lower().endswith(".json"):
                with open(p, encoding="utf-8-sig") as f:
                    yield os.path.relpath(p, SRC), json.load(f)


def split_of(path):
    s = path.lower()
    if "validation" in s or "/vl" in s or "vl_" in s or "검증" in path:
        return "Validation"
    if "training" in s or "/tl" in s or "tl_" in s or "학습" in path:
        return "Training"
    return "Unknown"


def build_row(path, d):
    part = find(d, "partNum")
    sents = find(d, "processSentenceInfo") or None
    is_part2 = (str(part).upper() in ("P2", "2", "PART2")) or (sents and find(d, "newTitle") in (None, ""))
    title_orig = find(d, "newsTitle", "") or ""
    if is_part2:
        title = title_orig
        ss = sents or find(d, "sentenceInfo") or []
        ss = sorted(ss, key=lambda s: sent_no(s, 0)) if isinstance(ss, list) else []
        body = " ".join(sent_text(s) for s in ss)
        off = [sent_no(s, i) for i, s in enumerate(ss) if isinstance(s, dict) and str(s.get("subjectConsistencyYn", "Y")).upper() == "N"]
    else:
        title = find(d, "newTitle") or title_orig
        body = find(d, "newsContent", "") or ""
        if not body:
            ss = find(d, "sentenceInfo") or []
            body = " ".join(sent_text(s) for s in ss)
        ref = find(d, "referSentenceInfo") or []
        off = [sent_no(s, i) for i, s in enumerate(ref) if isinstance(s, dict) and str(s.get("referSentenceyn", s.get("referSentenceYn", "N"))).upper() == "Y"]
    code = find(d, "clickbaitClass")
    ptype = str(find(d, "processType", "") or "")
    pattern = str(find(d, "processPattern", "") or "")
    return dict(
        news_id=str(find(d, "newsID", "") or os.path.basename(path)),
        split=split_of(path), part="Part2" if is_part2 else "Part1",
        category=str(find(d, "newsCategory", "") or ""),
        process_type="Direct" if ptype.upper().startswith("D") else ("Auto" if ptype else ""),
        pattern=pattern, level=str(find(d, "processLevel", "") or ""),
        clickbait_code=code, label=None if code is None else is_clickbait_aihub(code),
        title=str(title).strip(), body=re.sub(r"\s+", " ", str(body)).strip(),
        title_orig=str(title_orig).strip(), evidence_sent=",".join(map(str, off)),
        src_path=path,
    )


def main():
    rows = [build_row(p, d) for p, d in iter_json()]
    df = pd.DataFrame(rows)
    if df.empty:
        sys.exit(f"JSON 을 찾지 못했습니다: {SRC}")
    # 샘플(경량) 데이터처럼 공식 분할이 없으면 기사 ID 해시로 Training 90% / Validation 10% (공식 비율 8.9:1.1에 근접, 재현 가능)
    unk = df.split == "Unknown"
    if unk.any():
        import hashlib
        h = df.loc[unk, "news_id"].map(lambda x: int(hashlib.sha256(x.encode()).hexdigest()[:8], 16) % 10)
        df.loc[unk, "split"] = h.map(lambda v: "Validation" if v == 0 else "Training")
        df.loc[unk, "split_source"] = "id_hash_90_10"
    df["split_source"] = df.get("split_source", pd.Series(index=df.index, dtype=object)).fillna("official")
    # 입력 이상(RISK-05, DQ-04): 본문 누락, 너무 짧음, 한글 비율 낮음 → 판단 보류 플래그
    hangul = df.body.str.count(r"[가-힣]") / df.body.str.len().clip(lower=1)
    df["flag_empty"] = (df.title == "") | (df.body == "")
    df["flag_short"] = df.body.str.len() < 100
    df["flag_nonko"] = hangul < 0.3
    df["flag_pii"] = df.body.str.contains(r"[\w.+-]+@[\w-]+\.[\w.]+|01[016789]-?\d{3,4}-?\d{4}", regex=True)
    df["hold"] = df[["flag_empty", "flag_short", "flag_nonko"]].any(axis=1)
    df["title_len"], df["body_len"] = df.title.str.len(), df.body.str.len()
    df.to_parquet(os.path.join(OUT, "clean_main.parquet"), index=False)

    # 품질 지표(BaseKit 4 §4-6)
    dq = {
        "rows": len(df), "unique_ids": int(df.news_id.nunique()),
        "dup_ids_across_split": int(df.groupby("news_id").split.nunique().gt(1).sum()),
        "multi_class_ids": int(df.groupby("news_id").label.nunique().gt(1).sum()),
        "label_missing": int(df.label.isna().sum()),
        **{k: int(df[k].sum()) for k in ["flag_empty", "flag_short", "flag_nonko", "flag_pii", "hold"]},
    }
    pd.Series(dq).to_csv(os.path.join(OUT, "dq_metrics_summary.csv"), header=["value"])
    dist = df.groupby(["split", "category", "label"]).size().unstack(fill_value=0)
    dist.to_csv(os.path.join(OUT, "dq_category_distribution.csv"))
    df.groupby(["split", "label", "process_type", "part"]).size().to_csv(os.path.join(OUT, "dq_type_distribution.csv"))
    print(pd.Series(dq).to_string()); print(dist)


if __name__ == "__main__":
    main()
