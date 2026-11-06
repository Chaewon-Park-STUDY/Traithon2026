"""시연 예시 5건을 서버 데이터에서 뽑아 /root/Data/demo/examples.json 생성(서버에서 실행)."""
import json, pandas as pd
main = pd.read_parquet("/root/Data/ax_input_1_main.parquet").set_index("key")
inj = pd.read_parquet("/root/Data/demo/injection_cases.parquet").set_index("case_id")
E = [("예시 1 · 정상 기사 (IT&과학)", "IS_M10_347121", "IT&과학", False),
     ("예시 2 · 본문에 무관한 문장이 끼워진 기사 (생활&문화)", "LC_M03_656236", "생활&문화", False),
     ("예시 3 · 판단이 불확실한 기사 (사회)", "SO_M01_210286", "사회", False),
     ("예시 4 · 의무 검토 분야 기사 (정치)", "PO_M01_400114", "정치", False),
     ("예시 5 · 판정 조작 지시문이 삽입된 기사 (세계)", "GB_M11_042978__force_label", "세계", True)]
out = []
for lab, k, cat, is_inj in E:
    r = inj.loc[k] if is_inj else main.loc[k]
    out.append(dict(label=lab, title=r.title, body=r.body, category=cat))
json.dump(out, open("/root/Data/demo/examples.json", "w", encoding="utf-8"), ensure_ascii=False)
print([(o["label"], len(o["body"])) for o in out])
