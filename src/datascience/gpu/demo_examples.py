"""시연 예시 5건을 서버 데이터에서 뽑아 /root/Data/demo/examples.json 생성(서버에서 실행)."""
import json, pandas as pd
main = pd.read_parquet("/root/Data/ax_input_1_main.parquet").set_index("key")
inj = pd.read_parquet("/root/Data/demo/injection_cases.parquet").set_index("case_id")
E = [("① 정상 기사 (IT&과학)", "IS_M10_347121", "IT&과학", False),
     ("② 낚시성: 사람이 끼워 넣은 이탈 문장", "LC_M03_656236", "생활&문화", False),
     ("③ 애매한 기사 (사회)", "SO_M01_210286", "사회", False),
     ("④ 정치 기사 (의무 이관)", "PO_M01_400114", "정치", False),
     ("⑤ 인젝션: 판정 강요 지시문", "GB_M11_042978__force_label", "세계", True)]
out = []
for lab, k, cat, is_inj in E:
    r = inj.loc[k] if is_inj else main.loc[k]
    out.append(dict(label=lab, title=r.title, body=r.body, category=cat))
json.dump(out, open("/root/Data/demo/examples.json", "w", encoding="utf-8"), ensure_ascii=False)
print([(o["label"], len(o["body"])) for o in out])
