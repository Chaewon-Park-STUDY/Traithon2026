#!/usr/bin/env python3
"""요약 파일을 Discord 웹훅으로 보낸다. DISCORD_WEBHOOK_URL이 없으면 아무것도 하지 않는다."""
import json
import os
import sys
import urllib.request

url = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()
if not url:
    print("DISCORD_WEBHOOK_URL이 없어 알림을 건너뜁니다.")
    sys.exit(0)
text = open(sys.argv[1], encoding="utf-8").read()
text = text.replace("# ", "", 1)[:1900]
req = urllib.request.Request(url, data=json.dumps({"content": text}).encode("utf-8"),
                             headers={"Content-Type": "application/json", "User-Agent": "traithon-progress-bot"})
urllib.request.urlopen(req, timeout=20)
print("Discord 알림을 보냈습니다.")
