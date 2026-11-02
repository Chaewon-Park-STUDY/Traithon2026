"""Webis Clickbait 17 train zip(937MB)에서 jsonl만 HTTP Range로 추출한다(이미지 제외)."""
import io, sys, zipfile, urllib.request, os
URL = "https://zenodo.org/api/records/5530410/files/clickbait17-train-170630.zip/content"
OUT = sys.argv[1]
class HttpFile(io.RawIOBase):
    def __init__(s, url):
        s.url = url; s.pos = 0
        r = urllib.request.urlopen(urllib.request.Request(url, method="HEAD"))
        s.url = r.geturl(); s.size = int(r.headers["Content-Length"])
    def seekable(s): return True
    def readable(s): return True
    def tell(s): return s.pos
    def seek(s, off, wh=0):
        s.pos = off if wh == 0 else s.pos + off if wh == 1 else s.size + off; return s.pos
    def read(s, n=-1):
        if n < 0: n = s.size - s.pos
        if n == 0 or s.pos >= s.size: return b""
        end = min(s.pos + n, s.size) - 1
        r = urllib.request.urlopen(urllib.request.Request(s.url, headers={"Range": f"bytes={s.pos}-{end}"}))
        d = r.read(); s.pos += len(d); return d
    def readinto(s, b):
        d = s.read(len(b)); b[:len(d)] = d; return len(d)
f = io.BufferedReader(HttpFile(URL), buffer_size=1 << 20)
z = zipfile.ZipFile(f)
for i in z.infolist():
    if i.filename.endswith(".jsonl"):
        dst = os.path.join(OUT, os.path.basename(i.filename))
        with z.open(i) as src, open(dst, "wb") as o: o.write(src.read())
        print(i.filename, i.file_size)
