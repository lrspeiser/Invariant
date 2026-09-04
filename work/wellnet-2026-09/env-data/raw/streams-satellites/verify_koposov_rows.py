import os
import re
import tarfile

BASE = os.path.dirname(os.path.abspath(__file__))
t = tarfile.open(os.path.join(BASE, "koposov2019_paper_1812.08172.tar.gz"))
txt = t.extractfile([m for m in t.getmembers() if m.name == "main.tex"][0]).read().decode("utf-8", "replace")

BS = chr(92)
blocks = re.findall(BS + BS + r"begin\{table\*?\}(.*?)" + BS + BS + r"end\{table\*?\}", txt, re.S)
print("table environments:", len(blocks))
for i, b in enumerate(blocks):
    c = re.search(r"caption\{(.{0,95})", b, re.S)
    tab = re.search(BS + BS + r"begin\{tabular\}\{[^}]*\}(.*?)" + BS + BS + r"end\{tabular\}", b, re.S)
    body = tab.group(1) if tab else ""
    parts = [x.strip() for x in body.split(BS * 2)]
    numeric = 0
    other = 0
    for p in parts:
        p = re.sub(BS + BS + r"hline", "", p).strip()
        if not p:
            continue
        cells = [x.strip() for x in p.split("&")]
        try:
            [float(x.replace("$", "").replace(BS, "")) for x in cells]
            numeric += 1
        except ValueError:
            other += 1
    cap = (c.group(1)[:85].replace("\n", " ") if c else "")
    print("%d) numeric_rows=%3d non_numeric=%2d  %s" % (i, numeric, other, cap))
