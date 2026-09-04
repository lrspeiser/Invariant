# -*- coding: utf-8 -*-
import requests, hashlib, json, os, sys, datetime, io
BASE = r"C:\Users\henry\Documents\Codex\2026-08-21\Invariant-main-integration\work\wellnet-2026-09\cluster-data"
MEM  = os.path.join(BASE, "members")
os.makedirs(MEM, exist_ok=True)
VIZ = "https://vizier.cds.unistra.fr/viz-bin/asu-tsv"

def utcnow(): return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
def sha256f(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda: f.read(1<<20), b''): h.update(b)
    return h.hexdigest()

def parse_tsv(text):
    """VizieR asu-tsv: comment lines start with '#'. Then header, units, dashes, data."""
    lines = text.split("\n")
    body = [l for l in lines if not l.startswith("#")]
    # strip leading blanks
    while body and body[0].strip()=="": body.pop(0)
    if len(body)<3: return None,None,[]
    hdr = body[0].split("\t")
    units = body[1].split("\t")
    # line 2 should be dashes
    if not set(body[2].replace("\t","").strip()) <= set("-"):
        return None,None,[]
    data=[l for l in body[3:] if l.strip()!=""]
    return hdr, units, data

def pull(source, outname, note, extra=""):
    url = "%s?-source=%s&-out=**&-out.max=unlimited%s" % (VIZ, source, extra)
    print("GET", url)
    r = requests.get(url, timeout=1800)
    raw = os.path.join(MEM, outname + ".raw.tsv")
    with open(raw,"wb") as f: f.write(r.content)
    text = r.content.decode("utf-8","replace")
    ok_name = source in text
    hdr, units, data = parse_tsv(text)
    nrow = len(data) if data else 0
    print("  status=%d bytes=%d source_echoed=%s nrow=%s ncol=%s" % (r.status_code, len(r.content), ok_name, nrow, len(hdr) if hdr else 0))
    if r.status_code!=200 or not ok_name or nrow==0:
        print("  !! GENERIC/EMPTY -> NOT FOUND")
        return {"status":"NOT_FOUND","source":source,"http":r.status_code,"bytes":len(r.content),"source_echoed":ok_name,"row_count":nrow}
    man = {
      "file": outname + ".raw.tsv",
      "source_url": url,
      "exact_query": "VizieR asu-tsv GET -source=%s -out=** -out.max=unlimited%s" % (source, extra),
      "retrieved_utc": utcnow(),
      "sha256": sha256f(raw),
      "bytes": os.path.getsize(raw),
      "row_count": nrow,
      "column_count": len(hdr),
      "columns": [{"name":h,"unit":(units[i] if i<len(units) else "")} for i,h in enumerate(hdr)],
      "extraction": "Raw VizieR asu-tsv response saved verbatim. Rows counted after skipping '#' comment lines, the header line, the units line and the '---' separator line; blank trailing lines excluded.",
      "vizier_identifier_echoed_in_response": ok_name,
      "vizier_identifier": source,
      "note": note,
      "raw_response_file": outname + ".raw.tsv"
    }
    with open(os.path.join(MEM, outname + ".raw.tsv.manifest.json"),"w",encoding="utf-8") as f:
        json.dump(man,f,indent=2)
    print("  OK rows=%d cols=%d sha=%s" % (nrow,len(hdr),man["sha256"][:16]))
    return {"status":"OK","row_count":nrow,"column_count":len(hdr),"file":raw}

if __name__=="__main__":
    pass
