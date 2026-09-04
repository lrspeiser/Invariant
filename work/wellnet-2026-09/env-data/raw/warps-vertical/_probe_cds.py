"""Probe whether a CDS/VizieR catalogue designation exists, by fetching its ReadMe.

A real catalogue returns a plain-text ReadMe beginning with the catalogue
designation. A nonexistent one returns 404 or an HTML error page. We assert both.
"""
import sys
import requests

HDRS = {"User-Agent": "wellnet-gravity-acquisition/1.0 (research; contact via CDS)"}

def probe(cat):
    url = f"https://cdsarc.cds.unistra.fr/ftp/{cat}/ReadMe"
    try:
        r = requests.get(url, headers=HDRS, timeout=60)
    except Exception as e:
        return cat, "ERR", f"{type(e).__name__}: {e}", ""
    if r.status_code != 200:
        return cat, f"HTTP{r.status_code}", "", ""
    txt = r.text
    if "<html" in txt[:400].lower():
        return cat, "HTML", "generic HTML page - NOT a catalogue", ""
    # first non-empty line is normally "J/A+A/394/769   Title (Author+, year)"
    lines = [l for l in txt.splitlines() if l.strip()]
    head = lines[0][:110] if lines else ""
    # list the data files described
    files = []
    for l in lines:
        s = l.strip()
        if s.startswith(("table", "tab", "list", "catalog", "refs", "notes", "asu", "star")) and ".dat" in s:
            files.append(s.split()[0])
    return cat, "OK", head, ",".join(sorted(set(files)))

if __name__ == "__main__":
    for cat in sys.argv[1:]:
        c, st, head, files = probe(cat)
        print(f"{st:9s} {c:24s} {head}")
        if files:
            print(f"{'':9s} {'':24s} files: {files}")
