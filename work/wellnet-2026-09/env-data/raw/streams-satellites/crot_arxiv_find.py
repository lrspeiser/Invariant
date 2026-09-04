"""Resolve arXiv IDs for the misalignment / counter-rotation papers that are NOT
in VizieR, using the arXiv API (do not guess IDs). Polite backoff: arXiv returns
429 if queried faster than ~1 req / 3 s from a fresh client."""
import sys, os, json, time, urllib.parse
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import requests
import xml.etree.ElementTree as ET

NS = {"a": "http://www.w3.org/2005/Atom"}
API = "http://export.arxiv.org/api/query"
S = requests.Session()
S.headers.update({"User-Agent": "gravity-research-acquisition/1.0 (academic)"})

QUERIES = [
    ("Jin2016_MaNGA_misalign",   'all:"MaNGA" AND all:"kinematically decoupled" AND au:"Jin"'),
    ("Bryant2019_SAMI_misalign", 'au:"Bryant_J" AND all:"SAMI" AND all:"misalign"'),
    ("Chen2016_misaligned",      'all:"misaligned gas" AND all:"MaNGA" AND au:"Chen"'),
    ("BarreraBallesteros_CALIFA",'au:"Barrera-Ballesteros" AND all:"CALIFA"'),
    ("Xu_MaNGA_counterrot",      'all:"counter-rotating" AND all:"MaNGA"'),
    ("Bao_MaNGA_counterrot",     'au:"Bao_M" AND all:"counter-rotat"'),
    ("Zhou_MaNGA_counterrot",    'au:"Zhou_S" AND all:"counter-rotat"'),
    ("Duckworth_misalign",       'au:"Duckworth_C" AND all:"misalign"'),
    ("Moiseev_polarring",        'au:"Moiseev_A" AND all:"polar ring"'),
    ("polarring_catalogue",      'all:"polar ring" AND all:"catalogue"'),
    ("SAMI_misalignment",        'all:"SAMI" AND all:"kinematic misalignment"'),
]


def query(q, n=25, tries=6):
    for k in range(tries):
        try:
            r = S.get(API, params={"search_query": q, "start": 0,
                                   "max_results": n}, timeout=90)
            if r.status_code == 429:
                w = 12 * (k + 1)
                print("   429, sleeping %ds" % w); time.sleep(w); continue
            r.raise_for_status()
            return ET.fromstring(r.text)
        except ET.ParseError as e:
            print("   parse error: %s" % e); time.sleep(10)
        except Exception as e:
            print("   %s" % e); time.sleep(10 * (k + 1))
    return None


out = {}
for name, q in QUERIES:
    root = query(q)
    if root is None:
        print("FAILED %s" % name); out[name] = []; continue
    entries = []
    for e in root.findall("a:entry", NS):
        aid = e.find("a:id", NS).text.rsplit("/", 1)[-1]
        ti = " ".join(e.find("a:title", NS).text.split())
        auth = [a.find("a:name", NS).text for a in e.findall("a:author", NS)]
        jr = e.find("{http://arxiv.org/schemas/atom}journal_ref")
        entries.append({"arxiv": aid, "title": ti,
                        "first_author": auth[0] if auth else "",
                        "published": e.find("a:published", NS).text[:10],
                        "journal_ref": jr.text.strip().replace("\n", " ") if jr is not None else ""})
    out[name] = entries
    print("\n=== %s  (%d hits) ===" % (name, len(entries)))
    for e in entries[:14]:
        print("  %-12s %-10s %-30s %-18s %s"
              % (e["arxiv"], e["published"], (e["journal_ref"] or "-")[:30],
                 e["first_author"][:18], e["title"][:80]))
    time.sleep(4)

with open("crot_arxiv_search.json", "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=2)
print("\nWROTE crot_arxiv_search.json")
