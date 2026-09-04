"""arXiv API search to find the REAL arXiv ids (never guess them).

Writes extstream_arxiv_search.json with the full hit list per query.
"""
import sys, os, json, time, urllib.parse
import requests
import xml.etree.ElementTree as ET

BASE = r"C:\Users\henry\Documents\Codex\2026-08-21\Invariant-main-integration\work\wellnet-2026-09\env-data\raw\streams-satellites"
sys.path.insert(0, BASE)
HDR = {"User-Agent": "gravity-research-acquisition/1.0 (academic data acquisition)"}
NS = {"a": "http://www.w3.org/2005/Atom"}

QUERIES = [
    ("ssls_all", 'all:"Stellar Stream Legacy Survey"'),
    ("miro_carretero", 'au:"Miro-Carretero"'),
    ("md_streams_2010", 'au:"Martinez-Delgado" AND abs:"tidal stream"'),
    ("md_ngc5907", 'au:"Martinez-Delgado" AND all:"NGC 5907"'),
    ("dragonfly_5907", 'all:"NGC 5907" AND au:"van Dokkum"'),
    ("m31_gss_gilbert", 'au:"Gilbert" AND all:"giant southern stream"'),
    ("m31_gss_kinematics", 'all:"M31" AND all:"giant stellar stream" AND all:"kinematics"'),
    ("kalirai_m31", 'au:"Kalirai" AND all:"M31" AND all:"stream"'),
    ("escala_m31", 'au:"Escala" AND all:"M31"'),
    ("cunningham_m31", 'au:"Cunningham" AND all:"M31" AND all:"halo"'),
    ("ibata_m31_stream", 'au:"Ibata" AND all:"Andromeda" AND all:"stream"'),
    ("cena_pne", 'all:"NGC 5128" AND all:"planetary nebulae" AND all:"kinematics"'),
    ("cena_gc", 'all:"NGC 5128" AND all:"globular cluster" AND all:"kinematics"'),
    ("extgal_stream_kinematics", 'all:"stellar stream" AND all:"external galaxy" AND all:"velocities"'),
    ("umbrella_ngc4651", 'all:"NGC 4651" AND all:"umbrella"'),
    ("ngc5907_stream", 'all:"NGC 5907" AND all:"stream"'),
]


def arxiv(query, max_results=60):
    url = ("http://export.arxiv.org/api/query?search_query=%s&start=0&max_results=%d"
           "&sortBy=relevance&sortOrder=descending" % (urllib.parse.quote(query), max_results))
    r = requests.get(url, timeout=120, headers=HDR)
    r.raise_for_status()
    root = ET.fromstring(r.text)
    hits = []
    for e in root.findall("a:entry", NS):
        idu = e.find("a:id", NS).text
        aid = idu.rsplit("/", 1)[-1]
        title = " ".join(e.find("a:title", NS).text.split())
        pub = e.find("a:published", NS).text
        auths = [a.find("a:name", NS).text for a in e.findall("a:author", NS)][:6]
        jref = e.find("a:journal_ref", NS)
        summ = " ".join((e.find("a:summary", NS).text or "").split())
        hits.append({"arxiv_id": aid, "title": title, "published": pub,
                     "authors": auths,
                     "journal_ref": jref.text if jref is not None else None,
                     "abstract": summ[:900]})
    return url, hits


out = {}
for name, q in QUERIES:
    try:
        url, hits = arxiv(q)
        out[name] = {"query": q, "url": url, "n": len(hits), "hits": hits}
        print("=" * 78)
        print("QUERY %-24s  %s   -> %d hits" % (name, q, len(hits)))
        for h in hits[:12]:
            print("   %-14s %s  | %s" % (h["arxiv_id"], h["published"][:10], h["title"][:110]))
            if h["journal_ref"]:
                print("                  jref: %s" % h["journal_ref"][:100])
    except Exception as e:
        out[name] = {"query": q, "error": str(e)}
        print("QUERY %s FAILED: %s" % (name, e))
    time.sleep(3.2)

with open(os.path.join(BASE, "extstream_arxiv_search.json"), "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=2)
print("\nwrote extstream_arxiv_search.json")
