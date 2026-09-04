"""arXiv API search, round 2: targeted ids for the papers the brief names.

Round 1 crashed on a cp1252 print of a Greek alpha; here all printing is forced
to UTF-8 and non-encodable characters are replaced.
"""
import sys, os, json, time, urllib.parse, io
import requests
import xml.etree.ElementTree as ET

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE = r"C:\Users\henry\Documents\Codex\2026-08-21\Invariant-main-integration\work\wellnet-2026-09\env-data\raw\streams-satellites"
sys.path.insert(0, BASE)
HDR = {"User-Agent": "gravity-research-acquisition/1.0 (academic data acquisition)"}
NS = {"a": "http://www.w3.org/2005/Atom"}

QUERIES = [
    ("md2010_pilot_AJ140", 'au:"Martinez-Delgado" AND all:"pilot survey" AND all:"Local Volume"'),
    ("md2010_alt", 'au:"Martinez-Delgado" AND ti:"Stellar tidal streams in spiral galaxies"'),
    ("dragonfly_ngc5907_2019", 'all:"NGC 5907" AND all:"Dragonfly"'),
    ("vandokkum_5907", 'au:"van Dokkum" AND all:"5907"'),
    ("merritt_dragonfly", 'au:"Merritt" AND all:"Dragonfly" AND all:"stream"'),
    ("ngc4651_umbrella", 'all:"NGC 4651" AND all:"stream"'),
    ("foster_umbrella", 'au:"Foster" AND all:"umbrella"'),
    ("sombrero_stream", 'all:"Sombrero" AND all:"stream"'),
    ("cena_pne_peng", 'au:"Peng" AND all:"NGC 5128" AND all:"planetary nebulae"'),
    ("cena_woodley", 'au:"Woodley" AND all:"NGC 5128"'),
    ("cena_halo_stream", 'all:"Centaurus A" AND all:"stellar stream"'),
    ("ngc7241_megara", 'all:"NGC 7241" AND all:"stream"'),
    ("escala_m31_abund", 'au:"Escala" AND all:"Giant Stellar Stream"'),
    ("ibata_2001_andromeda", 'au:"Ibata" AND all:"giant stream" AND all:"Andromeda"'),
    ("extgal_stream_spectroscopy", 'all:"tidal stream" AND all:"external galaxies" AND all:"spectroscopy"'),
    ("gss_pn_kinematics", 'all:"Andromeda" AND all:"planetary nebulae" AND all:"stream"'),
    ("m31_rotation_curve", 'all:"M31" AND all:"rotation curve" AND all:"HI"'),
    ("ssls_ii_iii_iv", 'au:"Miro-Carretero" AND abs:"Stellar Stream Legacy Survey"'),
    ("stsurvey_2018", 'au:"Martinez-Delgado" AND ti:"Stellar Tidal Stream Survey"'),
]


def arxiv(query, max_results=60):
    url = ("http://export.arxiv.org/api/query?search_query=%s&start=0&max_results=%d"
           "&sortBy=relevance&sortOrder=descending" % (urllib.parse.quote(query), max_results))
    r = requests.get(url, timeout=120, headers=HDR)
    r.raise_for_status()
    root = ET.fromstring(r.text)
    hits = []
    for e in root.findall("a:entry", NS):
        aid = e.find("a:id", NS).text.rsplit("/", 1)[-1]
        title = " ".join(e.find("a:title", NS).text.split())
        pub = e.find("a:published", NS).text
        auths = [a.find("a:name", NS).text for a in e.findall("a:author", NS)][:8]
        jref = e.find("a:journal_ref", NS)
        doi = e.find("a:doi", NS)
        summ = " ".join((e.find("a:summary", NS).text or "").split())
        hits.append({"arxiv_id": aid, "title": title, "published": pub, "authors": auths,
                     "journal_ref": jref.text if jref is not None else None,
                     "doi": doi.text if doi is not None else None,
                     "abstract": summ[:1400]})
    return url, hits


out = {}
for name, q in QUERIES:
    try:
        url, hits = arxiv(q)
        out[name] = {"query": q, "url": url, "n": len(hits), "hits": hits}
        print("=" * 78)
        print("QUERY %-24s  %s   -> %d hits" % (name, q, len(hits)))
        for h in hits[:10]:
            print("   %-14s %s | %s" % (h["arxiv_id"], h["published"][:10], h["title"][:105]))
            if h["journal_ref"]:
                print("                  jref: %s" % h["journal_ref"][:110])
    except Exception as e:
        out[name] = {"query": q, "error": repr(e)}
        print("QUERY %s FAILED: %r" % (name, e))
    time.sleep(3.2)

with open(os.path.join(BASE, "extstream_arxiv_search2.json"), "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=2)
print("\nwrote extstream_arxiv_search2.json")
