# -*- coding: utf-8 -*-
"""Re-probe every VizieR catalogue tried for Products 2/3, record FOUND/NOT FOUND,
then verify every manifest against its data file."""
import os, json, hashlib, datetime, requests

ROOT = r"C:\Users\henry\Documents\Codex\2026-08-21\Invariant-main-integration\work\wellnet-2026-09\cluster-data"
NOW  = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
BASE = "https://vizier.cds.unistra.fr/viz-bin/asu-tsv"

PROBES = [
 ("J/ApJS/247/43",    "Kluge et al. 2020, ApJS 247, 43 -- BCG+ICL structure, WWFI g'"),
 ("J/ApJS/252/27",    "Kluge et al. 2021, ApJS 252, 27 -- photometric dissection of ICL"),
 ("J/ApJS/235/14",    "Shipley et al. 2018, ApJS 235, 14 -- HFF-DeepSpace photometry"),
 ("J/ApJS/195/15",    "Donzelli, Muriel & Madrid 2011, ApJS 195, 15 -- BCG luminosity profiles"),
 ("J/ApJ/797/82",     "Lauer et al. 2014, ApJ 797, 82 -- BCGs in Abell clusters"),
 ("J/MNRAS/474/917",  "Montes & Trujillo 2018, MNRAS 474, 917 -- ICL in the six HFF clusters"),
 ("J/MNRAS/516/1182", "Montes & Trujillo 2022, MNRAS 516, 1182"),
 ("J/MNRAS/442/L14",  "Montes & Trujillo 2014, MNRAS 442, L14 -- ICL in Abell 2744"),
 ("J/ApJ/857/79",     "Jimenez-Teja et al. 2018, ApJ 857, 79 -- ICL fractions in HFF/CLASH"),
 ("J/MNRAS/474/3009", "DeMaio et al. 2018, MNRAS 474, 3009 -- BCG+ICL profiles"),
 ("J/MNRAS/491/3751", "DeMaio et al. 2020, MNRAS 491, 3751 -- BCG+ICL growth"),
 ("J/ApJ/618/195",    "Gonzalez, Zabludoff & Zaritsky 2005, ApJ 618, 195 -- BCG+ICL profiles"),
 ("J/ApJ/666/147",    "Gonzalez, Zaritsky & Zabludoff 2007, ApJ 666, 147"),
 ("J/MNRAS/449/2353", "Burke, Hilton & Collins 2015, MNRAS 449, 2353 -- ICL in CLASH"),
]

print("=" * 78); print("VizieR probe log  ", NOW); print("=" * 78)
results = []
for cid, desc in PROBES:
    query = "-source=%s&-meta.all=&-out.max=1" % cid
    try:
        r = requests.get(BASE, params={"-source": cid, "-meta.all": "", "-out.max": "1"}, timeout=90)
        txt = r.text
        notfound = ("Error=Table or Catalog not found" in txt) or ("does not exist in catalog" in txt)
        tables = [l.split(":", 1)[1].strip() for l in txt.split("\n")
                  if l.startswith("#Name:") and "/" in l]
        tables = [t for t in tables if t.count("/") >= 3]
        status = "NOT FOUND" if notfound else "FOUND"
        results.append({"catalogue_id": cid, "reference": desc, "status": status,
                        "http_status": r.status_code, "response_bytes": len(txt),
                        "tables": tables,
                        "error_line": next((l.strip() for l in txt.split("\n") if "Error=" in l), None),
                        "exact_query": BASE + "?" + query, "probed_utc": NOW})
        print("  %-9s %-18s %s" % (status, cid, (",".join(tables) if tables else desc[:44])))
    except Exception as e:
        results.append({"catalogue_id": cid, "reference": desc, "status": "PROBE ERROR",
                        "error": str(e), "exact_query": BASE + "?" + query, "probed_utc": NOW})
        print("  ERROR    %-18s %s" % (cid, e))

log = {
  "generated_utc": NOW,
  "purpose": "Provenance record of every VizieR catalogue probed while acquiring Product 2 (BCG photometry / "
             "light profile) and Product 3 (intracluster light) for the seven target clusters. VizieR returns "
             "HTTP 200 for non-existent catalogues, so status is decided ONLY by the presence of an "
             "'#INFO Error=Table or Catalog not found' line, never by the HTTP code. No NOT FOUND catalogue was "
             "substituted with data from another cluster or another paper.",
  "targets": ["Abell 2744", "MACS J0416.1-2403", "MACS J0717.5+3745", "MACS J1149.5+2223",
              "Abell S1063 (RXC J2248.7-4431)", "Abell 370", "Abell 2029"],
  "vizier_probes": results,
  "papers_retrieved_from_arxiv_instead": [
    {"arxiv_id": "1710.03240", "reference": "Montes & Trujillo 2018, MNRAS 474, 917",
     "title_verified": "Intracluster Light at the Frontier II: The Frontier Fields Clusters",
     "reason": "VizieR J/MNRAS/474/917 NOT FOUND",
     "warning": "The id 1710.07300 is NOT this paper -- it resolves to the machine-learning paper "
                "'FigureQA: An Annotated Figure Dataset for Visual Reasoning'. Verified and discarded."},
    {"arxiv_id": "1803.04981", "reference": "Jimenez-Teja et al. 2018, ApJ 857, 79",
     "title_verified": "Unveiling the dynamical state of massive clusters through the ICL fraction",
     "reason": "VizieR J/ApJ/857/79 NOT FOUND"},
    {"arxiv_id": "2202.08289", "reference": "de Oliveira, Jimenez-Teja et al. 2022",
     "title_verified": "The intracluster light on Frontier Fields clusters Abell 370 and Abell S1063",
     "reason": "supplies the two HFF clusters absent from Jimenez-Teja et al. 2018"},
    {"arxiv_id": "1710.11313", "reference": "DeMaio et al. 2018, MNRAS 474, 3009",
     "title_verified": "Lost but not Forgotten: Intracluster Light in Galaxy Groups and Clusters",
     "reason": "VizieR J/MNRAS/474/3009 NOT FOUND"},
    {"arxiv_id": "2011.12992", "reference": "Kluge et al. 2021, ApJS 252, 27",
     "title_verified": "Photometric dissection of Intracluster Light and its correlations with host cluster properties",
     "reason": "consulted to confirm that NO per-cluster ICL fraction is published (sample averages only)"},
    {"arxiv_id": "1911.07911", "reference": "DeMaio et al. 2020, MNRAS 491, 3751",
     "title_verified": "The Growth of Brightest Cluster Galaxies and Intracluster Light Over the Past Ten Billion Years",
     "reason": "VizieR J/MNRAS/491/3751 NOT FOUND. NOT USED as a product: its only tabulated BCG+ICL "
               "luminosity/mass table (t2.tex) covers the 7 high-redshift (z=1.24-1.75) clusters only; none of "
               "the seven targets appear in it. Its sample table does list RXJ2248, MACS0416 and MACS1149."},
    {"arxiv_id": "2607.15340", "reference": "Ultra-deep INT/WFC imaging of IC 1101 (the Abell 2029 BCG)",
     "title_verified": "How large can galaxies be? Ultra-deep imaging of IC 1101, the most extended known galaxy",
     "reason": "checked for an Abell 2029 ICL profile. NOT USED as a product: the LaTeX source contains NO "
               "tabular environment at all -- every surface-brightness and colour profile is published as a "
               "figure only. It reports R_e = 73 +/- 2 kpc for IC 1101 and notes Uson et al. 1991 traced "
               "stellar emission to ~607 kpc, but gives no ICL fraction and performs no BCG/ICL separation."}
  ],
  "negative_results": [
    "Abell 2029: an arXiv full-text API search for 'Abell 2029' AND 'intracluster light' returned ZERO results. "
    "No per-cluster ICL fraction and no mu_ICL(r) with a stated BCG/ICL separation method was found for A2029 "
    "from any source probed.",
    "Kluge et al. 2020 (J/ApJS/247/43) CDS deposit contains only ReadMe, table1.dat and table4.dat -- the "
    "semimajor-axis BCG+ICL surface-brightness profiles measured down to 30 g' mag/arcsec2 are NOT deposited in "
    "machine-readable form anywhere, and the paper gives no data-availability statement for them.",
    "Kluge et al. 2021 publishes ICL fractions only as sample averages over 170 clusters, never per cluster.",
    "DeMaio et al. 2018 DELIBERATELY EXCLUDES MACS0717: 'we exclude MACS0717 from the final sample because it "
    "is a very dynamic systems of 4 merging clusters with no clear central BCG to which to anchor the radial "
    "profiles'.",
    "No tabulated mu_ICL(r) surface-brightness profile was found for ANY of the seven target clusters. Montes & "
    "Trujillo 2018, DeMaio et al. 2018 and the IC 1101 paper all publish their SB profiles as figures only. The "
    "only profile information obtained in parametric form is the Kluge+2020 and Donzelli+2011 Sersic fits."
  ]
}
p = os.path.join(ROOT, "bcg_icl_source_probe_log.json")
with open(p, "w", encoding="utf-8") as f:
    json.dump(log, f, indent=2, ensure_ascii=False)
print("\nWROTE", p)

# ---------------- verify all manifests ----------------
print("\n" + "=" * 78); print("MANIFEST VERIFICATION"); print("=" * 78)
bad = 0
tot = 0
for sub in ("bcg", "icl"):
    d = os.path.join(ROOT, sub)
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".manifest.json"):
            continue
        tot += 1
        mp = os.path.join(d, fn)
        m = json.load(open(mp, encoding="utf-8"))
        dp = os.path.join(d, m["file"])
        ok = os.path.exists(dp)
        h = sha256_ok = size_ok = rows_ok = None
        if ok:
            h = hashlib.sha256(open(dp, "rb").read()).hexdigest()
            sha256_ok = (h == m["sha256"])
            size_ok = (os.path.getsize(dp) == m["bytes"])
            n = sum(1 for _ in open(dp, encoding="utf-8")) - 1
            rows_ok = (n == m["row_count"])
        flag = "OK " if (ok and sha256_ok and size_ok and rows_ok) else "BAD"
        if flag == "BAD": bad += 1
        print("  %s %-52s rows=%-5s cols=%-4s %s" %
              (flag, m["file"], m["row_count"], m["column_count"],
               "" if flag == "OK " else "exists=%s sha=%s size=%s rows=%s" % (ok, sha256_ok, size_ok, rows_ok)))
print("\n%d manifests checked, %d bad" % (tot, bad))
assert bad == 0, "manifest verification failed"
print("ALL MANIFESTS VERIFIED")
