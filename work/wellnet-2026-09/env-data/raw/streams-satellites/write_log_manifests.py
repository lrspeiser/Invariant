"""Manifest the acquisition-process logs so every non-script file in the lane
carries provenance. These are NOT science tables: they are search results and
fetch logs produced while locating catalogues.
"""
import json
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from _manifest import write_manifest  # noqa: E402

LOGS = {
    "_vizier_fetch_log.json": (
        "Fetch log recording each VizieR request issued during satellite-catalogue "
        "acquisition and whether the payload validated.",
        "https://vizier.cds.unistra.fr/viz-bin/asu-tsv"),
    "crot_vizier_fetch_log.json": (
        "Fetch log for the counter-rotator VizieR requests.",
        "https://vizier.cds.unistra.fr/viz-bin/asu-tsv"),
    "crot_vizier_find_catalogs.json": (
        "Cached astroquery Vizier.find_catalogs() results used to locate "
        "counter-rotator catalogue identifiers instead of guessing them.",
        "https://vizier.cds.unistra.fr/ (astroquery find_catalogs)"),
    "crot_vizier_find2.json": (
        "Second pass of cached Vizier.find_catalogs() results.",
        "https://vizier.cds.unistra.fr/ (astroquery find_catalogs)"),
    "crot_vizier_find3.json": (
        "Third pass of cached Vizier.find_catalogs() results.",
        "https://vizier.cds.unistra.fr/ (astroquery find_catalogs)"),
    "crot_vizier_table_index.json": (
        "Index of the real table names inside each candidate VizieR catalogue. "
        "Needed because VizieR table names are frequently NOT table1/table2 but "
        "names such as 'catalog', 'dsph' or 'streams'.",
        "https://vizier.cds.unistra.fr/ (astroquery get_catalogs)"),
    "crot_arxiv_search.json": (
        "Cached arXiv API search results used to resolve paper identifiers.",
        "http://export.arxiv.org/api/query"),
    "crot_arxiv_search.log": (
        "Text log of the arXiv identifier resolution.",
        "http://export.arxiv.org/api/query"),
    "crot_eprint_fetch_log.json": (
        "Log of arXiv e-print tarball fetches.", "https://arxiv.org/e-print/"),
    "crot_eprint_fetch_log2.json": (
        "Second log of arXiv e-print tarball fetches.", "https://arxiv.org/e-print/"),
    "crot_polar_and_counterrotator_summary.json": (
        "Derived roll-up of polar / counter-rotator counts across the acquired "
        "counter-rotator catalogues.",
        "derived from the crot_* tables in this directory"),
    "crot_component_pair_counts.json": (
        "Derived counts of systems with BOTH components kinematically measured, "
        "per catalogue.",
        "derived from the crot_* tables in this directory"),
}

for fn, (desc, src) in LOGS.items():
    p = os.path.join(BASE, fn)
    if not os.path.exists(p):
        print("skip (absent):", fn)
        continue
    n = None
    try:
        with open(p, encoding="utf-8") as fh:
            obj = json.load(fh)
        if isinstance(obj, list):
            n = len(obj)
        elif isinstance(obj, dict):
            n = len(obj)
    except Exception:
        with open(p, encoding="utf-8", errors="replace") as fh:
            n = sum(1 for _ in fh)
    write_manifest(
        p, source_url=src, query="see 'extraction'; this file is an acquisition log",
        columns=[], row_count=n,
        extraction=desc,
        measurement_or_model="ACQUISITION LOG / SEARCH RESULT -- not a science table and not "
                             "usable as an observation.",
        note="Retained for provenance so that every catalogue identifier used in this lane can "
             "be traced back to the search that found it.",
    )
print("DONE")
