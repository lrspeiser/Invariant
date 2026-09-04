"""Retry J/ApJS/280/55 (MaNGA DynPop VII circular velocity curves, Zhu+ 2025).
The -out.all&-out.max=unlimited form returned the metadata header with ZERO data
rows. Try alternative VizieR query forms."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from _manifest import http_get, assert_vizier_tsv, write_manifest
from crot_parse import parse_vizier

D = os.path.dirname(os.path.abspath(__file__))
CAT = "J/ApJS/280/55"
FORMS = [
    ("outmax999999", "https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=J/ApJS/280/55&-out.all&-out.max=999999"),
    ("named_table",  "https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=J/ApJS/280/55/table1&-out.all&-out.max=unlimited"),
    ("no_outall",    "https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=J/ApJS/280/55&-out.max=unlimited&-out=**"),
    ("plain",        "https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=J/ApJS/280/55&-out.max=unlimited"),
]
ok = None
for tag, url in FORMS:
    dest = os.path.join(D, "crot_manga_dynpop_VII_zhu2025.%s.tsv" % tag)
    try:
        http_get(url, dest)
        cols, data = assert_vizier_tsv(dest, expect_catalog=CAT, min_rows=1)
        print("   >>> SUCCESS with form %r : %d rows" % (tag, len(data)))
        ok = (tag, url, dest); break
    except Exception as e:
        print("   form %-14s FAILED: %s" % (tag, e))
        try:
            os.remove(dest)
        except OSError:
            pass

if ok:
    tag, url, dest = ok
    cat, title, tables = parse_vizier(dest)
    total = sum(t["nrows"] for t in tables)
    print("\n%s | %s | tables: %s" % (cat, title,
          ["%s=%d" % (t["name"], t["nrows"]) for t in tables]))
    cols = []
    for t in tables:
        for c in t["columns"]:
            cols.append({"name": "%s.%s" % (t["name"].split("_")[-1], c["name"]),
                         "unit": c["unit"]})
    write_manifest(dest, source_url=url,
        query="HTTP GET %s [VizieR asu-tsv, retry form %r]" % (url, tag),
        columns=cols, row_count=total,
        measurement_or_model=("MODEL - DO NOT TREAT AS OBSERVATION. The circular "
            "velocity curves Vc(Re), Vc(amaj), Vcmax, Vc(rmax) are outputs of Jeans "
            "Anisotropic Modelling (JAM) of the MaNGA stellar kinematics; the table "
            "carries an explicit 'Qual = JAM model quality' column and logMBH is a "
            "model black-hole mass. A JAM circular-velocity curve embeds a total mass "
            "distribution and is NOT a direct measurement of the gravitational field. "
            "Re, amaj, FWHM, DA and the plate/IFU identifiers ARE measurements."),
        note=("Retry succeeded with query form %r after the -out.all&-out.max=unlimited "
              "form returned a header with zero data rows." % tag),
        extra={"vizier_catalog": cat, "vizier_title": title,
               "tables_detail": [{"table": t["name"], "nrows": t["nrows"],
                                  "ncols": len(t["columns"]),
                                  "columns": t["columns"]} for t in tables]})
else:
    print("\nALL FORMS FAILED for %s - recorded as a FAILURE, no substitute used." % CAT)
