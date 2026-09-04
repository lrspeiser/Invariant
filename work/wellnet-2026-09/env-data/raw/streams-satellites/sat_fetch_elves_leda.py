"""JOB2: HyperLEDA position angle + axis ratio for the 31 ELVES host galaxies.

The ELVES host table (Carlsten+2022 Table 1) carries no orientation information at
all. Host names are resolved to coordinates with CDS Sesame, then HyperLEDA
(VizieR VII/237/pgc) is cone-searched at 1 arcmin and the nearest match kept.
Everything recorded is a MEASUREMENT (isophotal shape from imaging).
"""
import os
import sys

HERE = r"C:\Users\henry\Documents\Codex\2026-08-21\Invariant-main-integration\work\wellnet-2026-09\env-data\raw\streams-satellites"
sys.path.insert(0, HERE)
from _manifest import write_manifest, http_get, assert_vizier_tsv  # noqa: E402
from sat_fetch_vizier import parse_tsv  # noqa: E402

import astropy.units as u  # noqa: E402
from astropy.coordinates import SkyCoord  # noqa: E402

ALIAS = {"M31": "NGC 224", "M81": "NGC 3031", "M104": "NGC 4594", "CENA": "NGC 5128",
         "NGC5457": "NGC 5457"}


def main():
    names = [l.split("\t")[0].strip() for l in
             open(os.path.join(HERE, "sat_elves_table1_hosts.tsv"),
                  encoding="utf-8").read().splitlines()[2:]]
    resolved, unresolved = [], []
    for n in names:
        if n == "MW":
            unresolved.append((n, "the Milky Way -- not an external galaxy"))
            continue
        q = ALIAS.get(n, n)
        try:
            c = SkyCoord.from_name(q)
            resolved.append((n, q, c))
        except Exception as e:
            unresolved.append((n, "%s: %s" % (type(e).__name__, e)))
    print("resolved %d / %d ELVES host names" % (len(resolved), len(names)))
    for n, why in unresolved:
        print("   UNRESOLVED %-10s %s" % (n, why))

    # one VizieR call, box-constrained per target via a position list is not supported;
    # use a single query with an OR of small boxes by issuing one cone per host.
    from astroquery.vizier import Vizier
    V = Vizier(columns=["**"], row_limit=-1)
    lines_out = []
    hdr = None
    matched = 0
    for n, q, c in resolved:
        try:
            res = V.query_region(c, radius=1.5 * u.arcmin, catalog="VII/237/pgc")
        except Exception as e:
            print("   QUERY FAIL %-10s %s" % (n, e))
            continue
        if not res or len(res[0]) == 0:
            print("   NO LEDA MATCH %-10s" % n)
            continue
        t = res[0]
        # nearest by angular separation
        sep = SkyCoord(t["RAJ2000"], t["DEJ2000"], unit=(u.hourangle, u.deg)).separation(c)
        k = int(sep.argmin())
        if hdr is None:
            hdr = ["elves_host"] + list(t.colnames) + ["sep_arcsec"]
        row = [n] + [str(t[cn][k]) for cn in t.colnames] + ["%.1f" % sep[k].arcsec]
        lines_out.append(row)
        matched += 1

    dest = os.path.join(HERE, "sat_hyperleda_elves_hosts_PA.tsv")
    units = ["", ""] + ["" for _ in range(len(hdr) - 2)]
    for i, cn in enumerate(hdr):
        if cn == "PA" or cn == "e_PA":
            units[i] = "deg"
        if cn == "sep_arcsec":
            units[i] = "arcsec"
    with open(dest, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\t".join(hdr) + "\n")
        fh.write("\t".join(units) + "\n")
        fh.write("\t".join(["-" * 3 for _ in hdr]) + "\n")
        for r in lines_out:
            fh.write("\t".join(r) + "\n")

    write_manifest(
        dest, "https://vizier.cds.unistra.fr/viz-bin/VizieR-3?-source=VII/237/pgc",
        query="astroquery.vizier query_region(SkyCoord.from_name(<host>), radius=1.5 arcmin, "
              "catalog='VII/237/pgc'); nearest match kept. Name aliases: %s" % ALIAS,
        columns=[{"name": c, "unit": u_} for c, u_ in zip(hdr, units)],
        row_count=len(lines_out),
        note="HyperLEDA logD25/logR25/PA for the ELVES host galaxies. The ELVES host table "
             "itself has NO orientation columns; this supplies them. %d of %d ELVES host "
             "entries matched (Milky Way excluded by construction)."
             % (matched, len(names)),
        measurement_or_model="MEASUREMENT -- isophotal major-axis position angle and axis "
                             "ratio from optical imaging. No dark matter assumed.",
        extra={"acquisition_job": "JOB2 streams-satellites",
               "elves_hosts_total": len(names), "elves_hosts_matched": matched,
               "unresolved": [n for n, _ in unresolved]})
    print("WROTE %s  rows=%d" % (dest, len(lines_out)))


if __name__ == "__main__":
    main()
