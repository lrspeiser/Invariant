"""JOB2 satellites lane: HOST DISK ORIENTATION.

The scientific goal needs, per host, the disk normal direction: an inclination (or
axis ratio b/a) AND a position angle. Neither the ELVES host table nor the SAGA DR2
host table carries these. This module acquires them from two independent sources:

  * HyperLEDA (VizieR VII/237/pgc): logR25 (-> b/a) and PA, queried by PGC number
    for the 101 SAGA DR3 hosts (which carry PGC ids natively).
  * Karachentsev+ 2013 Updated Nearby Galaxy Catalog (VizieR J/AJ/145/101/catalog):
    b/a AND inclination i AND the HI line width W50 / rotation amplitude vAmp --
    i.e. an IN-PLANE rotation measure for the same Local Volume hosts that ELVES
    surveys for satellites.

All quantities here are MEASUREMENTS (photometric isophote shape, HI line width).
"""
import os
import re
import sys

HERE = r"C:\Users\henry\Documents\Codex\2026-08-21\Invariant-main-integration\work\wellnet-2026-09\env-data\raw\streams-satellites"
sys.path.insert(0, HERE)
from _manifest import write_manifest, http_get, assert_vizier_tsv  # noqa: E402
from sat_fetch_vizier import parse_tsv  # noqa: E402

MEAS = ("MEASUREMENT -- isophotal axis ratio / position angle from imaging, and HI "
        "21cm line width. Inclination i is a GEOMETRIC INVERSION of the observed axis "
        "ratio assuming an oblate disk of finite intrinsic thickness (Karachentsev+2013 "
        "eq. therein); it is a measured shape, not a dark-matter-dependent quantity.")


def saga_pgcs():
    """PGC ids of the 101 SAGA DR3 hosts, read from the fixed-width MRT (bytes 22-26)."""
    path = os.path.join(HERE, "sat_saga_dr3_tableC1_hosts.mrt")
    lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
    rule = [i for i, l in enumerate(lines) if l.strip().startswith("---") and len(l.strip()) > 20]
    data = [l for l in lines[rule[-1] + 1:] if l.strip()]
    pgcs = []
    for l in data:
        s = l[21:26].strip()
        if s.isdigit():
            pgcs.append(s)
    assert len(pgcs) == len(data), "PGC parse lost rows: %d of %d" % (len(pgcs), len(data))
    return pgcs, len(data)


def fetch(url, name, note, expect_cat=None, min_rows=1):
    dest = os.path.join(HERE, name)
    http_get(url, dest)
    if expect_cat:
        assert_vizier_tsv(dest, expect_catalog=expect_cat, min_rows=min_rows)
    cols, units, n = parse_tsv(dest)
    write_manifest(dest, url, query="GET " + url,
                   columns=[{"name": c, "unit": u} for c, u in zip(cols, units)],
                   row_count=n, note=note, measurement_or_model=MEAS,
                   extra={"acquisition_job": "JOB2 streams-satellites",
                          "purpose": "host disk orientation (inclination / axis ratio / PA)"})
    return n, cols


def main():
    pgcs, nhosts = saga_pgcs()
    print("SAGA DR3 hosts parsed: %d, with PGC: %d" % (nhosts, len(pgcs)))

    url = ("https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=VII/237/pgc"
           "&-out.all&-out.max=unlimited&PGC=" + ",".join(pgcs))
    n, _ = fetch(url, "sat_hyperleda_saga_hosts_PA.tsv",
                 "HyperLEDA (Paturel+2003, VizieR VII/237/pgc) logD25, logR25 (axis ratio) "
                 "and PA for the %d SAGA DR3 host galaxies, queried by PGC id. Supplies the "
                 "host disk orientation and is an INDEPENDENT cross-check on the DESI "
                 "Legacy Imaging ba/PA already in the SAGA DR3 host table." % len(pgcs),
                 expect_cat="VII/237")
    print("HyperLEDA rows for SAGA hosts: %d (requested %d PGCs)" % (n, len(pgcs)))

    url = ("https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=J/AJ/145/101/catalog"
           "&-out.all&-out.max=unlimited")
    n2, _ = fetch(url, "sat_ungc_karachentsev2013_catalog.tsv",
                  "Karachentsev+2013 (AJ 145,101) Updated Nearby Galaxy Catalog: all Local "
                  "Volume galaxies with b/a, inclination i, HI line width W50 and rotation "
                  "amplitude vAmp, B and K photometry. Covers every ELVES host (all D<12Mpc) "
                  "and supplies BOTH the host disk inclination AND an in-plane rotation "
                  "measure for the same systems.",
                  expect_cat="J/AJ/145/101")
    print("UNGC rows: %d" % n2)


if __name__ == "__main__":
    main()
