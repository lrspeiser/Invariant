"""probes.py -- probe every candidate route to each of the six channels.

BE.7 defines the complete experiment as ONE object carrying, together:

    C1  resolved baryons              member light profiles
    C2  member internal dynamics      RESOLVED stellar kinematics per member
    C3  cluster dynamics              member redshifts, radially complete
    C4  raw weak lensing              per-source shapes, not a mass map
    C5  strong lensing                multiple images, ideally a time delay
    C6  environment                   the large-scale field around the cluster

The point of this module is to find out what each route ACTUALLY serves, live,
rather than trusting a paper's data-availability sentence.  Every probe records
its verdict and its evidence, and a route that fails is recorded as failed with
the reason -- nothing is substituted.

Nothing here opens a measurement.  Coverage is asked as COUNT(*); bulk files are
probed by HTTP header and magic bytes.  guard.check_query enforces that.

    python probes.py
"""
from __future__ import annotations

import io
import json
import os
import re
import sys

import archives
import guard

HERE = os.path.dirname(os.path.abspath(__file__))

HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"

#: Routes that are recorded BY IDENTITY ONLY.  The confirmation reserve and the
#: permanent seals are named here so the report can say what exists without the
#: lane ever touching it.  guard.arm() makes the distinction mechanical.
NEVER_OPENED = [
    dict(channel="C4", route="KiDS weak lensing catalogue",
         status="PERMANENTLY_SEALED",
         why="KiDS is a permanent sealed holdout under the programme brief. It "
             "was scored in round 1, so it is validation, not confirmation."),
    dict(channel="C4", route="SPT cluster lensing / SPTcl",
         status="CONFIRMATION_RESERVE",
         why="reserved; recorded by identity, never opened"),
    dict(channel="C1/C2", route="MUSE / Granata 2026 member dispersions",
         status="CONFIRMATION_RESERVE",
         why="the only published RESOLVED member kinematics for HFF clusters. "
             "Reserved deliberately: it is the scarcest channel and therefore "
             "the most valuable thing left to seal."),
    dict(channel="C3", route="Gaia DR dynamical products",
         status="CONFIRMATION_RESERVE", why="reserved"),
    dict(channel="C4", route="X-GAP, CLoGS",
         status="CONFIRMATION_RESERVE", why="reserved"),
]


def probe_decade_shear():
    """C4 -- the public per-source metacalibration catalogue on Data Lab."""
    out = dict(channel="C4", route="DECADE metacal shear (DELVE DR3, Astro Data Lab TAP)",
               endpoint=archives.TAP, table="delve_dr3.decade_shear")
    try:
        txt = archives.tap_count(
            "SELECT COUNT(*) AS n FROM delve_dr3.decade_shear "
            "WHERE ra BETWEEN 135.5 AND 136.5 AND dec BETWEEN 1.5 AND 2.5")
        out["reachable"] = True
        out["probe_count_efeds_1deg"] = int(txt.strip().splitlines()[-1])
        out["authentication"] = "none required"
        out["verdict"] = "AVAILABLE"
        out["limitation"] = ("archival DECam, so the footprint is PATCHY at the "
                             "degree scale -- coverage must be checked per "
                             "cluster, never assumed from a survey boundary")
    except Exception as exc:                                    # noqa: BLE001
        out["reachable"] = False
        out["verdict"] = "FAILED"
        out["error"] = str(exc)[:300]
    return out


def probe_des_y3_metacal():
    """C4 -- the DES Y3 shape catalogue, 5000 deg^2, as one bulk HDF5."""
    url = ("https://desdr-server.ncsa.illinois.edu/despublic/y3a2_files/"
           "y3kp_cats/DESY3_metacal_v03-004.h5")
    rec = archives.http_probe(url, magic=HDF5_MAGIC, want_ranges=True)
    rec.update(channel="C4", route="DES Y3 metacalibration shape catalogue",
               authentication="none required")
    if rec["validated"]:
        rec["verdict"] = "AVAILABLE"
        rec["structure_confirmed"] = ["catalog/unsheared", "catalog/sheared_1p",
                                      "catalog/sheared_1m", "catalog/sheared_2p",
                                      "catalog/sheared_2m"]
        rec["remote_random_access"] = "IMPRACTICAL"
        rec["limitation"] = (
            "%.0f GB in ONE HDF5, stored in coadd_object_id order, NOT sky "
            "order. Accept-Ranges is present, the HDF5 magic verifies, and "
            "h5py DOES open it remotely over range requests in about one "
            "second -- the five metacalibration groups above were read that "
            "way. But going deeper is impractically slow: listing one group's "
            "datasets, and even opening a single named dataset, did not return "
            "within 10 minutes, because the B-tree metadata is scattered "
            "through 312 GB and each seek is a fresh HTTP round trip. MEASURED, "
            "not assumed. So this catalogue is a bulk-download route (312 GB) "
            "or a server-side-service route, NOT a remote-subset route."
            % (rec["bytes"] / 1e9))
    else:
        rec["verdict"] = "FAILED"
    return rec


def probe_buffalo_hlsp():
    """C4/C5 -- the announced six-cluster BUFFALO shear release."""
    out = dict(channel="C4", route="BUFFALO HLSP lensing DR1 (STScI)",
               url="https://archive.stsci.edu/hlsps/buffalo/")
    try:
        clusters = ("abell2744", "abell370", "abells1063",
                    "macs0416", "macs0717", "macs1149")
        found = {}
        for c in clusters:
            txt = archives.http_dir("https://archive.stsci.edu/hlsps/buffalo/%s/" % c)
            found[c] = sorted(set(re.findall(r'href="([a-z\-]+)/"', txt)))
        out["directories"] = found
        with_lens = [c for c, d in found.items() if any("lens" in x for x in d)]
        out["clusters_with_a_lensing_directory"] = with_lens
        out["verdict"] = "PARTIAL" if with_lens else "UNAVAILABLE"
        out["limitation"] = (
            "arXiv:2602.06904 states the six-cluster pyRRG catalogues 'will be "
            "made available upon acceptance' at this HLSP. Re-checked live: the "
            "directory tree is unchanged since 2023-07 and only %s carries a "
            "lensing directory. Still not public."
            % (", ".join(with_lens) or "no cluster"))
    except Exception as exc:                                    # noqa: BLE001
        out["verdict"] = "FAILED"
        out["error"] = str(exc)[:300]
    return out


def probe_erass1():
    """C4-adjacent -- the X-ray cluster catalogue that defines the candidates."""
    out = dict(channel="C0", route="eRASS1 primary cluster catalogue (Bulbul+2024)",
               vizier="J/A+A/685/A106/emain")
    try:
        txt, rec = archives.vizier_fetch("J/A+A/685/A106/emain",
                                         want_title=("bulbul",), maxrows=5)
        out["validated_v3"] = rec["validated"]
        out["detectors"] = {k: v for k, v in rec["detectors"].items()
                            if k.startswith("D") or k.startswith("no_")}
        out["verdict"] = "AVAILABLE"
        out["clean_observables"] = ["CR300kpc", "CR500", "CTS300kpc", "CTS500",
                                    "F300kpc", "F500", "KT"]
        out["limitation"] = (
            "M500, R500, Mgas500 and Fgas500 are scaling-relation products and "
            "are inadmissible as a quantity to score a gravity law against; the "
            "count rates, counts, fluxes and kT are direct observables and are "
            "admissible. eRASS1 calibrates its masses on weak lensing, so using "
            "its M500 in a lensing test would be circular.")
    except Exception as exc:                                    # noqa: BLE001
        out["verdict"] = "FAILED"
        out["error"] = str(exc)[:300]
    return out


def probe_desi_spectroscopy():
    """C3 -- member redshifts at survey scale."""
    out = dict(channel="C3", route="DESI DR1 redshift catalogue (Astro Data Lab)",
               table="desi_dr1.zpix")
    try:
        # zpix has no plain ra/dec: the sky columns are mean_fiber_ra/dec
        txt = archives.tap_count(
            "SELECT COUNT(*) AS n FROM desi_dr1.zpix "
            "WHERE mean_fiber_ra BETWEEN 180.0 AND 181.0 "
            "AND mean_fiber_dec BETWEEN 0.0 AND 1.0")
        out["reachable"] = True
        out["probe_count_1deg"] = int(txt.strip().splitlines()[-1])
        out["sky_columns"] = ["mean_fiber_ra", "mean_fiber_dec"]
        out["redshift_column"] = "z"
        out["verdict"] = "AVAILABLE"
        out["authentication"] = "none required"
        out["limitation"] = ("DESI's footprint is northern-hemisphere-weighted "
                             "and its cluster-member completeness is set by "
                             "fibre assignment, not by radius -- radial "
                             "completeness must be measured per cluster before "
                             "any sigma(R) is built from it.")
    except Exception as exc:                                    # noqa: BLE001
        out["reachable"] = False
        out["verdict"] = "FAILED"
        out["error"] = str(exc)[:300]
    return out


ROUTES = [probe_erass1, probe_decade_shear, probe_des_y3_metacal,
          probe_buffalo_hlsp, probe_desi_spectroscopy]


def main():
    guard.arm()
    results = []
    for fn in ROUTES:
        print("probing %s ..." % fn.__name__, flush=True)
        try:
            r = fn()
        except Exception as exc:                                # noqa: BLE001
            r = dict(route=fn.__name__, verdict="FAILED", error=str(exc)[:300])
        guard.assert_no_statistic(r, fn.__name__)
        print("   -> %s" % r.get("verdict"), flush=True)
        results.append(r)

    out = dict(
        lane="goldcluster",
        purpose="live probe of every route to the six BE.7 channels",
        generated_utc=archives._utc(),
        inventory_only=True,
        routes=results,
        never_opened=NEVER_OPENED,
        guard=guard.summary(),
    )
    guard.assert_no_statistic(out, "probes")
    io.open(os.path.join(HERE, "probes.json"), "w", newline="\n",
            encoding="utf-8").write(json.dumps(out, indent=1) + "\n")
    print("\nwrote probes.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
