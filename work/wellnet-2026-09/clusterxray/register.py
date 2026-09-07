"""register.py -- register this run in the wellnet run registry BEFORE any work."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "registry"))
import registry  # noqa: E402

RUN_ID = "BR-clusterxray"
NOTE = ("Chandra ACIS level-2 EVENT LISTS for the six HFF/A2029 clusters, and 2D "
        "brightness and hardness maps built from them. Motivation: every thermal "
        "profile the programme held for these clusters (cluster-data/gas/accept_*) "
        "is a SPHERICAL DEPROJECTION, so the asymmetry is assumed away before any "
        "number reaches the table. Event lists are upstream of that. 27 "
        "observations, 3.3M photons after filtering, validated by three detectors "
        "(bytes vs Content-Length; EVENTS extension with X/Y/ENERGY; header OBS_ID "
        "and INSTRUME). Two traps recorded: the Ocat search endpoint silently "
        "IGNORES ra/dec/radius and returns HTTP 200 with unrelated engineering "
        "pointings, so targets are resolved by name and then rejected on position; "
        "and the raw hardness gradient (+0.124 centre to 8 arcmin, in EVERY cluster "
        "regardless of dynamical state) is ACIS vignetting, not thermal structure, "
        "and is divided out. Hardness is a temperature PROXY, not a temperature: no "
        "CIAO/CALDB, so no spectral fit. No gravity law is scored.")

if __name__ == "__main__":
    rec = registry.register(
        RUN_ID, "clusterxray",
        depends_on=("catalogue_validation", "temperature_support"),
        outputs=("work/wellnet-2026-09/clusterxray/raw_manifest.json",
                 "work/wellnet-2026-09/clusterxray/maps.json",
                 "work/wellnet-2026-09/clusterxray/cluster_xray_maps.png",
                 "work/wellnet-2026-09/clusterxray/REPORT.md"),
        note=NOTE)
    print("registered:", rec["run_id"], rec["status"])
