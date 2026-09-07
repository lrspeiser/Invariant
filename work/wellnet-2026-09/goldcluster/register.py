"""register.py -- register this run in the wellnet run registry BEFORE any work.

Run id letter is provisional: lanes self-register before the record assigns a
letter (see Run BM.3), and the record `gravity-discovery-program.md` is
authoritative.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "registry"))
import registry  # noqa: E402

RUN_ID = "BO-goldcluster"
NOTE = ("Gold Cluster Acquisition Specification (BE.7 instruction). INVENTORY "
        "ONLY: opens no pixel, shear, spectral or kinematic data and computes "
        "no gravity-relevant statistic on any cluster -- asserted mechanically "
        "by guard.py. Archive contact limited to VizieR -meta.all / METAcat "
        "metadata under the three-detector rule and HLSP directory listings. "
        "KiDS and the wide binaries sealed; SPT, X-GAP, CLoGS, Gaia dynamical "
        "products and MUSE/Granata dispersions recorded by identity only, never "
        "opened. Global parameters only; no theory tested. Record letter "
        "provisional.")

if __name__ == "__main__":
    rec = registry.register(
        RUN_ID, "goldcluster",
        depends_on=("catalogue_validation", "holdout_seal",
                    "confirmation_status"),
        outputs=("work/wellnet-2026-09/goldcluster/SPEC.json",
                 "work/wellnet-2026-09/goldcluster/SPEC.md",
                 "work/wellnet-2026-09/goldcluster/ranking.json",
                 "work/wellnet-2026-09/goldcluster/probes.json",
                 "work/wellnet-2026-09/goldcluster/spent.json",
                 "work/wellnet-2026-09/goldcluster/provenance_ledger.json",
                 "work/wellnet-2026-09/goldcluster/REPORT.md"),
        note=NOTE)
    print("registered:", rec["run_id"], rec["status"], rec["depends_on"])
