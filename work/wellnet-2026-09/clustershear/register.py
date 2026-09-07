"""register.py -- register this run in the wellnet run registry BEFORE any work."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "registry"))
import registry  # noqa: E402

RUN_ID = "BQ-clustershear"
NOTE = ("Raw weak-lensing profiles for the OPEN half of the gold-cluster pool "
        "(301 clusters; the 304 sealed are never queried, enforced by "
        "guard.check_target against holdout/loader before every archive call). "
        "Estimator reused unchanged from efeds-hsc/acquire_decade.py so the two "
        "lanes are comparable, including its MEASURED sign convention. Output is "
        "the profile, not the sources. Both nulls pass: cross component "
        "chi2/dof 13.6/10, random pointings 13.9/10. Stacked S/N 31.8 over 301 "
        "clusters. NO gravity law is scored and no mass is fitted; eRASS1 M500/"
        "R500/Mgas500 excluded as weak-lensing-calibrated and therefore circular. "
        "Every surviving trend has a conventional explanation this lane cannot "
        "exclude -- see controls.json.")

if __name__ == "__main__":
    rec = registry.register(
        RUN_ID, "clustershear",
        depends_on=("holdout_seal", "confirmation_status", "catalogue_validation"),
        outputs=("work/wellnet-2026-09/clustershear/profiles.jsonl",
                 "work/wellnet-2026-09/clustershear/analysis.json",
                 "work/wellnet-2026-09/clustershear/controls.json",
                 "work/wellnet-2026-09/clustershear/REPORT.md"),
        note=NOTE)
    print("registered:", rec["run_id"], rec["status"])
