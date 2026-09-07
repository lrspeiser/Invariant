"""register.py -- register this run in the wellnet run registry."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "registry"))
import registry  # noqa: E402
RUN_ID = "BT-clusterfirst"
NOTE = ("Gravity predicted from the OBSERVED baryons for the 11 open-half clusters "
        "carrying ACCEPT deprojected n_e(r), deep Chandra and public DECADE shear "
        "at once, the Bullet Cluster among them. g_bar computed by direct "
        "integration of the measured gas density -- no halo, no mass model, no "
        "scaling relation -- projected via Abel and compared to DeltaSigma_obs in "
        "projection, where lensing measures. Result reproduces the textbook cluster "
        "discrepancy on independent lensing data: Newton predicts 0.148x observed, "
        "the RAR 0.456x, i.e. baryons under-predict by ~7x and MOND-like laws still "
        "by ~2.2x. NOT a new result; the value is that it replicates with a survey "
        "and reduction pipeline never used on these objects. SEVERELY LIMITED: "
        "ACCEPT stops at 0.34-1.18 Mpc while the shear runs to 4.4, so only 1-3 "
        "lensing bins per cluster fall inside the measured gas and two clusters have "
        "none -- about 15 usable points, 0.65 dex scatter, per-cluster ratios "
        "spanning a factor 17. An uncut first version extrapolated the density past "
        "its data and supplied 3x-140x more mass than ACCEPT measured, giving BETTER "
        "looking numbers (0.145/0.765, 0.48 dex); refused under temperature_support "
        "v2. eRASS1 M500/R500/Mgas500 excluded as weak-lensing calibrated.")
if __name__ == "__main__":
    rec = registry.register(RUN_ID, "clusterfirst",
        depends_on=("holdout_seal", "confirmation_status", "temperature_support"),
        outputs=("work/wellnet-2026-09/clusterfirst/firstprinciples.json",
                 "work/wellnet-2026-09/clusterfirst/REPORT.md"),
        note=NOTE)
    print("registered:", rec["run_id"], rec["status"])
