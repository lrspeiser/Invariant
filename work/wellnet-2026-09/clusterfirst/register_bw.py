"""register_bw.py -- register the universality test."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "registry"))
import registry  # noqa: E402
RUN_ID = "BW-universality"
NOTE = ("Applies Milgrom's actual criterion -- universality, not goodness of fit -- to "
        "the cluster residual. MOND was derived by requiring one change to produce flat "
        "rotation curves AND Tully-Fisher at once, with one constant and zero free "
        "parameters per galaxy; halos permit both and predict neither. So the cluster "
        "question is whether the residual is ONE NUMBER. "
        "Found and fixed a SECOND broken null first: stacked.py permuted POINTS across "
        "clusters, breaking within-cluster correlation, inflating the null chi2 and "
        "returning p=1.00 on three variables -- real data more consistent than all 2000 "
        "permutations. Permuting CLUSTER LABELS fixes it. "
        "Ten per-cluster residuals span 0.04+-0.35 to 1.87+-0.51, grand mean 1.09. With "
        "shear noise alone chi2=22.0/9, p=0.009 -- which reads as a 2.6 sigma detection "
        "that the residual is NOT universal. IT DOES NOT SURVIVE: none of the gas-model "
        "uncertainty was in those bars. The beta-model normalisation agrees with ACCEPT "
        "only to 0.04-0.19 dex, a 10-56%% error on g_bar comparable to or larger than the "
        "shear errors. Adding it gives p=0.043; adding a conservative 15%% vignetting term "
        "gives p=0.086. VERDICT: consistent with a single universal value; the apparent "
        "variation is mostly my own gas model. "
        "One control passed: residual vs background-source count, a pure noise proxy, "
        "gives r=+0.07 p=0.89, so the variation is not simply the noisiest clusters. No "
        "physical variable tracks it (redshift r=-0.52 p=0.155; ten clusters need |r|>0.63; "
        "kT untestable, only 4 of 10 have one). "
        "BINDING CONSTRAINT is now the analytic ACIS vignetting stand-in: it is the "
        "largest term in the error budget, it sets beta, beta sets the gas mass, and the "
        "gas mass IS the prediction. Needs CIAO exposure maps -- an install and a "
        "re-reduction, not an analysis.")
if __name__ == "__main__":
    rec = registry.register(RUN_ID, "clusterfirst",
        depends_on=("holdout_seal", "confirmation_status", "temperature_support"),
        outputs=("work/wellnet-2026-09/clusterfirst/universality.json",
                 "work/wellnet-2026-09/clusterfirst/REPORT.md"),
        note=NOTE)
    print("registered:", rec["run_id"], rec["status"])
