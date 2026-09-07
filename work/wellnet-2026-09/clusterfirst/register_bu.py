"""register_bu.py -- register the modification search in the wellnet registry."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "registry"))
import registry  # noqa: E402

RUN_ID = "BV-retraction"
NOTE = (
    "RETRACTS Runs BT and BU. Their factor-of-two cluster residual was substantially an ESTIMATOR ARTEFACT: both filtered ds_obs > 0 before taking the ratio, dropping 10 of 65 rows whose observed lensing was negative (mean -0.53 sigma, ordinary downward noise on a small positive quantity), which removes only downward fluctuations and inflates the apparent missing gravity; and both took a MEDIAN of a ratio whose denominator carried >50%% error on 45 of 55 points. Corrected estimator -- every point kept, inverse-variance weighted, linear space -- gives observed/RAR-predicted = 1.09 +- 0.14, i.e. pred/obs 0.914, consistent with 1. Run BU was additionally VACUOUS: intrinsic scatter 0.000 dex, all 0.523 measurement error, so no modification could have reduced it whatever the physics. Redone as a STACKED test with power: binning 65 points four ways per variable gives per-bin errors of 0.2-0.5, and the residual is flat against gas density, kT, radius, redshift, gas mass and g_bar -- excluding a strong dependence, not a weak one. DOES NOT claim the RAR works on clusters: the literature result rests on far better data, the radial range misses core and far outskirts, and the outer beta of the extended gas model is set by surface brightness where the analytic ACIS vignetting stand-in is least reliable, biasing g_bar high and the residual low. Filed as failures/artefacts/10. Original note follows. "
    "Extended the gas density from the Chandra events already on disk out to "
    "0.94-4.6 Mpc: beta-model fits to the 0.5-2 keV surface brightness, anchored "
    "to ACCEPT where the two overlap (0.04-0.19 dex, 21-59 anchor bins each). One "
    "cluster REJECTED for a degenerate fit at Rc=0 kpc, n0=1.4e5 cm^-3 -- five "
    "orders of magnitude above any cluster core, reproducing the surface "
    "brightness while being physically impossible. Ten survive, taking Run BT's 15 "
    "usable points to 65 rows and sharpening the residual from 0.456 at 0.65 dex "
    "to 0.620 at 0.52 dex. "
    "Then seven one-parameter modification families multiplying the RAR: constant "
    "slip, a0 shift, density^p, temperature^p, radius^p, (1+z)^p, gas mass^p -- the "
    "axes this programme has spent eighteen months on. EVERY ONE drives the median "
    "to within a few percent of unity; NOT ONE reduces the scatter below 0.52 dex "
    "and four make it worse. Each refitted 300x with its variable permuted across "
    "clusters; none beats its own null. The discriminator is the design: anything "
    "with a free amplitude fixes a median, so only a scatter reduction is evidence "
    "of a real dependence. "
    "CONCLUSION: the cluster residual is an AMPLITUDE, not a dependence. It tracks "
    "neither local gas density, temperature, position, redshift nor gas mass. "
    "Consistent with a roughly universal missing-mass fraction; inconsistent with "
    "the environmental modifications tested. Binding limits, both stated: the 0.52 "
    "dex scatter (shape noise over 6,974-25,199 sources per cluster split ten "
    "ways), and the analytic ACIS vignetting model standing in for CIAO exposure "
    "maps, which biases the outer surface brightness, hence beta, hence the outer "
    "density. Sealed half never queried.")

if __name__ == "__main__":
    rec = registry.register(
        RUN_ID, "clusterfirst",
        depends_on=("holdout_seal", "confirmation_status", "temperature_support"),
        outputs=("work/wellnet-2026-09/clusterfirst/modifications.json",
                 "work/wellnet-2026-09/clusterfirst/gas_extended.json",
                 "work/wellnet-2026-09/clusterfirst/REPORT.md"),
        note=NOTE)
    print("registered:", rec["run_id"], rec["status"])
